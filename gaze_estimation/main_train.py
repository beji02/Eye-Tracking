from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from devtools import pprint
import shutil
import wandb
import logging

from config.config import load_config, Config
from config.args import parse_experiment_args
from data.data_transformation import create_data_transformations_for_resnet, create_mobileNetV2_transformations
from data.datasets import MPIIFaceGazeDataset
from model.models import (
    create_L2CS_model,
    get_fc_params,
    get_non_ignored_params,
    get_ignored_params,
)
from train_utils.OldTrainer import TrainerBuilder, Trainer
from utils.utils import get_experiment_path, login_wandb, set_randomness
import random
from gaze_utils.gaze_utils import angular, gazeto3d
from demo.visualize import add_gaze_to_image
from PyQt5 import QtCore
from model.models import MyLoveMobileNet as MobileNet

BATCHES_SEEN = 0
RANDOM_INDEXES = None
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath)

def setup_experiment_output_dir(experiment_path: Path, folds: list[int]):
    shutil.rmtree(experiment_path / "output", ignore_errors=True)
    os.makedirs(experiment_path / "output" / "models")

    if len(folds) > 1:
        for fold in folds:
            os.makedirs(experiment_path / "output" / "models" / f"fold_{fold}")
    else:
        fold = folds[0]
        os.makedirs(experiment_path / "output" / "models" / f"fold_{fold}")


def setup_device(config: Config) -> torch.device:
    device = torch.device("cuda:0" if config.train.use_gpu else "cpu")
    return device


def setup_train_dataloader(config: Config, fold: int) -> DataLoader:
    if config.data.dataset_name == "MPIIFaceGaze":
        if config.model.backbone.startswith("ResNet"):
            data_transformations = create_data_transformations_for_resnet()
        elif config.model.backbone.startswith("MobileNet"):
            data_transformations = create_mobileNetV2_transformations()
        image_dir = Path(config.data.dataset_dir) / "Image"
        label_dir = Path(config.data.dataset_dir) / "Label"

        train_dataset = MPIIFaceGazeDataset(
            config,
            image_dir=image_dir,
            label_dir=label_dir,
            train=True,
            is_pipeline_test=config.train.is_pipeline_test,
            fold=fold,
            data_transformations=data_transformations,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    return train_dataloader


def setup_test_dataloader(config: Config, fold: int) -> DataLoader:
    if config.data.dataset_name == "MPIIFaceGaze":
        if config.model.backbone.startswith("ResNet"):
            data_transformations = create_data_transformations_for_resnet()
        elif config.model.backbone.startswith("MobileNet"):
            data_transformations = create_mobileNetV2_transformations()
        image_dir = Path(config.data.dataset_dir) / "Image"
        label_dir = Path(config.data.dataset_dir) / "Label"

        test_dataset = MPIIFaceGazeDataset(
            config,
            image_dir=image_dir,
            label_dir=label_dir,
            train=False,
            is_pipeline_test=config.train.is_pipeline_test,
            fold=fold,
            data_transformations=data_transformations,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    return test_dataloader


def setup_optimizer_l2cs(model: torch.nn.Module, config: Config) -> torch.optim.Optimizer:
    if config.train.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            [
                {"params": get_ignored_params(model), "lr": 0},
                {
                    "params": get_non_ignored_params(model),
                    "lr": config.train.learning_rate,
                },
                {"params": get_fc_params(model), "lr": config.train.learning_rate},
            ],
            lr=config.train.learning_rate,
        )
    return optimizer


def compute_regression_loss(
    config, cont_labels, predictions, regression_criterion, softmax, device
):
    cont_label_pitch = Variable(cont_labels[:, 0]).cuda(device)
    cont_label_yaw = Variable(cont_labels[:, 1]).cuda(device)
    predicted_pitch, predicted_yaw = predictions

    predicted_pitch = softmax(predicted_pitch)
    predicted_yaw = softmax(predicted_yaw)

    idx_tensor = [idx for idx in range(config.model.bins)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(device)

    predicted_pitch = torch.sum(predicted_pitch * idx_tensor, 1) * 3 - 42 #- 45 # * 3 - 42
    predicted_yaw = torch.sum(predicted_yaw * idx_tensor, 1) * 3 - 42 #- 45 # * 3 - 42

    loss_regression_pitch = regression_criterion(predicted_pitch, cont_label_pitch)
    loss_regression_yaw = regression_criterion(predicted_yaw, cont_label_yaw)

    return loss_regression_pitch, loss_regression_yaw


def compute_binned_loss(binned_labels, predictions, binned_criterion, device):
    binned_label_pitch = Variable(binned_labels[:, 0]).cuda(device)
    binned_label_yaw = Variable(binned_labels[:, 1]).cuda(device)
    predicted_pitch, predicted_yaw = predictions

    loss_binned_pitch = binned_criterion(predicted_pitch, binned_label_pitch)
    loss_binned_yaw = binned_criterion(predicted_yaw, binned_label_yaw)

    return loss_binned_pitch, loss_binned_yaw


def compute_resnet_angular_error(config, cont_labels, predictions, softmax, device):
    predicted_pitch, predicted_yaw = predictions

    predicted_pitch = softmax(predicted_pitch)
    predicted_yaw = softmax(predicted_yaw)

    idx_tensor = [idx for idx in range(config.model.bins)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(device)

    predicted_pitch = torch.sum(predicted_pitch * idx_tensor, 1).cpu()* 3 - 42 #- 45 # * 3 - 42
    predicted_yaw = torch.sum(predicted_yaw * idx_tensor, 1).cpu() * 3 - 42 #- 45 # * 3 - 42

    predicted_pitch = predicted_pitch * np.pi / 180
    predicted_yaw = predicted_yaw * np.pi / 180

    cont_label_pitch = cont_labels[:, 0].float() * np.pi / 180
    cont_label_yaw = cont_labels[:, 1].float() * np.pi / 180

    angular_error = 0.0
    for p, y, pl, yl in zip(
        predicted_pitch, predicted_yaw, cont_label_pitch, cont_label_yaw
    ):
        angular_error += angular(gazeto3d([p, y]), gazeto3d([pl, yl]))

    return angular_error

def compute_mobilenet_angular_error(labels, predictions):
    preds_pitch = predictions[:, 0].float() * np.pi / 180
    preds_yaw = predictions[:, 1].float() * np.pi / 180

    label_pitch = labels[:, 0].float() * np.pi / 180
    label_yaw = labels[:, 1].float() * np.pi / 180

    angular_error = 0.0
    for p, y, pl, yl in zip(
        preds_pitch, preds_yaw, label_pitch, label_yaw
    ):
        angular_error += angular(gazeto3d([p, y]), gazeto3d([pl, yl]))

    return angular_error


def train_resnet_one_epoch(
    config: Config,
    train_dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    training_tools: dict,
    num_epochs_trained: int,
):
    regression_criterion = training_tools['reg_criterion']
    softmax = training_tools["softmax"]
    binned_criterion = training_tools["criterion"]
    optimizer = training_tools["optimizer"]
    global BATCHES_SEEN

    model.train()
    accumulating_loss_pitch = 0.0
    accumulating_loss_yaw = 0.0
    accumulating_loss = 0.0

    logging.info(f"Starting training")
    for i, (images, binned_labels, cont_labels) in enumerate(train_dataloader, start=1):
        BATCHES_SEEN += 1
        images = Variable(images).cuda(device)

        predictions = model(images)

        loss_regression_pitch, loss_regression_yaw = compute_regression_loss(
            config, cont_labels, predictions, regression_criterion, softmax, device
        )
        loss_binned_pitch, loss_binned_yaw = compute_binned_loss(
            binned_labels, predictions, binned_criterion, device
        )

        loss_pitch = loss_regression_pitch + loss_binned_pitch
        loss_yaw = loss_regression_yaw + loss_binned_yaw
        loss = loss_pitch + loss_yaw

        loss_seq = [loss_pitch, loss_yaw]
        grad_seq = [torch.tensor(1.0).cuda(device) for _ in range(len(loss_seq))]

        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss_seq, grad_seq)
        optimizer.step()

        accumulating_loss_pitch += loss_pitch
        accumulating_loss_yaw += loss_yaw
        accumulating_loss += loss

        if i % 100 == 0:
            wandb.log(
                {
                    "train_avg_batch_loss_pitch": accumulating_loss_pitch / i,
                    "train_avg_batch_loss_yaw": accumulating_loss_yaw / i,
                    "train_avg_batch_loss": accumulating_loss / i,
                },
                step=BATCHES_SEEN,
            )

            logging.info(
                "Epoch: [%d/%d], Batch: [%d/%d], Train loss: %.4f, Train pitch loss: %.4f, Train yaw loss: %.4f"
                % (
                    num_epochs_trained,
                    config.train.num_epochs,
                    i,
                    len(train_dataloader),
                    accumulating_loss / i,
                    accumulating_loss_pitch / i,
                    accumulating_loss_yaw / i,
                )
            )
    wandb.log(
        {
            "train_avg_batch_loss_pitch": accumulating_loss_pitch / i,
            "train_avg_batch_loss_yaw": accumulating_loss_yaw / i,
            "train_avg_batch_loss": accumulating_loss / i,
        },
        step=BATCHES_SEEN,
    )

    logging.info(
        "Epoch: [%d/%d], Batch: [%d/%d], Train loss: %.4f, Train pitch loss: %.4f, Train yaw loss: %.4f"
        % (
            num_epochs_trained,
            config.train.num_epochs,
            len(train_dataloader),
            len(train_dataloader),
            accumulating_loss / len(train_dataloader),
            accumulating_loss_pitch / len(train_dataloader),
            accumulating_loss_yaw / len(train_dataloader),
        )
    )


def get_k_unique_random_numbers(k, n):
    unique_numbers = set()
    while len(unique_numbers) < k:
        unique_numbers.add(random.randint(1, n))
    return list(unique_numbers)


def add_visual_evaluation_wandb_for_resnet(config, test_dataloader, device, model, softmax):
    global RANDOM_INDEXES
    test_dataset_len = len(test_dataloader.dataset)
    if RANDOM_INDEXES is None:
        RANDOM_INDEXES = get_k_unique_random_numbers(k=12, n=test_dataset_len - 1)
    random_indexes = RANDOM_INDEXES

    images = [test_dataloader.dataset.get_raw_image(index) for index in random_indexes]
    input_images = torch.stack(
        [(test_dataloader.dataset[index])[0] for index in random_indexes]
    ).to(
        device
    )  # Assuming the data is the first element
    all_gaze_truth = torch.stack(
        [(test_dataloader.dataset[index])[2] for index in random_indexes]
    ).to(
        device
    )  # Assuming the data is the first element
    # print(all_gaze_truth[0].shape)
    # print(all_gaze_truth[0])

    gaze_pitch, gaze_yaw = model(input_images)

    pitch_predicted = softmax(gaze_pitch)
    yaw_predicted = softmax(gaze_yaw)

    # mapping from binned (0 to 28) to angels (-42 to 42) or 0 90 to -45 45
    idx_tensor = [idx for idx in range(config.model.bins)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(device)

    pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 42 #- 45 # * 3 - 42
    yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 42 # * 3 - 42

    pitch_predicted = pitch_predicted * np.pi / 180
    yaw_predicted = yaw_predicted * np.pi / 180

    cont_label_pitch = all_gaze_truth[:, 0].float() * np.pi / 180
    cont_label_yaw = all_gaze_truth[:, 1].float() * np.pi / 180

    all_gaze_truth = [
        (pitch, gaze) for pitch, gaze in zip(cont_label_pitch, cont_label_yaw)
    ]
    all_gaze_predicted = [
        (pitch, gaze) for pitch, gaze in zip(pitch_predicted, yaw_predicted)
    ]
    # print(all_gaze_truth[0].shape)
    # print(all_gaze_truth[0])

    # Create a figure and axes for subplots
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))

    # Iterate over images and their corresponding gaze data
    for i, (image, gaze_truth, gaze_predicted) in enumerate(
        zip(images, all_gaze_truth, all_gaze_predicted)
    ):
        # print(gaze_predicted)
        image = np.array(image)
        image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = image.transpose(2, 0, 1)

        gaze_truth = (gaze_truth[0].cpu(), gaze_truth[1].cpu())
        gaze_predicted = (
            gaze_predicted[0].cpu().detach().numpy(),
            gaze_predicted[1].cpu().detach().numpy(),
        )
        # print(gaze_predicted)

        image_with_gaze_truth = add_gaze_to_image(
            image, ((112.0, 112.0), (112.0, 112.0)), gaze_truth, color=(0, 0, 255)
        )
        image_with_both = add_gaze_to_image(
            image_with_gaze_truth,
            ((112.0, 112.0), (112.0, 112.0)),
            gaze_predicted,
            color=(255, 0, 0),
        )

        image_with_both = image_with_both.transpose(1, 2, 0)
        image_with_both = cv2.cvtColor(image_with_both, cv2.COLOR_RGB2BGR)

        # Calculate subplot position
        row = i // 4
        col = i % 4

        # Plot image on the corresponding subplot
        axes[row, col].imshow(image_with_both)
        axes[row, col].axis("off")

    # Save the plot as an image
    plt.tight_layout()
    plt.savefig("wandb/plot.png")

    # Log the plot to wandb
    wandb.log({"plot": wandb.Image("wandb/plot.png")}, step=BATCHES_SEEN)

def add_visual_evaluation_wandb_for_mobilenet(test_dataloader, device, model):
    global RANDOM_INDEXES
    test_dataset_len = len(test_dataloader.dataset)
    if RANDOM_INDEXES is None:
        RANDOM_INDEXES = get_k_unique_random_numbers(k=12, n=test_dataset_len - 1)
    random_indexes = RANDOM_INDEXES

    images = [test_dataloader.dataset.get_raw_image(index) for index in random_indexes]
    input_images = torch.stack(
        [(test_dataloader.dataset[index])[0] for index in random_indexes]
    ).to(
        device
    )  # Assuming the data is the first element
    all_gaze_truth = torch.stack(
        [(test_dataloader.dataset[index])[2] for index in random_indexes]
    ).to(
        device
    )  # Assuming the data is the first element
    # print(all_gaze_truth[0].shape)
    # print(all_gaze_truth[0])

    predictions = model(input_images)

    pred_pitch = predictions[:, 0].float() * np.pi / 180
    pred_yaw = predictions[:, 1].float() * np.pi / 180

    label_pitch = all_gaze_truth[:, 0].float() * np.pi / 180
    label_yaw = all_gaze_truth[:, 1].float() * np.pi / 180

    all_gaze_truth = [
        (pitch, gaze) for pitch, gaze in zip(label_pitch, label_yaw)
    ]
    all_gaze_predicted = [
        (pitch, gaze) for pitch, gaze in zip(pred_pitch, pred_yaw)
    ]
    # print(all_gaze_truth[0].shape)
    # print(all_gaze_truth[0])

    # Create a figure and axes for subplots
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))

    # Iterate over images and their corresponding gaze data
    for i, (image, gaze_truth, gaze_predicted) in enumerate(
        zip(images, all_gaze_truth, all_gaze_predicted)
    ):
        # print(gaze_predicted)
        image = np.array(image)
        image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = image.transpose(2, 0, 1)

        gaze_truth = (gaze_truth[0].cpu(), gaze_truth[1].cpu())
        gaze_predicted = (
            gaze_predicted[0].cpu().detach().numpy(),
            gaze_predicted[1].cpu().detach().numpy(),
        )
        # print(gaze_predicted)

        image_with_gaze_truth = add_gaze_to_image(
            image, ((112.0, 112.0), (112.0, 112.0)), gaze_truth, color=(0, 0, 255)
        )
        image_with_both = add_gaze_to_image(
            image_with_gaze_truth,
            ((112.0, 112.0), (112.0, 112.0)),
            gaze_predicted,
            color=(255, 0, 0),
        )

        image_with_both = image_with_both.transpose(1, 2, 0)
        image_with_both = cv2.cvtColor(image_with_both, cv2.COLOR_RGB2BGR)

        # Calculate subplot position
        row = i // 4
        col = i % 4

        # Plot image on the corresponding subplot
        axes[row, col].imshow(image_with_both)
        axes[row, col].axis("off")

    # Save the plot as an image
    plt.tight_layout()
    plt.savefig("wandb/plot.png")

    # Log the plot to wandb
    wandb.log({"plot": wandb.Image("wandb/plot.png")}, step=BATCHES_SEEN)


def evaluate_resnet_on_test(
    config: Config,
    test_dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    training_tools: dict,
    num_epochs_trained: int,
):
    regression_criterion = training_tools['reg_criterion']
    softmax = training_tools["softmax"]
    binned_criterion = training_tools["criterion"]

    global BATCHES_SEEN

    model.eval()
    accumulating_loss_pitch = 0.0
    accumulating_loss_yaw = 0.0
    accumulating_loss = 0.0
    accumulating_angular_error = 0.0
    sample_counter = 0

    logging.info(f"Starting evaluation on test")
    with torch.no_grad():
        for i, (images, binned_labels, cont_labels) in enumerate(
            test_dataloader, start=1
        ):
            BATCHES_SEEN += 1
            sample_counter += images.size(0)
            images = Variable(images).cuda(device)

            predictions = model(images)

            loss_regression_pitch, loss_regression_yaw = compute_regression_loss(
                config, cont_labels, predictions, regression_criterion, softmax, device
            )
            loss_binned_pitch, loss_binned_yaw = compute_binned_loss(
                binned_labels, predictions, binned_criterion, device
            )

            loss_pitch = loss_regression_pitch + loss_binned_pitch
            loss_yaw = loss_regression_yaw + loss_binned_yaw
            loss = loss_pitch + loss_yaw

            angular_error = compute_resnet_angular_error(
                config, cont_labels, predictions, softmax, device
            )

            accumulating_loss_pitch += loss_pitch
            accumulating_loss_yaw += loss_yaw
            accumulating_loss += loss
            accumulating_angular_error += angular_error

            if i % 100 == 0:
                wandb.log(
                    {
                        "test_avg_batch_loss_pitch": accumulating_loss_pitch / i,
                        "test_avg_batch_loss_yaw": accumulating_loss_yaw / i,
                        "test_avg_batch_loss": accumulating_loss / i,
                        "test_avg_sample_angular_error": accumulating_angular_error / sample_counter,
                    },
                    step=BATCHES_SEEN,
                )

                logging.info(
                    "Epoch: [%d/%d], Batch: [%d/%d], Test loss: %.4f, Mae: %.4f"
                    % (
                        num_epochs_trained,
                        config.train.num_epochs,
                        i,
                        len(test_dataloader),
                        accumulating_loss / i,
                        accumulating_angular_error / sample_counter,
                    )
                )
    wandb.log(
        {
            "test_avg_batch_loss_pitch": accumulating_loss_pitch / len(test_dataloader),
            "test_avg_batch_loss_yaw": accumulating_loss_yaw / len(test_dataloader),
            "test_avg_batch_loss": accumulating_loss / len(test_dataloader),
            "test_avg_sample_angular_error": accumulating_angular_error / sample_counter,
        },
        step=BATCHES_SEEN,
    )

    logging.info(
        "Epoch: [%d/%d], Batch: [%d/%d], Test loss: %.4f, Mae: %.4f"
        % (
            num_epochs_trained,
            config.train.num_epochs,
            len(test_dataloader),
            len(test_dataloader),
            accumulating_loss / len(test_dataloader),
            accumulating_angular_error / sample_counter,
        )
    )
    add_visual_evaluation_wandb_for_resnet(config, test_dataloader, device, model, softmax)


def setup_logger(config: Config):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] -------- %(message)s --------",
        datefmt="%d/%m/%Y %H:%M:%S",
        handlers=[
            logging.FileHandler(config.experiment_path / "output" / "train.log"),
            logging.StreamHandler(),
        ],
    )


def init_wandb(config: Config, fold: int):
    wandb_config = dict(
        epochs=config.train.num_epochs,
        batch_size=config.train.batch_size,
        learning_rate=config.train.learning_rate,
        optimizer=config.train.optimizer,
        use_gpu=config.train.use_gpu,
        seed=config.train.seed,
        is_pipeline_test=config.train.is_pipeline_test,
        backbone=config.model.backbone,
        dataset_name=config.data.dataset_name,
        folding_strategy=config.data.folding_strategy,
        fold=fold
    )
    return wandb.init(project="gaze_estimation", config=wandb_config)


def get_images_from_dataloader(dataloader, device):
    for images, _, _ in dataloader:
        image = images[0]
        break
    images = torch.unsqueeze(image, 0)
    return Variable(images).to(device)

def create_model(config:Config, device):
    model = None
    if config.model.backbone.startswith("ResNet"):
        model = create_L2CS_model(
                arch=config.model.backbone, bin_count=config.model.bins, device=device
            )
    elif config.model.backbone.startswith("MobileNet"):
        model = MobileNet()
        model = model.to(device)
    return model

def create_training_tools(config:Config, model, device):
    training_tools = None
    if config.model.backbone.startswith("ResNet"):
        criterion = nn.CrossEntropyLoss().cuda(device)
        reg_criterion = nn.MSELoss().cuda(device)
        softmax = nn.Softmax(dim=1).cuda(device)
        optimizer = setup_optimizer_l2cs(model, config)
        training_tools = {
            "criterion": criterion,
            "reg_criterion": reg_criterion,
            "softmax": softmax,
            "optimizer": optimizer
        }
    elif config.model.backbone.startswith("MobileNet"):
        criterion = nn.MSELoss().cuda(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
        training_tools = {
            "criterion": criterion,
            "optimizer": optimizer
        }
    return training_tools

def train_resnet(config: Config, model: torch.nn.Module, device: torch.device, train_dataloader: DataLoader, test_dataloader: DataLoader, training_tools: dict):
    global BATCHES_SEEN
    epoch = 0
    evaluate_resnet_on_test(
        config,
        test_dataloader,
        model,
        device,
        training_tools,
        epoch,
    )
    for epoch in range(1, config.train.num_epochs + 1):
        wandb.log({"epoch": epoch}, step=BATCHES_SEEN)
        train_resnet_one_epoch(
            config,
            train_dataloader,
            model,
            device,
            training_tools,
            epoch,
        )
        evaluate_resnet_on_test(
            config,
            test_dataloader,
            model,
            device,
            training_tools,
            epoch,
        )

def evaluate_mobilenet_on_test(
    config: Config,
    test_dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    training_tools: dict,
    num_epochs_trained: int,
):
    criterion = training_tools['criterion']

    global BATCHES_SEEN

    model.eval()
    accumulating_loss_pitch = 0.0
    accumulating_loss_yaw = 0.0
    accumulating_loss = 0.0
    accumulating_angular_error = 0.0
    sample_counter = 0

    logging.info(f"Starting evaluation on test")
    with torch.no_grad():
        for i, (images, _, labels) in enumerate(
            test_dataloader, start=1
        ):
            BATCHES_SEEN += 1
            sample_counter += images.size(0)
            images = Variable(images).cuda(device)

            predictions = model(images).cpu()
            pitch_label, yaw_label = torch.split(labels, split_size_or_sections=1, dim=1)
            pitch_pred, yaw_pred = torch.split(predictions, split_size_or_sections=1, dim=1)

            loss_pitch = criterion(pitch_pred, pitch_label)
            loss_yaw = criterion(yaw_pred, yaw_label)
            loss = loss_pitch + loss_yaw

            angular_error = compute_mobilenet_angular_error(labels, predictions)

            accumulating_loss_pitch += loss_pitch
            accumulating_loss_yaw += loss_yaw
            accumulating_loss += loss
            accumulating_angular_error += angular_error

            if i % 100 == 0:
                wandb.log(
                    {
                        "test_avg_batch_loss_pitch": accumulating_loss_pitch / i,
                        "test_avg_batch_loss_yaw": accumulating_loss_yaw / i,
                        "test_avg_batch_loss": accumulating_loss / i,
                        "test_avg_sample_angular_error": accumulating_angular_error / sample_counter,
                    },
                    step=BATCHES_SEEN,
                )

                logging.info(
                    "Epoch: [%d/%d], Batch: [%d/%d], Test loss: %.4f, Mae: %.4f"
                    % (
                        num_epochs_trained,
                        config.train.num_epochs,
                        i,
                        len(test_dataloader),
                        accumulating_loss / i,
                        accumulating_angular_error / sample_counter,
                    )
                )
    wandb.log(
        {
            "test_avg_batch_loss_pitch": accumulating_loss_pitch / len(test_dataloader),
            "test_avg_batch_loss_yaw": accumulating_loss_yaw / len(test_dataloader),
            "test_avg_batch_loss": accumulating_loss / len(test_dataloader),
            "test_avg_sample_angular_error": accumulating_angular_error / sample_counter,
        },
        step=BATCHES_SEEN,
    )

    logging.info(
        "Epoch: [%d/%d], Batch: [%d/%d], Test loss: %.4f, Mae: %.4f"
        % (
            num_epochs_trained,
            config.train.num_epochs,
            len(test_dataloader),
            len(test_dataloader),
            accumulating_loss / len(test_dataloader),
            accumulating_angular_error / sample_counter,
        )
    )
    add_visual_evaluation_wandb_for_mobilenet(test_dataloader, device, model)

def train_mobilenet_one_epoch(
    config: Config,
    train_dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    training_tools: dict,
    num_epochs_trained: int,
):
    criterion = training_tools['criterion']
    optimizer = training_tools["optimizer"]
    global BATCHES_SEEN

    model.train()
    accumulating_loss_pitch = 0.0
    accumulating_loss_yaw = 0.0
    accumulating_loss = 0.0

    logging.info(f"Starting training")
    for i, (images, _, labels) in enumerate(train_dataloader, start=1):
        BATCHES_SEEN += 1
        images = Variable(images).cuda(device)
        labels = Variable(labels).cuda(device)

        predictions = model(images)

        pitch_label, yaw_label = torch.split(labels, split_size_or_sections=1, dim=1)
        pitch_pred, yaw_pred = torch.split(predictions, split_size_or_sections=1, dim=1)

        loss_pitch = criterion(pitch_pred, pitch_label)
        loss_yaw = criterion(yaw_pred, yaw_label)
        loss = loss_pitch + loss_yaw

        loss_back = criterion(predictions, labels)

        optimizer.zero_grad()
        torch.autograd.backward(loss_back)
        optimizer.step()

        accumulating_loss_pitch += loss_pitch
        accumulating_loss_yaw += loss_yaw
        accumulating_loss += loss

        if i % 100 == 0:
            wandb.log(
                {
                    "train_avg_batch_loss_pitch": accumulating_loss_pitch / i,
                    "train_avg_batch_loss_yaw": accumulating_loss_yaw / i,
                    "train_avg_batch_loss": accumulating_loss / i,
                },
                step=BATCHES_SEEN,
            )

            logging.info(
                "Epoch: [%d/%d], Batch: [%d/%d], Train loss: %.4f, Train pitch loss: %.4f, Train yaw loss: %.4f"
                % (
                    num_epochs_trained,
                    config.train.num_epochs,
                    i,
                    len(train_dataloader),
                    accumulating_loss / i,
                    accumulating_loss_pitch / i,
                    accumulating_loss_yaw / i,
                )
            )
    wandb.log(
        {
            "train_avg_batch_loss_pitch": accumulating_loss_pitch / i,
            "train_avg_batch_loss_yaw": accumulating_loss_yaw / i,
            "train_avg_batch_loss": accumulating_loss / i,
        },
        step=BATCHES_SEEN,
    )

    logging.info(
        "Epoch: [%d/%d], Batch: [%d/%d], Train loss: %.4f, Train pitch loss: %.4f, Train yaw loss: %.4f"
        % (
            num_epochs_trained,
            config.train.num_epochs,
            len(train_dataloader),
            len(train_dataloader),
            accumulating_loss / len(train_dataloader),
            accumulating_loss_pitch / len(train_dataloader),
            accumulating_loss_yaw / len(train_dataloader),
        )
    )


def train_mobilenet(config: Config, model: torch.nn.Module, device: torch.device, train_dataloader: DataLoader, test_dataloader: DataLoader, training_tools: dict):
    global BATCHES_SEEN
    epoch = 0
    evaluate_mobilenet_on_test(
        config,
        test_dataloader,
        model,
        device,
        training_tools,
        epoch,
    )
    for epoch in range(1, config.train.num_epochs + 1):
        wandb.log({"epoch": epoch}, step=BATCHES_SEEN)
        train_mobilenet_one_epoch(
            config,
            train_dataloader,
            model,
            device,
            training_tools,
            epoch,
        )
        evaluate_mobilenet_on_test(
            config,
            test_dataloader,
            model,
            device,
            training_tools,
            epoch,
        )

def train(config: Config, model: torch.nn.Module, device: torch.device, train_dataloader: DataLoader, test_dataloader: DataLoader, training_tools: dict):
    if config.model.backbone.startswith("ResNet"):
        train_resnet(config, model, device, train_dataloader, test_dataloader, training_tools)
    elif config.model.backbone.startswith("MobileNet"):
        train_mobilenet(config, model, device, train_dataloader, test_dataloader, training_tools)
    

def main():
    experiment_path = get_experiment_path()
    config = load_config(experiment_path)
    set_randomness(config)

    folds = (
        [i for i in range(15)]
        if config.data.folding_strategy == 0
        #else [random.randint(1, 15) - 1]
        else [10]
    )
    setup_experiment_output_dir(experiment_path, folds)

    device = setup_device(config)
    login_wandb()
    setup_logger(config)

    for fold in folds:
        with init_wandb(config, fold):
            train_dataloader = setup_train_dataloader(config, fold)
            test_dataloader = setup_test_dataloader(config, fold)

            model = create_model(config, device)
            training_tools = create_training_tools(config, model, device)
            train(config, model, device, train_dataloader, test_dataloader, training_tools)

            torch.save(
                model.state_dict(),
                config.experiment_path
                / "output"
                / "models"
                / f"fold_{fold}"
                / f"model.pkl",
            )

            images = get_images_from_dataloader(train_dataloader, device)
            torch.onnx.export(
                model,
                images,
                config.experiment_path
                / "output"
                / "models"
                / f"fold_{fold}"
                / "model.onnx",
            )
            wandb.save(
                config.experiment_path
                / "output"
                / "models"
                / f"fold_{fold}"
                / "model.onnx",
                base_path=config.experiment_path / "output" / "models" / f"fold_{fold}",
            )


if __name__ == "__main__":
    main()
