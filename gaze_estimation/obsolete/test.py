import torch.backends.cudnn as cudnn
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from model.models import (
    create_L2CS_model,
    get_fc_params,
    get_non_ignored_params,
    get_ignored_params,
)
import os
from gaze_utils.gaze_utils import angular, gazeto3d
from utils.utils import get_experiment_path, setup_device
from config.config import load_config, Config
from pathlib import Path
from data.datasets import MPIIFaceGazeDataset
from data.data_transformation import create_data_transformations_for_resnet
import logging
import shutil

def setup_experiment_output_dir(experiment_path: str) -> None:
    shutil.rmtree(experiment_path / "output" / "evaluation", ignore_errors=True)
    os.makedirs(experiment_path / "output" / "evaluation")


def setup_test_dataloader(config: Config, fold: int) -> DataLoader:
    if config.data.dataset_name == "MPIIFaceGaze":
        data_transformations = create_data_transformations_for_resnet()
        image_dir = Path(config.data.dataset_dir) / "Image"
        label_dir = Path(config.data.dataset_dir) / "Label"

        dataset = MPIIFaceGazeDataset(
            image_dir=str(image_dir),
            label_dir=str(label_dir),
            train=False,
            is_pipeline_test=config.train.is_pipeline_test,
            fold=fold,
            data_transformations=data_transformations,
        )
        test_dataloader = DataLoader(
            dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    return test_dataloader


def eval_fold_on_all_epochs(model, fold, test_dataloader, device, softmax, config):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] -------- %(message)s --------",
        datefmt="%d/%m/%Y %H:%M:%S",
        handlers=[
            logging.FileHandler(
                config.experiment_path / "output" / "evaluation" / f"eval.log"
            ),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Starting evaluation for fold: {fold}")

    mae_list = []
    epoch_list = []
    for epoch in range(config.train.num_epochs):
        saved_state_dict = torch.load(
            f"{config.experiment_path}/output/models/fold_{fold}/epoch_{epoch+1}.pkl"
        )
        model.load_state_dict(saved_state_dict)
        model.eval()

        mae = eval_model(model, test_dataloader, device, softmax)
        mae_list.append(mae)
        epoch_list.append(epoch + 1)

        logging.info(
            "Epoch: [%d/%d] Mae: %.4f"
            % (
                epoch+1,
                config.train.num_epochs,
                mae
            )
        )

    plt.set_loglevel(level = 'warning')
    fig = plt.figure(figsize=(14, 8))
    plt.xlabel("epoch")
    plt.ylabel("mae")
    plt.title("Gaze angular error")
    plt.plot(epoch_list, mae_list, color="blue", label="mae")
    plt.xticks(epoch_list)
    fig.savefig(f"{config.experiment_path}/output/evaluation/fold_{fold}.png", format="png")

    logging.info(f"Evaluation complete for model with fold: {fold}")



def eval_model(model, test_data_loader, device, softmax):
    mae = 0.0
    counter = 0
    idx_tensor = [idx for idx in range(28)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(device)

    with torch.no_grad():
        for j, (images, labels, cont_labels) in enumerate(test_data_loader):
            images = Variable(images).cuda(device)
            counter += cont_labels.size(0)

            label_pitch = cont_labels[:, 0].float() * np.pi / 180
            label_yaw = cont_labels[:, 1].float() * np.pi / 180

            gaze_pitch, gaze_yaw = model(images)

            # Binned predictions
            _, pitch_bpred = torch.max(gaze_pitch.data, 1)
            _, yaw_bpred = torch.max(gaze_yaw.data, 1)

            # Continuous predictions
            pitch_predicted = softmax(gaze_pitch)
            yaw_predicted = softmax(gaze_yaw)

            # mapping from binned (0 to 28) to angels (-42 to 42)
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 42
            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 42

            pitch_predicted = pitch_predicted * np.pi / 180
            yaw_predicted = yaw_predicted * np.pi / 180

            for p, y, pl, yl in zip(
                pitch_predicted, yaw_predicted, label_pitch, label_yaw
            ):
                mae += angular(gazeto3d([p, y]), gazeto3d([pl, yl]))
    return mae / counter


if __name__ == "__main__":
    experiment_path = get_experiment_path()
    config = load_config(experiment_path)
    setup_experiment_output_dir(experiment_path)

    device = setup_device(config)

    model = create_L2CS_model(arch=config.model.backbone, bin_count=28, device=device)
    softmax = nn.Softmax(dim=1).cuda(device)

    folds = [i for i in range(15)] if config.data.use_all_folds else [0]
    for fold in folds:
        test_dataloader = setup_test_dataloader(config, fold)
        eval_fold_on_all_epochs(model, fold, test_dataloader, device, softmax, config)
