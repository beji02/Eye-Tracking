# being in gaze_estimation
# python -m experiments.train
# (venv) deiubejan@DESKTOP-KHR914B:~/Thesis/MyWork/MPIIGaze/gaze_estimation$ 
# python -m train -experiment_path /home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/experiments/TestExperiment


from pathlib import Path
import torch.backends.cudnn as cudnn
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

from config.config import load_config, Config
from config.args import parse_experiment_args
from data.data_transformation import create_data_transformations_for_resnet
from data.datasets import MPIIFaceGazeDataset
from model.models import (
    create_L2CS_model,
    get_fc_params,
    get_non_ignored_params,
    get_ignored_params,
)
from train_utils.OldTrainer import TrainerBuilder, Trainer
from utils.utils import get_experiment_path, setup_device, login_wandb


def setup_train_dataloader(config: Config, fold: int) -> DataLoader:
    if config.data.dataset_name == "MPIIFaceGaze":
        data_transformations = create_data_transformations_for_resnet()
        image_dir = Path(config.data.dataset_dir) / "Image"
        label_dir = Path(config.data.dataset_dir) / "Label"

        dataset = MPIIFaceGazeDataset(
            image_dir=str(image_dir),
            label_dir=str(label_dir),
            train=True,
            is_pipeline_test=config.train.is_pipeline_test,
            fold=fold,
            data_transformations=data_transformations,
        )
        train_data_loader = DataLoader(
            dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    return train_data_loader


def setup_optimizer(model: torch.nn.Module, config: Config) -> torch.optim.Optimizer:
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


def setup_experiment_output_dir(experiment_path: str, folds: list[int]) -> None:
    shutil.rmtree(experiment_path / "output", ignore_errors=True)
    os.makedirs(experiment_path / "output" / "models")
    
    for fold in folds:
        os.makedirs(experiment_path / "output" / "models" / f"fold_{fold}")


def main():
    experiment_path = get_experiment_path()
    config = load_config(experiment_path)
    folds = [i for i in range(15)] if config.data.use_all_folds else [0]
    setup_experiment_output_dir(experiment_path, folds)

    device = setup_device(config)

    criterion = nn.CrossEntropyLoss().cuda(device)
    reg_criterion = nn.MSELoss().cuda(device)
    softmax = nn.Softmax(dim=1).cuda(device)

    for fold in folds:
        model = create_L2CS_model(
            arch=config.model.backbone, bin_count=28, device=device
        )
        optimizer = setup_optimizer(model, config)
        train_dataloader = setup_train_dataloader(config, fold)

        trainer = (
            TrainerBuilder()
                .new_session()
                .add_model(model)
                .add_train_dataloader(train_dataloader)
                .add_fold(fold)
                .add_device(device)
                .add_config(config)
                .add_criterion(criterion)
                .add_softmax(softmax)
                .add_reg_criterion(reg_criterion)
                .add_optimizer_gaze(optimizer)
                .build()
        )
        trainer.train()


if __name__ == "__main__":
    main()
