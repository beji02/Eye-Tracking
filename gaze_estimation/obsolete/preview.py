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


if __name__ == "__main__":
    experiment_path = get_experiment_path()
    config = load_config(experiment_path)

    device = setup_device(config)

    model = create_L2CS_model(arch=config.model.backbone, bin_count=28, device=device)
    softmax = nn.Softmax(dim=1).cuda(device)

    fold = 0
    epoch = 9
    saved_state_dict = torch.load(
            f"{config.experiment_path}/output/models/fold_{fold}/epoch_{epoch+1}.pkl"
        )
    model.load_state_dict(saved_state_dict)
    model.eval()

    # setup data
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
        first_data = dataset[1000]
        image = first_data[0]
        cont_labels = first_data[2]
        label_pitch = cont_labels[0].float() * np.pi / 180
        label_yaw = cont_labels[1].float() * np.pi / 180

        # print(image, "A\n", cont_labels, label_pitch, label_yaw)
        
        model.eval()
        image = torch.unsqueeze(image, 0)
        image = Variable(image).cuda(device)
        model.forward(image)

        gaze_pitch, gaze_yaw = model(image)

        # Binned predictions
        _, pitch_bpred = torch.max(gaze_pitch.data, 1)
        _, yaw_bpred = torch.max(gaze_yaw.data, 1)

        # Continuous predictions
        pitch_predicted = softmax(gaze_pitch)
        yaw_predicted = softmax(gaze_yaw)

        # mapping from binned (0 to 28) to angels (-42 to 42)
        idx_tensor = [idx for idx in range(28)]
        idx_tensor = torch.FloatTensor(idx_tensor).cuda(device)
        # 0-28, -> 0-84 -> 42

        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 42
        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 42

        pitch_predicted = pitch_predicted * np.pi / 180
        yaw_predicted = yaw_predicted * np.pi / 180

        print("Predictions: ", pitch_predicted, yaw_predicted)
        print("Groundtruth: ", label_pitch, label_yaw)

        image_np = image[0].cpu().detach().numpy()
        pitch_predicted = pitch_predicted[0].detach().numpy()
        yaw_predicted = yaw_predicted[0].detach().numpy()
        
        image_np = image_np.transpose(1, 2, 0)

        dx = -1000 * np.sin(pitch_predicted) * np.cos(yaw_predicted)
        dy = -1000 * np.sin(yaw_predicted)

        dx_label = -1000 * np.sin(label_pitch) * np.cos(label_yaw)
        dy_label = -1000 * np.sin(label_yaw)

        print(image_np.shape)
#        image_np = image_tensor.numpy()  # Convert to numpy array if not already
        # image_np = image_np.transpose(1, 2, 0)  # Change from CxHxW to HxWxC
        # image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        # image_np = (image_np * 255).astype(np.uint8)  # Assuming the tensor was in [0,1], scale to [0,255] and convert to uint8


        # image = cv2.arrowedLine(image_np, tuple(np.round(pos).astype(np.int32)),
        #     tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
        #     thickness, cv2.LINE_AA, tipLength=0.18)

        # # Display the image
        plt.imshow(image_np)
        plt.axis('off')  # Hide axis
        plt.arrow(244, 244, dx, dy, color='r')
        plt.arrow(244, 244, dx_label, dy_label, color='b')
        plt.show()



def eval_model(model, test_data_loader, device, softmax):
    idx_tensor = [idx for idx in range(28)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(device)

    with torch.no_grad():
        for images, labels, cont_labels in test_data_loader:
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