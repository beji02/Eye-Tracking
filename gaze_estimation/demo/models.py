from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple
from gui.Singleton import SingletonABC
from utils.utils import get_experiment_path, setup_device
from config.config import load_config, Config
from model.models import create_L2CS_model, MyLoveMobileNet as MobileNet
import torch.nn as nn
import torch
from torch.autograd import Variable
from pathlib import Path
from data.data_transformation import create_data_transformations_for_resnet, create_mobileNetV2_transformations
from data.datasets import InferenceDataset
from torch.utils.data import DataLoader
from face_detection import RetinaFace
import cv2
import matplotlib.pyplot as plt


class GazeEstimationModelWithResnet():
    def __init__(self, experiment_path) -> None:
        self._experiment_path = experiment_path
        self._config = load_config(self._experiment_path)
        self._device = setup_device(self._config)
        self._model = create_L2CS_model(arch=self._config.model.backbone, bin_count=self._config.model.bins, device=self._device)

        self._model.load_state_dict(torch.load(self._experiment_path / "output" / "models" / "fold_10" / "model.pkl", map_location=self._device), strict=True)
        self._model.to(self._device)
        self._model.eval()

        self._softmax = nn.Softmax(dim=1).cuda(self._device)
        self._data_transformations = create_data_transformations_for_resnet()

        idx_tensor = [idx for idx in range(self._config.model.bins)]
        self._idx_tensor = torch.FloatTensor(idx_tensor).cuda(self._device)

    def forward(self, image: np.ndarray) -> np.ndarray:
        image = image.transpose(1, 2, 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dataset = InferenceDataset(image, self._data_transformations)

        with torch.no_grad():
            images = torch.unsqueeze(dataset[0], dim=0)
            images = Variable(images).cuda(self._device)
            
            gaze_pitch, gaze_yaw = self._model(images)

            pitch_predicted = self._softmax(gaze_pitch)
            yaw_predicted = self._softmax(gaze_yaw)

            # mapping from binned (0 to 28) to angels (-42 to 42)
            pitch_predicted = torch.sum(pitch_predicted * self._idx_tensor, 1).cpu() - 45 #* 3 - 42
            yaw_predicted = torch.sum(yaw_predicted * self._idx_tensor, 1).cpu() * - 45 #3 - 42

            pitch_predicted = pitch_predicted * np.pi / 180
            yaw_predicted = yaw_predicted * np.pi / 180
        
        return (pitch_predicted, yaw_predicted)
    


class GazeEstimationModelWithMobilenet():
    def __init__(self, experiment_path) -> None:
        self._experiment_path = experiment_path
        self._config = load_config(self._experiment_path)
        self._device = setup_device(self._config)
        self._model = MobileNet()
        self._model.load_state_dict(torch.load(self._experiment_path / "output" / "models" / "fold_10" / "model.pkl", map_location=self._device), strict=True)
        self._model.to(self._device)
        self._model.eval()

        self._softmax = nn.Softmax(dim=1).cuda(self._device)
        self._data_transformations = create_mobileNetV2_transformations()

    def forward(self, image: np.ndarray) -> np.ndarray:
        image = image.transpose(1, 2, 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dataset = InferenceDataset(image, self._data_transformations)

        with torch.no_grad():
            images = torch.unsqueeze(dataset[0], dim=0)
            images = Variable(images).cuda(self._device)
            predictions = self._model(images)
            
            pred_pitch = predictions[:, 0].cpu() * np.pi / 180
            pred_yaw = predictions[:, 1].cpu() * np.pi / 180

        return (pred_pitch, pred_yaw)


class FaceDetectionModel():
    def __init__(self):
        self._face_detector = RetinaFace(gpu_id=0)
    
    def forward(self, image: np.ndarray) -> Tuple:
        image = image.copy().transpose(1, 2, 0)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        detected_faces = self._face_detector(image)
        cropped_face = None
        eye_positions = None

        for box, landmarks, score in detected_faces:
            if score > 0.9:
                x_min = max(0, int(box[0]))
                y_min = max(0, int(box[1]))
                x_max = int(box[2])
                y_max = int(box[3])

                cropped_image = image[y_min:y_max, x_min:x_max]
                resized_image = cv2.resize(cropped_image, (224, 224))
                cropped_face = resized_image
                eye_positions = landmarks[:2]
        
        if cropped_face is not None:
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            cropped_face = cropped_face.transpose(2, 0, 1)

        return cropped_face, eye_positions

class CombinedModel():
    def __init__(self, experiment_path):
        self._face_detector = FaceDetectionModel()
        self._gaze_estimator = GazeEstimationModelWithMobilenet(experiment_path)

    def forward(self, image: np.ndarray):
        cropped_face, eye_positions = self._face_detector.forward(image)
        gaze = None
        if cropped_face is not None:
            gaze = self._gaze_estimator.forward(cropped_face)
        return eye_positions, gaze

        


        