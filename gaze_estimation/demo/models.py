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
import time
from demo.model_with_timer import ModelWithTimer, ITimer

class GazeEstimationModelWithResnet():
    def __init__(self, config: Config) -> None:
        self._config = config
        self._device = setup_device(config)
        self._model = create_L2CS_model(arch=self._config.model.backbone, bin_count=self._config.model.bins, device=self._device)

        self._model.load_state_dict(torch.load(self._config.experiment_path / "output" / "models" / "fold_10" / "model.pkl", map_location=self._device), strict=True)
        self._model.to(self._device)
        self._model.eval()

        self._softmax = nn.Softmax(dim=1).cuda(self._device)
        self._data_transformations = create_data_transformations_for_resnet()

        idx_tensor = [idx for idx in range(self._config.model.bins)]
        self._idx_tensor = torch.FloatTensor(idx_tensor).cuda(self._device)

        self._inference_total_time = 0
        self._no_inferences = 0

    def forward(self, image: np.ndarray) -> np.ndarray:
        image = image.transpose(1, 2, 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dataset = InferenceDataset(image, self._data_transformations)

        with torch.no_grad():
            images = torch.unsqueeze(dataset[0], dim=0)
            images = Variable(images).cuda(self._device)
            
            start_time = time.time()
            gaze_pitch, gaze_yaw = self._model(images)
            end_time = time.time()
            execution_time = end_time - start_time
            self._inference_total_time += execution_time
            self._no_inferences += 1
            # print(f"(BETTER MODEL) Inference average time: {self._inference_total_time / self._no_inferences}s") 
            # print(f"No inferences: {self._no_inferences}")

            pitch_predicted = self._softmax(gaze_pitch)
            yaw_predicted = self._softmax(gaze_yaw)

            # mapping from binned (0 to 28) to angels (-42 to 42)
            pitch_predicted = torch.sum(pitch_predicted * self._idx_tensor, 1).cpu() - 45 #* 3 - 42
            yaw_predicted = torch.sum(yaw_predicted * self._idx_tensor, 1).cpu() * - 45 #3 - 42

            pitch_predicted = pitch_predicted * np.pi / 180
            yaw_predicted = yaw_predicted * np.pi / 180
        
        return (pitch_predicted, yaw_predicted)

class GazeEstimationModelWithMobilenet():
    def __init__(self, config) -> None:
        self._config = config
        self._device = setup_device(self._config)
        self._model = MobileNet()
        self._model.load_state_dict(torch.load(self._config.experiment_path / "output" / "models" / "fold_10" / "model.pkl", map_location=self._device), strict=True)
        self._model.to(self._device)
        self._model.eval()

        self._softmax = nn.Softmax(dim=1).cuda(self._device)
        self._data_transformations = create_mobileNetV2_transformations()

        self._inference_total_time = 0
        self._no_inferences = 0

    def forward(self, image: np.ndarray) -> np.ndarray:
        image = image.transpose(1, 2, 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dataset = InferenceDataset(image, self._data_transformations)

        with torch.no_grad():
            images = torch.unsqueeze(dataset[0], dim=0)
            images = Variable(images).cuda(self._device)

            start_time = time.time()
            predictions = self._model(images)
            end_time = time.time()
            execution_time = end_time - start_time
            self._inference_total_time += execution_time
            self._no_inferences += 1
            # print(f"(BETTER MODEL) Inference average time: {self._inference_total_time / self._no_inferences}s") 
            # print(f"No inferences: {self._no_inferences}")

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

class CombinedModel(ITimer):
    def __init__(self, experiment_path):
        self._face_detector = FaceDetectionModel()

        self._config = load_config(experiment_path)
        if self._config.model.backbone.startswith("ResNet"):
            self._gaze_estimator = GazeEstimationModelWithResnet(self._config)
        elif self._config.model.backbone.startswith("MobileNet"):
            self._gaze_estimator = GazeEstimationModelWithMobilenet(self._config)

        self._face_detector = ModelWithTimer(self._face_detector)
        self._gaze_estimator = ModelWithTimer(self._gaze_estimator)
        # total_params = sum(p.numel() for p in self._gaze_estimator._model.parameters())
        # print(total_params)

    def get_fps(self):
        return self._gaze_estimator.get_fps()
    
    def get_fps_face_detector(self):
        return self._face_detector.get_fps()

    def forward(self, image: np.ndarray):
        cropped_face, eye_positions = self._face_detector.forward(image)
        gaze = None
        if cropped_face is not None:
            gaze = self._gaze_estimator.forward(cropped_face)
        if eye_positions is None:
            return None
        return eye_positions, gaze

        


        