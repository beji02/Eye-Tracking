from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple
from gui.Singleton import SingletonABC
from utils.utils import get_experiment_path, setup_device
from config.config import load_config, Config
from model.models import create_L2CS_model
import torch.nn as nn
import torch
from torch.autograd import Variable
from pathlib import Path
from data.data_transformation import create_data_transformations_for_resnet
from data.datasets import InferenceDataset
from torch.utils.data import DataLoader
from demo.models import CombinedModel

MODEL_EXPERIMENT_DIR = Path("/home/deiubejan/Thesis/MyWork/MPIIGaze/gaze_estimation/experiments/TestExperiment")

class IModel(ABC):
    @abstractmethod
    def forward(self, image: np.ndarray) -> tuple:
        pass

# class MockModel(IModel):
#     def forward(self, image: np.ndarray) -> tuple:
#         return np.array([0.1985, -0.0526])
    

class Model(IModel, SingletonABC):
    def _initialize(self) -> None:
        # model should be all the pipeline not only the gaze estimation model
        self._combined_model = CombinedModel()

    def forward(self, image: np.ndarray) -> tuple:
        return self._combined_model.forward(image)
        
    
    def _destruct(self) -> None:
        pass