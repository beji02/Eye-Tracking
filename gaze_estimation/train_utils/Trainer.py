import torch
from torch.utils.data import DataLoader
from config.config import Config
from abc import ABC, abstractmethod
import logging

class Trainer:
    def __init__(
        self,
        config: Config,
        model: torch.nn.Module,
        test_dataloader: DataLoader,
        train_dataloader: DataLoader
    ):
        self._config = config
        self._model = model
        self._test_dataloader = test_dataloader
        self._train_dataloader = train_dataloader

    @abstractmethod
    def evaluate_on_test(self, num_epochs_trained: int):
        self._model.eval()
        logging.info(f"Starting evaluation on test")
        sample_counter = 0

        with torch.no_grad():
            for i, batch_data in enumerate(self._test_dataloader, start=1):
                sample_counter += batch_data.size(0)
                

                


    @abstractmethod
    def train_one_epoch(self):
        pass

    def evaluate_on_test(self):


