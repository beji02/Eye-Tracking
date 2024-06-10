from pathlib import Path
from pydantic import BaseModel
import json


class TrainConfig(BaseModel):
    num_epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    use_gpu: bool
    seed: int
    is_pipeline_test: bool

class ModelConfig(BaseModel):
    backbone: str
    bins: int

class DataConfig(BaseModel):
    folding_strategy: int
    dataset_name: str
    dataset_dir: Path

class Config(BaseModel):
    experiment_path: Path
    data: DataConfig
    model: ModelConfig
    train: TrainConfig

# "models: n, train: n-1, test: 1" -> 0
# "models: 1, train: n-1, test: 1" -> 1
# "models: 1, train: n-3, test: 3" -> 2

def load_config(experiment_path: Path) -> Config:
    config_path = experiment_path  / "config.json"
    with open(config_path, 'r') as file:
        config_data = json.load(file)
    return Config(experiment_path=experiment_path, **config_data)