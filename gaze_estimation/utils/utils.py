from pathlib import Path
from config.args import parse_experiment_args
import os
import shutil
from config.config import Config
import torch
import torch.backends.cudnn as cudnn
import wandb
import random
import numpy as np

def get_experiment_path() -> Path:
    args = parse_experiment_args()
    experiment_path = args.experiment_path
    return Path(experiment_path)

def setup_device(config: Config) -> torch.device:
    device = torch.device("cuda:0" if config.train.use_gpu else "cpu")
    return device

def login_wandb():
    with open("config/wandb_key.txt", 'r') as file:
        key = file.read()
    
    wandb.login(key=key)

def set_randomness(config: Config):
    random_seed = config.train.seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
