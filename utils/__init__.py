from typing import Callable, Optional
import torch
import os
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets

BATCH_SIZE = 128


# make output the same
def set_deterministic():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


def set_all_seeds(seed: int):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_dataset_fn(custom_transforms: Optional[Callable] = None) -> datasets.MNIST:
    # We do not need label, so we use tranining
    return datasets.MNIST(
        root="data", train=True, transform=custom_transforms, download=True
    )


def test_dataset_fn(custom_transforms: Optional[Callable] = None) -> datasets.MNIST:
    # We do not need label, so we use tranining
    return datasets.MNIST(root="data", train=False, transform=custom_transforms)
