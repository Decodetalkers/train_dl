from torch.utils.data import DataLoader
import torch
import torchvision
import time
import numpy as np
from torchvision import transforms
from torchvision import datasets
from utils import train_dataset_fn, test_dataset_fn

# NOTE: seed
RADOM_SEED = 123

LEARNING_RATE = 0.01

# NOTE: every batch, the iter size
BATCH_SIZE = 128

# NOTE: EPOCH count
NUM_EPOCHS = 10

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGE_CLASSES = 10

# The size of the picture
NUM_FUTURES = 28 * 28

train_dataset = train_dataset_fn(transforms.ToTensor())

test_dataset = test_dataset_fn(transforms.ToTensor())

train_loader: DataLoader[datasets.MNIST] = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_loader: DataLoader[datasets.MNIST] = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True
)

for images, labels in train_loader:
    print("Image batch dimensions:", images.shape)
    print("Image label dimensions:", labels.shape)
    break


# x -> conv -> batch_norm -> relu --> conv batch_norm --> relu -> linear
class ConvNetV1(torch.nn.Module):
    block_1: torch.nn.Sequential
    block_2: torch.nn.Sequential
    linear_1: torch.nn.Linear

    def __init__(self, num_classes):
        super(ConvNetV1, self).__init__()

        # I do not know math that well, but this flow kept the length
        # o = [(i + 2p - k) / s] + 1
        self.block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=4,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0,
            ),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                in_channels=4,
                out_channels=1,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            torch.nn.BatchNorm2d(1),
        )

        # I do not know math that well, but this flow kept the length
        # o = [(i + 2p - k) / s] + 1
        self.block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=4,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0,
            ),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                in_channels=4,
                out_channels=1,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            torch.nn.BatchNorm2d(1),
        )

        self.linear_1 = torch.nn.Linear(IMAGE_CLASSES, num_classes)

    def forward(self, x: torch.Tensor):

        shortcut = x
        x = self.block_1(x)
        x = torch.nn.functional.relu(x + shortcut)

        shortcut = x
        x = self.block_2(x)
        x = torch.nn.functional.relu(x + shortcut)

        # After conved, we have the data which has strong features
        logits = self.linear_1(x.view(-1, IMAGE_CLASSES))

        return logits
