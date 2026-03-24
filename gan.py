import time
from collections.abc import Callable
from typing import Dict, List, Tuple
from pyparsing import Optional
import numpy as np
import torch
import torchvision
import random
import os
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utils import set_deterministic, set_all_seeds

#########################
## SETTINGS
#########################

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
RANDOM_SEED = 42
GENERATOR_LEARNING_RATE = 0.0002
DISCRIMINATOR_LEARNING_RATE = 0.0002

NUM_EPOCHS = 100
BATCH_SIZE = 128

# Channels means grey picture
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 28, 28, 1

# Base set
set_deterministic()
set_all_seeds(RANDOM_SEED)

custom_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)

# We do not need label, so we use tranining
train_dataset = datasets.MNIST(
    root="data", train=True, transform=custom_transforms, download=True
)

train_loader: DataLoader[datasets.MNIST] = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True
)

# Checking the dataset
for images, labels in train_loader:
    print("Image batch dimensions:", images.shape)
    print("Image label dimensions:", labels.shape)
    break


# We need two model to train in on class
class GAN(torch.nn.Module):
    image_height: int
    image_width: int
    color_channels: int

    # Generate the image
    generator: nn.Sequential

    def __init__(
        self,
        laten_dim: int = 100,
        image_height: int = 28,
        image_width: int = 28,
        color_channels: int = 1,
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.color_channels = color_channels

        self.generator = nn.Sequential(
            # Layer one
            nn.Linear(laten_dim, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            # Upper back to image
            nn.Linear(128, image_height * image_height * color_channels),
            nn.Tanh(),
        )

        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_height * image_width * color_channels, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            # Only one feature, in 0..1, is true or false
            nn.Linear(128, 1),
        )

    def generator_forward(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.flatten(z, start_dim=1)
        img: torch.Tensor = self.generator(z)
        # Normalize the data to image
        img: torch.Tensor = img.view(
            z.size(0), self.color_channels, self.image_height, self.image_width
        )
        return img

    def discriminator_forward(self, img: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.discriminator(img)
        return logits


model = GAN()
model.to(DEVICE)

optim_gen = torch.optim.Adam(
    model.generator.parameters(), betas=(0.5, 0.999), lr=GENERATOR_LEARNING_RATE
)

optim_discr = torch.optim.Adam(
    model.discriminator.parameters(), betas=(0.5, 0.999), lr=DISCRIMINATOR_LEARNING_RATE
)


def train_gan_v1(
    num_epochs: int,
    model: GAN,
    optimizer_gen: torch.optim.Adam,
    optimizer_discr: torch.optim.Adam,
    latent_dim: int,
    device: torch.device,
    train_loader: DataLoader,
    loss_fn: Callable[..., torch.Tensor] | None = None,
    logging_interval: int = 100,
    save_model: str | None = None,
) -> Dict[str, List[int | float | torch.Tensor]]:
    log_dict: Dict[str, List[int | float | torch.Tensor]] = {
        "train_generator_loss_per_batch": [],
        "train_discriminator_loss_per_batch": [],
        "train_discriminator_real_acc_per_batch": [],
        "train_discriminator_fake_acc_per_batch": [],
        "images_from_noise_per_epoch": [],
    }
    if loss_fn is None:
        loss_fn = F.binary_cross_entropy_with_logits
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    start_time = time.time()

    for epoch in range(num_epochs):
        # Mark now we need grade
        model.train()
        for batch_idx, (features, _) in enumerate(train_loader):
            batch_size = features.size(0)

            # real images
            real_images = features.to(device)
            real_labels = torch.ones(batch_size, device=device)  # real label = 1

            # generated (fake) images
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)

            # format NCHW
            fake_images = model.generator_forward(noise)
            fake_labels = torch.zeros(batch_size, device=device)  # fake label = 0
            flipped_fake_labels = real_labels  # here, fake label = 1

            # --------------------------
            # Train Discriminator
            # --------------------------

            optimizer_discr.zero_grad()

            # get discriminator loss on real images
            discr_pred_real = model.discriminator_forward(real_images).view(-1)

            # Nx1 -> N
            real_loss = loss_fn(discr_pred_real, real_labels)
            # real_loss.backward()

            # get discriminator loss on fake images
            discr_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)
            fake_loss = loss_fn(discr_pred_fake, fake_labels)
            # fake_loss.backward()

            # combined loss
            discr_loss = 0.5 * (real_loss + fake_loss)
            discr_loss.backward()

            optimizer_discr.step()

            # --------------------------
            # Train Generator
            # --------------------------

            optimizer_gen.zero_grad()

            # get discriminator loss on fake images with flipped labels
            discr_pred_fake = model.discriminator_forward(fake_images).view(-1)
            gener_loss = loss_fn(discr_pred_fake, flipped_fake_labels)
            gener_loss.backward()

            optimizer_gen.step()

            # --------------------------
            # Logging
            # --------------------------
            log_dict["train_generator_loss_per_batch"].append(gener_loss.item())
            log_dict["train_discriminator_loss_per_batch"].append(discr_loss.item())

            predicted_labels_real = torch.where(
                discr_pred_real.detach() > 0.0, 1.0, 0.0
            )
            predicted_labels_fake = torch.where(
                discr_pred_fake.detach() > 0.0, 1.0, 0.0
            )
            acc_real = (predicted_labels_real == real_labels).float().mean() * 100.0
            acc_fake = (predicted_labels_fake == fake_labels).float().mean() * 100.0
            log_dict["train_discriminator_real_acc_per_batch"].append(acc_real.item())
            log_dict["train_discriminator_fake_acc_per_batch"].append(acc_fake.item())

            if not batch_idx % logging_interval:
                print(
                    "Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f"
                    % (
                        epoch + 1,
                        num_epochs,
                        batch_idx,
                        len(train_loader),
                        gener_loss.item(),
                        discr_loss.item(),
                    )
                )

        ### Save images for evaluation
        with torch.no_grad():
            fake_images = model.generator_forward(fixed_noise).detach().cpu()
            log_dict["images_from_noise_per_epoch"].append(
                torchvision.utils.make_grid(fake_images, padding=2, normalize=True)
            )

        print("Time elapsed: %.2f min" % ((time.time() - start_time) / 60))

    print("Total Training Time: %.2f min" % ((time.time() - start_time) / 60))

    if save_model is not None:
        torch.save(model.state_dict(), save_model)

    return log_dict


log_dict = train_gan_v1(
    num_epochs=NUM_EPOCHS,
    model=model,
    optimizer_gen=optim_gen,
    optimizer_discr=optim_discr,
    latent_dim=100,
    device=DEVICE,
    train_loader=train_loader,
    logging_interval=100,
    save_model="gan_mnist_01.pt",
)


def plot_multiple_training_losses(
    losses_list: Tuple[List[torch.Tensor], List[torch.Tensor]],
    num_epochs: int,
    averaging_iterations: int = 100,
    custom_labels_list: List[str] = [],
):
    for i, _ in enumerate(losses_list):
        if not len(losses_list[i]) == len(losses_list[0]):
            raise ValueError(
                "All loss tensors need to have the same number of elements."
            )

    iter_per_epoch = len(losses_list[0]) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)

    for i, minibatch_loss_tensor in enumerate(losses_list):
        ax1.plot(
            range(len(minibatch_loss_tensor)),
            (minibatch_loss_tensor),
            label=f"Minibatch Loss{custom_labels_list[i]}",
        )
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Loss")

        ax1.plot(
            np.convolve(
                minibatch_loss_tensor,
                np.ones(
                    averaging_iterations,
                )
                / averaging_iterations,
                mode="valid",
            ),
            color="black",
        )

    if len(losses_list[0]) < 1000:
        num_losses = len(losses_list[0]) // 2
    else:
        num_losses = 1000
    maxes = [np.max(losses_list[i][num_losses:]) for i, _ in enumerate(losses_list)]
    ax1.set_ylim(0, np.max(maxes) * 1.5)
    ax1.legend()

    ###################
    # Set second x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs + 1))

    newpos = [e * iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(str(i) for i in newlabel[::10])

    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("outward", 45))
    ax2.set_xlabel("Epochs")
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()


plot_multiple_training_losses(
    losses_list=(
        log_dict["train_discriminator_loss_per_batch"],
        log_dict["train_generator_loss_per_batch"],
    ),  # ty:ignore[invalid-argument-type]
    num_epochs=NUM_EPOCHS,
    custom_labels_list=[" -- Discriminator", " -- Generator"],
)

##########################
### VISUALIZATION
##########################

for i in range(0, NUM_EPOCHS, 5):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(f"Generated images at epoch {i}")
    plt.imshow(np.transpose(log_dict["images_from_noise_per_epoch"][i], (1, 2, 0)))
    plt.show()


plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title(f"Generated images after last epoch")
plt.imshow(np.transpose(log_dict["images_from_noise_per_epoch"][-1], (1, 2, 0)))
plt.show()
