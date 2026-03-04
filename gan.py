import time
from typing import Callable
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


def set_deterministic():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#########################
## SETTINGS
#########################

DEVICE = torch.device("cpu")

# Hyperparameters
RANDOM_SEED = 42
GENERATOR_LEARNING_RATE = 0.0002
DISCRIMINATOR_LEARNING_RATE = 0.0002

NUM_EPOCHS = 100
BATCH_SIZE = 128

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
):
    log_dict = {
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
            noise = torch.randn(
                batch_size, latent_dim, 1, 1, device=device
            )

            # format NCHW
            fake_images = model.generator_forward(noise)
            fake_labels = torch.zeros(batch_size, device=device)  # fake label = 0
            flipped_fake_labels = real_labels  # here, fake label = 1

            # --------------------------
            # Train Discriminator
            # --------------------------

            optimizer_discr.zero_grad()

            # get discriminator loss on real images
            discr_pred_real = model.discriminator_forward(real_images).view(
                -1
            )

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
