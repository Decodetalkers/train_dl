import torch
import torchvision

# NOTE: seed
RADOM_SEED = 123

# NOTE: every batch, the iter size
BATCH_SIZE = 256

# NOTE: EPOCH count
NUM_EPOCHS = 5

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MulitlayerPerceptron(torch.nn.Module):
    # TODO:
    def __init___(self):
        pass
