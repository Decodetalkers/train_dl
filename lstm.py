# It is a LSTM model
import torch
import torch.nn.functional as F
import time
import random
import pandas as pd
from utils.ttv import VOCABULARYL, TextToVector

torch.backends.cudnn.deterministic = True

RANDOM_SEED = 123
LEARNING_RATE = 0.005
BATCH_SIZE: int = 4
NUM_EPOCHS = 15
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 2

df = pd.read_csv("./misc/movie_data.csv")

DataLoader = TextToVector(dataset=df, batch_size=BATCH_SIZE)

for data, features in DataLoader:
    print(data)
    pass
