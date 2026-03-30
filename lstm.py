# It is a LSTM model
import torch
import torch.nn.functional as F
import time
import random
import pandas as pd
from utils.ttv import VOCABULARYL,VOCABULARY_LEN, TextToVector, SENTIMENT_FEATURES

torch.backends.cudnn.deterministic = True

RANDOM_SEED = 123
LEARNING_RATE = 0.005
BATCH_SIZE: int = 30
NUM_EPOCHS = 15
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 2

df = pd.read_csv("./misc/movie_data.csv")

DATA_LOADER = TextToVector(dataset=df, batch_size=BATCH_SIZE)


class MyLstm(torch.nn.Module):
    embendding: torch.nn.Embedding
    rnn: torch.nn.LSTM
    rc: torch.nn.Linear

    def __init__(
        self, input_dim: int, embendding_dim: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()

        self.embedding = torch.nn.Embedding(input_dim, embendding_dim)
        self.rnn = torch.nn.LSTM(embendding_dim, hidden_dim)

        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    # NOTE: Size([num_features, batch_size])
    def forward(self, text: torch.Tensor):
        embedded = self.embedding(text)
        _output, (hidden, _cell) = self.rnn(embedded)

        hidden.squeeze_(0)

        output = self.fc(hidden)

        return output


torch.manual_seed(RANDOM_SEED)

model = MyLstm(
    input_dim=VOCABULARY_LEN,
    embendding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=SENTIMENT_FEATURES,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def compute_accuracy(model: MyLstm, data_loader: TextToVector, device: torch.device):
    with torch.no_grad():
        correct_pred, num_example = 0, 0
        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.float().to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100.0  # ty:ignore[unresolved-attribute]


start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, (data, labels) in enumerate(DATA_LOADER):
        data = data.to(DEVICE).transpose_(0, 1)
        logits = model(data)
        loss = F.cross_entropy(logits, labels.to(DEVICE))

        optimizer.zero_grad()
        optimizer.step()

        ### LOGGING
        if not batch_idx % 50:
            print(
                f"Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} | "
                f"Batch {batch_idx:03d}/{DATA_LOADER.max_len:03d} | "
                f"Loss: {loss:.4f}"
            )

    with torch.set_grad_enabled(False):
        print(
            f"training accuracy: "
            f"{compute_accuracy(model, DATA_LOADER, DEVICE):.2f}%"
            f"\nvalid accuracy: "
            f"{compute_accuracy(model, DATA_LOADER, DEVICE):.2f}%"
        )

    print(f"Time elapsed: {(time.time() - start_time) / 60:.2f} min")
