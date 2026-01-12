from turtle import forward
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import time

import matplotlib
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed = 123
learning_rate = 0.1
num_epochs = 100
batch_size = 256

num_features = 784
num_classes = 10

train_dataset = datasets.MNIST(
    root="data", train=True, transform=transforms.ToTensor(), download=True
)

test_dataset = datasets.MNIST(root="data", train=False, transform=transforms.ToTensor())

train_loader: DataLoader[datasets.MNIST] = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

test_loader: DataLoader[datasets.MNIST] = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=True
)

for images, labels in train_loader:
    print("image batch dimensions:", images.shape)
    print("image label dimensions:", labels.shape)
    break


class SoftmaxRegission(torch.nn.Module):
    seq: torch.nn.Sequential
    act: torch.nn.Softmax

    def __init__(self, num_features: int, num_classes: int, hidden_layers=100):
        super(SoftmaxRegission, self).__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_layers),
            torch.nn.Softmax(dim=1),
            torch.nn.Linear(hidden_layers, num_classes),
        ).to(device)
        self.act = torch.nn.Softmax(dim=1).to(device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits: torch.Tensor = self.seq(x)
        probas = self.act(logits)
        return logits, probas


model = SoftmaxRegission(num_features=num_features, num_classes=num_classes)

model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
torch.manual_seed(random_seed)


def compute_accuracy(
    model: SoftmaxRegission, data_loader: DataLoader[datasets.MNIST]
) -> float:
    correct_pred, num_examples = torch.tensor(0, dtype=torch.int64).to(device), 0

    for features, targets in data_loader:
        features: torch.Tensor = features.view(-1, 28 * 28).to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)

        sum = (predicted_labels == targets).sum()
        correct_pred += sum

    return correct_pred.float().item() / num_examples * 100


start_time = time.time()
epoch_costs: list[torch.Tensor] = []

for epoch in range(num_epochs):
    avg_cost: torch.Tensor = torch.tensor(0.0, dtype=torch.float).to(device)
    for batch_idx, (features, targets) in enumerate(train_loader):
        features = features.view(-1, 28 * 28).to(device)
        targets = targets.to(device)
        logits, probas = model(features)

        cost = F.cross_entropy(logits, targets).to(device)
        optimizer.zero_grad()

        # Tensor has a callback to upgrade the grad, finally this will change the linear in model
        cost.backward()
        avg_cost += cost

        optimizer.step()
        ### LOGGING
        if not batch_idx % 50:
            print(
                "Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f"
                % (
                    epoch + 1,
                    num_epochs,
                    batch_idx,
                    len(train_dataset) // batch_size,
                    cost.item(),
                )
            )
    with torch.set_grad_enabled(False):
        avg_cost = avg_cost / len(train_dataset)
        epoch_costs.append(avg_cost.cpu())
        accuracy = compute_accuracy(model, train_loader)
        print(
            "Epoch: %03d/%03d training accuracy: %.2f%%"
            % (epoch + 1, num_epochs, accuracy)
        )
        print("Time elapsed: %.2f min" % ((time.time() - start_time) / 60))


plt.plot(epoch_costs)
plt.ylabel('Avg Cross Entropy Loss\n(approximated by averaging over minibatches)')
plt.xlabel('Epoch')
plt.show()

accuracy = compute_accuracy(model, test_loader)

print(f"Test accuracy: {accuracy:.2f}")

features, targets = next(iter(test_loader))

_, predictions = model.forward(features[:4].view(-1, 28 * 28).to(device))
predictions = torch.argmax(predictions, dim=1)
print(f"Predicted labels: {predictions}, targets:{targets[:4]}")
