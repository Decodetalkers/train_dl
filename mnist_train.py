from turtle import forward
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed = 123
learning_rate = 0.1
num_epochs = 25
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
    linear: torch.nn.Linear

    def __init__(self, num_features: int, num_classes: int):
        super(SoftmaxRegission, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)
        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits: torch.Tensor = self.linear(x)
        probas = F.softmax(logits, dim=1)
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
        epoch_costs.append(avg_cost)
        accuracy = compute_accuracy(model, train_loader)
        print(
            "Epoch: %03d/%03d training accuracy: %.2f%%"
            % (epoch + 1, num_epochs, accuracy)
        )
        print("Time elapsed: %.2f min" % ((time.time() - start_time) / 60))

features, targets = next(iter(test_loader))

_, predictions = model.forward(features[:4].view(-1, 28 * 28).to(device))
predictions = torch.argmax(predictions, dim=1)
print("Predicted labels", predictions)
