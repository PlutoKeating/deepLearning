"""Optimization loop adapted from PyTorch Learn the Basics."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Same MLP structure as the Learn the Basics optimization tutorial.
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The model returns logits; CrossEntropyLoss will apply the needed softmax math.
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> None:
    size = len(dataloader.dataset)
    # train() enables training-time behavior for layers such as dropout or batch norm.
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        # The optimization loop follows: compute gradients, update parameters,
        # then clear gradients so the next batch starts clean.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_value = loss.item()
            current = (batch + 1) * len(x)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")


def test(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str,
) -> None:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # eval() and no_grad() make evaluation faster and avoid storing gradients.
    model.eval()
    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f} \n"
    )


def main() -> None:
    torch.manual_seed(0)

    print("=== 06 Optimization Loop ===")

    data_root = Path(__file__).resolve().parents[1] / "data"
    # The tutorial trains and evaluates on separate FashionMNIST splits.
    training_data = datasets.FashionMNIST(
        root=str(data_root),
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.FashionMNIST(
        root=str(data_root),
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    device = get_device()
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    # CrossEntropyLoss is the standard classification loss for class-index labels.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 1
    for t in range(epochs):
        print(f"\nEpoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")


if __name__ == "__main__":
    main()
