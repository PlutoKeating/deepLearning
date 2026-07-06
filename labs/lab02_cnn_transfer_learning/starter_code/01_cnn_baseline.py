"""Train a small CNN baseline on FashionMNIST."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


BATCH_SIZE = 64
EPOCHS = 1
LEARNING_RATE = 1e-2
LAB_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = LAB_ROOT / "data"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # The baseline CNN learns local image features with convolution blocks.
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            # Pooling reduces 28x28 -> 14x14, then 14x14 -> 7x7.
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # The classifier flattens feature maps and outputs 10 FashionMNIST logits.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass: feature extractor first, classifier second.
        x = self.features(x)
        logits = self.classifier(x)
        return logits


def train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> None:
    size = len(dataloader.dataset)
    # train() enables gradient tracking and training-time layer behavior.
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagate, update parameters, then clear gradients for the next batch.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 200 == 0:
            current = (batch + 1) * len(x)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")


def test(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str,
) -> tuple[float, float]:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # Evaluation uses no_grad because we only measure loss and accuracy.
    model.eval()
    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    accuracy = correct / size
    print(
        f"Test Error: Accuracy: {(100 * accuracy):>0.1f}%, "
        f"Avg loss: {test_loss:>8f}"
    )
    return accuracy, test_loss


def main() -> None:
    torch.manual_seed(0)
    print("=== 01 CNN Baseline on FashionMNIST ===")

    # ToTensor is enough for the baseline; augmentation is added in the next script.
    training_data = datasets.FashionMNIST(
        root=str(DATA_ROOT),
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.FashionMNIST(
        root=str(DATA_ROOT),
        train=False,
        download=True,
        transform=ToTensor(),
    )

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    device = get_device()
    print(f"Using {device} device")
    print(f"Epochs: {EPOCHS}")

    model = SmallCNN().to(device)
    # CrossEntropyLoss consumes logits and class-index labels.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")


if __name__ == "__main__":
    main()
