"""Train a CNN on FashionMNIST with augmentation and dropout."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor


# You are encouraged to adjust the hyperparameters and observe how they affect the results.
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 1e-2
DROPOUT_P = 0.3
LAB_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = LAB_ROOT / "data"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class RegularizedCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # The convolution blocks match the baseline so changes are easier to compare.
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Dropout randomly disables activations during training to reduce overfitting.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=DROPOUT_P),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_P),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The output remains logits for 10 FashionMNIST classes.
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
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        # Same optimization loop as the baseline, so effects come from model/data changes.
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
    print("=== 02 CNN Augmentation and Regularization ===")

    # Random transforms are applied only to training images.
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(28, padding=4),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]
    )
    # Test data stays deterministic so validation is fair and repeatable.
    test_transform = ToTensor()

    print("Train transform:")
    print(train_transform)
    print(f"Dropout probability: {DROPOUT_P}")

    # Same dataset as the baseline, but train and test use different transforms.
    training_data = datasets.FashionMNIST(
        root=str(DATA_ROOT),
        train=True,
        download=True,
        transform=train_transform,
    )
    test_data = datasets.FashionMNIST(
        root=str(DATA_ROOT),
        train=False,
        download=True,
        transform=test_transform,
    )

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    device = get_device()
    print(f"Using {device} device")
    print(f"Epochs: {EPOCHS}")

    model = RegularizedCNN().to(device)
    # Printing the model shows where dropout was inserted.
    print("\nModel structure:")
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("Done!")


if __name__ == "__main__":
    main()
