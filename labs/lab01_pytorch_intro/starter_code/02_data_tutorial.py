"""FashionMNIST DataLoader example from PyTorch Learn the Basics."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def main() -> None:
    # The seed keeps the shuffled DataLoader batch reproducible for screenshots.
    torch.manual_seed(0)

    print("=== 02 Datasets and DataLoaders ===")

    data_root = Path(__file__).resolve().parents[1] / "data"

    # FashionMNIST is the same built-in dataset used in PyTorch Learn the Basics.
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

    # Dataset stores samples; DataLoader groups them into mini-batches.
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    print(f"Training dataset size: {len(training_data)}")
    print(f"Test dataset size: {len(test_data)}")
    print(f"Class names: {training_data.classes}")

    # One dataset item is an image tensor paired with a class label index.
    sample_img, sample_label = training_data[0]
    print(f"One sample image shape: {sample_img.shape}")
    print(f"One sample label index: {sample_label}")
    print(f"One sample label name: {training_data.classes[sample_label]}")

    # Iterating over a DataLoader returns a batch shaped [batch, channel, height, width].
    train_features, train_labels = next(iter(train_dataloader))
    print("\nOne training batch:")
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    test_features, test_labels = next(iter(test_dataloader))
    print("\nOne test batch:")
    print(f"Feature batch shape: {test_features.size()}")
    print(f"Labels batch shape: {test_labels.size()}")


if __name__ == "__main__":
    main()
