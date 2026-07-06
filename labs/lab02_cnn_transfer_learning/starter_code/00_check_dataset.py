"""Check datasets for Lab 02."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor


LAB_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = LAB_ROOT / "data"
HYMENOPTERA_ROOT = DATA_ROOT / "hymenoptera_data"
MISSING_DATA_MESSAGE = (
    "Please place hymenoptera_data under "
    "labs/lab02_cnn_transfer_learning/data/"
)


def check_fashionmnist() -> None:
    print("=== FashionMNIST Check ===")
    try:
        # FashionMNIST is used for the CNN baseline before transfer learning.
        train_data = datasets.FashionMNIST(
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
        print(f"FashionMNIST train size: {len(train_data)}")
        print(f"FashionMNIST test size: {len(test_data)}")

        # A small batch confirms the image and label tensor shapes.
        loader = DataLoader(train_data, batch_size=8, shuffle=True)
        images, labels = next(iter(loader))
        print(f"FashionMNIST one batch image shape: {images.shape}")
        print(f"FashionMNIST one batch label shape: {labels.shape}")
    except Exception as exc:
        print(f"FashionMNIST check failed: {exc}")


def check_hymenoptera() -> None:
    print("\n=== hymenoptera_data Check ===")
    # ImageFolder requires one subfolder per class inside each split.
    required_dirs = [
        HYMENOPTERA_ROOT / "train" / "ants",
        HYMENOPTERA_ROOT / "train" / "bees",
        HYMENOPTERA_ROOT / "val" / "ants",
        HYMENOPTERA_ROOT / "val" / "bees",
    ]
    if not all(path.is_dir() for path in required_dirs):
        print(MISSING_DATA_MESSAGE)
        return

    data_transform = transforms.Compose(
        [
            # ResNet-style transfer learning expects fixed-size RGB images.
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    # ImageFolder reads class names from folder names such as ants and bees.
    train_data = datasets.ImageFolder(
        str(HYMENOPTERA_ROOT / "train"),
        transform=data_transform,
    )
    val_data = datasets.ImageFolder(
        str(HYMENOPTERA_ROOT / "val"),
        transform=data_transform,
    )

    print(f"hymenoptera_data path: {HYMENOPTERA_ROOT}")
    print(f"hymenoptera train class names: {train_data.classes}")
    print(f"hymenoptera val class names: {val_data.classes}")
    print(f"hymenoptera train images: {len(train_data)}")
    print(f"hymenoptera val images: {len(val_data)}")

    # The batch should have shape [batch, 3, 224, 224] after the transform.
    loader = DataLoader(train_data, batch_size=4, shuffle=True)
    images, labels = next(iter(loader))
    print(f"hymenoptera one batch image shape: {images.shape}")
    print(f"hymenoptera one batch label shape: {labels.shape}")


def main() -> None:
    torch.manual_seed(0)
    print("=== 00 Check Dataset ===")
    print(f"Lab data root: {DATA_ROOT}")
    check_fashionmnist()
    check_hymenoptera()


if __name__ == "__main__":
    main()
