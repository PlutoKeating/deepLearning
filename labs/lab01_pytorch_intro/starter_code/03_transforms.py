"""Transform examples adapted from PyTorch Learn the Basics: Transforms."""

from pathlib import Path

import torch
from torchvision import datasets
from torchvision.transforms import Lambda, ToTensor


def main() -> None:
    print("=== 03 Transforms ===")

    data_root = Path(__file__).resolve().parents[1] / "data"

    # transform changes the image; target_transform changes the label.
    ds = datasets.FashionMNIST(
        root=str(data_root),
        train=True,
        download=True,
        # ToTensor converts PIL images to float tensors in [0, 1].
        transform=ToTensor(),
        # The Lambda transform mirrors the basics tutorial one-hot label example.
        target_transform=Lambda(
            lambda y: torch.zeros(10, dtype=torch.float).scatter_(
                0, torch.tensor(y), value=1
            )
        ),
    )

    image, label = ds[0]

    # The printed shape and dtype confirm the image transform was applied.
    print(f"Transformed image shape: {image.shape}")
    print(f"Transformed image dtype: {image.dtype}")
    print("Transformed one-hot label:")
    print(label)
    # A valid one-hot label has one 1 and the rest 0.
    print(f"One-hot label sum: {label.sum().item()}")
    print(f"Class index from one-hot label: {label.argmax().item()}")


if __name__ == "__main__":
    main()
