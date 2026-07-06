"""ResNet18 feature extractor on hymenoptera_data.

This script follows the official PyTorch transfer learning tutorial structure.
"""

from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


# You are encouraged to adjust the hyperparameters and observe how they affect the results.
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3
LAB_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = LAB_ROOT / "data" / "hymenoptera_data"
MISSING_DATA_MESSAGE = (
    "Please place hymenoptera_data under "
    "labs/lab02_cnn_transfer_learning/data/"
)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_dataset_exists(data_dir: Path) -> bool:
    required_dirs = [
        data_dir / "train" / "ants",
        data_dir / "train" / "bees",
        data_dir / "val" / "ants",
        data_dir / "val" / "bees",
    ]
    if not all(path.is_dir() for path in required_dirs):
        print(MISSING_DATA_MESSAGE)
        print(f"Expected dataset path: {data_dir}")
        return False
    return True


def build_resnet18() -> nn.Module:
    try:
        # DEFAULT weights load a ResNet18 pretrained on ImageNet.
        return models.resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception as exc:
        print(f"Warning: pretrained weights could not be loaded: {exc}")
        print("Falling back to weights=None.")
        return models.resnet18(weights=None)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    all_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return all_params, trainable_params


def train_model(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    dataset_sizes: dict[str, int],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_epochs: int,
) -> nn.Module:
    since = time.time()

    # The official transfer learning loop alternates train and val phases.
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Gradients are needed only during the training phase.
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    return model


def main() -> None:
    torch.manual_seed(0)
    print("=== 03 ResNet18 Feature Extractor ===")
    print(f"Dataset path: {DATA_DIR}")

    if not check_dataset_exists(DATA_DIR):
        return

    # Training uses random crop/flip augmentation; validation uses deterministic resize/crop.
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # ImageFolder maps train/ants, train/bees, val/ants, val/bees to class labels.
    image_datasets = {
        phase: datasets.ImageFolder(DATA_DIR / phase, data_transforms[phase])
        for phase in ["train", "val"]
    }
    dataloaders = {
        phase: DataLoader(
            image_datasets[phase],
            batch_size=BATCH_SIZE,
            shuffle=(phase == "train"),
        )
        for phase in ["train", "val"]
    }
    dataset_sizes = {phase: len(image_datasets[phase]) for phase in ["train", "val"]}
    class_names = image_datasets["train"].classes

    print(f"Class names: {class_names}")
    print(f"Train dataset size: {dataset_sizes['train']}")
    print(f"Val dataset size: {dataset_sizes['val']}")
    print(f"Batch size: {BATCH_SIZE}")

    device = get_device()
    print(f"Using {device} device")

    model_conv = build_resnet18()
    original_fc = model_conv.fc
    print("\nOriginal ResNet18 final layer:")
    print(original_fc)

    # Feature extractor mode freezes the pretrained convolutional backbone.
    for param in model_conv.parameters():
        param.requires_grad = False

    # Replace the ImageNet classifier with a two-class ants/bees classifier.
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)
    modified_fc = model_conv.fc
    print("Modified ResNet18 final layer:")
    print(modified_fc)

    all_params, trainable_params = count_parameters(model_conv)
    print(f"All parameter count: {all_params}")
    print(f"Trainable parameter count: {trainable_params}")

    model_conv = model_conv.to(device)
    criterion = nn.CrossEntropyLoss()
    # Only the new final layer is optimized because earlier layers are frozen.
    optimizer_conv = torch.optim.SGD(model_conv.fc.parameters(), lr=LEARNING_RATE)

    train_model(
        model_conv,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer_conv,
        device,
        EPOCHS,
    )
    print("Done!")


if __name__ == "__main__":
    main()
