"""Integrated Lab 02 experiment on hymenoptera_data.

Complete the TODO blocks, then run:
    python 04_integrated_experiment.py
"""

from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


# =========================
# Student Configuration Area
# You are encouraged to adjust the hyperparameters and observe how they affect the results.
# =========================
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE_CNN = 1e-3
LEARNING_RATE_FEATURE_EXTRACTOR = 1e-3
LEARNING_RATE_FINETUNE = 1e-4
NUM_CLASSES = 2
MODEL_TO_SAVE = "best"


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


def count_parameters(model: nn.Module) -> tuple[int, int]:
    all_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return all_params, trainable_params


def train_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    dataset_size = len(dataloader.dataset)
    # One train epoch updates parameters and accumulates loss/accuracy totals.
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size
    return epoch_loss, epoch_acc


def evaluate(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str,
) -> tuple[float, float]:
    dataset_size = len(dataloader.dataset)
    # Validation mirrors training metrics but does not update weights.
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size
    return epoch_loss, epoch_acc


def train_model(
    name: str,
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[nn.Module, float, float]:
    model = model.to(device)
    since = time.time()
    # Keep the best validation metric for final model comparison.
    best_val_acc = 0.0
    best_val_loss = 0.0

    print(f"\nTraining {name}")
    print("-" * 20)
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(
            dataloaders["train"],
            model,
            loss_fn,
            optimizer,
            device,
        )
        val_loss, val_acc = evaluate(dataloaders["val"], model, loss_fn, device)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss

        print(f"train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    elapsed = time.time() - since
    print(f"{name} training time: {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    return model, best_val_acc, best_val_loss


def build_resnet18() -> nn.Module:
    try:
        # Use pretrained ResNet18 when weights are available locally or online.
        return models.resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception as exc:
        print(f"Warning: pretrained weights could not be loaded: {exc}")
        print("Falling back to weights=None.")
        return models.resnet18(weights=None)


def save_model(model: nn.Module, save_path: Path) -> None:
    torch.save(model.state_dict(), save_path)


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


def create_transforms() -> dict[str, transforms.Compose]:
    # Match the official transfer learning train/val preprocessing pattern.
    # TODO 1: Complete train and val transforms.
    # Training transform:
    # - Apply RandomResizedCrop(224) to introduce scale and position variation.
    # - Apply RandomHorizontalFlip() for left-right augmentation.
    # - Convert the image to a tensor.
    # - Normalize using ImageNet mean and standard deviation.

    # Validation transform:
    # - Resize the image to 256 on the shorter side.
    # - Apply CenterCrop(224) for a consistent input size.
    # - Convert the image to a tensor.
    # - Normalize using ImageNet mean and standard deviation.
    raise NotImplementedError("Task 1 is not complete: create train and val transforms.")


def create_datasets_and_loaders(
    data_dir: Path,
    data_transforms: dict[str, transforms.Compose],
) -> tuple[dict[str, datasets.ImageFolder], dict[str, DataLoader], dict[str, int], list[str]]:
    # ImageFolder should read class labels from the train and val subfolders.
    # TODO 2: Complete ImageFolder datasets and DataLoaders.
    # Required:
    # - use train and val splits
    # - use BATCH_SIZE
    # - shuffle only the train DataLoader
    raise NotImplementedError("Task 2 is not complete: create datasets and DataLoaders.")


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # This baseline is trained from scratch for comparison with pretrained ResNet18.
        # TODO 3: Complete SmallCNN for RGB 224x224 input.
        # Required:
        # - input channels should be 3
        # - output classes should be NUM_CLASSES
        # - keep the model lightweight
        raise NotImplementedError("Task 3 is not complete: define SmallCNN.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return logits for NUM_CLASSES classes.
        raise NotImplementedError("Task 3 is not complete: define SmallCNN forward pass.")


def setup_feature_extractor() -> tuple[nn.Module, nn.Module, nn.Module, int]:
    # Feature extractor mode freezes the pretrained backbone and trains only the head.
    # TODO 4: Complete ResNet18 feature extractor setup.
    # Required:
    # - build ResNet18 with build_resnet18()
    # - save original final layer
    # - freeze all backbone parameters
    # - replace fc with nn.Linear(num_ftrs, NUM_CLASSES)
    # - return model, original_fc, modified_fc, trainable parameter count
    raise NotImplementedError("Task 4 is not complete: set up feature extractor.")


def setup_partial_finetuning() -> tuple[nn.Module, nn.Module, nn.Module, int]:
    # Partial fine-tuning keeps early features fixed and adapts later layers.
    # TODO 5: Complete ResNet18 partial fine-tuning setup.
    # Required:
    # - build ResNet18 with build_resnet18()
    # - save original final layer
    # - freeze early layers
    # - unfreeze layer4 and fc
    # - replace fc with nn.Linear(num_ftrs, NUM_CLASSES)
    # - return model, original_fc, modified_fc, trainable parameter count
    raise NotImplementedError("Task 5 is not complete: set up partial fine-tuning.")


def main() -> None:
    torch.manual_seed(0)

    print("=== 04 Integrated Experiment ===")
    print(f"Dataset path: {DATA_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate CNN: {LEARNING_RATE_CNN}")
    print(f"Learning rate feature extractor: {LEARNING_RATE_FEATURE_EXTRACTOR}")
    print(f"Learning rate fine-tuning: {LEARNING_RATE_FINETUNE}")
    print(f"Model to save: {MODEL_TO_SAVE}")

    if not check_dataset_exists(DATA_DIR):
        return

    device = get_device()
    print(f"Using {device} device")

    data_transforms = create_transforms()
    image_datasets, dataloaders, dataset_sizes, class_names = create_datasets_and_loaders(
        DATA_DIR,
        data_transforms,
    )
    print(f"Class names: {class_names}")
    print(f"Train dataset size: {dataset_sizes['train']}")
    print(f"Val dataset size: {dataset_sizes['val']}")

    loss_fn = nn.CrossEntropyLoss()

    # First comparison model: a lightweight CNN trained from scratch.
    # TODO 6: Train and validate SmallCNN.
    # small_cnn = SmallCNN()
    # optimizer_cnn = ...
    # small_cnn, small_cnn_acc, small_cnn_loss = train_model(...)
    raise NotImplementedError("Task 6 is not complete: train and validate SmallCNN.")

    feature_model, feature_original_fc, feature_modified_fc, feature_trainable = (
        setup_feature_extractor()
    )
    print("\nFeature extractor ResNet18 final layers")
    print("Original ResNet18 final layer:")
    print(feature_original_fc)
    print("Modified ResNet18 final layer:")
    print(feature_modified_fc)
    print(f"Feature extractor trainable parameter count: {feature_trainable}")

    # Second comparison model: frozen ResNet18 backbone plus new fc layer.
    # TODO 7: Train and validate ResNet18 feature extractor.
    # optimizer_feature = ...
    # feature_model, feature_acc, feature_loss = train_model(...)
    raise NotImplementedError(
        "Task 7 is not complete: train and validate ResNet18 feature extractor."
    )

    finetune_model, finetune_original_fc, finetune_modified_fc, finetune_trainable = (
        setup_partial_finetuning()
    )
    print("\nPartial fine-tuning ResNet18 final layers")
    print("Original ResNet18 final layer:")
    print(finetune_original_fc)
    print("Modified ResNet18 final layer:")
    print(finetune_modified_fc)
    print(f"Partial fine-tuning trainable parameter count: {finetune_trainable}")

    # Third comparison model: partly unfrozen ResNet18 for limited adaptation.
    # TODO 8: Train and validate ResNet18 partial fine-tuning.
    # optimizer_finetune = ...
    # finetune_model, finetune_acc, finetune_loss = train_model(...)
    raise NotImplementedError(
        "Task 8 is not complete: train and validate ResNet18 partial fine-tuning."
    )

    print("\nFinal comparison")
    print("-" * 20)
    print(f"SmallCNN validation accuracy: {100 * small_cnn_acc:.1f}%")
    print(f"ResNet18 feature extractor validation accuracy: {100 * feature_acc:.1f}%")
    print(f"ResNet18 partial fine-tuning validation accuracy: {100 * finetune_acc:.1f}%")

    # Compare the three validation accuracies before choosing what to save.
    # TODO 9: Compare validation accuracies and select the best model.
    # Required:
    # - support MODEL_TO_SAVE = "best", "cnn", "feature_extractor", or "finetune"
    # - set selected_name and selected_model
    raise NotImplementedError("Task 9 is not complete: compare and select a model.")

    # Save state_dict weights only, following the PyTorch save/load tutorial.
    # TODO 10: Save the selected model.
    # save_path = ...
    # save_model(selected_model, save_path)
    raise NotImplementedError("Task 10 is not complete: save the selected model.")

    print(f"Selected saved model: {selected_name}")
    print(f"Saved model path: {save_path}")
    print("Done!")


if __name__ == "__main__":
    main()
