"""Integrated FashionMNIST experiment for Lab 01.

Complete the six TODO blocks, then run:
    python 09_integrated_experiment.py
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.models import ResNet18_Weights
from torchvision.transforms import ToTensor


# =========================
# Student Configuration Area
# You are encouraged to adjust the hyperparameters and observe how they affect the results.
# =========================
BATCH_SIZE = 64
HIDDEN_DIM = 128
EPOCHS = 2
LEARNING_RATE = 1e-2
TRANSFER_NUM_CLASSES = 5
MODEL_TO_SAVE = "cnn"


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
    # This helper keeps the same training-loop pattern as the walkthrough scripts.
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        # Clear gradients after each update so batches do not accumulate gradients.
        optimizer.zero_grad()

        if batch % 200 == 0:
            loss_value = loss.item()
            current = (batch + 1) * len(x)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")


def test(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str,
) -> tuple[float, float]:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # Evaluation reports accuracy and average loss without tracking gradients.
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
        f"Test Result: Accuracy: {(100 * accuracy):>0.1f}%, "
        f"Avg loss: {test_loss:>8f}"
    )
    return accuracy, test_loss


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_dataloaders(data_root: Path) -> tuple[DataLoader, DataLoader]:
    # Students reuse the official FashionMNIST Dataset + DataLoader workflow here.
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

    # TODO 1: Create train and test DataLoaders with BATCH_SIZE.
    # train_dataloader = ...
    # test_dataloader = ...
    raise NotImplementedError("Task 1 is not complete: create the DataLoaders.")

    return train_dataloader, test_dataloader


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Build layers in __init__; keep forward() focused on data flow.
        # TODO 2: Complete an MLP model using HIDDEN_DIM.
        # Required idea:
        # - flatten 28x28 FashionMNIST images
        # - use HIDDEN_DIM in at least one hidden Linear layer
        # - output 10 classes
        raise NotImplementedError("Task 2 is not complete: define the MLP layers.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return the logits from your MLP.
        raise NotImplementedError("Task 2 is not complete: define the MLP forward pass.")


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # A CNN baseline should separate feature extraction from classification.
        # TODO 3: Complete a small CNN for FashionMNIST.
        # Required idea:
        # - input shape is [batch_size, 1, 28, 28]
        # - use Conv2d, ReLU, and a classifier ending in 10 classes
        raise NotImplementedError("Task 3 is not complete: define the SmallCNN layers.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return the logits from your CNN.
        raise NotImplementedError("Task 3 is not complete: define the SmallCNN forward pass.")


def run_model_experiment(
    name: str,
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> tuple[nn.Module, float, float]:
    model = model.to(device)
    # Each model gets its own optimizer so the comparison is independent.
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining {name}")
    print("-------------------------------")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}")
        train(train_dataloader, model, loss_fn, optimizer, device)

    accuracy, avg_loss = test(test_dataloader, model, loss_fn, device)
    return model, accuracy, avg_loss


def main() -> None:
    torch.manual_seed(0)

    print("=== 09 Integrated Experiment ===")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Hidden dim: {HIDDEN_DIM}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Transfer learning output classes: {TRANSFER_NUM_CLASSES}")
    print(f"Requested model to save: {MODEL_TO_SAVE}")

    device = get_device()
    print(f"Using {device} device")

    data_root = Path(__file__).resolve().parents[1] / "data"
    train_dataloader, test_dataloader = create_dataloaders(data_root)

    loss_fn = nn.CrossEntropyLoss()

    # Integrated task: run both models with the same data, loss, and epoch count.
    # TODO 4: Train and test both MLP and CNN, then compare their test accuracy.
    # mlp_model, mlp_accuracy, mlp_avg_loss = run_model_experiment(...)
    # cnn_model, cnn_accuracy, cnn_avg_loss = run_model_experiment(...)
    # Print which model has higher test accuracy.
    raise NotImplementedError("Task 4 is not complete: train, test, and compare both models.")

    print("\nFinal comparison")
    print("-------------------------------")
    print(f"MLP test accuracy: {100 * mlp_accuracy:.1f}%")
    print(f"MLP average loss: {mlp_avg_loss:.6f}")
    print(f"CNN test accuracy: {100 * cnn_accuracy:.1f}%")
    print(f"CNN average loss: {cnn_avg_loss:.6f}")

    # Save only the selected model weights, matching the official state_dict pattern.
    # TODO 5: Save the selected model to model_weights.pth.
    # Use MODEL_TO_SAVE to choose "mlp" or "cnn".
    # selected_model = ...
    # selected_name = ...
    # save_path = ...
    # torch.save(selected_model.state_dict(), save_path)
    raise NotImplementedError("Task 5 is not complete: save the selected model.")

    print(f"Selected saved model: {selected_name}")
    print(f"Saved model path: {save_path}")

    original_fc, modified_fc, trainable_params = setup_transfer_learning_preview()

    print("\nTransfer learning preview")
    print("-------------------------------")
    print("Original ResNet18 final layer:")
    print(original_fc)
    print("Modified ResNet18 final layer:")
    print(modified_fc)
    print(f"Trainable parameter count: {trainable_params}")
    print("\nDone!")


def setup_transfer_learning_preview() -> tuple[nn.Module, nn.Module, int]:
    # This mirrors the transfer learning preview: inspect fc, replace fc, count trainable params.
    # TODO 6: Complete the ResNet18 transfer learning preview.
    # Required steps:
    # - load ResNet18 pretrained weights
    # - save the original final layer
    # - replace the final layer with nn.Linear(num_ftrs, TRANSFER_NUM_CLASSES)
    # - save the modified final layer
    # - count trainable parameters
    # weights = ResNet18_Weights.DEFAULT
    # resnet18 = models.resnet18(weights=weights)
    # original_fc = ...
    # num_ftrs = ...
    # resnet18.fc = ...
    # modified_fc = ...
    # trainable_params = ...
    raise NotImplementedError("Task 6 is not complete: finish the ResNet18 preview.")

    return original_fc, modified_fc, trainable_params


if __name__ == "__main__":
    main()
