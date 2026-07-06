"""Build Model example adapted from PyTorch Learn the Basics."""

import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # nn.Module stores layers as attributes so PyTorch can track parameters.
        self.flatten = nn.Flatten()
        # The basics tutorial stacks Linear and ReLU layers for image classification.
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward defines how one input batch flows through the model.
        x = self.flatten(x)
        # The final Linear layer returns raw class scores called logits.
        logits = self.linear_relu_stack(x)
        return logits


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    torch.manual_seed(0)

    print("=== 04 Build Model ===")
    device = get_device()
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print("\nModel structure:")
    print(model)

    # A dummy image batch lets us inspect model output without training.
    x = torch.rand(1, 28, 28, device=device)
    logits = model(x)
    # Softmax converts logits to class probabilities along the class dimension.
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)

    print("\nLogits:")
    print(logits)
    print("\nPredicted probabilities:")
    print(pred_probab)
    print(f"\nPredicted class: {y_pred.item()}")


if __name__ == "__main__":
    main()
