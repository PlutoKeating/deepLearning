"""Save and Load Model example adapted from PyTorch Learn the Basics."""

from pathlib import Path

import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # The architecture must match when loading a saved state_dict.
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
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

    print("=== 07 Save and Load Model ===")

    device = get_device()
    model = NeuralNetwork().to(device)

    save_path = Path(__file__).resolve().parents[1] / "model_weights.pth"
    # The official tutorial recommends saving model weights with state_dict().
    torch.save(model.state_dict(), save_path)
    print(f"Saved PyTorch model state_dict to: {save_path}")

    # Recreate the same model class, then load the saved parameter tensors.
    loaded_model = NeuralNetwork().to(device)
    state_dict = torch.load(save_path, map_location=device, weights_only=True)
    loaded_model.load_state_dict(state_dict)
    # eval() is the normal mode for inference after loading.
    loaded_model.eval()

    x = torch.rand(1, 28, 28, device=device)
    with torch.no_grad():
        logits = loaded_model(x)
        predicted_class = logits.argmax(1).item()

    print("Loaded model and ran one prediction.")
    print("Logits:")
    print(logits)
    print(f"Predicted class after loading: {predicted_class}")


if __name__ == "__main__":
    main()
