"""Environment check based on the official PyTorch local installation page."""

import platform
import sys

import torch
import torchvision


def main() -> None:
    print("=== 00 Environment Check ===")

    # Basic version information helps confirm that Python, torch, and torchvision
    # were installed in the same environment, as on the official install page.
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Python platform: {platform.platform()}")
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")

    # Creating a random tensor is the smallest practical smoke test for PyTorch.
    print("\ntorch.rand(5, 3):")
    print(torch.rand(5, 3))

    # CUDA is optional in this lab; CPU output is still a successful check.
    cuda_available = torch.cuda.is_available()
    print(f"\ntorch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    # MPS is Apple's GPU backend. It is checked only when the backend exists.
    mps_available = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    print(f"torch.backends.mps.is_available(): {mps_available}")


if __name__ == "__main__":
    main()
