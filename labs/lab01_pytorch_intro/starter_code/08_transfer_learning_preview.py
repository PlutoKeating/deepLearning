"""Transfer learning preview adapted from the official PyTorch tutorial."""

import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def count_trainable_parameters(model: nn.Module) -> int:
    # Frozen parameters have requires_grad=False and will not be updated by optimizers.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    print("=== 08 Transfer Learning Preview ===")
    print("This script previews setup only; it does not run full training.\n")

    weights = ResNet18_Weights.DEFAULT

    print("Fine-tuning setup:")
    # Fine-tuning starts from pretrained ResNet18 and keeps all layers trainable.
    model_ft = models.resnet18(weights=weights)
    print("Original final fc layer:")
    print(model_ft.fc)

    # Replace the ImageNet classifier with a new classifier for two classes.
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    print("Modified final fc layer:")
    print(model_ft.fc)
    print(
        "Trainable parameters for fine-tuning: "
        f"{count_trainable_parameters(model_ft)}"
    )

    print("\nFixed feature extractor setup:")
    model_conv = models.resnet18(weights=weights)
    # Feature extractor mode freezes the convolutional backbone.
    for param in model_conv.parameters():
        param.requires_grad = False

    # The new final layer stays trainable because it is created after freezing.
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)
    print("Modified final fc layer for fixed feature extractor:")
    print(model_conv.fc)
    print(
        "Trainable parameters for fixed feature extractor: "
        f"{count_trainable_parameters(model_conv)}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nOptional training device if used later: {device}")


if __name__ == "__main__":
    main()
