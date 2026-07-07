"""Show the DCGAN Generator architecture for Lab 04."""

from pathlib import Path

import torch
from torch import nn
from torchvision.utils import save_image


# The output directory stores sample grids from the untrained Generator.
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
# NZ is the latent/noise vector channel count; the official DCGAN tutorial uses 100.
NZ = 100
# NGF controls Generator feature channels; larger values increase capacity and compute.
NGF = 64
# NC is the number of output channels; CelebA images are RGB, so NC is 3.
NC = 3
# The Generator outputs 64x64 images, matching the official tutorial and lab preprocessing.
IMAGE_SIZE = 64
# Generate 64 samples so they can be saved as an 8x8 grid.
FIXED_NOISE_SIZE = 64


class Generator(nn.Module):
    """DCGAN Generator from latent noise to a 64x64 RGB image."""

    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Sequential(
            # Input: [batch, NZ, 1, 1]. Expand a 1x1 noise vector into a 4x4 feature map.
            nn.ConvTranspose2d(NZ, NGF * 8, 4, 1, 0, bias=False),
            # BatchNorm stabilizes DCGAN training; it is shown here as part of the standard design.
            nn.BatchNorm2d(NGF * 8),
            # The Generator commonly uses ReLU in hidden layers for nonlinear feature growth.
            nn.ReLU(True),
            # 4x4 -> 8x8; channels decrease while spatial resolution doubles.
            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
            # 8x8 -> 16x16。
            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
            # 16x16 -> 32x32。
            nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            # 32x32 -> 64x64, then convert feature channels into 3 RGB channels.
            nn.ConvTranspose2d(NGF, NC, 4, 2, 1, bias=False),
            # Tanh outputs [-1, 1], which must match normalized training images.
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # nn.Sequential runs the transposed convolutions, normalizations, and activations in order.
        return self.main(x)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    # PNG saving expects [0, 1], so convert DCGAN tensors from [-1, 1] back to [0, 1].
    return (images * 0.5 + 0.5).clamp(0, 1)


def main() -> None:
    # Fix the random seed so the untrained sample grid is reproducible across runs.
    torch.manual_seed(0)
    print("=== 01 DCGAN Generator ===")
    print(f"Latent vector size NZ: {NZ}")
    print(f"Generator feature size NGF: {NGF}")
    print(f"Output channels NC: {NC}")
    print(f"Image size: {IMAGE_SIZE}")

    netG = Generator()
    # Noise must have shape [batch, NZ, 1, 1] to match the first ConvTranspose2d layer.
    fixed_noise = torch.randn(FIXED_NOISE_SIZE, NZ, 1, 1)

    with torch.no_grad():
        # This script only performs forward inference and shape inspection; gradients are unnecessary.
        fake_images = netG(fixed_noise)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "untrained_generator_samples.png"
    # An untrained Generator will not look realistic, but it verifies architecture and tensor shapes.
    save_image(denormalize(fake_images), output_path, nrow=8)

    print("\nGenerator structure:")
    print(netG)
    print(f"Fixed noise shape: {fixed_noise.shape}")
    print(f"Generated image tensor shape: {fake_images.shape}")
    print(f"Saved untrained fake image grid path: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
