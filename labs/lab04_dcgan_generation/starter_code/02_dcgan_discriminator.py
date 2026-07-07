"""Show the DCGAN Discriminator architecture for Lab 04."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ImageFolder root read by the Discriminator demo; it matches 00_check_dataset.py.
DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "faces"
# Discriminator input size is 64x64, matching the Generator output.
IMAGE_SIZE = 64
# This script only demonstrates one forward pass, so a small batch keeps output readable.
BATCH_SIZE = 8
# num_workers=0 avoids common Windows multiprocessing issues in classroom settings.
NUM_WORKERS = 0
# NDF controls the base Discriminator feature channels, matching the tutorial's ndf idea.
NDF = 64
# NC is the input channel count; RGB face images have 3 channels.
NC = 3
MISSING_DATA_MESSAGE = (
    "Please place the face dataset under "
    "labs/lab04_dcgan_generation/data/faces/train/faces/"
)


class Discriminator(nn.Module):
    """DCGAN Discriminator from a 64x64 RGB image to a real/fake score."""

    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Sequential(
            # Input: [batch, 3, 64, 64]. The first layer reduces spatial size to 32x32.
            nn.Conv2d(NC, NDF, 4, 2, 1, bias=False),
            # Discriminators usually use LeakyReLU so negative activations still carry gradients.
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32 -> 16x16 while channels increase to learn more abstract local features.
            nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 -> 8x8.
            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8 -> 4x4.
            nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4 -> 1x1, giving one real/fake score for each image.
            nn.Conv2d(NDF * 8, 1, 4, 1, 0, bias=False),
            # Sigmoid maps scores to [0, 1], interpretable as "probability of being real".
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Raw output is [batch, 1, 1, 1]; view(-1) gives [batch] for BCELoss.
        return self.main(x).view(-1)


def load_real_or_dummy_batch() -> tuple[torch.Tensor, str, int]:
    expected_dir = DATA_ROOT / "train" / "faces"
    if not expected_dir.is_dir():
        # Even without data, a dummy batch can demonstrate Discriminator shape behavior.
        print(MISSING_DATA_MESSAGE)
        print("Dataset not found for this demo; using a dummy image batch.")
        # torch.rand gives [0, 1]; multiplying by 2 and subtracting 1 gives DCGAN's [-1, 1].
        images = torch.rand(BATCH_SIZE, NC, IMAGE_SIZE, IMAGE_SIZE) * 2 - 1
        return images, "dummy", 0

    # Real-image preprocessing must match the Generator output range, or D/G training is mismatched.
    data_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.ImageFolder(DATA_ROOT / "train", transform=data_transform)
    if len(dataset) == 0:
        print("Dataset folder exists, but no images were found.")
        print("Using a dummy image batch.")
        # For an empty dataset, still return a correctly shaped dummy batch for the architecture demo.
        images = torch.rand(BATCH_SIZE, NC, IMAGE_SIZE, IMAGE_SIZE) * 2 - 1
        return images, "dummy", 0

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    # Take one batch for a forward pass only; no parameters are updated in this script.
    images, _ = next(iter(dataloader))
    return images, "real", len(dataset)


def main() -> None:
    # Fix the random seed so dummy data and initial-weight outputs are easier to reproduce.
    torch.manual_seed(0)
    print("=== 02 DCGAN Discriminator ===")
    print(f"Dataset path: {DATA_ROOT}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Input channels NC: {NC}")
    print(f"Discriminator feature size NDF: {NDF}")

    netD = Discriminator()
    images, batch_source, dataset_size = load_real_or_dummy_batch()

    with torch.no_grad():
        # This script only checks whether the Discriminator can process one batch.
        scores = netD(images)

    # Print only the first 5 scores to keep terminal output short for screenshots.
    preview_scores = [round(value, 4) for value in scores[:5].tolist()]

    print("\nDiscriminator structure:")
    print(netD)
    print(f"Input batch source: {batch_source}")
    print(f"Dataset size: {dataset_size}")
    print(f"Input image batch shape: {images.shape}")
    print(f"Discriminator output shape: {scores.shape}")
    print(f"Example discriminator scores: {preview_scores}")
    print("Done!")


if __name__ == "__main__":
    main()
