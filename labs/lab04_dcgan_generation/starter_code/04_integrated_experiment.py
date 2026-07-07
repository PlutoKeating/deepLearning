"""Integrated Lab 04 experiment: DCGAN image generation.

This script is the main student task for Lab 04. Use scripts 00-03 as
references, then complete the seven TODO blocks in this file.

The default configuration uses all prepared CelebA images and a medium-length
training run. It is longer than the walkthrough, but shorter than a full-quality
research run.
"""

from pathlib import Path
import shutil

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image


# =========================
# Student Configuration Area
# Medium default: intended to show visible improvement without making the lab
# unreasonably long. If CPU training is too slow, reduce NUM_EPOCHS.
# =========================
LAB_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = LAB_ROOT / "data" / "faces"
FLAT_CELEBA_DIR = LAB_ROOT / "data" / "img_align_celeba"
OUTPUT_DIR = LAB_ROOT / "outputs"

IMAGE_SIZE = 64
BATCH_SIZE = 128
NUM_WORKERS = 0
NZ = 100
NGF = 64
NDF = 64
NC = 3
NUM_EPOCHS = 20
LEARNING_RATE_D = 0.00005
LEARNING_RATE_G = 0.0002
BETA1 = 0.5
GRAD_CLIP_NORM = 10.0
# None means "train on all prepared ImageFolder images".
TRAIN_IMAGE_LIMIT: int | None = None
FIXED_NOISE_SIZE = 64
GENERATOR_STEPS = 1
PRINT_EVERY_BATCHES = 20
SAVE_EVERY_EPOCHS = 5


REAL_LABEL_RANGE = (0.85, 1.0)
FAKE_LABEL_RANGE = (0.0, 0.15)
GENERATOR_LABEL_RANGE = (0.90, 1.0)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REAL_GRID_NAME = "integrated_real_samples.png"
FAKE_GRID_NAME = "integrated_fake_samples.png"
LOSS_CURVE_NAME = "integrated_loss_curve.png"
GENERATOR_WEIGHTS_NAME = "generator_weights.pth"
DISCRIMINATOR_WEIGHTS_NAME = "discriminator_weights.pth"
MISSING_DATA_MESSAGE = (
    "Please place the face dataset under "
    "labs/lab04_dcgan_generation/data/faces/train/faces/ "
    "or extract CelebA under labs/lab04_dcgan_generation/data/img_align_celeba/"
)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        return []
    return [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def prepare_imagefolder_images(image_limit: int | None) -> None:
    """Copy flat CelebA images into the ImageFolder training directory."""
    imagefolder_class_dir = DATA_ROOT / "train" / "faces"
    flat_images = list_images(FLAT_CELEBA_DIR)
    if not flat_images:
        return

    imagefolder_class_dir.mkdir(parents=True, exist_ok=True)
    existing_images = list_images(imagefolder_class_dir)
    if image_limit is None:
        selected_images = flat_images
        target_description = "all available images"
    else:
        selected_images = flat_images[: min(image_limit, len(flat_images))]
        target_description = f"first {len(selected_images)} images"

    existing_names = {path.name for path in existing_images}
    images_to_copy = [
        path for path in selected_images if path.name not in existing_names
    ]
    if not images_to_copy:
        return

    print("Expanding ImageFolder images from the flat CelebA folder.")
    print(f"Source folder: {FLAT_CELEBA_DIR}")
    print(f"Target folder: {imagefolder_class_dir}")
    print(f"Existing ImageFolder images: {len(existing_images)}")
    print(f"Target prepared images: {len(selected_images)} ({target_description})")
    print(f"Images to copy now: {len(images_to_copy)}")
    for source_path in images_to_copy:
        shutil.copy2(source_path, imagefolder_class_dir / source_path.name)


def check_dataset_exists() -> bool:
    expected_dir = DATA_ROOT / "train" / "faces"
    prepare_imagefolder_images(TRAIN_IMAGE_LIMIT)

    image_count = len(list_images(expected_dir))
    if image_count == 0:
        print(MISSING_DATA_MESSAGE)
        print(f"Expected ImageFolder path: {expected_dir}")
        print(f"Flat CelebA path: {FLAT_CELEBA_DIR}")
        return False

    if TRAIN_IMAGE_LIMIT is not None and image_count < TRAIN_IMAGE_LIMIT:
        print(
            "Warning: fewer images than TRAIN_IMAGE_LIMIT are available; "
            f"using {image_count} images."
        )
    return True


def weights_init(module: nn.Module) -> None:
    """Initialize DCGAN convolution and batch-normalization layers."""
    class_name = module.__class__.__name__
    if class_name.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif class_name.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    return (images * 0.5 + 0.5).clamp(0, 1)


def save_image_grid(images: torch.Tensor, output_path: Path, nrow: int = 8) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = images.detach().cpu()
    save_image(denormalize(images), output_path, nrow=nrow)
    return output_path


def save_loss_curve(
    discriminator_losses: list[float],
    generator_losses: list[float],
    output_path: Path,
) -> Path:
    """Save a simple PNG loss curve without using extra plotting packages."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 640, 420
    margin = 55
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((margin, 18), "DCGAN loss curve", fill="black")
    draw.text((width - 150, 18), "D loss: blue", fill="blue")
    draw.text((width - 150, 38), "G loss: red", fill="red")

    x0, y0 = margin, height - margin
    x1, y1 = width - margin, margin
    draw.line((x0, y0, x1, y0), fill="black", width=2)
    draw.line((x0, y0, x0, y1), fill="black", width=2)
    draw.text((width // 2 - 25, height - 35), "batch", fill="black")
    draw.text((10, height // 2), "loss", fill="black")

    all_losses = discriminator_losses + generator_losses
    if not all_losses:
        draw.text((margin + 30, height // 2), "No loss values recorded.", fill="black")
        image.save(output_path)
        return output_path

    min_loss = min(0.0, min(all_losses))
    max_loss = max(1.0, max(all_losses))
    if max_loss == min_loss:
        max_loss = min_loss + 1.0

    plot_width = x1 - x0
    plot_height = y0 - y1

    def to_point(index: int, value: float, count: int) -> tuple[int, int]:
        if count <= 1:
            x = x0
        else:
            x = x0 + int(index * plot_width / (count - 1))
        y_ratio = (value - min_loss) / (max_loss - min_loss)
        y = y0 - int(y_ratio * plot_height)
        return x, y

    def draw_curve(values: list[float], color: str) -> None:
        if not values:
            return
        points = [to_point(index, value, len(values)) for index, value in enumerate(values)]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        else:
            draw.line(points, fill=color, width=2)

    draw_curve(discriminator_losses, "blue")
    draw_curve(generator_losses, "red")
    draw.text((margin, margin - 22), f"{max_loss:.2f}", fill="black")
    draw.text((margin, height - margin + 8), f"{min_loss:.2f}", fill="black")

    image.save(output_path)
    return output_path


def get_training_dataset(
    dataset: datasets.ImageFolder,
    image_limit: int | None,
) -> datasets.ImageFolder | Subset:
    if image_limit is None:
        return dataset

    actual_size = min(image_limit, len(dataset))
    if actual_size <= 0:
        raise ValueError("No images were found in the face dataset.")
    return Subset(dataset, list(range(actual_size)))


def checkpoint_config() -> dict[str, object]:
    """Store enough settings to understand the saved DCGAN weights later."""
    return {
        "image_size": IMAGE_SIZE,
        "nz": NZ,
        "ngf": NGF,
        "ndf": NDF,
        "nc": NC,
        "num_epochs": NUM_EPOCHS,
        "learning_rate_d": LEARNING_RATE_D,
        "learning_rate_g": LEARNING_RATE_G,
        "beta1": BETA1,
        "train_image_limit": TRAIN_IMAGE_LIMIT,
        "generator_steps": GENERATOR_STEPS,
        "real_label_range": REAL_LABEL_RANGE,
        "fake_label_range": FAKE_LABEL_RANGE,
        "generator_label_range": GENERATOR_LABEL_RANGE,
    }


def create_dataloader() -> tuple[datasets.ImageFolder, datasets.ImageFolder | Subset, DataLoader]:
    # TODO 1: Complete ImageFolder data loading for DCGAN.
    data_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.ImageFolder(DATA_ROOT / "train", transform=data_transform)
    training_dataset = get_training_dataset(dataset, TRAIN_IMAGE_LIMIT)
    dataloader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=(len(training_dataset) >= BATCH_SIZE),
    )

    return dataset, training_dataset, dataloader


class Generator(nn.Module):
    """DCGAN Generator mapping latent noise to 64x64 RGB images."""

    def __init__(self) -> None:
        super().__init__()
        # TODO 2: Complete the DCGAN Generator.
        self.main = nn.Sequential(
            # [batch, NZ, 1, 1] -> [batch, NGF*8, 4, 4]
            nn.ConvTranspose2d(NZ, NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),
            # 4x4 -> 8x8
            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            # 32x32 -> 64x64 RGB; Tanh range matches normalized real images.
            nn.ConvTranspose2d(NGF, NC, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class Discriminator(nn.Module):
    """DCGAN Discriminator mapping 64x64 RGB images to real/fake scores."""

    def __init__(self) -> None:
        super().__init__()
        # TODO 3: Complete the DCGAN Discriminator.
        self.main = nn.Sequential(
            # [batch, 3, 64, 64] -> [batch, NDF, 32, 32]
            nn.Conv2d(NC, NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32 -> 16x16
            nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 -> 8x8
            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8 -> 4x4
            nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4 -> 1x1; each image receives one real/fake probability.
            nn.Conv2d(NDF * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(-1)


def make_smoothed_labels(
    output: torch.Tensor,
    label_range: tuple[float, float],
) -> torch.Tensor:
    low, high = label_range
    return torch.empty_like(output).uniform_(low, high)


def train_one_batch(
    netG: nn.Module,
    netD: nn.Module,
    real_images: torch.Tensor,
    criterion: nn.Module,
    optimizerD: torch.optim.Optimizer,
    optimizerG: torch.optim.Optimizer,
    device: str,
) -> dict[str, float]:
    real_images = real_images.to(device)
    batch_size = real_images.size(0)

    # TODO 4: Complete the Discriminator update.
    netD.zero_grad(set_to_none=True)
    output_real = netD(real_images)
    real_labels = make_smoothed_labels(output_real, REAL_LABEL_RANGE)
    errD_real = criterion(output_real, real_labels)
    errD_real.backward()
    d_x = output_real.mean().item()

    noise = torch.randn(batch_size, NZ, 1, 1, device=device)
    fake_images = netG(noise)
    output_fake = netD(fake_images.detach())
    fake_labels = make_smoothed_labels(output_fake, FAKE_LABEL_RANGE)
    errD_fake = criterion(output_fake, fake_labels)
    errD_fake.backward()
    d_g_z1 = output_fake.mean().item()

    errD = errD_real + errD_fake
    torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=GRAD_CLIP_NORM)
    optimizerD.step()

    errG = torch.tensor(0.0, device=device)
    d_g_z2 = 0.0
    # TODO 5: Complete the Generator update.
    for _ in range(GENERATOR_STEPS):
        netG.zero_grad(set_to_none=True)
        generator_noise = torch.randn(batch_size, NZ, 1, 1, device=device)
        fake_for_generator = netG(generator_noise)
        output_for_generator = netD(fake_for_generator)
        generator_labels = make_smoothed_labels(
            output_for_generator,
            GENERATOR_LABEL_RANGE,
        )
        errG = criterion(output_for_generator, generator_labels)
        errG.backward()
        d_g_z2 = output_for_generator.mean().item()
        torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizerG.step()

    return {
        "D_loss": errD.item(),
        "G_loss": errG.item(),
        "D_x": d_x,
        "D_G_z_before": d_g_z1,
        "D_G_z_after": d_g_z2,
    }


def save_fixed_noise_samples(
    netG: nn.Module,
    fixed_noise: torch.Tensor,
    output_path: Path,
) -> Path:
    netG.eval()
    with torch.no_grad():
        # TODO 6: Save generated samples from fixed noise.
        fake_samples = netG(fixed_noise).detach().cpu()
        save_image_grid(fake_samples, output_path)
    netG.train()
    return output_path


def save_model_weights(netG: nn.Module, netD: nn.Module) -> tuple[Path, Path]:
    # TODO 7: Save Generator and Discriminator state_dict checkpoints.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = checkpoint_config()
    generator_path = OUTPUT_DIR / GENERATOR_WEIGHTS_NAME
    discriminator_path = OUTPUT_DIR / DISCRIMINATOR_WEIGHTS_NAME
    torch.save(
        {"model_state_dict": netG.state_dict(), "config": config},
        generator_path,
    )
    torch.save(
        {"model_state_dict": netD.state_dict(), "config": config},
        discriminator_path,
    )

    return generator_path, discriminator_path


def should_print_batch(batch_index: int, batch_count: int) -> bool:
    return (
        batch_index == 0
        or (batch_index + 1) % PRINT_EVERY_BATCHES == 0
        or (batch_index + 1) == batch_count
    )


def main() -> None:
    torch.manual_seed(0)

    print("=== 04 Integrated Experiment ===")
    print(f"Dataset path: {DATA_ROOT}")
    print(f"Flat CelebA path: {FLAT_CELEBA_DIR}")
    if TRAIN_IMAGE_LIMIT is None:
        print("Training image count: all available ImageFolder images")
    else:
        print(f"Training image count limit: {TRAIN_IMAGE_LIMIT}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Discriminator learning rate: {LEARNING_RATE_D}")
    print(f"Generator learning rate: {LEARNING_RATE_G}")
    print(f"Beta1: {BETA1}")
    print(f"Gradient clip norm: {GRAD_CLIP_NORM}")
    print(f"Generator steps per batch: {GENERATOR_STEPS}")
    print(f"Fixed noise size: {FIXED_NOISE_SIZE}")
    print(f"Save every epochs: {SAVE_EVERY_EPOCHS}")

    if not check_dataset_exists():
        return

    device = get_device()
    print(f"Device: {device}")

    dataset, training_dataset, dataloader = create_dataloader()
    print(f"Dataset size: {len(dataset)}")
    print(f"Actual training image count: {len(training_dataset)}")
    print(f"Actual number of batches: {len(dataloader)}")
    print(f"Class names: {dataset.classes}")

    real_images, _ = next(iter(dataloader))
    print(f"One training batch shape: {real_images.shape}")
    print(
        "One training batch normalized range: "
        f"min={real_images.min().item():.4f}, max={real_images.max().item():.4f}"
    )
    real_grid_path = save_image_grid(
        real_images[: min(FIXED_NOISE_SIZE, real_images.size(0))],
        OUTPUT_DIR / REAL_GRID_NAME,
    )
    print(f"Saved real sample grid path: {real_grid_path}")

    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    print("\nGenerator structure:")
    print(netG)
    print("\nDiscriminator structure:")
    print(netD)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(FIXED_NOISE_SIZE, NZ, 1, 1, device=device)
    optimizerD = torch.optim.Adam(
        netD.parameters(),
        lr=LEARNING_RATE_D,
        betas=(BETA1, 0.999),
    )
    optimizerG = torch.optim.Adam(
        netG.parameters(),
        lr=LEARNING_RATE_G,
        betas=(BETA1, 0.999),
    )

    discriminator_losses: list[float] = []
    generator_losses: list[float] = []

    print("\nStep 1 - longer DCGAN training")
    print("=" * 40)
    for epoch in range(NUM_EPOCHS):
        epoch_d_losses: list[float] = []
        epoch_g_losses: list[float] = []
        for batch_index, (images, _) in enumerate(dataloader):
            metrics = train_one_batch(
                netG,
                netD,
                images,
                criterion,
                optimizerD,
                optimizerG,
                device,
            )
            discriminator_losses.append(metrics["D_loss"])
            generator_losses.append(metrics["G_loss"])
            epoch_d_losses.append(metrics["D_loss"])
            epoch_g_losses.append(metrics["G_loss"])

            if should_print_batch(batch_index, len(dataloader)):
                print(
                    f"Epoch {epoch + 1}/{NUM_EPOCHS}, "
                    f"batch {batch_index + 1}/{len(dataloader)} | "
                    f"Discriminator loss: {metrics['D_loss']:.4f} | "
                    f"Generator loss: {metrics['G_loss']:.4f} | "
                    f"D(x): {metrics['D_x']:.4f} | "
                    f"D(G(z)): {metrics['D_G_z_before']:.4f} -> "
                    f"{metrics['D_G_z_after']:.4f}"
                )

        average_d_loss = sum(epoch_d_losses) / max(len(epoch_d_losses), 1)
        average_g_loss = sum(epoch_g_losses) / max(len(epoch_g_losses), 1)
        print(
            f"Epoch {epoch + 1} summary | "
            f"Average D loss: {average_d_loss:.4f} | "
            f"Average G loss: {average_g_loss:.4f}"
        )

        if (epoch + 1) % SAVE_EVERY_EPOCHS == 0 or (epoch + 1) == NUM_EPOCHS:
            epoch_grid_path = save_fixed_noise_samples(
                netG,
                fixed_noise,
                OUTPUT_DIR / f"integrated_fake_epoch_{epoch + 1:03d}.png",
            )
            print(f"Saved epoch fake grid path: {epoch_grid_path}")

    print("\nStep 2 - save generated samples and loss curve")
    print("=" * 40)
    fake_grid_path = save_fixed_noise_samples(
        netG,
        fixed_noise,
        OUTPUT_DIR / FAKE_GRID_NAME,
    )
    loss_curve_path = save_loss_curve(
        discriminator_losses,
        generator_losses,
        OUTPUT_DIR / LOSS_CURVE_NAME,
    )
    print(f"Saved fake sample grid path: {fake_grid_path}")
    print(f"Saved loss curve path: {loss_curve_path}")

    print("\nStep 3 - save final model weights")
    print("=" * 40)
    generator_path, discriminator_path = save_model_weights(netG, netD)
    print(f"Saved generator weights path: {generator_path}")
    print(f"Saved discriminator weights path: {discriminator_path}")
    print("Done!")


if __name__ == "__main__":
    main()
