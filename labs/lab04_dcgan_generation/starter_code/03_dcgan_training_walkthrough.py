"""Medium-length DCGAN training walkthrough for Lab 04."""

from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


# Locate the lab directory from this script path, so the script works from different terminals.
LAB_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = LAB_ROOT / "data" / "faces"
OUTPUT_DIR = LAB_ROOT / "outputs"
# Keep the image size, noise dimension, and base channel counts close to the official DCGAN tutorial.
IMAGE_SIZE = 64
BATCH_SIZE = 64
NUM_WORKERS = 0
NZ = 100
NGF = 64
NDF = 64
NC = 3
# A smaller Discriminator learning rate helps avoid D overpowering G too early.
LEARNING_RATE_D = 0.00005
LEARNING_RATE_G = 0.0002
# beta1=0.5 for Adam is the classic setting used in the official DCGAN tutorial.
BETA1 = 0.5
NUM_EPOCHS = 10
MAX_TRAIN_BATCHES_PER_EPOCH = 32
# Update G twice after each D step so G has a better chance to improve during a short run.
GENERATOR_STEPS = 2
FIXED_NOISE_SIZE = 64
# Smoothed labels make the adversarial game less brittle in short training.
REAL_LABEL_RANGE = (0.85, 1.0)
FAKE_LABEL_RANGE = (0.0, 0.15)
GENERATOR_LABEL_RANGE = (0.90, 1.0)
GRAD_CLIP_NORM = 10.0
PRINT_EVERY_BATCHES = 8
# White padding makes individual samples easier to see in saved image grids.
GRID_PADDING = 2
GRID_PAD_VALUE = 1.0
MISSING_DATA_MESSAGE = (
    "Please place the face dataset under "
    "labs/lab04_dcgan_generation/data/faces/train/faces/"
)


def get_device() -> str:
    # Prefer CUDA, then Apple Silicon MPS, and always fall back to CPU.
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_dataset_exists() -> bool:
    # Training expects ImageFolder structure: data/faces/train/faces/*.jpg.
    expected_dir = DATA_ROOT / "train" / "faces"
    if not DATA_ROOT.is_dir() or not expected_dir.is_dir():
        print(MISSING_DATA_MESSAGE)
        print(f"Expected dataset path: {expected_dir}")
        return False
    return True


def weights_init(module: nn.Module) -> None:
    # DCGAN initializes convolution layers with N(0, 0.02).
    class_name = module.__class__.__name__
    if class_name.find("Conv") != -1:
        # Initialize convolution and transposed-convolution weights from N(0, 0.02).
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif class_name.find("BatchNorm") != -1:
        # Initialize BatchNorm scale near 1 and bias at 0, following the DCGAN tutorial.
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
    def __init__(self) -> None:
        super().__init__()
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
        # Flatten [batch, 1, 1, 1] to [batch], matching the label tensor shape.
        return self.main(x).view(-1)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    # Training tensors are in [-1, 1]; convert back to [0, 1] before saving PNGs.
    return (images * 0.5 + 0.5).clamp(0, 1)


def make_smoothed_labels(
    output: torch.Tensor,
    label_range: tuple[float, float],
) -> torch.Tensor:
    low, high = label_range
    return torch.empty_like(output).uniform_(low, high)


def should_print_batch(batch_index: int, batch_count: int) -> bool:
    return (
        batch_index == 0
        or (batch_index + 1) % PRINT_EVERY_BATCHES == 0
        or (batch_index + 1) == batch_count
    )


def save_fixed_noise_grid(
    netG: nn.Module,
    fixed_noise: torch.Tensor,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep train mode for early DCGAN visualization so BatchNorm uses batch
    # statistics from fixed_noise, matching the official tutorial style.
    with torch.no_grad():
        # fixed_noise stays unchanged, so epoch grids show how the same latent vectors evolve.
        fake_samples = netG(fixed_noise).detach().cpu()
    save_image(
        denormalize(fake_samples),
        output_path,
        nrow=8,
        padding=GRID_PADDING,
        pad_value=GRID_PAD_VALUE,
    )
    return output_path


def save_real_image_grid(real_images: torch.Tensor, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # A real-image grid is a visual baseline for both data preprocessing and generated samples.
    save_image(
        denormalize(real_images.detach().cpu()),
        output_path,
        nrow=8,
        padding=GRID_PADDING,
        pad_value=GRID_PAD_VALUE,
    )
    return output_path


def save_loss_curve(
    discriminator_losses: list[float],
    generator_losses: list[float],
    output_path: Path,
) -> Path:
    """Save a simple PNG loss curve without external plotting packages."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Draw the plot with PIL so students do not need to install matplotlib.
    width, height = 640, 420
    margin = 55
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((margin, 18), "DCGAN walkthrough loss curve", fill="black")
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
        # Save an empty plot if no losses were recorded, making the failure easy to locate.
        image.save(output_path)
        return output_path

    # The y-axis covers at least [0, 1] and expands automatically for larger losses.
    min_loss = min(0.0, min(all_losses))
    max_loss = max(1.0, max(all_losses))
    if max_loss == min_loss:
        max_loss = min_loss + 1.0

    plot_width = x1 - x0
    plot_height = y0 - y1

    def to_point(index: int, value: float, count: int) -> tuple[int, int]:
        # Map one loss value to canvas coordinates: x is batch index, y is loss magnitude.
        if count <= 1:
            x = x0
        else:
            x = x0 + int(index * plot_width / (count - 1))
        y_ratio = (value - min_loss) / (max_loss - min_loss)
        y = y0 - int(y_ratio * plot_height)
        return x, y

    def draw_curve(values: list[float], color: str) -> None:
        # Draw a single value as a dot, or multiple values as a connected curve.
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


def create_dataloader() -> DataLoader:
    # Training transforms must match the data-check script and the Generator's output range.
    data_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    # ImageFolder sees train/faces as one class; class labels are not used for GAN real/fake training.
    dataset = datasets.ImageFolder(DATA_ROOT / "train", transform=data_transform)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )


def main() -> None:
    # Fix the random seed for more reproducible noise, shuffle order, and model initialization.
    torch.manual_seed(0)
    print("=== 03 DCGAN Training Walkthrough ===")
    print(f"Dataset path: {DATA_ROOT}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Maximum batches per epoch: {MAX_TRAIN_BATCHES_PER_EPOCH}")
    print(f"Discriminator learning rate: {LEARNING_RATE_D}")
    print(f"Generator learning rate: {LEARNING_RATE_G}")
    print(f"Generator updates per batch: {GENERATOR_STEPS}")
    print(f"Gradient clip norm: {GRAD_CLIP_NORM}")
    print(f"Print every batches: {PRINT_EVERY_BATCHES}")

    if not check_dataset_exists():
        return

    dataloader = create_dataloader()
    print(f"Dataset size: {len(dataloader.dataset)}")
    # Save one real batch first; this also proves the DataLoader can produce valid image tensors.
    preview_real_images, _ = next(iter(dataloader))

    device = get_device()
    print(f"Device: {device}")

    # Create Generator G and Discriminator D, then move both to the same device.
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    # apply recursively visits every submodule and runs DCGAN weight initialization.
    netG.apply(weights_init)
    netD.apply(weights_init)

    # BCELoss treats real/fake detection as binary classification.
    # G and D have separate optimizers because GAN training alternates between two objectives.
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(FIXED_NOISE_SIZE, NZ, 1, 1, device=device)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, 0.999))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    real_path = save_real_image_grid(
        preview_real_images[:FIXED_NOISE_SIZE],
        OUTPUT_DIR / "walkthrough_real_samples.png",
    )
    print(f"Saved walkthrough real image grid path: {real_path}")

    initial_path = save_fixed_noise_grid(
        netG,
        fixed_noise,
        OUTPUT_DIR / "walkthrough_fake_epoch_00.png",
    )
    print(f"Saved initial fixed-noise fake grid path: {initial_path}")

    discriminator_losses: list[float] = []
    generator_losses: list[float] = []
    final_d_loss = 0.0
    final_g_loss = 0.0
    final_d_x = 0.0
    final_d_g_z1 = 0.0
    final_d_g_z2 = 0.0

    print("\nStarting adversarial training walkthrough")
    print("-" * 50)
    print("Note: GAN losses are adversarial and do not need to decrease monotonically.")
    print("Use the saved image grids and loss curve to inspect the full process.")
    for epoch in range(NUM_EPOCHS):
        epoch_d_losses: list[float] = []
        epoch_g_losses: list[float] = []
        batches_this_epoch = min(MAX_TRAIN_BATCHES_PER_EPOCH, len(dataloader))

        for batch_index, (real_images, _) in enumerate(dataloader):
            # Limit batches per epoch so the walkthrough finishes in a predictable classroom time.
            if batch_index >= MAX_TRAIN_BATCHES_PER_EPOCH:
                break

            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # ---------------------------
            # Step A: update the Discriminator D.
            # Goal: make D(real_images) close to 1 and D(fake_images) close to 0.
            # Think of D as a binary classifier for this step: it receives both real
            # samples from the dataset and fake samples produced by G.
            # fake_images.detach() is essential here. It cuts the computation graph
            # between fake_images and G, so errD_fake updates only D's parameters.
            # The two backward calls accumulate gradients from real and fake examples
            # before optimizerD.step() applies one combined Discriminator update.
            # ---------------------------
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

            # ---------------------------
            # Step B: update the Generator G.
            # Goal: make D(G(z)) close to 1, meaning generated images are judged real.
            # D is used as a differentiable scoring network in this step. The gradient
            # flows through D's layers and into G's output, then into G's parameters.
            # Only optimizerG.step() is called, so this step changes G, not D.
            # The target range is near 1 because G is rewarded when D believes
            # generated images look like real training images.
            # ---------------------------
            errG = torch.tensor(0.0, device=device)
            d_g_z2 = 0.0
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

            # Record key metrics for terminal screenshots and the saved loss curve.
            # D(x) is D's average score on real images.
            # D(G(z)) before the G update shows how convincing fake images were to D.
            # D(G(z)) after the G update shows whether the latest G step improved its score.
            d_loss = errD.item()
            g_loss = errG.item()
            discriminator_losses.append(d_loss)
            generator_losses.append(g_loss)
            epoch_d_losses.append(d_loss)
            epoch_g_losses.append(g_loss)
            final_d_loss = d_loss
            final_g_loss = g_loss
            final_d_x = d_x
            final_d_g_z1 = d_g_z1
            final_d_g_z2 = d_g_z2

            if should_print_batch(batch_index, batches_this_epoch):
                print(
                    f"Epoch {epoch + 1}/{NUM_EPOCHS}, "
                    f"batch {batch_index + 1}/{batches_this_epoch} | "
                    f"Discriminator loss: {d_loss:.4f} | "
                    f"Generator loss: {g_loss:.4f} | "
                    f"D(x): {d_x:.4f} | "
                    f"D(G(z)): {d_g_z1:.4f} -> {d_g_z2:.4f}"
                )

        average_d_loss = sum(epoch_d_losses) / max(len(epoch_d_losses), 1)
        average_g_loss = sum(epoch_g_losses) / max(len(epoch_g_losses), 1)
        # Save samples from the same fixed noise after each epoch to inspect visual progress.
        epoch_grid_path = save_fixed_noise_grid(
            netG,
            fixed_noise,
            OUTPUT_DIR / f"walkthrough_fake_epoch_{epoch + 1:02d}.png",
        )
        print(
            f"Epoch {epoch + 1} summary | "
            f"Average D loss: {average_d_loss:.4f} | "
            f"Average G loss: {average_g_loss:.4f} | "
            f"Saved epoch fake grid path: {epoch_grid_path}"
        )

    # After training, save the final generated samples and the complete loss curve for reports.
    output_path = save_fixed_noise_grid(
        netG,
        fixed_noise,
        OUTPUT_DIR / "walkthrough_fake_samples.png",
    )
    loss_curve_path = save_loss_curve(
        discriminator_losses,
        generator_losses,
        OUTPUT_DIR / "walkthrough_loss_curve.png",
    )

    print("\nFinal walkthrough checkpoint")
    print("-" * 50)
    print(f"Final discriminator loss: {final_d_loss:.4f}")
    print(f"Final generator loss: {final_g_loss:.4f}")
    print(f"Final D(x): {final_d_x:.4f}")
    print(f"Final D(G(z)): {final_d_g_z1:.4f} -> {final_d_g_z2:.4f}")
    print(f"Saved walkthrough fake image grid path: {output_path}")
    print(f"Saved walkthrough loss curve path: {loss_curve_path}")
    print("Done!")


if __name__ == "__main__":
    main()
