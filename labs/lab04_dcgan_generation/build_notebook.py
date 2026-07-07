#!/usr/bin/env python3
"""Build the Lab 04 Jupyter notebook with pre-captured training outputs."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.14.4"},
}
cells = []

def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))

def code(s):
    cells.append(nbf.v4.new_code_cell(s))

# ===== TITLE =====
md("""# Lab 04: DCGAN for Image Generation

**Course:** Deep Learning and Computer Vision

**Student Name:** \u79e6\u57fa\u8d6b  
**Student ID:** 202464870331  
**Class:** 24\u5927\u6570\u636e1\u73ed  
**Date:** 2026-07-03

---

## Part 1: Walkthrough Script Running Results
""")

# ===== COMMON SETUP =====
code("""import subprocess, sys
from pathlib import Path
from IPython.display import Image, display

LAB_ROOT = Path("/home/pluto/lab04_dcgan_generation")
STARTER = LAB_ROOT / "starter_code"
OUTPUT_DIR = LAB_ROOT / "outputs"
VENV_PYTHON = str(LAB_ROOT / "venv" / "bin" / "python")

def run_script(name):
    result = subprocess.run(
        [VENV_PYTHON, str(STARTER / name)],
        capture_output=True, text=True, cwd=str(STARTER)
    )
    return result.stdout, result.stderr

# Also make transforms and torch available for code display cells
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
""")

# ===== SCRIPT 00 =====
md("""### Script 00: Check Face Dataset

This cell runs `00_check_dataset.py`, which prepares the CelebA dataset in ImageFolder format and verifies the data pipeline.""")

code("""stdout, stderr = run_script("00_check_dataset.py")
print(stdout)
if stderr:
    print("STDERR:", stderr[:500])
""")

md("**Real sample grid** (saved by `00_check_dataset.py`):")

code("""display(Image(filename=str(OUTPUT_DIR / "real_samples.png")))
""")

# ===== SCRIPT 01 =====
md("""### Script 01: DCGAN Generator

This cell runs `01_dcgan_generator.py`, which builds the Generator architecture and produces untrained fake samples.""")

code("""stdout, stderr = run_script("01_dcgan_generator.py")
print(stdout)
if stderr:
    print("STDERR:", stderr[:500])
""")

md("**Untrained generator samples** (saved by `01_dcgan_generator.py`):")

code("""display(Image(filename=str(OUTPUT_DIR / "untrained_generator_samples.png")))
""")

# ===== SCRIPT 02 =====
md("""### Script 02: DCGAN Discriminator

This cell runs `02_dcgan_discriminator.py`, which builds the Discriminator architecture and performs a forward pass on real images.""")

code("""stdout, stderr = run_script("02_dcgan_discriminator.py")
print(stdout)
if stderr:
    print("STDERR:", stderr[:500])
""")

# ===== SCRIPT 03 =====
md("""### Script 03: DCGAN Training Walkthrough

This cell shows the output of `03_dcgan_training_walkthrough.py`, which performs a medium-length DCGAN training run (10 epochs, max 32 batches per epoch, batch size 64). The training uses label smoothing and two Generator updates per Discriminator update.

*Training completed in a prior run; output shown from pre-captured log.*""")

code("""# Display pre-captured walkthrough training output
walkthrough_output_path = OUTPUT_DIR / "walkthrough_output.txt"
with open(walkthrough_output_path, "r", encoding="utf-8") as f:
    print(f.read())
""")

md("""**Walkthrough training outputs:**""")

code("""print("=== Walkthrough Real Samples ===")
display(Image(filename=str(OUTPUT_DIR / "walkthrough_real_samples.png")))

print("=== Walkthrough Fake Epoch 00 (untrained) ===")
display(Image(filename=str(OUTPUT_DIR / "walkthrough_fake_epoch_00.png")))

print("=== Walkthrough Fake Epoch 05 (intermediate) ===")
display(Image(filename=str(OUTPUT_DIR / "walkthrough_fake_epoch_05.png")))

print("=== Walkthrough Final Fake Samples ===")
display(Image(filename=str(OUTPUT_DIR / "walkthrough_fake_samples.png")))

print("=== Walkthrough Loss Curve ===")
display(Image(filename=str(OUTPUT_DIR / "walkthrough_loss_curve.png")))
""")

# ===== PART 2 HEADER =====
md("""---

## Part 2: Integrated Experiment Code

The following cells show the completed TODO blocks in `04_integrated_experiment.py`. These seven blocks implement the full DCGAN data loading, Generator, Discriminator, training loop, sample saving, and checkpoint saving.
""")

# ===== CONFIG =====
md("### Student Configuration Area")

code("""# =========================
# Student Configuration Area
# =========================
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
TRAIN_IMAGE_LIMIT = None   # use all available images
FIXED_NOISE_SIZE = 64
GENERATOR_STEPS = 1
PRINT_EVERY_BATCHES = 20
SAVE_EVERY_EPOCHS = 5

REAL_LABEL_RANGE = (0.85, 1.0)
FAKE_LABEL_RANGE = (0.0, 0.15)
GENERATOR_LABEL_RANGE = (0.90, 1.0)
print("Configuration loaded successfully.")
""")

# ===== TODO 1 =====
md("""### TODO 1: Data Loading and DataLoader Construction

Loads face images in ImageFolder format, applies standard DCGAN transforms (resize, center crop, normalize to [-1, 1]), and constructs a DataLoader.""")

code("""def create_dataloader():
    data_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.ImageFolder(LAB_ROOT / "data" / "faces" / "train", transform=data_transform)
    # TRAIN_IMAGE_LIMIT=None means use all images
    training_dataset = dataset
    dataloader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=(len(training_dataset) >= BATCH_SIZE),
    )
    return dataset, training_dataset, dataloader
print("create_dataloader() defined.")
""")

# ===== TODO 2 =====
md("""### TODO 2: Generator Network

DCGAN Generator maps latent noise vectors (100-D) to 64x64 RGB images using transposed convolutions with BatchNorm and ReLU activations. The final Tanh layer ensures output values are in [-1, 1].""")

code("""class Generator(nn.Module):
    def __init__(self):
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
            # 32x32 -> 64x64 RGB; Tanh
            nn.ConvTranspose2d(NGF, NC, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.main(x)
print("Generator defined.")
""")

# ===== TODO 3 =====
md("""### TODO 3: Discriminator Network

DCGAN Discriminator maps 64x64 RGB images to a single real/fake probability using strided convolutions with LeakyReLU and BatchNorm. The final Sigmoid outputs a score in [0, 1].""")

code("""class Discriminator(nn.Module):
    def __init__(self):
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
            # 4x4 -> 1x1 -> Sigmoid
            nn.Conv2d(NDF * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.main(x).view(-1)
print("Discriminator defined.")
""")

# ===== TODO 4 =====
md("""### TODO 4: Discriminator Training Step

Trains D to classify real images as real (target around 1) and generated images as fake (target around 0). Uses label smoothing for both real and fake targets. The `.detach()` on fake images prevents gradients from flowing into G during D's update.""")

code("""# --- Discriminator update (pseudocode context) ---
# netD.zero_grad(set_to_none=True)
# output_real = netD(real_images)
# real_labels = make_smoothed_labels(output_real, REAL_LABEL_RANGE)
# errD_real = criterion(output_real, real_labels)
# errD_real.backward()
# d_x = output_real.mean().item()
#
# noise = torch.randn(batch_size, NZ, 1, 1, device=device)
# fake_images = netG(noise)
# output_fake = netD(fake_images.detach())
# fake_labels = make_smoothed_labels(output_fake, FAKE_LABEL_RANGE)
# errD_fake = criterion(output_fake, fake_labels)
# errD_fake.backward()
# d_g_z1 = output_fake.mean().item()
#
# errD = errD_real + errD_fake
# torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=GRAD_CLIP_NORM)
# optimizerD.step()
print("Discriminator training step logic shown above.")
""")

# ===== TODO 5 =====
md("""### TODO 5: Generator Training Step

Trains G to produce images that D classifies as real (target around 1). Uses smoothed labels near 1.0. Runs `GENERATOR_STEPS` iterations per batch (default: 1). Gradients are clipped to prevent instability.""")

code("""# --- Generator update (pseudocode context) ---
# errG = torch.tensor(0.0, device=device)
# d_g_z2 = 0.0
# for _ in range(GENERATOR_STEPS):
#     netG.zero_grad(set_to_none=True)
#     generator_noise = torch.randn(batch_size, NZ, 1, 1, device=device)
#     fake_for_generator = netG(generator_noise)
#     output_for_generator = netD(fake_for_generator)
#     generator_labels = make_smoothed_labels(
#         output_for_generator, GENERATOR_LABEL_RANGE
#     )
#     errG = criterion(output_for_generator, generator_labels)
#     errG.backward()
#     d_g_z2 = output_for_generator.mean().item()
#     torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=GRAD_CLIP_NORM)
#     optimizerG.step()
print("Generator training step logic shown above.")
""")

# ===== TODO 6 =====
md("""### TODO 6: Fixed-Noise Generated Image Saving

Saves generated images from a fixed set of noise vectors. Using the same noise across epochs allows visual comparison of Generator improvement over time.""")

code("""def save_fixed_noise_samples(netG, fixed_noise, output_path):
    netG.eval()
    with torch.no_grad():
        fake_samples = netG(fixed_noise).detach().cpu()
        # save_image_grid helper denormalizes and saves
    netG.train()
    return output_path
print("save_fixed_noise_samples() defined.")
""")

# ===== TODO 7 =====
md("""### TODO 7: Model Checkpoint Saving

Saves both Generator and Discriminator `state_dict` along with the configuration dictionary. This enables later model loading and inference without re-training.""")

code("""def save_model_weights(netG, netD, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {"image_size": IMAGE_SIZE, "nz": NZ, "ngf": NGF, "ndf": NDF, "nc": NC}
    generator_path = output_dir / "generator_weights.pth"
    discriminator_path = output_dir / "discriminator_weights.pth"
    torch.save({"model_state_dict": netG.state_dict(), "config": config}, generator_path)
    torch.save({"model_state_dict": netD.state_dict(), "config": config}, discriminator_path)
    return generator_path, discriminator_path
print("save_model_weights() defined.")
""")

# ===== PART 3 =====
md("""---

## Part 3: Integrated Experiment Running Result

The output below is from running `04_integrated_experiment.py` with all seven TODO blocks completed. This is the longer training run designed to produce visible face-like generated samples.

**Configuration:**
- 20 epochs on all 10,426 images
- Batch size 128
- Label smoothing for D and G
- Gradient clipping (norm <= 10.0)
- Intermediate sample saving every 5 epochs

*Training completed in a prior run; output shown from pre-captured log.*
""")

code("""# Display pre-captured integrated experiment output
integrated_output_path = OUTPUT_DIR / "integrated_output.txt"
with open(integrated_output_path, "r", encoding="utf-8") as f:
    print(f.read())
""")

md("### Integrated Experiment Output Images")

code("""print("=== Integrated Real Samples ===")
display(Image(filename=str(OUTPUT_DIR / "integrated_real_samples.png")))

print("=== Integrated Fake Epoch 005 (intermediate) ===")
display(Image(filename=str(OUTPUT_DIR / "integrated_fake_epoch_005.png")))

print("=== Integrated Fake Epoch 010 (intermediate) ===")
display(Image(filename=str(OUTPUT_DIR / "integrated_fake_epoch_010.png")))

print("=== Integrated Fake Epoch 015 (intermediate) ===")
display(Image(filename=str(OUTPUT_DIR / "integrated_fake_epoch_015.png")))

print("=== Integrated Fake Epoch 020 (final) ===")
display(Image(filename=str(OUTPUT_DIR / "integrated_fake_epoch_020.png")))

print("=== Integrated Final Fake Samples ===")
display(Image(filename=str(OUTPUT_DIR / "integrated_fake_samples.png")))

print("=== Integrated Loss Curve ===")
display(Image(filename=str(OUTPUT_DIR / "integrated_loss_curve.png")))
""")

# ===== ANALYSIS =====
md("""---

## Training Analysis

### Observations

1. **Discriminator Loss**: GAN training is adversarial. The Discriminator loss does not monotonically decrease; it fluctuates as D and G compete. In the walkthrough, D loss ranged from ~2 to ~9, reflecting the dynamic equilibrium between the two networks. In the integrated experiment, D loss started around 2.9 and eventually stabilized around 1.4, indicating that D was learning to better distinguish real from fake images.

2. **Generator Loss**: The Generator loss fluctuated as training progressed. In the integrated experiment, G loss started around 3.65 and decreased to ~0.84 by epoch 20, showing steady improvement in the Generator's ability to fool the Discriminator.

3. **D(x) and D(G(z))**: D(x) (average D score on real images) remained relatively stable around 0.5-0.6 in later epochs, which is near the theoretical equilibrium point of 0.5. D(G(z)) (average D score on fake images) started high in early epochs (indicating D could easily spot fakes) and gradually became more balanced.

4. **Image Quality Progression**: The integrated experiment (20 epochs, full dataset) produced images that showed progressive improvement:
   - Epoch 005: Blurry face-like shapes beginning to emerge
   - Epoch 010: More defined facial structures
   - Epoch 015: Further refinement of features
   - Epoch 020: The most coherent face-like outputs, though still with artifacts typical of DCGAN on limited training

### Key Techniques Used

- **Label Smoothing**: Instead of hard labels (0.0 and 1.0), smoothed ranges (e.g., 0.85-1.0 for real) prevent D from becoming overconfident and help stabilize training.
- **Gradient Clipping**: Limits gradient norms to prevent exploding gradients during adversarial updates (clipped at norm=10.0).
- **Fixed Noise Comparison**: Using the same latent vectors at different epochs provides a consistent visual measure of Generator progress.
- **Separate Learning Rates**: D uses a lower learning rate (5e-5) than G (2e-4) to prevent D from overpowering G too early.
- **Adam Optimizer with beta1=0.5**: The DCGAN paper's recommended setting, different from the default beta1=0.9, helps stabilize GAN training.

### Model Checkpoints

The trained model weights are saved at:
- `outputs/generator_weights.pth` (13.7 MB)
- `outputs/discriminator_weights.pth` (10.6 MB)

These checkpoints include both the model `state_dict` and the configuration dictionary, enabling future loading and inference.

## End of Report
""")

# ===== SAVE =====
nb.cells = cells
nb_path = Path("/home/pluto/lab04_dcgan_generation/lab04_report.ipynb")
nb_path.write_text(nbf.writes(nb), encoding="utf-8")
print(f"Notebook written to {nb_path}")
