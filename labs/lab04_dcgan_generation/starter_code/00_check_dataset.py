"""Check the face ImageFolder dataset for Lab 04."""

from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


# All Lab 04 scripts locate the lab root from the script path, not from the terminal path.
LAB_ROOT = Path(__file__).resolve().parents[1]
# DATA_ROOT is the ImageFolder-style dataset directory used by the training scripts.
DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "faces"
# The provided CelebA image package is usually a flat folder: image files are placed directly here.
FLAT_CELEBA_DIR = LAB_ROOT / "data" / "img_align_celeba"
# The output directory stores image grids whose paths should appear in student screenshots.
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
# The official DCGAN tutorial uses 64x64 RGB images, so this lab keeps the same size.
IMAGE_SIZE = 64
# This script only checks data, not training, so a small batch keeps execution quick.
BATCH_SIZE = 16
# num_workers=0 is the safest classroom default on Windows teaching machines.
NUM_WORKERS = 0
# Copy all available flat CelebA images into ImageFolder format.
# Set this to an integer only if a smaller debug run is needed.
PREPARE_IMAGE_LIMIT: int | None = None
# Accept only common image suffixes so archives or notes are not treated as training images.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MISSING_DATA_MESSAGE = (
    "Face images were not found. Expected either "
    "labs/lab04_dcgan_generation/data/faces/train/faces/ "
    "or labs/lab04_dcgan_generation/data/img_align_celeba/"
)


def create_transform() -> transforms.Compose:
    # The preprocessing follows the official DCGAN tutorial:
    # 1. Resize and CenterCrop make every input exactly 64x64.
    # 2. ToTensor converts a PIL image to a [C, H, W] float tensor in [0, 1].
    # 3. Normalize shifts pixels to [-1, 1], matching the Generator's Tanh output.
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def list_flat_images(image_dir: Path) -> list[Path]:
    # The flat CelebA folder has no class subfolders, so we manually filter one directory level.
    if not image_dir.is_dir():
        return []
    return [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def prepare_imagefolder_from_flat_folder() -> bool:
    # torchvision.datasets.ImageFolder expects root/class_name/image.jpg.
    # This function copies the flat CelebA folder into data/faces/train/faces/.
    imagefolder_class_dir = DATA_ROOT / "train" / "faces"
    flat_images = list_flat_images(FLAT_CELEBA_DIR)
    if not flat_images:
        return False

    imagefolder_class_dir.mkdir(parents=True, exist_ok=True)
    existing_images = list_flat_images(imagefolder_class_dir)
    if PREPARE_IMAGE_LIMIT is None:
        selected_images = flat_images
        target_description = "all available images"
    else:
        selected_images = flat_images[: min(PREPARE_IMAGE_LIMIT, len(flat_images))]
        target_description = f"first {len(selected_images)} images"

    existing_names = {path.name for path in existing_images}
    images_to_copy = [
        source_path
        for source_path in selected_images
        if source_path.name not in existing_names
    ]

    if not images_to_copy:
        print("ImageFolder images are already prepared.")
        print(f"Existing ImageFolder images: {len(existing_images)}")
        print(f"Target prepared images: {len(selected_images)} ({target_description})")
        return True

    print("Preparing ImageFolder images from flat CelebA folder.")
    print(f"Source folder: {FLAT_CELEBA_DIR}")
    print(f"Target folder: {imagefolder_class_dir}")
    print(f"Existing ImageFolder images: {len(existing_images)}")
    print(f"Target prepared images: {len(selected_images)} ({target_description})")
    print(f"Images to copy now: {len(images_to_copy)}")

    for source_path in images_to_copy:
        # copy2 preserves file metadata; it does not affect training but keeps copies faithful.
        target_path = imagefolder_class_dir / source_path.name
        shutil.copy2(source_path, target_path)

    return True


def create_dataset(data_transform):
    # ImageFolder root should be train; the faces subfolder becomes one class.
    imagefolder_root = DATA_ROOT / "train"
    imagefolder_class_dir = imagefolder_root / "faces"

    if not imagefolder_class_dir.is_dir() or not list_flat_images(imagefolder_class_dir):
        # If students only extracted img_align_celeba, this script prepares ImageFolder format.
        prepare_imagefolder_from_flat_folder()

    if imagefolder_class_dir.is_dir():
        try:
            # ImageFolder returns (image_tensor, class_index), even when there is only one class.
            dataset = datasets.ImageFolder(imagefolder_root, transform=data_transform)
        except Exception as exc:
            print(f"ImageFolder failed to load the dataset: {exc}")
            dataset = None
        if dataset is not None and len(dataset) > 0:
            return dataset, imagefolder_class_dir, "ImageFolder"

    print(MISSING_DATA_MESSAGE)
    print(f"ImageFolder path: {imagefolder_class_dir}")
    print(f"Flat CelebA path: {FLAT_CELEBA_DIR}")
    return None, None, None


def denormalize(images: torch.Tensor) -> torch.Tensor:
    # DCGAN uses [-1, 1] image tensors; save_image expects [0, 1].
    return (images * 0.5 + 0.5).clamp(0, 1)


def main() -> None:
    # Fix the random seed so shuffled DataLoader screenshots are easier to reproduce.
    torch.manual_seed(0)
    print("=== 00 Check Face Dataset ===")
    print(f"ImageFolder dataset path: {DATA_ROOT / 'train' / 'faces'}")
    print(f"Flat CelebA dataset path: {FLAT_CELEBA_DIR}")
    if PREPARE_IMAGE_LIMIT is None:
        print("Auto-prepared image count: all available flat CelebA images")
    else:
        print(f"Auto-prepared image count limit: {PREPARE_IMAGE_LIMIT}")

    data_transform = create_transform()
    # create_dataset both checks an existing ImageFolder and prepares one from the flat folder.
    dataset, dataset_path, dataset_format = create_dataset(data_transform)
    if dataset is None:
        return

    if len(dataset) == 0:
        print("Dataset folder exists, but no images were found.")
        print(MISSING_DATA_MESSAGE)
        return

    try:
        # Taking one batch verifies image decoding, transforms, and batching; it is not training.
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        images, labels = next(iter(dataloader))
    except Exception as exc:
        print(f"Dataset loading failed: {exc}")
        print(MISSING_DATA_MESSAGE)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "real_samples.png"
    # Save a real-image grid so students can verify the data and denormalization visually.
    save_image(denormalize(images), output_path, nrow=8)

    print(f"Selected dataset format: {dataset_format}")
    print(f"Selected dataset path: {dataset_path}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Class names: {dataset.classes}")
    print(f"One batch image shape: {images.shape}")
    print(f"One batch label shape: {labels.shape}")
    print(
        "Normalized value range: "
        f"min={images.min().item():.4f}, max={images.max().item():.4f}"
    )
    print(f"Saved real sample grid path: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
