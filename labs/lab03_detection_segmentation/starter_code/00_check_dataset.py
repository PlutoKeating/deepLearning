"""Check the Penn-Fudan pedestrian dataset for Lab 03."""

from pathlib import Path

import numpy as np
from PIL import Image


LAB_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = LAB_ROOT / "data" / "PennFudanPed"
IMAGE_DIR = DATA_DIR / "PNGImages"
MASK_DIR = DATA_DIR / "PedMasks"
MISSING_DATA_MESSAGE = (
    "Please place PennFudanPed under labs/lab03_detection_segmentation/data/"
)


def check_dataset_exists() -> bool:
    # The official Penn-Fudan example uses separate folders for images and masks.
    if not DATA_DIR.is_dir() or not IMAGE_DIR.is_dir() or not MASK_DIR.is_dir():
        print(MISSING_DATA_MESSAGE)
        print(f"Expected dataset path: {DATA_DIR}")
        return False
    return True


def main() -> None:
    print("=== 00 Check Penn-Fudan Dataset ===")
    print(f"Dataset path: {DATA_DIR}")

    if not check_dataset_exists():
        return

    image_files = sorted(IMAGE_DIR.glob("*.png"))
    mask_files = sorted(MASK_DIR.glob("*.png"))

    # Each image should have a matching mask file with instance ids.
    print(f"Number of images: {len(image_files)}")
    print(f"Number of masks: {len(mask_files)}")

    if not image_files or not mask_files:
        print("Dataset folders exist, but no PNG images or masks were found.")
        return

    first_image = image_files[0]
    first_mask = mask_files[0]
    print(f"First image filename: {first_image.name}")
    print(f"First mask filename: {first_mask.name}")

    # Mask pixel values identify the background and each pedestrian instance.
    image = Image.open(first_image).convert("RGB")
    mask = Image.open(first_mask)
    image_array = np.array(image)
    mask_array = np.array(mask)
    unique_mask_ids = np.unique(mask_array)

    print(f"Sample image shape: {image_array.shape}")
    print(f"Sample mask shape: {mask_array.shape}")
    print(f"Unique mask ids: {unique_mask_ids.tolist()}")


if __name__ == "__main__":
    main()
