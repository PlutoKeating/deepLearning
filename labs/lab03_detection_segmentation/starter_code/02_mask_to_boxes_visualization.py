"""Visualize Penn-Fudan masks converted to bounding boxes."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import write_png
from torchvision.ops import masks_to_boxes
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


LAB_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = LAB_ROOT / "data" / "PennFudanPed"
OUTPUT_DIR = LAB_ROOT / "outputs"
MISSING_DATA_MESSAGE = (
    "Please place PennFudanPed under labs/lab03_detection_segmentation/data/"
)


def check_dataset_exists(data_dir: Path) -> bool:
    required_dirs = [data_dir / "PNGImages", data_dir / "PedMasks"]
    if not all(path.is_dir() for path in required_dirs):
        print(MISSING_DATA_MESSAGE)
        print(f"Expected dataset path: {data_dir}")
        return False
    return True


class PennFudanDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.root = root
        # The official tutorial pairs each image with a mask of instance ids.
        self.imgs = sorted((root / "PNGImages").glob("*.png"))
        self.masks = sorted((root / "PedMasks").glob("*.png"))

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img = Image.open(self.imgs[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])

        image_uint8 = F.pil_to_tensor(img)
        mask = torch.as_tensor(np.array(mask), dtype=torch.uint8)

        # Drop background id 0; every remaining id becomes one binary mask.
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        # masks_to_boxes gives one [xmin, ymin, xmax, ymax] box per instance mask.
        boxes = masks_to_boxes(masks)
        labels = torch.ones((len(obj_ids),), dtype=torch.int64)

        # The same target fields are used later by Mask R-CNN.
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(obj_ids),), dtype=torch.int64),
        }
        return image_uint8, target


def main() -> None:
    print("=== 02 Mask-to-Boxes Visualization ===")
    print(f"Dataset path: {DATA_DIR}")

    if not check_dataset_exists(DATA_DIR):
        return

    dataset = PennFudanDataset(DATA_DIR)
    if len(dataset) == 0:
        print("No images were found in the dataset.")
        return

    image, target = dataset[0]
    boxes = target["boxes"]
    masks = target["masks"]

    # Draw boxes and masks to make the target conversion visible for screenshots.
    labels = [f"person {idx + 1}" for idx in range(len(boxes))]
    drawn = draw_bounding_boxes(
        image,
        boxes,
        labels=labels,
        colors="red",
        width=3,
    )
    if len(masks) > 0:
        drawn = draw_segmentation_masks(
            drawn,
            masks=masks,
            alpha=0.25,
            colors=["blue"] * len(masks),
        )

    # Saving the image gives students a concrete artifact to inspect.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "sample_boxes.png"
    write_png(drawn, str(output_path))

    print(f"Number of objects in sample: {len(boxes)}")
    print(f"Boxes shape: {boxes.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Saved visualization path: {output_path}")


if __name__ == "__main__":
    main()
