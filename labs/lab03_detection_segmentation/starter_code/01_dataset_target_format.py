"""Show image-target pair format for detection and instance segmentation."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.ops import masks_to_boxes
from torchvision.transforms import functional as F


LAB_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = LAB_ROOT / "data" / "PennFudanPed"
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
        # Penn-Fudan stores RGB images and matching instance masks separately.
        self.imgs = sorted((root / "PNGImages").glob("*.png"))
        self.masks = sorted((root / "PedMasks").glob("*.png"))

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Detection models expect image tensors as float values in [0, 1].
        img = F.convert_image_dtype(F.pil_to_tensor(img), dtype=torch.float)
        mask = torch.as_tensor(np.array(mask), dtype=torch.uint8)

        # Unique mask ids represent background plus one id per object instance.
        obj_ids = torch.unique(mask)
        # Remove the background id so each remaining slice is one pedestrian mask.
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]

        # Convert instance masks to bounding boxes for the detection target.
        boxes = masks_to_boxes(masks)
        # Penn-Fudan has one foreground class: person.
        labels = torch.ones((len(obj_ids),), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(obj_ids),), dtype=torch.int64)

        # TorchVision detection models expect these target dictionary fields.
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }
        return img, target


def main() -> None:
    print("=== 01 Dataset Target Format ===")
    print(f"Dataset path: {DATA_DIR}")

    if not check_dataset_exists(DATA_DIR):
        return

    dataset = PennFudanDataset(DATA_DIR)
    if len(dataset) == 0:
        print("No images were found in the dataset.")
        return

    image, target = dataset[0]

    # The printed keys are the contract between Dataset and Mask R-CNN.
    print("Dataset.__getitem__ returns: image, target")
    print(f"Target dictionary keys: {list(target.keys())}")
    print(f"Image shape: {image.shape}")
    print(f"Boxes shape: {target['boxes'].shape}")
    print(f"Labels: {target['labels']}")
    print(f"Masks shape: {target['masks'].shape}")
    print(f"Area shape: {target['area'].shape}")
    print(f"iscrowd shape: {target['iscrowd'].shape}")


if __name__ == "__main__":
    main()
