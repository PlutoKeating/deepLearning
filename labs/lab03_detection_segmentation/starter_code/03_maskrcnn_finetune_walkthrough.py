"""Trainable Mask R-CNN fine-tuning walkthrough on Penn-Fudan.

This file is a step-by-step tutorial for fine-tuning Mask R-CNN on the
Penn-Fudan pedestrian dataset. It shows the complete workflow used in a small
object detection and instance segmentation experiment:

1. read RGB images and instance masks from the Penn-Fudan dataset;
2. convert mask pixel ids into the target dictionary required by Mask R-CNN;
3. build a TorchVision Mask R-CNN model and replace its prediction heads;
4. train the model for several short epochs on CPU or GPU;
5. use confidence scores to filter predictions during inference; and
6. save ground-truth, prediction, and model-weight files for inspection.

The default settings are designed for classroom use, so the script can run on
CPU with a small subset of images. To train longer without editing the file,
override the environment variables:

    $env:MASKRCNN_EPOCHS = "5"
    $env:MASKRCNN_TRAIN_SIZE = "48"
    $env:MASKRCNN_SCORE_THRESHOLD = "0.3"
    python 03_maskrcnn_finetune_walkthrough.py
"""

from pathlib import Path
import os

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.io import write_png
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import masks_to_boxes
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes


LAB_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = LAB_ROOT / "data" / "PennFudanPed"
OUTPUT_DIR = LAB_ROOT / "outputs"
PREDICTION_IMAGE_NAME = "walkthrough_prediction.png"
GROUND_TRUTH_IMAGE_NAME = "walkthrough_ground_truth.png"
MODEL_WEIGHTS_NAME = "walkthrough_maskrcnn_weights.pth"
MISSING_DATA_MESSAGE = (
    "Please place PennFudanPed under labs/lab03_detection_segmentation/data/"
)


def env_int(name: str, default: int) -> int:
    """Read an integer setting while keeping beginner-friendly defaults."""
    value = os.environ.get(name)
    return default if value is None else int(value)


def env_float(name: str, default: float) -> float:
    """Read a float setting from the shell environment."""
    value = os.environ.get(name)
    return default if value is None else float(value)


def env_bool(name: str, default: bool) -> bool:
    """Accept common true/false strings for simple command-line experiments."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# =========================
# Student Configuration Area
# =========================
# These defaults are a compromise: much better than a one-batch demo, but still
# reasonable on CPU.  For a stronger result, increase EPOCHS and TRAIN_SUBSET_SIZE.
BATCH_SIZE = env_int("MASKRCNN_BATCH_SIZE", 2)
EPOCHS = env_int("MASKRCNN_EPOCHS", 2)
TRAIN_SUBSET_SIZE = env_int("MASKRCNN_TRAIN_SIZE", 16)
VAL_SUBSET_SIZE = env_int("MASKRCNN_VAL_SIZE", 6)
LEARNING_RATE = env_float("MASKRCNN_LR", 0.005)
MOMENTUM = env_float("MASKRCNN_MOMENTUM", 0.9)
WEIGHT_DECAY = env_float("MASKRCNN_WEIGHT_DECAY", 0.0005)
SCORE_THRESHOLD = env_float("MASKRCNN_SCORE_THRESHOLD", 0.35)
FLIP_PROBABILITY = env_float("MASKRCNN_FLIP_PROB", 0.5)
FREEZE_BACKBONE = env_bool("MASKRCNN_FREEZE_BACKBONE", True)
NUM_WORKERS = env_int("MASKRCNN_NUM_WORKERS", 0)
NUM_CLASSES = 2  # background + person
PREVIEW_SCORE_COUNT = 10


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_dataset_exists(data_dir: Path) -> bool:
    required_dirs = [data_dir / "PNGImages", data_dir / "PedMasks"]
    if not all(path.is_dir() for path in required_dirs):
        print(MISSING_DATA_MESSAGE)
        print(f"Expected dataset path: {data_dir}")
        return False
    return True


class PennFudanDataset(Dataset):
    """Penn-Fudan pedestrian dataset in TorchVision detection target format.

    Mask R-CNN does not receive one fixed-size target tensor.  Instead, each
    image has a dictionary containing boxes, labels, masks, image_id, area, and
    iscrowd.  This mirrors the official TorchVision detection tutorial.
    """

    def __init__(
        self,
        root: Path,
        training: bool = False,
        flip_probability: float = 0.0,
    ) -> None:
        self.root = root
        self.training = training
        self.flip_probability = flip_probability
        self.imgs = sorted((root / "PNGImages").glob("*.png"))
        self.masks = sorted((root / "PedMasks").glob("*.png"))

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img = Image.open(self.imgs[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])

        # Mask R-CNN expects image tensors as float32 in [0, 1], not PIL images.
        img = F.convert_image_dtype(F.pil_to_tensor(img), dtype=torch.float)
        mask = torch.as_tensor(np.array(mask), dtype=torch.uint8)

        # In Penn-Fudan masks, pixel value 0 is background.  Each nonzero value
        # is a different pedestrian instance.
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]

        # A horizontal flip must be applied to both the image and masks.  Boxes
        # are recomputed from masks after the flip, which avoids box math errors.
        if (
            self.training
            and self.flip_probability > 0
            and torch.rand(()).item() < self.flip_probability
        ):
            img = F.hflip(img)
            masks = torch.flip(masks, dims=[2])

        if len(obj_ids) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = masks_to_boxes(masks)

        # Penn-Fudan has one foreground class: person.  Label 0 is reserved for
        # background by TorchVision detection models, so all people use label 1.
        labels = torch.ones((len(obj_ids),), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(obj_ids),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }
        return img, target


def collate_fn(batch):
    # Detection batches contain variable numbers of objects per image.  The
    # default PyTorch collate function tries to stack targets, so we keep lists.
    return tuple(zip(*batch))


def split_indices(total_size: int) -> tuple[list[int], list[int]]:
    """Create a deterministic train/validation split for reproducible output."""
    if total_size < 2:
        raise ValueError("PennFudanPed needs at least two samples for this lab.")

    train_size = min(TRAIN_SUBSET_SIZE, total_size - 1)
    val_size = min(VAL_SUBSET_SIZE, total_size - train_size)
    if val_size <= 0:
        train_size = total_size - 1
        val_size = 1

    generator = torch.Generator().manual_seed(0)
    indices = torch.randperm(total_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    return train_indices, val_indices


def get_model(num_classes: int) -> nn.Module:
    try:
        # COCO-pretrained Mask R-CNN gives much better starting features than a
        # randomly initialized detector.  If the weights are unavailable, the
        # script still runs, but the prediction quality will be much lower.
        model = maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
            weights_backbone=None,
        )
        print("Loaded COCO-pretrained Mask R-CNN weights.")
    except Exception as exc:
        print(f"Warning: pretrained Mask R-CNN weights could not be loaded: {exc}")
        print("Falling back to weights=None. Train longer for usable predictions.")
        model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    # Replace the COCO box classifier with a new classifier for this lab:
    # class 0 = background, class 1 = person.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor so it outputs one mask channel per lab class.
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes,
    )

    if FREEZE_BACKBONE:
        # For a short CPU lab, training the new detection heads while freezing
        # the feature extractor is faster and usually gives cleaner early output.
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False

    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: str,
    epoch: int,
) -> float:
    """Run one full training epoch and return the average total loss."""
    model.train()
    running_loss = 0.0

    for batch_index, (images, targets) in enumerate(data_loader):
        # Detection models receive a list of image tensors and a list of target
        # dictionaries.  Move every tensor to the selected device.
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        # In training mode, TorchVision Mask R-CNN returns a dictionary of loss
        # components instead of predictions.
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        if not torch.isfinite(losses):
            raise RuntimeError(f"Non-finite loss encountered: {losses.item()}")

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        running_loss += losses.item()
        loss_items = ", ".join(
            f"{key}: {value.item():.4f}" for key, value in loss_dict.items()
        )
        print(
            f"Epoch {epoch + 1}/{EPOCHS}, batch {batch_index + 1}/{len(data_loader)}, "
            f"total loss: {losses.item():.4f}"
        )
        print(f"  Loss components: {loss_items}")

    return running_loss / max(len(data_loader), 1)


def draw_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: list[str],
    output_path: Path,
    color: str,
) -> None:
    """Draw boxes on an image tensor and save a PNG artifact."""
    image_uint8 = (image * 255).to(torch.uint8).cpu()
    boxes = boxes.detach().cpu()

    if len(boxes) > 0:
        drawn = draw_bounding_boxes(
            image_uint8,
            boxes,
            labels=labels,
            colors=color,
            width=3,
        )
    else:
        drawn = image_uint8

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_png(drawn, str(output_path))


def select_demo_sample(dataset: Dataset) -> tuple[int, torch.Tensor, dict[str, torch.Tensor]]:
    """Choose a validation image with the most people for a useful visualization."""
    best_index = 0
    best_count = -1

    for index in range(len(dataset)):
        _, target = dataset[index]
        object_count = len(target["boxes"])
        if object_count > best_count:
            best_index = index
            best_count = object_count

    image, target = dataset[best_index]
    return best_index, image, target


def inference_and_visualize(model: nn.Module, dataset: Dataset, device: str) -> tuple[Path, Path]:
    model.eval()
    demo_index, image, target = select_demo_sample(dataset)
    image_on_device = image.to(device)

    with torch.no_grad():
        # In eval mode, Mask R-CNN returns predictions with boxes, labels, scores,
        # and masks.  Scores are sorted from high to low by TorchVision.
        prediction = model([image_on_device])[0]

    gt_boxes = target["boxes"].detach().cpu()
    gt_labels = [f"gt person {index + 1}" for index in range(len(gt_boxes))]
    ground_truth_path = OUTPUT_DIR / GROUND_TRUTH_IMAGE_NAME
    draw_boxes(image, gt_boxes, gt_labels, ground_truth_path, color="green")

    boxes = prediction["boxes"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    predicted_labels = prediction["labels"].detach().cpu()

    print(f"Validation demo sample index: {demo_index}")
    print(f"Ground-truth persons in demo image: {len(gt_boxes)}")
    print(f"Predicted boxes before threshold: {len(boxes)}")
    print(f"Score threshold: {SCORE_THRESHOLD}")
    if len(scores) > 0:
        preview_scores = scores[: min(PREVIEW_SCORE_COUNT, len(scores))]
        rounded_scores = [round(score, 4) for score in preview_scores.tolist()]
        print(f"Top prediction scores: {rounded_scores}")

    # Keep person predictions with high enough confidence.  The class check is
    # educational here: with two classes, valid foreground predictions use label 1.
    keep = (predicted_labels == 1) & (scores >= SCORE_THRESHOLD)
    filtered_boxes = boxes[keep]
    filtered_scores = scores[keep]

    print(f"Predicted boxes after threshold: {len(filtered_boxes)}")
    if len(filtered_scores) > 0:
        print(f"Highest kept score: {filtered_scores.max().item():.4f}")
        print(f"Lowest kept score: {filtered_scores.min().item():.4f}")
    else:
        print("No boxes kept. Try lowering MASKRCNN_SCORE_THRESHOLD for inspection.")

    prediction_labels = [
        f"person {score:.2f}" for score in filtered_scores.detach().cpu().tolist()
    ]
    prediction_path = OUTPUT_DIR / PREDICTION_IMAGE_NAME
    draw_boxes(image, filtered_boxes, prediction_labels, prediction_path, color="red")

    return ground_truth_path, prediction_path


def save_model_weights(model: nn.Module) -> Path:
    output_path = OUTPUT_DIR / MODEL_WEIGHTS_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    return output_path


def main() -> None:
    torch.manual_seed(0)

    print("=== 03 Mask R-CNN Fine-tuning Walkthrough ===")
    print(f"Dataset path: {DATA_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Train subset size: {TRAIN_SUBSET_SIZE}")
    print(f"Validation subset size: {VAL_SUBSET_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Score threshold: {SCORE_THRESHOLD}")
    print(f"Flip probability: {FLIP_PROBABILITY}")
    print(f"Freeze backbone: {FREEZE_BACKBONE}")
    print(f"DataLoader workers: {NUM_WORKERS}")

    if not check_dataset_exists(DATA_DIR):
        return

    device = get_device()
    print(f"Using {device} device")

    train_base = PennFudanDataset(
        DATA_DIR,
        training=True,
        flip_probability=FLIP_PROBABILITY,
    )
    val_base = PennFudanDataset(DATA_DIR, training=False)
    train_indices, val_indices = split_indices(len(train_base))
    train_dataset = Subset(train_base, train_indices)
    val_dataset = Subset(val_base, val_indices)

    print(f"Actual train images: {len(train_dataset)}")
    print(f"Actual validation images: {len(val_dataset)}")

    train_generator = torch.Generator().manual_seed(0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
        num_workers=NUM_WORKERS,
        pin_memory=device == "cuda",
        collate_fn=collate_fn,
    )

    model = get_model(NUM_CLASSES).to(device)
    total_parameters, trainable_parameters = count_parameters(model)
    print("Modified box predictor:")
    print(model.roi_heads.box_predictor)
    print("Modified mask predictor:")
    print(model.roi_heads.mask_predictor)
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(EPOCHS // 2, 1),
        gamma=0.1,
    )

    print("\nStep 1 - fine-tuning")
    print("=" * 40)
    for epoch in range(EPOCHS):
        average_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1} average loss: {average_loss:.4f}; "
            f"next learning rate: {current_lr:.6f}"
        )

    print("\nStep 2 - inference and visualization")
    print("=" * 40)
    ground_truth_path, prediction_path = inference_and_visualize(
        model,
        val_dataset,
        device,
    )

    print("\nStep 3 - save trained weights")
    print("=" * 40)
    weights_path = save_model_weights(model)
    print(f"Saved ground-truth visualization path: {ground_truth_path}")
    print(f"Saved prediction path: {prediction_path}")
    print(f"Saved model weights path: {weights_path}")
    print("Done!")


if __name__ == "__main__":
    main()
