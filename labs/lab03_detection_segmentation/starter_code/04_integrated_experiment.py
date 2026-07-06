"""Integrated Lab 03 experiment on PennFudanPed.

This script is the main student task for Lab 03.  Use the walkthrough scripts
00-03 as references, then complete the seven TODO blocks in this file.

The integrated experiment connects dataset target construction, Mask R-CNN head
replacement, short fine-tuning, validation sample selection, threshold
diagnostics, detection visualization, instance segmentation visualization, and
model-weight saving.

After completing the TODO blocks, run:
    python 04_integrated_experiment.py
"""

from pathlib import Path

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
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


# =========================
# Student Configuration Area
# You are encouraged to adjust the hyperparameters and observe how they affect the results.
# =========================
BATCH_SIZE = 2
EPOCHS = 2
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
TRAIN_SUBSET_SIZE = 40
VAL_SUBSET_SIZE = 10
NUM_CLASSES = 2
SCORE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
THRESHOLD_VALUES = (0.25, 0.35, 0.50)
FREEZE_BACKBONE = True
MODEL_TO_SAVE = "maskrcnn"


LAB_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = LAB_ROOT / "data" / "PennFudanPed"
OUTPUT_DIR = LAB_ROOT / "outputs"
PREDICTION_IMAGE_NAME = "integrated_prediction.png"
SEGMENTATION_IMAGE_NAME = "integrated_segmentation.png"
GROUND_TRUTH_IMAGE_NAME = "integrated_ground_truth.png"
MODEL_WEIGHTS_NAME = "model_weights.pth"
TARGET_KEYS = ("boxes", "labels", "masks", "image_id", "area", "iscrowd")
MISSING_DATA_MESSAGE = (
    "Please place PennFudanPed under labs/lab03_detection_segmentation/data/"
)


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


def get_transform():
    # The official detection model expects float image tensors in [0, 1].
    return lambda image: F.convert_image_dtype(F.pil_to_tensor(image), dtype=torch.float)


def summarize_target(image: torch.Tensor, target: dict[str, torch.Tensor]) -> None:
    """Print a short checkpoint for the custom detection target format."""
    print("\nDataset target checkpoint")
    print("-" * 30)
    print(f"Image tensor shape: {image.shape}")
    print(f"Target keys: {list(target.keys())}")

    missing_keys = [key for key in TARGET_KEYS if key not in target]
    if missing_keys:
        print(f"Missing target keys: {missing_keys}")
        return

    print(f"Boxes shape: {target['boxes'].shape}")
    print(f"Labels: {target['labels']}")
    print(f"Masks shape: {target['masks'].shape}")
    print(f"image_id: {target['image_id']}")
    print(f"Area shape: {target['area'].shape}")
    print(f"iscrowd shape: {target['iscrowd'].shape}")


class PennFudanDataset(Dataset):
    def __init__(self, root: Path, transform=None) -> None:
        self.root = root
        self.transform = transform
        # Penn-Fudan stores RGB images and instance masks in separate folders.
        self.imgs = sorted((root / "PNGImages").glob("*.png"))
        self.masks = sorted((root / "PedMasks").glob("*.png"))

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = F.convert_image_dtype(F.pil_to_tensor(img), dtype=torch.float)

        mask = torch.as_tensor(np.array(mask), dtype=torch.uint8)
        # Nonzero mask ids correspond to object instances.
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]

        # TODO 1: Build the target dictionary for one image.
        # Learning goal:
        # - connect instance masks to detection boxes and training metadata
        # Reference:
        # - compare with 01_dataset_target_format.py and 03_maskrcnn_finetune_walkthrough.py
        # Check after completion:
        # - boxes shape should be [num_objects, 4]
        # - masks shape should be [num_objects, height, width]
        # - target should contain every key in TARGET_KEYS
        # Steps to complete:
        # - convert instance masks to bounding boxes
        # - create one person label for each object
        # - create image_id, area, and iscrowd tensors
        # - put all six fields into target
        raise NotImplementedError("Task 1 is not complete: build the target dictionary.")

        return img, target


def collate_fn(batch):
    # Variable-size targets require a custom collate function for detection batches.
    return tuple(zip(*batch))


def split_dataset(dataset: Dataset) -> tuple[Subset, Subset]:
    total_size = len(dataset)
    if total_size < 2:
        raise ValueError("PennFudanPed needs at least two samples for this lab.")

    train_size = min(TRAIN_SUBSET_SIZE, total_size - 1)
    val_size = min(VAL_SUBSET_SIZE, total_size - train_size)
    if val_size <= 0:
        train_size = total_size - 1
        val_size = 1

    indices = list(range(total_size))
    # Keep the split deterministic so screenshots and runtimes are predictable.
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def get_model(num_classes: int) -> nn.Module:
    try:
        # Start from TorchVision's Mask R-CNN model, then adapt its predictors.
        model = maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
            weights_backbone=None,
        )
    except Exception as exc:
        print(f"Warning: pretrained Mask R-CNN weights could not be loaded: {exc}")
        print("Falling back to weights=None.")
        model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    original_box_predictor = model.roi_heads.box_predictor
    original_mask_predictor = model.roi_heads.mask_predictor
    print("\nOriginal Mask R-CNN predictors")
    print("-" * 30)
    print("Original box predictor:")
    print(original_box_predictor)
    print("Original mask predictor:")
    print(original_mask_predictor)

    # TODO 2: Replace the Mask R-CNN box predictor and mask predictor.
    # Learning goal:
    # - keep the pretrained Mask R-CNN body but adapt the two prediction heads
    # Reference:
    # - compare with 03_maskrcnn_finetune_walkthrough.py
    # Check after completion:
    # - the modified box and mask predictors should print NUM_CLASSES outputs
    # Steps to complete:
    # - read the input feature size from the existing box predictor
    # - replace roi_heads.box_predictor with FastRCNNPredictor
    # - read the input channel size from the existing mask predictor
    # - replace roi_heads.mask_predictor with MaskRCNNPredictor
    raise NotImplementedError("Task 2 is not complete: replace model predictors.")

    # TODO 3: Freeze the backbone when FREEZE_BACKBONE is True.
    # Learning goal:
    # - improve short CPU fine-tuning by training fewer parameters
    # Check after completion:
    # - when FREEZE_BACKBONE is True, backbone parameters should have requires_grad=False
    # - the printed trainable parameter count should be smaller than total parameters
    # Steps to complete:
    # - if FREEZE_BACKBONE is True, loop over model.backbone.parameters()
    # - set parameter.requires_grad = False for each backbone parameter
    raise NotImplementedError("Task 3 is not complete: freeze the backbone.")

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
    model.train()
    running_loss = 0.0

    for batch_index, (images, targets) in enumerate(data_loader):
        # Move each image and each target tensor because detection inputs are lists.
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        # In training mode, Mask R-CNN returns a loss dictionary.
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
        print(f"Loss values: {loss_items}")

    return running_loss / max(len(data_loader), 1)


def draw_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: list[str],
    color: str,
) -> torch.Tensor:
    image_uint8 = (image * 255).to(torch.uint8).cpu()
    if len(boxes) == 0:
        return image_uint8
    return draw_bounding_boxes(
        image_uint8,
        boxes.detach().cpu(),
        labels=labels,
        colors=color,
        width=3,
    )


def draw_masks(image: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    image_uint8 = (image * 255).to(torch.uint8).cpu()
    if len(masks) == 0:
        return image_uint8
    return draw_segmentation_masks(
        image_uint8,
        masks.detach().cpu(),
        alpha=0.45,
    )


def select_demo_sample(dataset: Dataset) -> tuple[int, torch.Tensor, dict[str, torch.Tensor]]:
    # TODO 4: Select the validation sample with the most ground-truth persons.
    # Learning goal:
    # - choose a useful image for visual inspection instead of always using dataset[0]
    # Check after completion:
    # - printed "Selected validation sample index" should be a valid dataset index
    # - printed "Ground-truth persons in selected image" should be the largest count
    # Steps to complete:
    # - loop over every validation sample
    # - count len(target["boxes"]) for each sample
    # - remember the index with the largest count
    # - return best_index, image, target for that sample
    raise NotImplementedError("Task 4 is not complete: select a validation sample.")


def summarize_threshold_counts(
    scores: torch.Tensor,
    thresholds: tuple[float, ...],
) -> list[tuple[float, int]]:
    # TODO 5: Count how many predictions remain under each score threshold.
    # Learning goal:
    # - observe how confidence thresholds control the number of displayed boxes
    # Check after completion:
    # - the returned list should contain one (threshold, count) pair per threshold
    # - larger thresholds should usually keep fewer or equal boxes
    # Steps to complete:
    # - create an empty list
    # - for each threshold, count scores >= threshold
    # - append (threshold, count) to the list
    # - return the list
    raise NotImplementedError("Task 5 is not complete: compare score thresholds.")


def inference_and_visualize(
    model: nn.Module,
    dataset: Dataset,
    device: str,
) -> tuple[Path, Path, Path]:
    model.eval()
    selected_index, image, target = select_demo_sample(dataset)
    image_on_device = image.to(device)

    with torch.no_grad():
        # In eval mode, Mask R-CNN returns prediction dictionaries.
        prediction = model([image_on_device])[0]

    gt_boxes = target["boxes"].detach().cpu()
    gt_labels = [f"gt person {index + 1}" for index in range(len(gt_boxes))]
    ground_truth_path = OUTPUT_DIR / GROUND_TRUTH_IMAGE_NAME
    ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_drawn = draw_boxes(image, gt_boxes, gt_labels, color="green")
    write_png(ground_truth_drawn, str(ground_truth_path))

    boxes = prediction["boxes"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    masks = prediction["masks"].detach().cpu()
    print(f"Selected validation sample index: {selected_index}")
    print(f"Ground-truth persons in selected image: {len(gt_boxes)}")
    print(f"Predicted boxes before threshold: {len(boxes)}")
    print(f"Predicted masks shape: {masks.shape}")
    print(f"Score threshold: {SCORE_THRESHOLD}")
    print(f"Mask threshold: {MASK_THRESHOLD}")
    if len(scores) > 0:
        preview_scores = scores[: min(5, len(scores))]
        print(f"First prediction scores: {preview_scores.tolist()}")

    threshold_counts = summarize_threshold_counts(scores, THRESHOLD_VALUES)
    print("Threshold comparison:")
    for threshold, kept_count in threshold_counts:
        print(f"threshold {threshold:.2f}: {kept_count} boxes kept")

    # TODO 6: Filter predictions and save detection plus segmentation images.
    # Learning goal:
    # - connect score filtering to detection boxes and segmentation masks
    # Reference:
    # - compare with 03_maskrcnn_finetune_walkthrough.py
    # Check after completion:
    # - len(filtered_boxes) should be less than or equal to len(boxes)
    # - the printed path should end with PREDICTION_IMAGE_NAME
    # - the segmentation path should end with SEGMENTATION_IMAGE_NAME
    # - the prediction image should appear under OUTPUT_DIR
    # - the segmentation image should appear under OUTPUT_DIR
    # Steps to complete:
    # - create keep = scores >= SCORE_THRESHOLD
    # - create filtered_boxes, filtered_scores, and filtered_masks using keep
    # - print the number of boxes after threshold
    # - print the number of masks after score threshold
    # - draw red prediction boxes with score labels
    # - save the drawn image to OUTPUT_DIR / PREDICTION_IMAGE_NAME
    # - convert filtered_masks[:, 0] to binary masks using MASK_THRESHOLD
    # - draw and save segmentation masks to OUTPUT_DIR / SEGMENTATION_IMAGE_NAME
    # - return ground_truth_path, prediction_path, and segmentation_path
    raise NotImplementedError("Task 6 is not complete: filter and save predictions.")


def main() -> None:
    torch.manual_seed(0)

    print("=== 04 Integrated Experiment ===")
    print(f"Dataset path: {DATA_DIR}")
    print(f"Train subset size: {TRAIN_SUBSET_SIZE}")
    print(f"Validation subset size: {VAL_SUBSET_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Score threshold: {SCORE_THRESHOLD}")
    print(f"Mask threshold: {MASK_THRESHOLD}")
    print(f"Threshold values: {THRESHOLD_VALUES}")
    print(f"Freeze backbone: {FREEZE_BACKBONE}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Model to save: {MODEL_TO_SAVE}")

    if not check_dataset_exists(DATA_DIR):
        return

    device = get_device()
    print(f"Device: {device}")

    dataset = PennFudanDataset(DATA_DIR, transform=get_transform())
    train_dataset, val_dataset = split_dataset(dataset)
    print(f"Actual train subset size: {len(train_dataset)}")
    print(f"Actual validation subset size: {len(val_dataset)}")

    print("\nStep 1 - custom Dataset target format")
    print("=" * 40)
    sample_image, sample_target = train_dataset[0]
    summarize_target(sample_image, sample_target)

    # The custom collate_fn preserves the list-of-images/list-of-targets format.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    print("\nStep 2 - Mask R-CNN predictor replacement and backbone freeze")
    print("=" * 40)
    model = get_model(NUM_CLASSES)
    print("Modified box predictor:")
    print(model.roi_heads.box_predictor)
    print("Modified mask predictor:")
    print(model.roi_heads.mask_predictor)
    total_parameters, trainable_parameters = count_parameters(model)
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    model = model.to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(EPOCHS // 2, 1),
        gamma=0.1,
    )

    print("\nStep 3 - short Mask R-CNN fine-tuning run")
    print("=" * 40)
    for epoch in range(EPOCHS):
        average_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1} average loss: {average_loss:.4f}; "
            f"next learning rate: {current_lr:.6f}"
        )

    print("\nStep 4 - inference threshold and visualization")
    print("=" * 40)
    # Run one selected validation image through the model and save visualizations.
    ground_truth_path, prediction_path, segmentation_path = inference_and_visualize(
        model,
        val_dataset,
        device,
    )

    print("\nStep 5 - save trained model weights")
    print("=" * 40)
    # TODO 7: Complete model saving to model_weights.pth.
    # Learning goal:
    # - save trained parameters, not the entire Python model object
    # Reference:
    # - use the same state_dict idea as Lab 01 save/load
    # Check after completion:
    # - the printed path should end with MODEL_WEIGHTS_NAME
    # - the file should appear under OUTPUT_DIR
    # Steps to complete:
    # - create save_path = OUTPUT_DIR / MODEL_WEIGHTS_NAME
    # - make sure save_path.parent exists
    # - call torch.save(model.state_dict(), save_path)
    raise NotImplementedError("Task 7 is not complete: save model weights.")

    print(f"Saved model path: {save_path}")
    print(f"Saved ground-truth visualization path: {ground_truth_path}")
    print(f"Saved prediction visualization path: {prediction_path}")
    print(f"Saved segmentation visualization path: {segmentation_path}")
    print("Done!")


if __name__ == "__main__":
    main()
