# Lab 03: Object Detection and Instance Segmentation with TorchVision

Course: Deep Learning and Computer Vision

## 1. Lab Goal

In this lab, you will:

- inspect an object detection and segmentation dataset
- understand image-target pairs
- convert instance masks to bounding boxes
- fine-tune Mask R-CNN on a small pedestrian dataset
- run inference and visualize predictions

Scripts `00_check_dataset.py` through `03_maskrcnn_finetune_walkthrough.py` are walkthrough scripts. Script `03_maskrcnn_finetune_walkthrough.py` is a complete tutorial reference. Script `04_integrated_experiment.py` is the main integrated task with TODO blocks for students to complete.

## 2. Official Reference

Use this official TorchVision tutorial as the main reference:

<https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html>

## 3. Data Preparation

This lab uses the Penn-Fudan pedestrian dataset.

Expected structure:

```text
data/PennFudanPed/
├── PNGImages/
└── PedMasks/
```

If the data folder is missing, please download the data package 

https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip

and place it under:

```text
labs/lab03_detection_segmentation/data/
```

The full expected path is:

```text
labs/lab03_detection_segmentation/data/PennFudanPed/
```

## 4. Files Provided

- `00_check_dataset.py`
- `01_dataset_target_format.py`
- `02_mask_to_boxes_visualization.py`
- `03_maskrcnn_finetune_walkthrough.py`
- `04_integrated_experiment.py`

Scripts `00` to `03` are walkthrough scripts. The file `03_maskrcnn_finetune_walkthrough.py` shows a complete trainable Mask R-CNN workflow.

`04_integrated_experiment.py` is the main integrated task. It asks you to reuse ideas from the walkthrough scripts, then complete target construction, model adaptation, backbone freezing, validation sample selection, threshold comparison, detection visualization, segmentation visualization, and model saving.

## 5. Four-Period Lab Plan

Period 1: Dataset check and target format.

Period 2: Mask-to-box conversion and visualization.

Period 3: Mask R-CNN fine-tuning walkthrough.

Period 4: Integrated experiment and PDF preparation.

## 6. Run the Lab Scripts

Open a terminal in the project root and enter:

```bash
cd labs/lab03_detection_segmentation/starter_code
```

Run the walkthrough scripts:

```bash
python 00_check_dataset.py
python 01_dataset_target_format.py
python 02_mask_to_boxes_visualization.py
python 03_maskrcnn_finetune_walkthrough.py
```

Then complete the seven TODO blocks in:

```text
04_integrated_experiment.py
```

Run the integrated experiment:

```bash
python 04_integrated_experiment.py
```

## 7. Submission

Use the same simplified PDF submission style as Lab 01 and Lab 02.

Submit exactly one PDF file:

```text
StudentID_Name_lab03.pdf
```

Do not submit:

- separate screenshot files
- `model_weights.pth` separately
- multiple modified Python files
- long reports

Your PDF should contain the following three parts.

### Part 1: Walkthrough Script Running Results

Include basic information:

- Course name
- Lab name
- Student name
- Student ID
- Class
- Date

Include screenshots showing that `00_check_dataset.py` to `03_maskrcnn_finetune_walkthrough.py` have run.

Combined terminal screenshots are allowed.

Screenshots should show:

- dataset information
- target dictionary keys
- boxes and masks shape
- saved sample visualization path
- Mask R-CNN modified predictor information
- walkthrough training loss
- walkthrough prediction visualization path

### Part 2: Integrated Experiment Code

Include screenshots of key completed parts of `04_integrated_experiment.py`.

Your code screenshots must show:

- Student Configuration Area
- completed target dictionary construction
- mask-to-box conversion
- model head replacement
- backbone freeze code
- validation sample selection code
- threshold comparison code
- prediction filtering
- visualization saving code
- segmentation mask saving code
- model saving code

Do not paste or screenshot the entire file.

### Part 3: Integrated Experiment Running Result

Include screenshots of running and the images in 'outputs/':

```bash
python 04_integrated_experiment.py
```

Screenshots must show:

- train subset size
- validation subset size
- loss values
- selected validation sample index
- ground-truth persons count
- threshold comparison output
- predicted boxes before threshold
- predicted boxes after threshold
- predicted masks shape
- masks after score threshold
- saved model path

And also the images in 'outputs/': 

- integrated_ground_truth.png
- integrated_prediction.png
- integrated_segmentation.png
