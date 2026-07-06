# Lab 02: CNN and Transfer Learning for Image Classification

Course: Deep Learning and Computer Vision

## 1. Lab Goal

In this lab, you will:

- train CNNs on FashionMNIST for CNN basics
- train ResNet18 as a feature extractor on a small real image dataset
- complete an integrated experiment on ants-vs-bees classification

Scripts `00_check_dataset.py` through `03_resnet18_feature_extractor.py` are walkthrough scripts. Script `04_integrated_experiment.py` is the main integrated task.

## 2. Official Reference

Use this official PyTorch tutorial as the main reference:

<https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html>

## 3. Data Preparation

The final integrated experiment uses the ants-vs-bees style dataset from the official PyTorch transfer learning tutorial.

Expected structure:

```text
data/hymenoptera_data/
├── train/
│   ├── ants/
│   └── bees/
└── val/
    ├── ants/
    └── bees/
```

If the data folder is missing, please download the data package 

https://download.pytorch.org/tutorial/hymenoptera_data.zip

and place it under:

```text
labs/lab02_cnn_transfer_learning/data/
```

The full expected path is:

```text
labs/lab02_cnn_transfer_learning/data/hymenoptera_data/
```

FashionMNIST can be downloaded by running `00_check_dataset.py`.

## 4. Files Provided

- `00_check_dataset.py`
- `01_cnn_baseline.py`
- `02_cnn_augmentation_regularization.py`
- `03_resnet18_feature_extractor.py`
- `04_integrated_experiment.py`

Scripts `00` to `03` are walkthrough scripts.

`04_integrated_experiment.py` is the main integrated task. It uses `hymenoptera_data`, not FashionMNIST.

## 5. Four-Period Lab Plan

Period 1: Dataset check and CNN baseline.

Period 2: CNN with augmentation and regularization.

Period 3: ResNet18 feature extractor on `hymenoptera_data`.

Period 4: Integrated experiment on `hymenoptera_data`.

## 6. Run the Lab Scripts

Open a terminal in the project root and enter:

```bash
cd labs/lab02_cnn_transfer_learning/starter_code
```

Run the walkthrough scripts:

```bash
python 00_check_dataset.py
python 01_cnn_baseline.py
python 02_cnn_augmentation_regularization.py
python 03_resnet18_feature_extractor.py
```

Then complete the TODO blocks in:

```text
04_integrated_experiment.py
```

Run the integrated experiment:

```bash
python 04_integrated_experiment.py
```

## 7. Submission

Use the same simplified PDF submission style as Lab 01.

Submit exactly one PDF file:

```text
StudentID_Name_lab02.pdf
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

Include screenshots showing that `00_check_dataset.py` to `03_resnet18_feature_extractor.py` have run.

Combined terminal screenshots are allowed.

Screenshots should show:

- dataset information
- CNN baseline result
- augmentation result
- ResNet18 feature extractor result

### Part 2: Integrated Experiment Code

Include screenshots of key completed parts of `04_integrated_experiment.py`.

Your code screenshots must show:

- Student Configuration Area
- train/val transforms
- ImageFolder DataLoaders
- completed SmallCNN
- ResNet18 feature extractor setup
- ResNet18 partial fine-tuning setup
- model comparison and saving code

Do not paste or screenshot the entire file.

### Part 3: Integrated Experiment Running Result

Include screenshots of running:

```bash
python 04_integrated_experiment.py
```

Screenshots must show:

- class names
- SmallCNN validation accuracy
- ResNet18 feature extractor validation accuracy
- ResNet18 partial fine-tuning validation accuracy
- selected saved model
- saved model path
- trainable parameter count
