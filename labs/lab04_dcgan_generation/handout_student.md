# Lab 04: DCGAN for Image Generation

Course: Deep Learning and Computer Vision

## 1. Lab Goal

In this lab, you will:

- inspect an image generation dataset
- understand the DCGAN Generator and Discriminator
- train a longer DCGAN on all prepared face images
- save real and generated image grids
- observe discriminator and generator losses during adversarial training

Scripts `00_check_dataset.py` through `03_dcgan_training_walkthrough.py` are walkthrough scripts. Script `03_dcgan_training_walkthrough.py` is a medium-length trainable DCGAN workflow for understanding the training loop. Script `04_integrated_experiment.py` is the longer integrated training script intended to produce more visible face-like generated samples.

## 2. Official Reference

Use this official PyTorch tutorial as the main reference:

<https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html>

## 3. Data Preparation

This lab uses face images from CelebA. The training scripts use ImageFolder format.

For this course, download the dataset package:

```text
File: img_align_celeba.zip
Link: https://pan.baidu.com/s/12QXFDtjfree8YR0dGFNUwA?pwd=scut
Extraction code: scut
```

Put the downloaded zip file here:

```text
labs/lab04_dcgan_generation/data/img_align_celeba.zip
```

Extract it under:

```text
labs/lab04_dcgan_generation/data/
```

After extraction, the flat CelebA image folder should be:

```text
labs/lab04_dcgan_generation/data/img_align_celeba/
```

It should contain image files directly:

```text
data/img_align_celeba/
+-- 000001.jpg
+-- 000002.jpg
+-- 000003.jpg
+-- ...
```

Then enter the starter code folder and run:

```bash
python 00_check_dataset.py
```

`00_check_dataset.py` will automatically copy all available images from the flat CelebA folder into the ImageFolder structure used by the later scripts:

```text
labs/lab04_dcgan_generation/data/faces/train/faces/
```

The original flat folder is kept unchanged. 

Final structure used by the training scripts:

```text
data/faces/
+-- train/
    +-- faces/
        +-- 000001.jpg
        +-- 000002.jpg
        +-- ...
```


## 4. Files Provided

- `00_check_dataset.py`
- `01_dcgan_generator.py`
- `02_dcgan_discriminator.py`
- `03_dcgan_training_walkthrough.py`
- `04_integrated_experiment.py`

Scripts `00` to `03` are walkthrough scripts. The file `03_dcgan_training_walkthrough.py` shows a medium-length trainable DCGAN workflow.

`04_integrated_experiment.py` is the main integrated task. Complete its seven TODO blocks to connect ImageFolder data loading, Generator construction, Discriminator construction, Discriminator training, Generator training, fixed-noise image saving, epoch sample saving, and checkpoint saving. The loss-curve saving function is already provided and will run after the TODO blocks are completed.

## 5. Four-Period Lab Plan

Period 1: Dataset check, automatic ImageFolder preparation, and real image visualization.

Period 2: Generator architecture and untrained fake image samples.

Period 3: Discriminator architecture and medium-length DCGAN training walkthrough.

Period 4: Integrated experiment TODO completion, training launch, and PDF preparation.

This lab is designed for four 45-minute periods, but the final integrated DCGAN training may run longer, especially on CPU. You may complete the TODO blocks during class, start the integrated run, and let it continue after class if needed.

## 6. Run the Lab Scripts

Open a terminal in the project root and enter:

```bash
cd labs/lab04_dcgan_generation/starter_code
```

Run the walkthrough scripts:

```bash
python 00_check_dataset.py
python 01_dcgan_generator.py
python 02_dcgan_discriminator.py
python 03_dcgan_training_walkthrough.py
```

Then complete the seven TODO blocks in:

```text
04_integrated_experiment.py
```

Run the integrated experiment:

```bash
python 04_integrated_experiment.py
```

Runtime note: the default configuration is intentionally longer than the walkthrough so the generated images have a better chance to show face-like structure. It trains on all prepared ImageFolder images, uses `NUM_EPOCHS = 20`, and saves intermediate generated grids every 5 epochs. On CPU this can still take a long time. If your computer is too slow, reduce `NUM_EPOCHS`.

Default integrated settings include:

- `BATCH_SIZE = 128`
- `NUM_EPOCHS = 20`
- `TRAIN_IMAGE_LIMIT = None`, meaning all prepared ImageFolder images are used
- `GENERATOR_STEPS = 1`
- separate learning rates for Discriminator and Generator
- label smoothing for real, fake, and Generator target labels

## 7. Submission

Use the same simplified PDF submission style as Lab 01, Lab 02, and Lab 03.

Submit exactly one PDF file:

```text
StudentID_Name_lab04.pdf
```

Do not submit:

- separate screenshot files
- model weights separately
- multiple modified Python files
- long reports

Your PDF should contain the following three parts. Keep screenshots clear and cropped: show the important terminal lines, code blocks, and output images instead of full-screen blank space.

### Part 1: Walkthrough Script Running Results

Include basic information:

- Course name
- Lab name
- Student name
- Student ID
- Class
- Date

Include screenshots showing that `00_check_dataset.py` to `03_dcgan_training_walkthrough.py` have run.

Combined terminal screenshots are allowed.

Required terminal screenshots:

- `00_check_dataset.py`: ImageFolder path, flat CelebA path, auto-prepared image count, dataset size, class names, one batch image shape, one batch label shape, normalized value range, and saved `real_samples.png` path.
- `01_dcgan_generator.py`: Generator structure, fixed noise shape, generated image tensor shape, and saved `untrained_generator_samples.png` path.
- `02_dcgan_discriminator.py`: Discriminator structure, input batch source, dataset size, input image batch shape, Discriminator output shape, and example Discriminator scores.
- `03_dcgan_training_walkthrough.py`: dataset size, batch size, epochs, maximum batches per epoch, Discriminator learning rate, Generator learning rate, Generator updates per batch, gradient clip norm, training loss lines, epoch summary lines, saved walkthrough fake image path, and saved walkthrough loss curve path.

And also the images in `outputs/`:

- real_samples.png
- untrained_generator_samples.png
- walkthrough_real_samples.png
- walkthrough_fake_epoch_00.png
- walkthrough_fake_samples.png
- walkthrough_loss_curve.png

You may also include one intermediate walkthrough image such as `walkthrough_fake_epoch_05.png` if it helps show the visual training process.

### Part 2: Integrated Experiment Code

Include screenshots of the completed code for the seven TODO blocks in `04_integrated_experiment.py`.

Your code screenshots should cover:

- Student Configuration Area
- TODO 1: Data loading and DataLoader construction
- TODO 2: Generator network
- TODO 3: Discriminator network
- TODO 4: Discriminator training step
- TODO 5: Generator training step
- TODO 6: fixed-noise generated image saving
- TODO 7: model checkpoint saving

Do not paste or screenshot the entire file. Crop each code screenshot around the relevant TODO block.

### Part 3: Integrated Experiment Running Result

Include screenshots of running and the images in `outputs/`:

```bash
python 04_integrated_experiment.py
```

Screenshots must show:

- dataset size
- actual training image count
- actual number of batches
- batch size
- number of epochs
- Discriminator learning rate
- Generator learning rate
- gradient clip norm
- Generator steps per batch
- save every epochs
- epoch number
- discriminator loss
- generator loss
- D(x)
- D(G(z))
- saved real sample grid path
- saved fake sample grid path
- saved epoch fake grid path, such as `integrated_fake_epoch_005.png`
- saved loss curve path
- saved generator weights path
- saved discriminator weights path

And also the images in `outputs/`:

- integrated_real_samples.png
- at least one intermediate image, such as integrated_fake_epoch_005.png, integrated_fake_epoch_010.png, or integrated_fake_epoch_015.png
- integrated_fake_samples.png
- integrated_loss_curve.png

The model weight files do not need to be submitted separately. It is enough that the PDF shows the terminal paths for `generator_weights.pth` and `discriminator_weights.pth`.

If the final image quality is still poor, compare the intermediate epoch images. GAN training is adversarial, so the loss curve may fluctuate. Judge progress mainly from the saved fixed-noise image grids, not from monotonically decreasing loss.
