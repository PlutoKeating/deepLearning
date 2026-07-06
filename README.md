# 🧠 Deep Learning & Computer Vision Exam Prep Suite

<p align="center">
  <img src="https://img.shields.io/badge/React-19-blue?style=for-the-badge&logo=react" alt="React" />
  <img src="https://img.shields.io/badge/Tailwind_CSS-v4-38bdf8?style=for-the-badge&logo=tailwindcss" alt="Tailwind CSS" />
  <img src="https://img.shields.io/badge/Vite-6-646cff?style=for-the-badge&logo=vite" alt="Vite" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Academic-Paper_Vault-red?style=for-the-badge&logo=arxiv" alt="Paper Vault" />
</p>

### 🚀 The Ultimate Study Portal, Interactive Quiz Engine & Research Vault for Deep Learning & CV

Preparing for a rigorous final exam in Deep Learning and Computer Vision? Look no further. **Deep Learning & Computer Vision Exam Prep Suite** is a state-of-the-art interactive study portal and document vault. It compiles core mathematical formulations, neural network architectures, cutting-edge papers, and hands-on PyTorch coding labs into an engaging React web application and structured Markdown wiki, guaranteeing you master the subject inside out.

> [!TIP]
> ⚡ **From Linear Classifiers to Diffusion & Foundation Models:** Covers a comprehensive 14-chapter academic curriculum and 3 major engineering labs, backed by a fully interactive self-test quiz engine.

---

## ✨ Key Features

- 📚 **Comprehensive 14-Chapter Academic Wiki:** Detailed explanations of loss functions, optimization, CNNs, RNNs, Transformers, Object Detection, Semantic Segmentation, Video, Self-Supervised Learning, Generative Models, and Foundation Models.
- 📝 **Interactive Exam-Style Quiz Bank:** Test your knowledge on both theoretical formulations (like backpropagation gradients and KL divergence) and practical code with a live-feedback React quiz engine.
- 🔬 **Hands-On PyTorch Lab Handouts & Starter Code:** Access guides and code for three key experiments: PyTorch Foundations, CNN Transfer Learning, and Object Detection / Segmentation.
- 📂 **Academic Paper Vault:** A collection of seminal deep learning papers (including AlexNet, ResNet, ViT, DINO, SimCLR, MAE, GAN, DDPM, Stable Diffusion, CLIP, LoRA, and SAM).
- 🎨 **Sleek, Modern Web Dashboard:** Fully decoupled React SPA designed with a clean, responsive layout using Tailwind CSS.

---

## 📅 The 14-Chapter Curriculum

| Phase | Chapter & Focus | Key Algorithms, Losses & Architectures Covered | Wiki Document |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Foundation & Classifiers | k-NN, Linear Classifiers, SVM Loss (Hinge), Softmax (Cross-Entropy), Regularization | `ch1_introduction_cv.md`<br/>`ch2_linear_classifiers.md` |
| **Phase 2** | Training & Optimization | SGD, Momentum, Adam, Batch Normalization, Dropout, Activation Functions (ReLU, LeakyReLU) | `ch3_regularization_optimization.md`<br/>`ch4_neural_networks.md` |
| **Phase 3** | Core Architectures | Convolution & Pooling, AlexNet, VGG, ResNet, RNNs, LSTM, GRU, Self-Attention, ViT | `ch5_cnn.md`, `ch6_rnn.md`<br/>`ch7_transformers.md` |
| **Phase 4** | Segmentation & Detection | FCN, U-Net, DeepLab (Atrous Conv, ASPP), IoU, Dice, R-CNN series, YOLO, NMS, Focal Loss | `ch8_semantic_segmentation.md`<br/>`ch9_object_detection.md` |
| **Phase 5** | Video & Self-Supervised | 3D CNN, Two-Stream, SlowFast, Contrastive Learning, SimCLR, MoCo, BYOL, MAE | `ch10_video_understanding.md`<br/>`ch11_self_supervised.md` |
| **Phase 6** | Generative Models | GAN, DCGAN, WGAN (Mode Collapse), VAE (ELBO), DDPM, DDIM, Stable Diffusion (LDM) | `ch12_generative_models.md`<br/>`ch13_diffusion_models.md` |
| **Phase 7** | Foundation Models & Review | CLIP (Contrastive Language-Image), Scaling Laws, LoRA (Low-Rank Adaptation), RLHF, SAM | `ch14_foundation_models.md` |

---

## 🔬 Hands-On PyTorch Labs

| Lab | Subject | Core Skills Covered | Wiki Review |
| :--- | :--- | :--- | :--- |
| **Lab 01** | **PyTorch Intro** | Tensor operations, Autograd mechanics, custom training loops, and validation checks. | `lab01_pytorch_intro.md` |
| **Lab 02** | **CNN Transfer Learning** | Fine-tuning pre-trained networks, feature extraction, and learning rate schedules. | `lab02_cnn_transfer.md` |
| **Lab 03** | **Object Detection & Seg** | Processing COCO datasets, bounding box visualization, and Mask R-CNN training. | `lab03_detection_seg.md` |

---

## 📁 Project Directory Layout

```bash
deepLearning/
├── exam-prep-site/         # 💻 Decoupled React+Vite web application
│   ├── src/                #   └─ Flashcards, quiz dashboard, and paper explorer
│   └── public/             #   └─ Static public assets
├── references/             # 📚 Study Raw Source Materials
│   ├── wiki/               #   └─ Highly detailed chapter-by-chapter academic summaries
│   └── quiz_bank.json      #   └─ Interactive exam-question database
├── course/                 # 📂 Core lecture slides in PDF format
├── labs/                   # 🔬 Lab starter codes and student handout manuals
├── materials/              # 📄 Seminal research papers (AlexNet, ResNet, YOLO, ViT, etc.)
├── study_plan.md           # 📅 Master study plan and suggested chapter reading order
└── study_progress.md       # ✍️ Personalized progress trackers and exam tips
```

---

## ⚡ Quick Start

### 1. Run the Web Dashboard Locally

Launch the interactive dashboard locally:

```bash
# Navigate to the site folder
cd exam-prep-site

# Install package dependencies
npm install

# Run the local Vite dev server
npm run dev
```

### 2. Compile Static Assets

To build the static application for production deployment:

```bash
npm run build
```
The optimized single-page app will be outputted to `dist/`, ready to upload and serve via any static platform.
