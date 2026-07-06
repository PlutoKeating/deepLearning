# Chapter 1: Introduction to Computer Vision

> 🟢 来自资料 — 综合自《Computer Vision: Algorithms and Applications》(Szeliski)、CS231n 课程讲义及课程介绍材料

---

## 1.1 What is Computer Vision?

计算机视觉（Computer Vision, CV）是人工智能的一个核心分支，其目标是赋予机器"看"的能力。一个经典的定义来自 David Marr：

> **"To see is to know what is where by looking."**
> — 看即是知：通过观察，知道什么东西在什么地方。

### 1.1.1 三个关键要素

| 要素 | 含义 | CV 中的对应概念 |
|------|------|----------------|
| **What** | 识别"是什么" | 物体分类、识别、语义理解 |
| **Where** | 确定"在哪里" | 定位、检测、分割 |
| **Looking** | 通过视觉传感器"观察" | 图像采集、特征提取 |

计算机视觉的数学本质可以表述为：给定二维图像 $I \in \mathbb{R}^{H \times W \times C}$，推断三维世界中的属性（物体身份、位姿、几何结构、运动等）。

---

## 1.2 Camera: Measurement Device vs Perception System

### 1.2.1 作为测量设备的相机

相机可以被建模为一个投影系统。针孔相机模型是最基础的数学模型：

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim K [R \mid t] \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

其中：
- $(X, Y, Z)$ 是世界坐标系中的 3D 点
- $(u, v)$ 是图像平面上的 2D 像素坐标
- $K$ 是内参矩阵（焦距、主点偏移）
- $[R \mid t]$ 是外参矩阵（旋转+平移）

🟡 AI补充：作为测量设备，相机提供的是精确的几何/光度测量——角度、距离、亮度等物理量。

### 1.2.2 作为感知系统的相机

当我们将相机与计算系统结合时，它变成了一个**感知系统**：

| 测量设备视角 | 感知系统视角 |
|-------------|-------------|
| 输出：像素值、深度图 | 输出：语义标签、物体属性、场景理解 |
| 评价标准：精度（accuracy） | 评价标准：准确率（precision/recall） |
| 关注几何与物理 | 关注语义与认知 |

🟢 来自资料：计算机视觉的本质是从 raw pixels（原始像素）到 high-level semantics（高层语义）的映射过程。这与人类视觉系统的"从视网膜到皮层"的信息处理流如出一辙。

---

## 1.3 History of Computer Vision

### 1.3.1 早期探索（1960s–1980s）

| 时期 | 里程碑 | 核心理念 |
|------|--------|---------|
| 1966 | MIT Summer Vision Project | 试图用一个暑假"解决"视觉问题 |
| 1970s | David Marr 的计算视觉理论 | 分层处理：primal sketch → 2.5D sketch → 3D model |
| 1980s | Lowe, Canny 边缘检测 | 从像素到特征的提取方法 |

🟡 AI补充：Marr 的理论奠定了 CV 作为独立学科的基础，他提出了信息处理系统的三个层次：计算理论层（做什么）、表示与算法层（怎么做）、硬件实现层（物理实现）。

### 1.3.2 特征工程时代（1990s–2011）

| 时期 | 里程碑 | 核心理念 |
|------|--------|---------|
| 1999 | SIFT (Lowe) | 尺度不变特征变换，手工设计特征 |
| 2001 | Viola-Jones 人脸检测 | 实时人脸检测，Haar-like 特征 + AdaBoost |
| 2005 | HOG (Dalal & Triggs) | 梯度方向直方图，行人检测 |
| 2007 | DPM (Felzenszwalb) | 可变形部件模型，目标检测的巅峰 |

🟢 来自资料：这一时期的范式是"手工设计特征 + 浅层分类器"，特征提取和分类是分离的两个步骤。

### 1.3.3 深度学习时代（2012–至今）

| 年份 | 里程碑 | 意义 |
|------|--------|------|
| **2012** | **AlexNet** | ImageNet 冠军，CNN 在 CV 领域的突破性应用 |
| 2014 | VGGNet, GoogLeNet (Inception) | 更深的网络，多尺度结构 |
| 2015 | ResNet (He et al.) | 残差连接，152层网络，解决退化问题 |
| 2014–2018 | GAN (Goodfellow), VAE | 生成模型兴起 |
| 2017–至今 | ViT, DINO, SAM | Transformer 从 NLP 跨界到 CV |
| 2020–至今 | Diffusion Models (DDPM) | 图像生成的新范式 |
| 2022–至今 | ChatGPT + CV → Multimodal | 多模态大模型整合视觉与语言 |

🟢 来自资料：深度学习的核心优势在于**端到端学习（end-to-end learning）**：系统直接从原始像素学习到最终任务输出，无需手工设计特征。这解释了为何 2012 年后 CV 领域发生了范式转变。

---

## 1.4 Core Computer Vision Tasks

### 1.4.1 任务全景

```
                ┌────────────────────────────────────┐
                │         Computer Vision Tasks       │
                └────────────────────────────────────┘
                                    │
        ┌───────────┬───────────────┼───────────────┬───────────┐
        ▼           ▼               ▼               ▼           ▼
   Classification  Detection    Segmentation    Generation   Understanding
    "是什么?"     "是什么+在哪?"  "像素级在哪?"   "创造新图像"   "更深层理解"
```

### 1.4.2 各任务详解

#### A. 图像分类 (Image Classification)

给定图像，预测其所属类别标签。

- **输入**：图像 $X \in \mathbb{R}^{H \times W \times 3}$
- **输出**：类别概率分布 $P(y|X)$，取 $\arg\max_y P(y|X)$
- **典型指标**：Top-1 Accuracy, Top-5 Accuracy
- **代表模型**：AlexNet → VGG → ResNet → ViT

#### B. 目标检测 (Object Detection)

在图像中同时定位和识别多个物体。

- **输入**：图像
- **输出**：Bounding Boxes + 类别标签
- **典型指标**：mAP (mean Average Precision) @ IoU thresholds
- **方法谱系**：
  - Two-stage: R-CNN → Fast R-CNN → Faster R-CNN
  - One-stage: YOLO, SSD, RetinaNet
  - Transformer-based: DETR

#### C. 语义分割 (Semantic Segmentation)

为图像中每个像素分配类别标签。

- **输入**：图像
- **输出**：dense label map, 尺寸与输入相同
- **关键思想**：全卷积网络 (FCN), Encoder-Decoder (U-Net)
- **典型指标**：mIoU (mean Intersection over Union)

#### D. 实例分割 (Instance Segmentation)

区分同一类别中的不同个体（比语义分割更细粒度）。

- **代表模型**：Mask R-CNN, YOLACT
- **输出**：每个实例的 mask + 类别

#### E. 图像识别与检索 (Recognition & Retrieval)

识别特定身份（人脸识别）或检索相似图像。

- **人脸识别流程**：检测 → 对齐 → 特征提取 → 比对
- **损失函数**：Triplet Loss, ArcFace, CosFace

#### F. 3D 视觉

从 2D 图像恢复 3D 信息。

- **任务**：深度估计、3D 重建、点云处理、NeRF
- **关键挑战**：从 2D 投影恢复 3D 的 ill-posed 问题

#### G. 图像生成 (Image Generation)

从噪声或条件输入生成逼真图像。

- **生成模型族**：
  - GAN: 生成器 + 判别器的对抗训练 → $\min_G \max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$
  - VAE: 变分自编码器 → 最大化 ELBO
  - Diffusion Models: 逐步去噪 → $\epsilon_\theta(x_t, t)$

#### H. 视频理解 (Video Understanding)

扩展到时间维度。

- **任务**：动作识别 (action recognition)、视频目标检测、跟踪
- **关键**：时序建模 — 3D CNN, Optical Flow, Transformer

---

## 1.5 Applications

| 领域 | 具体应用 | CV 任务 |
|------|---------|---------|
| **医疗健康 (Healthcare)** | 医学影像诊断、病理切片分析、手术导航 | 分类、分割、3D重建 |
| **机器人 (Robotics)** | 自主导航、物体抓取、SLAM | 检测、深度估计、位姿估计 |
| **自动驾驶 (Autonomous Driving)** | 车道检测、行人检测、交通标志识别 | 检测、分割、跟踪、深度估计 |
| **游戏娱乐 (Gaming & AR/VR)** | 体感交互、虚拟试穿、特效 | 姿态估计、人脸关键点、生成 |
| **无障碍 (Accessibility)** | 盲人导航、手语识别、屏幕朗读 | 识别、描述生成、OCR |
| **安防监控** | 异常行为检测、人脸识别、人数统计 | 检测、识别、跟踪 |
| **遥感** | 卫星图像分析、城市规划、灾害评估 | 分类、变化检测、分割 |

---

## 1.6 Course Overview

🟢 来自资料 — 课程介绍

### 1.6.1 评分结构

| 项目 | 占比 | 说明 |
|------|------|------|
| **出勤 (Attendance)** | **20%** | 课堂参与度 |
| **实验 (Experiments/Labs)** | **30%** | 3–4 个编程实验，使用 PyTorch |
| **期末考试 (Final Exam)** | **50%** | 闭卷笔试，覆盖全部章节 |

### 1.6.2 课程内容框架

| 章节 | 主题 | 关键词 |
|------|------|--------|
| Ch1 | Introduction to CV | Tasks, History, Applications |
| Ch2 | Linear Classifiers | SVM, Softmax, k-NN |
| Ch3 | Regularization & Optimization | SGD, Adam, BN, Dropout |
| Ch4 | Neural Networks | MLP, Backprop, Activation |
| Ch5 | CNN Architectures | AlexNet, VGG, ResNet, Inception |
| Ch6 | Training CNNs | Transfer Learning, Fine-tuning |
| Ch7 | Detection & Segmentation | R-CNN, YOLO, U-Net, FCN |
| Ch8 | Recurrent Networks | RNN, LSTM, GRU |
| Ch9 | Attention & Transformers | Self-Attention, ViT |
| Ch10 | Video Understanding | 3D CNN, Optical Flow, Action Recognition |
| Ch11 | Self-Supervised Learning | MoCo, SimCLR, MAE, DINO |
| Ch12–13 | Generative Models | GAN, VAE, Diffusion Models, Flow |
| Ch14 | Foundation Models | CLIP, GPT-4V, Multimodal Models |

---

## 1.7 Key Datasets

🟢 来自资料

### 1.7.1 图像分类

| 数据集 | 年份 | 规模 | 类别数 | 意义 |
|--------|------|------|--------|------|
| **MNIST** | 1998 | 60K train + 10K test | 10 | 手写数字，入门数据集 |
| **CIFAR-10/100** | 2009 | 50K train + 10K test | 10/100 | 32×32 自然图像小数据集 |
| **ImageNet** | 2009– | ~1.4M images | 1000 (ILSVRC) | 大规模图像分类基准，2012年深度学习突破 |
| **ImageNet-21K** | — | ~14M images | 21,841 | 大规模预训练 |

### 1.7.2 目标检测与分割

| 数据集 | 特点 |
|--------|------|
| **PASCAL VOC** (2005–2012) | 20 类，经典检测/分割基准 |
| **MS COCO** | 80 类，大规模检测+分割+描述，更关注场景中的小物体 |
| **Cityscapes** | 城市场景语义分割，自动驾驶相关 |
| **LVIS** | 大规模词汇实例分割，1203 类 |

### 1.7.3 其他

| 数据集 | 任务 |
|--------|------|
| **Kinetics** | 视频动作识别 |
| **CelebA** | 人脸属性，GAN 评估 |
| **LFW** | 人脸验证 |

---

## 1.8 CV, AI, and Big Data

🟢 来自资料 & 🟡 AI补充

### 1.8.1 三者关系

```
       ┌──────────────────────────┐
       │    Artificial Intelligence  │
       │  ┌──────────────────────┐ │
       │  │  Computer Vision     │ │
       │  │  ┌──────────────────┐│ │
       │  │  │  Deep Learning   ││ │
       │  │  └──────────────────┘│ │
       │  └──────────────────────┘ │
       │  NLP, Robotics, ...       │
       └──────────────────────────┘
              ▲           ▲
              │           │
         Big Data    Computing Power
```

- **Computer Vision ⊂ Artificial Intelligence**：CV 是 AI 的子领域
- **Deep Learning** 是当前 CV 的主流方法（但不唯一）
- **Big Data** 提供了训练深度模型所需的标注数据（如 ImageNet）
- **Compute Power** (GPU/TPU) 使训练大模型成为可能

### 1.8.2 数据驱动的 CV 范式

现代 CV 的核心公式可以概括为：

$$
\text{Performance} \propto \text{Data} \times \text{Compute} \times \text{Algorithms}
$$

深度学习需要大量标注数据来实现泛化。ImageNet 的百万级标注图像使 AlexNet 成为可能；如今的自监督学习试图摆脱对标注的依赖，利用海量无标注数据（如 DINO、MAE）。

---

## Practice Problems

### Problem 1: Task Identification

下列应用分别对应什么 CV 任务？

a) 自动驾驶中判断前方是否有人
b) 医学 CT 中圈出肿瘤区域
c) 手机相册中搜索"日落的照片"
d) 照相机自动对焦到人脸

**Solution:**
a) 目标检测 (Object Detection) — 定位+分类
b) 语义分割 (Semantic Segmentation) — 像素级分类
c) 图像检索 (Image Retrieval) — 基于内容的搜索
d) 人脸检测 (Face Detection) — 特定类别的检测

### Problem 2: 2012 Breakthrough

Why was AlexNet (2012) considered a breakthrough moment for CV?

**Solution:**
AlexNet 在 ImageNet 2012 上以 ~15.3% top-5 error（第二名 ~26.2%）大幅领先传统方法（SIFT+Fisher Vector）。关键因素：
1. **Deep CNN**：端到端学习，替代手工特征
2. **GPU 训练**：大规模并行计算
3. **ReLU + Dropout**：训练技巧创新
4. **数据增强**：减少过拟合
这标志着 CV 从特征工程时代进入深度学习时代。

### Problem 3: ImageNet vs COCO

What is the key difference between ImageNet classification and COCO object detection tasks?

**Solution:**
- **ImageNet 分类**：每张图一个标签（单标签分类），判断"这张图是什么"
- **COCO 检测**：每张图可能有多个物体（0到N个），需要同时给出类别和位置（bounding box），更接近真实世界场景。COCO 还包含分割标注和图像描述。

---

*Last updated: 2026-07-02*
