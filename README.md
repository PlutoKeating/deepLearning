# 🧠 深度学习与计算机视觉期末通关套件 (Deep Learning & CV Prep Suite)

<p align="center">
  <img src="https://img.shields.io/badge/React-19-blue?style=for-the-badge&logo=react" alt="React" />
  <img src="https://img.shields.io/badge/Tailwind_CSS-v4-38bdf8?style=for-the-badge&logo=tailwindcss" alt="Tailwind CSS" />
  <img src="https://img.shields.io/badge/Vite-6-646cff?style=for-the-badge&logo=vite" alt="Vite" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Academic-Paper_Vault-red?style=for-the-badge&logo=arxiv" alt="Paper Vault" />
</p>

### 🚀 一站式交互复习平台、自测题库与学术论文库（包含 14 章节核心大纲与 4 个 PyTorch 经典实验）

正在为高难度的深度学习与计算机视觉期末考试感到头疼？各种繁杂的数学推导、卷积特征图计算、多头自注意力公式让你无从下手？**深度学习与计算机视觉期末通关套件**是您的救星！它将精心归纳的核心数学公式、神经网络架构体系、地标性经典论文和 PyTorch 核心实验融为一体，配备精美的单页 React 前端应用与中文讲义 Wiki，保你高分通关！

> [!TIP]
> ⚡ **从线性分类器到扩散模型、基础模型：** 覆盖 14 章节完整的深度学习与 CV 专业学术大纲，更有 4 大 PyTorch 核心实验详细拆解（含最新集成的 **Lab 04: DCGAN 图像生成**），支持网页版真题极速演练！

---

## ✨ 核心特色

- 📚 **14 章节核心复习 Wiki 库：** 精炼总结损失函数、参数优化、三大核心架构（CNN/RNN/Transformer）、目标检测、语义分割、视频理解、自监督对比学习、生成对抗网络（GAN/VAE）、扩散模型以及多模态基础模型（CLIP、LoRA、SAM）。
- 📝 **交互式真题自测引擎：** 包含选择题与大图理解，支持即时自测反馈、深度原理解析（如反向传播计算图推导、ELBO 推导、KL 散度计算等）。
- 🔬 **PyTorch 实验大纲与 Starter Code：** 精心整理四个核心实验，配有完善的讲义说明、可运行脚本与完成度极佳的学术报告。
- 📂 **学术原版经典论文库：** 原生归档收集了 AlexNet, ResNet, ViT, DINO, SimCLR, MAE, GAN, DDPM, Stable Diffusion, CLIP, LoRA, SAM 等领域地标性论文的原版 PDF，方便随时考证。
- 🎨 **极简现代复习主页：** 基于完全解耦 React 前端架构与 Tailwind CSS 构建的精美、高响应式仪表盘。

---

## 📅 14 章节学术复习路线图

| 复习阶段 | 章节与核心主题 | 涵盖的核心算法、损失函数与网络架构 | 对应知识库文档 |
| :--- | :--- | :--- | :--- |
| **阶段 1** | CV 基础与线性分类器 | k-NN、线性分类器、SVM 损失函数（Hinge Loss）、Softmax 交叉熵损失、L1/L2 正则化 | `ch1_introduction_cv.md`<br/>`ch2_linear_classifiers.md` |
| **阶段 2** | 神经网络与训练技巧 | SGD、Momentum、Adam、批量归一化（BN 训推差异）、Dropout（倒置丢弃）、激活函数（ReLU/LeakyReLU/Sigmoid/Tanh） | `ch3_regularization_optimization.md`<br/>`ch4_neural_networks.md` |
| **阶段 3** | 三大经典核心架构 | 卷积与池化计算、感受野、AlexNet、VGG（3×3卷积核优势）、ResNet（残差连接）、RNN（LSTM 三门、GRU 两门）、Self-Attention (QKV 机制)、ViT 架构 | `ch5_cnn.md`<br/>`ch6_rnn.md`<br/>`ch7_transformers.md` |
| **阶段 4** | 分割与目标检测大题 | FCN、U-Net（对称编解码与跳跃连接）、DeepLab系列（空洞卷积、ASPP 模块）、IoU/mIoU、Dice系数、两阶段检测器（R-CNN/Fast/Faster R-CNN）、一阶段检测器（YOLO）、Focal Loss、NMS 极大值抑制、mAP 计算 | `ch8_semantic_segmentation.md`<br/>`ch9_object_detection.md` |
| **阶段 5** | 视频理解与自监督对比 | 3D CNN、双流网络（空间+时间）、SlowFast 架构、自监督对比学习 SimCLR (NT-Xent 损失)、MoCo (动量编码器+动态字典)、BYOL (无负样本机制)、MAE (高掩码率重建) | `ch10_video_understanding.md`<br/>`ch11_self_supervised.md` |
| **阶段 6** | 经典生成模型与扩散 | GAN 极大极小博弈、DCGAN 架构设计规范、WGAN（Wasserstein 距离与模式坍塌）、VAE（重参数化与 ELBO 重建损失）、DDPM 扩散过程（前向加噪与反向去噪）、DDIM 采样、CFG 引导、Stable Diffusion 隐空间扩散 | `ch12_generative_models.md`<br/>`ch13_diffusion_models.md` |
| **阶段 7** | 基础大模型与LoRA微调 | CLIP 双塔对比学习、Scaling Laws 标度律、LoRA 低秩分解微调、RLHF 三阶段训练、SAM 可提示分割基础模型 | `ch14_foundation_models.md` |

---

## 🔬 四大 PyTorch 核心实验回顾

| 实验编号 | 实验主题 | 核心技能与考点回顾 | 对应复习 Wiki |
| :--- | :--- | :--- | :--- |
| **Lab 01** | **PyTorch 基础入门** | Tensor 操作、Autograd 自动求导机制、自定义 Dataset 与网络训练循环、模型保存与加载 | `lab01_pytorch_intro.md` |
| **Lab 02** | **CNN 迁移学习** | 冻结特征提取层 (Feature Extractor) 与全网络微调 (Fine-tuning)、预训练 ResNet-18 权重调用、数据增强 | `lab02_cnn_transfer.md` |
| **Lab 03** | **目标检测与实例分割** | 行人检测 COCO 标注格式解析、掩码提取、RoIAlign 特征提取、两阶段 Mask R-CNN 模型推理与检测框可视化 | `lab03_detection_seg.md` |
| **Lab 04** | **DCGAN 图像生成** | 面部 CelebA 数据集加载、Generator（转置卷积上采样）与 Discriminator（步长卷积下采样）网络构建、对抗训练循环（判别器/生成器损失对抗波动）、Label Smoothing 标签平滑、固定噪声观察图像生成演变 | `lab04_dcgan_generation.md` |

---

## 📁 项目目录结构

```bash
deepLearning/
├── study-site/             # 💻 前端 React+Vite 交互复习网站（完全解耦）
│   ├── src/                #   └─ Flashcards 翻牌卡片、真题测验与文献浏览器源码
│   └── public/             #   └─ 静态图片、图标与分发资源
├── references/             # 📚 极速复习原始核心材料
│   ├── wiki/               #   └─ 精炼的 14 章节学术核心知识点 Wiki 知识库
│   └── quiz_bank.json      #   └─ 结构化的期末混合自测题库
├── course/                 # 📂 原版授课 PPT 课件对应的精炼 PDF 讲义
├── labs/                   # 🔬 4 大 PyTorch 实验的 Starter Code、学生讲义说明与完整 HTML 学术报告
├── materials/              # 📄 深度学习地标性经典学术论文原版 PDF (AlexNet, ResNet, YOLO, ViT, DDPM 等)
├── study_plan.md           # 📅 深度学习极速备考计划书与建议章节阅读顺序
└── study_progress.md       # ✍️ 个性化错题本、公式推导死磕笔记与学习追踪进度表
```

---

## ⚡ 快速开始

### 1. 本地启动网页复习终端

仅需一分钟，即可在本地运行高性能 React 备考网站：

```bash
# 进入网站项目文件夹
cd study-site

# 安装项目依赖
npm install

# 启动本地开发热更新服务器
npm run dev
```

### 2. 编译生产静态包

导出极致压缩的生产环境静态资源：

```bash
npm run build
```
编译生成的静态文件将输出至 `dist/` 文件夹下，随后即可直接上传并托管至任意静态网页平台。

---

## ☁️ Cloudflare Pages 部署说明

本项目的 `study-site/` 目录完美适配 **Cloudflare Pages** 自动化流水线，实现代码推送即部署（CI/CD）：

### Step-by-Step 部署配置步骤

1. **登录 Cloudflare 控制台：** 登录 [Cloudflare Dashboard](https://dash.cloudflare.com/)，导航至 **Workers & Pages** -> **Pages** -> **Connect to Git（连接到 Git）**。
2. **选择 GitHub 仓库：** 绑定您的 GitHub 账号，并选择本仓库 `deepLearning`。
3. **填写构建配置字段（核心要点）：** 请严格按照以下表格配置构建设置：

| 配置字段 (Configuration Field) | 目标设置值 (Value) | 字段详解与重要作用 |
| :--- | :--- | :--- |
| **Project Name (项目名称)** | `deep-learning-study-site` | 用于生成您的二级域名 (如 `deep-learning-study-site.pages.dev`) |
| **Production Branch (生产分支)** | `main` | 触发正式版部署的默认主分支 |
| **Framework Preset (框架预设)** | **Vite** | 自动加载 Vite 项目的标准依赖和环境预设 |
| **Root Directory (根目录)** | `study-site` | **[最关键]** 必须指定为解耦的网页子目录，否则构建会失败 |
| **Build Command (构建命令)** | `npm run build` | 运行 `tsc -b && vite build` 编译 React + TS + Vite 生产包 |
| **Build Output Directory (构建输出目录)** | `dist` | 编译后生成的静态资源分发包路径 |

4. **保存并部署：** 点击 "Save and Deploy"，Cloudflare 将拉取您的代码、自动安装 Node 依赖，并秒级部署上线。

> [!IMPORTANT]
> **关于根目录设置的声明：** 由于我们指定了 **Root Directory** 为 `study-site`，Cloudflare 会自动定位在该子目录中执行依赖安装和打包。请勿在构建命令中填写 `cd study-site && npm run build`。

> [!NOTE]
> 往后每一次向本仓库的 `main` 分支提交或合并代码，Cloudflare Pages 都会自动捕捉并静默更新部署，确保您的备考网站内容始终与本地错题笔记保持同步！
