# Chapter 8: Semantic Segmentation

> 🟢 来自资料 — 本章核心内容源自 FCN, U-Net, DeepLab 系列论文及课程讲义。🟡 AI补充 — MaskFormer/Mask2Former 及部分对比细节由AI辅助补全。

---

## 8.1 Task Definition（任务定义）

**语义分割 (Semantic Segmentation)**：为图像中的每个像素分配一个语义类别标签。

$$y_{ij} \in \{1, 2, ..., C\}, \quad \forall (i, j) \in \text{image}$$

与相关任务的对比：

| 任务 | 粒度 | 输出 |
|------|------|------|
| **图像分类** | 图像级 | 一个类别 |
| **目标检测** | 物体级 | 边界框 + 类别 |
| **语义分割** | 像素级 | 每个像素一个类别 |
| **实例分割** | 像素级 + 实例级 | 每个像素一个类别 + 实例ID |
| **全景分割** | 像素级 + 全面 | 语义(背景) + 实例(前景) |

🟢 来自资料

---

## 8.2 Evaluation Metrics（评估指标）

### 8.2.1 Pixel Accuracy（像素准确率）

$$\text{Pixel Acc} = \frac{\sum_i n_{ii}}{\sum_i \sum_j n_{ij}} = \frac{\text{正确分类像素数}}{\text{总像素数}}$$

**缺点**：类别不平衡时虚高（背景像素占绝大多数）。

### 8.2.2 Mean IoU (mIoU, 平均交并比)

对于类别 $i$：

$$\text{IoU}_i = \frac{TP_i}{TP_i + FP_i + FN_i} = \frac{n_{ii}}{\sum_j n_{ij} + \sum_j n_{ji} - n_{ii}}$$

$$\text{mIoU} = \frac{1}{C}\sum_{i=1}^C \text{IoU}_i$$

mIoU 是语义分割最核心的评估指标，对类别不平衡更 robust。

🟢 来自资料

### 8.2.3 Dice Coefficient (F1 Score)

$$\text{Dice}_i = \frac{2 \cdot TP_i}{2 \cdot TP_i + FP_i + FN_i} = \frac{2 \cdot |P_i \cap G_i|}{|P_i| + |G_i|}$$

常用于医学图像分割，对小目标更敏感。

🟢 来自资料

---

## 8.3 Fully Convolutional Networks (FCN, 2015)

### 8.3.1 核心思想

将分类网络（如 VGG）的**全连接层替换为卷积层**，实现任意尺寸输入的像素级预测。

🟢 来自资料

### 8.3.2 全连接 → 卷积

- FC(4096) → Conv($7 \times 7$, 4096)（假设 FC 前特征图 $7 \times 7$）
- FC(4096) → Conv($1 \times 1$, 4096)
- FC(1000) → Conv($1 \times 1$, num_classes)

### 8.3.3 上采样 (Upsampling)

| 方法 | 说明 |
|------|------|
| **转置卷积 (Transposed Convolution)** | 可学习的上采样，如 Conv2DTranspose |
| **双线性插值 (Bilinear Interpolation)** | 固定的上采样，无参数 |
| **Unpooling** | 配合 MaxPooling 的索引进行上采样 |

**转置卷积**：将输入值分散到输出，通过 stride 和 padding 控制输出尺寸。

$$W_{out} = (W_{in} - 1) \cdot S + F - 2P$$

🟢 来自资料

### 8.3.4 FCN Skip Connections

| 变体 | 上采样倍数 | 融合层 | 说明 |
|------|-----------|--------|------|
| **FCN-32s** | 32× | 无 | 仅最终层上采样，粗糙 |
| **FCN-16s** | 16× + 2× | pool4 | 融合 pool4 特征 |
| **FCN-8s** | 8× + 2× + 2× | pool4 + pool3 | 融合多层特征，最精细 |

跳跃连接融合公式（以 FCN-16s 为例）：

$$y = \text{Conv}(\text{Upsample}(\text{pool5\_score})) + \text{Conv}(\text{pool4\_score})$$

🟢 来自资料

**效果**：FCN-8s > FCN-16s > FCN-32s（融合更多低层空间信息 → 边缘更精细）。

---

## 8.4 U-Net (2015)

### 8.4.1 架构

U-Net 是经典的编码器-解码器结构，因其 U 形对称结构得名。

```
输入 → [Encoder 下采样路径] → 瓶颈 → [Decoder 上采样路径] → 输出
           |                    |              |
           └─── Skip Connection ──────────────┘
```

**Encoder（收缩路径）**：
- 重复 2 次 $3 \times 3$ Conv + ReLU + $2 \times 2$ MaxPool
- 每次池化后通道数翻倍（64→128→256→512→1024）

**Decoder（扩展路径）**：
- $2 \times 2$ 转置卷积上采样（通道减半）
- 与对应 Encoder 层特征图**拼接**（不是相加！）
- 2 次 $3 \times 3$ Conv + ReLU

**最终层**：$1 \times 1$ Conv 输出类别分数。

🟢 来自资料

### 8.4.2 Skip Connections 的作用

- 恢复在下采样过程中丢失的**空间细节**
- 梯度可以直接流向浅层
- 对**边界精确定位**至关重要

### 8.4.3 应用

- 生物医学图像分割（原始动机）
- 遥感图像分割
- 卫星图像分析
- 任何需要精确定位的像素级任务

🟢 来自资料

---

## 8.5 SegNet (2015)

### 8.5.1 核心创新：Max-Pooling Indices

SegNet 的上采样不使用转置卷积，而是记忆 MaxPooling 时最大值的位置（indices），上采样时将值放回原位，其余填 0。

$$\text{Upsample}(x) = \text{Unpooling}(x, \text{indices})$$

- **优点**：参数量少，内存效率高
- **缺点**：上采样后仍需卷积来平滑

🟢 来自资料

---

## 8.6 DeepLab 系列

### 8.6.1 DeepLab v1 (2015)

| 组件 | 说明 |
|------|------|
| **Atrous Convolution (空洞卷积)** | 带有空洞的卷积，在不增加参数的情况下扩大感受野 |
| **CRF (Conditional Random Field)** | 后处理，细化分割边界 |

**Atrous Convolution**：在卷积核元素之间插入空洞（rate $r$）。

- 有效卷积核尺寸：$F_{eff} = F + (F-1)(r-1)$
- 输出尺寸不变（same padding），感受野放大
- rate=1 退化为标准卷积

🟢 来自资料

### 8.6.2 DeepLab v2 (2017): ASPP

**ASPP (Atrous Spatial Pyramid Pooling)**：并行多个不同 rate 的空洞卷积，捕获多尺度上下文。

```
输入特征图
  ├── 1×1 Conv
  ├── 3×3 Conv, rate=6
  ├── 3×3 Conv, rate=12
  ├── 3×3 Conv, rate=18
  └── (Image-level features)
      → Concat → 1×1 Conv → 输出
```

🟢 来自资料

### 8.6.3 DeepLab v3 (2017)

**改进**：

- ASPP 中加入 **Global Average Pooling** 分支，捕获全局上下文
- 加入 **Batch Normalization**
- 移除 CRF 后处理（端到端训练足够好）

**扩展的 ASPP**：
```
输入 → [1×1 Conv, 3×3 rate=6, 3×3 rate=12, 3×3 rate=18, GAP + Upsample]
       → Concat → 1×1 Conv → 输出
```

🟢 来自资料

### 8.6.4 DeepLab v3+ (2018)

**引入编码器-解码器结构**：

- **Encoder**：DeepLab v3（ASPP + 主干网络）
- **Decoder**：融合低层特征 + 上采样
- 使用 **Atrous Separable Convolution**（空洞 + 深度可分离卷积）
- 主干网络可选 Xception 或 ResNet

🟢 来自资料

---

## 8.7 PSPNet (2017)

**Pyramid Scene Parsing Network**

**核心模块：Pyramid Pooling Module**

```
输入特征图
  ├── 1×1 Pool → Conv → Upsample
  ├── 2×2 Pool → Conv → Upsample
  ├── 3×3 Pool → Conv → Upsample
  ├── 6×6 Pool → Conv → Upsample
  └── 原始特征图
      → Concat → Conv → 输出
```

多尺度池化捕获不同大小的上下文信息。

🟢 来自资料

---

## 8.8 Beyond CNN-based Methods

### 8.8.1 MaskFormer (2021)

将语义分割重新定义为 **Mask Classification** 问题：

- 预测一组 $N$ 个二元 mask + 对应类别
- 使用 Transformer Decoder 输出 mask embeddings
- 通过匈牙利匹配进行训练

**优势**：统一语义分割和实例分割的框架。

🟢 来自资料

### 8.8.2 Mask2Former (2022)

**改进**：

- **Masked Attention**：Decoder 只关注预测 mask 区域附近，加速收敛
- **多尺度特征**：从不同层提取特征输入 Decoder
- **更好的收敛**：比 MaskFormer 快 3× 达到相同性能

🟢 来自资料

---

## 8.9 Loss Functions（损失函数）

### 8.9.1 Cross-Entropy Loss (逐像素)

$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log(p_{i,c})$$

**缺点**：类别不平衡时，大类别主导梯度。

### 8.9.2 Weighted Cross-Entropy

$$\mathcal{L}_{WCE} = -\frac{1}{N}\sum_{i=1}^N \sum_{c=1}^C w_c \cdot y_{i,c} \log(p_{i,c})$$

$w_c$ 与类别频率成反比（如 $w_c = 1/f_c$）。

### 8.9.3 Focal Loss

$$\mathcal{L}_{FL} = -\frac{1}{N}\sum_{i=1}^N \sum_{c=1}^C \alpha_c (1 - p_{i,c})^\gamma y_{i,c} \log(p_{i,c})$$

- $\gamma$：聚焦参数（典型值 2），降低易分类样本的权重
- $\alpha_c$：类别平衡权重

🟢 来自资料

### 8.9.4 Dice Loss

$$\mathcal{L}_{Dice} = 1 - \frac{1}{C}\sum_{c=1}^C \frac{2 \sum_i p_{i,c} \cdot y_{i,c}}{\sum_i p_{i,c} + \sum_i y_{i,c}}$$

- 直接优化 Dice 指标
- 对小目标更友好
- 常与 Cross-Entropy 组合使用：$\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{Dice}$

🟢 来自资料

---

## 8.10 Architecture Comparison

| 模型 | 年份 | 编码器 | 核心创新 | 特点 |
|------|------|--------|----------|------|
| **FCN** | 2015 | VGG | FC→Conv, Skip Connections | 开创性工作，奠基 |
| **U-Net** | 2015 | 自定义 | 对称Encoder-Decoder, 拼接Skip | 生物医学，精细边界 |
| **SegNet** | 2015 | VGG | MaxPooling Indices | 内存效率高 |
| **DeepLab v1** | 2015 | VGG | Atrous Conv + CRF | 大感受野 + 后处理 |
| **DeepLab v2** | 2017 | ResNet | ASPP | 多尺度上下文 |
| **DeepLab v3** | 2017 | ResNet | 改进ASPP + BN, 去CRF | 端到端，更强 |
| **DeepLab v3+** | 2018 | Xception/ResNet | Encoder-Decoder + Atrous Sep Conv | 速度+精度的平衡 |
| **PSPNet** | 2017 | ResNet | Pyramid Pooling Module | 全局+局部上下文 |
| **MaskFormer** | 2021 | ResNet/Swin | Mask Classification, Transformer | 统一语义+实例 |
| **Mask2Former** | 2022 | ResNet/Swin | Masked Attention, 多尺度 | 更快收敛，SOTA |

🟢 来自资料 / 🟡 AI补充

---

## 8.11 Practice Problems

### Problem 1: Transposed Convolution Output Size

> 输入 $4 \times 4$，使用 $3 \times 3$ 转置卷积，stride=2, padding=1。求输出尺寸。

**Solution**：

$$W_{out} = (W_{in} - 1) \cdot S + F - 2P$$
$$= (4 - 1) \times 2 + 3 - 2 \times 1 = 6 + 3 - 2 = 7$$

输出：$7 \times 7$

---

### Problem 2: mIoU Calculation

> 混淆矩阵（3类）：
> $$N = \begin{bmatrix} 50 & 10 & 5 \\ 8 & 40 & 2 \\ 3 & 7 & 30 \end{bmatrix}$$
> 计算每类 IoU 和 mIoU。

**Solution**：

- 类0：$IoU_0 = \frac{50}{50 + (10+5) + (8+3)} = \frac{50}{76} = 0.658$
- 类1：$IoU_1 = \frac{40}{40 + (8+2) + (10+7)} = \frac{40}{67} = 0.597$
- 类2：$IoU_2 = \frac{30}{30 + (3+7) + (5+2)} = \frac{30}{47} = 0.638$

$$mIoU = \frac{0.658 + 0.597 + 0.638}{3} = 0.631$$

Pixel Accuracy = $\frac{50+40+30}{50+10+5+8+40+2+3+7+30} = \frac{120}{155} = 0.774$

---

### Problem 3: Atrous Convolution Receptive Field

> 解释为何空洞卷积 rate=2, 3×3 卷积核的有效感受野是 $5 \times 5$。

**Solution**：

空洞卷积在卷积核元素间插入 rate-1 个零。rate=2 时，$3 \times 3$ 核中相邻元素间隔 1 个空洞，覆盖 $5 \times 5$ 区域：

```
X 0 X 0 X
0 0 0 0 0
X 0 X 0 X
0 0 0 0 0
X 0 X 0 X
```

其中 X 为实际参数位置。有效核尺寸 $F_{eff} = 3 + (3-1)(2-1) = 5$。

---

### Problem 4: U-Net Skip Connection

> U-Net 中为何使用拼接 (concatenation) 而非相加 (addition) 作为 skip connection？

**Solution**：

- **拼接**：保留 Encoder 和 Decoder 特征的独立性，后续卷积可学习自适应融合。通道数翻倍，信息更丰富。
- **相加**：强制两者维度匹配，将 Encoder 特征视为"残差修正"，信息混合更受限。

U-Net 是医学分割任务，需要精确的空间定位。拼接提供了更丰富的原始空间信息，比相加更适合边界恢复。

---

*Last updated: 2026-07-02*
