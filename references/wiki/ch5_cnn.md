# Chapter 5: Convolutional Neural Networks (CNNs)

> 🟢 来自资料 — 本章核心内容源自标准深度学习与计算机视觉教材及课程讲义。🟡 AI补充 — 架构对比细节与Grad-CAM公式推导由AI辅助补全。

---

## 5.1 Motivation: Why Convolution?

传统全连接网络处理图像时面临三个根本问题：

### 5.1.1 参数爆炸 (Parameter Explosion)

一张 $224 \times 224 \times 3$ 的图片，若第一层有 1000 个神经元，则参数量为：

$$224 \times 224 \times 3 \times 1000 \approx 1.5 \times 10^8$$

这导致内存巨大、极易过拟合。

🟢 来自资料

### 5.1.2 三大核心动机

| 动机 | 说明 | 实现方式 |
|------|------|----------|
| **局部连接 (Local Connectivity)** | 每个神经元只连接输入的局部区域（感受野, Receptive Field） | 卷积核仅覆盖 $K \times K$ 区域 |
| **参数共享 (Parameter Sharing)** | 同一特征图的所有位置共享同一组权重（卷积核） | 卷积核在整个输入上滑动 |
| **平移等变性 (Translation Equivariance)** | 输入平移导致输出同样平移：$f(T(x)) = T(f(x))$ | 卷积操作的固有属性 |

🟢 来自资料

> **平移等变性 vs 平移不变性**：卷积本身具有平移**等变性**（shift in → shift out）。平移**不变性**（shift in → same out）则由池化层（Pooling）提供。

---

## 5.2 Convolution Operation

### 5.2.1 定义

对于 2D 卷积，输入 $I$，卷积核（滤波器）$K$，在位置 $(i, j)$ 的输出：

$$S(i,j) = (I * K)(i,j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m,n)$$

实际深度学习框架中实现的是**互相关 (Cross-correlation)**（不翻转核），数学上等价。

🟢 来自资料

### 5.2.2 关键参数

| 参数 | 符号 | 含义 |
|------|------|------|
| 输入尺寸 | $W_{in} \times H_{in}$ | 输入特征图的宽高 |
| 卷积核尺寸 | $F \times F$ | 滤波器的空间尺寸 |
| 步长 (Stride) | $S$ | 卷积核每次移动的像素数 |
| 填充 (Padding) | $P$ | 在输入边界填充的像素数 |
| 卷积核数量 | $K$ | 输出通道数（特征图数量） |

### 5.2.3 输出尺寸公式

$$W_{out} = \left\lfloor \frac{W_{in} - F + 2P}{S} \right\rfloor + 1$$

$$H_{out} = \left\lfloor \frac{H_{in} - F + 2P}{S} \right\rfloor + 1$$

🟢 来自资料

### 5.2.4 Padding 类型

| 类型 | 公式 | 效果 |
|------|------|------|
| **Valid Padding** | $P = 0$ | 输出尺寸缩小，$W_{out} = \lfloor \frac{W_{in} - F}{S} \rfloor + 1$ |
| **Same Padding** | $P = \frac{F - 1}{2}$ | 输出尺寸不变（当 $S=1$），$W_{out} = W_{in}$ |
| **Full Padding** | $P = F - 1$ | 输出尺寸增大 |

🟢 来自资料

### 5.2.5 参数量和计算量

设输入通道为 $C_{in}$，输出通道为 $C_{out}$，卷积核尺寸 $F \times F$，输出尺寸 $W_{out} \times H_{out}$：

- **参数量**：$F^2 \times C_{in} \times C_{out} + C_{out}$（含偏置）
- **FLOPs**：$2 \times F^2 \times C_{in} \times C_{out} \times W_{out} \times H_{out}$（乘加各算一次）

🟡 AI补充

---

## 5.3 Special Convolution Types

### 5.3.1 $1 \times 1$ Convolution（逐点卷积）

$$y = W_{1 \times 1} \cdot x + b$$

- **作用**：跨通道信息融合，改变通道数，不改变空间尺寸
- **参数量**：$C_{in} \times C_{out}$
- **用途**：降维/升维、GoogLeNet 中的瓶颈层、增加非线性

🟢 来自资料

### 5.3.2 Depthwise Separable Convolution（深度可分离卷积）

将标准卷积分解为两步：

**Step 1 — Depthwise Convolution（逐通道卷积）**：
- 每个输入通道独立进行 $F \times F$ 卷积
- 卷积核尺寸：$F \times F \times 1$，$C_{in}$ 个
- 输出通道数 = 输入通道数

**Step 2 — Pointwise Convolution（逐点卷积）**：
- 用 $1 \times 1$ 卷积融合通道
- 卷积核尺寸：$1 \times 1 \times C_{in} \times C_{out}$

**计算量比较**：

$$\frac{\text{Depthwise Separable}}{\text{Standard}} = \frac{F^2 \cdot C_{in} \cdot H \cdot W + C_{in} \cdot C_{out} \cdot H \cdot W}{F^2 \cdot C_{in} \cdot C_{out} \cdot H \cdot W} = \frac{1}{C_{out}} + \frac{1}{F^2}$$

当 $C_{out}=64, F=3$ 时，计算量约为标准卷积的 $\frac{1}{9} + \frac{1}{64} \approx 12.7\%$。

🟢 来自资料（MobileNet 论文）

---

## 5.4 Pooling（池化）

### 5.4.1 类型

| 池化类型 | 操作 | 公式 | 特点 |
|----------|------|------|------|
| **Max Pooling** | 取局部区域最大值 | $y = \max(x_{i,i+F-1, j,j+F-1})$ | 保留最显著特征，提供平移不变性 |
| **Average Pooling** | 取局部区域均值 | $y = \frac{1}{F^2}\sum x$ | 平滑特征，减少方差 |
| **Global Average Pooling (GAP)** | 对整个特征图取均值 | $y_c = \frac{1}{H \times W}\sum_{i,j} x_{i,j,c}$ | 替代全连接层，每个通道一个输出 |

🟢 来自资料

### 5.4.2 GAP vs Flatten + FC

$$y_c^{\text{GAP}} = \frac{1}{H \times W}\sum_{i=1}^{H}\sum_{j=1}^{W} x_{i,j,c}$$

- **优点**：无参数，正则化效果好，输入尺寸灵活
- **缺点**：丢失空间位置信息

🟢 来自资料

---

## 5.5 Classic CNN Architectures

### 5.5.1 LeNet-5 (1998, LeCun)

用于手写数字识别 (MNIST) 的奠基性网络。

```
Input(32×32×1) → Conv(5×5, 6) → AvgPool(2×2) → Conv(5×5, 16) →
AvgPool(2×2) → Conv(5×5, 120) → FC(84) → FC(10)
```

**关键特征**：tanh 激活、平均池化、（在当时）开创性的卷积+下采样+全连接结构。

🟢 来自资料

### 5.5.2 AlexNet (2012, Krizhevsky et al.)

ImageNet 2012 冠军，标志着深度学习的复兴。

```
Input(227×227×3) → Conv(11×11, 96, S=4) → MaxPool(3×3, S=2) → [Norm]
→ Conv(5×5, 256, pad=2) → MaxPool(3×3, S=2) → [Norm]
→ Conv(3×3, 384, pad=1) → Conv(3×3, 384, pad=1) → Conv(3×3, 256, pad=1)
→ MaxPool(3×3, S=2)
→ FC(4096) → FC(4096) → FC(1000) → Softmax
```

**五大创新**：

| 创新 | 说明 |
|------|------|
| **ReLU** | 替代 sigmoid/tanh，缓解梯度消失，加速训练 |
| **Dropout** | 训练时随机失活 50% 神经元（仅前两个 FC 层），防止过拟合 |
| **Data Augmentation** | 随机裁剪、水平翻转、PCA 颜色扰动 |
| **Local Response Normalization** | 通道间局部归一化（后被 BatchNorm 取代） |
| **双 GPU 并行训练** | 模型分拆到两个 GPU，仅在某些层通信 |

🟢 来自资料

**参数量**：约 60M（5 个卷积层 + 3 个全连接层）

### 5.5.3 VGG (2014, Simonyan & Zisserman)

**核心设计哲学：深 + 简单 + 统一**

| 设计原则 | 说明 |
|----------|------|
| 只用 $3 \times 3$ 卷积 | 堆叠两个 $3 \times 3$ 等效于一个 $5 \times 5$ 的感受野，但参数更少 |
| 只用 $2 \times 2$ Max Pooling | 每次池化空间尺寸减半，通道翻倍 |
| 逐步加深 | VGG16 (16层)、VGG19 (19层) |

**感受野计算**：两个 $3 \times 3$ 卷积的感受野 = $5 \times 5$，三个 = $7 \times 7$

**参数量对比**：$5 \times 5$ 卷积参数量 = $25 C_{in} C_{out}$，两个 $3 \times 3$ = $2 \times 9 C_{in} C_{out} = 18 C_{in} C_{out}$（中间通道可能翻倍，但总体仍更高效且非线性更强）。

VGG16 结构：
```
[Conv3×3,64]×2 → MaxPool → [Conv3×3,128]×2 → MaxPool
→ [Conv3×3,256]×3 → MaxPool → [Conv3×3,512]×3 → MaxPool
→ [Conv3×3,512]×3 → MaxPool → FC(4096) → FC(4096) → FC(1000)
```

🟢 来自资料

### 5.5.4 GoogLeNet / Inception v1 (2014)

**核心思想**：Inception Module — 多尺度并行卷积

```
┌─────────────────────────────────────────────┐
│              Previous Layer                  │
├──────────┬──────────┬───────────┬────────────┤
│ 1×1 Conv │ 1×1 Conv │ 3×3 MaxPool│ 1×1 Conv  │
│          │ 3×3 Conv │ 1×1 Conv  │ 5×5 Conv   │
├──────────┴──────────┴───────────┴────────────┤
│              Filter Concatenation            │
└─────────────────────────────────────────────┘
```

**关键创新**：

1. **$1 \times 1$ 瓶颈层 (Bottleneck)**：在 $3 \times 3$ 和 $5 \times 5$ 卷积前使用 $1 \times 1$ 卷积降维，大幅减少计算量。
2. **辅助分类器 (Auxiliary Classifiers)**：在网络中间层添加分类器，提供额外梯度信号，缓解梯度消失。
3. **Global Average Pooling**：替代最后的全连接层。
4. **参数量仅约 5M**，远小于 AlexNet (60M) 和 VGG16 (138M)。

🟢 来自资料

**后续版本**：
- **Inception v2/v3**：引入 Batch Normalization，分解大卷积（$5 \times 5 \to 2$ 个 $3 \times 3$；$3 \times 3 \to 1 \times 3 + 3 \times 1$）
- **Inception v4 + Inception-ResNet**：结合 Residual Connection

### 5.5.5 ResNet (2015, He et al.)

**核心问题**：网络越深，训练误差反而增大（退化问题，非过拟合）。

**Residual Learning（残差学习）**：

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

其中 $\mathcal{F}$ 为残差映射（几层卷积），$\mathbf{x}$ 为恒等映射（shortcut/skip connection）。

🟢 来自资料

**为什么有效**：学习残差 $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$ 比直接学习 $\mathcal{H}(\mathbf{x})$ 更容易。在极端情况下，若最优映射接近恒等，残差块只需将 $\mathcal{F}$ 推向零。

**两种残差块**：

| 类型 | 结构 | 用途 |
|------|------|------|
| **Basic Block** | Conv3×3 → BN → ReLU → Conv3×3 → BN → (+) → ReLU | ResNet-18, ResNet-34 |
| **Bottleneck Block** | Conv1×1 → BN → ReLU → Conv3×3 → BN → ReLU → Conv1×1 → BN → (+) → ReLU | ResNet-50/101/152 |

Bottleneck 中 $1 \times 1$ 卷积先降维再升维（如 256 → 64 → 256），大幅减少计算量。

**BatchNorm 位置**：Conv → BN → ReLU（BN 在激活函数前）

**Shortcut 匹配**：当维度不匹配时，使用 $1 \times 1$ 卷积投影：
$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + W_s \mathbf{x}$$

🟢 来自资料

### 5.5.6 ResNet 变体

**ResNeXt (2017)**：
- 引入 **分组卷积 (Grouped Convolution)**，将通道分成 $G$ 组分别卷积
- 核心思想：通过 "cardinality"（分组数）扩展网络，而非深度/宽度
- Bottleneck 中 $3 \times 3$ 卷积替换为 $G$ 组分组卷积

**Wide ResNet (2016)**：
- 减少深度但增加每层宽度（通道数）
- 实验表明：宽而浅 > 窄而深（训练更快，梯度传播更好）

🟢 来自资料

### 5.5.7 DenseNet (2017)

**核心思想：密集连接 (Dense Connectivity)**

$$\mathbf{x}_l = H_l([\mathbf{x}_0, \mathbf{x}_1, ..., \mathbf{x}_{l-1}])$$

第 $l$ 层接收所有前层的特征图拼接作为输入。

**关键概念**：

| 概念 | 定义 | 典型值 |
|------|------|--------|
| **Growth Rate $k$** | 每层产生的**新**特征图数量 | 12, 24, 32 |
| **Bottleneck (DenseNet-B)** | $1 \times 1$ 卷积先降维 | 4k |
| **Transition Layer (DenseNet-C)** | 压缩通道，$\theta < 1$ | $\theta = 0.5$ |
| **Compression** | Transition 中通道压缩因子 | 0.5 |

**第 $l$ 层输入通道数**：$k_0 + k \times (l-1)$，其中 $k_0$ 为输入通道数。

🟢 来自资料

**优点**：缓解梯度消失、特征重用、参数效率高（$k$ 小但效果不差）

### 5.5.8 MobileNet (2017)

**目标**：适用于移动端的高效网络。

**核心：Depthwise Separable Convolution**

- MobileNet v1：深度可分离卷积 + 宽度乘子 $\alpha$ + 分辨率乘子 $\rho$
- MobileNet v2：引入 **Inverted Residual + Linear Bottleneck**
  - Expansion → Depthwise Conv → Projection（先升维、后降维）
  - 最后 $1 \times 1$ 投影无激活函数（linear）
- MobileNet v3：NAS 搜索 + 加入 SE (Squeeze-and-Excitation) 模块

🟢 来自资料

### 5.5.9 EfficientNet (2019)

**核心：Compound Scaling（复合缩放）**

传统方法单独缩放深度、宽度或分辨率。EfficientNet 统一三者：

$$
\begin{aligned}
d &= \alpha^\phi \quad &\text{(深度)} \\
w &= \beta^\phi \quad &\text{(宽度)} \\
r &= \gamma^\phi \quad &\text{(分辨率)}
\end{aligned}
$$

约束条件：$\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$（确保总计算量约翻倍）

- 先用 NAS 搜索得到 EfficientNet-B0 基础网络
- 然后通过复合缩放得到 B1–B7
- 使用 MBConv 块（MobileNet v2 的 inverted bottleneck + SE）

🟢 来自资料

---

## 5.6 Architecture Comparison

| 模型 | 年份 | 参数量 | Top-5 Error (ImageNet) | 层数 | 核心创新 |
|------|------|--------|------------------------|------|----------|
| LeNet-5 | 1998 | ~60K | — | 5 | 第一个成功的CNN |
| AlexNet | 2012 | ~60M | 15.3% | 8 | ReLU, Dropout, Data Aug, GPU |
| VGG-16 | 2014 | ~138M | 7.3% | 16 | 全3×3卷积, 深网络 |
| VGG-19 | 2014 | ~144M | 6.8% | 19 | 更深的VGG |
| GoogLeNet v1 | 2014 | ~5M | 6.7% | 22 | Inception Module, 1×1 Bottleneck |
| ResNet-50 | 2015 | ~25M | 5.3% | 50 | Residual Learning, Bottleneck |
| ResNet-101 | 2015 | ~44M | 4.6% | 101 | 更深的残差 |
| ResNet-152 | 2015 | ~60M | 4.5% | 152 | 极深残差网络 |
| DenseNet-121 | 2017 | ~8M | 5.0% | 121 | Dense Connections, Growth Rate |
| DenseNet-201 | 2017 | ~20M | 4.7% | 201 | 更深的密集连接 |
| ResNeXt-101 | 2017 | ~44M | 4.4% | 101 | Grouped Convolutions |
| MobileNet v1 | 2017 | ~4.2M | 10.5% | 28 | Depthwise Separable Conv |
| MobileNet v2 | 2018 | ~3.4M | ~8.0% | 53 | Inverted Residual, Linear Bottleneck |
| EfficientNet-B7 | 2019 | ~66M | 2.9% | — | Compound Scaling |

🟢 来自资料 / 🟡 AI补充（参数量与Top-5 Error数值综合文献）

---

## 5.7 CNN Visualization（CNN 可视化）

### 5.7.1 Filter Visualization（滤波器可视化）

直接可视化第一层卷积核（$3 \times 3 \times 3$ 可直接显示为彩色图案）：

- 低层卷积核：边缘检测器、颜色检测器、纹理检测器
- 高层卷积核：更复杂但更抽象，难以直接解释

更深层通过 **梯度上升 (Gradient Ascent)** 生成最大化激活的图像：
$$x^* = \arg\max_x \text{activation}_{neuron}(x) - \lambda \mathcal{R}(x)$$

🟢 来自资料

### 5.7.2 Saliency Maps（显著性图）

给定输入 $I$ 和类别 $c$，计算类别得分对输入像素的梯度：

$$S(I) = \left|\frac{\partial y_c}{\partial I}\right|$$

高梯度区域 = 对分类决策重要的区域。

🟢 来自资料

### 5.7.3 Grad-CAM (Gradient-weighted Class Activation Mapping)

**步骤**：

1. 计算类别 $c$ 得分对最后一层卷积特征图 $A^k$ 的梯度
2. 全局平均池化得到权重：
   $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y_c}{\partial A_{ij}^k}$$

3. 加权组合后 ReLU：
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

ReLU 确保只关注对类别有 **正向贡献** 的区域。

🟢 来自资料（Grad-CAM 论文）

---

## 5.8 Practice Problems

### Problem 1: Output Size Calculation

> 输入尺寸 $32 \times 32 \times 3$，使用 6 个 $5 \times 5$ 卷积核，步长 $S=1$，填充 $P=2$。求输出尺寸和参数量。

**Solution**：

$$W_{out} = \left\lfloor \frac{32 - 5 + 2 \times 2}{1} \right\rfloor + 1 = 32$$

输出尺寸：$32 \times 32 \times 6$。参数量：$5^2 \times 3 \times 6 + 6 = 456$。

---

### Problem 2: Receptive Field

> VGG 使用堆叠 $3 \times 3$ 卷积。经过三个 $3 \times 3$ 卷积（$S=1$）后，输出神经元在原始输入上的感受野大小是多少？等价于多大的单层卷积？

**Solution**：

$$\text{RF}_l = \text{RF}_{l-1} + (F - 1) \times \prod_{i=1}^{l-1} S_i$$

$$\text{RF}_1 = 3,\quad \text{RF}_2 = 3 + (3-1) \times 1 = 5,\quad \text{RF}_3 = 5 + (3-1) \times 1 = 7$$

感受野 = $7 \times 7$，等价于一层 $7 \times 7$ 卷积。

---

### Problem 3: Residual Block

> 解释为什么 Residual Block 中 BatchNorm 放在 ReLU 之前，以及为何 Bottleneck 最后 $1 \times 1$ 卷积后不加 ReLU。

**Solution**：
- **BN → ReLU**：先归一化再激活，确保 ReLU 输入分布稳定（预激活设计，Pre-activation）
- **Bottleneck 末尾不加 ReLU**：升维后的特征需与 shortcut 直接相加。最后是线性映射，ReLU 放在 shortcut 相加之后。

---

### Problem 4: MobileNet FLOPs

> 输入 $14 \times 14 \times 64$，输出 $14 \times 14 \times 128$，卷积核 $3 \times 3$。比较标准卷积和深度可分离卷积的参数量和计算量。

**Solution**：

**标准卷积**：参数量 = $3^2 \times 64 \times 128 = 73,728$，FLOPs ≈ $2 \times 73,728 \times 14 \times 14 = 28.9$M

**深度可分离卷积**：
- Depthwise：$3^2 \times 64$ 参数，FLOPs $2 \times 9 \times 64 \times 14 \times 14 = 0.23$M
- Pointwise：$1 \times 64 \times 128$ 参数，FLOPs $2 \times 8192 \times 14 \times 14 = 3.21$M
- 总 FLOPs ≈ 3.44M，约为标准卷积的 $11.9\%$。

---

*Last updated: 2026-07-02*
