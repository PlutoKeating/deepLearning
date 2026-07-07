# Ch10: Video Understanding (视频理解)

> 🟢 来自资料 — 基于课程讲义 `10_Video_Understanding.pdf` 及经典论文 (Two-Stream, C3D, I3D, TSN, SlowFast, TimeSformer)

---

## 1. Video as 3D Data (将视频作为 3D 数据)

与图像 ($H \times W \times 3$) 不同，视频引入了**时间维度 (Temporal Dimension)**：

$$\mathbf{V} \in \mathbb{R}^{T \times H \times W \times 3}$$

其中 $T$ 是帧数（时间步），$H$ 和 $W$ 是空间分辨率，3 代表 RGB 通道数。

**核心挑战 (Key Challenge)**：时间维度的增加大幅提升了计算开销，并要求模型对**运动 (Motion)** —— 即物体、人物和场景随时间推移的演变过程 —— 进行显式或隐式的建模。如果只是简单粗暴地独立处理每一帧，会丢失对动作理解极其关键的时序信息。

---

## 2. Key Tasks (核心任务)

| 任务名称 | 任务描述 | 示例输入 → 输出 |
|------|-------------|----------------------|
| **动作识别 (Action Recognition)** | 对一段剪辑好的短视频片段进行动作分类 | "弹吉他", "跑步" |
| **时序动作定位 (Temporal Action Localization)** | 在未经剪辑的长视频中检测动作的起止时间并分类 | 起止时间戳 + 动作标签 |
| **时空动作检测 (Spatio-temporal Action Detection)** | 在空间和时间上同时检测动作（输出边界框管道 Tube） | 每一帧的检测框序列 + 动作标签 |
| **视频描述生成 (Video Captioning)** | 生成一段描述视频内容的自然语言文本 | "一个男人在案板上切西红柿" |
| **视频问答 (Video Question Answering)** | 回答关于视频内容的问题 | "汽车是什么颜色的？" |
| **视频物体分割 (Video Object Segmentation)** | 在跨多帧的视频中对物体进行像素级分割 | 随时间变化的像素掩码 (Mask) |

> 🟢 来自资料 — 视频理解的核心是时空联合建模，动作识别是最经典的基础任务。

---

## 3. Early Approaches (早期方法)

### 3.1 Single-Frame CNN Baseline (单帧 CNN 基线)

最简单的方法：对视频的每一帧独立应用 2D CNN，然后将所有时间步的预测结果进行平均或池化。这种方法对于依赖静态背景和静态目标即可识别的动作（如“在森林里跋涉”）出奇地有效，但对于依赖运动时序的动作（如“开门”与“关门”）则完全失效。

### 3.2 Two-Stream Networks (双流网络, Simonyan & Zisserman, 2014)

使用两个并行的 CNN 分支，一个捕获**空间 (Spatial)** 外观信息，一个捕获**时间 (Temporal)** 运动信息：

- **空间流 (Spatial Stream)**：输入 = 单张 RGB 图像帧；捕获场景与物体的视觉外观。
- **时间流 (Temporal Stream)**：输入 = 连续多帧堆叠的**光流 (Optical Flow)** 图（通常是 10 帧连续光流）；直接显式地捕获运动模式。

这两个流都采用标准的 2D CNN（例如 VGG）。两者的预测结果在后期进行融合（后期融合 Late Fusion：如将得分做平均，或在拼接后的特征上使用 SVM）：

$$P(c|\mathbf{V}) = \frac{1}{2}\left[P_{\text{rgb}}(c|I) + P_{\text{flow}}(c|\Phi)\right]$$

**训练策略**：空间流通常使用 ImageNet 预训练模型进行微调。时间流则由于输入格式不同，通常需要从头训练，或者采用特殊的跨模态预训练。

**局限性：**
- 光流提取非常耗费计算资源，无法实现完全端到端 (End-to-end) 的训练与推理。
- 仅对短片段（~10 帧）进行建模，容易丢失长程时序结构 (Long-range Temporal Structure)。
- 两个流是完全独立的网络，计算无法共享。

> 🟢 来自资料 — Two-Stream 开创了显式建模运动信息（光流）的范式，在深度学习时代早期取得最佳结果。

---

## 4. 3D CNNs (三维卷积神经网络)

### 4.1 3D Convolution (3D 卷积)

通过增加一个时间方向上的卷积核维度，将 2D 卷积扩展到 3D 卷积：

$$\text{(3D Conv)} \quad y_{t,i,j} = \sum_{k=0}^{K_t-1} \sum_{u=0}^{K_h-1} \sum_{v=0}^{K_w-1} W_{k,u,v} \cdot x_{t+k, i+u, j+v} + b$$

其中 $K_t, K_h, K_w$ 是时间与空间维度的卷积核尺寸。一个 3D 卷积核的形状为 $K_t \times K_h \times K_w \times C_{\text{in}} \times C_{\text{out}}$。

与 2D 卷积相比，3D 卷积的参数量正比于 $K_t \times K_h \times K_w$（而 2D 仅为 $K_h \times K_w$）。

### 4.2 C3D (2014)

一种完全由 $3 \times 3 \times 3$ 卷积核构建的简单 3D CNN：
- 包含 8 个卷积层 + 5 个池化层 + 2 个全连接层。
- 输入：16 帧的短视频片段。
- 所有的卷积核尺寸均为 $3 \times 3 \times 3$，步长均为 $1 \times 1 \times 1$。
- 最大池化尺寸：$2 \times 2 \times 2$（第一层池化层除外，其尺寸为 $1 \times 2 \times 2$，以保留初始的时间步信号）。

**核心研究结论**：全网络采用固定的 $3 \times 3 \times 3$ 卷积核可以稳定获得比其他混用尺寸内核更好的表现。

**局限性**：网络容量较小，计算开销大，且只能处理 16 帧的极短时序。

### 4.3 I3D (Inflated 3D ConvNets / 膨胀 3D 卷积网络, 2017)

**核心思想**：通过将 2D 图像领域的 ImageNet 预训练模型"膨胀"为 3D 模型，实现参数复用。

**膨胀操作 (Inflation)**：获取一个预训练好的 2D 卷积滤波器 $W^{2D} \in \mathbb{R}^{K_h \times K_w \times C_{\text{in}} \times C_{\text{out}}}$，并将其沿时间轴**膨胀 (Inflate)** 为 3D：

$$W^{3D}_{k,u,v} = \frac{1}{K_t} W^{2D}_{u,v}$$

其实质是将 2D 权重复制 $K_t$ 次并除以 $K_t$ 做归一化（保证前向传播输出相同的方差）。对于池化层，则将 $2 \times 2$ 空间池化膨胀为 $1 \times 2 \times 2$（或 $2 \times 2 \times 2$）。

**双流 I3D (Two-Stream I3D)**：结合了膨胀 3D 网络和双流网络设计：
- RGB 流：膨胀的 Inception-v1 架构。
- 光流流：输入依然是连续 10 帧的光流，同样使用膨胀网络提取。
- 在最后对两个流的预测得分进行后期融合 (Late Fusion)。

**输入**：在 25 FPS 下提取的 64 帧完整分辨率 RGB 帧。

**核心优势**：借力于 ImageNet 的大规模图像数据预训练，大幅提升了模型在小视频数据集上的泛化性能。

> 🟢 来自资料 — I3D 的 "膨胀" 思想简单而有效，将 ImageNet 预训练迁移到 3D 是视频理解的重要突破。

---

## 5. (2+1)D Convolution ((2+1)D 卷积)

将 $K_t \times K_h \times K_w$ 的 3D 卷积核拆分为两个步骤：
- **空间卷积 (Spatial Convolution)**：尺寸为 $1 \times K_h \times K_w$（在每一帧上分别进行 2D 卷积）。
- **时间卷积 (Temporal Convolution)**：尺寸为 $K_t \times 1 \times 1$（沿时间维度进行 1D 卷积）。

并在两个步骤之间加入非线性激活函数 (ReLU)：

$$\text{(2+1)D Conv}(x) = \text{Conv}_{K_t \times 1 \times 1}\left(\text{ReLU}\left(\text{Conv}_{1 \times K_h \times K_w}(x)\right)\right)$$

**优势：**
- 相比完整 3D 卷积大幅减少参数量：从 $K_t K_h K_w C_i C_o$ 降为 $K_h K_w C_i C_{\text{mid}} + K_t C_{\text{mid}} C_o$。
- 中间增加的非线性激活函数 (ReLU) 使网络的表达能力翻倍。
- 参数被解耦，梯度路径更优，网络更容易优化。

**R(2+1)D 网络**：使用 (2+1)D 模块完全替换标准 ResNet 中的所有 3D 卷积得到的架构。

> 🟡 AI补充: (2+1)D 分解借鉴了 Inception 中的空间-通道可分离卷积思想，提供了一种参数高效的 3D 建模方案。

---

## 6. Temporal Segment Networks (TSN / 时序分段网络, 2016)

**研究痛点**：3D CNN 极其沉重，而双流网络只能建模极短的上下文。

**时序分段网络 (TSN)** 采用了一种**稀疏时间采样 (Sparse Temporal Sampling)** 的方案：

1. 将一段长视频均匀地切分为 $K$ 个等长的片段 (Segments)。
2. 从每个片段中随机抽取 1 个短视频帧对（Snippet）。
3. 抽取的每个短片段被输入到一个**权重共享**的双流网络（空间流 + 时间流）中。
4. **分段共识 (Segmental Consensus)**：将 $K$ 个短片段预测的类别得分进行聚合：
   - 方式包括对得分取平均、最大值池化、或者加权求和等。
   - 也可以在分类前将特征图进行拼接。

$$L(y, G) = -\sum_{i=1}^{C} y_i \left(G_i - \log \sum_{j=1}^{K} \exp(G_i^j)\right)$$

**核心优势**：仅仅通过 $K$ 个稀疏采样出的帧对就对整部视频的长程时序结构进行了建模，相比稠密帧计算，极大地节省了计算开销。

**训练技巧**：跨模态预训练、局部批归一化 (Partial BN：冻结空间流中除第一层外的所有批归一化层)、强 Dropout 等。

> 🟢 来自资料 — TSN 的稀疏采样策略使得模型可以高效地覆盖整个视频的时间跨度，是长视频理解的关键技术。

---

## 7. SlowFast (2019)

这是一种受生物视觉启发（灵长类动物视网膜中的小细胞 Parvocellular 和大细胞 Magnocellular 双通路机制）而设计的非对称双通路架构：

| 特性 | 慢速通路 (Slow Pathway) | 快速通路 (Fast Pathway) |
|--|-------------|--------------|
| **采样帧率** | 低（输入 $\tau$ 帧，如每 $\alpha=8$ 帧中抽 1 帧） | 高（输入 $\alpha\tau$ 帧） |
| **通道维度** | 宽（通道数 $C$ 大，代表高参数容量） | 窄（通道数 $\beta C$ 小，例如 $\beta = 1/8$） |
| **核心职责** | 捕获**空间语义信息 (Spatial Semantics)**（"是什么"） | 捕获**时间运动信息 (Temporal Motion)**（"怎么动"） |
| **输入特点** | 低时序分辨率，全空间特征提取 | 高时序分辨率，高帧率但计算量极轻 |

**核心设计点：**
- **横向连接 (Lateral Connections)**：只将快速通路的特征融合到慢速通路（由于慢速通路不具备细粒度的时序分辨率，故反向不成立）：
  
  快速通路的特征被重塑形状（从 $T \times C_{\text{fast}} \times H \times W$ 转为 $T/\alpha \times \alpha C_{\text{fast}} \times H \times W$），随后与慢速通路特征沿通道维度进行拼接 (Concatenation)。

- **快速通路不进行任何时间方向的下采样**：确保捕捉到精细的时间过渡与高速运动。

**实验结果**：在 Kinetics-400/600 视频数据集上性能强劲，而计算开销低于普通的 3D 卷积 ResNet。

> 🟢 来自资料 — SlowFast 用非对称的双通路设计高效解耦了空间语义和时间运动，是 3D ConvNet 时代的重要创新。

---

## 8. Video Transformers (视频 Transformer)

### 8.1 TimeSformer (2021)

将 Vision Transformer 成功应用至视频领域，引入了**分解时空注意力 (Divided Space-Time Attention)** 机制以降低二次方计算复杂度：

在 $T \times H \times W$ 张图片块 (Patches) 上进行标准的自注意力计算复杂度正比于 $O((S \cdot T)^2)$（其中 $S = H \times W$ 是每一帧的 Patch 数量）。TimeSformer 对自注意力计算进行了时间与空间方向的解耦：

1. **空间自注意力 (Spatial Attention)**：每一帧内的 Patch 仅在**当前帧内**相互计算注意力。
2. **时间自注意力 (Temporal Attention)**：相同的空间位置上的 Patch 沿**所有时间帧**跨帧计算注意力。

计算量对比：

$$O(S^2 \cdot T + T^2 \cdot S) \ll O((S \cdot T)^2)$$

**整体架构**：由 $L$ 个 Transformer 块堆叠而成，每个块内先后应用空间注意力与时间注意力（即“分解时空”注意力机制）。

### 8.2 ViViT (2021)

提出了几种拆分时空注意力机制的变体：

1. **先空间后时序 (Spatial-then-temporal)**（分解编码器）：与 TimeSformer 类似。
2. **分解点积 (Factorized dot-product)**：在计算注意力时，空间与时间权重被解耦为独立的点积，然后相乘。
3. **管状体嵌入 (Tubelet Embedding)**：从视频中提取 3D 的非重叠时空“管道”（即 3D 补丁块）进行初始的嵌入编码，从而减小输入序列长度，然后在简化的序列上应用全时空联合注意力。

> 🟢 来自资料 — Video Transformers 通过时空注意力解耦克服了 Transformer 二次复杂度在视频上的计算瓶颈。

---

## 9. Optical Flow (光流)

### 9.1 Definition (光流定义)

光流 (Optical Flow) 是指图像序列中亮度模式的表观运动。给定相邻的两帧图像 $I(x, y, t)$ 和 $I(x, y, t+1)$，光流场 $\mathbf{u} = (u, v)$ 描绘了每一个像素在相邻时间步内的位移向量。

**亮度恒定假设 (Brightness Constancy Assumption)**：

$$I(x, y, t) = I(x + u, y + v, t + 1)$$

**光流约束方程 (Optical Flow Constraint Equation)**（对其进行一阶泰勒展开）：

$$\frac{\partial I}{\partial x} u + \frac{\partial I}{\partial y} v + \frac{\partial I}{\partial t} = 0$$

$$\nabla I \cdot \mathbf{u} + I_t = 0$$

由于此方程是一个包含两个未知数 $(u, v)$ 的单方程，在局部尺度上无法求出唯一解。这被称为**孔径问题 (Aperture Problem)**。在传统物理方法中，需要加入额外的约束（如平滑性假设）来获得唯一解。

### 9.2 Learning-Based Methods (基于深度学习的光流估计)

| 算法名称 | 算法原理 |
|--------|----------|
| **FlowNet** (2015) | 首个端到端 CNN 光流模型：包括 FlowNetS（将图像堆叠作为输入）与 FlowNetC（引入专门的相关层 Correlation Layer），并使用 U-Net 架构进行微调。 |
| **FlowNet 2.0** (2017) | 通过多级 FlowNet 的串联与图像变形对光流进行递进估计，并针对大、小位移设计了专门的子网络。 |
| **PWC-Net** (2018) | 包含金字塔 (Pyramid)、变形 (Warping) 和代价体积 (Cost Volume)：先构建特征金字塔，利用前一级的估计变形特征，在当前层构建代价体积并估计流场，模型十分轻量高效。 |
| **RAFT** (2020) | 建立全像素对的相关体 (Correlation Volume)，并使用基于 GRU 的循环迭代结构从零初始流场逐步更新，大幅提升了光流估计的精度，代表当前最先进水平。 |

> 🟢 来自资料 — RAFT 通过全像素对相关体 + 迭代 GRU 更新取得了光流估计的突破性性能。

---

## 10. Self-Supervised Video Representation Learning (自监督视频表征学习)

视频固有的时间属性为自监督学习提供了海量且完全免费的天然监督信号：

| 算法名称 | 代理任务 (Pretext Task) | 任务原理描述 |
|--------|-------------|-------------|
| **Shuffle & Learn** | 时序顺序验证 | 判断给定的连续帧序列是正常的顺序，还是被随机打乱了？ |
| **Arrow of Time** | 时间方向预测 | 预测视频是在正放还是倒放？ |
| **SpeedNet** | 播放速度预测 | 预测视频是以几倍速（如正常、加速、减速）在播放？ |
| **VCOP** | 视频片段排序 | 给多段被打乱的视频片段进行升序排列 |
| **Pace Prediction** | 节奏预测 | 预测视频在时间维度上的膨胀因子 |
| **CVRL** | 对比学习 (SimCLR style) | 时序数据增强：从同一视频中截取的两个片段被视作正样本对。 |
| **VideoMAE** | 掩码自编码器 | 遮盖掉极其高比例（90%）的视频 Patch，要求网络重构缺失区域。 |

**核心发现**：视频中的时间连贯性（相邻帧高度相似）和时间动力学特征（如顺序、速度）提供了极其丰富且可靠的表征。

> 🟡 AI补充: 视频中的时间维度为自监督学习提供了丰富的免费监督信号，适合利用大规模无标注视频数据预训练。

---

## 11. Summary Formula Cheat Sheet (公式汇总速查)

| 核心概念 | 数学公式 |
|---------|---------|
| 3D 卷积计算 (3D Conv) | $y_{t,i,j} = \sum_k \sum_u \sum_v W_{k,u,v} \cdot x_{t+k, i+u, j+v}$ |
| (2+1)D 拆分卷积 | $\text{Conv}_{K_t \times 1 \times 1} \circ \text{ReLU} \circ \text{Conv}_{1 \times K_h \times K_w}$ |
| 光流基本约束方程 | $\nabla I \cdot \mathbf{u} + I_t = 0$ |
| 双流得分后期融合 | $P(c) = \frac{1}{2}[P_{\text{rgb}}(c) + P_{\text{flow}}(c)]$ |
| TimeSformer 复杂度 | $O(S^2 T + T^2 S)$ (相比完整时空的 $O(S^2 T^2)$) |

---

## 12. Practice Problems (练习题与详解)

### Problem 1: 3D Conv Parameters (3D 卷积参数量计算)
考虑一个 (2+1)D 卷积模块，其输入通道数为 $C_{\text{in}} = 64$，输出通道数为 $C_{\text{out}} = 128$，空间卷积核大小为 $K_h = K_w = 3$，时间卷积核大小为 $K_t = 3$。中间空间卷积输出的通道数（即中间层通道数）设为 $C_{\text{mid}} = 128$。请问：
a) 该 (2+1)D 卷积包含多少个可学习权重参数？
b) 相较于对应的标准 3D 卷积，该拆分操作能节省多少参数量？

**Solution (解析):**
- a) (2+1)D 包含两个步骤：
  - 空间卷积层参数（核大小为 $1 \times 3 \times 3$）：$1 \times 3 \times 3 \times C_{\text{in}} \times C_{\text{mid}} = 9 \times 64 \times 128 = 73,728$
  - 时间卷积层参数（核大小为 $3 \times 1 \times 1$）：$3 \times 1 \times 1 \times C_{\text{mid}} \times C_{\text{out}} = 3 \times 128 \times 128 = 49,152$
  - 总可学习权重参数量（忽略偏置）：$73,728 + 49,152 = 122,880$
- b) 对应的标准 3D 卷积（核大小为 $3 \times 3 \times 3$）的参数量为：
  - $3 \times 3 \times 3 \times C_{\text{in}} \times C_{\text{out}} = 27 \times 64 \times 128 = 221,184$
  - 参数节省比例：$\frac{221,184 - 122,880}{221,184} \approx 44.4\%$。

### Problem 2: TSN Snippet Aggregation (TSN 得分融合计算)
某 TSN 算法将视频分为 $K=3$ 个片段进行稀疏采样，模型在三分类任务上对这三个短片段输出的 Logits（未归一化得分）分别如下：
- 片段 1: $[2.0, 0.5, -1.0]$
- 片段 2: $[1.5, 1.0, 0.0]$
- 片段 3: $[0.8, 0.3, 1.5]$
请采用 **平均池化 (Average Pooling) + Softmax** 方式计算该视频最终被预测到各个类别的概率。

**Solution (解析):**
- 第一步，对 Logits 在所有片段维度上求均值（平均池化）：
  - 类别 0 的平均得分：$\frac{2.0 + 1.5 + 0.8}{3} = \frac{4.3}{3} \approx 1.433$
  - 类别 1 的平均得分：$\frac{0.5 + 1.0 + 0.3}{3} = \frac{1.8}{3} = 0.600$
  - 类别 2 的平均得分：$\frac{-1.0 + 0.0 + 1.5}{3} = \frac{0.5}{3} \approx 0.167$
  - 平均 Logits 向量：$[1.433, 0.600, 0.167]$
- 第二步，计算对应的指数值并进行归一化：
  - $e^{1.433} \approx 4.191$
  - $e^{0.600} \approx 1.822$
  - $e^{0.167} \approx 1.182$
  - 总和 $\sum = 4.191 + 1.822 + 1.182 = 7.195$
  - 归一化后的 Softmax 概率：
    - 类别 0：$\frac{4.191}{7.195} \approx 0.583$
    - 类别 1：$\frac{1.822}{7.195} \approx 0.253$
    - 类别 2：$\frac{1.182}{7.195} \approx 0.164$
- 最终各类别预测概率向量为 $[0.583, 0.253, 0.164]$。

### Problem 3: Aperture Problem (孔径问题原理分析)
若一条垂直的边缘正在图像中进行纯水平方向的运动，我们在时刻 $t$ 的像素位置 $(10, 5)$ 观测到的亮度和下一时刻 $t+1$ 的像素位置 $(11, 5)$ 观测到的亮度完全相同。
a) 请写出此时的光流约束方程。
b) 为什么该约束是不完备的？

**Solution (解析):**
- a) 由于这是一条纯垂直的像素边缘，在其垂直延伸方向上没有灰度变化，故空间纵向梯度 $\frac{\partial I}{\partial y} = 0$。
  - 代入光流约束方程：$\frac{\partial I}{\partial x} u + 0 \cdot v + I_t = 0$，整理得：$\frac{\partial I}{\partial x} u + I_t = 0$。
  - 我们可以据此求得水平运动速度 $u = -\frac{I_t}{\partial I / \partial x}$。
- b) 该约束不完备的原因在于：
  - 纵向速度分量 $v$ 的系数为 0，导致 $v$ 被完全隐藏在方程中。
  - 只要水平位移满足要求，垂直方向上的任何移动量 $v$ 都可以得到相同的亮度分布。这就是**孔径问题 (Aperture Problem)**。在局部视野较窄（如通过一个狭窄的狭缝或圆形孔径观察）的情况下，仅凭一维边缘无法确定物体的 2D 实际位移方向，只有观测到二维角点或复杂纹理结构时，方程系统才可能达到满秩状态以解出确定值。

### Problem 4: SlowFast Channel Distribution (SlowFast 通道分配)
某一 SlowFast 网络设置时间比率 $\alpha = 8$，通道权重比率 $\beta = 1/8$。如果慢速通路在 8 FPS 采样率下，首层卷积层的通道数配置为 64 维。请问：
a) 快速通路的采样帧率是多少？
b) 快速通路首层卷积层的通道数应该设计为多少维？

**Solution (解析):**
- a) 快速通路的采样帧率为慢速通路的 $\alpha$ 倍：
  - $\text{FPS}_{\text{fast}} = \text{FPS}_{\text{slow}} \times \alpha = 8 \times 8 = 64$ FPS。
- b) 快速通路的通道维数为慢速通路的 $\beta$ 倍：
  - $\text{Channel}_{\text{fast}} = \text{Channel}_{\text{slow}} \times \beta = 64 \times \frac{1}{8} = 8$ 维。

### Problem 5: Temporal Attention in TimeSformer (TimeSformer 注意力开销计算)
有一段 $T=96$ 帧的视频，每一帧被切分为 $14 \times 14 = 196$ 个空间 Patch 块。
a) 若进行标准的联合时空注意力 (Joint Space-Time Attention) 计算，输入序列的总块数 $N$ 是多少？
b) 请对比分析标准联合自注意力与分解时空自注意力 (Divided Space-Time Attention) 的基本乘加次数开销。

**Solution (解析):**
- a) 联合时空自注意力机制需要将所有帧的所有 Patch 拼接到一整条序列中：
  - 输入序列总块数 $N = T \times S = 96 \times 196 = 18,816$ 个 Patch。
- b) 两者乘加次数开销对比（与序列长度的平方成正比）：
  - **标准联合时空注意力**的复杂度为：
    - $O(N^2) = O(18,816^2) \approx 3.54 \times 10^8$ 次运算。
  - **分解时空自注意力**将注意力拆分为 $T$ 个空间注意力与 $S$ 个时间注意力：
    - 空间注意力复杂度：$T \times O(S^2) = 96 \times 196^2 = 3,687,936$ 次。
    - 时间注意力复杂度：$S \times O(T^2) = 196 \times 96^2 = 1,806,336$ 次。
    - 总复杂度：$3,687,936 + 1,806,336 = 5,494,272$ 次运算。
  - 在本题场景下，分解时空注意力比标准联合注意力在计算开销上节省了约 **64 倍**，这极大地增强了 Transformer 处理长视频的能力。

---

> 🟢 来自资料 — 视频理解从手工光流特征演进到端到端 3D CNN 再到 Video Transformer，核心是计算效率和长程时空建模的折中。