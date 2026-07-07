# Ch11: Self-Supervised Learning (自监督学习)

> 🟢 来自资料 — 基于课程讲义 `11_Self-Supervised Learning.pdf` 及 SimCLR, MoCo, BYOL, SimSiam, DINO, MAE, BEiT 等论文

---

## 1. Motivation: The Labeling Bottleneck (动机：标注瓶颈)

监督式深度学习需要海量的标注数据集（例如 ImageNet：120 万张图片，MS COCO：33 万张图片）。然而，人工数据标注面临着以下难以克服的痛点：
- **高昂的成本 (Expensive)**：每张图片都需要人工进行精细的分类或边界框标注。
- **范围局限性 (Limited)**：标注的数据无法轻松扩展到世界上所有的概念和专业领域（如医疗、罕见场景）。
- **主观偏置 (Biased)**：标注人员的主观想法会导致标注质量的不一致和标注误差。

自监督学习 (Self-Supervised Learning, SSL) 的核心理念在于**利用数据本身的结构信息作为监督信号** —— 模型无需外部标注，而是通过解决由数据自身自动生成的**代理任务 (Pretext Tasks / 前置任务)** 来学习高质量的特征表征。

**核心目标 (Core Objective)**：学习一个通用的特征表征函数 $f_\theta: \mathcal{X} \to \mathbb{R}^d$，捕获有益于下游任务（如分类、检测、分割）的深层语义信息。

> 🟢 来自资料 — SSL 用数据自身的结构信号取代人工标签，是缓解标注瓶颈的核心方法。

---

## 2. Pretext Tasks (代理任务 / 前置任务)

早期的自监督学习（自监督 1.0 时代）主要通过设计手工构造的代理任务来自动产生标签进行监督：

| 任务名称 | 自动生成的监督信号 | 代理任务工作原理描述 |
|------|-------------------|--------------|
| **旋转预测 (Rotation Prediction)** | 四分类 (0°, 90°, 180°, 270°) | 预测对图像应用了多少度的随机旋转角度 |
| **拼图游戏 (Jigsaw Puzzle)** | 随机排列索引 (Permutation Index) | 将图像切分成多个小块并打乱顺序，预测正确的排列索引 |
| **图像着色 (Colorization)** | 原始色彩通道值 | 输入去色后的灰度图像，预测其原始的彩色通道数据 |
| **图像修复 (Inpainting)** | 被遮掩区域的原始像素 | 随机遮掩图像中的某一块区域，让网络重构并填补缺失的像素值 |
| **相对位置预测 (Relative Patch Location)** | 八分类（周围八个方向） | 给定一个中心图像块，预测另一个随机邻近块相对于它的相对方向 |
| **样本卷积网络 (Exemplar-CNN)** | 代理类别标签 | 将每一张原始图像视为一个单独的类，通过数据增强创建变体并训练分类器 |
| **数量统计 (Counting)** | 目标物体合成数量 | 让模型预测图像中人工合成的多副本物体数量 |

**局限性 (Limitations)：**
- 手工设计的代理任务通常过于专注于局部或纹理特征，往往难以学到通用的高级高层语义信息。
- 这些任务专为特定的图像格式设计，很难推广到其他多模态领域。
- 特征表征的质量相比于全监督训练仍存在较大的性能落差。

> 🟡 AI补充: Pretext task 时代奠定了 SSL 的基本思想，但表示质量受限于任务设计，缺乏通用性。

---

## 3. Contrastive Learning Paradigm (对比学习范式)

### 3.1 Core Idea (核心思想)

对比学习的核心思想是：在特征嵌入空间中，**拉近正样本对 (Positive Pairs)**（即同一张图片的不同增强版本）的表示距离，同时**推开负样本对 (Negative Pairs)**（即不同图片）的表示距离。

$$\text{sim}(\mathbf{z}_i, \mathbf{z}_j) \text{ 对于正样本对 } (i,j) \text{ 极大，对于负样本对极小}$$

- **正样本对 (Positive Pairs)**：从同一张输入图像通过不同的随机数据增强组合（如随机裁剪、颜色抖动、高斯模糊）生成的两个视图（Views）。
- **负样本对 (Negative Pairs)**：来自不同源图像的视图。在训练中，通常从大批量（Large Mini-batch）或外部的内存池（Memory Bank）中进行采样。

### 3.2 InfoNCE Loss (InfoNCE 损失函数)

**InfoNCE (Noise Contrastive Estimation)** 损失函数是对比学习的标准损失目标：

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_i^+) / \tau)}{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_i^+) / \tau) + \sum_{j=1}^{K} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j^-) / \tau)}$$

其中：
- $\mathbf{z}_i$：当前查询的表示向量（锚点 Anchor）。
- $\mathbf{z}_i^+$：对应的正样本 Key（同一张图像的另一个增强视图）。
- $\mathbf{z}_j^-$：$K$ 个不相关的负样本 Key（其他不同图像的视图）。
- $\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a}^\top \mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$：代表余弦相似度 (Cosine Similarity)。
- $\tau$：温度超参数 (Temperature Hyperparameter)，较小的值会使概率分布更陡峭，让模型更加关注那些不易区分的**困难负样本 (Hard Negatives)**。

**物理本质**：这可以视为一个 $K+1$ 路的 Softmax 分类任务 —— 迫使模型在包含 1 个正样本和 $K$ 个负样本的集合中，精准地分类出谁是那个匹配的正样本。

> 🟢 来自资料 — InfoNCE 是对比学习的事实标准目标函数，源自 CPC (Contrastive Predictive Coding) 并广泛应用于视觉 SSL。

---

## 4. SimCLR (2020)

由 Chen 等人于 ICML 2020 提出的 **SimCLR (A Simple Framework for Contrastive Learning of Visual Representations)** 奠定了现代对比学习的极简范式。

### 4.1 Architecture (网络架构)

1. **组合数据增强 (Data Augmentation)**：对任意图像从一系列增强算子的**组合 (Composition)** 中随机抽取两次，生成两个不同的视图 $\tilde{x}_i$ 和 $\tilde{x}_j$：
   - 随机裁剪并缩放 (Random Crop + Resize)（核心的几何增强）。
   - 随机水平翻转 (Random Horizontal Flip)。
   - 随机颜色失真 (Color Distortion)（包括颜色抖动和灰度化 —— 极其关键，以防模型通过捷径学习颜色直方图！）。
   - 高斯模糊 (Gaussian Blur)。

2. **骨干特征编码器 (Encoder)** $f(\cdot)$：一个标准的 2D 卷积 ResNet-50，提取高维特征表示 $\mathbf{h}_i = f(\tilde{x}_i)$。

3. **非线性投影头 (Projection Head)** $g(\cdot)$：一个双层或三层的简单多层感知机 (MLP)，将高维特征映射到低维嵌入空间 $\mathbf{h}_i \to \mathbf{z}_i = g(\mathbf{h}_i)$。**极其重要**：对比学习损失是在 $\mathbf{z}$ 上进行计算的，但进行下游微调评估时会将投影头丢弃，直接采用骨干网络输出的特征 $\mathbf{h}$。

4. **双向对比损失**：对整个小批量 (Mini-batch) 计算互为正负样本的 InfoNCE 损失（对于含有 $N$ 张图的 Batch，产生 $2N$ 个增强样本，每个样本对应 1 个正样本和 $2N-2$ 个负样本）：

$$\ell(i,j) = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau)}$$

总损失：$\mathcal{L} = \frac{1}{2N} \sum_{k=1}^{N} [\ell(2k-1, 2k) + \ell(2k, 2k-1)]$

### 4.2 Key Design Choices (核心设计决策与消融结论)

| 关键设计 | 实验消融结论 |
|---------------|---------|
| **投影头 (Projection head)** | 引入非线性 MLP 头相较于直接使用特征，性能能提升高达 10%。而在评估时，将其舍弃比保留能在下游任务上获得额外 2% 的提升。 |
| **小批量大小 (Batch size)** | 越大的 Batch Size（从 256、4096 到 8192）效果越好。由于更大的 Batch 蕴含了更多负样本，训练需要特殊的 LARS 优化器以稳定大 Batch 梯度。 |
| **数据增强策略** | 多种增强算子的“组合拳”极其关键；单纯的随机裁剪或单纯的颜色失真都无法学到稳健的语义，只有两者结合才产生强大的互补性。 |
| **训练时长 (Epochs)** | 相比监督学习，自监督对比学习极度渴望更长的训练周期（800~1000 个 Epoch 才会完全收敛并提升质量）。 |
| **相似度正则化** | 在计算余弦相似度之前，必须将 $\mathbf{z}$ 进行 L2 归一化 (L2-normalize) —— 这是稳定优化的关键点。 |

> 🟢 来自资料 — SimCLR 通过系统消融实验确立了对比学习的核心组件：组合数据增强、非线性投影头、大 batch size。

---

## 5. MoCo (Momentum Contrast / 动量对比, 2020)

**研究痛点**：对比学习需要极其庞大的负样本集，但 SimCLR 式的 Batch 共享方式受制于显存限制，大 Batch（如 4096）只有依靠算力极其高昂的 TPU 集群。

**MoCo 解决方案**：将对比学习重构为**动态字典查询任务**，并用一个**先进先出队列 (Queue)** 异步维护大负样本字典，实现了负样本数与 Batch Size 的完美解耦。

### 5.1 Architecture (网络架构)

MoCo 包含三个核心组件：

1. **查询编码器 (Query Encoder)** $f_q$：标准的特征编码器，通过反向传播 (Backprop) 直接更新。
2. **键编码器 (Key Encoder)** $f_k$：其架构与查询编码器一致，但其参数通过**动量衰减 (Momentum Update)** 的方式平滑复制查询编码器的权重：
   $$\theta_k \leftarrow m \theta_k + (1 - m) \theta_q$$
   其中动量系数通常设为一个极大的值，如 $m = 0.999$，以确保 Key 表示的一致性。
3. **动态字典队列 (Queue)**：一个容量巨大的 FIFO 队列，用于存储最近几十个小批量的已编码 Key 向量（例如 $K = 65536$ 个特征向量）。

**训练演进步骤：**
1. 输入当前查询图像，由查询编码器输出：$\mathbf{q} = f_q(x^{\text{query}})$
2. 输入当前正样本图像，由键编码器输出正样本 Key：$\mathbf{k}_+ = f_k(x^{\text{key}})$
3. 与队列中存储的 $K$ 个历史 Key 计算余弦相似度，计算 InfoNCE 损失（$k_+$ 为唯一的正样本，队列中所有历史 Key 均为负样本）。
4. 将最新的 $\mathbf{k}_+$ 压入队列头部，并剔除队列尾部最陈旧的 Key。
5. 通过反向传播更新 $f_q$ 参数，随后依据动量公式更新 $f_k$ 参数。

**为什么要用动量键编码器？**
如果键编码器也和 $f_q$ 一样通过反向传播极快更新，由于队列包含多个历史 Batch，后面 Batch 的 Key 所基于的权重参数将与早期 Batch 完全不同，导致队列内的负样本表示**不一致（Representation Drift）**。使用缓慢变化的动量更新机制，保证了键编码器变化极其平缓，队列中所有 Key 的一致性得到了有力保证。

> 🟢 来自资料 — MoCo 的动量编码器 + 队列机制解耦了 batch size 与负样本数量，使得在普通 GPU 上也能使用大量负样本。

### 5.2 MoCo v2 (2020)

将 SimCLR 验证的高效组件快速吸收到了 MoCo 的框架下：
- 加入了 **MLP 非线性投影头**。
- 引入了**更强的数据增强算子**（如高斯模糊）。
- 引入了余弦退火学习率调度。

**效果**：仅仅在 8 张主流 GPU 上，即取得了胜过在 TPU 上进行巨大 Batch 训练的 SimCLR 的性能。

---

## 6. BYOL (Bootstrap Your Own Latent / 引导自身潜表征, 2020)

**核心突破：彻底颠覆了“对比学习必须依赖负样本对”的固有认知。**

### 6.1 Architecture (网络架构)

- **在线网络 (Online Network)** $f_\theta$：包含骨干编码器 + 投影头 + **预测器 (Predictor)** $q_\theta$（又一个额外的 MLP 结构）。
- **目标网络 (Target Network)** $f_\xi$：仅包含骨干编码器 + 投影头（**无预测器**），其参数不通过反向传播计算，而是通过在线网络的动量 EMA 形式平滑演进：
  $$\xi \leftarrow \tau \xi + (1 - \tau) \theta, \quad \tau = 0.996$$

**损失函数**（使用均方误差归一化后的预测向量）：

$$\mathcal{L}_{\text{BYOL}} = \left\|\frac{q_\theta(z_\theta)}{\|q_\theta(z_\theta)\|} - \frac{z_\xi}{\|z_\xi\|}\right\|_2^2 = 2 - 2 \cdot \frac{\langle q_\theta(z_\theta), z_\xi \rangle}{\|q_\theta(z_\theta)\| \cdot \|z_\xi\|}$$

**核心的不对称（Asymmetric）架构设计至关重要：**
- 在在线网络末端引入的**预测器** $q_\theta$ 打破了对称性，防止网络坍缩到退化解。
- 在目标网络分支上应用的**梯度截断 (Stop-gradient)** 机制切断了目标分支的常数退化路径。
- 如果去除上述任何一个组件，网络就会立刻陷入**坍缩 (Collapse)**（即对任意输入都预测输出恒定的全零或一常数）。

### 6.2 Why No Collapse? (为什么不会坍缩？)

动量目标网络的存在提供了一个缓慢演变（慢变）的表示 target，在线网络的预测器 $q_\theta$ 不断逼近这个 target。目标网络不直接通过优化损失更新，这在隐式上构成了一种**时序对比效果**，保证了特征表征的多样性与表达性。

> 🟢 来自资料 — BYOL 通过非对称架构（在线网络 + 动量目标网络）实现了无负样本的自监督学习，挑战了"负样本不可或缺"的认知。

---

## 7. SimSiam (2021)

**将自监督网络极致简化：不需要负样本对，也无需任何动量编码器！**

### 7.1 Architecture (网络架构)

- **两个完全对称且参数共享的编码器** $f$（无任何 EMA，两分支使用相同权重）。
- 两个增强后的视图在共享的编码器上运行，其中一个流引入了**预测器 (Predictor MLP)** 与**梯度截断 (Stop-gradient)** 机制。

对称的互预测损失函数定义：

$$\mathcal{L} = \frac{1}{2}\mathcal{D}(p_1, \text{sg}(z_2)) + \frac{1}{2}\mathcal{D}(p_2, \text{sg}(z_1))$$

其中 $\mathcal{D}$ 是负余弦相似度，$\text{sg}(\cdot)$ 代表不计算梯度的 **Stop-gradient (梯度截断)** 算子。

**核心研究结论**：**梯度截断 (Stop-gradient) 是防止自监督模型发生常数坍缩的决定性秘密武器**。SimSiam 的数学本质可以被合理抽象为一种类似期望最大化 (Expectation-Maximization, EM) 形式的交替优化求解过程。

> 🟢 来自资料 — SimSiam 证明了 stop-gradient 是防止坍缩的核心机制，动量编码器和负样本都不是必要的。

---

## 8. DINO (2021)

**将无监督条件下的“自蒸馏 (Self-distillation)”机制成功移植至 Vision Transformers (ViT) 架构。**

### 8.1 Architecture (网络架构)

- **学生网络 (Student Network)** $g_{\theta_s}$：包含 ViT 骨干和投影头，通过温度为 $\tau_s$ 的 Softmax 计算输出其 $K$ 维概率分布。
- **教师网络 (Teacher Network)** $g_{\theta_t}$：共享架构，参数通过学生网络的动量 EMA 复制，通过温度为 $\tau_t$ 的 Softmax 输出分布（设定 $\tau_t < \tau_s$ 以便获得更具判别性的**锐化 (Sharper)** 概率分布）。

### 8.2 Loss (损失函数)

计算学生网络与教师网络在不同裁剪裁剪组合（如多尺度全局、局部 Crop）下的交叉熵损失 (Cross Entropy Loss)：

$$\min_{\theta_s} -\mathbf{P}_t(\mathbf{x}) \log \mathbf{P}_s(\mathbf{x})$$

其中 $\mathbf{P}_t = \text{softmax}(g_{\theta_t}(\mathbf{x}) / \tau_t)$，$\mathbf{P}_s = \text{softmax}(g_{\theta_s}(\mathbf{x}) / \tau_s)$。

**教师更新方式**：$\theta_t \leftarrow \lambda \theta_t + (1 - \lambda) \theta_s$。

**防止坍缩的两大绝招：中心化与锐化 (Centering & Sharpening)**
- **中心化 (Centering)**：从教师的输出 Logits 中减去其小批量运行均值，防止某一个维度概率始终独占（防止统治坍缩）。
- **锐化 (Sharpening)**：使用极低的教师 Softmax 温度，防止所有的概率分布退化为均匀分布（防止平均坍缩）。

### 8.3 Emerging Properties (令人瞩目的涌现特性)

在 ViT 架构下，通过 DINO 自蒸馏训练得到的自注意力特征图（Self-Attention Maps），在其注意力头上会**自发地、不带任何像素级监督信号地涌现出清晰的“语义分割 (Semantic Segmentation)”图** —— 这极大地证明了自监督学习能够逼真地抽象出人类视野中的物体边界。

> 🟢 来自资料 — DINO 在 ViT 上的自蒸馏产生了涌现特性——自注意力图自动捕获语义分割，展示了 SSL 的潜力。

---

## 9. Masked Image Modeling (MIM / 掩码图像建模)

受到了自然语言处理领域中 BERT 模型的 Masked Language Modeling 机制启发，通过“完形填空”式的掩码图像重建进行自监督学习。

### 9.1 MAE (Masked Autoencoder / 掩码自编码器, 2022)

何恺明等人在 2022 年提出的 MAE 利用**非对称的编码器-解码器架构 (Asymmetric Encoder-Decoder)** 和**极高比例的掩码率 (High Mask Ratio)**，在视觉自监督上取得了重大成功：

1. **随机掩码 (Mask)**：将输入图片划分为不重叠的 Patches 块，并以**极高比例（如 75%）**随机丢弃、仅留存 25% 的可见块。
2. **轻量骨干编码器 (Encoder)**：使用 ViT 作为编码器，**仅仅对 25% 的可见 Patch** 进行特征计算。由于序列长度被压缩至 1/4，带来了多于 16 倍的极佳计算加速。
3. **轻量解码器 (Decoder)**：一个极其浅层的 ViT 架构，输入由“可见块编码器的输出特征”与“可学习的占位 Mask Tokens”拼接成的一整条完整长序列。该解码器仅在预训练阶段存在，下游评估时直接丢弃。
4. **重建均方损失 (MSE Loss)**：仅在**被遮蔽的 75% 图像块**上，计算预测像素值与原始像素值之间的均方误差：

$$\mathcal{L} = \frac{1}{|M|} \sum_{i \in M} \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2$$

**核心消融发现：**
- **75% 的极高掩码率最佳**：这与文本（BERT 掩码 15%）有根本区别。图像在空间上存在高度的冗余性，低掩码率会导致任务过于简单、模型只会进行插值。
- 解码器在特征重建中只需要是一个非常浅层、计算量占比低于 10% 的简单 ViT。
- 对目标 Patch 像素点进行局部均值与方差归一化后再进行重构，可以显著增强特征质量。

> 🟢 来自资料 — MAE 通过非对称编码器-解码器和高掩码率 (75%) 高效利用图像的视觉冗余，证明了掩码图像建模的可行性。

### 9.2 BEiT (2021)

将经典的 BERT 训练范式移植至 Vision Transformer：

1. 引入预训练好的**离散变分自编码器 (dVAE)**，将图像的每一个小 Patch 离散化映射为一个有限字典表中的“离散视觉 Token (Discrete Visual Token)”。
2. 遮蔽部分图像块，并将预测**对应的离散视觉 Token 的索引**作为分类任务（计算交叉熵分类损失），就像 BERT 预测缺失单词。
3. 同时结合图像整体级别的分类标签预测（在 CLS Token 上计算）。

MIM 范式属于**生成式自监督学习 (Generative SSL)**，与基于语义对比分类的对比学习形成了良好的技术互补。

> 🟡 AI补充: MIM 和对比学习是当前 SSL 的两大主流范式，MIM 更适合密集预测任务（如分割、检测），对比学习更适合语义判别任务（如分类）。

---

## 10. Joint Embedding Predictive Architecture (JEPA / 联合嵌入预测架构)

由人工智能先驱 Yann LeCun 极力推崇并提倡的架构，作为对生成式重建和语义对比学习的强力补充。

**核心思想**：避免在像素层面对高频噪声和细节进行无意义的建模，而是在**特征表征空间**对缺失的（被遮蔽的）目标区域进行语义特征预测：

$$\mathcal{L} = \|\text{Pred}(\mathbf{z}_{\text{context}}) - \mathbf{z}_{\text{target}}\|^2$$

并对目标特征分支 $\mathbf{z}_{\text{target}}$ 施加 **梯度截断 (Stop-gradient)** 以防特征坍缩。

**JEPA 架构的绝佳优势：**
- **相较于生成式模型 (MAE)**：不把宝贵的网络容量浪费在重构无意义的像素细节上。
- **相较于对比学习 (SimCLR/MoCo)**：无需经历繁琐、脆弱的增强算子设计与任何负样本对。

**I-JEPA (Image JEPA)**：使用在线上下文编码器 + 目标编码器 (EMA 更新) + 预测器。通过周围上下文的 Patch 表征预测被遮蔽目标 Patch 的深层表征，非常擅长学到高层抽象特征。

> 🟡 AI补充: JEPA 在表示空间进行预测，避免了像素空间的细节重建和负样本需求，是 LeCun 提出的通向世界模型的核心架构。

---

## 11. Comparison Table of SSL Methods (自监督算法终极对比表)

| 算法名称 | 算法机理 | 负样本依赖? | 动量更新? | 梯度截断 Stop-grad? | 网络架构基本结构 |
|--------|----------|-----------|-----------|-----------|--------------|
| **SimCLR** | 语义对比 | 是 (Batch共享) | 否 | 否 | 孪生共享网络 (Siamese) |
| **MoCo v1/v2** | 语义对比 | 是 (FIFO队列) | 是 ($\theta_k$) | 否 | 非对称双分支（在线、动量） |
| **BYOL** | 非对比表示学习 | 否 | 是 (目标分支) | 是 (目标分支) | 在线网络（含预测器）+ 动量目标网络 |
| **SimSiam** | 非对比表示学习 | 否 | 否 | 是 (一侧截断) | 孪生对称（带有单边预测器和梯度截断） |
| **DINO** | 自蒸馏机制 | 否 | 是 (教师分支) | 是 (教师分支) | 学生网络 - 教师网络（带中心化及锐化） |
| **MAE** | 掩码重建 (MIM) | 否 | 否 | 否 | 自编码器（可见块编码，完整块解码） |
| **BEiT** | 掩码分类 (MIM) | 否 | 否 | 否 | 采用 dVAE 的视觉词典预测 |
| **I-JEPA** | 联合嵌入预测 | 否 | 是 (目标分支) | 是 (目标分支) | 目标表示分支与上下文预测分支 |

### Paradigm Comparison (三大自监督技术路线对比)

| 技术路线 | 核心机制 | 典型算法 | 优势 | 劣势 |
|-----------|-----------|----------|------|------|
| **对比学习 (Contrastive)** | 推开负样本，拉近正样本 | SimCLR, MoCo | 提取的表征具有极强的语义判别力 | 极度依赖大量负样本和精巧设计的图像增强组合 |
| **生成式 (Generative / MIM)** | 从不完整的输入中重建原图 | MAE, BEiT | 空间定位强，非常适合密集预测任务（检测、分割） | 在像素级重建时，容易把模型容量浪费在不重要的噪声纹理上 |
| **联合嵌入预测 (Joint-Embedding)** | 在特征空间对目标进行预测 | BYOL, DINO, JEPA | 避开了复杂的负样本构建与像素微观重建，语义理解抽象度高 | 算法设计较为脆弱，不小心易陷入表示坍缩的死胡同 |

---

## 12. Practice Problems (练习题与详解)

### Problem 1: InfoNCE Temperature (InfoNCE 温度系数消融)
请详细阐述在 InfoNCE 损失函数中温度超参数 $\tau$ 所起到的具体作用。若 $\tau \to 0$ 和 $\tau \to \infty$ 时，模型的更新行为会分别发生什么变化？

**Solution (解析):**
- 温度超参数 $\tau$ 起到了**调节对难样本关注度**的作用。它作为分母的尺度因子，可以缩放余弦相似度得分的分布范围。
- 当 $\tau \to 0$ 时，Softmax 函数输出的概率分布会变得极其陡峭，几乎所有的梯度都将由与 Anchor 相似度最高的那个**最难负样本 (Hardest Negative)** 产生。这会使得模型具有类似最大间隔 (Max-margin) 的分类倾向。但是，这也极易带来数值不稳定性，使得少数噪声样本支配整个梯度方向，导致训练崩溃。
- 当 $\tau \to \infty$ 时，所有的指数项全部趋近于 1，Softmax 输出演变为均匀分布，InfoNCE 损失趋近于一常数 $\log(K+1)$。此时，梯度几乎为 0，网络将彻底无法学到任何有用的表征。
- 实际应用中，通常设定 $\tau \in [0.07, 0.5]$ 之间的较小值，以平衡难样本挖掘的广度与优化的平稳度。

### Problem 2: Momentum Update (动量键编码器演进计算)
MoCo 采用的键编码器动量更新公式为 $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$，其中动量系数 $m=0.999$。若在当前训练步中，查询编码器的参数通过梯度更新发生了大小为 $\Delta\theta = 0.01$ 的更新，试计算此时键编码器的参数相应变化了多少？

**Solution (解析):**
- 代入动量更新计算公式：
  $$\theta_k^{\text{new}} = m \cdot \theta_k^{\text{old}} + (1 - m) \cdot \theta_q^{\text{new}}$$
  $$\theta_k^{\text{new}} = m \cdot \theta_k^{\text{old}} + (1 - m) \cdot (\theta_k^{\text{old}} + \Delta\theta)$$
  $$\theta_k^{\text{new}} = \theta_k^{\text{old}} + (1 - m) \cdot \Delta\theta$$
- 代入数据 $m=0.999, \Delta\theta = 0.01$：
  $$\theta_k^{\text{new}} - \theta_k^{\text{old}} = (1 - 0.999) \times 0.01 = 0.001 \times 0.01 = 1 \times 10^{-5}$$
- 键编码器的权重在此步骤中仅发生了极其微弱的 $10^{-5}$ 程度的变化。这完美验证了动量更新机制能够确保 Key 的特征提取表示极其稳定、极缓地演化。

### Problem 3: Collapse Detection (表示坍缩的定义与检测)
在自监督学习中，所谓的“表示坍缩 (Representation Collapse)”是指什么现象？在实际训练中，你可以通过什么定量的数学指标来检测它？

**Solution (解析):**
- **表示坍缩 (Collapse)**：是指在缺失负样本（或者模型设计缺陷）的情况下，特征编码器“不费吹灰之力”地为所有的输入图像输出完全相同或者高度相似的低维特征向量。由于此时所有图片的特征均一致，它们在特征空间中的余弦距离为 0，模型可以轻松使损失降到极低值，但它却没有学到任何对识别图片有用的语义。
- **定量检测指标**：可以在一个 Mini-batch 内部，定量计算所有样本特征两两之间的**平均对偶余弦相似度 (Pairwise Cosine Similarity)**：
  $$\text{Metric} = \frac{2}{B(B-1)} \sum_{i=1}^{B} \sum_{j > i}^{B} \frac{\mathbf{z}_i^\top \mathbf{z}_j}{\|\mathbf{z}_i\|\|\mathbf{z}_j\|}$$
  - 如果计算出的均值极度逼近 1.0，说明网络已发生了灾难性的表示坍缩。
  - 在正常且高效的特征提取状态下，该对偶相似度均值应该保持在极低的非坍缩水平（例如 0.0 ~ 0.2 之间），仅在输入确为正样本对时相似度才处于高位。

### Problem 4: MAE Mask Ratio (MAE 与 BERT 掩码比例差异分析)
为什么 MAE 算法在图像预训练时推荐采用极高比例（如 75%）的掩码，而经典的 BERT 在文本中却仅采用 15% 的掩码比例？

**Solution (解析):**
- 图像与自然语言在信息密度上存在着质的差异。图像包含着**高度的空间冗余性 (Spatial Redundancy)**，相邻像素、甚至是相邻的图像块之间存在极强的连续性与相关性，如果掩码比例较低，ViT 能够轻而易举地通过周围插值完成重建，根本不需要逼迫网络理解图片的深层物体概念与高级语义。
- 采用 75% 的超高掩码率：
  - 能够完全打破像素级的局部相关性，极大地加深了重构难度，强迫模型必须去进行高级别的语义归纳。
  - 使得编码器只需处理 25% 的 Patch，大幅降低了运算和存储开销，获得了超过 4 倍的并行加速。
- 相比之下，文本作为人类的高度抽象结晶，**信息密度极高**。每一个词（Token）都在句中扮演着独特的语法和语义作用。如果在文本中随机抹去 75% 的词汇，剩下的词语将彻底支离破碎，人类也完全不可能推测出原始语义，更不具备重建的上下文条件。因此 15% 的掩码比例在文本中是信息留存的合理上限。

### Problem 5: BYOL vs SimSiam (无负样本抗坍缩机制消融)
BYOL 与 SimSiam 均能在不需要负样本对的前提下，完美克服特征坍缩。请指出这两个算法分别依靠何种独门机制来对抗坍缩？

**Solution (解析):**
- **BYOL** 依靠的是**非对称的双路自蒸馏演进**与**时序缓慢演变约束**。它使用了动量目标网络（EMA 平滑更新提供 Target 向量）与在线网络末端额外添加的 Predictor（一侧有、另一侧无），从而在机制上阻止了两端网络“联合走向”常数解。
- **SimSiam** 则展现了更为极致的机制。它证明了不需要任何动量编码器，仅仅需要**单边梯度截断 (Stop-gradient)** 机制，即可完全防止坍缩。Stop-gradient 将自监督损失函数的优化实质转变成了一个交替投影优化的机制，在这种数学框架下，常数退化解并非其稳定驻点，从而使模型具有天然抗坍缩特性。
- 这两个算法的对比消融表明，Stop-gradient 是底层最核心、最具决定意义的抗坍缩基石。

---

> 🟢 来自资料 — 自监督学习已从手工 pretext task 发展到对比学习 (SimCLR, MoCo)，再到无负样本方法 (BYOL, SimSiam) 和生成式方法 (MAE)，代表了表示学习的范式演进。