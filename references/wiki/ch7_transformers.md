# Chapter 7: Transformers

> 🟢 来自资料 — 本章核心内容源自 "Attention Is All You Need" (Vaswani et al., 2017) 以及后续 ViT/DeiT/Swin 论文。🟡 AI补充 — 数值示例与对比分析由AI辅助补全。

---

## 7.1 Limitations of RNNs

| 局限 | 说明 |
|------|------|
| **顺序处理 (Sequential Processing)** | $h_t$ 依赖 $h_{t-1}$，无法并行化 |
| **长程依赖 (Long-range Dependencies)** | 即使 LSTM/GRU，$t$ 和 $t+k$ 之间的信号路径长 $O(k)$ |
| **梯度问题** | BPTT 仍然受梯度消失/爆炸影响 |
| **Memory Bottleneck** | Seq2Seq 中固定长度的上下文向量成为信息瓶颈 |

🟢 来自资料

Transformer 的核心突破：**完全抛弃循环，纯基于注意力机制**。

---

## 7.2 Self-Attention（自注意力）

### 7.2.1 Query, Key, Value 形式

对于输入序列 $X \in \mathbb{R}^{n \times d_{model}}$（$n$ 个 token，每个维度 $d_{model}$）：

$$
\begin{aligned}
Q &= X W^Q \quad &(n \times d_k) \\
K &= X W^K \quad &(n \times d_k) \\
V &= X W^V \quad &(n \times d_v)
\end{aligned}
$$

其中 $W^Q, W^K \in \mathbb{R}^{d_{model} \times d_k}$，$W^V \in \mathbb{R}^{d_{model} \times d_v}$

**直觉**：
- **Query**：我在找什么？（当前 token 的诉求）
- **Key**：我有什么？（其他 token 的标签）
- **Value**：我贡献什么？（其他 token 的实际信息）

🟢 来自资料

### 7.2.2 Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

逐步解析（单头）：

1. **相似度矩阵**：$S = QK^\top \in \mathbb{R}^{n \times n}$，$S_{ij} = q_i^\top k_j$
2. **缩放**：$\frac{S}{\sqrt{d_k}}$，防止点积过大导致 softmax 梯度消失
3. **归一化**：$A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)$，每行是一个概率分布
4. **加权聚合**：$\text{Output} = A V$

🟢 来自资料

### 7.2.3 为什么除以 $\sqrt{d_k}$

当 $d_k$ 较大时，$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的方差为 $d_k$（假设独立标准正态），softmax 输入过大 → 梯度极端小。除以 $\sqrt{d_k}$ 将方差控制为 1。

🟢 来自资料

### 7.2.4 Self-Attention vs Cross-Attention

| 类型 | Q 来源 | K, V 来源 | 用途 |
|------|--------|-----------|------|
| **Self-Attention** | 同一序列 | 同一序列 | Encoder, Decoder 自层 |
| **Cross-Attention** | Decoder | Encoder | Decoder 关注 Encoder 输出 |
| **Masked Self-Attention** | Decoder（因果掩码） | Decoder | 自回归生成 |

---

## 7.3 Multi-Head Attention（多头注意力）

### 7.3.1 动机

单个注意力头可能只关注一种关系模式（如主语-谓语）。多个头可以并行关注不同子空间的信息。

### 7.3.2 公式

$$
\begin{aligned}
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O \\
\text{head}_i &= \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
\end{aligned}
$$

其中 $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$，$W_i^K \in \mathbb{R}^{d_{model} \times d_k}$，$W_i^V \in \mathbb{R}^{d_{model} \times d_v}$，$W^O \in \mathbb{R}^{h d_v \times d_{model}}$

通常取 $d_k = d_v = d_{model} / h$。

🟢 来自资料

### 7.3.3 计算复杂度

$$
\begin{aligned}
\text{Self-Attention} &: O(n^2 \cdot d_{model}) \quad (\text{序列长度的平方！}) \\
\text{RNN} &: O(n \cdot d^2)
\end{aligned}
$$

当 $n$ 很大时（长序列），Self-Attention 的内存和计算负担显著。

🟢 来自资料

---

## 7.4 Positional Encoding（位置编码）

Attention 本身是**置换不变的 (Permutation Invariant)**：打乱输入顺序，输出只是同样打乱。需要注入位置信息。

🟢 来自资料

### 7.4.1 Sinusoidal Positional Encoding

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{model}}}\right)$$

- $pos$：位置索引 (0, 1, 2, ...)
- $i$：维度索引 (0, 1, ..., $d_{model}/2 - 1$)

**关键性质**：
- $PE_{pos+k}$ 可表示为 $PE_{pos}$ 的线性函数 → 模型可学习相对位置
- 无需训练参数，可外推到训练时未见过的序列长度
- 添加到输入：$X_{\text{input}} = X_{embedding} + PE$

### 7.4.2 Learned Positional Embeddings

直接为每个位置学习一个可训练嵌入向量：

$$X_{\text{input}} = X_{embedding} + P_{pos}$$

- 简单直接
- 无法外推到更长序列
- 多数现代模型（GPT, BERT）使用此方法

🟢 来自资料

---

## 7.5 Transformer Block

### 7.5.1 标准 Transformer Block（Post-LN）

```
Input → Multi-Head Attention → (+) → LayerNorm → FFN → (+) → LayerNorm → Output
```

即：

$$
\begin{aligned}
x' &= \text{LayerNorm}(x + \text{MultiHeadAttention}(x)) \\
x'' &= \text{LayerNorm}(x' + \text{FFN}(x'))
\end{aligned}
$$

🟢 来自资料

### 7.5.2 Pre-LN vs Post-LN

| 变体 | 公式 | 特点 |
|------|------|------|
| **Post-LN** | $x_{l+1} = \text{LN}(x_l + F(x_l))$ | 原始 Transformer，训练可能不稳定 |
| **Pre-LN** | $x_{l+1} = x_l + F(\text{LN}(x_l))$ | 训练更稳定，收敛更快，现代模型更常用 |

🟢 来自资料

### 7.5.3 Feed-Forward Network (FFN)

$$\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2$$

或使用 GELU / SwiGLU 等现代激活函数：

$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

- 两层全连接，中间维度通常 $4 \times d_{model}$
- 每个 token 独立处理（位置间无交互）
- 提供模型的非线性变换能力

🟢 来自资料

### 7.5.4 Layer Normalization

$$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta$$

其中 $\mu, \sigma$ 沿特征维度（每个 token 独立）计算：

$$\mu = \frac{1}{d}\sum_{i=1}^d x_i, \quad \sigma = \sqrt{\frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2}$$

- LN 在每个 token 内部归一化，与 batch 大小无关
- 适合变长序列处理

🟢 来自资料

---

## 7.6 Numerical Example: Self-Attention

### 设定

$d_{model} = 4, d_k = 2, h = 2$（双头）。序列长度 $n = 3$，token 嵌入：

$$X = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix}$$

### 单头演示（Head 1）

$$W^Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}, \quad W^K = \begin{bmatrix} 0 & 0 \\ 0 & 0 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad W^V = I_{4 \times 2}$$

计算 Q, K, V：
$$Q = X W^Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad K = X W^K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}, \quad V = X W^V = X_{[:,:2]} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$$

注意力得分 $S = \frac{QK^\top}{\sqrt{2}}$：
$$QK^\top = \begin{bmatrix} 1 & 0 & 0 \\ 1 & 1 & 0 \\ 1 & 1 & 0 \end{bmatrix}, \quad S = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 0 & 0 \\ 1 & 1 & 0 \\ 1 & 1 & 0 \end{bmatrix}$$

Softmax（逐行）：
$$A = \begin{bmatrix} 0.58 & 0.29 & 0.13 \\ 0.42 & 0.42 & 0.16 \\ 0.42 & 0.42 & 0.16 \end{bmatrix}$$

输出：
$$\text{Output} = A V = \begin{bmatrix} 0.71 & 0.42 \\ 0.58 & 0.58 \\ 0.58 & 0.58 \end{bmatrix}$$

**解释**：Token 1 最关注自己（0.58），Token 2/3 同时关注 Token 1 和 2。

🟡 AI补充（数值示例）

---

## 7.7 Vision Transformer (ViT, 2021)

### 7.7.1 核心思想

> "An Image is Worth 16×16 Words"

直接将图像分割为固定大小的 patches，像 NLP token 一样处理。

🟢 来自资料

### 7.7.2 ViT 架构

```
Image (H×W×3)
  → 分割为 N 个 P×P patches → 展平为 P²×3 维向量
  → Linear Projection → d_model 维 patch embedding
  → 添加 [class] token + Position Embedding
  → Transformer Encoder × L
  → [class] token → MLP Head → 类别
```

**关键细节**：

| 组件 | 说明 |
|------|------|
| Patch Embedding | 每个 $P \times P$ patch 展平后线性投影到 $d_{model}$ |
| [class] token | 可学习的分类 token，类似 BERT [CLS] |
| Position Embedding | 可学习的位置嵌入（ViT 使用 1D 位置嵌入） |
| 无 Decoder | 纯 Encoder 架构（分类任务） |

🟢 来自资料

### 7.7.3 ViT 的归纳偏置 (Inductive Bias)

与 CNN 相比，ViT 几乎没有内置的视觉归纳偏置：

| 归纳偏置 | CNN | ViT |
|----------|-----|-----|
| 局部性 (Locality) | ✅ 卷积核只关注局部 | ❌ 全局自注意力 |
| 平移等变性 | ✅ 卷积固有 | ❌ 只有位置嵌入提供模糊的位置感 |
| 层次结构 | ✅ 池化逐步下采样 | ❌ 全层相同分辨率（标准 ViT） |

**结果**：ViT 需要更多数据才能匹敌 CNN 性能，但数据充足时表现更优。

🟢 来自资料

---

## 7.8 ViT Variants

### 7.8.1 DeiT (Data-efficient Image Transformers, 2021)

**解决 ViT 数据饥渴问题**：

| 技术 | 说明 |
|------|------|
| **Knowledge Distillation** | 用 CNN 教师模型引导 ViT 训练 |
| **Distillation Token** | 额外的可学习 token，接收教师信号 |
| **强数据增强** | RandAugment, Mixup, CutMix |

🟢 来自资料

### 7.8.2 Swin Transformer (2021)

**引入 CNN 的层次化 + 局部性**：

| 创新 | 说明 |
|------|------|
| **Shifted Windows** | 自注意力限制在局部窗口内，交替使用规则窗口和偏移窗口 |
| **Hierarchical Design** | 4 个阶段，每阶段 Patch Merging 下采样 2×，通道翻倍 |
| **线性复杂度** | 窗口内自注意力 → $O(n)$ 而非 $O(n^2)$ |

**Swin Transformer Block**（连续两个）：

```
W-MSA → LN → MLP → LN  (Window Multi-head Self-Attention)
SW-MSA → LN → MLP → LN (Shifted Window MSA)
```

🟢 来自资料

---

## 7.9 CNNs vs Transformers for Vision

| 维度 | CNN | Vision Transformer |
|------|-----|--------------------|
| **归纳偏置** | 强（局部性、平移等变性） | 弱（全靠数据学习） |
| **数据效率** | 高（中小数据集表现好） | 低（需要大量数据或蒸馏） |
| **感受野** | 局部 → 逐层扩大 | 全局（从第一层起） |
| **计算复杂度** | $O(HW \cdot K^2 \cdot C^2)$ | $O(N^2 \cdot d)$ ($N$ patches) |
| **多尺度特征** | 天然层次化 | 需要特殊设计（Swin, PVT） |
| **可解释性** | 成熟（Grad-CAM等） | 发展中（Attention Rollout） |
| **预训练范式** | ImageNet 监督预训练 | 大规模预训练 + 微调 |

🟢 来自资料 / 🟡 AI补充

---

## 7.10 Practice Problems

### Problem 1: Scaled Dot-Product Attention

> 给定 $Q = K = V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$（$d_k=2$），计算 Self-Attention 输出。

**Solution**：

$$QK^\top = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}$$

$$\frac{QK^\top}{\sqrt{2}} = \begin{bmatrix} 0.707 & 0 & 0.707 \\ 0 & 0.707 & 0.707 \\ 0.707 & 0.707 & 1.414 \end{bmatrix}$$

Softmax 逐行（保留两位）：
$$A = \begin{bmatrix} 0.42 & 0.21 & 0.42 \\ 0.21 & 0.42 & 0.42 \\ 0.23 & 0.23 & 0.54 \end{bmatrix}$$

$$\text{Output} = A V = \begin{bmatrix} 0.42 & 0.21 & 0.42 \\ 0.21 & 0.42 & 0.42 \\ 0.23 & 0.23 & 0.54 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} 0.84 & 0.63 \\ 0.63 & 0.84 \\ 0.77 & 0.77 \end{bmatrix}$$

---

### Problem 2: Multi-Head Design

> Transformer 中 $d_{model}=512, h=8$。每个 head 的 $d_k$ 是多少？为什么要取 $d_k = d_{model}/h$？

**Solution**：$d_k = d_v = 512/8 = 64$。

设计理由：
1. 总计算量与单头大致相同（$h \times d_k \times d_k = d_{model} \times d_k = d_{model}^2 / h$）
2. 保证每个 head 有不同的低维子空间
3. 实现简单，$W^Q \in \mathbb{R}^{d_{model} \times d_{model}}$ 重塑为 $d_{model} \times h \times d_k$

---

### Problem 3: ViT Position Embedding

> ViT 为什么需要位置嵌入？如果不用位置嵌入会发生什么？

**Solution**：

Self-Attention 是置换不变的：$f(PX) = f(X)$ 对任意置换矩阵 $P$ 成立。不加位置嵌入时，模型无法区分 "猫在狗左边" 和 "狗在猫左边"。

实际上，不加位置嵌入的 ViT 退化为 **Bag-of-Words 视觉模型**，仍有约 70% ImageNet 准确率（因为图像内容的空间分布本身带有统计模式），但远低于加位置嵌入的版本。

---

### Problem 4: Swin Complexity

> 比较标准 ViT 和 Swin Transformer 的 Self-Attention 计算复杂度。图像 $224 \times 224$，patch 大小 $16 \times 16$，Swin 窗口大小 $7 \times 7$。

**Solution**：

**ViT**：$N = (224/16)^2 = 196$ patches，复杂度 $O(N^2 d) = O(196^2 \cdot d) \approx O(38,416 \cdot d)$

**Swin**：每个窗口 $7 \times 7 = 49$ tokens，窗口数 $32 \times 32 / 49 \approx 21$，
复杂度 $O(\text{windows} \times \text{window size}^2 \cdot d) = O(21 \times 49^2 \cdot d) = O(50,421 \cdot d)$

在 stage 1 两者接近。但 stage 3/4 经过 Patch Merging 后 $N$ 更小，ViT 复杂度更高。

**关键差异**：Swin 是 $O(N)$（线性于 patch 数），原始 ViT 是 $O(N^2)$。

---

*Last updated: 2026-07-02*
