# Chapter 6: Recurrent Neural Networks (RNNs)

> 🟢 来自资料 — 本章核心内容源自标准深度学习教材（Goodfellow et al.）及课程讲义。🟡 AI补充 — 数值示例与部分应用场景由AI辅助补全。

---

## 6.1 Sequential Data and Recurrence

### 6.1.1 什么是序列数据

序列数据的特点是数据点之间存在时间/顺序依赖：

- **时序数据**：股票价格、天气记录、传感器读数
- **自然语言**：文本、语音
- **视频**：连续帧

🟢 来自资料

### 6.1.2 为何需要循环

前馈网络假设输入之间独立，无法建模序列依赖。RNN 通过 **隐藏状态 (Hidden State)** 在时间步之间传递信息：

$$h_t = f(h_{t-1}, x_t; \theta)$$

每个时间步共享相同参数 $\theta$，使其能处理任意长度序列。

🟢 来自资料

---

## 6.2 Basic RNN

### 6.2.1 数学公式

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

$$y_t = W_{hy} h_t + b_y$$

其中：
- $h_t \in \mathbb{R}^d$：$t$ 时刻的隐藏状态
- $x_t \in \mathbb{R}^m$：$t$ 时刻的输入
- $W_{hh} \in \mathbb{R}^{d \times d}$：隐藏到隐藏的权重矩阵（循环连接）
- $W_{xh} \in \mathbb{R}^{d \times m}$：输入到隐藏的权重矩阵
- $W_{hy} \in \mathbb{R}^{k \times d}$：隐藏到输出的权重矩阵

**展开 (Unrolling)**：将 RNN 沿时间轴展开形成一个深度共享权重的前馈网络。

🟢 来自资料

### 6.2.2 RNN 架构模式

| 模式 | 输入序列 | 输出序列 | 典型应用 |
|------|----------|----------|----------|
| **One-to-One** | 1 | 1 | 标准分类 |
| **One-to-Many** | 1 | $T$ | 图像描述 (Image Captioning) |
| **Many-to-One** | $T$ | 1 | 情感分析 (Sentiment Analysis) |
| **Many-to-Many (同步)** | $T$ | $T$ | 视频帧标注、词性标注 |
| **Many-to-Many (异步)** | $T$ | $T'$ | 机器翻译 (Seq2Seq) |

🟢 来自资料

```
One-to-One:       x → [RNN] → y
One-to-Many:      x → [RNN] → y₁→y₂→...→y_T
Many-to-One:      x₁→x₂→...→x_T → [RNN] → y
Many-to-Many:     x₁→x₂→...→x_T → [RNN] → y₁→y₂→...→y_T
Many-to-Many*:    x₁→...→x_T → [Encoder] → [Decoder] → y₁→...→y_T'
```

---

## 6.3 Backpropagation Through Time (BPTT)

### 6.3.1 原理

BPTT 是将 RNN 在时间轴上展开后应用反向传播：

1. 展开 RNN 为 $T$ 步
2. 计算损失 $\mathcal{L} = \sum_{t=1}^T \mathcal{L}_t(y_t, \hat{y}_t)$
3. 沿时间反向传播梯度，在共享权重处累加

梯度计算：

$$\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial \mathcal{L}_t}{\partial W_{hh}}$$

其中 $\frac{\partial \mathcal{L}_t}{\partial W_{hh}}$ 需要回溯到该时间步之前的所有时间步。

🟢 来自资料

### 6.3.2 Truncated BPTT

为处理极长序列，限制反向传播的时间步数（如 $k=100$）：

- 前向：处理完整序列
- 反向：仅回溯 $k$ 步
- 平衡了计算效率和长程依赖

---

## 6.4 Vanishing and Exploding Gradients

### 6.4.1 问题分析

BPTT 中，从时间步 $T$ 到 $t$ 的梯度包含连乘积：

$$\frac{\partial \mathcal{L}_T}{\partial h_t} = \frac{\partial \mathcal{L}_T}{\partial h_T} \cdot \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}$$

对于基本 RNN，$\frac{\partial h_k}{\partial h_{k-1}} = \text{diag}(\tanh'(\cdot)) \cdot W_{hh}^\top$

当 $W_{hh}$ 的特征值 $|\lambda| < 1$，连乘 → 0（**梯度消失**）
当 $|\lambda| > 1$，连乘 → ∞（**梯度爆炸**）

🟢 来自资料

### 6.4.2 解决方案

| 问题 | 解决方案 |
|------|----------|
| **梯度爆炸** | **Gradient Clipping**：$g \leftarrow \min\left(1, \frac{\theta}{\|g\|}\right) \cdot g$ |
| **梯度消失** | LSTM / GRU（门控机制）；残差连接；更好的初始化 |

### 6.4.3 Gradient Clipping（梯度裁剪）

$$\tilde{g} = \begin{cases} g & \text{if } \|g\|_2 \leq \tau \\ \frac{\tau}{\|g\|_2} \cdot g & \text{if } \|g\|_2 > \tau \end{cases}$$

$\tau$ 为裁剪阈值（典型值 1.0–5.0）。当梯度范数超过阈值时，按比例缩放至阈值。

🟢 来自资料

---

## 6.5 LSTM (Long Short-Term Memory)

### 6.5.1 核心思想

通过 **细胞状态 (Cell State)** $c_t$ 作为信息高速公路，配合门控机制控制信息流动。

🟢 来自资料

### 6.5.2 LSTM 完整公式

**遗忘门 (Forget Gate)**：决定丢弃哪些旧信息

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**输入门 (Input Gate)**：决定写入哪些新信息

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**候选细胞状态 (Candidate Cell State)**：创建候选值

$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

**细胞状态更新 (Cell State Update)**：

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**输出门 (Output Gate)**：决定输出什么

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**隐藏状态 (Hidden State)**：

$$h_t = o_t \odot \tanh(c_t)$$

🟢 来自资料

### 6.5.3 门控直觉

| 门 | 直觉 | 极端情况 |
|----|------|----------|
| $f_t$ | 遗忘旧记忆 | $f_t=0$：完全丢弃旧状态；$f_t=1$：完整保留 |
| $i_t$ | 写入新记忆 | $i_t=0$：忽略新输入；$i_t=1$：全部接受 |
| $o_t$ | 暴露记忆 | $o_t=0$：隐藏状态为零；$o_t=1$：完全暴露 |

### 6.5.4 梯度流动

由于细胞状态的线性更新 $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$，梯度可以通过 $c_t$ 路径无衰减地回传（当 $f_t \approx 1$ 时），这就是 LSTM 缓解梯度消失的关键。

🟢 来自资料

---

## 6.6 GRU (Gated Recurrent Unit)

### 6.6.1 动机

LSTM 的简化版：合并细胞状态和隐藏状态，将三个门简化为两个门。

🟢 来自资料

### 6.6.2 GRU 公式

**更新门 (Update Gate)**：控制保留多少旧状态 vs 接收多少新状态

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

**重置门 (Reset Gate)**：控制忽略多少旧状态（计算候选状态时）

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

**候选隐藏状态**：

$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$

**隐藏状态更新**：

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

🟢 来自资料

### 6.6.3 LSTM vs GRU 对比

| 特性 | LSTM | GRU |
|------|------|-----|
| 门数量 | 3 (f, i, o) | 2 (r, z) |
| 状态数量 | 2 (c, h) | 1 (h) |
| 参数量 | 较多 | 较少 (约 LSTM 的 75%) |
| 遗忘+输入 | 独立控制 | 耦合控制 ($1 - z_t$) |
| 性能 | 通常稍好 | 接近 LSTM，更快 |

---

## 6.7 高级 RNN 架构

### 6.7.1 Bidirectional RNN (BiRNN)

$$
\begin{aligned}
\overrightarrow{h}_t &= \text{RNN}_{\text{forward}}(x_t, \overrightarrow{h}_{t-1}) \\
\overleftarrow{h}_t &= \text{RNN}_{\text{backward}}(x_t, \overleftarrow{h}_{t+1}) \\
h_t &= [\overrightarrow{h}_t; \overleftarrow{h}_t]
\end{aligned}
$$

- 每个时间步同时获得过去和未来信息
- 适用于离线处理（需要完整序列）：NER、机器翻译的编码器
- 不适用于在线/流式处理

🟢 来自资料

### 6.7.2 Stacked (Deep) RNN

将多层 RNN 堆叠：

$$h_t^{(l)} = \text{RNN}^{(l)}(h_t^{(l-1)}, h_{t-1}^{(l)})$$

- 每层在不同时间尺度上学习特征
- 通常 2–4 层
- LSTM/GRU 常用于深层 RNN

🟢 来自资料

---

## 6.8 Applications

### 6.8.1 Language Modeling（语言建模）

**目标**：建模 $P(w_t | w_1, ..., w_{t-1})$

- Many-to-One (上下文 → 下一个词) 或 Many-to-Many (输入序列 → 输出序列)
- 评估指标：Perplexity = $e^{-\frac{1}{N}\sum \log P(w_i|context)}$

### 6.8.2 Machine Translation（机器翻译）

Seq2Seq 架构：
- **Encoder**：编码源语言 → 上下文向量 $c = h_T$
- **Decoder**：根据 $c$ 自回归生成目标语言
- 瓶颈：固定长度的 $c$ 压缩了所有源信息

### 6.8.3 Image Captioning（图像描述）

```
CNN (Encoder) → 特征向量 → RNN (Decoder) → 文字描述
```

- CNN 提取图像特征作为 RNN 的初始输入
- RNN 逐词生成描述
- 典型模型：Show and Tell (2015)

🟢 来自资料

---

## 6.9 Seq2Seq with Attention

### 6.9.1 Attention 动机

传统 Seq2Seq 用固定长度向量 $c$ 编码整个输入，对长序列性能急剧下降。

Attention 机制允许 Decoder 在每个时间步 **动态关注** Encoder 的不同位置。

🟢 来自资料

### 6.9.2 Bahdanau Attention（加性注意力）

$$
\begin{aligned}
e_{t,i} &= v_a^\top \tanh(W_a s_{t-1} + U_a h_i) \quad &\text{(对齐得分)} \\
\alpha_{t,i} &= \frac{\exp(e_{t,i})}{\sum_{j=1}^T \exp(e_{t,j})} \quad &\text{(注意力权重)} \\
c_t &= \sum_{i=1}^T \alpha_{t,i} h_i \quad &\text{(上下文向量)}
\end{aligned}
$$

Decoder 更新：
$$s_t = \text{RNN}(s_{t-1}, [y_{t-1}; c_t])$$

🟢 来自资料

### 6.9.3 Luong Attention（乘性注意力）

$$e_{t,i} = s_t^\top W_a h_i$$

更简单的点积/双线性形式，计算效率更高。

---

## 6.10 Numerical Example: LSTM

### 示例设置

假设 $d = 2$（隐藏维度），$m = 2$（输入维度），简化权重：

$$W_f = [I_2, I_2], \quad W_i = [I_2, I_2], \quad W_c = [I_2, I_2], \quad W_o = [I_2, I_2]$$

$$b_f = b_i = b_c = b_o = 0$$

初始：$h_0 = [0, 0]^\top, \quad c_0 = [0, 0]^\top$

输入：$x_1 = [1, -1]^\top$

### Step 1: 计算门值

$$[h_0; x_1] = [0, 0, 1, -1]^\top$$

遗忘门：
$$f_1 = \sigma\left(I_2 \cdot [0, 0]^\top + I_2 \cdot [1, -1]^\top\right) = \sigma([1, -1]^\top) = [0.731, 0.269]^\top$$

输入门：
$$i_1 = \sigma([1, -1]^\top) = [0.731, 0.269]^\top$$

候选细胞状态：
$$\tilde{c}_1 = \tanh([1, -1]^\top) = [0.762, -0.762]^\top$$

### Step 2: 更新细胞状态

$$c_1 = [0.731, 0.269]^\top \odot [0, 0]^\top + [0.731, 0.269]^\top \odot [0.762, -0.762]^\top$$

$$c_1 = [0.557, -0.205]^\top$$

### Step 3: 计算输出

输出门：
$$o_1 = \sigma([1, -1]^\top) = [0.731, 0.269]^\top$$

隐藏状态：
$$h_1 = [0.731, 0.269]^\top \odot \tanh([0.557, -0.205]^\top)$$

$$h_1 = [0.731, 0.269]^\top \odot [0.506, -0.202]^\top = [0.370, -0.054]^\top$$

🟡 AI补充（数值示例）

---

## 6.11 Practice Problems

### Problem 1: Vanilla RNN Gradient

> 证明：对于 vanilla RNN，BPTT 中 $\frac{\partial \mathcal{L}_T}{\partial h_t}$ 何时会消失。

**Solution**：

$$\frac{\partial \mathcal{L}_T}{\partial h_t} = \frac{\partial \mathcal{L}_T}{\partial h_T} \cdot \prod_{k=t+1}^{T} \left(\text{diag}(\tanh'(\cdot)) \cdot W_{hh}^\top\right)$$

$\tanh' \in (0, 1]$（最大在 $x=0$ 时为 1）。当 $W_{hh}$ 的最大奇异值 < 1 时，积的范数指数级衰减至 0，即梯度消失。

---

### Problem 2: LSTM Forget Gate

> LSTM 中，若 $f_t$ 始终为 1，$i_t$ 始终为 0，推断细胞状态和隐藏状态的变化趋势。

**Solution**：
- $c_t = 1 \cdot c_{t-1} + 0 \cdot \tilde{c}_t = c_{t-1}$ — 细胞状态完全保留
- 即使输出门 $o_t$ 变化，$h_t = o_t \odot \tanh(c_{t-1})$ 也只变化输出门控制的 "曝光度"
- 梯度可无衰减地通过 $c_t$ 路径传播

---

### Problem 3: GRU Update Gate

> 解释 GRU 中 $z_t = 0$ 和 $z_t = 1$ 两种极端情况。

**Solution**：

- $z_t = 0$：$h_t = h_{t-1}$，隐藏状态完全保留旧状态（类似 LSTM $f_t=1, i_t=0$）
- $z_t = 1$：$h_t = \tilde{h}_t$，完全使用候选新状态
- 由于 $z_t$ 同时控制 "保留旧" 和 "接受新"（耦合），GRU 比 LSTM 参数更少

---

### Problem 4: BiRNN Limitation

> BiLSTM 是否适合实时语音识别？为什么？

**Solution**：不适合。BiLSTM 需要完整序列（包括未来帧）才能计算每个时间步的 $\overleftarrow{h}_t$。实时系统只能使用单向 LSTM 或引入受控延迟（如等待几帧后再处理）。

---

*Last updated: 2026-07-02*
