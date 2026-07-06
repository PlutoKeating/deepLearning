# Chapter 3: Regularization and Optimization

> 🟢 来自资料 — 综合自 CS231n Lecture 3 & 7、Deep Learning Book (Goodfellow et al.)、Adam paper (Kingma & Ba 2014)、BatchNorm paper (Ioffe & Szegedy 2015)

---

## 3.1 Optimization Landscape

### 3.1.1 损失函数的几何性质

深度神经网络的损失面（loss landscape）具有以下特征：

| 特征 | 描述 |
|------|------|
| **非凸性 (Non-convexity)** | 存在大量局部极小值和鞍点 |
| **高维性 (High-dimensionality)** | 参数空间维度可高达 $10^6 \sim 10^9$ |
| **鞍点主导 (Saddle-dominated)** | 高维空间中鞍点比局部极小值更常见 |

#### 局部极小值 vs 鞍点

- **局部极小值 (Local Minimum)**：所有方向的梯度为零，且所有方向的曲率为正
- **鞍点 (Saddle Point)**：梯度为零，但某些方向曲率为正、某些为负

🟡 AI补充：在高维优化中，真正的"坏"局部极小值实际上罕见；大多数"卡住"的情况发生在鞍点附近，这推动了动量法等技术的发展。

### 3.1.2 梯度噪声

SGD 的随机梯度是真实梯度的噪声估计：

$$
g_t = \nabla L_{batch}(W_t) = \frac{1}{B} \sum_{i \in \text{batch}} \nabla L_i(W_t)
$$

噪声方差 $\propto \frac{1}{B}$。大 batch 减少噪声但计算量大；小 batch 噪声大但有隐式正则化效果。

---

## 3.2 SGD with Momentum

### 3.2.1 标准 SGD

$$
W_{t+1} = W_t - \alpha \nabla L(W_t)
$$

缺点：在峡谷地形（某些方向陡峭、某些方向平缓）中会震荡，收敛缓慢。

### 3.2.2 SGD + Momentum

Momentum 通过累积历史梯度来加速收敛于一致方向并抑制震荡：

$$
\boxed{v_{t+1} = \mu v_t - \alpha \nabla L(W_t)}
$$

$$
\boxed{W_{t+1} = W_t + v_{t+1}}
$$

其中：
- $v_t$：速度（velocity），累积的历史梯度
- $\mu \in [0, 1)$：动量系数，通常 $\mu = 0.9$ 或 $0.99$
- $\alpha$：学习率

**物理直觉**：想象一个球滚下山坡。动量使球保持前进方向，不会被小坑陷住；球在陡坡加速，在平缓处减速。

### 3.2.3 Nesterov Accelerated Gradient (NAG)

Nesterov 动量在"前瞻"位置计算梯度：

$$
\boxed{v_{t+1} = \mu v_t - \alpha \nabla L(W_t + \mu v_t)}
$$

$$
\boxed{W_{t+1} = W_t + v_{t+1}}
$$

与标准动量的区别：NAG 先沿动量方向移动一步，在"前瞻"位置评估梯度。这可以视为一种"先看再跳"的策略：

```
Standard Momentum:          Nesterov:
  当前位置 → 速度方向跳        当前位置 → 前瞻位置（速度方向）
  → 在终点计算梯度            → 在前瞻位置计算梯度
                             → 用前瞻梯度修正方向
```

🟢 来自资料：NAG 通常比标准动量收敛更快，特别是在凸优化的理论保证方面更强。在深度学习中，NAG 和标准动量实际表现接近。

---

## 3.3 Adaptive Learning Rate Methods

### 3.3.1 AdaGrad

AdaGrad 为每个参数独立调整学习率，适合稀疏特征：

$$
\boxed{G_t = G_{t-1} + (\nabla L(W_t))^2}
$$

$$
\boxed{W_{t+1} = W_t - \frac{\alpha}{\sqrt{G_t + \epsilon}} \odot \nabla L(W_t)}
$$

其中 $G_t$ 累积历史梯度的平方和（逐元素），$\epsilon = 10^{-8}$ 防止除零。

| 优点 | 缺点 |
|------|------|
| 适合稀疏数据（罕见特征获得更大更新） | $G_t$ 单调递增 → 学习率单调递减 → 可能过早停止 |
| 无需手动调整学习率衰减 | 不适合非凸优化（深度学习） |

### 3.3.2 RMSProp

RMSProp 通过指数移动平均解决了 AdaGrad 学习率单调递减的问题：

$$
\boxed{G_t = \beta G_{t-1} + (1 - \beta)(\nabla L(W_t))^2}
$$

$$
\boxed{W_{t+1} = W_t - \frac{\alpha}{\sqrt{G_t + \epsilon}} \odot \nabla L(W_t)}
$$

其中 $\beta$ 是衰减率，通常 $\beta = 0.9$ 或 $0.99$。

🟡 AI补充：RMSProp 的"移动平均"意味着旧的梯度平方信息逐渐被遗忘，学习率不会单调下降，适合深度网络的非凸优化。

### 3.3.3 Adam (Adaptive Moment Estimation)

🟢 来自资料：Adam（Kingma & Ba, 2014）结合了动量和自适应学习率，是当前最常用的优化器：

**Adam 算法**：

$$
\begin{aligned}
& m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(W_t) \quad &\text{(一阶矩，动量)} \\
& v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L(W_t))^2 \quad &\text{(二阶矩，自适应率)} \\
& \hat{m}_t = \frac{m_t}{1 - \beta_1^t} \quad &\text{(偏差修正)} \\
& \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad &\text{(偏差修正)} \\
& W_{t+1} = W_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

**默认超参数**：
- $\alpha = 0.001$（学习率）
- $\beta_1 = 0.9$（一阶矩衰减）
- $\beta_2 = 0.999$（二阶矩衰减）
- $\epsilon = 10^{-8}$

**偏差修正 (Bias Correction)**：由于 $m_0 = 0$ 和 $v_0 = 0$，初始估计偏向零。除以 $1 - \beta^t$ 对初始几步进行修正，后期该因子接近 1。

### 3.3.4 优化器对比

| 优化器 | 动量 | 自适应学习率 | 偏差修正 | 适用场景 |
|--------|------|-------------|---------|---------|
| SGD | ✗ | ✗ | ✗ | 简单任务，需精心调参 |
| SGD+Momentum | ✓ | ✗ | ✗ | CV 任务常用基线 |
| NAG | ✓ (Nesterov) | ✗ | ✗ | 凸优化理论保证好 |
| AdaGrad | ✗ | ✓ | ✗ | 稀疏数据 |
| RMSProp | ✗ | ✓ | ✗ | RNN、非平稳目标 |
| **Adam** | ✓ | ✓ | ✓ | **默认首选，通用性强** |
| AdamW | ✓ | ✓ | ✓ | Adam + decoupled weight decay |

🟢 来自资料：实践中，Adam 和 SGD+Momentum 是最常用的两种。Adam 对超参数更鲁棒；SGD+Momentum 经过精心调参后可以取得更好的泛化性能。

---

## 3.4 Learning Rate Schedules

### 3.4.1 为什么需要学习率调度？

- 训练初期：大学习率快速探索
- 训练后期：小学习率精细收敛
- 最佳策略随任务和模型而异

### 3.4.2 常见调度策略

#### Step Decay

每隔 $T$ 个 epoch 将学习率乘以 $\gamma$：

$$
\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t / T \rfloor}
$$

典型设置：$T = 30$ epochs，$\gamma = 0.1$

#### Cosine Annealing

$$
\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)
$$

🟢 来自资料：Cosine annealing 自 ResNet 以来被广泛使用，它在训练结束时平滑地将学习率降至极小值。

#### Linear Warmup

在训练初期线性增大学习率，避免大学习率破坏初始权重：

$$
\alpha_t = \alpha_{\text{target}} \cdot \frac{t}{T_{\text{warmup}}}
$$

通常在 Transformer 训练中使用（$T_{\text{warmup}} \approx 4000$ steps）。

#### Warmup + Cosine Decay（组合策略）

```
α
│     ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
│    ╱                    ╲
│   ╱                      ╲  ← Cosine
│  ╱                        ╲
│ ╱                          ╲___
│╱                                
└────────────────────────────────→ t
  warmup
```

这是现代 ViT、GPT 等模型的标准做法。

---

## 3.5 Weight Initialization

🟢 来自资料：权重的初始值对训练的收敛速度和最终性能有重大影响。

### 3.5.1 糟糕的初始化

- **全零初始化**：所有神经元学习相同的特征（对称性问题），梯度完全相同 → 无法打破对称性
- **过大的随机值**：激活值进入饱和区 → 梯度消失
- **过小的随机值**：深层网络的信号逐渐消失

### 3.5.2 Xavier/Glorot Initialization

$$
\boxed{W \sim \mathcal{U}\left(-\frac{\sqrt{6}}{\sqrt{n_{\text{in}} + n_{\text{out}}}}, \frac{\sqrt{6}}{\sqrt{n_{\text{in}} + n_{\text{out}}}}\right)}
$$

或正态分布版本：

$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)
$$

**推导动机**：保持前向传播和反向传播中激活值和梯度的方差不变。适用于 Sigmoid/Tanh 激活。

### 3.5.3 He/Kaiming Initialization

$$
\boxed{W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)}
$$

**推导动机**：考虑 ReLU 激活函数将一半的神经元输出置零（方差减半），因此需要 $\times 2$ 补偿。

### 3.5.4 初始化对比

| 方法 | 方差 | 适用激活函数 | 来源 |
|------|------|-------------|------|
| Xavier Uniform | $\frac{6}{n_{\text{in}}+n_{\text{out}}}$ | Sigmoid, Tanh | Glorot & Bengio 2010 |
| He Normal | $\frac{2}{n_{\text{in}}}$ | **ReLU, Leaky ReLU** | He et al. 2015 |

---

## 3.6 Batch Normalization

🟢 来自资料：Batch Normalization (Ioffe & Szegedy, 2015) 是深度学习中最实用的技术之一。

### 3.6.1 动机：Internal Covariate Shift

深度网络中，每层输入分布随前层参数更新而改变。这迫使后续层不断适应新分布，降低了训练效率。BatchNorm 通过标准化每层的输入来缓解此问题。

### 3.6.2 算法

对于 mini-batch $\mathcal{B} = \{x_1, \ldots, x_m\}$：

**Step 1: 计算 batch 统计量**

$$
\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i
$$

$$
\sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2
$$

**Step 2: 标准化**

$$
\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}
$$

**Step 3: 缩放和平移（可学习参数）**

$$
\boxed{y_i = \gamma \hat{x}_i + \beta \equiv \text{BN}_{\gamma, \beta}(x_i)}
$$

其中 $\gamma$（scale）和 $\beta$（shift）是可学习参数，恢复网络的表达能力。

### 3.6.3 训练 vs 推理

| 阶段 | $\mu$ | $\sigma^2$ |
|------|-------|------------|
| **训练** | 使用当前 mini-batch 的统计量 | 使用当前 mini-batch 的统计量 |
| **推理** | 使用训练期间累积的 running average | 使用训练期间累积的 running average |

推理时使用全局统计量（指数移动平均）：

$$
\mu_{\text{running}} \leftarrow \rho \cdot \mu_{\text{running}} + (1-\rho) \cdot \mu_{\mathcal{B}}
$$

### 3.6.4 BatchNorm 的优势

| 优点 | 机制 |
|------|------|
| 加速训练收敛 | 允许使用更大学习率 |
| 减少对初始化的敏感度 | 归一化减轻了初始值的影响 |
| 有正则化效果 | mini-batch 统计量的噪声类似 dropout |
| 缓解梯度消失/爆炸 | 保持激活值在合理范围 |

### 3.6.5 Layer Normalization

**Layer Normalization (LayerNorm)** 沿特征维度而非 batch 维度标准化：

$$
\mu_l = \frac{1}{H} \sum_{i=1}^{H} x_i, \quad \sigma_l^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu_l)^2
$$

| 特性 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 标准化维度 | Batch 维度 | Feature 维度 |
| 依赖 batch size | 是（小 batch 时统计量不稳定） | 否 |
| 适用场景 | CNN | **Transformer, RNN** |
| 训练/推理一致性 | 不一致（需 running stats） | 一致 |

🟢 来自资料：LayerNorm 是 Transformer 架构的标准选择，因为序列模型中 batch 维度可能很小且长度可变。

---

## 3.7 Dropout

🟢 来自资料：Dropout (Srivastava et al., 2014) 是一种强大的正则化技术。

### 3.7.1 动机

通过随机"丢弃"神经元来防止神经元之间的共适应（co-adaptation），强制网络学习冗余表示。

### 3.7.2 训练阶段

每个神经元以概率 $p$（keep probability）被保留，以概率 $1-p$ 被丢弃：

$$
\tilde{h} = h \odot m, \quad m_j \sim \text{Bernoulli}(p)
$$

### 3.7.3 推理阶段

使用全部神经元，但需要缩放以补偿训练时的丢弃：

**标准 Dropout**（原始版本）：

$$
y_{\text{test}} = p \cdot f(x)
$$

因为训练时只有 $p$ 比例的神经元活跃，期望输出被缩小了 $p$ 倍。

**Inverted Dropout**（更常用）：

训练时除以 $p$ 进行缩放：

$$
\tilde{h}_{\text{train}} = \frac{1}{p} \cdot (h \odot m)
$$

推理时不做任何修改：

$$
y_{\text{test}} = f(x)
$$

🟢 来自资料：Inverted Dropout 是 PyTorch/ TensorFlow 等框架的默认实现，因为推理时不需要额外操作。

### 3.7.4 Dropout 的工作原理

多个解释：
1. **集成学习 (Ensemble)**：每次 dropout 采样一个子网络，推理时隐式集成了 $2^n$ 个子网络
2. **打破共适应**：防止神经元依赖特定其他神经元的存在

---

## 3.8 Data Augmentation

🟢 来自资料：数据增强是一种隐式正则化，通过从现有数据生成新的训练样本来扩大训练集。

### 3.8.1 常用增强方法

#### 几何变换

| 方法 | 描述 |
|------|------|
| **Random Flip** | 水平/垂直翻转（概率 0.5） |
| **Random Crop** | 随机裁剪 + 缩放回原始尺寸 |
| **Random Rotation** | 随机旋转 ±$N$ 度 |
| **Random Affine** | 仿射变换（平移+缩放+旋转+剪切） |

#### 颜色变换

| 方法 | 描述 |
|------|------|
| **Color Jitter** | 随机调整亮度、对比度、饱和度、色调 |
| **PCA Color Augmentation** | 沿 PCA 主成分方向加噪声（AlexNet 使用） |

#### 高级增强

| 方法 | 描述 | 论文 |
|------|------|------|
| **Cutout** | 随机遮挡图像中一个方形区域 | DeVries & Taylor 2017 |
| **Mixup** | 两张图像及其标签的线性插值：$\tilde{x} = \lambda x_i + (1-\lambda) x_j$, $\tilde{y} = \lambda y_i + (1-\lambda) y_j$ | Zhang et al. 2018 |
| **CutMix** | 将一张图像的 patch 粘贴到另一张图像上，标签也按面积比例混合 | Yun et al. 2019 |
| **RandAugment** | 自动搜索增强策略组合 | Cubuk et al. 2020 |

### 3.8.2 Mixup 详解

$$
\begin{aligned}
\tilde{x} &= \lambda x_i + (1-\lambda) x_j \\
\tilde{y} &= \lambda y_i + (1-\lambda) y_j
\end{aligned}
$$

其中 $\lambda \sim \text{Beta}(\alpha, \alpha)$，通常 $\alpha = 0.2$ 或 $0.4$。

🟡 AI补充：Mixup 通过生成"中间样本"来鼓励模型在训练样本之间线性插值，从而提高泛化能力和对抗鲁棒性。

---

## 3.9 Early Stopping

训练过程中监控验证集损失：当验证损失不再下降（或开始上升）时停止训练。

```
Loss
 │  Train Loss ─────（持续下降）
 │  Val Loss    ╲
 │               ╲___（开始上升 → 过拟合）
 │                  ↑
 │             Early Stop Point
 └────────────────────────────────→ Epochs
```

🟢 来自资料：Early stopping 是一种简单高效的正则化策略，它通过限制优化步数来隐式约束参数范数，等价于某种 L2 正则化。

---

## Practice Problems

### Problem 1: Adam Update Step

Given current parameters $W = [1.0, -2.0]$，gradient $\nabla L = [0.3, -0.1]$，and Adam state $m_{t-1} = [0.1, 0.05]$, $v_{t-1} = [0.01, 0.02]$, $t=1$. Compute one step of Adam with $\alpha=0.001$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$.

**Solution:**

$$
\begin{aligned}
m_t &= 0.9 \cdot [0.1, 0.05] + 0.1 \cdot [0.3, -0.1] = [0.09+0.03, 0.045-0.01] = [0.12, 0.035] \\
v_t &= 0.999 \cdot [0.01, 0.02] + 0.001 \cdot [0.09, 0.01] = [0.00999+0.00009, 0.01998+0.00001] = [0.01008, 0.01999] \\
\hat{m}_t &= \frac{[0.12, 0.035]}{1-0.9^1} = \frac{[0.12, 0.035]}{0.1} = [1.2, 0.35] \\
\hat{v}_t &= \frac{[0.01008, 0.01999]}{1-0.999^1} = \frac{[0.01008, 0.01999]}{0.001} = [10.08, 19.99] \\
W_{t+1} &= [1.0, -2.0] - 0.001 \cdot \frac{[1.2, 0.35]}{\sqrt{[10.08, 19.99]}} \\
&= [1.0, -2.0] - 0.001 \cdot \left[\frac{1.2}{3.175}, \frac{0.35}{4.471}\right] \\
&= [1.0, -2.0] - [0.00038, 0.000078] \\
&= [0.99962, -2.00008]
\end{aligned}
$$

### Problem 2: Xavier vs He Initialization

For a layer with $n_{\text{in}} = 256$ and $n_{\text{out}} = 512$:
a) Compute the variance for Xavier normal initialization
b) Compute the variance for He normal initialization

**Solution:**
a) Xavier: $\sigma^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}} = \frac{2}{256+512} = \frac{2}{768} \approx 0.0026$
b) He: $\sigma^2 = \frac{2}{n_{\text{in}}} = \frac{2}{256} = 0.0078$

He initialization gives larger variance (~3×), compensating for ReLU's halving effect.

### Problem 3: BatchNorm Training vs Inference

A BatchNorm layer has learned $\gamma = 2$, $\beta = -1$, $\mu_{\text{running}} = 0.5$, $\sigma^2_{\text{running}} = 4$. At inference time, what is the output for input $x = 3.0$? (Assume $\epsilon = 0$)

**Solution:**

$$
\begin{aligned}
\hat{x} &= \frac{3.0 - 0.5}{\sqrt{4}} = \frac{2.5}{2} = 1.25 \\
y &= 2 \cdot 1.25 + (-1) = 2.5 - 1 = 1.5
\end{aligned}
$$

### Problem 4: Dropout Expected Value

At training time with inverted dropout ($p = 0.8$, keep probability), the output of a neuron is $h = 5.0$. If the dropout mask yields 0 (dropped), what is $\tilde{h}_{train}$? What is $\mathbb{E}[\tilde{h}_{train}]$?

**Solution:**

If dropped ($m = 0$): $\tilde{h}_{train} = \frac{1}{0.8} \cdot (5.0 \times 0) = 0$

Expected value:
$$
\mathbb{E}[\tilde{h}_{train}] = 0.8 \cdot \frac{5.0}{0.8} + 0.2 \cdot 0 = 5.0
$$

即 training 时的期望输出等于 $h$，这正是 inverted dropout 的设计目标。

---

*Last updated: 2026-07-02*
