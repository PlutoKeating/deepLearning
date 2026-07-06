# Chapter 4: Neural Networks

> 🟢 来自资料 — 综合自 CS231n Lecture 4 & 5、Deep Learning Book (Goodfellow et al.) Chapter 6、相关原始论文

---

## 4.1 Biological Inspiration

### 4.1.1 生物神经元

人脑包含约 $10^{11}$（千亿）个神经元。每个神经元的基本结构：

```
            Dendrites (树突)
               ← ← ←
               ← ← ←
    ┌──────────────────┐
    │   Cell Body      │────→ Axon (轴突) ──→ Synapses (突触)
    │   (细胞体)        │
    └──────────────────┘
```

- **树突 (Dendrites)**：接收来自其他神经元的信号
- **细胞体 (Soma)**：整合输入信号，如果总输入超过阈值则"发放"（fire）
- **轴突 (Axon)**：将信号传递给下游神经元
- **突触 (Synapses)**：神经元之间的连接点，可调节强度（可塑性）

🟡 AI补充：人工神经元是对生物神经元的极大简化抽象。真实神经元涉及复杂的电化学过程、多样的神经递质、精确的时间编码，而不仅仅是加权求和。

### 4.1.2 从生物到人工

| 生物概念 | 人工神经网络对应 |
|---------|----------------|
| 神经元 | 计算单元（activation unit） |
| 突触强度 | 权重 $w$ |
| 树突信号接收 | 加权求和 $\sum w_i x_i$ |
| 细胞体整合 + 发放 | 激活函数 $f(\sum w_i x_i + b)$ |
| 学习/可塑性 | 通过梯度下降更新权重 |

---

## 4.2 Artificial Neuron

### 4.2.1 数学模型

单个人工神经元的计算可以表达为：

$$
\boxed{y = f\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(w^T x + b)}
$$

其中：
- $x \in \mathbb{R}^n$：输入向量
- $w \in \mathbb{R}^n$：权重向量
- $b \in \mathbb{R}$：偏置（bias）
- $f(\cdot)$：激活函数（非线性）
- $y \in \mathbb{R}$：输出

```
   x₁ ──→ [w₁] ──╮
   x₂ ──→ [w₂] ──┤
   ...           ├──→ Σ + b ──→ f(·) ──→ y
   xₙ ──→ [wₙ] ──╯
```

---

## 4.3 Activation Functions

🟢 来自资料：激活函数引入非线性，使网络能够逼近任意函数。没有激活函数，多层网络等价于单层线性模型。

### 4.3.1 Sigmoid

$$
\boxed{\sigma(x) = \frac{1}{1 + e^{-x}}}
$$

- 输出范围：$(0, 1)$
- 导数：$\sigma'(x) = \sigma(x)(1 - \sigma(x))$

**优点**：
- 输出有界，适合输出概率
- 平滑可导

**缺点**：
- **梯度消失 (Vanishing Gradient)**：当 $|x|$ 很大时，$\sigma'(x) \to 0$，深层网络中梯度无法回传
- **非零中心 (Not Zero-centered)**：输出始终为正 → 后续层输入的均值始终为正 → 梯度更新呈 zig-zag 模式
- 指数运算计算量大

### 4.3.2 Tanh (Hyperbolic Tangent)

$$
\boxed{\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1}
$$

- 输出范围：$(-1, 1)$
- 导数：$\tanh'(x) = 1 - \tanh^2(x)$

**相比 Sigmoid 的改进**：零中心（zero-centered），缓解了 zig-zag 问题。但仍存在梯度消失问题。

### 4.3.3 ReLU (Rectified Linear Unit)

$$
\boxed{\text{ReLU}(x) = \max(0, x)}
$$

- 输出范围：$[0, +\infty)$
- 导数：$\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$

🟢 来自资料：ReLU 是当前最常用的激活函数，主要由 AlexNet (Krizhevsky et al., 2012) 推广。

**优点**：
- 计算简单（只需 max 操作），比 Sigmoid/Tanh 快得多
- 在 $x > 0$ 区域梯度恒为 1，**无梯度消失问题**
- 稀疏激活（约 50% 神经元输出为零），有隐式正则化效果

**缺点**：
- **Dead ReLU**：如果 $x$ 始终 $\leq 0$，梯度为零，权重永不更新
- 非零中心
- 无界输出（可能导致数值问题）

### 4.3.4 Leaky ReLU

$$
\boxed{\text{LeakyReLU}(x) = \max(0.01x, x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}}
$$

其中 $\alpha$ 是一个小常数（如 0.01）。

- 解决了 dead ReLU 问题：负区域有小梯度
- PReLU (Parametric ReLU) 将 $\alpha$ 设为可学习参数

### 4.3.5 ELU (Exponential Linear Unit)

$$
\boxed{\text{ELU}(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}}
$$

- 在负区域平滑地趋向 $-\alpha$
- 输出均值接近零（比 Leaky ReLU 更好的零中心属性）
- 计算量稍大（需要指数运算）

### 4.3.6 GELU (Gaussian Error Linear Unit)

$$
\boxed{\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)}
$$

其中 $\Phi(x)$ 是标准正态分布的 CDF。

🟢 来自资料：GELU 是现代 Transformer (BERT, GPT, ViT) 的默认激活函数。它在 ReLU 的基础上引入了随机正则化的思想——以概率 $\Phi(x)$ 乘以 $x$ 的期望值。

### 4.3.7 激活函数对比总结

| 函数 | 公式 | 输出范围 | 梯度消失 | 零中心 | Dead 问题 | 使用场景 |
|------|------|---------|---------|--------|----------|---------|
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | (0, 1) | ✗ 严重 | ✗ | ✗ | 历史/二分类输出 |
| **Tanh** | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | (-1, 1) | ✗ 严重 | ✓ | ✗ | 历史/RNN |
| **ReLU** | $\max(0, x)$ | $[0, \infty)$ | ✓ | ✗ | ✗ 可能 | CNN 默认 |
| **Leaky ReLU** | $\max(0.01x, x)$ | $(-\infty, \infty)$ | ✓ | ✗ | ✓ | ReLU 的改进 |
| **ELU** | $[\text{见上}]$ | $(-\alpha, \infty)$ | ✓ | ≈✓ | ✓ | ReLU 的平滑替代 |
| **GELU** | $x\Phi(x)$ | ≈$[-0.17, \infty)$ | ✓ | ≈✓ | ✓ | Transformer |

---

## 4.4 Multi-Layer Perceptron (MLP)

### 4.4.1 架构

MLP 由输入层、一个或多个隐藏层、输出层堆叠而成，每层与下一层全连接：

```
   Input      Hidden 1     Hidden 2     Output
   Layer      Layer        Layer        Layer

    x₁ ───→  h₁⁽¹⁾ ───→  h₁⁽²⁾ ───→  ŷ₁
    x₂ ───→  h₂⁽¹⁾ ───→  h₂⁽²⁾ ───→  ŷ₂
    x₃ ───→  h₃⁽¹⁾ ───→  h₃⁽²⁾
    ...       ...           ...
    xₙ ───→  hₘ⁽¹⁾ ───→  hₖ⁽²⁾
              W⁽¹⁾          W⁽²⁾         W⁽³⁾
```

### 4.4.2 数学表示

一个 $L$ 层 MLP 的前向传播：

$$
\begin{aligned}
z^{(1)} &= W^{(1)} x + b^{(1)}, & a^{(1)} &= f(z^{(1)}) \\
z^{(2)} &= W^{(2)} a^{(1)} + b^{(2)}, & a^{(2)} &= f(z^{(2)}) \\
&\vdots \\
z^{(L)} &= W^{(L)} a^{(L-1)} + b^{(L)}, & \hat{y} &= g(z^{(L)})
\end{aligned}
$$

其中：
- $z^{(\ell)}$：第 $\ell$ 层的 pre-activation（线性组合）
- $a^{(\ell)}$：第 $\ell$ 层的 post-activation（激活后输出），$a^{(0)} = x$
- $W^{(\ell)}, b^{(\ell)}$：第 $\ell$ 层的权重和偏置
- $f$：隐藏层激活函数（如 ReLU）
- $g$：输出层激活函数（如 Softmax 用于分类）

---

## 4.5 Forward Propagation

前向传播即从输入到输出逐层计算的过程。以 2 层网络为例：

**给定**：
- 输入 $x \in \mathbb{R}^D$
- 隐藏层权重 $W^{(1)} \in \mathbb{R}^{H \times D}$，偏置 $b^{(1)} \in \mathbb{R}^H$
- 输出层权重 $W^{(2)} \in \mathbb{R}^{K \times H}$，偏置 $b^{(2)} \in \mathbb{R}^K$

**前向传播**：

$$
\boxed{
\begin{aligned}
z^{(1)} &= W^{(1)} x + b^{(1)} \\
a^{(1)} &= \text{ReLU}(z^{(1)}) \\
z^{(2)} &= W^{(2)} a^{(1)} + b^{(2)} \\
\hat{y} &= \text{Softmax}(z^{(2)})
\end{aligned}
}
$$

---

## 4.6 Backpropagation

🟢 来自资料：反向传播 (Backpropagation) 是训练神经网络的核心算法，它利用链式法则高效计算所有参数的梯度。

### 4.6.1 链式法则 (Chain Rule)

对于复合函数 $z = f(g(x))$：

$$
\frac{dz}{dx} = \frac{dz}{dg} \cdot \frac{dg}{dx}
$$

对于多变量情况：

$$
\frac{\partial L}{\partial x_i} = \sum_j \frac{\partial L}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}
$$

### 4.6.2 计算图 (Computational Graph)

将神经网络表示为有向无环图，节点表示操作，边表示数据流：

```
   ╭───╮     ╭───╮     ╭───╮     ╭───╮
x ─→│ × │─→z¹─→│ + │─→a¹─→│ReLU│─→h¹─→ ...
   ╰─┬─╯     ╰─┬─╯     ╰────╯
    W¹        b¹
```

反向传播沿计算图反向遍历，利用局部梯度和上游梯度计算每个节点的梯度：

$$
\frac{\partial L}{\partial \text{node}} = \frac{\partial L}{\partial \text{output}} \cdot \frac{\partial \text{output}}{\partial \text{node}}
$$

### 4.6.3 反向传播详细推导（2 层网络）

设网络结构为：Input → Linear → ReLU → Linear → Softmax → Loss

**已知**：
- 损失函数 $L = -\log(\text{softmax}(z^{(2)})_{y})$
- 上游梯度（从 Loss 到 $z^{(2)}$ 的梯度）：$\frac{\partial L}{\partial z^{(2)}} = \hat{y} - y_{\text{one-hot}}$

**Step 1: 输出层梯度**

$$
\frac{\partial L}{\partial W^{(2)}} = \frac{\partial L}{\partial z^{(2)}} \cdot \frac{\partial z^{(2)}}{\partial W^{(2)}} = (\hat{y} - y_{\text{one-hot}}) \cdot (a^{(1)})^T
$$

$$
\frac{\partial L}{\partial b^{(2)}} = \frac{\partial L}{\partial z^{(2)}} = \hat{y} - y_{\text{one-hot}}
$$

**Step 2: 向隐藏层回传**

$$
\frac{\partial L}{\partial a^{(1)}} = (W^{(2)})^T \cdot \frac{\partial L}{\partial z^{(2)}}
$$

**Step 3: 通过 ReLU**

$$
\frac{\partial L}{\partial z^{(1)}} = \frac{\partial L}{\partial a^{(1)}} \odot \mathbb{1}[z^{(1)} > 0]
$$

**Step 4: 隐藏层梯度**

$$
\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial z^{(1)}} \cdot x^T
$$

$$
\frac{\partial L}{\partial b^{(1)}} = \frac{\partial L}{\partial z^{(1)}}
$$

### 4.6.4 关键求和规则

当梯度从多个路径汇聚到同一节点时，需要求和：

$$
\text{grad}(x) = \sum_{\text{所有从 }x\text{ 出发的路径}} \text{沿该路径的梯度积}
$$

---

## 4.7 Forward + Backward: Worked Example

🟢 来自资料：对一个具体的 2 层网络进行完整的前向和反向传播计算。

### 4.7.1 网络设置

```
Input:  x = [0.5, -0.3]^T   (D=2)
Hidden: W¹ = [[0.2, 0.8],    H=3
              [-0.5, 0.3],
              [0.7, -0.1]]
        b¹ = [0.1, -0.2, 0.3]^T
        f = ReLU

Output: W² = [[0.4, -0.3, 0.5],   K=2 (二分类)
              [-0.1, 0.6, 0.2]]
        b² = [0.0, 0.1]^T
        g = Softmax

True label: y = 0 (class 0)
```

### 4.7.2 Forward Pass

**Hidden layer pre-activation**:

$$
z^{(1)} = W^{(1)} x + b^{(1)} = \begin{bmatrix}
0.2 \cdot 0.5 + 0.8 \cdot (-0.3) + 0.1 \\
-0.5 \cdot 0.5 + 0.3 \cdot (-0.3) + (-0.2) \\
0.7 \cdot 0.5 + (-0.1) \cdot (-0.3) + 0.3
\end{bmatrix} = \begin{bmatrix}
0.1 - 0.24 + 0.1 \\
-0.25 - 0.09 - 0.2 \\
0.35 + 0.03 + 0.3
\end{bmatrix} = \begin{bmatrix}
-0.04 \\
-0.54 \\
0.68
\end{bmatrix}
$$

**Hidden layer post-activation (ReLU)**:

$$
a^{(1)} = \text{ReLU}(z^{(1)}) = [0, 0, 0.68]^T
$$

**Output scores**:

$$
z^{(2)} = W^{(2)} a^{(1)} + b^{(2)} = \begin{bmatrix}
0.4 \cdot 0 + (-0.3) \cdot 0 + 0.5 \cdot 0.68 + 0.0 \\
-0.1 \cdot 0 + 0.6 \cdot 0 + 0.2 \cdot 0.68 + 0.1
\end{bmatrix} = \begin{bmatrix}
0.34 \\
0.236
\end{bmatrix}
$$

**Softmax probabilities**:

$$
\begin{aligned}
e^{z^{(2)}} &= [e^{0.34}, e^{0.236}] = [1.405, 1.266] \\
\sum e^z &= 1.405 + 1.266 = 2.671 \\
\hat{y} &= [1.405/2.671, 1.266/2.671] = [0.526, 0.474]
\end{aligned}
$$

**Cross-entropy loss**:

$$
L = -\log(\hat{y}_0) = -\log(0.526) = 0.643
$$

### 4.7.3 Backward Pass

**Output layer gradient** (Softmax + CrossEntropy):

$$
\frac{\partial L}{\partial z^{(2)}} = \hat{y} - y_{\text{one-hot}} = [0.526, 0.474] - [1, 0] = [-0.474, 0.474]
$$

**Gradient for W²**:

$$
\frac{\partial L}{\partial W^{(2)}} = \frac{\partial L}{\partial z^{(2)}} \cdot (a^{(1)})^T = \begin{bmatrix}
-0.474 \cdot 0 & -0.474 \cdot 0 & -0.474 \cdot 0.68 \\
0.474 \cdot 0 & 0.474 \cdot 0 & 0.474 \cdot 0.68
\end{bmatrix} = \begin{bmatrix}
0 & 0 & -0.322 \\
0 & 0 & 0.322
\end{bmatrix}
$$

**Gradient for b²**: $\frac{\partial L}{\partial b^{(2)}} = [-0.474, 0.474]^T$

**Gradient back to a¹**:

$$
\frac{\partial L}{\partial a^{(1)}} = (W^{(2)})^T \cdot \frac{\partial L}{\partial z^{(2)}} = \begin{bmatrix} 0.4 & -0.1 \\ -0.3 & 0.6 \\ 0.5 & 0.2 \end{bmatrix} \begin{bmatrix} -0.474 \\ 0.474 \end{bmatrix} = \begin{bmatrix} -0.237 \\ 0.427 \\ -0.142 \end{bmatrix}
$$

**Gradient through ReLU**:

$$
\frac{\partial L}{\partial z^{(1)}} = \frac{\partial L}{\partial a^{(1)}} \odot \mathbb{1}[z^{(1)} > 0] = \begin{bmatrix} -0.237 \\ 0.427 \\ -0.142 \end{bmatrix} \odot \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ -0.142 \end{bmatrix}
$$

只有第三个神经元（$z^{(1)}_3 = 0.68 > 0$）的梯度通过。

**Gradient for W¹**:

$$
\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial z^{(1)}} \cdot x^T = \begin{bmatrix} 0 \\ 0 \\ -0.142 \end{bmatrix} \begin{bmatrix} 0.5 & -0.3 \end{bmatrix} = \begin{bmatrix} 0 & 0 \\ 0 & 0 \\ -0.071 & 0.043 \end{bmatrix}
$$

**Gradient for b¹**: $\frac{\partial L}{\partial b^{(1)}} = [0, 0, -0.142]^T$

### 4.7.4 结果分析

| 参数 | 梯度 |
|------|------|
| $W^{(1)}$ | 只有第三行非零（前两个神经元被 ReLU killed） |
| $b^{(1)}$ | 只有第三个偏置有非零梯度 |
| $W^{(2)}$ | 只有第三列非零（因为只有第三个隐藏神经元活跃） |
| $b^{(2)}$ | 两个都有梯度 |

🟢 来自资料：这个例子展示了 **Dead ReLU** 的潜在问题：前两个隐藏神经元的 $z^{(1)}$ 为负，ReLU 输出为 0，导致对应的权重梯度为 0，这些权重在本次更新中完全不更新。

---

## 4.8 Universal Approximation Theorem

### 4.8.1 定理陈述

**Universal Approximation Theorem**：具有至少一个隐藏层、足够多神经元的 MLP（使用任意"挤压"型激活函数如 Sigmoid）可以以任意精度逼近 $\mathbb{R}^n$ 的紧致子集上的任意连续函数。

### 4.8.2 关键含义

| 观点 | 含义 |
|------|------|
| **存在性 (Existence)** | 理论上 shallow network 足够 |
| **表达能力 (Expressiveness)** | 深度网络可以用指数级更少的神经元表达相同函数 |
| **学习 (Learning)** | 存在 ≠ 可学。优化和泛化是关键挑战 |

### 4.8.3 Deep vs Shallow

深层网络的优势：

1. **层次化特征表示**：从简单到复杂递进学习特征（边缘 → 纹理 → 部件 → 物体）
2. **参数效率**：深层网络可以指数级减少所需神经元数量
3. **经验证据**：在几乎所有任务上，深度网络显著优于浅层网络

🟢 来自资料：虽然理论上一层隐藏层足够，但实践中深网络更容易训练且性能更好。深度带来了**表示学习的层次化 (hierarchical representation learning)**。

---

## 4.9 Training Neural Networks

### 4.9.1 Loss Curves

```
Loss
 │
 │  ┌─────────────────  Train Loss（持续下降）
 │  │
 │  │        ╱‾‾‾‾‾‾‾‾‾  Val Loss（先降后升 → 过拟合）
 │  │       ╱
 │  │      ╱
 │  │     ╱
 │  │    ╱
 │  └───┘
 └──────────────────────────→ Epochs
```

### 4.9.2 诊断表

| 现象 | 含义 | 对策 |
|------|------|------|
| Train Loss 不下降 | 欠拟合 或 学习率太小/太大 | 调整学习率、检查梯度 |
| Train Loss ↓, Val Loss ↑ | **过拟合** | 更多数据、正则化、Dropout |
| Train Loss ↓, Val Loss ↓ | 良好拟合 | 继续训练 |
| Train Acc ≈ Val Acc，两者都很低 | 欠拟合 (high bias) | 更大模型、更多训练 |
| Train Acc >> Val Acc | 过拟合 (high variance) | 正则化、数据增强 |
| Train Loss NaN/Inf | 梯度爆炸 | 减小学习率、梯度裁剪 |

### 4.9.3 训练管线检查清单

🟡 AI补充：在开始训练之前，确保每个环节正确：

1. **梯度检查 (Gradient Check)**：用数值梯度验证解析梯度（如相对误差 $< 10^{-7}$）
2. **小数据集过拟合测试**：关掉正则化，网络应在小数据上达到 100% 准确率
3. **监控指标**：在训练期间实时观察 loss 曲线、梯度范数、激活分布
4. **调参顺序**：学习率 → 模型大小 → 正则化强度 → 其他超参数

---

## Practice Problems

### Problem 1: Activation Function Gradients

Compute the derivative at $x = -2$ for:
a) Sigmoid
b) Tanh
c) ReLU
d) Leaky ReLU ($\alpha = 0.01$)

**Solution:**
a) $\sigma(-2) = \frac{1}{1+e^2} \approx 0.119$, $\sigma'(-2) = 0.119 \cdot (1 - 0.119) \approx 0.105$
b) $\tanh(-2) \approx -0.964$, $\tanh'(-2) = 1 - (-0.964)^2 = 1 - 0.929 = 0.071$
c) $\text{ReLU}'(-2) = 0$ (since $-2 < 0$)
d) $\text{LeakyReLU}'(-2) = 0.01$

### Problem 2: Multi-Layer Gradient Flow

A 3-layer network: Input $x=2$, weights $w_1=3, w_2=0.5, w_3=4$, all with ReLU activation. Compute $\frac{\partial y}{\partial w_1}$.

**Solution:**

$$
\begin{aligned}
z_1 &= 3 \cdot 2 = 6, \quad a_1 = \text{ReLU}(6) = 6 \\
z_2 &= 0.5 \cdot 6 = 3, \quad a_2 = \text{ReLU}(3) = 3 \\
y &= 4 \cdot 3 = 12
\end{aligned}
$$

Backward:
$$
\begin{aligned}
\frac{\partial y}{\partial w_3} &= 3 \\
\frac{\partial y}{\partial w_2} &= 4 \cdot 6 = 24 \quad (\text{ReLU}'(3)=1) \\
\frac{\partial y}{\partial w_1} &= 4 \cdot \text{ReLU}'(3) \cdot 0.5 \cdot \text{ReLU}'(6) \cdot 2 = 4 \cdot 1 \cdot 0.5 \cdot 1 \cdot 2 = 4
\end{aligned}
$$

等价于链式法则：$\frac{\partial y}{\partial w_1} = w_3 \cdot \text{ReLU}'(a_2) \cdot w_2 \cdot \text{ReLU}'(a_1) \cdot x$

### Problem 3: Forward Pass with a Small MLP

Consider a 2-class MLP with 2 inputs, 2 hidden neurons (ReLU), 2 outputs (Softmax).

$$
W^{(1)} = \begin{bmatrix} 1 & -1 \\ 2 & 0 \end{bmatrix}, b^{(1)} = \begin{bmatrix} 0 \\ -1 \end{bmatrix}, W^{(2)} = \begin{bmatrix} 1 & -2 \\ -1 & 1 \end{bmatrix}, b^{(2)} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

For $x = [1, 2]^T$:
a) Compute $z^{(1)}, a^{(1)}, z^{(2)}, \hat{y}$
b) If true label $y=0$, compute cross-entropy loss

**Solution:**

a)
$$
z^{(1)} = \begin{bmatrix} 1 & -1 \\ 2 & 0 \end{bmatrix}\begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 0 \\ -1 \end{bmatrix} = \begin{bmatrix} -1 \\ 2 \end{bmatrix} + \begin{bmatrix} 0 \\ -1 \end{bmatrix} = \begin{bmatrix} -1 \\ 1 \end{bmatrix}
$$

$$
a^{(1)} = \text{ReLU}\left(\begin{bmatrix} -1 \\ 1 \end{bmatrix}\right) = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

$$
z^{(2)} = \begin{bmatrix} 1 & -2 \\ -1 & 1 \end{bmatrix}\begin{bmatrix} 0 \\ 1 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \end{bmatrix} = \begin{bmatrix} -2 \\ 1 \end{bmatrix}
$$

$$
e^{z^{(2)}} = [e^{-2}, e^1] = [0.135, 2.718], \quad \sum = 2.853
$$

$$
\hat{y} = \left[\frac{0.135}{2.853}, \frac{2.718}{2.853}\right] = [0.047, 0.953]
$$

b)
$$
L = -\log(0.047) = 3.058
$$

非常高的损失——模型严重错误地预测了类别 1 (0.953 vs 0.047)。

### Problem 4: Sigmoid Saturation

Show that $\sigma'(x) \leq 0.25$ for all $x$, and find where the maximum occurs.

**Solution:**

$$
\sigma'(x) = \sigma(x)(1-\sigma(x))
$$

令 $t = \sigma(x) \in (0,1)$，则 $\sigma'(x) = t(1-t)$。该二次函数在 $t = 0.5$ 时取最大值 $0.5 \times 0.5 = 0.25$。

当 $\sigma(x) = 0.5$ 时，$x = 0$。因此最大梯度在 $x=0$ 处，值为 0.25。

这意味着经过多层 Sigmoid 后，梯度被反复乘以 $\leq 0.25$ 的因子 → 指数衰减 → **梯度消失**。

---

*Last updated: 2026-07-02*
