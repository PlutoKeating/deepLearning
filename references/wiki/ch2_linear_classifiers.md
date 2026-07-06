# Chapter 2: Linear Classifiers

> 🟢 来自资料 — 综合自 CS231n 课程讲义 (Fei-Fei Li, Andrej Karpathy)、机器学习经典教材

---

## 2.1 Image Classification Task

### 2.1.1 问题定义

给定训练集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$，其中：
- $x_i \in \mathbb{R}^D$ 是输入图像（将 $H \times W \times C$ 展平为向量）
- $y_i \in \{1, 2, \ldots, K\}$ 是类别标签

目标是学习一个函数 $f: \mathbb{R}^D \to \{1, \ldots, K\}$ 能够对未见过的图像进行正确分类。

### 2.1.2 核心挑战

| 挑战 | 描述 |
|------|------|
| **语义鸿沟 (Semantic Gap)** | 像素值与高层语义概念之间的差距巨大 |
| **视角变化 (Viewpoint Variation)** | 同一物体从不同角度看起来完全不同 |
| **光照变化 (Illumination)** | 像素值随光照剧烈变化 |
| **类内差异 (Intra-class Variation)** | 同一类别内形态差异大（如不同品种的猫） |
| **遮挡 (Occlusion)** | 物体部分被遮挡 |
| **背景杂乱 (Background Clutter)** | 物体与背景难以区分 |
| **变形 (Deformation)** | 非刚性物体的形状变化 |

---

## 2.2 Nearest Neighbor Classifier

### 2.2.1 k-NN 原理

K-最近邻（k-Nearest Neighbor）是一种非参数分类器，核心思想极为简单：

**训练**：直接存储所有训练样本（无显式训练过程）

**预测**：对于测试样本 $x_{test}$：
1. 计算 $x_{test}$ 与所有训练样本的距离
2. 选择距离最近的 $k$ 个训练样本
3. 通过投票（分类）或平均（回归）得到预测结果

### 2.2.2 距离度量

#### L1 距离（Manhattan Distance）

$$
d_1(I_1, I_2) = \sum_{p} |I_1^p - I_2^p|
$$

其中 $p$ 遍历所有像素位置。L1 距离对坐标轴的选择敏感——如果旋转坐标系，距离值会改变。

#### L2 距离（Euclidean Distance）

$$
d_2(I_1, I_2) = \sqrt{\sum_{p} (I_1^p - I_2^p)^2}
$$

L2 距离对坐标轴旋转不变，但对离群点更敏感（平方放大误差）。

### 2.2.3 k-NN 的优缺点

| 优点 | 缺点 |
|------|------|
| 实现简单，无需训练 | 测试时计算量巨大 $\mathcal{O}(N)$ |
| 非参数，无假设 | 维度灾难 (Curse of Dimensionality) |
| 可解释性强 | 需要存储全部训练数据 |
| 对 $k$ 值的选择可调节平滑度 | 对像素级距离不敏感（两幅偏移一个像素的图像被认为很不同） |

🟢 来自资料：k-NN 在图像分类中很少使用，因为像素级距离不能很好地衡量语义相似性。但它是一个重要的 **baseline**。

### 2.2.4 超参数选择

- **k 的选择**：使用交叉验证 (cross-validation)。$k$ 越大，决策边界越平滑
- **距离度量**：L1 vs L2 取决于数据特性
- **关键原则**：永远不要用测试集调参！将数据划分为 **train / validation / test**

```
┌─────────────────────────────────────────────┐
│  原始数据                                    │
│  ├── Train (用于训练)                        │
│  ├── Validation (用于调超参数和模型选择)       │
│  └── Test (只在最后使用一次)                  │
└─────────────────────────────────────────────┘
```

---

## 2.3 Linear Classifier

### 2.3.1 Score Function

线性分类器的得分函数（score function）是最简单的参数化模型：

$$
\boxed{f(x_i, W, b) = Wx_i + b}
$$

其中：
- $x_i \in \mathbb{R}^D$：将图像展平为一维向量（如 CIFAR-10 中 $D = 32 \times 32 \times 3 = 3072$）
- $W \in \mathbb{R}^{K \times D}$：权重矩阵（也称参数）
- $b \in \mathbb{R}^K$：偏置向量
- 输出：$K$ 维得分向量，每维对应一个类别

### 2.3.2 矩阵形式

将训练集组织为矩阵 $X \in \mathbb{R}^{N \times D}$，则：

$$
S = XW^T + \mathbf{1}b^T
$$

其中 $S \in \mathbb{R}^{N \times K}$，$S_{ij}$ 是第 $i$ 个样本在第 $j$ 类的得分。

🟡 AI补充：偏置常被吸收进 $W$ 中，通过在 $x$ 末尾添加常数维度：$x' = [x; 1]$，$W' = [W, b]$，从而 $f(x) = W'x'$。

---

## 2.4 Interpreting Linear Classifiers

### 2.4.1 Template Matching 视角

将 $W$ 的每一行视为一个类别的"模板"（template）：

$$
\text{score}_k = w_k^T x + b_k
$$

这相当于将图像 $x$ 与第 $k$ 类的模板 $w_k$ 做内积（相似度匹配）。训练好的 $W$ 可视化后，每个 $w_k$ 看起来像一个模糊的类平均图像。

🟢 来自资料：这是线性分类器的一个重要直觉：它学习每个类别的一个模板，分类时计算图像与每个模板的匹配程度。

### 2.4.2 几何解释

在高维空间 $\mathbb{R}^D$ 中，每个类别的决策边界是一个 **超平面 (hyperplane)**：

$$
w_k^T x + b_k = w_j^T x + b_j
$$

这定义了一个 $(D-1)$ 维的超平面，将类别 $k$ 和 $j$ 的决策区域分开。

```
                    决策边界 (Hyperplane)
                         │
   Class A 区域          │    Class B 区域
       ○ ○ ○            │       △ △ △
         ○   ○          │      △     △
       ○     ○          │        △ △
                         │
              w^T x + b = 0
```

### 2.4.3 线性分类器的局限性

线性分类器无法处理线性不可分的问题。经典反例：

- **XOR 问题**：无法用一条直线分开
- **同心圆分布**：需要非线性边界
- **多模态分布**：同一类别分散在多个区域

🟡 AI补充：这正是引入神经网络（非线性激活函数 + 多层结构）的根本动机。

---

## 2.5 Loss Functions

损失函数量化了预测得分与真实标签之间的"不满意程度"。我们的目标是找到使损失最小化的参数 $W$。

### 2.5.1 Hinge Loss（多类 SVM 损失）

多类 SVM 损失（也称为 hinge loss）定义为：

$$
\boxed{L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)}
$$

其中：
- $s_j = f(x_i, W)_j$ 是第 $j$ 类的得分
- $s_{y_i}$ 是正确类别的得分
- $\Delta$ 是 margin（通常设为 1）
- $\max(0, \cdot)$ 称为 hinge 函数

**直觉**：希望正确类别的得分比所有其他类别高出至少 $\Delta$。

**完整损失**（含正则化）：

$$
L = \frac{1}{N} \sum_i L_i + \lambda R(W)
$$

#### SVM Loss 计算示例

🟢 来自资料：考虑三分类问题，某样本的得分为 $s = [13, -7, 11]$，正确类别 $y_i = 0$：

| 类别 j | $s_j$ | $s_{y_i}$ | $s_j - s_{y_i} + 1$ | $\max(0, \cdot)$ |
|--------|-------|-----------|----------------------|-------------------|
| 0 (正确) | 13 | — | — | — |
| 1 | -7 | 13 | $-7 - 13 + 1 = -19$ | 0 |
| 2 | 11 | 13 | $11 - 13 + 1 = -1$ | 0 |

$$
L_i = \max(0, -19) + \max(0, -1) = 0 + 0 = 0
$$

该样本分类完全正确，损失为 0。

#### 另一个示例

得分 $s = [10, 10, 10]$，正确类别 $y_i = 0$：

| 类别 j | $s_j - s_{y_i} + 1$ | Loss |
|--------|----------------------|------|
| 1 | $10 - 10 + 1 = 1$ | $\max(0, 1) = 1$ |
| 2 | $10 - 10 + 1 = 1$ | $\max(0, 1) = 1$ |

$$
L_i = 1 + 1 = 2
$$

即使得分相同，损失非零——鼓励正确类别得分独占鳌头。

#### Squared Hinge Loss

$$
L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)^2
$$

平方使惩罚非线性的增长，对违规更大的情况给予更多惩罚。

### 2.5.2 Cross-Entropy Loss（Softmax 分类器）

Softmax 将得分转换为概率分布，然后计算交叉熵损失。

#### Step 1: Softmax 函数

$$
\boxed{P(Y = k \mid X = x_i) = \frac{e^{s_k}}{\sum_{j=1}^{K} e^{s_j}}}
$$

其中 $s_k = f(x_i, W)_k$ 是得分（也称 logits）。

#### Step 2: Cross-Entropy Loss

$$
\boxed{L_i = -\log P(Y = y_i \mid X = x_i) = -\log\left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}}\right)}
$$

等价形式：

$$
L_i = -s_{y_i} + \log\sum_j e^{s_j}
$$

#### Softmax Loss 计算示例

🟢 来自资料：得分 $s = [3.2, 5.1, -1.7]$，正确类别 $y_i = 1$：

$$
\begin{aligned}
e^{s} &= [e^{3.2}, e^{5.1}, e^{-1.7}] = [24.53, 164.02, 0.18] \\
\sum e^{s} &= 24.53 + 164.02 + 0.18 = 188.73 \\
P(Y = 1) &= \frac{164.02}{188.73} = 0.869 \\
L_i &= -\log(0.869) = 0.140
\end{aligned}
$$

如果正确类别的概率接近 1，损失接近 0；接近 0，损失趋近 $\infty$。

#### SVM vs Softmax 对比

| 特性 | SVM (Hinge Loss) | Softmax (Cross-Entropy) |
|------|------------------|------------------------|
| 输出解释 | 得分（无概率意义） | 概率分布 |
| 满足条件后 | 一旦 margin 满足，损失为 0，不再优化 | 持续推动正确概率 → 1 |
| 对分数敏感度 | 只关心相对大小 | 关心具体概率值 |
| 可解释性 | 低 | 高（输出概率） |

🟢 来自资料：如果一个样本已被正确分类且 margin 足够大，SVM 不会再更新参数；但 Softmax 会继续推动 $P \to 1$。

---

## 2.6 Regularization

### 2.6.1 动机

正则化解决过拟合问题。完整的损失函数为：

$$
L(W) = \underbrace{\frac{1}{N} \sum_i L_i(f(x_i, W), y_i)}_{\text{数据损失}} + \underbrace{\lambda R(W)}_{\text{正则化项}}
$$

### 2.6.2 L2 Regularization (Weight Decay / Ridge)

$$
R(W) = \sum_k \sum_l W_{k,l}^2 = \|W\|_F^2
$$

- 惩罚大权重，鼓励权重分散到更多维度
- 解析解可导，梯度：$\nabla_W R(W) = 2W$
- 概率解释：对权重施加高斯先验 $W \sim \mathcal{N}(0, \frac{1}{2\lambda})$

### 2.6.3 L1 Regularization (Lasso)

$$
R(W) = \sum_k \sum_l |W_{k,l}| = \|W\|_1
$$

- 诱导稀疏性 (sparsity)：很多权重精确为零
- 概率解释：对权重施加拉普拉斯先验
- 可用于特征选择

### 2.6.4 Elastic Net

$$
R(W) = \alpha \|W\|_1 + (1-\alpha) \|W\|_F^2
$$

结合 L1 和 L2 的优点。

### 2.6.5 L1 vs L2 对比

| 特性 | L1 | L2 |
|------|----|----|
| 解的性质 | 稀疏解 | 稠密解 |
| 对异常值 | 更鲁棒 | 较敏感 |
| 梯度 | 常数（除零点） | 线性 |
| 优化难度 | 困难（不可导于0） | 简单（处处可导） |

---

## 2.7 Gradient Descent

### 2.7.1 基本思想

通过沿负梯度方向迭代更新来最小化损失：

$$
W \leftarrow W - \alpha \nabla_W L(W)
$$

其中 $\alpha$ 是学习率 (learning rate)。

### 2.7.2 数值梯度 vs 解析梯度

| 方法 | 公式 | 优点 | 缺点 |
|------|------|------|------|
| **数值梯度** | $\frac{df}{dx} \approx \frac{f(x+h)-f(x-h)}{2h}$ | 容易实现，用于梯度检查 | 慢，近似 |
| **解析梯度** | 通过微积分推导 | 精确，快速 | 需要推导，易出错 |

🟢 来自资料：实际中始终使用解析梯度，但用数值梯度做梯度检查 (gradient check) 来验证实现。

### 2.7.3 三种梯度下降变体

#### Batch Gradient Descent

$$
W \leftarrow W - \alpha \frac{1}{N} \sum_{i=1}^{N} \nabla_W L_i(W)
$$

- 每次使用全部数据
- 计算量大，不适合大数据集
- 确定性的收敛路径

#### Stochastic Gradient Descent (SGD)

$$
W \leftarrow W - \alpha \nabla_W L_i(W)
$$

- 每次随机选一个样本
- 更新非常快，但噪声大
- 收敛路径震荡

#### Mini-batch SGD

$$
W \leftarrow W - \alpha \frac{1}{B} \sum_{i=1}^{B} \nabla_W L_i(W)
$$

- 每次使用 $B$ 个样本（典型 $B = 32, 64, 128, 256$）
- **实际使用的标准方法**
- 平衡了计算效率和更新稳定性
- 可以利用 GPU 的并行计算能力

🟢 来自资料：Mini-batch SGD 是深度学习训练的默认选择。

---

## Practice Problems

### Problem 1: Linear Classifier Output

Given a 3-class linear classifier with:

$$
W = \begin{bmatrix} 0.1 & -0.5 & 0.3 \\ -0.2 & 0.8 & -0.1 \\ 0.4 & 0.1 & -0.6 \end{bmatrix}, \quad b = \begin{bmatrix} 0.1 \\ 0.0 \\ -0.2 \end{bmatrix}
$$

For input $x = [2.0, 1.0, 0.5]^T$, compute the scores and determine the predicted class.

**Solution:**

$$
s = Wx + b = \begin{bmatrix} 0.1 & -0.5 & 0.3 \\ -0.2 & 0.8 & -0.1 \\ 0.4 & 0.1 & -0.6 \end{bmatrix} \begin{bmatrix} 2.0 \\ 1.0 \\ 0.5 \end{bmatrix} + \begin{bmatrix} 0.1 \\ 0.0 \\ -0.2 \end{bmatrix}
$$

$$
s_0 = 0.2 - 0.5 + 0.15 + 0.1 = -0.05
$$
$$
s_1 = -0.4 + 0.8 - 0.05 + 0.0 = 0.35
$$
$$
s_2 = 0.8 + 0.1 - 0.30 - 0.2 = 0.40
$$

Predicted class: $\arg\max = 2$ (class 2, score 0.40).

### Problem 2: SVM Loss Calculation

For a 4-class problem with scores $s = [5, -3, 2, -1]$ and true class $y = 0$, compute the SVM loss with $\Delta = 1$.

**Solution:**

$$
\begin{aligned}
L &= \sum_{j \neq 0} \max(0, s_j - s_0 + 1) \\
&= \max(0, -3 - 5 + 1) + \max(0, 2 - 5 + 1) + \max(0, -1 - 5 + 1) \\
&= \max(0, -7) + \max(0, -2) + \max(0, -5) \\
&= 0 + 0 + 0 = 0
\end{aligned}
$$

正确类得分远高于其他，损失为 0。

### Problem 3: Softmax + Cross-Entropy

For scores $s = [2.0, 1.0, 0.1]$ and true class $y = 0$:

a) Compute softmax probabilities
b) Compute cross-entropy loss

**Solution:**

$$
\begin{aligned}
e^{2.0} &= 7.389, \quad e^{1.0} = 2.718, \quad e^{0.1} = 1.105 \\
\sum e^{s} &= 7.389 + 2.718 + 1.105 = 11.212 \\
P &= [0.659, 0.242, 0.099] \\
L &= -\log(0.659) = 0.417
\end{aligned}
$$

### Problem 4: L2 Regularization

If $W = \begin{bmatrix} 3 & -1 \\ 2 & 4 \end{bmatrix}$ and $\lambda = 0.1$, compute the L2 regularization loss and its gradient.

**Solution:**

$$
\begin{aligned}
R(W) &= \sum W_{ij}^2 = 9 + 1 + 4 + 16 = 30 \\
\lambda R(W) &= 3.0 \\
\nabla_W R(W) &= 2W = \begin{bmatrix} 6 & -2 \\ 4 & 8 \end{bmatrix}
\end{aligned}
$$

---

*Last updated: 2026-07-02*
