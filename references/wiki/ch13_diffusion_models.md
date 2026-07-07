# Ch13: Generative Models (Part 2) — Diffusion Models (扩散模型)

> 🟢 来自资料 — 基于课程讲义 `13_Generative Models_Part2.pdf` 及 DDPM, DDIM, Stable Diffusion, DiT, Rectified Flow 等论文

---

## 1. Diffusion Model Intuition (扩散模型直观物理原理)

扩散模型 (Diffusion Models) 的核心灵感源自非平衡热力学。它们通过学习**逆转一个逐渐给图像叠加噪声（加噪）的过程**，来掌握从纯随机噪声中生成逼真数据的本领。

模型包含两个基本物理过程：

1. **前向加噪过程 (Forward Diffusion)**（数学上固定的）：在 $T$ 个时间步内，逐渐向真实数据 $\mathbf{x}_0$ 中混入极微小的高斯噪声，直到其在第 $T$ 步彻底退化为一堆纯高斯随机噪声 $\mathbf{x}_T \sim \mathcal{N}(0, I)$。
2. **反向去噪过程 (Reverse Diffusion)**（可学习的）：训练一个深度神经网络（通常为 U-Net 或 Transformer），学习如何逐步将高斯随机噪声 $\mathbf{x}_T$ 进行一点点的去噪滤除，最终完美复原出干净的真实图像样本 $\mathbf{x}_0$。

$$\mathbf{x}_0 \xrightarrow{\text{前向逐步加噪}} \mathbf{x}_1 \xrightarrow{} \dots \xrightarrow{} \mathbf{x}_{T-1} \xrightarrow{} \mathbf{x}_T \xrightarrow{\text{反向逐步去噪}} \mathbf{x}_{T-1} \xrightarrow{} \dots \xrightarrow{} \mathbf{x}_0$$

**核心物理直觉**：相较于生成对抗网络 (GAN) 妄图通过单次前向网络就凭空“平地起高楼”地变出图片，扩散模型将这个极难的任务拆解成了 1000 步极其微小、在数学上极其容易近似的高斯去噪过渡步，因而获得了空前的训练稳定性和生成质量。

> 🟢 来自资料 — 扩散模型通过逐步加噪-去噪的过程生成数据，训练稳定，生成质量已超越 GAN。

---

## 2. DDPM (去噪扩散概率模型, 2020)

### 2.1 Forward Process (前向加噪过程)

前向加噪过程是一个特殊的**马尔可夫链 (Markov Chain)**，它严格遵循预先设定好的**方差调度表 (Variance Schedule)** $\beta_1, \beta_2, \dots, \beta_T \in (0, 1)$ 来逐步叠加微量高斯噪声：

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}\left(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}\right)$$

前向过程不含任何可学习的权重参数。通过应用变分推断和重参数化技巧，我们可以在 $O(1)$ 时间复杂度下，直接写出在任意时刻 $t$ 的加噪图像 $\mathbf{x}_t$ 的数学闭式解：

定义 $\alpha_t = 1 - \beta_t$，以及累乘项 $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$。则：

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I}\right)$$

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

当总时间步 $T \to \infty$ 时，累乘项 $\bar{\alpha}_T \to 0$，使得第 $T$ 步得到的图像退化为完全独立于原图的纯高斯白噪声：$\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$。

**方差调度策略 (Variance Schedules)**：包括线性调度（DDPM 原始版本采用）、改进的余弦调度（Cosine Schedule，由于能让加噪过渡更平缓而广受欢迎）或可学习调度。余弦调度数学公式为：$\bar{\alpha}_t = \frac{f(t)}{f(0)}$，其中 $f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$。

### 2.2 Reverse Process (反向去噪过程)

反向去噪也是一个马尔可夫链，但其状态转移概率是由神经网络**学习的高斯转移概率**：

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}\left(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \mathbf{\Sigma}_\theta(\mathbf{x}_t, t)\right)$$

在推理生成阶段，我们从一个纯随机高斯噪声 $\mathbf{x}_T \sim \mathcal{N}(0, I)$ 起步，通过神经网络迭代采样：

$$\mathbf{x}_{t-1} = \boldsymbol{\mu}_\theta(\mathbf{x}_t, t) + \mathbf{\Sigma}_\theta^{1/2}(\mathbf{x}_t, t) \cdot \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(0, I)$$

### 2.3 Training Objective (训练损失函数目标)

虽然变分下界推导看似复杂，但 DDPM 巧妙证明了：我们不需要让神经网络直接去预测去噪后的高斯均值 $\boldsymbol{\mu}_\theta$，而只需要让网络去**预测在前向过程中叠加到图像上的那个随机噪声项** $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$。

**简化的训练损失函数 (Simplified Loss)**（实际训练中唯一使用的金标准公式）：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$

其中：
- $t \sim \text{Uniform}(1, \dots, T)$（随机采样时间步）。
- $\mathbf{x}_0 \sim q(\mathbf{x}_0)$（输入真实的训练样本）。
- 随机噪声先验 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$。
- 混合得到的第 $t$ 步噪声图像为 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$。

这一损失在数学本质上等同于在多个不同噪声尺度上执行的**去噪分数匹配 (Denoising Score Matching)**。

一旦网络 $\boldsymbol{\epsilon}_\theta$ 训练完成，我们即可通过下式求出对应的反向均值：

$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)$$

而其反向方差项 $\mathbf{Sigma}_\theta$ 通常固定设置为常数 $\sigma_t^2 \mathbf{I}$，其中 $\sigma_t^2 = \beta_t$ 或较精细的后验方差下界 $\sigma_t^2 = \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$。

### 2.4 Sampling Algorithm (DDPM 迭代采样算法)

```
输入：已经训练完毕的噪声预测网络 ε_θ，总迭代去噪步数 T
从标准高斯分布中抽取纯随机噪声初始化：x_T ~ N(0, I)
for t = T, T-1, ..., 1:
    if t > 1:
        从高斯分布中抽取扰动噪声：z ~ N(0, I)
    else:
        令 z = 0 （最后一步不添加扰动项）
    依据均值和方差计算下一时间步的干净一点的图像：
    x_{t-1} = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t)) + σ_t · z
return 最终复原的高清图像：x_0
```

**为什么它表现极其稳定？**
因为网络只需要根据当前的加噪状态 $\mathbf{x}_t$ 去预测一个标准高斯噪声（在数学上这是一个定义良好的有监督 L2 连续值回归任务），避免了传统概率估计里的概率不重合、散度无穷大等问题。

> 🟢 来自资料 — DDPM 的核心贡献是简化目标函数：直接预测噪声而非均值，使训练极其稳定，且采样质量超过 GAN。

---

## 3. DDIM (去噪扩散隐式模型, 2021)

**研究痛点**：DDPM 在采样推理时，必须严格顺着马尔可夫链从 1000 到 1 逐个串行计算，这意味着生成一张图片需要多达 1000 次的网络前向推导，生成速度极慢。

**DDIM 核心突破**：将反向去噪过程重构为**非马尔可夫过程 (Non-Markovian)**，支持在保持相同边际概率分布的前提下，进行**确定性的、跳步（极少步骤）**的高速采样。

**DDIM 采样数学公式**（确定性形式，控制系数 $\eta = 0$）：

$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left(\frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{当前估计预测的干净原图 } \hat{\mathbf{x}}_0} + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + \sigma_t \mathbf{z}$$

其中扰动方差控制项为 $\sigma_t = \eta \sqrt{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}} \sqrt{1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}$。

- **确定性 DDIM ($\eta = 0$)**：反向扰动项完全为 0，反向路径变成一条完全确定性的轨道 —— 此时只要初始输入的噪声 $\mathbf{x}_T$ 固定，最终复原出的图像 $\mathbf{x}_0$ 保证一模一样。
- **随机性 DDIM ($\eta = 1$)**：等价于标准的随机马尔可夫 DDPM 采样。
- **跳步加速（Skip-step Sampling）**：由于不再依赖一步一高斯的马尔可夫推导，可以自由在 $\{1, \dots, T\}$ 中抽取子时序进行跳跃迭代（例如只在 $\{20, 40, \dots, 1000\}$ 的 50 个节点上计算）。

**核心结论**：DDIM 证明了扩散模型的采样本质属于常微分方程 (ODE) 轨迹求解。利用跳步技术，仅需 50~100 步即可生成质量几乎无损的精美画面。

> 🟢 来自资料 — DDIM 通过非马尔可夫公式实现确定性和加速采样，50-100 步即可得到优质结果。

---

## 4. Guidance (引导机制)

### 4.1 Classifier Guidance (分类器引导)

在推理去噪时，引入一个额外训练的、能对加噪图片进行正确分类的分类器 $p_\phi(c | \mathbf{x}_t)$，通过计算分类器对特定目标类别 $c$ 的梯度来修正去噪的方向。其数学分数函数（Score Function）修正如下：

$$\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | c) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log p(c | \mathbf{x}_t)$$

转换到噪声预测头上，带有分类器引导强度 $s$ 的校正公式为：

$$\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, c) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - s \cdot \sqrt{1 - \bar{\alpha}_t} \cdot \nabla_{\mathbf{x}_t} \log p_\phi(c | \mathbf{x}_t)$$

较大的引导强度 $s$ 会使生成的图片具有更强、更清晰的类别倾向性，但可能会稍微损害样本的多样性。

**最大局限**：必须额外花费精力在所有加噪噪声级别上，单独训练一个具备抗噪能力的分类器。

### 4.2 Classifier-Free Guidance (无分类器引导 / CFG)

CFG 在训练时，使用**单个**网络，在大部分时间传入正常条件 $c$，在 10%~20% 的时间随机丢弃条件（传入代表无条件的空字符 $\varnothing$）来进行多任务联合训练。在推理采样阶段，通过在有条件输出和无条件输出之间进行**线性外推（外插值）**，来放大引导效果：

$$\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, c) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing) + w \cdot \left[\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)\right]$$

其中 $w \geq 1$ 是**无分类器引导比例 (Guidance Scale)**：
- $w = 1$：标准的条件约束生成。
- $w = 0$：完全退化为无条件随机生成。
- $w > 1$：沿着条件所指示的方向进行放大外插（例如 Stable Diffusion 默认设为 $w = 7.5$），能让生成的细节与提示词语义（Prompt）高度贴合，极大提升视觉表现力。

**核心优势**：不需要单独训练分类器；训练逻辑极度简化，且能直接扩展到文本、图像、草图等任意控制条件上，是目前文生图模型的核心顶梁柱。

> 🟢 来自资料 — CFG 是当前文本到图像生成的核心技术，通过在训练时随机丢弃条件、推理时线性外推实现高质量可控生成。

---

## 5. Latent Diffusion Models (潜空间扩散模型 / Stable Diffusion, 2022)

**研究痛点**：直接在高维图像像素空间（例如 $512 \times 512 \times 3$ 的 RGB 空间，有多达 78 万个空间像素值）进行加噪去噪，其计算复杂度和内存开销极其高昂，难以在消费级显卡上运行。

**解决方案：潜空间扩散模型 (Latent Diffusion Models, LDM)**。先通过一个预训练好、编解码质量极高的自编码器，将图像压缩至一个低维特征**潜空间 (Latent Space)**，随后在此轻量级的特征图上执行扩散优化：

$$\mathbf{z} = \mathcal{E}(\mathbf{x}), \quad \hat{\mathbf{x}} = \mathcal{D}(\mathbf{z})$$

以著名的 Stable Diffusion 为例，它利用自编码器 $\mathcal{E}$ 将一幅 $512 \times 512 \times 3$ 的图像压缩至仅为 $64 \times 64 \times 4$ 的紧凑潜空间。这实现了 8 倍的空间下采样率（像素数压缩多于 64 倍），计算速度飙升。

**整体网络架构：**

1. **自编码器 (Autoencoder)**（常采用带 KL 散度约束或 VQ-GAN 约束的对称架构）：负责图像与潜空间的双向映射转换。使用重建、感知 (Perceptual) 及对抗损失进行端到端预训练。
2. **扩散骨干 U-Net (Latent U-Net)**：完全在 $64 \times 64$ 的潜特征图上进行高斯噪声的单步预测。
3. **多模态条件机制 (Cross-Attention)**：在 U-Net 的各个网络层中插入**交叉注意力 (Cross-Attention) 机制**，接收来自于 CLIP 文本编码器导出的提示词文本嵌入（Text Embeddings），进行细粒度的图文相关性注入：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V$$

其中查询向量 $Q$ 提取自潜特征图的当前注意力特征： $Q = W_Q^{(i)} \cdot \varphi_i(\mathbf{z}_t)$。
键向量 $K$ 和值向量 $V$ 则完全来自于多模态条件（如文本描述）的嵌入表征 $\tau_\theta(y)$：
$$K = W_K^{(i)} \cdot \tau_\theta(y), \quad V = W_V^{(i)} \cdot \tau_\theta(y)$$

**Latent 训练损失函数**：

$$\mathcal{L}_{\text{LDM}} = \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0,1), t, y}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \tau_\theta(y))\|^2\right]$$

> 🟢 来自资料 — Stable Diffusion 通过在潜空间进行扩散大幅降低了计算成本，配合交叉注意力实现了文本到图像生成。

---

## 6. DiT (Diffusion Transformers / 架构升级, 2022)

**革命性变革：在潜空间扩散模型框架下，完全用 Vision Transformer (ViT) 替换掉老旧的 U-Net。**

**核心系统流程：**

1. **补丁块拆分 (Patchify)**：将潜空间特征图 $\mathbf{z} \in \mathbb{R}^{I \times I \times C}$ 均匀切分成一系列 Patches 并通过线性投影投射为特征序列。
2. **DiT 特征块 (DiT Block)**：采用带有**自适应层归一化 (Adaptive Layer Norm, adaLN)** 机制的 Transformer 模块来平滑注入当前时间步信息：
   $$\text{adaLN}(h) = \gamma(t) \cdot \text{LayerNorm}(h) + \beta(t)$$
   其中缩放因子 $\gamma(t)$ 和平移因子 $\beta(t)$ 完全通过多层 MLP 接收时间步编码动态产生，并直接替换原 LayerNorm 中的可学习参数，以完成精准的时间和条件调度。
3. **线性预测层**：在序列末端恢复出特征形状并输出当前噪声。

**核心研究结论（Transformer 的 Scaling Laws 威力）**：DiT 展现出了强于 U-Net 的可扩展性。随着计算预算和网络参数量的成倍增加，DiT 的生成 FID 指标呈幂律平稳提升。增加网络参数体量比单纯增大潜空间特征尺度表现更佳。

> 🟢 来自资料 — DiT 证明了 Transformer 可以替代 U-Net 成为扩散模型的骨干，且具有更好的可扩展性。

---

## 7. Score-Based Models (基于分数的生成模型)

### 7.1 Score Function (分数函数定义)

(Stein) 分数函数定义为概率密度的对数关于自变量的梯度向量 $\nabla_{\mathbf{x}} \log p(\mathbf{x})$。它物理上指向了高维空间中，数据密度增长最陡峭的方向。

### 7.2 Score Matching (分数匹配训练)

建立一个神经网络 $\mathbf{s}_\theta(\mathbf{x})$ 来直接逼近这个高维的分数方向：

$$\mathcal{L}_{\text{SM}} = \mathbb{E}_{p_{\text{data}}(\mathbf{x})}\left[\|\mathbf{s}_\theta(\mathbf{x}) - \nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})\|^2\right]$$

鉴于真实的对数概率密度无法得知，通过采用**去噪分数匹配 (Denoising Score Matching, DSM)**（通过人为注入高斯扰动来构造目标项）来进行训练：

$$\mathcal{L}_{\text{DSM}} = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}, \tilde{\mathbf{x}} \sim \mathcal{N}(\mathbf{x}, \sigma^2 I)}\left[\|\mathbf{s}_\theta(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}} \log p(\tilde{\mathbf{x}}|\mathbf{x})\|^2\right]$$

如果引入多个不同的噪声尺度 $\{\sigma_i\}$ 进行联合回归，即构成了**噪声条件分数网络 (Noise Conditional Score Network, NCSN)**。

### 7.3 Langevin Dynamics Sampling (朗之万动力学采样)

在训练完成后，可以使用退火朗之万动力学 (Annealed Langevin Dynamics) 来引导噪声样本沿着高概率密度方向平滑逼近数据流：

$$\mathbf{x}_{t+1} = \mathbf{x}_t + \frac{\eta}{2} \nabla_{\mathbf{x}} \log p(\mathbf{x}_t) + \sqrt{\eta} \cdot \mathbf{z}_t, \quad \mathbf{z}_t \sim \mathcal{N}(0, I)$$

**与 DDPM 的深层数学底物理等价性**：
DDPM 的噪声预测网络 $\boldsymbol{\epsilon}_\theta$ 与基于分数的生成网络在本质上互为对立倒数：
$$\mathbf{s}_\theta(\mathbf{x}_t, t) = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

两者共同在一个统一的随机微分方程 (SDE) 数学分析框架下得到了融合。

> 🟡 AI补充: 基于分数的生成模型与 DDPM 在数学上等价 (SDE 框架)，提供了一种统一的连续时间视角。

---

## 8. Rectified Flow (2022)

**研究痛点**：DDPM 或者是 Wason-SDE 模型在概率路径中，去噪轨迹往往是弯曲、非线性的，导致求解数值离散时必须经过极其密集的小步长迭代才能确保精度。

**Rectified Flow 核心创新**：通过**线性插值 (Linear Interpolation)** 方案，直接学习并约束出一条连接数据与高斯噪声的**绝对直线（Straight Path）**传输路径。

构建其常微分方程 (ODE) 表达为：

$$\frac{d\mathbf{x}_t}{dt} = \mathbf{v}(\mathbf{x}_t, t)$$

网络需要直接逼近并预测出一个匀速直线运动的**速度向量场** $\mathbf{v}$：

$$\min_{\mathbf{v}} \mathbb{E}_{\mathbf{x}_0 \sim p_{\text{data}}, \mathbf{x}_1 \sim \mathcal{N}(0,I), t \sim U(0,1)}\left[\left\|(\mathbf{x}_1 - \mathbf{x}_0) - \mathbf{v}(\mathbf{x}_t, t)\right\|^2\right]$$

其中插值路径定义为绝对直线的两点插值： $\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\mathbf{x}_1$。

**突出优势：**
- **采样极速**：由于传输轨迹是一条完美的直线，求数值解时几乎没有任何曲率积累误差，在 1~10 步欧拉法单步推理下即可获得逼真的画质。
- **免去长时间模拟轨迹的训练成本**。
- 通过 Reflow（整流重对齐）迭代机制，能进一步拉直分布转换路径，代表当前业界极重要的前沿加速方案。

> 🟢 来自资料 — Rectified Flow 通过训练直线传输路径大幅减少了采样步数（1-10 步即可），是扩散模型加速的重要方向。

---

## 9. Applications (多模态应用矩阵)

| 核心领域 | 典型模型 | 机制特征描述 |
|-------------|--------|-------------|
| **文本生成图像 (Text-to-Image)** | Stable Diffusion, Imagen | 通过交叉注意力 (Cross-Attention) 融合 CLIP 提示词特征进行控制。 |
| **视频生成 (Text-to-Video)** | Sora, Video Diffusion Models | 在空间扩散上增加时间步注意力建模，保持时序连贯性。 |
| **局部图像修复 (Inpainting)** | RePaint, SD Inpainting | 固定非遮挡区并注入前向对应强度的加噪图，仅对缺失部分进行去噪。 |
| **超分辨率重建 (Super-Resolution)** | SR3, Cascaded Diffusion | 将低分辨率图拼接作为先验输入，自高斯噪声恢复高分辨率细节。 |
| **可控编辑 (Image Editing)** | InstructPix2Pix, SDEdit | 通过前向加一点噪声（如加噪到第 200 步），再在文本指示下进行重构去噪。 |
| **三维模型生成 (3D Generation)** | DreamFusion, Instant3D | 利用训练好的 2D 扩散模型作为 2D 分数蒸馏先验 (SDS Loss) 优化 NeRF 或 3D 属性。 |
| **生物分子生成** | Equivariant Diffusion Models | 联合等变物理约束，在三维空间中生成符合化学物理规律的分子空间构型。 |

---

## 10. Summary Formula Cheat Sheet (核心公式速查)

| 物理概念 | 数学公式 |
|---------|---------|
| 前向单步转移概率 | $q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}\left(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}\right)$ |
| 前向任意步直接加噪闭式解 | $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$ |
| DDPM 简化损失目标 | $\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$ |
| 反向均值解析恢复公式 | $\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)$ |
| 无分类器引导 (CFG) 外插公式 | $\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing) + w \cdot \left[\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)\right]$ |
| 确定性 DDIM 采样（无噪 $\eta=0$） | $\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{\mathbf{x}}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ |
| Stein 分数与噪声对应关系 | $\mathbf{s}_\theta(\mathbf{x}_t, t) = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$ |
| Rectified Flow 最小优化目标 | $\min_{\mathbf{v}} \mathbb{E}\left[\|(\mathbf{x}_1 - \mathbf{x}_0) - \mathbf{v}(\mathbf{x}_t, t)\|^2\right]$ |

---

## 11. Practice Problems (练习题与详解)

### Problem 1: Forward Process Derivation (前向任意时刻加噪公式推导)
请从马尔可夫单步加噪概率分布 $q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}\left(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}\right)$ 出发，运用数学归纳法与重参数化技巧，严谨推导出前向任意时刻直接加噪公式： $q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I}\right)$。

**Solution (解析):**
定义 $\alpha_t = 1 - \beta_t$。首先对单步转移概率应用重参数化采样：
$$\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_{t-1}, \quad \text{其中 } \boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(0, \mathbf{I})$$
类似地，将 $t-1$ 时刻的重参数化公式展开代入上式中：
$$\mathbf{x}_{t-1} = \sqrt{\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{1-\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}$$
代入后展开：
$$\mathbf{x}_t = \sqrt{\alpha_t}\left( \sqrt{\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{1-\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2} \right) + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_{t-1}$$
$$\mathbf{x}_t = \sqrt{\alpha_t\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{\alpha_t(1-\alpha_{t-1})}\boldsymbol{\epsilon}_{t-2} + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_{t-1}$$
由于自变量 $\boldsymbol{\epsilon}_{t-2}, \boldsymbol{\epsilon}_{t-1}$ 为两个相互独立、服从标准正态分布的多元高斯随机变量，根据多元高斯分布的加法性质（两个独立高斯变量的线性组合仍为高斯变量，其均值为线性组合均值，方差为线性组合系数的平方和）：
我们可以将后两项噪声合并为一个高斯项。合并后的高斯分布均值为 0，方差为：
$$\sigma^2 = \left( \sqrt{\alpha_t(1-\alpha_{t-1})} \right)^2 + \left( \sqrt{1-\alpha_t} \right)^2 = \alpha_t(1-\alpha_{t-1}) + 1 - \alpha_t$$
$$\sigma^2 = \alpha_t - \alpha_t\alpha_{t-1} + 1 - \alpha_t = 1 - \alpha_t\alpha_{t-1}$$
因此，合并两项噪声后，上式等价于：
$$\mathbf{x}_t = \sqrt{\alpha_t\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{1 - \alpha_t\alpha_{t-1}}\boldsymbol{\epsilon}'_{t-2}, \quad \text{其中 } \boldsymbol{\epsilon}'_{t-2} \sim \mathcal{N}(0, \mathbf{I})$$
通过数学归纳法一直向下迭代递推到起始时刻 $0$：
$$\mathbf{x}_t = \sqrt{\alpha_t\alpha_{t-1}\dots\alpha_1}\mathbf{x}_0 + \sqrt{1 - \alpha_t\alpha_{t-1}\dots\alpha_1}\boldsymbol{\epsilon}$$
代入累乘定义 $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$：
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \text{其中 } \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$
这对应概率密度表达式：
$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I}\right)$$
推导完毕。

### Problem 2: Noise Prediction vs. Mean Prediction (噪声预测与均值预测的等价性证明)
在变分下界的严谨推导中，最本质的反向去噪优化目标是拉近推断均值与真实后验均值的距离： $\mathbb{E}_{q}\left[ \|\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) - \boldsymbol{\mu}_\theta(\mathbf{x}_t, t)\|^2 \right]$。请证明：在 DDPM 框架下，将网络设为直接预测噪声 $\boldsymbol{\epsilon}_\theta$ 与预测均值 $\boldsymbol{\mu}_\theta$ 在数学优化本质上是完全等价的。

**Solution (解析):**
在已知真实干净原图 $\mathbf{x}_0$ 的条件下，反向加噪马尔可夫链的真实后验高斯均值可以通过贝叶斯法则严格导出：
$$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t$$
在前向加噪的重参数化定义中，我们可以通过反函数关系，将 $\mathbf{x}_0$ 表达为当前状态 $\mathbf{x}_t$ 与所叠加噪声 $\boldsymbol{\epsilon}$ 的自变量形式：
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon} \Rightarrow \mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\left(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}\right)$$
我们将此关于 $\mathbf{x}_0$ 的代数式代回到后验均值公式中：
$$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \left[ \frac{1}{\sqrt{\bar{\alpha}_t}}\left(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}\right) \right] + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t$$
$$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \left[ \frac{\beta_t}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)} + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \right]\mathbf{x}_t - \frac{\beta_t \sqrt{\bar{\alpha}_{t-1}}\sqrt{1 - \bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}(1 - \bar{\alpha}_t)}\boldsymbol{\epsilon}$$
合并第一项中 $\mathbf{x}_t$ 的系数（利用 $\alpha_t = 1 - \beta_t$）：
$$\text{Coef} = \frac{\beta_t + \alpha_t(1 - \bar{\alpha}_{t-1})}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)} = \frac{\beta_t + \alpha_t - \bar{\alpha}_t}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)} = \frac{1 - \bar{\alpha}_t}{\sqrt{\alpha_t}(1 - \bar{\alpha}_t)} = \frac{1}{\sqrt{\alpha_t}}$$
第二项中 $\boldsymbol{\epsilon}$ 的系数简化为：
$$\text{Coef}_2 = \frac{\beta_t \sqrt{\bar{\alpha}_{t-1}}}{\sqrt{\alpha_t \bar{\alpha}_{t-1}}\sqrt{1-\bar{\alpha}_t}} = \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}}$$
因此，真实后验均值数学上表现为：
$$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}\right)$$
由此可见，真实后验均值仅与当前的噪声项 $\boldsymbol{\epsilon}$ 线性相关。因此，我们只需设定可学习的参数化表示结构为相同的对应关系：
$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)$$
在优化损失中：
$$\|\tilde{\boldsymbol{\mu}}_t - \boldsymbol{\mu}_\theta\|^2 = \left\| \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon} - \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta \right\|^2 = C \cdot \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\|^2$$
其中 $C$ 仅是与时间相关的权重缩放项。这极其严谨地证明了，预测噪声和预测均值在数学本质上具有等价的一致性，且直接回归噪声形式具有更简单、分布固定的数学优点。

### Problem 3: CFG Scale Effect (CFG 比例调节物理行为分析)
在文生图扩散模型训练中应用了无分类器引导（CFG）策略。
a) 请问当引导比例 $w$ 从 $1$ 逐渐增加到正常值 $7.5$，再到过饱和的极大值 $20$ 时，采样生成的图像会分别表现出什么空间特征？
b) 为什么会发生这些现象？

**Solution (解析):**
- a) 物理生成的演进表现：
  - **$w = 1$ (无外推条件)**：标准的条件生成模式。图片能大致遵循提示词，但整体质感较为平庸、画面饱和度略低、细节清晰度一般，有时会遗漏一些关键词描述。
  - **$w = 7.5$ (正常范围值)**：提示词顺从度极强，主体结构高度清晰、色彩浓郁明快，视觉焦点对比强烈，是生成画质的黄金甜点区。
  - **$w = 20$ (极端外推值)**：图像产生灾难性的“过度曝光（Deep-fried）”伪影、出现极不自然的高饱和度和强烈对比，线条和轮廓出现硬断层和形状扭曲，完全丧失了写实照片该有的平滑过度与真实质感。
- b) 物理发生机理：
  - 引导比例 $w$ 的本质是在条件梯度方向与无条件梯度方向之间进行线性外插值。通过增大 $w$ 相当于不断拉大有无条件概率的差值，从而把数据密度的峰值（Mode）压缩并推向极高位（使分布极其陡峭），从而极大提升了图片的局部特征对比和线条明朗度。
  - 然而，当 $w$ 过大时，外推向量会粗暴地将低维潜特征向量强行推离“真实图像的流形空间（Image Manifold）”，使得特征均值和方差严重失衡、超出了自编码器解码重建的正常物理定义域，从而在最终图像上映射出灾难性的水噪和饱和伪影。

### Problem 4: Comparing DDPM and DDIM (跳步确定性轨迹还原消融)
设我们从一个完全相同的高斯初始白噪声输入 $\mathbf{x}_T$ 出发，调用相同的已经训练好的噪声网络。
a) 分别运行 DDPM (1000 步马尔可夫去噪) 以及确定性 DDIM (50 步 $\eta = 0$ 跳步去噪)。请问这两个生成的图像 $\mathbf{x}_0$ 之间存在什么空间数学关联？
b) 为什么能具备这种关联？

**Solution (解析):**
- a) 这两张图像在语义架构上会表现出**极其高度的相似性**（甚至达到肉眼难辨的局部一致），它们生成的是**同一个画面主题**（包括主体姿态、背景布局、光影和配色完全一致），仅仅只在极其微小的极高频纹理（如发丝边缘、噪点颗粒）上存在由于数值积分截断带来的一丝微弱偏差。
- b) 产生这一高度相似性关联的物理机理在于：
  - 在设定 $\eta = 0$ 的确定性 DDIM 状态下，反向去噪方程退化为一个完全确定性的概率常微分方程（Probability Flow ODE）。这意味着从高斯噪声 $\mathbf{x}_T$ 到生成图像 $\mathbf{x}_0$ 的映射是一个双向一一映射的单射曲线。
  - 尽管 50 步的确定性 DDIM 在时间步数上对该常微分方程进行了极度粗糙的跳步离散化（使用较大的 $\Delta t$ 求解），但它与 1000 步 DDPM 实质在沿着相同的常微分流动轨迹进行积分求解。因此，相同的初始边界条件 $\mathbf{x}_T$ 会将两个不同积分精度的算子约束到空间中的同一个吸引子中。
  - 这项重大发现也促成了“基于确定性 DDIM Inversion 机制的可控真实图像无损重建与精准编辑（Image Inversion & Editing）”的发展。

### Problem 5: Rectified Flow Advantage (Rectified Flow 直线高速传输加速的数学机理)
为什么 Rectified Flow 能够在仅仅需要 1 ~ 10 步欧拉迭代的极少步骤下即可生成极其高清逼真的原图，而经典的 DDPM 却雷打不动地必须执行数百至上千步去噪计算？

**Solution (解析):**
- **DDPM 的弯曲轨迹瓶颈**：在经典 DDPM 与 Score-based 扩散概率微分方程中，粒子在数据转换空间中的前向与反向传输轨道是**极度弯曲的非线性抛物曲线**。在进行数值微分计算时，弯曲轨迹意味着局部二阶及高阶导数极大。如果单步跨越步长 $\Delta t$ 稍大，一阶离散的局部截断误差会呈级数放大，使得路径完全失控、偏离原分布。因此必须使用高达 1000 步的微小微元步长来平滑逼近该弯曲轨道。
- **Rectified Flow 的绝对直线优势**：Rectified Flow 则直接通过变分训练重构了概率转换速度场，在数学上强制要求并保证了连接数据 $\mathbf{x}_0$ 和高斯噪声 $\mathbf{x}_1$ 之间的传输路径是一条**最简单、最平直的匀速直线路径**（通过严格逼近 $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$）。
- 直线轨迹的最突出数学性质就是高阶导数恒等于 0，这使得在数值离散化时，哪怕一阶欧拉步长设得极大，其局部截断误差也完全不会发生曲率扩散。粒子只需要顺着匀速直线速度方向大步迈进（例如在 $t=0$ 时一步大步踏到 $t=1$），也能极其精准地接近真实的流形分布，因而将扩散模型的生成采样效率提升到了极致。

---

> 🟢 来自资料 — 扩散模型从 DDPM 的基础框架，经过 DDIM (加速)、CFG (可控生成)、LDM (高效)、DiT (可扩展性) 等发展，已成为图像生成领域的主导范式。理解前向/反向过程、噪声预测损失、引导机制是核心考点。