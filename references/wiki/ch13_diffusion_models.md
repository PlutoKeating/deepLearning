# Ch13: Generative Models (Part 2) — Diffusion Models

> 🟢 来自资料 — 基于课程讲义 `13_Generative Models_Part2.pdf` 及 DDPM, DDIM, Stable Diffusion, DiT, Rectified Flow 等论文

---

## 1. Diffusion Model Intuition

Diffusion models are inspired by non-equilibrium thermodynamics. They learn to generate data by **reversing a gradual noising process**.

**Two processes**:

1. **Forward diffusion** (fixed): Gradually add Gaussian noise to data $\mathbf{x}_0$ over $T$ steps until it becomes pure noise $\mathbf{x}_T \sim \mathcal{N}(0, I)$.
2. **Reverse diffusion** (learned): Train a neural network to gradually denoise $\mathbf{x}_T$ back to a clean sample $\mathbf{x}_0$.

$$\mathbf{x}_0 \xrightarrow{\text{forward (noise)}} \mathbf{x}_1 \xrightarrow{} \dots \xrightarrow{} \mathbf{x}_{T-1} \xrightarrow{} \mathbf{x}_T \xrightarrow{\text{reverse (denoise)}} \mathbf{x}_{T-1} \xrightarrow{} \dots \xrightarrow{} \mathbf{x}_0$$

**Key insight**: Each denoising step is a small, tractable Gaussian transition, unlike GAN's one-shot generation.

> 🟢 来自资料 — 扩散模型通过逐步加噪-去噪的过程生成数据，训练稳定，生成质量已超越 GAN。

---

## 2. DDPM (Denoising Diffusion Probabilistic Models, 2020)

### 2.1 Forward Process

A Markov chain that gradually adds Gaussian noise according to a **variance schedule** $\beta_1, \beta_2, \dots, \beta_T \in (0, 1)$:

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}\left(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}\right)$$

The forward process has no learnable parameters. Using reparameterization trick, we can sample $\mathbf{x}_t$ at any arbitrary timestep directly:

Let $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$. Then:

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I}\right)$$

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

As $T \to \infty$, $\bar{\alpha}_T \to 0$, so $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$.

**Variance schedules**: Linear (DDPM original), cosine (improved), or learned. The cosine schedule is popular: $\bar{\alpha}_t = \frac{f(t)}{f(0)}$ where $f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$.

### 2.2 Reverse Process

Also a Markov chain, but with **learned Gaussian transitions**:

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}\left(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \mathbf{\Sigma}_\theta(\mathbf{x}_t, t)\right)$$

Starting from $\mathbf{x}_T \sim \mathcal{N}(0, I)$, iteratively sample:

$$\mathbf{x}_{t-1} = \boldsymbol{\mu}_\theta(\mathbf{x}_t, t) + \mathbf{\Sigma}_\theta^{1/2}(\mathbf{x}_t, t) \cdot \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(0, I)$$

### 2.3 Training Objective

The variational bound can be simplified. Instead of predicting $\boldsymbol{\mu}_\theta$, DDPM predicts the **noise** $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ that was added to $\mathbf{x}_0$:

**Simplified loss** (the one used in practice):

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$

where:
- $t \sim \text{Uniform}(1, \dots, T)$
- $\mathbf{x}_0 \sim q(\mathbf{x}_0)$ (training data)
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$
- $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$

This is equivalent to **denoising score matching** over multiple noise scales.

Once $\boldsymbol{\epsilon}_\theta$ is trained, the mean can be derived:

$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)$$

The variance $\mathbf{\Sigma}_\theta$ is typically fixed to $\sigma_t^2 \mathbf{I}$ where $\sigma_t^2 = \beta_t$ or $\sigma_t^2 = \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$.

### 2.4 Sampling Algorithm (DDPM)

```
Input: Noise predictor ε_θ, number of steps T
x_T ~ N(0, I)   // Start from pure noise
for t = T, T-1, ..., 1:
    z ~ N(0, I) if t > 1 else z = 0
    x_{t-1} = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t)) + σ_t · z
return x_0
```

**Why it works**: Each step removes a small amount of noise, and the network only needs to predict the noise component (a regression task), making training very stable.

> 🟢 来自资料 — DDPM 的核心贡献是简化目标函数：直接预测噪声而非均值，使训练极其稳定，且采样质量超过 GAN。

---

## 3. DDIM (Denoising Diffusion Implicit Models, 2021)

**Problem**: DDPM sampling requires $T = 1000$ sequential steps → slow.

**DDIM reformulates** the reverse process as a **non-Markovian** process, enabling **deterministic** and **fewer-step** sampling:

**DDIM sampling** (deterministic, $\eta = 0$):

$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left(\frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{predicted } \mathbf{x}_0} + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + \sigma_t \mathbf{z}$$

where $\sigma_t = \eta \sqrt{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}} \sqrt{1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}$.

- $\eta = 0$: **Deterministic DDIM** — no random noise added, consistent output for a given $\mathbf{x}_T$.
- $\eta = 1$: Recovered DDPM (stochastic).
- Can skip steps: use a subsequence $\tau \subset \{1, \dots, T\}$ (e.g., 50 steps instead of 1000).

**Key property**: DDIM is an implicit probabilistic model — the marginal distribution matches DDPM, but the forward process is not defined (hence "implicit").

> 🟢 来自资料 — DDIM 通过非马尔可夫公式实现确定性和加速采样，50-100 步即可得到优质结果。

---

## 4. Guidance

### 4.1 Classifier Guidance

Use a pre-trained classifier $p_\phi(c | \mathbf{x}_t)$ to guide generation toward a target class. The score function is modified:

$$\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | c) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log p(c | \mathbf{x}_t)$$

In terms of the noise predictor (with guidance scale $s$):

$$\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, c) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - s \cdot \sqrt{1 - \bar{\alpha}_t} \cdot \nabla_{\mathbf{x}_t} \log p_\phi(c | \mathbf{x}_t)$$

Higher $s$ → stronger conditioning, but may sacrifice diversity.

**Limitation**: Requires training a noise-aware classifier on noisy images.

### 4.2 Classifier-Free Guidance (CFG)

Train a **single** diffusion model as both conditional and unconditional by randomly dropping the condition during training (e.g., 10-20% of the time). At inference, interpolate between conditional and unconditional predictions:

$$\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, c) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing) + w \cdot \left[\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)\right]$$

where $w \geq 1$ is the **guidance scale**:
- $w = 1$: Standard conditional generation
- $w = 0$: Unconditional
- $w > 1$: Amplifies the effect of conditioning (typically $w = 7.5$ for Stable Diffusion)

**Advantages over classifier guidance**: No separate classifier needed; simpler training; works with any conditioning type (text, image, etc.).

> 🟢 来自资料 — CFG 是当前文本到图像生成的核心技术，通过在训练时随机丢弃条件、推理时线性外推实现高质量可控生成。

---

## 5. Latent Diffusion Models (Stable Diffusion, 2022)

**Problem**: Operating in pixel space is computationally expensive. A $512 \times 512$ image has 262K dimensions.

**Solution**: Perform diffusion in a **compressed latent space** using a pre-trained autoencoder:

$$\mathbf{z} = \mathcal{E}(\mathbf{x}), \quad \hat{\mathbf{x}} = \mathcal{D}(\mathbf{z})$$

where $\mathcal{E}$ encodes the image to a lower-dimensional latent (e.g., $64 \times 64 \times 4$ for SD, an 8× spatial compression).

**Architecture**:

1. **Autoencoder** (VQ-GAN or KL-regularized): Encodes images to latent space + decodes back. Trained with reconstruction loss + perceptual loss + adversarial loss.
2. **Diffusion Model (U-Net)**: Operates entirely in latent space — predicts noise in the latent.
3. **Conditioning**: Cross-attention layers in the U-Net attend to text embeddings (from CLIP text encoder).

**Cross-attention conditioning**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V$$

where $Q$ comes from the U-Net feature map, $K, V$ come from the text embedding $\tau_\theta(y)$:

$$Q = W_Q^{(i)} \cdot \varphi_i(\mathbf{z}_t), \quad K = W_K^{(i)} \cdot \tau_\theta(y), \quad V = W_V^{(i)} \cdot \tau_\theta(y)$$

**Loss**:

$$\mathcal{L}_{\text{LDM}} = \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0,1), t, y}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \tau_\theta(y))\|^2\right]$$

> 🟢 来自资料 — Stable Diffusion 通过在潜空间进行扩散大幅降低了计算成本，配合交叉注意力实现了文本到图像生成。

---

## 6. DiT (Diffusion Transformers, 2022)

**Replace the U-Net backbone with a Vision Transformer (ViT)** in the latent diffusion framework.

**Architecture**:

1. **Patchify**: Latent $\mathbf{z} \in \mathbb{R}^{I \times I \times C}$ → sequence of patches.
2. **DiT Block**: Transformer block with **adaptive layer norm (adaLN)** for time conditioning:
   $$\text{adaLN}(h) = \gamma(t) \cdot \text{LayerNorm}(h) + \beta(t)$$
   where $\gamma(t), \beta(t)$ are generated from the time embedding via an MLP. These replace the layer norm's learned affine parameters.
3. **Final layer norm + linear**: Predict noise and (optionally) diagonal covariance.

**Key result**: DiT scales better than U-Net with increased compute — larger DiT models produce higher-quality images. Scaling the transformer (more parameters) is more effective than scaling the latent dimension.

> 🟢 来自资料 — DiT 证明了 Transformer 可以替代 U-Net 成为扩散模型的骨干，且具有更好的可扩展性。

---

## 7. Score-Based Models

### 7.1 Score Function

The **(Stein) score function** $\nabla_{\mathbf{x}} \log p(\mathbf{x})$ points in the direction of higher data density. Score-based generative models learn this function directly.

### 7.2 Score Matching

Learn a score model $\mathbf{s}_\theta(\mathbf{x})$ to approximate $\nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})$:

$$\mathcal{L}_{\text{SM}} = \mathbb{E}_{p_{\text{data}}(\mathbf{x})}\left[\|\mathbf{s}_\theta(\mathbf{x}) - \nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})\|^2\right]$$

Since $\nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})$ is unknown, use **denoising score matching** (perturb data with Gaussian noise):

$$\mathcal{L}_{\text{DSM}} = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}, \tilde{\mathbf{x}} \sim \mathcal{N}(\mathbf{x}, \sigma^2 I)}\left[\|\mathbf{s}_\theta(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}} \log p(\tilde{\mathbf{x}}|\mathbf{x})\|^2\right]$$

With multiple noise scales $\{\sigma_i\}$, this is **Noise Conditional Score Network (NCSN)**, which is equivalent to DDPM.

### 7.3 Langevin Dynamics Sampling

Given a score model, generate samples via annealed Langevin dynamics:

$$\mathbf{x}_{t+1} = \mathbf{x}_t + \frac{\eta}{2} \nabla_{\mathbf{x}} \log p(\mathbf{x}_t) + \sqrt{\eta} \cdot \mathbf{z}_t, \quad \mathbf{z}_t \sim \mathcal{N}(0, I)$$

At each noise level, run $T$ Langevin steps, then reduce noise and repeat.

**Connection to diffusion**: DDPM's $\boldsymbol{\epsilon}_\theta$ and the score function are related by:
$$\mathbf{s}_\theta(\mathbf{x}_t, t) = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

> 🟡 AI补充: 基于分数的生成模型与 DDPM 在数学上等价 (SDE 框架)，提供了一种统一的连续时间视角。

---

## 8. Rectified Flow (2022)

**Problem**: DDPM's trajectories are curved (not straight), requiring many discretization steps.

**Rectified Flow** learns a **straight** path from data to noise. The goal is an ODE:

$$\frac{d\mathbf{x}_t}{dt} = \mathbf{v}(\mathbf{x}_t, t)$$

where $\mathbf{v}$ is a velocity field learned to transport $p_{\text{data}} \to \mathcal{N}(0, I)$ via:

$$\min_{\mathbf{v}} \mathbb{E}_{\mathbf{x}_0 \sim p_{\text{data}}, \mathbf{x}_1 \sim \mathcal{N}(0,I), t \sim U(0,1)}\left[\left\|(\mathbf{x}_1 - \mathbf{x}_0) - \mathbf{v}(\mathbf{x}_t, t)\right\|^2\right]$$

where $\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\mathbf{x}_1$ is the **linear interpolation** (straight line).

**Key advantages**:
- **Fast sampling**: Straight trajectories → few discretization steps needed (1-10 steps).
- **Simulation-free training**: No need to simulate trajectories for training.

**Reflow**: Apply rectified flow iteratively to further straighten the trajectories.

> 🟢 来自资料 — Rectified Flow 通过训练直线传输路径大幅减少了采样步数（1-10 步即可），是扩散模型加速的重要方向。

---

## 9. Applications

| Application | Method | Description |
|-------------|--------|-------------|
| **Text-to-Image** | Stable Diffusion, DALL·E 2, Imagen | Cross-attention or CLIP embedding conditioning |
| **Text-to-Video** | Video Diffusion Models, Make-A-Video | Temporal diffusion + text conditioning |
| **Image Inpainting** | RePaint | Replace known regions with noisy versions of ground truth, diffuse unknown regions |
| **Super-Resolution** | SR3, cascaded diffusion | Low-res as conditioning, diffuse high-res |
| **Image Editing** | SDEdit, InstructPix2Pix | Add noise + denoise with text/instruction conditioning |
| **3D Generation** | DreamFusion (Score Distillation Sampling) | Optimize NeRF/3DGS using pre-trained 2D diffusion prior |
| **Molecular Generation** | E(3)-equivariant diffusion | Generate 3D molecular conformations |
| **Audio Generation** | AudioLDM, MusicLDM | Diffusion in mel-spectrogram space |

> 🟢 来自资料 — 扩散模型已从图像生成扩展到视频、3D、音频、分子等多个领域，成为最通用的生成建模框架。

---

## 10. Summary Formula Cheat Sheet

| Concept | Formula |
|---------|---------|
| **Forward process** | $q(\mathbf{x}_t \| \mathbf{x}_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$ |
| **Forward (direct)** | $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$ |
| **Simplified loss** | $\mathcal{L} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2]$ |
| **Reverse mean** | $\boldsymbol{\mu}_\theta = \frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta)$ |
| **CFG** | $\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing) + w[\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)]$ |
| **DDIM (deterministic)** | $\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{\mathbf{x}}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_\theta$ (with $\eta=0$) |
| **Score-DDPM relation** | $\mathbf{s}_\theta(\mathbf{x}_t, t) = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1-\bar{\alpha}_t}}$ |
| **Rectified Flow** | $\mathbf{v}^* = \arg\min \mathbb{E}[\|(\mathbf{x}_1 - \mathbf{x}_0) - \mathbf{v}(\mathbf{x}_t, t)\|^2]$ |

---

## 11. Practice Problems

### Problem 1: Forward Process Derivation
Derive $q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I})$ from $q(\mathbf{x}_t | \mathbf{x}_{t-1})$.

**Solution:**
Using the reparameterization trick repeatedly:
$$\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_{t-1}$$
$$= \sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{1-\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}) + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_{t-1}$$
$$= \sqrt{\alpha_t\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{\alpha_t(1-\alpha_{t-1})}\boldsymbol{\epsilon}_{t-2} + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_{t-1}$$

Since $\boldsymbol{\epsilon}_{t-2}, \boldsymbol{\epsilon}_{t-1}$ are independent Gaussians, their sum is $\mathcal{N}(0, [\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t)]\mathbf{I}) = \mathcal{N}(0, (1 - \alpha_t\alpha_{t-1})\mathbf{I})$.

Continuing this induction:
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$$

### Problem 2: Noise Prediction vs. Mean Prediction
Show that predicting $\boldsymbol{\mu}_\theta$ and predicting $\boldsymbol{\epsilon}_\theta$ are equivalent in the DDPM framework.

**Solution:**
From $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$, the true posterior mean (in terms of $\mathbf{x}_0$ and $\mathbf{x}_t$) is:
$$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t$$

We can express $\mathbf{x}_0$ from $\mathbf{x}_t$ and noise: $\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon})$.

Substituting gives $\boldsymbol{\mu}_\theta$ in terms of $\boldsymbol{\epsilon}_\theta$:
$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)$$

Thus predicting noise or predicting the mean are equivalent; noise prediction is empirically simpler (the target has a known distribution).

### Problem 3: CFG Scale Effect
A diffusion model trained with CFG on text prompts. What happens to the generated image as $w$ increases from 1 to 20?

**Solution:**
- $w = 1$: Standard conditional generation — image follows the prompt but may be somewhat bland.
- $w \approx 7.5$: Typical CFG value — prompt adherence is strong, image details are crisp, colors are vivid (the extrapolation sharpens the distribution).
- $w = 20$: Over-saturation, artifacts, distorted shapes, unnatural colors. The model over-extrapolates — it amplifies the "conditional" direction so strongly that the image leaves the manifold of realistic images. Think "deep fried" or hallucinogenic appearance.

### Problem 4: Comparing DDPM and DDIM
An image generated with DDPM (1000 steps) and DDIM (50 steps, $\eta=0$) from the same initial noise $\mathbf{x}_T$. How do the outputs relate?

**Solution:**
For $\eta = 0$, DDIM is **deterministic**: the mapping $\mathbf{x}_T \to \mathbf{x}_0$ is a bijective function. If the same $\mathbf{x}_T$ and model are used, DDIM with any subset of steps produces **exactly the same output**. The 50-step DDIM will produce a nearly identical image (minor differences due to the discretization of the trajectory). For $\eta > 0$, the outputs diverge (stochasticity is introduced).

This property makes DDIM useful for tasks like **inversion** (finding the $\mathbf{x}_T$ that corresponds to the given image), which enables real image editing.

### Problem 5: Rectified Flow Advantage
Why can Rectified Flow generate images in 1-10 steps while DDPM needs 1000?

**Solution:**
DDPM's ODE trajectories are **curved** — the path from noise to data is not a straight line in the space. Discretizing a curved path requires many small steps for accuracy (local truncation error accumulates).

Rectified Flow explicitly trains the velocity field so that the ODE $\frac{d\mathbf{x}}{dt} = \mathbf{v}(\mathbf{x}, t)$ follows **straight lines** from data to noise. A straight line can be discretized in very few steps — a single Euler step at $t=0$ already reaches the target roughly. Training enforces $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$ to be an exact solution of the ODE, so the learned velocity field naturally supports coarse discretization.

---

> 🟢 来自资料 — 扩散模型从 DDPM 的基础框架，经过 DDIM (加速)、CFG (可控生成)、LDM (高效)、DiT (可扩展性) 等发展，已成为图像生成领域的主导范式。理解前向/反向过程、噪声预测损失、引导机制是核心考点。
