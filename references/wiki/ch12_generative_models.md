# Ch12: Generative Models (Part 1)

> 🟢 来自资料 — 基于课程讲义 `12_Generative Models.pdf` 及 GAN, DCGAN, WGAN, StyleGAN, VAE 等经典论文

---

## 1. Generative Model Taxonomy

Generative models aim to learn the true data distribution $p_{\text{data}}(\mathbf{x})$ from samples, enabling generation of new samples $\tilde{\mathbf{x}} \sim p_{\text{model}}(\mathbf{x})$.

```
Generative Models
├── Explicit Density Models
│   ├── Tractable Density: Autoregressive (PixelRNN, PixelCNN, WaveNet)
│   └── Approximate Density
│       ├── Variational: VAE, VQ-VAE
│       └── Energy-Based (Markov Chain): Boltzmann Machines
└── Implicit Density Models
    ├── GANs (sampling only, no density)
    └── Moment Matching
```

| Property | Explicit Density | Implicit Density (GANs) |
|----------|-----------------|-------------------------|
| Likelihood evaluation | Yes (or bound) | No |
| Sampling quality | Lower | Higher (sharper) |
| Mode coverage | Better | Worse (mode collapse) |
| Training stability | Stable | Unstable |

> 🟢 来自资料 — 显式密度模型可评估似然但生成质量通常较低；隐式密度模型（GAN）生成质量高但训练不稳定且无法评估似然。

---

## 2. Autoregressive Models

### 2.1 Formulation

Factor the joint distribution as a product of conditionals:

$$p(\mathbf{x}) = \prod_{i=1}^{n} p(x_i | x_1, x_2, \dots, x_{i-1})$$

For images, pixels are generated sequentially (e.g., row-major order with RGB channels).

### 2.2 PixelRNN (2016)

Uses **Row LSTM** and **Diagonal BiLSTM** to model spatial dependencies:
- Row LSTM: Each pixel depends on the row above (receptive field is a triangle).
- Diagonal BiLSTM: Skews the image so each pixel depends on all pixels above and to the left.

### 2.3 PixelCNN (2016)

Replaces RNN with **masked convolutions** for faster parallel training:
- **Mask A**: Applied to first layer; prevents seeing the current pixel.
- **Mask B**: Subsequent layers; allows seeing the current pixel's own channels but not future pixels.

PixelCNN is fully convolutional → much faster training than PixelRNN, but sequential at generation time.

**Gated PixelCNN**: Adds gating mechanism for better conditioning:

$$\mathbf{y} = \tanh(W_{k,f} * \mathbf{x}) \odot \sigma(W_{k,g} * \mathbf{x})$$

**Limitations**: Slow sampling (sequential), expensive for high-res images.

> 🟢 来自资料 — PixelCNN/RNN 开创了深度生成模型，但其逐像素生成的特性导致采样极慢。

---

## 3. Variational Autoencoders (VAE)

### 3.1 Probabilistic Formulation

VAE models the data generation process with a latent variable $\mathbf{z}$:

$$p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}$$

where $p(\mathbf{z}) = \mathcal{N}(0, I)$ is the prior.

The posterior $p_\theta(\mathbf{z}|\mathbf{x}) = \frac{p_\theta(\mathbf{x}|\mathbf{z})p(\mathbf{z})}{p_\theta(\mathbf{x})}$ is intractable → approximate with $q_\phi(\mathbf{z}|\mathbf{x})$.

### 3.2 ELBO Derivation

The Evidence Lower Bound:

$$\log p_\theta(\mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log p_\theta(\mathbf{x})\right]$$
$$= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{p_\theta(\mathbf{z}|\mathbf{x})}\right]$$
$$= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \cdot \frac{q_\phi(\mathbf{z}|\mathbf{x})}{p_\theta(\mathbf{z}|\mathbf{x})}\right]$$
$$= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right] + D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x}))$$
$$= \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]}_{\text{Reconstruction}} - \underbrace{D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{Regularization}} + D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x}))$$

Since the last KL term $\geq 0$:

$$\log p_\theta(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})) = \text{ELBO}$$

This is the **ELBO** — maximizing it also maximizes (a lower bound on) the log-likelihood.

### 3.3 Reparameterization Trick

To backpropagate through the stochastic sampling $\mathbf{z} \sim \mathcal{N}(\mu_\phi(\mathbf{x}), \sigma_\phi^2(\mathbf{x}))$, reparameterize:

$$\mathbf{z} = \mu_\phi(\mathbf{x}) + \sigma_\phi(\mathbf{x}) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Now gradients flow through $\mu_\phi$ and $\sigma_\phi$ deterministically.

### 3.4 VAE Loss

$$\mathcal{L}_{\text{VAE}}(\theta, \phi; \mathbf{x}) = -\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] + D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

- **Reconstruction loss**: $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$ (Gaussian decoder) or binary cross-entropy (Bernoulli decoder).
- **KL divergence** (Gaussian prior + Gaussian posterior):

$$D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})) = -\frac{1}{2}\sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

### 3.5 β-VAE

Adds a weight $\beta$ to the KL term to encourage **disentangled representations**:

$$\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})] - \beta \cdot D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

$\beta > 1$: Stronger pressure for factorized, disentangled latents (but may sacrifice reconstruction quality).

> 🟢 来自资料 — VAE 通过变分推断 + 重参数化技巧实现可微的隐式变量建模，但生成的图像通常较模糊（ELBO 的均值效应）。

---

## 4. GANs (Generative Adversarial Networks)

### 4.1 Min-Max Game

Two networks compete: a **Generator** $G$ and a **Discriminator** $D$.

- $G(\mathbf{z})$: Maps latent noise $\mathbf{z} \sim p_z$ to data space $\hat{\mathbf{x}}$.
- $D(\mathbf{x})$: Outputs probability that $\mathbf{x}$ is real (not generated).

**Adversarial objective**:

$$\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]$$

- $D$ wants to maximize: correctly classify real vs. fake.
- $G$ wants to minimize: fool $D$ (make $D(G(z)) \to 1$).

**Optimal Discriminator** (for fixed $G$):

$$D^*_G(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}$$

At optimal $D$, the game becomes minimizing the **Jensen-Shannon divergence** between $p_{\text{data}}$ and $p_g$:

$$V(G, D^*_G) = -\log 4 + 2 \cdot D_{\text{JS}}(p_{\text{data}} \| p_g)$$

### 4.2 Training Algorithm

```
for number of training iterations:
    # Step 1: Update Discriminator (k steps)
    for k steps:
        Sample minibatch of m noise samples {z^(1), ..., z^(m)} from p_z
        Sample minibatch of m examples {x^(1), ..., x^(m)} from p_data
        Update D by ascending its stochastic gradient:
            ∇_θd [1/m Σ log D(x^(i)) + 1/m Σ log(1 - D(G(z^(i))))]
    
    # Step 2: Update Generator (1 step)
    Sample minibatch of m noise samples {z^(1), ..., z^(m)} from p_z
    Update G by descending its stochastic gradient:
        ∇_θg [1/m Σ log(1 - D(G(z^(i))))]
```

**Practical trick**: Instead of minimizing $\log(1 - D(G(z)))$ (vanishing gradients when $G$ is bad), maximize $\log D(G(z))$ (non-saturating loss). This provides stronger gradients early in training.

### 4.3 Training Issues

| Problem | Description |
|---------|-------------|
| **Mode Collapse** | $G$ produces only a few modes of $p_{\text{data}}$; lacks diversity |
| **Instability** | Oscillating loss, non-convergence due to the min-max dynamics |
| **Vanishing Gradients** | When $D$ is too strong, gradients for $G$ vanish |
| **Evaluation Difficulty** | No single metric; Inception Score (IS) and Fréchet Inception Distance (FID) are proxies |

> 🟢 来自资料 — GAN 的 min-max 训练本质上不稳定，模式坍塌是其最核心的问题之一。

### 4.4 DCGAN (Deep Convolutional GAN, 2016)

Architectural guidelines for stable GAN training with CNNs:

1. Replace pooling with **strided convolutions** (discriminator) and **fractional-strided convolutions** (generator).
2. Use **Batch Normalization** in both $G$ and $D$ (except $G$ output and $D$ input layers).
3. Remove fully connected hidden layers.
4. **ReLU** in $G$ for all layers except output (**Tanh**).
5. **LeakyReLU** in $D$ for all layers (slope 0.2).

**Generator architecture** (DCGAN): Linear → Reshape → Fractional-strided Conv + BN + ReLU → ... → Conv + Tanh.

> 🟢 来自资料 — DCGAN 的架构指南极大地稳定了 GAN 训练，被后来几乎所有 GAN 变体继承。

### 4.5 Conditional GAN (cGAN)

Condition both $G$ and $D$ on auxiliary information $\mathbf{y}$ (class label, text, image):

$$\min_G \max_D V(D, G) = \mathbb{E}[\log D(\mathbf{x}|\mathbf{y})] + \mathbb{E}[\log(1 - D(G(\mathbf{z}|\mathbf{y})|\mathbf{y}))]$$

**Implementation**: Concatenate condition $\mathbf{y}$ with input ($\mathbf{z}$ for $G$, $\mathbf{x}$ for $D$).

### 4.6 WGAN (Wasserstein GAN, 2017)

**Problem**: JS divergence has discontinuities → may not provide useful gradients when $p_{\text{data}}$ and $p_g$ have disjoint supports.

**Wasserstein distance** (Earth Mover's Distance):

$$W(p_{\text{data}}, p_g) = \inf_{\gamma \in \Pi(p_{\text{data}}, p_g)} \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \gamma}[\|\mathbf{x} - \mathbf{y}\|]$$

The infimum over all joint distributions with marginals $p_{\text{data}}$ and $p_g$. Using Kantorovich-Rubinstein duality:

$$W(p_{\text{data}}, p_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[f(\mathbf{x})] - \mathbb{E}_{\mathbf{x} \sim p_g}[f(\mathbf{x})]$$

where $f$ is a 1-Lipschitz function (the "critic", replacing discriminator).

**WGAN-GP (Gradient Penalty)**: Enforces Lipschitz constraint via gradient penalty (instead of weight clipping):

$$\mathcal{L}_{\text{GP}} = \lambda \cdot \mathbb{E}_{\hat{\mathbf{x}} \sim p_{\hat{\mathbf{x}}}}\left[(\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1)^2\right]$$

where $\hat{\mathbf{x}}$ is a random interpolation between real and generated samples:
$\hat{\mathbf{x}} = \epsilon \mathbf{x} + (1 - \epsilon) G(\mathbf{z})$, with $\epsilon \sim U(0, 1)$.

**Benefits over vanilla GAN**:
- Meaningful loss curves (correlated with sample quality)
- More stable training
- Reduced mode collapse

> 🟢 来自资料 — WGAN 用 Wasserstein 距离 + 梯度惩罚取代 JS 散度，大幅提升了 GAN 训练的稳定性和损失函数的可解释性。

### 4.7 StyleGAN (2019, 2020)

**Mapping Network**: Latent $\mathbf{z} \in \mathbb{R}^{512}$ → 8-layer MLP → intermediate latent $\mathbf{w} \in \mathcal{W}$. This disentangles the latent space.

**Synthesis Network**: Starts from a learned constant (not noise input). Style is injected at each resolution via **Adaptive Instance Normalization (AdaIN)**:

$$\text{AdaIN}(\mathbf{x}_i, \mathbf{y}) = \mathbf{y}_{s,i} \cdot \frac{\mathbf{x}_i - \mu(\mathbf{x}_i)}{\sigma(\mathbf{x}_i)} + \mathbf{y}_{b,i}$$

where $\mathbf{y} = (\mathbf{y}_s, \mathbf{y}_b)$ is generated from $\mathbf{w}$ via an affine transform ("A" block). Style is applied per-channel at each resolution.

**Key features**:
- **Progressive growing** (StyleGAN v1): Train from low to high resolution.
- **Style mixing**: Use two different $\mathbf{w}$ for different layers → coarse styles (pose, face shape) from one, fine styles (color, texture) from another.
- **Stochastic variation**: Per-pixel noise added at each resolution for stochastic details (hair, freckles).
- **StyleGAN v2**: Redesigned normalization (demodulation), path length regularization.

> 🟢 来自资料 — StyleGAN 通过映射网络 + AdaIN 实现了对生成图像风格的精细解耦控制，是 GAN 在图像生成领域的巅峰之作。

### 4.8 Image-to-Image Translation

**pix2pix** (2017): Conditional GAN with paired training data ($\mathbf{x} \to \mathbf{y}$).

$$\mathcal{L}_{\text{pix2pix}} = \mathcal{L}_{\text{cGAN}}(G, D) + \lambda \mathcal{L}_{L1}(G)$$

where $\mathcal{L}_{L1} = \mathbb{E}[\|\mathbf{y} - G(\mathbf{x})\|_1]$ encourages the output to be close to the target in pixel space. Uses **U-Net** generator and **PatchGAN** discriminator (classifies at patch level).

**CycleGAN** (2017): Unpaired image translation using **cycle consistency loss**:

$$\mathcal{L}_{\text{CycleGAN}} = \mathcal{L}_{\text{GAN}}(G, D_Y) + \mathcal{L}_{\text{GAN}}(F, D_X) + \lambda \mathcal{L}_{\text{cyc}}(G, F)$$

where $\mathcal{L}_{\text{cyc}} = \mathbb{E}[\|F(G(\mathbf{x})) - \mathbf{x}\|_1] + \mathbb{E}[\|G(F(\mathbf{y})) - \mathbf{y}\|_1]$ — translating there and back should recover the original.

> 🟢 来自资料 — pix2pix 和 CycleGAN 拓展了 GAN 到图像翻译任务，CycleGAN 的创新在于无需成对训练数据。

---

## 5. Summary Formula Cheat Sheet

| Model | Key Formula |
|-------|-------------|
| **VAE ELBO** | $\log p(\mathbf{x}) \geq \mathbb{E}_q[\log p(\mathbf{x}|\mathbf{z})] - D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$ |
| **VAE KL (Gaussian)** | $-\frac{1}{2}\sum(1 + \log\sigma^2 - \mu^2 - \sigma^2)$ |
| **GAN Objective** | $\min_G \max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$ |
| **WGAN** | $\min_G \max_{\|D\|_L \leq 1} \mathbb{E}[D(x)] - \mathbb{E}[D(G(z))]$ |
| **WGAN-GP** | $+ \lambda \mathbb{E}[(\|\nabla_{\hat{x}}D(\hat{x})\|_2 - 1)^2]$ |
| **pix2pix** | $\mathcal{L}_{\text{cGAN}} + \lambda \mathcal{L}_{L1}$ |
| **CycleGAN** | $\mathcal{L}_{\text{GAN}}(G, D_Y) + \mathcal{L}_{\text{GAN}}(F, D_X) + \lambda \mathcal{L}_{\text{cyc}}$ |

---

## 6. Practice Problems

### Problem 1: GAN Optimal Discriminator
Derive the optimal discriminator $D^*_G(\mathbf{x})$ for a fixed generator $G$.

**Solution:**
For fixed $G$, the objective is:
$$V = \int_x \left[p_{\text{data}}(x) \log D(x) + p_g(x) \log(1 - D(x))\right] dx$$

For each $x$, maximize $f(D) = a \log D + b \log(1 - D)$ with $a = p_{\text{data}}(x)$, $b = p_g(x)$.

$$\frac{df}{dD} = \frac{a}{D} - \frac{b}{1-D} = 0 \Rightarrow D = \frac{a}{a+b}$$

Thus $D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$.

### Problem 2: VAE vs GAN Comparison
Compare VAE and GAN in terms of: (a) training objective, (b) sample quality, (c) mode coverage, (d) latent space properties.

**Solution:**
| Aspect | VAE | GAN |
|--------|-----|-----|
| Objective | Maximize ELBO (likelihood-based) | Adversarial min-max game |
| Sample quality | Blurrier (mean of posterior) | Sharper, more realistic |
| Mode coverage | Better (KL encourages coverage) | Worse (mode collapse) |
| Latent space | Smooth, continuous, interpolable | Less structured |
| Inference | Has encoder → latent inference | No encoder (except BiGAN/ALI) |
| Evaluation | Log-likelihood (lower bound) | FID, IS (proxy metrics) |

### Problem 3: Mode Collapse
What is mode collapse in GANs? Describe a scenario and a potential remedy.

**Solution:**
Mode collapse occurs when $G$ learns to produce only a small subset of the data distribution's modes — for example, on MNIST, generating only digit "1" in various styles but never other digits. This happens because $G$ finds a local optimum where it fools $D$ with limited diversity.

Remedies:
- **Minibatch discrimination** (allow $D$ to compare samples within a batch)
- **Unrolled GANs** ($G$ sees how $D$ would respond after $k$ optimization steps)
- **WGAN** (Wasserstein distance is more sensitive to mode dropping)
- **Packing** (pacGAN: $D$ sees multiple samples at once)
- **StyleGAN's style mixing**: Explicitly encourages diversity

### Problem 4: Reparameterization Trick
Why is the reparameterization trick necessary for VAE training?

**Solution:**
The VAE loss includes $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$. Computing this requires sampling $z \sim \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$, which is a stochastic node. If we naively sample, the gradient $\frac{\partial}{\partial \phi}$ cannot flow through this stochastic operation because the sampling distribution depends on $\phi$ in a non-differentiable way.

The reparameterization trick rewrites $z = \mu_\phi + \sigma_\phi \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0,I)$ is independent of $\phi$. Now the gradient flows through $\mu_\phi$ and $\sigma_\phi$ deterministically, and the stochasticity is isolated to $\epsilon$ (which doesn't affect gradients).

### Problem 5: WGAN-GP Gradient Penalty
Explain why WGAN-GP uses interpolation $\hat{x} = \epsilon x + (1-\epsilon)G(z)$ instead of just sampling from $p_{\text{data}}$ or $p_g$.

**Solution:**
The Lipschitz constraint $\|\nabla_x D(x)\|_2 \leq 1$ must hold **everywhere on the data manifold** and along paths between real and generated distributions (not just at data points). Enforcing it only on $p_{\text{data}}$ or $p_g$ would leave the critic unconstrained in between, allowing it to learn a function that satisfies the constraint at sampled points but violates it in between — defeating the purpose.

Interpolation $\hat{x} = \epsilon x + (1-\epsilon)G(z)$ with $\epsilon \sim U(0,1)$ samples points along lines between real and fake data, providing a soft constraint that is computationally efficient and empirically effective.

---

> 🟢 来自资料 — 生成模型两大主线：VAE (概率视角，稳定训练，模糊生成) 和 GAN (对抗视角，高质量生成，训练不稳定)。理解它们的数学基础和设计权衡是核心。
