# Ch12: Generative Models (Part 1) (经典生成模型：GAN 与 VAE)

> 🟢 来自资料 — 基于课程讲义 `12_Generative Models.pdf` 及 GAN, DCGAN, WGAN, StyleGAN, VAE 等经典论文

---

## 1. Generative Model Taxonomy (生成模型分类大纲)

生成模型 (Generative Models) 的根本目标是从给定的真实样本中，学习真实的潜在概率数据分布 $p_{\text{data}}(\mathbf{x})$，以便能够自主采样并生成高质量的新样本 $\tilde{\mathbf{x}} \sim p_{\text{model}}(\mathbf{x})$。

```
                               生成模型 (Generative Models)
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
显式密度模型 (Explicit Density Models)                           隐式密度模型 (Implicit Density Models)
  （显式定义并优化概率密度 f_θ(x)）                                （无显式密度，只支持从分布中采样）
         │                                                                   │
 ┌───────┴──────────────┐                                            ┌───────┴──────────────┐
 ▼                      ▼                                            ▼                      ▼
精确似然模型            近似似然模型                                     生成对抗网络           矩匹配模型
(Tractable Density)    (Approximate Density)                        (GANs)                 (Moment Matching)
 - 自回归模型           - 变分自编码器 (VAE, VQ-VAE)
 （PixelCNN/RNN）       - 基于能量的模型 (EM)
```

**核心特征对比：**

| 特性维度 | 显式密度模型 (Explicit Density) | 隐式密度模型 (Implicit Density / GANs) |
|----------|-----------------|-------------------------|
| **似然度评估 (Likelihood)** | 支持（或者支持计算其数学下界） | 不支持 |
| **采样生成的图像质量** | 通常较低（图像偏模糊） | 极高（细节清晰、逼真） |
| **数据多样性覆盖 (Mode Coverage)** | 极佳（受似然度约束，不易遗漏类别） | 较差（极易发生模式坍塌 Mode Collapse） |
| **训练稳定性** | 训练十分稳定（直接最大化似然） | 极其不稳定（对抗博弈极易失衡） |

> 🟢 来自资料 — 显式密度模型可评估似然但生成质量通常较低；隐式密度模型（GAN）生成质量高但训练不稳定且无法评估似然。

---

## 2. Autoregressive Models (自回归模型)

### 2.1 Formulation (数学建模)

自回归模型将多维联合概率分布，使用条件概率链式法则拆分为一系列一维条件概率的乘积：

$$p(\mathbf{x}) = \prod_{i=1}^{n} p(x_i | x_1, x_2, \dots, x_{i-1})$$

在图像生成任务中，模型按特定顺序（如从左到右、自上而下的行优先扫描，再按 R、G、B 通道顺序）逐个像素递进生成。

### 2.2 PixelRNN (2016)

使用 **行 LSTM (Row LSTM)** 和 **对角双向 LSTM (Diagonal BiLSTM)** 在空间上对长程上下文依赖进行建模：
- **Row LSTM**：当前像素取决于上一行的三角形感受野区域。
- **Diagonal BiLSTM**：将图像沿对角线进行扭曲投影，使当前像素的生成严格依赖于其上方和左侧的所有历史像素。

### 2.3 PixelCNN (2016)

使用**掩码卷积 (Masked Convolutions)** 替代了慢速的循环神经网络 (RNN)，以实现训练阶段的完全并行化：
- **掩码 A (Mask A)**：应用于卷积神经网络的第一层；将卷积核中心及右下位置权重置零，阻断当前像素“偷看”其自身。
- **掩码 B (Mask B)**：应用于后续的所有隐层；允许卷积核看见当前像素自身的中间通道特征，但依然阻断未来像素的信息。

PixelCNN 的全卷积设计使其在训练时速度极快，但在生成阶段（采样）由于必须逐像素自回归生成，速度依然缓慢。

**带门控的 PixelCNN (Gated PixelCNN)**：引入类似于 LSTM 的双分支门控机制以增强非线性拟合能力：

$$\mathbf{y} = \tanh(W_{k,f} * \mathbf{x}) \odot \sigma(W_{k,g} * \mathbf{x})$$

**局限性**：采样速度是硬伤（只能逐个像素串行推理），极难应用于高分辨率图像生成。

> 🟢 来自资料 — PixelCNN/RNN 开创了深度生成模型，但其逐像素生成的特性导致采样极慢。

---

## 3. Variational Autoencoders (VAE / 变分自编码器)

### 3.1 Probabilistic Formulation (变分概率建模)

变分自编码器通过引入连续的隐变量 (Latent Variable) $\mathbf{z}$ 来描述高维数据的生成过程：

$$p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}$$

其中隐变量的先验分布假定为标准多元高斯分布 $p(\mathbf{z}) = \mathcal{N}(0, I)$。

因为真实的后验概率分布 $p_\theta(\mathbf{z}|\mathbf{x}) = \frac{p_\theta(\mathbf{x}|\mathbf{z})p(\mathbf{z})}{p_\theta(\mathbf{x})}$ 在数学上是不可积（Intractable）的，VAE 引入了一个推断网络（编码器） $q_\phi(\mathbf{z}|\mathbf{x})$ 来逼近真实的后验。

### 3.2 ELBO Derivation (证据下界公式推导)

我们希望最大化对数似然 $\log p_\theta(\mathbf{x})$，通过引入变分近似后验 $q_\phi(\mathbf{z}|\mathbf{x})$：

$$\log p_\theta(\mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log p_\theta(\mathbf{x})\right]$$
$$= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{p_\theta(\mathbf{z}|\mathbf{x})}\right]$$
$$= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \cdot \frac{q_\phi(\mathbf{z}|\mathbf{x})}{p_\theta(\mathbf{z}|\mathbf{x})}\right]$$
$$= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right] + D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x}))$$
$$= \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]}_{\text{重建损失期望}} - \underbrace{D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{隐变量空间约束正则化}} + D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x}))$$

由于 KL 散度天然满足非负性 $D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x})) \geq 0$：

$$\log p_\theta(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})) \equiv \text{ELBO}$$

上式即为**证据下界 (Evidence Lower Bound, ELBO)** —— 在优化中通过最大化该下界，即可间接实现最大化数据似然度并缩小近似后验与真实后验的差距。

### 3.3 Reparameterization Trick (重参数化技巧)

编码器输出隐空间的均值 $\mu_\phi(\mathbf{x})$ 和方差 $\sigma_\phi^2(\mathbf{x})$。为了使得采样操作 $\mathbf{z} \sim \mathcal{N}(\mu_\phi(\mathbf{x}), \sigma_\phi^2(\mathbf{x}))$ 变得可导，引入重参数化：

$$\mathbf{z} = \mu_\phi(\mathbf{x}) + \sigma_\phi(\mathbf{x}) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

这样，随机性被剥离到了独立的标准噪声 $\epsilon$ 上，梯度可以平滑、确定性地回传流经 $\mu_\phi$ 与 $\sigma_\phi$。

### 3.4 VAE Loss (VAE 损失函数)

将最大化下界转化为最小化目标损失函数：

$$\mathcal{L}_{\text{VAE}}(\theta, \phi; \mathbf{x}) = -\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] + D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

- **重建损失 (Reconstruction Loss)**：如果生成器输出建模为高斯分布，则其等价于均方像素误差 $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$；若建模为伯努利分布，则为二分类交叉熵。
- **KL 散度正则化项 (KL Regularization)**：将高斯后验压缩至标准高斯先验，计算具有解析闭式解：

$$D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})) = -\frac{1}{2}\sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

### 3.5 β-VAE

通过在 KL 正则化项前引入缩放系数 $\beta$，以强迫模型提取**解耦表征 (Disentangled Representations)**：

$$\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})] - \beta \cdot D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

当设定 $\beta > 1$ 时，对隐空间的约束加大，迫使隐空间的各个维度相互解耦独立表示特定属性（如姿态、表情等），但可能会在一定程度上牺牲图像重建的清晰度。

> 🟢 来自资料 — VAE 通过变分推断 + 重参数化技巧实现可微的隐式变量建模，但生成的图像通常较模糊（ELBO 的均值效应）。

---

## 4. GANs (Generative Adversarial Networks / 生成对抗网络)

### 4.1 Min-Max Game (极大极小双人博弈)

GAN 引入两个网络相互竞争对抗：**生成器 (Generator, G)** 和**判别器 (Discriminator, D)**。

- $G(\mathbf{z})$：接收低维高斯噪声 $\mathbf{z} \sim p_z$，将其映射至高维数据图像空间 $\hat{\mathbf{x}}$。
- $D(\mathbf{x})$：接收图像，输出该图像属于“真实数据”而非“人工合成”的置信概率。

**生成对抗损失目标**：

$$\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]$$

- 判别器 $D$ 希望**极大化**该博弈值：即尽量把真实图片判为 1，生成图片判为 0。
- 生成器 $G$ 希望**极小化**该博弈值：即尽量让生成的图片骗过判别器，使 $D(G(z)) \to 1$（也即让 $\log(1 - D(G(z))) \to -\infty$）。

**最优判别器形式**（对于固定生成器 $G$）：

$$D^*_G(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}$$

将最优判别器代入博弈目标函数中，博弈的本质等价于最小化真实分布 $p_{\text{data}}$ 与生成分布 $p_g$ 之间的 **Jensen-Shannon 散度 (JS 散度)**：

$$V(G, D^*_G) = -\log 4 + 2 \cdot D_{\text{JS}}(p_{\text{data}} \| p_g)$$

### 4.2 Training Algorithm (博弈迭代训练算法)

```
对于每一次训练迭代周期：
    # 步骤 1：固定生成器，迭代更新判别器 D（更新 k 步）
    for k 步：
        从先验分布 p_z 中采样包含 m 个噪声样本的小批量 {z^(1), ..., z^(m)}
        从真实数据分布 p_data 中提取包含 m 个真实样本的小批量 {x^(1), ..., x^(m)}
        沿其随机梯度上升的方向更新判别器 D 的参数 θd：
            ∇_θd [ 1/m Σ log D(x^(i)) + 1/m Σ log(1 - D(G(z^(i)))) ]
    
    # 步骤 2：固定判别器，迭代更新生成器 G（更新 1 步）
    从先验分布 p_z 中采样包含 m 个噪声样本的小批量 {z^(1), ..., z^(m)}
    沿其随机梯度下降的方向更新生成器 G 的参数 θg：
        ∇_θg [ 1/m Σ log(1 - D(G(z^(i)))) ]
```

**非饱和损失技巧 (Practical non-saturating loss)**：在训练初期，$G$ 生成极差，$\log(1 - D(G(z)))$ 的梯度极其微弱（发生梯度饱和）。在实际编程中，我们通过**极大化** $\log D(G(z))$ 来代替极小化 $\log(1 - D(G(z)))$，从而为初始阶段的生成器提供充沛的引导梯度。

### 4.3 Training Issues (对抗训练三大死穴)

| 痛点问题 | 具体物理表现 |
|---------|-------------|
| **模式坍塌 (Mode Collapse)** | 生成器为了迎合判别器，只学会在几类极窄的局部高概率区域进行生成，导致生成的图像缺乏多样性。 |
| **训练失衡不收敛** | 极大极小值博弈（Min-max Dynamics）非常容易引起梯度振荡，使得损失曲线剧烈波动且网络难以收敛。 |
| **梯度消失** | 当判别器训练得太完美（$D$ 太强）时，生成器将无法获取任何有效的梯度，训练陷入停滞。 |
| **定量评估困难** | 由于缺乏可计算的概率密度，难以定量评估生成质量。目前常使用 Inception Score (IS) 或 Fréchet Inception Distance (FID) 作为代理指标。 |

> 🟢 来自资料 — GAN 的 min-max 训练本质上不稳定，模式坍塌是其最核心的问题之一。

### 4.4 DCGAN (深度卷积生成对抗网络, 2016)

DCGAN 为在图像生成中稳定地应用深度卷积层提供了一套极其高价值的经验性架构设计指南：

1. 在判别器中去除最大池化，改用**步长卷积 (Strided Convolutions)** 进行下采样；在生成器中使用**分数步长卷积 (Fractional-strided Convolutions / 转置卷积)** 进行上采样。
2. 在生成器和判别器中均使用**批归一化 (Batch Normalization)**（生成器输出层及判别器输入层除外）。
3. 剔除全连接隐层，全网络使用纯卷积流。
4. 生成器的所有隐层使用 **ReLU** 激活函数，输出层使用 **Tanh**（以便将图像归一化至 $[-1, 1]$）。
5. 判别器的所有网络层使用 **LeakyReLU** 激活函数（负斜率通常设为 0.2）。

**生成器核心架构图** (DCGAN)：输入低维噪声 $\to$ 线性投影 $\to$ 尺寸重构 $\to$ 多层转置卷积+BN+ReLU $\to$ 卷积输出层+Tanh。

> 🟢 来自资料 — DCGAN 的架构指南极大地稳定了 GAN 训练，被后来几乎所有 GAN 变体继承。

### 4.5 Conditional GAN (cGAN / 条件生成对抗网络)

在生成器和判别器的输入端，同时送入辅助的监督或控制信息 $\mathbf{y}$（如类别标签、文本描述、分割图像），以进行可控的定向图像生成：

$$\min_G \max_D V(D, G) = \mathbb{E}[\log D(\mathbf{x}|\mathbf{y})] + \mathbb{E}[\log(1 - D(G(\mathbf{z}|\mathbf{y})|\mathbf{y}))]$$

**物理实现**：将条件变量 $\mathbf{y}$（如单热编码或特征向量）与噪声 $\mathbf{z}$ 或图像 $\mathbf{x}$ 沿特征通道方向进行简单的拼接 (Concatenate)。

### 4.6 WGAN (Wasserstein GAN, 2017)

**研究痛点**：当真实分布 $p_{\text{data}}$ 与生成分布 $p_g$ 在高维空间分布不重合时（几乎必然发生），JS 散度值将恒为常数 $\log 2$，导致生成器梯度完全消失。

**解决方案：Wasserstein 距离 (推土机距离 Earth Mover's Distance)**：

$$W(p_{\text{data}}, p_g) = \inf_{\gamma \in \Pi(p_{\text{data}}, p_g)} \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \gamma}[\|\mathbf{x} - \mathbf{y}\|]$$

即为把一个概率分布搬迁运输转换为另一个分布所需的最小平均传输代价。通过 Kantorovich-Rubinstein 对偶性定理转换：

$$W(p_{\text{data}}, p_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[f(\mathbf{x})] - \mathbb{E}_{\mathbf{x} \sim p_g}[f(\mathbf{x})]$$

其中 $f$ 必须满足 **1-Lipschitz 连续性约束**（在 WGAN 中用一个“评论员 Critic”来替代普通的二分类判别器）。

**WGAN-GP (Gradient Penalty / 梯度惩罚)**：相比早期 WGAN 的权重强行裁剪 (Weight Clipping)，WGAN-GP 巧妙通过梯度惩罚的方式将判别器梯度的范数约束在 1 附近：

$$\mathcal{L}_{\text{GP}} = \lambda \cdot \mathbb{E}_{\hat{\mathbf{x}} \sim p_{\hat{\mathbf{x}}}}\left[(\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1)^2\right]$$

其中 $\hat{\mathbf{x}}$ 是在真实样本和生成样本之间连线上随机插值采样得到的样本点：$\hat{\mathbf{x}} = \epsilon \mathbf{x} + (1 - \epsilon) G(\mathbf{z})$，$\epsilon \sim U(0, 1)$。

**WGAN 带来的革命性优势：**
- 损失值（推土机距离）的大小与生成图像的视觉质量高度正相关，损失曲线第一次有了可解释的物理含义。
- 极具韧性的训练稳定性，几乎无需再担心生成器和判别器的博弈失衡。
- 在机制上大幅抑制了模式坍塌现象。

> 🟢 来自资料 — WGAN 用 Wasserstein 距离 + 梯度惩罚取代 JS 散度，大幅提升了 GAN 训练的稳定性和损失函数的可解释性。

### 4.7 StyleGAN (基于风格的生成对抗网络)

**映射网络 (Mapping Network)**：噪声 $\mathbf{z} \in \mathbb{R}^{512}$ 首先通过一个 8 层全连接 MLP 映射到解耦的中间风格特征空间 $\mathbf{w} \in \mathcal{W}$。这打破了极度纠缠的噪声先验。

**合成网络 (Synthesis Network)**：其输入起点不再是噪声，而是一个学到的恒定常数矩阵。在每一层不同分辨率的特征图上，通过**自适应实例归一化 (Adaptive Instance Normalization, AdaIN)** 动态注入提取自 $\mathbf{w}$ 的“风格信息”：

$$\text{AdaIN}(\mathbf{x}_i, \mathbf{y}) = \mathbf{y}_{s,i} \cdot \frac{\mathbf{x}_i - \mu(\mathbf{x}_i)}{\sigma(\mathbf{x}_i)} + \mathbf{y}_{b,i}$$

其中风格信息 $\mathbf{y} = (\mathbf{y}_s, \mathbf{y}_b)$ 是通过对风格向量 $\mathbf{w}$ 做仿射变换（"A" 模块）得到的，分别作用在特征图的特征通道级缩放和偏置平移上。

**三大特色技术：**
- **渐进式增长训练 (Progressive Growing)**（StyleGAN v1 引入）：随着训练进行，从极低分辨率（如 $4 \times 4$）逐渐平滑地引入更高层分辨率（直到 $1024 \times 1024$）。
- **风格混合 (Style Mixing)**：在合成不同分辨率时混用两个不同的风格向量 $\mathbf{w}_1, \mathbf{w}_2$。粗尺度（低分辨率层）主导宏观特征（如面部姿态、脸型），细尺度（高分辨率层）主导微观细节（如配色、皮肤纹理）。
- **随机噪声注入 (Stochastic Variation)**：在每一层特征图加上空间随机噪声，用以生成发丝、雀斑等纯随机的非语义微观细节。
- **StyleGAN v2 升级**：重新设计了特征归一化算法（解权重解调 Weight Demodulation），消除了特征图中的水滴状伪影，并加入了路径长度正则化。

> 🟢 来自资料 — StyleGAN 通过映射网络 + AdaIN 实现了对生成图像风格的精细解耦控制，是 GAN 在图像生成领域的巅峰之作。

### 4.8 Image-to-Image Translation (图像到图像翻译)

**pix2pix (2017)**：有监督条件下的图像翻译框架，需要严格的一对一配对数据 ($\mathbf{x} \to \mathbf{y}$)。

$$\mathcal{L}_{\text{pix2pix}} = \mathcal{L}_{\text{cGAN}}(G, D) + \lambda \mathcal{L}_{L1}(G)$$

其中 L1 像素均方误差项 $\mathcal{L}_{L1} = \mathbb{E}[\|\mathbf{y} - G(\mathbf{x}, \mathbf{z})\|_1]$ 强力迫使生成的图像紧扣真实目标边界，避免过度发散。生成器采用对称跳跃连接的 **U-Net** 结构，判别器采用空间局部分块判别的 **PatchGAN** 结构。

**CycleGAN (2017)**：用于**无配对数据**情况下的双向图像翻译。设计了互为倒数的两个生成器 $G: X \to Y$ 与 $F: Y \to X$，并引入**循环一致性损失 (Cycle Consistency Loss)**：

$$\mathcal{L}_{\text{CycleGAN}} = \mathcal{L}_{\text{GAN}}(G, D_Y) + \mathcal{L}_{\text{GAN}}(F, D_X) + \lambda \mathcal{L}_{\text{cyc}}(G, F)$$

其中循环一致性损失项为：

$$\mathcal{L}_{\text{cyc}}(G, F) = \mathbb{E}[\|F(G(\mathbf{x})) - \mathbf{x}\|_1] + \mathbb{E}[\|G(F(\mathbf{y})) - \mathbf{y}\|_1]$$

其物理本质是：将图片从域 $X$ 翻译到 $Y$，再反向翻译回 $X$，应该能够完整还原出原始图像。

> 🟢 来自资料 — pix2pix 和 CycleGAN 拓展了 GAN 到图像翻译任务，CycleGAN 的创新在于无需成对训练数据。

---

## 5. Summary Formula Cheat Sheet (数学公式速查大纲)

| 核心生成模型 | 物理公式 / 损失函数定义 |
|-------|-------------|
| **VAE 证据下界 (ELBO)** | $\log p(\mathbf{x}) \geq \mathbb{E}_q[\log p(\mathbf{x}|\mathbf{z})] - D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$ |
| **VAE 隐空间高斯 KL 散度** | $D_{\text{KL}} = -\frac{1}{2}\sum_j (1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$ |
| **GAN 对抗损失目标** | $\min_G \max_D \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1-D(G(z)))]$ |
| **WGAN 理论优化目标** | $\min_G \max_{\|D\|_L \leq 1} \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$ |
| **WGAN-GP 梯度惩罚损失** | $+ \lambda \mathbb{E}_{\hat{x} \sim p_{\hat{x}}}[(\|\nabla_{\hat{x}}D(\hat{x})\|_2 - 1)^2]$ |
| **pix2pix 损失构成** | $\mathcal{L}_{\text{cGAN}}(G, D) + \lambda \mathcal{L}_{L1}(G)$ |
| **CycleGAN 联合目标** | $\mathcal{L}_{\text{GAN}}(G, D_Y) + \mathcal{L}_{\text{GAN}}(F, D_X) + \lambda \mathcal{L}_{\text{cyc}}$ |

---

## 6. Practice Problems (练习题与详解)

### Problem 1: GAN Optimal Discriminator (GAN 最优判别器数学推导)
请在固定生成器 $G$ 的前提下，严谨地推导出使得对抗目标函数 $V(D, G)$ 达到极大值时的最优判别器表达式 $D^*_G(\mathbf{x})$。

**Solution (解析):**
当生成器 $G$ 固定的情况下，我们将极大极小目标展开为连续空间的积分形式：
$$V(D, G) = \int_x p_{\text{data}}(x) \log D(x) dx + \int_z p_z(z) \log(1 - D(G(z))) dz$$
根据测度映射，将第二项变换至图像概率空间 $p_g(x)$：
$$V(D, G) = \int_x \left[ p_{\text{data}}(x) \log D(x) + p_g(x) \log(1 - D(x)) \right] dx$$
为了使整个积分达到最大，我们只需在每一个局部自变量 $x$ 处，使得被积函数极大化。
令被积函数为 $f(D) = a \log D + b \log(1 - D)$，其中常量定义为 $a = p_{\text{data}}(x)$， $b = p_g(x)$。
对其关于自变量 $D$ 求偏导并令导数为零：
$$\frac{df}{dD} = \frac{a}{D} - \frac{b}{1 - D} = 0$$
$$\frac{a}{D} = \frac{b}{1 - D} \Rightarrow a(1 - D) = bD \Rightarrow a - aD = bD \Rightarrow D = \frac{a}{a + b}$$
将 $a, b$ 的物理意义代回，即得到最优判别器的数学解析闭式：
$$D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

### Problem 2: VAE vs GAN Comparison (VAE 与 GAN 技术特征终极对比)
请从训练损失函数目标、采样图像质量、概率多样性覆盖以及隐空间物理属性四个维度，对比剖析 VAE 与 GAN。

**Solution (解析):**
| 对比维度 | 变分自编码器 (VAE) | 生成对抗网络 (GAN) |
|--------|-----|-----|
| **训练基本准则** | 基于似然概率建模，直接最大化证据下界 (ELBO) | 纯基于无监督对抗博弈的双人极大极小竞争 (Min-max) |
| **采样生成的质量** | 图像表现较为模糊（受高斯重建损失倾向于平滑均值效应影响） | 细节极其逼真、边缘纹理清晰锐利 |
| **生成分布的多样性** | 极少出现模式丢失，能完整覆盖所有数据 Mode（受 KL 约束） | 极易面临局部常数退化的模式坍塌 (Mode Collapse) 风险 |
| **隐变量空间 (Latent Space)** | 具备良好的高斯平滑连续分布，高度可插值（Smooth, Interpolable） | 相对松散，不具备天生的标准多元高斯结构约束 |
| **特征后验推断** | 天生具有编码器网络，支持极其方便的特征推断 $\mathbf{x} \to \mathbf{z}$ | 无天生推断路径，只能通过额外引入架构（如 BiGAN）进行拟合 |
| **量化度量工具** | 可靠的对数似然概率下界（数学性质强） | 仅能依靠间接的图像特征分布度量（如 FID, IS 指标） |

### Problem 3: Mode Collapse (模式坍塌物理机制与拯救方案)
在 GAN 的实际训练中，所谓的“模式坍塌 (Mode Collapse)”是指什么现象？请设计一个现实发生的具体场景，并列举两种以上有效的算法拯救手段。

**Solution (解析):**
- **模式坍塌物理现象**：是指生成器通过训练走捷径，发现只需要输出极少数几个特定的极高质量样例，就能在极高概率下骗过当前的判别器。从而，生成器完全退化为重复生产这几种相同类型的样本，不再探索真实分布中其他多样的类别和模式。
- **现实发生的场景**：在手写数字 MNIST 数据集上训练 GAN。在理想情况下，模型应该产生 $0 \sim 9$ 的所有数字。但在发生模式坍塌时，生成器发现把所有的特征图全部印出数字 "1" 骗过判别器的概率最高，于是在随后的训练里它不论输入何种高斯噪声 $\mathbf{z}$，输出的全是各种粗细、微偏的数字 "1"，完全无法再生成数字 $2$、数字 $8$ 等其他数字类别。
- **算法拯救手段**：
  - **小批量判别机制 (Minibatch Discrimination)**：允许判别器不单张判定，而是对比一个 Batch 内部多张图像之间的余弦相似距离。如果生成器输出的图像高度重叠，判别器能一眼看穿并给予极大损失。
  - **采用 WGAN-GP 架构**：由于推土机距离对分布的平移和缺失极其敏感，具有天然抵抗模式坍塌的作用。
  - **风格混合机制 (Style Mixing)**：在 StyleGAN 中强制将不同的中间特征进行交叉注入，从而在底层机制上保障了输出的高多样性。

### Problem 4: Reparameterization Trick (重参数化技巧的必要性推导)
在 VAE 训练中，为什么必须引入重参数化技巧（Reparameterization Trick）？如果不引入，网络在进行反向传播更新时会面临什么数学阻碍？

**Solution (解析):**
- VAE 训练的根本阻碍在于**随机采样的不可导性**。VAE 的目标损失函数中需要计算 $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$。为了计算这个期望，前向传播必须从参数为 $\phi$ 的概率分布 $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$ 中采样出一个具体的隐特征向量 $\mathbf{z}$ 送入后续的生成网络。
- 如果直接进行采样操作，这一步骤在计算图上是一个**随机离散节点（Stochastic Node）**。当反向传播计算损失关于编码器参数的梯度 $\frac{\partial \mathcal{L}}{\partial \phi}$ 时，由于采样过程的输出并不以显式可微的函数形式依赖于 $\mu_\phi$ 和 $\sigma_\phi$，计算流在采样操作处会发生“断流”，使得梯度完全无法回传流经编码器，导致编码器的参数 $\phi$ 根本无法进行梯度优化。
- **重参数化技巧**通过将 $\mathbf{z}$ 变换为：
  $$\mathbf{z} = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
  成功将所有随机采样操作抽离成了外部独立的噪声 $\epsilon$。在新的计算图中，从 $\mu_\phi(x)$ 和 $\sigma_\phi(x)$ 到 $\mathbf{z}$ 之间现在是完全确定性、平滑可微的加法与点乘运算，这保证了梯度能够极其顺畅、确定地回传到编码器参数 $\phi$，使得端到端的变分推断联合训练得以顺利进行。

### Problem 5: WGAN-GP Gradient Penalty (WGAN-GP 随机插值梯度惩罚物理机理)
WGAN-GP 在强行实施 1-Lipschitz 连续性约束时，为什么偏偏要在真实样本 $p_{\text{data}}$ 和生成样本 $p_g$ 的随机插值线上进行梯度惩罚计算 $\hat{\mathbf{x}} = \epsilon \mathbf{x} + (1 - \epsilon) G(\mathbf{z})$？为什么仅仅只在真实和生成样本所在的局部点处施加惩罚是远远不够的？

**Solution (解析):**
- 1-Lipschitz 连续性约束 $\|\nabla_{\mathbf{x}} D(\mathbf{x})\|_2 \leq 1$ 必须在**整个高维数据空间的所有相连路径上**得到严格的物理保证，因为只有当函数处处平滑、且在连接两个分布的过渡带上不出现局部陡峭断层时，推土机距离的对偶优化形式才成立。
- 如果仅仅只是在真实样本点 $\mathbf{x}_{\text{real}} \sim p_{\text{data}}$ 或者仅仅在生成器产生的虚假点 $\mathbf{x}_{\text{fake}} \sim p_g$ 处施加梯度范数惩罚：
  - 判别器（评论员）依然可以在这两组独立的数据堆“空隙区域”中，学到极其陡峭、导数接近无穷大的断层函数。只要在采样点处的斜率满足要求，中间区域的失控依然会导致对偶定理崩溃、WGAN 发生训练不稳定或梯度消失。
- 由于高维空间的维度灾难，我们又不可能要求在整个高维空间中进行均匀的“处处施加惩罚”（计算代价不可承受）。
- **随机连线插值点采样** $\hat{\mathbf{x}} = \epsilon \mathbf{x} + (1 - \epsilon) G(\mathbf{z})$，通过对连接真实分布和生成分布的所有主要“传输路径”进行随机切片采样。这在数学上极佳、极低计算量地为那条最需要保持函数平滑的“运输主干道”施加了强大的平滑正则约束，从而在物理上保证了 WGAN 优化的完备性与高稳定性。

---

> 🟢 来自资料 — 生成模型两大主线：VAE (概率视角，稳定训练，模糊生成) 和 GAN (对抗视角，高质量生成，训练不稳定)。理解它们的数学基础和设计权衡是核心。