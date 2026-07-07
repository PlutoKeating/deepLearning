# Lab 04: DCGAN 图像生成 (Image Generation)

> 🟢 来自资料 &nbsp;|&nbsp; 🟡 AI补充

## 1. 生成对抗网络 (GAN) 理论基础

### 1.1 对抗博弈机制 (Adversarial Game Theory)

生成对抗网络 (Generative Adversarial Networks, GAN) 包含两个核心组件：
1. **生成器 G (Generator):** 接收一个随机噪声向量 $z \sim p_z$ (通常是高斯分布或均匀分布)，通过复杂的映射网络将其转换为逼真的合成图像 $G(z)$。
2. **判别器 D (Discriminator):** 接收输入图像 $x$（可能是真实图像，也可能是 $G(z)$ 生成的虚假图像），输出一个表示输入为真实图像概率的标量值 $D(x) \in [0, 1]$。

两者在训练过程中进行最大最小博弈 (Minimax Game)：
- **$D$ 的目标：** 最大化正确判别真实图像和生成图像的概率，即最大化 $D(x)$ 且最小化 $D(G(z))$。
- **$G$ 的目标：** 产生足够逼真的图像去欺骗 $D$，即最大化 $D(G(z))$。

### 1.2 极小化极大值目标函数 (Minimax Objective Function)

GAN 的数学目标函数如下：

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))]$$

* **第一项 $\mathbb{E}_{x \sim p_{data}}[\log D(x)]$:** 判别器在真实数据 $x$ 上的表现。为了让 $D$ 正确，期望其输出接近 1，从而使 $\log D(x)$ 极大化。
* **第二项 $\mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))]$:** 判别器在生成图像 $G(z)$ 上的表现。期望其输出接近 0，从而使 $\log(1-D(G(z)))$ 极大化。而生成器 $G$ 试图让 $D(G(z))$ 接近 1，从而极小化该项。

---

## 2. 深度卷积对抗网络 (DCGAN) 设计规范

DCGAN (Deep Convolutional GAN) 引入了全卷积神经网络架构，极大地提高了生成高分辨率、高质量图像的训练稳定性。其核心规范如下：

| 网络结构 | 设计规则 (Architectural Constraints) | 助记与原理 (Mnemonic & Rationale) |
| :--- | :--- | :--- |
| **层替换** | 移除 Pooling 层，完全改用卷积/转置卷积 | **全卷积层替换 (All-Convolutional Net)：** 允许网络自适应地学习空间下采样/上采样，避免 Pooling 导致的信息丢失。 |
| **归一化** | 广泛使用 **Batch Normalization** (批量归一化) | **Batch Normalization (BN)：** 稳定梯度流，防止生成器将所有 $z$ 映射到单一点（即**模式崩溃 (Mode Collapse)**）。*注意：G 的输出层和 D 的输入层不能加 BN，避免由于缩放导致输出分布饱和。* |
| **结构级** | 移除全连接层 (Fully Connected Layers) | **去FC化：** 防止参数量爆炸导致过拟合，增强多层空间表征能力。 |
| **G 激活函数** | 除输出层外使用 **ReLU**；输出层使用 **Tanh** | **Tanh 输出限制：** $G$ 的输出被限制在 $[-1, 1]$。数据加载时图像像素也需同步归一化至 $[-1, 1]$ 范围，使分布匹配。 |
| **D 激活函数** | 全部使用 **LeakyReLU** (带泄漏的修正线性单元) | **防止死神经元 (Dead Neurons)：** D 往往训练太快。使用 LeakyReLU (一般斜率 $\alpha = 0.2$) 允许负梯度微弱通过，提供稳定的学习反馈。 |

---

## 3. 维度变换数学计算 (Spatial Dimension Formulas)

在 PyTorch 中构建生成器和判别器时，必须精准计算特征图宽度的转换：

### 3.1 判别器：步长卷积下采样 (Strided Convolution)

用于对特征图进行二分之一尺寸缩减：

$$H_{out} = \lfloor \frac{H_{in} + 2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} \rfloor + 1$$

> 💡 **标准下采样参数（降为1/2）：** `kernel_size=4`, `stride=2`, `padding=1`。  
> 此时：$H_{out} = \frac{H_{in} - 4 + 2}{2} + 1 = \frac{H_{in}}{2}$。

### 3.2 生成器：转置卷积上采样 (Transposed Convolution)

用于对高维向量逐步进行空间放大，还原图像尺寸：

$$H_{out} = (H_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{kernel\_size} + \text{output\_padding}$$

> 💡 **标准上采样参数（放大为2倍）：** `kernel_size=4`, `stride=2`, `padding=1`, `output_padding=0`。  
> 此时：$H_{out} = (H_{in} - 1) \times 2 - 2 \times 1 + 4 = 2 \times H_{in}$。

---

## 4. 实验核心代码实现要点 (Key Implementations)

### 4.1 生成器设计 (Generator `nn.Module`)

生成器的结构是高维空间的扩展，通常从低维向量特征图 `[100, 1, 1]` 逐步扩张到通道为 `3`、大小为 `[64, 64]` 的真实图像：

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入为 nz=100 维的随机噪声向量 z，形状: [128, 100, 1, 1]
            # 输出通道数为 ngf*8 (512), 空间尺寸放大到 4x4
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # 转置卷积上采样: 4x4 -> 8x8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # 转置卷积上采样: 8x8 -> 16x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # 转置卷积上采样: 16x16 -> 32x32
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # 最后一层输出层: 32x32 -> 64x64
            # 输出通道数 nc=3 (RGB 图像)，注意不使用 BatchNorm，且激活函数使用 Tanh 限制到 [-1, 1]
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
```

### 4.2 判别器设计 (Discriminator `nn.Module`)

判别器是一个典型的二分类 CNN 结构，将一张 `[3, 64, 64]` 的图像不断下采样，最后压缩为形状 `[1]` 的概率标量：

```python
class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入图像形状 nc=3, 大小: [3, 64, 64]
            # 步长卷积下采样: 64x64 -> 32x32。注意第一层不加 BatchNorm
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 步长卷积下采样: 32x32 -> 16x16
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 步长卷积下采样: 16x16 -> 8x8
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 步长卷积下采样: 8x8 -> 4x4
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 最后一层：4x4 -> 1x1。输出一维标量
            # 使用 Sigmoid 激活函数将预测值映射到 [0, 1] 区间
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)
```

---

## 5. 对抗训练循环算法过程 (Adversarial Training Loop)

在每一轮 Batch 训练中，我们需要先后、分别训练判别器 $D$ 与生成器 $G$：

```python
# 初始化标准损失函数与全局真实/虚假标签
criterion = nn.BCELoss()
real_label = 1.0
fake_label = 0.0

# 优化器设置：DCGAN 经典学习率为 0.0002，Beta1 为 0.5
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
```

### 5.1 步骤 1：训练判别器 D (Minimize: $\text{Loss}_D = \text{Loss}_{D(real)} + \text{Loss}_{D(fake)}$)

判别器需要最大化正确判定，我们将输入分为真实图片与虚假图片两部分进行训练。

```python
# 1. 真实样本前向传播
netD.zero_grad()
real_images = data[0].to(device)
b_size = real_images.size(0)

# 使用标签平滑 (Label Smoothing) 来削弱 D：将 real_label 从 1.0 改为 0.9
label = torch.full((b_size,), 0.9, dtype=torch.float, device=device)
output = netD(real_images)
lossD_real = criterion(output, label)
lossD_real.backward()

# 2. 生成虚假样本并前向传播
noise = torch.randn(b_size, nz, 1, 1, device=device)
fake_images = netG(noise)

# fake_label 依然保持 0.0
label.fill_(0.0)
output = netD(fake_images.detach()) # 梯度不传回给 G，所以使用 .detach()
lossD_fake = criterion(output, label)
lossD_fake.backward()

# 更新判别器参数
optimizerD.step()
```

### 5.2 步骤 2：训练生成器 G (Minimize: $\text{Loss}_G$)

生成器需要最大化它所生成的图片在判别器中的得分（即最大化 $D(G(z))$，使其骗过 D）：

```python
netG.zero_grad()

# 因为刚才已经执行了 optimizerD.step()，所以对 fake_images 再次前向传播 D
output = netD(fake_images) # 不加 .detach()，以便梯度完整向后传递到 G

# G 的目标是使 D 预测其输出为 real (即 0.9 或 1.0)
label.fill_(0.9) # 也可采用标签平滑
lossG = criterion(output, label)
lossG.backward()

# 更新生成器参数
optimizerG.step()
```

---

## 6. 核心考点与常见问题 (Core Concepts & FAQs)

> [!IMPORTANT]
> 🧠 **对抗振荡现象 (Adversarial Oscillation):**  
> GAN 的训练不是单调收敛的！判别器损失 $L_D$ 和生成器损失 $L_G$ 并不一定会降到 0。最理想的状态是两者的 Loss 在某个平衡范围相互波动拉扯。若 $L_D$ 迅速降为 0，说明判别器过于强大，生成器将得不到有效梯度，导致训练彻底停滞。所以经常使用 **Label Smoothing (标签平滑)** 来稍稍削弱判别器的判断自信。

> [!WARNING]
> ⚠️ **模式崩溃 (Mode Collapse):**  
> 表现为生成器产生极度单一重复的样本（比如画出的所有人脸几乎一模一样），网络陷入局部最优。通过引入 `Batch Normalization` 和选择合适的经典优化参数 (`lr=0.0002`, `betas=(0.5, 0.999)`) 可以极大地缓解此问题。
