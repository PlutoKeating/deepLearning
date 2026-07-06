# Lab 02: CNN 与迁移学习图像分类

> 🟢 来自资料 &nbsp;|&nbsp; 🟡 AI补充

## 1. CNN 架构回顾

### 1.1 经典架构演进

| 架构 | 年份 | 核心创新 | 层数 | 参数量 |
|------|------|----------|------|--------|
| **LeNet-5** | 1998 | 卷积+池化+全连接基本范式 | 7 | ~60K |
| **AlexNet** | 2012 | ReLU、Dropout、GPU训练 | 8 | ~60M |
| **VGG-16** | 2014 | 小卷积核(3×3)堆叠、更深 | 16 | ~138M |
| **ResNet-18** | 2015 | 残差连接(Residual Connection) | 18 | ~11.7M |

### 1.2 VGG设计哲学

- 全部使用 $3 \times 3$ 卷积核（感受野与等效 $7 \times 7$ 相同但参数更少）
- 池化后通道数加倍，保持计算量平衡
- 简单堆叠的规则结构

### 1.3 ResNet 残差连接

残差块的核心公式：

$$y = \mathcal{F}(x, \{W_i\}) + x$$

其中 $\mathcal{F}(x, \{W_i\})$ 是待学习的残差映射，$+x$ 是恒等跳跃连接（skip connection）。

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 当维度不匹配时需要1x1投影
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)       # 残差连接
        out = F.relu(out)
        return out
```

**为什么ResNet有效？**

- 解决了深层网络的退化问题（degradation problem）
- 梯度可通过shortcut直接回传，缓解梯度消失
- 最差情况下残差块退化为恒等映射，不会比浅层网络更差

---

## 2. 数据增强技术

### 2.1 为什么需要数据增强？

增加训练数据的多样性，提升模型的泛化能力，减少过拟合。尤其在小数据集（如hymenoptera_data只有~240张训练图）上至关重要。

### 2.2 常用增强变换

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),  # 随机裁剪+缩放
    transforms.RandomHorizontalFlip(p=0.5),                  # 水平翻转
    transforms.RandomRotation(degrees=(-30, 30)),            # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2),    # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

### 2.3 各增强方法的作用

| 增强方法 | 解决的问题 | 典型参数 |
|----------|------------|----------|
| `RandomHorizontalFlip` | 左右朝向不变性 | `p=0.5` |
| `RandomResizedCrop` | 尺度和位置变化 | `size=224`, `scale=(0.08, 1.0)` |
| `RandomRotation` | 旋转不变性（部分） | `degrees=(-30, 30)` |
| `ColorJitter` | 颜色/光照变化 | `brightness=0.2` |
| `RandomErasing` | 遮挡鲁棒性 | `p=0.5` |

> 🟡 **关键原则**：增强只应用于训练集，验证集和测试集只用标准预处理（Resize + CenterCrop + Normalize）。

### 2.4 验证集变换

```python
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

验证时不使用随机增强，保证评估结果的可复现性。

---

## 3. 正则化技术

### 3.1 Dropout

训练时随机将神经元输出置零，防止特征共适应。

```python
model = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),     # 50%概率丢弃
    nn.Linear(256, 10),
)
```

- $p=0.5$ 是常用值
- 测试时自动关闭（Dropout层在 `model.eval()` 时会停止丢弃）
- 通常放在全连接层之后、激活函数之前

### 3.2 权重衰减 (Weight Decay / L2 正则化)

在优化器中添加惩罚项：

$$L_{\text{total}} = L_{\text{original}} + \lambda \sum w_i^2$$

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4,    # λ 的值
)
```

- `weight_decay` 等价于 L2 正则化的 $\lambda$
- 常用值：$10^{-4}$ 到 $10^{-3}$

### 3.3 Batch Normalization

对每个mini-batch进行归一化：

$$\hat{x} = \frac{x - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} \cdot \gamma + \beta$$

```python
nn.BatchNorm2d(num_features=32)   # CNN中的BN
nn.BatchNorm1d(num_features=128)  # 全连接层的BN
```

**BN的作用**：
- 加速收敛（允许更大学习率）
- 减轻对初始化的敏感性
- 有一定的正则化效果（减少对Dropout的依赖）
- 训练和评估时行为不同：训练时用batch统计量，评估时用运行均值/方差

> ⚠️ `model.train()` 和 `model.eval()` 主要就是控制BN和Dropout的行为。

---

## 4. 迁移学习

### 4.1 核心概念

**迁移学习 (Transfer Learning)**：将在源任务上学到的知识应用到目标任务上。

在本课程中的含义：
- **源任务**：ImageNet 1000类分类（120万+图片）
- **目标任务**：自定义小数据集分类（如ants vs bees，~240张训练图）

### 4.2 两种范式对比

| 方式 | 操作 | 学习率 | 可训练参数 | 适用场景 |
|------|------|--------|-----------|----------|
| **特征提取** | 冻结backbone，只训练新分类头 | 正常 (`1e-3`) | 极少（仅新的fc层） | 数据集很小、与源数据相似 |
| **微调** | 解冻部分/全部backbone，联合训练 | 较小 (`1e-4`) | 较多 | 数据集较大、与源数据有差异 |

### 4.3 特征提取器实现

```python
from torchvision import models
from torchvision.models import ResNet18_Weights

def setup_feature_extractor(num_classes):
    # Step 1: 加载预训练模型
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Step 2: 记录原始分类头（提交截图用）
    original_fc = model.fc

    # Step 3: 冻结所有backbone参数
    for param in model.parameters():
        param.requires_grad = False

    # Step 4: 替换分类头
    num_ftrs = model.fc.in_features    # 512
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model, original_fc
```

> 💡 为什么冻结backbone？因为浅层卷积学习的是通用特征（边缘、纹理），这些在ImageNet和ants/bees上都适用。

### 4.4 部分微调实现

```python
def setup_partial_finetuning(num_classes):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # 冻结早期层
    for param in model.parameters():
        param.requires_grad = False

    # 解冻最后几层（layer4和fc）
    for param in model.layer4.parameters():
        param.requires_grad = True

    # 替换分类头
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # requires_grad 默认 True

    return model
```

**ResNet18 结构速览**：

```
ResNet18
├── conv1 (7×7, stride 2)     ← 浅层，学习通用边缘/纹理
├── bn1, relu, maxpool
├── layer1 (2个BasicBlock)    ← 浅中层
├── layer2 (2个BasicBlock)    ← 中层
├── layer3 (2个BasicBlock)    ← 中深层
├── layer4 (2个BasicBlock)    ← 深层，更任务相关
├── avgpool
└── fc (512 → 1000)           ← 分类头
```

> 🟡 **解冻策略**：通常解冻 `layer4` 和 `fc`，保持前面的层冻结。深层特征更任务特定，微调收益更大。

### 4.5 ImageFolder 数据集加载

```python
from torchvision import datasets

image_datasets = {
    "train": datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        data_transforms["train"],
    ),
    "val": datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        data_transforms["val"],
    ),
}

dataloaders = {
    "train": DataLoader(image_datasets["train"], batch_size=8, shuffle=True),
    "val": DataLoader(image_datasets["val"], batch_size=8, shuffle=False),
}

class_names = image_datasets["train"].classes   # ["ants", "bees"]
```

`ImageFolder` 要求目录结构为：

```text
hymenoptera_data/
├── train/
│   ├── ants/   (124 images)
│   └── bees/   (121 images)
└── val/
    ├── ants/   (70 images)
    └── bees/   (83 images)
```

---

## 5. 实验：Ants vs Bees 分类

### 5.1 数据集说明

- 来自官方PyTorch迁移学习教程
- 训练集：~245张（ants ~124, bees ~121）
- 验证集：~153张（ants ~70, bees ~83）
- 图像尺寸不一，需预处理到224×224

### 5.2 四种方法对比

#### 方法A：SmallCNN 从头训练

```python
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

#### 方法B：SmallCNN + 数据增强

在方法A基础上添加增强变换，包括 `RandomResizedCrop`、`RandomHorizontalFlip` 等。

#### 方法C：ResNet18 特征提取器

- 加载ImageNet预训练权重
- 冻结所有backbone参数
- 替换 fc 为 `nn.Linear(512, 2)`
- 只训练新的fc层

#### 方法D：ResNet18 部分微调

- 加载ImageNet预训练权重
- 解冻 `layer4` 和新的 `fc`
- 其他层保持冻结
- 使用更小的学习率

### 5.3 预期准确率范围

| 方法 | 预期验证准确率 | 可训练参数 | 训练速度 |
|------|---------------|-----------|----------|
| SmallCNN（无增强） | 60-75% | ~1M | 快 |
| SmallCNN + 增强 | 70-82% | ~1M | 快 |
| ResNet18 特征提取 | 85-95% | ~1K | 很快 |
| ResNet18 部分微调 | 93-97% | ~3M | 中等 |

### 5.4 完整训练流程

```python
def train_model(name, model, dataloaders, loss_fn, optimizer, device, epochs):
    model = model.to(device)
    best_val_acc = 0.0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        for inputs, labels in dataloaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        # 验证阶段
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in dataloaders["val"]:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += loss_fn(outputs, labels).item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()

        val_acc = val_correct / len(dataloaders["val"].dataset)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"Epoch {epoch+1}/{epochs}, Val Acc: {val_acc:.4f}")

    return model, best_val_acc
```

### 5.5 模型选择与保存

```python
# 根据配置选择最优模型
if MODEL_TO_SAVE == "best":
    accuracies = {
        "cnn": cnn_acc,
        "feature_extractor": feature_acc,
        "finetune": finetune_acc,
    }
    selected_name = max(accuracies, key=accuracies.get)
elif MODEL_TO_SAVE == "feature_extractor":
    selected_name = "feature_extractor"
# ...

# 保存
torch.save(model.state_dict(), f"{selected_name}_weights.pth")
```

---

## 6. 全流程代码模板

```python
# Step 1: 数据变换
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# Step 2: 数据集
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ["train", "val"]
}

# Step 3: DataLoader
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=8, shuffle=(x == "train"))
    for x in ["train", "val"]
}

# Step 4: 模型
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 2)

# Step 5: 损失与优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.001,
    momentum=0.9,
)

# Step 6: 训练并保存
model, best_acc = train_model(...)
torch.save(model.state_dict(), "best_model.pth")
```

---

## 7. 考试要点 🎯

| 考点 | 关键知识 |
|------|----------|
| CNN架构 | LeNet→AlexNet→VGG→ResNet的演进和创新点 |
| 数据增强 | 只用于训练集；RandomResizedCrop, HorizontalFlip, ColorJitter作用 |
| 正则化 | Dropout位置和作用；weight_decay=L2；BN的训练/推理差异 |
| 特征提取 vs 微调 | 冻结策略、学习率差异、适用场景判断 |
| ResNet18结构 | conv1→layer1-4→avgpool→fc；各层通道数；残差连接公式 |
| 参数冻结 | `param.requires_grad = False`；优化器只传可训练参数 |
| ImageFolder | 目录结构要求；`dataset.classes` 获取类名 |
| 训练/验证变换 | train用RandomResizedCrop，val用Resize+CenterCrop |
| 迁移学习三步 | 1)加载预训练 2)冻结backbone 3)替换分类头 |
| 模型选择 | 对比验证准确率选最优；`torch.save(state_dict)` |

### 常见错误 ⚠️

1. **验证集用了增强**：验证时不能加 `RandomHorizontalFlip` 等随机变换
2. **冻结后优化器仍包含所有参数**：优化器应只传入 `requires_grad=True` 的参数
3. **忘记 `model.eval()`**：BN使用错误统计量导致验证不准确
4. **ImageNet mean/std 不匹配**：使用ImageNet预训练模型时Normalize必须使用ImageNet的统计值
5. **微调学习率与特征提取相同**：微调需要更小的学习率（`1e-4` vs `1e-3`）
