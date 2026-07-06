# Lab 01: PyTorch 入门与 FashionMNIST 实验

> 🟢 来自资料 &nbsp;|&nbsp; 🟡 AI补充

## 1. PyTorch 基础

### 1.1 张量 (Tensor)

张量是PyTorch的核心数据结构，本质上是一个多维数组，与NumPy的`ndarray`类似，但支持GPU加速和自动微分。

```python
import torch
import numpy as np

# 从Python data创建
data = [[1, 2], [3, 4]]
x = torch.tensor(data)

# 从NumPy创建（共享内存）
np_arr = np.array(data)
x_np = torch.from_numpy(np_arr)

# 常用工厂函数
x_ones = torch.ones_like(x)       # 全1
x_rand = torch.rand_like(x, dtype=torch.float)  # 随机
x_zeros = torch.zeros(3, 4)       # 全0
x_eye = torch.eye(4)              # 单位矩阵
```

**三大元数据**：`shape`、`dtype`、`device`

```python
tensor = torch.rand(3, 4)
print(tensor.shape)    # torch.Size([3, 4])
print(tensor.dtype)    # torch.float32
print(tensor.device)   # cpu
```

### 1.2 张量运算

```python
# 索引（与NumPy一致）
print(tensor[0])          # 第一行
print(tensor[:, 0])       # 第一列
print(tensor[..., -1])    # 最后一列

# 拼接
t1 = torch.cat([tensor, tensor], dim=1)   # shape变[3, 8]

# 矩阵运算
tensor.matmul(tensor.T)    # 矩阵乘法
tensor.mul(tensor)         # 逐元素乘法

# 原地操作（带下划线）
tensor.add_(5)             # tensor = tensor + 5
```

### 1.3 设备管理

```python
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"      # Apple Silicon GPU
    return "cpu"

device = get_device()
tensor = tensor.to(device)
model = model.to(device)  # 模型也需移到同一设备

# 在训练循环中
x, y = x.to(device), y.to(device)
```

**数据类型的常见选择**：

| dtype | 用途 | 典型场景 |
|-------|------|----------|
| `torch.float32` | 默认浮点型 | 模型参数、输入特征 |
| `torch.float64` | 双精度 | 需要高精度计算 |
| `torch.int64` / `torch.long` | 长整型 | 标签、索引 |
| `torch.uint8` | 无符号8位 | 图像像素值、mask |

---

## 2. 数据处理

### 2.1 Dataset 与 DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = ...  # 加载文件列表

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 返回 (image, label) 或 (image, target_dict)
        return image, label
```

`__len__` 和 `__getitem__` 是两个必须实现的方法。

```python
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

train_dataloader = DataLoader(
    training_data,
    batch_size=64,
    shuffle=True,     # 训练时打乱
)
```

### 2.2 DataLoader 关键参数

| 参数 | 说明 | 常用值 |
|------|------|--------|
| `batch_size` | 每批样本数 | 32, 64, 128 |
| `shuffle` | 是否打乱 | 训练True, 验证False |
| `num_workers` | 多进程加载 | 0 (CPU), 4+ (GPU) |
| `drop_last` | 丢弃不完整batch | False (默认) |
| `collate_fn` | 自定义batch拼接 | 检测任务需要 |

常见遍历模式：

```python
for batch, (x, y) in enumerate(train_dataloader):
    x, y = x.to(device), y.to(device)
    # x shape: [batch_size, channels, H, W]
    # y shape: [batch_size]
```

---

## 3. torchvision Transforms

### 3.1 基础变换

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),                    # PIL/ndarray → tensor, [0,255]→[0,1]
    transforms.Normalize(mean=[0.5], std=[0.5]),  # 归一化
])
```

`ToTensor` 将 PIL Image (H×W×C, uint8) 转为 tensor (C×H×W, float32)，并将像素范围从 `[0, 255]` 缩放至 `[0.0, 1.0]`。

`Normalize(mean, std)` 执行 $(x - \text{mean}) / \text{std}$，注意传入的mean和std应按通道数设置。

### 3.2 数据增强变换

| Transform | 作用 | 典型参数 |
|-----------|------|----------|
| `RandomHorizontalFlip` | 随机水平翻转 | `p=0.5` |
| `RandomVerticalFlip` | 随机垂直翻转 | `p=0.5` |
| `RandomRotation` | 随机旋转 | `degrees=(-30, 30)` |
| `RandomResizedCrop` | 随机裁剪后缩放 | `size=224, scale=(0.08, 1.0)` |
| `ColorJitter` | 颜色抖动 | `brightness=0.2, contrast=0.2` |
| `CenterCrop` | 中心裁剪 | `size=224` |
| `Resize` | 缩放 | `size=256` |

**训练 vs 验证变换示例**：

```python
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
```

> ImageNet 的 mean/std 值 `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` 是当使用ImageNet预训练权重时的标准归一化参数。

---

## 4. 构建模型

### 4.1 nn.Module

所有模型的基类，必须实现 `__init__` 和 `forward`。

```python
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)        # 返回logits
        return x
```

### 4.2 nn.Sequential

用于顺序叠加层的简单模型：

```python
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
```

### 4.3 常用层速查

| 层 | 用途 | 参数示例 |
|----|------|----------|
| `nn.Linear(in, out)` | 全连接 | `nn.Linear(784, 128)` |
| `nn.Conv2d(in_c, out_c, k)` | 2D卷积 | `nn.Conv2d(1, 32, 3, padding=1)` |
| `nn.MaxPool2d(k)` | 最大池化 | `nn.MaxPool2d(2)` |
| `nn.ReLU()` | 激活函数 | — |
| `nn.Sigmoid()` | 二分类输出 | — |
| `nn.BatchNorm2d(c)` | 批归一化 | `nn.BatchNorm2d(32)` |
| `nn.Dropout(p)` | Dropout | `nn.Dropout(0.5)` |
| `nn.Flatten()` | 展平 | — |
| `nn.AdaptiveAvgPool2d(size)` | 自适应池化 | `nn.AdaptiveAvgPool2d((1, 1))` |

**SmallCNN for FashionMNIST 示例**：

```python
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)            # 28→14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)            # 14→7
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # x: [B, 1, 28, 28]
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x
```

> 💡 `CrossEntropyLoss` 内置了 softmax，所以模型 `forward` 输出 **logits**（未归一化），不要加 softmax 层。

---

## 5. Autograd 自动微分

### 5.1 核心机制

```python
x = torch.randn(3, 4, requires_grad=True)
y = x.sum()        # 前向计算
y.backward()       # 反向传播
print(x.grad)      # dy/dx 的梯度
```

- `requires_grad=True` 标记需要跟踪梯度的张量
- `.backward()` 从标量开始反向计算所有梯度
- `.grad` 存储累积的梯度

### 5.2 梯度控制

```python
# 禁用梯度跟踪（推理/评估时）
with torch.no_grad():
    predictions = model(inputs)

# 原地操作前清空梯度
optimizer.zero_grad()   # 或 model.zero_grad()

# 冻结参数
for param in model.parameters():
    param.requires_grad = False
```

> ⚠️ 训练循环中必须先 `zero_grad()`，再 `backward()`，最后 `step()`，否则梯度会累积。

---

## 6. 损失函数与优化器

### 6.1 损失函数

| 损失函数 | 适用场景 | 调用方式 |
|----------|----------|----------|
| `nn.CrossEntropyLoss()` | 多分类（内置softmax） | `loss(pred_logits, labels)` |
| `nn.MSELoss()` | 回归 | `loss(pred, target)` |
| `nn.BCEWithLogitsLoss()` | 二分类（内置sigmoid） | `loss(pred_logits, labels)` |
| `nn.L1Loss()` | 回归（MAE） | `loss(pred, target)` |

```python
loss_fn = nn.CrossEntropyLoss()
pred = model(x)            # logits: [batch_size, num_classes]
loss = loss_fn(pred, y)    # y: [batch_size] 整数标签
```

### 6.2 优化器

```python
# SGD
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,         # 动量加速
    weight_decay=1e-4,    # L2正则化
)

# Adam（自适应学习率）
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
)
```

**SGD vs Adam 选择原则**：

| 优化器 | 优点 | 缺点 |
|--------|------|------|
| SGD + Momentum | 泛化能力通常更好 | 需要精细调参 |
| Adam | 收敛快，对学习率不敏感 | 可能过拟合 |

---

## 7. 训练循环

完整的训练循环包含四个核心步骤：

```python
def train(dataloader, model, loss_fn, optimizer, device):
    model.train()          # 切换到训练模式（启用dropout/batchnorm）
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        pred = model(x)                     # Step 1: 前向传播
        loss = loss_fn(pred, y)             # Step 2: 计算损失

        optimizer.zero_grad()               # Step 3a: 清空旧梯度
        loss.backward()                     # Step 3b: 反向传播
        optimizer.step()                    # Step 3c: 更新参数

def test(dataloader, model, loss_fn, device):
    model.eval()           # 切换到评估模式
    test_loss, correct = 0.0, 0
    with torch.no_grad():  # 不跟踪梯度
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    accuracy = correct / len(dataloader.dataset)
    return accuracy, test_loss / len(dataloader)
```

**四条铁律**（考试常考）：

1. `model.train()` 进入训练模式（启用 BatchNorm 和 Dropout）
2. `model.eval()` + `torch.no_grad()` 用于评估/推理
3. `zero_grad()` → `backward()` → `step()` 的顺序不可颠倒
4. `.to(device)` 必须在数据和模型上都执行

---

## 8. 模型保存与加载

### 8.1 state_dict

`model.state_dict()` 返回一个Python字典，将每层映射到其参数张量。这是PyTorch中保存和加载模型的推荐方式。

```python
# 保存
torch.save(model.state_dict(), "model_weights.pth")

# 加载（需要先实例化同结构模型）
model = MLP()
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()
```

### 8.2 保存整模型 vs 保存state_dict

| 方式 | 优点 | 缺点 |
|------|------|------|
| `torch.save(model.state_dict(), ...)` | 跨设备兼容、体积小 | 需重新构建模型结构 |
| `torch.save(model, ...)` | 一行加载 | Python版本/类定义依赖 |

> 🟢 考试中统一使用 `state_dict` 方式，这是官方推荐做法。

```python
# 加载时处理不同设备
model.load_state_dict(
    torch.load("weights.pth", map_location=torch.device("cpu"))
)
```

---

## 9. 迁移学习预览

### 9.1 核心思路

利用在大型数据集（如ImageNet）上预训练的模型，将其知识迁移到新任务上。

```python
from torchvision import models
from torchvision.models import ResNet18_Weights

# 加载预训练ResNet18
resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# 查看原始最终层
print(resnet18.fc)   # Linear(in_features=512, out_features=1000)

# 替换分类头（假设新任务有5个类别）
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 5)

# 冻结所有预训练参数
for param in resnet18.parameters():
    param.requires_grad = False

# 只训练新的分类头
# 需要将 requires_grad = True 的层的参数传给优化器
```

### 9.2 两种迁移范式

| 方式 | 操作 | 场景 |
|------|------|------|
| **特征提取 (Feature Extractor)** | 冻结backbone，只训练新分类头 | 小数据集、与预训练任务相似 |
| **微调 (Fine-tuning)** | 解冻部分/全部层，用更小学习率训练 | 数据集较大、与预训练任务有差异 |

> 🟡 参数统计代码在考试中常见：
> ```python
> trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
> ```

---

## 10. MLP vs CNN 对比实验

### 10.1 实验设置

- 数据集：FashionMNIST（10类，28×28灰度图）
- 训练样本：60,000，测试样本：10,000

### 10.2 MLP 结构

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, HIDDEN_DIM)   # 784 → 128
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_DIM, 10)

    def forward(self, x):
        x = self.flatten(x)     # [B, 784]
        x = self.relu(self.fc1(x))
        x = self.fc2(x)         # [B, 10]
        return x
```

### 10.3 CNN 结构

```python
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x
```

### 10.4 结果对比（参考值）

| 模型 | 参数量 | 测试准确率 | 特点 |
|------|--------|-----------|------|
| MLP (Hidden=128) | ~101K | ~86-88% | 全连接丢失空间结构 |
| SmallCNN (2 Conv) | ~15K | ~95-97% | 利用局部相关性和平移不变性 |

> 🟢 结论：CNN 用更少的参数获得了更高的准确率，因为卷积操作保留了图像的2D空间结构，并通过局部连接和权重共享大幅减少了参数量。

**为什么CNN优于MLP？**

- **局部连接 (Local Connectivity)**：每个神经元只连接局部区域，不需要看到整张图
- **权重共享 (Weight Sharing)**：同一个卷积核在整个图像上滑动，参数量恒定
- **平移不变性 (Translation Invariance)**：池化层使特征位置变化对输出影响减小
- **参数量级**：CNN的参数量与输入尺寸无关，而MLP随图像像素数平方增长

---

## 11. 考试要点 🎯

| 考点 | 关键知识 |
|------|----------|
| Tensor操作 | shape/dtype/device三属性；`.to()`设备转移；`matmul` vs `mul` |
| Dataset/DataLoader | 必须实现 `__len__` 和 `__getitem__`；`batch_size` 和 `shuffle` 参数 |
| Transforms | `ToTensor` 将 [0,255]→[0,1]；`Normalize` 公式；训练/验证变换的区别 |
| nn.Module | 必须实现 `__init__` 和 `forward`；`super().__init__()` 不可遗漏 |
| Autograd | `requires_grad` → `backward()` → `.grad`；`torch.no_grad()` 禁用计算图 |
| 训练循环 | `model.train()` → `zero_grad()` → `forward` → `loss` → `backward()` → `step()` |
| 损失函数 | `CrossEntropyLoss` 参数是 (logits, labels)；不要在前面加 softmax |
| model.save/load | 用 `state_dict()` 保存，用 `load_state_dict()` 加载 |
| 迁移学习 | 预训练权重 → 替换分类头 → 可选冻结backbone |
| MLP vs CNN | 参数量、空间结构保留、准确率差距原因 |

### 常见错误 ⚠️

1. **模型和数据在不同设备上**：必须都把 `.to(device)` 后再计算
2. **`CrossEntropyLoss` 的输入是logits**：不要在forward里加 `softmax`
3. **忘记 `zero_grad()`**：梯度会跨batch累积导致训练不收敛
4. **忘了 `model.eval()`**：推理时BatchNorm和Dropout行为不同
5. **`state_dict()` 而非整模型**：考试中只接受 `torch.save(model.state_dict(), ...)`
