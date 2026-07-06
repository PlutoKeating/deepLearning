# Lab 03: 目标检测与实例分割

> 🟢 来自资料 &nbsp;|&nbsp; 🟡 AI补充

## 1. 目标检测基础

### 1.1 与图像分类的区别

| 任务 | 输出 | 示例 |
|------|------|------|
| **图像分类** | 一个类别标签 | "这张图是猫" |
| **目标检测** | 每个目标的 (bbox + 类别) | "位置(10,20,100,150)有一只猫" |
| **实例分割** | 每个目标的 (bbox + 类别 + 逐像素mask) | 同上 + 精确到像素的猫形区域 |

### 1.2 检测数据集格式

一张图片对应一组标注目标，每个目标包含：

```python
target = {
    "boxes": torch.Tensor,      # [N, 4]  (x1, y1, x2, y2) 格式
    "labels": torch.Tensor,     # [N]     整数标签，从1开始(0是背景)
    "masks": torch.Tensor,      # [N, H, W]  二值mask（实例分割特有）
    "image_id": torch.Tensor,   # [1]     图片编号
    "area": torch.Tensor,       # [N]     目标面积（用于评价指标）
    "iscrowd": torch.Tensor,    # [N]     COCO格式：是否为密集人群
}
```

> ⚠️ boxes使用 `(x1, y1, x2, y2)` 格式，即左上角和右下角坐标，是整数或浮点数。

---

## 2. COCO格式与 Penn-Fudan 数据集

### 2.1 COCO 格式简介

COCO (Common Objects in Context) 是目标检测和分割的标准数据集格式：

- 使用 JSON 文件存储标注
- 包含 80 个物体类别 + 背景
- 支持目标检测、实例分割、关键点检测
- 评估指标：mAP（mean Average Precision）

### 2.2 Penn-Fudan 数据集

本实验使用的行人检测数据集：

```text
PennFudanPed/
├── PNGImages/          # 170张RGB图像
└── PedMasks/           # 对应的实例分割mask
```

- Mask格式：每个像素值为 `0`（背景）或 `object_id`（1, 2, 3...）
- 每个 `object_id` 代表一个独立的行人实例

```python
import numpy as np
from PIL import Image

mask = Image.open("PennFudanPed/PedMasks/FudanPed00001_mask.png")
mask_tensor = torch.as_tensor(np.array(mask), dtype=torch.uint8)

# 获取所有实例ID（排除0=背景）
obj_ids = torch.unique(mask_tensor)
obj_ids = obj_ids[1:]    # [1, 2, 3, ...]

# 转为二值mask [N, H, W]
masks = mask_tensor == obj_ids[:, None, None]
```

---

## 3. Mask R-CNN 架构

### 3.1 整体结构

Mask R-CNN 是一个两阶段检测器：

```
输入图像 [3, H, W]
    │
    ▼
┌─────────────────────────────┐
│  Backbone (ResNet50 + FPN)  │  特征提取
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  RPN (Region Proposal Net)  │  生成候选区域
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  RoIAlign                   │  精确的ROI特征提取
└──────────────┬──────────────┘
               ▼
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌──────┐ ┌──────┐ ┌──────┐
│Box   │ │Class │ │Mask  │      三个并行预测头
│Head  │ │Head  │ │Head  │
└──────┘ └──────┘ └──────┘
```

### 3.2 各组件详解

| 组件 | 作用 | 实现 |
|------|------|------|
| **Backbone (ResNet50+FPN)** | 提取多尺度特征 | ImageNet预训练，可选冻结 |
| **FPN (Feature Pyramid Net)** | 融合高低层特征，提升多尺度检测 | 自顶向下路径+横向连接 |
| **RPN** | 生成候选区域（proposals），输出 objectness + bbox回归 | 滑动窗口+anchor机制 |
| **RoIAlign** | 将不同尺寸的ROI映射为固定尺寸特征图 | 双线性插值，避免RoIPool的量化误差 |
| **Box Head** | 对proposal进行最终分类和bbox精修 | 全连接层 |
| **Mask Head** | 为每个正样本预测二值分割mask | 小型FCN网络 |

### 3.3 RoIAlign 与 RoIPool 的区别

> 🟡 这是一个常考的细致知识点。

| 方法 | 量化方式 | 精度 | 适用任务 |
|------|----------|------|----------|
| **RoIPool** | 两次量化（坐标取整→分bin取整） | 有偏差 | 目标检测 |
| **RoIAlign** | 双线性插值，无量化 | 像素级精确 | 实例分割 |

RoIAlign 是 Mask R-CNN 相对于 Faster R-CNN 的关键改进，使得 mask 预测成为可能。

---

## 4. 训练 Mask R-CNN

### 4.1 自定义 Dataset

```python
class PennFudanDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs = sorted((root / "PNGImages").glob("*.png"))
        self.masks = sorted((root / "PedMasks").glob("*.png"))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = F.to_tensor(img)

        mask = torch.as_tensor(np.array(mask), dtype=torch.uint8)

        # 提取实例 → 转为二值mask
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]   # 去除背景(0)
        masks = mask == obj_ids[:, None, None]

        # 构建目标字典
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = torch.where(masks[i])
            xmin, xmax = pos[1].min(), pos[1].max()
            ymin, ymax = pos[0].min(), pos[0].max()
            boxes.append([xmin, ymin, xmax, ymax])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.ones((num_objs,), dtype=torch.int64),  # 1 = 行人
            "masks": masks,                   # [num_objs, H, W]
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((num_objs,), dtype=torch.int64),
        }
        return img, target
```

### 4.2 Mask 转 Bounding Box

```python
from torchvision.ops import masks_to_boxes

boxes = masks_to_boxes(masks)
# masks shape: [N, H, W], bool/float
# boxes shape: [N, 4], float32
```

`masks_to_boxes` 对每个mask找到非零像素的最小包围框，返回 `(x1, y1, x2, y2)` 格式。

### 4.3 自定义 Collate 函数

由于每张图的标注目标数量不同（变长），需要自定义batch拼接：

```python
def collate_fn(batch):
    # batch = [(img1, target1), (img2, target2), ...]
    # 返回元组而不是拼接的tensor
    return tuple(zip(*batch))
    # → ([img1, img2, ...], [target1, target2, ...])
```

### 4.4 DataLoader

```python
train_loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,        # 必须指定！
    num_workers=0,                # 检测任务通常设为0
)
```

---

## 5. 模型适配

### 5.1 加载预训练 Mask R-CNN

```python
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)

model = maskrcnn_resnet50_fpn(
    weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
    weights_backbone=None,   # backbone已有COCO预训练，不额外加载
)
```

### 5.2 替换预测头

```python
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

NUM_CLASSES = 2   # 背景 + 行人

# 替换 Box Predictor
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

# 替换 Mask Predictor
in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(
    in_channels, hidden_layer, NUM_CLASSES
)
```

> 💡 `NUM_CLASSES` 必须包含**背景类**。Penn-Fudan 有 1 个前景类（person），所以 `NUM_CLASSES = 2`。

### 5.3 冻结 Backbone

```python
# 冻结 backbone 参数（推荐，尤其在小数据集上）
for param in model.backbone.parameters():
    param.requires_grad = False

# 优化器只传可训练参数
optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005,
)
```

**冻结策略对比**：

| 策略 | 可训练参数 | 训练时间 | 效果（小数据） |
|------|-----------|----------|---------------|
| 全冻结backbone | ~2M | 快 | 较好 |
| 部分解冻 | ~8M | 中等 | 稍微更好 |
| 全部训练 | ~30M+ | 慢 | 可能过拟合 |

---

## 6. 训练循环

### 6.1 损失函数

Mask R-CNN 在训练模式返回损失字典：

```python
model.train()
loss_dict = model(images, targets)
# loss_dict = {
#     "loss_classifier": ...,
#     "loss_box_reg": ...,
#     "loss_objectness": ...,
#     "loss_rpn_box_reg": ...,
#     "loss_mask": ...,
# }
```

**四个损失分量**：

| 损失项 | 含义 | 来源 |
|--------|------|------|
| `loss_objectness` | RPN前景/背景分类 | RPN |
| `loss_rpn_box_reg` | RPN bbox回归 | RPN |
| `loss_classifier` | 最终类别分类 | Box Head |
| `loss_box_reg` | 最终bbox回归 | Box Head |
| `loss_mask` | 逐像素mask二值交叉熵 | Mask Head |

总损失是各项的**加和**：

```python
losses = sum(loss for loss in loss_dict.values())
```

### 6.2 完整训练代码

```python
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, targets) in enumerate(data_loader):
        # 移动到设备（因为images和targets是列表，需逐个处理）
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 前向：返回loss_dict
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        running_loss += losses.item()

        if batch_idx % 5 == 0:
            loss_items = ", ".join(
                f"{k}: {v.item():.4f}" for k, v in loss_dict.items()
            )
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}: {loss_items}")

    return running_loss / len(data_loader)
```

> 🟡 检测模型中梯度裁剪 (`clip_grad_norm_`) 是常见做法，防止RPN区域导致的梯度爆炸。

---

## 7. 推理与可视化

### 7.1 推理模式

```python
model.eval()
with torch.no_grad():
    prediction = model([image])[0]
    # prediction = {
    #     "boxes": [M, 4],
    #     "labels": [M],
    #     "scores": [M],       # 置信度 [0, 1]
    #     "masks": [M, 1, H, W],  # sigmoid输出
    # }
```

推理模式与训练模式的关键区别：
- **训练时**：输入 `(images, targets)`，返回 `loss_dict`
- **推理时**：输入 `[images]`，返回 `prediction_dict`

### 7.2 分数阈值过滤

```python
SCORE_THRESHOLD = 0.5

keep = prediction["scores"] >= SCORE_THRESHOLD
filtered_boxes = prediction["boxes"][keep]
filtered_scores = prediction["scores"][keep]
filtered_masks = prediction["masks"][keep]

print(f"Boxes before threshold: {len(prediction['boxes'])}")
print(f"Boxes after threshold {SCORE_THRESHOLD}: {len(filtered_boxes)}")
```

**阈值选择的影响**：

| 阈值 | 效果 | 适用场景 |
|------|------|----------|
| 0.25 | 保留多，召回率高但假阳性多 | 不能漏检（安全关键） |
| 0.50 | 平衡 | 一般使用 |
| 0.75 | 保留少，精确率高但可能漏检 | 不能误报 |

> 🟡 考试中常见考点：NMS（非极大值抑制）已在 Mask R-CNN 内部自动执行，用户不需要手动调用。

### 7.3 绘制边界框

```python
from torchvision.utils import draw_bounding_boxes

# 分数标签
labels = [f"{s:.2f}" for s in filtered_scores]

# 绘制红色框
result = draw_bounding_boxes(
    (image * 255).to(torch.uint8),
    filtered_boxes,
    labels=labels,
    colors="red",
    width=3,
)

# 保存
from torchvision.io import write_png
write_png(result, "prediction.png")
```

### 7.4 绘制分割 Mask

```python
from torchvision.utils import draw_segmentation_masks

# masks shape: [N, 1, H, W]  需转为 [N, H, W] bool
binary_masks = filtered_masks[:, 0] > 0.5

result = draw_segmentation_masks(
    (image * 255).to(torch.uint8),
    binary_masks,
    alpha=0.45,         # 透明度
)

write_png(result, "segmentation.png")
```

---

## 8. 完整工作流总结

```text
Step 1: 加载Penn-Fudan数据集，将instance mask转为二值mask数组
    │
Step 2: 构建target字典（boxes, labels, masks, area, iscrowd, image_id）
    │  - masks_to_boxes() 从mask计算bbox
    │  - labels 全部为 1（person）
    │
Step 3: 分割训练/验证集（Subset），创建DataLoader（使用collate_fn）
    │
Step 4: 加载Mask R-CNN预训练模型，替换 box_predictor 和 mask_predictor
    │  - NUM_CLASSES = 背景(0) + 行人(1) = 2
    │
Step 5: 可选冻结backbone以加速训练
    │
Step 6: 训练几个epoch，监控各loss分量
    │
Step 7: 推理 → 分数阈值过滤 → 可视化bbox和mask
    │
Step 8: 保存 model.state_dict()
```

---

## 9. 考试要点 🎯

| 考点 | 关键知识 |
|------|----------|
| Target字典 | 6个字段；boxes=(x1,y1,x2,y2)；labels从1开始；masks=[N,H,W] |
| Dataset | `__getitem__` 返回 `(image, target)`；mask的unique→二值转换 |
| collate_fn | 检测任务必须自定义collate，返回元组而非拼接tensor |
| Mask R-CNN结构 | Backbone→RPN→RoIAlign→{Box,Class,Mask} Heads |
| 模型训练/推理差异 | 训练返回loss_dict；推理返回prediction_dict（含scores） |
| 损失分量 | loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg + loss_mask |
| NUM_CLASSES | 必须包含背景类（0=background）|
| 预测头替换 | `FastRCNNPredictor` 替换box；`MaskRCNNPredictor` 替换mask |
| 分数阈值 | `scores >= threshold` 过滤；阈值升高→框减少 |
| 可视化 | `draw_bounding_boxes` + `draw_segmentation_masks` |
| Grad Clip | `clip_grad_norm_` 防止RPN梯度爆炸 |

### 常见错误 ⚠️

1. **忘记 `collate_fn`**：检测任务DataLoader不指定collate_fn会报错（tensor形状不一致）
2. **NUM_CLASSES 不含背景**：`NUM_CLASSES = 前景类别数 + 1`
3. **target中boxes用 `(cx, cy, w, h)` 格式**：PyTorch检测模型要求 `(x1, y1, x2, y2)`
4. **训练和推理调用方式混淆**：训练传 `model(images, targets)`，推理传 `model([images])`
5. **评估时未过滤低分预测**：不做score threshold会有大量假阳性框
6. **mask预测的sigmoid阈值**：`masks[:, 0] > 0.5` 将sigmoid输出二值化

---

## 10. 关键代码速查

### 构建Target字典

```python
obj_ids = torch.unique(mask_tensor)
obj_ids = obj_ids[1:]                        # 去除背景0
masks = mask_tensor == obj_ids[:, None, None]  # [N, H, W]
boxes = masks_to_boxes(masks)                # [N, 4]
labels = torch.ones((len(obj_ids),), dtype=torch.int64)
area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
iscrowd = torch.zeros((len(obj_ids),), dtype=torch.int64)

target = {
    "boxes": boxes,
    "labels": labels,
    "masks": masks,
    "image_id": torch.tensor([idx]),
    "area": area,
    "iscrowd": iscrowd,
}
```

### 替换预测头

```python
# Box predictor
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

# Mask predictor
in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, 256, NUM_CLASSES)
```

### 冻结 Backbone

```python
for param in model.backbone.parameters():
    param.requires_grad = False
```

### 分数过滤 + 可视化

```python
keep = prediction["scores"] >= SCORE_THRESHOLD
boxes = prediction["boxes"][keep]
scores = prediction["scores"][keep]
masks = prediction["masks"][keep]

# 边界框
pred_image = draw_bounding_boxes(
    (img * 255).to(torch.uint8), boxes,
    labels=[f"{s:.2f}" for s in scores], colors="red", width=3,
)

# 分割mask
seg_image = draw_segmentation_masks(
    (img * 255).to(torch.uint8),
    masks[:, 0] > 0.5, alpha=0.45,
)
```
