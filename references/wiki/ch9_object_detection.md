# Ch9: Object Detection (目标检测)

> 🟢 来自资料 — 基于课程讲义 `9_Object_Detection.pdf` 及经典论文 (R-CNN, Fast R-CNN, Faster R-CNN, YOLO, SSD, RetinaNet, DETR)

---

## 1. Task Definition (任务定义)

目标检测 (Object Detection) 联合解决两个子问题：

- **定位 (Localization)**：预测边界框 (Bounding Box) $\mathbf{b} = (x, y, w, h)$ —— 即中心坐标、宽度和高度（或角点坐标 $(x_1, y_1, x_2, y_2)$）。
- **分类 (Classification)**：分配类别标签 (Category Label) $c \in \{1, \dots, C\}$（另外通常包含一个"背景"类，用以拒绝非物体区域）。

对于输入图像 $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$，检测器输出一组预测结果：

$$\{(\mathbf{b}_i, c_i, s_i)\}_{i=1}^{K}$$

其中 $s_i$ 是置信度得分 (Confidence Score)，且预测数量 $K$ 随图像不同而变化。这使得目标检测在本质上成为一个**集合预测 (Set Prediction)** 问题 —— 物体的数量与身份类别在先验上都是未知的。

> 🟡 AI补充: 目标检测是分类+定位的联合任务，输出是不定长的集合，这与分类（固定长度输出）有本质区别。

---

## 2. Evaluation Metrics (评估指标)

### 2.1 IoU (Intersection over Union / 交并比)

交并比 (IoU) 用于衡量预测边界框 (Predicted Box) $B_p$ 与真实边界框 (Ground-truth Box) $B_{gt}$ 之间的重叠程度：

$$\text{IoU}(B_p, B_{gt}) = \frac{|B_p \cap B_{gt}|}{|B_p \cup B_{gt}|} \in [0, 1]$$

如果 $\text{IoU} \geq \tau$（通常设 $\tau = 0.5$），则该预测被视为一个**真正例 (True Positive, TP)**；否则，被视为一个**假正例 (False Positive, FP)**。

### 2.2 Precision-Recall Curve (精确率-召回率曲线 / PR 曲线)

对于给定的类别，将所有预测结果按置信度得分降序排列。在每个置信度阈值下：

- **精确率 (Precision)** = $\frac{\text{TP}}{\text{TP} + \text{FP}}$ —— 预测为物体的检测中正确的比例。
- **召回率 (Recall)** = $\frac{\text{TP}}{\text{TP} + \text{FN}}$ —— 所有真实物体中被成功检测到的比例。

精确率-召回率曲线 (PR Curve) 描绘了精确率与召回率之间的关系。如果一个检测器的 PR 曲线完全在另一个检测器的曲线之上，则说明该检测器更优。

### 2.3 mAP (mean Average Precision / 平均精度均值)

**单类平均精度 (Average Precision, AP)**：PR 曲线下的面积，在实际中通常通过 11 点插值法或全点平均法（曲线下面积 AUC）进行近似。

**平均精度均值 (mAP)** 是所有 $C$ 个类别 AP 的平均值：

$$\text{mAP} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c$$

常见变体：
- **mAP@0.5**：IoU 阈值设为 0.5。
- **mAP@[0.5:0.95]**（MS COCO 评估标准）：IoU 阈值从 0.5 到 0.95，以步长 0.05 递增并计算所有 mAP 的平均值。

> 🟢 来自资料 — mAP 是目标检测最核心的聚合指标，COCO mAP@[0.5:0.95] 是当前事实标准。

---

## 3. Two-Stage Detectors (双阶段检测器)

双阶段检测器 (Two-stage Detectors) 首先生成**区域建议 (Region Proposals)**，然后对每个建议区域进行分类和边界框微调。

### 3.1 R-CNN (2014)

**处理流程 (Pipeline)：**

1. **选择性搜索 (Selective Search)**：从输入图像中生成约 2000 个与类别无关的区域建议 (Region Proposals)。
2. **区域变形 (Warp)**：将每个建议区域变形为固定大小（例如 $227 \times 227$）以作为 CNN 的输入。
3. **CNN 特征提取 (CNN Feature Extraction)**：使用 AlexNet 架构的 CNN 提取每个变形区域的 4096 维特征向量。
4. **SVM 分类 (SVM Classification)**：使用一组特定于类别的线性支持向量机 (Linear SVMs) 对每个区域进行评分。
5. **边界框回归 (Bounding Box Regression)**：使用线性回归器精细调整区域的坐标（特定于类别）。

**损失函数 (Losses)：**

- 分类：使用 SVM 铰链损失 (Hinge Loss)。
- 定位：使用边界框偏移量 $t^i = (t_x^i, t_y^i, t_w^i, t_h^i)$ 的 L2 回归损失。

**坐标转换关系 (边界框回归目标)：**

$$t_x = \frac{G_x - P_x}{P_w}, \quad t_y = \frac{G_y - P_y}{P_h}$$
$$t_w = \log\frac{G_w}{P_w}, \quad t_h = \log\frac{G_h}{P_h}$$

**局限性 (Limitations)：**

- 训练步骤多阶段（CNN → SVM → 回归器），无法实现端到端训练 (End-to-end Training)。
- 每张图像需要约 2000 次前向传播 —— **极其缓慢**（测试时每张图约需 47 秒）。
- 需要将提取出的特征写入磁盘以训练 SVM 和回归器，占用极多存储。
- 图像变形 (Warping) 会导致原图的长宽比失真。

> 🟢 来自资料 — R-CNN 开创了两阶段检测范式，但速度是致命短板。

### 3.2 Fast R-CNN (2015)

**核心创新：感兴趣区域池化 (RoI Pooling)** —— 在所有建议区域之间共享卷积层计算。

**处理流程 (Pipeline)：**

1. **整图 CNN 前向传播 (Whole-image CNN)**：将整张图像输入 CNN 中运行一次，提取出特征图 (Feature Map)。
2. **RoI 投影 (RoI Projection)**：将（由选择性搜索生成的）每个候选区域投影到特征图上。
3. **RoI 池化 (RoI Pooling)**：将投影后的每个 RoI 划分为固定的网格（例如 $7 \times 7$），并在每个网格块（Bin）内应用最大值池化 (Max Pooling) —— 从而对任意大小的输入 RoI 都得到相同尺寸的固定大小输出。
4. **全连接层与双输出分支 (FC layers + two sibling output layers)**：分类分支（在 $C+1$ 个类别上计算 Softmax）与边界框回归分支（输出特定于类别的边界框偏移量）。

**多任务损失 (Multi-task Loss)（联合训练）：**

$$L(p, u, t^u, v) = L_{\text{cls}}(p, u) + \lambda [u \geq 1] \cdot L_{\text{loc}}(t^u, v)$$

- $L_{\text{cls}}(p, u) = -\log p_u$（真实类别 $u$ 的对数损失）。
- $L_{\text{loc}}(t^u, v) = \sum_{i \in \{x,y,w,h\}} \text{smooth}_{L_1}(t_i^u - v_i)$

其中平滑 L1 损失 (Smooth L1 Loss) 的定义为：

$$\text{smooth}_{L_1}(x) = \begin{cases} 0.5x^2 & |x| < 1 \\ |x| - 0.5 & \text{其他情况} \end{cases}$$

- 指示器 $[u \geq 1]$：对于背景样本（$u=0$），定位损失为 0。
- 参数 $\lambda$ 用于平衡分类损失和定位损失（通常设为 $\lambda = 1$）。

**相较于 R-CNN 的优势：**

- 单阶段训练（除了生成候选区域外，基本实现端到端训练）。
- 不需要将中间特征缓存在磁盘上。
- 训练速度提高约 9 倍；测试推理速度提高约 213 倍。

> 🟢 来自资料 — Fast R-CNN 的核心贡献是 RoI Pooling 让 CNN 特征可以共享，大幅加速训练 and 推理。

### 3.3 Faster R-CNN (2015)

**核心创新：区域建议网络 (Region Proposal Network, RPN)** —— 用一个可学习的神经网络替代慢速的选择性搜索 (Selective Search)。

**系统架构：**

1. **骨干网络 (Backbone CNN)**：提取特征图（例如使用 VGG-16 或 ResNet）。
2. **区域建议网络 (Region Proposal Network, RPN)**：在特征图上进行滑动的轻量全卷积网络。
3. **RoI 池化与检测头 (RoI Pooling + detection head)**：与 Fast R-CNN 相同。

**锚框 (Anchor Boxes)**：在特征图的每个滑动窗口位置 $(i, j)$（步长 stride $s=16$）处，RPN 会考虑 $k$ 个具有不同尺度 (Scales) 和长宽比 (Aspect Ratios) 的锚框（参考框）。典型取值通常为：

- 3 种尺度：$\{128^2, 256^2, 512^2\}$
- 3 种长宽比：$\{1:1, 1:2, 2:1\}$
- 每个位置共 $k = 9$ 个锚框。

对于大小为 $H \times W$ 的特征图，总共生成 $H \times W \times k$ 个锚框。

**RPN 结构：**

- 使用一个 $3 \times 3$ 卷积滑动窗口 —— 在每个位置提取 256 维（或 512 维）特征。
- 连接两个并行的 $1 \times 1$ 卷积层：
  - **分类层 (cls layer)**：输出 $2k$ 个得分（指示物体性 Objectness：是否为物体）。
  - **回归层 (reg layer)**：输出 $4k$ 个坐标偏移量（每个锚框的边界框调整量）。

**RPN 损失函数：**

$$L(\{p_i\}, \{t_i\}) = \frac{1}{N_{\text{cls}}} \sum_i L_{\text{cls}}(p_i, p_i^*) + \lambda \frac{1}{N_{\text{reg}}} \sum_i p_i^* \cdot L_{\text{reg}}(t_i, t_i^*)$$

- $p_i$：预测的锚框 $i$ 的物体性概率。
- $p_i^* \in \{0, 1\}$：真实标签（如果锚框与任一真实边界框的 IoU > 0.7 则设为 1；如果与所有真实边界框的 IoU < 0.3 则设为 0）。
- $t_i$：预测的边界框偏移量；$t_i^*$：真实的边界框偏移量（相对于锚框）。
- $L_{\text{cls}}$：二分类交叉熵损失 (Binary Cross-Entropy Loss)。
- $L_{\text{reg}}$：平滑 L1 损失 (Smooth L1 Loss)。
- 乘数 $p_i^*$：代表仅正样本（即含有物体的锚框）对边界框回归损失有贡献。
- $N_{\text{cls}}$：分类小批量大小 (Mini-batch Size, 约为 256)；$N_{\text{reg}}$：锚框位置的总数 (约为 2400)。
- $\lambda \approx 10$ 用于平衡两项损失的权重。

**训练方式**：四步交替训练 (4-step Alternating Training) 或近似联合训练 (Approximate Joint Training，此方式更常用)。

> 🟢 来自资料 — Faster R-CNN 的 RPN + anchor 机制是里程碑式的贡献，首次实现了纯神经网络端到端的目标检测。

### 3.4 FPN (Feature Pyramid Network / 特征金字塔网络, 2017)

**研究痛点**：目标物体的大小千差万别。由于高层特征图尺度较小（例如步长为 32），对于小尺寸物体的检测精度极差。

**解决方案**：构建具有**自上而下路径和横向连接 (Top-down Pathway and Lateral Connections)** 的特征金字塔。

**金字塔架构：**

1. **自下而上路径 (Bottom-up Pathway)**：标准的 CNN 前向传播，产生不同分辨率的特征图组合 $\{C_2, C_3, C_4, C_5\}$，对应的特征图步长分别为 $\{4, 8, 16, 32\}$（以 ResNet 为例）。
2. **自上而下路径 (Top-down Pathway)**：从顶层的 $C_5$ 开始，先进行 2 倍上采样 (Upsample)，再与侧边投影分支相结合：
   $$P_i = \text{Conv}_{3\times3}(\text{Conv}_{1\times1}(C_i) + \text{Upsample}(P_{i+1}))$$
3. **RPN 与检测头 (RPN + Detection Heads)**：分别在特征金字塔的每一层 $\{P_2, \dots, P_6\}$ 上独立运行并进行预测。

**锚框分配策略**：根据其面积大小将不同尺寸的锚框分配到对应的金字塔特征层上：
$$k = k_0 + \log_2(\sqrt{w h} / 224)$$

> 🟢 来自资料 — FPN 以极小计算开销解决了多尺度检测问题，成为几乎所有现代检测器的基础组件。

---

## 4. One-Stage Detectors (单阶段检测器)

单阶段检测器 (One-stage Detectors) 直接从图像像素中预测类别得分和边界框，省去了显式的生成候选区域步骤。

### 4.1 YOLO (You Only Look Once, 2016)

**核心思想**：将目标检测重构为一个单一的回归问题，直接从输入图像的像素值回归出边界框和类别概率。

**YOLOv1 架构：**

1. 将输入图像划分为 $S \times S$ 的网格（例如 $7 \times 7$）。
2. 每个网格单元预测：
   - $B$ 个边界框参数 $(x, y, w, h, \text{confidence})$ —— 每个单元预测最多 2 个边界框。
   - $C$ 个类别条件概率 $\text{Pr}(\text{Class}_i | \text{Object})$。
3. 置信度计算公式：Confidence = $\text{Pr}(\text{Object}) \times \text{IoU}_{\text{pred}}^{\text{truth}}$

**输出张量**：$S \times S \times (B \times 5 + C)$ 维度的张量。在 PASCAL VOC 上为 $7 \times 7 \times 30$ 维。

**YOLOv1 损失函数（加权平方误差和）：**

$$L = \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2\right]$$
$$+ \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2\right]$$
$$+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2$$
$$+ \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{noobj}} (C_i - \hat{C}_i)^2$$
$$+ \sum_{i=0}^{S^2} \mathbb{1}_{i}^{\text{obj}} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2$$

其中，$\lambda_{\text{coord}} = 5$（加大坐标预测的惩罚力度），$\lambda_{\text{noobj}} = 0.5$（降低不含物体区域的损失权重）。对边界框的宽和高采用平方根形式，以补偿欧氏距离损失在不同目标尺寸下的敏感度差异。

**YOLOv1 的局限性：**

- 每个网格单元最多只能预测 2 个物体，限制了其对密集小目标的检测能力。
- 没有使用锚框，完全从头学习边界框的先验。
- $7 \times 7$ 的粗糙网格在一定程度上限制了空间预测的精度。

**YOLO 架构演进：**
| 版本 | 主要改进点 |
|---------|-----------------|
| YOLOv2 (YOLO9000) | 引入锚框 (Anchor Boxes)、批归一化 (Batch Norm)、多尺度训练 (Multi-scale Training)、采用 Darknet-19 骨干网络 |
| YOLOv3 | 金字塔式的多尺度检测结构、Darknet-53 骨干网络、每类采用独立的二分类交叉熵 (Binary CE) 以支持多标签分类 |
| YOLOv4 | CSPDarknet53、PANet、Mosaic 数据增强、CIoU 损失等先进技巧 (Bag of Freebies/Specials) |
| YOLOv5 | 纯 PyTorch 实现、自动锚框机制 (Auto-anchor)、极其丰富的数据增强 |
| YOLOv8 / v9 / v10 | 无锚框设计 (Anchor-free)、解耦预测头 (Decoupled Heads)、先进的标签分配策略 |

> 🟢 来自资料 — YOLO 系列以速度见长，不断融合学术界最佳实践，在实时检测领域占据主导地位。

### 4.2 SSD (Single Shot MultiBox Detector / 单阶段多框检测器, 2016)

**核心思想：**

- 利用 CNN 中不同尺度（即不同网络层）的**多尺度特征图 (Multiple Feature Maps)** 来进行检测。
- 每一个特征图的网格单元都会预测一系列相对于**默认框 (Default Boxes)**（即预先设定的不同长宽比的锚框）的坐标偏移量和类别得分。
- 不使用单独的候选区域生成步骤，直接输出类别置信度和边界框偏移。

**默认框尺寸设定**：在第 $k$ 层特征图上，默认框的基准尺寸计算如下：
$$s_k = s_{\text{min}} + \frac{s_{\text{max}} - s_{\text{min}}}{m - 1}(k - 1), \quad k \in [1, m]$$

结合设定的长宽比 $\{1, 2, 3, \frac{1}{2}, \frac{1}{3}\}$，默认框的宽为 $w = s_k\sqrt{a_r}$，高为 $h = s_k / \sqrt{a_r}$。

**损失函数**：与 Faster R-CNN 的多任务损失类似，包含分类得分损失 (Softmax) 和定位回归损失 (Smooth L1)，并在训练中使用困难负样本挖掘 (Hard Negative Mining)，使正负样本比例保持在约 1:3值。

> 🟢 来自资料 — SSD 证明了多尺度特征图 + default boxes 可以高效地进行单阶段检测，速度优于 Faster R-CNN。

### 4.3 RetinaNet (2017) —— Focal Loss (焦点损失)

**研究痛点：单阶段检测器面临的极端类别不平衡 (Class Imbalance) 问题**。在一张普通的图片中，虽然包含近 10 万个候选位置，但往往只有 10~100 个真实的物体。在训练中，极其大量的"简单负样本"（即不含物体的背景区域）产生的损失会彻底淹没真实目标的损失。

**Focal Loss (焦点损失)** 重构了标准的交叉熵损失，极大地降低了简单样本的损失权重：

标准交叉熵 (Cross Entropy) 损失：$\text{CE}(p_t) = -\log(p_t)$，其中
$$p_t = \begin{cases} p & \text{若 } y = 1 \\ 1 - p & \text{其他情况} \end{cases}$$

**Focal Loss 定义：**

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- $(1 - p_t)^\gamma$：**调制因子 (Modulating Factor)**。当样本分类正确（$p_t \to 1$）时，调制因子趋近于 0，因此这部分"简单样本"的损失被大大衰减；当样本分类错误或难以置信（$p_t \ll 1$）时，调制因子接近 1，损失基本保持。
- $\alpha_t$：类别平衡权重（用于调节正负样本的比例，类似于平衡交叉熵中的 $\alpha$）。
- $\gamma \geq 0$：聚焦参数 (Focusing Parameter)。$\gamma = 0$ 时即为标准交叉熵损失，在实际应用中 $\gamma = 2$ 效果最佳。

**网络架构**：RetinaNet 采用 FPN 骨干网络加上两个并行的预测子网络（分别用于分类和边界框回归，它们均是连接在每一层特征金字塔之上的小型全卷积网络）。

> 🟢 来自资料 — Focal Loss 解决了单阶段检测器训练时的极端正负样本不平衡问题，使 RetinaNet 在精度上首次匹敌两阶段方法。

---

## 5. Anchor-Free Methods (无锚框目标检测)

摒弃复杂的预设锚框 (Anchor Boxes) 设计，简化了算法超参数。

### 5.1 CornerNet (2018)

将检测目标建模为**一对关键点**（即左上角角点和右下角角点）。通过预测热力图 (Heatmap) 检测这些角点，并输出一个嵌入向量 (Embedding Vector) 来匹配属于同一个物体的两个对应角点。

**核心组件：**
- **角点池化 (Corner Pooling)**：一种新型的池化层，通过沿特征图整行/整列进行最大值计算来帮助网络更好地定位边界处的角点。
- **嵌入损失 (Embedding Loss)**：拉近属于同一个物体的角点嵌入距离，推开不同物体的角点距离，包含拉近损失 $L_{\text{pull}}$ 和推开损失 $L_{\text{push}}$。

### 5.2 CenterNet (2019)

将物体建模为**中心点**。通过热力图定位物体的中心点，并直接在中心点位置回归目标的宽度和高度。

**处理流程 (Pipeline)：**
1. 预测物体中心点的热力图（关键点估计任务）。
2. 在预测的中心点位置，回归出物体的宽和高 $(w, h)$，并估计局部的坐标偏移量以纠正由于下采样带来的亚像素误差。
3. **完全省去了 NMS (非极大值抑制) 操作** —— 只需要从热力图中提取局部峰值（Local Maxima）即可直接得出候选框。

其逻辑极为简洁高效，非常适合应用于关键点估计和实时检测任务。

### 5.3 FCOS (Fully Convolutional One-Stage, 2019)

**核心思想**：在特征图上的每个像素位置，直接预测一个四维向量 $(l, t, r, b)$，其代表当前位置到目标物体四个边界的距离。

**核心创新：**
- **中心度分支 (Center-ness Branch)**：预测一个介于 0 和 1 之间的"中心度"得分，用于抑制远离物体中心的低质量低置信度候选框：
  $$\text{centerness} = \sqrt{\frac{\min(l, r)}{\max(l, r)} \times \frac{\min(t, b)}{\max(t, b)}}$$
- **基于 FPN 的多尺度预测**：将不同尺寸范围的物体分配给特征金字塔的不同特征层进行回归，巧妙解决了特征位置上由于边界框重叠而产生的二义性问题。

> 🟢 来自资料 — Anchor-free 方法去成了 anchor 设计的超参数负担，FCOS 和 CenterNet 证明了 anchor-free 可以达到甚至超越 anchor-based 方法的性能。

---

## 6. DETR (DEtection TRansformer, 2020)

**首个端到端基于 Transformer 的目标检测器**，摒弃了传统目标检测中的大量手工设计组件（如 NMS、锚框生成等）。

**网络架构：**

1. **CNN 骨干网络 (CNN Backbone)**：用于提取初始的高维特征图。
2. **Transformer 编码器 (Transformer Encoder)**：将特征图拉平并加入位置编码 (Positional Encoding) 作为输入，使用自注意力机制 (Self-Attention) 建模全局上下文信息。
3. **Transformer 解码器 (Transformer Decoder)**：接受 $N$ 个可学习的**物体查询 (Object Queries)**（位置编码）作为输入，在编码器输出的键值对上进行交叉注意力计算，每个查询预测一个检测物体（或预测为“无物体”的空类别）。
4. **预测输出头 (Prediction Heads)**：使用简单的前馈网络 (FFN) 将解码器的输出映射为类别概率和对应的边界框。

**二分匹配损失 (Bipartite Matching Loss)**：DETR 预测一组成员固定为 $N$ 的预测集，而真实物体的数量为 $M$ 且满足 $M < N$。在训练中，使用匈牙利算法 (Hungarian Algorithm) 寻找预测集与真实集之间的一对一最优二分匹配关系：

$$\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^{N} \mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)})$$

$$\mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)}) = -\mathbb{1}_{\{c_i \neq \varnothing\}} \hat{p}_{\sigma(i)}(c_i) + \mathbb{1}_{\{c_i \neq \varnothing\}} \lambda_{\text{box}} \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\sigma(i)})$$

在匹配完成后，仅针对成功匹配的一对一正负样本对计算 Hungarian 损失，包括分类损失（交叉熵）与边界框损失（L1 损失和 GIoU 损失）。

**关键特性：**
- **无需 NMS**：因为每个物体查询 (Object Query) 保证最多只匹配一个目标。
- **无需设定复杂的锚框参数**。
- 收敛速度极其缓慢：相比传统检测器的 12~36 轮 Epoch，DETR 通常需要训练多达 500 轮 Epoch。后续的 Deformable DETR（可变形 DETR）利用可变形注意力机制，大幅缓解了收敛慢的缺点。

> 🟢 来自资料 — DETR 将目标检测重构为集合预测问题，用 Transformer + 匈牙利匹配替代了大量手工设计组件。

---

## 7. NMS and Soft-NMS (非极大值抑制与软抑制)

### 7.1 Non-Maximum Suppression (NMS / 非极大值抑制)

目标检测器往往会对同一个物体输出大量重叠的预测框。NMS 的作用就是滤除这些冗余的边界框：

**标准 NMS 算法流程：**
1. 将所有输出框按照置信度得分降序排序。
2. 选取当前得分最高的边界框 $B$，将其加入最终的检测结果集中。
3. 移除剩余边界框中与 $B$ 之间的交并比满足 $\text{IoU}(B, B_i) \geq \tau$（通常 $\tau = 0.5$）的所有冗余框。
4. 重复以上步骤，直至没有剩余边界框。

**缺点**：硬阈值设计。在密集场景下（如人流密集区），两个真实相近的物体很有可能因为交并比大而被 NMS 错误地过滤掉。

### 7.2 Soft-NMS (软非极大值抑制)

不同于 NMS 直接粗暴地滤除重叠框，Soft-NMS 选择根据它们与高分框的重叠度来**衰减其置信度得分**：

线性得分衰减公式：

$$s_i = \begin{cases} s_i & \text{若 } \text{IoU}(M, b_i) < N_t \\ s_i \cdot (1 - \text{IoU}(M, b_i)) & \text{若 } \text{IoU}(M, b_i) \geq N_t \end{cases}$$

或高斯衰减形式：
$$s_i = s_i \cdot e^{-\frac{\text{IoU}(M, b_i)^2}{\sigma}}$$

Soft-NMS 能够有效保留密集分布的相邻物体，从而在拥挤场景下显著提升检测器的召回率。

> 🟢 来自资料 — Soft-NMS 通过分数衰减替代硬删除，在密集场景下显著改善了召回率。

---

## 8. Comparison Table of Detection Architectures (目标检测架构对比表)

| 算法名称 | 算法类型 | 候选框机制 | 骨干网络 | 推理速度 | 核心创新 |
|--------|------|-----------|----------|-------|---------------|
| R-CNN (2014) | 双阶段 | 选择性搜索 | AlexNet | ~47秒/张 | 卷积神经网络与区域提取相结合 |
| Fast R-CNN (2015) | 双阶段 | 选择性搜索 | VGG-16 | ~2秒/张 | RoI池化、多任务联合损失函数 |
| Faster R-CNN (2015) | 双阶段 | 区域建议网络 (RPN) | VGG-16/ResNet | ~0.2秒/张 | RPN网络、基于锚框的设计 |
| FPN (2017) | 双阶段 | 区域建议网络 (RPN) | ResNet-FPN | ~0.2秒/张 | 多尺度特征金字塔网络 |
| YOLOv1 (2016) | 单阶段 | 无（网格化回归） | 订制网络 | ~45 FPS | 统一的全局回归问题建模 |
| YOLOv3 (2018) | 单阶段 | 无（网格+锚框） | Darknet-53 | ~30-60 FPS | 多尺度预测、引入多标签独立二分类交叉熵 |
| SSD (2016) | 单阶段 | 默认框 (Default Boxes) | VGG-16 | ~46-59 FPS | 多尺度卷积特征图上的直接预测 |
| RetinaNet (2017) | 单阶段 | 锚框 | ResNet-FPN | ~5-12 FPS | Focal Loss 解决正负样本极度不平衡 |
| CornerNet (2018) | 无锚框 | 无（角点对匹配） | Hourglass | ~4 FPS | 基于角点检测的热力图与嵌入向量 |
| CenterNet (2019) | 无锚框 | 无（中心点估计） | DLA/Hourglass | ~28-45 FPS | 基于中心点定位及尺寸直接回归（免NMS） |
| FCOS (2019) | 无锚框 | 无 | ResNet-FPN | ~15-22 FPS | 每个空间位置预测4D边界框相对距离 |
| DETR (2020) | Transformer 架构 | 物体查询 (Object Queries) | ResNet | ~28 FPS | 匈牙利二分匹配损失、完全不需要 NMS |

---

## 9. Practice Problems (练习题与详解)

### Problem 1: IoU Calculation (IoU 计算)
有两个边界框：预测框 $B_1 = [10, 10, 50, 60]$ (格式为 $[x_1, y_1, x_2, y_2]$) 以及真实框 $B_2 = [30, 20, 70, 80]$。请计算它们之间的交并比 (IoU)。

**Solution (解析):**
- 相交区域 (Intersection) 的坐标：$[\max(10, 30), \max(10, 20), \min(50, 70), \min(60, 80)] = [30, 20, 50, 60]$
- 相交面积 (Intersection Area)：$(50-30) \times (60-20) = 20 \times 40 = 800$
- 边界框 $B_1$ 面积：$(50-10) \times (60-10) = 40 \times 50 = 2000$
- 边界框 $B_2$ 面积：$(70-30) \times (80-20) = 40 \times 60 = 2400$
- 相并面积 (Union Area)：$\text{Area}(B_1) + \text{Area}(B_2) - \text{Intersection Area} = 2000 + 2400 - 800 = 3600$
- 交并比 $\text{IoU} = \frac{800}{3600} \approx 0.222$

### Problem 2: Focal Loss Behavior (Focal Loss 行为分析)
设定超参数为 $\gamma = 2$ 且类别调节权重 $\alpha = 1$。请对比分析模型在预测概率为 $p_t = 0.9$ (分类正确的简单样本) 与 $p_t = 0.1$ (分类错误的困难样本) 时的 Focal Loss 损失大小。

**Solution (解析):**
- 分类正确样本（$p_t = 0.9$）：$\text{FL} = -(1-0.9)^2 \log(0.9) = -(0.01)(-0.105) \approx 0.00105$
- 分类错误样本（$p_t = 0.1$）：$\text{FL} = -(1-0.1)^2 \log(0.1) = -(0.81)(-2.303) \approx 1.865$
- 对比之下，分类正确的样本其 Focal Loss 被极度压低，仅为对应标准交叉熵损失（$-\log(0.9) \approx 0.105$）的 $\frac{1}{100}$ 左右。而困难样本的损失依然保持在较大状态（缩减为标准交叉熵的 0.81 倍），促使模型将训练焦点集中在难分类样本上。

### Problem 3: Anchor Assignment (锚框总数计算)
某 Faster R-CNN 框架在步长 (Stride) 为 16 的特征图上进行目标检测，设定了 3 种尺度 $\{128, 256, 512\}^2$ 及 3 种长宽比 $\{1:1, 1:2, 2:1\}$ 的锚框。
a) 请问特征图上的每个空间网格点处对应生成多少个锚框？
b) 如果输入特征图的尺寸大小为 $40 \times 60$，该网络总共生成多少个锚框？

**Solution (解析):**
- a) 每个空间位置处的锚框数量：$3 \text{ (尺度)} \times 3 \text{ (长宽比)} = 9$ 个。
- b) 锚框总数：$40 \times 60 \times 9 = 21600$ 个。

### Problem 4: RPN Loss Components (RPN 损失成分分析)
请详细解释在 RPN 网络或者目标检测多任务联合损失函数中指示器 $\mathbb{1}_{ij}^{\text{obj}}$（在 RPN 中通常表示为真实分类标签 $p_i^*$）所起到的具体作用。

**Solution (解析):**
指示器 $\mathbb{1}_{ij}^{\text{obj}}$（或 $p_i^*$）在损失函数中起到**门控函数 (Gating Function)** 的关键作用：只有当网格单元 $i$ 的第 $j$ 个锚框被分配为**正样本**（即它与任意一个真实目标边界框成功匹配）时，其取值才为 1，其他情况（负样本/背景）取值均为 0。这确保了：
- 仅有匹配成功的正样本才参与计算**定位回归损失 (Bounding Box Regression Loss)** —— 我们显然不可能让没有物体的背景锚框去向任意真实物体的坐标偏移靠拢。
- **分类损失 (Classification/Confidence Loss)** 则对所有正、负样本（即前景和背景区域）一并进行惩罚。
- 引入该门控机制避免了无物体背景锚框在回归坐标时产生杂乱的梯度更新，确保了训练过程的平稳收敛。

### Problem 5: DETR Matching (DETR 匈牙利匹配机制)
为什么 DETR 模型在计算损失前需要引入二分匹配 (Bipartite Matching)？如果去除该机制，训练过程会发生什么现象？

**Solution (解析):**
DETR 模型直接输出一组大小固定为 $N$ 的预测边界框集合，而输入的真实图像中往往仅包含 $M$ 个真实物体（其中 $M < N$，剩余未匹配的 $N-M$ 个槽位被分类标记为“无物体”）。**二分匹配**（即匈牙利算法）正是在这两个集合之间，建立起一一对应的最优匹配连接，以使总匹配代价最低。如果去除这一匹配机制：
- 多个物体查询 (Object Queries) 可能会全部被分配去回归同一幅真实目标物体。
- 回归损失 and 分类损失的计算对象将变得无法定义（哪个具体的预测槽位该去惩罚哪个真实的物体？）。
- 整个训练过程将极其容易坍塌到一个退化解中（例如所有的物体查询最终都预测出一个重合的相同候选框），因为标准的交叉熵损失在不加入二分匹配约束时，本身并不具备强迫“预测输出保持不重复、一对一映射”的物理先验。

这是由于目标检测本质是一个**集合预测问题 (Set Prediction Problem)**：整体损失不仅需要具有对预测排列顺序的对称不变性，也必须强制约束每一个真实目标仅有唯一的预测与之对应。

---

> 🟢 来自资料 — 以上内容综合自 Faster R-CNN、YOLO、SSD、RetinaNet 及 DETR 等经典论文，覆盖目标检测领域的核心技术演进。