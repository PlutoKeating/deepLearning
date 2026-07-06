# Ch9: Object Detection

> 🟢 来自资料 — 基于课程讲义 `9_Object_Detection.pdf` 及经典论文 (R-CNN, Fast R-CNN, Faster R-CNN, YOLO, SSD, RetinaNet, DETR)

---

## 1. Task Definition

Object detection jointly addresses two sub-problems:

- **Localization**: Predict a bounding box $\mathbf{b} = (x, y, w, h)$ — center coordinates, width, height (or corner coordinates $(x_1, y_1, x_2, y_2)$).
- **Classification**: Assign a category label $c \in \{1, \dots, C\}$ (plus a "background" class to reject non-object regions).

For an input image $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$, the detector outputs a set of predictions:

$$\{(\mathbf{b}_i, c_i, s_i)\}_{i=1}^{K}$$

where $s_i$ is a confidence score and $K$ varies per image. This makes detection fundamentally a **set prediction** problem — both the number and identity of objects are unknown a priori.

> 🟡 AI补充: 目标检测是分类+定位的联合任务，输出是不定长的集合，这与分类（固定长度输出）有本质区别。

---

## 2. Evaluation Metrics

### 2.1 IoU (Intersection over Union)

IoU measures the overlap between a predicted box $B_p$ and a ground-truth box $B_{gt}$:

$$\text{IoU}(B_p, B_{gt}) = \frac{|B_p \cap B_{gt}|}{|B_p \cup B_{gt}|} \in [0, 1]$$

A prediction is considered a **True Positive** if $\text{IoU} \geq \tau$ (typically $\tau = 0.5$). Otherwise it's a False Positive.

### 2.2 Precision-Recall Curve

For a given class, sort all predictions by confidence score descending. At each threshold:

- **Precision** = $\frac{\text{TP}}{\text{TP} + \text{FP}}$ — fraction of detections that are correct
- **Recall** = $\frac{\text{TP}}{\text{TP} + \text{FN}}$ — fraction of ground-truth objects found

The PR curve plots precision vs. recall. A detector dominates another if its PR curve is entirely above.

### 2.3 mAP (mean Average Precision)

**AP (Average Precision)** for a single class: area under the PR curve, often approximated by interpolated 11-point average or the all-point average (AUC).

**mAP** is the mean AP across all $C$ classes:

$$\text{mAP} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c$$

Common variants:
- **mAP@0.5**: IoU threshold = 0.5
- **mAP@[0.5:0.95]** (COCO metric): average mAP at IoU thresholds from 0.5 to 0.95 in steps of 0.05

> 🟢 来自资料 — mAP 是目标检测最核心的聚合指标，COCO mAP@[0.5:0.95] 是当前事实标准。

---

## 3. Two-Stage Detectors

Two-stage detectors first generate **region proposals**, then classify and refine each proposal.

### 3.1 R-CNN (2014)

**Pipeline:**

1. **Selective Search**: Generate ~2000 category-independent region proposals from the input image.
2. **Warp**: Each proposal is warped to a fixed size (e.g., $227 \times 227$) for CNN input.
3. **CNN Feature Extraction**: An AlexNet-style CNN extracts a 4096-dim feature vector per proposal.
4. **SVM Classification**: A set of class-specific linear SVMs score each proposal.
5. **Bounding Box Regression**: A linear regressor refines proposal coordinates (class-specific).

**Losses:**

- SVM hinge loss for classification
- L2 regression loss for bounding box offsets: $t^i = (t_x^i, t_y^i, t_w^i, t_h^i)$

**Transformations (box regression targets):**

$$t_x = \frac{G_x - P_x}{P_w}, \quad t_y = \frac{G_y - P_y}{P_h}$$
$$t_w = \log\frac{G_w}{P_w}, \quad t_h = \log\frac{G_h}{P_h}$$

**Limitations:**
- Training is multi-stage (CNN → SVM → regressor), not end-to-end
- ~2000 forward passes per image → **very slow** (~47s/image at test time)
- Features must be written to disk for SVM/regressor training
- Warping distorts aspect ratios

> 🟢 来自资料 — R-CNN 开创了两阶段检测范式，但速度是致命短板。

### 3.2 Fast R-CNN (2015)

**Key Innovation: RoI Pooling** — share convolution computation across all proposals.

**Pipeline:**

1. **Whole-image CNN**: Run the CNN once on the entire image → feature map.
2. **RoI Projection**: Each region proposal (from Selective Search) is projected onto the feature map.
3. **RoI Pooling**: Each RoI is divided into a fixed grid (e.g., $7 \times 7$), and max pooling is applied within each bin → fixed-size output regardless of input RoI size.
4. **FC layers + two sibling output layers**: classification (softmax over $C+1$ classes) + bbox regression (class-specific offsets).

**Multi-task Loss (joint training):**

$$L(p, u, t^u, v) = L_{\text{cls}}(p, u) + \lambda [u \geq 1] \cdot L_{\text{loc}}(t^u, v)$$

- $L_{\text{cls}}(p, u) = -\log p_u$ (log loss for true class $u$)
- $L_{\text{loc}}(t^u, v) = \sum_{i \in \{x,y,w,h\}} \text{smooth}_{L_1}(t_i^u - v_i)$

Where smooth L1 is:

$$\text{smooth}_{L_1}(x) = \begin{cases} 0.5x^2 & |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}$$

- $[u \geq 1]$ indicator: localization loss is 0 for background ($u=0$)
- $\lambda$ balances classification and localization (typically $\lambda = 1$)

**Advantages over R-CNN:**
- Single-stage training (end-to-end except for region proposals)
- No disk caching of features
- Training speed: ~9× faster; Test speed: ~213× faster

> 🟢 来自资料 — Fast R-CNN 的核心贡献是 RoI Pooling 让 CNN 特征可以共享，大幅加速训练和推理。

### 3.3 Faster R-CNN (2015)

**Key Innovation: Region Proposal Network (RPN)** — replace slow Selective Search with a learnable neural network.

**Architecture:**

1. **Backbone CNN**: Extract feature map (e.g., from VGG-16, ResNet).
2. **Region Proposal Network (RPN)**: A small fully-convolutional network that slides over the feature map.
3. **RoI Pooling + detection head**: Same as Fast R-CNN.

**Anchor Boxes**: At each sliding window position $(i, j)$ on the feature map (stride $s=16$), the RPN considers $k$ anchor boxes (reference boxes) at different scales and aspect ratios. Typical values:

- 3 scales: $\{128^2, 256^2, 512^2\}$
- 3 aspect ratios: $\{1:1, 1:2, 2:1\}$
- Total $k = 9$ anchors per position

For a feature map of size $H \times W$, there are $H \times W \times k$ anchors.

**RPN Architecture:**

- A $3 \times 3$ conv sliding window → 256-d (or 512-d) feature per position
- Two sibling $1 \times 1$ conv layers:
  - **cls layer**: $2k$ scores (objectness: object vs. not-object)
  - **reg layer**: $4k$ coordinates (bounding box offsets for each anchor)

**RPN Loss:**

$$L(\{p_i\}, \{t_i\}) = \frac{1}{N_{\text{cls}}} \sum_i L_{\text{cls}}(p_i, p_i^*) + \lambda \frac{1}{N_{\text{reg}}} \sum_i p_i^* \cdot L_{\text{reg}}(t_i, t_i^*)$$

- $p_i$: predicted objectness for anchor $i$
- $p_i^* \in \{0, 1\}$: ground-truth label (1 if IoU with any GT > 0.7, 0 if IoU < 0.3 for all GT)
- $t_i$: predicted offset; $t_i^*$: ground-truth offset (relative to anchor)
- $L_{\text{cls}}$: binary cross-entropy (log loss)
- $L_{\text{reg}}$: smooth L1 loss
- $p_i^*$ multiplier: only positive anchors contribute to regression loss
- $N_{\text{cls}}$: mini-batch size (~256); $N_{\text{reg}}$: number of anchor locations (~2400)
- $\lambda \approx 10$ for balanced weighting

**Training**: 4-step alternating training or approximate joint training (more common).

> 🟢 来自资料 — Faster R-CNN 的 RPN + anchor 机制是里程碑式的贡献，首次实现了纯神经网络端到端的目标检测。

### 3.4 FPN (Feature Pyramid Network, 2017)

**Problem**: Objects exist at multiple scales. A single-scale feature map (e.g., stride 32) has poor resolution for small objects.

**Solution**: Build a feature pyramid with a **top-down pathway and lateral connections**.

**Architecture:**

1. **Bottom-up pathway**: Standard CNN forward pass produces feature maps $\{C_2, C_3, C_4, C_5\}$ at strides $\{4, 8, 16, 32\}$ (ResNet naming).
2. **Top-down pathway**: Starting from $C_5$, upsample (2×) and combine with lateral connections:
   $$P_i = \text{Conv}_{3\times3}(\text{Conv}_{1\times1}(C_i) + \text{Upsample}(P_{i+1}))$$
3. **RPN + Detection Heads**: Applied independently at each pyramid level $\{P_2, \dots, P_6\}$.

**Anchor assignment**: Anchors of different scales are assigned to different pyramid levels according to their area:
$$k = k_0 + \log_2(\sqrt{w h} / 224)$$

> 🟢 来自资料 — FPN 以极小计算开销解决了多尺度检测问题，成为几乎所有现代检测器的基础组件。

---

## 4. One-Stage Detectors

One-stage detectors directly predict class and bounding box without an explicit proposal stage.

### 4.1 YOLO (You Only Look Once, 2016)

**Core idea**: Frame detection as a single regression problem from image pixels to bounding boxes and class probabilities.

**YOLOv1 Architecture:**

1. Divide input image into $S \times S$ grid (e.g., $7 \times 7$).
2. Each grid cell predicts:
   - $B$ bounding boxes $(x, y, w, h, \text{confidence})$ — up to 2 boxes
   - $C$ class probabilities $\text{Pr}(\text{Class}_i | \text{Object})$
3. Confidence = $\text{Pr}(\text{Object}) \times \text{IoU}_{\text{pred}}^{\text{truth}}$

**Output**: $S \times S \times (B \times 5 + C)$ tensor = $7 \times 7 \times 30$ for PASCAL VOC.

**Loss Function (YOLOv1) — sum-squared error with weighting:**

$$L = \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2\right]$$
$$+ \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2\right]$$
$$+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2$$
$$+ \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{noobj}} (C_i - \hat{C}_i)^2$$
$$+ \sum_{i=0}^{S^2} \mathbb{1}_{i}^{\text{obj}} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2$$

Where $\lambda_{\text{coord}} = 5$ (emphasize localization), $\lambda_{\text{noobj}} = 0.5$ (down-weight empty boxes). The $\sqrt{w}$ and $\sqrt{h}$ terms compensate for size sensitivity of Euclidean loss.

**Limitations of YOLOv1:**
- Each grid cell predicts at most 2 objects → struggles with dense small objects
- No anchor boxes, purely learned bounding box priors
- Coarse grid ($7 \times 7$) limits spatial precision

**YOLO Evolution:**
| Version | Key Improvements |
|---------|-----------------|
| YOLOv2 (YOLO9000) | Anchor boxes, batch norm, multi-scale training, Darknet-19 |
| YOLOv3 | FPN-style multi-scale, Darknet-53, binary cross-entropy per class (multi-label) |
| YOLOv4 | Bag of freebies/specials: CSPDarknet53, PANet, Mosaic augmentation, CIoU loss |
| YOLOv5 | PyTorch implementation, auto-anchor, extensive augmentation |
| YOLOv8/YOLOv9/YOLOv10 | Anchor-free, decoupled heads, advanced label assignment strategies |

> 🟢 来自资料 — YOLO 系列以速度见长，不断融合学术界最佳实践，在实时检测领域占据主导地位。

### 4.2 SSD (Single Shot MultiBox Detector, 2016)

**Core ideas:**
- Use **multiple feature maps** at different scales (from different layers) instead of a single scale
- Each feature map cell predicts offsets relative to **default boxes** (pre-computed anchor boxes) at multiple aspect ratios
- No objectness proposal — directly predict class scores + box offsets

**Default boxes**: At each feature map cell, consider boxes of different scales:
$$s_k = s_{\text{min}} + \frac{s_{\text{max}} - s_{\text{min}}}{m - 1}(k - 1), \quad k \in [1, m]$$

Aspect ratios $\{1, 2, 3, \frac{1}{2}, \frac{1}{3}\}$ produce width $w = s_k\sqrt{a_r}$, height $h = s_k / \sqrt{a_r}$.

**Loss**: Similar to Faster R-CNN's multi-task loss: confidence loss (softmax) + localization loss (smooth L1), with hard negative mining (3:1 ratio of negatives to positives).

> 🟢 来自资料 — SSD 证明了多尺度特征图 + default boxes 可以高效地进行单阶段检测，速度优于 Faster R-CNN。

### 4.3 RetinaNet (2017) — Focal Loss

**Problem: Extreme class imbalance in one-stage detectors.** For a typical image, ~100K candidate locations but only ~10-100 objects. The vast majority are easy negatives that overwhelm the loss.

**Focal Loss** reshapes the standard cross-entropy loss to down-weight easy examples:

Standard cross-entropy: $\text{CE}(p_t) = -\log(p_t)$ where
$$p_t = \begin{cases} p & \text{if } y = 1 \\ 1 - p & \text{otherwise} \end{cases}$$

**Focal Loss:**

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- $(1 - p_t)^\gamma$: **modulating factor**. For well-classified examples ($p_t \to 1$), $(1-p_t)^\gamma \to 0$ → loss is down-weighted. For misclassified examples ($p_t \ll 1$), the factor is near 1.
- $\alpha_t$: class-balancing weight (similar to $\alpha$-balanced CE).
- $\gamma \geq 0$: focusing parameter. $\gamma = 0$ recovers standard CE; $\gamma = 2$ works best in practice.

**Architecture**: FPN backbone + two sub-networks (classification and box regression, both are small FCNs attached to each pyramid level).

> 🟢 来自资料 — Focal Loss 解决了单阶段检测器训练时的极端正负样本不平衡问题，使 RetinaNet 在精度上首次匹敌两阶段方法。

---

## 5. Anchor-Free Methods

Eliminate the need for pre-defined anchor boxes.

### 5.1 CornerNet (2018)

Detect objects as **pairs of keypoints** (top-left and bottom-right corners). Each corner is detected via a heatmap, and an embedding vector associates the two corners belonging to the same object.

**Key components:**
- **Corner Pooling**: A specialized pooling layer that helps better localize corners (pools from entire row/column direction).
- **Embedding loss**: Pull embeddings of the same object's corners together, push different objects apart: $L_{\text{pull}} + L_{\text{push}}$.

### 5.2 CenterNet (2019)

Model objects as **single points** — the center point. From each center, regress width and height directly.

**Pipeline:**
1. Predict a heatmap of object centers (keypoint estimation).
2. At each center, regress the object size $(w, h)$ and a local offset for sub-pixel accuracy.
3. No NMS needed (peak extraction from heatmap suffices).

Extremely simple and fast. Adapted from keypoint estimation frameworks.

### 5.3 FCOS (Fully Convolutional One-Stage, 2019)

**Core idea**: Predict a 4D vector $(l, t, r, b)$ at each feature map location — distances to the four sides of the bounding box.

**Key innovations:**
- **Center-ness branch**: Predict a "center-ness" score (0 to 1) to suppress low-quality predictions far from object centers:
  $$\text{centerness} = \sqrt{\frac{\min(l, r)}{\max(l, r)} \times \frac{\min(t, b)}{\max(t, b)}}$$
- **FPN for multi-scale**: Assign different object sizes to different pyramid levels, addressing ambiguity from overlapping boxes.

> 🟢 来自资料 — Anchor-free 方法去除了 anchor 设计的超参数负担，FCOS 和 CenterNet 证明了 anchor-free 可以达到甚至超越 anchor-based 方法的性能。

---

## 6. DETR (DEtection TRansformer, 2020)

**End-to-end transformer-based detector** that eliminates many hand-crafted components (NMS, anchor generation).

**Architecture:**

1. **CNN Backbone**: Extract feature map.
2. **Transformer Encoder**: Flatten features + positional encoding → self-attention to model global context.
3. **Transformer Decoder**: Takes $N$ learned **object queries** (positional encodings) as input. Each query attends to encoder output and predicts one object (or "no object").
4. **Prediction Heads**: FFN predicts class + bounding box for each query.

**Bipartite Matching Loss**: The fundamental challenge is that DETR predicts a fixed-size set of $N$ objects, but ground truth has $M$ objects ($M < N$). Hungarian algorithm finds the optimal bipartite matching between predictions and ground truth:

$$\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^{N} \mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)})$$

$$\mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)}) = -\mathbb{1}_{\{c_i \neq \varnothing\}} \hat{p}_{\sigma(i)}(c_i) + \mathbb{1}_{\{c_i \neq \varnothing\}} \lambda_{\text{box}} \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\sigma(i)})$$

The full Hungarian loss then sums classification (cross-entropy) + box loss (L1 + GIoU) for matched pairs.

**Key properties:**
- No NMS (each query predicts at most one object)
- No anchors
- Slow convergence (500 epochs vs. 12-36 for traditional detectors)
- Deformable DETR improves convergence with deformable attention

> 🟢 来自资料 — DETR 将目标检测重构为集合预测问题，用 Transformer + 匈牙利匹配替代了大量手工设计组件。

---

## 7. NMS and Soft-NMS

### 7.1 Non-Maximum Suppression (NMS)

After detection, multiple overlapping boxes often predict the same object. NMS removes redundant detections:

**Algorithm:**
1. Sort all detections by confidence score descending.
2. Select the highest-scoring box $B$ and add to output.
3. Remove all boxes with $\text{IoU}(B, B_i) \geq \tau$ (typically $\tau = 0.5$).
4. Repeat until no boxes remain.

**Drawback**: Hard threshold — if a legitimate object is close to another (crowded scenes), it may be incorrectly suppressed.

### 7.2 Soft-NMS

Instead of removing overlapping boxes, **decay their scores** proportionally to IoU:

$$s_i = \begin{cases} s_i & \text{if } \text{IoU}(M, b_i) < N_t \\ s_i \cdot (1 - \text{IoU}(M, b_i)) & \text{if } \text{IoU}(M, b_i) \geq N_t \end{cases}$$

Or with a Gaussian penalty:
$$s_i = s_i \cdot e^{-\frac{\text{IoU}(M, b_i)^2}{\sigma}}$$

Soft-NMS preserves nearby objects, improving recall in crowded scenes.

> 🟢 来自资料 — Soft-NMS 通过分数衰减替代硬删除，在密集场景下显著改善了召回率。

---

## 8. Comparison Table of Detection Architectures

| Method | Type | Proposals | Backbone | Speed | Key Innovation |
|--------|------|-----------|----------|-------|---------------|
| R-CNN (2014) | Two-stage | Selective Search | AlexNet | ~47s/img | CNN + region proposals |
| Fast R-CNN (2015) | Two-stage | Selective Search | VGG-16 | ~2s/img | RoI Pooling, multi-task loss |
| Faster R-CNN (2015) | Two-stage | RPN | VGG-16/ResNet | ~0.2s/img | RPN, anchor boxes |
| FPN (2017) | Two-stage | RPN | ResNet-FPN | ~0.2s/img | Multi-scale feature pyramids |
| YOLOv1 (2016) | One-stage | None (grid) | Custom | ~45 FPS | Unified regression |
| YOLOv3 (2018) | One-stage | None (grid+anchor) | Darknet-53 | ~30-60 FPS | Multi-scale, binary CE |
| SSD (2016) | One-stage | Default boxes | VGG-16 | ~46-59 FPS | Multi-scale feature maps |
| RetinaNet (2017) | One-stage | Anchors | ResNet-FPN | ~5-12 FPS | Focal Loss |
| CornerNet (2018) | Anchor-free | None | Hourglass | ~4 FPS | Corner keypoint pairs |
| CenterNet (2019) | Anchor-free | None | DLA/Hourglass | ~28-45 FPS | Center point + size |
| FCOS (2019) | Anchor-free | None | ResNet-FPN | ~15-22 FPS | Per-location 4D vector |
| DETR (2020) | Transformer | Object queries | ResNet | ~28 FPS | Bipartite matching, no NMS |

---

## 9. Practice Problems

### Problem 1: IoU Calculation
Two boxes: $B_1 = [10, 10, 50, 60]$ (x1, y1, x2, y2) and $B_2 = [30, 20, 70, 80]$. Calculate IoU.

**Solution:**
- Intersection: $[\max(10, 30), \max(10, 20), \min(50, 70), \min(60, 80)] = [30, 20, 50, 60]$
- Intersection area: $(50-30) \times (60-20) = 20 \times 40 = 800$
- $B_1$ area: $40 \times 50 = 2000$
- $B_2$ area: $40 \times 60 = 2400$
- Union: $2000 + 2400 - 800 = 3600$
- $\text{IoU} = \frac{800}{3600} \approx 0.222$

### Problem 2: Focal Loss Behavior
For $\gamma = 2$, $\alpha = 1$, compare FL loss for $p_t = 0.9$ (well-classified) vs. $p_t = 0.1$ (misclassified).

**Solution:**
- $p_t = 0.9$: $\text{FL} = -(1-0.9)^2 \log(0.9) = -(0.01)(-0.105) \approx 0.00105$
- $p_t = 0.1$: $\text{FL} = -(1-0.1)^2 \log(0.1) = -(0.81)(-2.303) \approx 1.865$
- The loss for the well-classified example is ~1800× smaller than the standard CE loss of $-\log(0.9) \approx 0.105$.

### Problem 3: Anchor Assignment
A Faster R-CNN with feature map stride 16, anchors $\{128, 256, 512\}^2$ and ratios $\{1:1, 1:2, 2:1\}$. How many anchors per spatial location? If the feature map is $40 \times 60$, what is the total number of anchors?

**Solution:**
- $3 \times 3 = 9$ anchors per location
- Total: $40 \times 60 \times 9 = 21600$ anchors

### Problem 4: RPN Loss Components
Explain the role of $\mathbb{1}_{ij}^{\text{obj}}$ in the RPN / detection loss function.

**Solution:**
The indicator $\mathbb{1}_{ij}^{\text{obj}}$ (or $p_i^*$) serves as a **gating function**: it is 1 only when the $j$-th anchor in cell $i$ is assigned as a **positive** (matched to a ground-truth object) and 0 otherwise. This ensures that:
- Only positive anchors contribute to the **localization (bounding box regression) loss** — we cannot regress box offsets for anchors with no associated object.
- **Classification/confidence loss** applies to both positive and negative anchors.
- Without this gating, the regression loss on background anchors would try to push boxes toward arbitrary locations, destabilizing training.

### Problem 5: DETR Matching
Why does DETR need bipartite matching? What would happen without it?

**Solution:**
DETR predicts a fixed-size set of $N$ boxes (with class labels). The ground truth contains $M < N$ objects (plus "no object" for the remaining $N-M$). The **bipartite matching** (Hungarian algorithm) finds a one-to-one assignment between predictions and ground-truth labels that minimizes a matching cost. Without it:
- Multiple predictions could all try to match the same ground-truth object.
- The loss would be undefined (which prediction is penalized for which object?).
- Training would collapse to predicting all objects in a single box (a degenerate solution), since standard cross-entropy does not enforce one-to-one correspondence.

This is fundamentally the **set prediction problem**: the loss must be invariant to permutations of predictions, and each ground-truth object must be assigned to exactly one prediction.

---

> 🟢 来自资料 — 以上内容综合自 Faster R-CNN、YOLO、SSD、RetinaNet 及 DETR 等经典论文，覆盖目标检测领域的核心技术演进。
