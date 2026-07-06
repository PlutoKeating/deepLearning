# Ch10: Video Understanding

> 🟢 来自资料 — 基于课程讲义 `10_Video_Understanding.pdf` 及经典论文 (Two-Stream, C3D, I3D, TSN, SlowFast, TimeSformer)

---

## 1. Video as 3D Data

Unlike images ($H \times W \times 3$), video introduces a **temporal dimension**:

$$\mathbf{V} \in \mathbb{R}^{T \times H \times W \times 3}$$

where $T$ is the number of frames (time steps), $H$ and $W$ are spatial dimensions, and 3 represents RGB channels.

**Key challenge**: The temporal dimension adds significant computational cost and requires modeling **motion** — how objects, people, and scenes evolve over time. Naively treating each frame independently loses temporal information critical for action understanding.

---

## 2. Key Tasks

| Task | Description | Example Input → Output |
|------|-------------|----------------------|
| **Action Recognition** | Classify the action in a trimmed video clip | "Playing guitar", "Running" |
| **Temporal Action Localization** | Detect start/end times + classify actions in untrimmed video | Timestamps + labels |
| **Spatio-temporal Action Detection** | Detect actions in space and time (bounding box tubes) | Box sequences per action |
| **Video Captioning** | Generate natural language description | "A man is slicing a tomato on a cutting board" |
| **Video Question Answering** | Answer questions about video content | "What color is the car?" |
| **Video Object Segmentation** | Segment objects across frames | Pixel masks over time |

> 🟢 来自资料 — 视频理解的核心是时空联合建模，动作识别是最经典的基础任务。

---

## 3. Early Approaches

### 3.1 Single-Frame CNN Baseline

The simplest approach: apply a 2D CNN to individual frames, then average or pool predictions across time. This is surprisingly competitive for static-heavy actions but fails on motion-dependent actions (e.g., "opening" vs. "closing" a door).

### 3.2 Two-Stream Networks (Simonyan & Zisserman, 2014)

Two parallel CNNs, one for **spatial** appearance, one for **temporal** motion:

- **Spatial Stream**: Input = single RGB frame; captures scene/object appearance.
- **Temporal Stream**: Input = stack of **optical flow** frames (10 consecutive frames); captures motion patterns explicitly.

Both streams use standard 2D CNNs (e.g., VGG). Outputs are fused (late fusion: average or SVM on concatenated scores):

$$P(c|\mathbf{V}) = \frac{1}{2}\left[P_{\text{rgb}}(c|I) + P_{\text{flow}}(c|\Phi)\right]$$

**Training**: Spatial stream can be pre-trained on ImageNet. Temporal stream is typically trained from scratch or with cross-modal pre-training.

**Limitations**:
- Optical flow extraction is computationally expensive (not end-to-end)
- Only models short clips (~10 frames), misses long-range temporal structure
- Two separate networks, no shared computation

> 🟢 来自资料 — Two-Stream 开创了显式建模运动信息（光流）的范式，在深度学习时代早期取得最佳结果。

---

## 4. 3D CNNs

### 4.1 3D Convolution

Extend 2D convolution by adding a temporal kernel dimension:

$$\text{(3D Conv)} \quad y_{t,i,j} = \sum_{k=0}^{K_t-1} \sum_{u=0}^{K_h-1} \sum_{v=0}^{K_w-1} W_{k,u,v} \cdot x_{t+k, i+u, j+v} + b$$

where $K_t, K_h, K_w$ are temporal and spatial kernel sizes. A 3D conv kernel has shape $K_t \times K_h \times K_w \times C_{\text{in}} \times C_{\text{out}}$.

Compared to 2D conv: parameters $\propto K_t \times K_h \times K_w$ vs. $K_h \times K_w$.

### 4.2 C3D (2014)

Simple 3D CNN with $3 \times 3 \times 3$ kernels throughout:
- 8 conv layers + 5 pooling layers + 2 FC layers
- Input: 16-frame clips
- All kernels are $3 \times 3 \times 3$, stride $1 \times 1 \times 1$
- Pooling: $2 \times 2 \times 2$ (except first layer: $1 \times 2 \times 2$ to preserve temporal info)

**Key finding**: $3 \times 3 \times 3$ kernels consistently outperform varied kernel sizes.

**Limitations**: Small network, heavy compute, limited clip length (16 frames).

### 4.3 I3D (Inflated 3D ConvNets, 2017)

**Core insight**: Bootstrap 3D CNNs from 2D ImageNet pre-trained models.

**Inflation**: Take a pre-trained 2D conv filter $W^{2D} \in \mathbb{R}^{K_h \times K_w \times C_{\text{in}} \times C_{\text{out}}}$ and **inflate** it to 3D:

$$W^{3D}_{k,u,v} = \frac{1}{K_t} W^{2D}_{u,v}$$

Essentially, repeat the 2D filter along the temporal dimension and normalize. For pooling layers, inflate $2 \times 2$ → $1 \times 2 \times 2$ (or $2 \times 2 \times 2$).

**Two-Stream I3D**: Combines inflated 3D ConvNet with two-stream design:
- RGB stream: Inflated Inception-v1
- Flow stream: Input = 10-frame optical flow stack, also inflated
- Late fusion of scores from both streams

**Input**: 64 RGB frames at 25 FPS, full-resolution.

**Key advantage**: Leverages ImageNet pre-training → better generalization with limited video data.

> 🟢 来自资料 — I3D 的 "膨胀" 思想简单而有效，将 ImageNet 预训练迁移到 3D 是视频理解的重要突破。

---

## 5. (2+1)D Convolution

Decompose 3D convolution $K_t \times K_h \times K_w$ into:
- **Spatial convolution**: $1 \times K_h \times K_w$ (2D conv on each frame)
- **Temporal convolution**: $K_t \times 1 \times 1$ (1D conv along time)

With a non-linearity (ReLU) between them:

$$\text{(2+1)D Conv}(x) = \text{Conv}_{K_t \times 1 \times 1}\left(\text{ReLU}\left(\text{Conv}_{1 \times K_h \times K_w}(x)\right)\right)$$

**Advantages**:
- Fewer parameters than full 3D conv: $K_h K_w C_i C_o + K_t C_{\text{mid}} C_o$ vs. $K_t K_h K_w C_i C_o$
- Extra non-linearity doubles the network's representational capacity
- Easier optimization (factorized gradient paths)

**R(2+1)D**: ResNet architecture with (2+1)D blocks replacing 3D convolutions throughout.

> 🟡 AI补充: (2+1)D 分解借鉴了 Inception 中的空间-通道可分离卷积思想，提供了一种参数高效的 3D 建模方案。

---

## 6. Temporal Segment Networks (TSN, 2016)

**Problem**: 3D CNNs are heavy; two-stream models only capture short clips.

**Temporal Segment Networks** use **sparse temporal sampling**:

1. Divide the video into $K$ equal-length segments.
2. Randomly sample 1 snippet (short clip) from each segment.
3. Each snippet goes through a **shared** two-stream network (spatial + temporal).
4. **Segmental consensus**: Aggregate class scores from all $K$ snippets:
   - Averaging, max-pooling, or weighted averaging of scores
   - Concatenation of features before classification

$$L(y, G) = -\sum_{i=1}^{C} y_i \left(G_i - \log \sum_{j=1}^{K} \exp(G_i^j)\right)$$

**Key benefit**: Models long-range temporal structure across the entire video with only $K$ sparse samples, making it computationally efficient compared to dense frame processing.

**Training strategies**: Cross-modality pre-training, partial BN (freeze BN in spatial stream except first layer), dropout.

> 🟢 来自资料 — TSN 的稀疏采样策略使得模型可以高效地覆盖整个视频的时间跨度，是长视频理解的关键技术。

---

## 7. SlowFast (2019)

Two-pathway architecture inspired by biological vision (parvocellular + magnocellular pathways in the primate retina):

| | Slow Pathway | Fast Pathway |
|--|-------------|--------------|
| **Frame rate** | Low ($\tau$ frames, e.g., 1 frame every $\alpha=8$ frames) | High ($\alpha\tau$ frames) |
| **Channels** | Many ($C$ channels, high capacity) | Few ($\beta C$ channels, e.g., $\beta = 1/8$) |
| **Purpose** | Capture **spatial semantics** (what) | Capture **temporal motion** (how) |
| **Input** | Low temporal resolution, full spatial | High temporal resolution, lightweight processing |

**Key design**:
- **Lateral connections**: Fuse features from fast pathway to slow pathway (not vice versa, since slow pathway doesn't have fine temporal info):
  
  Fast features are reshaped ($T \times C_{\text{fast}} \times H \times W$ → $T/\alpha \times \alpha C_{\text{fast}} \times H \times W$) and concatenated with slow features.

- **No temporal downsampling in fast pathway**: Preserves fine temporal resolution.

**Result**: Strong performance on Kinetics-400/600 with lower compute than full 3D ResNets.

> 🟢 来自资料 — SlowFast 用非对称的双通路设计高效解耦了空间语义和时间运动，是 3D ConvNet 时代的重要创新。

---

## 8. Video Transformers

### 8.1 TimeSformer (2021)

Apply Vision Transformer to video with **divided space-time attention** to reduce computation:

Standard self-attention on $T \times H \times W$ patches costs $O((S \cdot T)^2)$ where $S = H \times W$ patches per frame. TimeSformer decomposes this:

1. **Spatial attention** (within each frame): Each patch attends to all patches *in the same frame only*.
2. **Temporal attention** (across frames): Each patch attends to patches at the *same spatial location across all frames*.

$$O(S^2 \cdot T + T^2 \cdot S) \ll O((S \cdot T)^2)$$

**Architecture**: $L$ transformer blocks, each applying spatial then temporal attention (or vice versa; "divided space-time").

### 8.2 ViViT (2021)

Several variants for factorizing spatial-temporal attention:

1. **Spatial-then-temporal** (factorized encoder): Like TimeSformer.
2. **Factorized dot-product**: Separate attention weights for space and time, multiplying them.
3. **Tubelet embedding**: Extract non-overlapping spatio-temporal "tubes" (3D patches) for initial embedding, then apply full joint attention on the reduced sequence.

> 🟢 来自资料 — Video Transformers 通过时空注意力解耦克服了 Transformer 二次复杂度在视频上的计算瓶颈。

---

## 9. Optical Flow

### 9.1 Definition

Optical flow is the apparent motion of brightness patterns in an image sequence. Given two consecutive frames $I(x, y, t)$ and $I(x, y, t+1)$, the flow field $\mathbf{u} = (u, v)$ describes the displacement of each pixel.

**Brightness Constancy Assumption**:

$$I(x, y, t) = I(x + u, y + v, t + 1)$$

**Optical Flow Constraint Equation** (first-order Taylor expansion):

$$\frac{\partial I}{\partial x} u + \frac{\partial I}{\partial y} v + \frac{\partial I}{\partial t} = 0$$

$$\nabla I \cdot \mathbf{u} + I_t = 0$$

This is one equation in two unknowns $(u, v)$ → **aperture problem**. Additional constraints (smoothness) are needed for a unique solution.

### 9.2 Learning-Based Methods

| Method | Approach |
|--------|----------|
| **FlowNet** (2015) | End-to-end CNN: FlowNetS (stacked images as input) and FlowNetC (correlation layer). U-Net-style refinement. |
| **FlowNet 2.0** (2017) | Stacked FlowNets with warping; specialized sub-networks for small/large displacements. |
| **PWC-Net** (2018) | Pyramid, Warping, Cost Volume: feature pyramid → warp features → construct cost volume → estimate flow. Compact and fast. |
| **RAFT** (2020) | Recurrent All-Pairs Field Transforms: compute all-pairs correlation volume → iterative GRU-based updates from an initial flow. State-of-the-art accuracy. |

> 🟢 来自资料 — RAFT 通过全像素对相关体 + 迭代 GRU 更新取得了光流估计的突破性性能。

---

## 10. Self-Supervised Video Representation Learning

Video's temporal structure provides a natural source of free supervisory signal:

| Method | Pretext Task | Description |
|--------|-------------|-------------|
| **Shuffle & Learn** | Temporal order verification | Is this clip in correct temporal order? |
| **Arrow of Time** | Temporal direction | Is the video playing forward or backward? |
| **SpeedNet** | Playback speed prediction | At what speed is this video playing? |
| **VCOP** | Clip order prediction | Permutation of shuffled clips |
| **Pace Prediction** | Pace ratio | Predict temporal dilation factor |
| **CVRL** | Contrastive (SimCLR-style) | Temporal augmentations: two clips from same video = positive |
| **VideoMAE** | Masked autoencoding | Mask large portions (90%) of video patches, reconstruct |

**Key insight**: Temporal coherence (nearby frames are similar) and temporal dynamics (order, speed) are abundant and reliable self-supervisory signals.

> 🟡 AI补充: 视频中的时间维度为自监督学习提供了丰富的免费监督信号，适合利用大规模无标注视频数据预训练。

---

## 11. Summary Formula Cheat Sheet

| Concept | Formula |
|---------|---------|
| 3D Convolution | $y_{t,i,j} = \sum_k \sum_u \sum_v W_{k,u,v} \cdot x_{t+k, i+u, j+v}$ |
| (2+1)D Decomposition | $\text{Conv}_{K_t \times 1 \times 1} \circ \text{ReLU} \circ \text{Conv}_{1 \times K_h \times K_w}$ |
| Optical Flow Constraint | $\nabla I \cdot \mathbf{u} + I_t = 0$ |
| Two-Stream Fusion | $P(c) = \frac{1}{2}[P_{\text{rgb}}(c) + P_{\text{flow}}(c)]$ |
| TimeSformer Complexity | $O(S^2 T + T^2 S)$ vs. full $O(S^2 T^2)$ |

---

## 12. Practice Problems

### Problem 1: 3D Conv Parameters
A (2+1)D block with input channels $C_{in} = 64$, output $C_{out} = 128$, $K_h = K_w = 3$, $K_t = 3$. The spatial conv uses $C_{mid} = 128$ intermediate channels. Count the parameters.

**Solution:**
- Spatial conv ($1 \times 3 \times 3$): $1 \times 3 \times 3 \times 64 \times 128 = 73,728$
- Temporal conv ($3 \times 1 \times 1$): $3 \times 1 \times 1 \times 128 \times 128 = 49,152$
- Total: $122,880$

Compare with full 3D conv: $3 \times 3 \times 3 \times 64 \times 128 = 221,184$ → (2+1)D saves ~44%.

### Problem 2: TSN Snippet Aggregation
A TSN with $K=3$ segments produces logits per segment: segment 1: $[2.0, 0.5, -1.0]$; segment 2: $[1.5, 1.0, 0.0]$; segment 3: $[0.8, 0.3, 1.5]$. Compute the final class probabilities with average pooling + softmax.

**Solution:**
- Averaged logits: $[\frac{2.0+1.5+0.8}{3}, \frac{0.5+1.0+0.3}{3}, \frac{-1.0+0.0+1.5}{3}] = [1.433, 0.600, 0.167]$
- Softmax: $\exp(1.433) = 4.19$, $\exp(0.600) = 1.82$, $\exp(0.167) = 1.18$
- Sum: $7.19$
- Probabilities: $[0.583, 0.253, 0.164]$

### Problem 3: Aperture Problem
A vertical edge moving horizontally: pixel $I(10, 5)$ at time $t$ has the same intensity as $I(11, 5)$ at time $t+1$. What does the optical flow constraint equation give us? Why is this insufficient?

**Solution:**
- The edge is vertical → $\frac{\partial I}{\partial y} = 0$ (no gradient along the edge)
- $\frac{\partial I}{\partial x} u + 0 \cdot v + I_t = 0$
- $\frac{\partial I}{\partial x} u + I_t = 0$ → we can solve for $u$ (horizontal motion)
- But $v$ (vertical motion) is **unconstrained** — the observed brightness pattern is consistent with any vertical displacement, hence the **aperture problem**: we cannot determine the full 2D motion from a local window that contains only a 1D edge. A corner or textured region (2D structure) is needed.

### Problem 4: SlowFast Channel Distribution
A SlowFast model uses $\alpha = 8$, $\beta = 1/8$. The slow pathway operates at 8 FPS with 64 channels in the first layer. What is the fast pathway's frame rate and channel count?

**Solution:**
- Fast pathway frame rate: $8 \times 8 = 64$ FPS
- Fast pathway channels: $64 \times 1/8 = 8$ channels

### Problem 5: Temporal Attention in TimeSformer
A video has $T=96$ frames, each divided into $14 \times 14 = 196$ spatial patches. Compare the FLOPs of joint space-time attention vs. divided space-time attention.

**Solution:**
- Total patches: $N = 96 \times 196 = 18,816$
- Joint attention complexity: $O(N^2) = O(18,816^2) \approx 3.54 \times 10^8$ operations
- Divided attention: $T \cdot O(S^2) + S \cdot O(T^2) = 96 \cdot 196^2 + 196 \cdot 96^2 = 96 \cdot 38,416 + 196 \cdot 9,216 = 3,687,936 + 1,806,336 \approx 5.49 \times 10^6$ operations
- Divided attention is ~64× cheaper in this case.

---

> 🟢 来自资料 — 视频理解从手工光流特征演进到端到端 3D CNN 再到 Video Transformer，核心是计算效率和长程时空建模的折中。
