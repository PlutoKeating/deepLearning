# Ch11: Self-Supervised Learning

> 🟢 来自资料 — 基于课程讲义 `11_Self-Supervised Learning.pdf` 及 SimCLR, MoCo, BYOL, SimSiam, DINO, MAE, BEiT 等论文

---

## 1. Motivation: The Labeling Bottleneck

Supervised deep learning requires massive labeled datasets (ImageNet: 1.2M images, COCO: 330K images). Labeling is:
- **Expensive**: Requires human annotators for each image
- **Limited**: Cannot scale to all concepts and domains
- **Biased**: Annotator subjectivity and labeling errors

Self-supervised learning (SSL) **leverages the data itself as supervision** — the model learns representations by solving **pretext tasks** derived from the data structure without external labels.

**Core objective**: Learn a representation function $f_\theta: \mathcal{X} \to \mathbb{R}^d$ that captures semantic information useful for downstream tasks.

> 🟢 来自资料 — SSL 用数据自身的结构信号取代人工标签，是缓解标注瓶颈的核心方法。

---

## 2. Pretext Tasks

Early SSL approaches designed hand-crafted tasks where labels are generated automatically:

| Task | Supervision Signal | How It Works |
|------|-------------------|--------------|
| **Rotation Prediction** | 4-class (0°, 90°, 180°, 270°) | Predict rotation angle applied to the image |
| **Jigsaw Puzzle** | Permutation index | Shuffle patches, predict the correct permutation |
| **Colorization** | Original colors | Colorize grayscale images |
| **Inpainting** | Masked region | Reconstruct randomly masked image regions |
| **Relative Patch Location** | 8-class | Predict relative position of two patches |
| **Exemplar-CNN** | Surrogate classes | Each image treated as its own class (data augmentation creates variants) |
| **Counting** | Count number | Predict object count in synthetically generated images |

**Limitations**:
- Pretext tasks may not teach representations useful for semantic recognition
- Hand-crafted tasks are specific to image domain, hard to generalize across modalities
- Performance lags behind fully-supervised methods

> 🟡 AI补充: Pretext task 时代奠定了 SSL 的基本思想，但表示质量受限于任务设计，缺乏通用性。

---

## 3. Contrastive Learning Paradigm

### 3.1 Core Idea

Pull representations of **positive pairs** (different views of the same instance) close together in the embedding space, while pushing **negative pairs** (views of different instances) apart.

$$\text{sim}(\mathbf{z}_i, \mathbf{z}_j) \text{ is large for } (i,j) \text{ positive, small for negative}$$

**Positive pairs**: Two augmented views of the same image (random crop, color jitter, blur, etc.).  
**Negative pairs**: Views from different source images. Typically sampled from a large memory bank or the current mini-batch.

### 3.2 InfoNCE Loss

The **InfoNCE (Noise Contrastive Estimation)** loss is the standard contrastive objective:

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_i^+) / \tau)}{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_i^+) / \tau) + \sum_{j=1}^{K} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j^-) / \tau)}$$

where:
- $\mathbf{z}_i$: query representation (anchor)
- $\mathbf{z}_i^+$: positive key (augmented view of same image)
- $\mathbf{z}_j^-$: $K$ negative keys
- $\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a}^\top \mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$: cosine similarity
- $\tau$: temperature hyperparameter (lower = sharper distribution, more focus on hard negatives)

**Interpretation**: This is a $(K+1)$-way softmax classification — the model must identify which of the $K+1$ keys corresponds to the positive pair.

> 🟢 来自资料 — InfoNCE 是对比学习的事实标准目标函数，源自 CPC (Contrastive Predictive Coding) 并广泛应用于视觉 SSL。

---

## 4. SimCLR (2020)

**A Simple Framework for Contrastive Learning of Visual Representations** (Chen et al., ICML 2020).

### 4.1 Architecture

1. **Data Augmentation**: Generate two views $\tilde{x}_i$ and $\tilde{x}_j$ from the same image via a **composition** of augmentations:
   - Random crop + resize (key augmentation)
   - Random horizontal flip
   - Color distortion (color jittering + color dropping — crucial!)
   - Gaussian blur

2. **Encoder** $f(\cdot)$: A ResNet (e.g., ResNet-50) extracts representations $\mathbf{h}_i = f(\tilde{x}_i)$.

3. **Projection Head** $g(\cdot)$: A small MLP (2-3 layers) maps $\mathbf{h}_i \to \mathbf{z}_i = g(\mathbf{h}_i)$. The contrastive loss is applied on $\mathbf{z}$, but downstream tasks use $\mathbf{h}$.

4. **Loss**: InfoNCE on the mini-batch ($2N$ examples → $2N-2$ negatives per positive):

$$\ell(i,j) = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau)}$$

Final loss: $\mathcal{L} = \frac{1}{2N} \sum_{k=1}^{N} [\ell(2k-1, 2k) + \ell(2k, 2k-1)]$

### 4.2 Key Design Choices

| Design Choice | Finding |
|---------------|---------|
| **Projection head** | Non-linear head (2-layer MLP) → +10% improvement; discarding the head at downstream → +2% over keeping it |
| **Batch size** | Larger is better (256 → 4096 → 8192). Large batch = more negatives. Requires LARS optimizer for large-batch training. |
| **Augmentation** | Composition is crucial; random crop + color distortion individually insufficient, together strongly complementary |
| **Training length** | Longer training (800-1000 epochs) significantly improves representation quality |
| **Normalization** | L2-normalize embeddings before computing similarity → critical |

> 🟢 来自资料 — SimCLR 通过系统消融实验确立了对比学习的核心组件：组合数据增强、非线性投影头、大 batch size。

---

## 5. MoCo (Momentum Contrast, 2020)

**Problem**: Contrastive learning needs many negatives, but batch size is limited by GPU memory. SimCLR uses huge batches (4096-8192) requiring TPUs.

**MoCo's solution**: Maintain a **dynamic dictionary** (queue) of encoded keys from recent mini-batches.

### 5.1 Architecture

Three components:

1. **Query Encoder** $f_q$: Standard encoder (updated by backprop).
2. **Key Encoder** $f_k$: **Momentum-updated** copy of the query encoder.
3. **Dynamic Dictionary (Queue)**: Stores the last $K$ encoded keys (FIFO queue, e.g., $K = 65536$).

**Training step**:
1. Encode current query: $\mathbf{q} = f_q(x^{\text{query}})$
2. Encode current key: $\mathbf{k}_+ = f_k(x^{\text{key}})$
3. Compute loss using $\mathbf{k}_+$ (positive) and the $K$ keys in the queue (negatives)
4. Enqueue $\mathbf{k}_+$, dequeue oldest
5. Update $f_q$ by backprop; update $f_k$ by momentum:

$$\theta_k \leftarrow m \theta_k + (1 - m) \theta_q$$

with $m = 0.999$ (very slow update for key consistency).

**InfoNCE Loss** (with queue negatives):

$$\mathcal{L}_q = -\log \frac{\exp(\mathbf{q} \cdot \mathbf{k}_+ / \tau)}{\exp(\mathbf{q} \cdot \mathbf{k}_+ / \tau) + \sum_{i=1}^{K} \exp(\mathbf{q} \cdot \mathbf{k}_i / \tau)}$$

**Why momentum?** A rapidly changing key encoder would make keys across batches inconsistent → the queue would contain stale representations from different encoder versions. Slow momentum update ensures keys evolve smoothly, maintaining a consistent dictionary.

> 🟢 来自资料 — MoCo 的动量编码器 + 队列机制解耦了 batch size 与负样本数量，使得在普通 GPU 上也能使用大量负样本。

### 5.2 MoCo v2 (2020)

Integrates SimCLR's improvements into the MoCo framework:
- **MLP projection head** (2-layer, with BN)
- **Stronger data augmentation** (adding blur, stronger color distortion)
- **Cosine learning rate schedule**

Result: Outperforms SimCLR with smaller batches on standard GPUs (8 GPUs → competitive with SimCLR on TPUs).

---

## 6. BYOL (Bootstrap Your Own Latent, 2020)

**Key claim: Contrastive learning works WITHOUT negative pairs.**

### 6.1 Architecture

- **Online network** $f_\theta$: encoder + projector + predictor
- **Target network** $f_\xi$: encoder + projector (no predictor); same architecture, **momentum-updated**:
  $$\xi \leftarrow \tau \xi + (1 - \tau) \theta, \quad \tau = 0.996$$

**Loss** (simple MSE between normalized predictions):

$$\mathcal{L}_{\text{BYOL}} = \left\|\frac{q_\theta(z_\theta)}{\|q_\theta(z_\theta)\|} - \frac{z_\xi}{\|z_\xi\|}\right\|_2^2 = 2 - 2 \cdot \frac{\langle q_\theta(z_\theta), z_\xi \rangle}{\|q_\theta(z_\theta)\| \cdot \|z_\xi\|}$$

**Asymmetric architecture is crucial**:
- The **predictor** $q_\theta$ (extra MLP on the online branch) prevents collapse.
- The **stop-gradient** on the target branch prevents trivial solutions.
- Without either → collapse (constant representation for all inputs).

### 6.2 Why No Collapse?

The momentum target network provides a slowly-moving target, combined with the asymmetric online predictor. The target network's parameters are not optimized to minimize BYOL loss directly — they follow via EMA. This **implicitly introduces contrast** even without explicit negative pairs.

> 🟢 来自资料 — BYOL 通过非对称架构（在线网络 + 动量目标网络）实现了无负样本的自监督学习，挑战了"负样本不可或缺"的认知。

---

## 7. SimSiam (2021)

**Further simplification: No momentum encoder, no negative pairs.**

### 7.1 Architecture

- **Two identical encoders** $f$ (shared weights, no momentum).
- Two augmented views go through the same encoder.
- One branch has a **predictor MLP + stop-gradient**.

$$\mathcal{L} = \frac{1}{2}\mathcal{D}(p_1, \text{sg}(z_2)) + \frac{1}{2}\mathcal{D}(p_2, \text{sg}(z_1))$$

where $\mathcal{D}$ is negative cosine similarity and $\text{sg}(\cdot)$ = stop-gradient.

**Key insight**: **Stop-gradient is the essential ingredient** for preventing collapse. SimSiam formulates SSL as an Expectation-Maximization (EM)-like algorithm where stop-gradient naturally emerges.

> 🟢 来自资料 — SimSiam 证明了 stop-gradient 是防止坍缩的核心机制，动量编码器和负样本都不是必要的。

---

## 8. DINO (2021)

**Self-distillation with no labels**, applied to Vision Transformers.

### 8.1 Architecture

- **Student network** $g_{\theta_s}$: ViT encoder + projection head (outputs $K$-dim distribution via softmax with temperature $\tau_s$).
- **Teacher network** $g_{\theta_t}$: Same architecture, **momentum-updated**, outputs distribution with temperature $\tau_t$ ($\tau_t < \tau_s$, sharper distribution).

### 8.2 Loss

Cross-entropy between student and teacher outputs (same image, different crops):

$$\min_{\theta_s} -\mathbf{P}_t(\mathbf{x}) \log \mathbf{P}_s(\mathbf{x})$$

where $\mathbf{P}_t = \text{softmax}(g_{\theta_t}(\mathbf{x}) / \tau_t)$ and $\mathbf{P}_s = \text{softmax}(g_{\theta_s}(\mathbf{x}) / \tau_s)$.

**Teacher update**: $\theta_t \leftarrow \lambda \theta_t + (1 - \lambda) \theta_s$ (EMA).

**Centering + Sharpening**: The teacher output is centered (subtract running mean) and sharpened (low temperature) to avoid collapse.

### 8.3 Emerging Properties

When applied to ViT, DINO attention maps spontaneously produce **semantic segmentation** in the self-attention heads — the model learns to segment objects without any pixel-level supervision.

> 🟢 来自资料 — DINO 在 ViT 上的自蒸馏产生了涌现特性——自注意力图自动捕获语义分割，展示了 SSL 的潜力。

---

## 9. Masked Image Modeling (MIM)

Inspired by BERT's masked language modeling, MIM learns by reconstructing masked image content.

### 9.1 MAE (Masked Autoencoder, 2022)

**Asymmetric encoder-decoder** with high mask ratio:

1. **Mask**: Divide the image into patches, randomly mask 75% of them.
2. **Encoder**: ViT that processes **only visible patches** (25% of total). This drastically reduces computation.
3. **Decoder**: Lightweight ViT that processes all patches (visible encoder outputs + learnable mask tokens). Only used during pre-training, discarded for downstream tasks.
4. **Loss**: MSE between predicted and original pixel values in the masked patches only:

$$\mathcal{L} = \frac{1}{|M|} \sum_{i \in M} \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2$$

**Key findings**:
- 75% mask ratio is optimal — images have heavy spatial redundancy (unlike language).
- The decoder can be very lightweight (small, shallow).
- Normalized pixel values as reconstruction targets work well.

> 🟢 来自资料 — MAE 通过非对称编码器-解码器和高掩码率 (75%) 高效利用图像的视觉冗余，证明了掩码图像建模的可行性。

### 9.2 BEiT (2021)

**BERT-style pre-training** for vision:

1. Use a pre-trained **dVAE (discrete VAE)** tokenizer to convert image patches to a vocabulary of visual tokens.
2. During pre-training, mask patches and **predict the discrete visual token** (cross-entropy classification, like BERT predicting masked words).
3. Also trains for **visual token prediction + image-level classification** (CLS token).

MIM is a **generative SSL** approach, complementary to contrastive methods.

> 🟡 AI补充: MIM 和对比学习是当前 SSL 的两大主流范式，MIM 更适合密集预测任务，对比学习更适合语义判别任务。

---

## 10. Joint Embedding Predictive Architecture (JEPA)

Proposed by Yann LeCun as an alternative to both generative and contrastive approaches.

**Core idea**: Predict the **representation** of a target signal (e.g., a masked region) from the representation of a context signal, but predict in **representation space** rather than pixel space:

$$\mathcal{L} = \|\text{Pred}(\mathbf{z}_{\text{context}}) - \mathbf{z}_{\text{target}}\|^2$$

with **stop-gradient** on $\mathbf{z}_{\text{target}}$ to prevent collapse.

**Advantages over generative models**: No need to model high-frequency pixel details.  
**Advantages over contrastive**: No need for negative pairs.

**I-JEPA (Image JEPA)**: Context encoder + target encoder (EMA) + predictor. Masks target blocks, predicts their representation from surrounding context blocks.

> 🟡 AI补充: JEPA 在表示空间进行预测，避免了像素空间的细节重建和负样本需求，是 LeCun 提出的通向世界模型的核心架构。

---

## 11. Comparison Table of SSL Methods

| Method | Approach | Negatives? | Momentum? | Stop-Grad? | Architecture |
|--------|----------|-----------|-----------|-----------|--------------|
| **SimCLR** | Contrastive | Yes (in-batch) | No | No | Siamese (shared) |
| **MoCo v1/v2** | Contrastive | Yes (queue) | Yes ($\theta_k$) | No | Asymmetric ($\theta_q$, $\theta_k$) |
| **BYOL** | Non-contrastive | No | Yes (target) | Yes (target) | Asymmetric + predictor |
| **SimSiam** | Non-contrastive | No | No | Yes | Siamese + predictor |
| **DINO** | Self-distillation | No | Yes (teacher) | Yes (teacher) | Student-Teacher |
| **MAE** | Generative (MIM) | No | No | No (no dual) | AE (enc visible, dec all) |
| **BEiT** | Generative (MIM) | No | No | No | BERT-style + dVAE |
| **I-JEPA** | Joint-embedding pred. | No | Yes (target) | Yes (target) | Context-Target + Pred |

### Paradigm Comparison

| Paradigm | Mechanism | Examples | Pros | Cons |
|-----------|-----------|----------|------|------|
| **Contrastive** | Push neg. apart, pull pos. together | SimCLR, MoCo | Strong semantics | Needs many negatives, data aug. sensitive |
| **Generative** | Reconstruct input from partial observation | MAE, BEiT | Great for dense tasks | Wastes capacity on pixel details |
| **Joint-Embedding** | Predict representation of target from context | BYOL, SimSiam, DINO, I-JEPA | No negatives, no pixel reconstruction | Risk of collapse |

---

## 12. Practice Problems

### Problem 1: InfoNCE Temperature
Explain the effect of the temperature $\tau$ in InfoNCE loss. What happens as $\tau \to 0$ and $\tau \to \infty$?

**Solution:**
- **$\tau \to 0$**: The softmax distribution becomes very peaked — the model focuses almost exclusively on the hardest negative (the one with highest similarity). This can lead to instability as a single negative dominates gradient. The loss becomes more like max-margin.
- **$\tau \to \infty$**: The distribution becomes uniform — the loss becomes $\log(K+1)$, essentially constant. No learning occurs.
- **Typical values**: $\tau \in [0.07, 0.5]$. Lower $\tau$ = more focus on hard negatives, higher = softer contrast.

### Problem 2: Momentum Update
MoCo uses $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$ with $m=0.999$. If the query encoder updates by $\Delta\theta = 0.01$ in one step, how much does the key encoder change?

**Solution:**
$$\theta_k^{\text{new}} = 0.999 \cdot \theta_k^{\text{old}} + 0.001 \cdot (\theta_k^{\text{old}} + 0.01)$$
$$= 0.999 \cdot \theta_k^{\text{old}} + 0.001 \cdot \theta_k^{\text{old}} + 0.001 \cdot 0.01$$
$$= \theta_k^{\text{old}} + 1 \times 10^{-5}$$

The key encoder changes by only $10^{-5}$ per step, ensuring slow evolution and a consistent queue.

### Problem 3: Collapse Detection
What does it mean when a SSL model "collapses"? How would you detect it?

**Solution:**
**Collapse**: The model outputs the same (or trivially similar) representation for all inputs, regardless of content. This achieves zero training loss but learns nothing.
**Detection**: Compute pairwise cosine similarity of representations across a batch. If $\text{mean}(\text{sim}(\mathbf{z}_i, \mathbf{z}_j)) \approx 1.0$ for all pairs, the model has collapsed. A healthy model shows similarity near 1.0 only for positive pairs and much lower (0.0-0.2) for negatives.

### Problem 4: MAE Mask Ratio
Why does MAE use 75% masking for images while BERT only masks 15% for text?

**Solution:**
Images have **much higher spatial redundancy** than text — neighboring pixels are highly correlated, and a 256×256 image carries far more redundant information than a 256-token text sequence. With 75% masking:
- The visible patches provide sufficient context for reconstruction.
- The encoder only processes 25% of patches → 4× faster pre-training.
- At 75% ratio, the reconstruction task is challenging enough to force semantic understanding (the model cannot simply interpolate from neighbors).

Text tokens are more information-dense — masking 75% of words would make sentences unreadable, removing too much semantic content for meaningful reconstruction.

### Problem 5: BYOL vs SimSiam
Both BYOL and SimSiam avoid collapse without negative pairs. What mechanisms does each use?

**Solution:**
- **BYOL**: Uses **two mechanisms**: (1) momentum target network (EMA update) provides a slowly-evolving target; (2) asymmetric predictor on the online branch. The combination prevents both the online and target networks from jointly finding the trivial solution.
- **SimSiam**: Uses **one mechanism**: **stop-gradient**. The stop-gradient operation makes the optimization an EM-like procedure — it implicitly introduces an alternating optimization that prevents collapse. SimSiam shows that the momentum encoder in BYOL was a bonus (improved performance) but not the core anti-collapse mechanism.

---

> 🟢 来自资料 — 自监督学习已从手工 pretext task 发展到对比学习 (SimCLR, MoCo)，再到无负样本方法 (BYOL, SimSiam) 和生成式方法 (MAE)，代表了表示学习的范式演进。
