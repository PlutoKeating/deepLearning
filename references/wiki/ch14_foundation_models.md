# Ch14: Foundation Models

> 🟢 来自资料 — 基于课程讲义 `14_Foundation Models.pdf` 及 GPT, BERT, CLIP, SAM, LLaVA, LoRA, RLHF 等经典论文

---

## 1. What Are Foundation Models?

**Definition** (Bommasani et al., 2021): Foundation models are models trained on **broad data at scale** and **adaptable to a wide range of downstream tasks**.

Key characteristics:
- **Scale**: Billions of parameters, trained on internet-scale data.
- **Emergence**: Capabilities not explicitly trained for emerge at scale.
- **Homogenization**: A few models serve as the foundation for many applications.
- **Adaptation**: Can be fine-tuned, prompted, or used in-context for diverse tasks.

> 🟢 来自资料 — 基础模型代表了从"每个任务训练一个模型"到"一个模型服务所有任务"的范式转变。

---

## 2. Language Models

### 2.1 GPT Series

| Model | Year | Parameters | Key Innovation |
|-------|------|-----------|---------------|
| **GPT-1** | 2018 | 117M | Generative pre-training + supervised fine-tuning. 12-layer Transformer decoder. |
| **GPT-2** | 2019 | 1.5B | Zero-shot task transfer; larger scale; "language models are unsupervised multitask learners" |
| **GPT-3** | 2020 | 175B | In-context learning (few-shot); no fine-tuning needed for many tasks |
| **InstructGPT** | 2022 | 175B | RLHF alignment; instruction following |
| **GPT-4** | 2023 | Undisclosed (~1.7T est.) | Multimodal (text + image input); strong reasoning; longer context |

**GPT Architecture**: Autoregressive Transformer decoder with masked self-attention:

$$p(x) = \prod_{i=1}^{n} p(x_i | x_1, \dots, x_{i-1})$$

Training objective (next-token prediction):
$$\mathcal{L}_{\text{LM}} = -\sum_{i} \log p_\theta(x_i | x_{<i})$$

### 2.2 BERT (2018)

**Bidirectional Encoder Representations from Transformers**. Unlike GPT, BERT uses a Transformer **encoder** with bidirectional attention.

**Pre-training objectives**:
1. **Masked Language Modeling (MLM)**: Mask 15% of tokens, predict masked tokens from context (both left and right).
2. **Next Sentence Prediction (NSP)**: Binary classification — does sentence B follow sentence A?

**Fine-tuning**: Add task-specific heads — single sentence classification, sentence pair classification, question answering, sequence tagging.

**Key insight**: Bidirectional context is crucial for understanding (vs. generation).

> 🟢 来自资料 — BERT 的双向编码器 + MLM 预训练范式启发了视觉领域的 BEiT 和 MAE。

---

## 3. Scaling Laws

### 3.1 Kaplan et al. (2020)

For autoregressive language models, the cross-entropy loss $L$ follows a power law:

$$L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D}$$

where $N$ = model parameters, $D$ = training tokens, $N_c, D_c, \alpha_N, \alpha_D$ are constants. Loss decreases predictably with scale.

### 3.2 Chinchilla Scaling Laws (2022)

**Key finding**: For compute-optimal training, model size and data should be scaled **equally**:
- Optimal: $N_{\text{opt}} \propto C^{0.5}$, $D_{\text{opt}} \propto C^{0.5}$
- Kaplan suggested $N \propto C^{0.73}$, $D \propto C^{0.27}$ (underestimated data needs)

**Practical implication**: For a 70B parameter model, you need ~1.4T tokens for compute-optimal training. Most models were **undertrained** (had too many parameters for their data).

### 3.3 Emergent Abilities

Abilities that are **not present in smaller models** but emerge at scale. Examples:
- Few-shot in-context learning
- Chain-of-thought reasoning
- Instruction following
- Arithmetic, translation, code generation

**Key property**: Performance on many benchmarks is near-random until a certain scale threshold, then jumps dramatically.

> 🟢 来自资料 — Scaling laws 和 emergent abilities 解释了为什么更大的模型能解锁全新能力，但 Chinchilla 告诉我们数据同样重要。

---

## 4. Vision-Language Models

### 4.1 CLIP (Contrastive Language-Image Pre-training, 2021)

**Core idea**: Learn a joint embedding space for images and text via contrastive learning on 400M (image, text) pairs from the internet.

**Architecture**:
- **Image Encoder**: ViT or ResNet.
- **Text Encoder**: Transformer (GPT-2 style).
- Both encoders project to a shared $d$-dimensional embedding space.

**Training**: Given a batch of $N$ (image, text) pairs, maximize cosine similarity for the $N$ correct pairs and minimize for the $N^2 - N$ incorrect pairs:

$$\mathcal{L}_{\text{CLIP}} = -\frac{1}{2N} \sum_{i=1}^{N} \left[\log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)} + \log \frac{\exp(\text{sim}(T_i, I_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(T_i, I_j) / \tau)}\right]$$

This is a **symmetric InfoNCE loss** — both image→text and text→image.

**Zero-shot classification**: For $K$ classes, compute similarity between image embedding and text embeddings of prompts like "a photo of a {class}". Predict the class with highest similarity.

$$\hat{y} = \arg\max_k \text{sim}(f_{\text{img}}(\mathbf{x}), f_{\text{text}}(\text{"a photo of } c_k\text{"}))$$

**Key properties**:
- Matches fully-supervised ResNet-50 on ImageNet **without any ImageNet labels**.
- Robust to distribution shift (doesn't overfit to ImageNet-specific features).
- Enables zero-shot transfer to any classification task describable in language.

> 🟢 来自资料 — CLIP 通过对比学习在 400M 图文对上训练，实现了强大的零样本分类和跨模态理解，是 VLM 的奠基之作。

### 4.2 Other VLMs

| Model | Key Features |
|-------|-------------|
| **ALIGN** | Larger, noisier dataset (1.8B pairs), simpler architecture, strong zero-shot |
| **BLIP** | Unified vision-language understanding + generation. Multimodal mixture of encoder-decoder (MED). Bootstraps captions for data. |
| **BLIP-2** | Q-Former: lightweight querying transformer bridges frozen image encoder + frozen LLM. Efficient cross-modal alignment. |

---

## 5. Multi-Modal Models

### 5.1 Flamingo (2022)

- **Architecture**: Frozen vision encoder (NFNet) + frozen LLM (Chinchilla), connected by trainable **Perceiver Resampler** and **gated cross-attention** layers.
- Enables few-shot visual tasks by interleaving image-text tokens and prompting the LLM.

### 5.2 GPT-4V (2023)

OpenAI's multimodal GPT-4 with vision capabilities:
- Accepts images + text as input.
- Visual reasoning, chart reading, screenshot understanding, visual question answering.
- Capabilities are emergent from scale; details of architecture undisclosed.

### 5.3 LLaVA (Large Language and Vision Assistant, 2023)

**Visual instruction tuning**: Fine-tune an LLM to follow multimodal instructions.

**Architecture**:
1. Vision encoder (CLIP ViT) → linear projection → LLM (Vicuna/Llama).
2. Train on **visual instruction data**: GPT-4 generates instruction-following conversations about images.
3. Two-stage training: (a) feature alignment (projection only), (b) end-to-end fine-tuning.

**Training data**: 158K language-image instruction-following samples.

> 🟢 来自资料 — LLaVA 将视觉编码器 + LLM 结合，通过 GPT-4 生成的指令数据训练，实现了强大的多模态对话能力。

---

## 6. Segment Anything Model (SAM, 2023)

**Promptable segmentation**: A model that can segment any object in any image given a prompt (point, box, mask, or text).

**Architecture**:
1. **Image Encoder**: ViT (MAE pre-trained) → dense image embedding.
2. **Prompt Encoder**: Encodes sparse (points, boxes) or dense (masks) prompts.
3. **Mask Decoder**: Lightweight Transformer that combines image + prompt embeddings → multiple mask predictions.

**Training**: Trained on **SA-1B** dataset — 11M images, 1.1B masks, generated through a data engine (model-in-the-loop annotation):
- **Stage 1**: Assisted manual annotation.
- **Stage 2**: Semi-automatic (model predicts, human corrects).
- **Stage 3**: Fully automatic (model generates masks on all images).

**Zero-shot generalization**: SAM can segment objects in entirely new image domains without fine-tuning.

> 🟢 来自资料 — SAM 通过大规模数据引擎 + 可提示架构实现了通用分割的零样本泛化，代表了视觉基础模型的成功范式。

---

## 7. Large Vision Models

### 7.1 DINOv2 (2023)

Self-supervised learning at scale for vision:
- Discriminative self-supervised pre-training (DINO + iBOT) on curated large-scale dataset (LVD-142M).
- Produces **general-purpose visual features** that work well for classification, segmentation, depth estimation, and retrieval without fine-tuning.
- ViT-based, trained with student-teacher distillation + masked image modeling objectives.

### 7.2 Other Large Vision Models

| Model | Approach |
|-------|----------|
| **ViT-22B** | Scaling ViT to 22B parameters with improved training stability |
| **EVA-02** | Masked image modeling + CLIP features as targets |
| **InternImage** | Large-scale CNN with deformable convolution, scaling to 3B params |

---

## 8. Prompt Engineering

### 8.1 Zero-Shot Prompting

Provide only the task description, no examples:

> "Classify the sentiment: 'This movie was fantastic!' →"

### 8.2 Few-Shot (In-Context Learning)

Provide $K$ examples in the prompt:

> "Great movie → Positive
> Terrible film → Negative
> It was okay, I guess → Neutral
> Absolutely loved it! →"

### 8.3 Chain-of-Thought (CoT)

Prompt the model to reason step-by-step:

> "Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 balls. How many balls does he have now?
> A: Let's think step by step. Roger started with 5 balls. 2 cans × 3 balls each = 6 balls. 5 + 6 = 11 balls. Answer: 11."

**Variants**: Zero-shot CoT ("Let's think step by step"), self-consistency (majority vote over multiple CoT samples), tree-of-thought.

> 🟢 来自资料 — Prompt engineering 是利用基础模型能力的关键技能，CoT 推理显著提升复杂推理任务的性能。

---

## 9. Parameter-Efficient Fine-Tuning (PEFT)

### 9.1 Motivation

Full fine-tuning of billion-parameter models is impractical:
- Storing separate copies for each task is expensive.
- Training all parameters requires massive compute.

PEFT methods adapt models by training only a small fraction of parameters.

### 9.2 LoRA (Low-Rank Adaptation, 2021)

Decomposes weight updates into **low-rank matrices**:

$$W' = W_0 + \Delta W = W_0 + BA$$

where $W_0 \in \mathbb{R}^{d \times k}$ is frozen, $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$ (e.g., $r = 8$ or $16$).

**Forward pass**:
$$h = W_0 x + BAx = W_0 x + \frac{\alpha}{r} \cdot BAx$$

where $\alpha$ is a scaling factor. Only $A$ and $B$ are trained — typically applied to attention projection matrices ($W_Q, W_K, W_V, W_O$).

**Advantages**:
- No inference latency (can merge $\Delta W$ into $W_0$).
- Small storage: ~0.1-1% of full parameters.
- Task switching by swapping $A, B$ matrices.

### 9.3 Other PEFT Methods

| Method | Mechanism |
|--------|-----------|
| **Adapters** | Insert small bottleneck layers between Transformer layers; train only the adapters |
| **Prefix Tuning** | Learn continuous "virtual tokens" prepended to the input; keep model frozen |
| **Prompt Tuning** | Learn soft prompts in the embedding space; simpler than prefix tuning |
| **IA³** | Learn rescaling vectors for activations (key, value, FFN) |

> 🟢 来自资料 — LoRA 通过低秩分解实现参数高效微调，已成为大模型适配的事实标准。

---

## 10. RLHF (Reinforcement Learning from Human Feedback)

Align language models with human preferences using reinforcement learning.

### 10.1 Three-Stage Process

**Stage 1: Supervised Fine-Tuning (SFT)**
- Collect human-written demonstrations (prompt → desired response).
- Fine-tune the pre-trained LM on these demonstrations.

**Stage 2: Reward Model Training**
- For a given prompt, generate multiple responses from the SFT model.
- Human labelers rank responses (better/worse).
- Train a **reward model** $r_\phi(x, y)$ to predict human preferences (typically as a Bradley-Terry model):

$$P(y_w \succ y_l | x) = \frac{\exp(r_\phi(x, y_w))}{\exp(r_\phi(x, y_w)) + \exp(r_\phi(x, y_l))}$$

- Loss: $-\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$ (binary cross-entropy on pairwise comparisons).

**Stage 3: PPO Fine-Tuning**
- Fine-tune the SFT model with Proximal Policy Optimization (PPO).
- The reward model scores outputs; a KL penalty prevents the model from diverging too far from SFT:

$$R(x, y) = r_\phi(x, y) - \beta \cdot D_{\text{KL}}(\pi_\theta(\cdot|x) \| \pi_{\text{SFT}}(\cdot|x))$$

### 10.2 DPO (Direct Preference Optimization, 2023)

Simplifies RLHF by directly optimizing the policy from preferences, **without a separate reward model**:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

This is equivalent to the RLHF objective under the Bradley-Terry model, but much simpler to implement.

> 🟢 来自资料 — RLHF 是将语言模型与人类价值观对齐的关键技术，InstructGPT/ChatGPT 的成功证明了对齐的重要性。

---

## 11. Multimodal Agents and Embodied AI

### 11.1 LLM-Based Agents

Foundation models increasingly serve as the "brain" of autonomous agents:
- **Planning**: Decompose high-level goals into sub-tasks.
- **Tool use**: Call APIs, run code, search the web.
- **Memory**: Maintain conversation history and knowledge bases.
- **Reflection**: Self-critique and iterative improvement.

**Examples**: AutoGPT, ReAct (Reasoning + Acting), Code Interpreter.

### 11.2 Embodied AI

Extending foundation models to physical world interaction:
- **Vision-Language-Action models**: RT-1, RT-2 (Robotics Transformer) — map vision + language directly to robot actions.
- **World models**: Predict future states given actions → plan in the learned model.
- **Sim-to-real transfer**: Train in simulation, deploy on real robots.

> 🟡 AI补充: 多模态智能体将基础模型作为推理-执行核心，是通向通用人工智能 (AGI) 的关键路径之一。

---

## 12. Ethical Considerations

| Concern | Description |
|---------|-------------|
| **Bias and Fairness** | Models reflect and amplify training data biases (gender, race, culture). Mitigation requires careful data curation and debiasing techniques. |
| **Hallucination** | Models generate plausible-sounding but factually incorrect content. Particularly dangerous in medical, legal, and educational applications. |
| **Environmental Cost** | Training large models consumes massive energy (GPT-3 estimated at ~550 tCO2). Efficient architectures and renewable energy are important. |
| **Privacy** | Training data may contain personal information; models can memorize and leak it. Differential privacy and data filtering are partial solutions. |
| **Misuse** | Deepfakes, disinformation at scale, automated social engineering. Watermarking, detection tools, and access controls are being developed. |
| **Accountability** | Who is responsible for model outputs? Opacity of training data and architectures complicates attribution. |
| **Economic Impact** | Automation of knowledge work; job displacement concerns. |

> 🟢 来自资料 — 基础模型的伦理问题（偏见、幻觉、隐私、环境影响）是课程考试中可能涉及的重要话题。

---

## 13. Future Directions

- **AGI**: Foundation models as a path toward general intelligence — debate about whether scaling alone suffices.
- **World Models**: Models that learn causal, physical, and commonsense understanding of the world.
- **Multimodal Foundation Models**: Unifying vision, language, audio, video, code, and actions.
- **Efficient Architectures**: State-space models (Mamba), mixture-of-experts (MoE), reducing compute needs while maintaining quality.
- **Alignment and Safety**: Scalable oversight, interpretability, mechanistic understanding of model internals.
- **Personalized Foundation Models**: Adaptation to individual users while preserving privacy.
- **Scientific Discovery**: Foundation models for protein folding (AlphaFold), drug discovery, materials science, and mathematical reasoning.

---

## 14. Practice Problems

### Problem 1: CLIP Zero-Shot Classification
CLIP achieves 76.2% accuracy on ImageNet zero-shot. A standard ResNet-50 achieves 76.2% when trained on the full ImageNet dataset. Explain why this is significant.

**Solution:**
This is significant because CLIP achieved this accuracy **without using any ImageNet training labels**. A standard ResNet-50 required 1.2M labeled images to reach the same accuracy. CLIP's performance demonstrates:
- **Transfer from natural language supervision**: The 400M noisy (image, text) pairs from the internet provide sufficient signal to learn visual concepts general enough to classify ImageNet categories.
- **Robustness**: CLIP's performance degrades far less on distribution shifts (ImageNet-V2, ImageNet-R, ImageNet-Sketch) compared to supervised models, suggesting it learns more robust visual features.
- **Zero-shot capability**: The model can be applied to any class describable in natural language without any training examples — a qualitatively different capability from supervised models.

### Problem 2: LoRA Parameter Count
A Transformer with $d_{\text{model}} = 4096$, 32 attention heads. Apply LoRA with rank $r = 16$ to $W_Q, W_K, W_V, W_O$ (each is $4096 \times 4096$). How many trainable parameters does LoRA add? What fraction of the total?

**Solution:**
Each weight matrix $W \in \mathbb{R}^{4096 \times 4096}$ with LoRA rank $r = 16$ adds $A \in \mathbb{R}^{16 \times 4096}$ and $B \in \mathbb{R}^{4096 \times 16}$:
- Per matrix: $16 \times 4096 + 4096 \times 16 = 131,072$ parameters.
- For 4 matrices ($W_Q, W_K, W_V, W_O$): $4 \times 131,072 = 524,288$ parameters.
- Original 4 matrices: $4 \times 4096 \times 4096 = 67,108,864$ parameters.
- Fraction: $\frac{524,288}{67,108,864} \approx 0.78\%$ (less than 1%).

### Problem 3: RLHF Reward Model
The reward model is trained on human pairwise preferences. A preference dataset has: for prompt $x$, human prefers $y_w$ over $y_l$. The reward model $r_\phi$ scores each. Write the loss function and explain its interpretation.

**Solution:**
$$\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_w, y_l)}[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))]$$

This is logistic regression: predict the probability that $y_w$ is preferred over $y_l$ as $\sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$. The loss encourages $r_\phi(x, y_w) > r_\phi(x, y_l)$ — the winning response should get a higher reward. The magnitude of the difference determines confidence.

### Problem 4: Emergent Abilities
Provide an example of an emergent ability in language models and explain what "emergent" means in this context.

**Solution:**
**Example**: Arithmetic (e.g., 3-digit addition). Models with < 1B parameters perform at near-chance level. At ~10B parameters, performance jumps to > 80%. This is not a gradual improvement — it's a qualitative change in behavior at a specific scale threshold.

**"Emergent" means**: The ability is **not explicitly trained** and is **not present** (or is negligible) in smaller versions of the same architecture trained on the same data distribution. It "emerges" solely from scaling up model size — the model develops new capabilities that were not engineered. This is significant because it suggests that scaling alone might unlock unforeseen capabilities, which has implications for AI safety (unpredictable capabilities).

### Problem 5: SAM Prompt Engineering
A user clicks one point on a dog in an image. SAM outputs three masks (whole, part, subpart). Explain why SAM produces multiple outputs and how the ambiguity is resolved.

**Solution:**
A single point is inherently **ambiguous** — it could refer to:
1. The dog's nose (subpart)
2. The dog's head (part)
3. The entire dog (whole)

SAM is designed to handle this **inherent ambiguity** by predicting multiple valid masks and ranking them by confidence. The ambiguity is resolved by the **confidence score**: SAM outputs masks with associated IoU predictions; the highest-confidence mask is typically the "whole" object. For more precise control, users can provide additional prompts (another point, a bounding box, or a rough mask) to disambiguate.

This design reflects SAM's philosophy: the model should handle the ambiguity internally rather than forcing a single arbitrary output, and users can iteratively refine.

### Problem 6: Chinchilla vs. GPT-3
GPT-3 (175B parameters) was trained on ~300B tokens. According to Chinchilla scaling laws, was GPT-3 under-trained or over-trained? How many more tokens would be optimal?

**Solution:**
According to Chinchilla, for compute-optimal training, parameters and tokens should scale equally: $N_{\text{opt}} \propto C^{0.5}$, $D_{\text{opt}} \propto C^{0.5}$.

For a 175B parameter model, Chinchilla's formula suggests approximately $175\text{B} \times 20 \approx 3.5\text{T}$ tokens would be optimal (roughly 20× more tokens than parameters). GPT-3 was trained on only ~300B tokens — it was **significantly undertrained**. The Chinchilla model (70B parameters) was trained on 1.4T tokens and outperformed GPT-3 despite being smaller, demonstrating the importance of adequate data.

---

> 🟢 来自资料 — 基础模型代表了 AI 从"专用"到"通用"的范式转变。从 GPT/BERT 的语言模型，到 CLIP/SAM 的视觉基础模型，再到多模态和对齐技术，理解其架构、训练范式和能力边界是当代 AI 的核心知识。
