# 📚 深度学习与计算机视觉 期末极速通关计划

建议复习时长：14 小时 (可根据距离考试的时间缩短)
Wiki 知识库路径：`references/wiki/`
题库路径：`references/quiz_bank.json`

---

## 📅 阶段复习进度表

<!-- 下方 PHASE_TABLE 为 scripts/ingest.py 的插入锚点，请勿删除 -->
<!-- PHASE_TABLE -->

| 阶段 | 内容 | 章节 | 建议时长 | 关联 Wiki | 关键知识点 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Phase 1 | Foundation — CV基础与线性分类器 | Ch1–Ch2 | 2h | `references/wiki/ch1_introduction_cv.md` `references/wiki/ch2_linear_classifiers.md` | CV定义与应用、k-NN、线性分类器、SVM/Softmax损失、正则化 |
| Phase 2 | Training & Neural Nets — 优化与神经网络 | Ch3–Ch4 | 2h | `references/wiki/ch3_regularization_optimization.md` `references/wiki/ch4_neural_networks.md` | SGD/Adam优化器、BatchNorm、Dropout、激活函数、反向传播、MLP |
| Phase 3 | CNN, RNN, Transformer — 三大核心架构 | Ch5–Ch7 | 3h | `references/wiki/ch5_cnn.md` `references/wiki/ch6_rnn.md` `references/wiki/ch7_transformers.md` | 卷积/池化、AlexNet/VGG/ResNet、LSTM/GRU、自注意力、ViT |
| Phase 4 | Segmentation & Detection — 分割与检测 | Ch8–Ch9 | 2h | `references/wiki/ch8_semantic_segmentation.md` `references/wiki/ch9_object_detection.md` | FCN/U-Net/DeepLab、IoU/Dice、R-CNN系列/YOLO、mAP/NMS |
| Phase 5 | Video & SSL — 视频理解与自监督 | Ch10–Ch11 | 2h | `references/wiki/ch10_video_understanding.md` `references/wiki/ch11_self_supervised.md` | 3D CNN/Two-Stream/SlowFast、SimCLR/MoCo/BYOL/MAE |
| Phase 6 | Generative Models — 生成模型 | Ch12–Ch13 | 2h | `references/wiki/ch12_generative_models.md` `references/wiki/ch13_diffusion_models.md` | GAN/DCGAN/WGAN、VAE/ELBO、DDPM/DDIM/CFG、Stable Diffusion |
| Phase 7 | Foundation Models + Review — 基础模型与总复习 | Ch14 + 综合 | 2h | `references/wiki/ch14_foundation_models.md` | CLIP、Scaling Laws、LoRA、RLHF、SAM、错题重做、综合测验 |

---

## 📋 各阶段详细任务

### Phase 1: Foundation（Ch1–Ch2）
- [ ] 阅读 Ch1 课件：CV定义、应用场景、发展历史（侧重2012年AlexNet里程碑）
- [ ] 阅读 Ch2 课件：k-NN分类器原理与k值选择
- [ ] 掌握线性分类器：参数化方法、SVM损失（hinge loss）、Softmax损失（交叉熵）
- [ ] 理解正则化：L1 vs L2 正则化，在损失函数中的体现
- [ ] 练习 quiz_bank.json 中 q1–q10

### Phase 2: Training & Neural Nets（Ch3–Ch4）
- [ ] 阅读 Ch3 课件：SGD、Momentum、Adam 优化器原理
- [ ] 掌握训练技巧：Batch Normalization（训推差异）、Dropout（inverted dropout）、数据增强
- [ ] 阅读 Ch4 课件：激活函数（ReLU/LeakyReLU/Sigmoid/Tanh）对比
- [ ] 理解反向传播与计算图，能手动推导简单网络的梯度
- [ ] 练习 quiz_bank.json 中 q11–q20

### Phase 3: CNN, RNN, Transformer（Ch5–Ch7）
- [ ] 阅读 Ch5 课件：卷积运算、池化、感受野、参数共享
- [ ] 掌握经典CNN：AlexNet、VGG（3×3堆叠）、ResNet（跳跃连接）
- [ ] 阅读 Ch6 课件：RNN梯度消失、LSTM三门控、GRU两门控、BPTT
- [ ] 阅读 Ch7 课件：自注意力QKV、多头注意力、位置编码、ViT架构
- [ ] 练习 quiz_bank.json 中 q21–q35

### Phase 4: Segmentation & Detection（Ch8–Ch9）
- [ ] 阅读 Ch8 课件：语义分割任务、FCN（全卷积+转置卷积）
- [ ] 掌握 U-Net（对称编码解码+跳跃连接）、DeepLab（空洞卷积+ASPP）
- [ ] 理解评估指标：IoU（交并比）、Dice系数、mIoU
- [ ] 阅读 Ch9 课件：两阶段检测器（R-CNN/Fast/Faster R-CNN）
- [ ] 掌握一阶段检测器（YOLO）、Focal Loss、mAP计算、NMS
- [ ] 练习 quiz_bank.json 中 q36–q45

### Phase 5: Video & SSL（Ch10–Ch11）
- [ ] 阅读 Ch10 课件：3D CNN、双流网络（空间+时间流）、光流
- [ ] 掌握 SlowFast（慢路径+快路径架构）
- [ ] 阅读 Ch11 课件：对比学习框架 SimCLR（NT-Xent loss）
- [ ] 掌握 MoCo（动量编码器+动态字典）、BYOL（无负样本）
- [ ] 理解 MAE（掩码自编码器，高掩码率重建）
- [ ] 练习 quiz_bank.json 中 q46–q55

### Phase 6: Generative Models（Ch12–Ch13）
- [ ] 阅读 Ch12 课件：GAN（极大极小博弈）、DCGAN架构规范
- [ ] 理解 VAE（变分自编码器）ELBO：重建损失+KL散度
- [ ] 掌握 WGAN（Wasserstein距离）、模式坍塌问题
- [ ] 阅读 Ch13 课件：DDPM扩散过程（前向加噪+反向去噪）
- [ ] 掌握 DDIM（加速采样）、CFG（无分类器引导）、Stable Diffusion（LDM隐空间扩散）
- [ ] 练习 quiz_bank.json 中 q56–q65

### Phase 7: Foundation Models + Review（Ch14 + 综合）
- [ ] 阅读 Ch14 课件：CLIP双塔对比学习、Scaling Laws
- [ ] 理解 LoRA（低秩分解微调）、RLHF三阶段
- [ ] 掌握 SAM（可提示分割基础模型）
- [ ] 查看 study_progress.md 中的错题档案，重新练习错题
- [ ] 综合测验：随机抽取跨章节混合题目
- [ ] 练习 quiz_bank.json 中 q66–q73

---

## 🧪 Lab 知识点回顾

| Lab | 主题 | 核心技能 | 关联题目 |
| :--- | :--- | :--- | :--- |
| Lab01 | PyTorch入门 | Tensor操作、自动求导、`torch.no_grad()`、训练循环 | q71 |
| Lab02 | CNN迁移学习 | 冻结(freeze)与微调(fine-tune)、预训练模型使用 | q72 |
| Lab03 | 检测与分割 | 图像预处理、归一化、Faster R-CNN推理 | q73 |
