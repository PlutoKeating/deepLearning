export interface Chapter {
  id: number;
  title: string;
  subtitle: string;
  file: string;
  labLinks?: { name: string; file: string }[];
}

export const chapters: Chapter[] = [
  { id: 1, title: 'Introduction', subtitle: '计算机视觉导论', file: 'ch1_introduction_cv.md' },
  { id: 2, title: 'Linear Classifiers', subtitle: '线性分类器', file: 'ch2_linear_classifiers.md' },
  { id: 3, title: 'Regularization & Optimization', subtitle: '正则化与优化', file: 'ch3_regularization_optimization.md' },
  { id: 4, title: 'Neural Networks', subtitle: '神经网络', file: 'ch4_neural_networks.md' },
  { id: 5, title: 'CNN', subtitle: '卷积神经网络', file: 'ch5_cnn.md' },
  { id: 6, title: 'RNN', subtitle: '循环神经网络', file: 'ch6_rnn.md' },
  { id: 7, title: 'Transformer', subtitle: 'Transformer架构', file: 'ch7_transformers.md' },
  { id: 8, title: 'Semantic Segmentation', subtitle: '语义分割', file: 'ch8_semantic_segmentation.md' },
  { id: 9, title: 'Object Detection', subtitle: '目标检测', file: 'ch9_object_detection.md' },
  { id: 10, title: 'Video Understanding', subtitle: '视频理解', file: 'ch10_video_understanding.md' },
  { id: 11, title: 'Self-Supervised Learning', subtitle: '自监督学习', file: 'ch11_self_supervised.md' },
  { id: 12, title: 'Generative Models I', subtitle: '生成模型 (GAN/VAE)', file: 'ch12_generative_models.md' },
  { id: 13, title: 'Generative Models II', subtitle: '扩散模型', file: 'ch13_diffusion_models.md' },
  { id: 14, title: 'Foundation Models', subtitle: '基础模型', file: 'ch14_foundation_models.md' },
];

export const labs = [
  { id: 1, name: 'Lab 1: PyTorch入门', file: 'lab01_pytorch_intro.md' },
  { id: 2, name: 'Lab 2: CNN与迁移学习', file: 'lab02_cnn_transfer.md' },
  { id: 3, name: 'Lab 3: 检测与分割', file: 'lab03_detection_seg.md' },
];
