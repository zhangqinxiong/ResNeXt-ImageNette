# ResNeXt - ImageNette 图像分类

基于 [timm](https://github.com/huggingface/pytorch-image-models) 库的 ResNeXt50\_32x4d 预训练模型，在 [ImageNette](https://github.com/fastai/imagenette) 数据集上进行全参数微调。

## 项目结构

```
ResNeXt/
├── train.py          # 训练脚本
├── best_model.pth    # 最佳验证准确率模型权重（自动生成）
├── runs/             # TensorBoard 日志目录（自动生成）
├── README.md         # 本文件
└── .gitignore        # Git 忽略规则
```

## 环境要求

- Python ≥ 3.8
- PyTorch ≥ 1.10（推荐 2.x）
- torchvision（匹配 PyTorch 版本）
- timm
- tensorboard
- tqdm

### 安装依赖

```bash
pip install torch torchvision timm tensorboard tqdm
```

## 数据集

[ImageNette](https://github.com/fastai/imagenette) 是 ImageNet 的一个子集，包含 10 个易于区分的类别：

| 类别 ID      | 描述         |
|-------------|-------------|
| n01440764   | tench       |
| n02102040   | English springer |
| n02979186   | cassette player |
| n03000684   | chain saw   |
| n03028079   | church      |
| n03394916   | French horn |
| n03417042   | garbage truck |
| n03425413   | gas pump    |
| n03445777   | golf ball   |
| n03888257   | parachute   |

数据集目录结构：
```
/path/to/ImageNette/
├── train/
│   ├── n01440764/
│   ├── n02102040/
│   └── ...
└── val/
    ├── n01440764/
    ├── n02102040/
    └── ...
```

## 配置说明

所有超参数在 `train.py` 头部集中定义：

| 参数              | 值      | 说明                          |
|-------------------|---------|-------------------------------|
| NUM_CLASSES       | 10      | 分类数                        |
| EPOCHS            | 50      | 总训练轮数                    |
| WARMUP_EPOCHS     | 3       | 预热轮数                      |
| LR                | 1e-4    | 初始学习率                    |
| WEIGHT_DECAY      | 0.05    | 权重衰减                      |
| BATCH_SIZE        | 64      | 批次大小                      |
| NUM_WORKERS       | 4       | 数据加载进程数                 |
| TENSORBOARD_PORT  | 6006    | TensorBoard 端口              |

## 训练

### 启动训练

```bash
python train.py
```

训练过程会自动：
1. 加载 timm 预训练的 `resnext50_32x4d` 权重
2. 替换全连接层为 10 类分类头
3. 全参数微调（所有层均参与训练）
4. 每个 epoch 结束后在验证集上评估
5. 保存验证准确率最高的模型到 `best_model.pth`

### 学习率调度策略

- **前 3 个 epoch (Warmup)**：学习率从 0 线性增长到 `LR`
- **后 47 个 epoch (Cosine Annealing)**：学习率按余弦曲线从 `LR` 衰减到 0

### AMP 混合精度

使用 `torch.cuda.amp` 自动混合精度训练：
- 前向传播自动在 fp16 下计算
- loss 使用 GradScaler 缩放，防止梯度下溢
- 权重更新时自动反缩放

## 可视化

启动 TensorBoard：

```bash
tensorboard --logdir=runs --port=6006
```

打开浏览器访问 `http://localhost:6006`，可查看：
- Loss/train, Loss/val
- Acc/train, Acc/val
- LR 变化曲线

## 实验结果

在 ImageNette 上训练 50 epoch 后的典型结果：

| 指标              | 值       |
|-------------------|----------|
| 最佳验证准确率     | ~99.7%   |
| 训练 Loss         | ~0.03    |
| 验证 Loss         | ~0.01    |

## 参考

- [Aggregated Residual Transformations for Deep Neural Networks (ResNeXt)](https://arxiv.org/abs/1611.05431)
- [timm: PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
- [ImageNette](https://github.com/fastai/imagenette)
