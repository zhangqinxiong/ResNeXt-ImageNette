"""
ResNeXt - ImageNette 图像分类训练脚本
======================================
基于 timm 库的 ResNeXt50_32x4d 预训练模型，在 ImageNette 数据集上进行全参数微调。
支持 AMP 混合精度加速、TensorBoard 可视化、Warmup + Cosine Annealing 学习率调度。

Usage:
    python train.py
"""
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import os
from tqdm import tqdm
import math

# ======================== 超参数配置 ========================
DATA_ROOT = "/home/ivi/zqx/ImageNette"   # ImageNette 数据集根目录（含 train/ 和 val/）
NUM_CLASSES = 10                          # 分类数（ImageNette 共 10 类）
EPOCHS = 50                               # 总训练轮数
WARMUP_EPOCHS = 3                         # 预热轮数（学习率从 0 线性增长）
LR = 1e-4                                 # 初始学习率
WEIGHT_DECAY = 0.05                       # AdamW 权重衰减系数
BATCH_SIZE = 64                           # 批次大小
NUM_WORKERS = 4                           # DataLoader 子进程数
TENSORBOARD_PORT = 6006                   # TensorBoard 端口号

# ======================== 设备 ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======================== 数据预处理 ========================
# 训练增强：随机裁剪 + 水平翻转
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 验证增强：缩放 + 中心裁剪
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ImageFolder 自动按子目录名读取标签
train_dataset = ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_transform)
val_dataset = ImageFolder(os.path.join(DATA_ROOT, "val"), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

# ======================== 模型 ========================
# ResNeXt50_32x4d：50 层，32 个分组，每组 4 维，timm 预训练权重
model = timm.create_model("resnext50_32x4d", pretrained=True, num_classes=NUM_CLASSES)
model = model.to(device)

# ======================== 优化器 & 损失函数 & AMP ========================
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()                     # AMP 梯度缩放器（防止 fp16 下溢出）

# ======================== 学习率调度 ========================
total_steps = len(train_loader) * EPOCHS            # 总迭代步数
warmup_steps = len(train_loader) * WARMUP_EPOCHS    # 预热步数

def get_lr(step):
    """自定义学习率函数：前 warmup_steps 步线性增长，之后余弦退火衰减到 0。"""
    if step < warmup_steps:
        return LR * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return LR * 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                              lr_lambda=lambda step: get_lr(step) / LR)

# ======================== TensorBoard ========================
writer = SummaryWriter()
print(f"TensorBoard: tensorboard --logdir=runs --port={TENSORBOARD_PORT}")

# ======================== 训练循环 ========================
best_acc = 0.0
global_step = 0

for epoch in range(1, EPOCHS + 1):
    # ---- 训练阶段 ----
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # AMP 混合精度前向传播
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()      # 缩放 loss 后反向传播
        scaler.step(optimizer)             # 优化器更新参数（自动反缩放）
        scaler.update()                    # 更新缩放因子
        scheduler.step()                   # 每步更新学习率
        global_step += 1

        # 统计
        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

        pbar.set_postfix(loss=loss.item())

    train_acc = train_correct / train_total
    train_loss_avg = train_loss / train_total

    # ---- 验证阶段 ----
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    val_loss_avg = val_loss / val_total

    # ---- 记录到 TensorBoard ----
    writer.add_scalar("Loss/train", train_loss_avg, epoch)
    writer.add_scalar("Loss/val", val_loss_avg, epoch)
    writer.add_scalar("Acc/train", train_acc, epoch)
    writer.add_scalar("Acc/val", val_acc, epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

    # ---- 打印日志 ----
    print(f"Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, "
          f"Train Loss={train_loss_avg:.4f}, Val Loss={val_loss_avg:.4f}, "
          f"LR={optimizer.param_groups[0]['lr']:.2e}")

    # ---- 保存最佳模型 ----
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  -> Saved best model (acc={best_acc:.4f})")

writer.close()
print(f"Done! Best Val Acc: {best_acc:.4f}")
