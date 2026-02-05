"""
训练CRNN文本识别模型的主脚本。
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# -- 动态添加项目根目录到 sys.path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

from recognition.converter import LabelConverter
from recognition.dataset import create_dataloader
from recognition.model import CRNN

# -- 全局配置常量 --
TRAIN_ANNOTATION = "recognition/data/train_annotation.txt"
VALID_ANNOTATION = "recognition/data/valid_annotation.txt"
KEYS_PATH = "recognition/keys.txt"
IMG_HEIGHT = 32
IMG_WIDTH = 1800
NUM_CHANNELS = 1

# 训练配置
BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 3e-4 # 稍微提高初始 LR，配合 Scheduler 使用
CHECKPOINT_DIR = "recognition/checkpoints/"
SAVE_INTERVAL = 5

# 硬件配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4


def debug_visualization(model, converter, dataset_loader):
    """
    可视化调试：从 Loader 中取一个 Batch，打印前 3 个样本的 GT 和 Pred
    """
    model.eval()
    print("\n[Debug] 采样检查预测效果:")
    try:
        # 获取一个 Batch
        # 注意：dataset_loader 已经配置了 collate_fn，返回的是 (images, labels)
        batch = next(iter(dataset_loader))
        images, labels = batch
        images = images.to(DEVICE)

        with torch.no_grad():
            preds = model(images)  # 输出形状: (Seq, Batch, C)

            # --- [修复] ---
            # 不要在这里转置！converter.decode 内部会自动处理 (Seq, Batch, C) 的形状。
            # 如果这里转置了，converter 内部误判后会再次转置，导致维度错乱。
            # preds = preds.transpose(0, 1) <--- 删除这行

        # 直接传入 (Seq, Batch, C)
        decoded_preds = converter.decode(preds.cpu())

        for i in range(min(3, len(labels))):
            print(f"  样本 {i + 1}:")
            print(f"    GT  : {labels[i]}")
            print(f"    Pred: {decoded_preds[i]}")

            # 简单对比一下，给个直观的正确/错误标记
            status = "✅" if labels[i] == decoded_preds[i] else "❌"
            print(f"    Result: {status}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Debug 失败: {e}")
    print("-" * 30)

def train():
    print(f"使用设备: {DEVICE}")

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    writer = SummaryWriter(log_dir=f"recognition/runs/ocr_experiment_{int(time.time())}")

    converter = LabelConverter(KEYS_PATH)
    num_classes = converter.num_classes
    print(f"类别数量: {num_classes}")

    # 初始化模型
    model = CRNN(
        img_height=IMG_HEIGHT,
        nc=NUM_CHANNELS,
        n_class=num_classes,
    ).to(DEVICE)

    # 损失函数
    criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(DEVICE)

    # 优化器: 使用 AdamW 替代 Adam，泛化性更好
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # [新增] 学习率调度器: 当 Loss 不再下降时，减少 LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )

    train_loader = create_dataloader(
        annotation_path=TRAIN_ANNOTATION,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        num_channels=NUM_CHANNELS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        is_train=True
    )

    valid_loader = None
    if VALID_ANNOTATION and os.path.exists(VALID_ANNOTATION):
        valid_loader = create_dataloader(
            annotation_path=VALID_ANNOTATION,
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH,
            num_channels=NUM_CHANNELS,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            is_train=False
        )
        print("验证集已加载。")
    else:
        print("警告: 未找到验证集标注文件。")

    print("开始训练...")
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]", unit="batch")

        for i, (images, labels) in enumerate(pbar):
            images = images.to(DEVICE)

            preds = model(images)

            text, length = converter.encode(labels)
            text = text.to(DEVICE)
            length = length.to(DEVICE)

            preds_size = torch.IntTensor([preds.size(0)] * images.size(0)).to(DEVICE)

            loss = criterion(preds, text, preds_size, length)

            optimizer.zero_grad()
            loss.backward()

            # [新增] 梯度裁剪: 防止 LSTM 梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss

            pbar.set_postfix(loss=f"{current_loss:.4f}", lr=optimizer.param_groups[0]['lr'])
            writer.add_scalar('Train/Batch_Loss', current_loss, global_step)
            global_step += 1

        avg_loss = total_loss / len(train_loader)
        print(f"第 {epoch} 轮平均损失: {avg_loss:.4f}")
        writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)

        # 更新学习率
        scheduler.step(avg_loss)

        # 每 2 个 Epoch 进行一次可视化 Debug
        if epoch % 2 == 0:
            debug_visualization(model, converter, train_loader)

        if valid_loader:
            val_acc = evaluate(model, valid_loader, converter)
            writer.add_scalar('Validation/Accuracy', val_acc, epoch)

        if epoch % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"recognition_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"检查点已保存至 {checkpoint_path}")

    writer.close()

def evaluate(model, dataloader, converter):
    model.eval()
    correct_predictions = 0
    total_samples = 0

    print("\n--- 开始验证 ---")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(DEVICE)
            preds = model(images)
            decoded_preds = converter.decode(preds.cpu())

            total_samples += len(labels)
            for pred, gt in zip(decoded_preds, labels):
                # 简单去除空格比较，防止 padding 引起的误差
                if pred.strip() == gt.strip():
                    correct_predictions += 1

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    print(f"验证集准确率: {accuracy:.4f}")
    if total_samples > 0:
        print(f"  示例 GT: {labels[0]}")
        print(f"  示例 Pred: {decoded_preds[0]}\n")

    return accuracy

if __name__ == "__main__":
    train()