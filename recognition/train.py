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
# 这解决了直接运行脚本时 "ModuleNotFoundError: No module named 'recognition'" 的问题。
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

from recognition.converter import LabelConverter
from recognition.dataset import create_dataloader
from recognition.model import CRNN

# -- 全局配置常量 --
# 数据配置
TRAIN_ANNOTATION = "recognition/data/train_annotation.txt"
VALID_ANNOTATION = "recognition/data/valid_annotation.txt"
KEYS_PATH = "recognition/keys.txt"
IMG_HEIGHT = 32
IMG_WIDTH = 1800
NUM_CHANNELS = 1

# 训练配置
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-4
PRETRAINED_MODEL = ""
CHECKPOINT_DIR = "recognition/checkpoints/"
SAVE_INTERVAL = 5
LOG_INTERVAL = 10

# 模型配置
HIDDEN_SIZE = 256
NUM_LAYERS = 2

# 硬件配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

def train():
    """
    主训练循环。
    """
    # -- 初始化 --
    print(f"使用设备: {DEVICE}")

    # 如果检查点目录不存在，则创建它
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # 初始化 TensorBoard Writer
    # 日志将保存在 runs/ 目录下，您可以根据需要修改 log_dir
    writer = SummaryWriter(log_dir=f"recognition/runs/ocr_experiment_{int(time.time())}")

    # 初始化标签转换器
    converter = LabelConverter(KEYS_PATH)
    num_classes = converter.num_classes
    print(f"类别数量: {num_classes}")

    # 初始化模型
    model = CRNN(
        num_classes=num_classes,
        img_height=IMG_HEIGHT,
        num_channels=NUM_CHANNELS,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    # 损失函数
    criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(DEVICE)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 数据加载器 ---
    # 加载训练集
    train_loader = create_dataloader(
        annotation_path=TRAIN_ANNOTATION,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        num_channels=NUM_CHANNELS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        is_train=True
    )
    
    # 检查并加载验证集
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
        print("警告: 未找到验证集标注文件，将跳过验证步骤。")
    
    print("开始训练...")
    
    global_step = 0

    # -- 训练循环 --
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        
        # 使用 tqdm 显示进度条
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]", unit="batch")
        
        for i, (images, labels) in enumerate(pbar):
            images = images.to(DEVICE)
            
            # 前向传播
            preds = model(images)
            
            # 为CTC Loss准备标签
            text, length = converter.encode(labels)
            text = text.to(DEVICE)
            length = length.to(DEVICE)
            
            preds_size = torch.IntTensor([preds.size(0)] * BATCH_SIZE).to(DEVICE)

            # 计算损失
            loss = criterion(preds, text, preds_size, length)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            total_loss += current_loss
            
            # 更新 tqdm 进度条显示的 loss
            pbar.set_postfix(loss=f"{current_loss:.4f}")

            # 记录 Batch Loss 到 TensorBoard
            writer.add_scalar('Train/Batch_Loss', current_loss, global_step)
            global_step += 1

        avg_loss = total_loss / len(train_loader)
        print(f"第 {epoch} 轮平均损失: {avg_loss:.4f}")
        
        # 记录 Epoch Loss 到 TensorBoard
        writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)

        # 验证 (可选)
        if valid_loader:
            val_acc = evaluate(model, valid_loader, converter)
            # 记录验证准确率到 TensorBoard
            writer.add_scalar('Validation/Accuracy', val_acc, epoch)

        # 保存检查点
        if epoch % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"recognition_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"检查点已保存至 {checkpoint_path}")

    # 关闭 writer
    writer.close()

def evaluate(model, dataloader, converter):
    """
    在验证集上评估模型。
    返回:
        float: 验证集准确率
    """
    model.eval()
    correct_predictions = 0
    total_samples = 0
    
    print("\n--- 开始验证 ---")
    with torch.no_grad():
        # 验证过程也可以加个简单的进度条，或者保持静默
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(DEVICE)
            preds = model(images)
            
            # 解码预测结果
            decoded_preds = converter.decode(preds.cpu())
            
            total_samples += len(labels)
            for pred, gt in zip(decoded_preds, labels):
                if pred == gt:
                    correct_predictions += 1
    
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    print(f"验证集准确率: {accuracy:.4f} ({correct_predictions}/{total_samples})")
    if total_samples > 0:
        print("预测示例:")
        print(f"  - 真实标签: {labels[0]}")
        print(f"  - 预测结果:   {decoded_preds[0]}\n")
    
    return accuracy

if __name__ == "__main__":
    train()
