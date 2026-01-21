import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
from torch.utils.tensorboard import SummaryWriter

from utils.dataset import OCRDataset
from utils.transforms import get_transforms
from models.east_model import EASTModel
from utils.loss import EASTLoss

# --- 超参数 ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 4 # 根据你的 GPU 显存调整
EPOCHS = 100
INPUT_SIZE = 1024
TRAIN_IMAGE_DIR = "detection/data/train_images"
TRAIN_LABEL_DIR = "detection/data/train_labels"
VAL_IMAGE_DIR = "detection/data/val_images"
VAL_LABEL_DIR = "detection/data/val_labels"
MAX_KEEP_CHECKPOINTS = 10 # 保留最近的多少个模型检查点
VAL_INTERVAL = 10 # 每多少个 epoch 进行一次验证

def validate(model, val_loader, criterion, device):
    """
    在验证集上评估模型。
    """
    model.eval() # 切换到评估模式
    val_loss = 0
    val_score_loss = 0
    val_geo_loss = 0
    
    with torch.no_grad(): # 不计算梯度
        for batch in val_loader:
            images = batch['image'].to(device)
            true_score_maps = batch['score_map'].to(device).unsqueeze(1)
            true_geo_maps = batch['geo_map'].to(device)
            
            pred_score_maps, pred_geo_maps = model(images)
            
            loss, score_loss, geo_loss = criterion(
                true_score_maps, pred_score_maps,
                true_geo_maps, pred_geo_maps
            )
            
            val_loss += loss.item()
            val_score_loss += score_loss.item()
            val_geo_loss += geo_loss.item()
            
    avg_loss = val_loss / len(val_loader)
    avg_score_loss = val_score_loss / len(val_loader)
    avg_geo_loss = val_geo_loss / len(val_loader)
    
    return avg_loss, avg_score_loss, avg_geo_loss

def main():
    """
    主训练函数。
    """
    # --- TensorBoard 设置 ---
    # 2. 初始化 SummaryWriter，日志会保存在 detection/runs/ 目录下
    log_dir = f"detection/runs/ocr_experiment_{int(time.time())}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志将保存在: {log_dir}")

    # --- 设备设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 数据加载 ---
    train_transform = get_transforms(is_train=True, input_size=INPUT_SIZE)
    train_dataset = OCRDataset(image_dir=TRAIN_IMAGE_DIR, label_dir=TRAIN_LABEL_DIR, transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    val_loader = None
    if os.path.exists(VAL_IMAGE_DIR) and os.path.exists(VAL_LABEL_DIR):
        print("发现验证集，正在加载...")
        val_transform = get_transforms(is_train=False, input_size=INPUT_SIZE)
        val_dataset = OCRDataset(image_dir=VAL_IMAGE_DIR, label_dir=VAL_LABEL_DIR, transform=val_transform)
        if len(val_dataset) > 0:
            val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            print(f"验证集加载完成，共 {len(val_dataset)} 张图片。")

    # --- 模型、损失函数和优化器 ---
    model = EASTModel(pretrained=True).to(device)
    criterion = EASTLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. 将模型图写入 TensorBoard (仅一次)
    if train_loader:
        first_batch = next(iter(train_loader))
        writer.add_graph(model, first_batch['image'].to(device))

    saved_checkpoints = []
    best_val_loss = float('inf')

    # --- 训练循环 ---
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_score_loss = 0
        epoch_geo_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device)
            true_score_maps = batch['score_map'].to(device).unsqueeze(1)
            true_geo_maps = batch['geo_map'].to(device)

            optimizer.zero_grad()
            pred_score_maps, pred_geo_maps = model(images)
            
            loss, score_loss, geo_loss = criterion(true_score_maps, pred_score_maps, true_geo_maps, pred_geo_maps)
            
            loss.backward()
            optimizer.step()
            
            loss_val, score_loss_val, geo_loss_val = loss.item(), score_loss.item(), geo_loss.item()
            
            # 3. 记录每个 batch 的训练损失到 TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/train_batch', loss_val, global_step)
            writer.add_scalar('Loss/score_loss_batch', score_loss_val, global_step)
            writer.add_scalar('Loss/geo_loss_batch', geo_loss_val, global_step)

            epoch_loss += loss_val
            epoch_score_loss += score_loss_val
            epoch_geo_loss += geo_loss_val
            
            progress_bar.set_postfix({'loss': f"{loss_val:.4f}", 'score': f"{score_loss_val:.4f}", 'geo': f"{geo_loss_val:.4f}"})
            
        # --- Epoch 结束统计 ---
        avg_loss = epoch_loss / len(train_loader)
        avg_score_loss = epoch_score_loss / len(train_loader)
        avg_geo_loss = epoch_geo_loss / len(train_loader)
        
        print(f"Epoch {epoch+1} 训练结束: Loss: {avg_loss:.4f} (Score: {avg_score_loss:.4f}, Geo: {avg_geo_loss:.4f})")

        # --- 验证步骤 ---
        # 每 VAL_INTERVAL 个 epoch 进行一次验证
        if val_loader and (epoch + 1) % VAL_INTERVAL == 0:
            print("正在验证...")
            val_loss, val_score, val_geo = validate(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1} 验证结果: Loss: {val_loss:.4f} (Score: {val_score:.4f}, Geo: {val_geo:.4f})")
            
            # 4. 记录训练和验证的 Epoch 平均损失
            writer.add_scalars('Loss/epoch', {'train': avg_loss, 'validation': val_loss}, epoch + 1)
            writer.add_scalars('Score Loss/epoch', {'train': avg_score_loss, 'validation': val_score}, epoch + 1)
            writer.add_scalars('Geo Loss/epoch', {'train': avg_geo_loss, 'validation': val_geo}, epoch + 1)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = "detection/checkpoints/best_model.pth"
                torch.save(model.state_dict(), best_model_path)
                print(f"★ 发现新最佳模型 (Val Loss: {val_loss:.4f})，已保存至 {best_model_path}")
        else:
             # 如果不进行验证，只记录训练损失
             writer.add_scalar('Loss/epoch_train', avg_loss, epoch + 1)

        # --- 保存常规检查点 ---
        checkpoint_dir = "detection/checkpoints/"
        if not os.path.exists(checkpoint_dir):
             os.makedirs(checkpoint_dir)
             
        checkpoint_filename = os.path.join(checkpoint_dir, f"detection_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_filename)
        
        saved_checkpoints.append(checkpoint_filename)
        if len(saved_checkpoints) > MAX_KEEP_CHECKPOINTS:
            oldest_checkpoint = saved_checkpoints.pop(0)
            if os.path.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)

    writer.close() # 6. 关闭 writer
    print("训练完成!")

if __name__ == '__main__':
    if not os.path.exists(TRAIN_IMAGE_DIR) or not os.path.exists(TRAIN_LABEL_DIR):
        print(f"错误: 请确保你的数据分别在 '{TRAIN_IMAGE_DIR}' 和 '{TRAIN_LABEL_DIR}' 目录下")
    else:
        main()
