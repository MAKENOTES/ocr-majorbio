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

def validate(model, val_loader, criterion, device, writer, epoch):
    """
    在验证集上评估模型，并记录可视化结果。
    """
    model.eval() # 切换到评估模式
    val_loss = 0
    val_score_loss = 0
    val_geo_loss = 0
    
    # 用于可视化的样本 (只取第一个 batch)
    vis_images = None
    vis_true_score = None
    vis_pred_score = None
    vis_pred_geo = None
    
    with torch.no_grad(): # 不计算梯度
        for batch_idx, batch in enumerate(val_loader):
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
            
            # 抓取第一个 batch 用于可视化
            if batch_idx == 0:
                vis_images = images
                vis_true_score = true_score_maps
                vis_pred_score = pred_score_maps
                # 取 Geo Map 的第一个通道 (dist_top) 进行可视化，归一化以便显示
                vis_pred_geo = pred_geo_maps[:, 0:1, :, :] 
                
    avg_loss = val_loss / len(val_loader)
    avg_score_loss = val_score_loss / len(val_loader)
    avg_geo_loss = val_geo_loss / len(val_loader)
    
    # --- TensorBoard 可视化 ---
    if writer and vis_images is not None:
        # 1. 原始图像
        writer.add_images('Vis/Input_Images', vis_images, epoch)
        
        # 2. Score Maps (True vs Pred)
        # 将 Score Map 扩展为 3 通道以便显示 (灰度图)
        writer.add_images('Vis/True_Score_Map', vis_true_score.repeat(1, 3, 1, 1), epoch)
        writer.add_images('Vis/Pred_Score_Map', vis_pred_score.repeat(1, 3, 1, 1), epoch)
        
        # 3. Geo Map (Pred - Top Distance)
        # 简单的归一化用于显示: div by max
        if vis_pred_geo.max() > 0:
            vis_pred_geo = vis_pred_geo / vis_pred_geo.max()
        writer.add_images('Vis/Pred_Geo_Map_Top', vis_pred_geo.repeat(1, 3, 1, 1), epoch)

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
    best_model_path = None # 记录当前最佳模型的路径

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
            writer.add_scalar('batch/loss', loss_val, global_step)
            writer.add_scalar('batch/score_loss', score_loss_val, global_step)
            writer.add_scalar('batch/geo_loss', geo_loss_val, global_step)

            epoch_loss += loss_val
            epoch_score_loss += score_loss_val
            epoch_geo_loss += geo_loss_val
            
            progress_bar.set_postfix({'loss': f"{loss_val:.4f}", 'score': f"{score_loss_val:.4f}", 'geo': f"{geo_loss_val:.4f}"})
            
        # --- Epoch 结束统计 ---
        avg_loss = epoch_loss / len(train_loader)
        avg_score_loss = epoch_score_loss / len(train_loader)
        avg_geo_loss = epoch_geo_loss / len(train_loader)
        
        print(f"Epoch {epoch+1} 训练结束: Loss: {avg_loss:.4f} (Score: {avg_score_loss:.4f}, Geo: {avg_geo_loss:.4f})")
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch + 1)

        # --- 验证步骤 ---
        # 每 VAL_INTERVAL 个 epoch 进行一次验证
        if val_loader and (epoch + 1) % VAL_INTERVAL == 0:
            print("正在验证...")
            # 传递 writer 和 epoch 以便记录可视化图像
            val_loss, val_score, val_geo = validate(model, val_loader, criterion, device, writer, epoch + 1)
            print(f"Epoch {epoch+1} 验证结果: Loss: {val_loss:.4f} (Score: {val_score:.4f}, Geo: {val_geo:.4f})")
            
            # 4. 记录训练和验证的 Epoch 平均损失
            writer.add_scalars('epoch/loss', {'train': avg_loss, 'validation': val_loss}, epoch + 1)
            writer.add_scalars('epoch/score Loss', {'train': avg_score_loss, 'validation': val_score}, epoch + 1)
            writer.add_scalars('epoch/geo Loss', {'train': avg_geo_loss, 'validation': val_geo}, epoch + 1)
            
            if val_loss < best_val_loss:
                # 删除旧的最佳模型
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                    print(f"已删除旧的最佳模型: {best_model_path}")

                best_val_loss = val_loss
                # 新的文件名包含 epoch 和 loss
                best_model_path = f"detection/checkpoints/best_detection_{epoch+1}_loss_{val_loss:.4f}.pth"
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
