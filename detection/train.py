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
LEARNING_RATE =  1e-5  #1e-4 3e-5
BATCH_SIZE = 4 # 根据你的 GPU 显存调整
EPOCHS = 500
INPUT_SIZE = 1024
TRAIN_IMAGE_DIR = "detection/data/train_images"
TRAIN_LABEL_DIR = "detection/data/train_labels"
VAL_IMAGE_DIR = "detection/data/val_images"
VAL_LABEL_DIR = "detection/data/val_labels"

# --- 模型保存策略优化 ---
MAX_KEEP_CHECKPOINTS = 10       # 常规检查点：保留最近的 5 个 (用于断点续训)
BEST_MODELS_TO_KEEP = 3        # 最佳检查点：保留验证集 Loss 最低的 Top 3
VAL_INTERVAL = 10              # 前期：每 10 个 epoch 验证一次
START_DENSE_VAL_EPOCH = 500    # 策略切换点：从第 500 epoch 开始
                               # 500 以后，每个 epoch 都验证 (忽略 VAL_INTERVAL)

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


def save_best_checkpoint(model, epoch, val_loss, best_checkpoints_list):
    """
    管理 Top-K 最佳模型的保存与删除
    best_checkpoints_list: [(loss, path), ...]
    """
    checkpoint_dir = "detection/checkpoints/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 保存当前模型
    file_name = f"best_detection_epoch_{epoch}_loss_{val_loss:.5f}.pth"
    save_path = os.path.join(checkpoint_dir, file_name)
    torch.save(model.state_dict(), save_path)

    # 加入列表并排序 (按 loss 从小到大)
    best_checkpoints_list.append((val_loss, save_path))
    best_checkpoints_list.sort(key=lambda x: x[0])

    # 如果超过数量限制，删除 loss 最大的那个 (列表最后一个)
    if len(best_checkpoints_list) > BEST_MODELS_TO_KEEP:
        worst_model = best_checkpoints_list.pop()  # 弹出最后一个 (loss 最大的)
        worst_path = worst_model[1]
        if os.path.exists(worst_path):
            os.remove(worst_path)
            print(f"移除旧的 Top-{BEST_MODELS_TO_KEEP + 1} 模型: {os.path.basename(worst_path)}")

    print(f"★ 当前 Top-{BEST_MODELS_TO_KEEP} 模型已更新 (本轮 Loss: {val_loss:.5f})")

def main():
    """
    主训练函数。
    """
    # --- TensorBoard 设置 ---
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
    # model.load_state_dict(torch.load("detection/checkpoints/detection_epoch_1200.pth"))
    criterion = EASTLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 将模型图写入 TensorBoard (仅一次)
    if train_loader:
        first_batch = next(iter(train_loader))
        writer.add_graph(model, first_batch['image'].to(device))


    regular_checkpoints = []  # 用于保存最近 N 个 epoch (用于 resume)
    best_checkpoints = []  # 用于保存 Top K best models (用于 inference)
    # --- 训练循环 ---
    for epoch in range(1, EPOCHS + 1): # 从 1 开始计数比较直观
        model.train()
        epoch_loss = 0
        epoch_score_loss = 0
        epoch_geo_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device)
            true_score_maps = batch['score_map'].to(device).unsqueeze(1)
            true_geo_maps = batch['geo_map'].to(device)

            optimizer.zero_grad()
            pred_score_maps, pred_geo_maps = model(images)
            loss, score_loss, geo_loss = criterion(true_score_maps, pred_score_maps, true_geo_maps, pred_geo_maps)
            loss.backward()
            
            # --- 记录梯度信息 (每 100 个 batch 记录一次，避免日志过大) ---
            if batch_idx % 100 == 0:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, epoch * len(train_loader) + batch_idx)
                        writer.add_scalar(f'Gradient_Norm/{name}', param.grad.norm(), epoch * len(train_loader) + batch_idx)

            #梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            loss_val, score_loss_val, geo_loss_val = loss.item(), score_loss.item(), geo_loss.item()
            # 记录每个 batch 的训练损失到 TensorBoard
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
        print(f"Epoch {epoch} 训练结束: Loss: {avg_loss:.4f} (Score: {avg_score_loss:.4f}, Geo: {avg_geo_loss:.4f})")
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch)

        # --- 验证步骤 ---
        should_validate = False
        if epoch >= START_DENSE_VAL_EPOCH:
            should_validate = True
        elif epoch % VAL_INTERVAL == 0:
            should_validate = True
        if val_loader and should_validate:
            val_loss, val_score, val_geo = validate(model, val_loader, criterion, device, writer, epoch)
            print(f"Epoch {epoch} Val Loss: {val_loss:.4f} (Score: {val_score:.4f}, Geo: {val_geo:.4f})")
            writer.add_scalars('epoch/loss', {'train': avg_loss, 'validation': val_loss}, epoch)
            writer.add_scalars('epoch/score Loss', {'train': avg_score_loss, 'validation': val_score}, epoch)
            writer.add_scalars('epoch/geo Loss', {'train': avg_geo_loss, 'validation': val_geo}, epoch)
            # 只有当列表未满，或者当前 loss 小于列表中最差的 loss 时，才保存
            if len(best_checkpoints) < BEST_MODELS_TO_KEEP or val_loss < best_checkpoints[-1][0]:
                save_best_checkpoint(model, epoch, val_loss, best_checkpoints)
        else:
            # 如果没验证，只记录训练 loss
            writer.add_scalars('epoch/loss_train', {'train': avg_loss}, epoch)

        # --- 常规检查点保存
        checkpoint_dir = "detection/checkpoints/"
        if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
        checkpoint_filename = os.path.join(checkpoint_dir, f"detection_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_filename)
        regular_checkpoints.append(checkpoint_filename)
        if len(regular_checkpoints) > MAX_KEEP_CHECKPOINTS:
            oldest = regular_checkpoints.pop(0)
            if os.path.exists(oldest):
                os.remove(oldest)

    writer.close()
    print("训练完成!")

if __name__ == '__main__':
    if not os.path.exists(TRAIN_IMAGE_DIR) or not os.path.exists(TRAIN_LABEL_DIR):
        print(f"错误: 请确保你的数据分别在 '{TRAIN_IMAGE_DIR}' 和 '{TRAIN_LABEL_DIR}' 目录下")
    else:
        main()
