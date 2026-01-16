import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import OCRDataset
from utils.transforms import get_transforms
from models.east_model import EASTModel
from utils.loss import EASTLoss

# --- 超参数 ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 4 # 根据你的 GPU 显存调整
EPOCHS = 100
INPUT_SIZE = 512
TRAIN_IMAGE_DIR = "data/train_images"
TRAIN_LABEL_DIR = "data/train_labels"
# 可选: 添加验证集数据目录
# VAL_IMAGE_DIR = "data/val_images"
# VAL_LABEL_DIR = "data/val_labels"

def main():
    """
    主训练函数。
    该函数负责设置设备、加载数据、初始化模型、定义损失函数和优化器，
    并执行主训练循环。
    """
    # --- 设备设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 数据加载 ---
    train_transform = get_transforms(is_train=True, input_size=INPUT_SIZE)
    train_dataset = OCRDataset(
        image_dir=TRAIN_IMAGE_DIR,
        label_dir=TRAIN_LABEL_DIR,
        transform=train_transform
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 可选: 创建验证集加载器
    # val_transform = get_transforms(is_train=False, input_size=INPUT_SIZE)
    # val_dataset = OCRDataset(...)
    # val_loader = DataLoader(...)

    # --- 模型、损失函数和优化器 ---
    model = EASTModel(pretrained=True).to(device)
    criterion = EASTLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 训练循环 ---
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_score_loss = 0
        epoch_geo_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            images = batch['image'].to(device)
            true_score_maps = batch['score_map'].to(device).unsqueeze(1) # 增加通道维度
            # 已修正: ToTensor 转换已产生正确的 (C, H, W) 形状,
            # DataLoader 将其批处理为 (B, C, H, W)。不再需要 permute 操作。
            true_geo_maps = batch['geo_map'].to(device)

            # 清零参数梯度
            optimizer.zero_grad()
            
            # 前向传播 会调用 模型的 forward 方法
            pred_score_maps, pred_geo_maps = model(images)
            
            # 计算损失
            loss, score_loss, geo_loss = criterion(
                true_score_maps, pred_score_maps,
                true_geo_maps, pred_geo_maps
            )
            
            # 反向传播和优化 计算梯度
            loss.backward()
            # 学习发生的关键时刻, 每一个 batch,  优化器都会根据计算出的梯度微调模型的权重
            optimizer.step()
            
            # 更新 epoch 的总损失
            epoch_loss += loss.item()
            epoch_score_loss += score_loss.item()
            epoch_geo_loss += geo_loss.item()
            
            # 更新进度条显示
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'score_loss': f"{score_loss.item():.4f}",
                'geo_loss': f"{geo_loss.item():.4f}"
            })
            
        # 打印 epoch 的统计数据
        avg_loss = epoch_loss / len(train_loader)
        avg_score_loss = epoch_score_loss / len(train_loader)
        avg_geo_loss = epoch_geo_loss / len(train_loader)
        print(f"Epoch {epoch+1} 结束: avg_loss: {avg_loss:.4f}, avg_score_loss: {avg_score_loss:.4f}, avg_geo_loss: {avg_geo_loss:.4f}")

        # --- 验证步骤 (可选) ---
        # model.eval()
        # with torch.no_grad():
        #     # ... 此处编写验证逻辑 ...
        
        # --- 保存模型检查点 ---
        # 最佳实践是在验证集上性能有提升后才保存模型
        torch.save(model.state_dict(), f"east_model_epoch_{epoch+1}.pth")
        print(f"已保存模型检查点至 east_model_epoch_{epoch+1}.pth")

    print("训练完成!")

if __name__ == '__main__':
    # 确保你的数据目录存在
    import os
    if not os.path.exists(TRAIN_IMAGE_DIR) or not os.path.exists(TRAIN_LABEL_DIR):
        print(f"错误: 请确保你的数据分别在 '{TRAIN_IMAGE_DIR}' 和 '{TRAIN_LABEL_DIR}' 目录下")
    else:
        main()
