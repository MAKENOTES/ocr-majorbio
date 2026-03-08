watch -n 1 nvidia-smi

pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130


torch==2.9.0.*
torchvision==0.24.0.*
torchaudio==2.9.0.*
opencv-python==4.12.0.88
Pillow==12.0.0
tqdm==4.67.1

pip install albumentations


tensorboard
tensorboard --logdir=detection/runs
tensorboard --logdir=recognition/runs
标注数据优化
训练优化方案
    1、调整权重3--->2  学习率1e-5  epoch 至少1500+
    2、调整iou 算法  diou
    3、两个阶段训练，先1e-4 权重3--->1  得到小的score， 然后调追加训练 整权重3--->2  学习率1e-5  降低geo
    4、调整模型构建



def diou_loss(self, pred_aabb, true_aabb):
        """
        计算 DIoU Loss
        输入形状: (M, 4) -> top, right, bottom, left
        """
        eps = 1e-6

        # 1. 拆分边界距离
        d_top_p, d_right_p, d_bottom_p, d_left_p = pred_aabb[:, 0], pred_aabb[:, 1], pred_aabb[:, 2], pred_aabb[:, 3]
        d_top_t, d_right_t, d_bottom_t, d_left_t = true_aabb[:, 0], true_aabb[:, 1], true_aabb[:, 2], true_aabb[:, 3]

        # 2. 计算 IoU (这部分与标准 IoU 一样)
        # 交集
        inter_w = torch.min(d_left_p, d_left_t) + torch.min(d_right_p, d_right_t)
        inter_h = torch.min(d_top_p, d_top_t) + torch.min(d_bottom_p, d_bottom_t)
        inter_area = inter_w * inter_h
        
        # 并集
        area_pred = (d_left_p + d_right_p) * (d_top_p + d_bottom_p)
        area_true = (d_left_t + d_right_t) * (d_top_t + d_bottom_t)
        union_area = area_pred + area_true - inter_area
        union_area = torch.clamp(union_area, min=eps)
        
        iou = (inter_area + eps) / union_area

        # 3. 计算 DIoU 的惩罚项
        
        # --- A. 计算两个框的中心点 (相对于当前像素的坐标) ---
        # 设当前像素为原点 (0,0)
        # 预测框中心: x = (right - left)/2, y = (bottom - top)/2
        # 注意方向：通常 top/bottom 指的是距离，所以 y 轴方向需要统一逻辑，
        # 但这里计算的是距离差，只要统一即可。
        # 假设 top 是向上，bottom 是向下：
        # center_y_pred = (d_bottom_p - d_top_p) / 2.0
        # center_x_pred = (d_right_p - d_left_p) / 2.0
        
        # center_y_true = (d_bottom_t - d_top_t) / 2.0
        # center_x_true = (d_right_t - d_left_t) / 2.0
        
        # 中心点欧氏距离的平方 (rho^2)
        # Simplify: ( (db_p - dt_p)/2 - (db_t - dt_t)/2 )^2 = 1/4 * ( (db_p - dt_p) - (db_t - dt_t) )^2
        center_dist_sq = 0.25 * (
            (d_bottom_p - d_top_p - (d_bottom_t - d_top_t))**2 + 
            (d_right_p - d_left_p - (d_right_t - d_left_t))**2
        )

        # --- B. 计算最小闭包区域 (Smallest Enclosing Box) 的对角线平方 (c^2) ---
        # 闭包宽 = max(左) + max(右)
        enclose_w = torch.max(d_left_p, d_left_t) + torch.max(d_right_p, d_right_t)
        enclose_h = torch.max(d_top_p, d_top_t) + torch.max(d_bottom_p, d_bottom_t)
        
        # 对角线平方
        diag_dist_sq = enclose_w**2 + enclose_h**2
        
        # 防止除零
        diag_dist_sq = torch.clamp(diag_dist_sq, min=eps)

        # 4. 计算 DIoU Loss
        # DIoU = IoU - (rho^2 / c^2)
        # Loss = 1 - DIoU = 1 - IoU + (rho^2 / c^2)
        diou_loss_value = 1.0 - iou + (center_dist_sq / diag_dist_sq)

        return diou_loss_value

120
150
160
240
340
380



44
152
196
248
288