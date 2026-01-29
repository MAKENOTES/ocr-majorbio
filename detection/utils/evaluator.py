import cv2
import numpy as np

class Evaluator:
    @staticmethod
    def get_boxes_from_maps(score_map, geo_map, score_thresh=0.8, nms_thresh=0.2):
        """
        从 score_map 和 geo_map 中恢复边界框 (支持 ReLU + 4通道模式)
        """
        # score_map: (H, W), geo_map: (4, H, W)
        if score_map.max() < score_thresh:
            return np.array([])

        # 1. 获取满足阈值的像素点
        xy_text = np.argwhere(score_map > score_thresh)
        if len(xy_text) == 0:
            return np.array([])

        # 2. 恢复框
        boxes = []
        scores = []

        # 批量处理以加速 (Vectorized implementation)
        # xy_text[:, 0] is y, xy_text[:, 1] is x
        y = xy_text[:, 0]
        x = xy_text[:, 1]

        scores = score_map[y, x]

        # geo_map shape is (4, H, W), taking values at y, x
        # d_top, d_right, d_bottom, d_left
        geo_d = geo_map[:, y, x]
        d_top = geo_d[0]
        d_right = geo_d[1]
        d_bottom = geo_d[2]
        d_left = geo_d[3]

        # 像素中心对齐 (x + 0.5)
        pixel_x = x + 0.5
        pixel_y = y + 0.5

        # 恢复特征图上的坐标
        x_min = pixel_x - d_left
        x_max = pixel_x + d_right
        y_min = pixel_y - d_top
        y_max = pixel_y + d_bottom

        # 映射回原图尺度 (Stride = 4)
        boxes = np.stack([x_min, y_min, x_max, y_max], axis=1) * 4.0

        # 3. NMS 后处理
        # cv2.dnn.NMSBoxes 需要 [x, y, w, h] 格式
        x_min = boxes[:, 0]
        y_min = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        # 构造 input for NMS
        boxes_xywh = np.stack([x_min, y_min, w, h], axis=1).tolist()
        scores_list = scores.tolist()

        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores_list, score_thresh, nms_thresh)

        if len(indices) > 0:
            indices = indices.flatten()
            return boxes[indices]  # 返回 [N, 4] (x1, y1, x2, y2)
        else:
            return np.array([])

    @staticmethod
    def calculate_iou(box1, box2):
        """计算两个矩形框的 IoU (Box格式: x1, y1, x2, y2)"""
        xx1 = max(box1[0], box2[0])
        yy1 = max(box1[1], box2[1])
        xx2 = min(box1[2], box2[2])
        yy2 = min(box1[3], box2[3])

        w = max(0.0, xx2 - xx1)
        h = max(0.0, yy2 - yy1)
        inter = w * h

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / (union + 1e-6)

    @staticmethod
    def compute_batch_metrics(pred_boxes_list, gt_boxes_list, iou_thresh=0.5):
        """
        计算一个 Batch 的 TP, FP, FN
        pred_boxes_list: list of np.array, 每個元素是一张图的预测框
        gt_boxes_list: list of np.array, 每個元素是一张图的真值框
        """
        tp_all = 0
        fp_all = 0
        fn_all = 0

        for pred_boxes, gt_boxes in zip(pred_boxes_list, gt_boxes_list):
            # 处理空数据情况
            if len(gt_boxes) == 0:
                fp_all += len(pred_boxes)
                continue
            if len(pred_boxes) == 0:
                fn_all += len(gt_boxes)
                continue

            # 标记哪些 GT 已经被匹配了
            gt_matched = np.zeros(len(gt_boxes), dtype=bool)

            # 对每个预测框，寻找最大 IoU 的 GT
            for p_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1

                for i, g_box in enumerate(gt_boxes):
                    if gt_matched[i]:  # 如果这个 GT 已经被其他预测框匹配了，跳过 (一对一匹配)
                        continue

                    iou = Evaluator.calculate_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

                if best_iou >= iou_thresh:
                    tp_all += 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp_all += 1

            # 剩下的未匹配 GT 都是 FN
            fn_all += np.sum(~gt_matched)

        return tp_all, fp_all, fn_all