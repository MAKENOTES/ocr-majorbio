# OCR 项目方案（当前详细版）

> 场景：合同 / 投标文件，版式相对固定，几乎无旋转。
> 重点：降低标注成本、提升召回，便于后续识别与结构化。

---

## 1. 总体目标
- 输入：扫描 PDF / 图片型 PDF。
- 输出：结构化 JSON（页码 + 文字框 + 文本 + 置信度）。
- 架构：两阶段 OCR（检测 + 识别），便于稳定训练与分阶段微调。

---

## 2. 端到端流程
1) **PDF 渲染**（PyMuPDF）
   - 目的：生成高质量页面图（PNG/JPG）。
   - 分辨率建议：300 dpi（小字号可 400 dpi）。

2) **Text Detection（DBNet）**
   - 目的：检测行级文字框。
   - 输出：四点矩形（quad）或 bbox。

3) **Crop 裁剪**
   - 目的：将行框裁剪为识别输入。

4) **Text Recognition（CRNN + CTC）**
   - 目的：识别行文本。

5) **结构化输出**
   - 目的：按阅读顺序排序并导出 JSON。

---

## 3. Detection 方案（当前实现）

### 3.1 模型与框架
- 模型：DBNet（MMOCR 实现）
- 主干：ResNet18 + FPNC
- 输出框类型：`quad`（矩形，方便后处理）

### 3.2 标注与格式
- 设计要求：ICDAR Localization 格式（多边形）。
- 实际情况：你的标注为 **矩形 bbox（x1 y1 x2 y2）**。
- 处理方式：将矩形转换为四点 polygon（训练 DBNet 必需）。

**说明（关键）：**
DBNet 训练需要 polygon 生成文本区域与阈值图。如果只有矩形框，需要先转换成四点 polygon。即使最终输出矩形，也要保留 polygon 参与训练。

### 3.3 数据转换（已实现）
- 工具：`tools/convert_icdar_to_mmocr.py`
- 关键参数：
  - `--bbox_only`：标注仅含 x1 y1 x2 y2 时启用
  - `--label_prefix gt_`：适配你的文件前缀

示例：
```
python tools/convert_icdar_to_mmocr.py \
  --images_dir data/images \
  --labels_dir data/det_labels \
  --label_prefix gt_ \
  --out_train_json data/mmocr/textdet_train.json \
  --out_val_json data/mmocr/textdet_val.json \
  --val_ratio 0.1 \
  --bbox_only
```

---

## 4. Detection 训练配置（当前）
配置文件：`configs/mmocr_dbnet_resnet18_contract.py`

### 4.1 训练策略
- `max_epochs = 300`
- `val_interval = 10`
- `PolyLR end = 300`

### 4.2 数据增强（轻量，无旋转）
- 亮度/饱和度轻微变化（ColorJitter）
- 轻微缩放（0.8 ~ 1.2）
- **不使用旋转**（符合你的场景）

### 4.3 推理阈值（为提高召回）
- `mask_thr = 0.2`
- `min_text_score = 0.2`
- `min_text_width = 3`

**说明（关键）：**
降低阈值可提高召回，但会带来更多误检。建议在业务上用后处理过滤。

---

## 5. Detection 训练执行
```
source .venv/bin/activate
python detection/train_mmocr.py \
  --config configs/mmocr_dbnet_resnet18_contract.py \
  --work_dir outputs/det_preds/mmocr_dbnet
```

如果中断，可恢复：
```
python detection/train_mmocr.py \
  --config configs/mmocr_dbnet_resnet18_contract.py \
  --work_dir outputs/det_preds/mmocr_dbnet \
  --resume
```

---

## 6. 自动选择最佳 Epoch
- 工具：`tools/select_best_epoch.py`
- 逻辑：解析日志中的 `icdar/hmean`，创建 best.pth 软链接。

```
source .venv/bin/activate
python tools/select_best_epoch.py --work_dir outputs/det_preds/mmocr_dbnet
```

---

## 7. 推理验证召回
建议使用低阈值进行验证：
```
source .venv/bin/activate
python third_party/mmocr/tools/test.py \
  configs/mmocr_dbnet_resnet18_contract.py \
  outputs/det_preds/mmocr_dbnet/epoch_XX.pth
```

---

## 8. Recognition 阶段（待实施）
- 模型：CRNN + CTC
- 输入：检测行框裁剪图
- 输出：行级文本
- 训练数据：通过检测框 crop + 手工标注文本

---

## 9. 输出结构
- JSON 字段：`page`, `bbox`, `text`, `score`
- 排序规则：按 y 再 x

---

## 10. 关键注意事项（请重点关注）
1) **DBNet 必须用 polygon 监督**：即便标注是矩形，也必须转为四点 polygon。
2) **阈值影响召回**：低阈值能提高召回，需结合后处理过滤误检。
3) **轻量增强更稳定**：无旋转场景下，不建议加角度增强。
4) **最佳 epoch 不一定是最后一个**：建议自动挑 best。
