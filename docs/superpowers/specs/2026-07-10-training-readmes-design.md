# 检测与识别训练 README 设计

## 目标

在 `detection/README.md` 和 `recognition/README.md` 中准确梳理当前训练代码的完整执行过程，使维护者能够理解配置、数据、模型、损失、优化、验证、日志和 checkpoint 的关系，并提供可执行、可验证的训练流程优化方案。

## 文档原则

1. 只描述仓库当前代码能够证明的行为，不把注释中的设想写成已实现功能。
2. 将“当前流程”“已知风险”“优化建议”分开，避免读者混淆现状与规划。
3. 复杂流程使用 Mermaid，普通配置和输入输出契约使用表格或列表。
4. 优化建议按 P0、P1、P2 排序，每项说明问题依据、修改位置、建议做法、验证指标和预期收益。
5. 本任务只修改两个 README，不修改训练代码、超参数、数据、模型或 checkpoint。

## Detection README 结构

### 项目定位与入口

说明 `detection/train_resnet.py` 使用 ResNet18 + FPN 风格融合生成 1/4 尺度的 score map 和四通道 geometry map，并列出其直接依赖：

- `components/dataset.py`
- `components/transforms.py`
- `components/ground_truth.py`
- `models/east_model.py`
- `components/loss.py`

### 配置和输入契约

用表格记录训练脚本当前常量，包括数据目录、输入尺寸、batch size、epoch、学习率、验证频率、早停和 checkpoint 保留数量。说明标签文件名映射规则 `图片名 -> gt_图片名.txt`，以及当前标签行按 `x_min y_min x_max y_max [text]` 解析。

### 数据与张量流程

使用 Mermaid 展示：

```text
图片/标签 -> OCRDataset -> 训练增强 -> ResizeAndPad
-> GenerateGT -> ToTensor -> DataLoader
-> EASTModel -> score/geometry -> EASTLoss
```

补充关键张量形状：输入 `[B,3,640,640]`，输出和真值 score `[B,1,160,160]`，geometry `[B,4,160,160]`。解释四边形收缩、正样本 mask、top/right/bottom/left 距离和 `log1p` 几何目标。

### 训练、验证和保存

逐步说明 optimizer 清零、前向、Dice 与 masked Smooth L1、反向、梯度记录、裁剪、Adam step、OneCycleLR step、epoch 汇总。使用 Mermaid 展示验证频率切换、Top-K 保存、早停和常规 checkpoint 的控制流。

### 当前风险与优化建议

明确记录以下代码现状：

- 模型 geometry head 没有显式非负激活，而损失注释称输出经过 ReLU。
- ground truth 使用 `log1p`，需要确认推理恢复与损失空间一致。
- 没有验证集时训练结束摘要会格式化 `None` 风险。
- 常规 checkpoint 仅在本次进程内维护列表，重启后不会继续清理历史文件。
- checkpoint 只保存模型权重，不能完整恢复 optimizer、scheduler、epoch 和早停状态。
- 随机水平裁剪按中心保留框并裁剪坐标，需要用可视化确认边界目标质量。

优化方案按以下优先级组织：

- P0：修正验证集缺失路径、统一 geometry 输出/目标/推理契约、增加非有限损失保护和可恢复 checkpoint。
- P1：加入检测指标驱动的模型选择、数据与 ground truth 单元测试、可复现实验设置、AMP 与日志降采样。
- P2：分阶段训练、骨干冻结/解冻、损失权重实验、增强消融和困难样本采样。

## Recognition README 结构

### 项目定位与入口

说明 `recognition/train.py` 使用 MobileNet 风格 CNN、单层双向 GRU、线性分类头和 CTC Loss，并列出其直接依赖：

- `components/dataset.py`
- `components/converter.py`
- `models/model.py`
- `keys.txt`

### 配置和输入契约

用表格记录标注路径、字典、图像高度、兼容性宽度参数、通道数、batch size、epoch、学习率、验证/保存起点和 Top-K。说明标注文件以首个逗号分隔路径，其余字段重新拼接为标签。

### 数据与张量流程

使用 Mermaid 展示：

```text
标注 -> TextRecognitionDataset -> 去红章/缩放/增强
-> BucketBatchSampler -> dynamic padding
-> Mobile CNN -> padding mask -> BiGRU -> CTC head
-> CTCLoss / decode -> Loss、Accuracy、CER
```

补充关键张量形状：动态输入 `[B,1,32,Wmax]`，CNN 输出 `[B,256,1,W/4]`，序列 `[T,B,256]`，最终输出 `[T,B,C]`。解释 `keys.txt` 中 blank、UNK 和普通字符索引，以及有效宽度对 mask 和 CTC input length 的作用。

### 训练、验证和保存

逐步说明标签编码、前向、CTC Loss、反向、梯度裁剪与监控、AdamW、epoch 汇总、验证解码、Accuracy/CER/Loss、ReduceLROnPlateau 和周期/Top-K checkpoint。使用 Mermaid 展示保存条件和指标流。

### 当前风险与优化建议

明确记录以下代码现状：

- `TRAIN_ANNOTATION` 与 `VALID_ANNOTATION` 当前指向同一文件，验证指标存在数据泄漏，且训练与验证预处理不同。
- 训练脚本使用 `img_widths // 4` 计算 CTC input length，而模型实现了更精确的 `_calc_cnn_output_width`。
- 非有限 loss 只在日志累计时跳过，但反向传播和 optimizer step 已经执行。
- 图片读取异常时返回 dummy tensor，同时保留原标签，可能污染训练。
- `IMG_WIDTH` 参数目前只为兼容保留，实际宽度由动态 padding 决定。
- scheduler 注释和 Top-K 注释仍称 Valid Loss，但实际使用 CER。
- checkpoint 只保存模型权重，不能完整恢复训练状态。

优化方案按以下优先级组织：

- P0：拆分独立验证集、统一有效序列长度计算、在反向前拦截非有限 loss、处理坏图和实现完整断点续训。
- P1：增加标签/字典审计、按字符数与宽度分桶、AMP、固定随机种子、CER 与长文本分层指标。
- P2：增强消融、难例重采样、语言约束解码、模型容量和 dropout 对照实验。

## 验证

文档完成后执行以下检查：

1. Mermaid 代码块数量和 `flowchart` 语法检查。
2. README 中出现的文件路径、类名、函数名和配置常量均能在仓库中检索到。
3. 当前风险逐项与源码核对，优化建议不得声称已实现。
4. 检查标题层级、占位符、尾随空格和 `git diff --check`。
5. 只暂存两个 README 及本任务设计/计划文件，不纳入其他工作区改动。
