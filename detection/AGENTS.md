# Detection 模块指南

## 作用域

本文件仅适用于 `detection/` 目录，并继承仓库根目录 `AGENTS.md` 的全部规则。若两者存在表述差异，以根目录的全局约束和用户当前明确要求为准。

## 模块概述

本模块实现 EAST 风格的文本区域检测。核心输出通常包括文本区域 `score map` 和描述文本框边界的 `geometry map`，再经阈值筛选、几何恢复和 NMS 等后处理得到四边形文本框。

## 模块结构

- `models/`：EAST 网络及 ResNet18、ResNet50、MobileNet 等骨干变体。
- `components/dataset.py`：检测数据索引、图像读取和样本组织。
- `components/transforms.py`：训练与验证阶段的图像、文本框联合变换。
- `components/ground_truth*.py`：不同模型尺度对应的 score map 和 geometry map 生成逻辑。
- `components/loss.py`：分类分数和文本框几何损失。
- `components/eval_metrics.py`：四边形匹配及检测指标计算。
- `train_resnet*.py`、`train_mobilenet.py`：不同骨干网络的训练入口。
- `inference*.py`：PyTorch 检测推理和可视化入口。
- 根目录 `detection_export_onnx*.py`、`detection_quantize_int8*.py`、`detection_inference*.py`：导出、量化及 ONNX/OpenVINO 推理入口。

## 模型与数据契约

- 修改模型时确认所选骨干网络、ground-truth 生成尺度、训练脚本、checkpoint 加载、导出脚本和推理脚本属于同一变体。
- 标签文件中的四边形顶点顺序、坐标系和图像文件对应关系属于数据契约。不得在未同步上下游的情况下改变解析顺序或坐标含义。
- 旋转、缩放、裁剪、padding 和 resize 必须对图像与四边形坐标执行一致变换，并保持 score map、geometry map 与模型输出分辨率一致。
- 几何距离的方向和通道顺序必须与 ground-truth、损失函数和后处理恢复算法一致。调整 IoU、DIoU 或其他几何损失时，先确认通道语义，不只比较张量形状。
- 空标注、无效坐标、退化多边形和完全位于裁剪区域外的文本框必须有明确处理路径，不能生成 NaN、负面积或越界索引。
- checkpoint 可能保存为纯 `state_dict` 或包含 `model_state_dict`、`state_dict` 等字段的字典。修改加载逻辑时保持现有格式兼容，并明确使用的模型类。

## 修改要求

- 修改 `models/` 后同步检查对应训练、PyTorch 推理、ONNX 导出及量化入口的输入尺寸和输出结构。
- 修改 `components/transforms.py` 或 `components/ground_truth*.py` 后，使用可人工核对的少量样本检查变换后的四边形、score map 和 geometry map 是否对齐。
- 修改 `components/loss.py` 时检查空正样本、极小框、数值稳定项和反向传播，避免除零及非有限损失。
- 修改后处理、阈值或缩放逻辑时，分别检查原始图像尺寸、模型输入尺寸和输出特征图尺寸，避免重复缩放坐标。
- 修改检测核心流程时，按根目录规则创建或更新 `ocr_detection_plan_current.md`，记录采用的模型变体、数据契约、关键参数和验证结果。
- 训练超参数继续使用脚本顶部常量配置。除非用户明确要求，不引入命令行参数，也不擅自开始长时间训练。

## 导出与量化

- 导出前将模型切换为评估模式，并使用与目标推理入口一致的输入通道数、尺寸和预处理。
- 导出后至少检查模型可加载，输入输出名称、数据类型、静态或动态维度以及输出特征图形状符合推理代码预期。
- 量化校准数据必须复用生产推理的预处理语义。不得为了让量化命令运行而静默改变颜色空间、归一化或 resize 方式。
- 验证导出和量化逻辑时输出到新的临时路径，不覆盖现有 `.onnx`、`.xml`、`.bin` 或 INT8 文件。

## 验证要求

根据改动内容选择最小但有针对性的验证：

- 数据或几何变换：覆盖有标注、空标注、退化多边形、不同宽高比和裁剪边界样本。
- 模型或损失：使用小型合成张量检查输出形状、有限损失和反向传播；同时确认 CPU 行为，涉及 CUDA 时再补充 GPU 检查。
- 后处理：覆盖无候选框、单框、多框重叠和不同原图尺寸。
- 导出或量化：检查产物可加载，并用同一输入比较 PyTorch 与目标运行时的输出形状及基本数值一致性。
- 文档或纯配置：至少执行引用检查和 `git diff --check`。

除非用户明确要求并已确认资源，不运行完整训练。缺少数据集、checkpoint、CUDA 或目标运行时时，应报告对应验证未执行及原因。
