## 3. Detection 阶段（初始要求）

### 已完成
- 在 `.venv` 中安装 docTR：`pip install python-doctr`。
- 更新依赖清单：`req.txt` 中新增 `python-doctr`，并放宽 `opencv-python` 版本约束到 `<5`。

### 训练实现
- 使用 docTR 的 `db_resnet50` 作为默认检测模型。
- 使用 `data/images` + `data/det_labels`（`gt_` 前缀）加载 ICDAR 标注，并在训练脚本中转换为 docTR 需要的归一化多边形格式。
- 训练入口：`detection/train.py`（训练参数固定在代码中的配置函数内，不使用命令行传参）。
- 训练脚本已补充中文注释，覆盖数据解析、归一化处理、训练流程与保存逻辑。
- 新增训练/验证日志输出与验证流程，验证集路径可在配置中单独指定。
- 模型按每 20 个 epoch 保存一次。

### 训练执行（已跑通）
- 在 `.venv` 中执行：`python detection/train.py`，输出模型 `outputs/doctr_det/doctr_det_epoch_1.pth`。

### 规则遵循
- 已确认遵循 `context.txt` 规则：中文回复、中文注释、依赖与执行在 `.venv`、核心步骤及时记录、训练脚本不使用命令行传参、函数注释使用正规多行说明。
