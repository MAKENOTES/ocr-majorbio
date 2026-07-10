# 分级 AGENTS.md 初始化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建根目录及检测、识别模块的分级代理协作指南。

**Architecture:** 根目录 `AGENTS.md` 定义整个仓库的共同约束；`detection/AGENTS.md` 和 `recognition/AGENTS.md` 继承全局规则，只补充各模块特有的结构、模型契约和验证要求。所有文件仅包含文档，不改变运行时代码。

**Tech Stack:** Markdown、Python 3、PyTorch、ONNX、OpenVINO、Git

## Global Constraints

- 所有回复使用中文。
- 所有新增代码注释和函数文档使用中文。
- 所有 Python 命令和依赖安装必须在项目虚拟环境中执行。
- 依赖变化必须同步更新 `requirements.txt` 和相关说明。
- 训练脚本和工具使用文件内常量配置，不新增命令行参数，除非用户明确要求。
- 涉及检测核心流程时创建或更新 `ocr_detection_plan_current.md`。
- 不提交或覆盖数据集、checkpoint、运行日志及生成模型。
- 不启动完整训练任务作为常规验证手段。

---

### Task 1: 创建仓库级协作指南

**Files:**
- Create: `AGENTS.md`

**Interfaces:**
- Consumes: `context.txt` 中的既有协作约束，以及仓库根目录脚本和 `requirements.txt`。
- Produces: `detection/AGENTS.md` 与 `recognition/AGENTS.md` 共同继承的全局规则。

- [x] **Step 1: 创建根目录说明**

写入以下章节和明确规则：

```markdown
# AGENTS.md

## 项目概述
说明检测、识别、导出和量化流程，以及根目录执行约定。

## 仓库结构
列出 detection、recognition、根目录推理/导出/量化脚本、requirements.txt。

## 通用协作规则
写入中文交流、虚拟环境、依赖同步、文件内常量配置、正式中文 docstring、检测计划更新要求。

## 开发与验证
要求先做针对性检查，禁止用完整训练作为常规测试，并如实报告缺失的数据、模型或设备。

## 数据和生成文件
保护数据集、checkpoint、runs、ONNX、OpenVINO IR 及 INT8 文件，避免覆盖现有产物。

## Git 约束
保持改动聚焦，不回滚他人修改，提交前运行 git diff --check 和 git status --short。
```

- [x] **Step 2: 检查根目录指南内容**

Run: `rg -n "虚拟环境|requirements.txt|命令行参数|ocr_detection_plan_current.md|git diff --check" AGENTS.md`

Expected: 每项全局约束至少匹配一次，且措辞与 `context.txt` 不冲突。

- [x] **Step 3: 提交根目录指南**

```bash
git add AGENTS.md
git commit -m "docs: 添加仓库级代理指南"
```

### Task 2: 创建检测模块指南

**Files:**
- Create: `detection/AGENTS.md`

**Interfaces:**
- Consumes: 根目录 `AGENTS.md`、`detection/models/`、`detection/components/`、现有训练和推理脚本。
- Produces: 检测模块内修改所需的 EAST 数据、模型、损失、导出和验证约束。

- [x] **Step 1: 创建检测模块说明**

写入以下章节和明确规则：

```markdown
# Detection 模块指南

## 作用域
声明继承根目录 AGENTS.md，仅覆盖 detection 目录。

## 模块结构
说明 models、components、训练、推理和文档文件职责。

## 模型与数据约束
保持模型变体配套；保持四边形坐标、score map、geometry map 和缩放逻辑一致；保护数据及 checkpoint。

## 修改要求
修改损失、数据增强、后处理、导出或量化时检查上下游契约，并更新检测计划。

## 验证要求
覆盖空标注、退化多边形、不同尺寸、CPU/GPU；导出后检查输入输出名称和形状；不启动完整训练。
```

- [x] **Step 2: 检查检测指南引用**

Run: `rg -n "models/|components/|score map|geometry map|ocr_detection_plan_current.md|完整训练" detection/AGENTS.md`

Expected: 每个检测模块关键约束至少匹配一次。

- [x] **Step 3: 提交检测模块指南**

```bash
git add detection/AGENTS.md
git commit -m "docs: 添加检测模块代理指南"
```

### Task 3: 创建识别模块指南

**Files:**
- Create: `recognition/AGENTS.md`

**Interfaces:**
- Consumes: 根目录 `AGENTS.md`、`recognition/models/`、`recognition/components/`、`recognition/utils/`、`recognition/keys.txt`。
- Produces: 识别模块内修改所需的 CRNN/CTC、字典、标签、动态宽度和导出约束。

- [x] **Step 1: 创建识别模块说明**

写入以下章节和明确规则：

```markdown
# Recognition 模块指南

## 作用域
声明继承根目录 AGENTS.md，仅覆盖 recognition 目录。

## 模块结构
说明 models、components、utils、训练、评估和文档文件职责。

## 模型与数据约束
将 keys.txt 字符顺序定义为模型契约；保持图像宽度、序列长度、CTC target length 和解码一致；规范化时关注符号、空格、下划线与印章干扰。

## 修改要求
字典变化必须检查类别数、checkpoint 和解码兼容性；动态 padding、批采样和标签转换变化必须联动验证。

## 验证要求
覆盖不同宽度、批量大小、未知字符和空标签；导出后检查动态宽度轴及解码；不启动完整训练。
```

- [x] **Step 2: 检查识别指南引用**

Run: `rg -n "keys.txt|CTC|动态|未知字符|checkpoint|完整训练" recognition/AGENTS.md`

Expected: 每个识别模块关键约束至少匹配一次。

- [x] **Step 3: 提交识别模块指南**

```bash
git add recognition/AGENTS.md
git commit -m "docs: 添加识别模块代理指南"
```

### Task 4: 验证分级指南

**Files:**
- Verify: `AGENTS.md`
- Verify: `detection/AGENTS.md`
- Verify: `recognition/AGENTS.md`

**Interfaces:**
- Consumes: 三份已创建的指南及 `context.txt`。
- Produces: 无占位符、无空白符错误、作用域清晰的最终文档集合。

- [x] **Step 1: 检查文件和作用域声明**

Run: `rg -n "根目录.*AGENTS.md|适用于整个仓库|仅适用于" AGENTS.md detection/AGENTS.md recognition/AGENTS.md`

Expected: 根目录文件声明全局作用域，两个子目录文件声明继承关系和局部作用域。

- [x] **Step 2: 检查文档质量**

Run: `rg -n "TBD|TODO|待补充|稍后实现" AGENTS.md detection/AGENTS.md recognition/AGENTS.md`

Expected: 无输出。

Run: `git diff --check`

Expected: 无输出，退出码为 0。

- [x] **Step 3: 检查最终状态**

Run: `git status --short`

Expected: 实施计划文件可能尚未提交；三份 `AGENTS.md` 均已提交，不存在其他意外修改。
