_base_ = [
    # DBNet 主干与默认组件配置
    '../third_party/mmocr/configs/textdet/dbnet/_base_dbnet_resnet18_fpnc.py',
    # 默认运行时（日志、Hook 等）
    '../third_party/mmocr/configs/textdet/_base_/default_runtime.py',
    # 训练计划（SGD + PolyLR）
    '../third_party/mmocr/configs/textdet/_base_/schedules/schedule_sgd_1200e.py',
]

train_pipeline = [
    # 读取图像（忽略 EXIF 旋转）
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    # 读取标注（bbox + polygon）
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
    ),
    # 轻量亮度/饱和度增强（不含旋转）
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=16.0 / 255,
        saturation=0.2),
    # 轻量遮挡增强（模拟印章遮挡文字）
    dict(
        type='ImgAugWrapper',
        args=[dict(cls='CoarseDropout', p=0.15, size_percent=[0.02, 0.08])]),
    # 轻微缩放增强（不含翻转/旋转）
    dict(
        type='ImgAugWrapper',
        args=[['Resize', [0.8, 1.2]]]),
    # 缩放到固定尺寸并填充，保持输入尺度稳定
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640)),
    # 打包输入给模型
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape'))
]

test_pipeline = [
    # 测试/验证不做增强
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1333, 736), keep_ratio=True),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
    ),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# 数据根目录，这里使用项目根路径
data_root = '.'

# 使用矩形框时，仍以 polygon 形式输入（由 bbox 转成四点矩形）
# DBNet 训练依赖 polygon 生成监督图，因此这里保持 with_polygon=True

# 训练集配置（使用转换后的 MMOCR JSON）
train_dataset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='data/mmocr/textdet_train.json',
    # 适配小字场景，降低最小尺寸过滤阈值
    filter_cfg=dict(filter_empty_gt=True, min_size=10),
    pipeline=train_pipeline,
)

# 验证集配置
val_dataset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='data/mmocr/textdet_val.json',
    test_mode=True,
    pipeline=test_pipeline,
)

# 数据加载参数
train_dataloader = dict(
    # 训练 batch size（显存不足可降为 4 或 2）
    batch_size=8,
    # 数据加载进程数
    num_workers=4,
    # 训练期间保持 worker 常驻
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset,
)

val_dataloader = dict(
    # 验证 batch size
    batch_size=1,
    # 验证数据加载进程数
    num_workers=2,
    # 验证期间保持 worker 常驻
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset,
)

test_dataloader = val_dataloader

# 自动学习率缩放基准
auto_scale_lr = dict(base_batch_size=8)

# 降低训练轮数，适配小数据/低成本训练
train_cfg = dict(
    type='EpochBasedTrainLoop',
    # 总训练轮数（需要多少 epoch 就改这里）
    max_epochs=300,
    # 每隔多少轮做一次验证
    val_interval=10,
)
param_scheduler = [
    dict(
        # Poly 学习率衰减
        type='PolyLR',
        power=0.9,
        # 最小学习率
        eta_min=1e-7,
        # 结束 epoch（需与 max_epochs 对齐）
        end=300,
    ),
]

# 推理/验证阈值（偏向提升召回）
model = dict(
    det_head=dict(
        postprocessor=dict(
            # 输出四点矩形（不需要任意多边形）
            text_repr_type='quad',
            # 二值化阈值，降低可提高召回
            mask_thr=0.2,
            # 最低文本分数阈值，降低可提高召回
            min_text_score=0.2,
            # 最小文本宽度，适当降低避免过滤过多小框
            min_text_width=3,
        )
    )
)
