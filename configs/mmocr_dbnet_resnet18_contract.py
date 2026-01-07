_base_ = [
    '../third_party/mmocr/configs/textdet/dbnet/_base_dbnet_resnet18_fpnc.py',
    '../third_party/mmocr/configs/textdet/_base_/default_runtime.py',
    '../third_party/mmocr/configs/textdet/_base_/schedules/schedule_sgd_1200e.py',
]

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
    ),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640)),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape'))
]

test_pipeline = [
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

data_root = '.'

train_dataset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='data/mmocr/textdet_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
)

val_dataset = dict(
    type='OCRDataset',
    data_root=data_root,
    ann_file='data/mmocr/textdet_val.json',
    test_mode=True,
    pipeline=test_pipeline,
)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset,
)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=8)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=10)
param_scheduler = [
    dict(type='PolyLR', power=0.9, eta_min=1e-7, end=200),
]
