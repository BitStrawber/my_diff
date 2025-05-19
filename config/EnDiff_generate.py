# config/EnDiff_generate.py

_base_ = [
    './_base_/default_runtime.py'
]

# 模型设置
model = dict(
    type='EnDiff',
    net=dict(
        type='PM',
        channels=16,
        time_channels=16
    ),
    T=1000,
    diffuse_ratio=0.6,
    sample_times=15,
    land_loss_weight=1,
    uw_loss_weight=10
)

# 数据集设置
dataset_type = 'HqLqCocoDataset'
data_root = './data/mydata/'
hq_img_prefix = './data/enhance/'
test_ann = './data/mydata/annotations/instances_test.json'
classes = ['echinus', 'holothurian', 'scallop', 'starfish']

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadLqHqImages'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1333, 800)],
        flip=True,
        transforms=[
            dict(type='ResizeLqHqImages', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'hq_img']),
            dict(type='Collect', keys=['img', 'hq_img'], meta_keys=['filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'hq_img_filename'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    test=dict(
        type=dataset_type,
        hq_img_prefix=hq_img_prefix,
        ann_file=test_ann,
        classes=classes,
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline
    )
)

# 评估设置
evaluation = None

# 混合精度训练
fp16 = None