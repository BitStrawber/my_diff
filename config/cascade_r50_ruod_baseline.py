_base_ = [
    './_base_/models/cascade_rcnn_r50_fpn.py',
    './_base_/schedules/schedule_2x.py',
    './_base_/default_runtime.py'
]

# RUOD dataset - 修改为你的实际路径
data_root = '/path/to/data/RUOD_pic/'
train_ann = '/path/to/data/RUOD_ANN/instances_train.json'
test_ann = '/path/to/data/RUOD_ANN/instances_test.json'
classes = [
    'holothurian', 'echinus', 'scallop', 'starfish', 'fish', 'corals',
    'diver', 'cuttlefish', 'turtle', 'jellyfish'
]
num_classes = 10

# Model: standard Cascade RCNN, no diffusion
model = dict(
    init_cfg=dict(
        type='Pretrained',
        checkpoint='torchvision://resnet50'
    ),
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ])
)

# Data: 8 GPUs, bs=2 per GPU (total BS=16), image size 1920x1080
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1920, 1080), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug', img_scale=(1920, 1080), flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ]),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        pipeline=train_pipeline,
        ann_file=train_ann,
        img_prefix=data_root + 'images/',
        classes=classes),
    val=dict(
        pipeline=test_pipeline,
        ann_file=test_ann,
        img_prefix=data_root + 'images/',
        classes=classes),
    test=dict(
        pipeline=test_pipeline,
        ann_file=test_ann,
        img_prefix=data_root + 'images/',
        classes=classes))

# LR: 8 GPUs, total BS=16, same as official mmdetection
auto_scale_lr = dict(enable=False, base_batch_size=16)
evaluation = dict(interval=1, save_best='auto', classwise=True)
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=24)
