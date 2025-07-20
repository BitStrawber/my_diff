_base_ = [
    './_base_/models/cascade_rcnn_r50_fpn.py',
    './_base_/schedules/schedule_2x.py', './_base_/default_runtime.py'
]

# model
num_classes = 30
model = dict(
    type='EnDiffDet',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='torchvision://resnet50'
    ),
    diff_cfg=dict(
        type='EnDiff',
        net=dict(type='PM', channels=16, time_channels=16),
        diffuse_ratio=0.6,
        sample_times=15,
        land_loss_weight=1,
        uw_loss_weight=1),
    backbone=dict(frozen_stages=-1, init_cfg=None),
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
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
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
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
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
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='soft_nms', iou_threshold=0.5),
            max_per_img=100))
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadLqHqImages'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='ResizeLqHqImages',
        img_scale=(1920, 1080),
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='FormatBundle'),
    dict(type='Collect', keys=['img', 'hq_img', 'gt_bboxes', 'gt_labels'], meta_keys=['filename', 'ori_filename',
         'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'hq_img_filename']),
]
test_pipeline = [
    dict(type='LoadLqHqImages'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1920, 1080)],
        flip=True,
        transforms=[
            dict(type='ResizeLqHqImages', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'hq_img']),
            dict(type='Collect', keys=['img', 'hq_img'], meta_keys=['filename', 'ori_filename', 'ori_shape',
                 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'hq_img_filename']),
        ])
]
dataset_type = 'HqLqCocoDataset'
hq_img_prefix = '/media/HDD0/XCX/backgrounds/'
data_root = '/media/HDD0/XCX/synthetic_dataset/'
train_ann = '/media/HDD0/XCX/synthetic_dataset/annotations/split_results/train.json'
test_ann = '/media/HDD0/XCX/synthetic_dataset/annotations/split_results/test.json'
# hq_img_prefix = '/home/xcx/桌面/ddpm/en-diff/data/enhance/'
# data_root = '/home/xcx/桌面/ddpm/en-diff/data/mydata/'
# train_ann = '/home/xcx/桌面/ddpm/en-diff/data/mydata/annotations/instances_train.json'
# test_ann = '/home/xcx/桌面/ddpm/en-diff/data/mydata/annotations/instances_test.json'

classes = [
    'anemone fish', 'coho', 'gar', 'leatherback turtle', 'rock beauty', 'stingray',
    'barracouta', 'conch', 'goldfish', 'lionfish', 'sea anemone', 'sturgeon',
    'box turtle', 'coral reef', 'great white shark', 'loggerhead', 'sea slug', 'tench',
    'brain coral', 'eel', 'hammerhead', 'mud turtle', 'sea urchin', 'terrapin',
    'chiton', 'electric ray', 'jellyfish', 'puffer', 'starfish', 'tiger shark'
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        hq_img_prefix=hq_img_prefix,
        ann_file=train_ann,
        classes=classes,
        img_prefix=data_root+'images/',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        hq_img_prefix=hq_img_prefix,
        ann_file=test_ann,
        classes=classes,
        img_prefix=data_root+'images/',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        hq_img_prefix=hq_img_prefix,
        ann_file=test_ann,
        classes=classes,
        img_prefix=data_root+'images/',
        pipeline=test_pipeline
    ))
evaluation = dict(interval=1, save_best='auto', classwise=True)

optimizer = dict(
    lr=0.0002,
    paramwise_cfg=dict(
        custom_keys=dict(diffusion=dict(lr_mult=0.1, decay_mult=5.0))))

epoch_iter = 2262
lr_config = dict(
    _delete_=True,
    policy='MulStep',
    step=[0, 6 * epoch_iter, 12 * epoch_iter, 20 * epoch_iter, 23 * epoch_iter],
    lr_mul=[1, 0.1, 1, 0.1, 0.01],
    by_epoch=False,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    warmup_start=0)

auto_scale_lr = dict(enable=True, base_batch_size=8)

runner = dict(max_epochs=300)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(
        type='TrainModeControlHook', train_modes=['sample', 'det'], num_epoch=[6, 6])
]
fp16 = dict(loss_scale=512.0)

find_unused_parameters = True
# ====== 配置区域 ======
# CONFIG_PATH = './config/EnDiff_r50_diff.py'
# CHECKPOINT_PATH = './work_dirs/EnDiff_r50_diff/epoch_9.pth'
# INPUT_DIR = '/media/HDD0/XCX/synthetic_dataset/images'
# OUTPUT_DIR = '/media/HDD0/XCX/new_dataset'
# ANNOTATION_PATH = '/media/HDD0/XCX/synthetic_dataset/annotations/split_results/part2.json'