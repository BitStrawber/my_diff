_base_ = [
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

# model
model = dict(
    type='EnDiff',
    net=dict(type='PM', channels=16, time_channels=16),
    diffuse_ratio=0.6,
    sample_times=15,
    land_loss_weight=1,
    uw_loss_weight=10,
    init_cfg=dict(type='Pretrained', checkpoint=None)
)

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadLqHqImages'),
    dict(
        type='ResizeLqHqImages',
        img_scale=(1920, 1080),
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='FormatBundle'),
    dict(type='Collect', keys=['img', 'hq_img'], meta_keys=['filename', 'ori_filename',
         'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction',
         'img_norm_cfg', 'hq_img_filename']),
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
            dict(type='Collect', keys=['img', 'hq_img'], meta_keys=['filename', 'ori_filename',
                 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip',
                 'flip_direction', 'img_norm_cfg', 'hq_img_filename']),
        ])
]

dataset_type = 'HqLqCocoDataset'
hq_img_prefix = '/media/HDD0/XCX/backgrounds/'
data_root = '/media/HDD0/XCX/synthetic_dataset/'
train_ann = '/media/HDD0/XCX/synthetic_dataset/annotations/split_results/train.json'
test_ann = '/media/HDD0/XCX/synthetic_dataset/annotations/split_results/test.json'

classes = [
    'anemone fish', 'coho', 'gar', 'leatherback turtle', 'rock beauty', 'stingray',
    'barracouta', 'conch', 'goldfish', 'lionfish', 'sea anemone', 'sturgeon',
    'box turtle', 'coral reef', 'great white shark', 'loggerhead', 'sea slug', 'tench',
    'brain coral', 'eel', 'hammerhead', 'mud turtle', 'sea urchin', 'terrapin',
    'chiton', 'electric ray', 'jellyfish', 'puffer', 'starfish', 'tiger shark'
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
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

evaluation = dict(interval=1, save_best='auto')

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0025,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            diffusion=dict(lr_mult=0.1, decay_mult=5.0)
        )
    )
)

# learning policy
epoch_iter = 2262
lr_config = dict(
    policy='MulStep',
    step=[0, 6 * epoch_iter, 12 * epoch_iter, 20 * epoch_iter, 23 * epoch_iter],
    lr_mul=[1, 0.1, 1, 0.1, 0.01],
    by_epoch=False,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    warmup_start=0)

runner = dict(type='EpochBasedRunner', max_epochs=24)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

# runtime settings
checkpoint_config = dict(interval=1)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]