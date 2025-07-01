_base_ = [
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

# 模型配置
model = dict(
    type='EnDiff',
    net=dict(
        type='PM',
        channels=16,
        time_channels=16,
    ),
    diffuse_ratio=0.6,
    sample_times=15,
    land_loss_weight=1.0,
    uw_loss_weight=10.0,
    init_cfg=None
)

# 数据配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadLqHqImages'),
    dict(type='ResizeLqHqImages', img_scale=(1920, 1080), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='FormatBundle'),
    dict(type='Collect', keys=['img', 'hq_img'])
]

test_pipeline = [
    dict(type='LoadLqHqImages'),
    dict(type='ResizeLqHqImages', img_scale=(1920, 1080), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='FormatBundle'),
    dict(type='Collect', keys=['img', 'hq_img'])
]

dataset_type = 'HqLqCocoDataset'
hq_img_prefix = '/media/HDD0/XCX/backgrounds/'
data_root = '/media/HDD0/XCX/synthetic_dataset/'
train_ann = '/media/HDD0/XCX/synthetic_dataset/annotations/split_results/train.json'
test_ann = '/media/HDD0/XCX/synthetic_dataset/annotations/split_results/test.json'

data = dict(
    samples_per_gpu=2,  # 根据显存调整
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        hq_img_prefix=hq_img_prefix,
        img_prefix=data_root+'images/',
        ann_file=train_ann,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        hq_img_prefix=hq_img_prefix,
        img_prefix=data_root+'images/',
        ann_file=test_ann,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        hq_img_prefix=hq_img_prefix,
        img_prefix=data_root+'images/',
        ann_file=test_ann,
        pipeline=test_pipeline
    )
)

# 优化器配置
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'net.time_emb': dict(lr_mult=0.1),
        }))

epoch_iter = 2262
# 学习率调度
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

# 训练配置
runner = dict(max_epochs=24)
# 在配置文件末尾添加
checkpoint_config = dict(
    interval=5,
    max_keep_ckpts=3,  # 最多保留3个检查点
    save_optimizer=True,  # 保存优化器状态
    by_epoch=True
)

# 自动恢复设置
auto_resume = True
resume_from = None  # 如果指定路径则从该检查点恢复

evaluation = dict(interval=1, metric='loss')
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# 其他设置
fp16 = dict(loss_scale=512.0)
dist_params = dict(backend='nccl')
workflow = [('train', 1)]