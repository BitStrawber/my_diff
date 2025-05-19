# 导入基础配置文件，包括模型结构、训练计划和运行时设置
_base_ = [
    './_base_/models/cascade_rcnn_r50_fpn.py',  # 基础模型结构
    './_base_/schedules/schedule_2x.py',  # 训练计划
    './_base_/default_runtime.py'  # 默认运行时设置
]

# 模型配置
num_classes = 4  # 定义数据集的类别数量
model = dict(
    type='EnDiffDet',  # 自定义模型类型，结合了扩散模型和目标检测
    init_cfg=dict(  # 初始化配置，加载预训练模型
        type='Pretrained',  # 使用预训练模型进行初始化
        checkpoint='checkpoints/cascade_rcnn_r50_fpn_20e_coco_bbox_mAP-0.41_20200504_175131-e9872a90.pth'  # 预训练模型的路径
    ),
    diff_cfg=dict(  # 扩散模型的配置
        type='EnDiff',  # 扩散模型类型
        net=dict(type='PM', channels=16, time_channels=16),  # 扩散网络结构
        diffuse_ratio=0.6,  # 扩散比例
        sample_times=15,  # 采样次数
        land_loss_weight=1,  # 土地损失权重
        uw_loss_weight=10  # 水下损失权重
    ),
    backbone=dict(  # 骨干网络配置
        frozen_stages=-1,  # 不冻结任何阶段
        init_cfg=None  # 不使用特定的初始化配置
    )
)

# 图像归一化配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  # 均值
    std=[58.395, 57.12, 57.375],  # 标准差
    to_rgb=True  # 转换为RGB格式
)

# 训练数据处理流程
train_pipeline = [
    dict(type='LoadLqHqImages'),  # 加载低质量（LQ）和高质量（HQ）图像
    dict(
        type='ResizeLqHqImages',  # 调整LQ和HQ图像大小
        img_scale=(1333, 800),
        keep_ratio=True
    ),
    dict(type='RandomFlip', flip_ratio=0.5),  # 随机翻转
    dict(type='Normalize', **img_norm_cfg),  # 图像归一化
    dict(type='Pad', size_divisor=32),  # 填充图像
    dict(type='FormatBundle'),  # 格式化数据
    dict(
        type='Collect',  # 收集数据
        keys=['img', 'hq_img', 'gt_bboxes', 'gt_labels'],
        meta_keys=[
            'filename', 'ori_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
            'img_norm_cfg', 'hq_img_filename'
        ]
    )
]

# 测试数据处理流程
test_pipeline = [
    dict(type='LoadLqHqImages'),  # 加载LQ和HQ图像
    dict(
        type='MultiScaleFlipAug',  # 多尺度翻转增强
        img_scale=[(1333, 800)],
        flip=True,
        transforms=[
            dict(type='ResizeLqHqImages', keep_ratio=True),  # 保持比例调整大小
            dict(type='RandomFlip'),  # 随机翻转
            dict(type='Normalize', **img_norm_cfg),  # 归一化
            dict(type='Pad', size_divisor=32),  # 填充
            dict(type='ImageToTensor', keys=['img', 'hq_img']),  # 图像转为张量
            dict(
                type='Collect',  # 收集数据
                keys=['img', 'hq_img'],
                meta_keys=[
                    'filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor',
                    'flip', 'flip_direction', 'img_norm_cfg',
                    'hq_img_filename'
                ]
            )
        ]
    )
]

# 数据集配置
dataset_type = 'HqLqCocoDataset'  # 自定义数据集类型
hq_img_prefix = './data/enhance/'  # 高质量图像前缀路径
data_root = './data/mydata/'  # 数据根目录
train_ann = './data/mydata/annotations/instances_train.json'
test_ann = './data/mydata/annotations/instances_test.json'

# 数据加载器配置
classes = ['echinus', 'holothurian',  'scallop', 'starfish']
data = dict(
    samples_per_gpu=3,
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

# 评估配置
evaluation = dict(
    interval=1,  # 评估间隔
    save_best='auto',  # 自动保存最佳模型
    classwise=True  # 类别-wise评估
)

# 优化器配置
optimizer = dict(
    lr=0.0025,  # 学习率
    paramwise_cfg=dict(  # 参数优化配置
        custom_keys=dict(
            diffusion=dict(lr_mult=0.1, decay_mult=5.0)  # 扩散模型部分的学习率和衰减率
        )
    )
)

# 学习率调整配置
epoch_iter = 2262
lr_config = dict(
    _delete_=True,  # 删除基配置中的学习率配置
    policy='MulStep',  # 多步学习率策略
    step=[0, 6 * epoch_iter, 12 * epoch_iter, 20 * epoch_iter, 23 * epoch_iter],  # 学习率调整步骤
    lr_mul=[1, 0.1, 1, 0.1, 0.01],  # 学习率乘数
    by_epoch=False,  # 按迭代次数调整学习率
    warmup='linear',  # 预热策略
    warmup_iters=500,  # 预热迭代次数
    warmup_ratio=0.001,  # 预热比率
    warmup_start=0  # 预热起始迭代次数
)

# 运行配置
runner = dict(max_epochs=24)  # 最大训练周期数
log_config = dict(
    interval=50,  # 日志记录间隔
    hooks=[dict(type='TextLoggerHook')]  # 日志挂钩类型
)

# 自定义挂钩
custom_hooks = [
    dict(type='NumClassCheckHook'),  # 检查类别数量的钩子
    dict(
        type='TrainModeControlHook',  # 控制训练模式的钩子
        train_modes=['sample'],  # 训练模式列表
        num_epoch=[12, 12]  # 每个训练模式的周期数
    )
]

# 混合精度训练配置
fp16 = dict(loss_scale=512.0)  # 损失缩放因子