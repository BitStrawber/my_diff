_base_ = [
    './_base_/models/cascade_rcnn_r50_fpn.py',
    './_base_/datasets/coco_detection.py',
    './_base_/schedules/schedule_2x.py',
    './_base_/default_runtime.py'
]

# 自定义数据集路径
data_root = '/media/HDD0/XCX/RUOD_pic/'
train_ann = '/media/HDD0/XCX/RUOD_ANN/instances_train.json'
test_ann = '/media/HDD0/XCX/RUOD_ANN/instances_test.json'
classes = [
    'holothurian', 'echinus', 'scallop', 'starfish', 'fish', 'corals',
    'diver', 'cuttlefish', 'turtle', 'jellyfish'
]
num_classes = 10

# 模型和数据集配置
model = dict(
    init_cfg=dict(
        type='Pretrained',
        checkpoint='./work_dirs/EnDiff_r50_ruod/best_bbox_mAP_epoch_24.pth'
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
        ])
)
gpu_num = 3
gpu_ids = range(gpu_num)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        ann_file=train_ann,
        img_prefix=data_root+'images/',
        classes=classes,
    ),
    val=dict(
        ann_file=test_ann,
        img_prefix=data_root+'images/',
        classes=classes,
    ),
    test=dict(
        ann_file=test_ann,
        img_prefix=data_root+'images/',
        classes=classes,
    )
)

# 其他自定义参数
optimizer = dict(type='SGD', lr=0.02*gpu_num, momentum=0.9, weight_decay=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=24)