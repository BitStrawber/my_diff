_base_ = [
    './_base_/models/cascade_rcnn_r50_fpn.py',
    './_base_/datasets/coco_detection.py',
    './_base_/schedules/schedule_2x.py',
    './_base_/default_runtime.py'
]

# 自定义数据集路径
# data_root = '/media/HDD0/XCX/synthetic_dataset/'
data_root = '/media/HDD0/XCX/compare/det05/'
train_ann = '/media/HDD0/XCX/synthetic_dataset/annotations/split_results/part2_train_merge.json'
test_ann = '/media/HDD0/XCX/synthetic_dataset/annotations/split_results/part2_val_merge.json'
classes = [
    'holothurian', 'echinus', 'scallop', 'starfish', 'fish', 'corals',
    'diver', 'cuttlefish', 'turtle', 'jellyfish'
]
num_classes = 10

# 模型和数据集配置
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


data = dict(
    samples_per_gpu=2,     # 根据GPU显存调整
    workers_per_gpu=4,     # 数据加载线程数
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

auto_scale_lr = dict(enable=True, base_batch_size=2)

# 其他自定义参数
evaluation = dict(interval=1, save_best='auto', classwise=True)
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=24)