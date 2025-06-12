_base_ = [
    './_base_/models/cascade_rcnn_r50_fpn.py',
    './_base_/datasets/coco_detection.py',
    './_base_/schedules/schedule_2x.py',
    './_base_/default_runtime.py'
]

# 自定义数据集路径
data_root = '/media/HDD0/XCX/new_dataset/'
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
        checkpoint='torchvision://resnet50',
        prefix='backbone'
    ),
    roi_head=dict(
        bbox_head=[
            dict(num_classes=num_classes),
            dict(num_classes=num_classes),
            dict(num_classes=num_classes),
        ]
    )
)
gpu_ids = range(3)

data = dict(
    samples_per_gpu=4,     # 根据GPU显存调整
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

# 其他自定义参数
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=24)