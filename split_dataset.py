import os
import json
import random
import numpy as np
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO


def stratified_split(coco_ann_file, output_dir, test_size=0.2, val_size=0.25, seed=42):
    """分层随机划分COCO数据集"""
    os.makedirs(output_dir, exist_ok=True)

    # 加载原始标注
    with open(coco_ann_file) as f:
        ann_data = json.load(f)

    # 按类别分层抽样
    coco = COCO(coco_ann_file)
    img_ids = sorted(coco.getImgIds())
    cat_ids = coco.getCatIds()

    # 构建类别到图像的映射
    cat_to_imgs = {cat_id: [] for cat_id in cat_ids}
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            cat_to_imgs[ann['category_id']].append(img_id)

    # 分层抽样：先抽20%作为第一部分
    part1_imgs = set()
    for cat_id, imgs in cat_to_imgs.items():
        sampled = random.sample(imgs, int(len(imgs) * 0.2))
        part1_imgs.update(sampled)
    part1_imgs = list(part1_imgs)
    part2_imgs = list(set(img_ids) - set(part1_imgs))

    # 在第一部分中划分train/val
    train_imgs, val_imgs = train_test_split(
        part1_imgs,
        test_size=val_size,
        random_state=seed
    )

    # 构建各分组的标注数据
    def _build_anns(img_ids):
        img_ids = set(img_ids)
        new_anns = [ann for ann in ann_data['annotations']
                    if ann['image_id'] in img_ids]
        new_imgs = [img for img in ann_data['images']
                    if img['id'] in img_ids]
        return {
            'images': new_imgs,
            'annotations': new_anns,
            'categories': ann_data['categories'],
            'info': ann_data.get('info', {}),
            'licenses': ann_data.get('licenses', [])
        }

    # 保存划分结果
    splits = {
        'part1_train': _build_anns(train_imgs),
        'part1_val': _build_anns(val_imgs),
        'part2': _build_anns(part2_imgs)
    }

    for name, data in splits.items():
        output_path = os.path.join(output_dir, f'{name}.json')
        with open(output_path, 'w') as f:
            json.dump(data, f)

    print(f"划分完成！结果保存在 {output_dir}")
    print(f"Part1 (20%): 训练集 {len(train_imgs)}张, 验证集 {len(val_imgs)}张")
    print(f"Part2 (80%): {len(part2_imgs)}张")


if __name__ == '__main__':
    # 使用示例
    stratified_split(
        coco_ann_file='/media/HDD0/XCX/synthetic_dataset/annotations/instances_all.json',
        output_dir='/media/HDD0/XCX/synthetic_dataset/annotations/split_results',
        test_size=0.2,
        val_size=0.25,
        seed=42
    )