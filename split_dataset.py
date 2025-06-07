import os
import json
import random
from sklearn.model_selection import train_test_split


def split_coco_dataset(ann_file, output_dir, part1_ratio=0.2, test_ratio=0.25, seed=42):
    """
    划分COCO数据集：
    1. 随机划分为两部分（1:4比例）
    2. 将较小的部分再划分为训练集和测试集

    Args:
        ann_file: COCO标注文件路径
        output_dir: 输出目录
        part1_ratio: 第一部分占比（默认0.2，即1:4）
        test_ratio: 测试集占第一部分的比率（默认0.25）
        seed: 随机种子
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载原始标注
    with open(ann_file) as f:
        ann_data = json.load(f)

    # 获取所有图像ID并随机打乱
    all_images = ann_data['images']
    random.seed(seed)
    random.shuffle(all_images)

    # 计算划分点
    split_idx = int(len(all_images) * part1_ratio)

    # 划分两部分
    part1_images = all_images[:split_idx]  # 20%数据
    part2_images = all_images[split_idx:]  # 80%数据

    # 在第一部分中划分训练集和测试集
    train_images, test_images = train_test_split(
        part1_images,
        test_size=test_ratio,
        random_state=seed
    )

    # 构建标注子集函数
    def build_subset(images):
        image_ids = {img['id'] for img in images}
        return {
            'images': images,
            'annotations': [ann for ann in ann_data['annotations'] if ann['image_id'] in image_ids],
            'categories': ann_data['categories'],
            'info': ann_data.get('info', {}),
            'licenses': ann_data.get('licenses', [])
        }

    # 构建各数据集
    part2_ann = build_subset(part2_images)
    train_ann = build_subset(train_images)
    test_ann = build_subset(test_images)

    # 保存结果
    with open(os.path.join(output_dir, 'part2.json'), 'w') as f:
        json.dump(part2_ann, f)
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_ann, f)
    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(test_ann, f)

    # 打印统计信息
    print(f"划分完成！结果保存在 {output_dir}")
    print(f"总图像数: {len(all_images)}")
    print(f"Part1 (20%): {len(part1_images)}张")
    print(f"  ├─ 训练集: {len(train_images)}张")
    print(f"  └─ 测试集: {len(test_images)}张")
    print(f"Part2 (80%): {len(part2_images)}张")


if __name__ == '__main__':
    # 使用示例
    split_coco_dataset(
        ann_file='/media/HDD0/XCX/synthetic_dataset/annotations/instances_all.json',
        output_dir='/media/HDD0/XCX/synthetic_dataset/annotations/split_results',
        part1_ratio=0.2,
        test_ratio=0.25,
        seed=42
    )