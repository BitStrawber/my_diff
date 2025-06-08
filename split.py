import json
import random
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm


def split_coco_annotations(
        annotation_path: str,
        output_dir: str,
        train_ratio: float = 0.8,
        shuffle: bool = True,
        seed: int = 42,
        min_val_samples: int = 1
) -> Dict[str, str]:
    """
    划分COCO标注文件为训练集和验证集（不处理图像文件）

    参数:
        annotation_path: COCO标注文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例 (0-1)
        shuffle: 是否打乱数据
        seed: 随机种子
        min_val_samples: 验证集最小样本数

    返回:
        包含生成文件路径的字典 {
            'train': 训练集路径,
            'val': 验证集路径
        }
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载原始标注
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    # 获取图像ID列表
    images = coco_data['images']
    image_ids = [img['id'] for img in images]

    # 确保验证集有足够样本
    n_val = max(int(len(image_ids) * (1 - train_ratio)), min_val_samples)
    n_train = len(image_ids) - n_val

    if shuffle:
        random.seed(seed)
        random.shuffle(image_ids)

    # 划分ID集合
    train_ids = set(image_ids[:n_train])
    val_ids = set(image_ids[n_train:])

    # 构建新标注结构
    def build_subset(ids: set) -> Dict:
        subset = {
            "info": coco_data.get("info", {}),
            "licenses": coco_data.get("licenses", []),
            "categories": coco_data["categories"],
            "images": [img for img in images if img['id'] in ids],
            "annotations": [ann for ann in coco_data['annotations']
                            if ann['image_id'] in ids]
        }
        return subset

    # 创建子集
    train_ann = build_subset(train_ids)
    val_ann = build_subset(val_ids)

    # 保存文件
    train_path = Path(output_dir) / "part2_train.json"
    val_path = Path(output_dir) / "part2_val.json"

    with open(train_path, 'w') as f:
        json.dump(train_ann, f, indent=2)
    with open(val_path, 'w') as f:
        json.dump(val_ann, f, indent=2)

    # 打印统计信息
    print("\nCOCO annotation split completed:")
    print(f"Total images: {len(image_ids)}")
    print(f"Train set: {len(train_ann['images'])} images, {len(train_ann['annotations'])} annotations")
    print(f"Val set: {len(val_ann['images'])} images, {len(val_ann['annotations'])} annotations")
    print(f"Files saved to:\n- {train_path}\n- {val_path}")

    return {
        'train': str(train_path),
        'val': str(val_path)
    }


# 使用示例
if __name__ == "__main__":
    # 配置参数
    ANNOTATION_PATH = "/media/HDD0/XCX/synthetic_dataset/annotations/split_results/part2.json"
    OUTPUT_DIR = "/media/HDD0/XCX/synthetic_dataset/annotations/split_results"
    TRAIN_RATIO = 0.8  # 80%训练集，20%验证集

    # 执行划分
    split_coco_annotations(
        annotation_path=ANNOTATION_PATH,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO
    )