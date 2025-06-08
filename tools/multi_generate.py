import os
import torch
import mmcv
import cv2
import numpy as np
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose
from pycocotools.coco import COCO
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from torch.nn.parallel import DataParallel
from torch.cuda.amp import autocast

# ====== 配置区域 ======
CONFIG_PATH = './config/EnDiff_r50_diff.py'
CHECKPOINT_PATH = './work_dirs/EnDiff_r50_diff/epoch_9.pth'
INPUT_DIR = '/media/HDD0/XCX/synthetic_dataset/images'
OUTPUT_DIR = '/media/HDD0/XCX/generate_dataset'
ANNOTATION_PATH = '/media/HDD0/XCX/synthetic_dataset/annotations/split_results/part2.json'
GPU_IDS = [0, 1, 2, 3]  # 使用的GPU设备ID
BATCH_SIZE = 16  # 根据GPU显存调整
NUM_WORKERS = 4  # 数据加载线程数


# =====================

class CocoProcessor:
    """线程安全的COCO标注处理器"""

    def __init__(self, annotation_path: str):
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")

        self.coco = COCO(annotation_path)
        self._build_filename_index()
        self._load_categories()
        self.lock = threading.Lock()

    def _build_filename_index(self):
        """构建文件名到图像元数据的索引"""
        with self.lock:
            self.filename_to_info = {img['file_name']: img for img in self.coco.dataset['images']}

    def _load_categories(self):
        """加载类别ID到名称的映射"""
        with self.lock:
            self.cat_id_to_name = {
                cat['id']: cat['name'] for cat in self.coco.dataset['categories']
            }

    def get_annotations(self, filename: str) -> Tuple[List[Dict], Optional[Dict]]:
        """
        线程安全地获取标注信息
        返回: (annotations, image_info)
        """
        with self.lock:
            img_info = self.filename_to_info.get(filename)
            if img_info is None:
                return [], None

            ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
            return self.coco.loadAnns(ann_ids), img_info


def load_model(config_path, checkpoint_path, device_ids):
    """加载模型并支持多GPU"""
    cfg = mmcv.Config.fromfile(config_path)
    model = build_detector(cfg.model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']

    # 多GPU包装
    if len(device_ids) > 1:
        model = DataParallel(model, device_ids=device_ids)
        device = f'cuda:{device_ids[0]}'
    else:
        device = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    model.eval()
    return model, device, cfg


def build_test_pipeline(cfg):
    """构建测试数据流水线"""
    return Compose([
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1980, 1080),  # 根据实际需求调整
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='Normalize', **cfg.img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'], meta_keys=[])
            ])
    ])


def process_batch(model, batch_imgs, device, cfg):
    """批量处理图像增强（支持混合精度）"""
    with torch.no_grad(), autocast():
        if isinstance(model, DataParallel):
            # 多GPU模式
            enhanced_imgs = model.module.diffusion.forward_test(batch_imgs)
        else:
            # 单GPU模式
            enhanced_imgs = model.diffusion.forward_test(batch_imgs)

    # 转换为numpy数组
    enhanced_imgs = enhanced_imgs.float().cpu().permute(0, 2, 3, 1).numpy()
    enhanced_imgs = (enhanced_imgs * cfg.img_norm_cfg['std'] + cfg.img_norm_cfg['mean'])
    enhanced_imgs = np.clip(enhanced_imgs, 0, 255).astype(np.uint8)
    return [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in enhanced_imgs]


def save_results(filename, enhanced_img, output_dir, coco_processor=None):
    """线程安全的结果保存函数"""
    try:
        # 创建输出目录
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visible'), exist_ok=True)

        # 保存增强后的图像
        img_path = os.path.join(output_dir, 'images', filename)
        cv2.imwrite(img_path, enhanced_img)

        # 处理标注可视化
        if coco_processor:
            annotations, img_info = coco_processor.get_annotations(filename)
            if annotations:
                visualized_img = enhanced_img.copy()
                for ann in annotations:
                    x, y, w, h = map(int, ann['bbox'])
                    cv2.rectangle(visualized_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    # 添加标签
                    cat_name = coco_processor.cat_id_to_name.get(ann['category_id'], 'unknown')
                    label = f"{cat_name}"
                    if 'score' in ann:
                        label += f" {ann['score']:.2f}"
                    cv2.putText(visualized_img, label, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # 保存可视化结果
                visible_path = os.path.join(output_dir, 'visible', filename)
                cv2.imwrite(visible_path, visualized_img)
    except Exception as e:
        print(f"Error saving {filename}: {str(e)}")


def main():
    # 初始化环境
    torch.backends.cudnn.benchmark = True  # 加速卷积运算
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型
    print("Loading model...")
    model, device, cfg = load_model(CONFIG_PATH, CHECKPOINT_PATH, GPU_IDS)
    test_pipeline = build_test_pipeline(cfg)

    # 初始化COCO处理器
    coco_processor = None
    if ANNOTATION_PATH and os.path.exists(ANNOTATION_PATH):
        coco_processor = CocoProcessor(ANNOTATION_PATH)

    # 获取所有图像文件
    img_files = [f for f in os.listdir(INPUT_DIR)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not img_files:
        print("No images found in input directory!")
        return

    # 使用线程池并行加载和保存
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 分批处理
        for i in tqdm(range(0, len(img_files), BATCH_SIZE), desc="Processing batches"):
            batch_files = img_files[i:i + BATCH_SIZE]

            try:
                # 并行加载图像
                load_tasks = []
                for filename in batch_files:
                    img_path = os.path.join(INPUT_DIR, filename)
                    load_tasks.append(executor.submit(mmcv.imread, img_path))

                # 获取加载结果
                batch_imgs = [task.result() for task in load_tasks]
                batch_imgs = [img for img in batch_imgs if img is not None]

                if not batch_imgs:
                    continue

                # 预处理
                batch_data = [{'img': img, 'img_info': {'filename': f}}
                              for img, f in zip(batch_imgs, batch_files) if img is not None]
                batch_processed = test_pipeline(batch_data)

                # 转换为tensor
                img_tensor = torch.stack([data['img'] for data in batch_processed]).to(device)

                # 处理增强
                enhanced_batch = process_batch(model, img_tensor, device, cfg)

                # 并行保存结果
                for enhanced_img, filename in zip(enhanced_batch, batch_files):
                    executor.submit(save_results, filename, enhanced_img,
                                    OUTPUT_DIR, coco_processor)

                # 清理显存
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\nError processing batch {i}-{i + BATCH_SIZE}: {str(e)}")
                continue

    print(f"\nProcessing completed! Results saved to:")
    print(f"- Enhanced images: {os.path.join(OUTPUT_DIR, 'images')}")
    print(f"- Visualized images: {os.path.join(OUTPUT_DIR, 'visible')}")


if __name__ == '__main__':
    import threading

    main()