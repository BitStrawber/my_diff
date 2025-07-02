import os
import torch
import torch.distributed as dist
import mmcv
import cv2
import numpy as np
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose
from pycocotools.coco import COCO
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm  # 导入进度条库
import sys
sys.path.append('./')
from EnDiff import *
import argparse


# ====== 配置区域 ======
# CONFIG_PATH = './config/EnDiff_r50_diff.py'
# CHECKPOINT_PATH = './work_dirs/EnDiff_r50_diff/epoch_9.pth'
# INPUT_DIR = '/media/HDD0/XCX/synthetic_dataset/images'
# OUTPUT_DIR = '/media/HDD0/XCX/new_dataset'
# ANNOTATION_PATH = '/media/HDD0/XCX/synthetic_dataset/annotations/split_results/part2.json'
CONFIG_PATH = './config/EnDiff_r50_diff.py'
CHECKPOINT_PATH = './work_dirs/EnDiff_r50/epoch_1.pth'
INPUT_DIR = '/home/xcx/桌面/synthetic_dataset/blended_images'
OUTPUT_DIR = '/home/xcx/桌面/temp'
ANNOTATION_PATH = '/home/xcx/桌面/synthetic_dataset/annotations/instances_all.json'

# =====================

class CocoProcessor:
    """专业的COCO标注处理器（保持标注原始尺寸）"""

    def __init__(self, annotation_path: str):
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")

        self.coco = COCO(annotation_path)
        self._build_filename_index()
        self._load_categories()

    def _build_filename_index(self):
        """构建文件名到图像元数据的索引"""
        self.filename_to_info = {img['file_name']: img for img in self.coco.dataset['images']}

    def _load_categories(self):
        """加载类别ID到名称的映射"""
        self.cat_id_to_name = {
            cat['id']: cat['name'] for cat in self.coco.dataset['categories']
        }

    def get_annotations(self, filename: str) -> Tuple[List[Dict], Optional[Dict]]:
        """
        获取指定文件名的标注信息和图像元数据
        返回: (annotations, image_info)
        """
        img_info = self.filename_to_info.get(filename)
        if img_info is None:
            return [], None

        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        return self.coco.loadAnns(ann_ids), img_info


def resize_to_annotation_size(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    将图像调整到标注文件中指定的尺寸（保持宽高比）
    参数:
        img: 输入图像 (H,W,C)
        target_size: 目标尺寸 (width, height)
    返回:
        调整后的图像
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size

    # 计算缩放比例（保持宽高比）
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 调整图像尺寸
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 填充到目标尺寸（如果需要）
    if new_w != target_w or new_h != target_h:
        padded_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded_img[:new_h, :new_w] = resized_img
        return padded_img

    return resized_img


def process_and_save_images(model, input_dir: str, output_dir: str,
                            annotation_path: Optional[str] = None, rank: int = 0, world_size: int = 1):
    """
    完整的处理流程：
    1. 生成增强图像
    2. 裁剪到目标尺寸
    3. 调整图像到标注文件中的尺寸
    4. 分别保存原始图像和带标注图像
    """
    # 创建输出目录结构（仅rank 0创建）
    if rank == 0:
        visible_dir = os.path.join(output_dir, 'visible')
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(visible_dir, exist_ok=True)
    else:
        visible_dir = os.path.join(output_dir, 'visible')
        images_dir = os.path.join(output_dir, 'images')

    # 初始化COCO处理器（仅rank 0执行）
    coco_processor = None
    if rank == 0 and annotation_path and os.path.exists(annotation_path):
        try:
            coco_processor = CocoProcessor(annotation_path)
            print(f"Loaded COCO annotations for {len(coco_processor.filename_to_info)} images")
            if annotation_path and os.path.exists(annotation_path):
                try:
                    coco_processor = CocoProcessor(annotation_path)
                    print(f"Loaded COCO annotations for {len(coco_processor.filename_to_info)} images")
                except Exception as e:
                    print(f"Warning: Failed to load COCO annotations - {str(e)}")
        except Exception as e:
            print(f"Warning: Failed to load COCO annotations - {str(e)}")


    # 获取模型配置
    cfg = mmcv.Config.fromfile(CONFIG_PATH)
    test_pipeline = build_test_pipeline(cfg)

    # 获取所有要处理的图像文件
    img_files = [f for f in os.listdir(input_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not img_files:
        if rank == 0:
            print("No images found in input directory!")
        return

    # 分布式数据分配
    local_files = [f for i, f in enumerate(img_files) if i % world_size == rank]

    # 创建进度条（各rank独立显示）
    progress_bar = tqdm(local_files, desc=f"Rank {rank}", position=rank, disable=(rank != 0))

    # 处理每张图像
    for filename in progress_bar:
        try:
            # 更新进度条描述
            if rank == 0:
                progress_bar.set_postfix(file=filename[:15] + "..." if len(filename) > 15 else filename)

            # ===== 1. 生成增强图像 =====
            img_path = os.path.join(input_dir, filename)
            img = mmcv.imread(img_path)
            if img is None:
                continue

            data = {
                'img_prefix': input_dir,
                'img_info': {'filename': filename},
                'img': img
            }
            data = test_pipeline(data)

            img_tensor = data['img'][0].unsqueeze(0) if isinstance(data['img'], list) \
                else data['img'].unsqueeze(0)
            img_tensor = img_tensor.to(device)

            with torch.no_grad():
                enhanced_img = model.module.diffusion.forward_test(img_tensor)

            # 转换为numpy并反归一化
            enhanced_img = enhanced_img.squeeze().permute(1, 2, 0).cpu().numpy()
            enhanced_img = (enhanced_img * cfg.img_norm_cfg['std'] + cfg.img_norm_cfg['mean'])
            enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)
            enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)

            # ===== 3. 调整图像到标注尺寸 =====
            final_img = enhanced_img.copy()
            if coco_processor:
                annotations, img_info = coco_processor.get_annotations(filename)
                if img_info:
                    # 获取标注文件中指定的原始尺寸
                    target_size = (img_info['width'], img_info['height'])
                    final_img = resize_to_annotation_size(enhanced_img, target_size)

            # ===== 4. 保存调整后的图像 =====
            images_path = os.path.join(images_dir, filename)
            cv2.imwrite(images_path, final_img)

            # ===== 5. 处理标注 =====
            if coco_processor and annotations:
                # 直接使用原始标注（不调整）
                visualized_img = final_img.copy()
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

                # 保存带标注的图像
                visible_path = os.path.join(visible_dir, filename)
                cv2.imwrite(visible_path, visualized_img)

        except Exception as e:
            print(f"\nRank {rank} error processing {filename}: {str(e)}")

def load_model(config_path, checkpoint_path, device=None):
    """加载训练好的模型"""
    cfg = mmcv.Config.fromfile(config_path)
    model = build_detector(cfg.model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location=device)
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    model.to(device)
    model.eval()
    return model


def build_test_pipeline(cfg):
    """构建正确的测试数据流水线"""
    return Compose([
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1920, 1080),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='Normalize', **cfg.img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='CropPadding'),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'], meta_keys=[])
            ])
    ])

def init_distributed():
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend='nccl', init_method='env://')
        return True, rank, world_size, gpu
    return False, 0, 1, 0

def parse_args():
    parser = argparse.ArgumentParser(description='图像增强与标注处理工具')
    parser.add_argument('--config', default=CONFIG_PATH, help='模型配置文件路径')
    parser.add_argument('--checkpoint', default=CHECKPOINT_PATH, help='模型权重文件路径')
    parser.add_argument('--input', default=INPUT_DIR, help='输入图像目录路径')
    parser.add_argument('--output', default=OUTPUT_DIR, help='输出目录路径')
    parser.add_argument('--annotation', default=ANNOTATION_PATH, help='COCO标注文件路径')
    return parser.parse_args()


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()

    # 初始化分布式环境
    distributed, rank, world_size, gpu = init_distributed()
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    # 加载模型（仅在rank 0打印信息）
    if rank == 0:
        print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.config, args.checkpoint, device)

    # 模型分布式包装
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device())
    else:
        model = torch.nn.DataParallel(model)
    model.eval()

    # 处理图像
    if rank == 0:
        print(f"\nProcessing images from {args.input}...")

    if not os.path.isdir(args.input):
        raise FileNotFoundError(f"Input directory not found: {args.input}")

    process_and_save_images(model, args.input, args.output, args.annotation,
                            rank=rank, world_size=world_size)

    if rank == 0:
        print(f"\nAll done! Results saved to:")
        print(f"- Resized images: {os.path.join(args.output, 'images')}")
        print(f"- Annotated images: {os.path.join(args.output, 'visible')}")