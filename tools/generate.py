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
from tqdm import tqdm  # 导入进度条库
from EnDiff import *


# ====== 配置区域 ======
CONFIG_PATH = '../config/EnDiff_r50_diff.py'
CHECKPOINT_PATH = '../work_dirs/EnDiff_r50_diff/epoch_9.pth'
INPUT_DIR = '/media/HDD0/XCX/synthetic_dataset/images'
OUTPUT_DIR = '/media/HDD0/XCX/new_dataset'
ANNOTATION_PATH = '/media/HDD0/XCX/synthetic_dataset/annotations/split_results/part2.json'


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

def crop_to_fixed_size(image, target_width=1980, target_height=1080):
    """
    从左上角开始裁剪图像为固定尺寸
    :param image: 输入图像
    :param target_width: 目标宽度
    :param target_height: 目标高度
    :return: 裁剪后的图像
    """
    height, width, _ = image.shape

    # 从左上角开始裁剪
    cropped_image = image[:target_height, :target_width]

    return cropped_image

def process_and_save_images(model, input_dir: str, output_dir: str,
                            annotation_path: Optional[str] = None):
    """
    完整的处理流程：
    1. 生成增强图像
    2. 裁剪到目标尺寸
    3. 调整图像到标注文件中的尺寸
    4. 分别保存原始图像和带标注图像
    """
    # 创建输出目录结构
    visible_dir = os.path.join(output_dir, 'visible')
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(visible_dir, exist_ok=True)

    # 初始化COCO处理器
    coco_processor = None
    if annotation_path and os.path.exists(annotation_path):
        try:
            coco_processor = CocoProcessor(annotation_path)
            print(f"Loaded COCO annotations for {len(coco_processor.filename_to_info)} images")
        except Exception as e:
            print(f"Warning: Failed to load COCO annotations - {str(e)}")

    # 获取模型配置
    cfg = mmcv.Config.fromfile(CONFIG_PATH)
    test_pipeline = build_test_pipeline(cfg)

    # 获取所有要处理的图像文件
    img_files = [f for f in os.listdir(input_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not img_files:
        print("No images found in input directory!")
        return

    # 创建进度条
    progress_bar = tqdm(img_files, desc="Processing images", unit="image")

    # 处理每张图像
    for filename in progress_bar:
        try:
            # 更新进度条描述
            progress_bar.set_postfix(file=filename[:15]+"..." if len(filename)>15 else filename)

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
                enhanced_img = model.diffusion.forward_test(img_tensor)

            # 转换为numpy并反归一化
            enhanced_img = enhanced_img.squeeze().permute(1, 2, 0).cpu().numpy()
            enhanced_img = (enhanced_img * cfg.img_norm_cfg['std'] + cfg.img_norm_cfg['mean'])
            enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)
            enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)

            # ===== 2. 初始裁剪（可选） =====
            cropped_image = crop_to_fixed_size(enhanced_img)

            # ===== 3. 调整图像到标注尺寸 =====
            final_img = cropped_image.copy()
            if coco_processor:
                annotations, img_info = coco_processor.get_annotations(filename)
                if img_info:
                    # 获取标注文件中指定的原始尺寸
                    target_size = (img_info['width'], img_info['height'])
                    final_img = resize_to_annotation_size(cropped_image, target_size)

            # ===== 4. 保存调整后的图像 =====
            images_dir = os.path.join(images_dir, filename)
            cv2.imwrite(images_dir, final_img)

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
            print(f"\nError processing {filename}: {str(e)}")

def load_model(config_path, checkpoint_path, device='cuda:0'):
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
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'], meta_keys=[])
            ])
    ])


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    print("Loading model...")
    model = load_model(CONFIG_PATH, CHECKPOINT_PATH, device)

    print(f"\nProcessing images from {INPUT_DIR}...")
    process_and_save_images(model, INPUT_DIR, OUTPUT_DIR, ANNOTATION_PATH)

    print(f"\nAll done! Results saved to:")
    print(f"- Resized images: {os.path.join(OUTPUT_DIR, 'images')}")
    print(f"- Annotated images: {os.path.join(OUTPUT_DIR, 'visible')}")