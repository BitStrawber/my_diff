import os
import torch
import mmcv
import cv2
import numpy as np
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose
from tqdm import tqdm
from typing import Optional
from EnDiff import *


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
                dict(type='CropPadding'),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'], meta_keys=[])
            ])
    ])


def generate_enhanced_images(model, input_dir: str, output_dir: str,
                             config_path: str, device: str = 'cuda:0'):
    """
    生成增强图像并保存到输出目录
    :param model: 加载的模型
    :param input_dir: 输入图像目录
    :param output_dir: 输出目录
    :param config_path: 配置文件路径
    :param device: 使用的设备
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取模型配置
    cfg = mmcv.Config.fromfile(config_path)
    test_pipeline = build_test_pipeline(cfg)

    # 获取所有图像文件
    img_files = [f for f in os.listdir(input_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not img_files:
        print("No images found in input directory!")
        return

    # 处理每张图像
    progress_bar = tqdm(img_files, desc="Generating enhanced images", unit="image")
    for filename in progress_bar:
        try:
            progress_bar.set_postfix(file=filename[:15] + "..." if len(filename) > 15 else filename)

            # 读取图像
            img_path = os.path.join(input_dir, filename)
            img = mmcv.imread(img_path)
            if img is None:
                continue

            # 预处理
            data = {
                'img_prefix': input_dir,
                'img_info': {'filename': filename},
                'img': img
            }
            data = test_pipeline(data)

            # 模型推理
            img_tensor = data['img'][0].unsqueeze(0) if isinstance(data['img'], list) \
                else data['img'].unsqueeze(0)
            img_tensor = img_tensor.to(device)

            with torch.no_grad():
                enhanced_img = model.diffusion.forward_test(img_tensor)

            # 后处理
            enhanced_img = enhanced_img.squeeze().permute(1, 2, 0).cpu().numpy()
            enhanced_img = (enhanced_img * cfg.img_norm_cfg['std'] + cfg.img_norm_cfg['mean'])
            enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)
            enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)

            # 保存增强图像
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, enhanced_img)

        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")


if __name__ == '__main__':
    # 配置参数
    CONFIG_PATH = '../config/EnDiff_r50_diff.py'
    CHECKPOINT_PATH = '../work_dirs/EnDiff_r50/epoch_1.pth'
    INPUT_DIR = '/home/xcx/桌面/synthetic_dataset/blended_images'
    OUTPUT_DIR = '/home/xcx/桌面/enhanced_images'  # 增强图像输出目录
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 检查输入目录
    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    # 加载模型并生成增强图像
    print("Loading model...")
    model = load_model(CONFIG_PATH, CHECKPOINT_PATH, DEVICE)

    print(f"\nGenerating enhanced images from {INPUT_DIR}...")
    generate_enhanced_images(model, INPUT_DIR, OUTPUT_DIR, CONFIG_PATH, DEVICE)

    print(f"\nEnhanced images saved to: {OUTPUT_DIR}")