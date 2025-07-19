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
from tqdm import tqdm
import sys
import argparse
import traceback
import pprint  # 用于美观地打印配置

# 确保项目路径被包含
sys.path.append('./')
try:
    from EnDiff import *
except ImportError:
    print("Warning: Could not import custom modules from EnDiff.py.")

# ====== 配置区域 ======
DEFAULT_CONFIG_PATH = './config/EnDiff_r50_diff.py'
DEFAULT_CHECKPOINT_PATH = './work_dirs/EnDiff_r50_diff/epoch_9.pth'
DEFAULT_INPUT_DIR = '/media/HDD0/XCX/synthetic_dataset/images'
DEFAULT_OUTPUT_DIR = '/media/HDD0/XCX/new_dataset'
DEFAULT_ANNOTATION_PATH = '/media/HDD0/XCX/synthetic_dataset/annotations/split_results/part2.json'


# =====================

# --- CocoProcessor 和 resize_to_annotation_size 函数保持不变 ---
class CocoProcessor:
    def __init__(self, annotation_path: str):
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")
        self.coco = COCO(annotation_path)
        self.filename_to_info = {img['file_name']: img for img in self.coco.dataset['images']}
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.coco.dataset['categories']}

    def get_annotations(self, filename: str) -> Tuple[List[Dict], Optional[Dict]]:
        img_info = self.filename_to_info.get(filename)
        if img_info is None: return [], None
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        return self.coco.loadAnns(ann_ids), img_info


def resize_to_annotation_size(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    h, w = img.shape[:2];
    target_w, target_h = target_size;
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded_img = np.zeros((target_h, target_w, 3), dtype=np.uint8);
    padded_img[:new_h, :new_w] = resized_img
    return padded_img


def build_inference_pipeline_dynamically(cfg: mmcv.Config) -> Compose:
    """
    强制动态构建一个干净的、仅用于推理的 pipeline，忽略配置文件中的 test_pipeline。
    这是解决问题的最稳健方法，无需修改配置文件。
    """
    if rank == 0:
        print("\n" + "=" * 20 + " Building pipeline dynamically " + "=" * 20)
        print("Ignoring pipeline from config file to ensure inference compatibility.")

    # 直接定义一个标准的、安全的推理 pipeline
    inference_pipeline_cfg = [
        # 第1步：使用标准加载器
        dict(type='LoadImageFromFile'),

        # 第2步：使用标准的数据增强/转换容器
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1920, 1080),
            flip=False,  # 推理时关闭翻转
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='Normalize', **cfg.img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

    if rank == 0:
        print("--- Using the following pipeline for inference: ---")
        pprint.pprint(inference_pipeline_cfg)
        print("=" * 65 + "\n")

    return Compose(inference_pipeline_cfg)


def process_images_loop(model, input_dir: str, output_dir: str, config_path: str,
                        annotation_path: Optional[str], rank: int, world_size: int, device: torch.device):
    if rank == 0:
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visible'), exist_ok=True)
    if dist.is_initialized(): dist.barrier()

    coco_processor = None
    if annotation_path and os.path.exists(annotation_path):
        try:
            coco_processor = CocoProcessor(annotation_path)
        except Exception as e:
            if rank == 0: print(f"Warning: Failed to load COCO - {e}")

    cfg = mmcv.Config.fromfile(config_path)

    # === 关键：调用动态构建 pipeline 的函数 ===
    test_pipeline = build_inference_pipeline_dynamically(cfg)

    all_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not all_files:
        if rank == 0: print("No images found!"); return
    local_files = [f for i, f in enumerate(all_files) if i % world_size == rank]

    progress_bar = tqdm(total=len(local_files), desc=f"Rank {rank}", position=rank, disable=(rank != 0))

    for filename in local_files:
        try:
            # 准备一个只包含文件信息的“原材料”字典
            data = {'img_prefix': input_dir, 'img_info': {'filename': filename}}

            # 让干净的 pipeline 完成所有工作
            processed_data = test_pipeline(data)

            img_tensor = processed_data['img'][0].unsqueeze(0).to(device)

            with torch.no_grad():
                enhanced_img_tensor = model.module.forward_test(img_tensor)

            # --- 后处理和保存逻辑不变 ---
            enhanced_img = enhanced_img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            enhanced_img = (enhanced_img * cfg.img_norm_cfg['std'] + cfg.img_norm_cfg['mean'])
            enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)
            final_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)

            images_path = os.path.join(output_dir, 'images', filename)
            if coco_processor:
                annotations, img_info = coco_processor.get_annotations(filename)
                if img_info:
                    final_img = resize_to_annotation_size(final_img, (img_info['width'], img_info['height']))
                    cv2.imwrite(images_path, final_img)
                    if annotations:
                        visible_dir = os.path.join(output_dir, 'visible')
                        visualized_img = final_img.copy()
                        for ann in annotations:
                            x, y, w, h = map(int, ann['bbox']);
                            cv2.rectangle(visualized_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cat_name = coco_processor.cat_id_to_name.get(ann['category_id'], 'unknown')
                            cv2.putText(visualized_img, cat_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 0, 255), 1)
                        cv2.imwrite(os.path.join(visible_dir, filename), visualized_img)
            else:
                cv2.imwrite(images_path, final_img)

        except Exception as e:
            print(f"\nRank {rank} error processing {filename}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        if rank == 0: progress_bar.update(1)
    if rank == 0: progress_bar.close()


# --- load_model, init_distributed, parse_args 函数保持不变 ---
def load_model(config_path, checkpoint_path, device=None):
    cfg = mmcv.Config.fromfile(config_path);
    if 'custom_imports' in cfg: mmcv.utils.import_modules_from_strings(**cfg.custom_imports)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location=device)
    if 'CLASSES' in checkpoint.get('meta', {}): model.CLASSES = checkpoint['meta']['CLASSES']
    model.to(device);
    model.eval()
    return model


def init_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK']);
        world_size = int(os.environ['WORLD_SIZE']);
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu);
        dist.init_process_group(backend='nccl', init_method='env://')
        return True, rank, world_size, gpu
    return False, 0, 1, 0


def parse_args():
    parser = argparse.ArgumentParser(description='图像增强与标注处理工具 (最终鲁棒版)')
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH, help='模型配置文件路径')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT_PATH, help='模型权重文件路径')
    parser.add_argument('--input', default=DEFAULT_INPUT_DIR, help='输入图像目录路径')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR, help='输出目录路径')
    parser.add_argument('--annotation', default=DEFAULT_ANNOTATION_PATH, help='COCO标注文件路径')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    distributed, rank, world_size, gpu = init_distributed()
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    try:
        if rank == 0: print(f"Loading model...")
        model = load_model(args.config, args.checkpoint, device)
        if distributed: model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu],
                                                                          find_unused_parameters=True)
        model.eval()
        if rank == 0: print(f"\nProcessing images from {args.input}...")
        if not os.path.isdir(args.input): raise FileNotFoundError(f"Input directory not found: {args.input}")
        process_images_loop(model, args.input, args.output, args.config, args.annotation, rank, world_size, device)
        if dist.is_initialized(): dist.barrier()
        if rank == 0:
            print("\n" + "=" * 50);
            print("All tasks completed successfully!")
            print(f"Results saved to: {os.path.join(args.output, 'images')}")
            if args.annotation and os.path.exists(args.annotation): print(
                f"Visualizations saved to: {os.path.join(args.output, 'visible')}")
            print("=" * 50)
    except Exception as e:
        print(f"\nAn error occurred on Rank {rank}: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        if dist.is_initialized(): dist.destroy_process_group()