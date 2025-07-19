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
from torch.utils.data import Dataset, DataLoader

# 确保项目路径被包含，以便导入自定义模块
sys.path.append('./')
from EnDiff import *  # 假设 EnDiff 等模块在此文件中

# ==============================================================================
# ======                          配置区域                           ======
# ==============================================================================
CONFIG_PATH = './config/EnDiff_r50_diff.py'
CHECKPOINT_PATH = './work_dirs/EnDiff_r50_diff/epoch_9.pth'
INPUT_DIR = '/media/HDD0/XCX/synthetic_dataset/images'
OUTPUT_DIR = '/media/HDD0/XCX/new_dataset'
ANNOTATION_PATH = '/media/HDD0/XCX/synthetic_dataset/annotations/split_results/part2.json'


# ==============================================================================

class CocoProcessor:
    """专业的COCO标注处理器（保持标注原始尺寸）"""

    def __init__(self, annotation_path: str):
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")
        self.coco = COCO(annotation_path)
        self.filename_to_info = {img['file_name']: img for img in self.coco.dataset['images']}
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.coco.dataset['categories']}

    def get_annotations(self, filename: str) -> Tuple[List[Dict], Optional[Dict]]:
        """获取指定文件名的标注信息和图像元数据"""
        img_info = self.filename_to_info.get(filename)
        if img_info is None:
            return [], None
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        return self.coco.loadAnns(ann_ids), img_info


class InferenceDataset(Dataset):
    """为高效推理定制的数据集"""

    def __init__(self, file_paths: List[str], pipeline: List[Dict]):
        self.file_paths = file_paths
        self.pipeline = pipeline

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.file_paths[idx]
        filename = os.path.basename(img_path)
        try:
            img = mmcv.imread(img_path)
            if img is None:
                raise IOError(f"Failed to read image: {img_path}")

            data = {'img': img, 'img_info': {'filename': filename}, 'img_prefix': None}
            processed_data = self.pipeline(data)

            img_tensor = processed_data['img'][0] if isinstance(processed_data['img'], list) else processed_data['img']

            return {'filename': filename, 'img': img_tensor, 'status': 'ok'}
        except Exception as e:
            print(f"Warning: Error processing {img_path}, skipping. Error: {e}", file=sys.stderr)
            return {'filename': filename, 'img': torch.zeros(3, 32, 32), 'status': 'error'}


def post_process_output(tensor_img: torch.Tensor, cfg: mmcv.Config) -> np.ndarray:
    """将模型输出的Tensor后处理为可保存的Numpy图像"""
    img_np = tensor_img.squeeze().permute(1, 2, 0).cpu().numpy()
    mean = np.array(cfg.img_norm_cfg['mean'], dtype=np.float32)
    std = np.array(cfg.img_norm_cfg['std'], dtype=np.float32)
    img_np = (img_np * std + mean)
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


def resize_to_annotation_size(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """将图像调整到标注文件中指定的尺寸（保持宽高比并填充）"""
    h, w = img.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    padded_img[:new_h, :new_w] = resized_img
    return padded_img


def save_and_visualize(image: np.ndarray, filename: str, output_dir: str, coco_processor: Optional[CocoProcessor]):
    """保存增强后的图像，并（如果可能）保存带标注的可视化版本"""
    images_dir = os.path.join(output_dir, 'images')
    images_path = os.path.join(images_dir, filename)
    cv2.imwrite(images_path, image)

    if coco_processor:
        annotations, _ = coco_processor.get_annotations(filename)
        if annotations:
            visible_dir = os.path.join(output_dir, 'visible')
            visualized_img = image.copy()
            for ann in annotations:
                x, y, w, h = map(int, ann['bbox'])
                cv2.rectangle(visualized_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cat_name = coco_processor.cat_id_to_name.get(ann['category_id'], 'unknown')
                cv2.putText(visualized_img, cat_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            visible_path = os.path.join(visible_dir, filename)
            cv2.imwrite(visible_path, visualized_img)


def process_images_dist(model, input_dir: str, output_dir: str, annotation_path: Optional[str],
                        cfg: mmcv.Config, rank: int, world_size: int, device: torch.device):
    """分布式处理图像的核心函数"""
    if rank == 0:
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visible'), exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    coco_processor = None
    if annotation_path and os.path.exists(annotation_path):
        try:
            coco_processor = CocoProcessor(annotation_path)
            if rank == 0:
                print(f"Loaded COCO annotations for {len(coco_processor.filename_to_info)} images.")
        except Exception as e:
            if rank == 0:
                print(f"Warning: Failed to load COCO annotations - {e}", file=sys.stderr)

    test_pipeline = build_test_pipeline(cfg).transforms[1].transforms
    all_files = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    # 每个进程只处理自己的文件子集
    local_files = [f for i, f in enumerate(all_files) if i % world_size == rank]

    dataset = InferenceDataset(local_files, test_pipeline)
    # 根据你的GPU显存调整 batch_size 和 num_workers
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    progress_bar = tqdm(total=len(all_files), desc="Processing", position=0, disable=(rank != 0))

    for batch_data in data_loader:
        valid_mask = [status == 'ok' for status in batch_data['status']]
        if not any(valid_mask):
            if rank == 0:
                progress_bar.update(len(batch_data['filename']))
            continue

        filenames = [fn for fn, m in zip(batch_data['filename'], valid_mask) if m]
        img_tensors = torch.stack([img for img, m in zip(batch_data['img'], valid_mask) if m]).to(device)

        with torch.no_grad():
            enhanced_batch = model.module.diffusion.forward_test(img_tensors)

        for i, filename in enumerate(filenames):
            enhanced_img_tensor = enhanced_batch[i]
            final_img = post_process_output(enhanced_img_tensor, cfg)

            if coco_processor:
                _, img_info = coco_processor.get_annotations(filename)
                if img_info:
                    target_size = (img_info['width'], img_info['height'])
                    final_img = resize_to_annotation_size(final_img, target_size)

            save_and_visualize(final_img, filename, output_dir, coco_processor)

        if rank == 0:
            progress_bar.update(len(batch_data['filename']))

    progress_bar.close()


def load_model_and_cfg(config_path: str, checkpoint_path: str, device: torch.device):
    """加载模型和配置"""
    cfg = mmcv.Config.fromfile(config_path)
    model = build_detector(cfg.model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    model.to(device)
    model.eval()
    return model, cfg


def build_test_pipeline(cfg: mmcv.Config) -> Compose:
    """构建测试数据流水线"""
    return Compose([
        dict(type='LoadImageFromFile'),
        dict(type='MultiScaleFlipAug',
             img_scale=(1920, 1080),
             flip=False,
             transforms=[
                 dict(type='Resize', keep_ratio=True),
                 dict(type='Normalize', **cfg.img_norm_cfg),
                 dict(type='Pad', size_divisor=32),
                 dict(type='ImageToTensor', keys=['img']),
                 dict(type='Collect', keys=['img'])
             ])
    ])


def init_distributed() -> Tuple[bool, int, int, int]:
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend='nccl', init_method='env://')
        return True, rank, world_size, gpu
    return False, 0, 1, 0


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='分布式图像增强与标注处理工具')
    parser.add_argument('--config', default=CONFIG_PATH, help='模型配置文件路径')
    parser.add_argument('--checkpoint', default=CHECKPOINT_PATH, help='模型权重文件路径')
    parser.add_argument('--input', default=INPUT_DIR, help='输入图像目录路径')
    parser.add_argument('--output', default=OUTPUT_DIR, help='输出目录路径')
    parser.add_argument('--annotation', default=ANNOTATION_PATH, help='COCO标注文件路径')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    distributed, rank, world_size, gpu = init_distributed()
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    try:
        if rank == 0:
            print(f"Loading model from {args.checkpoint}...")
        model, cfg = load_model_and_cfg(args.config, args.checkpoint, device)

        if distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

        model.eval()

        if rank == 0:
            print(f"\nProcessing images from {args.input}...")
        if not os.path.isdir(args.input):
            raise FileNotFoundError(f"Input directory not found: {args.input}")

        process_images_dist(model, args.input, args.output, args.annotation, cfg, rank, world_size, device)

        if dist.is_initialized():
            dist.barrier()

        if rank == 0:
            print("\n" + "=" * 50)
            print("All tasks completed successfully!")
            print(f"Results saved to:")
            print(f"- Resized enhanced images: {os.path.join(args.output, 'images')}")
            print(f"- Annotated visualized images: {os.path.join(args.output, 'visible')}")
            print("=" * 50)

    except Exception as e:
        print(f"\nAn error occurred on Rank {rank}: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()