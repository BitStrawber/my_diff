import argparse
import os
import os.path as osp
import torch
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import argparse  # 用于解析命令行参数
import copy  # 用于复制对象
import os  # 用于操作文件和目录
import os.path as osp  # 用于操作路径
import time  # 用于获取时间
import warnings  # 用于处理警告信息

import torch  # 用于深度学习模型的构建和训练
import mmcv  # MMDetection的依赖库，用于配置管理、日志记录等
import torch.distributed as dist  # 用于分布式训练
from mmcv import Config, DictAction  # 用于配置文件的加载和解析
from mmcv.runner import get_dist_info, init_dist  # 用于分布式训练的初始化和信息获取
from mmcv.utils import get_git_hash  # 用于获取Git哈希值

from mmdet import __version__  # 用于获取MMDetection的版本信息
from mmdet.apis import init_random_seed, set_random_seed, train_detector  # 用于初始化随机种子和训练检测器
from mmdet.datasets import build_dataset  # 用于构建数据集
from mmdet.models import build_detector  # 用于构建检测模型
from mmdet.utils import (collect_env, get_device, get_root_logger,  # 用于收集环境信息、获取设备信息和根日志记录器
                         replace_cfg_vals, setup_multi_processes,  # 用于替换配置值和设置多进程
                         update_data_root)  # 用于更新数据根目录
import sys  # 用于系统相关的操作
sys.path.append('./')  # 将当前目录添加到系统路径中
from EnDiff import *  # 导入自定义的EnDiff模块

def parse_args():
    parser = argparse.ArgumentParser(description='Generate high-quality images')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('input_image', help='path to the low-quality input image')
    parser.add_argument('output_image', help='path to save the generated high-quality image')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # 加载配置文件
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # 构建模型
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location=args.device)
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.cfg = cfg  # 保存配置文件到模型中
    model.to(args.device)
    model.eval()

    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 替换为目标图像大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载低质量图像
    low_quality_image = Image.open(args.input_image)
    low_quality_tensor = transform(low_quality_image).unsqueeze(0).to(args.device)  # 添加批次维度并移动到设备

    # 推理生成高质量图像
    with torch.no_grad():
        high_quality_tensor = model.forward_test(low_quality_tensor)

    # 反归一化
    def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    # 可视化
    high_quality_image = unnormalize(high_quality_tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    plt.imshow(high_quality_image)
    plt.axis('off')
    plt.show()

    # 保存生成的高质量图像
    save_image(high_quality_tensor, args.output_image)

if __name__ == '__main__':
    main()

