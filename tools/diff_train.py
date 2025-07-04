# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import json
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
import sys
sys.path.append('./')
from EnDiff import *


def check_data_paths(cfg):
    """检查所有数据路径是否存在"""
    required_paths = [
        cfg.data.train.ann_file,
        cfg.data.train.img_prefix,
        cfg.data.val.ann_file,
        cfg.data.val.img_prefix
    ]

    missing_paths = []
    for path in required_paths:
        if not osp.exists(path):
            missing_paths.append(path)

    if missing_paths:
        raise FileNotFoundError(
            f"以下路径不存在:\n{'n'.join(missing_paths)}\n"
            f"当前工作目录: {os.getcwd()}"
        )
    else:
        print("✅ 所有数据路径验证通过")


def validate_dataset(cfg):
    """全面验证数据集完整性"""
    # 1. 路径检查
    print("\n=== 正在检查数据路径 ===")
    check_data_paths(cfg)

    # 2. 标注文件检查
    print("\n=== 正在检查标注文件 ===")
    for phase in ['train', 'val']:
        ann_file = getattr(cfg.data, phase).ann_file
        try:
            with open(ann_file) as f:
                ann = json.load(f)
                print(f"{phase}标注文件: 包含 {len(ann['images'])} 张图像, {len(ann['annotations'])} 个标注")
        except Exception as e:
            raise ValueError(f"{phase}标注文件解析失败: {str(e)}")

    # 3. 图像采样检查
    print("\n=== 正在抽样检查图像文件 ===")
    for phase in ['train', 'val']:
        dataset = build_dataset(getattr(cfg.data, phase))
        print(f"{phase}数据集总样本数: {len(dataset)}")

        # 检查前5个样本
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            img_path = sample['filename']
            hq_path = sample['hq_img_filename']

            print(f"\n样本 {i}:")
            print(f"LQ图像路径: {img_path} -> {'存在' if osp.exists(img_path) else '缺失'}")
            print(f"HQ图像路径: {hq_path} -> {'存在' if osp.exists(hq_path) else '缺失'}")

            if 'gt_bboxes' in sample:
                print(f"标注框数量: {len(sample['gt_bboxes'])}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train a diffusion model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file (deprecate), '
             'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)


    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 确保配置文件中没有检测相关的设置
    if 'detector' in cfg.model or 'roi_head' in cfg.model:
        warnings.warn('Detector related configurations found in the config file. '
                      'These will be ignored for diffusion model training.')

    # 自动调整学习率
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            warnings.warn('Can not find "auto_scale_lr" or '
                          '"auto_scale_lr.enable" or '
                          '"auto_scale_lr.base_batch_size" in your'
                          ' configuration file.')

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs_diff',
                                osp.splitext(osp.basename(args.config))[0])

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume

    # GPU设置
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated. Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    cfg.device = get_device()
    # set random seeds
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    # build model - 这里会构建你的扩散模型
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    # 添加DDP兼容性初始化
    if distributed:
        # 确保所有rank同步随机种子
        set_random_seed(cfg.seed + dist.get_rank())

        # 模型预处理
        model._is_init = True
        if hasattr(model, '_setup_trainable_params'):
            model._setup_trainable_params()

        # 分布式数据并行包装
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(),
            find_unused_parameters=True
        )

    # 修改数据加载部分
    def build_compatible_dataset(cfg):
        dataset = build_dataset(cfg.data.train)
        if distributed:
            from mmcv.parallel import collate
            dataset = torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg.data.samples_per_gpu,
                sampler=torch.utils.data.distributed.DistributedSampler(dataset),
                collate_fn=collate,
                num_workers=cfg.data.workers_per_gpu,
                pin_memory=True
            )
        return dataset

    datasets = [build_compatible_dataset(cfg)]

    # 修改训练配置
    cfg.optimizer_config.grad_clip = dict(max_norm=1.0)
    if distributed:
        cfg.optimizer_config.update(dict(
            _delete_=True,
            type='DistOptimizerHook',
            update_interval=1,
            grad_clip=dict(max_norm=1.0)
        ))

    # 验证集设置
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    # checkpoint配置
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)

    # 训练扩散模型
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()