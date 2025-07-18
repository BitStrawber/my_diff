# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import torch
import mmcv
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
# 【修改1】移除 mmdet.apis.train_detector，因为我们将使用 mmcv 的 runner
# from mmdet.apis import train_detector
from mmdet.apis import init_random_seed, set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
# 【修改2】导入 mmcv 的 EpochBasedRunner 和 build_optimizer
from mmcv.runner import (EpochBasedRunner, build_optimizer,
                         build_runner)


# 【修改3】移除不必要的导入
# import sys
# sys.path.append('./')
# from EnDiff import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train a diffusion model using MMDetection framework')
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
    # 【修改4】移除 train-mode 和 test-num 参数，因为它们不再需要
    # parser.add_argument(
    #     '--train-mode', ... )
    parser.add_argument(
        '--test-num', ... )

    # auto-scale-lr 参数可以保留，因为它很有用
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

    # 【修改5】 auto-scale-lr 逻辑可以简化或直接使用
    # mmdet 的 auto_scale_lr 是在 train.py 的顶层实现的，我们在这里手动实现一个简化版
    if args.auto_scale_lr and 'auto_scale_lr' in cfg and cfg.auto_scale_lr.get('enable', False):
        # get num of gpus
        if args.launcher == 'none':
            gpus = len(cfg.gpu_ids)
        else:
            _, world_size = get_dist_info()
            gpus = world_size

        # calculate total batch size
        total_batch_size = cfg.data.samples_per_gpu * gpus
        base_batch_size = cfg.auto_scale_lr.get('base_batch_size')
        if base_batch_size:
            factor = total_batch_size / base_batch_size
            cfg.optimizer.lr = cfg.optimizer.lr * factor
            print(f'LR scaled by {factor}, new LR is {cfg.optimizer.lr}')

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume

    # GPU IDs setup - this part is mostly fine
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('... single GPU mode ...')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('... use --gpu-id ...')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # 【修改6】移除所有 train_mode 相关的逻辑
    # if args.train_mode == 'diff':
    #     ...

    # init distributed env first
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        torch.cuda.set_device(args.local_rank)
        init_dist(args.launcher, **cfg.dist_params)
        rank, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir and logger
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    if hasattr(cfg.model, 'diff_cfg') and 'uw_loss_weight' in cfg.model.diff_cfg:
        original_weight = cfg.model.diff_cfg['uw_loss_weight']
        cfg.model.diff_cfg['uw_loss_weight'] = original_weight * args.test_num
        logger.info(
            f'Adjusted uw_loss_weight: {original_weight} * {args.test_num} = {cfg.model.diff_cfg["uw_loss_weight"]}')

    # log env info
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    cfg.device = get_device()
    # set random seeds
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed and distributed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    # Build the model
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # Add CLASSES to the model
    if hasattr(datasets[0], 'CLASSES'):
        model.CLASSES = datasets[0].CLASSES

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES if hasattr(datasets[0], 'CLASSES') else None)

    # 【修改8】使用更通用的 MMCV 训练流程，而不是 mmdet.apis.train_detector
    # DDP wrapping
    if distributed:
        device = torch.device('cuda', args.local_rank)
        model = model.to(device)
        # find_unused_parameters 仍然是好主意，因为扩散模型可能有些复杂
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    # Build the runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    runner.register_training_hooks(
        cfg.lr_config,
        cfg.optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if args.resume_from:
        runner.resume(args.resume_from)
    elif cfg.get('auto_resume'):
        runner.resume(osp.join(cfg.work_dir, 'latest.pth'))

    runner.run(datasets, cfg.workflow, max_epochs=cfg.runner.max_epochs)


if __name__ == '__main__':
    main()