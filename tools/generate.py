# Copyright (c) OpenMMLab. All rights reserved.
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
    # 定义一个函数用于解析命令行参数
    parser = argparse.ArgumentParser(description='Train a detector')  # 创建一个参数解析器
    parser.add_argument('config', help='train config file path')  # 添加一个位置参数，用于指定训练配置文件路径
    parser.add_argument('--work-dir', help='the dir to save logs and models')  # 添加一个可选参数，用于指定保存日志和模型的目录
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')  # 添加一个可选参数，用于指定恢复训练的检查点文件
    parser.add_argument('--auto-resume', action='store_true', help='resume from the latest checkpoint automatically')  # 添加一个可选参数，用于自动恢复最近的检查点
    parser.add_argument('--no-validate', action='store_true', help='whether not to evaluate the checkpoint during training')  # 添加一个可选参数，用于在训练期间是否不评估检查点
    group_gpus = parser.add_mutually_exclusive_group()  # 添加一个互斥参数组，用于GPU相关的选项
    group_gpus.add_argument('--gpus', type=int, help='number of gpus to use (only applicable to non-distributed training)')  # 添加一个可选参数，用于指定使用的GPU数量
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+', help='ids of gpus to use (only applicable to non-distributed training)')  # 添加一个可选参数，用于指定使用的GPU IDs
    group_gpus.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use (only applicable to non-distributed training)')  # 添加一个可选参数，用于指定使用的GPU ID，默认为0
    parser.add_argument('--seed', type=int, default=None, help='random seed')  # 添加一个可选参数，用于指定随机种子
    parser.add_argument('--diff-seed', action='store_true', help='Whether or not set different seeds for different ranks')  # 添加一个可选参数，用于是否为不同的进程设置不同的种子
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')  # 添加一个可选参数，用于是否设置确定性选项以提高可重复性
    parser.add_argument('--options', nargs='+', action=DictAction, help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file (deprecate), change to --cfg-options instead.')  # 添加一个可选参数，用于覆盖配置文件中的某些设置（已弃用）
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white space is allowed.')  # 添加一个可选参数，用于覆盖配置文件中的某些设置
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')  # 添加一个可选参数，用于指定作业启动器
    parser.add_argument('--local_rank', type=int, default=0)  # 添加一个可选参数，用于指定本地进程的排名
    parser.add_argument('--auto-scale-lr', action='store_true', help='enable automatically scaling LR.')  # 添加一个可选参数，用于是否自动缩放学习率
    args = parser.parse_args()  # 解析命令行参数
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)  # 设置环境变量LOCAL_RANK

    if args.options and args.cfg_options:
        raise ValueError('--options and --cfg-options cannot be both specified, --options is deprecated in favor of --cfg-options')  # 如果同时指定了--options和--cfg-options，则抛出错误
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')  # 如果指定了--options，则发出警告
        args.cfg_options = args.options

    return args


def main():
    # 定义主函数
    args = parse_args()  # 解析命令行参数

    cfg = Config.fromfile(args.config)  # 从配置文件加载配置

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)  # 替换配置文件中的变量

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)  # 根据MMDET_DATASETS更新数据根目录

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)  # 如果指定了--cfg-options，则合并到配置中

    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and 'enable' in cfg.auto_scale_lr and 'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            warnings.warn('Can not find "auto_scale_lr" or "auto_scale_lr.enable" or "auto_scale_lr.base_batch_size" in your configuration file. Please update all the configuration files to mmdet >= 2.24.1.')

    # set multi-process settings
    setup_multi_processes(cfg)  # 设置多进程环境

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True  # 启用cudnn基准测试以加速训练

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])  # 设置默认的工作目录

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support single GPU mode in non-distributed training. Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. Because we only support single GPU mode in non-distributed training. Use the first GPU in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)  # 初始化分布式环境
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))  # 创建工作目录
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))  # 将配置文件保存到工作目录
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)  # 初始化日志记录器

    # init the meta dict to record some important information such as environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    cfg.device = get_device()  # 获取设备信息
    # set random seeds
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))  # 根据配置构建检测模型
    model.init_weights()  # 初始化模型权重

    datasets = [build_dataset(cfg.data.train)]  # 构建训练数据集
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))  # 构建验证数据集
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)  # 开始训练检测模型


if __name__ == '__main__':
    main()  # 执行主函数