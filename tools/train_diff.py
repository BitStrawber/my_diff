import os
import torch
import mmcv
from mmcv.runner import init_dist
from mmcv.utils import Config
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import get_root_logger


def main():
    # 初始化配置
    config = 'config/diff.py'
    cfg = Config.fromfile(config)

    # 工作目录设置
    cfg.work_dir = 'work_dirs/diff01'
    mmcv.mkdir_or_exist(cfg.work_dir)

    # 初始化日志
    logger = get_root_logger(log_file=os.path.join(cfg.work_dir, 'train.log'))

    # 设置随机种子
    seed = 42
    set_random_seed(seed, deterministic=True)
    logger.info(f'Set random seed to {seed}')

    # 构建模型
    model = build_detector(cfg.model)
    logger.info(f'Model architecture:\n{model}')

    # 构建数据集
    datasets = [build_dataset(cfg.data.train)]

    # 分布式训练设置
    if torch.cuda.is_available():
        distributed = True
        init_dist('pytorch', **cfg.dist_params)
    else:
        distributed = False
        logger.warning('CUDA not available, using CPU mode')

    # 训练参数优化
    cfg.optimizer_config = dict(grad_clip=dict(max_norm=1.0))
    cfg.evaluation.save_best = 'loss'

    # 开始训练
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=True,
        timestamp=mmcv.get_time_str(),
        meta={'exp_name': 'diff_pm'}
    )


if __name__ == '__main__':
    main()