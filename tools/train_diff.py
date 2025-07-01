import os
import torch
import mmcv
import glob
from mmcv.runner import init_dist
from mmcv.utils import Config
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import get_root_logger


def find_latest_checkpoint(work_dir):
    """自动查找最新的检查点文件"""
    checkpoint_files = glob.glob(os.path.join(work_dir, 'epoch_*.pth'))
    if not checkpoint_files:
        return None

    # 按epoch数字排序
    checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    return checkpoint_files[-1]

def main():
    # 初始化配置
    config = 'config/diff.py'
    cfg = Config.fromfile(config)

    # 工作目录设置
    cfg.work_dir = 'work_dirs/diff01'
    mmcv.mkdir_or_exist(cfg.work_dir)

    # 初始化日志
    logger = get_root_logger(log_file=os.path.join(cfg.work_dir, 'train.log'))

    # 自动恢复逻辑
    if cfg.auto_resume:
        latest_checkpoint = find_latest_checkpoint(cfg.work_dir)
        if latest_checkpoint:
            cfg.resume_from = latest_checkpoint
            logger.info(f'Auto-resume found checkpoint: {latest_checkpoint}')

    # 手动指定的恢复路径优先级更高
    if cfg.get('resume_from'):
        cfg.auto_resume = False  # 禁用自动恢复
        logger.info(f'Manual resume from: {cfg.resume_from}')

        # 设置随机种子
        seed = cfg.get('seed', 42)
        set_random_seed(seed, deterministic=True)
        logger.info(f'Set random seed to {seed}')

        # 构建模型
        model = build_detector(cfg.model)
        logger.info(f'Model architecture:\n{model}')

        # 如果从检查点恢复，加载模型和优化器状态
        if cfg.get('resume_from'):
            load_checkpoint(model, cfg.resume_from, map_location='cpu')
            logger.info(f'Loaded checkpoint from {cfg.resume_from}')

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