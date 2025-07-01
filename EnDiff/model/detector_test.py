import torch
import torch.nn.functional as F

from mmdet.models.builder import DETECTORS, MODELS
from mmdet.models.detectors import CascadeRCNN

@DETECTORS.register_module()
class EnDiffDet0(CascadeRCNN):
    def __init__(self, backbone, diff_cfg, **kwargs):
        super().__init__(backbone, **kwargs)
        self.diffusion = MODELS.build(diff_cfg)
        self.train_mode = 'det'

    def train_mode_control(self, train_mode):
        def freeze_module(module):
            for p in module.parameters():
                p.requires_grad = False
            module.eval()
        def unfreeze_module(module):
            for p in module.parameters():
                p.requires_grad = True
            module.train()

        assert train_mode in ['det', 'sample']

        # 使用DistributedDataParallel的no_sync上下文管理器
        if hasattr(self, 'module'):  # 处理DDP包装的情况
            model = self.module
        else:
            model = self

        if train_mode == 'det':  # train detection
            # 先解冻所有参数
            for p in model.parameters():
                p.requires_grad = True

            # 然后冻结扩散部分
            for p in model.diffusion.parameters():
                p.requires_grad = False
            model.diffusion.eval()

        elif train_mode == 'sample':  # train diffusion
            # 先冻结所有参数
            for p in model.parameters():
                p.requires_grad = False

            # 然后解冻扩散网络
            for p in model.diffusion.net.parameters():
                p.requires_grad = True
            model.diffusion.net.train()

        # 确保所有进程完成模式切换
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # 额外的同步检查
        if torch.distributed.is_initialized():
            # 检查关键参数的requires_grad状态是否一致
            key_param = next(model.parameters())
            requires_grad = torch.tensor(key_param.requires_grad,
                                         device=key_param.device)
            torch.distributed.broadcast(requires_grad, src=0)
            assert requires_grad.item() == key_param.requires_grad

    def extract_feat(self, x):
        x = self.diffusion(x, return_loss=False)
        x = self.backbone(x)

        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      hq_img=None,
                      **kwargs):
        if self.train_mode == 'det':
            return super().forward_train(img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, proposals, **kwargs)
        elif self.train_mode == 'sample':
            loss = self.diffusion(img, hq_img)
            return loss

    def forward_test(self, imgs, img_metas, hq_img=None, **kwargs):
        _ = hq_img
        return super().forward_test(imgs, img_metas, **kwargs)