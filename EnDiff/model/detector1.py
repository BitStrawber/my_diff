import torch
import torch.nn.functional as F

from mmdet.models.builder import DETECTORS, MODELS
from mmdet.models.detectors import CascadeRCNN
from torch import nn


class DummyModule(nn.Module):
    """合法的虚拟模块"""
    def forward(self, *args, **kwargs):
        return None

@DETECTORS.register_module()
class EnDiffO(CascadeRCNN):
    def __init__(self, backbone, diff_cfg, **kwargs):
        super().__init__(backbone, **kwargs)
        self.diffusion = MODELS.build(diff_cfg)

        # 显式冻结所有非扩散参数
        for name, param in self.named_parameters():
            if 'diffusion' not in name:
                param.requires_grad = False

        # 替换为合法的虚拟模块
        self.rpn_head = DummyModule()
        self.roi_head = DummyModule()
        self.bbox_head = DummyModule()


    def extract_feat(self, x):
        return self.diffusion(x, return_loss=False)

    def forward_train(self,
                      img,
                      img_metas,
                      hq_img=None,
                      **kwargs):
        loss = self.diffusion(img, hq_img)
        return loss

    def forward_test(self, imgs, img_metas, hq_img=None, **kwargs):
        _ = hq_img
        return super().forward_test(imgs, img_metas, **kwargs)