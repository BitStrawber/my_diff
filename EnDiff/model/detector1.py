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

        # 构建扩散模型
        self.diffusion = MODELS.build(diff_cfg)

        # 冻结非扩散参数（兼容DDP）
        self._freeze_non_diffusion_params()

        # 替换检测头为虚拟模块
        self._replace_detection_heads()

        # 标记需要梯度更新的参数（DDP必需）
        self._setup_trainable_params()

    def _freeze_non_diffusion_params(self):
        """安全冻结参数（兼容DDP）"""
        for name, param in self.named_parameters():
            if 'diffusion' not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

    def _replace_detection_heads(self):
        """替换检测头并确保参数存在"""
        self.rpn_head = DummyModule()
        self.roi_head = DummyModule()
        self.bbox_head = DummyModule()

        # 确保参数注册到模型中
        for name, module in [('rpn_head', self.rpn_head),
                             ('roi_head', self.roi_head),
                             ('bbox_head', self.bbox_head)]:
            self.add_module(name, module)

    def _setup_trainable_params(self):
        """显式声明需要梯度的参数（DDP必需）"""
        self._trainable_params = [p for p in self.parameters() if p.requires_grad]

    def trainable_parameters(self):
        """DDP兼容的参数获取方法"""
        return (p for p in self.parameters() if p.requires_grad)

    def extract_feat(self, x):
        """重写特征提取方法"""
        return self.diffusion(x, return_loss=False)

    def forward_train(self, img, img_metas, hq_img=None, **kwargs):
        """训练前向传播（DDP兼容）"""
        if isinstance(img, list):  # 处理DDP的数据包装
            img = img[0]
            hq_img = hq_img[0] if hq_img is not None else None

        loss = self.diffusion(img, hq_img)
        return {'loss': loss}

    def forward_test(self, imgs, img_metas, hq_img=None, **kwargs):
        """测试前向传播（DDP兼容）"""
        if isinstance(imgs, list):  # 处理DDP的数据包装
            imgs = imgs[0]

        return self.diffusion(imgs, return_loss=False)

    def forward(self, img, img_metas=None, hq_img=None, return_loss=True, **kwargs):
        """统一入口（兼容MMDetection接口）"""
        if img_metas is None:
            img_metas = [{} for _ in range(len(img))]

        if return_loss:
            return self.forward_train(img, img_metas, hq_img, **kwargs)
        else:
            return self.forward_test(img, img_metas, hq_img, **kwargs)