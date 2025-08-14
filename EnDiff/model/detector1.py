import torch
import torch.nn.functional as F

from mmdet.models.builder import DETECTORS, MODELS
from mmdet.models.detectors.base import BaseDetector


@DETECTORS.register_module()
class DiffusionOnlyWrapper(BaseDetector):
    """
    一个专门用于运行扩散模型的极简包装器。
    它继承自 BaseDetector 以便能被 MMDetection 的 runner 识别和训练。
    """

    def __init__(self, diff_cfg, init_cfg=None, train_cfg=None, test_cfg=None, **kwargs):
        super().__init__(init_cfg)
        # 唯一的组件就是我们的扩散模型
        self.diffusion = MODELS.build(diff_cfg)

        # 移除 backbone, neck, head 等不必要的属性
        # BaseDetector 会尝试调用它们，我们设为 None 来避免错误
        self.backbone = nn.Identity()  # 使用一个占位符，什么都不做
        self.neck = None
        self.rpn_head = None
        self.roi_head = None
        self.bbox_head = None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        # 这个方法必须实现，但我们在这里什么都不做，因为特征提取在 diffusion 模型内部
        # 或者直接返回图像本身，传递给 forward_train/forward_test
        return img

    def forward_train(self, img, hq_img, **kwargs):
        """
        训练时，直接调用扩散模型的 forward_train (即 loss 计算)
        img 是低质量图像 u0, hq_img 是高质量图像 h0
        """
        # **kwargs 会包含 img_metas, gt_bboxes 等，我们直接忽略它们
        losses = self.diffusion(u0=img, h0=hq_img, return_loss=True)
        return losses

    def simple_test(self, img, **kwargs):
        """
        测试时，直接调用扩散模型的 forward_test (即推理)
        """
        # **kwargs 会包含 img_metas 等，我们忽略
        # diffusion.forward_test 只需要低质量图像 u0
        enhanced_img = self.diffusion(u0=img, return_loss=False)

        # MMDetection 的测试流程期望返回一个特定格式的列表
        # 这里我们可以返回增强后的图像，但需要包装一下
        # 或者，如果只是想生成图像而不进行评估，可以返回一个空列表
        # 如果你想在MMDet的评估流程中保存图像，需要自定义评估钩子

        # 为了能跑通测试流程，返回一个符合格式的空结果
        # [B, C, H, W] -> list of [C, H, W]
        results = [img for img in enhanced_img]
        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        # 如果使用 MultiScaleFlipAug，需要实现 aug_test
        # 这里我们只对第一张（非增强）图像进行处理
        return self.simple_test(imgs[0], img_metas[0], **kwargs)

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        MMDetection 的测试入口
        """
        return self.simple_test(imgs, **kwargs)