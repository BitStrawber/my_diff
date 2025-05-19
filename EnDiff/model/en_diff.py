import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os
import torchvision.utils as vutils

from mmcv.runner import BaseModule
from mmdet.models.builder import MODELS


@MODELS.register_module()
class EnDiff(BaseModule):
    def __init__(
            self,
            net,
            T=1000,
            diffuse_ratio=0.6,
            sample_times=10,
            land_loss_weight=1,
            uw_loss_weight=1,
            init_cfg=None,
    ):
        super(EnDiff, self).__init__(init_cfg)
        self.net = MODELS.build(net)
        self.T = T
        self.diffuse_ratio = diffuse_ratio
        self.sample_times = sample_times
        self.t_end = int(self.T * self.diffuse_ratio)
        self.land_loss_weight = land_loss_weight
        self.uw_loss_weight = uw_loss_weight

        self.f_0 = math.cos(0.008 / 1.008 * math.pi / 2) ** 2
        self.t_list = list(range(0, self.t_end, self.t_end // self.sample_times)) + [self.t_end]

        # 初始化计数器
        self.counter = 0
    
    def get_alpha_cumprod(self, t: torch.Tensor):
        return (torch.cos((t / self.T + 0.008) / 1.008 * math.pi / 2) ** 2 / self.f_0)[:, None, None, None] 

    def q_diffuse(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        t = t.reshape(-1)
        alpha_cumprod = self.get_alpha_cumprod(t).to(x0.device)
        noise = torch.randn_like(x0) if noise is None else noise
        xt =  torch.sqrt(alpha_cumprod) * x0 + torch.sqrt(1 - alpha_cumprod) * noise
        return xt, noise
    # 根据图像与时间步长返回加噪后的图像与加上的噪声

    # 在 predict 方法中添加调试信息
    def predict(self, et: torch.Tensor, u0: torch.Tensor, t: torch.Tensor):
        e0 = self.net(et, t, u0)
        alpha_cumprod = self.get_alpha_cumprod(t)
        alpha_cumprod_prev = self.get_alpha_cumprod(t - (self.t_end // self.sample_times))
        noise = (et - torch.sqrt(alpha_cumprod) * e0) / torch.sqrt(1 - alpha_cumprod)
        e_prev = torch.sqrt(alpha_cumprod_prev) * e0 + torch.sqrt(1 - alpha_cumprod_prev) * noise
        return e0, e_prev, noise
    
    def loss(self, r_prev: torch.Tensor, h_prev: torch.Tensor, noise_pred: torch.Tensor, noise_gt: torch.Tensor):
        r = torch.flatten(r_prev, 1)
        h = torch.flatten(h_prev, 1)
        r = F.log_softmax(r, dim=-1)
        h = F.log_softmax(h, dim=-1)
        land_loss =  F.kl_div(r, h, log_target=True, reduction='batchmean') * self.land_loss_weight

        uw_loss = F.mse_loss(noise_pred, noise_gt, reduction='mean') * self.uw_loss_weight
        return dict(land_loss = land_loss, uw_loss = uw_loss)
    
    def forward_train(self, u0: torch.Tensor, h0: torch.Tensor):
        train_idx = random.randint(1, len(self.t_list) - 1)
        rs, noise_gt = self.q_diffuse(u0, torch.full((1,), self.t_end, device=u0.device))# 低质量图像加噪结果，加噪到底

        rt_prev = rs
        for i, t in enumerate(self.t_list[:train_idx - 1:-1]):
            _, rt_prev, noise_pred = self.predict(rt_prev, u0, torch.full((1,), t, device=u0.device))

        ht_prev, _ = self.q_diffuse(h0, torch.full((1,), self.t_list[train_idx - 1], device=u0.device))# 高质量图像加噪结果
        return self.loss(rt_prev, ht_prev, noise_pred, noise_gt)
    # 输入第质量与高质量图像，向loss中输入（预测去噪结果，真实目标数据，预测噪声，真实噪声）：
    
    def forward_test(self, u0):

        save_path = '/home/xcx/桌面/temp'
        rs, _ = self.q_diffuse(u0, torch.full((1,), self.t_end, device=u0.device))

        rt_prev = rs
        for i, t in enumerate(self.t_list[:1:-1]):
            r0, rt_prev, _ = self.predict(rt_prev, u0, torch.full((1,), t, device=u0.device))

        temp = r0
        # 将 r0 转换为 [0, 1] 范围
        temp = (temp - temp.min()) / (temp.max() - temp.min())
        # 生成文件名
        filename = f'output_image_{self.counter:04d}.png'  # 按序号生成文件名
        self.counter += 1  # 递增计数器
        # 保存图像
        save_path_full = os.path.join(save_path, filename)
        vutils.save_image(temp, save_path_full, normalize=True)

        return r0

    def forward_test0(self, u0, meta):
        save_path = '/home/xcx/桌面/temp'
        os.makedirs(save_path, exist_ok=True)  # 确保保存路径存在

        # 扩散过程
        rs, _ = self.q_diffuse(u0, torch.full((u0.size(0),), self.t_end, device=u0.device))

        rt_prev = rs
        for i, t in enumerate(self.t_list[:1:-1]):
            r0, rt_prev, _ = self.predict(rt_prev, u0, torch.full((u0.size(0),), t, device=u0.device))

        # 对生成的图像进行归一化
        temp = r0
        temp = (temp - temp.min()) / (temp.max() - temp.min())

        # 从元数据中提取文件名和原始尺寸
        filenames = [meta[i]['ori_filename'] for i in range(u0.size(0))]
        original_shapes = [meta[i]['ori_shape'] for i in range(u0.size(0))]

        # 保存每个生成的图像
        for i in range(u0.size(0)):
            filename = filenames[i]
            output_filename = f"output_{filename}"
            save_path_full = os.path.join(save_path, output_filename)

            # 确保生成的图像与原图像尺寸一致
            original_shape = original_shapes[i]
            temp_i = F.interpolate(temp[i].unsqueeze(0), size=original_shape[:2], mode='bilinear', align_corners=False)
            vutils.save_image(temp_i.squeeze(0), save_path_full, normalize=True)

        return r0

    
    def forward(self, u0: torch.Tensor, h0: torch.Tensor = None, return_loss: bool = True):
        if return_loss:
            assert h0 is not None
            return self.forward_train(u0, h0)
        else:
            return self.forward_test(u0)
