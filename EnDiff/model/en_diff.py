import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os
import torchvision.utils as vutils
import numpy as np

from mmcv.runner import BaseModule
from mmdet.models.builder import MODELS

# 全局变量 r 和 s 在你的类中没有被使用，可以安全地保留或删除
r = 0
s = [15, 60, 90]
test = 0.1


class MyGaussianBlur(torch.nn.Module):
    # 初始化
    def __init__(self, radius=1, sigema=1.5):
        super(MyGaussianBlur, self).__init__()
        self.radius = radius
        self.sigema = sigema

    # 高斯的计算公式
    def calc(self, x, y):
        res1 = 1 / (2 * math.pi * self.sigema * self.sigema)
        res2 = math.exp(-(x * x + y * y) / (2 * self.sigema * self.sigema))
        return res1 * res2

    # 滤波模板
    def template(self):
        sideLength = self.radius * 2 + 1
        result = np.zeros((sideLength, sideLength))
        for i in range(0, sideLength):
            for j in range(0, sideLength):
                result[i, j] = self.calc(i - self.radius, j - self.radius)
        all = result.sum()
        return result / all

    # 滤波函数
    def filter(self, image, template):
        # 确保在 CUDA 环境下运行
        if image.is_cuda and not torch.cuda.is_available():
            raise RuntimeError("Image is on CUDA but CUDA is not available.")

        device = image.device
        kernel = torch.FloatTensor(template).to(device)
        kernel = kernel.view(1, 1, *kernel.shape)  # 将二维张量扩展为四维张量

        # 扩展到3个通道
        # 如果image是单通道，则不需要expand
        if image.shape[1] == 3:
            kernel = kernel.expand(3, 1, *kernel.shape[2:])  # 扩展为 [3, 1, H, W]
            groups = 3
        else:
            groups = 1

        weight = torch.nn.Parameter(data=kernel, requires_grad=False)
        # 使用 image.shape[1] 自动确定 groups
        new_pic2 = torch.nn.functional.conv2d(image, weight, padding=self.radius, groups=groups)
        return new_pic2


# =============================================================================
# === 修正后的 sinkhorn_wasserstein 函数 ===
# =============================================================================
def sinkhorn_wasserstein(x, y, p=2, blur=0.05, scaling=0.5, iterations=50):
    """
    改进的Sinkhorn近似，支持多通道空间特征
    Args:
        x, y: (B, C, H, W) 的光场特征
        blur: 熵正则化系数（建议0.01~0.1）
    """
    B, num_channels, H, W = x.shape  # <--- 修正1：将 C 重命名为 num_channels
    x = x.reshape(B, num_channels, -1)  # (B, C, H*W)
    y = y.reshape(B, num_channels, -1)

    # 计算逐通道的Wasserstein距离
    total_dist = 0
    for c in range(num_channels):  # <--- 使用 num_channels
        x_c = x[:, c]  # (B, H*W)
        y_c = y[:, c]  # (B, H*W)

        # 计算成本矩阵（L2距离）
        cost_matrix = torch.cdist(x_c.unsqueeze(0), y_c.unsqueeze(0), p=p).squeeze(0)  # 更高效的方式计算成本矩阵

        # Sinkhorn迭代
        K = torch.exp(-cost_matrix / blur)
        u = torch.ones(B, H * W, device=x.device) / (H * W)
        v = torch.ones(B, H * W, device=y.device) / (H * W)

        for _ in range(iterations):
            u_ = u.transpose(0, 1)  # (H*W, B)
            v_ = v.transpose(0, 1)  # (H*W, B)

            # 为了数值稳定性，添加一个小的epsilon
            v_ = 1.0 / (K.T @ u_).clamp_min(1e-6)
            u_ = 1.0 / (K @ v_).clamp_min(1e-6)

            u = u_.T
            v = v_.T

        # 计算当前通道的距离
        P = u.unsqueeze(2) * K * v.unsqueeze(1)
        total_dist += (P * cost_matrix).sum(dim=(1, 2)).mean()  # 批平均

    return total_dist / num_channels  # <--- 修正2：除以正确的通道数


@MODELS.register_module()
class EnDiff(BaseModule):
    def __init__(
            self,
            net,
            T=1000,
            diffuse_ratio=0.6,
            sample_times=10,
            land_loss_weight=1,
            uw_loss_weight=1 * test,
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
        self.L1 = nn.L1Loss()
        self.counter = 0

    def MutiScaleLuminanceEstimation(self, img):
        # 懒加载高斯模糊模块以确保它们在正确的设备上
        if not hasattr(self, 'guas_15'):
            self.guas_15 = MyGaussianBlur(radius=15, sigema=15).to(img.device)
            self.temp_15 = self.guas_15.template()
            self.guas_60 = MyGaussianBlur(radius=60, sigema=60).to(img.device)
            self.temp_60 = self.guas_60.template()
            self.guas_90 = MyGaussianBlur(radius=90, sigema=90).to(img.device)
            self.temp_90 = self.guas_90.template()

        x_15 = self.guas_15.filter(img, self.temp_15)
        x_60 = self.guas_60.filter(img, self.temp_60)
        x_90 = self.guas_90.filter(img, self.temp_90)
        return (x_15 + x_60 + x_90) / 3

    def get_alpha_cumprod(self, t: torch.Tensor):
        return (torch.cos((t / self.T + 0.008) / 1.008 * math.pi / 2) ** 2 / self.f_0)[:, None, None, None]

    def q_diffuse(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        t = t.reshape(-1)
        alpha_cumprod = self.get_alpha_cumprod(t).to(x0.device)
        noise = torch.randn_like(x0) if noise is None else noise
        xt = torch.sqrt(alpha_cumprod) * x0 + torch.sqrt(1 - alpha_cumprod) * noise
        return xt, noise

    def predict(self, et: torch.Tensor, u0: torch.Tensor, t: torch.Tensor):
        e0 = self.net(et, t, u0)
        alpha_cumprod = self.get_alpha_cumprod(t).to(et.device)
        alpha_cumprod_prev = self.get_alpha_cumprod(t - (self.t_end // self.sample_times)).to(et.device)

        # 防止 alpha_cumprod 接近 1 时分母为0
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod).clamp_min(1e-6)
        noise = (et - torch.sqrt(alpha_cumprod) * e0) / sqrt_one_minus_alpha_cumprod

        e_prev = torch.sqrt(alpha_cumprod_prev) * e0 + torch.sqrt(1.0 - alpha_cumprod_prev) * noise
        return e0, e_prev, noise

    # =============================================================================
    # === 修正后的 loss 函数 ===
    # =============================================================================
    def loss(self, r_prev: torch.Tensor, h_prev: torch.Tensor, noise_pred: torch.Tensor, noise_gt: torch.Tensor):
        # 提取光场特征
        r_prev_lf = self.MutiScaleLuminanceEstimation(r_prev)
        h_prev_lf = self.MutiScaleLuminanceEstimation(h_prev)

        # 修正1：在使用 H 和 W 之前，从张量形状中获取它们
        B, C, H, W = r_prev_lf.shape

        # 对每个空间位置独立归一化（保持空间结构）
        r_normalized = F.layer_norm(r_prev_lf, [H, W])
        h_normalized = F.layer_norm(h_prev_lf, [H, W])

        # 计算光场特征的 Wasserstein 距离
        land_loss = sinkhorn_wasserstein(r_normalized, h_normalized) * self.land_loss_weight

        uw_loss = F.mse_loss(noise_pred, noise_gt, reduction='mean') * self.uw_loss_weight
        return dict(land_loss=land_loss, uw_loss=uw_loss)

    def forward_train(self, u0: torch.Tensor, h0: torch.Tensor):
        # 确保t_list不为空
        if len(self.t_list) <= 1:
            raise ValueError("t_list is too short. Check T, diffuse_ratio, and sample_times.")

        train_idx = random.randint(1, len(self.t_list) - 1)

        # 确保张量在同一设备上
        device = u0.device

        rs, noise_gt = self.q_diffuse(u0, torch.full((u0.shape[0],), self.t_end, device=device))

        rt_prev = rs
        # 从高到低遍历时间步
        for t_val in reversed(self.t_list[train_idx:]):
            _, rt_prev, noise_pred = self.predict(rt_prev, u0, torch.full((u0.shape[0],), t_val, device=device))

        ht_prev, _ = self.q_diffuse(h0, torch.full((h0.shape[0],), self.t_list[train_idx - 1], device=device))

        return self.loss(rt_prev, ht_prev, noise_pred, noise_gt)

    def forward_test(self, u0):
        device = u0.device
        rs, _ = self.q_diffuse(u0, torch.full((u0.shape[0],), self.t_end, device=device))

        rt_prev = rs
        r0 = None  # 初始化r0
        # 从高到低遍历时间步，但不包括 t=0
        for t_val in reversed(self.t_list[1:]):  # t_list[0] is 0
            r0, rt_prev, _ = self.predict(rt_prev, u0, torch.full((u0.shape[0],), t_val, device=device))

        # 如果循环没有执行（t_list太短），则返回 u0
        if r0 is None:
            r0, _, _ = self.predict(rt_prev, u0, torch.full((u0.shape[0],), self.t_list[1], device=device))

        return r0

    def forward(self, u0: torch.Tensor, h0: torch.Tensor = None, return_loss: bool = True):
        if return_loss:
            assert h0 is not None
            return self.forward_train(u0, h0)
        else:
            return self.forward_test(u0)