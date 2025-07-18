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

r = 0
s = [15,60,90]
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
        kernel = torch.FloatTensor(template).cuda()
        kernel2 = kernel.view(1, 1, *kernel.shape)  # 将二维张量扩展为四维张量
        kernel2 = kernel2.expand(3, 1, *kernel.shape)  # 扩展为 [3, 1, 2 * r + 1, 2 * r + 1]
        weight = torch.nn.Parameter(data=kernel2, requires_grad=False)
        new_pic2 = torch.nn.functional.conv2d(image, weight, padding=self.radius, groups=3)
        return new_pic2

@MODELS.register_module()
class EnDiff(BaseModule):
    def __init__(
            self,
            net,
            T=1000,
            diffuse_ratio=1.0,
            sample_times=10,
            land_loss_weight=1,
            uw_loss_weight=1*test,
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
        # 初始化 L1 损失
        self.L1 = nn.L1Loss()

        # 初始化计数器
        self.counter = 0

    # 光场滤波器，获得光场信息并返回
    def MutiScaleLuminanceEstimation(self, img):
        guas_15 = MyGaussianBlur(radius=15, sigema=15).cuda()
        temp_15 = guas_15.template()

        guas_60 = MyGaussianBlur(radius=60, sigema=60).cuda()
        temp_60 = guas_60.template()

        guas_90 = MyGaussianBlur(radius=90, sigema=90).cuda()
        temp_90 = guas_90.template()

        x_15 = guas_15.filter(img, temp_15)
        x_60 = guas_60.filter(img, temp_60)
        x_90 = guas_90.filter(img, temp_90)

        img = (x_15 + x_60 + x_90) / 3
        return img

    def get_alpha_cumprod(self, t: torch.Tensor):
        return (torch.cos((t / self.T + 0.008) / 1.008 * math.pi / 2) ** 2 / self.f_0)[:, None, None, None]

    def q_diffuse(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        t = t.reshape(-1)
        alpha_cumprod = self.get_alpha_cumprod(t).to(x0.device)
        noise = torch.randn_like(x0) if noise is None else noise
        xt = torch.sqrt(alpha_cumprod) * x0 + torch.sqrt(1 - alpha_cumprod) * noise
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

        # # 提取光场特征
        r_prev_lf = self.MutiScaleLuminanceEstimation(r_prev)
        h_prev_lf = self.MutiScaleLuminanceEstimation(h_prev)

        r = torch.flatten(r_prev_lf, 1)
        h = torch.flatten(h_prev_lf, 1)
        r = F.log_softmax(r, dim=-1)
        h = F.log_softmax(h, dim=-1)
        # 计算光场特征的 L1 损失
        land_loss = F.kl_div(r, h, log_target=True, reduction='batchmean') * self.land_loss_weight

        uw_loss = F.mse_loss(noise_pred, noise_gt, reduction='mean') * self.uw_loss_weight
        return dict(land_loss=land_loss, uw_loss=uw_loss)

    def train_step(self, u0: torch.Tensor, h0: torch.Tensor):
        """
        专门用于训练的函数，计算 loss。
        这个逻辑之前在 forward_train 中。
        """
        train_idx = random.randint(1, len(self.t_list) - 1)
        # 如果是 DDP，需要同步 train_idx
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                tensor_idx = torch.tensor(train_idx, device=u0.device)
            else:
                tensor_idx = torch.zeros(1, dtype=torch.long, device=u0.device)
            torch.distributed.broadcast(tensor_idx, src=0)
            train_idx = tensor_idx.item()

        rs, noise_gt = self.q_diffuse(u0, torch.full((1,), self.t_end, device=u0.device))
        rt_prev = rs
        noise_pred = noise_gt  # 初始化

        if train_idx < len(self.t_list):
            for i, t in enumerate(self.t_list[:train_idx - 1:-1]):
                _, rt_prev, noise_pred = self.predict(rt_prev, u0, torch.full((1,), t, device=u0.device))

        return self.loss(rt_prev, h0, noise_pred, noise_gt)

    def inference_step(self, u0: torch.Tensor):
        """
        专门用于推理的函数，生成增强图像。
        这个逻辑之前在 forward_test 中。
        """
        rs, _ = self.q_diffuse(u0, torch.full((1,), self.t_end, device=u0.device))
        rt_prev = rs
        r0 = u0  # 初始化

        for i, t in enumerate(self.t_list[:0:-1]):  # 修正循环，不包括 t=0
            r0, rt_prev, _ = self.predict(rt_prev, u0, torch.full((1,), t, device=u0.device))

        return r0

    def forward(self, u0: torch.Tensor, h0: torch.Tensor = None, return_loss: bool = True, **kwargs):
        """
        统一的 forward 入口，由包装器调用。
        """
        if return_loss:
            assert h0 is not None, "High-quality image (h0) is required for training."
            return self.train_step(u0, h0)
        else:
            return self.inference_step(u0)

