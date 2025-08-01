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
test = 1.0
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

        s = [15, 60, 90]
        self.guas_15 = MyGaussianBlur(radius=s[0], sigema=s[0]).cuda()
        self.guas_60 = MyGaussianBlur(radius=s[1], sigema=s[1]).cuda()
        self.guas_90 = MyGaussianBlur(radius=s[2], sigema=s[2]).cuda()

        self.temp_15 = self.guas_15.template()
        self.temp_60 = self.guas_60.template()
        self.temp_90 = self.guas_90.template()

    # 光场滤波器，获得光场信息并返回
    def MutiScaleLuminanceEstimation(self, img):
        x_15 = self.guas_15.filter(img, self.temp_15)
        x_60 = self.guas_60.filter(img, self.temp_60)
        x_90 = self.guas_90.filter(img, self.temp_90)
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

    def get_x0_from_noise(self, xt, t, noise):
        """根据 xt, t 和预测的噪声，反解出 x0"""
        alpha_cumprod = self.get_alpha_cumprod(t).to(xt.device)
        x0_pred = (xt - torch.sqrt(1 - alpha_cumprod) * noise) / torch.sqrt(alpha_cumprod)
        return x0_pred

    def forward_train(self, u0: torch.Tensor, h0: torch.Tensor):
        # 1. 为当前 batch 中的每个样本随机采样一个时间步 t
        t = torch.randint(1, self.t_end + 1, (u0.shape[0],), device=u0.device).long()

        # 2. 对低质量图像 u0 进行一步加噪，得到加噪图 ut 和真实噪声 noise_gt
        ut, noise_gt = self.q_diffuse(u0, t)

        # 3. 让网络从 (ut, t) 预测噪声，以 u0 作为条件
        noise_pred = self.net(ut, t, u0)

        # 4. (可选但推荐) 从预测的噪声反解出预测的 x0
        x0_pred = self.get_x0_from_noise(ut, t, noise_pred)
        # 也许需要对 x0_pred 进行 clamp，以匹配图像的范围，例如 [-1, 1] 或 [0, 1]
        # x0_pred.clamp_(-1., 1.)

        # ht, _ = self.q_diffuse(h0, t, noise=None)

        # 5. 将所有需要的张量传递给独立的 loss 函数
        return self.loss(x0_pred, h0, noise_pred, noise_gt)

    # 输入第质量与高质量图像，向loss中输入（预测去噪结果，真实目标数据，预测噪声，真实噪声）：

    def forward_test(self, u0):

        # save_path = '/home/xcx/桌面/temp0'
        rs, _ = self.q_diffuse(u0, torch.full((1,), self.t_end, device=u0.device))

        rt_prev = rs
        # 迭代采样生成最终图像
        # 注意 t_list 应该从 t_end 遍历到 1
        for t_val in reversed(self.t_list):
            if t_val == 0: continue
            t = torch.full((u0.shape[0],), t_val, device=u0.device)
            r0, rt_prev, _ = self.predict(rt_prev, u0, t)

        final_r0, _, _ = self.predict(rt_prev, u0, torch.full((u0.shape[0],), self.t_list[0], device=u0.device))

        return final_r0

    def forward(self, u0: torch.Tensor, h0: torch.Tensor = None, return_loss: bool = True):
        if return_loss:
            assert h0 is not None
            return self.forward_train(u0, h0)
        else:
            return self.forward_test(u0)

