import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

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
        # 根据配置构建基础网络
        self.net = MODELS.build(net)
        # 扩散总步数
        self.T = T
        # 扩散比例，决定扩散结束步长
        self.diffuse_ratio = diffuse_ratio
        # 采样次数
        self.sample_times = sample_times
        # 扩散结束步长
        self.t_end = int(self.T * self.diffuse_ratio)
        # 土地损失的权重
        self.land_loss_weight = land_loss_weight
        # 水下损失的权重
        self.uw_loss_weight = uw_loss_weight

        # 计算初始频率参数
        self.f_0 = math.cos(0.008 / 1.008 * math.pi / 2) ** 2
        # 生成时间步长列表，用于扩散和采样过程
        self.t_list = list(range(0, self.t_end, self.t_end // self.sample_times)) + [self.t_end]

    def get_alpha_cumprod(self, t: torch.Tensor):
        # 根据时间步长 t 计算累积 alpha 值，用于扩散和去噪过程
        return (torch.cos((t / self.T + 0.008) / 1.008 * math.pi / 2) ** 2 / self.f_0)[:, None, None, None]

    def q_diffuse(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        # 前向扩散过程，将输入 x0 扩散到时间步长 t
        t = t.reshape(-1)
        # 获取累积 alpha 值
        alpha_cumprod = self.get_alpha_cumprod(t).to(x0.device)
        # 如果未提供噪声，则生成随机噪声
        noise = torch.randn_like(x0) if noise is None else noise
        # 计算扩散后的特征 xt
        xt = torch.sqrt(alpha_cumprod) * x0 + torch.sqrt(1 - alpha_cumprod) * noise
        return xt, noise

    def predict(self, et: torch.Tensor, u0: torch.Tensor, t: torch.Tensor):
        # 噪声预测过程，用于去噪
        # 使用基础网络预测噪声
        e0 = self.net(et, t, u0)
        # 获取当前时间步长的累积 alpha 值
        alpha_cumprod = self.get_alpha_cumprod(t)
        # 获取前一时间步长的累积 alpha 值
        alpha_cumprod_prev = self.get_alpha_cumprod(t - (self.t_end // self.sample_times))
        # 计算噪声
        noise = (et - torch.sqrt(alpha_cumprod) * e0) / torch.sqrt(1 - alpha_cumprod)
        # 计算前一时间步长的特征 e_prev
        e_prev = torch.sqrt(alpha_cumprod_prev) * e0 + torch.sqrt(1 - alpha_cumprod_prev) * noise
        return e0, e_prev, noise

    def loss(self, r_prev: torch.Tensor, h_prev: torch.Tensor, noise_pred: torch.Tensor, noise_gt: torch.Tensor):
        # 计算损失函数，包括土地损失和水下损失
        # 展平特征
        r = torch.flatten(r_prev, 1)
        h = torch.flatten(h_prev, 1)
        # 计算 log 软概率
        r = F.log_softmax(r, dim=-1)
        h = F.log_softmax(h, dim=-1)
        # 计算 KL 散度作为土地损失
        land_loss = F.kl_div(r, h, log_target=True, reduction='batchmean') * self.land_loss_weight

        # 计算均方误差作为水下损失
        uw_loss = F.mse_loss(noise_pred, noise_gt, reduction='mean') * self.uw_loss_weight
        return dict(land_loss=land_loss, uw_loss=uw_loss)

    def forward_train(self, u0: torch.Tensor, h0: torch.Tensor):
        # 训练时的前向传播过程
        # 随机选择一个时间步长用于训练
        train_idx = random.randint(1, len(self.t_list) - 1)
        # 对输入 u0 进行前向扩散
        rs, noise_gt = self.q_diffuse(u0, torch.full((1,), self.t_end, device=u0.device))

        rt_prev = rs
        # 根据选定的时间步长进行反向推导
        for i, t in enumerate(self.t_list[:train_idx - 1:-1]):
            _, rt_prev, noise_pred = self.predict(rt_prev, u0, torch.full((1,), t, device=u0.device))

        # 获取前一时间步长的特征
        ht_prev, _ = self.q_diffuse(h0, torch.full((1,), self.t_list[train_idx - 1], device=u0.device))
        # 计算损失
        return self.loss(rt_prev, ht_prev, noise_pred, noise_gt)

    def forward_test(self, u0):
        # 测试时的前向传播过程，用于生成增强特征
        # 对输入 u0 进行前向扩散
        rs, _ = self.q_diffuse(u0, torch.full((1,), self.t_end, device=u0.device))

        rt_prev = rs
        # 逐步进行去噪操作
        for i, t in enumerate(self.t_list[:1:-1]):
            r0, rt_prev, _ = self.predict(rt_prev, u0, torch.full((1,), t, device=u0.device))
        # 返回最终的增强特征
        return r0

    def forward(self, u0: torch.Tensor, h0: torch.Tensor = None, return_loss: bool = True):
        # 统一的前向传播接口
        if return_loss:
            # 训练时返回损失
            assert h0 is not None
            return self.forward_train(u0, h0)
        else:
            # 测试时返回增强特征
            return self.forward_test(u0)