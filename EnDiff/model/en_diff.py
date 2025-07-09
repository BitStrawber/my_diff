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

# 全局变量 r, s, test 在类中没有直接使用，但我们保留它们
r = 0
s = [15, 60, 90]
test = 0.1


# =============================================================================
# === 新增：高效稳定的 Sinkhorn-Wasserstein 距离函数 ===
# =============================================================================
def sinkhorn_wasserstein(x, y, p=2, blur=0.05, iterations=20, reach=0.5):
    """
    计算两个特征图 (B, C, H, W) 之间的 Sinkhorn-Wasserstein 距离。
    这是一个近似的Wasserstein距离，计算效率更高。

    Args:
        x, y: 输入的特征图，形状为 (B, C, H, W)。
        p: 成本矩阵中使用的范数 (通常为1或2)。
        blur: 熵正则化系数，值越小越接近真实Wasserstein距离，但可能不稳定。
        iterations: Sinkhorn算法的迭代次数。
        reach: Sinkhorn算法的缩放因子，用于提高收敛性。
    """
    # 将特征图展平为 (B, C, N)，其中 N = H * W
    x = x.flatten(2)
    y = y.flatten(2)

    # 计算成本矩阵 C(i,j) = ||x_i - y_j||_p^p
    # torch.cdist 会自动处理批次和通道维度
    cost_matrix = torch.cdist(x, y, p=p)  # Shape: (B, C, N, N)

    # 计算核矩阵 K = exp(-C / blur)
    K = torch.exp(-cost_matrix / blur)

    # 初始化均匀分布的边际概率 a 和 b
    B, C, N, _ = K.shape
    a = torch.ones(B, C, N, device=x.device) / N
    b = torch.ones(B, C, N, device=y.device) / N

    v = b  # 初始化Sinkhorn迭代中的 v

    # Sinkhorn迭代
    for _ in range(iterations):
        u = a / (K @ v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-6)
        v = b / (K.transpose(-2, -1) @ u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-6)

    # 计算传输矩阵 P = diag(u) * K * diag(v)
    P = u.unsqueeze(-1) * K * v.unsqueeze(-2)

    # 计算Wasserstein距离 sum(P * C)
    dist = torch.sum(P * cost_matrix, dim=(-1, -2))

    # 对批次和通道取平均
    return dist.mean()


class MyGaussianBlur(torch.nn.Module):
    def __init__(self, radius=1, sigema=1.5):
        super(MyGaussianBlur, self).__init__()
        self.radius = radius
        self.sigema = sigema

    def calc(self, x, y):
        res1 = 1 / (2 * math.pi * self.sigema * self.sigema)
        res2 = math.exp(-(x * x + y * y) / (2 * self.sigema * self.sigema))
        return res1 * res2

    def template(self):
        sideLength = self.radius * 2 + 1
        result = np.zeros((sideLength, sideLength))
        for i in range(sideLength):
            for j in range(sideLength):
                result[i, j] = self.calc(i - self.radius, j - self.radius)
        all = result.sum()
        return result / all

    # === 修改点1：修复 filter 函数中的设备硬编码问题 ===
    def filter(self, image, template):
        # 让 kernel 自动适应 image 所在的设备 (cpu或cuda:n)
        device = image.device
        kernel = torch.FloatTensor(template).to(device)
        kernel = kernel.view(1, 1, *kernel.shape)

        num_channels = image.shape[1]
        kernel = kernel.expand(num_channels, 1, *kernel.shape[2:])
        weight = torch.nn.Parameter(data=kernel, requires_grad=False)

        # groups 参数应等于通道数，以实现逐通道卷积
        return F.conv2d(image, weight, padding=self.radius, groups=num_channels)


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

        # 确保 t_end // sample_times 不为0
        step = max(1, self.t_end // self.sample_times)
        self.t_list = list(range(0, self.t_end, step)) + [self.t_end]

        self.L1 = nn.L1Loss()
        self.counter = 0

    # === 修改点2：懒加载高斯模块，确保在正确的设备上初始化 ===
    def MutiScaleLuminanceEstimation(self, img):
        # 第一次调用时，在 img 所在的设备上创建高斯模糊模块
        if not hasattr(self, 'guas_15'):
            device = img.device
            self.guas_15 = MyGaussianBlur(radius=15, sigema=15).to(device)
            self.temp_15 = self.guas_15.template()
            self.guas_60 = MyGaussianBlur(radius=60, sigema=60).to(device)
            self.temp_60 = self.guas_60.template()
            self.guas_90 = MyGaussianBlur(radius=90, sigema=90).to(device)
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
        device = et.device
        alpha_cumprod = self.get_alpha_cumprod(t).to(device)

        time_step = max(1, self.t_end // self.sample_times)
        t_prev = (t - time_step).clamp_min(0)  # 确保 t_prev 不为负
        alpha_cumprod_prev = self.get_alpha_cumprod(t_prev).to(device)

        # 为数值稳定性添加 clamp_min
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod).clamp_min(1e-6)
        noise = (et - torch.sqrt(alpha_cumprod) * e0) / sqrt_one_minus_alpha_cumprod

        e_prev = torch.sqrt(alpha_cumprod_prev) * e0 + torch.sqrt(1.0 - alpha_cumprod_prev) * noise
        return e0, e_prev, noise

    # === 修改点3：将 land_loss 修改为 Wasserstein 距离 ===
    def loss(self, r_prev: torch.Tensor, h_prev: torch.Tensor, noise_pred: torch.Tensor, noise_gt: torch.Tensor):
        # 1. 提取光场特征 (与原来相同)
        r_prev_lf = self.MutiScaleLuminanceEstimation(r_prev)
        h_prev_lf = self.MutiScaleLuminanceEstimation(h_prev)

        # 2. 计算 land_loss (使用 Wasserstein 距离)
        #    - 不再需要 flatten 和 log_softmax
        #    - 直接将特征图传入 sinkhorn_wasserstein 函数
        land_loss = sinkhorn_wasserstein(r_prev_lf, h_prev_lf) * self.land_loss_weight

        # 3. 计算 uw_loss (与原来相同)
        uw_loss = F.mse_loss(noise_pred, noise_gt, reduction='mean') * self.uw_loss_weight

        return dict(land_loss=land_loss, uw_loss=uw_loss)

    def forward_train(self, u0: torch.Tensor, h0: torch.Tensor):
        device = u0.device
        batch_size = u0.shape[0]

        train_idx = random.randint(1, len(self.t_list) - 1)

        rs, noise_gt = self.q_diffuse(u0, torch.full((batch_size,), self.t_end, device=device, dtype=torch.long))

        rt_prev = rs
        noise_pred = None  # 初始化以防循环不执行

        # 确保循环至少执行一次以定义 noise_pred
        for t_val in reversed(self.t_list[train_idx:]):
            _, rt_prev, noise_pred = self.predict(rt_prev, u0,
                                                  torch.full((batch_size,), t_val, device=device, dtype=torch.long))

        if noise_pred is None:
            _, rt_prev, noise_pred = self.predict(rt_prev, u0, torch.full((batch_size,), self.t_list[-1], device=device,
                                                                          dtype=torch.long))

        ht_prev, _ = self.q_diffuse(h0, torch.full((batch_size,), self.t_list[train_idx - 1], device=device,
                                                   dtype=torch.long))

        return self.loss(rt_prev, ht_prev, noise_pred, noise_gt)

    def forward_test(self, u0):
        device = u0.device
        batch_size = u0.shape[0]
        rs, _ = self.q_diffuse(u0, torch.full((batch_size,), self.t_end, device=device, dtype=torch.long))

        rt_prev = rs
        r0 = rt_prev  # 默认值
        for t_val in reversed(self.t_list[1:]):  # t_list[0] is 0
            r0, rt_prev, _ = self.predict(rt_prev, u0,
                                          torch.full((batch_size,), t_val, device=device, dtype=torch.long))
        return r0

    def forward(self, u0: torch.Tensor, h0: torch.Tensor = None, return_loss: bool = True):
        # 兼容 MMDetection 的调用方式
        if isinstance(u0, dict):
            # 如果输入是字典 (来自 data_loader), 解包
            data = u0
            u0 = data['img']
            h0 = data.get('hq_img')  # 使用 .get 防止测试时出错
            # img_metas = data['img_metas'] # 如果需要元信息

        if return_loss:
            assert h0 is not None, "High-quality image 'h0' or 'hq_img' must be provided for training."
            return self.forward_train(u0, h0)
        else:
            return self.forward_test(u0)