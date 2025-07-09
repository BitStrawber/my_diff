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

test = 0.1


# =============================================================================
# === NEW LOSS FUNCTION: Sliced-Wasserstein Distance (Computationally Feasible) ===
# =============================================================================
def sliced_wasserstein_distance(x, y, n_projections=64, p=2):
    """
    Computes the Sliced-Wasserstein Distance between two feature maps.
    Args:
        x, y: Input feature maps of shape (B, C, H, W).
        n_projections: The number of random projections to use.
        p: The norm to use for the 1D Wasserstein distance (1 or 2).
    """
    B, C, H, W = x.shape
    device = x.device

    # Flatten the spatial dimensions
    x_flat = x.view(B, C, -1)  # (B, C, N)
    y_flat = y.view(B, C, -1)  # (B, C, N)

    # Generate random projection vectors
    projections = torch.randn(C, n_projections, device=device).T  # (n_projections, C)
    projections = F.normalize(projections, p=2, dim=1)  # Normalize projections

    # Project the features
    # (B, C, N) -> (B, N, C) @ (C, n_projections) -> (B, N, n_projections)
    x_proj = x_flat.permute(0, 2, 1).matmul(projections.T)
    y_proj = y_flat.permute(0, 2, 1).matmul(projections.T)

    # Sort the projections to compute the 1D Wasserstein distance
    x_proj_sorted, _ = torch.sort(x_proj, dim=1)
    y_proj_sorted, _ = torch.sort(y_proj, dim=1)

    # Calculate the L_p distance between sorted projections
    # This is the 1D Wasserstein distance
    dist = torch.abs(x_proj_sorted - y_proj_sorted)
    if p == 1:
        w_dist = torch.mean(dist, dim=1)
    elif p == 2:
        w_dist = torch.sqrt(torch.mean(dist ** 2, dim=1))
    else:
        raise ValueError("p must be 1 or 2")

    # Average over all projections and the batch
    return w_dist.mean()


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

    def filter(self, image, template):
        device = image.device
        kernel = torch.FloatTensor(template).to(device)
        kernel = kernel.view(1, 1, *kernel.shape)

        num_channels = image.shape[1]
        kernel = kernel.expand(num_channels, 1, *kernel.shape[2:])
        weight = torch.nn.Parameter(data=kernel, requires_grad=False)
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
        step = max(1, self.t_end // self.sample_times)
        self.t_list = list(range(0, self.t_end, step)) + [self.t_end]
        self.L1 = nn.L1Loss()
        self.counter = 0

    def MutiScaleLuminanceEstimation(self, img):
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
        t_prev = (t - time_step).clamp_min(0)
        alpha_cumprod_prev = self.get_alpha_cumprod(t_prev).to(device)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod).clamp_min(1e-6)
        noise = (et - torch.sqrt(alpha_cumprod) * e0) / sqrt_one_minus_alpha_cumprod
        e_prev = torch.sqrt(alpha_cumprod_prev) * e0 + torch.sqrt(1.0 - alpha_cumprod_prev) * noise
        return e0, e_prev, noise

    # =============================================================================
    # === UPDATED LOSS: Calling the new Sliced-Wasserstein function             ===
    # =============================================================================
    def loss(self, r_prev: torch.Tensor, h_prev: torch.Tensor, noise_pred: torch.Tensor, noise_gt: torch.Tensor):
        r_prev_lf = self.MutiScaleLuminanceEstimation(r_prev)
        h_prev_lf = self.MutiScaleLuminanceEstimation(h_prev)

        # Calculate land_loss using the new, efficient Sliced-Wasserstein Distance
        land_loss = sliced_wasserstein_distance(r_prev_lf, h_prev_lf) * self.land_loss_weight

        uw_loss = F.mse_loss(noise_pred, noise_gt, reduction='mean') * self.uw_loss_weight

        return dict(land_loss=land_loss, uw_loss=uw_loss)

    def forward_train(self, u0: torch.Tensor, h0: torch.Tensor):
        # This function should be fine as it is.
        device = u0.device
        batch_size = u0.shape[0]

        train_idx = random.randint(1, len(self.t_list) - 1)
        rs, noise_gt = self.q_diffuse(u0, torch.full((batch_size,), self.t_end, device=device, dtype=torch.long))

        rt_prev = rs
        noise_pred = None

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
        r0 = rt_prev
        for t_val in reversed(self.t_list[1:]):
            r0, rt_prev, _ = self.predict(rt_prev, u0,
                                          torch.full((batch_size,), t_val, device=device, dtype=torch.long))
        return r0

    def forward(self, u0: torch.Tensor, h0: torch.Tensor = None, return_loss: bool = True):
        # Your working forward logic
        if return_loss:
            assert h0 is not None
            return self.forward_train(u0, h0)
        else:
            return self.forward_test(u0)