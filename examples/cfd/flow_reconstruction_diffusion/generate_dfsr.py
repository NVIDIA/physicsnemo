# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle  # TODO remove

import hydra
import numpy as np
import PIL.Image
import torch
import torchvision.utils as tvu
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
from omegaconf import DictConfig
from utils import StackedRandomGenerator, open_url

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from utils import EasyDict, construct_class_by_name
import copy
import logging
import matplotlib.pyplot as plt
from einops import rearrange
from mpl_toolkits.axes_grid1 import ImageGrid
import math
import pickle
import yaml

from utils import (
    InfiniteSampler,
    check_ddp_consistency,
    construct_class_by_name,
    copy_params_and_buffers,
    ddp_sync,
    format_time,
    open_url,
    print_module_summary,
)


def sampler(
    net,
    latents,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=None,
    sigma_max=None,
    rho=7,
    solver="heun",
    discretization="edm",
    schedule="linear",
    scaling="none",
    epsilon_s=1e-3,
    C_1=0.001,
    C_2=0.008,
    M=1000,
    alpha=1,
    s_churn=0,
    s_min=0,
    s_max=float("inf"),
    s_noise=1,
):
    """
    Generalized sampler, representing the superset of all sampling methods discussed
    in the paper "Elucidating the Design Space of Diffusion-Based Generative Models"
    """
    if solver not in ["euler", "heun"]:
        raise ValueError(f'Invalid solver "{solver}"')
    if discretization not in ["vp", "ve", "iddpm", "edm"]:
        raise ValueError(f'Invalid discretization "{discretization}"')
    if schedule not in ["vp", "ve", "linear"]:
        raise ValueError(f'Invalid schedule "{schedule}"')
    if scaling is not None and scaling not in ["vp"]:
        raise ValueError(f'Invalid scaling "{scaling}"')

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = (
        lambda beta_d, beta_min: lambda t: (
            np.e ** (0.5 * beta_d * (t**2) + beta_min * t) - 1
        )
        ** 0.5
    )
    vp_sigma_deriv = (
        lambda beta_d, beta_min: lambda t: 0.5
        * (beta_min + beta_d * t)
        * (sigma(t) + 1 / sigma(t))
    )
    vp_sigma_inv = (
        lambda beta_d, beta_min: lambda sigma: (
            (beta_min**2 + 2 * beta_d * (sigma**2 + 1).log()).sqrt() - beta_min
        )
        / beta_d
    )
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma**2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {"vp": vp_def, "ve": 0.02, "iddpm": 0.002, "edm": 0.002}[
            discretization
        ]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {"vp": vp_def, "ve": 100, "iddpm": 81, "edm": 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = (
        2
        * (np.log(sigma_min**2 + 1) / epsilon_s - np.log(sigma_max**2 + 1))
        / (epsilon_s - 1)
    )
    vp_beta_min = np.log(sigma_max**2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == "vp":
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == "ve":
        orig_t_steps = (sigma_max**2) * (
            (sigma_min**2 / sigma_max**2) ** (step_indices / (num_steps - 1))
        )
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == "iddpm":
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device):  # M, ..., 1
            u[j - 1] = (
                (u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1
            ).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[
            ((len(u_filtered) - 1) / (num_steps - 1) * step_indices)
            .round()
            .to(torch.int64)
        ]
    else:  # edm sigma steps
        sigma_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho

    # Define noise level schedule.
    if schedule == "vp":
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == "ve":
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == "vp":
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = (
            min(s_churn / num_steps, np.sqrt(2) - 1)
            if s_min <= sigma(t_cur) <= s_max
            else 0
        )
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (
            sigma(t_hat) ** 2 - sigma(t_cur) ** 2
        ).clip(min=0).sqrt() * s(t_hat) * s_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (
            sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)
        ) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == "euler" or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == "heun"
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(
                torch.float64
            )
            d_prime = (
                sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)
            ) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * (
                (1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime
            )

    return x_next


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def ddim_steps(x, seq, model, b, **kwargs):
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    dx_func = kwargs.get("dx_func", None)
    clamp_func = kwargs.get("clamp_func", None)
    cache = kwargs.get("cache", False)

    logger = kwargs.get("logger", None)
    if logger is not None:
        logger.update(x=xs[-1])

    for i, j in zip(reversed(seq), reversed(seq_next)):
        with torch.no_grad():
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to("cuda")

            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to("cpu"))

            c2 = (1 - at_next).sqrt()
        if dx_func is not None:
            dx = dx_func(xt)
        else:
            dx = 0
        with torch.no_grad():
            xt_next = at_next.sqrt() * x0_t + c2 * et - dx
            if clamp_func is not None:
                xt_next = clamp_func(xt_next)
            xs.append(xt_next.to("cpu"))

        if logger is not None:
            logger.update(x=xs[-1])

        if not cache:
            xs = xs[-1:]
            x0_preds = x0_preds[-1:]

    return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    xs = [x]
    x0_preds = []
    betas = b
    dx_func = kwargs.get("dx_func", None)
    cache = kwargs.get("cache", False)
    clamp_func = kwargs.get("clamp_func", None)

    for i, j in zip(reversed(seq), reversed(seq_next)):
        with torch.no_grad():

            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to("cuda")

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to("cpu"))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e
                + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()

        if dx_func is not None:
            dx = dx_func(x)
        else:
            dx = 0
        with torch.no_grad():
            sample = mean + mask * torch.exp(0.5 * logvar) * noise - dx
            if clamp_func is not None:
                sample = clamp_func(sample)
            xs.append(sample.to("cpu"))
        if not cache:
            xs = xs[-1:]
            x0_preds = x0_preds[-1:]

    return xs, x0_preds


def guided_ddpm_steps(x, seq, model, b, **kwargs):
    """Guided DDPM steps"""
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    xs = [x]
    x0_preds = []
    betas = b
    dx_func = kwargs.get("dx_func", None)
    if dx_func is None:
        raise ValueError("dx_func is required for guided denoising")
    clamp_func = kwargs.get("clamp_func", None)
    cache = kwargs.get("cache", False)
    w = kwargs.get("w", 3.0)

    for i, j in zip(reversed(seq), reversed(seq_next)):
        with torch.no_grad():

            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to("cuda")

        dx = dx_func(x)
        with torch.no_grad():

            output = (w + 1) * model(x, t.float(), dx) - w * model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to("cpu"))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e
                + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()

        with torch.no_grad():
            sample = mean + mask * torch.exp(0.5 * logvar) * noise - dx
            if clamp_func is not None:
                sample = clamp_func(sample)
            xs.append(sample.to("cpu"))
        if not cache:
            xs = xs[-1:]
            x0_preds = x0_preds[-1:]

    return xs, x0_preds


def guided_ddim_steps(x, seq, model, b, **kwargs):
    """Guided DDiM steps"""
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    dx_func = kwargs.get("dx_func", None)
    if dx_func is None:
        raise ValueError("dx_func is required for guided denoising")
    clamp_func = kwargs.get("clamp_func", None)
    cache = kwargs.get("cache", False)
    w = kwargs.get("w", 3.0)
    logger = kwargs.get("logger", None)
    if logger is not None:
        logger.update(x=xs[-1])

    for i, j in zip(reversed(seq), reversed(seq_next)):
        with torch.no_grad():
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to("cuda")

        dx = dx_func(xt)

        et = (w + 1) * model(xt, t, dx) - w * model(xt, t)
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        x0_preds.append(x0_t.to("cpu"))
        c2 = (1 - at_next).sqrt()

        with torch.no_grad():
            xt_next = at_next.sqrt() * x0_t + c2 * et - dx
            if clamp_func is not None:
                xt_next = clamp_func(xt_next)
            xs.append(xt_next.to("cpu"))

        if logger is not None:
            logger.update(x=xs[-1])

        if not cache:
            xs = xs[-1:]
            x0_preds = x0_preds[-1:]

    return xs, x0_preds


class MetricLogger(object):
    """metric logger"""

    def __init__(self, metric_fn_dict):
        self.metric_fn_dict = metric_fn_dict
        self.metric_dict = {}
        self.reset()

    def reset(self):
        for key in self.metric_fn_dict.keys():
            self.metric_dict[key] = []

    @torch.no_grad()
    def update(self, **kwargs):
        for key in self.metric_fn_dict.keys():
            self.metric_dict[key].append(self.metric_fn_dict[key](**kwargs))

    def get(self):
        return self.metric_dict.copy()

    def log(self, outdir, postfix=""):
        with open(os.path.join(outdir, f"metric_log_{postfix}.pkl"), "wb") as f:
            pickle.dump(self.metric_dict, f)


def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    """Get beta schedule"""
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def load_flow_data(path, stat_path=None):
    """Loads the flow data"""
    # load flow data from path
    data = np.load(path)  # [N, T, h, w]

    # print('Original data shape:', data.shape)
    data_mean, data_scale = np.mean(data[:-4]), np.std(data[:-4])
    print(f"Data range: mean: {data_mean} scale: {data_scale}")
    data = data[-4:, ...].copy().astype(np.float32)  # only take the test set
    data = torch.as_tensor(data, dtype=torch.float32)
    flattened_data = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1] - 2):
            flattened_data.append(data[i, j : j + 3, ...])
    flattened_data = torch.stack(flattened_data, dim=0)
    print(f"data shape: {flattened_data.shape}")
    return flattened_data, data_mean.item(), data_scale.item()


def load_recons_data(ref_path, sample_path, data_kw, smoothing, smoothing_scale):
    """Loads recons data"""
    print("Loading low-res input data from: ", sample_path)
    with np.load(sample_path, allow_pickle=True) as f:
        sampled_data = f[data_kw][-4:, ...].copy().astype(np.float32)
    sampled_data = torch.as_tensor(sampled_data, dtype=torch.float32)
    print("Loading high-res reference data from: ", ref_path)
    ref_data = np.load(ref_path).astype(np.float32)
    # ref_data = np.load(ref_path).astype(np.float32)[:,:,::4,::4]
    data_mean, data_scale = np.mean(ref_data[:-4]), np.std(ref_data[:-4])

    ref_data = ref_data[-4:, ...].copy().astype(np.float32)  # only take the test set
    ref_data = torch.as_tensor(ref_data, dtype=torch.float32)

    flattened_sampled_data = []
    flattened_ref_data = []

    for i in range(ref_data.shape[0]):
        for j in range(ref_data.shape[1] - 2):
            flattened_ref_data.append(ref_data[i, j : j + 3, ...])
            flattened_sampled_data.append(sampled_data[i, j : j + 3, ...])

    flattened_ref_data = torch.stack(flattened_ref_data, dim=0)
    flattened_sampled_data = torch.stack(flattened_sampled_data, dim=0)
    if smoothing:
        arr = flattened_sampled_data
        ker_size = smoothing_scale
        # peridoic padding
        arr = F.pad(
            arr,
            pad=(
                (ker_size - 1) // 2,
                (ker_size - 1) // 2,
                (ker_size - 1) // 2,
                (ker_size - 1) // 2,
            ),
            mode="circular",
        )
        arr = transforms.GaussianBlur(kernel_size=ker_size, sigma=ker_size)(
            arr
        )  # F.avg_pool2d(arr, (ker_size, ker_size), stride=1, count_include_pad=False)
        flattened_sampled_data = arr[
            ...,
            (ker_size - 1) // 2 : -(ker_size - 1) // 2,
            (ker_size - 1) // 2 : -(ker_size - 1) // 2,
        ]

    # print(f'data shape: {flattened_ref_data.shape}')

    return (
        flattened_ref_data,
        flattened_sampled_data,
        data_mean.item(),
        data_scale.item(),
    )


class MinMaxScaler(object):
    """minmax scaler"""

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, x):
        return x - self.min  # / (self.max - self.min)

    def inverse(self, x):
        return x * (self.max - self.min) + self.min

    def scale(self):
        return self.max - self.min


class StdScaler(object):
    """Std scaler"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    def scale(self):
        return self.std


def make_image_grid(images, out_path, ncols=8):
    """Make image grid"""
    # assume images in the shape of (N, T, H, W)
    t, h, w = images.shape
    images = images.detach().cpu().numpy()
    b = t // ncols
    fig = plt.figure(figsize=(8.0, 8.0))
    # print("t, h, w, ncols: ", t, h, w, ncols)
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(b, ncols),  # creates 2x2 grid of axes
    )

    for ax, im_no in zip(grid, np.arange(b * ncols)):
        # Iterating over the grid returns the Axes.
        ax.imshow(images[im_no, :, :], cmap="twilight", vmin=-23, vmax=23)
        ax.axis("off")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def ensure_dir(path):
    """ensure dir"""
    if not os.path.exists(path):
        os.makedirs(path)


def slice2sequence(data):
    """slice to sequence"""
    data = rearrange(data[:, 1:2], "t f h w -> (t f) h w")
    return data


def l1_loss(x, y):
    """l1 loss"""
    return torch.mean(torch.abs(x - y))


def l2_loss(x, y):
    """l2 loss"""
    return ((x - y) ** 2).mean((-1, -2)).sqrt().mean()


def vorticity_residual(w, re=1000.0, dt=1 / 32, calc_grad=True):
    """Velocity residual"""
    # w [b t h w]
    # print("#### in def vorticity_residual() ####")
    batchsize = w.size(0)
    w = w.clone()
    w.requires_grad_(True)
    nx = w.size(2)
    ny = w.size(3)
    device = w.device

    w_h = torch.fft.fft2(w[:, 1:-1], dim=[2, 3])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = (
        torch.cat(
            (
                torch.arange(start=0, end=k_max, step=1, device=device),
                torch.arange(start=-k_max, end=0, step=1, device=device),
            ),
            0,
        )
        .reshape(N, 1)
        .repeat(1, N)
        .reshape(1, 1, N, N)
    )
    k_y = (
        torch.cat(
            (
                torch.arange(start=0, end=k_max, step=1, device=device),
                torch.arange(start=-k_max, end=0, step=1, device=device),
            ),
            0,
        )
        .reshape(1, N)
        .repeat(N, 1)
        .reshape(1, 1, N, N)
    )
    # Negative Laplacian in Fourier space
    lap = k_x**2 + k_y**2
    lap[..., 0, 0] = 1.0
    psi_h = w_h / lap

    u_h = 1j * k_y * psi_h
    v_h = -1j * k_x * psi_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    u = torch.fft.irfft2(u_h[..., :, : k_max + 1], dim=[2, 3])
    v = torch.fft.irfft2(v_h[..., :, : k_max + 1], dim=[2, 3])
    wx = torch.fft.irfft2(wx_h[..., :, : k_max + 1], dim=[2, 3])
    wy = torch.fft.irfft2(wy_h[..., :, : k_max + 1], dim=[2, 3])
    wlap = torch.fft.irfft2(wlap_h[..., :, : k_max + 1], dim=[2, 3])
    advection = u * wx + v * wy

    wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)

    # establish forcing term
    x = torch.linspace(0, 2 * np.pi, nx + 1, device=device)
    x = x[0:-1]
    X, Y = torch.meshgrid(x, x)
    f = -4 * torch.cos(4 * Y)

    residual = wt + (advection - (1.0 / re) * wlap + 0.1 * w[:, 1:-1]) - f

    # Add scaling factor
    eps = 1e-6
    w_norm = torch.norm(w[:, 1, :, :], dim=(-1, -2))
    residual_loss = (residual**2).sum(dim=(-1, -2, -3)) / ((w_norm**2) + eps)
    residual_loss = residual_loss.mean()
    if calc_grad:
        dw = torch.autograd.grad(residual_loss, w)[0]
        return dw, residual_loss
    else:
        return residual_loss


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """

    # Initialize distributed manager.
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize logger.
    logger = logging.getLogger("LOG")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(
        "%s/%s.txt"
        % (
            cfg.outdir,
            "logging_info_{}_s{}_t{}_r{}".format(
                cfg.data_kw, cfg.smoothing_scale, cfg.t, cfg.r
            ),
        )
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    device = dist.device
    logger.info("Using device: {}".format(device))

    print(">" * 80)
    logger.info("Exp instance id = {}".format(os.getpid()))
    # logger.info("Exp comment = {}".format(args.comment))
    logger.info("Config =")
    print("<" * 80)

    logger.info("Doing sparse reconstruction task")
    logger.info("Loading model")

    # Load test data.

    # Define and load model.
    network_kwargs = EasyDict()
    if cfg.arch == "dfsr":
        network_kwargs.update(
            model_type="SongUNet",
            embedding_type="positional",
            encoder_type="standard",
            decoder_type="standard",
        )
        network_kwargs.update(
            channel_mult_noise=1,
            resample_filter=[1, 1],
            model_channels=64,
            channel_mult=[1, 1, 1, 2],
        )
    else:
        assert cfg.arch == "adm"
        network_kwargs.update(
            model_type="DhariwalUNet", model_channels=192, channel_mult=[1, 2, 3, 4]
        )

    # Preconditioning & loss function.
    loss_kwargs = EasyDict()
    if cfg.precond == "dfsr":
        network_kwargs.class_name = "physicsnemo.models.diffusion.VEPrecond_dfsr"
        loss_kwargs.class_name = "physicsnemo.metrics.diffusion.VELoss_dfsr"
    elif cfg.precond == "dfsr_cond":
        network_kwargs.class_name = "physicsnemo.models.diffusion.VEPrecond_dfsr_cond"
        loss_kwargs.class_name = "physicsnemo.metrics.diffusion.VELoss_dfsr_cond"
    loss_fn = construct_class_by_name(**loss_kwargs)  # training.loss.(VP|VE|EDM)Loss

    # Network options.
    if cfg.cbase is not None:
        network_kwargs.model_channels = cfg.cbase
    if cfg.cres is not None:
        network_kwargs.channel_mult = cfg.cres
    if cfg.augment:
        raise NotImplementedError("Augmentation is not implemented")
    network_kwargs.update(dropout=cfg.dropout, use_fp16=cfg.fp16)

    interface_kwargs = dict(
        img_resolution=cfg.img_resolution,
        img_channels=cfg.img_channels,
        label_dim=cfg.label_dim,
    )

    net = construct_class_by_name(
        **network_kwargs, **interface_kwargs
    )  # subclass of torch.nn.Module

    # Load non-EMA weights
    ckpt_filename = "training-state-003435.pt"
    ckpt_data = torch.load(
        os.path.join(cfg.outdir, ckpt_filename), map_location=torch.device("cpu")
    )
    # print("list(ckpt_data.keys()):\n", list(ckpt_data.keys()))
    copy_params_and_buffers(
        src_module=ckpt_data["net"], dst_module=net, require_all=True
    )
    # optimizer.load_state_dict(ckpt_data["optimizer_state"])
    del ckpt_data  # conserve memory

    # Load EMA weights
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    ckpt_filename = "network-snapshot-003435.pkl"
    with open_url(
        os.path.join(cfg.outdir, ckpt_filename), verbose=(dist.rank == 0)
    ) as f:
        # net = pickle.load(f)["ema"].to(device)
        net = pickle.load(f)["ema"]
        net.train().requires_grad_(True).to(device)

    logger.info("Preparing data")
    ref_data, blur_data, data_mean, data_std = load_recons_data(
        cfg.data,
        cfg.sample_data,
        cfg.data_kw,
        smoothing=cfg.smoothing,
        smoothing_scale=cfg.smoothing_scale,
    )

    scaler = StdScaler(data_mean, data_std)

    # pack data loader
    testset = torch.utils.data.TensorDataset(blur_data, ref_data)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.batch, shuffle=False, num_workers=cfg.workers
    )

    l2_loss_all = np.zeros((ref_data.shape[0], cfg.repeat_run, cfg.sample_step))
    residual_loss_all = np.zeros((ref_data.shape[0], cfg.repeat_run, cfg.sample_step))

    betas = get_beta_schedule(
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        num_diffusion_timesteps=cfg.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device)

    print("cfg.precond: ", cfg.precond)
    print("cfg.smoothing: ", cfg.smoothing)
    print("cfg.smoothing_scale: ", cfg.smoothing_scale)

    smoothing_kernel_size = cfg.smoothing_scale if cfg.smoothing else 0
    # Create image sample directory
    if cfg.precond == "dfsr_cond":
        dir_name = "recons_{}_{}_t{}_r{}_w{}_s{}".format(
            cfg.data_kw,
            cfg.data_kw,
            cfg.t,
            cfg.r,
            cfg.guidance_weight,
            smoothing_kernel_size,
        )
    else:
        dir_name = "recons_{}_{}_t{}_r{}_lam{}_s{}".format(
            cfg.data_kw, cfg.data_kw, cfg.t, cfg.r, cfg.lambda_, smoothing_kernel_size
        )
    if cfg.precond == "dfsr_cond":
        print("Use residual gradient guidance during sampling")
        dir_name = "guided_" + dir_name
    else:
        print("Not use physical gradient during sampling")
    image_sample_dir = os.path.join(cfg.outdir, dir_name)
    if not os.path.exists(image_sample_dir):
        os.makedirs(image_sample_dir)
    with open(os.path.join(image_sample_dir, "config.yml"), "w") as outfile:
        # yaml.dump(cfg, outfile)
        OmegaConf.save(config=cfg, f=outfile)

    for batch_index, (blur_data, data) in enumerate(test_loader):
        print("Sampling Batch: ", batch_index)
        logger.info("Batch: {} / Total batch {}".format(batch_index, len(test_loader)))
        x0 = blur_data.to(device)

        gt = data.to(device)

        logger.info("Preparing reference image")
        logger.info("Dumping visualization...")

        sample_folder = "sample_batch{}".format(batch_index)
        ensure_dir(os.path.join(image_sample_dir, sample_folder))

        sample_img_filename = "input_image.png"
        path_to_dump = os.path.join(
            image_sample_dir, sample_folder, sample_img_filename
        )
        x0_masked = x0.clone()

        make_image_grid(slice2sequence(x0_masked), path_to_dump, cfg.batch)
        sample_img_filename = "reference_image.png"
        path_to_dump = os.path.join(
            image_sample_dir, sample_folder, sample_img_filename
        )
        make_image_grid(slice2sequence(gt), path_to_dump, cfg.batch)

        # save as array
        if cfg.dump_arr:
            np.save(
                os.path.join(image_sample_dir, sample_folder, "input_arr.npy"),
                slice2sequence(x0).cpu().numpy(),
            )
            np.save(
                os.path.join(image_sample_dir, sample_folder, "reference_arr.npy"),
                slice2sequence(data).cpu().numpy(),
            )

        # calculate initial loss
        # l1_loss_init = l1_loss(x0, gt)
        l2_loss_init = l2_loss(x0, gt)
        # print("l2_loss_init :", l2_loss_init)

        logger.info("L2 loss init: {}".format(l2_loss_init))
        gt_residual = vorticity_residual(gt)[1].detach()
        init_residual = vorticity_residual(x0)[1].detach()
        logger.info("Residual init: {}".format(init_residual))
        logger.info("Residual reference: {}".format(gt_residual))

        x0 = scaler(x0)
        xinit = x0.clone()

        # prepare loss function
        if cfg.log_loss:
            l2_loss_fn = lambda x: l2_loss(scaler.inverse(x).to(gt.device), gt)
            equation_loss_fn = lambda x: vorticity_residual(
                scaler.inverse(x), calc_grad=False
            )

            metric_logger = MetricLogger(
                {"l2 loss": l2_loss_fn, "residual loss": equation_loss_fn}
            )

        # we repeat the sampling for multiple times
        for repeat in range(cfg.repeat_run):
            logger.info(f"Run No.{repeat}:")
            x0 = xinit.clone()
            for it in range(cfg.sample_step):  # we run the sampling for three times
                e = torch.randn_like(x0)
                total_noise_levels = int(cfg.t * (0.7**it))

                a = (1 - betas).cumprod(dim=0)
                x = (
                    x0 * a[total_noise_levels - 1].sqrt()
                    + e * (1.0 - a[total_noise_levels - 1]).sqrt()
                )

                if cfg.precond == "dfsr_cond":
                    physical_gradient_func = (
                        lambda x: vorticity_residual(scaler.inverse(x))[0]
                        / scaler.scale()
                    )

                num_of_reverse_steps = int(cfg.r * (0.7**it))
                betas = betas.to(device)
                skip = total_noise_levels // num_of_reverse_steps
                seq = range(0, total_noise_levels, skip)

                if cfg.precond == "dfsr_cond":
                    xs, _ = guided_ddim_steps(
                        x,
                        seq,
                        net,
                        betas,
                        w=cfg.guidance_weight,
                        dx_func=physical_gradient_func,
                        cache=False,
                        logger=metric_logger,
                    )
                else:
                    xs, _ = ddim_steps(
                        x, seq, net, betas, cache=False, logger=metric_logger
                    )

                x = xs[-1]
                x0 = xs[-1].cuda()

                l2_loss_f = l2_loss(scaler.inverse(x.clone()).to(gt.device), gt)
                logger.info("L2 loss it{}: {}".format(it, l2_loss_f))
                residual_loss_f = vorticity_residual(
                    scaler.inverse(x.clone()), calc_grad=False
                ).detach()
                logger.info("Residual it{}: {}".format(it, residual_loss_f))

                l2_loss_all[
                    batch_index * x.shape[0] : (batch_index + 1) * x.shape[0],
                    repeat,
                    it,
                ] = l2_loss_f.item()
                residual_loss_all[
                    batch_index * x.shape[0] : (batch_index + 1) * x.shape[0],
                    repeat,
                    it,
                ] = residual_loss_f.item()
                # print("l2_loss_all.shape: ", l2_loss_all.shape) # default shape: (1272, 1, 1)

                if cfg.dump_arr:
                    np.save(
                        os.path.join(
                            image_sample_dir,
                            sample_folder,
                            f"sample_arr_run_{repeat}_it{it}.npy",
                        ),
                        slice2sequence(scaler.inverse(x)).cpu().numpy(),
                    )

                if cfg.log_loss:
                    test1 = f"run_{repeat}_it{it}"
                    metric_logger.log(
                        os.path.join(image_sample_dir, sample_folder),
                        f"run_{repeat}_it{it}",
                    )
                    metric_logger.reset()
                print("gt_residual: ", gt_residual)
                print("init_residual: ", init_residual)
                print("residual_loss_f.item(): ", residual_loss_f.item())
                # torch.cuda.empty_cache()
                # exit("####")

        logger.info("Finished batch {}".format(batch_index))
        logger.info("========================================================")

    logger.info("Finished sampling")
    logger.info(f"mean l2 loss: {l2_loss_all[..., -1].mean()}")
    logger.info(f"std l2 loss: {l2_loss_all[..., -1].std(axis=1).mean()}")
    logger.info(f"mean residual loss: {residual_loss_all[..., -1].mean()}")
    logger.info(f"std residual loss: {residual_loss_all[..., -1].std(axis=1).mean()}")
    logger.info("cfg.precond: {}".format(cfg.precond))
    logger.info("cfg.smoothing: {}".format(cfg.smoothing))
    logger.info("cfg.smoothing_scale: {}".format(cfg.smoothing_scale))


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        help="directory to load configuration file",
    )
    args = parser.parse_args()
    loaded_config = OmegaConf.load(args.config)
    main(loaded_config)

# ----------------------------------------------------------------------------
