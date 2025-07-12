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
import tqdm
from omegaconf import DictConfig
from physicsnemo.utils.diffusion.utils import StackedRandomGenerator

from misc import open_url

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper


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


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """
    ckpt_filename = cfg.ckpt_filename
    img_outdir = cfg.img_outdir
    subdirs = cfg.subdirs
    gen_seeds = cfg.gen_seeds
    class_idx = cfg.class_idx
    max_batch_size = cfg.max_batch_size

    # Initialize distributed manager.
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # Initialize logger.
    logger = PythonLogger("main")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging()

    num_batches = (
        (len(gen_seeds) - 1) // (max_batch_size * dist.world_size) + 1
    ) * dist.world_size
    all_batches = torch.as_tensor(gen_seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.rank :: dist.world_size]

    # Rank 0 goes first.
    if dist.world_size > 1 and dist.rank != 0:
        torch.distributed.barrier()

    # Load network.
    logger0.info(f'Loading network from "{ckpt_filename}"...')
    with open_url(ckpt_filename, verbose=(dist.rank == 0)) as f:
        net = pickle.load(f)["ema"].to(device)

    # Other ranks follow.
    if dist.world_size > 1 and dist.rank == 0:
        torch.distributed.barrier()

    # Loop over batches.
    logger0.info(f'Generating {len(gen_seeds)} images to "{img_outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=(dist.rank != 0)):
        if dist.world_size > 1:
            torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn(
            [batch_size, net.img_channels, net.img_resolution, net.img_resolution],
            device=device,
        )
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[
                rnd.randint(net.label_dim, size=[batch_size], device=device)
            ]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        images = sampler(
            net,
            latents,
            class_labels,
            randn_like=rnd.randn_like,
            num_steps=cfg.num_steps,
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
            rho=cfg.rho,
            solver=cfg.solver,
            discretization=cfg.discretization,
            schedule=cfg.schedule,
            scaling=cfg.scaling,
            epsilon_s=1e-3,
            C_1=0.001,
            C_2=0.008,
            M=1000,
            alpha=1,
            s_churn=cfg.s_churn,
            s_min=cfg.s_min,
            s_max=cfg.s_max,
            s_noise=cfg.s_noise,
        )

        # Save images.
        images_np = (
            (images * 127.5 + 128)
            .clip(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = (
                os.path.join(img_outdir, f"{seed-seed%1000:06d}")
                if subdirs
                else img_outdir
            )
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f"{seed:06d}.png")
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], "L").save(image_path)
            else:
                PIL.Image.fromarray(image_np, "RGB").save(image_path)

    # Done.
    if dist.world_size > 1:
        torch.distributed.barrier()
    logger0.info("Done.")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
