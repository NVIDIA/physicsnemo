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

from typing import Callable, Literal, Optional

import numpy as np
import nvtx
import torch

from physicsnemo.models.diffusion import EDMPrecond

# ruff: noqa: E731


@nvtx.annotate(message="deterministic_sampler", color="red")
def deterministic_sampler(
    net: torch.nn.Module,
    latents: torch.Tensor,
    img_lr: torch.Tensor,
    class_labels: Optional[torch.Tensor] = None,
    randn_like: Callable = torch.randn_like,
    num_steps: int = 18,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    rho: float = 7.0,
    solver: Literal["heun", "euler"] = "heun",
    discretization: Literal["vp", "ve", "iddpm", "edm"] = "edm",
    schedule: Literal["vp", "ve", "linear"] = "linear",
    scaling: Literal["vp", "none"] = "none",
    epsilon_s: float = 1e-3,
    C_1: float = 0.001,
    C_2: float = 0.008,
    M: int = 1000,
    alpha: float = 1.0,
    S_churn: int = 0,
    S_min: float = 0.0,
    S_max: float = float("inf"),
    S_noise: float = 1.0,
) -> torch.Tensor:
    r"""
    Generalized sampler, representing the superset of all sampling methods
    discussed in the paper `Elucidating the Design Space of Diffusion-Based
    Generative Models (EDM) <https://arxiv.org/abs/2206.00364>`_.

    This function integrates an ODE (probability flow) or SDE over multiple
    time-steps to generate samples from the diffusion model provided by the
    argument 'net'. It can be used to combine multiple choices to
    design a custom sampler, including multiple integration solver,
    discretization method, noise schedule, and so on.

    Parameters
    ----------
    net : torch.nn.Module
        The diffusion model to use in the sampling process.
    latents : torch.Tensor
        The latent random noise used as the initial condition for the
        stochastic ODE.
    img_lr : torch.Tensor
        Low-resolution input image for conditioning the diffusion process.
        Passed as a keywork argument to the model ``net``.
    class_labels : Optional[torch.Tensor]
        Labels of the classes used as input to a class-conditionned
        diffusion model. Passed as a keyword argument to the model ``net``.
        If provided, it must be a tensor containing  integer values.
        Defaults to ``None``, in which case it is ignored.
    randn_like: Callable
        Random Number Generator to generate random noise that is added
        during the stochastic sampling. Must have the same signature as
        torch.randn_like and return torch.Tensor. Defaults to
        torch.randn_like.
    num_steps : Optional[int]
        Number of time-steps for the stochastic ODE integration. Defaults
        to 18.
    sigma_min : Optional[float]
        Minimum noise level for the diffusion process. ``sigma_min``,
        ``sigma_max``, and ``rho`` are used to compute the time-step
        discretization, based on the choice of discretization. For the
        default choice (``discretization='heun'``), the noise level schedule
        is computed as:
        :math:`\sigma_i = (\sigma_{max}^{1/\rho} + i / (\text{num_steps} - 1) * (\sigma_{min}^{1/\rho} - \sigma_{max}^{1/\rho}))^{\rho}`.
        For other choices of ``discretization``, see details in the EDM
        paper. Defaults to ``None``, in which case defaults values depending
        of the specified discretization are used.
    sigma_max : Optional[float]
        Maximum noise level for the diffusion process. See ``sigma_min`` for
        details. Defaults to ``None``, in which case defaults values depending
        of the specified discretization are used.
    rho : float, optional
        Exponent used in the noise schedule. See ``sigma_min`` for details.
        Only used when ``discretization="heun"``. Values in the range
        [5, 10] produce better images. Lower values lead to truncation errors
        equalized over all time steps. Defaults to 7.
    solver : Literal["heun", "euler"]
        The numerical method used to integrate the stochastic ODE. ``"euler"``
        is 1st order solver, which is faster but produces lower-quality
        images. ``"heun"`` is 2nd order, more expensive, but produces
        higher-quality images. Defaults to ``"heun"``.
    discretization : Literal["vp", "ve", "iddpm", "edm"]
        The method to discretize time-steps :math:`t_i` in the
        diffusion process. See the EDM paper for details. Defaults to
        ``"edm"``.
    schedule : Literal["vp", "ve", "linear"]
        The type of noise level schedule.  Defaults to ``"linear"``. If
        ``schedule="ve"``, then :math:`\sigma(t) = \sqrt{t}`. If
        ``schedule="linear"``, then :math:`\sigma(t) = t`. If ``schedule="vp"``,
        see EDM paper for details. Defaults to ``"linear"``.
    scaling : Literal["vp", "none"]
        The type of time-dependent signal scaling :math:`s(t)`, such that
        :math:`x = s(t) \hat{x}`. See EDM paper for details on the ``"vp"``
        scaling. Defaults to ``"none"``, in which case :math:`s(t)=1`.
    epsilon_s : float, optional
        Parameter to compute both the noise level schedule and the
        time-step discetization. Only used when ``discretization="vp"`` or
        ``schedule="vp"``. Ignored in other cases. Defaults to 1e-3.
    C_1 : float, optional
        Parameters to compute the time-step discetization. Only used when
        ``discretization="iddpm"``. Defaults to 0.001.
    C_2 : float, optional
        Same as for C_1. Only used when ``discretization="iddpm"``. Defaults to
        0.008.
    M : int, optional
        Same as for C_1 and C_2. Only used when ``discretization="iddpm"``.
        Defaults to 1000.
    alpha : float, optional
        Controls (i.e. multiplies) the step size :math:`t_{i+1} -
        \hat{t}_i` in the stochastic sampler, where :math:`\hat{t}_i` is
        the temporarily increased noise level. Defaults to 1.0, which is
        the recommended value.
    S_churn : int, optional
        Controls the amount of stochasticty injected in the SDE in the
        stochatsic sampler. Larger values of ``S_churn`` lead to larger values
        of :math:`\hat{t}_i`, which in turn lead to injecting more
        stochasticity in the SDE by  Defaults to 0, which means no
        stochasticity is injected.
    S_min : float, optional
        ``S_min`` and ``S_max`` control the time-step range over which
        stochasticty is injected in the SDE. Stochasticity is injected
        through :math:`\hat{t}_i` for time-steps :math:`t_i` such that
        :math:`S_{min} \leq t_i \leq S_{max}`. Defaults to 0.0.
    S_max : float, optional
        See ``S_min``. Defaults to ``float("inf")``.
    S_noise : float, optional
        Controls the amount of stochasticty injected in the SDE in the
        stochatsic sampler. Added signal noise is proportinal to
        :math:`\epsilon_i` where :math:`\epsilon_i \sim \mathcal{N}(0, S_{noise}^2)`. Defaults
        to 1.0.

    Returns
    -------
        torch.Tensor:
            Generated batch of samples. Same shape as the input ``latents``.
    """

    # conditioning
    x_lr = img_lr

    if solver not in ["euler", "heun"]:
        raise ValueError(f"Unknown solver {solver}")
    if discretization not in ["vp", "ve", "iddpm", "edm"]:
        raise ValueError(f"Unknown discretization {discretization}")
    if schedule not in ["vp", "ve", "linear"]:
        raise ValueError(f"Unknown schedule {schedule}")
    if scaling not in ["vp", "none"]:
        raise ValueError(f"Unknown scaling {scaling}")

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

    # Select default noise level range based on the specified
    # time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=epsilon_s)
        sigma_min = {"vp": vp_def, "ve": 0.02, "iddpm": 0.002, "edm": 0.002}[
            discretization
        ]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=1)
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
    else:
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
            min(S_churn / num_steps, np.sqrt(2) - 1)
            if S_min <= sigma(t_cur) <= S_max
            else 0
        )
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (
            sigma(t_hat) ** 2 - sigma(t_cur) ** 2
        ).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        if isinstance(net, EDMPrecond):
            # Conditioning info is passed as keyword arg
            denoised = net(
                x_hat / s(t_hat),
                sigma(t_hat),
                condition=x_lr,
                class_labels=class_labels,
            ).to(torch.float64)
        else:
            denoised = net(x_hat / s(t_hat), x_lr, sigma(t_hat), class_labels).to(
                torch.float64
            )
        d_cur = (
            sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)
        ) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == "euler" or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            if isinstance(net, EDMPrecond):
                # Conditioning info is passed as keyword arg
                denoised = net(
                    x_prime / s(t_prime),
                    sigma(t_prime),
                    condition=x_lr,
                    class_labels=class_labels,
                ).to(torch.float64)
            else:
                denoised = net(
                    x_prime / s(t_prime), x_lr, sigma(t_prime), class_labels
                ).to(torch.float64)
            d_prime = (
                sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)
            ) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * (
                (1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime
            )

    return x_next
