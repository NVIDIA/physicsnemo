# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

import warnings
from typing import Callable, Literal, Optional

import numpy as np
import nvtx
import torch

from physicsnemo.core.warnings import LegacyFeatureWarning
from physicsnemo.diffusion.multi_diffusion import GridPatching2D
from physicsnemo.diffusion.preconditioners import EDMPrecond, EDMPreconditioner
from tensordict import TensorDict
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)

from physicsnemo.domain_parallel.shard_tensor import scatter_tensor
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard

warnings.warn(
    "The 'deterministic_sampler' function from 'physicsnemo.diffusion.samplers' "
    "is a legacy implementation that will be deprecated in a future release. "
    "Updated implementations will be provided in an upcoming version.",
    LegacyFeatureWarning,
    stacklevel=2,
)

# ruff: noqa: E731


# NOTE: use two wrappers for apply, to avoid recompilation when input shape changes
@torch.compile()
def _apply_wrapper_Cin_channels(patching, input, additional_input=None):
    """
    Apply the patching operation to the input tensor with :math:`C_{in}` channels.
    """
    return patching.apply(input=input, additional_input=additional_input)


@torch.compile()
def _apply_wrapper_Cout_channels_no_grad(patching, input, additional_input=None):
    """
    Apply the patching operation to an input tensor with :math:`C_{out}`
    channels that does not require gradients.
    """
    return patching.apply(input=input, additional_input=additional_input)


@torch.compile()
def _apply_wrapper_Cout_channels_grad(patching, input, additional_input=None):
    """
    Apply the patching operation to an input tensor with :math:`C_{out}`
    channels that requires gradients.
    """
    return patching.apply(input=input, additional_input=additional_input)


@torch.compile()
def _fuse_wrapper(patching, input, batch_size):
    return patching.fuse(input=input, batch_size=batch_size)


def _apply_wrapper_select(
    input: torch.Tensor, patching: GridPatching2D | None
) -> Callable:
    """
    Select the correct patching wrapper based on the input tensor's requires_grad attribute.
    If patching is None, return the identity function.
    If patching is not None, return the appropriate patching wrapper.
    If input.requires_grad is True, return _apply_wrapper_Cout_channels_grad.
    If input.requires_grad is False, return
    _apply_wrapper_Cout_channels_no_grad.
    """
    if patching:
        if input.requires_grad:
            return _apply_wrapper_Cout_channels_grad
        else:
            return _apply_wrapper_Cout_channels_no_grad
    else:
        return lambda patching, input, additional_input=None: input


@nvtx.annotate(message="deterministic_sampler", color="red")
def deterministic_sampler(
    net: torch.nn.Module,
    latents: torch.Tensor,
    img_lr: torch.Tensor,
    class_labels: Optional[torch.Tensor] = None,
    randn_like: Callable = torch.randn_like,
    patching: Optional[GridPatching2D] = None,
    mean_hr: Optional[torch.Tensor] = None,
    lead_time_label: Optional[torch.Tensor] = None,
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
    dtype: torch.dtype = torch.float64,
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
    patching : Optional[GridPatching2D], default=None
        A patching utility for patch-based diffusion. Implements methods to
        extract patches from an image and batch the patches along dim=0.
        Should also implement a ``fuse`` method to reconstruct the original
        image from a batch of patches. See
        :class:`~physicsnemo.diffusion.multi_diffusion.GridPatching2D` for details. By
        default ``None``, in which case non-patched diffusion is used.
    mean_hr : Optional[Tensor], optional
        Optional tensor containing mean high-resolution images for
        conditioning. Must have same height and width as ``img_lr``, with shape
        :math:`(B_{hr}, C_{hr}, H, W)`  where the batch dimension
        :math:`B_{hr}` can be either 1, either equal to ``batch_size``, or can be omitted. If
        :math:`B_{hr} = 1` or is omitted, ``mean_hr`` will be expanded to match the shape
        of ``img_lr``. By default ``None``.
    lead_time_label : Optional[Tensor], optional
        Lead-time labels to pass to the model, shape ``(batch_size,)``.
        If not provided, the model is called without a lead-time label input.
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
    dtype : torch.dtype, optional
        Controls the precision used for sampling
    Returns
    -------
        torch.Tensor:
            Generated batch of samples. Same shape as the input ``latents``.
    """
    if isinstance(img_lr, TensorDict):
        cond_vec = img_lr["cond_vec"]
        img_lr = img_lr["cond_concat"]
    else:
        cond_vec = None

    is_sharded = isinstance(img_lr, DTensor)

    # conditioning = [mean_hr, img_lr, global_lr]
    x_lr = img_lr
    if mean_hr is not None:
        if mean_hr.shape[-2:] != img_lr.shape[-2:]:
            raise ValueError(
                f"mean_hr and img_lr must have the same height and width, "
                f"but found {mean_hr.shape[-2:]} vs {img_lr.shape[-2:]}."
            )
        x_lr = torch.cat((mean_hr.expand(x_lr.shape[0], -1, -1, -1), x_lr), dim=1)

    # Safety check on type of patching
    if patching is not None and not isinstance(patching, GridPatching2D):
        raise ValueError("patching must be an instance of GridPatching2D.")

    # Safety check: if patching is used then img_lr and latents must have same
    # height and width, otherwise there is mismatch in the number
    # of patches extracted to form the final batch_size.
    if patching:
        if img_lr.shape[-2:] != latents.shape[-2:]:
            raise ValueError(
                f"img_lr and latents must have the same height and width, "
                f"but found {img_lr.shape[-2:]} vs {latents.shape[-2:]}. "
            )
    # img_lr and latents must also have the same batch_size, otherwise mismatch
    # when processed by the network
    if img_lr.shape[0] != latents.shape[0]:
        raise ValueError(
            f"img_lr and latents must have the same batch size, but found "
            f"{img_lr.shape[0]} vs {latents.shape[0]}."
        )

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

    # Select default noise level range based on the specified time step discretization.
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

    batch_size = img_lr.shape[0]
    # input and position padding + patching
    if patching:
        # Patched conditioning [x_lr, mean_hr]
        # (batch_size * patch_num, C_in + C_out, patch_shape_y, patch_shape_x)
        x_lr = _apply_wrapper_Cin_channels(
            patching=patching, input=x_lr, additional_input=img_lr
        )

        # Function to select the correct positional embedding for each patch
        def patch_embedding_selector(emb):
            # emb: (N_pe, image_shape_y, image_shape_x)
            # return: (batch_size * patch_num, N_pe, patch_shape_y, patch_shape_x)
            return patching.apply(emb.expand(batch_size, -1, -1, -1))

    else:
        patch_embedding_selector = None

    # Compute corresponding betas for VP.
    vp_beta_d = (
        2
        * (np.log(sigma_min**2 + 1) / epsilon_s - np.log(sigma_max**2 + 1))
        / (epsilon_s - 1)
    )
    vp_beta_min = np.log(sigma_max**2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=dtype, device=latents.device)
    if discretization == "vp":
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == "ve":
        orig_t_steps = (sigma_max**2) * (
            (sigma_min**2 / sigma_max**2) ** (step_indices / (num_steps - 1))
        )
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == "iddpm":
        u = torch.zeros(M + 1, dtype=dtype, device=latents.device)
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

    if is_sharded:
        t_steps = _shard_in_mesh(t_steps, x_lr, shard_dim=None)
        latents = _shard_in_mesh(latents, x_lr, shard_dim=2)

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(dtype) * (sigma(t_next) * s(t_next))

    optional_args = {}
    if lead_time_label is not None:
        optional_args["lead_time_label"] = lead_time_label
    if patching:
        optional_args["embedding_selector"] = patch_embedding_selector

    if cond_vec is not None:
        x_lr = TensorDict({"cond_concat": x_lr, "cond_vec": cond_vec})

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

        # Euler step. Perform patching operation on score tensor if patch-based
        # generation is used denoised = net(x_hat, t_hat,
        # class_labels,lead_time_label=lead_time_label)

        h = t_next - t_hat
        x_hat_batch = _apply_wrapper_select(input=x_hat, patching=patching)(
            patching=patching, input=x_hat
        ).to(latents.device)

        if isinstance(net, EDMPrecond):
            # Conditioning info is passed as keyword arg
            denoised = net(
                x_hat_batch / s(t_hat),
                sigma(t_hat),
                condition=x_lr,
                class_labels=class_labels,
                **optional_args,
            ).to(dtype)
        elif isinstance(net, (EDMPreconditioner, FSDP)):
            denoised = net(
                x_hat_batch / s(t_hat),
                sigma(t_hat).reshape(1).repeat(x_hat_batch.shape[0]),
                condition=x_lr,
                **optional_args,
            ).to(dtype)
        else:
            denoised = net(
                x_hat_batch / s(t_hat),
                x_lr,
                sigma(t_hat),
                class_labels,
                **optional_args,
            ).to(dtype)

        if patching:
            # Un-patch the denoised image
            # (batch_size, C_out, img_shape_y, img_shape_x)
            denoised = _fuse_wrapper(
                patching=patching, input=denoised, batch_size=batch_size
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
            # Patched input
            # (batch_size * patch_num, C_out, patch_shape_y, patch_shape_x)
            x_prime_batch = _apply_wrapper_select(input=x_prime, patching=patching)(
                patching=patching, input=x_prime
            ).to(latents.device)

            if isinstance(net, EDMPrecond):
                # Conditioning info is passed as keyword arg
                denoised = net(
                    x_prime_batch / s(t_prime),
                    sigma(t_prime),
                    condition=x_lr,
                    class_labels=class_labels,
                    **optional_args,
                ).to(dtype)
            elif isinstance(net, (EDMPreconditioner, FSDP)):
                denoised = net(
                    x_prime_batch / s(t_prime),
                    sigma(t_prime).reshape(1).repeat(x_hat_batch.shape[0]),
                    condition=x_lr,
                    **optional_args,
                ).to(dtype)
            else:
                denoised = net(
                    x_prime_batch / s(t_prime),
                    x_lr,
                    sigma(t_prime),
                    class_labels,
                    **optional_args,
                ).to(dtype)

            if patching:
                # Un-patch the denoised image
                # (batch_size, C_out, img_shape_y, img_shape_x)
                denoised = _fuse_wrapper(
                    patching=patching, input=denoised, batch_size=batch_size
                )

            d_prime = (
                sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)
            ) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * (
                (1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime
            )

    return x_next


def _shard_in_mesh(
    x: torch.Tensor, y: torch.Tensor, shard_dim: int | None = None
) -> torch.Tensor:
    if (hasattr(y, "_local_tensor") or hasattr(y, "placements")) and hasattr(
        y, "device_mesh"
    ):
        # y is sharded - need to replicate sigma to be compatible with DTensor operations
        source_rank = torch.distributed.get_global_rank(y.device_mesh.get_group(), 0)
        # Get the source rank for replication
        # Replicate sigma across the domain mesh
        # This ensures all spatial shards see the same sigma values
        placement = Shard(shard_dim) if (shard_dim is not None) else Replicate()
        x = scatter_tensor(
            x,
            source_rank,
            y.device_mesh,
            placements=(placement,),
            global_shape=x.shape,
            dtype=x.dtype,
        )
    return x
