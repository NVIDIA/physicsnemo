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

from typing import Any, Literal

from jaxtyping import Float
from tensordict import TensorDict
import torch
from torch import Tensor

from physicsnemo.core import Module
from physicsnemo.diffusion.noise_schedulers import RectifiedFlowNoiseScheduler
from physicsnemo.diffusion.samplers import sample
from physicsnemo.diffusion.utils import ConcatConditionWrapper
from physicsnemo.models.dit import DiT


class TimeScaledModel(Module):
    r"""Scale flow-matching times before passing them to the backbone.

    Flow-matching times live in :math:`[0, 1]`, while the sinusoidal timestep
    embedder of the DiT resolves inputs spanning roughly :math:`[0, 1000]`.
    The scale is part of the saved model, so training and sampling always
    agree.

    Parameters
    ----------
    model : Module
        Backbone called as ``model(x, t, condition=condition)``.
    time_scale : float, optional
        Factor applied to the time before calling the backbone, by default 999.

    Forward
    -------
    x : torch.Tensor
        Noisy state of shape :math:`(B, C, H, W)`.
    t : torch.Tensor
        Flow-matching times of shape :math:`(B,)`.
    condition : torch.Tensor | TensorDict | None, optional
        Condition passed to the backbone, see :func:`build_condition`.
    **model_kwargs : Any
        Other keyword arguments passed to the backbone.

    Outputs
    -------
    torch.Tensor
        Backbone output of shape :math:`(B, C, H, W)`, the predicted clean
        state.
    """

    def __init__(self, model: Module, time_scale: float = 999.0):
        super().__init__()
        self.model = model
        self.time_scale = time_scale

    def forward(
        self,
        x: Float[Tensor, "B C H W"],
        t: Float[Tensor, " B"],
        condition: Tensor | TensorDict | None = None,
        **model_kwargs: Any,
    ) -> Float[Tensor, "B C H W"]:
        if not torch.compiler.is_compiling():
            if x.ndim != 4:
                raise ValueError(
                    f"Expected tensor of shape (B, C, H, W) but got tensor of shape {tuple(x.shape)}"
                )
            if t.shape != (x.shape[0],):
                raise ValueError(
                    f"Expected tensor of shape ({x.shape[0]},) but got tensor of shape {tuple(t.shape)}"
                )
        # Scale in float32: a reduced precision t would quantize the scaled
        # time to a coarser grid than in training
        return self.model(
            x, t.float() * self.time_scale, condition=condition, **model_kwargs
        )


def get_dit(
    target_channels: int,
    conditional_channels: int,
    scalar_condition_channels: int,
    img_resolution: tuple[int, int],
    attn_kernel_size: int = 31,
    qk_norm: bool = True,
    time_scale: float = 999.0,
    **dit_kwargs: Any,
) -> TimeScaledModel:
    """
    Create the diffusion transformer (DiT) that predicts the clean HRRR state.

    Args:
        target_channels: The number of channels in the target
        conditional_channels: The number of channels in the conditioning
        scalar_condition_channels: The number of scalar condition channels
        img_resolution: Resolution of the data (DiT inputs/outputs)
        attn_kernel_size: The neighborhood size of the NATTEN attention
        qk_norm: Whether to normalize the queries and keys of the attention
        time_scale: The factor applied to the times before the DiT embedder
        **dit_kwargs: Any other parameters of the DiT
    Returns:
        TimeScaledModel: a wrapped torch module net(x_t, t, condition) -> x0
    """
    attn_kwargs = {"attn_kernel": attn_kernel_size, "qk_norm": qk_norm}
    if qk_norm:
        # Under bf16 autocast the torch LayerNorm returns float32 queries and
        # keys, while NATTEN needs the same dtype for queries, keys, and values
        attn_kwargs["norm_layer"] = "apex"
    dit = DiT(
        input_size=tuple(img_resolution),
        in_channels=target_channels + conditional_channels,
        out_channels=target_channels,
        condition_dim=scalar_condition_channels,
        attn_kwargs=attn_kwargs,
        **dit_kwargs,
    )
    return TimeScaledModel(ConcatConditionWrapper(dit), time_scale=time_scale)


def build_condition(
    background: torch.Tensor, scalar_conditions: torch.Tensor | None = None
) -> torch.Tensor | TensorDict:
    """Build the model condition: the model concatenates the background with
    the noisy state along the channels, and feeds the scalar conditions to
    the DiT conditioning embedding."""
    if scalar_conditions is None:
        return background
    return TensorDict(
        {"cond_concat": background, "cond_vec": scalar_conditions},
        batch_size=[background.shape[0]],
        device=background.device,
    )


@torch.no_grad()
def sample_flow(
    model: Module,
    condition: torch.Tensor | TensorDict,
    shape: tuple[int, ...],
    noise_scheduler: RectifiedFlowNoiseScheduler,
    num_steps: int = 18,
    solver: Literal["heun", "euler"] = "heun",
    time_shift: float = 1.0,
) -> torch.Tensor:
    r"""
    Sample from an x0-prediction model by integrating the rectified-flow ODE
    from :math:`t = t_{\max}` to :math:`t = 0`.

    For the path :math:`\mathbf{x}_t = (1 - t) \mathbf{x}_0 + t
    \boldsymbol{\epsilon}`, the probability-flow ODE is
    :math:`d\mathbf{x}/dt = \mathbf{v}` with the flow
    :math:`\mathbf{v} = (\mathbf{x}_t - \hat{\mathbf{x}}_0) / t`, which the
    noise scheduler computes from the model output with
    :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.x0_to_flow`.
    The flow stays finite at :math:`t = 1` and the solvers never compute it
    at :math:`t = 0`, so sampling can start from pure noise.

    Parameters
    ----------
    model : Module
        x0-prediction model called as ``model(x_t, t, condition=condition)``.
    condition : torch.Tensor | TensorDict
        Model condition, see :func:`build_condition`.
    shape : tuple[int, ...]
        Shape ``(B, C, H, W)`` of the samples.
    noise_scheduler : RectifiedFlowNoiseScheduler
        Noise scheduler; sampling starts at its ``t_max``.
    num_steps : int, optional
        Number of solver steps, by default 18.
    solver : {"heun", "euler"}, optional
        ODE solver, by default "heun" (two model evaluations per step).
    time_shift : float, optional
        Resolution-dependent time shift :math:`a`. Values above 1 warp the
        time steps as :math:`t \to a t / (1 + (a - 1) t)` to spend more steps
        at high noise levels, which helps when sampling at a higher resolution
        than used in training. By default 1 (no shift).

    Returns
    -------
    torch.Tensor
        Samples of shape ``shape``.
    """
    device = next(model.parameters()).device
    t_steps = noise_scheduler.timesteps(num_steps, device=device)
    t_steps = time_shift * t_steps / (1.0 + (time_shift - 1.0) * t_steps)
    xN = noise_scheduler.init_latents(
        tuple(shape[1:]), t_steps[0].expand(shape[0]), device=device
    )

    def flow(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x0 = model(x, t, condition=condition).float()
        return noise_scheduler.x0_to_flow(x0, x, t)

    # Note: For rectified flow, the right-hand side of the probability-flow ODE
    # is the flow itself, so the sampler uses the flow directly as the denoiser.
    # Converting through noise_scheduler.get_denoiser would divide by
    # alpha(t) = 1 - t, which vanishes at t = 1.
    return sample(
        flow, xN, noise_scheduler, num_steps, solver=solver, time_steps=t_steps
    )
