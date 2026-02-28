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

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal

import numpy as np
from tensordict import TensorDict
import torch

from physicsnemo.core import Module
from physicsnemo.models.diffusion_unets import StormCastUNet
from physicsnemo.diffusion.preconditioners import EDMPrecond, EDMPreconditioner
from physicsnemo.diffusion.utils import ConcatConditionWrapper

# from physicsnemo.diffusion.samplers import deterministic_sampler
from physicsnemo.models.dit import DiT

from utils.sampler import deterministic_sampler
import utils.apex  # do not remove, enables Apex LayerNorm with ShardTensor


def get_preconditioned_unet(
    name: str,
    target_channels: int,
    conditional_channels: int = 0,
    spatial_embedding: bool = True,
    img_resolution: tuple = (512, 640),
    model_type: str | None = None,
    channel_mult: list = [1, 2, 2, 2, 2],
    attn_resolutions: list = [],
    lead_time_steps: int = 0,
    lead_time_channels: int = 4,
    amp_mode: bool = False,
    **model_kwargs,
) -> EDMPrecond | StormCastUNet:
    """
    Create a preconditioner-wrapped SongUNet network.

    Args:
        name: 'regression' or 'diffusion' to select between either model type
        target_channels: The number of channels in the target
        conditional_channels: The number of channels in the conditioning
        spatial_embedding: whether or not to use the additive spatial embedding in the U-Net
        img_resolution: resolution of the data (U-Net inputs/outputs)
        model_type: the model class to use, or None to select it automatically
        channel_mult: the channel multipliers for the different levels of the U-Net
        attn_resolutions: resolution of internal U-Net stages to use self-attention
        lead_time_steps: the number of possible lead time steps, if 0 lead time embedding will be disabled
        lead_time_channels: the number of channels to use for each lead time embedding
    Returns:
        EDMPrecond or StormCastUNet: a wrapped torch module net(x+n, sigma, condition, class_labels) -> x
    """

    if model_type is None:
        model_type = "SongUNetPosLtEmbd" if lead_time_steps else "SongUNet"

    model_params = {
        "img_resolution": img_resolution,
        "img_out_channels": target_channels,
        "model_type": model_type,
        "channel_mult": channel_mult,
        "attn_resolutions": attn_resolutions,
        "additive_pos_embed": spatial_embedding,
        "amp_mode": amp_mode,
    }
    model_params.update(model_kwargs)

    if lead_time_steps:
        model_params["N_grid_channels"] = 0
        model_params["lead_time_channels"] = lead_time_channels
        model_params["lead_time_steps"] = lead_time_steps
    else:
        lead_time_channels = 0

    if name == "diffusion":
        return EDMPrecond(
            img_channels=target_channels + conditional_channels + lead_time_channels,
            **model_params,
        )

    elif name == "regression":
        return StormCastUNet(
            img_in_channels=conditional_channels + lead_time_channels,
            embedding_type="zero",
            **model_params,
        )


def get_preconditioned_natten_dit(
    target_channels: int,
    conditional_channels: int = 0,
    scalar_condition_channels: int = 0,
    img_resolution: tuple = (512, 640),
    hidden_size: int = 768,
    depth: int = 16,
    num_heads: int = 16,
    patch_size: int = 4,
    attn_kernel_size: int = 31,
    lead_time_steps: int = 0,
    layernorm_backend: Literal["torch", "apex"] = "apex",
    conditioning_embedder: Literal["dit", "edm", "zero"] = "dit",
    **model_kwargs,
) -> EDMPreconditioner:
    """
    Create a preconditioner-wrapped Diffusion Transformer (DiT) network.

    Args:
        target_channels: The number of channels in the target
        conditional_channels: The number of channels in the conditioning
        scalar_condition_channels: The number of scalar condition channels
        img_resolution: Resolution of the data (DiT inputs/outputs)
        hidden_size: The number of channels in the internal layers of the DiT
        depth: The number of transformer blocks in the DiT
        num_heads: number of heads in multi-head attention
        patch_size: the patch size used by the DiT embedder
        attn_kernel_size: the attention neighborhood size
        lead_time_steps: the number of possible lead time steps, if 0 lead time embedding will be disabled
        **model_kwargs: any additional parameters to the model
    Returns:
        EDMPrecond or StormCastUNet: a wrapped torch module net(x+n, sigma, condition, class_labels) -> x
    """

    condition_dim = scalar_condition_channels + lead_time_steps
    attn_kwargs = {"attn_kernel": attn_kernel_size}
    dit = DiT(
        input_size=img_resolution,
        in_channels=target_channels + conditional_channels,
        out_channels=target_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        patch_size=patch_size,
        attention_backend="natten2d",
        layernorm_backend=layernorm_backend,
        attn_kwargs=attn_kwargs,
        condition_dim=condition_dim,
        conditioning_embedder=conditioning_embedder,
        **model_kwargs,
    )
    return EDMPreconditioner(model=ConcatConditionWrapper(dit))


def build_network_condition_and_target(
    background: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor],
    invariant_tensor: torch.Tensor | None,
    scalar_conditions: torch.Tensor | None = None,
    lead_time_label: torch.Tensor | None = None,
    regression_net: Module | None = None,
    condition_list: Iterable[str] = ("state", "background"),
    regression_condition_list: Iterable[str] = ("state", "background"),
) -> tuple[torch.Tensor | TensorDict, torch.Tensor, torch.Tensor | None]:
    """Build the condition and target tensors for the network.

    Args:
        background: background tensor
        state: tuple of previous state and target state
        invariant_tensor: invariant tensor or None if no invariant is used
        lead_time_label: lead time label or None if lead time embedding is not used
        regression_net: regression model, can be None if 'regression' is not in condition_list
        condition_list: list of conditions to include, may include 'state', 'background', 'regression' and 'invariant'
        regression_condition_list: list of conditions for the regression network, may include 'state', 'background', and 'invariant'
            This is only used if regression_net is set.
    Returns:
        A tuple of tensors: (
            condition: model condition concatenated from conditions specified in condition_list,
            target: training target,
            regression: regression model output
        ). The regression model output will be None if 'regression' is not in condition_list.
    """
    if ("regression" in condition_list) and (regression_net is None):
        raise ValueError(
            "regression_net must be provided if 'regression' is in condition_list"
        )
    target = state[1]

    condition_tensors = {
        "state": state[0],
        "background": background,
        "invariant": invariant_tensor,
        "regression": None,
    }

    with torch.no_grad():
        if "regression" in condition_list:
            # Inference regression model
            condition_tensors["regression"] = regression_model_forward(
                regression_net,
                state[0],
                background,
                invariant_tensor,
                lead_time_label=lead_time_label,
                condition_list=regression_condition_list,
            )
            target = target - condition_tensors["regression"]

        condition = [
            y for c in condition_list if (y := condition_tensors[c]) is not None
        ]
        condition = torch.cat(condition, dim=1)

    if scalar_conditions is not None:
        condition = TensorDict(
            {"cond_concat": condition, "cond_vec": scalar_conditions},
            device=condition.device,
        ).to(dtype=condition.dtype)

    return (condition, target, condition_tensors["regression"])


def unpack_batch(
    batch: dict[str, Any],
    device: torch.device | str,
    memory_format: torch.memory_format = torch.preserve_format,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack a data batch into background, state and lead time label with the correct
    device and data types.
    """
    (background, state, mask) = nested_to(
        (batch["background"], batch["state"], batch.get("mask")),
        device=device,
        dtype=torch.float32,
        non_blocking=True,
        memory_format=memory_format,
    )

    lead_time_label = batch.get("lead_time_label")
    if lead_time_label is not None:
        lead_time_label = lead_time_label.to(
            device=device, dtype=torch.int64, non_blocking=True
        )
    scalar_conditions = batch.get("scalar_conditions")
    if scalar_conditions is not None:
        scalar_conditions = scalar_conditions.to(
            device=device, dtype=torch.float32, non_blocking=True
        )

    return (background, state, mask, lead_time_label, scalar_conditions)


def diffusion_model_forward(
    model: Module,
    condition: torch.Tensor,
    shape: Iterable[int],
    lead_time_label: torch.Tensor | None = None,
    sampler_args: dict[str, Any] = {},
) -> torch.Tensor:
    """Helper function to run diffusion model sampling"""

    # TODO: avoid creating full tensor here when sharding
    latents = torch.randn(*shape, device=condition.device, dtype=condition.dtype)

    if not hasattr(model, "sigma_min"):
        model.sigma_min = 0.0
    if not hasattr(model, "sigma_max"):
        model.sigma_max = np.inf
    if not hasattr(model, "round_sigma"):
        model.round_sigma = torch.as_tensor

    return deterministic_sampler(
        model,
        latents=latents,
        img_lr=condition,
        lead_time_label=lead_time_label,
        dtype=condition.dtype,
        **sampler_args,
    )


def regression_model_forward(
    model: Module,
    state: torch.Tensor,
    background: torch.Tensor,
    invariant_tensor: torch.Tensor,
    lead_time_label: torch.Tensor | None = None,
    condition_list: Iterable[str] = ("state", "background"),
) -> torch.Tensor:
    """Helper function to run regression model forward pass in inference"""

    (x, _, _) = build_network_condition_and_target(
        background,
        (state, None),
        invariant_tensor,
        lead_time_label=lead_time_label,
        condition_list=condition_list,
    )

    labels = {} if lead_time_label is None else {"lead_time_label": lead_time_label}
    return model(x, **labels)


def regression_loss_fn(
    net: Module,
    images: torch.Tensor,
    condition: torch.Tensor,
    class_labels: None = None,
    lead_time_label: torch.Tensor | None = None,
    augment_pipe: Callable | None = None,
    return_model_outputs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Helper function for training the StormCast regression model, so that it has a similar call signature as
    the EDMLoss and the same training loop can be used to train both regression and diffusion models

    Args:
        net: physicsnemo.models.diffusion.StormCastUNet
        images: Target data, shape [batch_size, target_channels, w, h]
        condition: input to the model, shape=[batch_size, condition_channel, w, h]
        class_labels: unused (applied to match EDMLoss signature)
        lead_time_label: lead time label or None if lead time embedding is not used
        augment_pipe: optional data augmentation pipe
        return_model_outputs: If True, will return the generated outputs
    Returns:
        out: loss function with shape [batch_size, target_channels, w, h]
            This should be averaged to get the mean loss for gradient descent.
    """

    y, augment_labels = (
        augment_pipe(images) if augment_pipe is not None else (images, None)
    )

    labels = {} if lead_time_label is None else {"lead_time_label": lead_time_label}
    D_yn = net(x=condition, **labels)
    loss = (D_yn - y) ** 2
    if return_model_outputs:
        return loss, D_yn
    else:
        return loss


def nested_to(
    x: torch.Tensor | Mapping | list | tuple | Any, **kwargs
) -> torch.Tensor | dict | list | Any:
    """Move tensors in nested structures to a device/dtype.

    Parameters
    ----------
    x : torch.Tensor or Mapping or list or tuple
        Input structure.
    **kwargs
        Keyword arguments forwarded to ``Tensor.to``.

    Returns
    -------
    torch.Tensor or dict or list
        Structure with tensors moved.
    """
    if isinstance(x, Mapping):
        return {k: nested_to(v, **kwargs) for (k, v) in x.items()}
    elif isinstance(x, (list, tuple)):
        return [nested_to(v, **kwargs) for v in x]
    else:
        if not isinstance(x, torch.Tensor):
            return x
        return x.to(**kwargs)
