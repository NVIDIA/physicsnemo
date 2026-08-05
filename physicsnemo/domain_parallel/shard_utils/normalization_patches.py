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


r"""Normalization operation patches for ShardTensor.

This module provides custom implementations of normalization operations
that work correctly with ``ShardTensor`` objects. The key challenge with
normalization on sharded tensors is that statistics (mean, variance) must
be computed globally across all ranks, not just locally.

The module provides:

- ``PartialGroupNorm``: Custom autograd function for group normalization
- ``group_norm_wrapper``: Function handler for ``torch.nn.functional.group_norm``
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed.tensor import DTensor

from physicsnemo.domain_parallel import ShardTensor, ShardTensorSpec
from physicsnemo.domain_parallel.shard_utils.patch_core import MissingShardPatch

__all__ = [
    "group_norm_wrapper",
]


class PartialGroupNorm(torch.autograd.Function):
    r"""Custom autograd function for applying group normalization to sharded tensors.

    Implements group normalization from first principles so that all
    statistics are computed globally (across ranks) without relying on
    ``aten.native_group_norm`` / ``aten.native_group_norm_backward``.

    The math is straightforward:

    .. math::

        \mu_g      &= \frac{1}{D}\sum_{i \in g} x_i              \\
        \sigma^2_g &= \frac{1}{D}\sum_{i \in g} (x_i - \mu_g)^2  \\
        \hat{x}_i  &= (x_i - \mu_g) / \sqrt{\sigma^2_g + \varepsilon}  \\
        y_i        &= \gamma_c \hat{x}_i + \beta_c

    where :math:`D = \text{cpg} \times HxW_{\text{global}}` and the sums
    are computed via **local partial sums + all-reduce**.

    Forward: 1 all-reduce (sum and sum-of-squares, concatenated).
    Backward: 2 all-reduces (grad correction terms; grad_weight/grad_bias).
    """

    @staticmethod
    def forward(
        input: torch.Tensor,
        spec: ShardTensorSpec,
        num_groups: int,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        eps: float,
    ) -> tuple[ShardTensor, torch.Tensor, torch.Tensor]:
        r"""Apply group normalization over a sharded tensor.

        Parameters
        ----------
        input : torch.Tensor
            Local input tensor of shape :math:`(N, C, *)`.
        spec : ShardTensorSpec
            Sharding specification for the input tensor.
        num_groups : int
            Number of groups to separate the channels into.
        weight : Optional[torch.Tensor]
            Optional scale parameter of shape :math:`(C,)`.
        bias : Optional[torch.Tensor]
            Optional bias parameter of shape :math:`(C,)`.
        eps : float
            Small constant added to denominator for numerical stability.

        Returns
        -------
        tuple[ShardTensor, torch.Tensor, torch.Tensor]
            Tuple of ``(normalized_output, global_mean, global_rstd)`` where
            the trailing two tensors of shape ``(N, G)`` are intermediate
            statistics needed by the backward pass; the public wrapper
            ``group_norm_wrapper`` discards them, and ``setup_context``
            marks them non-differentiable.
        """
        # These are local shapes:
        N, C = input.shape[0], input.shape[1]
        channels_per_group = C // num_groups
        HxW_local = input.numel() // (N * C)

        # some Consistency checks:

        # Not supporting more than one sharded dimension at the moment:
        if spec.mesh.ndim > 1:
            raise MissingShardPatch(
                "Group Normalization is not implemented for sharded tensors with more than one sharded dimension"
            )

        sharded_dim = spec.placements[0].dim
        if sharded_dim == 1:
            raise MissingShardPatch(
                "Group normalization is not implemented for sharded tensors along the channel dimension"
            )

        # Cast weight/bias to input dtype once.
        if weight is not None:
            weight = weight.to(input.dtype)
        if bias is not None:
            bias = bias.to(input.dtype)

        # -- Global statistics via a single all-reduce ----------------------
        # Reshape: (N, C, *spatial) -> (N, G, cpg*HxW_local)
        x = input.view(N, num_groups, -1)

        # Total elements in reduction dimension (correct for uneven sharding).
        global_spatial = spec.tensor_meta.shape[2:]
        global_spatial_numel = 1
        for s in global_spatial:
            global_spatial_numel *= s
        D_global = channels_per_group * global_spatial_numel

        # Local partial sums.
        local_sum = x.sum(dim=2)  # (N, G)
        local_sum_sq = x.pow(2).sum(dim=2)  # (N, G)

        # Fuse into one all-reduce for lower latency. We use the functional
        # collective (parameterized by (mesh, mesh_dim)) rather than
        # ``dist.all_reduce(group=...)`` so the AOT-captured backward graph
        # holds a ``DeviceMesh`` reference instead of a C++ ``ProcessGroup``
        # ScriptObject; AOTAutograd ``deepcopy``s the backward GraphModule
        # during caching and ``ProcessGroup`` has no ``__getstate__``.
        # ``mesh_dim=0`` is safe here because ``mesh.ndim > 1`` is rejected
        # above.
        packed = torch.stack([local_sum, local_sum_sq], dim=0)  # (2, N, G)
        packed = funcol.all_reduce(packed, "sum", (spec.mesh, 0))
        if isinstance(packed, funcol.AsyncCollectiveTensor):
            packed = packed.wait()
        global_sum, global_sum_sq = packed[0], packed[1]

        global_mean = (global_sum / D_global).unsqueeze(2)  # (N, G, 1)
        global_var = (global_sum_sq / D_global) - global_mean.squeeze(2).pow(2)
        global_rstd = torch.rsqrt(
            global_var.unsqueeze(2).clamp(min=0.0) + eps
        )  # (N, G, 1)

        # -- Normalize directly with global stats --------------------------
        y = (x - global_mean) * global_rstd  # (N, G, cpg*HxW_local)

        # Apply per-channel affine: weight (C,) and bias (C,).
        if weight is not None:
            w = (
                weight.view(1, num_groups, channels_per_group, 1)
                .expand(1, num_groups, channels_per_group, HxW_local)
                .reshape(1, num_groups, -1)
            )
            y = y * w
        if bias is not None:
            b = (
                bias.view(1, num_groups, channels_per_group, 1)
                .expand(1, num_groups, channels_per_group, HxW_local)
                .reshape(1, num_groups, -1)
            )
            y = y + b

        local_output = y.view(input.shape)

        shard_output = ShardTensor.from_local(
            local_output,
            spec.mesh,
            spec.placements,
            sharding_shapes=spec.sharding_shapes(),
        )

        # Return statistics so setup_context can save them; the public
        # wrapper discards these extras and they are marked non-diff.
        return shard_output, global_mean.squeeze(2), global_rstd.squeeze(2)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        r"""Save tensors and metadata for the backward pass.

        ``global_mean`` and ``global_rstd`` are intermediate statistics
        computed by ``forward`` and returned as extra outputs; we save them
        via ``save_for_backward`` and mark them non-differentiable so they
        don't appear in the autograd graph as live tensors.
        """
        input, spec, num_groups, weight, bias, eps = inputs
        _shard_output, global_mean, global_rstd = output

        # Re-cast weight/bias to match the input dtype so the backward sees
        # the same dtype as forward used.
        if weight is not None:
            weight = weight.to(input.dtype)
        if bias is not None:
            bias = bias.to(input.dtype)

        ctx.save_for_backward(input, weight, bias, global_mean, global_rstd)
        ctx.num_groups = num_groups
        ctx.eps = eps
        ctx.spec = spec
        ctx.mark_non_differentiable(global_mean, global_rstd)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: ShardTensor,
        _grad_mean: torch.Tensor | None = None,
        _grad_rstd: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        None,
        None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
    ]:
        r"""Backward pass for distributed group normalization.

        Two all-reduces are needed:

        1. ``sum(dx_hat)`` and ``sum(dx_hat * y)`` — for the ``grad_input``
           correction terms (fused into one call).
        2. ``grad_weight`` and ``grad_bias`` — simple per-channel sums
           (fused into one call).

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context containing saved variables.
        grad_output : ShardTensor
            Gradient of the loss with respect to the output.

        Returns
        -------
        Tuple[torch.Tensor, None, None, Optional[torch.Tensor], Optional[torch.Tensor], None]
            Tuple containing gradients for (input, spec, num_groups, weight, bias, eps).
            ``None`` values indicate non-differentiable parameters.
        """
        input, weight, bias, global_mean, global_rstd = ctx.saved_tensors
        num_groups = ctx.num_groups
        N, C = input.shape[0], input.shape[1]
        channels_per_group = C // num_groups
        HxW_local = input.numel() // (N * C)

        local_grad_output = grad_output._local_tensor.contiguous()

        # Ensure grad dtype matches saved input dtype.
        if local_grad_output.dtype != input.dtype:
            local_grad_output = local_grad_output.to(input.dtype)

        spec = ctx.spec

        # Total elements in reduction dimension (correct for uneven sharding).
        global_spatial = spec.tensor_meta.shape[2:]
        global_spatial_numel = 1
        for s in global_spatial:
            global_spatial_numel *= s
        D_global = channels_per_group * global_spatial_numel

        # Reshape to (N, G, cpg * HxW_local) for per-group math.
        x = input.view(N, num_groups, -1)
        grad_out_g = local_grad_output.view(N, num_groups, -1)
        mean_v = global_mean.view(N, num_groups, 1)
        rstd_v = global_rstd.view(N, num_groups, 1)

        # Normalised input: y = (x - mean) * rstd
        y = (x - mean_v) * rstd_v

        # dx_hat = grad_output * weight  (per-channel, broadcast over spatial)
        if weight is not None:
            w_expanded = (
                weight.view(1, num_groups, channels_per_group, 1)
                .expand(1, num_groups, channels_per_group, HxW_local)
                .reshape(1, num_groups, -1)
            )
            dx_hat = grad_out_g * w_expanded
        else:
            dx_hat = grad_out_g

        # -- All-reduce 1: correction terms for grad_input ------------------
        sum_dx_hat = dx_hat.sum(dim=2, keepdim=True)  # (N, G, 1)
        sum_dx_hat_y = (dx_hat * y).sum(dim=2, keepdim=True)  # (N, G, 1)

        # Functional collective: keeps the AOT backward graph free of raw
        # ProcessGroup references (see forward for the full rationale).
        packed_sums = torch.cat([sum_dx_hat, sum_dx_hat_y], dim=2)  # (N, G, 2)
        packed_sums = funcol.all_reduce(packed_sums, "sum", (spec.mesh, 0))
        if isinstance(packed_sums, funcol.AsyncCollectiveTensor):
            packed_sums = packed_sums.wait()
        sum_dx_hat = packed_sums[:, :, :1]  # (N, G, 1)
        sum_dx_hat_y = packed_sums[:, :, 1:]  # (N, G, 1)

        # grad_input = rstd * (dx_hat - mean(dx_hat) - y * mean(dx_hat * y))
        grad_input = rstd_v * (
            dx_hat - sum_dx_hat / D_global - y * sum_dx_hat_y / D_global
        )
        grad_input = grad_input.view(input.shape)

        # -- All-reduce 2: grad_weight and grad_bias ------------------------
        grad_weight = None
        grad_bias = None

        if weight is not None and ctx.needs_input_grad[3]:
            # grad_weight_c = sum_{n, spatial} grad_output * y  (per-channel)
            y_c = y.view(N, C, HxW_local)
            grad_out_c = local_grad_output.view(N, C, HxW_local)
            grad_weight = (grad_out_c * y_c).sum(dim=(0, 2))  # (C,)

        if bias is not None and ctx.needs_input_grad[4]:
            grad_out_c = local_grad_output.view(N, C, HxW_local)
            grad_bias = grad_out_c.sum(dim=(0, 2))  # (C,)

        # Fuse the two small all-reduces when both are needed. Same functional
        # collective rationale as above.
        if grad_weight is not None and grad_bias is not None:
            packed_wb = torch.stack([grad_weight, grad_bias], dim=0)  # (2, C)
            packed_wb = funcol.all_reduce(packed_wb, "sum", (spec.mesh, 0))
            if isinstance(packed_wb, funcol.AsyncCollectiveTensor):
                packed_wb = packed_wb.wait()
            grad_weight, grad_bias = packed_wb[0], packed_wb[1]
        elif grad_weight is not None:
            grad_weight = funcol.all_reduce(grad_weight, "sum", (spec.mesh, 0))
            if isinstance(grad_weight, funcol.AsyncCollectiveTensor):
                grad_weight = grad_weight.wait()
        elif grad_bias is not None:
            grad_bias = funcol.all_reduce(grad_bias, "sum", (spec.mesh, 0))
            if isinstance(grad_bias, funcol.AsyncCollectiveTensor):
                grad_bias = grad_bias.wait()

        return grad_input, None, None, grad_weight, grad_bias, None


def group_norm_wrapper(
    func: Callable,
    types: tuple[Any, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ShardTensor:
    r"""Wrapper for ``torch.nn.functional.group_norm`` that handles ShardTensor inputs.

    This function intercepts calls to group_norm and handles ShardTensor inputs
    with the ``PartialGroupNorm`` custom implementation.

    Parameters
    ----------
    func : Callable
        Original group_norm function.
    types : Any
        Types of the arguments (unused).
    args : Tuple
        Positional arguments to group_norm.
    kwargs : dict
        Keyword arguments to group_norm.

    Returns
    -------
    ShardTensor
        Normalized tensor with the same sharding as input.
    """
    input, num_groups, weight, bias, eps = repackage_group_norm_args(*args, **kwargs)

    # Gather any distributed weights/bias
    if isinstance(weight, (ShardTensor, DTensor)):
        weight = weight.full_tensor()
    if isinstance(bias, (ShardTensor, DTensor)):
        bias = bias.full_tensor()

    output_spec = input._spec
    # PartialGroupNorm returns (output, global_mean, global_rstd); the two
    # extras are intermediate statistics marked non-differentiable and only
    # needed by its backward pass.
    x, _, _ = PartialGroupNorm.apply(
        input.to_local(), output_spec, num_groups, weight, bias, eps
    )

    return x


def repackage_group_norm_args(
    input: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-05,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, int, torch.Tensor | None, torch.Tensor | None, float]:
    r"""Repackage arguments for group_norm function into a standardized format.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape :math:`(N, C, *)`.
    num_groups : int
        Number of groups to separate the channels into.
    weight : Optional[torch.Tensor], optional
        Scale parameter of shape :math:`(C,)`.
    bias : Optional[torch.Tensor], optional
        Bias parameter of shape :math:`(C,)`.
    eps : float, default=1e-05
        Small constant added to denominator for numerical stability.
    *args : Any
        Additional positional arguments (unused).
    **kwargs : Any
        Additional keyword arguments (unused).

    Returns
    -------
    Tuple[torch.Tensor, int, Optional[torch.Tensor], Optional[torch.Tensor], float]
        Tuple of (input, num_groups, weight, bias, eps).
    """
    return input, num_groups, weight, bias, eps


ShardTensor.register_function_handler(
    torch.nn.functional.group_norm, group_norm_wrapper
)
