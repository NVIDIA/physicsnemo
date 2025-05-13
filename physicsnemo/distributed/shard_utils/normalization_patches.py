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

from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from physicsnemo.distributed import ShardTensor, ShardTensorSpec
from physicsnemo.distributed.manager import DistributedManager

__all__ = [
    "group_norm_wrapper",
]


aten = torch.ops.aten


class PartialGroupNorm(torch.autograd.Function):
    """Custom autograd function for applying group normalization to sharded tensors.

    This implementation extends group normalization functionality to work with distributed
    ShardTensor inputs by:
    1. Computing local statistics on each shard
    2. Synchronizing statistics across all shards
    3. Applying the global statistics to normalize each local shard

    The implementation ensures that the result is mathematically equivalent to running
    group normalization on the full, unsharded tensor, while maintaining the distributed
    nature of the computation.

    This class is used by the group_norm_wrapper function to intercept and handle
    torch.nn.functional.group_norm calls with ShardTensor inputs.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        spec: ShardTensorSpec,
        num_groups: int,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        eps: float,
    ) -> ShardTensor:
        """Applies group normalization over a sharded tensor.

        Args:
            ctx: Autograd context
            input: Input tensor of shape [N, C, *]
            spec: Sharding specification for the input tensor
            num_groups: Number of groups to separate the channels into
            weight: Optional scale parameter of shape [C]
            bias: Optional bias parameter of shape [C]
            eps: Small constant added to denominator for numerical stability

        Returns:
            Normalized tensor of same shape as input
        """
        # Save for backward
        ctx.num_groups = num_groups
        ctx.eps = eps
        ctx.spec = spec

        # The syntax is:
        # local_output, mean, rstd = torch.ops.aten.native_group_norm(
        #     input: Tensor, # [N, C, *spatial]
        #     weight: Optional[Tensor],
        #     bias: Optional[Tensor],
        #     N: int,
        #     C: int,
        #     HxW: int,
        #     group: int,
        #     eps: float
        # )

        N, C = input.shape[0], input.shape[1]

        HxW = input.numel() // (N * C)

        local_output, mean, rstd = aten.native_group_norm(
            input, weight, bias, N, C, HxW, num_groups, eps
        )

        # Sync the mean and rstd across all ranks
        # Note that the variance has to be inverted to make it a linear sync:

        global_mean = mean.clone()
        global_var = (1.0 / (rstd**2)) - eps

        # If the mesh is 2D, we still want to reduce this over entire tensor.
        # The DistributedManager provides a caching mechanism for getting a mesh-wide group:
        group = DistributedManager().get_mesh_group(spec.mesh)

        # TODO - unevenly sharded tensors need a *weighted* reduction here!!
        count = len(dist.get_process_group_ranks(group))

        # Could merge these if needed.  They are probably small,
        # so paying more for latency than bandwidth.
        dist.all_reduce(global_mean, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(global_var, op=dist.ReduceOp.SUM, group=group)

        # Compute final global statistics
        global_mean = global_mean / count
        global_var = global_var / count

        global_rstd = torch.rsqrt(global_var + eps)

        # Correct the output from global stats:

        original_shape = input.shape

        broadcast_shape = (N, num_groups, -1)

        scale_factor = (global_rstd / rstd).view(broadcast_shape)

        # Correct to the globally normalized output:
        local_output = (
            local_output.view(broadcast_shape)
            - global_mean.view(broadcast_shape)
            + mean.view(broadcast_shape)
        ) * scale_factor

        local_output = local_output.view(original_shape)

        # Now, apply the weight and
        if weight is not None:
            local_output = local_output * weight.view(1, -1, *([1] * (input.dim() - 2)))
        if bias is not None:
            local_output = local_output + bias.view(1, -1, *([1] * (input.dim() - 2)))

        ctx.save_for_backward(input, weight, bias)
        ctx.global_mean = global_mean
        ctx.global_invstd = global_rstd

        ctx.grad_mask = (
            input.requires_grad,
            weight is not None and weight.requires_grad,
            bias is not None and bias.requires_grad,
        )

        return ShardTensor.from_local(
            local_output,
            spec.mesh,
            spec.placements,
            sharding_shapes=spec.sharding_shapes(),
        )

    @staticmethod
    def backward(
        ctx, grad_output: ShardTensor
    ) -> Tuple[
        torch.Tensor, None, None, Optional[torch.Tensor], Optional[torch.Tensor], None
    ]:
        """Backward pass for group normalization.

        Args:
            ctx: Autograd context containing saved variables
            grad_output: Gradient of the loss with respect to the output

        Returns:
            Tuple containing gradients for inputs, None, None, weights, bias, and None
        """
        input, weight, _ = ctx.saved_tensors
        num_groups = ctx.num_groups
        N, C = input.shape[0], input.shape[1]
        HxW = input.numel() // (N * C)

        local_grad_output = grad_output._local_tensor.contiguous()

        grad_input, grad_weight, grad_bias = aten.native_group_norm_backward(
            local_grad_output,
            input=input,
            mean=ctx.global_mean,
            rstd=ctx.global_invstd,
            weight=weight,
            # bias,
            N=N,
            C=C,
            HxW=HxW,
            group=num_groups,
            output_mask=ctx.grad_mask,
        )

        spec = ctx.spec
        group = DistributedManager().get_mesh_group(spec.mesh)

        # Only reduce if grad_weight or grad_bias is not None
        if grad_weight is not None:
            dist.all_reduce(grad_weight, group=group)
        if grad_bias is not None:
            dist.all_reduce(grad_bias, group=group)

        return grad_input, None, None, grad_weight, grad_bias, None


def group_norm_wrapper(func, types, args, kwargs) -> ShardTensor:
    """Wrapper for torch.nn.functional.group_norm that handles ShardTensor inputs.

    This function intercepts calls to group_norm and handles ShardTensor inputs
    with the PartialGroupNorm custom implementation

    Args:
        func: Original group_norm function
        types:  (unused)
        args: Positional arguments to group_norm
        kwargs: Keyword arguments to group_norm

    Returns:
        Normalized tensor ( ShardTensor)
    """
    input, num_groups, weight, bias, eps = repackage_group_norm_args(*args, **kwargs)

    # Gather any distributed weights/bias
    if isinstance(weight, (ShardTensor, DTensor)):
        weight = weight.full_tensor()
    if isinstance(bias, (ShardTensor, DTensor)):
        bias = bias.full_tensor()

    output_spec = input._spec
    x = PartialGroupNorm.apply(
        input.to_local(), output_spec, num_groups, weight, bias, eps
    )

    return x


def repackage_group_norm_args(
    input: torch.Tensor,
    num_groups: int,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-05,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor], Optional[torch.Tensor], float]:
    """Repackage arguments for group_norm function into a standardized format.

    Args:
        input: Input tensor of shape [N, C, *]
        num_groups: Number of groups to separate the channels into
        weight: Optional scale parameter of shape [C]
        bias: Optional bias parameter of shape [C]
        eps: Small constant added to denominator for numerical stability
        *args: Additional positional arguments (unused)
        **kwargs: Additional keyword arguments (unused)

    Returns:
        Tuple of (input, num_groups, weight, bias, eps)
    """
    return input, num_groups, weight, bias, eps


ShardTensor.register_function_handler(
    torch.nn.functional.group_norm, group_norm_wrapper
)
