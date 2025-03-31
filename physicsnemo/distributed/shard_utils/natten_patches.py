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

import importlib.util
from typing import Any, Tuple, Union

import torch
import wrapt

from physicsnemo.utils.version_check import check_module_requirements

check_module_requirements("physicsnemo.distributed.shard_tensor")

from torch.distributed.tensor.placement_types import Shard  # noqa: E402

from physicsnemo.distributed import ShardTensor  # noqa: E402
from physicsnemo.distributed.shard_utils.patch_core import (  # noqa: E402
    MissingShardPatch,
    UndeterminedShardingError,
)

from .halo import (  # noqa: E402
    halo_padding_1d,
    halo_unpadding_1d,
)

__all__ = ["na2d_wrapper"]


def compute_halo_from_kernel_and_dilation(kernel_size: int, dilation: int) -> int:
    """Compute the halo size needed for neighborhood attention along a single dimension.

    For neighborhood attention, the halo size is determined by the kernel size and dilation.
    Currently only supports odd kernel sizes with dilation=1.

    Args:
        kernel_size: Size of attention kernel window along this dimension
        dilation: Dilation factor for attention kernel

    Returns:
        Required halo size on each side of a data chunk

    Raises:
        MissingShardPatch: If kernel configuration is not supported for sharding
            - Even kernel sizes not supported
            - Dilation != 1 not supported
    """
    # Currently, reject even kernel_sizes and dilation != 1:
    if kernel_size % 2 == 0:
        raise MissingShardPatch(
            "Neighborhood Attention is not implemented for even kernels"
        )
    if dilation != 1:
        raise MissingShardPatch(
            "Neighborhood Attention is not implemented for dilation != 1"
        )

    # For odd kernels with dilation=1, halo is half the kernel size (rounded down)
    halo = int(kernel_size // 2)

    return halo


def shard_to_haloed_local(
    q: ShardTensor, k: ShardTensor, v: ShardTensor, kernel_size: int, dilation: int = 1
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], list[int]]:
    """Add halo regions to query, key and value tensors for neighborhood attention.

    For neighborhood attention, each tensor needs access to neighboring values within
    the kernel window. This function adds halo regions to sharded q/k/v tensors by
    gathering values from adjacent ranks.

    Args:
        q: Query tensor, sharded across device mesh
        k: Key tensor, must be sharded same as query
        v: Value tensor, must be sharded same as query
        kernel_size: Size of attention window
        dilation: Dilation factor for attention window, must be 1

    Returns:
        Tuple containing:
        - Tuple of (padded_q, padded_k, padded_v) local tensors with halos
        - List of halo sizes for each mesh dimension

    Raises:
        ValueError: If q/k/v are not sharded on same mesh
    """
    # Verify q/k/v use same device mesh
    if q._spec.mesh != k._spec.mesh:
        raise ValueError("Mismatched mesh not supported in na2d")
    if q._spec.mesh != v._spec.mesh:
        raise ValueError("Mismatched mesh not supported in na2d")

    # Compute required halo size from kernel parameters
    halo_size = compute_halo_from_kernel_and_dilation(kernel_size, dilation)

    # Get device mesh and create halo params
    mesh = q._spec.mesh
    halo = [halo_size] * mesh.ndim
    edge_padding_s = [0] * mesh.ndim
    edge_padding_t = "none"

    # TODO: Verify q/k/v have identical sharding

    # Add halos to each tensor
    local_padded_q = HaloPaddingND.apply(
        q,
        halo,
        edge_padding_t,
        edge_padding_s,
    )

    local_padded_k = HaloPaddingND.apply(
        k,
        halo,
        edge_padding_t,
        edge_padding_s,
    )

    local_padded_v = HaloPaddingND.apply(
        v,
        halo,
        edge_padding_t,
        edge_padding_s,
    )

    return (local_padded_q, local_padded_k, local_padded_v), halo


class HaloPaddingND(torch.autograd.Function):
    """Autograd wrapper for distributed halo padding.

    Handles halo padding for distributed tensors using ShardTensor concept
    (local tensor + device mesh + shard placements). Forward pass gathers adjacent regions
    from neighboring devices. Backward pass distributes gradients outward.

    Supports multi-dimensional halo passing with compatible mesh and halo parameters.
    """

    @staticmethod
    def forward(
        ctx,
        stensor: ShardTensor,
        halo: Tuple[int, ...],
        edge_padding_t: str,
        edge_padding_s: Tuple[int, ...],
    ) -> torch.Tensor:
        """Forward pass of distributed halo padding.

        Args:
            ctx: Autograd context for saving tensors
            stensor: Input ShardTensor
            halo: Halo sizes for each dimension
            edge_padding_t: Edge padding type ("zeros" or "none")
            edge_padding_s: Edge padding sizes

        Returns:
            Padded local tensor

        Raises:
            ValueError: If halo size doesn't match mesh dimensions
        """
        mesh = stensor.device_mesh
        if len(halo) != mesh.ndim:
            raise ValueError(
                f"Halo size ({len(halo)}) must match mesh rank ({mesh.ndim})"
            )

        placements = stensor.placements
        local_tensor = stensor.to_local()

        # Apply halo padding for each sharded dimension
        for mesh_dim in range(mesh.ndim):
            if isinstance(placements[mesh_dim], Shard):
                tensor_dim = placements[mesh_dim].dim
                local_tensor = halo_padding_1d(
                    local_tensor,
                    mesh,
                    mesh_dim,
                    tensor_dim,
                    halo[mesh_dim],
                    edge_padding_t,
                    edge_padding_s[0],
                )

        # Save context for backward pass
        ctx.halo = halo
        ctx.spec = stensor._spec
        ctx.requires_input_grad = stensor.requires_grad

        return local_tensor

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[ShardTensor, None, None, None]:
        """Backward pass of distributed halo padding.

        Args:
            ctx: Autograd context containing saved tensors
            grad_output: Gradient tensor from downstream

        Returns:
            Tuple containing:
            - Gradient for input tensor
            - None for other inputs (halo, padding_type, padding_size)
        """
        spec = ctx.spec
        mesh = spec.mesh
        placements = spec.placements
        halo = ctx.halo

        # Remove halos from gradients in reverse order
        for mesh_dim in range(mesh.ndim):
            if isinstance(placements[mesh_dim], Shard):
                tensor_dim = placements[mesh_dim].dim
                grad_output = halo_unpadding_1d(
                    grad_output, mesh, mesh_dim, tensor_dim, halo[mesh_dim]
                )

        # Wrap gradient in ShardTensor
        grad_tensor = ShardTensor(
            grad_output,
            spec,
            requires_grad=grad_output.requires_grad,
        )

        return grad_tensor, None, None, None


class UnSliceHaloND(torch.autograd.Function):
    """Autograd function to remove halo regions from a tensor after halo computation.

    Used to trim off unnecessary halo sections after operations like neighborhood attention
    that require halo regions for computation but not in the final output.

    Forward pass removes halo regions by unpadding along sharded dimensions.
    Backward pass adds halo regions back via padding to match the original shape.
    """

    @staticmethod
    def forward(
        ctx,
        local_tensor: torch.Tensor,
        halo: tuple[int, ...],
        mesh: torch.distributed.device_mesh.DeviceMesh,
        placements: tuple[torch.distributed.tensor.placement_types.Placement, ...],
    ) -> "ShardTensor":
        """Forward pass to remove halo regions.

        Args:
            ctx: Autograd context for saving tensors
            local_tensor: Input tensor with halo regions
            halo: Tuple of halo sizes for each mesh dimension
            mesh: Device mesh for distributed computation
            placements: Tuple of placement specs for each mesh dimension

        Returns:
            ShardTensor with halo regions removed

        Raises:
            ValueError: If halo size does not match mesh rank
        """
        # Save context for backward pass
        ctx.halo = halo
        ctx.mesh = mesh
        ctx.placements = placements

        if len(halo) != mesh.ndim:
            raise ValueError(
                f"Halo size ({len(halo)}) must match mesh rank ({mesh.ndim})"
            )

        # Remove halos along sharded dimensions
        for mesh_dim in range(mesh.ndim):
            if isinstance(placements[mesh_dim], Shard):
                tensor_dim = placements[mesh_dim].dim
                local_tensor = halo_unpadding_1d(
                    local_tensor, mesh, mesh_dim, tensor_dim, halo[mesh_dim]
                )

        # Convert to ShardTensor
        stensor = ShardTensor.from_local(local_tensor, mesh, placements)
        return stensor

    @staticmethod
    def backward(
        ctx, grad_output: "ShardTensor"
    ) -> tuple[torch.Tensor, None, None, None]:
        """Backward pass to add halo regions back.

        Args:
            ctx: Autograd context with saved tensors
            grad_output: Gradient tensor from downstream

        Returns:
            Tuple containing:
            - Gradient tensor with halo regions added back
            - None for other inputs (halo, mesh, placements)
        """
        mesh = ctx.mesh
        halo = ctx.halo
        placements = ctx.placements

        # Configure padding parameters
        edge_padding_s = [0] * len(halo)
        edge_padding_t = "none"

        # Add halos back via padding
        local_tensor = grad_output.to_local()
        for mesh_dim in range(mesh.ndim):
            if isinstance(placements[mesh_dim], Shard):
                tensor_dim = placements[mesh_dim].dim
                local_tensor = halo_padding_1d(
                    local_tensor,
                    mesh,
                    mesh_dim,
                    tensor_dim,
                    halo[mesh_dim],
                    edge_padding_t,
                    edge_padding_s[mesh_dim],
                )

        return local_tensor, None, None, None


# Make sure the module exists before importing it:

natten_spec = importlib.util.find_spec("natten")
if natten_spec is not None:

    @wrapt.patch_function_wrapper(
        "natten.functional", "na2d", enabled=ShardTensor.patches_enabled
    )
    def na2d_wrapper(
        wrapped: Any, instance: Any, args: tuple, kwargs: dict
    ) -> Union[torch.Tensor, ShardTensor]:
        """Wrapper for natten.functional.na2d to support sharded tensors.

        Handles both regular torch.Tensor inputs and distributed ShardTensor inputs.
        For regular tensors, passes through to the wrapped na2d function.
        For ShardTensor inputs, handles adding halos and applying distributed na2d.

        Args:
            wrapped: Original na2d function being wrapped
            instance: Instance the wrapped function is bound to
            args: Positional arguments containing query, key, value tensors
            kwargs: Keyword arguments including kernel_size and dilation

        Returns:
            Result tensor as either torch.Tensor or ShardTensor depending on input types

        Raises:
            UndeterminedShardingError: If input tensor types are mismatched
        """

        def fetch_qkv(
            q: Any, k: Any, v: Any, *args: Any, **kwargs: Any
        ) -> Tuple[Any, Any, Any]:
            """Helper to extract query, key, value tensors from args."""
            return q, k, v

        q, k, v = fetch_qkv(*args)

        # Get kernel parameters
        dilation = kwargs.get("dilation", 1)
        kernel_size = kwargs["kernel_size"]

        if all([type(_t) == torch.Tensor for _t in (q, k, v)]):
            return wrapped(*args, **kwargs)
        elif all([type(_t) == ShardTensor for _t in (q, k, v)]):
            # This applies a halo layer and returns local torch tensors:
            (lq, lk, lv), halo = shard_to_haloed_local(q, k, v, kernel_size, dilation)

            # Apply native na2d operation
            x = wrapped(lq, lk, lv, kernel_size, dilation)

            # Remove halos and convert back to ShardTensor
            x = UnSliceHaloND.apply(x, halo, q._spec.mesh, q._spec.placements)
            return x

        else:
            raise UndeterminedShardingError(
                "q, k, and v must all be the same types (torch.Tensor or ShardTensor)"
            )

else:

    def na2d_wrapper(*args: Any, **kwargs: Any) -> None:
        """Placeholder wrapper when natten module is not installed."""
        raise Exception(
            "na2d_wrapper not supported because module 'natten' not installed"
        )
