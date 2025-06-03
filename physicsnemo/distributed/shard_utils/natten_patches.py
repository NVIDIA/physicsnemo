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
from typing import Any, Callable, List, Tuple, Union

import torch
import wrapt

from physicsnemo.utils.version_check import check_module_requirements

check_module_requirements("physicsnemo.distributed.shard_tensor")

from torch.distributed.tensor.placement_types import Shard  # noqa: E402

from physicsnemo.distributed import ShardTensor  # noqa: E402
from physicsnemo.distributed.shard_utils.halo import (  # noqa: E402
    HaloConfig,
    halo_padding,
    unhalo_padding,
)
from physicsnemo.distributed.shard_utils.patch_core import (  # noqa: E402
    MissingShardPatch,
    UndeterminedShardingError,
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


def compute_halo_configs_from_natten_args(
    example_input: ShardTensor,
    kernel_size: int,
    dilation: int,
) -> List[HaloConfig]:
    """Compute halo configurations for a sharded tensor based on convolution arguments.

    Args:
        example_input: The sharded tensor that will be used in neighborhood attention
        kernel_size: Size of attention kernel window
        dilation: Dilation factor for attention kernel

    Returns:
        List of HaloConfig objects for each sharded dimension
    """
    # Compute required halo size from kernel parameters
    halo_size = compute_halo_from_kernel_and_dilation(kernel_size, dilation)

    placements = example_input._spec.placements

    halo_configs = []

    for mesh_dim, p in enumerate(placements):
        if not isinstance(p, Shard):
            continue

        tensor_dim = p.dim
        if tensor_dim in [
            0,
        ]:  # Skip batch dim
            continue

        # Compute required halo size from kernel parameters
        halo_size = compute_halo_from_kernel_and_dilation(kernel_size, dilation)

        if halo_size > 0:
            # Create a halo config for this dimension
            halo_configs.append(
                HaloConfig(
                    mesh_dim=mesh_dim,
                    tensor_dim=tensor_dim,
                    halo_size=halo_size,
                    edge_padding_size=0,  # Always 0 for natten
                    communication_method="a2a",
                )
            )

    return halo_configs


def partial_na2d(
    q: ShardTensor,
    k: ShardTensor,
    v: ShardTensor,
    kernel_size: int,
    dilation: int,
    base_func: Callable,
) -> ShardTensor:
    """
    High Level, differentiable function to compute neighborhood attention on a sharded tensor.

    Operation works like so:
    - Figure out the size of halos needed.
    - Apply the halo padding (differentiable)
    - Perform the neighborhood attention on the padded tensor. (differentiable)
    - "UnHalo" the output tensor (different from, say, convolutions)
    - Return the updated tensor as a ShardTensor.

    Args:
        q: Query tensor as ShardTensor
        k: Key tensor as ShardTensor
        v: Value tensor as ShardTensor
        kernel_size: Size of attention kernel window
        dilation: Dilation factor for attention kernel
        base_func: The base neighborhood attention function to call with padded tensors

    Returns:
        ShardTensor containing the result of neighborhood attention

    Raises:
        MissingShardPatch: If kernel configuration is not supported for sharding
        UndeterminedShardingError: If input tensor types are mismatched
    """

    # First, get the tensors locally and perform halos:
    lq, lk, lv = q.to_local(), k.to_local(), v.to_local()

    # Compute halo configs for these tensors.  We can assume
    # the halo configs are the same for q/k/v and just do it once:

    halo_configs = compute_halo_configs_from_natten_args(q, kernel_size, dilation)

    # Apply the halo padding to the input tensor
    for halo_config in halo_configs:
        lq = halo_padding(lq, q._spec.mesh, halo_config)
        lk = halo_padding(lk, k._spec.mesh, halo_config)
        lv = halo_padding(lv, v._spec.mesh, halo_config)

    # Apply native na2d operation
    x = base_func(lq, lk, lv, kernel_size, dilation)

    # Remove halos and convert back to ShardTensor
    # x = UnSliceHaloND.apply(x, halo, q._spec)
    for halo_config in halo_configs:
        x = unhalo_padding(x, q._spec.mesh, halo_config)

    # Convert back to ShardTensor
    x = ShardTensor.from_local(
        x, q._spec.mesh, q._spec.placements, q._spec.sharding_shapes()
    )
    return x


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

            return partial_na2d(q, k, v, kernel_size, dilation, base_func=wrapped)

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
