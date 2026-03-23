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

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.domain_parallel.shard_utils.halo import (
    HaloConfig,
    halo_padding,
    unhalo_padding,
)
from physicsnemo.domain_parallel.shard_utils.patch_core import (
    MissingShardPatch,
    UndeterminedShardingError,
)
from physicsnemo.nn.functional.natten import na1d, na2d, na3d

_natten = OptionalImport("natten")
_raw_func_map = {
    na1d: lambda: _natten.functional.na1d,
    na2d: lambda: _natten.functional.na2d,
    na3d: lambda: _natten.functional.na3d,
}

__all__ = ["na1d_wrapper", "na2d_wrapper", "na3d_wrapper"]


def compute_halo_from_kernel_and_dilation(kernel_size: int, dilation: int) -> int:
    r"""Compute the halo size needed for neighborhood attention along a single dimension.

    For neighborhood attention, the halo size is determined by the kernel size and dilation.
    Currently only supports odd kernel sizes with dilation=1.

    Parameters
    ----------
    kernel_size : int
        Size of attention kernel window along this dimension.
    dilation : int
        Dilation factor for attention kernel.

    Returns
    -------
    int
        Required halo size on each side of a data chunk.

    Raises
    ------
    MissingShardPatch
        If kernel configuration is not supported for sharding:
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
) -> list[HaloConfig]:
    r"""Compute halo configurations for a sharded tensor based on neighborhood attention arguments.

    Parameters
    ----------
    example_input : ShardTensor
        The sharded tensor that will be used in neighborhood attention.
    kernel_size : int
        Size of attention kernel window.
    dilation : int
        Dilation factor for attention kernel.

    Returns
    -------
    List[HaloConfig]
        List of HaloConfig objects for each sharded dimension.
    """
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


def _partial_natten(
    q: ShardTensor,
    k: ShardTensor,
    v: ShardTensor,
    kernel_size: int,
    dilation: int,
    base_func: Callable,
    **natten_kwargs: Any,
) -> ShardTensor:
    r"""Compute neighborhood attention on a sharded tensor with halo exchange.

    1. Figure out the size of halos needed.
    2. Apply the halo padding (differentiable)
    3. Perform the neighborhood attention on the padded tensor. (differentiable)
    4. "UnHalo" the output tensor (different from, say, convolutions)
    5. Return the updated tensor as a ShardTensor.

    Parameters
    ----------
    q : ShardTensor
        Query tensor as ShardTensor.
    k : ShardTensor
        Key tensor as ShardTensor.
    v : ShardTensor
        Value tensor as ShardTensor.
    kernel_size : int
        Size of attention kernel window.
    dilation : int
        Dilation factor for attention kernel.
    base_func : Callable
        The base neighborhood attention function to call with padded tensors. Called as
        ``base_func(lq, lk, lv, kernel_size=kernel_size, dilation=dilation, **natten_kwargs)``.
    **natten_kwargs : Any
        Additional keyword arguments passed through to ``base_func`` (e.g. ``is_causal``, ``scale``, ``stride``).

    Returns
    -------
    ShardTensor
        ShardTensor containing the result of neighborhood attention.

    Raises
    ------
    MissingShardPatch
        If kernel configuration is not supported for sharding.
    """
    # First, get the tensors locally and perform halos:
    lq, lk, lv = q.to_local(), k.to_local(), v.to_local()

    # Compute halo configs for these tensors.  We can assume
    # the halo configs are the same for q/k/v and just do it once:
    halo_configs = compute_halo_configs_from_natten_args(q, kernel_size, dilation)

    # Apply the halo padding to the input tensors
    for halo_config in halo_configs:
        lq = halo_padding(lq, q._spec.mesh, halo_config)
        lk = halo_padding(lk, k._spec.mesh, halo_config)
        lv = halo_padding(lv, v._spec.mesh, halo_config)

    # Apply native na2d operation (dilation explicit; other options via natten_kwargs)
    x = base_func(
        lq, lk, lv, kernel_size=kernel_size, dilation=dilation, **natten_kwargs
    )

    # Remove halos and convert back to ShardTensor
    for halo_config in halo_configs:
        x = unhalo_padding(x, q._spec.mesh, halo_config)

    # Convert back to ShardTensor
    x = ShardTensor.from_local(
        x, q._spec.mesh, q._spec.placements, q._spec.sharding_shapes()
    )
    return x


def _natten_wrapper(
    func: Callable,
    types: tuple[Any, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor | ShardTensor:
    r"""Shared wrapper for natten functions to support sharded tensors.

    Registered with :meth:`ShardTensor.register_function_handler` so that calls
    to :func:`~physicsnemo.nn.functional.natten.na1d`,
    :func:`~physicsnemo.nn.functional.natten.na2d`, or
    :func:`~physicsnemo.nn.functional.natten.na3d` automatically route through
    this handler when any argument is a :class:`ShardTensor`.

    Parameters
    ----------
    func : Callable
        The wrapped natten function (passed by ``__torch_function__``).
    types : tuple[Any, ...]
        The types of the inputs (unused).
    args : tuple[Any, ...]
        Positional arguments containing query, key, value tensors and kernel_size.
    kwargs : dict[str, Any]
        Keyword arguments including ``dilation``.

    Returns
    -------
    Union[torch.Tensor, ShardTensor]
        Result tensor as either ``torch.Tensor`` or ShardTensor depending on input types.

    Raises
    ------
    UndeterminedShardingError
        If input tensor types are mismatched.
    """
    q, k, v, kernel_size = args[0], args[1], args[2], args[3]

    dilation = kwargs.get("dilation", 1)
    natten_kwargs = {_k: _v for _k, _v in kwargs.items() if _k != "dilation"}

    if all(type(_t) is torch.Tensor for _t in (q, k, v)):
        return func(
            q, k, v, kernel_size=kernel_size, dilation=dilation, **natten_kwargs
        )
    elif all(isinstance(_t, ShardTensor) for _t in (q, k, v)):
        raw_func = _raw_func_map[func]()
        return _partial_natten(
            q, k, v, kernel_size, dilation, base_func=raw_func, **natten_kwargs
        )
    else:
        raise UndeterminedShardingError(
            "q, k, and v must all be the same types (torch.Tensor or ShardTensor)"
        )


# Public aliases for explicit registration
na1d_wrapper = _natten_wrapper
na2d_wrapper = _natten_wrapper
na3d_wrapper = _natten_wrapper

ShardTensor.register_function_handler(na1d, na1d_wrapper)
ShardTensor.register_function_handler(na2d, na2d_wrapper)
ShardTensor.register_function_handler(na3d, na3d_wrapper)
