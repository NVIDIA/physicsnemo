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

wrapt = OptionalImport("wrapt")
natten = OptionalImport("natten")


__all__ = ["na2d_wrapper"]


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
    **na2d_kwargs: Any,
) -> ShardTensor:
    r"""High-level, differentiable function to compute neighborhood attention on a sharded tensor.

    Operation works like so:

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
        ``base_func(lq, lk, lv, kernel_size, dilation=dilation, **na2d_kwargs)``.
    **na2d_kwargs : Any
        Additional keyword arguments passed through to ``base_func`` (e.g. ``is_causal``, ``scale``, ``stride``).

    Returns
    -------
    ShardTensor
        ShardTensor containing the result of neighborhood attention.

    Raises
    ------
    MissingShardPatch
        If kernel configuration is not supported for sharding.
    UndeterminedShardingError
        If input tensor types are mismatched.
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

    # Apply native na2d operation (dilation explicit; other options via na2d_kwargs)
    x = base_func(lq, lk, lv, kernel_size, dilation=dilation, **na2d_kwargs)

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
if natten.available and wrapt.available:

    @wrapt.patch_function_wrapper(
        "natten.functional", "na2d", enabled=ShardTensor.patches_enabled
    )
    def na2d_wrapper(
        wrapped: Callable, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> torch.Tensor | ShardTensor:
        r"""Wrapper for ``natten.functional.na2d`` to support sharded tensors.

        Handles both regular ``torch.Tensor`` inputs and distributed ShardTensor inputs.
        For regular tensors, passes through to the wrapped na2d function.
        For ShardTensor inputs, handles adding halos and applying distributed na2d.

        Parameters
        ----------
        wrapped : Callable
            Original na2d function being wrapped.
        instance : Any
            Instance the wrapped function is bound to (unused).
        args : tuple[Any, ...]
            Positional arguments containing query, key, value tensors.
        kwargs : dict[str, Any]
            Keyword arguments including ``kernel_size`` and ``dilation``.

        Returns
        -------
        Union[torch.Tensor, ShardTensor]
            Result tensor as either ``torch.Tensor`` or ShardTensor depending on input types.

        Raises
        ------
        UndeterminedShardingError
            If input tensor types are mismatched.
        """

        def fetch_qkv(
            q: Any, k: Any, v: Any, *args: Any, **kwargs: Any
        ) -> tuple[Any, Any, Any]:
            r"""Extract query, key, value tensors from args.

            Parameters
            ----------
            q : Any
                Query tensor.
            k : Any
                Key tensor.
            v : Any
                Value tensor.
            *args : Any
                Additional positional arguments (unused).
            **kwargs : Any
                Additional keyword arguments (unused).

            Returns
            -------
            Tuple[Any, Any, Any]
                Tuple of (query, key, value) tensors.
            """
            return q, k, v

        q, k, v = fetch_qkv(*args)

        # Get kernel parameters (keep explicit); pass remaining kwargs through to na2d
        dilation = kwargs.get("dilation", 1)
        kernel_size = kwargs["kernel_size"]
        na2d_kwargs = {
            k: v for k, v in kwargs.items() if k not in ("kernel_size", "dilation")
        }

        if all([isinstance(_t, torch.Tensor) for _t in (q, k, v)]):
            return wrapped(*args, **kwargs)
        elif all([isinstance(_t, ShardTensor) for _t in (q, k, v)]):
            return partial_na2d(
                q, k, v, kernel_size, dilation, base_func=wrapped, **na2d_kwargs
            )

        else:
            raise UndeterminedShardingError(
                "q, k, and v must all be the same types (torch.Tensor or ShardTensor)"
            )

else:

    def na2d_wrapper(*args: Any, **kwargs: Any) -> None:
        r"""Placeholder wrapper when natten module is not installed.

        Parameters
        ----------
        *args : Any
            Positional arguments (unused).
        **kwargs : Any
            Keyword arguments (unused).

        Raises
        ------
        Exception
            Always raised indicating natten is not installed.
        """
        raise Exception(
            "na2d_wrapper not supported because module 'natten' not installed"
        )


# Clean up OptionalImport references from module namespace.
# inspect.unwrap (used by doctest collection) checks hasattr(obj, '__wrapped__')
# on every module-level object; on OptionalImport this triggers __getattr__ which
# raises RuntimeError (not AttributeError) when the package is missing, crashing
# the doctest collector.  The references are no longer needed after the if/else.
del wrapt, natten
