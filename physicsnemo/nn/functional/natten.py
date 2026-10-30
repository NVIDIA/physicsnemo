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

from typing import Any, Optional

import torch

from physicsnemo.core.version_check import OptionalImport

_natten = OptionalImport("natten")

# Options the ``physicsnemo::na{1,2,3}d`` ops carry, with NATTEN's defaults.
_OP_OPTIONS = {"is_causal": False, "scale": None, "backend": None}


def _register_na_op(name: str) -> torch._ops.OpOverload:
    r"""Define ``physicsnemo::<name>``: NATTEN's ``<name>`` as a graph op.

    Tensor subclasses such as ``ShardTensor`` intercept the op by name in
    ``__torch_function__`` (``register_named_function_handler``), in eager
    and under ``torch.compile`` alike, and handle autograd themselves.
    """

    @torch.library.custom_op(f"physicsnemo::{name}", mutates_args=())
    def op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kernel_size: int,
        dilation: int,
        is_causal: bool,
        scale: Optional[float],
        backend: Optional[str],
    ) -> torch.Tensor:
        return getattr(_natten.functional, name)(
            q,
            k,
            v,
            kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            scale=scale,
            backend=backend,
        )

    @op.register_fake
    def _(q, k, v, kernel_size, dilation, is_causal, scale, backend):
        return torch.empty_like(q)

    return op._opoverload


_NA_OPS = {name: _register_na_op(name) for name in ("na1d", "na2d", "na3d")}


def _na(name, q, k, v, kernel_size, dilation, kwargs):
    r"""Call NATTEN on plain tensors; route tensor subclasses through the op."""
    if all(type(t) in (torch.Tensor, torch.nn.Parameter) for t in (q, k, v)):
        return getattr(_natten.functional, name)(
            q, k, v, kernel_size, dilation=dilation, **kwargs
        )
    unsupported = set(kwargs) - set(_OP_OPTIONS)
    if unsupported:
        raise TypeError(
            f"{name} on tensor subclasses supports only {sorted(_OP_OPTIONS)}, "
            f"got {sorted(unsupported)}"
        )
    opts = {**_OP_OPTIONS, **kwargs}
    return _NA_OPS[name](
        q,
        k,
        v,
        kernel_size,
        dilation,
        opts["is_causal"],
        opts["scale"],
        opts["backend"],
    )


def na1d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
    **kwargs: Any,
) -> torch.Tensor:
    r"""Compute 1D neighborhood attention, with tensor-subclass dispatch.

    This is a thin wrapper around :func:`natten.functional.na1d`. Tensor
    subclass operands (e.g. ``ShardTensor``) go through the
    ``physicsnemo::na1d`` op, which subclasses intercept by name; there only
    ``is_causal``, ``scale`` and ``backend`` are accepted as extra options.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape :math:`(B, L, \text{heads}, D)`.
    k : torch.Tensor
        Key tensor of shape :math:`(B, L, \text{heads}, D)`.
    v : torch.Tensor
        Value tensor of shape :math:`(B, L, \text{heads}, D)`.
    kernel_size : int
        Size of the attention kernel window.
    dilation : int, default=1
        Dilation factor for the attention kernel.
    **kwargs : Any
        Additional keyword arguments forwarded to :func:`natten.functional.na1d`
        (e.g. ``is_causal``, ``scale``).

    Returns
    -------
    torch.Tensor
        Output tensor of the same shape as ``q``.
    """
    return _na("na1d", q, k, v, kernel_size, dilation, kwargs)


def na2d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
    **kwargs: Any,
) -> torch.Tensor:
    r"""Compute 2D neighborhood attention, with tensor-subclass dispatch.

    This is a thin wrapper around :func:`natten.functional.na2d`. Tensor
    subclass operands (e.g. ``ShardTensor``) go through the
    ``physicsnemo::na2d`` op, which subclasses intercept by name; there only
    ``is_causal``, ``scale`` and ``backend`` are accepted as extra options.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape :math:`(B, H, W, \text{heads}, D)`.
    k : torch.Tensor
        Key tensor of shape :math:`(B, H, W, \text{heads}, D)`.
    v : torch.Tensor
        Value tensor of shape :math:`(B, H, W, \text{heads}, D)`.
    kernel_size : int
        Size of the attention kernel window.
    dilation : int, default=1
        Dilation factor for the attention kernel.
    **kwargs : Any
        Additional keyword arguments forwarded to :func:`natten.functional.na2d`
        (e.g. ``is_causal``, ``scale``).

    Returns
    -------
    torch.Tensor
        Output tensor of the same shape as ``q``.
    """
    return _na("na2d", q, k, v, kernel_size, dilation, kwargs)


def na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
    **kwargs: Any,
) -> torch.Tensor:
    r"""Compute 3D neighborhood attention, with tensor-subclass dispatch.

    This is a thin wrapper around :func:`natten.functional.na3d`. Tensor
    subclass operands (e.g. ``ShardTensor``) go through the
    ``physicsnemo::na3d`` op, which subclasses intercept by name; there only
    ``is_causal``, ``scale`` and ``backend`` are accepted as extra options.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape :math:`(B, X, Y, Z, \text{heads}, D)`.
    k : torch.Tensor
        Key tensor of shape :math:`(B, X, Y, Z, \text{heads}, D)`.
    v : torch.Tensor
        Value tensor of shape :math:`(B, X, Y, Z, \text{heads}, D)`.
    kernel_size : int
        Size of the attention kernel window.
    dilation : int, default=1
        Dilation factor for the attention kernel.
    **kwargs : Any
        Additional keyword arguments forwarded to :func:`natten.functional.na3d`
        (e.g. ``is_causal``, ``scale``).

    Returns
    -------
    torch.Tensor
        Output tensor of the same shape as ``q``.
    """
    return _na("na3d", q, k, v, kernel_size, dilation, kwargs)


__all__ = ["na1d", "na2d", "na3d"]
