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

from typing import Any

import torch
from torch.overrides import handle_torch_function, has_torch_function

from physicsnemo.core.version_check import OptionalImport

_natten = OptionalImport("natten")


def _subclass_under_compile(*tensors: torch.Tensor) -> bool:
    r"""True while Dynamo traces a call with a tensor-subclass operand.

    Dynamo folds ``has_torch_function`` to ``False`` for traceable wrapper
    subclasses such as ``ShardTensor``, so the check above alone would hand the
    subclass to the raw kernel. ``type()`` still reports the subclass.
    """
    return torch.compiler.is_compiling() and any(
        type(t) is not torch.Tensor for t in tensors
    )


@torch._dynamo.allow_in_graph
def _dispatch_in_graph(
    name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    dilation: int,
    kwargs: dict[str, Any],
) -> torch.Tensor:
    r"""Subclass dispatch as one opaque graph node.

    Dynamo records the call and runs it under fake propagation, where
    ``__torch_function__`` dispatches to the subclass handler (for
    ``ShardTensor``, the halo-exchange path), as for ``torch`` ops.
    """
    func = _NA_FUNCS[name]
    return handle_torch_function(
        func, (q, k, v), q, k, v, kernel_size, dilation=dilation, **kwargs
    )


def na1d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
    **kwargs: Any,
) -> torch.Tensor:
    r"""Compute 1D neighborhood attention, with ``__torch_function__`` dispatch.

    This is a thin wrapper around :func:`natten.functional.na1d` that enables
    automatic dispatch through PyTorch's ``__torch_function__`` protocol. When
    called with a tensor subclass (e.g. ``ShardTensor``), the registered handler
    is invoked instead of the underlying natten implementation.

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
    if has_torch_function((q, k, v)):
        return handle_torch_function(
            na1d,
            (q, k, v),
            q,
            k,
            v,
            kernel_size,
            dilation=dilation,
            **kwargs,
        )
    if _subclass_under_compile(q, k, v):
        return _dispatch_in_graph("na1d", q, k, v, kernel_size, dilation, kwargs)
    return _natten.functional.na1d(q, k, v, kernel_size, dilation=dilation, **kwargs)


def na2d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
    **kwargs: Any,
) -> torch.Tensor:
    r"""Compute 2D neighborhood attention, with ``__torch_function__`` dispatch.

    This is a thin wrapper around :func:`natten.functional.na2d` that enables
    automatic dispatch through PyTorch's ``__torch_function__`` protocol. When
    called with a tensor subclass (e.g. ``ShardTensor``), the registered handler
    is invoked instead of the underlying natten implementation.

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
    if has_torch_function((q, k, v)):
        return handle_torch_function(
            na2d,
            (q, k, v),
            q,
            k,
            v,
            kernel_size,
            dilation=dilation,
            **kwargs,
        )
    if _subclass_under_compile(q, k, v):
        return _dispatch_in_graph("na2d", q, k, v, kernel_size, dilation, kwargs)
    return _natten.functional.na2d(q, k, v, kernel_size, dilation=dilation, **kwargs)


def na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
    **kwargs: Any,
) -> torch.Tensor:
    r"""Compute 3D neighborhood attention, with ``__torch_function__`` dispatch.

    This is a thin wrapper around :func:`natten.functional.na3d` that enables
    automatic dispatch through PyTorch's ``__torch_function__`` protocol. When
    called with a tensor subclass (e.g. ``ShardTensor``), the registered handler
    is invoked instead of the underlying natten implementation.

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
    if has_torch_function((q, k, v)):
        return handle_torch_function(
            na3d,
            (q, k, v),
            q,
            k,
            v,
            kernel_size,
            dilation=dilation,
            **kwargs,
        )
    if _subclass_under_compile(q, k, v):
        return _dispatch_in_graph("na3d", q, k, v, kernel_size, dilation, kwargs)
    return _natten.functional.na3d(q, k, v, kernel_size, dilation=dilation, **kwargs)


_NA_FUNCS = {"na1d": na1d, "na2d": na2d, "na3d": na3d}

__all__ = ["na1d", "na2d", "na3d"]
