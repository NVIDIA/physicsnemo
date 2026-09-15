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

r"""Variable-size row exchange primitives shared by the ShardTensor row ops.

Both the halo scatter correction (:mod:`halo_scatter`) and the routed gather
(:mod:`index_ops`) move rows between ranks with a functional-collective
``all_to_all`` whose split sizes are decided at run time, fold contributions
with a configurable accumulator, and hand a plain group-name token to
``custom_op`` bodies. Those pieces live here; the routing that decides *which*
rows move stays with each op.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh

__all__ = [
    "funcol_all_to_all_v_rows",
    "get_fp32_scatter_accumulator",
    "resolve_group_name",
    "set_fp32_scatter_accumulator",
]


def _funcol_group_arg(group: object) -> object:
    r"""Return *group* in the form functional collectives accept.

    Parameters
    ----------
    group : DeviceMesh or ProcessGroup or str or None
        Group specification. ``None`` means the default world group, which
        funcol does not accept directly.

    Returns
    -------
    object
        ``(mesh, 0)`` for a mesh, the default group for ``None``, otherwise
        *group* unchanged.
    """
    if isinstance(group, DeviceMesh):
        return (group, 0)
    if group is None:
        return dist.distributed_c10d._get_default_group()
    return group


# Accumulator used when folding ``float32`` scatter contributions (halo
# corrections, routed scatter-add). ``float64`` by default: row folds sum many
# contributions and downstream applications rely on the extra significance.
# ``float32`` trades that for speed and memory on the accumulation buffer.
_FP32_SCATTER_ACCUMULATOR: torch.dtype = torch.float64


def set_fp32_scatter_accumulator(dtype: torch.dtype) -> None:
    r"""Choose the accumulator for ``float32`` scatter folds.

    Parameters
    ----------
    dtype : torch.dtype
        ``torch.float64`` (default; highest significance) or ``torch.float32``
        (accumulate in place; cheaper on large local shards).

    Raises
    ------
    ValueError
        For any other dtype.
    """
    global _FP32_SCATTER_ACCUMULATOR
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"fp32 scatter accumulator must be float32 or float64, got {dtype}"
        )
    _FP32_SCATTER_ACCUMULATOR = dtype


def get_fp32_scatter_accumulator() -> torch.dtype:
    r"""Accumulator currently used for ``float32`` scatter folds.

    Returns
    -------
    torch.dtype
        ``torch.float64`` unless changed by :func:`set_fp32_scatter_accumulator`.
    """
    return _FP32_SCATTER_ACCUMULATOR


def _accumulator_dtype(dtype: torch.dtype) -> torch.dtype:
    r"""Dtype for folding scatter contributions of *dtype*.

    Row folds sum many contributions, so accumulating in the input precision
    loses significance for the smaller float types.

    Parameters
    ----------
    dtype : torch.dtype
        Dtype of the contributions.

    Returns
    -------
    torch.dtype
        :func:`get_fp32_scatter_accumulator` for ``float32``, ``float32`` for
        ``float16`` / ``bfloat16``, otherwise *dtype* itself.
    """
    if dtype == torch.float32:
        return _FP32_SCATTER_ACCUMULATOR
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def resolve_group_name(group: object) -> str:
    r"""Resolve *group* to a c10d group-name string.

    The name is the traceable token a ``custom_op`` can carry in place of a
    ``ProcessGroup``.

    Parameters
    ----------
    group : DeviceMesh or ProcessGroup or str or None
        A 1-D device mesh (its dim-0 group is used), a process group, an
        existing group name, or ``None`` for the default world group.

    Returns
    -------
    str
        The group name; ``""`` denotes the default world group.
    """
    if group is None:
        return ""
    if isinstance(group, str):
        return group
    if isinstance(group, DeviceMesh):
        return group.get_group(0).group_name
    return group.group_name


def funcol_all_to_all_v_rows(
    send_rows: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
    group: object = None,
) -> torch.Tensor:
    r"""AOT-traceable variable-size row ``all_to_all``.

    Parameters
    ----------
    send_rows : torch.Tensor
        Rows to send, shape ``(sum(send_counts), *trailing)``, ordered by
        destination rank.
    send_counts : list[int]
        Number of rows sent to each rank.
    recv_counts : list[int]
        Number of rows received from each rank.
    group : DeviceMesh or ProcessGroup or str or None, optional
        Group to exchange over; ``None`` is the default world group.

    Returns
    -------
    torch.Tensor
        Received rows, shape ``(sum(recv_counts), *trailing)``, ordered by
        source rank.
    """
    trailing = tuple(send_rows.shape[1:])
    row_size = 1
    for d in trailing:
        row_size *= d
    flat_send = send_rows.contiguous().reshape(-1)
    send_flat = [c * row_size for c in send_counts]
    recv_flat = [c * row_size for c in recv_counts]
    total_recv = sum(recv_counts)
    flat_recv = funcol.wait_tensor(
        funcol.all_to_all_single(
            flat_send, recv_flat, send_flat, _funcol_group_arg(group)
        )
    )
    return flat_recv.reshape((total_recv,) + trailing)
