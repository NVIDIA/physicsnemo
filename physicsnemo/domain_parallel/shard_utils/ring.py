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


r"""Ring communication utilities for distributed tensor operations.

This module provides utilities for ring-based collective communication patterns.
Ring communication is useful for operations where data needs to be passed around
in a circular fashion between processes, such as ring attention.

The module provides:

- ``RingPassingConfig``: Configuration dataclass for ring communication parameters
- ``perform_ring_iteration_funcol``: Ring step via functional collectives; the
  preferred primitive. Overlap comes from issuing the shift before the local
  compute and calling ``finish_ring_iteration`` after it -- no caller-managed
  streams or work handles.
- ``finish_ring_iteration``: Synchronization point for an in-flight funcol step
- ``perform_ring_iteration``: Blocking single step over raw ``dist.*`` p2p/a2a
  collectives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh


@dataclass
class RingPassingConfig:
    r"""Configuration for ring-based communication operations.

    This class encapsulates all parameters needed for ring communication patterns,
    making it easier to pass consistent configurations between functions.

    Attributes
    ----------
    mesh_dim : int
        Mesh dimension for the ring communication.
    mesh_size : int
        Size of the mesh for this communication.
    ring_direction : Literal["forward", "backward"]
        Direction of ring communication. ``"forward"`` sends to rank+1,
        ``"backward"`` sends to rank-1. Default is ``"forward"``.
    communication_method : Literal["p2p", "a2a"]
        Method for exchanging data. ``"p2p"`` uses point-to-point operations,
        ``"a2a"`` uses all-to-all. Default is ``"a2a"``.
    """

    VALID_COMM_METHODS = ["p2p", "a2a"]
    VALID_RING_DIRECTIONS = ["forward", "backward"]

    mesh_dim: int
    mesh_size: int
    ring_direction: Literal["forward", "backward"] = "forward"

    communication_method: Literal["p2p", "a2a"] = "a2a"

    def __post_init__(self) -> None:
        r"""Validate configuration parameters after initialization.

        Raises
        ------
        ValueError
            If invalid communication method or ring direction is specified.
        """

        if self.communication_method not in self.VALID_COMM_METHODS:
            raise ValueError(
                f"Invalid communication method: {self.communication_method}. "
                f"Must be one of {self.VALID_COMM_METHODS}"
            )

        if self.ring_direction not in self.VALID_RING_DIRECTIONS:
            raise ValueError(
                f"Invalid ring direction: {self.ring_direction}. "
                f"Must be one of {self.VALID_RING_DIRECTIONS}"
            )


def _get_ring_comm_ranks(
    mesh: DeviceMesh,
    ring_config: RingPassingConfig,
) -> tuple[dist.ProcessGroup, int, int, int, int, int]:
    r"""Compute the send/recv neighbor ranks for a ring communication step.

    Parameters
    ----------
    mesh : DeviceMesh
        Device mesh that defines the distributed process group.
    ring_config : RingPassingConfig
        Configuration for the ring communication pattern.

    Returns
    -------
    tuple[dist.ProcessGroup, int, int, int, int, int]
        Tuple of (local_group, local_size, local_id_for_send, local_id_for_recv,
        global_id_for_send, global_id_for_recv).
    """
    local_group = mesh.get_group(ring_config.mesh_dim)
    local_rank = mesh.get_local_rank(ring_config.mesh_dim)
    local_size = dist.get_world_size(group=local_group)

    local_id_for_send = local_rank + 1 if local_rank < local_size - 1 else 0
    local_id_for_recv = local_rank - 1 if local_rank > 0 else local_size - 1

    if ring_config.ring_direction == "backward":
        local_id_for_send, local_id_for_recv = local_id_for_recv, local_id_for_send

    id_for_send = dist.get_global_rank(group=local_group, group_rank=local_id_for_send)
    id_for_recv = dist.get_global_rank(group=local_group, group_rank=local_id_for_recv)

    return (
        local_group,
        local_size,
        local_id_for_send,
        local_id_for_recv,
        id_for_send,
        id_for_recv,
    )


def perform_ring_iteration(
    tensor: torch.Tensor,
    mesh: DeviceMesh,
    ring_config: RingPassingConfig,
    recv_shape: torch.Size | None = None,
) -> torch.Tensor:
    r"""Perform a single step of ring collective communication.

    Tensors are sent to the next rank in the ring, and wrap around from rank N-1
    to rank 0. This implements a single step of ring communication where each
    process sends data to its neighbor and receives data from its other neighbor.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to be sent in this ring communication step.
    mesh : DeviceMesh
        Device mesh that defines the distributed process group.
    ring_config : RingPassingConfig
        Configuration for the ring communication pattern.
    recv_shape : Union[torch.Size, None], optional
        Shape of the tensor to receive. If ``None``, assumes same shape as
        the tensor being sent.

    Returns
    -------
    torch.Tensor
        The tensor received from the previous rank in the ring.
    """

    dtype = tensor.dtype
    device = tensor.device

    (
        local_group,
        local_size,
        local_id_for_send,
        local_id_for_recv,
        id_for_send,
        id_for_recv,
    ) = _get_ring_comm_ranks(mesh, ring_config)

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    if recv_shape is None:
        tensor_recv = torch.empty_like(tensor)
    else:
        tensor_recv = torch.empty(recv_shape, dtype=dtype, device=device)

    if ring_config.communication_method == "p2p":
        p2p_op_list = []
        torch.cuda.set_device(tensor.device)

        # Post receive
        p2p_op_list.append(
            dist.P2POp(
                op=dist.irecv,
                tensor=tensor_recv,
                peer=id_for_recv,
                group=local_group,
            )
        )

        # Post sends
        p2p_op_list.append(
            dist.P2POp(
                op=dist.isend,
                tensor=tensor,
                peer=id_for_send,
                group=local_group,
            )
        )

        # Ensure all communication completes
        reqs = dist.batch_isend_irecv(p2p_op_list)
        for req in reqs:
            req.wait()

    elif ring_config.communication_method == "a2a":
        # All-to-all communication
        all_to_all_send = [
            torch.empty(0, dtype=dtype, device=device) for _ in range(local_size)
        ]
        all_to_all_recv = [
            torch.empty(0, dtype=dtype, device=device) for _ in range(local_size)
        ]

        # Use local ranks as indices into the arrays, not global ranks
        all_to_all_recv[local_id_for_recv] = tensor_recv
        all_to_all_send[local_id_for_send] = tensor

        # Perform exchange
        dist.all_to_all(all_to_all_recv, all_to_all_send, group=local_group)

    return tensor_recv


def perform_ring_iteration_funcol(
    tensor: torch.Tensor,
    mesh: DeviceMesh,
    ring_config: RingPassingConfig,
    recv_shape: torch.Size | None = None,
    wait: bool = True,
) -> torch.Tensor:
    r"""Single ring step via functional collectives -- no explicit streams or work handles.
    CUDA streams give great overlap but cannot be compiled.  ``funcol`` can
    be compiled, so it's the preferred path here.

    The shift is an ``all_to_all_single`` whose split sizes are zero everywhere
    except the send/recv neighbor slots. Functional collectives are issued
    asynchronously on the communicator's internal stream, so overlap with
    compute comes from *when the result is waited on*, not from caller-managed
    CUDA streams: issue the shift before the local compute and wait after it.

    ``ring_config.communication_method`` is not consulted: the functional
    collective is always an all-to-all with two nonzero splits. (There is
    no P2P equivalent)

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to send to the next rank in the ring.
    mesh : DeviceMesh
        Device mesh that defines the distributed process group.
    ring_config : RingPassingConfig
        Configuration for the ring communication pattern.
    recv_shape : torch.Size | None, optional
        Shape of the incoming tensor (shards may be uneven). If ``None``,
        assumes the same shape as ``tensor``.
    wait : bool, optional
        If ``True`` (default), wait and return the reshaped received tensor --
        a drop-in replacement for ``perform_ring_iteration``. If ``False``,
        return the *flat* unwaited tensor; complete it later with
        ``finish_ring_iteration``.

    Returns
    -------
    torch.Tensor
        The received tensor (``wait=True``), or the flat in-flight tensor
        (``wait=False``).
    """
    local_group, local_size, local_id_for_send, local_id_for_recv, _, _ = (
        _get_ring_comm_ranks(mesh, ring_config)
    )

    if recv_shape is None:
        recv_shape = tensor.shape

    input_split_sizes = [0] * local_size
    output_split_sizes = [0] * local_size
    input_split_sizes[local_id_for_send] = tensor.numel()
    output_split_sizes[local_id_for_recv] = math.prod(recv_shape)

    flat_recv = funcol.all_to_all_single(
        tensor.contiguous().reshape(-1),
        output_split_sizes,
        input_split_sizes,
        local_group,
    )

    if wait:
        return finish_ring_iteration(flat_recv, recv_shape)
    return flat_recv


def finish_ring_iteration(
    flat_recv: torch.Tensor,
    recv_shape: torch.Size,
) -> torch.Tensor:
    r"""Complete an in-flight ring step started with ``wait=False``.

    Waits on the functional collective and reshapes the flat buffer to
    ``recv_shape``. This is the synchronization point: place it after the
    compute that should overlap the communication.
    """
    return funcol.wait_tensor(flat_recv).reshape(recv_shape)
