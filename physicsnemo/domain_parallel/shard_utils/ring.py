# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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


from dataclasses import dataclass
from typing import Literal, Union

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh


@dataclass
class RingPassingConfig:
    """Configuration for ring-based communication operations.

    This class encapsulates all parameters needed for ring communication patterns,
    making it easier to pass consistent configurations between functions.

    Attributes:
        mesh_dim (int): Mesh dimension for the ring communication
        tensor_dim (int): Tensor dimension involved in the communication
        mesh_size (int): Size of the mesh for this communication
        communication_method (str): Method for exchanging data ("p2p" or "a2a")
    """

    VALID_COMM_METHODS = ["p2p", "a2a"]
    VALID_RING_DIRECTIONS = ["forward", "backward"]

    mesh_dim: int
    mesh_size: int
    ring_direction: Literal["forward", "backward"] = "forward"

    communication_method: Literal["p2p", "a2a"] = "a2a"

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization.

        Raises:
            ValueError: If invalid communication method is specified
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


def perform_ring_iteration(
    tensor: torch.Tensor,
    mesh: DeviceMesh,
    ring_config: RingPassingConfig,
    recv_shape: Union[torch.Size, None] = None,
) -> torch.Tensor:
    """
    Performs a ring collective communication where all tensors are the same size.

    Tensors are sent to the next rank in the ring, and wrap around from rank N-1 to rank 0.
    This implements a single step of ring communication where each process sends data to
    its neighbor and receives data from its other neighbor.

    Args:
        tensor (torch.Tensor): The tensor to be sent in this ring communication step
        mesh (DeviceMesh): Device mesh that defines the distributed process group
        ring_config (RingPassingConfig): Configuration for the ring communication pattern

    Returns:
        torch.Tensor: The tensor received from the previous rank in the ring
    """

    dtype = tensor.dtype
    device = tensor.device

    # Get process group info
    local_group = mesh.get_group(ring_config.mesh_dim)
    local_rank = mesh.get_local_rank(ring_config.mesh_dim)
    local_size = dist.get_world_size(group=local_group)

    # Point-to-point communication
    local_id_for_send = local_rank + 1 if local_rank < local_size - 1 else 0
    local_id_for_recv = local_rank - 1 if local_rank > 0 else local_size - 1

    if ring_config.ring_direction == "backward":
        # Swap send/recv directions for backward ring
        local_id_for_send, local_id_for_recv = local_id_for_recv, local_id_for_send

    id_for_send = dist.get_global_rank(group=local_group, group_rank=local_id_for_send)
    id_for_recv = dist.get_global_rank(group=local_group, group_rank=local_id_for_recv)

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
