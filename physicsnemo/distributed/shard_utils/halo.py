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

from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from physicsnemo.utils.version_check import check_module_requirements

check_module_requirements("physicsnemo.distributed.shard_tensor")


def halo_unpadding_1d(
    local_tensor: torch.Tensor,
    mesh: DeviceMesh,
    mesh_dim: int,
    tensor_dim: int,
    halo_t: int,
    edge_padding_t: Optional[str] = "zeros",
    edge_padding_s: Optional[int] = 0,
    return_slices: bool = False,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
]:
    """Removes halo padding from a tensor in 1D.

    This is the backward pass of distributed halo padding. Can be chained to remove
    halo padding in multiple dimensions if needed.

    Args:
        local_tensor: Local tensor chunk to unpad
        mesh: Device mesh containing sharding information
        mesh_dim: Mesh dimension for this unpadding operation
        tensor_dim: Tensor dimension to unpad
        halo_t: Size of halo padding to remove (assumed symmetric)
        edge_padding_t: Edge padding type (currently unused)
        edge_padding_s: Edge padding size (only valid with zeros padding, currently unused)
        return_slices: Whether to return removed halo slices

    Returns:
        Unpadded tensor if return_slices=False, otherwise tuple of:
        - Unpadded tensor
        - Tuple of (front slice, end slice) containing removed halos
    """
    # Validate mesh dimension
    if mesh_dim is None:
        if mesh.ndim != 1:
            raise ValueError(
                f"Halo padding requires `dim` for mesh size > 1 (got shape {mesh.shape})"
            )
        mesh_dim = 0

    # Get process group info
    local_group = mesh.get_group(mesh_dim)
    local_rank = mesh.get_local_rank(mesh_dim)
    local_size = dist.get_world_size(group=local_group)

    # Get shape of dimension being unpadded
    dim_shape = local_tensor.shape[tensor_dim]

    # Calculate slice boundaries
    start = halo_t if local_rank != 0 else 0
    end = dim_shape - halo_t if local_rank != local_size - 1 else dim_shape

    if return_slices:
        # Get removed halo slices for non-edge ranks
        front_slice = None
        if local_rank != 0:
            front_indices = torch.arange(0, start).to(local_tensor.device)
            front_slice = local_tensor.index_select(
                tensor_dim, front_indices
            ).contiguous()

        end_slice = None
        if local_rank != local_size - 1:
            end_indices = torch.arange(end, dim_shape).to(local_tensor.device)
            end_slice = local_tensor.index_select(tensor_dim, end_indices).contiguous()

    # Remove halo padding
    indices = torch.arange(start, end).to(local_tensor.device)
    local_tensor = local_tensor.index_select(tensor_dim, indices).contiguous()

    if return_slices:
        return local_tensor, (front_slice, end_slice)

    return local_tensor


def halo_padding_1d(
    local_tensor: torch.Tensor,
    mesh: DeviceMesh,
    mesh_dim: int,
    tensor_dim: int,
    halo_t: int,
    edge_padding_t: Optional[str] = "zeros",
    edge_padding_s: Optional[int] = 0,
) -> torch.Tensor:  # pragma: no cover
    """Adds halo padding to a tensor in 1D.

    This is the forward pass of distributed halo padding. Can be chained to add
    halo padding in multiple dimensions if needed.

    Args:
        local_tensor: Local tensor chunk to pad
        mesh: Device mesh containing sharding information
        mesh_dim: Mesh dimension for this padding operation
        tensor_dim: Tensor dimension to pad
        halo_t: Size of halo padding to add (assumed symmetric)
        edge_padding_t: Edge padding type (zeros, reflect, replicate, circular, none)
        edge_padding_s: Edge padding size (only valid with zeros padding)

    Returns:
        Padded tensor with halos added locally to each chunk

    Note:
        Coalescing the padded tensor directly without consuming the halo will produce
        invalid results.
    """
    valid_padding = ["zeros", "reflect", "replicate", "circular", "none"]
    if edge_padding_t not in valid_padding:
        raise ValueError(f"Invalid edge padding: {edge_padding_t}")

    if edge_padding_s != 0 and edge_padding_t != "zeros":
        raise NotImplementedError(
            f"Edge padding size != 0 only supported with zeros padding "
            f"(got size={edge_padding_s}, type={edge_padding_t})"
        )

    # Validate mesh dimension
    if mesh_dim is None:
        if mesh.ndim != 1:
            raise ValueError(
                f"Halo padding requires `dim` for mesh size > 1 (got shape {mesh.shape})"
            )

    # Select halo regions to exchange
    left_indices = torch.arange(0, halo_t).to(local_tensor.device)
    max_index = local_tensor.shape[tensor_dim]
    right_indices = max_index - 1 - left_indices
    right_indices = torch.flip(right_indices, (0,))

    halo_to_left = local_tensor.index_select(tensor_dim, left_indices).contiguous()
    halo_to_right = local_tensor.index_select(tensor_dim, right_indices).contiguous()

    # Exchange halos between ranks
    halo_from_left, halo_from_right = perform_halo_collective(
        mesh, mesh_dim, halo_to_left, halo_to_right
    )

    # Combine local tensor with received halos
    padded_output = unpack_halo_tensors(
        mesh,
        mesh_dim,
        tensor_dim,
        halo_from_left,
        halo_from_right,
        local_tensor,
        edge_padding_s,
        edge_padding_t,
    )

    return torch.cat(padded_output, dim=tensor_dim)


def perform_halo_collective(
    mesh: DeviceMesh,
    mesh_dim: int,
    halo_to_left: torch.Tensor,
    halo_to_right: torch.Tensor,
    method: str = "a2a",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Performs collective communication to exchange halo regions between ranks.

    Args:
        mesh: Device mesh for communication
        mesh_dim: Mesh dimension for exchange
        halo_to_left: Halo tensor to send left
        halo_to_right: Halo tensor to send right
        method: Communication method ("p2p" or "a2a")

    Returns:
        Tuple of (halo from left, halo from right) tensors
    """
    template_halo = next(
        (x for x in [halo_to_left, halo_to_right] if x is not None), None
    )
    if template_halo is None:
        raise ValueError(
            "At least one of halo_to_left or halo_to_right must not be None"
        )

    # Get process group info
    local_group = mesh.get_group(mesh_dim)
    local_rank = mesh.get_local_rank(mesh_dim)
    local_size = dist.get_world_size(group=local_group)

    if method == "p2p":
        # Point-to-point communication
        id_of_right = local_rank + 1 if local_rank < local_size - 1 else None
        id_of_left = local_rank - 1 if local_rank > 0 else None

        halo_from_right = torch.empty_like(template_halo)
        halo_from_left = torch.empty_like(template_halo)

        p2p_op_list = []
        torch.cuda.set_device(template_halo.device)

        # Post receives
        if id_of_right is not None:
            p2p_op_list.append(
                dist.P2POp(
                    op=dist.irecv,
                    tensor=halo_from_right,
                    peer=id_of_right,
                    group=local_group,
                )
            )

        if id_of_left is not None:
            p2p_op_list.append(
                dist.P2POp(
                    op=dist.irecv,
                    tensor=halo_from_left,
                    peer=id_of_left,
                    group=local_group,
                )
            )

        # Post sends
        if id_of_left is not None:
            p2p_op_list.append(
                dist.P2POp(
                    op=dist.isend,
                    tensor=halo_to_left,
                    peer=id_of_left,
                    group=local_group,
                )
            )

        if id_of_right is not None:
            p2p_op_list.append(
                dist.P2POp(
                    op=dist.isend,
                    tensor=halo_to_right,
                    peer=id_of_right,
                    group=local_group,
                )
            )

        # Ensure all communication completes
        if len(p2p_op_list) > 0:
            reqs = dist.batch_isend_irecv(p2p_op_list)
            for req in reqs:
                req.wait()

    elif method == "a2a":
        # All-to-all communication
        all_to_all_send = [
            torch.empty(0, dtype=template_halo.dtype, device=template_halo.device)
            for _ in range(local_size)
        ]
        all_to_all_recv = [
            torch.empty(0, dtype=template_halo.dtype, device=template_halo.device)
            for _ in range(local_size)
        ]

        # Set up send/recv buffers
        if local_rank != 0:
            # Send one left
            all_to_all_send[local_rank - 1] = halo_to_left
            # Receive one right (need to initialize an empty buffer of the right size):
            all_to_all_recv[local_rank - 1] = torch.zeros_like(
                template_halo
            ).contiguous()

        if local_rank != local_size - 1:
            # Send one to the right:
            all_to_all_send[local_rank + 1] = halo_to_right
            # Receive one from the right:
            all_to_all_recv[local_rank + 1] = torch.zeros_like(
                template_halo
            ).contiguous()

        # Perform exchange
        dist.all_to_all(all_to_all_recv, all_to_all_send, group=local_group)

        # Extract received halos
        halo_from_left = all_to_all_recv[local_rank - 1] if local_rank != 0 else None
        halo_from_right = (
            all_to_all_recv[local_rank + 1] if local_rank != local_size - 1 else None
        )

    return halo_from_left, halo_from_right


def unpack_halo_tensors(
    mesh: DeviceMesh,
    mesh_dim: int,
    target_dim: int,
    halo_from_left: Optional[torch.Tensor],
    halo_from_right: Optional[torch.Tensor],
    local_tensor: torch.Tensor,
    edge_padding_s: Optional[int],
    edge_padding_t: str,
) -> List[torch.Tensor]:
    """Combines local tensor with received halos and edge padding.

    Args:
        mesh: Device mesh for process info
        mesh_dim: Mesh dimension being padded
        target_dim: Tensor dimension being padded
        halo_from_left: Halo received from left rank
        halo_from_right: Halo received from right rank
        local_tensor: Local tensor chunk
        edge_padding_s: Edge padding size
        edge_padding_t: Edge padding type

    Returns:
        List of tensors to concatenate for final padded result
    """
    # Get process group info
    local_group = mesh.get_group(mesh_dim)
    local_rank = mesh.get_local_rank(mesh_dim)
    local_size = dist.get_world_size(group=local_group)

    padded_output = []

    # Add left padding
    if local_rank != 0:
        padded_output.append(halo_from_left)
    else:
        if edge_padding_t == "zeros":
            if edge_padding_s is None:
                padded_output.append(torch.zeros_like(halo_from_right))
            else:
                shape = list(halo_from_right.shape)
                shape[target_dim] = edge_padding_s
                zeros = torch.zeros(
                    shape, device=halo_from_right.device, dtype=halo_from_right.dtype
                )
                padded_output.append(zeros)
        elif edge_padding_t == "reflect":
            padded_output.append(halo_from_right.flip(target_dim))
        elif edge_padding_t == "replicate":
            raise NotImplementedError("Replicate padding not implemented")
        elif edge_padding_t == "circular":
            padded_output.append(halo_from_right)
        elif edge_padding_t == "none":
            pass

    # Add local tensor
    padded_output.append(local_tensor)

    # Add right padding
    if local_rank != local_size - 1:
        padded_output.append(halo_from_right)
    else:
        if edge_padding_t == "zeros":
            if edge_padding_s is None:
                padded_output.append(torch.zeros_like(halo_from_left))
            else:
                shape = list(halo_from_left.shape)
                shape[target_dim] = edge_padding_s
                zeros = torch.zeros(
                    shape, device=halo_from_left.device, dtype=halo_from_left.dtype
                )
                padded_output.append(zeros)
        elif edge_padding_t == "reflect":
            padded_output = halo_from_left.flip(target_dim)
        elif edge_padding_t == "replicate":
            raise NotImplementedError("Replicate padding not implemented")
        elif edge_padding_t == "circular":
            padded_output.append(halo_from_left)
        elif edge_padding_t == "none":
            pass

    return padded_output


def apply_grad_halo(
    mesh: DeviceMesh,
    mesh_dim: int,
    tensor_dim: int,
    grad_input: torch.Tensor,
    halo_from_left: torch.Tensor,
    halo_from_right: torch.Tensor,
) -> torch.Tensor:
    """Applies halo gradients to input gradient tensor.

    Args:
        mesh: Device mesh for process info
        mesh_dim: Mesh dimension for halo
        tensor_dim: Tensor dimension for halo
        grad_input: Input gradient tensor
        halo_from_left: Gradient from left halo
        halo_from_right: Gradient from right halo

    Returns:
        Updated gradient tensor with halo gradients applied
    """
    # Get process group info
    local_group = mesh.get_group(mesh_dim)
    local_rank = mesh.get_local_rank(mesh_dim)
    local_size = dist.get_world_size(group=local_group)

    # Apply right halo gradient
    if local_rank != local_size - 1:
        start_idx = grad_input.shape[tensor_dim] - halo_from_right.shape[tensor_dim]
        length = halo_from_right.shape[tensor_dim]
        grad_input.narrow(tensor_dim, start_idx, length).add_(halo_from_right)

    # Apply left halo gradient
    if local_rank != 0:
        start_idx = 0
        length = halo_from_left.shape[tensor_dim]
        grad_input.narrow(tensor_dim, start_idx, length).add_(halo_from_left)

    return grad_input
