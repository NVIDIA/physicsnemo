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


from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.distributed as dist
from torch.autograd.profiler import record_function
from torch.distributed.device_mesh import DeviceMesh

from physicsnemo.utils.profiling import profile

"""Halo exchange utilities for distributed tensor operations.

This module provides functionality for halo padding operations in distributed computing
environments. Halo padding is a technique used in distributed tensor operations where
each process needs access to a small region of data (the "halo") from neighboring
processes to perform local computations correctly.

The module includes:
- HaloConfig: Configuration class for halo exchange parameters
- Autograd-compatible functions for forward and backward passes
- Primitives for halo exchange operations
- Utility functions for slicing and applying halo regions

"""


@dataclass
class HaloConfig:
    """Configuration for halo padding operations.

    This class encapsulates all parameters needed for halo exchange operations,
    making it easier to pass consistent configurations between functions.

    Attributes:
        mesh_dim (int): Mesh dimension for this padding operation
        tensor_dim (int): Tensor dimension to pad/unpad
        halo_size (int): Size of halo padding (assumed symmetric)
        edge_padding_size (int): Edge padding size (puts 0s on the edge tensors)
        communication_method (str): Method for exchanging halos ("p2p" or "a2a")
    """

    mesh_dim: int
    tensor_dim: int
    halo_size: int
    edge_padding_size: int = 0
    async_op: bool = False

    CommMethod = Literal["p2p", "a2a"]
    VALID_COMM_METHODS = ["p2p", "a2a"]
    communication_method: CommMethod = "a2a"

    def __post_init__(self):
        """Validate configuration parameters after initialization.

        Raises:
            ValueError: If invalid padding type or communication method is specified
        """

        if self.communication_method not in self.VALID_COMM_METHODS:
            raise ValueError(
                f"Invalid communication method: {self.communication_method}. "
                f"Must be one of {self.VALID_COMM_METHODS}"
            )

        if self.async_op and self.communication_method == "p2p":
            raise ValueError(
                "Async halo padding is not supported with p2p communication. "
                "Must be a2a."
            )


@profile
def halo_padding(
    tensor: torch.Tensor,
    mesh: DeviceMesh,
    halo_config: HaloConfig,
) -> torch.Tensor:
    """High-level, differentiable function to apply halo padding with gradient support.

    Args:
        tensor: torch.Tensor to apply halo padding to
        mesh: DeviceMesh containing device information that the halo is performed on
        halo_config: Configuration object containing all halo parameters

    Returns:
        Padded tensor with halos added locally to each chunk.  This is *not* a ShardTensor - it
        is a torch.Tensor that has had local edges replicated from neighboring ranks.
    """

    return HaloPadding.apply(tensor, mesh, halo_config)


@profile
def unhalo_padding(
    tensor: torch.Tensor,
    mesh: DeviceMesh,
    halo_config: HaloConfig,
) -> torch.Tensor:
    """High-level, differentiable function to apply unhalo padding with gradient support.

    This function removes halo regions from a tensor according to the provided configuration.
    It is the inverse operation of halo_padding and maintains differentiability for gradients.

    Args:
        tensor: Padded tensor with halos to be removed
        mesh: DeviceMesh containing device information for the operation
        halo_config: Configuration object containing all halo parameters

    Returns:
        Tensor with halo regions removed according to the configuration
    """

    return UnHaloPadding.apply(tensor, mesh, halo_config)


class HaloPadding(torch.autograd.Function):
    """Autograd Function for applying and removing halo padding.

    This class handles the forward and backward passes for halo padding operations,
    maintaining proper gradient flow between distributed tensors.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        config: HaloConfig,
    ) -> torch.Tensor:
        """Add halo padding to a tensor in the forward pass.

        Args:
            ctx: Autograd context for saving tensors/variables for backward
            tensor: torch.Tensor to apply halo padding to
            mesh: DeviceMesh containing device information that the halo is performed on
            config: HaloConfig defining padding parameters

        Returns:
            Padded tensor with halos added locally to each chunk
        """

        # Save context for backward pass
        ctx.mesh = mesh
        ctx.config = config

        padded_tensor = halo_padding_fwd_primitive(
            tensor,
            mesh,
            config,
        )
        return padded_tensor

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None]:
        """Handle gradients by removing halo padding and applying halo gradients.

        Args:
            ctx: Autograd context from forward pass
            grad_output: Gradient tensor with halo padding

        Returns:
            Tuple of (gradient for input Tensor, None)
        """
        mesh = ctx.mesh
        config = ctx.config

        grad_input = halo_padding_bwd_primitive(
            grad_output,
            mesh,
            config,
        )

        return grad_input, None, None


class UnHaloPadding(torch.autograd.Function):
    """Autograd Function for removing halo padding with gradient support.

    This class implements the forward and backward passes for unhalo padding operations.
    In the forward pass, it removes halo regions from the input tensor according to the
    configuration. In the backward pass, it adds zero padding in the halo regions to
    maintain the correct shape for gradient propagation.

    This is the inverse operation of HaloPadding and maintains differentiability.
    """

    @staticmethod
    def forward(
        ctx, tensor: torch.Tensor, mesh: DeviceMesh, config: HaloConfig
    ) -> torch.Tensor:
        """
        Forward pass for unhalo padding.

        Conceptually, this is a truncated version of the bwd pass of halo padding.
        It is actually collective-free in the forward pass, since we just cut pieces off.

        We still require the mesh to save it for the backward pass.

        Args:
            ctx: Autograd context for saving tensors/variables for backward
            tensor: torch.Tensor to apply halo padding to
            mesh: DeviceMesh containing device information that the halo is performed on
            config: HaloConfig defining padding parameters
        """

        # Save context for backward pass
        ctx.mesh = mesh
        ctx.config = config

        # Chop off the halos
        _left, unpadded_tensor, _right = slice_halo_regions(
            tensor,
            mesh,
            config,
        )

        ctx.left_shape = _left.shape
        ctx.right_shape = _right.shape

        return unpadded_tensor

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, None]:
        """
        Backward pass for unhalo padding.

        In the backward pass, we need to add zero tensors where we previously
        removed the halo regions in the forward pass. This effectively pads
        the gradient with zeros in the halo regions.

        Args:
            ctx: Autograd context containing saved tensors/variables from forward
            grad_output: Gradient of the loss with respect to the output of forward

        Returns:
            Tuple containing:
            - Gradient with respect to the input tensor
            - None for mesh parameter (not differentiable)
            - None for config parameter (not differentiable)
        """

        left_zeros = torch.zeros(
            ctx.left_shape, device=grad_output.device, dtype=grad_output.dtype
        )
        right_zeros = torch.zeros(
            ctx.right_shape, device=grad_output.device, dtype=grad_output.dtype
        )

        grad_input = apply_halo_tensors(
            ctx.mesh,
            ctx.config,
            grad_output,
            left_zeros,
            right_zeros,
        )

        return grad_input, None, None


@profile
def halo_padding_fwd_primitive(
    local_tensor: torch.Tensor,
    mesh: DeviceMesh,
    halo_config: HaloConfig,
) -> torch.Tensor:
    """
    Forward primitive for halo padding.

    Halo padding is meant for operations
    that are applying a localized function (like convolution, but need not be conv)
    to a spatially sharded tensor.  During the forward pass, the inputs from the
    neighboring tensors are copied from remote regions and appended to the local image.

    Args:
        local_tensor: The local tensor chunk to pad with halos
        mesh: Device mesh containing sharding information
        halo_config: HaloConfig defining padding parameters

    Returns:
        Padded tensor with halos from neighboring ranks
    """
    # It's not optimized, but we pull of the halo from both sides currently.  One
    # gets discarded on the edge ranks.  But, it would have to wait
    # for the other ranks to make this selection anyways.

    # Select halo regions to exchange
    left_indices = torch.arange(0, halo_config.halo_size, device=local_tensor.device)
    max_index = local_tensor.shape[halo_config.tensor_dim]
    right_indices = max_index - 1 - left_indices
    right_indices = torch.flip(right_indices, (0,))

    # Collectives need contiguous data.  So we enforce that here.
    halo_to_left = local_tensor.index_select(
        halo_config.tensor_dim, left_indices
    ).contiguous()
    halo_to_right = local_tensor.index_select(
        halo_config.tensor_dim, right_indices
    ).contiguous()

    # Exchange halos between ranks
    halo_from_left, halo_from_right = perform_halo_collective(
        mesh,
        halo_config.mesh_dim,
        halo_to_left,
        halo_to_right,
        halo_config.communication_method,
        halo_config.async_op,
    )

    # Combine local tensor with received halos
    padded_output = apply_halo_tensors(
        mesh,
        halo_config,
        local_tensor,
        halo_from_left,
        halo_from_right,
    )

    return padded_output


@profile
def slice_halo_regions(
    local_tensor: torch.Tensor,
    mesh: DeviceMesh,
    halo_config: HaloConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splits a tensor into left halo, center, and right halo regions.

    This primitive function divides the input tensor along the specified dimension into
    three parts: left halo region, central tensor (without halos), and right halo region.
    The slicing boundaries are determined based on the rank in the mesh and the halo configuration.

    "left" and "right" do not necessarily correspond to spatial locations, but instead
    think of "left" as the region closer to rank 0 and "right" closer to rank N-1.

    Args:
        local_tensor: Input tensor to be sliced
        mesh: DeviceMesh containing device information
        halo_config: Configuration defining halo parameters and dimensions

    Returns:
        Tuple of (left_slice, central_slice, right_slice) tensors
    """

    # Get process group info
    local_group = mesh.get_group(halo_config.mesh_dim)
    local_rank = mesh.get_local_rank(halo_config.mesh_dim)
    local_size = dist.get_world_size(group=local_group)

    # Get shape of dimension being unpadded
    dim_shape = local_tensor.shape[halo_config.tensor_dim]

    # Calculate slice boundaries
    start = halo_config.halo_size if local_rank != 0 else halo_config.edge_padding_size
    end = (
        dim_shape - halo_config.halo_size
        if local_rank != local_size - 1
        else dim_shape - halo_config.edge_padding_size
    )

    left_slice, central_slice, right_slice = torch.tensor_split(
        local_tensor, [start, end], dim=halo_config.tensor_dim
    )

    return left_slice, central_slice, right_slice


@profile
def halo_padding_bwd_primitive(
    grad_output: torch.Tensor,
    mesh: DeviceMesh,
    halo_config: HaloConfig,
) -> torch.Tensor:
    """Backward primitive for halo padding.

    Recall the forward pass is a concatenation of neighboring regions.
    The backward pass takes the gradients of the padded images, slices
    off the pieces that represent the halos, performs a halo collective,
    and *adds* the gradients to their original positions in the local grads.

    Args:
        grad_output: Gradient tensor from upstream operations
        mesh: Device mesh containing sharding information
        halo_config: HaloConfig defining padding parameters

    Returns:
        Gradient tensor with halo contributions applied
    """

    grad_to_left, local_grad, grad_to_right = slice_halo_regions(
        grad_output,
        mesh,
        halo_config,
    )

    # Exchange halos between ranks
    grad_from_left, grad_from_right = perform_halo_collective(
        mesh,
        halo_config.mesh_dim,
        grad_to_left.contiguous(),
        grad_to_right.contiguous(),
        halo_config.communication_method,
    )

    # Apply halo gradients
    final_grad_local = apply_grad_halo(
        mesh,
        halo_config,
        local_grad,
        grad_from_left,
        grad_from_right,
    )

    return final_grad_local


@profile
def perform_halo_collective(
    mesh: DeviceMesh,
    mesh_dim: int,
    halo_to_left: torch.Tensor,
    halo_to_right: torch.Tensor,
    method: Literal["p2p", "a2a"] = "a2a",
    async_op: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Performs collective communication to exchange halo regions between ranks.

    There is an assumption made here that messages are symmetric between paired
    processes in terms of message size.  So, size(message_to_right) == size(message_from_left).
    This assumption is used when preparing buffers for the incoming messages.

    If messages aren't being sent in one direction, it's expected to take
    in an empty tensor with the proper device and dtype still.

    Args:
        mesh: Device mesh for communication
        mesh_dim: Mesh dimension for exchange
        halo_to_left: Halo tensor to send left
        halo_to_right: Halo tensor to send right
        method: Communication method ("p2p" or "a2a")

    Returns:
        Tuple of (halo from left, halo from right) tensors
    """

    # We get the dtype and device from the first non-None tensor
    # Do not use this as the generalized template - we don't assume a
    # rank is sending equal amounts of data left and right.  Only
    # assume the messages are symmetric between
    template_halo = next(
        (x for x in [halo_to_left, halo_to_right] if x is not None), None
    )

    dtype = template_halo.dtype
    device = template_halo.device

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
            torch.empty(0, dtype=dtype, device=device) for _ in range(local_size)
        ]
        all_to_all_recv = [
            torch.empty(0, dtype=dtype, device=device) for _ in range(local_size)
        ]

        # Set up send/recv buffers
        if local_rank != 0:
            # Send one left
            all_to_all_send[local_rank - 1] = halo_to_left
            # Receive one right (need to initialize an empty buffer of the right size):
            all_to_all_recv[local_rank - 1] = torch.zeros_like(
                halo_to_left
            ).contiguous()

        if local_rank != local_size - 1:
            # Send one to the right:
            all_to_all_send[local_rank + 1] = halo_to_right
            # Receive one from the right:
            all_to_all_recv[local_rank + 1] = torch.zeros_like(
                halo_to_right
            ).contiguous()

        # Perform exchange
        with record_function("all_to_all_queue_and_wait"):
            request = dist.all_to_all(
                all_to_all_recv, all_to_all_send, group=local_group, async_op=async_op
            )

            if async_op:
                # According to the docs, this will wait until the collectives are enqueued and it's safe to use the recv buffers.
                request.wait()

        # Extract received halos
        halo_from_left = all_to_all_recv[local_rank - 1] if local_rank != 0 else None
        halo_from_right = (
            all_to_all_recv[local_rank + 1] if local_rank != local_size - 1 else None
        )

    return halo_from_left, halo_from_right


@profile
def apply_halo_tensors(
    mesh: DeviceMesh,
    halo_config: HaloConfig,
    local_tensor: torch.Tensor,
    halo_from_left: Optional[torch.Tensor],
    halo_from_right: Optional[torch.Tensor],
) -> torch.Tensor:
    """Combines local tensor with received halos and edge padding.

    Args:
        mesh: Device mesh for process info
        halo_config: HaloConfig defining padding parameters
        local_tensor: Local tensor chunk
        halo_from_left: Halo received from left rank
        halo_from_right: Halo received from right rank

    Returns:
        Padded tensor with halos from neighboring ranks
    """
    # Get process group info
    local_group = mesh.get_group(halo_config.mesh_dim)
    local_rank = mesh.get_local_rank(halo_config.mesh_dim)
    local_size = dist.get_world_size(group=local_group)

    padded_output = []

    # Add left padding
    if local_rank != 0:
        padded_output.append(halo_from_left)
    else:
        if halo_config.edge_padding_size > 0:
            shape = list(halo_from_right.shape)
            shape[halo_config.tensor_dim] = halo_config.edge_padding_size
            zeros = torch.zeros(
                shape, device=halo_from_right.device, dtype=halo_from_right.dtype
            )
            padded_output.append(zeros)

    # Add the original, now central tensor
    padded_output.append(local_tensor)

    # Add right padding
    if local_rank != local_size - 1:
        padded_output.append(halo_from_right)
    else:
        if halo_config.edge_padding_size > 0:
            shape = list(halo_from_left.shape)
            shape[halo_config.tensor_dim] = halo_config.edge_padding_size
            zeros = torch.zeros(
                shape, device=halo_from_left.device, dtype=halo_from_left.dtype
            )
            padded_output.append(zeros)

    return torch.cat(padded_output, dim=halo_config.tensor_dim)


@profile
def apply_grad_halo(
    mesh: DeviceMesh,
    halo_config: HaloConfig,
    grad_input: torch.Tensor,
    halo_from_left: torch.Tensor,
    halo_from_right: torch.Tensor,
) -> torch.Tensor:
    """Applies halo gradients to input gradient tensor.

    The forward pass of a halo is padding to edges.  The backward
    pass is to trim add the halos to the edges of the local region
    (in the same locations that were sent previously).

    Args:
        mesh: Device mesh for process info
        halo_config: HaloConfig defining padding parameters
        grad_input: Input gradient tensor
        halo_from_left: Gradient from left halo
        halo_from_right: Gradient from right halo

    Returns:
        Updated gradient tensor with halo gradients applied
    """
    # Get process group info
    local_group = mesh.get_group(halo_config.mesh_dim)
    local_rank = mesh.get_local_rank(halo_config.mesh_dim)
    local_size = dist.get_world_size(group=local_group)

    # Apply right halo gradient
    if local_rank != local_size - 1:
        start_idx = (
            grad_input.shape[halo_config.tensor_dim]
            - halo_from_right.shape[halo_config.tensor_dim]
        )
        length = halo_from_right.shape[halo_config.tensor_dim]
        grad_input.narrow(halo_config.tensor_dim, start_idx, length).add_(
            halo_from_right
        )

    # Apply left halo gradient
    if local_rank != 0:
        start_idx = 0
        length = halo_from_left.shape[halo_config.tensor_dim]
        grad_input.narrow(halo_config.tensor_dim, start_idx, length).add_(
            halo_from_left
        )

    return grad_input
