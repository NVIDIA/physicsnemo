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

from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import wrapt

from physicsnemo.utils.version_check import check_module_requirements

check_module_requirements("physicsnemo.distributed.shard_tensor")


from torch.distributed.tensor import DTensor  # noqa: E402
from torch.distributed.tensor.placement_types import (  # noqa: E402
    Shard,
)

from physicsnemo.distributed import ShardTensor, ShardTensorSpec  # noqa: E402
from physicsnemo.distributed.shard_utils.patch_core import (  # noqa: E402
    MissingShardPatch,
    UndeterminedShardingError,
)

from .halo import (  # noqa: E402
    apply_grad_halo,
    halo_padding_1d,
    halo_unpadding_1d,
    perform_halo_collective,
)
from .patch_core import promote_to_iterable  # noqa: E402

__all__ = [
    "conv1d_wrapper",
    "conv2d_wrapper",
    "conv3d_wrapper",
]


def conv_output_shape(L_in, p, s, k, d):
    L_out = (L_in + 2 * p - d * (k - 1) - 1) / s + 1
    return int(L_out)


def compute_halo_from_kernel_stride_and_dilation(
    kernel_size: int, stride: int, dilation: int
) -> int:
    """Compute the halo size needed for a convolution kernel along a single dimension.

    Args:
        kernel_size: Size of convolution kernel along this dimension
        stride: Convolution stride along this dimension
        dilation: Convolution dilation parameter

    Returns:
        Required halo size on each side of a data chunk

    Raises:
        MissingShardPatch: If kernel configuration is not supported for sharding
    """
    # Special case: even kernel with matching stride and no dilation needs no halo
    if kernel_size % 2 == 0:
        if kernel_size == stride and dilation == 1:
            return 0
        else:
            raise MissingShardPatch(
                "Sharded Convolution is not implemented for even kernels without matching stride"
            )

    if dilation != 1:
        raise MissingShardPatch(
            "Sharded Convolution is not implemented for dilation != 1"
        )

    # The receptive field is how far in the input a pixel in the output can see
    # It's used to calculate how large the halo computation has to be
    receptive_field = dilation * (kernel_size - 1) + 1

    # The number of halo pixels is the casting `int(receptive field/2)`
    # Why?  Assuming a filter in the output image is centered in the input image,
    # we have only half of it's filter to the left.
    # Even kernels:
    if kernel_size % 2 == 0:
        halo_size = int(receptive_field / 2 - 1)
    else:
        halo_size = int(receptive_field / 2)

    return halo_size


def shard_to_haloed_local_for_convNd(
    input: ShardTensor,
    kernel_shape: Tuple[int, ...],
    stride: Union[int, Tuple[int, ...]],
    padding: Union[int, Tuple[int, ...]],
    dilation: Union[int, Tuple[int, ...]],
    groups: int = 1,
    **extra_kwargs,
) -> torch.Tensor:
    """Converts a sharded tensor to a local tensor with halo regions for convolution.

    Takes a sharded tensor and adds appropriate halo regions based on the convolution
    parameters, so that when the convolution is applied locally it produces the same
    result as if applied globally then sharded.

    Args:
        input: ShardTensor to add halos to
        kernel_shape: Shape of convolution kernel
        stride: Convolution stride (int or tuple matching kernel dims)
        padding: Convolution padding (int or tuple matching kernel dims)
        dilation: Convolution dilation (int or tuple matching kernel dims)
        groups: Number of convolution groups (default: 1)

    Returns:
        Local torch.Tensor with added halo regions

    Raises:
        ValueError: If tensor is sharded along batch/channel dims or invalid dims
    """
    mesh = input._spec.mesh
    placements = input._spec.placements

    if mesh.ndim != len(placements):
        raise ValueError("Mesh dimensions must match number of placements")

    # Extract parameters for sharded dimensions only
    h_kernel = []
    h_stride = []
    h_padding = []
    h_dilation = []

    for p in placements:
        if not isinstance(p, Shard):
            continue

        tensor_dim = p.dim
        if tensor_dim in [0, 1]:
            raise ValueError("Cannot shard convolution along batch/channel dimensions")

        if tensor_dim >= len(kernel_shape) + 2:
            raise ValueError("Invalid tensor dimension for kernel rank")

        # Convert from NCHW indexing to kernel indexing
        kernel_idx = tensor_dim - 2
        h_kernel.append(kernel_shape[kernel_idx])
        h_stride.append(stride[kernel_idx])
        h_padding.append(padding[kernel_idx])
        h_dilation.append(dilation[kernel_idx])

    # Compute required halo size for each sharded dim
    halo_size = tuple(
        compute_halo_from_kernel_stride_and_dilation(k, s, d)
        for k, s, d in zip(h_kernel, h_stride, h_dilation)
    )

    # Set edge padding type based on convolution padding
    edge_padding_t = "zeros" if any(h_padding) else "none"

    # Add halos via collective communication
    local_input = HaloPaddingConvND.apply(input, halo_size, edge_padding_t, padding)

    return local_input


class HaloPaddingConvND(torch.autograd.Function):
    """Autograd wrapper for distributed convolution-centric halo padding.

    Handles halo padding for distributed convolutions using ShardTensor concept
    (local tensor + device mesh + shard placements). Forward pass gathers adjacent regions
    from neighboring devices. Backward pass distributes gradients outward.

    Supports multi-dimensional halo passing with compatible mesh and halo parameters."""

    @staticmethod
    def forward(
        ctx,
        stensor: ShardTensor,
        halo: tuple[int, ...],
        edge_padding_t: str,
        edge_padding_s: tuple[int, ...],
    ) -> torch.Tensor:
        """Forward pass of distributed halo padding.

        Args:
            stensor: Input ShardTensor
            halo: Halo sizes for each dimension
            edge_padding_t: Edge padding type ("zeros" or "none")
            edge_padding_s: Edge padding sizes

        Returns:
            Padded local tensor

        Raises:
            ValueError: If halo size does not match mesh rank
        """
        mesh = stensor.device_mesh
        if len(halo) != mesh.ndim:
            raise ValueError(
                f"Halo size ({len(halo)}) must match mesh rank ({mesh.ndim})"
            )

        placements = stensor.placements
        local_tensor = stensor.to_local()

        # Apply halo padding for each sharded dimension
        for mesh_dim in range(mesh.ndim):
            if isinstance(placements[mesh_dim], Shard):
                tensor_dim = placements[mesh_dim].dim
                local_tensor = halo_padding_1d(
                    local_tensor,
                    mesh,
                    mesh_dim,
                    tensor_dim,
                    halo[mesh_dim],
                    edge_padding_t,
                    edge_padding_s[mesh_dim],
                )

        ctx.halo = halo
        ctx.spec = stensor._spec
        ctx.requires_input_grad = stensor.requires_grad

        return local_tensor

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[ShardTensor, None, None, None]:
        """Backward pass of distributed halo padding.

        Args:
            grad_output: Gradient tensor from downstream

        Returns:
            Tuple of (gradient ShardTensor, None, None, None)
        """
        spec = ctx.spec
        mesh = spec.mesh
        placements = spec.placements
        halo = ctx.halo

        # Process gradients for each sharded dimension
        for mesh_dim in range(mesh.ndim):
            if isinstance(placements[mesh_dim], Shard):
                tensor_dim = placements[mesh_dim].dim

                # Unpad gradients and get halo slices
                grad_input, grad_halos = halo_unpadding_1d(
                    grad_output,
                    mesh,
                    mesh_dim,
                    tensor_dim,
                    halo[mesh_dim],
                    return_slices=True,
                )

                # Exchange and accumulate gradient halos
                halo_from_left, halo_from_right = perform_halo_collective(
                    mesh, mesh_dim, *grad_halos
                )
                grad_input = apply_grad_halo(
                    mesh,
                    mesh_dim,
                    tensor_dim,
                    grad_input,
                    halo_from_left,
                    halo_from_right,
                )

        # Wrap gradient in ShardTensor
        grad_tensor = ShardTensor(
            grad_input,
            spec,
            requires_grad=grad_input.requires_grad,
        )

        return grad_tensor, None, None, None


aten = torch.ops.aten


class PartialConvND(torch.autograd.Function):
    """Sharded convolution operation that uses halo message passing for distributed computation.

    This class implements a distributed convolution primitive that operates on sharded tensors.
    It handles both forward and backward passes while managing communication between shards.

    Leverages torch.ops.aten.convolution.default for generic convolutions.
    """

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        weights: torch.nn.Parameter,
        bias: Optional[torch.nn.Parameter],
        output_spec: "ShardTensorSpec",
        conv_kwargs: dict,
    ) -> "ShardTensor":
        """Forward pass of the distributed convolution.

        Args:
            ctx: Context object for saving tensors needed in backward pass
            inputs: Input tensor to convolve
            weights: Convolution filter weights
            bias: Optional bias tensor
            output_spec: Specification for output ShardTensor
            conv_kwargs: Dictionary of convolution parameters (stride, padding, etc.)

        Returns:
            ShardTensor containing the convolution result
        """
        # Save spec for backward pass
        ctx.spec = output_spec

        # Save local tensors to avoid distributed dispatch in backward pass
        ctx.save_for_backward(inputs, weights, bias)

        # Get sharded output dimensions by checking placements
        sharded_output_dims = []
        for i, placement in enumerate(output_spec.placements):
            if isinstance(placement, Shard):
                sharded_output_dims.append(placement.dim)

        # Force padding to 0 _along sharded dims only_ since padding is
        # handled by halo exchange
        # Check if input is channels first (NCHW) or channels last (NHWC)
        if inputs.is_contiguous(memory_format=torch.contiguous_format):
            offset = 2
        elif inputs.is_contiguous(memory_format=torch.channels_last):
            offset = 1
        else:
            raise ValueError("Input tensor must be channels first or channels last")

        padding = list(conv_kwargs["padding"])

        # Update padding arguments.  Set to 0 on sharded dims only:
        for i, p in enumerate(padding):
            if i + offset in sharded_output_dims:
                padding[i] = 0

        conv_kwargs["padding"] = tuple(padding)
        ctx.conv_kwargs = conv_kwargs
        # Perform local convolution on this shard
        local_chunk = aten.convolution.default(inputs, weights, bias, **conv_kwargs)

        # Wrap result in ShardTensor with specified distribution
        output = ShardTensor.from_local(
            local_chunk, output_spec.mesh, output_spec.placements
        )

        ctx.requires_input_grad = inputs.requires_grad
        return output

    @staticmethod
    def backward(
        ctx, grad_output: "ShardTensor"
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None]:
        """Backward pass for distributed convolution.

        Args:
            ctx: Context object containing saved tensors
            grad_output: Gradient of the loss with respect to convolution output

        Returns:
            Tuple containing gradients for inputs, weights, and bias (plus None values for other args)
        """
        spec = ctx.spec
        conv_kwargs = ctx.conv_kwargs
        local_chunk, weight, bias = ctx.saved_tensors

        # Specify which inputs need gradients
        output_mask = (
            ctx.requires_input_grad,  # input gradient
            True,  # weight gradient always needed
            bias is not None,  # bias gradient if bias exists
        )

        # Compute local gradients
        local_grad_output = grad_output._local_tensor
        grad_input, grad_weight, grad_bias = aten.convolution_backward(
            local_grad_output,
            local_chunk,
            weight,
            bias,
            output_mask=output_mask,
            **conv_kwargs,
        )

        # Synchronize weight and bias gradients across all ranks
        group = spec.mesh.get_group()
        dist.all_reduce(grad_weight, group=group)
        if grad_bias is not None:
            dist.all_reduce(grad_bias, group=group)

        return grad_input, grad_weight, grad_bias, None, None


@wrapt.patch_function_wrapper(
    "torch.nn.functional", "conv1d", enabled=ShardTensor.patches_enabled
)
def conv1d_wrapper(wrapped, instance, args, kwargs):

    return generic_conv_nd_wrapper(wrapped, instance, args, kwargs)


@wrapt.patch_function_wrapper(
    "torch.nn.functional", "conv2d", enabled=ShardTensor.patches_enabled
)
def conv2d_wrapper(wrapped, instance, args, kwargs):

    return generic_conv_nd_wrapper(wrapped, instance, args, kwargs)


@wrapt.patch_function_wrapper(
    "torch.nn.functional", "conv3d", enabled=ShardTensor.patches_enabled
)
def conv3d_wrapper(wrapped, instance, args, kwargs):

    return generic_conv_nd_wrapper(wrapped, instance, args, kwargs)


def generic_conv_nd_wrapper(wrapped, instance, args, kwargs):
    """Generic wrapper for torch N-dimensional convolution operations.

    Handles both regular torch.Tensor inputs and distributed ShardTensor inputs.
    For regular tensors, passes through to the wrapped convolution.
    For ShardTensor inputs, handles gathering weights/bias and applying distributed
    convolution with halo regions.

    Args:
        wrapped: Original convolution function being wrapped
        instance: Instance the wrapped function is bound to
        args: Positional arguments for convolution
        kwargs: Keyword arguments for convolution

    Returns:
        Convolution result as either torch.Tensor or ShardTensor

    Raises:
        UndeterminedShardingError: If input tensor types are invalid
    """
    input, weight, bias, conv_kwargs = repackage_conv_args(*args, **kwargs)

    # Handle regular torch tensor inputs
    if (
        type(input) == torch.Tensor
        and type(weight) == torch.nn.parameter.Parameter
        and (bias is None or type(bias) == torch.nn.parameter.Parameter)
    ):
        return wrapped(*args, **kwargs)

    # Handle distributed ShardTensor inputs
    elif type(input) == ShardTensor:
        # Gather any distributed weights/bias
        if isinstance(weight, (ShardTensor, DTensor)):
            weight = weight.full_tensor()
        if isinstance(bias, (ShardTensor, DTensor)):
            bias = bias.full_tensor()

        kernel_shape = weight.shape[2:]

        # Promote scalar args to match kernel dimensions
        promotables = ["stride", "padding", "dilation", "output_padding"]
        conv_kwargs = {
            key: promote_to_iterable(p, kernel_shape) if key in promotables else p
            for key, p in conv_kwargs.items()
        }

        # Add halos and perform distributed convolution
        local_input = shard_to_haloed_local_for_convNd(
            input, kernel_shape, **conv_kwargs
        )
        output_spec = input._spec
        x = PartialConvND.apply(local_input, weight, bias, output_spec, conv_kwargs)
        return x

    else:
        msg = (
            "input, weight, bias (if not None) must all be the valid types "
            "(torch.Tensor or ShardTensor), but got "
            f"{type(input)}, "
            f"{type(weight)}, "
            f"{type(bias)}, "
        )
        raise UndeterminedShardingError(msg)


def repackage_conv_args(
    input: Union[torch.Tensor, ShardTensor],
    weight: Union[torch.Tensor, DTensor],
    bias: Union[torch.Tensor, DTensor, None] = None,
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
    transposed: bool = False,
    output_padding: Union[int, Tuple[int, ...]] = 0,
    *args,
    **kwargs,
) -> Tuple[
    Union[torch.Tensor, ShardTensor],
    Union[torch.Tensor, DTensor],
    Union[torch.Tensor, DTensor, None],
    dict,
]:
    """Repackages convolution arguments into standard format.

    Takes the full set of arguments that could be passed to a convolution operation
    and separates them into core tensor inputs (input, weight, bias) and
    configuration parameters packaged as a kwargs dict.

    Args:
        input: Input tensor to convolve
        weight: Convolution kernel weights
        bias: Optional bias tensor
        stride: Convolution stride length(s)
        padding: Input padding size(s)
        dilation: Kernel dilation factor(s)
        groups: Number of convolution groups
        transposed: Whether this is a transposed convolution
        output_padding: Additional output padding for transposed convs
        *args: Additional positional args (unused)
        **kwargs: Additional keyword args (unused)

    Returns:
        Tuple containing:
        - Input tensor
        - Weight tensor
        - Bias tensor (or None)
        - Dict of convolution configuration parameters
    """
    # Package all non-tensor parameters into a kwargs dictionary
    return_kwargs = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "transposed": transposed,
        "output_padding": output_padding,
        "groups": groups,
    }

    return input, weight, bias, return_kwargs


# This will become the future implementation, or similar.
# Why not today?  Because the backwards pass in DTensor has an explicit (and insufficient)
# hard coded implementation for the backwards pass.
# When that switch happens, the order in the arg repackaging will need to be updated.
# ShardTensor.register_function_handler(aten.convolution.default, generic_conv_nd_wrapper)
