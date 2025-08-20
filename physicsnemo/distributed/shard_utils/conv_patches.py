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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from physicsnemo.utils.profiling import annotate, profile
from physicsnemo.utils.version_check import check_module_requirements

check_module_requirements("physicsnemo.distributed.shard_tensor")


from torch.distributed.tensor import DTensor  # noqa: E402
from torch.distributed.tensor.placement_types import (  # noqa: E402
    Shard,
)

from physicsnemo.distributed import ShardTensor, ShardTensorSpec  # noqa: E402
from physicsnemo.distributed.shard_utils.patch_core import (  # noqa: E402
    MissingShardPatch,
)

from .halo import HaloConfig, halo_padding  # noqa: E402
from .patch_core import promote_to_iterable  # noqa: E402

aten = torch.ops.aten


@profile
def conv_output_shape(
    L_in: int, padding: int, stride: int, kernel_size: int, dilation: int
) -> int:
    """Calculate the output length of a 1D convolution operation.

    This function computes the resulting length of a 1D tensor after applying
    a convolution with the given parameters.

    Args:
        L_in: Input length
        padding: Padding size (on each side)
        stride: Convolution stride
        kernel_size: Size of the convolution kernel
        dilation: Dilation factor for the kernel

    Returns:
        The length of the output tensor after convolution
    """
    L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    return int(L_out)


@profile
def compute_halo_from_kernel_stride_and_dilation(
    kernel_size: int,
    stride: int,
    dilation: int,
    padding: Union[int, str],
    transposed: bool,
) -> int:
    """Compute the halo size needed for a convolution kernel along a single dimension.

    At a high level, the halo is equal to half the receptive field of the kernel.
    There are some subtleties with even vs odd kernel sizes and the conventions of
    where a kernel starts getting applied.

    Args:
        kernel_size: Size of convolution kernel along this dimension
        stride: Convolution stride along this dimension
        dilation: Convolution dilation parameter

    Returns:
        Required halo size on each side of a data chunk

    Raises:
        MissingShardPatch: If kernel configuration is not supported for sharding,
                          specifically for even kernels without matching stride
    """
    # Special case: even kernel with matching stride and no dilation needs no halo
    if kernel_size % 2 == 0:
        if kernel_size == stride and dilation == 1 and padding == 0:
            return 0
        else:
            raise MissingShardPatch(
                "Sharded Convolution is not implemented for even kernels without matching stride and padding 0. "
                "If you need this functionality, please open an issue at https://github.com/NVIDIA/PhysicsNemo/issues"
            )

    if transposed:
        # Support currently only for even kernels with padding 0 and stride = kernel_size
        if kernel_size % 2 != 0 or padding != 0 or stride != kernel_size:
            raise MissingShardPatch(
                "Sharded Convolution is not implemented for transposed convolutions with non-matching stride or padding. "
                "If you need this functionality, please open an issue at https://github.com/NVIDIA/PhysicsNemo/issues"
            )

    # The receptive field is how far in the input a pixel in the output can see
    # It's used to calculate how large the halo computation has to be
    receptive_field = dilation * (kernel_size - 1) + 1

    # For odd kernels, the halo size is half the receptive field (integer division)
    # This represents how many pixels we need from neighboring ranks on each side
    halo_size = receptive_field // 2

    return halo_size


@profile
def padding_from_str_and_params(
    padding: str,
    input_shape: Tuple[int, ...],
    kernel_size: int,
    stride: int,
    dilation: int,
) -> int:
    """Convert a string padding specification to a numerical value.

    Args:
        padding: String padding specification
        conv_kwargs: Dictionary of convolution arguments
        dim: Dimension index
    """

    if padding == "same":
        total_padding = max(
            0,
            (
                (input_shape - 1) * stride
                + 1
                + (kernel_size - 1) * dilation
                - input_shape
            ),
        )
        return total_padding // 2
    elif padding == "valid":
        return 0
    elif padding == "none":
        return 0
    else:
        raise ValueError(f"Invalid padding specification: {padding}")


@profile
def compute_halo_configs_from_conv_args(
    input: ShardTensor,
    kernel_size: Tuple[int, ...],
    conv_kwargs: Dict[str, Any],
) -> List[HaloConfig]:
    """Compute halo configurations for a sharded tensor based on convolution arguments.

    Args:
        input: The sharded tensor that will be used in convolution
        kernel_size: Tuple of kernel dimensions for the convolution
        conv_kwargs: Dictionary of convolution arguments including stride,
                    padding, dilation, and groups

    Returns:
        List of HaloConfig objects for each sharded dimension

    Note:
        This function updates conv_kwargs in place, setting padding to 0 for sharded dimensions.
    """

    placements = input._spec.placements

    stride = conv_kwargs["stride"]
    dilation = conv_kwargs["dilation"]

    # This is to update and set the padding to 0 on the sharded dims:
    padding = conv_kwargs["padding"]

    if isinstance(padding, str):
        # Convert this to numerical values:
        padding = [
            padding_from_str_and_params(
                padding, input.shape[i], kernel_size[i], stride[i], dilation[i]
            )
            for i in range(len(kernel_size))
        ]
    else:
        # Ensure it's a list:
        padding = list(padding)

    # All parameters are assumed to be iterables of the same length
    halo_configs = []

    for mesh_dim, p in enumerate(placements):
        if not isinstance(p, Shard):
            continue

        tensor_dim = p.dim
        if tensor_dim in [0, 1]:  # Skip batch and channel dimensions
            continue

        # Map tensor dimension to kernel dimension (accounting for batch, channel dims)
        kernel_dim = tensor_dim - 2
        if kernel_dim >= len(kernel_size):
            continue

        # Compute halo size for this dimension
        halo_size = compute_halo_from_kernel_stride_and_dilation(
            kernel_size[kernel_dim],
            stride[kernel_dim],
            dilation[kernel_dim],
            padding[kernel_dim],
            conv_kwargs["transposed"],
        )

        if halo_size > 0:
            # Create a halo config for this dimension

            halo_configs.append(
                HaloConfig(
                    mesh_dim=mesh_dim,
                    tensor_dim=tensor_dim,
                    halo_size=halo_size,
                    edge_padding_size=padding[kernel_dim],
                    communication_method="a2a",
                    async_op=True,
                )
            )
            # Set the padding to 0 on the sharded dims:
            padding[kernel_dim] = 0

    # Update the padding before returning:
    conv_kwargs["padding"] = tuple(padding)

    return halo_configs


@profile
def compute_output_shape(
    sharding_shape: Tuple[int, ...],
    conv_kwargs: Dict[str, Any],
    kernel_size: Tuple[int, ...],
) -> Tuple[int, ...]:
    """
    For a specified input shape, determine the output shape after a convolution.
    Handles both regular and transposed convolutions.
    """
    output_shape = []
    tensor_rank = len(sharding_shape[2:])
    for tensor_dim in range(tensor_rank):
        if not conv_kwargs["transposed"]:
            # Regular convolution
            num = (
                sharding_shape[tensor_dim + 2]
                + 2 * conv_kwargs["padding"][tensor_dim]
                - (kernel_size[tensor_dim] - 1) * conv_kwargs["dilation"][tensor_dim]
                - 1
            )
            o = num / conv_kwargs["stride"][tensor_dim] + 1
        else:
            # Transposed convolution
            output_padding = conv_kwargs.get("output_padding", (0,) * tensor_rank)[
                tensor_dim
            ]
            o = (sharding_shape[tensor_dim + 2] - 1) * conv_kwargs["stride"][tensor_dim]
            o = o - 2 * conv_kwargs["padding"][tensor_dim]
            o = o + conv_kwargs["dilation"][tensor_dim] * (kernel_size[tensor_dim] - 1)
            o = o + output_padding + 1

        output_shape.append(int(o))

    return tuple(output_shape)


@profile
def partial_conv_nd(
    input: ShardTensor,
    weight: torch.nn.Parameter,
    bias: Optional[torch.nn.Parameter],
    conv_kwargs: Dict[str, Any],
) -> ShardTensor:
    """Perform a convolution on a sharded tensor with halo exchange.

    This high-level, differentiable function computes a convolution on a sharded tensor
    by performing these steps:
    1. Calculate the size of halos needed
    2. Apply halo padding (differentiable)
    3. Perform convolution on the padded tensor with padding=0 on sharded dimensions
    4. Return the result as a ShardTensor

    Args:
        input: The sharded input tensor
        weight: Convolution filter weights
        bias: Optional bias parameter
        conv_kwargs: Dictionary of convolution parameters (stride, padding, etc.)

    Returns:
        Resulting ShardTensor after convolution operation
    """
    with annotate("partial_conv_nd"):
        kernel_size = weight.shape[2:]

        # This will produce one config per sharded dim
        # It also *updates* conv_kwargs in place to set padding to 0 on the sharded dims
        halo_configs = compute_halo_configs_from_conv_args(
            input, kernel_size, conv_kwargs
        )

        # We get one halo_config per sharded dim.
        sharding_shapes = input._spec.sharding_shapes()
        # # First, update the shapes to take into account the halo and edge paddings:

        # Create a mapping from mesh_dim to halo_config for easy lookup
        halo_config_map = {config.mesh_dim: config for config in halo_configs}

        real_input_shapes = {}
        for mesh_dim, sharing_tuple in sharding_shapes.items():
            # If this mesh_dim doesn't need halos, just copy the original shapes
            if mesh_dim not in halo_config_map:
                real_input_shapes[mesh_dim] = sharing_tuple
                continue

            tensor_dim = halo_config_map[mesh_dim].tensor_dim
            real_input_shapes[mesh_dim] = []
            for i, s in enumerate(sharing_tuple):
                padding = halo_config_map[mesh_dim].halo_size

                if i == 0 or i == len(sharing_tuple) - 1:
                    # On the edge of the split, the additional size is halo + edge padding
                    padding += halo_config_map[mesh_dim].edge_padding_size
                else:
                    # Otherwise, its 2xhalo size added on.
                    padding += halo_config_map[mesh_dim].halo_size

                updated_shape = list(s)
                updated_shape[tensor_dim] += padding

                real_input_shapes[mesh_dim].append(tuple(updated_shape))

        input_spec = input._spec
        local_input = input.to_local()

        with annotate("halo_padding"):
            # Apply the halo padding to the input tensor
            for halo_config in halo_configs:
                local_input = halo_padding(local_input, input._spec.mesh, halo_config)

        with annotate("perform_convolution"):
            # Perform the convolution on the padded tensor
            local_output = perform_convolution(
                local_input, weight, bias, input_spec, conv_kwargs
            )

        batch_channel_shape = tuple(local_output.shape[:2])
        # Update the output shapes to take into account the batch anc channel dims:
        real_output_shapes = {
            dim: tuple(
                batch_channel_shape + compute_output_shape(s, conv_kwargs, kernel_size)
                for s in real_input_shapes[dim]
            )
            for dim in real_input_shapes
        }

        # Convert the local output to a ShardTensor
        with annotate("partial_conv_nd.from_local"):
            output = ShardTensor.from_local(
                local_output,
                input_spec.mesh,
                input_spec.placements,
                sharding_shapes=real_output_shapes,
            )

        return output


@profile
def perform_convolution(
    inputs: torch.Tensor,
    weights: torch.nn.Parameter,
    bias: Optional[torch.nn.Parameter],
    input_spec: "ShardTensorSpec",
    conv_kwargs: Dict[str, Any],
) -> torch.Tensor:
    """Apply a convolution operation using the PartialConvND autograd function.

    Args:
        inputs: Input tensor to convolve
        weights: Convolution filter weights
        bias: Optional bias tensor
        input_spec: Specification for output ShardTensor
        conv_kwargs: Dictionary of convolution parameters

    Returns:
        ShardTensor containing the convolution result
    """
    return PartialConvND.apply(inputs, weights, bias, input_spec, conv_kwargs)


class PartialConvND(torch.autograd.Function):
    """Sharded convolution operation that uses halo message passing for distributed computation.

    This class implements a distributed convolution primitive that operates on sharded tensors.
    It handles both forward and backward passes while managing communication between shards.

    Leverages torch.ops.aten.convolution.default for generic convolutions.
    """

    @staticmethod
    @profile
    def forward(
        ctx,
        inputs: torch.Tensor,
        weights: torch.nn.Parameter,
        bias: Optional[torch.nn.Parameter],
        input_spec: "ShardTensorSpec",
        conv_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """Forward pass of the distributed convolution.

        Args:
            ctx: Context object for saving tensors needed in backward pass
            inputs: Input tensor to convolve
            weights: Convolution filter weights
            bias: Optional bias tensor
            input_spec: Specification for output ShardTensor
            conv_kwargs: Dictionary of convolution parameters (stride, padding, etc.)

        Returns:
            ShardTensor containing the convolution result
        """
        # Save spec for backward pass
        ctx.spec = input_spec

        # Save local tensors to avoid distributed dispatch in backward pass
        ctx.save_for_backward(inputs, weights, bias)

        # print type of inputs
        ctx.conv_kwargs = conv_kwargs
        # Perform local convolution on this shard
        local_chunk = aten.convolution.default(inputs, weights, bias, **conv_kwargs)

        ctx.requires_input_grad = inputs.requires_grad
        # return output
        return local_chunk

    @staticmethod
    @profile
    def backward(
        ctx, grad_output: torch.Tensor
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

        # Cast, in case the precision is off:
        weight = weight.to(dtype=grad_output.dtype)
        bias = bias.to(dtype=grad_output.dtype) if bias is not None else None
        local_chunk = local_chunk.to(dtype=grad_output.dtype)
        # Compute local gradients
        # local_grad_output = grad_output._local_tensor
        grad_input, grad_weight, grad_bias = aten.convolution_backward(
            grad_output,
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


def generic_conv_nd_wrapper(func: callable, types: tuple, args: tuple, kwargs: dict):
    """Wrapper function for N-dimensional convolution operations supporting shardtensors.

    This function dispatches convolution operations to appropriate implementations based on input types.
    It handles both regular and transposed convolutions, and supports both torch.Tensor and ShardTensor inputs.

    Args:
        func: The convolution function to be wrapped (conv1d, conv2d, etc.)
        types: Tuple of input types (unused)
        args: Positional arguments to the convolution function
        kwargs: Keyword arguments to the convolution function

    Returns:
        The result of the convolution operation

    """

    if "transpose" in func.__name__:
        input, weight, bias, conv_kwargs = repackage_conv_transposed_args(
            *args, **kwargs
        )
    else:
        input, weight, bias, conv_kwargs = repackage_conv_args(*args, **kwargs)

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

    # Use the convolution args to compute the sharded halo
    return partial_conv_nd(input, weight, bias, conv_kwargs)


@profile
def repackage_conv_args(
    input: Union[torch.Tensor, ShardTensor],
    weight: Union[torch.Tensor, DTensor],
    bias: Union[torch.Tensor, DTensor, None] = None,
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
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
        "transposed": False,
        "groups": groups,
        "output_padding": output_padding,
    }

    return input, weight, bias, return_kwargs


@profile
def repackage_conv_transposed_args(
    input: Union[torch.Tensor, ShardTensor],
    weight: Union[torch.Tensor, DTensor],
    bias: Union[torch.Tensor, DTensor, None] = None,
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    output_padding: Union[int, Tuple[int, ...]] = 0,
    groups: int = 1,
    dilation: Union[int, Tuple[int, ...]] = 1,
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
        "output_padding": output_padding,
        "groups": groups,
        "transposed": True,
    }

    return input, weight, bias, return_kwargs


ShardTensor.register_function_handler(
    torch.nn.functional.conv1d, generic_conv_nd_wrapper
)
ShardTensor.register_function_handler(
    torch.nn.functional.conv2d, generic_conv_nd_wrapper
)
ShardTensor.register_function_handler(
    torch.nn.functional.conv3d, generic_conv_nd_wrapper
)
ShardTensor.register_function_handler(
    torch.nn.functional.conv_transpose1d, generic_conv_nd_wrapper
)
ShardTensor.register_function_handler(
    torch.nn.functional.conv_transpose2d, generic_conv_nd_wrapper
)
ShardTensor.register_function_handler(
    torch.nn.functional.conv_transpose3d, generic_conv_nd_wrapper
)
