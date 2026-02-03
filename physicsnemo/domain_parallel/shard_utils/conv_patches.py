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

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import (
    Shard,
)

from physicsnemo.domain_parallel import ShardTensor, ShardTensorSpec
from physicsnemo.domain_parallel.shard_utils.patch_core import (
    MissingShardPatch,
)
from physicsnemo.utils.profiling import profile

from .halo import HaloConfig, halo_padding
from .patch_core import promote_to_iterable


@profile
def conv_output_shape(
    L_in: int, padding: int, stride: int, kernel_size: int, dilation: int
) -> int:
    r"""Calculate the output length of a 1D convolution operation.

    This function computes the resulting length of a 1D tensor after applying
    a convolution with the given parameters.

    Parameters
    ----------
    L_in : int
        Input length.
    padding : int
        Padding size (on each side).
    stride : int
        Convolution stride.
    kernel_size : int
        Size of the convolution kernel.
    dilation : int
        Dilation factor for the kernel.

    Returns
    -------
    int
        The length of the output tensor after convolution.
    """
    L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    return int(L_out)


@profile
def compute_halo_from_kernel_stride_and_dilation(
    kernel_size: int,
    stride: int,
    dilation: int,
    padding: int | str,
    transposed: bool,
) -> int:
    r"""Compute the halo size needed for a convolution kernel along a single dimension.

    At a high level, the halo is equal to half the receptive field of the kernel.
    There are some subtleties with even vs odd kernel sizes and the conventions of
    where a kernel starts getting applied.

    Parameters
    ----------
    kernel_size : int
        Size of convolution kernel along this dimension.
    stride : int
        Convolution stride along this dimension.
    dilation : int
        Convolution dilation parameter.
    padding : Union[int, str]
        Padding specification for the convolution.
    transposed : bool
        Whether this is a transposed convolution.

    Returns
    -------
    int
        Required halo size on each side of a data chunk.

    Raises
    ------
    MissingShardPatch
        If kernel configuration is not supported for sharding,
        specifically for even kernels without matching stride.
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
    input_shape: tuple[int, ...],
    kernel_size: int,
    stride: int,
    dilation: int,
) -> int:
    r"""Convert a string padding specification to a numerical value.

    Parameters
    ----------
    padding : str
        String padding specification (``"same"``, ``"valid"``, or ``"none"``).
    input_shape : Tuple[int, ...]
        Shape of the input tensor.
    kernel_size : int
        Size of the convolution kernel.
    stride : int
        Convolution stride.
    dilation : int
        Convolution dilation factor.

    Returns
    -------
    int
        Numerical padding value.

    Raises
    ------
    ValueError
        If an invalid padding specification is provided.
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
    kernel_size: tuple[int, ...],
    conv_kwargs: dict[str, Any],
    transposed: bool = False,
) -> list[HaloConfig]:
    r"""Compute halo configurations for a sharded tensor based on convolution arguments.

    Parameters
    ----------
    input : ShardTensor
        The sharded tensor that will be used in convolution.
    kernel_size : Tuple[int, ...]
        Tuple of kernel dimensions for the convolution.
    conv_kwargs : Dict[str, Any]
        Dictionary of convolution arguments including stride, padding,
        dilation, and groups.
    transposed : bool, default=False
        Whether this is a transposed convolution.

    Returns
    -------
    List[HaloConfig]
        List of HaloConfig objects for each sharded dimension.

    Notes
    -----
    This function updates ``conv_kwargs`` in place, setting padding to 0
    for sharded dimensions.
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
            transposed,
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
    sharding_shape: tuple[int, ...],
    conv_kwargs: dict[str, Any],
    kernel_size: tuple[int, ...],
    transposed: bool = False,
) -> tuple[int, ...]:
    r"""Determine the output shape after a convolution for a specified input shape.

    Handles both regular and transposed convolutions.

    Parameters
    ----------
    sharding_shape : Tuple[int, ...]
        Input tensor shape.
    conv_kwargs : Dict[str, Any]
        Dictionary of convolution parameters (stride, padding, dilation, etc.).
    kernel_size : Tuple[int, ...]
        Tuple of kernel dimensions for the convolution.
    transposed : bool, default=False
        Whether this is a transposed convolution.

    Returns
    -------
    Tuple[int, ...]
        Output shape after the convolution operation.
    """
    output_shape = []
    tensor_rank = len(sharding_shape[2:])
    for tensor_dim in range(tensor_rank):
        if not transposed:
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


def compute_haloed_and_padded_input_shape(
    input_shape: tuple[int, ...],
    target_mesh_dim: int,
    mesh_coords: tuple[int, ...],
    mesh_sizes: tuple[int, ...],
    halo_config_map: dict[int, HaloConfig],
) -> tuple[int, ...]:
    r"""Determine the output shape after halo and edge padding is applied.

    Given an input shape, a list of halo configs, and the rank of this
    input in the input tensor, determine the output shape for this input
    after the halo and edge padding is applied.

    Parameters
    ----------
    input_shape : Tuple[int, ...]
        The shape of the input tensor.
    target_mesh_dim : int
        The dimension of the mesh that this input is along.
    mesh_coords : Tuple[int, ...]
        The coordinates of this input in the mesh.
    mesh_sizes : Tuple[int, ...]
        The sizes of the mesh.
    halo_config_map : Dict[int, HaloConfig]
        A map from halo mesh dim to HaloConfig.

    Returns
    -------
    Tuple[int, ...]
        The shape of the input tensor after the halo and edge padding is applied.
    """
    output_shape = list(input_shape)
    # Loop over the halo configs:
    for halo_config in halo_config_map.values():
        # This function must be careful.
        # We have two concepts of mesh dim, here
        # First, the tensor itself

        # What is the mesh size and rank along this mesh dim?

        # Always apply the halo padding at least one time:
        padding = halo_config.halo_size

        # Determine if this tensor is on the edge for this halo:
        halo_mesh_dim = halo_config.mesh_dim
        if (
            mesh_coords[halo_mesh_dim] == 0
            or mesh_coords[halo_mesh_dim] == mesh_sizes[halo_mesh_dim] - 1
        ):
            # apply edge padding instead:
            padding += halo_config.edge_padding_size
        else:
            # apply halo padding twice:
            padding += halo_config.halo_size

        output_shape[halo_config.tensor_dim] += padding

    return tuple(output_shape)


class ConvGradReducer(torch.autograd.Function):
    r"""Custom autograd function that performs an allreduce on gradients in backward pass.

    This makes defining a forward-only shard patch easier. If you need to allreduce
    weight grads in the backward pass, call this on the weight in the forward pass.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        weight_or_bias: torch.Tensor,
        spec: ShardTensorSpec,
    ) -> torch.Tensor:
        r"""Forward pass that saves the spec for backward.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context for saving variables for backward.
        weight_or_bias : torch.Tensor
            The weight or bias tensor to pass through.
        spec : ShardTensorSpec
            Shard spec of the convolutional input (not the weight_or_bias).

        Returns
        -------
        torch.Tensor
            The input tensor unchanged.
        """
        ctx.spec = spec
        return weight_or_bias

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_weight_or_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        r"""Backward pass that performs allreduce on gradients.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context containing saved variables from forward.
        grad_weight_or_bias : torch.Tensor
            Gradient of the loss with respect to weight or bias.

        Returns
        -------
        Tuple[torch.Tensor, None]
            Tuple of (reduced gradient, ``None`` for spec).
        """
        for mesh_dim in range(ctx.spec.mesh.ndim):
            if ctx.spec.placements[mesh_dim].is_shard():
                group = ctx.spec.mesh.get_group(mesh_dim)
                dist.all_reduce(grad_weight_or_bias, group=group)

        return grad_weight_or_bias, None


@profile
def partial_conv_nd(
    func: callable,
    conv_input: ShardTensor,
    weight: torch.nn.Parameter,
    bias: torch.nn.Parameter | None,
    conv_kwargs: dict[str, Any],
    transposed: bool = False,
) -> ShardTensor:
    r"""Perform a convolution on a sharded tensor with halo exchange.

    This high-level, differentiable function computes a convolution on a sharded tensor
    by performing these steps:

    1. Calculate the size of halos needed
    2. Apply halo padding (differentiable)
    3. Perform convolution on the padded tensor with padding=0 on sharded dimensions
    4. Return the result as a ShardTensor

    Parameters
    ----------
    func : callable
        The function to be called (``conv1d``, ``conv2d``, etc.).
    conv_input : ShardTensor
        The sharded input tensor.
    weight : torch.nn.Parameter
        Convolution filter weights.
    bias : Optional[torch.nn.Parameter]
        Optional bias parameter.
    conv_kwargs : Dict[str, Any]
        Dictionary of convolution parameters (stride, padding, etc.).
    transposed : bool, default=False
        Whether this is a transposed convolution.

    Returns
    -------
    ShardTensor
        Resulting ShardTensor after convolution operation.
    """

    input_spec = conv_input._spec

    # Get the spatial size of the kernel, which excludes conv_input/output channel sizes
    kernel_size = weight.shape[2:]

    #####################################################################
    # Halo computations and metad data
    #####################################################################

    # Compute the halo configs, one per sharded dim
    # It also *updates* conv_kwargs in place to set padding to 0 on the sharded dims
    halo_configs = compute_halo_configs_from_conv_args(
        conv_input, kernel_size, conv_kwargs, transposed
    )

    # We need to know not just the local shape after the halo,
    # but EVERY sharded shape after the halo

    # Get the shapes:
    sharding_shapes = input_spec.sharding_shapes()

    # Create a mapping from mesh_dim to halo_config for easy lookup
    halo_config_map = {config.mesh_dim: config for config in halo_configs}

    # This will be the sharded shapes after the halo:
    haloed_and_padded_input_shapes = {}

    # This loop will compute, for every shard of the input, what it's shape will
    # be after the halo layer.  This includes potential edge paddings.

    # We loop over the mesh and then each shard shape along that mesh dim

    # A subroutine will compute, given the halo configs, how a tensor will
    # be updated given it's mesh dimension and location on the mesh:

    haloed_and_padded_input_shapes = {}

    # The indexing here can get really tricky if you're not careful.
    # Sharding shapes are only stored for tensors shards that are along
    # the same dimension as our local tensor.
    # So, when computing edge/not edge for halo sizes,
    # all indexing starts with the mesh index of our local tensor:
    mesh_sizes = [len(sharding_tuple) for sharding_tuple in sharding_shapes.values()]

    self_mesh_coords = tuple(
        input_spec.mesh.get_local_rank(m) for m in range(input_spec.mesh.ndim)
    )

    for mesh_dim, sharding_tuple in sharding_shapes.items():
        haloed_and_padded_input_shapes[mesh_dim] = []

        for i, shard_shape in enumerate(sharding_tuple):
            # For this function to evaluate correctly, we need to pass
            # for each tensor it's coordinates in the mesh and the mesh_sizes
            # Starting from the local tensor's index, update the index
            # along the mesh dimension we're currently investigating:
            mesh_coords = list(self_mesh_coords)
            mesh_coords[mesh_dim] = i
            output_shape = compute_haloed_and_padded_input_shape(
                shard_shape,
                mesh_dim,
                mesh_coords,
                mesh_sizes,
                halo_config_map,
            )
            haloed_and_padded_input_shapes[mesh_dim].append(output_shape)

    local_input = conv_input.to_local()

    # Finally, Actually APPLY the halo padding to the conv_input tensor
    for halo_config in halo_configs:
        local_input = halo_padding(local_input, input_spec.mesh, halo_config)

    #####################################################################
    # Apply the convolution locally
    #####################################################################
    # For the backward pass: on input sharded dimensions, we need to to reduce the
    # the grads for weight and bias:
    weight = ConvGradReducer.apply(weight, input_spec)
    if bias is not None:
        bias = ConvGradReducer.apply(bias, input_spec)

    # Perform the convolution on the padded tensor
    local_output = func(local_input, weight, bias, **conv_kwargs)

    #####################################################################
    # Create the output shard tensor
    #####################################################################
    batch_channel_shape = tuple(local_output.shape[:2])
    # Update the output shapes to take into account the batch anc channel dims:
    real_output_shapes = {
        dim: tuple(
            batch_channel_shape
            + compute_output_shape(s, conv_kwargs, kernel_size, transposed)
            for s in haloed_and_padded_input_shapes[dim]
        )
        for dim in haloed_and_padded_input_shapes
    }

    # Convert the local output to a ShardTensor
    output = ShardTensor.from_local(
        local_output,
        input_spec.mesh,
        input_spec.placements,
        sharding_shapes=real_output_shapes,
    )

    return output


def generic_conv_nd_wrapper(
    func: Callable,
    types: tuple[Any, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ShardTensor:
    r"""Wrapper function for N-dimensional convolution operations supporting ShardTensors.

    This function dispatches convolution operations to appropriate implementations
    based on input types. It handles both regular and transposed convolutions.

    Parameters
    ----------
    func : callable
        The convolution function to be wrapped (``conv1d``, ``conv2d``, etc.).
    types : tuple
        Tuple of input types (unused).
    args : tuple
        Positional arguments to the convolution function.
    kwargs : dict
        Keyword arguments to the convolution function.

    Returns
    -------
    ShardTensor
        The result of the convolution operation.
    """

    if "transpose" in func.__name__:
        inputs, weight, bias, conv_kwargs = repackage_conv_transposed_args(
            *args, **kwargs
        )
        transposed = True
    else:
        inputs, weight, bias, conv_kwargs = repackage_conv_args(*args, **kwargs)
        transposed = False

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
    return partial_conv_nd(func, inputs, weight, bias, conv_kwargs, transposed)


@profile
def repackage_conv_args(
    input: torch.Tensor | ShardTensor,
    weight: torch.Tensor | DTensor,
    bias: torch.Tensor | DTensor | None = None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    groups: int = 1,
    output_padding: int | tuple[int, ...] = 0,
    *args,
    **kwargs,
) -> tuple[
    torch.Tensor | ShardTensor,
    torch.Tensor | DTensor,
    torch.Tensor | DTensor | None,
    dict,
]:
    r"""Repackage convolution arguments into standard format.

    Takes the full set of arguments that could be passed to a convolution operation
    and separates them into core tensor inputs (input, weight, bias) and
    configuration parameters packaged as a kwargs dict.

    Parameters
    ----------
    input : Union[torch.Tensor, ShardTensor]
        Input tensor to convolve.
    weight : Union[torch.Tensor, DTensor]
        Convolution kernel weights.
    bias : Union[torch.Tensor, DTensor, None], optional
        Optional bias tensor.
    stride : Union[int, Tuple[int, ...]], default=1
        Convolution stride length(s).
    padding : Union[int, Tuple[int, ...]], default=0
        Input padding size(s).
    dilation : Union[int, Tuple[int, ...]], default=1
        Kernel dilation factor(s).
    groups : int, default=1
        Number of convolution groups.
    output_padding : Union[int, Tuple[int, ...]], default=0
        Additional output padding for transposed convolutions.
    *args : Any
        Additional positional arguments (unused).
    **kwargs : Any
        Additional keyword arguments (unused).

    Returns
    -------
    Tuple[Union[torch.Tensor, ShardTensor], Union[torch.Tensor, DTensor], Union[torch.Tensor, DTensor, None], dict]
        Tuple containing (input tensor, weight tensor, bias tensor or ``None``,
        dict of convolution configuration parameters).
    """
    # Package all non-tensor parameters into a kwargs dictionary
    return_kwargs = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        # "transposed": False,
        "groups": groups,
        # "output_padding": output_padding,
    }

    return input, weight, bias, return_kwargs


@profile
def repackage_conv_transposed_args(
    input: torch.Tensor | ShardTensor,
    weight: torch.Tensor | DTensor,
    bias: torch.Tensor | DTensor | None = None,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    output_padding: int | tuple[int, ...] = 0,
    groups: int = 1,
    dilation: int | tuple[int, ...] = 1,
    *args,
    **kwargs,
) -> tuple[
    torch.Tensor | ShardTensor,
    torch.Tensor | DTensor,
    torch.Tensor | DTensor | None,
    dict,
]:
    r"""Repackage transposed convolution arguments into standard format.

    Takes the full set of arguments that could be passed to a transposed convolution
    operation and separates them into core tensor inputs (input, weight, bias) and
    configuration parameters packaged as a kwargs dict.

    Parameters
    ----------
    input : Union[torch.Tensor, ShardTensor]
        Input tensor to convolve.
    weight : Union[torch.Tensor, DTensor]
        Convolution kernel weights.
    bias : Union[torch.Tensor, DTensor, None], optional
        Optional bias tensor.
    stride : Union[int, Tuple[int, ...]], default=1
        Convolution stride length(s).
    padding : Union[int, Tuple[int, ...]], default=0
        Input padding size(s).
    output_padding : Union[int, Tuple[int, ...]], default=0
        Additional output padding for transposed convolutions.
    groups : int, default=1
        Number of convolution groups.
    dilation : Union[int, Tuple[int, ...]], default=1
        Kernel dilation factor(s).
    *args : Any
        Additional positional arguments (unused).
    **kwargs : Any
        Additional keyword arguments (unused).

    Returns
    -------
    Tuple[Union[torch.Tensor, ShardTensor], Union[torch.Tensor, DTensor], Union[torch.Tensor, DTensor, None], dict]
        Tuple containing (input tensor, weight tensor, bias tensor or ``None``,
        dict of convolution configuration parameters).
    """
    # Package all non-tensor parameters into a kwargs dictionary
    return_kwargs = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "output_padding": output_padding,
        "groups": groups,
        # "transposed": True,
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
