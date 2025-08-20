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

from collections.abc import Iterable
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, cast
from warnings import warn

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, _mesh_resources

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.profiling import annotate, profile
from physicsnemo.utils.version_check import check_module_requirements

# Prevent importing this module if the minimum version of pytorch is not met.
check_module_requirements("physicsnemo.distributed.shard_tensor")

from torch.distributed.tensor import DTensor  # noqa: E402
from torch.distributed.tensor._dtensor_spec import (  # noqa: E402
    TensorMeta,
)
from torch.distributed.tensor.placement_types import (  # noqa: E402
    Placement,
    Replicate,
    Shard,
)

from physicsnemo.distributed._shard_redistribute import (  # noqa: E402
    ShardRedistribute,
)
from physicsnemo.distributed._shard_tensor_spec import (  # noqa: E402
    ShardTensorSpec,
    _infer_shard_tensor_spec_from_local_chunks,
    _stride_from_contiguous_shape_C_style,
)

aten = torch.ops.aten


class _ToTorchTensor(torch.autograd.Function):
    """Autograd function to convert a ShardTensor to a regular PyTorch tensor.

    This class handles the conversion from ShardTensor to torch.Tensor in both forward
    and backward passes, maintaining proper gradient flow.  Slices the ShardTensor
    to the local component only on the current rank.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: "ShardTensor",
        grad_placements: Optional[Sequence[Placement]] = None,
    ) -> torch.Tensor:
        """Convert ShardTensor to torch.Tensor in forward pass.

        Args:
            ctx: Autograd context for saving tensors/variables for backward
            input: ShardTensor to convert
            grad_placements: Optional sequence of placements to use for gradients

        Returns:
            torch.Tensor: Local tensor representation of the ShardTensor
        """
        ctx.shard_tensor_spec = input._spec
        ctx.grad_placements = grad_placements
        local_tensor = input._local_tensor

        # JUST LIKE DTENSOR:
        # We need to return a fresh Tensor object there as autograd metadata
        # will be inplaced into it. So we don't want to pollute the Tensor
        # object stored in the _local_tensor of this ShardTensor.
        return local_tensor.view_as(local_tensor)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple["ShardTensor", None]:
        """Convert gradient torch.Tensor back to ShardTensor in backward pass.

        Args:
            ctx: Autograd context containing saved tensors/variables from forward
            grad_output: Gradient tensor to convert back to ShardTensor

        Returns:
            Tuple containing:
            - ShardTensor gradient
            - None for grad_placements gradient (not needed)
        """
        shard_tensor_spec = ctx.shard_tensor_spec
        mesh = shard_tensor_spec.mesh
        if ctx.grad_placements is not None:
            if ctx.grad_placements != shard_tensor_spec.placements:
                grad_placements = ctx.grad_placements
                grad_sharding_shapes = "infer"
            else:
                # If the placements are the same as the input placements,
                # we reuse the sharding sizes from the input placements.
                grad_placements = ctx.grad_placements
                grad_sharding_shapes = shard_tensor_spec._sharding_shapes
        else:
            grad_placements = shard_tensor_spec.placements
            grad_sharding_shapes = shard_tensor_spec._sharding_shapes
        if grad_sharding_shapes is None:
            grad_sharding_shapes = "infer"
        # Generate a spec based on grad outputs and the expected placements:
        grad_tensor_spec = _infer_shard_tensor_spec_from_local_chunks(
            grad_output, mesh, grad_placements, grad_sharding_shapes
        )

        return (
            ShardTensor(
                grad_output, grad_tensor_spec, requires_grad=grad_output.requires_grad
            ),
            None,
        )


class _FromTorchTensor(torch.autograd.Function):
    """Autograd function for converting a torch.Tensor to a ShardTensor.

    This class handles the forward and backward passes for converting between
    torch.Tensor and ShardTensor types, maintaining gradient information.

    Global shape information is inferred using collective communication on
    the specified device mesh.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        local_input: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: Tuple[Placement, ...],
        sharding_shapes: Union[str, Dict[int, List[Tuple[int, ...]]]] = "chunk",
    ) -> "ShardTensor":
        """Convert a local torch.Tensor to a ShardTensor in forward pass.

        Args:
            ctx: Autograd context for saving tensors/variables for backward
            local_input: Local tensor to convert to ShardTensor
            device_mesh: Device mesh specifying process groups
            placements: Tuple of placement rules for sharding
            sharding_shapes: Controls how shard tensor spec is generated:
                - "chunk": Use torch.chunk shapes to infer shapes from global shape (no communication)
                - "infer": Use collective communication to infer shapes from mesh neighbors.
                - Manual dict mapping mesh dim to list of shard shapes: Use provided shapes.  Must pass on each rank!

        Returns:
            ShardTensor constructed from the local input tensor
        """
        ctx.previous_placement = placements
        ctx.previous_mesh = device_mesh

        # This function is simpler than the corresponding DTensor implementation on the surface
        # because under the hood, we have some logic here to infer the sharding shapes.
        shard_tensor_spec = _infer_shard_tensor_spec_from_local_chunks(
            local_input, device_mesh, placements, sharding_shapes
        )

        shard_tensor = ShardTensor(
            local_input,
            shard_tensor_spec,
            requires_grad=local_input.requires_grad,
        )

        return shard_tensor

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: "ShardTensor",
    ) -> Tuple[torch.Tensor, None, None]:
        """Convert gradient ShardTensor back to torch.Tensor in backward pass.

        Args:
            ctx: Autograd context containing saved tensors/variables from forward
            grad_output: Gradient ShardTensor to convert back to torch.Tensor

        Returns:
            Tuple containing:
            - Local tensor gradient
            - None for device_mesh gradient (not needed)
            - None for placements gradient (not needed)

        Raises:
            RuntimeError: If gradient tensor has different placement than original
        """
        previous_placement = ctx.previous_placement
        if grad_output.placements != previous_placement:
            # Automatically redistribute to the previous placement as long as it's not a partial.
            if not any(p.is_partial() for p in previous_placement):
                grad_output = grad_output.redistribute(
                    grad_output._spec.mesh, previous_placement
                )
            else:
                raise RuntimeError(
                    "Resharding gradients with partial placements not implemented"
                )

        return grad_output.to_local(), None, None, None


class ShardTensor(DTensor):
    """
    A class similar to pytorch's native DTensor but with more
    flexibility for uneven data sharding.

    Leverages very similar API to DTensor (identical, where possible)
    but deliberately tweaking routines to avoid implicit assumptions
    about tensor sharding.

    The key differences from DTensor are:
    - Supports uneven sharding where different ranks can have different local tensor sizes
    - Tracks and propagates shard size information across operations
    - Handles redistribution of unevenly sharded tensors
    - Provides custom collective operations optimized for uneven sharding

    Like DTensor, operations are dispatched through PyTorch's dispatcher system.
    Most operations work by:
    1. Converting inputs to local tensors
    2. Performing the operation locally
    3. Constructing a new ShardTensor with appropriate sharding spec
    4. Handling any needed communication between ranks

    The class provides methods for:
    - Converting to/from local tensors
    - Redistributing between different sharding schemes
    - Performing collective operations like all_gather and reduce_scatter
    - Basic tensor operations that maintain sharding information
    """

    _local_tensor: torch.Tensor
    _spec: ShardTensorSpec
    __slots__ = ["_local_tensor", "_spec"]

    # For torch.ops.aten operators (low-level dispatch)
    _dispatch_registry: Dict[torch._ops.OpOverload, Callable] = {}

    # For Python-level functions (torch.mean, tensor.mean, etc.)
    _function_registry: Dict[Callable, Callable] = {}

    # For custom functions registered with PyTorch,
    # it is sometimes necessary to match by name.
    # For instance, if you declare an op with
    #
    # @torch.library.custom_op(
    #    "module::function_name", mutates_args=()
    # )
    # def function_external_to_torch(
    #
    # Then, you likely want to register the handler with
    #
    # ShardTensor.register_named_function_handler("module.function_name.default", handler)
    _named_function_registry: dict[str, Callable] = {}

    # Upon construction of any ShardTensor objects, this will be set to true.
    # Wrappers are triggered dynamically, so the wrapping will be pass-through
    # exclusively until true.
    _enable_shard_patches: bool = False

    @classmethod
    def patches_enabled(cls) -> bool:
        """
        Whether to enable patches for this class.

        Default is False, but can be changed by the user.
        """
        return cls._enable_shard_patches

    @classmethod
    def register_dispatch_handler(
        cls, op: torch._ops.OpOverload, handler: Callable
    ) -> None:
        """Register a handler for a specific PyTorch operator in the dispatch system."""
        cls._dispatch_registry[op] = handler

    @classmethod
    def register_function_handler(cls, func: Callable, handler: Callable) -> None:
        """Register a handler for a Python-level function or method."""
        cls._function_registry[func] = handler

    @classmethod
    def register_named_function_handler(cls, func_name: str, handler: Callable) -> None:
        """Register a named function that has been named via torch.library.custom_op"""
        cls._named_function_registry[func_name] = handler

    @staticmethod
    def __new__(
        cls,
        local_tensor: torch.Tensor,
        spec: ShardTensorSpec,
        *,
        requires_grad: bool,
    ) -> "ShardTensor":
        """
        Construct a new Shard Tensor from a local tensor, device mesh, and placement.

        Note that unlike DTensor, ShardTensor will automatically collect the Shard size
        information from all participating devices. This is to enable uneven and
        dynamic sharding.

        Heavily derived from torch DTensor

        Args:
            local_tensor: Local tensor to use as the data
            spec: ShardTensorSpec defining the sharding scheme
            requires_grad: Whether the tensor requires gradients

        Returns:
            A new ShardTensor instance
        """
        if local_tensor.requires_grad and not requires_grad:
            warn(
                "To construct a new ShardTensor from torch.Tensor, "
                "it's recommended to use local_tensor.detach() and "
                "make requires_grad consistent."
            )

        if spec.tensor_meta is None:
            raise ValueError("TensorMeta should not be None!")

        # Check the sharding information is known:
        ret = torch.Tensor._make_wrapper_subclass(
            cls,
            spec.tensor_meta.shape,
            strides=spec.tensor_meta.stride,
            dtype=local_tensor.dtype,
            device=local_tensor.device,
            layout=local_tensor.layout,
            requires_grad=requires_grad,
        )

        ret._spec = spec
        ret._local_tensor = local_tensor

        cls._enable_shard_patches = True

        return ret

    def __repr__(self) -> str:
        return f"ShardTensor(local_tensor={self._local_tensor}, device_mesh={self._spec.mesh}, placements={self._spec.placements})"

    @classmethod
    def from_dtensor(cls, dtensor: DTensor) -> "ShardTensor":
        """
        Convert a DTensor to a ShardTensor.  We assume the DTensor is properly constructed.

        Args:
            dtensor: DTensor to convert

        Returns:
            Equivalent ShardTensor
        """
        # Always ensure sharding is turned on:
        cls._enable_shard_patches = True

        # DTensor is locked to sharding a tensor according to chunk format.
        # We can use that to infer sharding sizes with no communication.

        # Create the spec by inferring the sharding sizes from the DTensor:
        spec = _infer_shard_tensor_spec_from_local_chunks(
            dtensor._local_tensor,
            dtensor._spec.mesh,
            dtensor._spec.placements,
            sharding_shapes="chunk",
            global_shape=dtensor.shape,
        )

        return ShardTensor.__new__(
            cls,
            local_tensor=dtensor._local_tensor,
            spec=spec,
            requires_grad=dtensor.requires_grad,
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs={}):
        with annotate(f"__torch_function___{func.__name__}"):
            # Check for overrides:
            if func in cls._function_registry and cls._enable_shard_patches:
                res = cls._function_registry[func](func, types, args, kwargs)
                return res
            elif (
                str(func) in cls._named_function_registry and cls._enable_shard_patches
            ):
                res = cls._named_function_registry[str(func)](func, types, args, kwargs)
                return res
            # Fall back to the default behavior:
            return super().__torch_function__(func, types, args, kwargs)

    @classmethod
    @torch._disable_dynamo
    @profile
    def __torch_dispatch__(
        cls,
        func: torch._ops.OpOverload,
        types: Tuple[type, ...],
        args: Tuple[object, ...] = (),
        kwargs: Optional[Dict[str, object]] = None,
    ) -> Union["ShardTensor", Iterable["ShardTensor"], object]:
        with annotate(f"__torch_dispatch___{func.__name__}"):
            # Leverage DTensor Dispatch as much as possible, but, enable
            # the ability to operate on this output in the future:
            if func in cls._dispatch_registry:
                res = cls._dispatch_registry[func](*args, **kwargs)
                return res

            # We assume that if we reach this point, the operator has not been
            # intercepted by a wrapper or in the registry.  So the DTensor
            # default behavior is likely to be correct.

            if func == aten.view.default:
                # For view, we need input tensors to be contiguous:
                for arg in args:
                    if isinstance(arg, ShardTensor) or isinstance(arg, DTensor):
                        if not arg._local_tensor.is_contiguous():
                            arg._local_tensor = arg._local_tensor.contiguous()

            dispatch_res = DTensor._op_dispatcher.dispatch(func, args, kwargs or {})

            # Return a shard tensor instead of a dtensor.
            def _convert_dtensor_with_input_check(dtensor, input_args):
                """
                This function searches the input for ShardTensors that match output shapes.
                It prevents collectives, since we can copy the sharding shapes for irregular shards.
                The idea here is that, if the global shape is unchanged, and
                the placements are unchanged, the sharding shapes should be unchanged.

                If no matches are found, it falls back to inference based on DTensor.

                This is only used when we already went back through the DTensor dispatch.
                """
                # Check if this matches any input ShardTensor
                for arg in input_args:
                    if (
                        isinstance(arg, ShardTensor)
                        and dtensor._spec.tensor_meta == arg._spec.tensor_meta
                        and dtensor._spec.placements == arg._spec.placements
                    ):
                        return ShardTensor.__new__(
                            ShardTensor,
                            local_tensor=dtensor._local_tensor,
                            spec=arg._spec,
                            requires_grad=dtensor.requires_grad,
                        )

                # Fall back to default conversion
                return ShardTensor.from_dtensor(dtensor)

            if isinstance(dispatch_res, DTensor):
                return _convert_dtensor_with_input_check(dispatch_res, args)

            if isinstance(dispatch_res, Iterable):
                return type(dispatch_res)(
                    _convert_dtensor_with_input_check(d, args)
                    if isinstance(d, DTensor)
                    else d
                    for d in dispatch_res
                )

            return dispatch_res

    @staticmethod
    def from_local(
        local_tensor: torch.Tensor,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
        sharding_shapes: Union[str, Dict[int, List[Tuple[int, ...]]]] = "infer",
    ) -> "ShardTensor":
        """
        Generate a new ShardTensor from local torch tensors. Uses
        device mesh and placements to infer global tensor properties.

        No restriction is made on forcing tensors to have equal shapes
        locally. Instead, the requirement is that tensor shapes could
        be concatenated into a single tensor according to the placements.

        Args:
            local_tensor: Local chunk of tensor. All participating tensors must be
                of the same rank and concatable across the mesh dimensions
            device_mesh: Target Device Mesh, if not specified will use the current mesh
            placements: Target placements, must have same number of elements as device_mesh.ndim
            sharding_shapes: Controls how shard tensor spec is generated:
                - "chunk": Use torch.chunk shapes to infer shapes from global shape (no communication)
                - "infer": Use collective communication to infer shapes from mesh neighbors.
                - Manual dict mapping mesh dim to list of shard shapes: Use provided shapes.  Must pass on each rank!
        Returns:
            A new ShardTensor instance
        """

        # this turns on shard patches globally for this process.
        ShardTensor._enable_shard_patches = True

        # This implementation follows the pytorch DTensor Implementation Closely.
        device_mesh = device_mesh or _mesh_resources.get_current_mesh()
        device_type = device_mesh.device_type

        # convert the local tensor to desired device base on device mesh's device_type
        if device_type != local_tensor.device.type and not local_tensor.is_meta:
            local_tensor = local_tensor.to(device_type)

        # set default placements to replicated if not specified
        if placements is None:
            placements = [Replicate() for _ in range(device_mesh.ndim)]
        else:
            placements = list(placements)
            for idx, placement in enumerate(placements):
                # normalize shard dim to be positive
                if placement.is_shard():
                    placement = cast(Shard, placement)
                    if placement.dim < 0:
                        placements[idx] = Shard(placement.dim + local_tensor.ndim)

        # `from_local` is differentiable, and the gradient of the dist tensor this function
        # created should flow back the gradients to the local_tensor, so we call an autograd
        # function to construct the dist tensor instead.
        return _FromTorchTensor.apply(  # pyre-ignore[16]: autograd func
            local_tensor,
            device_mesh,
            tuple(placements),
            sharding_shapes,
        )

    def offsets(self, mesh_dim: Optional[int] = None) -> List[int]:
        """
        Get offsets of shards along a mesh dimension.

        Args:
            mesh_dim: Mesh dimension to get offsets for. If None, returns all offsets.

        Returns:
            List of offsets for shards along specified dimension
        """
        return self._spec.offsets(mesh_dim)

    def redistribute(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
        *,
        async_op: bool = False,
    ) -> "ShardTensor":
        """
        Redistribute tensor across device mesh with new placement scheme.
        Like DTensor redistribute but uses custom layer for shard redistribution.

        Args:
            device_mesh: Target device mesh. Uses current if None.
            placements: Target placement scheme. Required.
            async_op: Whether to run asynchronously

        Returns:
            Redistributed ShardTensor

        Raises:
            RuntimeError: If placements not specified or invalid
        """

        # if device_mesh is not specified, use the current device_mesh
        device_mesh = device_mesh or self.device_mesh
        # raise error if new placements not specified
        if placements is None:
            raise RuntimeError("placements is needed for redistribute!")

        placements = list(placements)
        for i, placement in enumerate(placements):
            if placement.is_partial():
                raise RuntimeError(
                    "Can not redistribute to Partial, redistributing to Partial is for internal use only!"
                )
            elif isinstance(placement, Shard) and placement.dim < 0:
                # normalize shard dim to be positive
                placements[i] = Shard(placement.dim + self.ndim)
        placements = tuple(placements)

        return ShardRedistribute.apply(self, device_mesh, placements, async_op)

    def to_local(
        self, *, grad_placements: Optional[Sequence[Placement]] = None
    ) -> torch.Tensor:
        """
        Get local tensor from this ShardTensor.

        Args:
            grad_placements: Future layout of gradients. Optional.

        Returns:
            Local torch.Tensor. Shape may vary between ranks for sharded tensors.
        """

        if not torch.is_grad_enabled():
            return self._local_tensor

        if grad_placements is not None and not isinstance(grad_placements, tuple):
            grad_placements = tuple(grad_placements)

        return _ToTorchTensor.apply(self, grad_placements)

    def full_tensor(
        self, *, grad_placements: Optional[Sequence[Placement]] = None
    ) -> torch.Tensor:
        """
        Need to re-implement here to ensure a ShardTensor is used as the output
        of redistribute.
        """

        redist_res = self.redistribute(
            placements=[Replicate()] * self.device_mesh.ndim, async_op=False
        )
        return _ToTorchTensor.apply(redist_res, grad_placements)

    def backward(self, *args, **kwargs):
        """
        Backward pass for ShardTensor.

        This method is used to perform the backward pass for a ShardTensor.
        It handles the redistribution of the tensor to the desired placements
        and then calls the backward pass on the local tensor.
        """

        # Before calling backward, we need to resolve any partial placements.
        new_placements = []
        # grad_placements = []
        needs_redistribute = False
        for i, placement in enumerate(self._spec.placements):
            if placement.is_partial():
                new_placements.append(Replicate())
                # grad_placements.append(Shard(i))
                needs_redistribute = True
            else:
                new_placements.append(placement)
                # grad_placements.append(placement)

        if needs_redistribute:
            self = self.redistribute(placements=new_placements)

        return self.to_local().backward(*args, **kwargs)


def scatter_tensor(
    tensor: torch.Tensor,
    global_src: int,
    mesh: DeviceMesh,
    placements: Tuple[Placement, ...],
    global_shape: Optional[torch.Size] = None,
    dtype: Optional[torch.dtype] = None,
    requires_grad: bool = False,
) -> "ShardTensor":
    """
    Take a tensor from source rank and distribute it across devices on the mesh according to placements.

    This function takes a tensor that exists on a single source rank and distributes it across
    a device mesh according to the specified placement scheme. For multi-dimensional meshes,
    it performs a flattened scatter operation before constructing the sharded tensor.

    Args:
        tensor: The tensor to distribute, must exist on source rank
        global_src: Global rank ID of the source process
        mesh: Device mesh defining the process topology
        placements: Tuple of placement specifications defining how to distribute the tensor

    Returns:
        ShardTensor: The distributed tensor with specified placements

    Raises:
        ValueError: If global_src is not an integer or not in the mesh
    """
    dm = DistributedManager()

    if not isinstance(global_src, int):
        raise ValueError("Global source must be an integer rank")
    if global_src not in mesh.mesh:
        raise ValueError("Please specify a tensor source in this mesh")

    is_src = dm.rank == global_src

    # For multi-dimensional meshes, we use a flattened process group
    mesh_group = dm.get_mesh_group(mesh)

    # Broadcast tensor metadata from source
    if global_shape is None or dtype is None:
        if dm.rank == global_src:
            meta = [TensorMeta(tensor.shape, tensor.stride(), tensor.dtype)]
        else:
            meta = [None]

        dist.broadcast_object_list(meta, src=global_src, group=mesh_group)

        local_meta = meta[0]
    else:
        stride = _stride_from_contiguous_shape_C_style(global_shape)
        local_meta = TensorMeta(global_shape, stride, dtype)

    # This needs to be optimized, but I want to get the whole pipeline optimized first.
    # This only gets done when scatter_tensor is called and it should be relatively small
    # in full applications.

    # What isn't optimmized?  Broadcasting the full tensor when placement is likely
    # Shard on at least one mesh dimension.  It would be more efficient to iteratively
    # scatter along Shard dimensions.  BUT, the focus is on performance of full applications
    # and this is a once-per-iteration cost.

    # Broadcast the tensor to all ranks
    if tensor is None and not is_src:
        # Tensor is allowed to be none if not on the root rank
        tensor = torch.empty(local_meta.shape, dtype=local_meta.dtype, device=dm.device)

    dist.broadcast(tensor, src=global_src, group=mesh_group)

    # Create a fully-replicated spec:
    spec = ShardTensorSpec(
        mesh=mesh,
        placements=[Replicate() for _ in range(mesh.ndim)],
        tensor_meta=local_meta,
        _sharding_shapes={},
    )

    # Make a "fully-replicated" tensor on all ranks:
    st = ShardTensor.__new__(
        ShardTensor,
        local_tensor=tensor,
        spec=spec,
        requires_grad=requires_grad,
    )

    # Redistribute the tensor to the desired placements:
    st = st.redistribute(mesh, placements, async_op=False)
    # This is an unoptimal step but is functional:
    if requires_grad:
        st = st.detach()
        st.requires_grad = True

    return st
