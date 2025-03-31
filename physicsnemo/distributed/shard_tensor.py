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
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast
from warnings import warn

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, _mesh_resources

from physicsnemo.distributed import DistributedManager
from physicsnemo.distributed.utils import compute_split_shapes, split_tensor_along_dim
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
)


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

        grad_placements = ctx.grad_placements or shard_tensor_spec.placements

        # Generate a spec based on grad outputs and the expected placements:
        grad_tensor_spec = _infer_shard_tensor_spec_from_local_chunks(
            grad_output, mesh, grad_placements
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
    ) -> "ShardTensor":
        """Convert a local torch.Tensor to a ShardTensor in forward pass.

        Args:
            ctx: Autograd context for saving tensors/variables for backward
            local_input: Local tensor to convert to ShardTensor
            device_mesh: Device mesh specifying process groups
            placements: Tuple of placement rules for sharding

        Returns:
            ShardTensor constructed from the local input tensor
        """
        ctx.previous_placement = placements
        ctx.previous_mesh = device_mesh
        # This function is simpler than the corresponding DTensor implementation on the surface
        # because under the hood, we always do checks here.

        shard_tensor_spec = _infer_shard_tensor_spec_from_local_chunks(
            local_input, device_mesh, placements
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
            raise RuntimeError("Resharding gradients not yet implemented")

        return grad_output.to_local(), None, None


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

    _function_registry: Dict[torch._ops.OpOverload, callable] = {}

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
    def register_function_handler(cls, func: torch._ops.OpOverload, handler: callable):
        """
        Register a custom handler for a specific function.

        Args:
            func: The function to intercept.
            handler: The custom handler to call instead of the default dispatch.
        """
        cls._function_registry[func] = handler

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
    def from_dtensor(
        cls, dtensor: DTensor, force_sharding_inference: bool = False
    ) -> "ShardTensor":
        """
        Convert a DTensor to a ShardTensor.

        Args:
            dtensor: DTensor to convert

        Returns:
            Equivalent ShardTensor
        """

        # DTensor is locked to sharding a tensor according to chunk format.
        # We can use that to infer sharding sizes with no communication.

        mesh = dtensor._spec.mesh
        placements = dtensor._spec.placements

        if force_sharding_inference:
            shard_tensor_spec = _infer_shard_tensor_spec_from_local_chunks(
                dtensor._local_tensor, dtensor._spec.mesh, dtensor._spec.placements
            )
            return ShardTensor.__new__(
                cls,
                local_tensor=dtensor._local_tensor,
                spec=shard_tensor_spec,
                requires_grad=dtensor.requires_grad,
            )
        else:
            temp_sharding_sizes = {}
            for i in range(mesh.ndim):
                if isinstance(placements[i], Shard):
                    # Compute the chunk size for this dimension:
                    input_dim = dtensor.shape[placements[i].dim]
                    chunked_shapes = compute_split_shapes(input_dim, mesh.size(i))
                    # This needs to be a tuple of torch.Size

                    temp_sharding_sizes[i] = chunked_shapes

            # To create the full, final sharding shapes, we update the global shape with
            sharding_sizes = {}
            # Initialize sharding_sizes with same keys as temp_sharding_sizes
            # Each value is a list of torch.Size equal to mesh size for that dimension
            for mesh_dim in temp_sharding_sizes.keys():
                placement = placements[mesh_dim]
                # We should not have the mesh dim in this dict if it wasn't sharded above:
                tensor_dim = placement.dim

                sharding_sizes[mesh_dim] = [
                    torch.Size(dtensor.shape) for _ in temp_sharding_sizes[mesh_dim]
                ]
                # For each shard along this mesh dimension
                for i, shard_size in enumerate(temp_sharding_sizes[mesh_dim]):
                    # Replace size at sharded dim with actual shard size
                    updated_shard_size = torch.Size(
                        tuple(
                            (
                                shard_size if j == tensor_dim else s
                                for j, s in enumerate(sharding_sizes[mesh_dim][i])
                            )
                        )
                    )
                    sharding_sizes[mesh_dim][i] = updated_shard_size

            # Cast to tuples:
            for mesh_dim in temp_sharding_sizes.keys():
                sharding_sizes[mesh_dim] = tuple(sharding_sizes[mesh_dim])

            spec = ShardTensorSpec(
                mesh=dtensor._spec.mesh,
                placements=dtensor._spec.placements,
                tensor_meta=dtensor._spec.tensor_meta,
                _sharding_sizes=sharding_sizes,  # Leave this to none for a lazy init and assume it's not breaking to make this cast.
                _local_shape=dtensor._local_tensor.shape,
            )

            cls._enable_shard_patches = True

            return ShardTensor.__new__(
                cls,
                local_tensor=dtensor._local_tensor,
                spec=spec,
                requires_grad=dtensor.requires_grad,
            )

    @classmethod
    @torch._disable_dynamo
    def __torch_dispatch__(
        cls,
        func: torch._ops.OpOverload,
        types: Tuple[type, ...],
        args: Tuple[object, ...] = (),
        kwargs: Optional[Dict[str, object]] = None,
    ) -> Union["ShardTensor", Iterable["ShardTensor"], object]:
        # Leverage DTensor Dispatch as much as possible, but, enable
        # the ability to operate on this output in the future:

        if func in cls._function_registry:
            return cls._function_registry[func](*args, **kwargs)

        dispatch_res = DTensor._op_dispatcher.dispatch(func, args, kwargs or {})

        # dispatch_res = ShardTensor._op_dispatcher.dispatch(func, args, kwargs or {})

        # Return a shard tensor instead of a dtensor.
        # ShardTensor inherits from DTensor and can lazy-init from for efficiency
        if isinstance(dispatch_res, DTensor):
            return ShardTensor.from_dtensor(dispatch_res, force_sharding_inference=True)

        if isinstance(dispatch_res, Iterable):
            return type(dispatch_res)(
                ShardTensor.from_dtensor(d, force_sharding_inference=True)
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
        infer_shape: Optional[bool] = True,
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
            infer_shape: If False, assumes even distribution like DTensor. Default True.

        Returns:
            A new ShardTensor instance
        """

        if infer_shape:

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
            ShardTensor._enable_shard_patches = True
            return _FromTorchTensor.apply(  # pyre-ignore[16]: autograd func
                local_tensor,
                device_mesh,
                tuple(placements),
            )
        else:
            ShardTensor._enable_shard_patches = True
            return ShardTensor.from_dtensor(
                DTensor.from_local(local_tensor, device_mesh, placements)
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


def scatter_tensor(
    tensor: torch.Tensor,
    global_src: int,
    mesh: DeviceMesh,
    placements: Tuple[Placement, ...],
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

    # For multi-dimensional meshes, create a flattened process group
    if mesh.ndim != 1:
        global_ranks = mesh.mesh.flatten().tolist()
        mesh_group = dist.new_group(ranks=global_ranks, use_local_synchronization=True)
    else:
        mesh_group = mesh.get_group()

    # Broadcast tensor metadata from source
    axis_rank = dist.get_rank(mesh_group)
    if dm.rank == global_src:
        meta = [TensorMeta(tensor.shape, tensor.stride(), tensor.dtype)]
    else:
        meta = [None]

    dist.broadcast_object_list(meta, src=global_src, group=mesh_group)
    dist.barrier(group=mesh_group)
    local_meta = meta[0]

    # Cast the shape to a list to be mutable:
    local_shape = list(local_meta.shape)

    if is_src:
        chunks = [tensor]
    else:
        chunks = None

    # Split tensor according to shard placements
    for dim, placement in enumerate(placements):
        if isinstance(placement, Shard):
            tensor_dim = placement.dim
            axis_rank = dist.get_rank(group=mesh.get_group(dim))
            axis_size = dist.get_world_size(group=mesh.get_group(dim))

            sections = compute_split_shapes(local_shape[tensor_dim], axis_size)

            if is_src:
                new_chunks = []
                for t in chunks:
                    new_chunks += split_tensor_along_dim(t, tensor_dim, axis_size)
                chunks = new_chunks
            local_shape[tensor_dim] = sections[axis_rank]

    # Convert the shape back to a tuple:
    local_shape = tuple(local_shape)

    # Allocate local tensor
    local_chunk = torch.empty(
        local_shape,
        dtype=local_meta.dtype,
        device=torch.device(f"cuda:{dm.local_rank}"),
    )

    # Scatter chunks across mesh
    dist.scatter(local_chunk, chunks, src=global_src, group=mesh_group)

    # Construct ShardTensor from local tensor
    return ShardTensor.from_local(
        local_tensor=local_chunk,
        device_mesh=mesh,
        placements=placements,
    )
