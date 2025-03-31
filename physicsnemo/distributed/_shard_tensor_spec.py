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

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from physicsnemo.utils.version_check import check_module_requirements

check_module_requirements("physicsnemo.distributed.shard_tensor")


from torch.distributed.tensor._dtensor_spec import (  # noqa: E402
    DTensorSpec,
    TensorMeta,
)
from torch.distributed.tensor.placement_types import (  # noqa: E402
    Placement,
    Shard,
)


@dataclass(kw_only=True)
class ShardTensorSpec(DTensorSpec):
    """A distributed tensor specification that tracks sharding information.

    This class extends DTensorSpec to include information about global placements of shards.
    This is useful when the tensor is distributed in an uneven or unexpected way.

    Attributes
    ----------
    _local_shape : Optional[torch.Size]
        The shape of the local shard of the tensor
    _sharding_sizes : Optional[dict[int, Tuple[torch.Size, ...]]]
        Mapping from mesh dimension to shard sizes. Keys are mesh dimensions,
        values are tuples of torch.Size representing shard shapes along that dimension.
        Shard sizes are only tracked along the sharded dimensions, not replicated dimensions.
    """

    _local_shape: Optional[torch.Size] = field(default_factory=lambda: None)
    # This dict is a mapping from the mesh dimension to the shard sizes, _not_ the tensor index
    _sharding_sizes: Optional[dict[int, Tuple[torch.Size, ...]]] = field(
        default_factory=lambda: None
    )

    def _hash_impl(self) -> int:
        """Implements hashing for the spec including sharding information.

        Based on DTensor hash spec but explicitly including shard size information.

        Returns
        -------
        int
            Hash value incorporating mesh, placements, tensor metadata and sharding sizes
        """

        hash_items = []
        hash_items.append(self.mesh)
        hash_items.append(self.placements)

        if self.tensor_meta is not None:
            hash_items.append(self.tensor_meta.shape)
            hash_items.append(self.tensor_meta.stride)
            hash_items.append(self.tensor_meta.dtype)
        if self._sharding_sizes is not None:
            hash_items.append(tuple(sorted(self._sharding_sizes.items())))
        hash_tuple = tuple(hash_items)
        return hash(hash_tuple)

    def __hash__(self) -> int:
        """
        Just like the parent class: compute the hash lazily.
        See torch.distributed.tensor._dtensor_spec.py for more information.
        """
        if self._hash is None:
            self._hash = self._hash_impl()
        return self._hash

    def sharding_sizes(
        self, mesh_dim: Optional[int] = None
    ) -> dict[int, Tuple[torch.Size, ...]] | Tuple[torch.Size, ...]:
        """Get the sizes of shards along specified mesh dimensions.

        Parameters
        ----------
        mesh_dim : Optional[int]
            If provided, return sizes only for this mesh dimension

        Returns
        -------
        dict[int, Tuple[torch.Size, ...]] | Tuple[torch.Size, ...]
            Dictionary of shard sizes by mesh dim, or tuple of sizes for specific dim
        """
        if self._sharding_sizes is None:
            shard_shapes_by_dim, global_shape = _all_gather_shard_shapes(
                self._local_shape, self.placements, self.mesh
            )
            self._sharding_sizes = shard_shapes_by_dim
            self.tensor_meta = self.tensor_meta._replace(shape=global_shape)
        if mesh_dim is not None:
            if mesh_dim in self._sharding_sizes:
                return self._sharding_sizes[mesh_dim]
        return self._sharding_sizes

    def __eq__(self, other: object) -> bool:
        """Check if two ShardTensorSpecs are equal.

        Parameters
        ----------
        other : object
            The other object to compare to
        """
        if not isinstance(other, ShardTensorSpec):
            return False
        if not super().__eq__(other):
            return False
        if self._sharding_sizes != other._sharding_sizes:
            return False
        return True

    @property
    def local_shape(self) -> torch.Size:
        """Get the shape of the local shard.

        Returns
        -------
        torch.Size
            Shape of local tensor shard

        Raises
        ------
        Exception
            If local shape has not been set
        """
        if self._local_shape is None:
            raise Exception("Missing local shape!")
        return self._local_shape

    @local_shape.setter
    def local_shape(self, value: torch.Size) -> None:
        """Set the local shard shape.

        Parameters
        ----------
        value : torch.Size
            Shape to set for local shard

        Raises
        ------
        TypeError
            If value is not a torch.Size
        """
        if not isinstance(value, torch.Size):
            raise TypeError("Local shape must be instance of torch.Size")
        self._local_shape = value

    def offsets(self, mesh_dim: Optional[int] = None) -> Tuple[int, ...] | int:
        """Calculate offsets for the local shard within the global tensor.

        Returns the effective offset of this tensor along sharded dimensions, as if it
        was all collected into one device and you wanted to slice it
        to recover the local slice.

        Parameters
        ----------
        mesh_dim : Optional[int]
            If provided, return offset only for this mesh dimension

        Returns
        -------
        Tuple[int, ...] | int
            Tuple of offsets for each mesh dimension, or single offset if mesh_dim specified
        """
        offsets = []
        for loop_mesh_dim in range(self.mesh.ndim):
            coord = self.mesh.get_coordinate()[loop_mesh_dim]
            placement = self.placements[loop_mesh_dim]
            # If the placement is not shard, offset is 0:
            if isinstance(placement, Shard):
                shards = self._sharding_sizes[loop_mesh_dim]
                tensor_dim = placement.dim
                o = sum([s[tensor_dim] for s in shards[:coord]])
                offsets.append(o)
            else:
                offsets.append(0)

        if mesh_dim is not None:
            return offsets[mesh_dim]

        return tuple(offsets)  # Fixed: Return tuple instead of list


def _stride_from_contiguous_shape_C_style(
    shape: Tuple[
        int,
    ]
) -> Tuple[int]:
    """
    Compute and return the stride from a tensor shape, assuming it is
    both contiguous and laid out in C-style

    Parameters
    ----------
    shape : Tuple[int,]
        input shape as Tuple or torch.Size

    Returns
    -------
    Tuple[int]
        list of strides of same length as input
    """

    # For scalars, stride is empty:
    if len(shape) == 0:
        return ()

    # Implicitly, assume sharding only happens over specified placements
    # To compute strides, we make the assumption that the tensors are in the "C" style layout (default)
    # So, all strides at the deepest level are 1.
    stride = [
        1,
    ]
    for axis_len in reversed(shape[1:]):
        next_stride = stride[-1] * axis_len
        stride.append(next_stride)

    stride = tuple(reversed(stride))
    return stride


def _all_gather_shard_shapes(
    local_shape: Tuple[int],
    placements: Tuple[Placement],
    target_mesh: DeviceMesh,
):

    shard_shapes_by_dim = {}
    global_shape = [s for s in local_shape]
    # We start by assuming the global shape is the local shape and fix it on sharded axes
    for mesh_axis, placement in enumerate(placements):

        if isinstance(placement, Shard):
            tensor_dim = placement.dim

            local_group = target_mesh.get_group(mesh_axis)

            local_size = dist.get_world_size(group=local_group)

            all_shapes = [torch.Size()] * local_size

            # First, allgather the dimensions of each tensor to each rank:
            # Possible collective of CPU-based objects!  Could be slow if using separate hosts!
            dist.all_gather_object(all_shapes, local_shape, local_group)

            # Check that all shapes are the same rank:
            if not all([len(local_shape) == len(all_s) for all_s in all_shapes]):
                raise ValueError(
                    "Rank mismatch detected when attempting to infer shapes and sizes"
                )

            # Every dimension must be equal for this list, along the sharded axis
            for d in range(len(local_shape)):
                if d == tensor_dim:
                    continue  # skip the sharded dimension
                if not all([local_shape[d] == all_s[d] for all_s in all_shapes]):
                    raise ValueError(
                        f"Dimension mismatch detected at non-sharded dimension {d}. "
                        "All local shapes must match except along sharded dimension."
                    )

            # Build a list of local torch.Size on this axis for each shard to store:
            local_meta = tuple(
                # torch.Size(tuple(s)) for s in zip(all_shapes)
                all_shapes
            )

            shard_shapes_by_dim[mesh_axis] = local_meta

            # To infer the global shape _for this axis_,
            # we have to loop over each axis in the rank list
            # To check what placement is there.
            # This assumes full sharding:
            global_shape[tensor_dim] = sum([all_s[tensor_dim] for all_s in all_shapes])

    return shard_shapes_by_dim, tuple(global_shape)


def _infer_shard_tensor_spec_from_local_chunks(
    local_chunk: torch.Tensor,
    target_mesh: DeviceMesh,
    placements: Tuple[Placement, ...],
) -> ShardTensorSpec:
    """
    Use local sizes, target mesh, and specified placements to build a
    ShardTensorSpec.  Performs checks that all local tensors are compatible

    Parameters
    ----------
    local_chunk : torch.Tensor
        local tensor to be used as a shard of a global tensor
    target_mesh : DeviceMesh
        Device mesh object to build this ShardTensor on
    placements : Tuple[Placement, ...]
        Specified placements of this tensor

    Returns
    -------
    ShardTensorSpec
        Specification to be used in creating a ShardTensor.  Key feature
        of this spec is that each ShardTensor knows the shape and size of
        other shards, and can compute global offsets and reductions properly
    """

    # # Only accept sharding placements (not replications or partial (aka pending))
    # if not all([p.is_shard() for p in placements]):
    #     raise ValueError(
    #         "Shard Tensor will only infer shape and strides for sharded tensors,"
    #         "for replication use DTensor"
    #     )

    # Need to infer the placements on each dimension of the mesh.
    if len(placements) != target_mesh.ndim:
        raise ValueError("Mesh dimension must match placements length")

    local_shape = local_chunk.shape

    shard_shapes_by_dim, global_shape = _all_gather_shard_shapes(
        local_shape,
        placements,
        target_mesh,
    )

    stride = _stride_from_contiguous_shape_C_style(global_shape)

    # # Finally, build a tensor spec to return:
    global_meta = TensorMeta(
        shape=tuple(global_shape), stride=stride, dtype=local_chunk.dtype
    )
    # all_shard_meta = local_meta

    sharding_sizes = {dim: tuple(s) for dim, s in shard_shapes_by_dim.items()}

    return ShardTensorSpec(
        mesh=target_mesh,
        placements=placements,
        tensor_meta=global_meta,
        _local_shape=local_shape,
        _sharding_sizes=sharding_sizes,
    )
