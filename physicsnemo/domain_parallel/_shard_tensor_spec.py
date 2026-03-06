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

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import (
    DTensorSpec,
    TensorMeta,
)
from torch.distributed.tensor.placement_types import (
    Placement,
    Shard,
)

from physicsnemo.distributed.utils import compute_split_shapes


@dataclass(kw_only=True)
class ShardTensorSpec(DTensorSpec):
    r"""A distributed tensor specification that tracks sharding information.

    This class extends ``DTensorSpec`` to include information about global
    placements of shards. This is useful when the tensor is distributed in
    an uneven or unexpected way.

    Attributes
    ----------
    _local_shape : Optional[torch.Size]
        The shape of the local shard of the tensor.
    _sharding_shapes : Optional[dict[int, Tuple[torch.Size, ...]]]
        Mapping from mesh dimension to shard shapes. Keys are mesh dimensions,
        values are tuples of ``torch.Size`` representing shard shapes along
        that dimension. Shard shapes are only tracked along the sharded
        dimensions, not replicated dimensions.
    """

    _local_shape: torch.Size | None = field(default_factory=lambda: None)
    # This dict is a mapping from the mesh dimension to the shard shapes, _not_ the tensor index
    _sharding_shapes: dict[int, tuple[torch.Size, ...]] | None = field(
        default_factory=lambda: None
    )

    def _hash_impl(self) -> int:
        r"""Implement hashing for the spec including sharding information.

        Based on ``DTensor`` hash spec but explicitly including shard size
        information.

        Returns
        -------
        int
            Hash value incorporating mesh, placements, tensor metadata, and
            sharding shapes.
        """

        hash_items = []
        hash_items.append(self.mesh)
        hash_items.append(self.placements)

        if self.tensor_meta is not None:
            hash_items.append(self.tensor_meta.shape)
            hash_items.append(self.tensor_meta.stride)
            hash_items.append(self.tensor_meta.dtype)
        if self._sharding_shapes is not None:
            hash_items.append(tuple(sorted(self._sharding_shapes.items())))
        hash_tuple = tuple(hash_items)
        return hash(hash_tuple)

    def __hash__(self) -> int:
        r"""Compute the hash lazily.

        Just like the parent class, the hash is computed lazily and cached.
        See ``torch.distributed.tensor._dtensor_spec.py`` for more information.

        Returns
        -------
        int
            The hash value for this spec.
        """
        if self._hash is None:
            self._hash = self._hash_impl()
        return self._hash

    def sharding_shapes(
        self, mesh_dim: int | None = None
    ) -> dict[int, tuple[torch.Size, ...]] | tuple[torch.Size, ...]:
        r"""Get the shapes of shards along specified mesh dimensions.

        Parameters
        ----------
        mesh_dim : Optional[int], optional
            If provided, return shapes only for this mesh dimension.

        Returns
        -------
        Union[Dict[int, Tuple[torch.Size, ...]], Tuple[torch.Size, ...]]
            Dictionary of shard shapes by mesh dim if ``mesh_dim`` is ``None``,
            or tuple of shapes for the specific mesh dimension.
        """
        if self._sharding_shapes is None:
            if mesh_dim is None:
                shard_shapes_by_dim, global_shape = _all_gather_shard_shapes(
                    self._local_shape, self.placements, self.mesh
                )
                self._sharding_shapes = shard_shapes_by_dim
                self.tensor_meta = self.tensor_meta._replace(shape=global_shape)
            else:
                return _gather_shard_shapes_for_dim(
                    self._local_shape,
                    mesh_dim,
                    self.mesh.get_group(mesh_dim),
                    do_checks=False,
                )
        if mesh_dim is not None:
            if mesh_dim in self._sharding_shapes:
                return self._sharding_shapes[mesh_dim]
        return self._sharding_shapes

    def __eq__(self, other: object) -> bool:
        r"""Check if two ShardTensorSpecs are equal.

        Parameters
        ----------
        other : object
            The other object to compare to.

        Returns
        -------
        bool
            ``True`` if the specs are equal, ``False`` otherwise.
        """
        if not isinstance(other, ShardTensorSpec):
            return False
        if not super().__eq__(other):
            return False
        if self._sharding_shapes != other._sharding_shapes:
            return False
        return True

    @property
    def local_shape(self) -> torch.Size:
        r"""Get the shape of the local shard.

        Returns
        -------
        torch.Size
            Shape of local tensor shard.

        Raises
        ------
        RuntimeError
            If local shape has not been set.
        """
        if self._local_shape is None:
            raise Exception("Missing local shape!")
        return self._local_shape

    @local_shape.setter
    def local_shape(self, value: torch.Size) -> None:
        r"""Set the local shard shape.

        Parameters
        ----------
        value : torch.Size
            Shape to set for local shard.

        Raises
        ------
        TypeError
            If value is not a ``torch.Size``.
        """
        if not isinstance(value, torch.Size):
            raise TypeError("Local shape must be instance of torch.Size")
        self._local_shape = value

    def offsets(self, mesh_dim: int | None = None) -> tuple[int, ...] | int:
        r"""Calculate offsets for the local shard within the global tensor.

        Returns the effective offset of this tensor along sharded dimensions,
        as if it was all collected into one device and you wanted to slice it
        to recover the local slice.

        Parameters
        ----------
        mesh_dim : Optional[int], optional
            If provided, return offset only for this mesh dimension.

        Returns
        -------
        Union[Tuple[int, ...], int]
            Tuple of offsets for each mesh dimension, or single offset if
            ``mesh_dim`` is specified.
        """
        offsets = []
        for loop_mesh_dim in range(self.mesh.ndim):
            coord = self.mesh.get_coordinate()[loop_mesh_dim]
            placement = self.placements[loop_mesh_dim]
            # If the placement is not shard, offset is 0:
            if isinstance(placement, Shard):
                shards = self._sharding_shapes[loop_mesh_dim]
                tensor_dim = placement.dim
                o = sum([s[tensor_dim] for s in shards[:coord]])
                offsets.append(o)
            else:
                offsets.append(0)

        if mesh_dim is not None:
            return offsets[mesh_dim]

        return tuple(offsets)  # Fixed: Return tuple instead of list


def _stride_from_contiguous_shape_C_style(shape: tuple[int, ...]) -> tuple[int, ...]:
    r"""Compute strides from a tensor shape assuming contiguous C-style layout.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Input shape as tuple or ``torch.Size``.

    Returns
    -------
    Tuple[int, ...]
        Tuple of strides of same length as input.
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


def _gather_shard_shapes_for_dim(
    local_shape: torch.Size | torch.Tensor,
    tensor_dim: int,
    local_group: dist.ProcessGroup,
    do_checks: bool = False,
) -> tuple[torch.Size, ...]:
    r"""Gather tensor shapes from all ranks in a process group for a given dimension.

    This function collects the shapes of tensor shards from all ranks in a
    process group and performs optional validation checks on the gathered
    shapes. Uses NCCL, which requires two-way transfers between host and device.

    Parameters
    ----------
    local_shape : Union[torch.Size, torch.Tensor]
        Shape of the local tensor shard, either as ``torch.Size`` or tensor.
    tensor_dim : int
        The tensor dimension being sharded.
    local_group : dist.ProcessGroup
        Process group to gather shapes from.
    do_checks : bool, default=False
        Whether to validate shape consistency across ranks.

    Returns
    -------
    Tuple[torch.Size, ...]
        Tuple of ``torch.Size`` objects containing gathered shapes from all ranks.

    Raises
    ------
    ValueError
        If shape validation fails when ``do_checks=True``:

        - Ranks have different tensor dimensions.
        - Non-sharded dimensions don't match across ranks.
    """
    local_size = dist.get_world_size(group=local_group)

    if not isinstance(local_shape, torch.Tensor):
        shape = torch.tensor(local_shape, device="cpu", pin_memory=True)

    local_shape = shape.to(device="cuda", non_blocking=True)

    all_shapes = [
        torch.zeros_like(local_shape, device="cuda") for _ in range(local_size)
    ]

    dist.all_gather(all_shapes, local_shape, group=local_group)

    all_shapes = [torch.Size(s.cpu().tolist()) for s in all_shapes]

    if do_checks:
        # Check that all shapes are the same rank
        if not all(len(local_shape) == len(all_s) for all_s in all_shapes):
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

    return tuple(all_shapes)


def _all_gather_shard_shapes(
    local_shape: torch.Size,
    placements: tuple[Placement, ...],
    target_mesh: DeviceMesh,
    do_checks: bool = False,
) -> tuple[dict[int, tuple[torch.Size, ...]], tuple[int, ...]]:
    r"""Gather shard shapes from all ranks across all sharded mesh dimensions.

    Parameters
    ----------
    local_shape : torch.Size
        Shape of the local tensor shard.
    placements : Tuple[Placement, ...]
        Tuple of placement specifications for each mesh dimension.
    target_mesh : DeviceMesh
        Device mesh containing process groups.
    do_checks : bool, default=False
        Whether to validate shape consistency across ranks.

    Returns
    -------
    Tuple[Dict[int, Tuple[torch.Size, ...]], Tuple[int, ...]]
        Tuple containing:

        - Dictionary mapping mesh dimensions to tuples of shard shapes.
        - The inferred global shape as a tuple.
    """
    shard_shapes_by_dim = {}
    global_shape = [s for s in local_shape]
    # We start by assuming the global shape is the local shape and fix it on sharded axes
    for mesh_axis, placement in enumerate(placements):
        if isinstance(placement, Shard):
            tensor_dim = placement.dim
            local_group = target_mesh.get_group(mesh_axis)

            shard_shapes_for_dim = _gather_shard_shapes_for_dim(
                local_shape, tensor_dim, local_group, do_checks
            )
            local_meta = tuple(
                # torch.Size(tuple(s)) for s in zip(all_shapes)
                shard_shapes_for_dim
            )

            shard_shapes_by_dim[mesh_axis] = local_meta

            # To infer the global shape _for this axis_,
            # we have to loop over each axis in the rank list
            # To check what placement is there.
            # This assumes full sharding:
            global_shape[tensor_dim] = sum([all_s[tensor_dim] for all_s in local_meta])

    return shard_shapes_by_dim, tuple(global_shape)


def compute_sharding_shapes_from_chunking_global_shape(
    mesh: DeviceMesh,
    placements: tuple[Placement, ...],
    global_shape: tuple[int, ...],
) -> dict[int, list[torch.Size]]:
    r"""Compute shard sizes for each mesh dimension based on global shape.

    For each sharded dimension in the mesh, computes the chunk sizes that
    would result from evenly dividing the global tensor shape. Returns a
    mapping from mesh dimensions to lists of ``torch.Size`` objects
    representing the shape of each shard.

    Parameters
    ----------
    mesh : DeviceMesh
        Device mesh defining the process topology.
    placements : Tuple[Placement, ...]
        Tuple of placement specifications for each mesh dimension.
    global_shape : Tuple[int, ...]
        Global shape of the full tensor before sharding.

    Returns
    -------
    Dict[int, List[torch.Size]]
        Dictionary mapping mesh dimensions to lists of ``torch.Size`` objects
        representing shard shapes for that dimension.

    Raises
    ------
    ValueError
        If placements length doesn't match mesh dimensions.
    """
    if len(placements) != mesh.ndim:
        raise ValueError("Number of placements must match mesh dimensions")

    # First compute raw chunk sizes for each sharded dimension
    temp_sharding_shapes: dict[int, list[int]] = {}
    for i in range(mesh.ndim):
        if isinstance(placements[i], Shard):
            # Compute the chunk size for this dimension:
            input_dim = global_shape[placements[i].dim]
            chunked_shapes = compute_split_shapes(input_dim, mesh.size(i))

            # for each tensor in the list

            temp_sharding_shapes[i] = chunked_shapes

    # Temp sharding shapes always has a key for each mesh dim.
    # Each is a list with length = size of that mesh dim.

    # Initialize shapes for all sharded dimensions, but using the global shape.
    # We will update next.
    sharding_shapes = {
        mesh_dim: [list(global_shape) for _ in chunks]
        for mesh_dim, chunks in temp_sharding_shapes.items()
    }

    # Go through and reduce each mesh dim to the right shape for _this_ rank
    for mesh_dim, shape_list in temp_sharding_shapes.items():
        this_rank = mesh.get_local_rank(mesh_dim)
        temp_sharding_shapes[mesh_dim] = shape_list[this_rank]

    # Finally, update the sharded shape with the right chunk size:
    for shape_list in sharding_shapes.values():
        for inner_mesh_dim, chunk_size in temp_sharding_shapes.items():
            tensor_dim = placements[inner_mesh_dim].dim
            for shape in shape_list:
                shape[tensor_dim] = chunk_size

    # Convert to immutable torch.Size
    return {
        mesh_dim: [torch.Size(tuple(size)) for size in sizes]
        for mesh_dim, sizes in sharding_shapes.items()
    }


def _infer_shard_tensor_spec_from_local_chunks(
    local_chunk: torch.Tensor,
    target_mesh: DeviceMesh,
    placements: tuple[Placement, ...],
    sharding_shapes: str | dict[int, list[tuple[int, ...]]] = "chunk",
    global_shape: tuple[int, ...] | None = None,
) -> ShardTensorSpec:
    r"""Build a ShardTensorSpec from local sizes, target mesh, and placements.

    Performs checks that all local tensors are compatible with the
    specified sharding configuration.

    Parameters
    ----------
    local_chunk : torch.Tensor
        Local tensor to be used as a shard of a global tensor.
    target_mesh : DeviceMesh
        Device mesh object to build this ShardTensor on.
    placements : Tuple[Placement, ...]
        Specified placements of this tensor.
    sharding_shapes : Union[str, Dict[int, List[Tuple[int, ...]]]], default="chunk"
        Controls how shard tensor spec is generated:

        - ``"chunk"``: Use ``torch.chunk`` shapes to infer shapes from
          global shape (no communication). Requires ``global_shape``.
        - ``"infer"``: Use collective communication to infer shapes from
          mesh neighbors.
        - Manual dict mapping mesh dim to list of shard shapes: Use
          provided shapes directly.
    global_shape : Optional[Tuple[int, ...]], optional
        Global shape of the tensor. Required if ``sharding_shapes="chunk"``.

    Returns
    -------
    ShardTensorSpec
        Specification to be used in creating a ShardTensor. Key feature
        of this spec is that each ShardTensor knows the shape and size of
        other shards, and can compute global offsets and reductions properly.

    Raises
    ------
    ValueError
        If ``sharding_shapes`` is an invalid string, if ``"chunk"`` is used
        without ``global_shape``, if placements length doesn't match mesh
        dimensions, or if inferred shapes don't match local tensor shape.
    """
    # Sharding_shapes, if a string, must be one of "chunk" "infer"
    if isinstance(sharding_shapes, str) and sharding_shapes not in [
        "chunk",
        "infer",
    ]:
        raise ValueError(
            "If sharding_shapes is a string, it must be one of: 'chunk', 'infer'"
        )

    # if sharding_shapes is a chunk, global_shape must be provided
    if sharding_shapes == "chunk" and global_shape is None:
        raise ValueError("If sharding_shapes is 'chunk', global_shape must be provided")

    # Check if sharding_shapes is an empty dict
    if isinstance(sharding_shapes, dict) and not sharding_shapes:
        # Raise an error only if the placements contains a shard:
        if any(isinstance(placement, Shard) for placement in placements):
            raise ValueError("sharding_shapes as a dict cannot be empty")

    # Need to infer the placements on each dimension of the mesh.
    if len(placements) != target_mesh.ndim:
        raise ValueError("Mesh dimension must match placements length")
    # If sharding_shapes is chunk, compute the chunk sizes from the global shape
    if isinstance(sharding_shapes, str):
        if sharding_shapes == "chunk":
            # This is communication-free.  It's the path from a properly-formated DTensorSpec.
            shard_shapes_by_dim = compute_sharding_shapes_from_chunking_global_shape(
                target_mesh,
                placements,
                list(global_shape),
            )
            # Basic sanity check, make sure the inferred shape matches the
            # local shape on the first sharded mesh dimension
            mesh_rank = None
            for mesh_dim, p in enumerate(placements):
                if isinstance(p, Shard):
                    mesh_rank = target_mesh.get_coordinate()[mesh_dim]
                    break

            if mesh_rank is not None:
                inferred_local_shape = shard_shapes_by_dim[mesh_dim][mesh_rank]
                if inferred_local_shape != local_chunk.shape:
                    raise ValueError(
                        f"Rank {dist.get_rank()} expected local shape {inferred_local_shape} does not match tensor's local shape {local_chunk.shape}"
                    )

        if sharding_shapes == "infer":
            # When unsure, this is a good option.
            shard_shapes_by_dim, global_shape = _all_gather_shard_shapes(
                local_chunk.shape,
                placements,
                target_mesh,
            )
    else:
        # We have been passed sharding shapes manually (yay!  best performance)
        # so infer the global shape from them
        global_shape = list(local_chunk.shape)
        for i in range(target_mesh.ndim):
            if isinstance(placements[i], Shard):
                # Sum the sides for this axis:
                tensor_dim = placements[i].dim
                global_shape[tensor_dim] = sum(
                    [s[tensor_dim] for s in sharding_shapes[i]]
                )

        shard_shapes_by_dim = sharding_shapes

    stride = _stride_from_contiguous_shape_C_style(global_shape)

    # # Finally, build a tensor spec to return:
    global_meta = TensorMeta(
        shape=tuple(global_shape), stride=stride, dtype=local_chunk.dtype
    )

    sharding_shapes = {dim: tuple(s) for dim, s in shard_shapes_by_dim.items()}
    return ShardTensorSpec(
        mesh=target_mesh,
        placements=placements,
        tensor_meta=global_meta,
        _local_shape=local_chunk.shape,
        _sharding_shapes=sharding_shapes,
    )
