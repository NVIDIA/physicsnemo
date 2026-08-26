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

from itertools import accumulate
from math import prod
from typing import cast

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import (
    TensorMeta,
)
from torch.distributed.tensor._redistribute import (
    _gen_transform_infos,
)
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)

import physicsnemo.domain_parallel.shard_tensor as shard_tensor
from physicsnemo.domain_parallel._shard_tensor_spec import (
    ShardTensorSpec,
    compute_sharding_shapes_from_chunking_global_shape,
)

# TODO:
# DTensor makes assumptions about sharding sizes.
# I need to figure out the target spec  manually, based on input/output placements.
# I'm already intercepting the collectives and using the right input sizes.
# But the output placements are containing the wrong sharding sizes.
# It should all "just work" once that's fixed.


# Worker functions for the collectives specific to uneven shaped tensors:
def _to_replicate_tensor(
    local_tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    mesh_dim: int,
    tensor_dim: int,
    current_spec: ShardTensorSpec,
) -> torch.Tensor:
    r"""Convert a sharded tensor to a replicated tensor by gathering all shards.

    Parameters
    ----------
    local_tensor : torch.Tensor
        The local shard of the tensor to replicate.
    device_mesh : DeviceMesh
        The device mesh containing process groups.
    mesh_dim : int
        The mesh dimension along which to gather.
    tensor_dim : int
        The tensor dimension along which data is sharded.
    current_spec : ShardTensorSpec
        Specification of current sharding scheme.

    Returns
    -------
    torch.Tensor
        The fully replicated tensor on this rank.

    Note
    ----
    This function handles uneven sharding by using ``all_gather_v`` instead of
    regular ``all_gather``.
    """
    # Get the mesh for the group:
    mesh = current_spec.mesh

    # Ensure contiguous data for the reduction:
    local_tensor = local_tensor.contiguous()

    # # Get all sizes:
    # TODO: We don't need to summon all sizes across all mesh dimensions.
    # Optimize the spec function to only get the sizes for the relevant mesh dimensions.
    sizes = current_spec.sharding_shapes()

    # Consecutive redistributes _don't_ update full sizes.
    # So, extract the shape from this tensor, and assume all other tensor
    # dims match.
    tensor_dim_shapes = tuple(s[tensor_dim] for s in sizes[mesh_dim])
    base_shapes = [list(local_tensor.shape) for _ in tensor_dim_shapes]
    for i, t in enumerate(tensor_dim_shapes):
        base_shapes[i][tensor_dim] = tensor_dim_shapes[i]

    # Gather with funcol rather than dist.all_gather so a captured graph
    # holds a DeviceMesh reference instead of a c10d ProcessGroup
    # ScriptObject: AOTAutograd deepcopies the backward GraphModule during
    # caching, and ProcessGroup has no __getstate__. funcol.all_gather_tensor
    # requires equal shapes across ranks, so pad the flattened shard to the
    # largest shard's numel and slice per-rank afterwards.
    numels = [prod(s) for s in base_shapes]
    max_numel = max(numels)
    send = local_tensor.reshape(-1)
    if send.numel() < max_numel:
        send = torch.nn.functional.pad(send, (0, max_numel - send.numel()))
    gathered = funcol.all_gather_tensor(send, gather_dim=0, group=(mesh, mesh_dim))
    if isinstance(gathered, funcol.AsyncCollectiveTensor):
        gathered = gathered.wait()

    shards = [
        gathered[i * max_numel : i * max_numel + numels[i]].view(shape)
        for i, shape in enumerate(base_shapes)
    ]
    return torch.cat(shards, dim=tensor_dim).contiguous()


def _select_slice_from_replicate(
    local_tensor: torch.Tensor,
    target_spec: ShardTensorSpec,
    mesh_dim: int,
    mesh_coord: int,
    sizes: list[int] | None = None,
) -> tuple[torch.Tensor, list[int] | None]:
    r"""Select the appropriate slice from a replicated tensor to create a shard.

    Parameters
    ----------
    local_tensor : torch.Tensor
        The replicated tensor to slice from.
    target_spec : ShardTensorSpec
        Specification of target sharding scheme.
    mesh_dim : int
        The mesh dimension along which to shard.
    mesh_coord : int
        The coordinate of this rank in the mesh dimension.
    sizes : Optional[List[int]], optional
        Size hint for chunking. If provided and matches mesh size, uses
        these sizes for splitting.

    Returns
    -------
    Tuple[torch.Tensor, Optional[List[int]]]
        Tuple containing the selected slice that will become this rank's
        shard, and the sizes used (or ``None`` if chunk was used).

    Note
    ----
    This function handles uneven sharding by using the sharding sizes from
    the target spec to split the tensor into potentially uneven chunks.
    """

    # TODO - This needs a rework to enable caching of shapes for a grad pass.
    # We really only need the sizes from this dimension:
    tensor_dim = target_spec.placements[mesh_dim].dim
    mesh_size = target_spec.mesh.size(mesh_dim=mesh_dim)

    # Can we use the size hint here?
    if sizes is not None and len(sizes) != mesh_size:
        sizes = None

    # Split the tensor:
    if sizes is None:
        # Use chunk, not split, when dividing without a plan
        chunks = torch.chunk(local_tensor, mesh_size, dim=tensor_dim)
    else:
        # Convert sizes to cumulative sum using basic Python
        chunk_starts = []
        running_sum = 0
        for size in sizes[:-1]:
            running_sum += size
            chunk_starts.append(running_sum)
        chunks = torch.tensor_split(local_tensor, chunk_starts, dim=tensor_dim)
    return chunks[mesh_coord], sizes


def _to_new_shard_dim(
    local_tensor: torch.Tensor,
    current_spec: ShardTensorSpec,
    target_spec: ShardTensorSpec,
    mesh_dim: int,
    size_hint: list[int] | None,
    spec_shapes_are_current: bool,
    current_dim: int,
    target_dim: int,
) -> tuple[torch.Tensor, list[int] | None]:
    r"""Transpose tensor sharding from one dimension to another.

    Reshards a tensor from being sharded on ``current_dim`` to being sharded
    on ``target_dim``. Uses all-to-all communication which is more efficient
    than all_gather followed by scatter.

    Parameters
    ----------
    local_tensor : torch.Tensor
        The local shard of the tensor to reshard.
    current_spec : ShardTensorSpec
        Specification of current sharding scheme.
    target_spec : ShardTensorSpec
        Specification of target sharding scheme.
    mesh_dim : int
        The device mesh dimension on which we're transposing.
    size_hint : Optional[List[int]]
        If provided, use this to chunk the tensor for both send and recv.
    spec_shapes_are_current : bool
        Whether ``current_spec``'s recorded per-rank sharding shapes still
        describe ``local_tensor``. The caller sets this from the hop
        sequence of the enclosing redistribute (true only until the first
        transform mutates the local tensor), which is derived from
        placements and therefore identical on every rank -- the fast path
        below must be taken by all ranks or none, so this decision may not
        depend on rank-local state.
    current_dim : int
        Currently sharded on this tensor dimension.
    target_dim : int
        Want to be sharded on this tensor dimension.

    Returns
    -------
    Tuple[torch.Tensor, Optional[List[int]]]
        Tuple containing the resharded tensor and the size hint used
        (may be ``None`` if it was discarded).
    """
    # We're essentially transposing the tensor here.
    # We could implement this as an all_gather_v / scatter_v, but
    # it's more efficient to do an all_to_all.

    device_mesh = target_spec.mesh
    mesh_size = device_mesh.size(mesh_dim=mesh_dim)

    # To use the size hint, and preserve the original sharding, we need to insist that
    # the mesh_size and the length of size hint is equal
    if size_hint is not None and mesh_size != len(size_hint):
        # Setting to None will prevent it being used further
        size_hint = None

    # First, we need to split the tensor along the target dimension:
    if size_hint is None:
        chunks = torch.chunk(local_tensor, mesh_size, dim=target_dim)
    else:
        chunk_starts = list(accumulate(size_hint))
        chunks = torch.tensor_split(local_tensor, chunk_starts[:-1], dim=target_dim)

    # MUST be contiguous for all_to_all:
    # Also, cast to list for all_to_all:
    chunks = [c.contiguous() for c in chunks]

    # Try to compute recv shapes analytically, with no communication:
    # - the target-dim chunk size we're about to receive is whatever *we*
    #   compute for our own slot, since every rank derives its chunk sizes
    #   from the same (already-replicated) global extent via the same
    #   deterministic chunking rule -- it's not actually rank-dependent.
    # - the current-dim extent of the sender is already recorded per-rank
    #   on `current_spec`.
    # Whether the fast path applies is decided by `spec_shapes_are_current`,
    # which the caller derives from the hop sequence (identical on all
    # ranks by construction). It must NOT be decided from rank-local state
    # such as `local_tensor.shape`: ranks disagreeing here means a subset
    # skips the shape-negotiation all_to_all below and the collectives
    # mismatch. The shape comparison is therefore an assertion (corrupt or
    # inconsistent specs should fail loudly), not a fallback condition.
    recv_shapes = None
    if (
        spec_shapes_are_current
        and current_spec._sharding_shapes is not None
        and mesh_dim in current_spec._sharding_shapes
    ):
        current_shapes_by_rank = current_spec._sharding_shapes[mesh_dim]
        my_coord = device_mesh.get_coordinate()[mesh_dim]
        if len(current_shapes_by_rank) != mesh_size:
            raise RuntimeError(
                f"current_spec records {len(current_shapes_by_rank)} shapes "
                f"for mesh dim {mesh_dim}, expected {mesh_size}."
            )
        if torch.Size(current_shapes_by_rank[my_coord]) != torch.Size(
            local_tensor.shape
        ):
            raise RuntimeError(
                f"current_spec records shape {current_shapes_by_rank[my_coord]} "
                f"for this rank on mesh dim {mesh_dim}, but the local tensor "
                f"has shape {tuple(local_tensor.shape)}. The spec's sharding "
                "shapes are stale or corrupt."
            )
        my_target_chunk_size = chunks[my_coord].shape[target_dim]
        recv_shapes = []
        for sender_shape in current_shapes_by_rank:
            if len(sender_shape) != local_tensor.ndim:
                raise RuntimeError(
                    f"current_spec records shape {tuple(sender_shape)} for a "
                    f"peer rank on mesh dim {mesh_dim}, which has different "
                    f"rank than the local tensor shape "
                    f"{tuple(local_tensor.shape)}."
                )
            recv_shape = list(sender_shape)
            recv_shape[target_dim] = my_target_chunk_size
            recv_shapes.append(recv_shape)

    if recv_shapes is None:
        # Fallback: negotiate recv shapes with the sender ranks directly.
        # funcol (with a mesh group, not a ProcessGroup) keeps this legal
        # inside AOT-captured backward graphs, which get deepcopied.
        # Every per-destination shape tensor has ndim elements, so the
        # exchange is an even all_to_all_single (None split sizes).
        ndim = local_tensor.ndim
        send_shape_buf = torch.cat(
            [
                torch.tensor(c.shape, device=local_tensor.device, dtype=torch.int64)
                for c in chunks
            ]
        )
        recv_shape_buf = funcol.all_to_all_single(
            send_shape_buf, None, None, (device_mesh, mesh_dim)
        )
        if isinstance(recv_shape_buf, funcol.AsyncCollectiveTensor):
            recv_shape_buf = recv_shape_buf.wait()

        # Turn the recv_shapes back into plain int shape lists.
        recv_shapes = [
            recv_shape_buf[i * ndim : (i + 1) * ndim].tolist() for i in range(mesh_size)
        ]

    # Exchange the data itself as one flattened all_to_all_single (funcol for
    # the same compile/deepcopy reason as above): concatenate the flattened
    # per-destination chunks in ascending rank order, with explicit uneven
    # split sizes on both sides.
    input_split_sizes = [c.numel() for c in chunks]
    output_split_sizes = [prod(shape) for shape in recv_shapes]
    send_buf = torch.cat([c.reshape(-1) for c in chunks])
    recv_buf = funcol.all_to_all_single(
        send_buf, output_split_sizes, input_split_sizes, (device_mesh, mesh_dim)
    )
    if isinstance(recv_buf, funcol.AsyncCollectiveTensor):
        recv_buf = recv_buf.wait()

    recv_buffers = [
        chunk.view(shape)
        for chunk, shape in zip(torch.split(recv_buf, output_split_sizes), recv_shapes)
    ]

    # Take the received tensors and stack them along the target dimension:
    stacked_tensor = torch.cat(recv_buffers, dim=current_dim).contiguous()

    # Return the size hint in case we discarded it
    return stacked_tensor, size_hint


def redistribute_local_shard_tensor(
    local_tensor: torch.Tensor,
    current_spec: ShardTensorSpec,
    target_spec: ShardTensorSpec,
    *,
    async_op: bool = False,
    is_backward: bool = False,
    target_sharding_shapes: dict[int, list[int]] | None = None,
) -> torch.Tensor:
    r"""Redistribute a local tensor between different ShardTensorSpec configurations.

    This redistributes the local tensor (``torch.Tensor``) from the current
    ShardTensorSpec to the target ShardTensorSpec, which involves the necessary
    collective calls to transform the local shard of the ShardTensor from its
    current spec to the target spec.

    The collective operations are implemented in the Placement classes, which
    we avoid modifying. To get around that, we mimic the logic from PyTorch's
    original redistribute. But in cases where a tensor is sharded and the
    shards are uneven, we intercept and replace the collectives:

    - ``Shard(dim)`` -> ``Replicate()``: ``all_gather_v`` instead of ``all_gather``
    - ``Shard(src_dim)`` -> ``Shard(dst_dim)``: remains all_to_all but
      reimplemented to handle sizes correctly
    - ``Replicate()`` -> ``Shard(dim)``: local chunking is unchanged but return
      value is ShardTensorSpec instead
    - ``Partial()`` -> ``Replicate()``: ``all_reduce`` needs to become a weighted
      ``all_reduce``, depending on operation
    - ``Partial()`` -> ``Shard(dim)``: ``reduce_scatter`` needs to become a
      weighted ``reduce_scatter``, depending on operation

    Parameters
    ----------
    local_tensor : torch.Tensor
        The local tensor shard to redistribute.
    current_spec : ShardTensorSpec
        Specification of current sharding scheme.
    target_spec : ShardTensorSpec
        Specification of target sharding scheme.
    async_op : bool, default=False
        Whether to run asynchronously.
    is_backward : bool, default=False
        Whether this is a backward redistribution. For example, a forward
        ``Partial`` to ``Replicate`` redistribution combines every rank's
        contribution. In backward, each original contribution receives the
        same full gradient. Converting that gradient back to ``Partial`` would
        split it up only for a later operation to combine it again, so backward
        keeps it replicated instead. The caller must therefore label the
        returned gradient ``Replicate``. Labeling it ``Partial`` would falsely
        request another reduction and could multiply the gradient by the mesh
        size. This exception applies only while reversing a redistribution;
        ordinary ``Partial`` tensors still represent pending reductions.
    target_sharding_shapes : Optional[Dict[int, List[int]]], optional
        Per-rank shard sizes keyed by tensor dimension. Default is ``None``.

    Returns
    -------
    torch.Tensor
        The redistributed local tensor.

    Raises
    ------
    NotImplementedError
        If cross device mesh communication is attempted.
    RuntimeError
        If redistribution fails for any reason.
    """
    if target_sharding_shapes is None:
        target_sharding_shapes = {}

    if current_spec.mesh != target_spec.mesh:
        # TODO: alltoall/permute reshuffling to change device_mesh if they are not the same
        raise NotImplementedError("Cross device mesh comm not supported yet!")

    new_local_tensor = None
    device_mesh = current_spec.mesh

    my_coordinate = device_mesh.get_coordinate()

    if my_coordinate is None:
        # if rank is not part of mesh, we skip redistribute and simply return local_tensor,
        # which should be an empty tensor
        return local_tensor

    # This is an internal-focused step.  If the target_spec has the same placements and mesh
    # as the current, but is missing sharding sizes, we can use the current spec's sharding sizes.
    # if target_spec._sharding_sizes is None:
    #     if target_spec.placements == current_spec.placements and target_spec.mesh == current_spec.mesh:
    #         target_spec._sharding_sizes = current_spec.sharding_shapes()

    # For sharded tensors, we use the same order of transformation as DTensor.
    # However, often we need to ignore the provided logical shape and substitute
    # a sharded shape instead.
    # This is done by providing a target_sharding_shapes dict above.

    transform_infos = _gen_transform_infos(current_spec, target_spec)

    if len(transform_infos) == 0:
        return local_tensor

    # `current_spec`'s recorded per-rank sharding shapes describe the local
    # tensors only until the first hop below mutates them. This flag is a
    # function of the hop sequence alone (derived from placements, so
    # identical on every rank) -- see `_to_new_shard_dim` for why the
    # fast-path decision must be rank-uniform.
    spec_shapes_are_current = True

    for transform_info in transform_infos:
        i = transform_info.mesh_dim
        current, target = transform_info.src_dst_placements
        device_mesh.size(mesh_dim=i)

        if current == target:
            # short cut, just use the original local tensor
            new_local_tensor = local_tensor
            continue

        # logger.debug("redistribute from %s to %s on mesh dim %s", current, target, i)
        if target.is_replicate():
            # Case 1: target is Replicate
            if current.is_partial():
                partial_spec = cast(Partial, current)
                new_local_tensor = partial_spec._reduce_value(
                    local_tensor, device_mesh, i
                )
            elif current.is_shard():
                current_placement = cast(Shard, current)
                new_local_tensor = _to_replicate_tensor(
                    local_tensor,
                    device_mesh,
                    mesh_dim=i,
                    tensor_dim=current_placement.dim,
                    current_spec=current_spec,
                )
            else:
                raise RuntimeError(
                    f"redistribute from {current} to {target} not supported yet"
                )
        elif target.is_shard():
            # Case 2: target is Shard
            target_placement = cast(Shard, target)
            if current.is_partial():
                partial_spec = cast(Partial, current)
                new_local_tensor = partial_spec._reduce_shard_value(
                    local_tensor, device_mesh, i, target_placement
                )
            elif current.is_replicate():
                # split the tensor and return the corresponding cloned local shard
                # Are there suggested placements for the shards?
                if target_placement.dim in target_sharding_shapes:
                    size_hint = target_sharding_shapes[target_placement.dim]
                else:
                    size_hint = None
                new_local_tensor, size_hint = _select_slice_from_replicate(
                    local_tensor,
                    target_spec,
                    i,
                    my_coordinate[i],
                    size_hint,
                )
                if (
                    size_hint is not None
                    and target_placement.dim in target_sharding_shapes
                ):
                    target_sharding_shapes[target_placement.dim] = size_hint

            else:
                if not current.is_shard():
                    raise RuntimeError(
                        f"Current placement should be shard but found {current}"
                    )
                shard_spec = cast(Shard, current)
                if shard_spec.dim != target_placement.dim:
                    # Here we need to essentially transpose the tensor along two dimensions.
                    # We cached shardings that appear in both the input and output shards, along tensor dimensions.
                    # So, if the target tensor dimension is in there,
                    # That is how we're going to shard the local tensor on the tensor_dim,
                    # and it also defines how we'll receive the tensor .
                    if target_placement.dim in target_sharding_shapes:
                        size_hint = target_sharding_shapes[target_placement.dim]
                    else:
                        size_hint = None

                    new_local_tensor, size_hint = _to_new_shard_dim(
                        local_tensor,
                        current_spec,  # Known per-rank shapes, to avoid negotiating recv sizes.
                        target_spec,  # Send the whole spec so we can infer full recv sizes.
                        i,  # The mesh dim we're transposing sharding on.
                        size_hint,
                        spec_shapes_are_current,  # Rank-uniform fast-path gate.
                        current.dim,  # Current tensor dimension.
                        target_placement.dim,  # Target tensor dimension.
                    )
                    if (
                        size_hint is None
                        and target_placement.dim in target_sharding_shapes
                    ):
                        target_sharding_shapes.pop(target_placement.dim)
                    if size_hint is not None and current.dim in target_sharding_shapes:
                        target_sharding_shapes.pop(current.dim)

        elif target.is_partial():
            if current.is_replicate():
                partial_spec = cast(Partial, target)
                # In a reverse redistribution, keep the fully reduced gradient
                # instead of partitioning it only to reduce it again later.
                # ShardRedistribute.backward normalizes the emitted placement to
                # Replicate so the returned tensor's metadata matches this data.
                new_local_tensor = (
                    partial_spec._partition_value(local_tensor, device_mesh, i)
                    if not is_backward
                    else local_tensor
                )
            elif current.is_shard():
                if not is_backward:
                    raise RuntimeError(
                        f"redistribute from {current} to {target} not supported yet"
                    )
                # Gather the reverse-path shard into complete data. The caller
                # emits Replicate metadata rather than the temporary Partial
                # target used to describe the inverse forward transform.
                current_placement = cast(Shard, current)
                new_local_tensor = current_placement._to_replicate_tensor(
                    local_tensor, device_mesh, i, transform_info.logical_shape
                )
            else:
                # partial -> partial no op, should never hit
                new_local_tensor = local_tensor

        if new_local_tensor is None:
            raise RuntimeError(
                "Failed to create new local tensor during redistribution"
            )
        local_tensor = new_local_tensor
        # This hop transformed the local tensor (the `current == target`
        # shortcut above skips this point), so the spec's recorded shapes
        # no longer describe it.
        spec_shapes_are_current = False

    if new_local_tensor is None:
        raise RuntimeError("redistribute failed!")

    if not async_op and isinstance(new_local_tensor, funcol.AsyncCollectiveTensor):
        new_local_tensor = new_local_tensor.wait()

    return new_local_tensor


def get_tensor_sharding_shapes_by_dim(
    current_spec: ShardTensorSpec,
    target_placements: tuple[Placement, ...],
) -> dict[int, list[int]]:
    r"""Extract sharding shapes that are preserved between current and target placements.

    For shardings that exist in both the current spec and target placements on
    the same tensor dimension, this function extracts and returns those shapes.

    Parameters
    ----------
    current_spec : ShardTensorSpec
        The current sharding specification.
    target_placements : Tuple[Placement, ...]
        The target placement specifications.

    Returns
    -------
    Dict[int, List[int]]
        Dictionary mapping tensor dimensions to lists of shard sizes for
        dimensions that are sharded in both current and target placements.
    """

    target_sharding_shapes = {}
    # Look through the target placements for shardings:
    for target_mesh_dim, target_placement in enumerate(target_placements):
        if isinstance(target_placement, Shard):
            # If the target tensor dim is in the current target_placements,
            # Maintain that sharding.
            target_tensor_dim = target_placement.dim
            # Find if this tensor dim is in the current spec's placements:
            for current_mesh_dim, current_placement in enumerate(
                current_spec.placements
            ):
                if (
                    isinstance(current_placement, Shard)
                    and target_tensor_dim == current_placement.dim
                ):
                    # The tensor dim is the same in both current and target,
                    # But the rest of the tensors dimensions may change.
                    # Therefore only save the dimension on this axis.
                    current_shardings = current_spec.sharding_shapes()[current_mesh_dim]
                    target_sharding_shapes[target_tensor_dim] = [
                        c[target_tensor_dim] for c in current_shardings
                    ]

    return target_sharding_shapes


class ShardRedistribute(torch.autograd.Function):
    r"""ShardTensor-enhanced version of redistribute with autograd support.

    Extends the functionality in ``DTensor`` to allow redistribution of
    sharded tensors with uneven sharding. This autograd function handles
    both forward and backward passes for redistributing sharded tensors
    between different sharding schemes.
    """

    @staticmethod
    def forward(
        input: "shard_tensor.ShardTensor",
        device_mesh: DeviceMesh,
        placements: tuple[Placement, ...],
        async_op: bool = False,
    ) -> "shard_tensor.ShardTensor":
        r"""Forward pass for redistributing a sharded tensor.

        Parameters
        ----------
        input : ShardTensor
            Input sharded tensor to redistribute.
        device_mesh : DeviceMesh
            Target device mesh for redistribution.
        placements : Tuple[Placement, ...]
            Target placement scheme for redistribution.
        async_op : bool, default=False
            Whether to perform redistribution asynchronously.

        Returns
        -------
        ShardTensor
            Redistributed sharded tensor with new placement scheme.
        """
        current_spec = input._spec

        if current_spec.placements != placements:
            # We have to assume, here, that the current spec has correct sharding_shapes.
            # Therefore, we can use the target placement + current sharding_shapes
            # to get the target sharding sizes correctly.

            # target_spec = generate_target_spec_from_current_and_placements(
            #     current_spec,
            #     placements,
            # )

            target_spec = ShardTensorSpec(
                device_mesh,
                placements,
                tensor_meta=input._spec.tensor_meta,
            )

            # The target sharding sizes are potentially incomplete.
            # They're only provided for shardings that are the same in input/output.
            target_sharding_shapes = get_tensor_sharding_shapes_by_dim(
                current_spec, placements
            )
            # ctx.target_sharding_shapes = target_sharding_shapes
            local_tensor = input._local_tensor
            output = redistribute_local_shard_tensor(
                local_tensor,
                current_spec,
                target_spec,
                async_op=async_op,
                target_sharding_shapes=target_sharding_shapes,
            )
            # Set the local shape:
            target_spec._local_shape = output.shape

            # Populate _sharding_shapes on the target spec so downstream
            # consumers (especially under torch.compile) don't trip
            # `_all_gather_shard_shapes` -- a blocking collective that is
            # not AOT-traceable. Start from chunk semantics (pure
            # arithmetic, no comms) and override preserved-shard tensor
            # dims with the precomputed per-rank sizes from
            # `target_sharding_shapes` so uneven sharding is preserved.
            global_shape = tuple(input._spec.tensor_meta.shape)
            chunk_shapes = compute_sharding_shapes_from_chunking_global_shape(
                device_mesh, placements, global_shape
            )
            mesh_coordinate = device_mesh.get_coordinate()
            for mesh_dim, shapes in chunk_shapes.items():
                overridden = []
                for rank, chunk_shape in enumerate(shapes):
                    rank_shape = list(chunk_shape)
                    # Preserve every uneven shard in this cross-section. The
                    # current mesh dim varies with ``rank``; all other mesh
                    # dims remain fixed at this rank's coordinates.
                    for preserved_mesh_dim, placement in enumerate(placements):
                        if not isinstance(placement, Shard):
                            continue
                        tensor_dim = placement.dim
                        per_rank_sizes = target_sharding_shapes.get(tensor_dim)
                        if per_rank_sizes is None or len(
                            per_rank_sizes
                        ) != device_mesh.size(preserved_mesh_dim):
                            continue
                        size_rank = (
                            rank
                            if preserved_mesh_dim == mesh_dim
                            else mesh_coordinate[preserved_mesh_dim]
                        )
                        rank_shape[tensor_dim] = int(per_rank_sizes[size_rank])
                    overridden.append(tuple(rank_shape))
                chunk_shapes[mesh_dim] = overridden
            target_spec._sharding_shapes = {
                mesh_dim: tuple(tuple(s) for s in shapes)
                for mesh_dim, shapes in chunk_shapes.items()
            }
        else:
            # use the same local tensor if placements are the same.
            output = input._local_tensor
            target_spec = current_spec

        return shard_tensor.ShardTensor(
            output.contiguous(),
            target_spec,
            requires_grad=input.requires_grad,
        )

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        r"""Save the source spec and ``async_op`` flag for the backward redistribute.

        ``DisableTorchFunctionSubclass`` shielding avoids re-entering the
        ShardTensor ``__torch_function__`` fallback while reading
        ``input._spec`` -- the same AOT-hostile bridge motivated the
        shielding in ``ShardedSum.setup_context``.
        """
        input, _device_mesh, _placements, async_op = inputs
        with torch._C.DisableTorchFunctionSubclass():
            ctx.current_spec = input._spec
        ctx.async_op = async_op

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: "shard_tensor.ShardTensor",
    ) -> tuple["shard_tensor.ShardTensor", None, None, None]:
        r"""Backward pass for redistributing a sharded tensor.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context containing saved tensors/variables from forward.
        grad_output : ShardTensor
            Gradient output tensor to redistribute back.

        Returns
        -------
        Tuple[ShardTensor, None, None, None]
            Tuple containing the redistributed gradient tensor and ``None``
            for device_mesh, placements, and async_op gradients (not
            differentiable).
        """
        previous_spec = ctx.current_spec
        current_spec = grad_output._spec

        async_op = ctx.async_op

        local_tensor = grad_output._local_tensor
        target_sharding_shapes = get_tensor_sharding_shapes_by_dim(
            previous_spec, previous_spec.placements
        )
        output = redistribute_local_shard_tensor(
            local_tensor,
            current_spec,
            previous_spec,
            async_op=async_op,
            is_backward=True,
            target_sharding_shapes=target_sharding_shapes,
        )

        # Reverse redistribution deliberately keeps complete gradients instead
        # of manufacturing pending reductions. Label those values Replicate so
        # every emitted Partial continues to mean "reduction required."
        normalized_placements: list[Placement] = []
        for previous_placement in previous_spec.placements:
            if previous_placement.is_partial():
                normalized_placements.append(Replicate())
            else:
                normalized_placements.append(previous_placement)

        # Carry the source spec's per-rank shard shapes onto the grad: the
        # normalization above only rewrites Partial -> Replicate, so the Shard
        # placements (the only mesh dims with entries) are previous_spec's
        # verbatim. Omitting them leaves an uneven grad with a shapeless spec,
        # which __tensor_unflatten__ rejects at compile boundaries (even-chunk
        # assumption) and which would otherwise gather shapes lazily.
        spec = ShardTensorSpec(
            previous_spec.device_mesh,
            tuple(normalized_placements),
            tensor_meta=TensorMeta(
                shape=grad_output.shape,
                stride=grad_output.stride(),
                dtype=grad_output.dtype,
            ),
            _local_shape=output.shape,
            _sharding_shapes=(
                dict(previous_spec._sharding_shapes)
                if previous_spec._sharding_shapes is not None
                else None
            ),
        )
        output_shard_tensor = shard_tensor.ShardTensor(
            output,
            spec,
            requires_grad=grad_output.requires_grad,
        )
        return (
            output_shard_tensor,
            None,
            None,
            None,
        )
