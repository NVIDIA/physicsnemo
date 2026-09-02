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

r"""Weight synchronization utilities for domain-parallel training.

In hybrid (data x domain) parallelism, every rank in a domain group holds a
full replica of the model's plain (non-distributed) weights, and those
replicas must start identical.  Neither DDP nor FSDP2 does this for you on
the domain axis: DDP broadcasts only over *its* process group (the data-
parallel axis), and FSDP2 performs no initial synchronization at all.
:func:`sync_module_over_mesh` fills that gap.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from .shard_tensor import ShardTensor

__all__ = ["sync_module_over_mesh"]

# Same default bucket size DDP uses for its construction-time sync (250MB).
_BROADCAST_BUCKET_SIZE = int(250 * 1024 * 1024)


def _validate_1d_mesh(mesh: DeviceMesh, caller: str) -> dist.ProcessGroup:
    """Validate a synchronization mesh and return its process group."""
    if mesh.ndim != 1:
        raise ValueError(
            f"{caller} expects a 1-D mesh, got {mesh.ndim}-D. "
            f'Pass the axis explicitly, e.g. mesh["domain"].'
        )
    return mesh.get_group()


def _is_distributed_tensor(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` has an explicit distributed layout."""
    return isinstance(tensor, (DTensor, ShardTensor))


def _plain_module_tensors(
    module: nn.Module,
) -> list[tuple[str, torch.Tensor]]:
    """Collect plain parameters followed by plain buffers."""
    tensors: list[tuple[str, torch.Tensor]] = []
    for name, tensor in module.named_parameters():
        if not _is_distributed_tensor(tensor):
            tensors.append((name, tensor.detach()))
    for name, tensor in module.named_buffers():
        if not _is_distributed_tensor(tensor):
            tensors.append((name, tensor.detach()))
    return tensors


def _validate_plain_tensors(
    named_tensors: list[tuple[str, torch.Tensor]], mesh: DeviceMesh
) -> None:
    """Reject state that cannot participate in the mesh collective."""
    for name, tensor in named_tensors:
        if tensor.is_meta:
            raise RuntimeError(
                "sync_module_over_mesh cannot synchronize meta tensors; "
                f'materialize "{name}" first.'
            )
        if tensor.device.type != mesh.device_type:
            raise RuntimeError(
                f'sync_module_over_mesh found "{name}" on {tensor.device.type}, '
                f"but the mesh device type is {mesh.device_type}."
            )


def _verify_matching_metadata(
    named_tensors: list[tuple[str, torch.Tensor]], group: dist.ProcessGroup
) -> None:
    """Collectively verify corresponding module state before broadcasting."""
    local_metadata = [
        (name, tuple(tensor.shape), str(tensor.dtype), str(tensor.layout))
        for name, tensor in named_tensors
    ]
    gathered_metadata: list[object] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered_metadata, local_metadata, group=group)
    reference = gathered_metadata[0]
    if any(metadata != reference for metadata in gathered_metadata[1:]):
        raise RuntimeError(
            "sync_module_over_mesh: parameter/buffer metadata differs across "
            f"the mesh: {gathered_metadata}."
        )


def _broadcast_coalesced(
    group: dist.ProcessGroup,
    tensors: list[torch.Tensor],
    src_mesh_rank: int,
) -> None:
    """Broadcast tensors efficiently, with a public-API compatibility fallback."""
    broadcast_coalesced = getattr(dist, "_broadcast_coalesced", None)
    if broadcast_coalesced is not None:
        broadcast_coalesced(group, tensors, _BROADCAST_BUCKET_SIZE, src_mesh_rank)
        return

    src_global_rank = dist.get_global_rank(group, src_mesh_rank)
    for tensor in tensors:
        dist.broadcast(tensor, src=src_global_rank, group=group)


def _tensor_checksum(tensor: torch.Tensor) -> torch.Tensor:
    """Return a small diagnostic checksum that is stable in the presence of NaNs."""
    values = torch.view_as_real(tensor) if tensor.is_complex() else tensor
    values = values.double()
    finite = torch.isfinite(values)
    finite_values = torch.where(finite, values, torch.zeros_like(values))
    return torch.stack(
        (
            finite_values.sum(),
            finite_values.abs().sum(),
            torch.isnan(values).sum().double(),
            torch.isposinf(values).sum().double(),
            torch.isneginf(values).sum().double(),
        )
    )


def _verify_matching_values(
    named_tensors: list[tuple[str, torch.Tensor]], group: dist.ProcessGroup
) -> None:
    """Verify per-tensor diagnostic checksums across a process group."""
    for name, tensor in named_tensors:
        checksum = _tensor_checksum(tensor)
        lo, hi = checksum.clone(), checksum.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX, group=group)
        if not torch.equal(lo, hi):
            raise RuntimeError(
                "sync_module_over_mesh: parameter/buffer checksum differs "
                f'across the mesh for "{name}" '
                f"(min={lo.tolist()}, max={hi.tolist()})."
            )


def sync_module_over_mesh(
    module: nn.Module,
    mesh: DeviceMesh,
    src_mesh_rank: int = 0,
    verify: bool = False,
) -> None:
    r"""Broadcast a module's plain parameters and buffers over a mesh axis.

    Synchronizes every plain (non-distributed) parameter **and buffer** of
    ``module`` across ``mesh``'s process group, from the rank at position
    ``src_mesh_rank`` of the mesh.  Distributed tensors (``DTensor``,
    ``ShardTensor``) are skipped because arbitrary distributed layouts cannot
    safely be broadcast by this utility.  Their initialization remains the
    caller's responsibility.  Source-based constructors such as
    ``distribute_tensor`` and :func:`scatter_tensor` synchronize values, while
    ``DTensor.from_local`` and ``ShardTensor.from_local`` do not.

    Call this whenever ``domain_size > 1``, regardless of the data-parallel
    wrapper: DDP broadcasts weights only over the data-parallel group at
    construction, and FSDP2 (``fully_shard``) does not synchronize initial
    weights on any axis.  The ordering relative to DDP construction does not
    affect correctness (the two axes compose), but on the FSDP2 path this
    must run *before* ``fully_shard``, while the parameters are still plain.

    Parameters
    ----------
    module : torch.nn.Module
        The module whose plain parameters/buffers are synchronized in place.
    mesh : DeviceMesh
        A 1-D device mesh (typically the ``"domain"`` submesh of a larger
        mesh) whose process group the broadcast runs over.
    src_mesh_rank : int, default=0
        The source position *within the mesh* to broadcast from.
    verify : bool, default=False
        If ``True``, collectively compare tensor metadata before the broadcast
        and per-tensor diagnostic checksums afterward.  The checksum detects
        rank disagreement but is not a cryptographic integrity check.

    Raises
    ------
    ValueError
        If ``mesh`` is not 1-D.  For a multi-dim mesh, pass the axis you
        want explicitly, e.g. ``mesh["domain"]``.
    RuntimeError
        If called after FSDP2 wrapping, if plain state is incompatible with the
        mesh, if all registered state is already distributed, or if
        ``verify=True`` finds differing metadata or values.
    """
    group = _validate_1d_mesh(mesh, "sync_module_over_mesh")
    group_size = dist.get_world_size(group)
    if src_mesh_rank < 0 or src_mesh_rank >= group_size:
        raise ValueError(
            f"src_mesh_rank must be in [0, {group_size}), got {src_mesh_rank}."
        )

    if isinstance(module, FSDPModule):
        raise RuntimeError(
            "sync_module_over_mesh must run before fully_shard, while replicated "
            "module state is still plain."
        )

    named_tensors = _plain_module_tensors(module)
    _validate_plain_tensors(named_tensors, mesh)

    if not named_tensors:
        if any(True for _ in module.parameters()) or any(
            True for _ in module.buffers()
        ):
            raise RuntimeError(
                "sync_module_over_mesh found module state, but none of it is "
                "plain. Distributed tensors are caller-managed; call this "
                "utility before converting replicated state."
            )
        return

    if verify:
        _verify_matching_metadata(named_tensors, group)

    with torch.no_grad():
        # The same coalesced broadcast DDP uses at construction; note its
        # src is the rank *within the group*, unlike dist.broadcast.
        _broadcast_coalesced(
            group, [tensor for _, tensor in named_tensors], src_mesh_rank
        )

    if verify:
        _verify_matching_values(named_tensors, group)
