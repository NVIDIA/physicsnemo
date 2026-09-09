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

r"""Shared machinery for domain-parallel (rank-local) reading in datapipes.

Readers that support domain-parallel reading accept a ``domain_parallel``
configuration dict plus a 1-D ``device_mesh`` (constructed and injected at
runtime in Python -- it is not serializable config). Each rank reads only
its share of the sharded batch axes; the local pieces move to the GPU inside a
:class:`ShardedProto`; and :meth:`ShardedProto.assemble` builds the sample
with ``Shard(0)`` ShardTensors via the communication-free chunk path of
``ShardTensor.from_local``.

Configuration
-------------
::

    domain_parallel = {
        # Auto gate: shard a batch axis when its length (the size of tensor
        # dim 0) is at least this many entries; shorter axes replicate.
        "auto_shard_size": 1024,
        # Optional overrides, keyed by axis name: "shard" | "replicate".
        # Anything not named falls back to the auto gate.
        "placements": {"interior.points": "shard", "boundaries.stl": "replicate"},
    }

``auto_shard_size`` looks at the length of dim 0 for each tensor, not the
total number of elements. With the default of 1024: a point cloud of shape
``(200_000, 3)`` shards, while a ``(512, 512)`` image and a ``(7, 100000)``
table replicate (pin them with ``placements`` to shard them along dim 0). An
axis shorter than the world size always replicates, so no rank is left with
an empty shard. ``auto_shard_size: 1`` shards everything except scalars and
axes shorter than the world size.

Decisions are made per **batch axis**, not per tensor. Tensors that share a
batch axis are co-indexed and placed together. A mesh has one batch axis:
``cells`` when it has cells (``cells``, every ``cell_data`` leaf, and its
``points`` / ``point_data`` all chunk together; ``cells`` keeps global vertex
ids and ``points[cells]`` is the routed gather), or ``points`` for a point
cloud. For a flat dataset the axes are the groups of leaves that share a
dim-0 length. For example, a zarr dataset with ``coords`` and ``fields``
arrays of the same length shards both even if you only specify
``placements: {"coords": "shard"}``. Subsampling often forces the same dim-0
length across a dataset, and the gate is decided *after* subsampling: the
axis length it sees is the subsampled length, not the stored one.

Axis names are dotted paths. An override applies to the axis it names and,
by prefix, to every axis beneath it: ``boundaries.stl: replicate`` pins both
``boundaries.stl.points`` and ``boundaries.stl.cells``; for a flat sample
``solution: shard`` pins every leaf under ``solution``. Overrides that match
nothing are reported as warnings.

Placement convention
--------------------
A sharded leaf is a ``ShardTensor`` with ``Shard(0)``. A replicated leaf is a
**plain tensor** that is identical on every rank of the device mesh -- it is
never wrapped as a ``Replicate`` ShardTensor. Mixed ops rely on ShardTensor
promoting plain operands, and reductions over a sharded axis yield
``Partial`` results that callers materialize (``full_tensor``) only where a
rank-identical value is required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Literal, Mapping

import torch
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.datapipes.keys import NestedKey, key_to_str
from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.domain_parallel._shard_tensor_spec import (
    compute_sharding_shapes_from_chunking_global_shape,
)

logger = logging.getLogger(__name__)

# The only placement domain-parallel readers produce: sharded on the batch
# (dim-0) axis of a 1-D device mesh.
PLACEMENTS = (Shard(0),)

# A sharded tensor pays a fixed per-op dispatch cost regardless of size, so
# short batch axes are strictly cheaper replicated. This default keeps
# per-sample metadata (freestream vectors, parameter tables) plain while
# sharding anything point- or cell-sized.
DEFAULT_AUTO_SHARD_SIZE = 1024

# Placement names as they appear in configuration. The ``domain_parallel``
# dict is parsed from YAML (Hydra), so overrides are strings rather than
# ``torch.distributed`` ``Placement`` objects; they are promoted to the real
# placement at assembly time, where the only outcomes are ``Shard(0)`` for a
# sharded axis and a plain (rank-identical) tensor for a replicated one.
PlacementName = Literal["shard", "replicate"]
_VALID_PLACEMENTS: tuple[PlacementName, ...] = ("shard", "replicate")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainParallelConfig:
    r"""Parsed ``domain_parallel`` configuration bound to its device mesh.

    Built once per reader by :meth:`from_dict`; every placement decision goes
    through :meth:`decide`, so the raw dict is never re-read per sample.

    Parameters
    ----------
    device_mesh : DeviceMesh
        1-D device mesh the batch axes are sharded over.
    auto_shard_size : int
        Auto gate: a batch axis shards when its length (tensor dim 0) is at
        least this many entries and at least the world size.
    placements : Mapping[str, PlacementName]
        User overrides, dotted axis name -> ``"shard"`` | ``"replicate"``;
        an entry applies to the axis it names and, by prefix, to every axis
        beneath it.
    """

    device_mesh: DeviceMesh
    auto_shard_size: int = DEFAULT_AUTO_SHARD_SIZE
    placements: Mapping[str, PlacementName] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls, config: Mapping[str, Any] | None, device_mesh: DeviceMesh | None
    ) -> DomainParallelConfig | None:
        r"""Validate and parse the ``domain_parallel`` dict / ``device_mesh`` pair.

        Parameters
        ----------
        config : Mapping or None
            The ``domain_parallel`` configuration dict (see module docstring).
        device_mesh : DeviceMesh or None
            The device mesh the batch axes would be sharded over.

        Returns
        -------
        DomainParallelConfig or None
            ``None`` when both are absent (domain parallelism off).

        Raises
        ------
        ValueError
            On a missing/extra pairing, a non-1-D mesh, an unknown key, a
            non-positive ``auto_shard_size``, or a bad placements entry.
        """
        if config is None and device_mesh is None:
            return None
        if (config is None) != (device_mesh is None):
            raise ValueError(
                "domain_parallel and device_mesh must be provided together"
            )
        if device_mesh.ndim != 1:
            raise ValueError(f"device_mesh must be 1-D, got {device_mesh.ndim} dims")

        unknown = set(config) - {"auto_shard_size", "placements"}
        if unknown:
            raise ValueError(
                f"unknown domain_parallel keys {sorted(unknown)}; "
                'expected "auto_shard_size" and/or "placements"'
            )
        size = config.get("auto_shard_size", DEFAULT_AUTO_SHARD_SIZE)
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError(f"auto_shard_size must be a positive int, got {size!r}")
        placements = config.get("placements") or {}
        if not isinstance(placements, Mapping):
            raise ValueError(
                f'placements must be a dict of axis -> "shard"|"replicate", '
                f"got {placements!r}"
            )
        bad = {k: v for k, v in placements.items() if v not in _VALID_PLACEMENTS}
        if bad:
            raise ValueError(f'placements values must be "shard"|"replicate": {bad}')
        return cls(
            device_mesh=device_mesh, auto_shard_size=size, placements=dict(placements)
        )

    @property
    def world_size(self) -> int:
        r"""Number of ranks the batch axes are sharded over."""
        return self.device_mesh.size(0)

    def chunk_bounds(self, global_n: int) -> tuple[int, int]:
        r"""This rank's ``[start, stop)`` share of an axis of length *global_n*."""
        return chunk_bounds(global_n, self.device_mesh)

    def placement_for(
        self, axis: str, pinned: Mapping[str, PlacementName] | None = None
    ) -> PlacementName | None:
        r"""The override that applies to *axis*, if any.

        User ``placements`` win over reader *pinned* entries; within each, the
        longest dotted prefix of *axis* wins.

        Parameters
        ----------
        axis : str
            Dotted axis name.
        pinned : Mapping[str, PlacementName], optional
            Reader-supplied structural pins (``"boundaries.stl": "replicate"``).

        Returns
        -------
        PlacementName or None
            ``"shard"`` / ``"replicate"`` if an override applies, else ``None``.
        """
        overrides = {**(pinned or {}), **self.placements}
        parts = axis.split(".")
        for n in range(len(parts), 0, -1):
            hit = overrides.get(".".join(parts[:n]))
            if hit is not None:
                return hit
        return None

    def warn_unmatched(self, names: Iterable[str]) -> None:
        r"""Log a warning for configured overrides that match none of *names*.

        Parameters
        ----------
        names : Iterable[str]
            Dotted names an override could legitimately target.
        """
        names = list(names)
        for key in self.placements:
            if not any(n == key or n.startswith(key + ".") for n in names):
                logger.warning(
                    "domain_parallel.placements[%r] matches no batch axis of this "
                    "sample (axes: %s); check the key",
                    key,
                    sorted(names),
                )

    def decide(
        self,
        axes: Mapping[str, int],
        pinned: Mapping[str, PlacementName] | None = None,
        *,
        check_unmatched: bool = True,
    ) -> dict[str, bool]:
        r"""Decide shard-vs-replicate for each named batch axis.

        Parameters
        ----------
        axes : Mapping[str, int]
            Length (tensor dim 0) per axis name, from metadata; no data read.
        pinned : Mapping[str, PlacementName], optional
            Reader-supplied structural pins, overridden by ``placements``.
        check_unmatched : bool, default=True
            Warn about ``placements`` keys that match none of *axes*.

        Returns
        -------
        dict[str, bool]
            ``True`` to shard the axis, ``False`` to replicate it. Under the
            gate an axis shards when its length is at least
            ``auto_shard_size`` and at least the world size.

        Raises
        ------
        ValueError
            An axis pinned to ``"shard"`` with fewer entries than the world
            size, which would leave a rank with an empty shard.
        """
        if check_unmatched:
            self.warn_unmatched(axes)
        world_size = self.world_size
        decisions: dict[str, bool] = {}
        for axis, length in axes.items():
            pin = self.placement_for(axis, pinned)
            if pin == "shard":
                if length < world_size:
                    raise ValueError(
                        f"axis {axis!r} is pinned to shard but has {length} entries "
                        f"< world size {world_size}"
                    )
                decisions[axis] = True
            elif pin == "replicate":
                decisions[axis] = False
            else:
                decisions[axis] = length >= max(self.auto_shard_size, world_size)
        return decisions


# ---------------------------------------------------------------------------
# Chunk arithmetic
# ---------------------------------------------------------------------------


def chunk_bounds(global_n: int, device_mesh: DeviceMesh) -> tuple[int, int]:
    r"""This rank's ``[start, stop)`` share of an axis under ``torch.chunk`` semantics.

    Uses the same shard-shape arithmetic as
    ``ShardTensor.from_local(sharding_shapes="chunk")``, so entries selected
    with these bounds are exactly the local shard the later wrap declares.

    Parameters
    ----------
    global_n : int
        Global length of the batch axis being sharded.
    device_mesh : DeviceMesh
        1-D device mesh the axis is sharded over.

    Returns
    -------
    tuple[int, int]
        Half-open ``[start, stop)`` range of entries owned by this rank.
    """
    shapes = compute_sharding_shapes_from_chunking_global_shape(
        device_mesh, PLACEMENTS, (global_n,)
    )
    sizes = [s[0] for s in shapes[0]]
    rank = device_mesh.get_local_rank(0)
    start = sum(sizes[:rank])
    return start, start + sizes[rank]


# ---------------------------------------------------------------------------
# Placement resolution
# ---------------------------------------------------------------------------


def resolve_leaf_placements(
    meta: dict[NestedKey, tuple[int, ...]], config: DomainParallelConfig
) -> dict[NestedKey, bool]:
    r"""Decide shard-vs-replicate per leaf of a flat sample, by axis group.

    Leaves sharing a dim-0 length form one batch axis and are decided
    together, so co-indexed arrays never end up with mismatched placements.
    A ``placements`` entry naming any leaf pins its whole group; two leaves
    of one group pinned differently is an error. Scalars always replicate.

    Parameters
    ----------
    meta : dict[NestedKey, tuple[int, ...]]
        Global shape per leaf (from store metadata; no data read).
    config : DomainParallelConfig
        Parsed configuration bound to the device mesh.

    Returns
    -------
    dict[NestedKey, bool]
        Per-leaf sharding decision.
    """
    groups: dict[int, list[NestedKey]] = {}
    for key, shape in meta.items():
        if len(shape) > 0:
            groups.setdefault(shape[0], []).append(key)

    # One axis per dim-0 length, named by its leaves so errors and warnings
    # can point at keys the user configured. Overrides apply by prefix.
    axes: dict[str, int] = {}
    pinned: dict[str, str] = {}
    axis_of: dict[int, str] = {}
    for length, keys in groups.items():
        names = sorted(key_to_str(k) for k in keys)
        axis = "|".join(names)
        axis_of[length] = axis
        axes[axis] = length
        pins = {}
        for name in names:
            pin = config.placement_for(name)
            if pin is not None:
                pins[name] = pin
        if len(set(pins.values())) > 1:
            raise ValueError(
                f"leaves {sorted(pins)} share a batch axis (length {length}) but "
                f"are pinned to different placements: {pins}"
            )
        if pins:
            pinned[axis] = next(iter(pins.values()))

    # Overrides were already applied per leaf above (they are pins now), so
    # decide without the user placements; warn against the leaf names.
    config.warn_unmatched(key_to_str(k) for k, shape in meta.items() if len(shape) > 0)
    axis_decisions = replace(config, placements={}).decide(
        axes, pinned, check_unmatched=False
    )
    return {
        key: (len(shape) > 0 and axis_decisions[axis_of[shape[0]]])
        for key, shape in meta.items()
    }


# ---------------------------------------------------------------------------
# Host-stage payload
# ---------------------------------------------------------------------------

# kind -> function rebuilding the final sample from the assembled TensorDict.
_REBUILDERS: dict[str, Callable[[TensorDict], Any]] = {
    "tensordict": lambda td: td,
}


def register_proto_kind(kind: str, rebuild: Callable[[TensorDict], Any]) -> None:
    r"""Register how a :class:`ShardedProto` of *kind* becomes a sample.

    Internal: the datapipes' own readers register ``"mesh"`` and
    ``"domain_mesh"``; it is not a public extension point.

    Parameters
    ----------
    kind : str
        Payload kind tag (``"mesh"``, ``"domain_mesh"``, ...).
    rebuild : Callable[[TensorDict], Any]
        Builds the sample from the nested TensorDict whose sharded leaves
        have already been wrapped as ShardTensors.
    """
    _REBUILDERS[kind] = rebuild


@dataclass(frozen=True)
class ShardedProto:
    r"""This rank's share of one sample, before ShardTensor assembly.

    The host-stage payload of a domain-parallel read. ``tensors`` is a
    nested TensorDict mirroring the final sample's structure (for a mesh:
    ``points``, ``cells``, ``point_data``, ``cell_data``, ``global_data``;
    for a domain mesh: ``interior``, ``boundaries.<name>``, ``global_data``).
    Sharded leaves hold this rank's share; replicated leaves are complete.

    Readers return a proto instead of a finished sample; datasets move it to
    the device (``to`` / ``pin_memory``, the same seam every sample flows
    through) and then call :meth:`assemble`.

    Users aren't expected to interact with a ShardedProto, unless you're
    assembling your own domain-parallel reader.

    Parameters
    ----------
    tensors : TensorDict
        Nested local tensors.
    sharded : dict[tuple[str, ...], tuple[int, ...]]
        Global shape per sharded leaf, keyed by nested tuple key. Leaves
        absent from this map are replicated.
    device_mesh : DeviceMesh
        1-D device mesh the selection was taken against; the wrap reuses it.
    kind : str
        Which registered rebuild turns the assembled TensorDict into the
        sample (``"tensordict"``, ``"mesh"``, ``"domain_mesh"``).
    """

    tensors: TensorDict
    sharded: dict[tuple[str, ...], tuple[int, ...]]
    device_mesh: DeviceMesh
    kind: str = "tensordict"

    def _replace_tensors(self, tensors: TensorDict) -> "ShardedProto":
        return ShardedProto(
            tensors=tensors,
            sharded=self.sharded,
            device_mesh=self.device_mesh,
            kind=self.kind,
        )

    def to(self, device: torch.device, non_blocking: bool = False) -> "ShardedProto":
        r"""Return a copy with the local tensors moved to *device*.

        Parameters
        ----------
        device : torch.device
            Target device.
        non_blocking : bool, default=False
            Passed through to ``TensorDict.to`` for async H2D copies.
        """
        return self._replace_tensors(self.tensors.to(device, non_blocking=non_blocking))

    def pin_memory(self) -> "ShardedProto":
        r"""Return a copy with the local tensors in pinned host memory."""
        return self._replace_tensors(self.tensors.pin_memory())

    def assemble(self) -> Any:
        r"""Wrap sharded leaves as ``Shard(0)`` ShardTensors and rebuild the sample.

        The wrap is communication-free: every rank derives identical shard
        shapes from the global shapes carried in :attr:`sharded`.
        """
        try:
            rebuild = _REBUILDERS[self.kind]
        except KeyError:
            raise ValueError(
                f"no rebuild registered for proto kind {self.kind!r}; "
                f"known kinds: {sorted(_REBUILDERS)}"
            ) from None
        return rebuild(
            wrap_sharded_leaves(self.tensors, self.sharded, self.device_mesh)
        )


def assemble_if_proto(data: Any) -> Any:
    r"""Assemble a :class:`ShardedProto` payload; pass anything else through."""
    return data.assemble() if isinstance(data, ShardedProto) else data


def wrap_sharded_leaves(
    tensors: TensorDict,
    sharded: dict[tuple[str, ...], tuple[int, ...]],
    device_mesh: DeviceMesh,
) -> TensorDict:
    r"""Rebuild *tensors* with every leaf in *sharded* wrapped as a ShardTensor.

    Structure (including empty sub-TensorDicts) is preserved; the result has
    ``batch_size=[]`` at every level since sharded leaves carry global batch
    lengths that the local sub-TensorDict batch sizes no longer match. The
    wrap is the communication-free chunk path: every rank derives identical
    shard shapes from the global shapes.

    Parameters
    ----------
    tensors : TensorDict
        Nested local tensors, already on the target device.
    sharded : dict[tuple[str, ...], tuple[int, ...]]
        Global shape per sharded leaf.
    device_mesh : DeviceMesh
        1-D device mesh for the ``Shard(0)`` wrap.

    Returns
    -------
    TensorDict
        Same structure; sharded leaves are ``Shard(0)`` ShardTensors,
        replicated leaves pass through as plain tensors.
    """

    def wrap(td: TensorDict, prefix: tuple[str, ...]) -> TensorDict:
        out: dict[str, Any] = {}
        for key, value in td.items():
            path = (*prefix, key)
            if isinstance(value, TensorDict):
                out[key] = wrap(value, path)
            elif path in sharded:
                out[key] = ShardTensor.from_local(
                    value,
                    device_mesh,
                    PLACEMENTS,
                    sharding_shapes="chunk",
                    global_shape=sharded[path],
                )
            else:
                out[key] = value
        return TensorDict(out, batch_size=[])

    return wrap(tensors, ())


def as_leaf_key(key: NestedKey) -> tuple[str, ...]:
    r"""Normalize a TensorDict leaf key to the tuple form :class:`ShardedProto` uses.

    A string is one component (it is a TensorDict key, not a dotted config
    path); a tuple passes through.
    """
    return (key,) if isinstance(key, str) else tuple(key)
