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

r"""Pure-logic tests for the domain-parallel reader configuration.

Placement resolution reads only global metadata and the mesh size, so it is
testable single-rank with a stub device mesh; the distributed read/wrap
behavior is covered in ``test/domain_parallel/datapipes/``.
"""

import dataclasses
import logging

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.datapipes._domain_parallel import (
    DomainParallelConfig,
    ShardedProto,
    assemble_if_proto,
    resolve_leaf_placements,
)
from physicsnemo.datapipes.readers.base import Reader
from physicsnemo.datapipes.readers.mesh import (
    mesh_selection_to_proto,
    plan_mesh_selection,
    resolve_mesh_placements,
)
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import MEASURE_WEIGHTS_KEY


class _StubMesh:
    """Duck-typed 1-D device mesh: just a world size."""

    def __init__(self, world_size: int):
        self._world_size = world_size

    @property
    def ndim(self) -> int:
        return 1

    def size(self, dim: int = 0) -> int:
        return self._world_size


def test_validate_requires_pairing():
    DomainParallelConfig.from_dict(None, None)  # both absent: fine
    with pytest.raises(ValueError, match="together"):
        DomainParallelConfig.from_dict({}, None)
    with pytest.raises(ValueError, match="together"):
        DomainParallelConfig.from_dict(None, _StubMesh(2))


def test_validate_rejects_bad_config():
    mesh = _StubMesh(2)
    with pytest.raises(ValueError, match="unknown"):
        DomainParallelConfig.from_dict({"nonsense": {}}, mesh)
    with pytest.raises(ValueError, match="auto_shard_size"):
        DomainParallelConfig.from_dict({"auto_shard_size": 0}, mesh)
    with pytest.raises(ValueError, match="placements must be a dict"):
        DomainParallelConfig.from_dict({"placements": "auto"}, mesh)
    with pytest.raises(ValueError, match="shard"):
        DomainParallelConfig.from_dict({"placements": {"x": "banana"}}, mesh)
    DomainParallelConfig.from_dict(
        {"auto_shard_size": 8, "placements": {"x": "shard"}}, mesh
    )
    DomainParallelConfig.from_dict({}, mesh)


@pytest.mark.parametrize(
    ("rows", "config", "world", "expected"),
    [
        # Default gate: dim 0 at least 1024 long, whatever the world size.
        (1024, {}, 4, True),
        (1023, {}, 4, False),
        (200_000, {}, 2, True),
        # Explicit gate.
        (64, {"auto_shard_size": 32}, 2, True),
        (31, {"auto_shard_size": 32}, 2, False),
        # Never shorter than the world size under the gate.
        (3, {"auto_shard_size": 1}, 4, False),
        (4, {"auto_shard_size": 1}, 4, True),
        # Pins bypass the gate.
        (3, {"auto_shard_size": 10**6, "placements": {"a": "shard"}}, 2, True),
        (10**9, {"placements": {"a": "replicate"}}, 2, False),
    ],
)
def test_axis_gate(rows, config, world, expected):
    assert DomainParallelConfig.from_dict(config, _StubMesh(world)).decide(
        {"a": rows}
    ) == {"a": expected}


def test_axis_pin_below_world_size_raises():
    with pytest.raises(ValueError, match="world size"):
        DomainParallelConfig.from_dict(
            {"placements": {"a": "shard"}}, _StubMesh(4)
        ).decide({"a": 3})


def test_axis_pin_by_prefix_and_reader_pins():
    axes = {
        "interior.points": 10**6,
        "interior.cells": 10**6,
        "boundaries.stl.points": 10**6,
        "boundaries.stl.cells": 10**6,
        "boundaries.wing.points": 10**6,
    }
    mesh = _StubMesh(2)
    # Reader pin on the sub-mesh applies to both of its axes.
    pinned = {"boundaries.stl": "replicate"}
    decisions = DomainParallelConfig.from_dict({}, mesh).decide(axes, pinned)
    assert decisions["boundaries.stl.points"] is False
    assert decisions["boundaries.stl.cells"] is False
    assert decisions["interior.points"] is True
    assert decisions["boundaries.wing.points"] is True
    # User config overrides the reader pin; a deeper key beats a shallower one.
    config = {
        "placements": {
            "boundaries.stl": "shard",
            "interior": "replicate",
            "interior.points": "shard",
        }
    }
    decisions = DomainParallelConfig.from_dict(config, mesh).decide(axes, pinned)
    assert decisions["boundaries.stl.points"] is True
    assert decisions["interior.points"] is True
    assert decisions["interior.cells"] is False


def test_leaf_placements_group_by_axis():
    mesh = _StubMesh(4)
    meta = {
        "coords": (5000, 3),
        "fields": (5000, 4),
        "mask": (5000,),
        "params": (7,),
        "scalar": (),
    }
    # One decision per shared dim-0 length; scalars always replicate.
    decisions = resolve_leaf_placements(meta, DomainParallelConfig.from_dict({}, mesh))
    assert decisions == {
        "coords": True,
        "fields": True,
        "mask": True,
        "params": False,
        "scalar": False,
    }
    # Pinning one leaf pins its whole group.
    decisions = resolve_leaf_placements(
        meta,
        DomainParallelConfig.from_dict({"placements": {"mask": "replicate"}}, mesh),
    )
    assert not any(decisions[k] for k in ("coords", "fields", "mask"))
    # Conflicting pins within a group are an error.
    with pytest.raises(ValueError, match="different placements"):
        resolve_leaf_placements(
            meta,
            DomainParallelConfig.from_dict(
                {"placements": {"coords": "shard", "fields": "replicate"}}, mesh
            ),
        )


def test_mesh_placements_axes():
    mesh = _StubMesh(2)
    shapes = {"interior": (10**5, 0), "boundaries.wing": (10**5, 5 * 10**4)}
    decisions = resolve_mesh_placements(
        shapes, DomainParallelConfig.from_dict({}, mesh)
    )
    # A point cloud is gated on points and never shards cells; a mesh with
    # cells is gated on cells and its vertices follow.
    assert decisions["interior"] == (True, False)
    assert decisions["boundaries.wing"] == (True, True)
    decisions = resolve_mesh_placements(
        shapes,
        DomainParallelConfig.from_dict(
            {"placements": {"boundaries.wing.cells": "replicate"}}, mesh
        ),
    )
    assert decisions["boundaries.wing"] == (False, False)
    # A few cells but many points: the cells axis decides, so it replicates.
    decisions = resolve_mesh_placements(
        {"": (10**6, 3)}, DomainParallelConfig.from_dict({}, mesh)
    )
    assert decisions[""] == (False, False)


class _StubDeviceMesh(_StubMesh):
    """Stub with a rank, enough for ``chunk_bounds``."""

    def __init__(self, world_size: int, rank: int):
        super().__init__(world_size)
        self._rank = rank

    def get_local_rank(self, dim: int = 0) -> int:
        return self._rank

    def get_coordinate(self):
        return [self._rank]


def _small_mesh() -> Mesh:
    torch.manual_seed(0)
    return Mesh(
        points=torch.randn(10, 3),
        cells=torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [2, 3, 9]]),
        point_data={"t": torch.randn(10)},
        cell_data={"p": torch.randn(4, 2)},
        global_data={"Re": torch.tensor(1.0)},
    )


def test_mesh_selection_to_proto_point_cloud_window():
    """A point cloud reads its chunk of the (windowed) point range and reports
    the window length as the global count."""
    mesh = _small_mesh()
    cloud = Mesh(points=mesh.points, point_data={"t": mesh.point_data["t"]})
    dm = _StubDeviceMesh(2, 1)
    plan = plan_mesh_selection(10, 0, True, False, dm)
    tensors, sharded = mesh_selection_to_proto(cloud, plan, ("interior",))
    torch.testing.assert_close(tensors["points"], mesh.points[5:10])
    torch.testing.assert_close(tensors["point_data", "t"], mesh.point_data["t"][5:10])
    assert sharded == {
        ("interior", "points"): (10, 3),
        ("interior", "point_data", "t"): (10,),
    }

    window = torch.tensor([7, 8, 9, 0, 1, 2])
    plan = plan_mesh_selection(10, 0, True, False, dm, point_window=window)
    tensors, sharded = mesh_selection_to_proto(cloud, plan)
    torch.testing.assert_close(tensors["points"], mesh.points[[0, 1, 2]])
    assert sharded[("points",)] == (6, 3)


def test_mesh_selection_to_proto_cells_no_window():
    """Full resolution: chunk of the cell range and chunk of the point range;
    cells keep their global vertex ids."""
    mesh = _small_mesh()
    dm = _StubDeviceMesh(2, 0)
    plan = plan_mesh_selection(10, 4, True, True, dm)
    tensors, sharded = mesh_selection_to_proto(mesh, plan, ("b",))
    torch.testing.assert_close(tensors["points"], mesh.points[:5])
    torch.testing.assert_close(tensors["cells"], mesh.cells[:2])
    torch.testing.assert_close(tensors["cell_data", "p"], mesh.cell_data["p"][:2])
    assert sharded == {
        ("b", "points"): (10, 3),
        ("b", "point_data", "t"): (10,),
        ("b", "cells"): (4, 3),
        ("b", "cell_data", "p"): (4, 2),
    }
    # Replicated: everything, untouched.
    plan = plan_mesh_selection(10, 4, False, False, dm)
    tensors, sharded = mesh_selection_to_proto(mesh, plan)
    torch.testing.assert_close(tensors["points"], mesh.points)
    torch.testing.assert_close(tensors["cells"], mesh.cells)
    assert sharded == {}


def test_mesh_selection_to_proto_cell_window_compacts_globally():
    """A cell window compacts onto the referenced vertices (eager semantics);
    this rank keeps its chunk of the remapped cells and of the vertex set."""
    mesh = _small_mesh()
    window = torch.tensor([2, 3])  # cells [6,7,8], [2,3,9] -> vertices 2,3,6,7,8,9
    ref = _subsample_reference(mesh, window)
    for rank in (0, 1):
        dm = _StubDeviceMesh(2, rank)
        plan = plan_mesh_selection(10, 4, True, True, dm, cell_window=window)
        tensors, sharded = mesh_selection_to_proto(mesh, plan)
        torch.testing.assert_close(tensors["cells"], ref.cells[rank : rank + 1])
        torch.testing.assert_close(
            tensors["points"], ref.points[3 * rank : 3 * rank + 3]
        )
        torch.testing.assert_close(
            tensors["cell_data", "p"], ref.cell_data["p"][rank : rank + 1]
        )
        torch.testing.assert_close(
            tensors["cell_data", MEASURE_WEIGHTS_KEY],
            ref.cell_data[MEASURE_WEIGHTS_KEY][rank : rank + 1],
        )
        assert sharded[("points",)] == (6, 3) and sharded[("cells",)] == (2, 3)
    assert plan.measure_factor == 2.0
    with pytest.raises(ValueError, match="point subsampling"):
        plan_mesh_selection(10, 4, True, True, dm, point_window=window)
    with pytest.raises(ValueError, match="cell_window"):
        plan_mesh_selection(10, 0, True, False, dm, cell_window=window)


def _subsample_reference(mesh: Mesh, window: torch.Tensor) -> Mesh:
    """Eager cell subsample: slice_cells + compaction + measure weights."""
    from physicsnemo.mesh.calculus.measure import compose_measure_weights

    out = mesh.slice_cells(window)
    referenced = torch.unique(out.cells)
    out = out.slice_points(referenced)
    compose_measure_weights(out, mesh.n_cells / len(window))
    return out


def test_assemble_if_proto_and_rebuild_registry():
    """Plain samples pass through; an unknown kind fails loudly."""
    plain = object()
    assert assemble_if_proto(plain) is plain
    proto = ShardedProto(
        tensors=TensorDict({"x": torch.zeros(2)}, batch_size=[]),
        sharded={},
        device_mesh=None,
        kind="nonexistent",
    )
    with pytest.raises(ValueError, match="nonexistent"):
        proto.assemble()
    # With nothing sharded the wrap is a pure copy; the tensordict kind is
    # the identity rebuild.
    out = dataclasses.replace(proto, kind="tensordict").assemble()
    torch.testing.assert_close(out["x"], torch.zeros(2))


def test_leaf_placements_pin_by_prefix_and_warn_unmatched(caplog):
    """A dotted override pins every leaf beneath it; a key matching nothing warns."""
    mesh = _StubMesh(2)
    meta = {
        ("solution", "pressure"): (5000, 1),
        ("solution", "velocity"): (5000, 3),
        "coords": (5000, 3),
        "params": (7,),
    }
    decisions = resolve_leaf_placements(
        meta,
        DomainParallelConfig.from_dict({"placements": {"solution": "replicate"}}, mesh),
    )
    # The pin reaches both nested leaves, and through the shared axis, coords.
    assert decisions[("solution", "pressure")] is False
    assert decisions[("solution", "velocity")] is False
    assert decisions["coords"] is False

    with caplog.at_level(logging.WARNING):
        resolve_leaf_placements(
            meta,
            DomainParallelConfig.from_dict({"placements": {"coordz": "shard"}}, mesh),
        )
    assert any("coordz" in rec.getMessage() for rec in caplog.records)


class _NoDomainParallelReader(Reader):
    """A reader that does not implement rank-local reading."""

    def _load_sample(self, index):
        return {}

    def __len__(self):
        return 1


def test_unsupported_reader_rejects_domain_parallel():
    """Asking a non-supporting reader for domain-parallel reading is an error,
    not a silent full read on every rank."""
    with pytest.raises(ValueError, match="does not support domain-parallel"):
        _NoDomainParallelReader(domain_parallel={}, device_mesh=_StubMesh(2))
    _NoDomainParallelReader()  # no domain parallelism: fine
