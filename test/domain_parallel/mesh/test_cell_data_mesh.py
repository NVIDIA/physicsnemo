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

r"""ShardTensor support for ``cell_data`` on a ``Mesh`` with cells.

Counterpart of ``test_point_cloud_mesh.py`` for the cell batch dim: points
are ``Shard(0)`` over n_points, ``cell_data`` entries are ``Shard(0)`` over
n_cells, and cells stay replicated. Covers construction, the elementwise
transforms (rotate with ``transform_cell_data``, field normalization), and
pins that lazily computed cell caches coexist with sharded ``cell_data`` in
the same cell-batch TensorDict.

Pattern: build the full mesh identically on all ranks (seeded), shard,
run the op, gather, compare.
"""

import math

import pytest
import torch
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.datapipes.transforms.mesh.transforms import NormalizeMeshFields
from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.mesh.mesh import Mesh
from test.domain_parallel.mesh.conftest import shard_queries

# Uneven on 2/4/8 ranks for both batch dims (n_points = 3 * _N_CELLS).
_N_CELLS = 611


def _gather(t: torch.Tensor) -> torch.Tensor:
    return t.full_tensor() if hasattr(t, "full_tensor") else t


def _build_full_cell_mesh(device):
    r"""Identical full triangle soup on every rank: one well-conditioned
    triangle per cell, copies offset along x so no cell is degenerate."""
    torch.manual_seed(31)
    base = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device
    )
    offsets = torch.zeros(_N_CELLS, 1, 3, device=device)
    offsets[:, 0, 0] = 2.0 * torch.arange(_N_CELLS, device=device)
    points = (base.unsqueeze(0) + offsets).reshape(-1, 3)
    cells = torch.arange(3 * _N_CELLS, device=device, dtype=torch.int64).reshape(-1, 3)
    cell_data = {
        "pressure": torch.randn(_N_CELLS, device=device),
        "wall_shear": torch.randn(_N_CELLS, 3, device=device),
    }
    return points, cells, cell_data


def _shard_cell_mesh(mesh_1d, device):
    points, cells, cell_data = _build_full_cell_mesh(device)
    return Mesh(
        points=shard_queries(points, mesh_1d),
        cells=cells,
        cell_data={
            key: shard_queries(value, mesh_1d) for key, value in cell_data.items()
        },
    )


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_cell_data_mesh_construction_sharded(distributed_mesh):
    r"""Construction: global n_cells batch, sharded entries preserved as
    ShardTensors alongside sharded points and replicated cells."""
    dm = DistributedManager()
    mesh = _shard_cell_mesh(distributed_mesh, dm.device)

    assert mesh.n_points == 3 * _N_CELLS
    assert mesh.n_cells == _N_CELLS
    assert isinstance(mesh.points, ShardTensor)
    assert not isinstance(mesh.cells, ShardTensor)
    # __post_init__ sets the cell_data batch_size from n_cells; ShardTensor
    # entries report GLOBAL shapes, so the batch is the global cell count.
    assert tuple(mesh.cell_data.batch_size) == (_N_CELLS,)
    for key in ("pressure", "wall_shear"):
        assert isinstance(mesh.cell_data[key], ShardTensor)
        assert mesh.cell_data[key]._spec.placements == (Shard(0),)

    # Round trip: gathered entries match the full build.
    _, _, full_cell_data = _build_full_cell_mesh(dm.device)
    for key in ("pressure", "wall_shear"):
        torch.testing.assert_close(
            mesh.cell_data[key].full_tensor(), full_cell_data[key]
        )


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_rotate_transforms_cell_data_sharded(distributed_mesh):
    r"""rotate(transform_cell_data=True): elementwise per-cell math; vector
    fields rotate, scalars pass through, sharding is preserved."""
    dm = DistributedManager()

    points, cells, cell_data = _build_full_cell_mesh(dm.device)
    reference = Mesh(points=points, cells=cells, cell_data=cell_data).rotate(
        angle=math.pi / 2, axis="z", transform_cell_data=True
    )

    sharded = _shard_cell_mesh(distributed_mesh, dm.device).rotate(
        angle=math.pi / 2, axis="z", transform_cell_data=True
    )

    assert isinstance(sharded.cell_data["wall_shear"], ShardTensor)
    assert sharded.cell_data["wall_shear"]._spec.placements == (Shard(0),)
    torch.testing.assert_close(
        sharded.points.full_tensor(), reference.points, atol=1e-5, rtol=1e-5
    )
    for key in ("pressure", "wall_shear"):
        torch.testing.assert_close(
            sharded.cell_data[key].full_tensor(),
            reference.cell_data[key],
            atol=1e-5,
            rtol=1e-5,
        )


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_normalize_cell_data_sharded(distributed_mesh):
    r"""NormalizeMeshFields on sharded cell_data: pure elementwise math,
    must preserve sharding and match the unsharded transform."""
    dm = DistributedManager()

    fields = {
        "pressure": {"type": "scalar", "mean": 101325.0, "std": 250.0},
        "wall_shear": {"type": "vector", "mean": [1.0, 0.0, 0.0], "std": 0.5},
    }
    transform = NormalizeMeshFields(association="cell_data", fields=fields)
    transform.to(dm.device)

    points, cells, cell_data = _build_full_cell_mesh(dm.device)
    reference = transform(Mesh(points=points, cells=cells, cell_data=cell_data))

    sharded = transform(_shard_cell_mesh(distributed_mesh, dm.device))

    for key in ("pressure", "wall_shear"):
        assert isinstance(sharded.cell_data[key], ShardTensor)
        assert sharded.cell_data[key]._spec.placements == (Shard(0),)
        torch.testing.assert_close(
            sharded.cell_data[key].full_tensor(),
            reference.cell_data[key],
            atol=1e-5,
            rtol=1e-5,
        )


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_lazy_cell_cache_coexists_with_sharded_cell_data(distributed_mesh):
    r"""Lazily computed cell quantities (from sharded points) and sharded
    cell_data live in the same cell-batch TensorDict.

    Pins the mixed-distribution case: whatever placement the derived caches
    come out with, their values must match the unsharded reference and the
    user's cell_data entries must stay sharded.
    """
    dm = DistributedManager()

    points, cells, cell_data = _build_full_cell_mesh(dm.device)
    reference = Mesh(points=points, cells=cells, cell_data=cell_data)

    sharded = _shard_cell_mesh(distributed_mesh, dm.device)

    torch.testing.assert_close(
        _gather(sharded.cell_centroids), reference.cell_centroids, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        _gather(sharded.cell_areas), reference.cell_areas, atol=1e-4, rtol=1e-4
    )
    for key in ("pressure", "wall_shear"):
        assert isinstance(sharded.cell_data[key], ShardTensor)
