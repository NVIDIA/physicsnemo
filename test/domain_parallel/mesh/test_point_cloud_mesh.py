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

r"""ShardTensor support for point-cloud ``Mesh``.

A ``Mesh`` with no cells whose ``points`` and ``point_data`` entries are
``Shard(0)`` ShardTensors over the domain mesh behaves like the unsharded
mesh — construction succeeds with *global* ``n_points`` / TensorDict batch
size, and the recipe-relevant transforms (CenterMesh, NormalizeMeshFields)
produce results whose ``full_tensor()`` matches the unsharded reference.

Pattern: build the full mesh identically on all ranks (seeded), shard,
run the op, gather, compare.
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.datapipes.transforms.mesh.transforms import (
    CenterMesh,
    NormalizeMeshFields,
)
from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor, scatter_tensor
from physicsnemo.mesh.mesh import Mesh

# Uneven on 2/4/8 ranks: exercises the uneven-shard bookkeeping.
_N_POINTS = 1234


def _build_full_point_cloud(device):
    r"""Identical full point cloud on every rank (autouse seed fixture)."""
    torch.manual_seed(17)
    points = torch.randn(_N_POINTS, 3, device=device)
    point_data = {
        "pressure": torch.randn(_N_POINTS, 1, device=device),
        "velocity": torch.randn(_N_POINTS, 3, device=device),
    }
    global_data = {"u_inf": torch.tensor([30.0, 0.0, 0.0], device=device)}
    return points, point_data, global_data


def _shard_point_cloud(mesh_1d, device):
    points, point_data, global_data = _build_full_point_cloud(device)
    sharded_points = scatter_tensor(points, 0, mesh_1d, (Shard(0),))
    sharded_point_data = {
        key: scatter_tensor(value, 0, mesh_1d, (Shard(0),))
        for key, value in point_data.items()
    }
    # global_data stays plain / replicated — scalar batch, per-sample data.
    return Mesh(
        points=sharded_points,
        point_data=sharded_point_data,
        global_data=global_data,
    )


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_point_cloud_mesh_construction_sharded(distributed_mesh):
    r"""Construction: global n_points, consistent TensorDict batch, sharded
    entries preserved as ShardTensors."""
    dm = DistributedManager()
    mesh = _shard_point_cloud(distributed_mesh, dm.device)

    assert mesh.n_points == _N_POINTS
    assert mesh.n_cells == 0
    assert mesh.n_spatial_dims == 3
    assert isinstance(mesh.points, ShardTensor)
    assert mesh.points._spec.placements == (Shard(0),)
    # __post_init__ sets TensorDict batch_size from points.shape[0]; with
    # ShardTensors that is the GLOBAL count, and all point_data entries must
    # be consistently sharded to satisfy it.
    assert tuple(mesh.point_data.batch_size) == (_N_POINTS,)
    assert isinstance(mesh.point_data["pressure"], ShardTensor)

    # Round trip: gathered points match the full build.
    full_points, _, _ = _build_full_point_cloud(dm.device)
    torch.testing.assert_close(mesh.points.full_tensor(), full_points)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_center_mesh_sharded(distributed_mesh):
    r"""CenterMesh on a sharded point cloud: the COM (``points.mean(dim=0)``,
    the n_cells==0 path) is a Partial->Replicate reduction; the translated
    points must match the unsharded transform."""
    dm = DistributedManager()

    full_points, full_point_data, full_global = _build_full_point_cloud(dm.device)
    reference = CenterMesh()(
        Mesh(points=full_points, point_data=full_point_data, global_data=full_global)
    )

    sharded = CenterMesh()(_shard_point_cloud(distributed_mesh, dm.device))

    assert isinstance(sharded.points, ShardTensor)
    assert sharded.points._spec.placements == (Shard(0),)
    torch.testing.assert_close(
        sharded.points.full_tensor(), reference.points, atol=1e-5, rtol=1e-5
    )


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_normalize_mesh_fields_sharded(distributed_mesh):
    r"""NormalizeMeshFields on sharded point_data: pure elementwise math,
    must preserve sharding and match the unsharded transform."""
    dm = DistributedManager()

    fields = {
        "pressure": {"type": "scalar", "mean": 101325.0, "std": 250.0},
        "velocity": {"type": "vector", "mean": [30.0, 0.0, 0.0], "std": 5.0},
    }
    transform = NormalizeMeshFields(association="point_data", fields=fields)
    transform.to(dm.device)

    full_points, full_point_data, full_global = _build_full_point_cloud(dm.device)
    reference = transform(
        Mesh(points=full_points, point_data=full_point_data, global_data=full_global)
    )

    sharded = transform(_shard_point_cloud(distributed_mesh, dm.device))

    for key in ("pressure", "velocity"):
        assert isinstance(sharded.point_data[key], ShardTensor)
        assert sharded.point_data[key]._spec.placements == (Shard(0),)
        torch.testing.assert_close(
            sharded.point_data[key].full_tensor(),
            reference.point_data[key],
            atol=1e-5,
            rtol=1e-5,
        )


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_translate_preserves_sharding(distributed_mesh):
    r"""Basic geometric op sanity: translate is elementwise on points and
    must keep the Shard(0) placement and shard shapes intact."""
    dm = DistributedManager()
    mesh = _shard_point_cloud(distributed_mesh, dm.device)
    offset = torch.tensor([1.0, -2.0, 0.5], device=dm.device)

    translated = mesh.translate(offset)

    assert isinstance(translated.points, ShardTensor)
    assert translated.points._spec.placements == (Shard(0),)
    assert (
        translated.points._spec.sharding_shapes() == mesh.points._spec.sharding_shapes()
    )
    full_points, _, _ = _build_full_point_cloud(dm.device)
    torch.testing.assert_close(translated.points.full_tensor(), full_points + offset)
