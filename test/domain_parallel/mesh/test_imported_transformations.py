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

r"""Reruns the geometric-transformation suite on meshes with sharded points.

The base tests in ``test/mesh/transformations/test_transformations.py``
validate ``translate``/``rotate``/``scale``/``transform`` numerics and cache
propagation. Here the four core test classes rerun with the mesh points (a
point quantity) sharded ``Shard(0)`` over the domain mesh: an autouse
fixture rebinds the base module's ``create_mesh_with_caches`` factory to a
variant that warms the caches on the full mesh — the warmed caches are all
cell quantities and stay plain — and then rebuilds the mesh with sharded
points. Tests inside these classes that construct meshes inline run on
plain tensors and simply re-prove the ops are unaffected by the distributed
session.

Not imported: the error/edge/matrix-helper classes and the deformation-
adjacent tests (out of scope), and classes whose meshes are all built
inline, which would add no distributed coverage.
"""

import pytest
import torch

import test.mesh.transformations.test_transformations as base
from physicsnemo.mesh.mesh import Mesh
from test.domain_parallel.mesh.conftest import shard_queries

pytestmark = [pytest.mark.multigpu_static, pytest.mark.timeout(300)]


def _tile_mesh(base_mesh: Mesh, n_spatial_dims: int, n_points_target: int) -> Mesh:
    """Tile ``base_mesh`` along the first axis to reach ``n_points_target`` points.

    Copies are offset by 20 units in the first spatial dimension so they stay
    geometrically disjoint.  Cells whose vertex indices exceed
    ``n_points_target - 1`` (the final partial copy) are dropped.  All
    geometry comes from the well-conditioned literal base mesh, so no
    degenerate cells are introduced.
    """
    base_n = base_mesh.n_points
    n_copies = max(1, (n_points_target + base_n - 1) // base_n)

    pts_parts = [base_mesh.points]
    cell_parts = [base_mesh.cells]
    for k in range(1, n_copies):
        offset = torch.zeros(n_spatial_dims, device=base_mesh.points.device)
        offset[0] = k * 20.0
        pts_parts.append(base_mesh.points + offset)
        cell_parts.append(base_mesh.cells + k * base_n)

    points = torch.cat(pts_parts, dim=0)[:n_points_target]
    cells = torch.cat(cell_parts, dim=0)
    valid = (cells < n_points_target).all(dim=-1)
    return Mesh(points=points, cells=cells[valid])


@pytest.fixture(autouse=True)
def _sharded_mesh_factory(distributed_mesh, monkeypatch):
    # 8 points per rank + 3 keeps an uneven split on all power-of-2 world sizes.
    world_size = distributed_mesh.size(0)
    n_points = 8 * world_size + 3

    def sharded_create_mesh_with_caches(n_spatial_dims, n_manifold_dims, device="cpu"):
        # Tile the literal base mesh (well-conditioned geometry, no random cells).
        base = base_module_factory(n_spatial_dims, n_manifold_dims, device=device)
        full = _tile_mesh(base, n_spatial_dims, n_points)
        # Warm caches on the full plain mesh before sharding.
        _ = full.cell_areas
        _ = full.cell_centroids
        if full.codimension == 1:
            _ = full.cell_normals
        return Mesh(
            points=shard_queries(full.points, distributed_mesh),
            cells=full.cells,
            point_data=full.point_data,
            cell_data=full.cell_data,
            global_data=full.global_data,
            _cache=full._cache,
        )

    base_module_factory = base.create_mesh_with_caches
    monkeypatch.setattr(
        base, "create_mesh_with_caches", sharded_create_mesh_with_caches
    )


class TestTranslation(base.TestTranslation):
    pass


class TestRotation(base.TestRotation):
    pass


class TestScale(base.TestScale):
    pass


class TestTransform(base.TestTransform):
    pass
