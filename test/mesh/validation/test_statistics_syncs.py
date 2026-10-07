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

"""Mesh statistics are read in one transfer, with unchanged values."""

import math
import warnings

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.planar import structured_grid
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.mesh.primitives.volumes import cube_volume
from physicsnemo.mesh.validation import compute_mesh_statistics
from physicsnemo.mesh.validation.quality import compute_quality_metrics


def _reference_statistics(mesh: Mesh, tolerance: float = 1e-10) -> dict:
    """The statistics as computed one ``.item()`` at a time."""
    stats = {
        "n_points": mesh.n_points,
        "n_cells": mesh.n_cells,
        "n_manifold_dims": mesh.n_manifold_dims,
        "n_spatial_dims": mesh.n_spatial_dims,
    }
    areas = mesh.cell_areas
    stats["n_degenerate_cells"] = (areas < tolerance).sum().item()
    stats["n_isolated_vertices"] = mesh.n_points - len(torch.unique(mesh.cells))

    def summary(x):
        return (
            x.min().item(),
            x.mean().item(),
            x.max().item(),
            x.std(correction=0).item(),
        )

    stats["cell_area_stats"] = summary(areas)
    quality = compute_quality_metrics(mesh)
    min_edge, max_edge = quality["min_edge_length"], quality["max_edge_length"]
    stats["edge_length_stats"] = (
        min_edge.min().item(),
        (min_edge.mean().item() + max_edge.mean().item()) / 2.0,
        max_edge.max().item(),
        max_edge.std(correction=0).item(),
    )
    for key in ("aspect_ratio", "quality_score"):
        if key in quality.keys():
            stats[f"{key}_stats"] = summary(quality[key])
    return stats


def _with_defects(mesh: Mesh) -> Mesh:
    """Add two isolated vertices and a cell with a repeated vertex."""
    points = torch.cat([mesh.points, mesh.points[:2] + 10.0])
    degenerate = mesh.cells[:1].clone()
    degenerate[0, 1] = degenerate[0, 0]
    return Mesh(points=points, cells=torch.cat([mesh.cells, degenerate]))


MESHES = {
    "segments": lambda d: structured_grid.load(n_x=5, n_y=4, device=d).get_facet_mesh(),
    "triangles_2d": lambda d: structured_grid.load(n_x=6, n_y=5, device=d),
    "triangles_3d": lambda d: sphere_icosahedral.load(subdivisions=2, device=d),
    "tetrahedra": lambda d: cube_volume.load(subdivisions=2, device=d),
}


@pytest.mark.parametrize("name", MESHES)
@pytest.mark.parametrize("defects", [False, True])
def test_statistics_match_reference(name, defects, device):
    mesh = MESHES[name](device)
    if defects:
        mesh = _with_defects(mesh)

    stats = compute_mesh_statistics(mesh)
    expected = _reference_statistics(mesh)

    assert list(stats) == list(expected)
    for key, value in expected.items():
        assert type(stats[key]) is type(value)
        # Exact equality, where NaN (std of infinite aspect ratios) equals NaN
        flat = value if isinstance(value, tuple) else (value,)
        actual = stats[key] if isinstance(value, tuple) else (stats[key],)
        assert all(
            a == b or (math.isnan(a) and math.isnan(b)) for a, b in zip(actual, flat)
        ), (key, stats[key], value)


def test_isolated_vertices_are_counted(device):
    mesh = Mesh(
        points=torch.rand(6, 2, device=device),
        cells=torch.tensor([[0, 1, 2], [2, 1, 4]], device=device),
    )
    assert compute_mesh_statistics(mesh)["n_isolated_vertices"] == 2


def _count_cuda_syncs(fn) -> int:
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn()
        finally:
            torch.cuda.set_sync_debug_mode(previous)
    return sum("synchronizing CUDA operation" in str(w.message) for w in caught)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("name", MESHES)
def test_statistics_add_one_sync_to_quality_metrics(name):
    mesh = MESHES[name]("cuda")
    _ = mesh.cell_areas
    compute_mesh_statistics(mesh)  # warm up

    quality_syncs = _count_cuda_syncs(lambda: compute_quality_metrics(mesh))
    total_syncs = _count_cuda_syncs(lambda: compute_mesh_statistics(mesh))
    assert total_syncs <= quality_syncs + 1
