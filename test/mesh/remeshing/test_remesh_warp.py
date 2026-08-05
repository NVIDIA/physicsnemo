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

"""Correctness and API tests for Warp-accelerated surface remeshing."""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.boundaries import is_watertight
from physicsnemo.mesh.primitives.surfaces import plane, sphere_icosahedral, torus
from physicsnemo.mesh.remeshing import remesh
from physicsnemo.nn.functional.geometry.remeshing import remeshing

pytestmark = pytest.mark.cuda


def _euler_characteristic(mesh: Mesh) -> int:
    edges = torch.cat(
        [
            mesh.cells[:, [0, 1]],
            mesh.cells[:, [1, 2]],
            mesh.cells[:, [2, 0]],
        ]
    )
    n_edges = torch.unique(torch.sort(edges, dim=1).values, dim=0).shape[0]
    return mesh.n_points - n_edges + mesh.n_cells


def _assert_clean_topology(
    mesh: Mesh,
    *,
    check_geometric_validation: bool = True,
) -> None:
    assert mesh.n_cells > 0
    assert mesh.cells.dtype == torch.int64
    assert int(mesh.cells.min()) >= 0
    assert int(mesh.cells.max()) < mesh.n_points
    assert torch.unique(mesh.cells).numel() == mesh.n_points

    sorted_cells = torch.sort(mesh.cells, dim=1).values
    assert not (sorted_cells[:, 1:] == sorted_cells[:, :-1]).any()
    assert torch.unique(sorted_cells, dim=0).shape[0] == mesh.n_cells

    report = mesh.validate(check_manifoldness=True)
    assert report["is_manifold"]
    if check_geometric_validation:
        assert report["valid"]


def test_warp_remesh_closed_sphere_contract():
    source = sphere_icosahedral.load(subdivisions=3, device="cuda")
    source.point_data["temperature"] = torch.arange(
        source.n_points, dtype=source.points.dtype, device="cuda"
    )
    source.cell_data["pressure"] = torch.ones(source.n_cells, device="cuda")
    source.global_data["case_id"] = torch.tensor(7, device="cuda")
    original_points = source.points.clone()
    original_cells = source.cells.clone()

    output = remesh(source, 128)

    assert 3 <= output.n_points <= 128
    assert output.n_manifold_dims == 2 and output.n_spatial_dims == 3
    assert output.points.device == source.points.device
    assert output.points.dtype == source.points.dtype
    assert not output.points.requires_grad
    assert len(output.point_data.keys(include_nested=True, leaves_only=True)) == 0
    assert len(output.cell_data.keys(include_nested=True, leaves_only=True)) == 0
    assert int(output.global_data["case_id"]) == 7
    torch.testing.assert_close(source.points, original_points)
    torch.testing.assert_close(source.cells, original_cells)

    _assert_clean_topology(output)
    assert is_watertight(output)
    assert _euler_characteristic(output) == 2
    outward = (output.cell_normals * output.cell_centroids).sum(dim=1)
    assert (outward > 0).all()
    # Projected output points lie on the piecewise-linear source sphere.
    assert float((output.points.norm(dim=1) - 1.0).abs().max()) < 0.01

    edges = torch.cat(
        [
            output.cells[:, [0, 1]],
            output.cells[:, [1, 2]],
            output.cells[:, [2, 0]],
        ]
    )
    edges = torch.unique(torch.sort(edges, dim=1).values, dim=0)
    edge_lengths = torch.linalg.vector_norm(
        output.points[edges[:, 0]] - output.points[edges[:, 1]], dim=1
    )
    assert float(edge_lengths.std() / edge_lengths.mean()) < 0.25
    relative_area_error = (
        output.cell_areas.sum() / source.cell_areas.sum() - 1.0
    ).abs()
    assert float(relative_area_error) < 0.05


def test_cuda_default_dispatch_and_mesh_method():
    source = sphere_icosahedral.load(subdivisions=2, device="cuda")
    direct = remesh(source, 48)
    method = source.remesh(48)

    torch.testing.assert_close(direct.cells, method.cells)
    _assert_clean_topology(direct)
    _assert_clean_topology(method)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_warp_remesh_preserves_dtype_and_accepts_noncontiguous_geometry(dtype):
    base = sphere_icosahedral.load(subdivisions=3)
    storage = torch.empty(base.n_points, 6, dtype=dtype, device="cuda")
    points = storage[:, ::2]
    points.copy_(base.points.to(device="cuda", dtype=dtype))
    assert not points.is_contiguous()
    source = Mesh(points=points, cells=base.cells.to(device="cuda", dtype=torch.int32))

    output = remesh(source, 96)

    assert output.points.dtype == dtype
    assert output.cells.dtype == torch.int64
    _assert_clean_topology(output)


@pytest.mark.parametrize("scale", [1.0e-4, 1.0e4])
def test_warp_remesh_is_scale_equivariant(scale):
    unit_source = sphere_icosahedral.load(subdivisions=3, device="cuda")
    source = Mesh(points=scale * unit_source.points, cells=unit_source.cells.clone())

    output = remesh(source, 128)

    assert 3 <= output.n_points <= 128
    _assert_clean_topology(output)
    assert is_watertight(output)
    assert _euler_characteristic(output) == 2
    normalized_radius_error = (output.points.norm(dim=1) / scale - 1.0).abs()
    assert float(normalized_radius_error.max()) < 0.01


@pytest.mark.parametrize(
    ("offset", "scale"),
    [(1.0e7, 1.0), (1.0e100, 1.0e90), (0.0, 1.0e-100)],
)
def test_warp_remesh_normalizes_world_coordinates(offset, scale):
    unit_source = sphere_icosahedral.load(subdivisions=3, device="cuda")
    source = Mesh(
        points=offset + scale * unit_source.points.to(torch.float64),
        cells=unit_source.cells.clone(),
    )

    output = remesh(source, 128)

    assert 3 <= output.n_points <= 128
    assert output.points.dtype == torch.float64
    _assert_clean_topology(
        output,
        check_geometric_validation=scale >= 1.0e-50,
    )
    normalized_radius_error = (
        ((output.points - offset) / scale).norm(dim=1) - 1.0
    ).abs()
    assert float(normalized_radius_error.max()) < 0.01


def test_warp_remesh_preserves_valid_thin_faces():
    source = Mesh(
        points=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0e-7, 0.0],
                [0.0, 1.0e-7, 0.0],
            ],
            device="cuda",
            dtype=torch.float64,
        ),
        cells=torch.tensor([[0, 1, 2], [0, 2, 3]], device="cuda"),
    )

    output = remesh(source, 4, max_iterations=0)

    assert output.n_points == 4
    assert output.n_cells == 2
    assert (output.cell_areas > 0.0).all()
    _assert_clean_topology(output)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "search_radius_scale": 2.0,
            "voxel_width_scale": 0.9,
            "hash_grid_resolution": 64,
            "farthest_point_threshold": 64,
            "farthest_point_oversampling": 2,
        },
        {
            "search_radius_scale": 1.8,
            "voxel_width_scale": 1.0,
            "hash_grid_resolution": 96,
            "farthest_point_threshold": 0,
            "farthest_point_oversampling": 3,
        },
    ],
    ids=["farthest-point-initialization", "voxel-initialization"],
)
def test_warp_remesh_accepts_custom_tuning(kwargs):
    source = sphere_icosahedral.load(subdivisions=2, device="cuda")

    points, cells = remeshing(
        source.points,
        source.cells,
        48,
        max_iterations=1,
        **kwargs,
    )
    output = Mesh(points=points, cells=cells)

    assert output.points.device.type == "cuda"
    _assert_clean_topology(output)


@pytest.mark.parametrize(
    ("surface", "target", "expected_euler", "watertight"),
    [
        ("plane", 100, 1, False),
        ("torus", 256, 0, True),
    ],
)
def test_warp_remesh_surface_topology(surface, target, expected_euler, watertight):
    if surface == "plane":
        source = plane.load(subdivisions=20, device="cuda")
    else:
        source = torus.load(n_major=48, n_minor=24, device="cuda")
    output = remesh(source, target)

    _assert_clean_topology(output)
    assert _euler_characteristic(output) == expected_euler
    assert bool(is_watertight(output)) is watertight


def test_warp_remesh_preserves_separated_components():
    first = sphere_icosahedral.load(subdivisions=3, device="cuda")
    second = Mesh(
        points=first.points + torch.tensor([4.0, 0.0, 0.0], device="cuda"),
        cells=first.cells.clone(),
    )
    source = Mesh.merge([first, second])

    output = remesh(source, 128)

    _assert_clean_topology(output)
    assert is_watertight(output)
    assert _euler_characteristic(output) == 4


def test_warp_remesh_is_nondifferentiable():
    source = sphere_icosahedral.load(subdivisions=3, device="cuda")
    source.points.requires_grad_(True)

    first = remesh(source, 96)
    second = remesh(source, 96)

    assert not first.points.requires_grad
    assert torch.isfinite(first.points).all()
    assert torch.isfinite(second.points).all()


def test_warp_remesh_uses_current_cuda_stream():
    source = sphere_icosahedral.load(subdivisions=3, device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        output = remesh(source, 96)
        marker = output.points.square().sum()
    stream.synchronize()

    assert torch.isfinite(marker)
    _assert_clean_topology(output)


@pytest.mark.parametrize(
    ("n_clusters", "error", "match"),
    [
        (True, TypeError, "integer"),
        (2, ValueError, "at least 3"),
        (10_000, ValueError, "cannot exceed"),
    ],
)
def test_warp_remesh_rejects_invalid_cluster_counts(n_clusters, error, match):
    source = sphere_icosahedral.load(subdivisions=2, device="cuda")
    with pytest.raises(error, match=match):
        remesh(source, n_clusters)


@pytest.mark.parametrize("max_iterations", [-1, 1.5, True, None])
def test_warp_remesh_rejects_invalid_iteration_counts(max_iterations):
    source = sphere_icosahedral.load(subdivisions=2, device="cuda")
    with pytest.raises((TypeError, ValueError), match="max_iterations"):
        remesh(
            source,
            32,
            max_iterations=max_iterations,
        )


def test_warp_remesh_rejects_unsafe_geometry():
    nonfinite = sphere_icosahedral.load(subdivisions=2, device="cuda")
    nonfinite.points[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        remesh(nonfinite, 32)

    invalid_cells = sphere_icosahedral.load(subdivisions=2, device="cuda")
    invalid_cells.cells[0, 0] = invalid_cells.n_points
    with pytest.raises(ValueError, match="indices"):
        remesh(invalid_cells, 32)
