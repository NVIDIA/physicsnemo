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

"""Tests for the tensor-level Warp remeshing functional."""

import inspect
import subprocess
import sys
from typing import Literal, get_type_hints

import pytest
import torch
import warp as wp

from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.nn.functional import remeshing
from physicsnemo.nn.functional.geometry import Remeshing
from physicsnemo.nn.functional.geometry.remeshing._warp_impl import launch_forward
from physicsnemo.nn.functional.geometry.remeshing._warp_impl._kernels import (
    assign_vertices,
    project_centroids_to_surface,
    update_centroids,
)
from physicsnemo.nn.functional.geometry.remeshing._warp_impl.launch_forward import (
    _remove_nonmanifold_faces,
    _voxel_representatives,
    _weighted_sample_without_replacement,
)


def test_remeshing_function_spec_contract():
    assert Remeshing.implementations() == ("warp",)
    implementation = Remeshing._get_impls()["warp"]
    assert implementation.rank == 0
    assert implementation.baseline
    assert list(inspect.signature(remeshing).parameters) == [
        "mesh_vertices",
        "mesh_indices",
        "n_clusters",
        "max_iterations",
        "search_radius_scale",
        "voxel_width_scale",
        "hash_grid_resolution",
        "farthest_point_threshold",
        "farthest_point_oversampling",
        "implementation",
    ]
    assert get_type_hints(remeshing)["implementation"] == Literal["warp"] | None

    label, args, kwargs = next(iter(Remeshing.make_inputs_forward(device="cpu")))
    assert label == "small-v482-k64"
    assert args[0].shape == (482, 3)
    assert args[1].ndim == 2 and args[1].shape[1] == 3
    assert args[2] == 64
    assert kwargs == {}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_remeshing_public_api_fake_tensor_propagation(device):
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, statically_known_true

    with FakeTensorMode(shape_env=ShapeEnv()):
        vertices = torch.empty((16, 3), dtype=torch.float64, device=device)
        indices = torch.empty((20, 3), dtype=torch.int32, device=device)
        output_vertices, output_indices = remeshing(
            vertices,
            indices,
            8,
            implementation="warp",
        )

    assert isinstance(output_vertices, FakeTensor)
    assert isinstance(output_indices, FakeTensor)
    assert output_vertices.shape[1:] == (3,)
    assert output_indices.shape[1:] == (3,)
    assert output_vertices.dtype == vertices.dtype
    assert output_indices.dtype == torch.int64
    assert output_vertices.device == vertices.device
    assert output_indices.device == indices.device
    assert statically_known_true(output_vertices.shape[0] >= 3)
    assert statically_known_true(output_vertices.shape[0] <= 8)
    assert statically_known_true(output_indices.shape[0] >= 1)


def test_remeshing_custom_op_tags():
    tags = torch.ops.physicsnemo.remeshing_warp.default.tags
    assert torch.Tag.nondeterministic_bitwise in tags
    assert torch.Tag.cudagraph_unsafe in tags


@pytest.mark.parametrize(
    ("keyword", "value", "error", "match"),
    [
        ("search_radius_scale", 0.0, ValueError, "finite and positive"),
        ("search_radius_scale", torch.inf, ValueError, "finite and positive"),
        ("search_radius_scale", True, TypeError, "real number"),
        ("voxel_width_scale", torch.nan, ValueError, "finite and positive"),
        ("voxel_width_scale", "1.0", TypeError, "real number"),
        ("hash_grid_resolution", 0, ValueError, "at least 1"),
        ("hash_grid_resolution", 64.0, TypeError, "integer"),
        ("hash_grid_resolution", 257, ValueError, "at most 256"),
        ("farthest_point_threshold", -1, ValueError, "at least 0"),
        ("farthest_point_threshold", False, TypeError, "integer"),
        ("farthest_point_oversampling", 0, ValueError, "at least 1"),
        ("farthest_point_oversampling", 2.0, TypeError, "integer"),
    ],
)
def test_remeshing_rejects_invalid_tuning_values_and_types(
    keyword, value, error, match
):
    vertices = torch.rand(16, 3)
    indices = torch.tensor([[0, 1, 2], [2, 3, 0]])
    with pytest.raises(error, match=match):
        remeshing(vertices, indices, 8, **{keyword: value})


def test_remeshing_rejects_unrepresentable_integer_scale():
    vertices = torch.rand(16, 3)
    indices = torch.tensor([[0, 1, 2], [2, 3, 0]])
    with pytest.raises(ValueError, match="finite and positive"):
        remeshing(vertices, indices, 8, search_radius_scale=10**1_000)


def test_remeshing_rejects_unknown_implementation():
    vertices = torch.rand(16, 3)
    indices = torch.tensor([[0, 1, 2], [2, 3, 0]])
    with pytest.raises(KeyError, match="No implementation named 'torch'"):
        remeshing(vertices, indices, 8, implementation="torch")


def test_remeshing_rejects_invalid_tensor_inputs():
    vertices = torch.rand(16, 3)
    indices = torch.tensor([[0, 1, 2], [2, 3, 0]])

    with pytest.raises(ValueError, match="mesh_vertices must have shape"):
        remeshing(vertices[:, :2], indices, 8)
    with pytest.raises(ValueError, match="mesh_indices must have shape"):
        remeshing(vertices, indices.reshape(-1), 8)
    meta_vertices = torch.empty(16, 3, device="meta")
    meta_indices = torch.empty(2, 3, dtype=torch.int64, device="meta")
    with pytest.raises(ValueError, match="supports CPU and CUDA tensors"):
        remeshing(meta_vertices, meta_indices, 8)


@pytest.mark.parametrize(
    "imports",
    [
        "import physicsnemo.mesh.remeshing; import physicsnemo.nn.functional.geometry",
        "import physicsnemo.nn.functional.geometry; import physicsnemo.mesh.remeshing",
    ],
)
def test_remeshing_import_order(imports):
    subprocess.run(  # noqa: S603 - interpreter and snippets are test constants
        [sys.executable, "-c", imports],
        check=True,
        capture_output=True,
        text=True,
    )


def test_remeshing_cpu_tensor_api_contract():
    source = sphere_icosahedral.load(subdivisions=1)
    output_vertices, output_indices = remeshing(
        source.points,
        source.cells,
        24,
        max_iterations=1,
        search_radius_scale=2.0,
        farthest_point_threshold=0,
        implementation="warp",
    )

    assert 3 <= output_vertices.shape[0] <= 24
    assert output_vertices.device.type == "cpu"
    assert output_vertices.dtype == source.points.dtype
    assert output_indices.ndim == 2 and output_indices.shape[1] == 3
    assert output_indices.device.type == "cpu"
    assert output_indices.dtype == torch.int64


def test_remeshing_cpu_torch_compile():
    source = sphere_icosahedral.load(subdivisions=1)
    compiled = torch.compile(remeshing, backend="eager", fullgraph=True, dynamic=True)

    output_vertices, output_indices = compiled(
        source.points,
        source.cells,
        24,
        max_iterations=1,
        search_radius_scale=2.0,
        farthest_point_threshold=0,
        implementation="warp",
    )

    assert 3 <= output_vertices.shape[0] <= 24
    assert output_indices.ndim == 2 and output_indices.shape[1] == 3
    assert output_indices.dtype == torch.int64


def test_remeshing_cpu_custom_op_contract():
    from physicsnemo.nn.functional.geometry.remeshing._warp_impl import remeshing_warp

    source = sphere_icosahedral.load(subdivisions=1)
    torch.library.opcheck(
        remeshing_warp,
        args=(source.points, source.cells, 24, 1, 1.6, 1.15, 128, 256, 4),
        rtol=1.0e-4,
        atol=1.0e-4,
    )


def test_weighted_sampling_avoids_torch_category_limit(monkeypatch):
    def reject_multinomial(*args, **kwargs):
        raise AssertionError("torch.multinomial must not be used")

    monkeypatch.setattr(torch, "multinomial", reject_multinomial)
    monkeypatch.setattr(launch_forward, "_WEIGHTED_SAMPLE_CHUNK_SIZE", 2)
    weights = torch.tensor([1.0, 0.0, 4.0, 2.0, 3.0])

    first = _weighted_sample_without_replacement(weights, 3)
    second = _weighted_sample_without_replacement(weights, 3)

    torch.testing.assert_close(first, second)
    assert torch.unique(first).numel() == 3
    assert 1 not in first


def test_assign_vertices_brute_force_fallback():
    points = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [10.0, 0.0, 0.0]])
    centroids = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vertex_areas = torch.ones(3)
    labels = torch.empty(3, dtype=torch.int32)
    centroid_sums = torch.zeros(2, 3)
    centroid_areas = torch.zeros(2)
    search_radius = 1.0e-4

    grid = wp.HashGrid(dim_x=8, dim_y=8, dim_z=8, device="cpu")
    wp_centroids = wp.from_torch(centroids, dtype=wp.vec3f)
    grid.build(wp_centroids, radius=search_radius)
    wp.launch(
        assign_vertices,
        dim=points.shape[0],
        inputs=[
            grid.id,
            wp.from_torch(points, dtype=wp.vec3f),
            wp_centroids,
            wp.from_torch(vertex_areas, dtype=wp.float32),
            wp.from_torch(labels, dtype=wp.int32),
            wp.from_torch(centroid_sums, dtype=wp.float32),
            wp.from_torch(centroid_areas, dtype=wp.float32),
            search_radius,
            0,
        ],
        device="cpu",
    )

    torch.testing.assert_close(labels, torch.tensor([0, 1, 1], dtype=torch.int32))


def test_update_centroids_resets_accumulators():
    centroids = torch.tensor([[0.0, 0.0, 0.0], [9.0, 9.0, 9.0]])
    centroid_sums = torch.tensor([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]])
    centroid_areas = torch.tensor([2.0, 0.0])

    wp.launch(
        update_centroids,
        dim=centroids.shape[0],
        inputs=[
            wp.from_torch(centroids, dtype=wp.vec3f),
            wp.from_torch(centroid_sums, dtype=wp.float32),
            wp.from_torch(centroid_areas, dtype=wp.float32),
        ],
        device="cpu",
    )

    torch.testing.assert_close(
        centroids,
        torch.tensor([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]]),
    )
    torch.testing.assert_close(centroid_sums, torch.zeros_like(centroid_sums))
    torch.testing.assert_close(centroid_areas, torch.zeros_like(centroid_areas))


def test_project_centroids_uses_warp_barycentric_convention():
    points = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    indices = torch.tensor([0, 1, 2], dtype=torch.int32)
    centroids = torch.tensor([[0.6, 0.2, 1.0]])
    source_surface = wp.Mesh(
        points=wp.from_torch(points, dtype=wp.vec3f),
        indices=wp.from_torch(indices, dtype=wp.int32),
    )

    wp.launch(
        project_centroids_to_surface,
        dim=1,
        inputs=[
            source_surface.id,
            wp.from_torch(centroids, dtype=wp.vec3f),
            float(1.0e30),
        ],
        device="cpu",
    )

    torch.testing.assert_close(centroids, torch.tensor([[0.6, 0.2, 0.0]]))


@pytest.mark.cuda
def test_remeshing_tensor_api_contract():
    source = sphere_icosahedral.load(subdivisions=2, device="cuda")
    output_vertices, output_indices = remeshing(
        source.points,
        source.cells,
        48,
        implementation="warp",
    )

    assert output_vertices.device == source.points.device
    assert output_vertices.dtype == source.points.dtype
    assert not output_vertices.requires_grad
    assert 3 <= output_vertices.shape[0] <= 48
    assert output_indices.device == source.cells.device
    assert output_indices.dtype == torch.int64
    assert output_indices.ndim == 2 and output_indices.shape[1] == 3
    assert int(output_indices.min()) >= 0
    assert int(output_indices.max()) < output_vertices.shape[0]


@pytest.mark.cuda
def test_remeshing_public_api_torch_compile():
    source = sphere_icosahedral.load(subdivisions=1, device="cuda")
    compiled = torch.compile(remeshing, backend="eager", fullgraph=True, dynamic=True)

    output_vertices, output_indices = compiled(
        source.points,
        source.cells,
        24,
        max_iterations=1,
        implementation="warp",
    )

    assert 3 <= output_vertices.shape[0] <= 24
    assert output_indices.ndim == 2 and output_indices.shape[1] == 3
    assert output_indices.dtype == torch.int64


@pytest.mark.cuda
def test_remeshing_custom_op_contract():
    from physicsnemo.nn.functional.geometry.remeshing._warp_impl import remeshing_warp

    source = sphere_icosahedral.load(subdivisions=2, device="cuda")
    # Break exact symmetries that put projected centroids on triangle ties.
    # The operation is tagged as bitwise nondeterministic because it uses Warp
    # atomics, but opcheck should still catch meaningful AOT dispatch errors.
    ramp = torch.linspace(-1.0, 1.0, source.n_points, device="cuda")
    source.points[:, 0].add_(1.0e-3 * ramp)
    source.points[:, 1].add_(3.7e-4 * ramp.square())
    torch.library.opcheck(
        remeshing_warp,
        args=(source.points, source.cells, 32, 1, 1.6, 1.15, 128, 256, 4),
        rtol=1.0e-4,
        atol=1.0e-4,
    )


@pytest.mark.cuda
def test_voxel_representatives_avoid_packed_key_overflow():
    points = sphere_icosahedral.load(subdivisions=3, device="cuda").points

    representatives = _voxel_representatives(
        points,
        points.amin(dim=0),
        points.amax(dim=0),
        torch.finfo(torch.float32).tiny,
    )

    assert representatives.numel() == points.shape[0]
    assert torch.unique(representatives).numel() == points.shape[0]


@pytest.mark.cuda
def test_nonmanifold_cleanup_handles_high_edge_incidence():
    n_faces = 10
    points = torch.zeros(n_faces + 2, 3, device="cuda")
    points[1, 0] = 1.0
    angles = torch.arange(n_faces, device="cuda") * 0.2
    points[2:, 1] = torch.cos(angles)
    points[2:, 2] = torch.sin(angles)
    cells = torch.stack(
        [
            torch.zeros(n_faces, dtype=torch.int64, device="cuda"),
            torch.ones(n_faces, dtype=torch.int64, device="cuda"),
            torch.arange(2, n_faces + 2, device="cuda"),
        ],
        dim=1,
    )

    cleaned = _remove_nonmanifold_faces(points, cells, points.shape[0])

    edges = torch.cat([cleaned[:, [0, 1]], cleaned[:, [1, 2]], cleaned[:, [2, 0]]])
    _, counts = torch.unique(
        torch.sort(edges, dim=1).values,
        dim=0,
        return_counts=True,
    )
    assert int(counts.max()) <= 2
