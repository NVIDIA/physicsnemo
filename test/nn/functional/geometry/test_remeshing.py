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

from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.nn.functional.geometry.remeshing import (
    Remeshing,
    WarpRemeshOptions,
    remeshing,
)
from physicsnemo.nn.functional.geometry.remeshing._warp_impl.launch_forward import (
    _remove_nonmanifold_faces,
    _voxel_representatives,
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
        "warp_options",
        "implementation",
    ]
    assert get_type_hints(remeshing)["implementation"] == Literal["warp"] | None

    label, args, kwargs = next(iter(Remeshing.make_inputs_forward(device="cpu")))
    assert label == "small-v482-k64"
    assert args[0].shape == (482, 3)
    assert args[1].ndim == 2 and args[1].shape[1] == 3
    assert args[2] == 64
    assert kwargs == {}


def test_remeshing_public_api_fake_tensor_propagation():
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, statically_known_true

    with FakeTensorMode(shape_env=ShapeEnv()):
        vertices = torch.empty((16, 3), dtype=torch.float64, device="cuda")
        indices = torch.empty((20, 3), dtype=torch.int32, device="cuda")
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
def test_warp_remesh_options_reject_invalid_values_and_types(
    keyword, value, error, match
):
    with pytest.raises(error, match=match):
        WarpRemeshOptions(**{keyword: value})


def test_warp_remesh_options_reject_unrepresentable_integer_scale():
    with pytest.raises(ValueError, match="finite and positive"):
        WarpRemeshOptions(search_radius_scale=10**1_000)


def test_remeshing_rejects_cpu_and_invalid_tensor_inputs():
    vertices = torch.rand(16, 3)
    indices = torch.tensor([[0, 1, 2], [2, 3, 0]])

    with pytest.raises(ValueError, match="requires CUDA tensors"):
        remeshing(vertices, indices, 8)
    with pytest.raises(ValueError, match="mesh_vertices must have shape"):
        remeshing(vertices[:, :2], indices, 8)
    with pytest.raises(ValueError, match="mesh_indices must have shape"):
        remeshing(vertices, indices.reshape(-1), 8)
    with pytest.raises(TypeError, match="WarpRemeshOptions instance.*dict"):
        remeshing(vertices, indices, 8, warp_options={})


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
        2.0,
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
