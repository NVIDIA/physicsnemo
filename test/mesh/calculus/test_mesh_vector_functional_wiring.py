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

import inspect

import pytest
import torch

from physicsnemo.mesh.calculus.curl import compute_curl_cells_lsq
from physicsnemo.mesh.calculus.divergence import (
    compute_divergence_cells_lsq,
    compute_divergence_points_lsq,
)
from physicsnemo.mesh.calculus.gradient import compute_gradient_points_lsq
from physicsnemo.mesh.calculus.laplacian import (
    compute_laplacian_cells_lsq,
    compute_laplacian_points_dec,
    compute_laplacian_points_lsq,
)
from physicsnemo.mesh.domain_mesh import DomainMesh
from physicsnemo.mesh.mesh import Mesh
from test.conftest import requires_module


def _tet_mesh(device: str):
    torch_device = torch.device(device)
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
        device=torch_device,
    )
    cells = torch.tensor(
        [
            [0, 1, 2, 4],
            [0, 1, 3, 4],
            [0, 2, 3, 4],
            [1, 2, 3, 4],
        ],
        dtype=torch.int64,
        device=torch_device,
    )
    return Mesh(points=points, cells=cells)


def _triangle_fan_mesh_2d(device: str):
    torch_device = torch.device(device)
    points = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.8660254],
            [-0.5, 0.8660254],
            [-1.0, 0.0],
            [-0.5, -0.8660254],
            [0.5, -0.8660254],
        ],
        dtype=torch.float32,
        device=torch_device,
    )
    cells = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            [0, 5, 6],
            [0, 6, 1],
        ],
        dtype=torch.int64,
        device=torch_device,
    )
    return Mesh(points=points, cells=cells)


@pytest.mark.parametrize(
    "wrapper",
    [
        Mesh.compute_point_derivatives,
        Mesh.compute_cell_derivatives,
        Mesh.gradient,
        Mesh.divergence,
        Mesh.curl,
        Mesh.laplacian,
        DomainMesh.compute_point_derivatives,
        DomainMesh.compute_cell_derivatives,
    ],
    ids=lambda wrapper: wrapper.__qualname__,
)
def test_mesh_calculus_wrapper_implementation_defaults_to_torch(wrapper):
    """All Mesh and DomainMesh functional wrappers pin the Torch backend."""
    implementation = inspect.signature(wrapper).parameters["implementation"]

    assert implementation.default == "torch"


@requires_module("warp")
def test_mesh_gradient_exposes_warp_implementation(device: str):
    mesh = _tet_mesh(device)
    values = mesh.points[:, 0] + 2.0 * mesh.points[:, 1] - mesh.points[:, 2]

    output = mesh.gradient(
        values,
        gradient_type="extrinsic",
        implementation="warp",
    )
    expected = compute_gradient_points_lsq(
        mesh,
        values,
        intrinsic=False,
        implementation="warp",
    )

    torch.testing.assert_close(output, expected)


@requires_module("warp")
def test_mesh_compute_point_derivatives_exposes_warp_implementation(device: str):
    mesh = _tet_mesh(device)
    mesh.point_data["f"] = mesh.points[:, 0] + mesh.points[:, 1]

    output = mesh.compute_point_derivatives(
        keys="f",
        gradient_type="extrinsic",
        implementation="warp",
    )
    expected = compute_gradient_points_lsq(
        mesh,
        mesh.point_data["f"],
        intrinsic=False,
        implementation="warp",
    )

    torch.testing.assert_close(output.point_data["f_gradient"], expected)


@pytest.mark.parametrize("data_source", ["points", "cells"])
@requires_module("warp")
def test_mesh_warp_gradient_uses_derivative_first_layout(device: str, data_source: str):
    mesh = _tet_mesh(device)
    coordinates = mesh.points if data_source == "points" else mesh.cell_centroids
    derivatives = coordinates.new_tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]
    )
    vector_field = coordinates @ derivatives

    gradient = mesh.gradient(
        vector_field,
        gradient_type="extrinsic",
        data_source=data_source,
        implementation="warp",
    )
    expected = derivatives.expand_as(gradient)
    torch.testing.assert_close(gradient, expected, atol=5.0e-3, rtol=5.0e-3)

    if data_source == "points":
        mesh.point_data["affine_vector"] = vector_field
        derived = mesh.compute_point_derivatives(
            keys="affine_vector",
            method="lsq",
            gradient_type="extrinsic",
            implementation="warp",
        )
        stored_gradient = derived.point_data["affine_vector_gradient"]
    else:
        mesh.cell_data["affine_vector"] = vector_field
        derived = mesh.compute_cell_derivatives(
            keys="affine_vector",
            method="lsq",
            gradient_type="extrinsic",
            implementation="warp",
        )
        stored_gradient = derived.cell_data["affine_vector_gradient"]
    torch.testing.assert_close(stored_gradient, expected, atol=5.0e-3, rtol=5.0e-3)


def test_intrinsic_point_gradient_rejects_warp_implementation(device: str):
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
        device=torch.device(device),
    )
    cells = torch.tensor(
        [[0, 1, 2], [1, 3, 2]], dtype=torch.int64, device=points.device
    )
    mesh = Mesh(points=points, cells=cells)
    values = points[:, 0]

    with pytest.raises(NotImplementedError, match="intrinsic tangent-space"):
        mesh.gradient(values, implementation="warp")


def test_mesh_divergence_uses_functional_wiring(device: str):
    mesh = _tet_mesh(device)
    vector_field = mesh.points.clone()

    output = mesh.divergence(vector_field, implementation="torch")
    expected = compute_divergence_points_lsq(
        mesh,
        vector_field,
        implementation="torch",
    )

    torch.testing.assert_close(output, expected)


@requires_module("warp")
def test_mesh_divergence_none_uses_automatic_dispatch(device: str):
    mesh = _tet_mesh(device)
    vector_field = mesh.points.clone()

    output = mesh.divergence(vector_field, implementation=None)
    expected = compute_divergence_points_lsq(
        mesh,
        vector_field,
        implementation="warp",
    )

    torch.testing.assert_close(output, expected)


def test_mesh_dec_gradient_rejects_warp_implementation(device: str):
    mesh = _triangle_fan_mesh_2d(device)
    values = mesh.points[:, 0]

    with pytest.raises(NotImplementedError, match="DEC gradients"):
        mesh.gradient(values, method="dec", implementation="warp")

    with pytest.raises(ValueError, match="Invalid implementation='invalid'"):
        mesh.gradient(values, method="dec", implementation="invalid")


@pytest.mark.parametrize("data_source", ["points", "cells"])
def test_mesh_curl_exposes_2d_scalar_curl(device: str, data_source: str):
    mesh = _triangle_fan_mesh_2d(device)
    locations = mesh.points if data_source == "points" else mesh.cell_centroids
    vector_field = torch.stack((-locations[:, 1], locations[:, 0]), dim=-1)

    output = mesh.curl(
        vector_field,
        data_source=data_source,
        implementation="torch",
    )

    assert output.shape == (locations.shape[0],)
    torch.testing.assert_close(output, torch.full_like(output, 2.0))


def test_mesh_laplacian_cotan_uses_functional_wiring(device: str):
    mesh = _tet_mesh(device)
    values = mesh.points.square().sum(dim=-1)

    output = mesh.laplacian(values, implementation="torch")
    expected = compute_laplacian_points_dec(mesh, values, implementation="torch")

    torch.testing.assert_close(output, expected)


def test_mesh_laplacian_preserves_positional_data_source(device: str):
    mesh = _tet_mesh(device)
    values = mesh.points.square().sum(dim=-1)

    output = mesh.laplacian(values, "points", implementation="torch")
    expected = compute_laplacian_points_dec(mesh, values, implementation="torch")

    torch.testing.assert_close(output, expected)


def test_mesh_laplacian_lsq_method_uses_functional_wiring(device: str):
    mesh = _tet_mesh(device)
    values = mesh.points.square().sum(dim=-1)

    output = mesh.laplacian(values, method="lsq", implementation="torch")
    expected = compute_laplacian_points_lsq(
        mesh,
        values,
        implementation="torch",
    )

    torch.testing.assert_close(output, expected)


def test_mesh_laplacian_distinguishes_invalid_arguments(device: str):
    mesh = _tet_mesh(device)
    values = mesh.points.square().sum(dim=-1)

    with pytest.raises(ValueError, match="Invalid method='invalid'"):
        mesh.laplacian(values, method="invalid")
    with pytest.raises(ValueError, match="Invalid data_source='invalid'"):
        mesh.laplacian(values, data_source="invalid")


def test_cell_lsq_operators_expose_min_neighbors(device: str):
    mesh = _tet_mesh(device)
    min_neighbors = mesh.n_cells
    vector_field = mesh.cell_centroids.clone()
    rotational_field = torch.stack(
        [
            -mesh.cell_centroids[:, 1],
            mesh.cell_centroids[:, 0],
            torch.zeros_like(mesh.cell_centroids[:, 2]),
        ],
        dim=-1,
    )
    values = mesh.cell_centroids.square().sum(dim=-1)

    divergence = compute_divergence_cells_lsq(
        mesh,
        vector_field,
        min_neighbors=min_neighbors,
        implementation="torch",
    )
    curl = compute_curl_cells_lsq(
        mesh,
        rotational_field,
        min_neighbors=min_neighbors,
        implementation="torch",
    )
    laplacian = compute_laplacian_cells_lsq(
        mesh,
        values,
        min_neighbors=min_neighbors,
        implementation="torch",
    )

    torch.testing.assert_close(divergence, torch.zeros_like(divergence))
    torch.testing.assert_close(curl, torch.zeros_like(curl))
    torch.testing.assert_close(laplacian, torch.zeros_like(laplacian))
