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

"""Cache-lifecycle tests for public cotangent DEC operators."""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh.calculus.divergence import compute_divergence_points_dec
from physicsnemo.mesh.calculus.laplacian import compute_laplacian_points_dec
from physicsnemo.mesh.geometry.dual_meshes import (
    get_or_compute_cotan_weights_fem,
    get_or_compute_dual_volumes_0,
)
from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.utilities._topology import get_or_compute_unique_edges


def _two_triangles(*, requires_grad: bool = False) -> Mesh:
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        requires_grad=requires_grad,
    )
    cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
    return Mesh(points=points, cells=cells)


def test_cotan_geometry_and_topology_are_reused() -> None:
    mesh = _two_triangles()

    weights_1, edges_1 = get_or_compute_cotan_weights_fem(mesh)
    volumes_1 = get_or_compute_dual_volumes_0(mesh)
    weights_2, edges_2 = get_or_compute_cotan_weights_fem(mesh)
    volumes_2 = get_or_compute_dual_volumes_0(mesh)

    assert weights_2 is weights_1
    assert edges_2 is edges_1
    assert volumes_2 is volumes_1
    assert mesh._cache["geometry", "cotan_weights_fem"] is weights_1
    assert mesh._cache["geometry", "dual_volumes_0"] is volumes_1


def test_public_dec_operators_share_preprocessing_cache() -> None:
    mesh = _two_triangles()
    compute_laplacian_points_dec(mesh, torch.arange(4.0))

    weights = mesh._cache["geometry", "cotan_weights_fem"]
    volumes = mesh._cache["geometry", "dual_volumes_0"]
    edges = mesh._cache["topology", "unique_edges"]
    compute_divergence_points_dec(mesh, mesh.points)

    assert mesh._cache["geometry", "cotan_weights_fem"] is weights
    assert mesh._cache["geometry", "dual_volumes_0"] is volumes
    assert mesh._cache["topology", "unique_edges"] is edges


def test_transform_invalidates_geometry_but_preserves_topology() -> None:
    mesh = _two_triangles()
    get_or_compute_cotan_weights_fem(mesh)
    get_or_compute_dual_volumes_0(mesh)
    edges = mesh._cache["topology", "unique_edges"]

    translated = mesh.translate([2.0, -3.0])

    assert len(translated._cache["geometry"]) == 0
    assert translated._cache["topology", "unique_edges"] is edges
    new_weights, new_edges = get_or_compute_cotan_weights_fem(translated)
    assert new_edges is edges
    torch.testing.assert_close(
        new_weights, mesh._cache["geometry", "cotan_weights_fem"]
    )


def test_with_data_preserves_cached_geometry_with_independent_container() -> None:
    mesh = _two_triangles()
    weights, _ = get_or_compute_cotan_weights_fem(mesh)
    volumes = get_or_compute_dual_volumes_0(mesh)

    updated = mesh.with_data(point_data={"value": torch.arange(4.0)})

    assert updated._cache is not mesh._cache
    assert updated._cache["geometry", "cotan_weights_fem"] is weights
    assert updated._cache["geometry", "dual_volumes_0"] is volumes


def test_to_converts_geometry_cache_without_casting_topology() -> None:
    mesh = _two_triangles()
    get_or_compute_cotan_weights_fem(mesh)
    get_or_compute_dual_volumes_0(mesh)

    converted = mesh.to(torch.float64)

    assert converted._cache["geometry", "cotan_weights_fem"].dtype == torch.float64
    assert converted._cache["geometry", "dual_volumes_0"].dtype == torch.float64
    assert converted._cache["topology", "unique_edges"].dtype == torch.int64


@pytest.mark.parametrize("operation", ["slice", "pad"])
def test_topology_changing_operations_clear_cotan_caches(operation: str) -> None:
    mesh = _two_triangles()
    get_or_compute_cotan_weights_fem(mesh)
    get_or_compute_dual_volumes_0(mesh)

    if operation == "slice":
        result = mesh.slice_cells(torch.tensor([0]))
    else:
        result = mesh.pad(target_n_points=5, target_n_cells=3)

    assert len(result._cache["topology"]) == 0
    assert len(result._cache["geometry"]) == 0


def test_differentiable_geometry_bypasses_geometry_cache() -> None:
    mesh = _two_triangles(requires_grad=True)

    weights_1, edges_1 = get_or_compute_cotan_weights_fem(mesh)
    volumes_1 = get_or_compute_dual_volumes_0(mesh)
    weights_2, edges_2 = get_or_compute_cotan_weights_fem(mesh)
    volumes_2 = get_or_compute_dual_volumes_0(mesh)

    assert edges_2 is edges_1
    assert weights_2 is not weights_1
    assert volumes_2 is not volumes_1
    assert len(mesh._cache["geometry"]) == 0
    assert mesh._cache.get(("cell", "areas"), None) is None


def test_differentiable_geometry_supports_repeated_backward() -> None:
    mesh = _two_triangles(requires_grad=True)

    for _ in range(2):
        values = torch.randn(mesh.n_points, requires_grad=True)
        result = compute_laplacian_points_dec(mesh, values)
        result.square().sum().backward()
        assert mesh.points.grad is not None
        assert torch.isfinite(mesh.points.grad).all()
        mesh.points.grad = None


def test_legacy_cache_schema_is_upgraded_lazily() -> None:
    points = _two_triangles().points
    old_cache = TensorDict(
        {
            "cell": TensorDict({}, batch_size=[2]),
            "point": TensorDict({}, batch_size=[4]),
        }
    )
    mesh = Mesh(
        points=points,
        cells=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        _cache=old_cache,
    )

    get_or_compute_cotan_weights_fem(mesh)
    get_or_compute_dual_volumes_0(mesh)

    assert "topology" in mesh._cache
    assert "geometry" in mesh._cache
    assert "unique_edges" in mesh._cache["topology"]
    assert "cotan_weights_fem" in mesh._cache["geometry"]


def test_zero_manifold_has_well_shaped_empty_edge_cache() -> None:
    mesh = Mesh(points=torch.randn(3, 2))

    edges, inverse = get_or_compute_unique_edges(mesh)
    weights, cached_edges = get_or_compute_cotan_weights_fem(mesh)

    assert edges.shape == (0, 2)
    assert inverse.shape == (0,)
    assert weights.shape == (0,)
    assert cached_edges is edges


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("operator", ["laplacian", "divergence"])
def test_inference_warmup_then_training_backward(
    implementation: str, operator: str
) -> None:
    mesh = _two_triangles()

    with torch.inference_mode():
        if operator == "laplacian":
            compute_laplacian_points_dec(
                mesh, torch.randn(mesh.n_points), implementation=implementation
            )
        else:
            compute_divergence_points_dec(
                mesh, torch.randn(mesh.n_points, 2), implementation=implementation
            )

    assert mesh._cache["topology", "unique_edges"].is_inference()
    assert mesh._cache["geometry", "cotan_weights_fem"].is_inference()
    assert mesh._cache["geometry", "dual_volumes_0"].is_inference()

    if operator == "laplacian":
        field = torch.randn(mesh.n_points, requires_grad=True)
        output = compute_laplacian_points_dec(
            mesh, field, implementation=implementation
        )
    else:
        field = torch.randn(mesh.n_points, 2, requires_grad=True)
        output = compute_divergence_points_dec(
            mesh, field, implementation=implementation
        )
    output.square().sum().backward()

    assert field.grad is not None
    assert torch.isfinite(field.grad).all()
    assert not mesh._cache["topology", "unique_edges"].is_inference()
    assert not mesh._cache["geometry", "cotan_weights_fem"].is_inference()
    assert not mesh._cache["geometry", "dual_volumes_0"].is_inference()
