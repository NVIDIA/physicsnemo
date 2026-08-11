# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mesh integration tests for nearest-surface shrinkwrap."""

import importlib

import pytest
import torch

from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.transformations.deform import shrinkwrap


def _target_plane(*, requires_grad: bool = False) -> Mesh:
    return Mesh(
        points=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            requires_grad=requires_grad,
        ),
        cells=torch.tensor([[0, 1, 2], [0, 2, 3]]),
        cell_data={"surface_id": torch.tensor([4, 4])},
    )


def _source_patch(*, requires_grad: bool = False) -> Mesh:
    return Mesh(
        points=torch.tensor(
            [
                [0.1, 0.1, 0.4],
                [0.9, 0.1, 0.5],
                [0.9, 0.9, 0.6],
                [0.1, 0.9, 0.7],
            ],
            requires_grad=requires_grad,
        ),
        cells=torch.tensor([[0, 1, 2], [0, 2, 3]]),
        point_data={
            "temperature": torch.tensor([10.0, 20.0, 30.0, 40.0]),
            "controls": {"weight": torch.tensor([0.0, 0.5, 1.0, 1.0])},
        },
        cell_data={"material": torch.tensor([7, 7])},
        global_data={"case_id": torch.tensor(12)},
    )


def test_shrinkwrap_namespace_and_bound_method_are_canonical():
    transformations = importlib.import_module("physicsnemo.mesh.transformations")
    deform_module = importlib.import_module("physicsnemo.mesh.transformations.deform")
    implementation_module = importlib.import_module(
        "physicsnemo.mesh.transformations.deform.shrinkwrap"
    )

    assert deform_module.shrinkwrap is shrinkwrap
    assert implementation_module.shrinkwrap is shrinkwrap
    assert shrinkwrap.__module__ == (
        "physicsnemo.mesh.transformations.deform.shrinkwrap"
    )
    assert "shrinkwrap" in deform_module.__all__
    assert not hasattr(transformations, "shrinkwrap")
    assert Mesh.shrinkwrap is shrinkwrap
    assert not hasattr(DomainMesh, "shrinkwrap")


@pytest.mark.parametrize(
    "index_dtype",
    [
        torch.int8,
        torch.int16,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    ],
)
def test_mesh_shrinkwrap_normalizes_integer_target_connectivity(
    index_dtype: torch.dtype,
):
    source = _source_patch()
    target_template = _target_plane()
    target = Mesh(
        points=target_template.points,
        cells=target_template.cells.to(index_dtype),
        cell_data=target_template.cell_data,
    )

    output = source.shrinkwrap(target, implementation="torch")

    expected = source.points.clone()
    expected[:, 2] = 0.0
    torch.testing.assert_close(output.points, expected)
    assert target.cells.dtype == index_dtype


@pytest.mark.parametrize("index_dtype", [torch.bool, torch.complex64])
def test_mesh_shrinkwrap_rejects_non_integer_target_connectivity(
    index_dtype: torch.dtype,
):
    target_template = _target_plane()
    target = Mesh(
        points=target_template.points,
        cells=target_template.cells.to(index_dtype),
    )

    with pytest.raises(TypeError, match="non-bool integer dtype"):
        _source_patch().shrinkwrap(target, implementation="torch")


def test_mesh_shrinkwrap_normalized_connectivity_keeps_range_validation():
    target_template = _target_plane()
    target = Mesh(
        points=target_template.points,
        cells=torch.tensor(
            [[0, 1, torch.iinfo(torch.uint64).max]],
            dtype=torch.uint64,
        ),
    )

    with pytest.raises(ValueError, match="indices outside the target point range"):
        _source_patch().shrinkwrap(target, implementation="torch")


def test_mesh_shrinkwrap_preserves_data_and_cache_contract():
    source = _source_patch()
    target = _target_plane()
    source_points = source.points.clone()
    target_points = target.points.clone()
    original_areas = source.cell_areas.clone()
    _ = source.cell_centroids
    topology = source.get_point_to_points_adjacency()

    output = source.shrinkwrap(target, implementation="torch")

    expected = source.points.clone()
    expected[:, 2] = 0.0
    torch.testing.assert_close(output.points, expected)
    torch.testing.assert_close(source.points, source_points)
    torch.testing.assert_close(target.points, target_points)
    assert output is not source
    assert torch.equal(output.cells, source.cells)
    assert torch.equal(
        output.point_data["temperature"],
        source.point_data["temperature"],
    )
    assert torch.equal(output.cell_data["material"], source.cell_data["material"])
    assert torch.equal(output.global_data["case_id"], source.global_data["case_id"])

    assert list(output._cache["cell"].keys()) == []
    assert list(output._cache["point"].keys()) == []
    assert output.get_point_to_points_adjacency().to_list() == topology.to_list()
    torch.testing.assert_close(source.cell_areas, original_areas)
    assert source._cache.get(("cell", "areas")) is not None
    assert source._cache.get(("cell", "centroids")) is not None


def test_mesh_shrinkwrap_resolves_nested_point_weight_key():
    source = _source_patch()
    target = _target_plane()
    weights = source.point_data["controls", "weight"]

    output = source.shrinkwrap(
        target,
        point_weights=("controls", "weight"),
        implementation="torch",
    )

    expected = source.points.clone()
    expected[:, 2] *= 1.0 - weights
    torch.testing.assert_close(output.points, expected)


def test_mesh_shrinkwrap_supports_a_tetrahedral_source():
    source = Mesh(
        points=torch.tensor(
            [
                [0.2, 0.2, 0.4],
                [0.8, 0.2, 0.4],
                [0.2, 0.8, 0.4],
                [0.2, 0.2, 1.0],
            ]
        ),
        cells=torch.tensor([[0, 1, 2, 3]]),
    )
    boundary_selection = torch.tensor([True, True, True, False])

    output = source.shrinkwrap(
        _target_plane(),
        point_weights=boundary_selection,
        implementation="torch",
    )

    expected = source.points.clone()
    expected[:3, 2] = 0.0
    torch.testing.assert_close(output.points, expected)
    assert torch.equal(output.cells, source.cells)
    assert output.n_spatial_dims == 3
    assert output.n_manifold_dims == 3


def test_mesh_shrinkwrap_max_distance_preserves_misses_exactly():
    source = _source_patch()
    target = _target_plane()

    output = source.shrinkwrap(
        target,
        max_distance=0.55,
        implementation="torch",
    )

    torch.testing.assert_close(output.points[:2, 2], torch.zeros(2))
    assert torch.equal(output.points[2:], source.points[2:])


def test_mesh_shrinkwrap_preserves_all_requested_gradients():
    source = _source_patch(requires_grad=True)
    target = _target_plane(requires_grad=True)
    weights = torch.tensor([0.2, 0.4, 0.6, 0.8], requires_grad=True)
    offset = torch.tensor(0.05, requires_grad=True)

    output = source.shrinkwrap(
        target,
        point_weights=weights,
        offset=offset,
        implementation="torch",
    )
    output.points.square().sum().backward()

    for tensor in (source.points, target.points, weights, offset):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    assert target.points.grad.abs().sum() > 0


def test_mesh_shrinkwrap_rejects_non_mesh_target():
    source = _source_patch()

    with pytest.raises(TypeError, match="target must be a Mesh"):
        source.shrinkwrap(torch.zeros(3, 3), implementation="torch")


def test_mesh_shrinkwrap_missing_weight_key_has_actionable_error():
    source = _source_patch()

    with pytest.raises(KeyError, match="point_weights field 'missing'.*Available keys"):
        source.shrinkwrap(
            _target_plane(),
            point_weights="missing",
            implementation="torch",
        )
