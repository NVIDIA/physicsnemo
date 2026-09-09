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

"""Cache preservation for datapipe transforms that only replace mesh data."""

import pytest
import torch

from physicsnemo.datapipes.transforms.mesh import (
    ComputeSurfaceNormals,
    DropMeshFields,
    NormalizeMeshFields,
    RenameMeshFields,
    SetGlobalField,
)
from physicsnemo.mesh import Mesh, MeshFieldAssociation


def _mesh_with_populated_caches() -> Mesh:
    mesh = Mesh(
        points=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        cells=torch.tensor([[0, 1, 2], [0, 2, 3]]),
        point_data={
            "old_point": torch.arange(4),
            "drop_point": torch.ones(4),
        },
        cell_data={
            "old_cell": torch.arange(2),
            "drop_cell": torch.ones(2),
        },
        global_data={
            "old_global": torch.tensor(1.0),
            "drop_global": torch.tensor(2.0),
        },
    )
    _ = mesh.cell_centroids
    _ = mesh.cell_areas
    _ = mesh.cell_normals
    _ = mesh.point_normals
    mesh._cache["topology", "sentinel"] = torch.tensor(7)
    return mesh


def _assert_caches_preserved_and_independent(source: Mesh, result: Mesh) -> None:
    cache_keys = (
        ("cell", "centroids"),
        ("cell", "areas"),
        ("cell", "normals"),
        ("point", "normals"),
        ("topology", "sentinel"),
    )
    for key in cache_keys:
        source_value = source._cache.get(key, None)
        result_value = result._cache.get(key, None)
        assert source_value is not None
        assert result_value is not None
        torch.testing.assert_close(result_value, source_value)

    result._cache["cell", "derived"] = torch.zeros(result.n_cells)
    result._cache["topology", "derived"] = torch.tensor(8)
    assert source._cache.get(("cell", "derived"), None) is None
    assert source._cache.get(("topology", "derived"), None) is None


def test_drop_mesh_fields_preserves_caches() -> None:
    mesh = _mesh_with_populated_caches()

    result = DropMeshFields(
        point_data=["drop_point"],
        cell_data=["drop_cell"],
        global_data=["drop_global"],
    )(mesh)

    assert set(result.point_data.keys()) == {"old_point"}
    assert set(result.cell_data.keys()) == {"old_cell"}
    assert set(result.global_data.keys()) == {"old_global"}
    _assert_caches_preserved_and_independent(mesh, result)


def test_rename_mesh_fields_preserves_caches() -> None:
    mesh = _mesh_with_populated_caches()

    result = RenameMeshFields(
        point_data={"old_point": "new_point"},
        cell_data={"old_cell": "new_cell"},
        global_data={"old_global": "new_global"},
    )(mesh)

    assert "new_point" in result.point_data
    assert "old_point" not in result.point_data
    assert "new_cell" in result.cell_data
    assert "old_cell" not in result.cell_data
    assert "new_global" in result.global_data
    assert "old_global" not in result.global_data
    _assert_caches_preserved_and_independent(mesh, result)


def test_set_global_field_preserves_caches() -> None:
    mesh = _mesh_with_populated_caches()

    result = SetGlobalField({"new_global": [3.0, 4.0]})(mesh)

    torch.testing.assert_close(
        result.global_data["new_global"], torch.tensor([3.0, 4.0])
    )
    assert "new_global" not in mesh.global_data
    _assert_caches_preserved_and_independent(mesh, result)


@pytest.mark.parametrize(
    ("association", "field_name", "untouched_field", "expected"),
    [
        (
            "point_data",
            "old_point",
            "drop_point",
            torch.tensor([-0.5, 0.0, 0.5, 1.0]),
        ),
        ("cell_data", "old_cell", "drop_cell", torch.tensor([-0.5, 0.0])),
        ("global_data", "old_global", "drop_global", torch.tensor(0.0)),
    ],
)
def test_normalize_mesh_fields_preserves_caches(
    association: MeshFieldAssociation,
    field_name: str,
    untouched_field: str,
    expected: torch.Tensor,
) -> None:
    mesh = _mesh_with_populated_caches()
    source_value = getattr(mesh, association)[field_name].clone()
    source_untouched = getattr(mesh, association)[untouched_field]
    normalizer = NormalizeMeshFields(
        association=association,
        fields={
            field_name: {
                "type": "scalar",
                "mean": 1.0,
                "std": 2.0,
            }
        },
    )

    result = normalizer(mesh)

    torch.testing.assert_close(getattr(result, association)[field_name], expected)
    torch.testing.assert_close(getattr(mesh, association)[field_name], source_value)
    result_untouched = getattr(result, association)[untouched_field]
    torch.testing.assert_close(result_untouched, source_untouched)
    assert result_untouched.data_ptr() != source_untouched.data_ptr()
    _assert_caches_preserved_and_independent(mesh, result)


def test_compute_cell_normals_preserves_the_computed_cache() -> None:
    mesh = _mesh_with_populated_caches().strip_caches(
        keep=[("cell", "areas"), "topology"]
    )
    assert mesh._cache.get(("cell", "normals"), None) is None

    result = ComputeSurfaceNormals(store_as="cell_data", field_name="surface_normal")(
        mesh
    )

    cached_normals = result._cache.get(("cell", "normals"), None)
    assert cached_normals is not None
    torch.testing.assert_close(result.cell_data["surface_normal"], cached_normals)
    _assert_computed_cache_is_independent(mesh, result, "cell")


def test_compute_point_normals_preserves_the_computed_cache() -> None:
    mesh = _mesh_with_populated_caches().strip_caches(keep=["topology"])
    assert mesh._cache.get(("point", "normals"), None) is None

    result = ComputeSurfaceNormals(store_as="point_data", field_name="surface_normal")(
        mesh
    )

    cached_normals = result._cache.get(("point", "normals"), None)
    assert cached_normals is not None
    torch.testing.assert_close(result.point_data["surface_normal"], cached_normals)
    _assert_computed_cache_is_independent(mesh, result, "point")


def _assert_computed_cache_is_independent(
    source: Mesh, result: Mesh, association: str
) -> None:
    batch_size = source.n_cells if association == "cell" else source.n_points
    result._cache[association, "derived"] = torch.full((batch_size,), 9)
    assert source._cache.get((association, "derived"), None) is None
    result._cache["topology", "derived"] = torch.tensor(10)
    assert source._cache.get(("topology", "derived"), None) is None
