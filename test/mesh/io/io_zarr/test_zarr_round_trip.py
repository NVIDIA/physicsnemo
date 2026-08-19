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

"""Round-trip tests for physicsnemo.mesh.io.to_zarr / from_zarr."""

import pytest
import torch
from conftest import (
    assert_meshes_equal,
    make_domain_mesh,
    make_mesh,
    make_point_cloud,
)

from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.io import from_zarr, to_zarr


def test_mesh_round_trip(tmp_path):
    mesh = make_mesh()
    to_zarr(mesh, tmp_path / "mesh.zarr")
    back = from_zarr(tmp_path / "mesh.zarr")
    assert isinstance(back, Mesh)
    assert_meshes_equal(mesh, back)


def test_nested_tensordict_fields_round_trip(tmp_path):
    """Nested TensorDicts in field trees must survive the round trip."""
    mesh = make_mesh()
    to_zarr(mesh, tmp_path / "mesh.zarr")
    back = from_zarr(tmp_path / "mesh.zarr")
    assert torch.equal(
        back.point_data["flow", "velocity"], mesh.point_data["flow", "velocity"]
    )
    assert torch.equal(
        back.point_data["flow", "moments", "second"],
        mesh.point_data["flow", "moments", "second"],
    )


def test_point_cloud_round_trip(tmp_path):
    """A mesh with no cells comes back as a clean point cloud."""
    pc = make_point_cloud()
    to_zarr(pc, tmp_path / "pc.zarr")
    back = from_zarr(tmp_path / "pc.zarr")
    assert back.n_cells == 0
    assert_meshes_equal(pc, back)


def test_domain_mesh_round_trip(tmp_path):
    dm = make_domain_mesh()
    to_zarr(dm, tmp_path / "dm.zarr")
    back = from_zarr(tmp_path / "dm.zarr")
    assert isinstance(back, DomainMesh)
    assert set(back.boundary_names) == set(dm.boundary_names)
    assert_meshes_equal(dm.interior, back.interior)
    for name in dm.boundary_names:
        assert_meshes_equal(dm.boundaries[name], back.boundaries[name])
    assert torch.equal(back.global_data["rho_inf"], dm.global_data["rho_inf"])


def test_boundary_subgroup_loads_as_mesh(tmp_path):
    """A mesh subgroup inside a DomainMesh store is directly loadable."""
    dm = make_domain_mesh()
    to_zarr(dm, tmp_path / "dm.zarr")
    wall = from_zarr(tmp_path / "dm.zarr" / "boundaries" / "wall")
    assert isinstance(wall, Mesh)
    assert_meshes_equal(dm.boundaries["wall"], wall)


def test_chunking_and_compression_applied(tmp_path):
    """Layout policy must reach nested leaves (tensordict#1758 workaround)."""
    import zarr

    to_zarr(make_domain_mesh(), tmp_path / "dm.zarr", chunk_rows=16)
    g = zarr.open_group(str(tmp_path / "dm.zarr"), mode="r")
    arr = g["boundaries/wall/points"]  # nested leaf
    assert arr.chunks[0] == 16
    assert any(type(c).__name__ == "ZstdCodec" for c in arr.metadata.codecs)


def test_dtypes_preserved(tmp_path):
    mesh = make_mesh()
    to_zarr(mesh, tmp_path / "mesh.zarr")
    back = from_zarr(tmp_path / "mesh.zarr")
    assert back.points.dtype == mesh.points.dtype
    assert back.cells.dtype == mesh.cells.dtype


def test_non_mesh_store_raises(tmp_path):
    import zarr

    g = zarr.open_group(str(tmp_path / "not_a_mesh.zarr"), mode="w")
    g.create_array("x", shape=(3,), dtype="f4")
    with pytest.raises(ValueError, match="not written by"):
        from_zarr(tmp_path / "not_a_mesh.zarr")


def test_wrong_type_raises(tmp_path):
    with pytest.raises(TypeError, match="Expected Mesh or DomainMesh"):
        to_zarr(torch.zeros(3), tmp_path / "x.zarr")
