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

import re

import pytest
import torch
from conftest import (
    assert_meshes_equal,
    make_domain_mesh,
    make_mesh,
    make_point_cloud,
    write_unfinished_store,
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
    """A mesh subgroup inside a DomainMesh store is directly loadable, also
    through a symlink."""
    dm = make_domain_mesh()
    to_zarr(dm, tmp_path / "dm.zarr")
    wall = from_zarr(tmp_path / "dm.zarr" / "boundaries" / "wall")
    assert isinstance(wall, Mesh)
    assert_meshes_equal(dm.boundaries["wall"], wall)
    (tmp_path / "wall.zarr").symlink_to(tmp_path / "dm.zarr" / "boundaries" / "wall")
    assert_meshes_equal(dm.boundaries["wall"], from_zarr(tmp_path / "wall.zarr"))


def test_chunking_and_compression_applied(tmp_path):
    """Layout policy must reach nested leaves."""
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


def test_0d_uint64_leaf_round_trip(tmp_path):
    """A 0-d uint64 leaf reads back as a 0-d uint64 tensor."""
    mesh = make_mesh()
    mesh.global_data["seed"] = torch.tensor(2**40, dtype=torch.uint64)
    to_zarr(mesh, tmp_path / "mesh.zarr")
    seed = from_zarr(tmp_path / "mesh.zarr").global_data["seed"]
    assert seed.dtype == torch.uint64
    assert seed.shape == ()
    assert seed.item() == 2**40


def test_unsupported_dtype_raises_before_writing(tmp_path):
    """A field NumPy cannot hold (bfloat16) raises and writes no partial store."""
    mesh = make_mesh()
    mesh.point_data["pred"] = torch.zeros(mesh.n_points, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match=r"'point_data/pred'.*bfloat16"):
        to_zarr(mesh, tmp_path / "mesh.zarr")
    assert not (tmp_path / "mesh.zarr").exists()


class _PlainMesh(Mesh):
    """A Mesh subclass without fields of its own."""


class _PlainDomainMesh(DomainMesh):
    """A DomainMesh subclass without fields of its own."""


class _TaggedMesh(Mesh):
    """A Mesh subclass with a field that from_zarr cannot restore."""

    tag: torch.Tensor | None = None


def test_subclass_round_trips_as_base_class(tmp_path):
    """Subclasses without fields of their own load back as their base class."""
    mesh = make_mesh()
    to_zarr(_PlainMesh(points=mesh.points, cells=mesh.cells), tmp_path / "m.zarr")
    back = from_zarr(tmp_path / "m.zarr")
    assert type(back) is Mesh
    assert torch.equal(back.points, mesh.points)
    assert torch.equal(back.cells, mesh.cells)

    dm = make_domain_mesh()
    to_zarr(
        _PlainDomainMesh(
            interior=dm.interior,
            boundaries=dict(dm.boundaries.items()),
            global_data=dm.global_data,
        ),
        tmp_path / "dm.zarr",
    )
    assert type(from_zarr(tmp_path / "dm.zarr")) is DomainMesh
    wall = from_zarr(tmp_path / "dm.zarr" / "boundaries" / "wall")
    assert_meshes_equal(dm.boundaries["wall"], wall)


@pytest.mark.parametrize("in_domain", [False, True])
def test_subclass_with_fields_raises_before_writing(tmp_path, in_domain):
    """A subclass field that from_zarr could not restore raises, writing nothing."""
    mesh = make_mesh()
    obj = _TaggedMesh(points=mesh.points, cells=mesh.cells, tag=torch.ones(2))
    if in_domain:
        obj = DomainMesh(interior=make_point_cloud(), boundaries={"wall": obj})
    with pytest.raises(TypeError, match=r"_TaggedMesh.*'tag'"):
        to_zarr(obj, tmp_path / "x.zarr")
    assert not (tmp_path / "x.zarr").exists()


@pytest.mark.parametrize("n_vertices_per_cell", [2, 3, 4])
def test_mesh_without_cells_keeps_cell_shape(tmp_path, n_vertices_per_cell):
    """A cell mesh with zero cells keeps its cell shape, dtype and dimension."""
    mesh = Mesh(
        points=torch.rand(5, 3),
        cells=torch.empty((0, n_vertices_per_cell), dtype=torch.int32),
    )
    dm = DomainMesh(interior=make_point_cloud(), boundaries={"empty": mesh})
    to_zarr(mesh, tmp_path / "mesh.zarr")
    to_zarr(dm, tmp_path / "dm.zarr")
    for back in (
        from_zarr(tmp_path / "mesh.zarr"),
        from_zarr(tmp_path / "dm.zarr").boundaries["empty"],
    ):
        assert back.cells.shape == (0, n_vertices_per_cell)
        assert back.cells.dtype == torch.int32
        assert back.n_manifold_dims == n_vertices_per_cell - 1


def _with_point_data_key(mesh: Mesh, name: str) -> Mesh:
    mesh.point_data[name] = mesh.points
    return mesh


def _with_group_key(mesh: Mesh, name: str) -> Mesh:
    mesh.global_data[name] = {"x": torch.ones(1)}
    return mesh


def _with_boundary_name(mesh: Mesh, name: str) -> DomainMesh:
    return DomainMesh(interior=mesh, boundaries={name: mesh})


@pytest.mark.parametrize("name", ["U [m/s]", "U [m\\s]", "", ".", "..", "zarr.json"])
@pytest.mark.parametrize(
    "make,parent",
    [
        (_with_point_data_key, "point_data"),
        (_with_group_key, "global_data"),
        (_with_boundary_name, "boundaries"),
    ],
)
def test_invalid_name_raises_before_writing(tmp_path, make, parent, name):
    """A key or boundary name that zarr cannot store raises, writing nothing."""
    with pytest.raises(ValueError, match=re.escape(f"key {(parent, name)}")):
        to_zarr(make(make_mesh(), name), tmp_path / "x.zarr")
    assert not (tmp_path / "x.zarr").exists()


def test_dtype_check_ignores_default_device(tmp_path):
    """A non-CPU default device does not make supported dtypes look unsupported."""
    mesh = make_mesh()
    with torch.device("meta"):
        to_zarr(mesh, tmp_path / "mesh.zarr")
    assert_meshes_equal(mesh, from_zarr(tmp_path / "mesh.zarr"))


@pytest.mark.parametrize(
    "make,groups", [(make_mesh, [""]), (make_domain_mesh, ["", "interior"])]
)
def test_unfinished_store_raises(tmp_path, make, groups):
    """A store whose write failed partway raises on load. So does a mesh group
    inside it, even one that was written in full."""
    store = tmp_path / "store.zarr"
    write_unfinished_store(make(), store)
    for group in groups:
        with pytest.raises(ValueError, match="did not finish"):
            from_zarr(store / group)


def test_non_mesh_store_raises(tmp_path):
    import zarr

    g = zarr.open_group(str(tmp_path / "not_a_mesh.zarr"), mode="w")
    g.create_array("x", shape=(3,), dtype="f4")
    with pytest.raises(ValueError, match="not written by"):
        from_zarr(tmp_path / "not_a_mesh.zarr")


def test_wrong_type_raises(tmp_path):
    with pytest.raises(TypeError, match="Expected Mesh or DomainMesh"):
        to_zarr(torch.zeros(3), tmp_path / "x.zarr")
