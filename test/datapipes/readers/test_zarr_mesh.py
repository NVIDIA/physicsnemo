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

"""Tests for ZarrMeshReader and save_mesh_to_zarr."""

import pytest
import torch

pytest.importorskip("zarr")

import physicsnemo.datapipes as dp
from physicsnemo.mesh import Mesh

BACKENDS = ["zarr"]
try:
    import tensorstore  # noqa: F401

    BACKENDS.append("tensorstore")
except ImportError:
    pass


def _make_mesh(n_cells: int = 100, seed: int = 0) -> Mesh:
    gen = torch.Generator().manual_seed(seed)
    n_points = 3 * n_cells
    return Mesh(
        points=torch.randn(n_points, 3, generator=gen),
        cells=torch.arange(n_points, dtype=torch.int64).reshape(n_cells, 3),
        cell_data={
            "pressure": torch.randn(n_cells, generator=gen),
            "wss": torch.randn(n_cells, 3, generator=gen),
        },
        point_data={"disp": torch.randn(n_points, 3, generator=gen)},
        # One 1-d and one 0-d entry: freestream metadata commonly arrives
        # as 0-d scalars (e.g. DrivAerML DomainMesh global_data).
        global_data={"U_inf": torch.tensor([38.9]), "L_ref": torch.tensor(2.786)},
    )


@pytest.fixture
def zarr_mesh_dir(tmp_path):
    """Directory with three saved mesh groups."""
    for i in range(3):
        dp.save_mesh_to_zarr(
            _make_mesh(seed=i), tmp_path / f"sample_{i:04d}.zarr",
            chunk_cells=32, chunk_points=96,
        )
    return tmp_path


@pytest.mark.parametrize("backend", BACKENDS)
def test_full_roundtrip(zarr_mesh_dir, backend):
    reader = dp.ZarrMeshReader(zarr_mesh_dir, backend=backend)
    assert len(reader) == 3
    for i in range(3):
        ref = _make_mesh(seed=i)
        mesh, metadata = reader[i]
        assert torch.equal(mesh.points, ref.points)
        assert torch.equal(mesh.cells, ref.cells)
        assert torch.equal(mesh.cell_data["pressure"], ref.cell_data["pressure"])
        assert torch.equal(mesh.cell_data["wss"], ref.cell_data["wss"])
        assert torch.equal(mesh.point_data["disp"], ref.point_data["disp"])
        assert torch.equal(mesh.global_data["U_inf"], ref.global_data["U_inf"])
        assert torch.equal(mesh.global_data["L_ref"], ref.global_data["L_ref"])
        assert mesh.global_data["L_ref"].ndim == 0
        assert metadata["index"] == i
        assert "sample_" in metadata["source_path"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_subsample_n_cells(zarr_mesh_dir, backend):
    reader = dp.ZarrMeshReader(
        zarr_mesh_dir, backend=backend, subsample_n_cells=10
    )
    gen = torch.Generator().manual_seed(0)
    reader.set_generator(gen)
    mesh, _ = reader[0]
    assert mesh.n_cells == 10
    assert mesh.cell_data["pressure"].shape[0] == 10
    assert mesh.cell_data["wss"].shape == (10, 3)
    # Remapped indices must be valid for the point window read.
    assert mesh.cells.min() >= 0
    assert mesh.cells.max() < mesh.n_points
    # point_data is sliced with the same point window.
    assert mesh.point_data["disp"].shape[0] == mesh.n_points
    # global_data always loads in full.
    assert torch.equal(mesh.global_data["U_inf"], torch.tensor([38.9]))


def test_subsample_geometry_matches_full_read(zarr_mesh_dir):
    """The subsampled block must be a verbatim slice of the full mesh."""
    full, _ = dp.ZarrMeshReader(zarr_mesh_dir)[0]
    reader = dp.ZarrMeshReader(zarr_mesh_dir, subsample_n_cells=10)
    gen = torch.Generator().manual_seed(3)
    reader.set_generator(gen)
    sub, _ = reader[0]
    # Locate the block via its cell_data slice, then compare geometry.
    p = full.cell_data["pressure"]
    start = (p == sub.cell_data["pressure"][0]).nonzero()[0].item()
    sl = slice(start, start + 10)
    assert torch.equal(sub.cell_data["pressure"], p[sl])
    sub_vertices = sub.points[sub.cells]
    full_vertices = full.points[full.cells[sl]]
    assert torch.equal(sub_vertices, full_vertices)


def test_subsample_deterministic_and_epoch_reseed(zarr_mesh_dir):
    def block(seed_epoch):
        reader = dp.ZarrMeshReader(zarr_mesh_dir, subsample_n_cells=10)
        gen = torch.Generator().manual_seed(0)
        reader.set_generator(gen)
        reader.set_epoch(seed_epoch)
        mesh, _ = reader[0]
        return mesh.cell_data["pressure"]

    assert torch.equal(block(1), block(1))
    assert not torch.equal(block(1), block(2))


def test_backends_agree(zarr_mesh_dir):
    if "tensorstore" not in BACKENDS:
        pytest.skip("tensorstore not installed")
    a, _ = dp.ZarrMeshReader(zarr_mesh_dir, backend="zarr")[1]
    b, _ = dp.ZarrMeshReader(zarr_mesh_dir, backend="tensorstore")[1]
    assert torch.equal(a.points, b.points)
    assert torch.equal(a.cells, b.cells)
    assert torch.equal(a.cell_data["pressure"], b.cell_data["pressure"])


def test_point_cloud_roundtrip_and_subsample(tmp_path):
    gen = torch.Generator().manual_seed(0)
    cloud = Mesh(
        points=torch.randn(200, 3, generator=gen),
        point_data={"v": torch.randn(200, 3, generator=gen)},
    )
    dp.save_mesh_to_zarr(cloud, tmp_path / "cloud.zarr", chunk_points=64)
    mesh, _ = dp.ZarrMeshReader(tmp_path)[0]
    assert mesh.n_cells == 0
    assert torch.equal(mesh.points, cloud.points)

    reader = dp.ZarrMeshReader(tmp_path, subsample_n_points=50)
    reader.set_generator(torch.Generator().manual_seed(0))
    sub, _ = reader[0]
    assert sub.n_points == 50
    assert sub.point_data["v"].shape == (50, 3)


def test_invalid_configurations(zarr_mesh_dir, tmp_path):
    with pytest.raises(NotImplementedError):
        dp.ZarrMeshReader(
            zarr_mesh_dir, subsample_n_cells=5, subsample_n_points=5
        )
    reader = dp.ZarrMeshReader(zarr_mesh_dir, subsample_n_points=5)
    with pytest.raises(NotImplementedError):
        reader[0]  # points-subsample on a mesh with cells
    with pytest.raises(FileNotFoundError):
        dp.ZarrMeshReader(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        dp.ZarrMeshReader(tmp_path / "empty")


def test_registry_resolution(zarr_mesh_dir):
    from physicsnemo.datapipes.registry import COMPONENT_REGISTRY

    assert COMPONENT_REGISTRY.get("ZarrMeshReader") is dp.ZarrMeshReader


def _make_domain_mesh(seed: int = 0):
    from physicsnemo.mesh import DomainMesh

    gen = torch.Generator().manual_seed(seed + 100)
    interior = Mesh(
        points=torch.randn(50, 3, generator=gen),
        point_data={"velocity": torch.randn(50, 3, generator=gen)},
    )
    return DomainMesh(
        interior=interior,
        boundaries={"vehicle": _make_mesh(n_cells=40, seed=seed)},
        global_data={
            "rho_inf": torch.tensor(1.205),
            "p_inf": torch.tensor(0.0),
        },
    )


@pytest.fixture
def domain_zarr_dir(tmp_path):
    """Directory with two DomainMesh-schema groups (souped boundaries)."""
    for i in range(2):
        dp.save_domain_mesh_to_zarr(
            _make_domain_mesh(seed=i), tmp_path / f"run_{i}.zarr",
            chunk_cells=16, chunk_points=48, soup_boundaries=True,
        )
    return tmp_path


def test_domain_mesh_schema_roundtrip(domain_zarr_dir):
    import zarr as zarr_mod

    root = zarr_mod.open_group(str(domain_zarr_dir / "run_0.zarr"), mode="r")
    assert root.attrs["format"] == "physicsnemo-domainmesh-zarr"
    assert root.attrs["boundary_names"] == ["vehicle"]
    assert root["boundaries/vehicle"].attrs["layout"] == "soup"
    assert root["interior"].attrs["n_cells"] == 0
    # Case-level metadata lives ONLY at the root (single-sourced): the
    # boundary keeps its own keys (U_inf) but case keys are not copied in.
    assert "rho_inf" in root["global_data"].array_keys()
    boundary_gd = root["boundaries/vehicle"]["global_data"]
    assert "U_inf" in boundary_gd.array_keys()
    assert "rho_inf" not in boundary_gd.array_keys()


def test_subpath_and_merge_global_data(domain_zarr_dir):
    ref = _make_domain_mesh(seed=0)
    reader = dp.ZarrMeshReader(
        domain_zarr_dir,
        subpath="boundaries/vehicle",
        merge_global_data_from="../../global_data",
    )
    assert len(reader) == 2
    mesh, metadata = reader[0]
    soup_ref = ref.boundaries["vehicle"]
    # Souped boundary: cell_data and per-cell vertex geometry must match.
    assert torch.equal(mesh.cell_data["pressure"], soup_ref.cell_data["pressure"])
    assert torch.equal(
        mesh.points[mesh.cells], soup_ref.points[soup_ref.cells]
    )
    # Boundary's own global_data merged with case-level metadata.
    assert torch.equal(mesh.global_data["U_inf"], torch.tensor([38.9]))
    assert torch.equal(mesh.global_data["rho_inf"], torch.tensor(1.205))


def test_merge_collision_raises(tmp_path):
    from physicsnemo.mesh import DomainMesh

    dm = _make_domain_mesh(seed=0)
    # Case-level key colliding with the boundary's own U_inf.
    dm = DomainMesh(
        interior=dm.interior,
        boundaries=dm.boundaries,
        global_data={"U_inf": torch.tensor(1.0)},
    )
    dp.save_domain_mesh_to_zarr(dm, tmp_path / "run_0.zarr")
    reader = dp.ZarrMeshReader(
        tmp_path, subpath="boundaries/vehicle",
        merge_global_data_from="../../global_data",
    )
    with pytest.raises(ValueError, match="collision"):
        reader[0]


def test_soup_layout_skips_cells_and_matches_indexed(tmp_path):
    """Soup fast path must produce the same sample as the indexed path."""
    mesh = _make_mesh(n_cells=100, seed=3)  # already soup by construction
    dp.save_mesh_to_zarr(mesh, tmp_path / "s.zarr", chunk_cells=32, chunk_points=96)
    import zarr as zarr_mod

    g = zarr_mod.open_group(str(tmp_path / "s.zarr"), mode="r")
    assert g.attrs["layout"] == "soup"

    def read(subsample):
        reader = dp.ZarrMeshReader(tmp_path, subsample_n_cells=subsample)
        reader.set_generator(torch.Generator().manual_seed(7))
        return reader[0]

    sub, _ = read(10)
    full, _ = read(10**9)
    assert torch.equal(full.cells, mesh.cells)  # synthesized == stored
    p = full.cell_data["pressure"]
    start = (p == sub.cell_data["pressure"][0]).nonzero()[0].item()
    assert torch.equal(
        sub.points[sub.cells], full.points[full.cells[start : start + 10]]
    )


def test_to_cell_soup_and_layout_detection():
    mesh = _make_mesh(n_cells=20, seed=0)
    shared = Mesh(  # indexed mesh with genuinely shared vertices
        points=mesh.points[:33],
        cells=torch.randint(0, 33, (20, 3), generator=torch.Generator().manual_seed(1)),
        cell_data={"f": torch.randn(20)},
    )
    soup = dp.readers.zarr_mesh.to_cell_soup(shared)
    assert soup.n_points == 60
    assert torch.equal(soup.points[soup.cells], shared.points[shared.cells])
    assert dp.readers.zarr_mesh._is_soup(soup)
    assert not dp.readers.zarr_mesh._is_soup(shared)


def test_zarr_domain_mesh_reader(domain_zarr_dir):
    reader = dp.ZarrDomainMeshReader(domain_zarr_dir)
    assert len(reader) == 2
    domain, metadata = reader[0]
    ref = _make_domain_mesh(seed=0)
    assert metadata["boundary_names"] == ["vehicle"]
    assert torch.equal(domain.interior.points, ref.interior.points)
    assert torch.equal(
        domain.interior.point_data["velocity"], ref.interior.point_data["velocity"]
    )
    veh, ref_veh = domain.boundaries["vehicle"], ref.boundaries["vehicle"]
    assert torch.equal(veh.points[veh.cells], ref_veh.points[ref_veh.cells])
    assert torch.equal(domain.global_data["rho_inf"], torch.tensor(1.205))


def test_zarr_domain_mesh_reader_subsample_and_full_res(domain_zarr_dir):
    reader = dp.ZarrDomainMeshReader(
        domain_zarr_dir, subsample_n_points=10, subsample_n_cells=8
    )
    reader.set_generator(torch.Generator().manual_seed(0))
    domain, _ = reader[0]
    assert domain.interior.n_points == 10  # point cloud -> n_points
    assert domain.boundaries["vehicle"].n_cells == 8  # has cells -> n_cells

    full = dp.ZarrDomainMeshReader(
        domain_zarr_dir, subsample_n_points=10, subsample_n_cells=8,
        full_resolution_boundaries=["vehicle"],
    )
    domain_full, _ = full[0]
    assert domain_full.boundaries["vehicle"].n_cells == 40  # untouched

    with pytest.raises(ValueError, match="full_resolution_boundaries"):
        dp.ZarrDomainMeshReader(
            domain_zarr_dir, full_resolution_boundaries=["nope"]
        )


def test_zarr_domain_mesh_reader_rejects_mesh_groups(zarr_mesh_dir):
    with pytest.raises(ValueError, match="format"):
        dp.ZarrDomainMeshReader(zarr_mesh_dir)


def test_writer_accepts_store_object(tmp_path):
    import zarr as zarr_mod

    store = zarr_mod.storage.LocalStore(str(tmp_path / "s.zarr"))
    dp.save_mesh_to_zarr(_make_mesh(n_cells=10), store)
    mesh, _ = dp.ZarrMeshReader(tmp_path)[0]
    assert mesh.n_cells == 10


def test_schema_version_rejection(tmp_path):
    import zarr as zarr_mod

    dp.save_mesh_to_zarr(_make_mesh(n_cells=10), tmp_path / "s.zarr")
    g = zarr_mod.open_group(str(tmp_path / "s.zarr"), mode="r+")
    g.attrs["schema_version"] = 999
    reader = dp.ZarrMeshReader(tmp_path)
    with pytest.raises(ValueError, match="schema_version=999"):
        reader[0]
    with pytest.raises(ValueError, match="schema_version=999"):
        dp.validate_mesh_zarr(tmp_path / "s.zarr")


def test_validate_mesh_zarr(tmp_path, domain_zarr_dir):
    dp.save_mesh_to_zarr(_make_mesh(n_cells=25), tmp_path / "ok.zarr")
    summary = dp.validate_mesh_zarr(tmp_path / "ok.zarr")
    assert summary["layout"] == "soup" and summary["n_cells"] == 25

    domain_summary = dp.validate_mesh_zarr(domain_zarr_dir / "run_0.zarr")
    assert domain_summary["format"] == "physicsnemo-domainmesh-zarr"
    assert domain_summary["boundaries"]["vehicle"]["layout"] == "soup"

    # Corrupt an attr and a soup claim; both must be caught.
    import zarr as zarr_mod

    g = zarr_mod.open_group(str(tmp_path / "ok.zarr"), mode="r+")
    g.attrs["n_cells"] = 24
    with pytest.raises(ValueError, match="n_cells"):
        dp.validate_mesh_zarr(tmp_path / "ok.zarr")
    g.attrs["n_cells"] = 25
    g["cells"][0] = [2, 1, 0]  # break the arange while layout says soup
    with pytest.raises(ValueError, match="not arange"):
        dp.validate_mesh_zarr(tmp_path / "ok.zarr")


def test_mesh_dataset_pipeline(zarr_mesh_dir):
    """ZarrMeshReader drops into MeshDataset + mesh transforms."""
    from physicsnemo.datapipes.transforms.mesh import MeshToDomainMesh

    reader = dp.ZarrMeshReader(zarr_mesh_dir, subsample_n_cells=10)
    dataset = dp.MeshDataset(
        reader,
        transforms=[
            dp.CenterMesh(use_area_weighting=False),
            MeshToDomainMesh(
                cell_data_targets=["pressure"],
                interior_points="cell_centroids",
                boundary_name="vehicle",
            ),
        ],
    )
    domain, _ = dataset[0]
    assert domain.interior.n_points == 10
    assert "pressure" in domain.interior.point_data.keys()
    assert "pressure" not in domain.boundaries["vehicle"].cell_data.keys()
    assert "wss" in domain.boundaries["vehicle"].cell_data.keys()


def _make_nested_mesh(n_cells: int = 60, seed: int = 0) -> Mesh:
    """Mesh with nested TensorDicts in every data container."""
    gen = torch.Generator().manual_seed(seed)
    n_points = 3 * n_cells
    return Mesh(
        points=torch.randn(n_points, 3, generator=gen),
        cells=torch.arange(n_points, dtype=torch.int64).reshape(n_cells, 3),
        point_data={
            "disp": torch.randn(n_points, 3, generator=gen),
            "turbulence": {
                "k": torch.randn(n_points, generator=gen),
                "wall": {"omega": torch.randn(n_points, generator=gen)},
            },
        },
        cell_data={"stress": {"max": torch.randn(n_cells, generator=gen)}},
        global_data={
            "freestream": {"U": torch.tensor(38.9), "p": torch.tensor(0.0)},
        },
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_nested_tensordict_roundtrip(tmp_path, backend):
    """Nested TensorDicts map to nested zarr groups and back exactly."""
    ref = _make_nested_mesh()
    dp.save_mesh_to_zarr(ref, tmp_path / "n.zarr", chunk_cells=16, chunk_points=48)
    dp.validate_mesh_zarr(tmp_path / "n.zarr")
    mesh, _ = dp.ZarrMeshReader(tmp_path, backend=backend)[0]
    assert torch.equal(
        mesh.point_data["turbulence", "wall", "omega"],
        ref.point_data["turbulence", "wall", "omega"],
    )
    assert torch.equal(
        mesh.cell_data["stress", "max"], ref.cell_data["stress", "max"]
    )
    assert torch.equal(
        mesh.global_data["freestream", "U"], ref.global_data["freestream", "U"]
    )


def test_nested_subsample_slices_all_leaves(tmp_path):
    """One block subsample must slice every nested leaf consistently."""
    dp.save_mesh_to_zarr(
        _make_nested_mesh(), tmp_path / "n.zarr", chunk_cells=16, chunk_points=48
    )
    full, _ = dp.ZarrMeshReader(tmp_path)[0]
    reader = dp.ZarrMeshReader(tmp_path, subsample_n_cells=10)
    reader.set_generator(torch.Generator().manual_seed(0))
    sub, _ = reader[0]
    assert sub.n_cells == 10
    assert sub.point_data["turbulence", "wall", "omega"].shape[0] == sub.n_points
    p = full.cell_data["stress", "max"]
    start = (p == sub.cell_data["stress", "max"][0]).nonzero()[0].item()
    assert torch.equal(sub.cell_data["stress", "max"], p[start : start + 10])


def test_nested_case_global_data_merge(tmp_path):
    """Nested case-level global_data survives both read-time merge paths."""
    from physicsnemo.mesh import DomainMesh

    dm = _make_domain_mesh(seed=0)
    dm = DomainMesh(
        interior=dm.interior,
        boundaries=dm.boundaries,
        global_data={
            "freestream": {"U": torch.tensor(38.9)},
            "rho_inf": torch.tensor(1.205),
        },
    )
    dp.save_domain_mesh_to_zarr(dm, tmp_path / "run_0.zarr")
    mesh, _ = dp.ZarrMeshReader(
        tmp_path,
        subpath="boundaries/vehicle",
        merge_global_data_from="../../global_data",
    )[0]
    assert torch.equal(mesh.global_data["freestream", "U"], torch.tensor(38.9))
    domain, _ = dp.ZarrDomainMeshReader(tmp_path)[0]
    assert torch.equal(domain.global_data["freestream", "U"], torch.tensor(38.9))


def test_slash_field_name_rejected(tmp_path):
    """'/' is the zarr path separator; writing it would silently nest."""
    mesh = Mesh(
        points=torch.randn(30, 3),
        cells=torch.arange(30, dtype=torch.int64).reshape(10, 3),
        cell_data={"stress/max": torch.randn(10)},
    )
    with pytest.raises(ValueError, match="must not contain"):
        dp.save_mesh_to_zarr(mesh, tmp_path / "bad.zarr")


def test_non_tensor_field_rejected(tmp_path):
    """NonTensorData is not representable in schema v1: clear error."""
    from tensordict import NonTensorData, TensorDict

    mesh = Mesh(
        points=torch.randn(30, 3),
        cells=torch.arange(30, dtype=torch.int64).reshape(10, 3),
        global_data=TensorDict(
            {"case_name": NonTensorData("run_042")}, batch_size=[]
        ),
    )
    with pytest.raises(TypeError, match="NonTensorData"):
        dp.save_mesh_to_zarr(mesh, tmp_path / "bad.zarr")


def test_to_cell_soup_nested_point_data():
    gen = torch.Generator().manual_seed(1)
    shared = Mesh(
        points=torch.randn(50, 3, generator=gen),
        cells=torch.randint(0, 50, (30, 3), generator=gen),
        point_data={"turbulence": {"k": torch.randn(50, generator=gen)}},
    )
    soup = dp.readers.zarr_mesh.to_cell_soup(shared)
    assert soup.n_points == 90
    assert torch.equal(
        soup.point_data["turbulence", "k"].reshape(30, 3),
        shared.point_data["turbulence", "k"][shared.cells],
    )


def _make_indexed_mesh(seed: int = 0) -> Mesh:
    """Shared-vertex mesh whose cells are NOT arange (layout=indexed)."""
    gen = torch.Generator().manual_seed(seed)
    n_points, n_cells = 200, 150
    return Mesh(
        points=torch.randn(n_points, 3, generator=gen),
        cells=torch.randint(0, n_points, (n_cells, 3), generator=gen),
        cell_data={"pressure": torch.randn(n_cells, generator=gen)},
        point_data={"disp": torch.randn(n_points, 3, generator=gen)},
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_indexed_subsample_matches_full_read(tmp_path, backend):
    """The dependent-read path (indexed layout) must slice verbatim."""
    import zarr as zarr_mod

    dp.save_mesh_to_zarr(
        _make_indexed_mesh(), tmp_path / "i.zarr", chunk_cells=32, chunk_points=64
    )
    g = zarr_mod.open_group(str(tmp_path / "i.zarr"), mode="r")
    assert g.attrs["layout"] == "indexed"

    full, _ = dp.ZarrMeshReader(tmp_path, backend=backend)[0]
    reader = dp.ZarrMeshReader(tmp_path, backend=backend, subsample_n_cells=20)
    reader.set_generator(torch.Generator().manual_seed(5))
    sub, _ = reader[0]
    p = full.cell_data["pressure"]
    start = (p == sub.cell_data["pressure"][0]).nonzero()[0].item()
    sl = slice(start, start + 20)
    assert torch.equal(sub.cell_data["pressure"], p[sl])
    assert torch.equal(sub.points[sub.cells], full.points[full.cells[sl]])
    # point_data is the [cells.min(), cells.max()] window of the block.
    off = full.cells[sl].min()
    assert torch.equal(
        sub.point_data["disp"], full.point_data["disp"][off : off + sub.n_points]
    )


def test_set_epoch_resume_reproducible(zarr_mesh_dir):
    """set_epoch(n) after resume == reaching epoch n sequentially."""

    def block(epochs):
        reader = dp.ZarrMeshReader(zarr_mesh_dir, subsample_n_cells=10)
        reader.set_generator(torch.Generator().manual_seed(0))
        for e in epochs:
            reader.set_epoch(e)
        mesh, _ = reader[0]
        return mesh.cell_data["pressure"]

    assert torch.equal(block([1, 2, 3]), block([3]))


def test_mesh_reader_rejects_domainmesh_group(domain_zarr_dir):
    """Forgetting subpath= on a DomainMesh dataset gives a clear error."""
    reader = dp.ZarrMeshReader(domain_zarr_dir)
    with pytest.raises(ValueError, match="subpath"):
        reader[0]


def test_heterogeneous_boundary_names_raise(tmp_path):
    """Extra boundaries in a later case must not be silently dropped."""
    from physicsnemo.mesh import DomainMesh

    dp.save_domain_mesh_to_zarr(_make_domain_mesh(seed=0), tmp_path / "run_0.zarr")
    dm1 = _make_domain_mesh(seed=1)
    dm1 = DomainMesh(
        interior=dm1.interior,
        boundaries={**dm1.boundaries, "extra": _make_mesh(n_cells=10, seed=2)},
        global_data=dm1.global_data,
    )
    dp.save_domain_mesh_to_zarr(dm1, tmp_path / "run_1.zarr")
    reader = dp.ZarrDomainMeshReader(tmp_path)
    reader[0]  # first case defines the structure and reads fine
    with pytest.raises(ValueError, match="boundary_names"):
        reader[1]


def test_zarr_v2_store_rejected(tmp_path):
    """Schema requires zarr format 3; v2 stores fail with a clear error."""
    import zarr as zarr_mod

    g = zarr_mod.open_group(str(tmp_path / "v2.zarr"), mode="w", zarr_format=2)
    g.create_array("points", shape=(4, 3), dtype="float32")
    g.attrs["format"] = "physicsnemo-mesh-zarr"
    reader = dp.ZarrMeshReader(tmp_path)
    with pytest.raises(ValueError, match="format 2"):
        reader[0]


def test_validate_nested_leaf_dims(tmp_path):
    """The validator checks leading dims of nested leaves, any depth."""
    import zarr as zarr_mod

    dp.save_mesh_to_zarr(_make_nested_mesh(), tmp_path / "n.zarr")
    g = zarr_mod.open_group(str(tmp_path / "n.zarr"), mode="r+")
    g["point_data/turbulence"].create_array("bad", shape=(7,), dtype="float32")
    with pytest.raises(ValueError, match="turbulence/bad"):
        dp.validate_mesh_zarr(tmp_path / "n.zarr")
