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

"""Tests for MeshReader, DomainMeshReader, and DomainMesh transform integration."""

import pytest
import torch

from physicsnemo.datapipes.mesh_dataset import MeshDataset
from physicsnemo.datapipes.readers.mesh import (
    DomainMeshReader,
    MeshReader,
    _subsample_mesh_points,
)
from physicsnemo.datapipes.transforms.mesh import (
    CenterMesh,
    RandomScaleMesh,
    ScaleMesh,
    SubsampleMesh,
)
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import (
    EFFECTIVE_MEASURE_KEY,
    cell_measures,
)
from physicsnemo.mesh.primitives.basic import (
    single_triangle_3d,
    two_triangles_2d,
)


class TestMeshReader:
    """Tests for MeshReader (single-mesh)."""

    def test_len_and_getitem(self, tmp_path):
        mesh = two_triangles_2d.load()
        mesh.save(tmp_path / "a.pmsh")
        mesh.save(tmp_path / "b.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh")
        assert len(reader) == 2
        m, meta = reader[0]
        assert isinstance(m, Mesh)
        assert m.n_points == mesh.n_points
        assert "source_path" in meta
        assert "index" in meta
        assert meta["index"] == 0

    def test_negative_index(self, tmp_path):
        mesh = two_triangles_2d.load()
        mesh.save(tmp_path / "single.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh")
        m1, _ = reader[0]
        m2, _ = reader[-1]
        assert m1.n_points == m2.n_points

    def test_iter(self, tmp_path):
        mesh = two_triangles_2d.load()
        for i in range(3):
            mesh.save(tmp_path / f"m{i}.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh")
        samples = list(reader)
        assert len(samples) == 3
        for m, meta in samples:
            assert isinstance(m, Mesh)
            assert isinstance(meta, dict)

    def test_subsample_n_points(self, tmp_path):
        mesh = Mesh(points=torch.randn(50, 3))
        mesh.save(tmp_path / "m.pt")
        reader = MeshReader(tmp_path, pattern="*.pt", subsample_n_points=10)
        reader.set_generator(torch.Generator().manual_seed(0))
        loaded, _ = reader[0]
        assert loaded.n_points == 10

    def test_subsample_n_points_wraps_cyclically(self):
        mesh = Mesh(points=torch.arange(10, dtype=torch.float32).unsqueeze(-1))
        sampled = _subsample_mesh_points(
            mesh,
            4,
            generator=torch.Generator().manual_seed(2),
        )
        torch.testing.assert_close(
            sampled.points.squeeze(-1),
            torch.tensor([8.0, 9.0, 0.0, 1.0]),
        )


class TestDomainMeshReader:
    """Tests for DomainMeshReader (DomainMesh per sample)."""

    def _make_domain_mesh(self):
        """Create a simple DomainMesh for testing."""
        interior = Mesh(points=torch.randn(10, 3))
        wall = single_triangle_3d.load()
        inlet = single_triangle_3d.load()
        return DomainMesh(
            interior=interior,
            boundaries={"wall": wall, "inlet": inlet},
            global_data={"Re": torch.tensor(1e6)},
        )

    def test_len_and_getitem(self, tmp_path):
        dm = self._make_domain_mesh()
        dm.save(tmp_path / "sample_a.pdmsh")
        dm.save(tmp_path / "sample_b.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        assert len(reader) == 2
        loaded, meta = reader[0]
        assert isinstance(loaded, DomainMesh)
        assert loaded.interior.n_points == dm.interior.n_points
        assert "source_path" in meta
        assert "index" in meta
        assert meta["index"] == 0

    def test_boundary_names_in_metadata(self, tmp_path):
        dm = self._make_domain_mesh()
        dm.save(tmp_path / "dm.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        _, meta = reader[0]
        assert sorted(meta["boundary_names"]) == ["inlet", "wall"]

    def test_no_boundaries(self, tmp_path):
        dm = DomainMesh(interior=Mesh(points=torch.randn(5, 3)))
        dm.save(tmp_path / "bare.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        loaded, meta = reader[0]
        assert loaded.n_boundaries == 0
        assert meta["boundary_names"] == []

    def test_iter(self, tmp_path):
        dm = self._make_domain_mesh()
        for i in range(3):
            dm.save(tmp_path / f"dm{i}.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        samples = list(reader)
        assert len(samples) == 3
        for loaded, meta in samples:
            assert isinstance(loaded, DomainMesh)
            assert isinstance(meta, dict)

    def test_global_data_preserved(self, tmp_path):
        dm = self._make_domain_mesh()
        dm.save(tmp_path / "dm.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        loaded, _ = reader[0]
        assert "Re" in loaded.global_data.keys()

    def _save_point_cloud_with_triangulated_wall(self, tmp_path, n_cells, n_interior):
        """Point-cloud interior + disjoint-triangle wall with scattered vertex ids.

        Vertex ids are permuted so a cell's three vertices are not stored
        adjacently, as on a real surface; a contiguous point block then
        retains a cell only if all three of its vertices happen to fall in it.
        """
        gen = torch.Generator().manual_seed(0)
        wall = Mesh(
            points=torch.randn(3 * n_cells, 3, generator=gen),
            cells=torch.randperm(3 * n_cells, generator=gen).reshape(n_cells, 3),
        )
        dm = DomainMesh(
            interior=Mesh(points=torch.randn(n_interior, 3, generator=gen)),
            boundaries={"wall": wall},
        )
        dm.save(tmp_path / "dm.pdmsh")

    def test_boundary_subsample_modes(self, tmp_path):
        """``boundary_subsample`` selects which subsample(s) hit the boundaries."""
        k = 500
        self._save_point_cloud_with_triangulated_wall(tmp_path, 2000, 5000)

        def load(mode):
            reader = DomainMeshReader(
                tmp_path,
                pattern="*.pdmsh",
                subsample_n_cells=k,
                subsample_n_points=k,
                boundary_subsample=mode,
            )
            reader.set_generator(torch.Generator().manual_seed(0))
            return reader[0][0]

        ### "both" (default): cells-then-points on a triangulated boundary keeps
        ### only cells whose three vertices all survive the point cut, ~k/27.
        both = load("both")
        assert both.interior.n_points == k
        assert both.boundaries["wall"].n_cells < 100

        cells = load("cells")
        assert cells.interior.n_points == k
        assert cells.boundaries["wall"].n_cells == k

        points = load("points")
        assert points.interior.n_points == k
        assert points.boundaries["wall"].n_points == k

    def test_boundary_subsample_invalid(self, tmp_path):
        self._make_domain_mesh().save(tmp_path / "dm.pdmsh")
        with pytest.raises(ValueError, match="boundary_subsample"):
            DomainMeshReader(tmp_path, pattern="*.pdmsh", boundary_subsample="neither")

    def test_interior_n_points_range(self, tmp_path):
        """Query-count augmentation: the interior count is drawn per sample
        from ``[lo, hi]``, reproducibly under ``set_epoch``; boundaries keep
        their fixed sizes; ``None`` leaves the fixed behaviour unchanged."""
        self._save_point_cloud_with_triangulated_wall(tmp_path, 2000, 5000)
        lo, hi = 300, 900
        reader = DomainMeshReader(
            tmp_path,
            pattern="*.pdmsh",
            subsample_n_cells=500,
            interior_n_points_range=(lo, hi),
        )
        reader.set_generator(torch.Generator().manual_seed(0))
        fixed = DomainMeshReader(
            tmp_path,
            pattern="*.pdmsh",
            subsample_n_cells=500,
            subsample_n_points=500,
        )
        fixed.set_generator(torch.Generator().manual_seed(0))
        counts = []
        for epoch in range(6):
            reader.set_epoch(epoch)
            dm = reader[0][0]
            counts.append(dm.interior.n_points)
            assert lo <= dm.interior.n_points <= hi
            # Boundaries are not touched by the range: the cell subsample
            # alone keeps exactly ``subsample_n_cells`` wall cells.
            assert dm.boundaries["wall"].n_cells == 500
        assert len(set(counts)) > 1, counts
        # Reproducible: same epoch, same draw.
        reader.set_epoch(3)
        a = reader[0][0].interior.n_points
        reader.set_epoch(4)
        reader.set_epoch(3)
        assert reader[0][0].interior.n_points == a
        # Default None: fixed count unchanged.
        assert fixed[0][0].interior.n_points == 500
        with pytest.raises(ValueError):
            DomainMeshReader(
                tmp_path, pattern="*.pdmsh", interior_n_points_range=(900, 300)
            )

    def test_interior_n_points_range_overrides_subsample_n_points(self, tmp_path):
        """The range replaces ``subsample_n_points`` for the interior only;
        boundaries are still subsampled to ``subsample_n_points``."""
        self._save_point_cloud_with_triangulated_wall(tmp_path, 2000, 5000)
        lo, hi, k = 300, 900, 1000
        reader = DomainMeshReader(
            tmp_path,
            pattern="*.pdmsh",
            subsample_n_points=k,
            interior_n_points_range=[lo, hi],
        )
        reader.set_generator(torch.Generator().manual_seed(0))
        for epoch in range(4):
            reader.set_epoch(epoch)
            dm = reader[0][0]
            assert lo <= dm.interior.n_points <= hi
            assert dm.boundaries["wall"].n_points == k

    @pytest.mark.parametrize(
        "bad", [(300.9, 900.2), (300, 900, 5), (300,), (0, 10), (900, 300), 300]
    )
    def test_interior_n_points_range_invalid(self, tmp_path, bad):
        self._make_domain_mesh().save(tmp_path / "dm.pdmsh")
        with pytest.raises(ValueError, match="interior_n_points_range"):
            DomainMeshReader(tmp_path, pattern="*.pdmsh", interior_n_points_range=bad)


class TestMeshDataset:
    """Tests for MeshDataset with mesh transforms."""

    def test_single_mesh_with_transform(self, tmp_path):
        mesh = two_triangles_2d.load()
        mesh.save(tmp_path / "m.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh")
        ds = MeshDataset(reader, transforms=[ScaleMesh(2.0)])
        m, meta = ds[0]
        assert isinstance(m, Mesh)
        assert m.n_points == mesh.n_points

    def test_domain_mesh_with_transform(self, tmp_path):
        interior = Mesh(points=torch.randn(10, 3))
        wall = single_triangle_3d.load()
        dm = DomainMesh(
            interior=interior,
            boundaries={"wall": wall},
        )
        dm.save(tmp_path / "dm.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        ds = MeshDataset(reader, transforms=[ScaleMesh(0.5)])
        loaded, meta = ds[0]
        assert isinstance(loaded, DomainMesh)
        assert loaded.interior.n_points == interior.n_points
        assert loaded.n_boundaries == 1

    def test_domain_mesh_transform_applies_to_all(self, tmp_path):
        interior = Mesh(
            points=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        )
        wall = Mesh(
            points=torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        )
        dm = DomainMesh(interior=interior, boundaries={"wall": wall})
        dm.save(tmp_path / "dm.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        ds = MeshDataset(reader, transforms=[ScaleMesh(2.0)])
        loaded, _ = ds[0]
        assert torch.allclose(
            loaded.interior.points,
            torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        )
        assert torch.allclose(
            loaded.boundaries["wall"].points,
            torch.tensor([[2.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
        )

    def test_close_closes_mesh_readers(self, tmp_path):
        """MeshDataset.close() closes MeshReader and DomainMeshReader."""
        two_triangles_2d.load().save(tmp_path / "m.pmsh")
        DomainMesh(interior=Mesh(points=torch.randn(10, 3))).save(tmp_path / "dm.pdmsh")
        for reader in (
            MeshReader(tmp_path, pattern="*.pmsh"),
            DomainMeshReader(tmp_path, pattern="*.pdmsh"),
        ):
            ds = MeshDataset(reader)
            ds[0]
            ds.close()


class TestDomainMeshTransforms:
    """Tests for DomainMesh-aware transform behavior via apply_to_domain."""

    def test_scale_transforms_domain_global_data(self, tmp_path):
        """ScaleMesh with transform_global_data=True should scale domain global_data."""
        dm = DomainMesh(
            interior=Mesh(
                points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            ),
            global_data={"velocity": torch.tensor([1.0, 0.0, 0.0])},
        )
        dm.save(tmp_path / "dm.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        ds = MeshDataset(
            reader,
            transforms=[ScaleMesh(2.0, transform_global_data=True)],
        )
        loaded, _ = ds[0]
        assert torch.allclose(
            loaded.global_data["velocity"],
            torch.tensor([2.0, 0.0, 0.0]),
        )

    def test_scale_preserves_domain_global_data_by_default(self, tmp_path):
        """ScaleMesh without transform_global_data leaves domain global_data unchanged."""
        dm = DomainMesh(
            interior=Mesh(
                points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            ),
            global_data={"velocity": torch.tensor([1.0, 0.0, 0.0])},
        )
        dm.save(tmp_path / "dm.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        ds = MeshDataset(reader, transforms=[ScaleMesh(2.0)])
        loaded, _ = ds[0]
        assert torch.allclose(
            loaded.global_data["velocity"],
            torch.tensor([1.0, 0.0, 0.0]),
        )

    def test_random_scale_consistent_across_meshes(self, tmp_path):
        """RandomScaleMesh should apply the same factor to interior and boundaries."""
        interior = Mesh(
            points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        wall = Mesh(
            points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        dm = DomainMesh(interior=interior, boundaries={"wall": wall})
        dm.save(tmp_path / "dm.pdmsh")

        aug = RandomScaleMesh(
            distribution=torch.distributions.Uniform(0.5, 2.0),
        )
        aug.set_generator(torch.Generator().manual_seed(42))
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        ds = MeshDataset(
            reader,
            transforms=[aug],
        )
        loaded, _ = ds[0]

        interior_factor = loaded.interior.points[1, 0].item()
        wall_factor = loaded.boundaries["wall"].points[1, 0].item()
        assert interior_factor == pytest.approx(wall_factor)

    def test_center_mesh_uses_interior_com(self, tmp_path):
        """CenterMesh should center by interior COM, not per-mesh COM."""
        interior = Mesh(
            points=torch.tensor(
                [
                    [2.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                ]
            ),
        )
        wall = Mesh(
            points=torch.tensor(
                [
                    [10.0, 0.0, 0.0],
                    [12.0, 0.0, 0.0],
                ]
            ),
        )
        dm = DomainMesh(interior=interior, boundaries={"wall": wall})
        dm.save(tmp_path / "dm.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        ds = MeshDataset(
            reader,
            transforms=[CenterMesh(use_area_weighting=False)],
        )
        loaded, _ = ds[0]

        interior_com = loaded.interior.points.mean(dim=0)
        assert torch.allclose(interior_com, torch.zeros(3), atol=1e-6)

        expected_wall = torch.tensor(
            [
                [10.0 - 3.0, 0.0, 0.0],
                [12.0 - 3.0, 0.0, 0.0],
            ]
        )
        assert torch.allclose(loaded.boundaries["wall"].points, expected_wall)


class TestMeshDatasetStreams:
    """Tests for MeshDataset prefetching with real CUDA streams."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prefetch_with_stream(self, tmp_path):
        """Prefetch with a CUDA stream transfers data to GPU."""
        mesh = two_triangles_2d.load()
        mesh.save(tmp_path / "m.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh", pin_memory=True)
        ds = MeshDataset(reader, device="cuda:0")

        stream = torch.cuda.Stream()
        ds.prefetch(0, stream=stream)

        data, _ = ds[0]
        assert data.points.device.type == "cuda"
        torch.cuda.synchronize()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prefetch_with_stream_and_transforms(self, tmp_path):
        """Prefetch with CUDA stream applies transforms on GPU."""
        mesh = Mesh(
            points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        mesh.save(tmp_path / "m.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh", pin_memory=True)
        ds = MeshDataset(reader, transforms=[ScaleMesh(2.0)], device="cuda:0")

        stream = torch.cuda.Stream()
        ds.prefetch(0, stream=stream)

        data, _ = ds[0]
        assert data.points.device.type == "cuda"
        expected = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], device="cuda:0")
        assert torch.allclose(data.points, expected)
        torch.cuda.synchronize()


class TestMeshReaderSubsamplingRNG:
    """Tests for MeshReader subsampling RNG reproducibility."""

    def test_subsample_reproducible(self, tmp_path):
        """Same generator seed yields identical subsampled points."""
        mesh = Mesh(points=torch.randn(100, 3))
        mesh.save(tmp_path / "m.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh", subsample_n_points=10)

        reader.set_generator(torch.Generator().manual_seed(42))
        data1, _ = reader[0]

        reader.set_generator(torch.Generator().manual_seed(42))
        data2, _ = reader[0]

        assert torch.equal(data1.points, data2.points)

    def test_subsample_epoch_changes_output(self, tmp_path):
        """Different epochs produce different subsampled slices."""
        mesh = Mesh(points=torch.randn(100, 3))
        mesh.save(tmp_path / "m.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh", subsample_n_points=10)

        reader.set_generator(torch.Generator().manual_seed(42))
        reader.set_epoch(0)
        data_e0, _ = reader[0]

        reader.set_generator(torch.Generator().manual_seed(42))
        reader.set_epoch(1)
        data_e1, _ = reader[0]

        assert not torch.equal(data_e0.points, data_e1.points)


class TestDomainMeshReaderExtraBoundaries:
    """Tests for DomainMeshReader extra_boundaries feature."""

    def test_extra_boundaries_loaded(self, tmp_path):
        """Extra boundary mesh is loaded alongside the DomainMesh."""
        interior = Mesh(points=torch.randn(10, 3))
        dm = DomainMesh(interior=interior)
        case_dir = tmp_path / "case"
        case_dir.mkdir()
        dm.save(case_dir / "domain.pdmsh")

        farfield = Mesh(points=torch.randn(7, 3))
        farfield.save(case_dir / "farfield_001.pmsh")

        reader = DomainMeshReader(
            tmp_path,
            pattern="**/*.pdmsh",
            extra_boundaries={"farfield": {"pattern": "farfield*.pmsh"}},
        )
        assert len(reader) == 1

        loaded, meta = reader[0]
        assert "farfield" in loaded.boundary_names
        assert loaded.boundaries["farfield"].n_points == 7

    def test_extra_boundaries_missing_raises(self, tmp_path):
        """Missing extra boundary file raises FileNotFoundError."""
        interior = Mesh(points=torch.randn(10, 3))
        dm = DomainMesh(interior=interior)
        dm.save(tmp_path / "domain.pdmsh")

        reader = DomainMeshReader(
            tmp_path,
            pattern="*.pdmsh",
            extra_boundaries={"missing_bnd": {"pattern": "nonexistent*.pmsh"}},
        )

        with pytest.raises(FileNotFoundError, match="nonexistent"):
            _ = reader[0]


class TestReadersKeepDatasetFilesUnchanged:
    """Readers map saved meshes copy-on-write: an in-place edit of a loaded
    sample stays in memory and never reaches the dataset on disk."""

    def test_mesh_reader(self, tmp_path):
        mesh = single_triangle_3d.load()
        mesh.point_data["p"] = torch.ones(mesh.n_points)
        mesh.save(tmp_path / "m.pmsh")

        loaded, _ = MeshReader(tmp_path, pattern="*.pmsh")[0]
        loaded.points.add_(1.0)
        loaded.point_data["p"].mul_(0.0)

        reloaded = Mesh.load(tmp_path / "m.pmsh")
        torch.testing.assert_close(reloaded.points, mesh.points)
        torch.testing.assert_close(reloaded.point_data["p"], mesh.point_data["p"])

    def test_domain_mesh_reader_with_extra_boundary(self, tmp_path):
        case_dir = tmp_path / "case"
        case_dir.mkdir()
        dm = DomainMesh(
            interior=Mesh(points=torch.randn(10, 3)),
            boundaries={"wall": single_triangle_3d.load()},
        )
        dm.save(case_dir / "domain.pdmsh")
        farfield = Mesh(points=torch.randn(7, 3))
        farfield.save(case_dir / "farfield.pmsh")

        reader = DomainMeshReader(
            tmp_path,
            pattern="**/*.pdmsh",
            extra_boundaries={"farfield": {"pattern": "farfield*.pmsh"}},
        )
        loaded, _ = reader[0]
        loaded.interior.points.add_(1.0)
        loaded.boundaries["wall"].points.add_(1.0)
        loaded.boundaries["farfield"].points.add_(1.0)

        reloaded = DomainMesh.load(case_dir / "domain.pdmsh")
        torch.testing.assert_close(reloaded.interior.points, dm.interior.points)
        torch.testing.assert_close(
            reloaded.boundaries["wall"].points, dm.boundaries["wall"].points
        )
        torch.testing.assert_close(
            Mesh.load(case_dir / "farfield.pmsh").points, farfield.points
        )


class TestTensorDictMeshApply:
    """Verifies the recipe-pipeline contract: ``td.apply(transform,
    call_on_nested=True)`` invokes the transform on each top-level
    ``Mesh`` value rather than recursing into its tensor leaves.
    """

    def test_scale_each(self):
        from tensordict import TensorDict

        mesh = two_triangles_2d.load()
        original_points = mesh.points.clone()
        td = TensorDict({"x": mesh, "y": mesh.clone()}, batch_size=[])
        out = td.apply(ScaleMesh(3.0), call_on_nested=True)
        assert out["x"].n_points == mesh.n_points
        assert "x" in out
        assert "y" in out
        assert torch.allclose(out["x"].points, original_points * 3.0)
        assert torch.allclose(out["y"].points, original_points * 3.0)


class TestCellSubsampleMeasureWeights:
    """Reader-side cell subsampling records inverse inclusion probabilities as measure weights."""

    def _make_strip(self, n_cells: int) -> Mesh:
        """Disjoint unit right triangles (area 0.5 each) along the x-axis."""
        pts = []
        for i in range(n_cells):
            pts += [
                [float(i), 0.0, 0.0],
                [float(i) + 1.0, 0.0, 0.0],
                [float(i), 1.0, 0.0],
            ]
        return Mesh(
            points=torch.tensor(pts),
            cells=torch.arange(3 * n_cells).reshape(n_cells, 3),
        )

    def test_reader_records_weight(self, tmp_path):
        n, k = 40, 12
        self._make_strip(n).save(tmp_path / "m.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh", subsample_n_cells=k)
        reader.set_generator(torch.Generator().manual_seed(0))
        mesh, _ = reader[0]
        assert mesh.n_cells == k
        torch.testing.assert_close(cell_measures(mesh), mesh.cell_areas * (n / k))

    def test_reader_noop_below_threshold(self, tmp_path):
        n = 8
        self._make_strip(n).save(tmp_path / "m.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh", subsample_n_cells=20)
        mesh, _ = reader[0]
        assert mesh.n_cells == n
        assert EFFECTIVE_MEASURE_KEY not in mesh.cell_data.keys()

    def test_equal_area_mesh_recovers_total_exactly(self, tmp_path):
        # With identical triangles, ANY cyclic block reproduces the full
        # area: sum(area * N/k) == k * 0.5 * N/k == full total.
        n, k = 30, 7
        full = self._make_strip(n)
        full.save(tmp_path / "m.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh", subsample_n_cells=k)
        reader.set_generator(torch.Generator().manual_seed(0))
        for _ in range(5):
            mesh, _ = reader[0]
            corrected = cell_measures(mesh).sum()
            torch.testing.assert_close(corrected, full.cell_areas.sum())

    def test_composes_with_subsample_mesh_transform(self, tmp_path):
        # Reader keeps k1 of N (weight N/k1); SubsampleMesh keeps k2 of k1
        # (weight x k1/k2). Composed weight must be exactly N/k2.
        n, k1, k2 = 60, 20, 5
        self._make_strip(n).save(tmp_path / "m.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh", subsample_n_cells=k1)
        reader.set_generator(torch.Generator().manual_seed(0))
        mesh, _ = reader[0]
        mesh = SubsampleMesh(n_cells=k2)(mesh)
        assert mesh.n_cells == k2
        torch.testing.assert_close(cell_measures(mesh), mesh.cell_areas * (n / k2))

    def test_domain_mesh_reader_records_weights_on_boundaries(self, tmp_path):
        interior = Mesh(points=torch.randn(10, 3))
        wall = self._make_strip(24)
        dm = DomainMesh(interior=interior, boundaries={"wall": wall})
        dm.save(tmp_path / "dm.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh", subsample_n_cells=6)
        loaded, _ = reader[0]
        assert loaded.boundaries["wall"].n_cells == 6
        torch.testing.assert_close(
            cell_measures(loaded.boundaries["wall"]),
            loaded.boundaries["wall"].cell_areas * (24 / 6),
        )
        ### Interior is a point cloud: no cells, no weights.
        assert EFFECTIVE_MEASURE_KEY not in loaded.interior.cell_data.keys()

    def test_seeded_reproducibility(self, tmp_path):
        ### Reader RNG is derived per-sample from (base_seed, epoch, index):
        ### the same triple must reproduce the same block; varying the epoch
        ### must produce a variety of blocks.
        self._make_strip(50).save(tmp_path / "m.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh", subsample_n_cells=10)
        reader.set_generator(torch.Generator().manual_seed(123))
        m1, _ = reader[0]
        m2, _ = reader[0]
        torch.testing.assert_close(m1.points, m2.points)
        distinct_blocks = set()
        for epoch in range(20):
            reader.set_epoch(epoch)
            m, _ = reader[0]
            distinct_blocks.add(round(float(m.points[0, 0])))
        assert len(distinct_blocks) > 5


class TestMeshReaderLifecycle:
    """``close()`` on the mesh readers and its propagation from MeshDataset."""

    def test_mesh_reader_close_is_noop_and_idempotent(self, tmp_path):
        two_triangles_2d.load().save(tmp_path / "m.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh")
        reader.close()
        reader.close()
        # Still usable: there is no reader-level resource to release.
        m, _ = reader[0]
        assert isinstance(m, Mesh)

    def test_domain_mesh_reader_close_is_noop_and_idempotent(self, tmp_path):
        DomainMesh(interior=Mesh(points=torch.randn(5, 3))).save(tmp_path / "d.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")
        reader.close()
        reader.close()
        dm, _ = reader[0]
        assert isinstance(dm, DomainMesh)

    @pytest.mark.parametrize("kind", ["mesh", "domain"])
    def test_mesh_dataset_close_closes_reader(self, tmp_path, monkeypatch, kind):
        if kind == "mesh":
            two_triangles_2d.load().save(tmp_path / "m.pmsh")
            reader = MeshReader(tmp_path, pattern="*.pmsh")
        else:
            DomainMesh(interior=Mesh(points=torch.randn(5, 3))).save(
                tmp_path / "d.pdmsh"
            )
            reader = DomainMeshReader(tmp_path, pattern="*.pdmsh")

        calls: list[int] = []
        monkeypatch.setattr(reader, "close", lambda: calls.append(1))
        ds = MeshDataset(reader)
        _ = ds[0]
        ds.close()
        assert calls == [1]


class TestMeshReaderMetadata:
    """Metadata is assembled before pinning and honours include_index_in_metadata."""

    @pytest.mark.parametrize("include_index", [True, False])
    def test_mesh_reader_include_index(self, tmp_path, include_index):
        two_triangles_2d.load().save(tmp_path / "m.pmsh")
        reader = MeshReader(
            tmp_path, pattern="*.pmsh", include_index_in_metadata=include_index
        )
        _, meta = reader[0]
        assert "source_path" in meta
        assert ("index" in meta) is include_index

    @pytest.mark.parametrize("include_index", [True, False])
    def test_domain_mesh_reader_include_index(self, tmp_path, include_index):
        DomainMesh(interior=Mesh(points=torch.randn(5, 3))).save(tmp_path / "d.pdmsh")
        reader = DomainMeshReader(
            tmp_path, pattern="*.pdmsh", include_index_in_metadata=include_index
        )
        _, meta = reader[0]
        assert "source_path" in meta
        assert "boundary_names" in meta
        assert ("index" in meta) is include_index

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mesh_reader_pin_memory_keeps_metadata(self, tmp_path):
        two_triangles_2d.load().save(tmp_path / "m.pmsh")
        reader = MeshReader(tmp_path, pattern="*.pmsh", pin_memory=True)
        m, meta = reader[0]
        assert m.points.is_pinned()
        assert meta["index"] == 0
        assert "source_path" in meta

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_domain_mesh_reader_pin_memory_keeps_metadata(self, tmp_path):
        dm = DomainMesh(
            interior=Mesh(points=torch.randn(5, 3)),
            boundaries={"wall": single_triangle_3d.load()},
        )
        dm.save(tmp_path / "d.pdmsh")
        reader = DomainMeshReader(tmp_path, pattern="*.pdmsh", pin_memory=True)
        loaded, meta = reader[0]
        assert loaded.interior.points.is_pinned()
        assert meta["boundary_names"] == ["wall"]
        assert meta["index"] == 0


class TestDomainMeshReaderExtraBoundaryLoading:
    """The split between locating/loading extra meshes and attaching them."""

    def _case(self, tmp_path, *, n_extra: int = 1):
        case_dir = tmp_path / "case"
        case_dir.mkdir()
        dm = DomainMesh(
            interior=Mesh(points=torch.randn(10, 3)),
            boundaries={"wall": single_triangle_3d.load()},
        )
        dm.save(case_dir / "domain.pdmsh")
        for i in range(n_extra):
            Mesh(points=torch.randn(7 + i, 3)).save(case_dir / f"farfield_{i:03d}.pmsh")
        return case_dir

    def test_load_extra_boundary_meshes_returns_only_extras(self, tmp_path):
        self._case(tmp_path)
        reader = DomainMeshReader(
            tmp_path,
            pattern="**/*.pdmsh",
            extra_boundaries={"farfield": {"pattern": "farfield*.pmsh"}},
        )
        extras = reader._load_extra_boundary_meshes(0)
        assert set(extras) == {"farfield"}
        assert extras["farfield"].n_points == 7

    def test_load_extra_boundaries_keeps_existing_boundaries(self, tmp_path):
        self._case(tmp_path)
        reader = DomainMeshReader(
            tmp_path,
            pattern="**/*.pdmsh",
            extra_boundaries={"farfield": {"pattern": "farfield*.pmsh"}},
        )
        loaded, meta = reader[0]
        assert sorted(loaded.boundary_names) == ["farfield", "wall"]
        assert sorted(meta["boundary_names"]) == ["farfield", "wall"]
        assert loaded.boundaries["wall"].n_points == 3

    def test_multiple_matches_warns_and_uses_first_sorted(self, tmp_path, caplog):
        self._case(tmp_path, n_extra=2)
        reader = DomainMeshReader(
            tmp_path,
            pattern="**/*.pdmsh",
            extra_boundaries={"farfield": {"pattern": "farfield*.pmsh"}},
        )
        with caplog.at_level("WARNING"):
            extras = reader._load_extra_boundary_meshes(0)
        assert extras["farfield"].n_points == 7  # farfield_000
        assert any("Multiple meshes" in rec.getMessage() for rec in caplog.records)

    def test_zarr_directory_extra_boundary_uses_from_zarr(self, tmp_path):
        pytest.importorskip("zarr")
        from physicsnemo.mesh.io import to_zarr

        case_dir = tmp_path / "case"
        case_dir.mkdir()
        DomainMesh(interior=Mesh(points=torch.randn(10, 3))).save(
            case_dir / "domain.pdmsh"
        )
        to_zarr(Mesh(points=torch.randn(9, 3)), case_dir / "farfield.zarr")

        reader = DomainMeshReader(
            tmp_path,
            pattern="**/*.pdmsh",
            extra_boundaries={"farfield": {"pattern": "farfield*.zarr"}},
        )
        loaded, _ = reader[0]
        assert loaded.boundaries["farfield"].n_points == 9
