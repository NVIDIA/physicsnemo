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

"""Tests for MeshReader / DomainMeshReader with DatasetCache."""

import pytest
import torch

from physicsnemo.datapipes.caching import DatasetCache
from physicsnemo.datapipes.readers.mesh import DomainMeshReader, MeshReader
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.primitives.basic import two_triangles_2d


def _assert_mesh_equal(a: Mesh, b: Mesh):
    assert torch.equal(a.points, b.points)
    assert torch.equal(a.cells, b.cells)
    for key in a.point_data.keys(include_nested=True, leaves_only=True):
        assert torch.equal(a.point_data[key], b.point_data[key])
    for key in a.cell_data.keys(include_nested=True, leaves_only=True):
        assert torch.equal(a.cell_data[key], b.cell_data[key])
    for key in a.global_data.keys(include_nested=True, leaves_only=True):
        assert torch.equal(a.global_data[key], b.global_data[key])


def _assert_domain_mesh_equal(a: DomainMesh, b: DomainMesh):
    assert a.boundary_names == b.boundary_names
    _assert_mesh_equal(a.interior, b.interior)
    for name in a.boundary_names:
        _assert_mesh_equal(a.boundaries[name], b.boundaries[name])
    for key in a.global_data.keys(include_nested=True, leaves_only=True):
        assert torch.equal(a.global_data[key], b.global_data[key])


@pytest.fixture
def mesh_root(tmp_path):
    """Three cases, each with a mesh sample and a sibling geometry mesh."""
    root = tmp_path / "data"
    base = two_triangles_2d.load()
    geo = two_triangles_2d.load()
    geo.points = geo.points + 10.0
    for i in range(3):
        case = root / f"run_{i}"
        case.mkdir(parents=True)
        mesh = base.clone()
        mesh.point_data["p"] = torch.randn(mesh.n_points, 2)
        mesh.global_data["Re"] = torch.tensor(1.0e6 + i)
        mesh.save(case / "sample.pmsh")
        dm = DomainMesh(
            interior=mesh.clone(),
            boundaries={"wall": geo.clone()},
            global_data={"U_inf": torch.tensor(30.0 + i)},
        )
        dm.save(case / "domain.pdmsh")
        geo.save(case / "geo_single_solid.stl.pmsh")
    return root


@pytest.fixture(
    params=["ram_only", "disk_only", "ram_and_disk"],
)
def cache(request, tmp_path):
    ram = None if request.param == "disk_only" else 2**24
    disk = None if request.param == "ram_only" else tmp_path / "cache"
    return DatasetCache(ram_bytes_limit=ram, disk_dir=disk)


class TestMeshReaderCache:
    def test_matches_uncached(self, mesh_root, cache):
        plain = MeshReader(mesh_root, pattern="**/sample.pmsh")
        cached = MeshReader(mesh_root, pattern="**/sample.pmsh", cache=cache)
        assert len(plain) == len(cached) == 3
        for _ in range(2):  # cold then warm
            for i in range(len(plain)):
                a, meta_a = plain[i]
                b, meta_b = cached[i]
                _assert_mesh_equal(a, b)
                assert meta_a == meta_b

    def test_subsampling_matches_uncached_across_epochs(self, mesh_root, cache):
        kwargs = dict(pattern="**/sample.pmsh", subsample_n_points=3)
        plain = MeshReader(mesh_root, **kwargs)
        cached = MeshReader(mesh_root, **kwargs, cache=cache)
        gen = torch.Generator()
        gen.manual_seed(1234)
        plain.set_generator(gen)
        cached.set_generator(gen)
        for epoch in range(3):
            plain.set_epoch(epoch)
            cached.set_epoch(epoch)
            for i in range(len(plain)):
                a, _ = plain[i]
                b, _ = cached[i]
                _assert_mesh_equal(a, b)

    def test_warm_read_skips_stock_loader(self, mesh_root, monkeypatch):
        cache = DatasetCache(ram_bytes_limit=2**24)
        reader = MeshReader(mesh_root, pattern="**/sample.pmsh", cache=cache)
        expected, _ = reader[0]  # populate

        def boom(path):
            raise AssertionError(f"Mesh.load called on warm read: {path}")

        monkeypatch.setattr(Mesh, "load", boom)
        warm, _ = reader[0]
        _assert_mesh_equal(expected, warm)

    def test_discovery_attributes_unchanged(self, mesh_root, cache):
        cached = MeshReader(mesh_root, pattern="**/sample.pmsh", cache=cache)
        assert cached._root == mesh_root
        assert len(cached._paths) == 3


class TestDomainMeshReaderCache:
    def test_matches_uncached(self, mesh_root, cache):
        plain = DomainMeshReader(mesh_root)
        cached = DomainMeshReader(mesh_root, cache=cache)
        for _ in range(2):  # cold then warm
            for i in range(len(plain)):
                a, meta_a = plain[i]
                b, meta_b = cached[i]
                _assert_domain_mesh_equal(a, b)
                assert meta_a == meta_b

    def test_full_pipeline_matches_uncached(self, mesh_root, cache):
        """Subsampling + drops + extra boundaries, cached vs uncached, 2 epochs."""
        kwargs = dict(
            subsample_n_points=3,
            drop_interior_cells=True,
            extra_boundaries={"geo": {"pattern": "geo_*.pmsh"}},
        )
        plain = DomainMeshReader(mesh_root, **kwargs)
        cached = DomainMeshReader(mesh_root, **kwargs, cache=cache)
        gen = torch.Generator()
        gen.manual_seed(7)
        plain.set_generator(gen)
        cached.set_generator(gen)
        for epoch in range(2):
            plain.set_epoch(epoch)
            cached.set_epoch(epoch)
            for i in range(len(plain)):
                a, _ = plain[i]
                b, _ = cached[i]
                _assert_domain_mesh_equal(a, b)

    def test_warm_read_skips_stock_loaders(self, mesh_root, monkeypatch):
        cache = DatasetCache(ram_bytes_limit=2**24)
        reader = DomainMeshReader(
            mesh_root,
            extra_boundaries={"geo": {"pattern": "geo_*.pmsh"}},
            cache=cache,
        )
        expected, _ = reader[0]  # populate: sample + glob + extra boundary

        def boom(path):
            raise AssertionError(f"stock load called on warm read: {path}")

        monkeypatch.setattr(DomainMesh, "load", boom)
        monkeypatch.setattr(Mesh, "load", boom)
        warm, _ = reader[0]
        _assert_domain_mesh_equal(expected, warm)

    def test_disk_persistence_across_reader_instances(self, mesh_root, tmp_path):
        """A fresh reader + fresh cache on a warm disk dir loads via mirrors."""
        disk = tmp_path / "cache"
        warm_cache = DatasetCache(ram_bytes_limit=None, disk_dir=disk)
        warm_reader = DomainMeshReader(mesh_root, cache=warm_cache)
        expected = [warm_reader[i][0] for i in range(len(warm_reader))]

        fresh_cache = DatasetCache(ram_bytes_limit=None, disk_dir=disk)
        fresh_reader = DomainMeshReader(mesh_root, cache=fresh_cache)
        for i in range(len(fresh_reader)):
            _assert_domain_mesh_equal(expected[i], fresh_reader[i][0])
        assert fresh_cache.stats()["disk"]["hits"] >= len(fresh_reader)

    def test_fallback_when_mirror_destroyed(self, mesh_root, tmp_path):
        """Deleting mirror internals mid-run degrades to source loads."""
        cache = DatasetCache(ram_bytes_limit=None, disk_dir=tmp_path / "cache")
        reader = DomainMeshReader(mesh_root, cache=cache)
        expected, _ = reader[0]

        for mirror in (tmp_path / "cache").rglob("*.tree"):
            for meta in mirror.rglob("meta.json"):
                meta.unlink()

        again, _ = reader[0]
        _assert_domain_mesh_equal(expected, again)

    def test_repeated_reads_do_not_leak_subsampling_state(self, mesh_root):
        """Cached raw meshes must be unaffected by downstream subsampling."""
        cache = DatasetCache(ram_bytes_limit=2**24)
        reader = DomainMeshReader(mesh_root, subsample_n_points=3, cache=cache)
        gen = torch.Generator()
        gen.manual_seed(0)
        reader.set_generator(gen)
        a, _ = reader[0]
        b, _ = reader[0]  # same epoch, same index -> identical subsample
        _assert_domain_mesh_equal(a, b)
        # And the full-resolution cached entry is still intact.
        full = DomainMeshReader(mesh_root, cache=cache)[0][0]
        plain = DomainMeshReader(mesh_root)[0][0]
        _assert_domain_mesh_equal(full, plain)
