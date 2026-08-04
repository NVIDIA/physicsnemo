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

"""DatasetCache integration tests for the non-mesh readers.

Mesh reader integration is covered in ``test_mesh_reader_cache.py``; this
file covers the level-1 (blob) opt-ins: zarr key/attr discovery, tensorstore
attributes, and the uniform ``cache`` kwarg on the remaining readers.
"""

import numpy as np
import pytest
import torch

from physicsnemo.datapipes.caching import DatasetCache, cached_or_load
from physicsnemo.datapipes.readers.base import Reader
from test.conftest import requires_module


@pytest.fixture
def zarr_dir_with_attrs(tmp_path):
    """Directory of zarr groups with both arrays and attributes."""
    zarr = pytest.importorskip("zarr", minversion="3.0")
    for i in range(3):
        root = zarr.open(tmp_path / f"sample_{i}.zarr", mode="w")
        root.create_array("field", data=np.random.randn(20, 3).astype(np.float32))
        root.attrs["timestep"] = float(i)
    return tmp_path


class TestCachedOrLoadHelper:
    def test_no_cache_is_transparent(self, tmp_path):
        assert cached_or_load(None, "k/v1", tmp_path, lambda: 41) == 41
        assert cached_or_load(
            None, "k/v1", tmp_path, lambda p: str(p), src=tmp_path
        ) == str(tmp_path)

    def test_keys_by_resolved_path(self, tmp_path):
        (tmp_path / "real").mkdir()
        (tmp_path / "link").symlink_to(tmp_path / "real")
        cache = DatasetCache(ram_bytes_limit=2**20)
        calls = []

        def loader():
            calls.append(1)
            return "v"

        cached_or_load(cache, "k/v1", tmp_path / "real", loader)
        cached_or_load(cache, "k/v1", tmp_path / "link", loader)
        assert len(calls) == 1  # symlink and target share one entry


class _TinyReader(Reader):
    """Minimal Reader to exercise the base-class cache plumbing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.field_name_computations = 0

    def _load_sample(self, index):
        return {"x": torch.tensor([float(index)])}

    def _get_field_names(self):
        self.field_name_computations += 1
        return super()._get_field_names()

    def __len__(self):
        return 4


class TestReaderBasePlumbing:
    def test_field_names_memoized(self):
        reader = _TinyReader()
        assert reader.field_names == ["x"]
        assert reader.field_names == ["x"]
        assert reader.field_name_computations == 1

    def test_negative_indexing_still_works(self):
        reader = _TinyReader()
        data, meta = reader[-1]
        assert meta["index"] == 3
        with pytest.raises(IndexError):
            reader[-5]

    def test_cached_helper_uses_reader_cache(self, tmp_path):
        cache = DatasetCache(ram_bytes_limit=2**20)
        reader = _TinyReader(cache=cache)
        calls = []

        def loader():
            calls.append(1)
            return {"v": 1}

        a = reader._cached("k/v1", tmp_path, loader)
        b = reader._cached("k/v1", tmp_path, loader)
        assert a == b == {"v": 1}
        assert len(calls) == 1


@requires_module("zarr>=3.0.0")
class TestZarrReaderCache:
    def test_matches_uncached(self, zarr_dir_with_attrs):
        from physicsnemo.datapipes.readers.zarr import ZarrReader

        cache = DatasetCache(ram_bytes_limit=2**24)
        plain = ZarrReader(zarr_dir_with_attrs, fields=["field", "timestep"])
        cached = ZarrReader(
            zarr_dir_with_attrs, fields=["field", "timestep"], cache=cache
        )
        for _ in range(2):  # cold then warm
            for i in range(len(plain)):
                a, _ = plain[i]
                b, _ = cached[i]
                assert torch.equal(a["field"], b["field"])
                assert torch.equal(a["timestep"], b["timestep"])
        assert cache.stats()["ram"]["hits"] > 0

    def test_key_discovery_cached(self, zarr_dir_with_attrs):
        from physicsnemo.datapipes.readers.zarr import ZarrReader

        cache = DatasetCache(ram_bytes_limit=2**24)
        reader = ZarrReader(zarr_dir_with_attrs, cache=cache)
        reader[0]
        misses_after_first = cache.stats()["ram"]["misses"]
        reader[0]
        assert cache.stats()["ram"]["misses"] == misses_after_first


@requires_module("tensorstore")
class TestTensorStoreZarrReaderCache:
    def test_attrs_matches_uncached(self, zarr_dir_with_attrs):
        from physicsnemo.datapipes.readers.tensorstore_zarr import (
            TensorStoreZarrReader,
        )

        cache = DatasetCache(ram_bytes_limit=2**24)
        plain = TensorStoreZarrReader(zarr_dir_with_attrs, fields=["field", "timestep"])
        cached = TensorStoreZarrReader(
            zarr_dir_with_attrs, fields=["field", "timestep"], cache=cache
        )
        for _ in range(2):  # cold then warm
            for i in range(len(plain)):
                a, _ = plain[i]
                b, _ = cached[i]
                assert torch.equal(a["field"], b["field"])
                assert torch.equal(a["timestep"], b["timestep"])
        assert cache.stats()["ram"]["hits"] > 0

    def test_attrs_read_once_per_group(self, zarr_dir_with_attrs, monkeypatch):
        from physicsnemo.datapipes.readers.tensorstore_zarr import (
            TensorStoreZarrReader,
        )

        cache = DatasetCache(ram_bytes_limit=2**24)
        reader = TensorStoreZarrReader(
            zarr_dir_with_attrs, fields=["field", "timestep"], cache=cache
        )
        expected, _ = reader[0]

        def boom(group_path):
            raise AssertionError("attrs re-read on warm sample")

        monkeypatch.setattr(reader, "_read_attributes_from_store", boom)
        warm, _ = reader[0]
        assert torch.equal(expected["timestep"], warm["timestep"])

    def test_warm_open_skips_array_metadata_files(self, zarr_dir_with_attrs):
        """assume_metadata: warm reads work even with array metadata deleted."""
        from physicsnemo.datapipes.readers.tensorstore_zarr import (
            TensorStoreZarrReader,
        )

        cache = DatasetCache(ram_bytes_limit=2**24)
        reader = TensorStoreZarrReader(
            zarr_dir_with_attrs, fields=["field"], cache=cache
        )
        expected = [reader[i][0] for i in range(len(reader))]  # populate specs

        # Remove every array-level metadata file: only cached specs remain.
        for group in zarr_dir_with_attrs.glob("*.zarr"):
            for name in ("zarr.json", ".zarray"):
                meta = group / "field" / name
                if meta.exists():
                    meta.unlink()

        for i in range(len(reader)):
            warm, _ = reader[i]
            assert torch.equal(expected[i]["field"], warm["field"])

    def test_subsampling_matches_uncached(self, zarr_dir_with_attrs):
        from physicsnemo.datapipes.readers.tensorstore_zarr import (
            TensorStoreZarrReader,
        )

        kwargs = dict(
            fields=["field"],
            coordinated_subsampling={"n_points": 8, "target_keys": ["field"]},
        )
        plain = TensorStoreZarrReader(zarr_dir_with_attrs, **kwargs)
        cached = TensorStoreZarrReader(
            zarr_dir_with_attrs, **kwargs, cache=DatasetCache(ram_bytes_limit=2**24)
        )
        gen = torch.Generator()
        gen.manual_seed(3)
        plain.set_generator(gen)
        cached.set_generator(gen)
        for epoch in range(2):
            plain.set_epoch(epoch)
            cached.set_epoch(epoch)
            for i in range(len(plain)):
                a, _ = plain[i]
                b, _ = cached[i]
                assert torch.equal(a["field"], b["field"])


@requires_module("h5py")
class TestHDF5ReaderCache:
    def test_accepts_cache_kwarg(self, tmp_path):
        import h5py

        from physicsnemo.datapipes.readers.hdf5 import HDF5Reader

        path = tmp_path / "data.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("x", data=np.random.randn(5, 8).astype(np.float32))

        reader = HDF5Reader(path, cache=DatasetCache(ram_bytes_limit=2**20))
        data, _ = reader[0]
        assert data["x"].shape == (8,)


@requires_module("pyvista")
class TestVTKReaderCache:
    def test_matches_uncached_and_caches_file_resolution(self, tmp_path):
        import pyvista as pv

        from physicsnemo.datapipes.readers.vtk import VTKReader

        for i in range(2):
            case = tmp_path / f"case_{i}"
            case.mkdir()
            pv.Sphere(radius=1.0 + i).extract_surface().triangulate().save(
                case / "geometry.stl"
            )

        cache = DatasetCache(ram_bytes_limit=2**24)
        plain = VTKReader(tmp_path)
        cached = VTKReader(tmp_path, cache=cache)
        for _ in range(2):  # cold then warm
            for i in range(len(plain)):
                a, _ = plain[i]
                b, _ = cached[i]
                assert torch.equal(a["stl_coordinates"], b["stl_coordinates"])
                assert torch.equal(a["stl_faces"], b["stl_faces"])
        # File resolution was cached (one entry per case dir).
        assert cache.stats()["ram"]["hits"] >= 2


class TestNumpyReaderCache:
    def test_accepts_cache_kwarg(self, tmp_path):
        from physicsnemo.datapipes.readers.numpy import NumpyReader

        np.savez(tmp_path / "s0.npz", x=np.random.randn(8).astype(np.float32))
        np.savez(tmp_path / "s1.npz", x=np.random.randn(8).astype(np.float32))
        reader = NumpyReader(tmp_path, cache=DatasetCache(ram_bytes_limit=2**20))
        data, _ = reader[1]
        assert data["x"].shape == (8,)
