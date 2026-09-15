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

import numpy as np
import pytest

pytest.importorskip("zarr")
pytest.importorskip("obstore")

import zarr  # noqa: E402

from physicsnemo.experimental.datapipes.healda.loaders.zarr_loader import (  # noqa: E402
    NO_LEVEL,
    ZarrLoader,
    _open_remote_store,
)


def _make_store(path, times=4, levels=(1000, 850, 500), ny=3, nx=4):
    """Write a small ERA5-like zarr store and return its consolidated root."""
    root = zarr.open_group(str(path), mode="w")
    t = root.create_array("time", shape=(times,), dtype="datetime64[ns]")
    t[:] = np.datetime64("2000-01-01") + np.arange(times) * np.timedelta64(1, "h")
    lev = root.create_array("level", shape=(len(levels),), dtype="i8")
    lev[:] = np.asarray(levels)
    z = root.create_array("z", shape=(times, len(levels), ny, nx), dtype="f4")
    z[:] = np.arange(z.size, dtype="f4").reshape(z.shape)
    sp = root.create_array("sp", shape=(times, ny, nx), dtype="f4")
    sp[:] = np.arange(sp.size, dtype="f4").reshape(sp.shape)
    zarr.consolidate_metadata(root.store)
    return root


def test_local_load(tmp_path):
    """Baseline: local paths keep working through the plain zarr path."""
    store_path = tmp_path / "data.zarr"
    _make_store(store_path)
    loader = ZarrLoader(
        path=str(store_path),
        variables_3d=["z"],
        variables_2d=["sp"],
        levels=[1000, 500],
        level_coord_name="level",
    )
    out = zarr.core.sync.sync(loader.sel_time(loader.times[:2]))
    assert ("z", 1000) in out and ("z", 500) in out and ("sp", NO_LEVEL) in out
    assert out[("z", 1000)].shape == (2, 3, 4)
    assert out[("sp", NO_LEVEL)].shape == (2, 3, 4)


def test_remote_load_via_obstore(tmp_path):
    """A file:// URL is non-local, exercising the obstore-backed store path."""
    store_path = tmp_path / "data.zarr"
    _make_store(store_path)
    loader = ZarrLoader(
        path=f"file://{store_path}",
        variables_3d=["z"],
        variables_2d=["sp"],
        levels=[850],
        level_coord_name="level",
    )
    # The remote path must have been swapped for an obstore-backed zarr store
    assert isinstance(loader.group.store_path.store, zarr.storage.ObjectStore)
    out = zarr.core.sync.sync(loader.sel_time(loader.times[1:3]))
    np.testing.assert_array_equal(
        out[("z", 850)],
        np.arange(4 * 3 * 3 * 4, dtype="f4").reshape(4, 3, 3, 4)[1:3, 1],
    )


def test_storage_options_translation(tmp_path, monkeypatch):
    """fsspec-style keys are translated to obstore config names."""
    captured = {}
    import physicsnemo.experimental.datapipes.healda.loaders.zarr_loader as zl

    real_from_url = zl._obstore_store.from_url

    def spy_from_url(url, **kwargs):
        captured.update(kwargs)
        return real_from_url(url)

    monkeypatch.setattr(zl._obstore_store, "from_url", spy_from_url, raising=False)
    store_path = tmp_path / "data.zarr"
    _make_store(store_path)
    ZarrLoader(
        path=f"file://{store_path}",
        variables_3d=[],
        variables_2d=["sp"],
        levels=[],
        storage_options={"anon": True, "endpoint_url": "https://ex.io", "foo": 1},
    )
    assert captured == {"skip_signature": True, "endpoint": "https://ex.io", "foo": 1}


def test_open_remote_store_is_read_only(tmp_path):
    store_path = tmp_path / "data.zarr"
    _make_store(store_path)
    store = _open_remote_store(f"file://{store_path}", None)
    assert store.read_only
