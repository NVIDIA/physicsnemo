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

"""Tests for DatasetCache: blob/tree entries, tiers, eviction, concurrency."""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.datapipes.caching import (
    DatasetCache,
    decode_blob,
    encode_blob,
    estimate_resident_size,
)


class TestBlobCodec:
    @pytest.mark.parametrize(
        "value",
        [
            None,
            True,
            42,
            3.5,
            "hello",
            b"raw-bytes",
            [1, "a", None, [2.0]],
            (1, 2),
            {"nested": {"x": 1}},
        ],
        ids=type,
    )
    def test_json_like_roundtrip(self, value):
        assert decode_blob(encode_blob(value, kind="t/v1", identity="i")) == value

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float64, torch.int64, torch.bfloat16, torch.bool]
    )
    def test_tensor_roundtrip(self, dtype):
        t = (torch.rand(3, 4) * 10).to(dtype)
        out = decode_blob(encode_blob(t, kind="t/v1", identity="i"))
        assert out.dtype == dtype
        assert torch.equal(out, t)

    def test_scalar_and_empty_tensor_roundtrip(self):
        for t in [torch.tensor(1.0e6), torch.empty(0, 3)]:
            out = decode_blob(encode_blob(t, kind="t/v1", identity="i"))
            assert out.shape == t.shape
            assert torch.equal(out, t)

    def test_tensordict_roundtrip(self):
        td = TensorDict(
            {"U_inf": torch.tensor(30.0), "rho": torch.tensor(1.2)}, batch_size=[]
        )
        out = decode_blob(encode_blob(td, kind="t/v1", identity="i"))
        assert isinstance(out, TensorDict)
        assert torch.equal(out["U_inf"], td["U_inf"])
        assert torch.equal(out["rho"], td["rho"])

    def test_no_pickle_for_arbitrary_objects(self):
        class Foo:
            pass

        with pytest.raises(TypeError):
            encode_blob(Foo(), kind="t/v1", identity="i")

    def test_corrupt_data_raises(self):
        with pytest.raises(ValueError):
            decode_blob(b"not a cache blob")


class TestBlobEntries:
    def test_ram_hit_runs_loader_once(self):
        cache = DatasetCache(ram_bytes_limit=2**20)
        calls = []

        def loader():
            calls.append(1)
            return {"x": 1}

        a = cache.get_or_load(("k/v1", "id"), loader)
        b = cache.get_or_load(("k/v1", "id"), loader)
        assert a == b == {"x": 1}
        assert len(calls) == 1
        assert cache.stats()["ram"]["hits"] == 1

    def test_distinct_keys_are_distinct_entries(self):
        cache = DatasetCache(ram_bytes_limit=2**20)
        a = cache.get_or_load(("k/v1", "a"), lambda: 1)
        b = cache.get_or_load(("k/v1", "b"), lambda: 2)
        c = cache.get_or_load(("other/v1", "a"), lambda: 3)
        assert (a, b, c) == (1, 2, 3)

    def test_disk_roundtrip_across_instances(self, tmp_path):
        """Disk entries persist: a fresh cache (fresh process stand-in) hits."""
        value = {"Re": torch.tensor(1.0e6)}
        cache1 = DatasetCache(disk_dir=tmp_path / "c", ram_bytes_limit=None)
        cache1.get_or_load(("k/v1", "id"), lambda: value)

        cache2 = DatasetCache(disk_dir=tmp_path / "c", ram_bytes_limit=None)
        out = cache2.get_or_load(
            ("k/v1", "id"), lambda: pytest.fail("loader must not run")
        )
        assert torch.equal(out["Re"], value["Re"])

    def test_disk_hit_promotes_to_ram(self, tmp_path):
        cache1 = DatasetCache(disk_dir=tmp_path / "c", ram_bytes_limit=None)
        cache1.get_or_load(("k/v1", "id"), lambda: [1, 2, 3])

        cache2 = DatasetCache(disk_dir=tmp_path / "c", ram_bytes_limit=2**20)
        cache2.get_or_load(("k/v1", "id"), lambda: pytest.fail("no loader"))
        assert cache2.stats()["ram"]["entries"] == 1
        cache2.get_or_load(("k/v1", "id"), lambda: pytest.fail("no loader"))
        assert cache2.stats()["ram"]["hits"] == 1

    def test_max_item_bytes_bypasses_cache(self, tmp_path):
        cache = DatasetCache(
            ram_bytes_limit=2**30, disk_dir=tmp_path / "c", max_item_bytes=1024
        )
        big = torch.zeros(1_000_000)
        calls = []

        def loader():
            calls.append(1)
            return big

        cache.get_or_load(("k/v1", "big"), loader)
        cache.get_or_load(("k/v1", "big"), loader)
        assert len(calls) == 2  # never admitted, loads every time
        assert cache.stats()["ram"]["entries"] == 0
        assert cache.stats()["disk"]["entries"] == 0

    def test_invalidate_and_clear(self, tmp_path):
        cache = DatasetCache(ram_bytes_limit=2**20, disk_dir=tmp_path / "c")
        cache.get_or_load(("a/v1", "x"), lambda: 1)
        cache.get_or_load(("b/v1", "y"), lambda: 2)

        cache.invalidate(("a/v1", "x"))
        assert cache.get_or_load(("a/v1", "x"), lambda: 10) == 10

        cache.clear(kind="b/v1")
        assert cache.get_or_load(("b/v1", "y"), lambda: 20) == 20

        cache.clear()
        assert cache.stats()["ram"]["entries"] == 0
        assert cache.stats()["disk"]["entries"] == 0

    def test_single_flight(self):
        """Concurrent gets for one key run the loader exactly once."""
        cache = DatasetCache(ram_bytes_limit=2**20)
        calls = []

        def slow_loader():
            calls.append(1)
            time.sleep(0.05)
            return "value"

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(
                    lambda _: cache.get_or_load(("k/v1", "id"), slow_loader), range(8)
                )
            )
        assert results == ["value"] * 8
        assert len(calls) == 1

    def test_thread_safety_many_keys(self):
        cache = DatasetCache(ram_bytes_limit=2**20)

        def work(i):
            key = ("k/v1", f"id-{i % 7}")
            return cache.get_or_load(key, lambda: i % 7)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(work, range(200)))
        assert all(r == i % 7 for i, r in enumerate(results))


class TestEviction:
    def _fill(self, cache, sizes):
        for name, size in sizes.items():
            cache.get_or_load(
                ("k/v1", name), lambda size=size: bytes(size), size_hint=size
            )

    def _resident(self, cache, names):
        found = []
        for name in names:
            sentinel = object()
            v = cache.get_or_load(("k/v1", name), lambda: sentinel)
            if v is not sentinel:
                found.append(name)
            cache.invalidate(("k/v1", name)) if v is sentinel else None
        return found

    def test_largest_evicted_first(self):
        cache = DatasetCache(ram_bytes_limit=1000, eviction="largest")
        self._fill(cache, {"small-1": 100, "small-2": 100, "big": 700, "small-3": 200})
        # 1100 > 1000: the largest entry ("big") goes first.
        assert self._resident(cache, ["small-1", "small-2", "big", "small-3"]) == [
            "small-1",
            "small-2",
            "small-3",
        ]

    def test_fifo_evicts_oldest_first(self):
        cache = DatasetCache(ram_bytes_limit=1000, eviction="fifo")
        self._fill(cache, {"first": 400, "second": 400, "third": 400})
        assert self._resident(cache, ["first", "second", "third"]) == [
            "second",
            "third",
        ]

    def test_lru_evicts_least_recent_first(self):
        cache = DatasetCache(ram_bytes_limit=1000, eviction="lru")
        self._fill(cache, {"a": 400, "b": 400})
        cache.get_or_load(("k/v1", "a"), lambda: pytest.fail("no loader"))  # touch a
        self._fill(cache, {"c": 400})  # evicts b (least recently used)
        assert self._resident(cache, ["a", "b", "c"]) == ["a", "c"]

    def test_disk_eviction_deletes_files(self, tmp_path):
        cache = DatasetCache(
            ram_bytes_limit=None, disk_dir=tmp_path / "c", disk_bytes_limit=2048
        )
        for i in range(8):
            cache.get_or_load(("k/v1", f"id-{i}"), lambda: bytes(512))
        assert cache.stats()["disk"]["bytes"] <= 2048
        assert cache.stats()["disk"]["evictions"] > 0

    def test_unknown_policy_rejected(self):
        with pytest.raises(ValueError):
            DatasetCache(eviction="nope")


def _make_tree(root, n_small=3, large_bytes=256 * 1024):
    """Directory tree with small metadata files and one large payload."""
    (root / "sub").mkdir(parents=True)
    for i in range(n_small):
        (root / "sub" / f"meta_{i}.json").write_text('{"k": %d}' % i)
    (root / "small.bin").write_bytes(b"x" * 100)
    (root / "large.bin").write_bytes(b"y" * large_bytes)
    return root


def _tree_loader(path):
    """Stock stand-in loader: reads every file, returns name -> bytes."""
    return {
        str(p.relative_to(path)): p.read_bytes()
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


class TestTreeEntries:
    def test_mirror_copies_small_and_symlinks_large(self, tmp_path):
        src = _make_tree(tmp_path / "src")
        cache = DatasetCache(
            ram_bytes_limit=None, disk_dir=tmp_path / "c", small_file_bytes=1024
        )
        expected = _tree_loader(src)
        out = cache.get_or_load(("tree/v1", str(src)), _tree_loader, src=src)
        assert out == expected

        mirrors = list((tmp_path / "c").rglob("*.tree"))
        assert len(mirrors) == 1
        mirror = mirrors[0]
        assert not (mirror / "small.bin").is_symlink()
        assert not (mirror / "sub" / "meta_0.json").is_symlink()
        assert (mirror / "large.bin").is_symlink()
        assert (mirror / "large.bin").resolve() == (src / "large.bin").resolve()

    def test_loader_receives_mirror_when_disk_configured(self, tmp_path):
        src = _make_tree(tmp_path / "src")
        cache = DatasetCache(ram_bytes_limit=None, disk_dir=tmp_path / "c")
        seen = []

        def loader(path):
            seen.append(path)
            return _tree_loader(path)

        cache.get_or_load(("tree/v1", str(src)), loader, src=src)
        assert seen[0] != src
        assert str(seen[0]).startswith(str(tmp_path / "c"))

    @pytest.mark.parametrize(
        "ram,disk", [(True, True), (True, False), (False, True), (False, False)]
    )
    def test_all_tier_combinations(self, tmp_path, ram, disk):
        src = _make_tree(tmp_path / "src")
        cache = DatasetCache(
            ram_bytes_limit=2**20 if ram else None,
            disk_dir=(tmp_path / "c") if disk else None,
        )
        expected = _tree_loader(src)
        for _ in range(3):
            out = cache.get_or_load(("tree/v1", str(src)), _tree_loader, src=src)
            assert out == expected

    def test_ram_hit_skips_loader_and_filesystem(self, tmp_path):
        src = _make_tree(tmp_path / "src")
        cache = DatasetCache(ram_bytes_limit=2**20)
        calls = []

        def loader(path):
            calls.append(path)
            return _tree_loader(path)

        cache.get_or_load(("tree/v1", str(src)), loader, src=src)
        cache.get_or_load(("tree/v1", str(src)), loader, src=src)
        assert len(calls) == 1
        assert cache.stats()["ram"]["hits"] == 1

    def test_ram_hit_returns_structural_copy(self, tmp_path):
        src = _make_tree(tmp_path / "src")
        cache = DatasetCache(ram_bytes_limit=2**20)
        a = cache.get_or_load(("tree/v1", str(src)), _tree_loader, src=src)
        a["injected"] = b"mutation"  # dict.copy() protects structure
        b = cache.get_or_load(("tree/v1", str(src)), _tree_loader, src=src)
        assert "injected" not in b

    def test_fallback_to_source_on_bad_mirror(self, tmp_path, caplog):
        src = _make_tree(tmp_path / "src")
        cache = DatasetCache(ram_bytes_limit=None, disk_dir=tmp_path / "c")
        expected = _tree_loader(src)
        cache.get_or_load(("tree/v1", str(src)), _tree_loader, src=src)

        # Corrupt the mirror: a loader that requires a file that's now gone.
        mirror = next((tmp_path / "c").rglob("*.tree"))
        (mirror / "small.bin").unlink()

        def strict_loader(path):
            out = _tree_loader(path)
            if "small.bin" not in out:
                raise FileNotFoundError("small.bin missing")
            return out

        out = cache.get_or_load(("tree/v1", str(src)), strict_loader, src=src)
        assert out == expected  # fell back to source
        # Entry was invalidated; the next read re-mirrors cleanly.
        out = cache.get_or_load(("tree/v1", str(src)), strict_loader, src=src)
        assert out == expected

    def test_shared_disk_dir_between_instances(self, tmp_path):
        """Two caches on one directory (multi-rank stand-in)."""
        src = _make_tree(tmp_path / "src")
        caches = [
            DatasetCache(ram_bytes_limit=None, disk_dir=tmp_path / "c")
            for _ in range(2)
        ]
        expected = _tree_loader(src)

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(
                pool.map(
                    lambda i: caches[i % 2].get_or_load(
                        ("tree/v1", str(src)), _tree_loader, src=src
                    ),
                    range(8),
                )
            )
        assert all(r == expected for r in results)
        assert len(list((tmp_path / "c").rglob("*.tree"))) == 1

    def test_validate_mtime_invalidates_stale_mirror(self, tmp_path):
        src = _make_tree(tmp_path / "src")
        warm = DatasetCache(ram_bytes_limit=None, disk_dir=tmp_path / "c")
        warm.get_or_load(("tree/v1", str(src)), _tree_loader, src=src)

        time.sleep(0.02)
        (src / "small.bin").write_bytes(b"z" * 100)
        far_future = time.time() + 3600
        import os

        os.utime(src, (far_future, far_future))

        checked = DatasetCache(
            ram_bytes_limit=None, disk_dir=tmp_path / "c", validate="mtime"
        )
        out = checked.get_or_load(("tree/v1", str(src)), _tree_loader, src=src)
        assert out["small.bin"] == b"z" * 100


class TestSizeEstimation:
    def test_plain_tensor_counts_nbytes(self):
        t = torch.zeros(1000, dtype=torch.float32)
        assert estimate_resident_size(t) >= 4000

    def test_memmap_leaves_do_not_count(self, tmp_path):
        td = TensorDict(
            {"big": torch.zeros(100_000), "small": torch.tensor(1.0)}, batch_size=[]
        )
        td.memmap_(str(tmp_path / "td"))
        loaded = TensorDict.load_memmap(str(tmp_path / "td"))
        size = estimate_resident_size(loaded, small_file_bytes=64 * 1024)
        assert size < 10_000  # 400 KB memmap leaf not counted as resident
