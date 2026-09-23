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

import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.datapipes import caching
from physicsnemo.datapipes.caching import (
    _PER_OBJECT_OVERHEAD,
    DatasetCache,
    _dir_size,
    cached_or_load,
    decode_blob,
    encode_blob,
    estimate_resident_size,
)


class TestBlobCodec:
    @pytest.mark.parametrize(
        "value",
        [None, True, 42, 3.5, "hello", [1, "a", None, [2.0]], {"nested": {"x": 1}}],
        ids=type,
    )
    def test_json_roundtrip(self, value):
        assert decode_blob(encode_blob(value)) == value

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float64, torch.bfloat16, torch.int64, torch.bool]
    )
    def test_small_tensor_roundtrip(self, dtype):
        t = (torch.rand(1, 3) * 10).to(dtype)
        out = decode_blob(encode_blob({"U_inf": t}))["U_inf"]
        assert out.dtype == dtype and out.shape == t.shape
        assert torch.equal(out, t)

    def test_scalar_tensor_roundtrip(self):
        t = torch.tensor(1.0e6)
        out = decode_blob(encode_blob(t))
        assert out.shape == () and torch.equal(out, t)

    def test_tensordict_roundtrip(self):
        td = TensorDict(
            {"Re": torch.tensor(1.0e6), "U": torch.tensor([[1.0, 0.0, 0.0]])},
            batch_size=[],
        )
        out = decode_blob(encode_blob(td))
        assert isinstance(out, TensorDict)
        assert torch.equal(out["Re"], td["Re"]) and torch.equal(out["U"], td["U"])

    def test_bulk_tensor_refused(self):
        with pytest.raises(TypeError, match="bulk"):
            encode_blob(torch.zeros(100_000))

    def test_arbitrary_object_refused(self):
        class Foo:
            pass

        with pytest.raises(TypeError):
            encode_blob(Foo())

    def test_plain_dict_that_looks_like_a_tag_is_left_alone(self):
        # Two keys, so it is not a tagged node.
        v = {"__tensor__": 1, "other": 2}
        assert decode_blob(encode_blob(v)) == v

    def test_corrupt_data_raises(self):
        with pytest.raises(ValueError):
            decode_blob(b"not json")

    def test_non_json_values_are_ram_only(self, tmp_path, caplog):
        cache = DatasetCache(ram_bytes_limit=2**20, disk_dir=tmp_path / "c")
        calls = []

        def loader():
            calls.append(1)
            return torch.zeros(100_000)  # bulk: over the JSON tensor cap

        with caplog.at_level("WARNING"):
            a = cache.get_or_load(("t/v1", "x"), loader)
            b = cache.get_or_load(("t/v1", "x"), loader)
            cache.get_or_load(("t/v1", "y"), loader)
        assert torch.equal(a, b) and len(calls) == 2  # RAM hit for the repeat
        assert cache.stats()["disk"]["entries"] == 0
        assert list((tmp_path / "c").rglob("*.json")) == []
        assert caplog.text.count("RAM tier only") == 1  # warned once per kind

    def test_small_tensor_blob_persists_to_disk(self, tmp_path):
        value = {"Re": torch.tensor(1.0e6), "U_inf": torch.tensor([[30.0, 0.0, 0.0]])}
        DatasetCache(ram_bytes_limit=None, disk_dir=tmp_path / "c").get_or_load(
            ("global/v1", "run_1"), lambda: value
        )
        later = DatasetCache(ram_bytes_limit=None, disk_dir=tmp_path / "c")
        out = later.get_or_load(("global/v1", "run_1"), lambda: pytest.fail("miss"))
        assert torch.equal(out["Re"], value["Re"])
        assert torch.equal(out["U_inf"], value["U_inf"])


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
        value = {"Re": 1.0e6, "keys": ["p", "U"], "nested": {"n": None}}
        cache1 = DatasetCache(disk_dir=tmp_path / "c", ram_bytes_limit=None)
        cache1.get_or_load(("k/v1", "id"), lambda: value)

        cache2 = DatasetCache(disk_dir=tmp_path / "c", ram_bytes_limit=None)
        out = cache2.get_or_load(
            ("k/v1", "id"), lambda: pytest.fail("loader must not run")
        )
        assert out == value

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
            # RAM sizing of a bytes value is len + per-object overhead; make
            # the accounted size exactly *size* so the arithmetic below is exact.
            payload = bytes(size - _PER_OBJECT_OVERHEAD)
            cache.get_or_load(("k/v1", name), lambda p=payload: p)

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
        cache = DatasetCache(ram_bytes_limit=10_000, eviction="largest")
        self._fill(
            cache, {"small-1": 1000, "small-2": 1000, "big": 7000, "small-3": 2000}
        )
        # 11000 > 10000: the largest entry ("big") goes first.
        assert self._resident(cache, ["small-1", "small-2", "big", "small-3"]) == [
            "small-1",
            "small-2",
            "small-3",
        ]

    def test_fifo_evicts_oldest_first(self):
        cache = DatasetCache(ram_bytes_limit=10_000, eviction="fifo")
        self._fill(cache, {"first": 4000, "second": 4000, "third": 4000})
        assert self._resident(cache, ["first", "second", "third"]) == [
            "second",
            "third",
        ]

    def test_lru_evicts_least_recent_first(self):
        cache = DatasetCache(ram_bytes_limit=10_000, eviction="lru")
        self._fill(cache, {"a": 4000, "b": 4000})
        cache.get_or_load(("k/v1", "a"), lambda: pytest.fail("no loader"))  # touch a
        self._fill(cache, {"c": 4000})  # evicts b (least recently used)
        assert self._resident(cache, ["a", "b", "c"]) == ["a", "c"]

    def test_disk_eviction_deletes_files(self, tmp_path):
        cache = DatasetCache(
            ram_bytes_limit=None, disk_dir=tmp_path / "c", disk_bytes_limit=2048
        )
        for i in range(8):
            cache.get_or_load(("k/v1", f"id-{i}"), lambda: "x" * 512)
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

    def test_memmap_leaf_bytes_do_not_count_but_mappings_do(self, tmp_path):
        td = TensorDict(
            {"big": torch.zeros(100_000), "small": torch.tensor(1.0)}, batch_size=[]
        )
        td.memmap_(str(tmp_path / "td"))
        loaded = TensorDict.load_memmap(str(tmp_path / "td"))
        size = estimate_resident_size(loaded, small_file_bytes=64 * 1024)
        # The 400 KB leaf's bytes are not resident, but each of the two
        # leaves is one mmap region and is charged the mapping cost.
        assert 2 * caching._MAPPING_CHARGE <= size < 400_000

    def test_ram_budget_bounds_mapping_count(self, tmp_path):
        """Many small memmap trees must evict, not accumulate mmap regions."""
        for i in range(8):
            TensorDict({"x": torch.zeros(4)}, batch_size=[]).memmap_(
                str(tmp_path / f"td{i}")
            )
        cache = DatasetCache(ram_bytes_limit=4 * caching._MAPPING_CHARGE)
        for i in range(8):
            p = tmp_path / f"td{i}"
            cache.get_or_load(("td/v1", str(p)), TensorDict.load_memmap, src=p)
        assert cache.stats()["ram"]["entries"] <= 4
        assert cache.stats()["ram"]["evictions"] > 0


class TestMultiProcessSharing:
    """Two instances on one disk_dir stand in for two ranks on a node."""

    def test_shared_dir_adopts_other_process_entries(self, tmp_path):
        src = _make_tree(tmp_path / "src")
        writer = DatasetCache(ram_bytes_limit=None, disk_dir=tmp_path / "c")
        reader = DatasetCache(ram_bytes_limit=None, disk_dir=tmp_path / "c")
        assert reader.stats()["disk"]["entries"] == 0

        writer.get_or_load(("tree/v1", str(src)), _tree_loader, src=src)
        # The reader never wrote or scanned this entry, but a hit adopts it
        # into its own accounting so the shared byte limit is enforced.
        reader.get_or_load(("tree/v1", str(src)), _tree_loader, src=src)
        disk = reader.stats()["disk"]
        assert disk["hits"] == 1
        assert disk["entries"] == 1
        assert disk["bytes"] == writer.stats()["disk"]["bytes"] > 0

    def test_scan_uses_sidecar_size(self, tmp_path):
        src = _make_tree(tmp_path / "src")
        cache = DatasetCache(ram_bytes_limit=None, disk_dir=tmp_path / "c")
        cache.get_or_load(("tree/v1", str(src)), _tree_loader, src=src)
        mirror = next((tmp_path / "c").rglob("*.tree"))
        sidecar = mirror.with_suffix(".size")
        assert int(sidecar.read_text()) == _dir_size(mirror)

        # A later process adopts persisted entries at startup from the
        # sidecar, without walking the mirror.
        later = DatasetCache(ram_bytes_limit=None, disk_dir=tmp_path / "c")
        assert later.stats()["disk"]["entries"] == 1
        assert later.stats()["disk"]["bytes"] == _dir_size(mirror)

    def test_mirror_deleted_mid_read_falls_back(self, tmp_path, caplog):
        """Another rank evicting our mirror between lookup and load."""
        src = _make_tree(tmp_path / "src")
        cache = DatasetCache(ram_bytes_limit=None, disk_dir=tmp_path / "c")
        expected = _tree_loader(src)
        cache.get_or_load(("tree/v1", str(src)), _tree_loader, src=src)
        cache_root = tmp_path / "c"

        def racing_loader(path):
            if cache_root in path.parents:  # reading the mirror: yank it
                shutil.rmtree(path)
                raise FileNotFoundError(path)
            return _tree_loader(path)

        with caplog.at_level("WARNING"):
            out = cache.get_or_load(("tree/v1", str(src)), racing_loader, src=src)
        assert out == expected
        assert "falling back to source" in caplog.text
        # The next read re-mirrors cleanly.
        cache.get_or_load(("tree/v1", str(src)), _tree_loader, src=src)
        assert len(list(cache_root.rglob("*.tree"))) == 1


class TestIdentity:
    def test_identity_is_abspath_not_realpath(self, tmp_path, monkeypatch):
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        cache = DatasetCache(ram_bytes_limit=2**20)
        calls = []

        def loader():
            calls.append(1)
            return {"k": 1}

        cached_or_load(cache, "k/v1", real, loader)
        cached_or_load(cache, "k/v1", link, loader)
        assert len(calls) == 2  # symlink and target are distinct entries

        # Relative paths normalize to the same absolute identity (no I/O).
        monkeypatch.chdir(tmp_path)
        cached_or_load(cache, "k/v1", "real", loader)
        assert len(calls) == 2


class TestDiskDirChecks:
    def test_network_filesystem_warns_but_works(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(caching, "_filesystem_type", lambda p: "lustre")
        with caplog.at_level("WARNING"):
            cache = DatasetCache(disk_dir=tmp_path / "c")
        assert cache._disk is not None
        assert "lustre" in caplog.text and "node-local" in caplog.text

    def test_tmpfs_warns(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(caching, "_filesystem_type", lambda p: "tmpfs")
        with caplog.at_level("WARNING"):
            DatasetCache(disk_dir=tmp_path / "c")
        assert "RAM-backed" in caplog.text

    def test_budget_clamped_to_free_space(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(caching, "_filesystem_type", lambda p: "ext4")
        usage = shutil.disk_usage(tmp_path)
        monkeypatch.setattr(
            caching.shutil, "disk_usage", lambda p: usage._replace(free=10 * 2**30)
        )
        with caplog.at_level("WARNING"):
            cache = DatasetCache(disk_dir=tmp_path / "c", disk_bytes_limit=200 * 2**30)
        assert cache._disk.limit_bytes == int(10 * 2**30 * 0.8)
        assert "exceeds" in caplog.text

    def test_detection_failure_is_permissive(self, tmp_path, monkeypatch):
        monkeypatch.setattr(caching, "_filesystem_type", lambda p: None)
        cache = DatasetCache(disk_dir=tmp_path / "c", disk_bytes_limit=1024)
        assert cache._disk.limit_bytes == 1024
