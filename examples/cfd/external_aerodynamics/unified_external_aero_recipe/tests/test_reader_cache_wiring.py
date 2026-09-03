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

"""Wiring tests for the recipe's `dataloader.cache` reader-cache option.

`build_dataloaders` requires repo-local dataset YAMLs and real data
directories, so these tests exercise the two pieces it composes:
Hydra-instantiating a ``DatasetCache`` from a config block (the
``cfg.dataloader.cache`` path), and `build_dataset` threading a shared
cache instance into the instantiated reader.
"""

from __future__ import annotations

import hydra
import pytest
import torch
from datasets import build_dataset
from omegaconf import OmegaConf

from physicsnemo.datapipes.caching import DatasetCache
from physicsnemo.mesh import DomainMesh
from physicsnemo.mesh.primitives.basic import two_triangles_2d


@pytest.fixture
def volume_data_root(tmp_path):
    """Minimal drivaer-volume-style layout: run_*/domain.pdmsh."""
    root = tmp_path / "data"
    m = two_triangles_2d.load()
    for i in range(2):
        case = root / f"run_{i}"
        case.mkdir(parents=True)
        DomainMesh(
            interior=m.clone(),
            boundaries={"wall": m.clone()},
            global_data={"U_inf": torch.tensor(30.0 + i)},
        ).save(case / "domain.pdmsh")
    return root


def _ds_yaml(root):
    return OmegaConf.create(
        {
            "pipeline": {
                "reader": {
                    "_target_": "${dp:DomainMeshReader}",
                    "path": str(root),
                    "pattern": "run_*/domain.pdmsh",
                },
                "transforms": [],
            },
        }
    )


class TestCacheConfigInstantiation:
    """Hydra instantiation of the ``cfg.dataloader.cache`` config block."""

    def test_hydra_block_builds_dataset_cache(self, tmp_path):
        """The documented cfg.dataloader.cache block instantiates cleanly."""
        cache_cfg = OmegaConf.create(
            {
                "_target_": "${dp:DatasetCache}",
                "ram_bytes_limit": 2**24,
                "disk_dir": str(tmp_path / "pn-cache"),
                "disk_bytes_limit": 2**26,
                "eviction": "largest",
            }
        )
        cache = hydra.utils.instantiate(cache_cfg)
        assert isinstance(cache, DatasetCache)
        assert cache.get_or_load(("k/v1", "x"), lambda: 7) == 7


class TestBuildDatasetCacheWiring:
    """`build_dataset` threads a shared DatasetCache into its reader."""

    def test_reader_receives_shared_cache(self, volume_data_root):
        """Two datasets built with one cache share that exact instance."""
        cache = DatasetCache(ram_bytes_limit=2**24)
        train = build_dataset(_ds_yaml(volume_data_root), device=None, cache=cache)
        val = build_dataset(_ds_yaml(volume_data_root), device=None, cache=cache)
        assert train.reader._cache is cache
        assert val.reader._cache is cache

    def test_cache_default_is_off(self, volume_data_root):
        """Omitting `cache` preserves today's uncached reader exactly."""
        dataset = build_dataset(_ds_yaml(volume_data_root), device=None)
        assert dataset.reader._cache is None

    def test_collect_reader_caches_finds_shared_instance(self, volume_data_root):
        """benchmark_io's stats hook finds one shared cache across loaders."""
        pytest.importorskip("tensorboard")  # train.py imports SummaryWriter
        from train import _collect_reader_caches

        from physicsnemo.datapipes import DataLoader, MultiDataset

        cache = DatasetCache(ram_bytes_limit=2**24)
        ds_a = build_dataset(_ds_yaml(volume_data_root), device=None, cache=cache)
        ds_b = build_dataset(_ds_yaml(volume_data_root), device=None, cache=cache)
        val = build_dataset(_ds_yaml(volume_data_root), device=None, cache=cache)

        train_loader = DataLoader(MultiDataset(ds_a, ds_b), batch_size=1)
        val_loader = DataLoader(val, batch_size=1)
        caches = _collect_reader_caches(train_loader, val_loader)
        assert len(caches) == 1
        assert caches[0] is cache

    def test_cached_samples_match_uncached(self, volume_data_root):
        """Cached reads are tensor-identical to uncached, cold and warm."""
        cache = DatasetCache(ram_bytes_limit=2**24)
        plain = build_dataset(_ds_yaml(volume_data_root), device=None)
        cached = build_dataset(_ds_yaml(volume_data_root), device=None, cache=cache)
        for _ in range(2):  # cold then warm
            for i in range(len(plain.reader)):
                a, _meta = plain.reader[i]
                b, _meta = cached.reader[i]
                assert torch.equal(a.interior.points, b.interior.points)
                assert torch.equal(a.global_data["U_inf"], b.global_data["U_inf"])
        assert cache.stats()["ram"]["hits"] > 0
