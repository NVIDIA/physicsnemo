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

"""CPU tests for the recipe's domain-parallel plumbing.

Config translation to the readers and sampling over the data-parallel axis
only. The gradient-equivalence check across a domain group lives in
``test_domain_parallel_gradients.py`` (multi-GPU).
"""

from __future__ import annotations

from datasets import (
    _build_directory_samplers,
    _build_manifest_samplers,
    _domain_parallel_from_cfg,
)
from omegaconf import OmegaConf


class _FakeMesh:
    """Stand-in for a 1-D DeviceMesh axis: a size and this rank's position."""

    def __init__(self, size: int, rank: int):
        self._size, self._rank = size, rank

    def size(self, dim: int = 0) -> int:
        return self._size

    def get_local_rank(self, dim: int = 0) -> int:
        return self._rank


# ---------------------------------------------------------------------------
# Config translation
# ---------------------------------------------------------------------------


def test_domain_parallel_policy_passes_through_except_domain_size():
    """Everything in ``domain_parallelism`` but ``domain_size`` reaches the readers
    unchanged, so their own validation sees what the user wrote (typos included)."""
    cfg = OmegaConf.create(
        {
            "domain_parallelism": {
                "domain_size": 4,
                "auto_shard_size": 2048,
                "placements": {"boundaries.stl_geometry": "replicate"},
                "typo_key": 1,
            }
        }
    )
    policy = _domain_parallel_from_cfg(cfg)
    assert policy == {
        "auto_shard_size": 2048,
        "placements": {"boundaries.stl_geometry": "replicate"},
        "typo_key": 1,
    }
    assert isinstance(policy["placements"], dict)  # plain, not DictConfig

    # Unset block: an empty policy (readers apply their defaults).
    assert _domain_parallel_from_cfg(OmegaConf.create({})) == {}


# ---------------------------------------------------------------------------
# Samplers over the data-parallel axis
# ---------------------------------------------------------------------------


def test_directory_samplers_shard_over_data_mesh_only():
    """Every rank of a domain group (same ddp rank) gets the same indices."""
    train_ds = list(range(40))
    val_ds = list(range(12))
    seqs = {}
    for ddp_rank in (0, 1):
        train_sampler, val_sampler = _build_directory_samplers(
            train_ds,
            val_ds,
            use_distributed=True,
            sampler_seed=7,
            data_mesh=_FakeMesh(size=2, rank=ddp_rank),
        )
        train_sampler.set_epoch(0)
        seqs[ddp_rank] = (list(train_sampler), list(val_sampler))
    # Two ddp ranks partition the data ...
    assert not set(seqs[0][0]) & set(seqs[1][0])
    assert sorted(seqs[0][1] + seqs[1][1]) == val_ds
    # ... and are deterministic, so two domain ranks with the same ddp rank
    # (which build the very same sampler) agree.
    again, _ = _build_directory_samplers(
        train_ds,
        val_ds,
        use_distributed=True,
        sampler_seed=7,
        data_mesh=_FakeMesh(size=2, rank=0),
    )
    again.set_epoch(0)
    assert list(again) == seqs[0][0]


def test_manifest_samplers_shard_over_data_mesh_only():
    """Manifest indices are split by ddp rank, not by world rank."""
    train_idx = list(range(30))
    val_idx = list(range(30, 40))
    parts = []
    for ddp_rank in (0, 1, 2):
        train_sampler, val_sampler = _build_manifest_samplers(
            train_idx,
            val_idx,
            dist_manager=None,  # unused when a data_mesh is given
            sampler_seed=3,
            data_mesh=_FakeMesh(size=3, rank=ddp_rank),
        )
        parts.append((set(train_sampler), set(val_sampler)))
    assert set().union(*(p[1] for p in parts)) == set(val_idx)
    for a in range(3):
        for b in range(a + 1, 3):
            assert not parts[a][0] & parts[b][0]
