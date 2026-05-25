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

"""Tests for AeroJEPA prototype anchor build / load helpers."""

import pytest
import torch

from physicsnemo.experimental.nn.aerojepa import (
    build_context_prototype_anchors,
    build_target_prototype_anchors,
    ensure_context_prototype_anchors,
    ensure_target_prototype_anchors,
    load_context_prototype_anchors,
    load_target_prototype_anchors,
)


class _FakeDataset:
    """Tiny in-memory dataset of dict samples for the build helpers."""

    def __init__(self, n_samples: int = 3, n_surface: int = 60, n_volume: int = 80):
        self.samples = [
            {
                "surface_pos": torch.randn(n_surface, 3),
                "surface_feat": torch.randn(n_surface, 4),
                "volume_pos": torch.randn(n_volume, 3),
                "volume_feat": torch.randn(n_volume, 4),
            }
            for _ in range(n_samples)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def _tiny_cfg(**overrides) -> dict:
    """Smallest config that drives the build pipeline end-to-end."""
    cfg = {
        "prototype_anchor_count": 8,
        "prototype_num_passes": 1,
        "prototype_max_candidate_coords": 256,
        "prototype_kmeans_iters": 3,
        "prototype_assignment_chunk_size": 32,
        "prototype_source_strategy": "fps_cluster",
        "prototype_source_max_point_tokens": 16,
        "prototype_source_cluster_size": 4,
        "prototype_source_knn_chunk_size": 32,
    }
    cfg.update(overrides)
    return cfg


def test_build_context_writes_npz_and_returns_anchors(tmp_path):
    """Build pass writes an ``.npz`` and returns a ``(P, 3)`` float32 tensor."""
    ds = _FakeDataset()
    out = tmp_path / "ctx.npz"
    coords = build_context_prototype_anchors(
        train_dataset=ds, context_cfg=_tiny_cfg(), output_path=out, seed=42
    )
    assert out.exists()
    assert coords.shape == (8, 3)
    assert coords.dtype == torch.float32


def test_build_target_writes_npz_and_returns_anchors(tmp_path):
    """Target build pass mirrors the context one with target-side point loading."""
    ds = _FakeDataset()
    out = tmp_path / "tgt.npz"
    coords = build_target_prototype_anchors(
        train_dataset=ds, target_cfg=_tiny_cfg(), output_path=out, seed=42
    )
    assert out.exists()
    assert coords.shape == (8, 3)
    assert coords.dtype == torch.float32


def test_load_round_trips_built_anchors(tmp_path):
    """Loading the ``.npz`` produced by ``build_*`` returns the same tensor."""
    ds = _FakeDataset()
    out = tmp_path / "ctx.npz"
    coords = build_context_prototype_anchors(
        train_dataset=ds, context_cfg=_tiny_cfg(), output_path=out, seed=42
    )
    loaded_ctx = load_context_prototype_anchors(out)
    loaded_tgt = load_target_prototype_anchors(out)
    assert torch.equal(coords, loaded_ctx)
    assert torch.equal(loaded_ctx, loaded_tgt)


def test_load_rejects_wrong_shape(tmp_path):
    """A file whose coords are not ``(P, 3)`` is rejected with a clear message."""
    import json

    import numpy as np

    bad_path = tmp_path / "bad.npz"
    np.savez_compressed(
        bad_path,
        coords=np.zeros((4, 2), dtype=np.float32),
        metadata_json=np.asarray(json.dumps({})),
    )
    with pytest.raises(ValueError, match=r"coords with shape"):
        load_target_prototype_anchors(bad_path)


def test_ensure_loads_when_file_exists(tmp_path):
    """``ensure_*`` loads the cached file rather than rebuilding."""
    ds = _FakeDataset()
    out = tmp_path / "tgt.npz"
    built = build_target_prototype_anchors(
        train_dataset=ds, target_cfg=_tiny_cfg(), output_path=out, seed=42
    )
    cfg = _tiny_cfg(prototype_anchor_path=str(out))
    loaded = ensure_target_prototype_anchors(train_dataset=ds, target_cfg=cfg, seed=42)
    assert torch.equal(built, loaded)


def test_ensure_rebuilds_when_flag_set(tmp_path):
    """``prototype_rebuild=True`` forces a fresh build even when the file exists."""
    ds = _FakeDataset()
    out = tmp_path / "ctx.npz"
    build_context_prototype_anchors(
        train_dataset=ds, context_cfg=_tiny_cfg(), output_path=out, seed=1
    )
    original_mtime = out.stat().st_mtime_ns
    cfg = _tiny_cfg(prototype_anchor_path=str(out), prototype_rebuild=True)
    ensure_context_prototype_anchors(train_dataset=ds, context_cfg=cfg, seed=2)
    assert out.stat().st_mtime_ns >= original_mtime


def test_ensure_requires_path():
    """No ``prototype_anchor_path`` in cfg is rejected with a helpful error."""
    ds = _FakeDataset()
    with pytest.raises(ValueError, match=r"prototype_anchor_path"):
        ensure_context_prototype_anchors(
            train_dataset=ds, context_cfg=_tiny_cfg(), seed=42
        )
