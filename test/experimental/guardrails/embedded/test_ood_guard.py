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

"""Tests for the embedded OOD guardrail (``OODGuard``) and its config."""

import logging

import pytest
import torch

from physicsnemo.experimental.guardrails.embedded import OODGuard, OODGuardConfig

_GUARD_LOGGER = "physicsnemo.experimental.guardrails.embedded.ood_guard"

_DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA not available"
        ),
        id="cuda",
    ),
]


def _populate(
    guard: OODGuard,
    n_samples: int,
    device: str,
    *,
    global_dim: int | None = None,
    geo_dim: int | None = None,
    batch_size: int = 4,
    seed: int = 0,
) -> None:
    """Feed ``n_samples`` in-distribution samples into ``guard`` via ``collect``."""
    gen = torch.Generator(device=device).manual_seed(seed)
    remaining = n_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        g = (
            torch.randn(b, global_dim, device=device, generator=gen)
            if global_dim is not None
            else None
        )
        z = (
            torch.randn(b, geo_dim, device=device, generator=gen)
            if geo_dim is not None
            else None
        )
        guard.collect(global_embedding=g, geometry_latent=z)
        remaining -= b


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_requires_buffer_size_and_applies_defaults():
    """``OODGuardConfig`` requires ``buffer_size``; other fields have defaults."""
    with pytest.raises(TypeError):
        OODGuardConfig()  # buffer_size is required

    cfg = OODGuardConfig(buffer_size=32)
    assert cfg.buffer_size == 32
    assert cfg.knn_k == 10
    assert cfg.sensitivity == 1.5

    with pytest.raises(TypeError):
        OODGuardConfig(buffer_size=32, unknown_field=1)


# ---------------------------------------------------------------------------
# Construction: surfaces can be independently enabled/disabled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "global_dim,geo_dim",
    [
        (3, 8),  # both enabled
        (3, None),  # global only
        (None, 8),  # geometry only
        (None, None),  # fully disabled (still valid)
    ],
)
def test_construction_buffers(global_dim, geo_dim):
    """Each surface's buffers are allocated iff its dim is set."""
    guard = OODGuard(buffer_size=16, global_dim=global_dim, geometry_embed_dim=geo_dim)

    if global_dim is not None:
        assert guard.global_min.shape == (global_dim,)
        assert guard.global_max.shape == (global_dim,)
        assert torch.isinf(guard.global_min).all()
    else:
        assert guard.global_min is None
        assert guard.global_max is None

    if geo_dim is not None:
        assert guard.geo_embeddings.shape == (16, geo_dim)
        assert guard.geo_ptr.item() == 0
        assert not guard.geo_full.item()
        assert torch.isinf(guard.knn_threshold)
    else:
        assert guard.geo_embeddings is None
        assert guard.geo_ptr is None
        assert guard.geo_full is None
        assert guard.knn_threshold is None


# ---------------------------------------------------------------------------
# Global-parameter bounding box
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", _DEVICES)
def test_global_bounds_collect_and_detect(device, caplog):
    """Collect shrinks bounds; out-of-box batch at check-time warns."""
    guard = OODGuard(buffer_size=8, global_dim=2).to(device)

    # In-distribution: all values in [-1, 1].
    for _ in range(5):
        vals = torch.rand(4, 2, device=device) * 2 - 1  # uniform in [-1, 1)
        guard.collect(global_embedding=vals)

    assert (guard.global_min >= -1.0).all()
    assert (guard.global_max <= 1.0).all()

    # In-distribution check: no warnings emitted.
    with caplog.at_level(logging.WARNING, logger=_GUARD_LOGGER):
        guard.check(global_embedding=torch.zeros(2, 2, device=device))
    assert not any(
        "OOD Guard: global_embedding" in r.getMessage() for r in caplog.records
    )

    # OOD check: value well above the collected max on dim 1.
    caplog.clear()
    ood = torch.tensor([[0.0, 10.0]], device=device)
    with caplog.at_level(logging.WARNING, logger=_GUARD_LOGGER):
        guard.check(global_embedding=ood)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("dim 1" in m and "above training max" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# Geometry FIFO + threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", _DEVICES)
def test_geometry_fifo_wraps_and_latches_full(device):
    """Pointer wraps modulo buffer_size; ``geo_full`` latches on first wrap."""
    buffer_size = 5
    guard = OODGuard(buffer_size=buffer_size, geometry_embed_dim=4).to(device)

    # 3 samples: ptr advances, not yet full.
    _populate(guard, 3, device=device, geo_dim=4, batch_size=1, seed=1)
    assert guard.geo_ptr.item() == 3
    assert not guard.geo_full.item()

    # 4 more samples (total 7) → wrap past capacity.
    _populate(guard, 4, device=device, geo_dim=4, batch_size=1, seed=2)
    assert guard.geo_full.item()
    assert guard.geo_ptr.item() == 7 % buffer_size  # == 2
    assert 0 <= guard.geo_ptr.item() < buffer_size


@pytest.mark.parametrize("device", _DEVICES)
def test_geometry_threshold_computes_and_detects_ood(device, caplog):
    """Threshold becomes finite after calibration; far-OOD queries warn.

    The guard L2-normalises both buffer and query, so distances are bounded
    in ``[0, 2]`` on the unit sphere.  We seed the buffer with a cluster
    biased toward ``+e_0`` so coverage is local; a query pointing along
    ``-e_0`` is then ~2 away — reliably over any reasonable threshold.
    """
    guard = OODGuard(buffer_size=32, geometry_embed_dim=8, knn_k=4, sensitivity=1.5).to(
        device
    )

    # Clustered in-distribution buffer: a tight Gaussian offset along +e_0.
    gen = torch.Generator(device=device).manual_seed(11)
    shift = torch.zeros(8, device=device)
    shift[0] = 3.0
    in_dist = torch.randn(32, 8, device=device, generator=gen) * 0.3 + shift
    for i in range(0, 32, 4):
        guard.collect(geometry_latent=in_dist[i : i + 4])

    # First check triggers lazy threshold calibration.
    guard.check(geometry_latent=in_dist[:1].clone())
    assert torch.isfinite(guard.knn_threshold)

    # Far-OOD query: unit vector pointing along -e_0, antipodal to the cluster.
    z_ood = torch.zeros(1, 8, device=device)
    z_ood[0, 0] = -1.0
    with caplog.at_level(logging.WARNING, logger=_GUARD_LOGGER):
        guard.check(geometry_latent=z_ood)
    assert any(
        "OOD Guard: geometry sample" in r.getMessage()
        and "above threshold" in r.getMessage()
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "global_embedding,geometry_latent,match",
    [
        # Rank-1 global_embedding: batch+channel confusion hazard.
        (torch.zeros(3), None, "at least 2 dims"),
        # Channel-dim mismatch on global_embedding.
        (torch.zeros(2, 5), None, "last-dim mismatch"),
        # Rank-3 geometry_latent (pooling missed at caller).
        (None, torch.zeros(2, 3, 8), "must be rank-2"),
        # Channel mismatch on geometry_latent.
        (None, torch.zeros(2, 5), "channel dim mismatch"),
    ],
)
def test_shape_validation(global_embedding, geometry_latent, match):
    """Bad-shape inputs raise ``ValueError`` with actionable messages."""
    guard = OODGuard(buffer_size=8, global_dim=3, geometry_embed_dim=8)
    with pytest.raises(ValueError, match=match):
        guard.collect(
            global_embedding=global_embedding, geometry_latent=geometry_latent
        )
    with pytest.raises(ValueError, match=match):
        guard.check(global_embedding=global_embedding, geometry_latent=geometry_latent)


# ---------------------------------------------------------------------------
# AMP dtype robustness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [pytest.param(torch.float16, id="fp16"), pytest.param(torch.bfloat16, id="bf16")],
)
def test_amp_inputs_upcast_into_fp32_buffers(dtype):
    """AMP (fp16/bf16) inputs are accepted; buffers stay fp32."""
    guard = OODGuard(buffer_size=4, global_dim=2, geometry_embed_dim=4)
    g = torch.ones(2, 2, dtype=dtype)
    z = torch.ones(2, 4, dtype=dtype)

    guard.collect(global_embedding=g, geometry_latent=z)

    assert guard.global_min.dtype == torch.float32
    assert guard.geo_embeddings.dtype == torch.float32
    # Values propagated despite dtype mismatch at the input boundary.
    assert torch.allclose(guard.global_min, torch.ones(2))
    assert torch.allclose(guard.geo_embeddings[:2], torch.ones(2, 4))


# ---------------------------------------------------------------------------
# Threshold staleness across train → eval → train → eval
# ---------------------------------------------------------------------------


def test_threshold_restales_after_collect_then_check():
    """Running ``collect`` after ``check`` re-marks the threshold stale."""
    guard = OODGuard(buffer_size=16, geometry_embed_dim=4, knn_k=3)
    _populate(guard, n_samples=16, device="cpu", geo_dim=4, seed=7)

    # First check computes the threshold.
    guard.check(geometry_latent=torch.randn(1, 4))
    t0 = guard.knn_threshold.clone()
    assert torch.isfinite(t0)
    assert guard._threshold_stale is False

    # More collection invalidates the stale flag.
    _populate(guard, n_samples=8, device="cpu", geo_dim=4, seed=8)
    assert guard._threshold_stale is True

    # Next check recomputes.
    guard.check(geometry_latent=torch.randn(1, 4))
    assert guard._threshold_stale is False
    # Buffer differs from iteration 1, so threshold should change in general.
    # (Identity not guaranteed; compare for finiteness and plausibility.)
    assert torch.isfinite(guard.knn_threshold)


# ---------------------------------------------------------------------------
# Sensitivity multiplier
# ---------------------------------------------------------------------------


def test_sensitivity_scales_threshold_linearly():
    """Doubling ``sensitivity`` doubles the computed kNN threshold."""
    # Deterministic collection so both guards see the same buffer.
    gen = torch.Generator(device="cpu").manual_seed(42)
    samples = torch.randn(32, 8, generator=gen)

    def _calibrate(sensitivity: float) -> torch.Tensor:
        g = OODGuard(
            buffer_size=32, geometry_embed_dim=8, knn_k=4, sensitivity=sensitivity
        )
        g.collect(geometry_latent=samples.clone())
        g.compute_threshold()
        return g.knn_threshold.clone()

    t1 = _calibrate(1.0)
    t2 = _calibrate(2.0)
    assert torch.isfinite(t1) and torch.isfinite(t2)
    assert torch.allclose(t2, t1 * 2.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Checkpoint round-trip via state_dict
# ---------------------------------------------------------------------------


def test_state_dict_roundtrip_preserves_calibration():
    """``state_dict`` captures all guard state; reload reproduces threshold."""
    src = OODGuard(buffer_size=16, global_dim=2, geometry_embed_dim=4, knn_k=3)
    _populate(src, n_samples=16, device="cpu", global_dim=2, geo_dim=4, seed=5)
    src.compute_threshold()
    threshold_src = src.knn_threshold.clone()

    dst = OODGuard(buffer_size=16, global_dim=2, geometry_embed_dim=4, knn_k=3)
    dst.load_state_dict(src.state_dict())

    # Buffers transferred verbatim; threshold identical.
    assert torch.equal(dst.geo_embeddings, src.geo_embeddings)
    assert dst.geo_full.item() == src.geo_full.item()
    assert dst.geo_ptr.item() == src.geo_ptr.item()
    assert torch.equal(dst.global_min, src.global_min)
    assert torch.equal(dst.global_max, src.global_max)
    assert torch.allclose(dst.knn_threshold, threshold_src)
