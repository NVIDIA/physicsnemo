# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Figure 2 + Figure 3 validation diagnostics in
``utils/metrics.py``.

The lightweight torch kernels here mirror the coord-aware
``earth2studio.statistics.{rmse, spread_skill_ratio, rank_histogram, crps}``
family; canonical numerics are cross-checked against hand computation
and the paper's eq. (4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from utils.metrics import (
    crps_per_variable_per_lead,
    derived_variable_crps,
    ensemble_rmse_per_variable_per_lead,
    flag_bad_seeds,
    plot_metric_vs_lead,
    plot_power_spectra,
    plot_rank_histograms,
    power_spectra_per_variable,
    rank_histogram_per_variable,
    save_summary,
    spread_skill_per_variable_per_lead,
)


def _fixture(seed: int = 0, B: int = 2, K: int = 3, M: int = 4, C: int = 5):
    torch.manual_seed(seed)
    ensemble = torch.randn(B, K, M, C, 6, 6, dtype=torch.float64)
    target = torch.randn(B, K, C, 6, 6, dtype=torch.float64)
    return ensemble, target


def test_crps_per_variable_per_lead_shape_and_finite():
    ens, tgt = _fixture()
    out = crps_per_variable_per_lead(ens, tgt)
    assert out.shape == (ens.shape[1], ens.shape[3])
    assert np.all(np.isfinite(out))


def test_ensemble_rmse_matches_hand():
    # With a single-member ensemble (M=1) RMSE reduces to the deterministic
    # RMSE of that single prediction against the target.
    torch.manual_seed(0)
    B, K, C, H, W = 1, 1, 2, 3, 3
    pred = torch.randn(B, K, 1, C, H, W, dtype=torch.float64)
    tgt = torch.randn(B, K, C, H, W, dtype=torch.float64)
    out = ensemble_rmse_per_variable_per_lead(pred, tgt)
    expected = ((pred[:, :, 0] - tgt) ** 2).mean(dim=(0, -2, -1)).sqrt().numpy()
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_spread_skill_degenerate_ensemble_is_zero_over_skill():
    # Identical members → variance is 0 → spread=0, ratio=0.
    torch.manual_seed(0)
    B, K, M, C, H, W = 1, 2, 3, 2, 4, 4
    base = torch.randn(B, K, C, H, W, dtype=torch.float64)
    pred = base.unsqueeze(2).expand(B, K, M, C, H, W)
    tgt = base + 1.0  # non-zero skill
    spread, skill, ratio = spread_skill_per_variable_per_lead(pred, tgt)
    np.testing.assert_allclose(spread, np.zeros_like(spread), atol=1e-12)
    np.testing.assert_allclose(ratio, np.zeros_like(ratio), atol=1e-12)
    # Skill is nonzero when ensemble mean misses target.
    assert np.all(skill > 0)


def test_rank_histogram_sums_to_total_positions():
    ens, tgt = _fixture()
    hist = rank_histogram_per_variable(ens, tgt)
    # Sum per channel == B * K * H * W.
    B, K, _, C, H, W = ens.shape
    assert hist.shape == (C, ens.shape[2] + 1)
    expected_total = B * K * H * W
    np.testing.assert_array_equal(hist.sum(axis=1), np.full(C, expected_total))


def test_derived_variable_crps_returns_wspd_and_dz_when_present():
    # Fabricate a tensor with a variable ordering that contains all the
    # components needed by both derived quantities.
    variables = ["u10m", "v10m", "z300", "z500", "t2m"]
    ens, tgt = _fixture(C=5)
    derived = derived_variable_crps(ens, tgt, variables)
    assert set(derived) == {"wspd10m", "z300_minus_z500"}
    for arr in derived.values():
        assert arr.shape == (ens.shape[1],)
        assert np.all(np.isfinite(arr))


def test_derived_variable_crps_skips_missing_components():
    variables = ["t2m", "msl"]
    ens, tgt = _fixture(C=2)
    derived = derived_variable_crps(ens, tgt, variables)
    assert derived == {}


def test_power_spectra_shape():
    torch.manual_seed(0)
    B, K, C, H, W = 1, 2, 3, 32, 32
    ens_mean = torch.randn(B, K, C, H, W, dtype=torch.float32)
    tgt = torch.randn(B, K, C, H, W, dtype=torch.float32)
    k, ens_pow, tgt_pow = power_spectra_per_variable(ens_mean, tgt)
    assert k.ndim == 1
    assert ens_pow.shape == (K, C, k.size)
    assert tgt_pow.shape == (K, C, k.size)
    assert np.all(ens_pow >= 0) and np.all(tgt_pow >= 0)


def test_flag_bad_seeds_picks_amplified_tail_only():
    # 3 seeds, 2 leads, 2 channels, 8 wavenumber bins. Truth has a flat
    # low-power tail (top 20% = last 2 bins).
    K, C, B = 2, 2, 8
    truth = np.ones((K, C, B), dtype=np.float64)
    truth[..., -2:] = 0.1  # small truth tail

    fore = np.tile(truth, (3, 1, 1, 1)).astype(np.float64)  # (S, K, C, B)
    # Seed 1 amplifies the tail of one channel / one lead 10x — above 3x
    # threshold.
    fore[1, 1, 0, -2:] = 1.0  # 10x the truth tail
    # Seed 2 amplifies modestly (2x) — below threshold.
    fore[2, :, :, -2:] = 0.2

    bad = flag_bad_seeds(fore, truth, tail_fraction=0.25, threshold=3.0)
    assert bad == [1]

    # Tighter threshold catches seed 2 too.
    bad_strict = flag_bad_seeds(fore, truth, tail_fraction=0.25, threshold=1.5)
    assert bad_strict == [1, 2]


def test_flag_bad_seeds_validates_shapes_and_knobs():
    truth = np.ones((1, 1, 4))
    good = np.ones((1, 1, 1, 4))
    with pytest.raises(ValueError, match="shape"):
        flag_bad_seeds(good.squeeze(0), truth)  # wrong ndim
    with pytest.raises(ValueError, match="shape"):
        flag_bad_seeds(good, truth[:, :, :3])  # mismatched bins
    with pytest.raises(ValueError, match="tail_fraction"):
        flag_bad_seeds(good, truth, tail_fraction=0.0)
    with pytest.raises(ValueError, match="threshold"):
        flag_bad_seeds(good, truth, threshold=0.0)


def test_flag_bad_seeds_returns_empty_when_all_ok():
    truth = np.ones((1, 1, 4))
    fore = truth[None].repeat(4, axis=0)  # (4, 1, 1, 4), all matching truth
    assert flag_bad_seeds(fore, truth, threshold=1.5) == []


def test_plot_and_save_roundtrip(tmp_path: Path):
    ens, tgt = _fixture()
    crps = crps_per_variable_per_lead(ens, tgt)
    variables = [f"v{i}" for i in range(ens.shape[3])]
    leads = np.arange(1, ens.shape[1] + 1)

    crps_path = tmp_path / "crps.png"
    was_plotted = plot_metric_vs_lead(
        crps, variables, leads, "CRPS", "test crps", str(crps_path)
    )
    if was_plotted:  # matplotlib available
        assert crps_path.is_file() and crps_path.stat().st_size > 0

    hist = rank_histogram_per_variable(ens, tgt)
    hist_path = tmp_path / "hist.png"
    was_plotted = plot_rank_histograms(hist, variables, str(hist_path))
    if was_plotted:
        assert hist_path.is_file() and hist_path.stat().st_size > 0

    B, K, C, H, W = 1, 1, 2, 32, 32
    ens_mean = torch.randn(B, K, C, H, W, dtype=torch.float32)
    tgt2 = torch.randn(B, K, C, H, W, dtype=torch.float32)
    k, ens_pow, tgt_pow = power_spectra_per_variable(ens_mean, tgt2)
    spec_path = tmp_path / "spec.png"
    was_plotted = plot_power_spectra(
        k, ens_pow, tgt_pow, ["v0", "v1"], lead_idx=0, out_path=str(spec_path)
    )
    if was_plotted:
        assert spec_path.is_file() and spec_path.stat().st_size > 0

    summary_path = tmp_path / "summary.npz"
    save_summary(
        {"crps_per_lead_per_channel": crps, "rank_histograms": hist}, str(summary_path)
    )
    loaded = np.load(summary_path)
    assert "crps_per_lead_per_channel" in loaded.files
    assert "rank_histograms" in loaded.files
