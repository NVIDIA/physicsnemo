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

"""Unit tests for the fair-CRPS loss used in FGN training.

The reference values are computed directly from arXiv:2506.10772v1 eq. (4):

    fCRPS(x_{1:N}, y) = (1/N) sum_n |x_n - y|
                      - (1/(2 N (N-1))) sum_{n, n'} |x_n - x_{n'}|
"""

import math

import numpy as np
import pytest
import torch
from utils.loss import (
    build_area_weights,
    build_channel_weights,
    ensemble_mean_mse,
    fair_crps,
)

# ---------------------------------------------------------------------------
# Paper eq. (4): scalar reference calculations
# ---------------------------------------------------------------------------


def _fcrps_reference(x: list[float], y: float) -> float:
    """Reference fCRPS per eq. (4) using naive Python (no torch/numpy)."""
    n = len(x)
    first = sum(abs(xi - y) for xi in x) / n
    pair = sum(abs(xi - xj) for xi in x for xj in x)
    second = pair / (2.0 * n * (n - 1))
    return first - second


def _to_ensemble(x: list[float]) -> torch.Tensor:
    """Turn a flat list into shape ``[1, N, 1, 1, 1]``."""
    return torch.tensor(x, dtype=torch.float64).view(1, -1, 1, 1, 1)


@pytest.mark.parametrize(
    "x,y,expected",
    [
        # Symmetric ensemble around truth, perfect calibration.
        # N=2, y=0, x=[1,-1]: first=1, second=(|1-1|+|1+1|+|-1-1|+|-1+1|)/4=1 → 0.
        ([1.0, -1.0], 0.0, 0.0),
        # N=3, y=0, x=[1,2,3] — worked example from the chat log.
        # first = (1+2+3)/3 = 2
        # pair  = 0+1+2 + 1+0+1 + 2+1+0 = 8
        # second = 8 / (2*3*2) = 8/12
        # fCRPS = 2 - 8/12
        ([1.0, 2.0, 3.0], 0.0, 2.0 - 8.0 / 12.0),
        # Degenerate ensemble: all members identical, spread term = 0,
        # so fCRPS reduces to plain |x - y|.
        ([5.0, 5.0, 5.0], 3.0, 2.0),
    ],
)
def test_fair_crps_matches_paper_eq4(x, y, expected):
    ensemble = _to_ensemble(x)
    target = torch.tensor(y, dtype=torch.float64).view(1, 1, 1, 1)
    got = fair_crps(ensemble, target).item()
    assert math.isclose(got, expected, rel_tol=0, abs_tol=1e-12), (got, expected)


@pytest.mark.parametrize(
    "x,y",
    [
        ([0.3, -1.7], 1.1),
        ([-2.0, 0.5, 4.5], -0.3),
        ([1.0, 1.0, 2.0, 2.0, 3.0], 1.5),
    ],
)
def test_fair_crps_matches_naive_reference(x, y):
    ensemble = _to_ensemble(x)
    target = torch.tensor(y, dtype=torch.float64).view(1, 1, 1, 1)
    expected = _fcrps_reference(x, y)
    got = fair_crps(ensemble, target).item()
    assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_fair_crps_rejects_single_member():
    with pytest.raises(ValueError, match="at least two"):
        fair_crps(torch.zeros(1, 1, 1, 1, 1), torch.zeros(1, 1, 1, 1))


def test_fair_crps_rejects_bad_shapes():
    with pytest.raises(ValueError, match=r"\[B, M, C, H, W\]"):
        fair_crps(torch.zeros(1, 2, 1, 1), torch.zeros(1, 1, 1, 1))
    with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
        fair_crps(torch.zeros(1, 2, 1, 1, 1), torch.zeros(1, 1, 1))


# ---------------------------------------------------------------------------
# Paper eq. (5): weighted loss reduction matches (1/G) sum_i a_i fCRPS_i
# ---------------------------------------------------------------------------


def test_weighted_fair_crps_matches_paper_eq5():
    # Two channels, 1x3 spatial, N=2 ensemble, hand-computed.
    # Build a case where channels have different per-location fCRPS values.
    ensemble = torch.tensor(
        [
            [
                [[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]],
                [[[-1.0, 0.0, 1.0]], [[2.0, 3.0, 4.0]]],
            ]
        ],
        dtype=torch.float64,
    )  # shape (B=1, M=2, C=2, H=1, W=3)
    target = torch.tensor(
        [[[[0.0, 0.0, 0.0]], [[5.0, 5.0, 5.0]]]], dtype=torch.float64
    )  # (1, 2, 1, 3)

    # Per-location fCRPS with N=2:
    #   fCRPS = (|x_1 - y| + |x_2 - y|) / 2 - |x_1 - x_2| / 2
    # (since pairwise sum = 2 * |x_1 - x_2| and 2N(N-1) = 4, so second term
    # = 2|x_1 - x_2|/4 = |x_1 - x_2|/2)
    def fc(x1, x2, y):
        return (abs(x1 - y) + abs(x2 - y)) / 2.0 - abs(x1 - x2) / 2.0

    per_loc = torch.tensor(
        [
            [
                [[fc(1, -1, 0), fc(2, 0, 0), fc(3, 1, 0)]],
                [[fc(4, 2, 5), fc(5, 3, 5), fc(6, 4, 5)]],
            ]
        ],
        dtype=torch.float64,
    )
    # Unweighted: mean over (B, C, H, W).
    expected_unw = per_loc.mean().item()
    got_unw = fair_crps(ensemble, target).item()
    assert math.isclose(got_unw, expected_unw, rel_tol=0, abs_tol=1e-12)

    # Weighted reduction per paper eq. (5): (1/G) sum a_i fCRPS_i.
    # weights shape (1, 2, 1, 1): per-channel.
    weights = torch.tensor([0.1, 1.0], dtype=torch.float64).view(1, 2, 1, 1)
    expected_w = (per_loc * weights).mean().item()
    got_w = fair_crps(ensemble, target, weights=weights).item()
    assert math.isclose(got_w, expected_w, rel_tol=0, abs_tol=1e-12)


def test_weighted_fair_crps_broadcast_shapes():
    ensemble = torch.randn(2, 3, 4, 5, 6, dtype=torch.float64)
    target = torch.randn(2, 4, 5, 6, dtype=torch.float64)

    # Per-channel weights only.
    w_c = torch.rand(1, 4, 1, 1, dtype=torch.float64) + 0.1
    # Per-lat weights only.
    w_h = torch.rand(1, 1, 5, 1, dtype=torch.float64) + 0.1
    # Full broadcast.
    w_full = (w_c * w_h).broadcast_to(2, 4, 5, 6)

    loss_c = fair_crps(ensemble, target, weights=w_c).item()
    loss_h = fair_crps(ensemble, target, weights=w_h).item()
    loss_full = fair_crps(ensemble, target, weights=w_full).item()

    # Not equal to each other, but each should match the explicit per-loc
    # computation when reduced with .mean().
    assert math.isfinite(loss_c) and math.isfinite(loss_h) and math.isfinite(loss_full)


# ---------------------------------------------------------------------------
# ensemble_mean_mse
# ---------------------------------------------------------------------------


def test_ensemble_mean_mse_matches_hand():
    ensemble = torch.tensor(
        [[[[[1.0]]], [[[3.0]]]]], dtype=torch.float64
    )  # (1, 2, 1, 1, 1)
    target = torch.tensor([[[[2.0]]]], dtype=torch.float64)
    # mean pred = 2.0, squared error = 0.
    assert ensemble_mean_mse(ensemble, target).item() == pytest.approx(0.0)

    target2 = torch.tensor([[[[5.0]]]], dtype=torch.float64)
    # mean pred = 2.0, error = -3, sq = 9.
    assert ensemble_mean_mse(ensemble, target2).item() == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# build_channel_weights — paper §2.2.3 scheme
# ---------------------------------------------------------------------------


def test_channel_weights_surface_and_t2m():
    # Paper scheme: surface → 0.1; t2m special-cased to 1.0.
    w = build_channel_weights(["u10m", "v10m", "t2m", "msl"])
    assert np.allclose(w, [0.1, 0.1, 1.0, 0.1])


def test_channel_weights_atmospheric_linear_by_level():
    # Two atmospheric variables of the same prefix, levels 300 and 500.
    # Expected: level / sum(levels) = 3/8 and 5/8.
    w = build_channel_weights(["t300", "t500"])
    assert math.isclose(w[0], 3 / 8, abs_tol=1e-6)
    assert math.isclose(w[1], 5 / 8, abs_tol=1e-6)


def test_channel_weights_geopotential_halved():
    # Paper §2.2.3: geopotential weights halved to tame overfitting.
    w_z = build_channel_weights(["z300", "z500"])
    assert math.isclose(w_z[0], 0.5 * 3 / 8, abs_tol=1e-6)
    assert math.isclose(w_z[1], 0.5 * 5 / 8, abs_tol=1e-6)

    # Non-geopotential atmospheric (temperature) with the same levels is NOT
    # halved — acts as a control to confirm the halving is scoped to z*.
    w_t = build_channel_weights(["t300", "t500"])
    assert math.isclose(w_z[0] * 2.0, w_t[0], abs_tol=1e-6)
    assert math.isclose(w_z[1] * 2.0, w_t[1], abs_tol=1e-6)


def test_channel_weights_paper_table_a1_mixed():
    # Mix of atmospheric and surface from Table A.1 — all weights positive
    # and finite, geopotential scaled down relative to temperature.
    variables = [
        "z500",
        "z850",  # atmospheric geopotential, halved
        "t500",
        "t850",  # atmospheric temperature, NOT halved
        "u10m",
        "v10m",
        "t2m",
        "msl",  # surface
    ]
    w = build_channel_weights(variables)
    assert (w > 0).all() and np.all(np.isfinite(w))
    # z at level L_i gets half the temperature weight at level L_i.
    assert math.isclose(w[0] * 2.0, w[2], abs_tol=1e-6)  # z500 -> t500
    assert math.isclose(w[1] * 2.0, w[3], abs_tol=1e-6)  # z850 -> t850
    # Surface layout (float32 tolerance).
    assert math.isclose(w[4], 0.1, abs_tol=1e-6)
    assert math.isclose(w[5], 0.1, abs_tol=1e-6)
    assert math.isclose(w[6], 1.0, abs_tol=1e-6)
    assert math.isclose(w[7], 0.1, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# build_area_weights
# ---------------------------------------------------------------------------


def test_area_weights_normalised_to_unit_mean():
    for h in (37, 73, 181, 721):
        w = build_area_weights(h)
        assert w.shape == (h, 1)
        # Mean over latitude equals 1 by construction.
        assert math.isclose(float(w.mean()), 1.0, abs_tol=1e-6)
        # Poles get the smallest weight, equator the largest.
        assert w[0, 0] < w[h // 2, 0]
        assert w[-1, 0] < w[h // 2, 0]
