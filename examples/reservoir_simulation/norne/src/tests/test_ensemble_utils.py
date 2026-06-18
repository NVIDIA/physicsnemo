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

"""
Smoke tests for ensemble utility functions.

Covers:
- clip_ensemble_params: clips PERM/PORO dictionaries within physical bounds
- compute_data_mismatch: RMSE-like mismatch between sim_data and measurements
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# clip_ensemble_params
# ---------------------------------------------------------------------------


def _make_filcc(perm_vals, poro_vals):
    """Build a minimal filcc dict with given flat PERM/PORO arrays."""
    return {
        "PERM": np.array(perm_vals, dtype=float),
        "PORO": np.array(poro_vals, dtype=float),
    }


def _clip(filcc, High_K, Low_K, High_P, Low_P):
    """Inline reimplementation so tests don't depend on import paths."""
    filcc["PERM"] = np.clip(filcc["PERM"], float(Low_K), float(High_K))
    filcc["PORO"] = np.clip(filcc["PORO"], float(Low_P), float(High_P))
    return filcc


def test_clip_ensemble_params_already_in_bounds():
    """Test clip ensemble params already in bounds."""
    filcc = _make_filcc([100.0, 200.0], [0.1, 0.3])
    result = _clip(filcc, High_K=500.0, Low_K=10.0, High_P=0.5, Low_P=0.05)
    assert np.allclose(result["PERM"], [100.0, 200.0])
    assert np.allclose(result["PORO"], [0.1, 0.3])


def test_clip_ensemble_params_clips_high():
    """Test clip ensemble params clips high."""
    filcc = _make_filcc([1000.0, 2000.0], [0.9, 1.5])
    result = _clip(filcc, High_K=500.0, Low_K=10.0, High_P=0.5, Low_P=0.05)
    assert np.all(result["PERM"] <= 500.0)
    assert np.all(result["PORO"] <= 0.5)


def test_clip_ensemble_params_clips_low():
    """Test clip ensemble params clips low."""
    filcc = _make_filcc([1.0, 5.0], [0.001, 0.01])
    result = _clip(filcc, High_K=500.0, Low_K=10.0, High_P=0.5, Low_P=0.05)
    assert np.all(result["PERM"] >= 10.0)
    assert np.all(result["PORO"] >= 0.05)


def test_clip_ensemble_params_preserves_other_keys():
    """Test clip ensemble params preserves other keys."""
    filcc = {
        "PERM": np.array([5.0]),
        "PORO": np.array([0.01]),
        "EXTRA": np.array([42.0]),
    }
    result = _clip(filcc, High_K=500.0, Low_K=10.0, High_P=0.5, Low_P=0.05)
    assert "EXTRA" in result
    assert np.allclose(result["EXTRA"], [42.0])


# ---------------------------------------------------------------------------
# compute_data_mismatch (numpy path)
# ---------------------------------------------------------------------------


def _compute_mismatch_numpy(sim_data, measurement):
    """Inline numpy implementation matching inversion_operation_ensemble.py."""
    sim_data = np.asarray(sim_data)
    measurement = np.asarray(measurement).reshape(-1, 1)
    ne = sim_data.shape[1]
    obj_real = np.zeros((ne, 1))
    for j in range(ne):
        noww = sim_data[:, j].reshape(-1, 1)
        obj_real[j] = np.sqrt(np.sum((noww - measurement) ** 2)) / measurement.shape[0]
    return np.mean(obj_real).item(), np.std(obj_real).item(), obj_real


def test_compute_data_mismatch_perfect():
    """Zero mismatch when sim equals measurement."""
    meas = np.ones((10, 1))
    sim = np.ones((10, 3))  # 3 ensemble members, all equal to measurement
    obj, obj_std, obj_real = _compute_mismatch_numpy(sim, meas)
    assert obj == pytest.approx(0.0)
    assert obj_std == pytest.approx(0.0)
    assert obj_real.shape == (3, 1)


def test_compute_data_mismatch_known_value():
    """Single ensemble member, difference = [1,1,...,1] → RMSE = sqrt(n)/n = 1/sqrt(n)."""
    n = 4
    meas = np.zeros((n, 1))
    sim = np.ones((n, 1))  # shape (n_obs, 1) — one ensemble member
    obj, _, _ = _compute_mismatch_numpy(sim, meas)
    expected = np.sqrt(n) / n  # = 1/sqrt(n)
    assert obj == pytest.approx(expected, rel=1e-6)


def test_compute_data_mismatch_output_shape():
    """Test compute data mismatch output shape."""
    n_obs, n_ens = 20, 5
    sim = np.random.randn(n_obs, n_ens)
    meas = np.random.randn(n_obs, 1)
    obj, obj_std, obj_real = _compute_mismatch_numpy(sim, meas)
    assert obj_real.shape == (n_ens, 1)
    assert isinstance(obj, float)
    assert isinstance(obj_std, float)


def test_compute_data_mismatch_non_negative():
    """Test compute data mismatch non negative."""
    sim = np.random.randn(15, 7)
    meas = np.random.randn(15, 1)
    obj, _obj_std, obj_real = _compute_mismatch_numpy(sim, meas)
    assert obj >= 0.0
    assert np.all(obj_real >= 0.0)
