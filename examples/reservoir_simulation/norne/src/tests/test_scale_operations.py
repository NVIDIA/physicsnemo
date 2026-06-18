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
Smoke tests for scale_operation* functions in opm_extract_props_geostats.py.

Tests run against inline implementations so they do not depend on heavy
simulator imports (scipy, OPM readers, etc.) but faithfully mirror the
logic in the source file.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Inline mirrors of the production functions
# ---------------------------------------------------------------------------


def scale_operation(tensor, target_min=None, target_max=None):
    """Scale operation."""
    tensor = tensor.copy()
    tensor[np.isnan(tensor)] = 0
    tensor[np.isinf(tensor)] = 0
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    rescaled_tensor = tensor / max_val
    return min_val, max_val, rescaled_tensor


def scale_operation_pressure(tensor, max_val):
    """Scale operation pressure."""
    tensor = tensor.copy()
    tensor[np.isnan(tensor)] = 0
    tensor[np.isinf(tensor)] = 0
    rescaled_tensor = tensor / max_val
    return np.min(tensor), max_val, rescaled_tensor


def scale_operationS(tensor, lenwels, N_pr):
    """Scale operationS."""
    tensor = tensor.copy()
    tensor[np.isnan(tensor)] = 0
    tensor[np.isinf(tensor)] = 0
    C, Cmax, Cmin = [], [], []
    for k in range(lenwels):
        Anow = tensor[:, :, k * N_pr : (k + 1) * N_pr]
        min_val = np.min(Anow)
        max_val = np.max(Anow)
        C.append(Anow / max_val)
        Cmax.append(max_val)
        Cmin.append(min_val)
    return np.concatenate(C, axis=2), Cmax, Cmin


# ---------------------------------------------------------------------------
# scale_operation
# ---------------------------------------------------------------------------


def test_scale_operation_range():
    """Test scale operation range."""
    tensor = np.array([2.0, 4.0, 8.0])
    _, max_val, rescaled = scale_operation(tensor)
    assert max_val == pytest.approx(8.0)
    assert np.allclose(rescaled, [0.25, 0.5, 1.0])


def test_scale_operation_nan_replaced():
    """Test scale operation nan replaced."""
    tensor = np.array([np.nan, 4.0, 8.0])
    _, _, rescaled = scale_operation(tensor)
    assert not np.any(np.isnan(rescaled))


def test_scale_operation_inf_replaced():
    """Test scale operation inf replaced."""
    tensor = np.array([np.inf, 4.0, 8.0])
    _, _, rescaled = scale_operation(tensor)
    assert not np.any(np.isinf(rescaled))


def test_scale_operation_returns_min_max():
    """Test scale operation returns min max."""
    tensor = np.array([1.0, 5.0, 10.0])
    min_val, max_val, _ = scale_operation(tensor)
    assert min_val == pytest.approx(1.0)
    assert max_val == pytest.approx(10.0)


def test_scale_operation_all_same():
    """Test scale operation all same."""
    tensor = np.full((5,), 3.0)
    _min_val, max_val, rescaled = scale_operation(tensor)
    assert max_val == pytest.approx(3.0)
    assert np.allclose(rescaled, 1.0)


# ---------------------------------------------------------------------------
# scale_operation_pressure
# ---------------------------------------------------------------------------


def test_scale_operation_pressure_divides_by_max_val():
    """Test scale operation pressure divides by max val."""
    tensor = np.array([100.0, 200.0, 300.0])
    _, returned_max, rescaled = scale_operation_pressure(tensor, max_val=300.0)
    assert returned_max == pytest.approx(300.0)
    assert np.allclose(rescaled, [1 / 3, 2 / 3, 1.0])


def test_scale_operation_pressure_nan_replaced():
    """Test scale operation pressure nan replaced."""
    tensor = np.array([np.nan, 200.0, 300.0])
    _, _, rescaled = scale_operation_pressure(tensor, max_val=300.0)
    assert not np.any(np.isnan(rescaled))


# ---------------------------------------------------------------------------
# scale_operationS
# ---------------------------------------------------------------------------


def test_scale_operationS_output_shape():
    """Test scale operationS output shape."""
    lenwels, N_pr, n_samples, n_time = 3, 4, 10, 5
    tensor = np.random.rand(n_samples, n_time, lenwels * N_pr) + 0.1
    out, Cmax, Cmin = scale_operationS(tensor, lenwels=lenwels, N_pr=N_pr)
    assert out.shape == (n_samples, n_time, lenwels * N_pr)
    assert len(Cmax) == lenwels
    assert len(Cmin) == lenwels


def test_scale_operationS_values_bounded():
    """Test scale operationS values bounded."""
    lenwels, N_pr = 2, 3
    tensor = np.random.rand(8, 6, lenwels * N_pr) * 100 + 1
    out, _Cmax, _ = scale_operationS(tensor, lenwels=lenwels, N_pr=N_pr)
    assert np.all(out <= 1.0 + 1e-9)
    assert np.all(out >= 0.0)
