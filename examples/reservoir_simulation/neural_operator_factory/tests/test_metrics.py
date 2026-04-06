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

"""Unit tests for evaluation metrics (numpy and torch)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.metrics import (
    compute_r2_score,
    compute_relative_l1_error,
    compute_relative_l2_error,
    mae_torch,
    max_absolute_error,
    max_error_torch,
    mean_absolute_error,
    mean_plume_error,
    mean_relative_error,
    mse_torch,
    normalized_mse,
    peak_signal_to_noise_ratio,
    psnr_torch,
    r2_score_torch,
    relative_l1_torch,
    relative_l2_torch,
    rmse_torch,
)


class TestNumpyMetrics:
    """Tests for numpy-based metrics."""

    def test_mae_known_value(self):
        """Verify MAE computes expected value for known inputs."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        assert abs(mean_absolute_error(y_pred, y_true) - 0.5) < 1e-8

    def test_mae_identical(self):
        """Verify MAE is zero when prediction equals target."""
        y = np.random.randn(100)
        assert mean_absolute_error(y, y) == 0.0

    def test_max_absolute_error(self):
        """Verify max absolute error returns the largest element-wise difference."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 3.0, 2.0])
        assert abs(max_absolute_error(y_pred, y_true) - 3.0) < 1e-8

    def test_mre_known_value(self):
        """Verify mean relative error computes expected value for known inputs."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 21.0, 31.0])
        data_range = 30.0 - 10.0  # 20
        expected = np.mean(np.abs(y_pred - y_true)) / data_range  # 1/20 = 0.05
        assert abs(mean_relative_error(y_pred, y_true) - expected) < 1e-8

    def test_mpe_only_plume_region(self):
        """Verify mean plume error only considers cells above plume threshold."""
        y_true = np.array([0.0, 0.0, 0.5, 0.8])
        y_pred = np.array([0.0, 0.0, 0.6, 0.9])
        mpe = mean_plume_error(y_pred, y_true)
        assert abs(mpe - 0.1) < 1e-8

    def test_mpe_no_plume(self):
        """Verify mean plume error is zero when no plume region exists."""
        y_true = np.zeros(10)
        y_pred = np.zeros(10)
        assert mean_plume_error(y_pred, y_true) == 0.0

    def test_r2_perfect(self):
        """Verify R2 score is 1.0 for a perfect prediction."""
        y = np.random.randn(50)
        assert abs(compute_r2_score(y, y) - 1.0) < 1e-8

    def test_r2_constant_prediction(self):
        """Verify R2 score is zero when prediction equals the target mean."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full_like(y_true, y_true.mean())
        assert abs(compute_r2_score(y_pred, y_true)) < 1e-8

    def test_r2_negative_for_bad_prediction(self):
        """Verify R2 score is negative for a prediction worse than the mean."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        assert compute_r2_score(y_pred, y_true) < 0

    def test_relative_l2_known(self):
        """Verify relative L2 error equals 1.0 when prediction is all zeros."""
        y_true = np.array([3.0, 4.0])  # norm = 5
        y_pred = np.array([0.0, 0.0])  # diff norm = 5
        assert abs(compute_relative_l2_error(y_pred, y_true) - 1.0) < 1e-6

    def test_relative_l1_known(self):
        """Verify relative L1 error equals 1.0 when prediction is all zeros."""
        y_true = np.array([1.0, 2.0, 3.0])  # L1 norm = 6
        y_pred = np.array([0.0, 0.0, 0.0])  # diff L1 = 6
        assert abs(compute_relative_l1_error(y_pred, y_true) - 1.0) < 1e-6

    def test_nmse_variance(self):
        """Verify normalized MSE with variance normalization for known inputs."""
        y_true = np.array([1.0, 3.0])  # var = 1
        y_pred = np.array([2.0, 2.0])  # mse = 1
        assert abs(normalized_mse(y_pred, y_true, "variance") - 1.0) < 1e-6

    def test_psnr_perfect(self):
        """Verify PSNR is infinite for a perfect prediction."""
        y = np.random.randn(100)
        assert peak_signal_to_noise_ratio(y, y) == float("inf")

    def test_psnr_known(self):
        """Verify PSNR computes expected dB value for known MSE and data range."""
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.0, 0.9])  # mse = 0.005, range = 1
        psnr = peak_signal_to_noise_ratio(y_pred, y_true)
        expected = 20 * np.log10(1.0 / np.sqrt(0.005))
        assert abs(psnr - expected) < 1e-4


class TestTorchMetrics:
    """Tests for torch-based metrics."""

    def test_mse_torch_value(self):
        """Verify torch MSE computes expected value for known inputs."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 3.5])
        assert torch.isclose(mse_torch(pred, target), torch.tensor(0.25))

    def test_rmse_torch_value(self):
        """Verify torch RMSE computes expected value for known inputs."""
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([2.0, 3.0])
        assert torch.isclose(rmse_torch(pred, target), torch.tensor(1.0))

    def test_mae_torch_value(self):
        """Verify torch MAE computes expected value for known inputs."""
        pred = torch.tensor([0.0, 0.0])
        target = torch.tensor([1.0, 3.0])
        assert torch.isclose(mae_torch(pred, target), torch.tensor(2.0))

    def test_relative_l2_torch_known(self):
        """Verify torch relative L2 error equals 1.0 for zero prediction."""
        target = torch.tensor([3.0, 4.0])  # norm = 5
        pred = torch.zeros(2)
        assert torch.isclose(
            relative_l2_torch(pred, target), torch.tensor(1.0), atol=1e-6
        )

    def test_relative_l1_torch_known(self):
        """Verify torch relative L1 error equals 1.0 for zero prediction."""
        target = torch.tensor([1.0, 2.0, 3.0])
        pred = torch.zeros(3)
        assert torch.isclose(
            relative_l1_torch(pred, target), torch.tensor(1.0), atol=1e-6
        )

    def test_r2_torch_perfect(self):
        """Verify torch R2 score is 1.0 for a perfect prediction."""
        y = torch.randn(50)
        assert torch.isclose(r2_score_torch(y, y), torch.tensor(1.0))

    def test_r2_torch_negative(self):
        """Verify torch R2 score is negative for a prediction worse than the mean."""
        target = torch.tensor([1.0, 2.0, 3.0])
        pred = torch.tensor([10.0, 20.0, 30.0])
        assert r2_score_torch(pred, target) < 0

    def test_max_error_torch(self):
        """Verify torch max error returns the largest element-wise difference."""
        pred = torch.tensor([0.0, 0.0])
        target = torch.tensor([1.0, 5.0])
        assert torch.isclose(max_error_torch(pred, target), torch.tensor(5.0))

    def test_psnr_torch_perfect(self):
        """Verify torch PSNR is infinite for a perfect prediction."""
        y = torch.randn(50)
        assert psnr_torch(y, y) == float("inf")


class TestTorchNumpyConsistency:
    """Verify torch and numpy metrics agree on the same data."""

    def test_mae_consistency(self):
        """Verify numpy and torch MAE agree on the same random data."""
        pred_np = np.random.randn(100)
        target_np = np.random.randn(100)
        np_val = mean_absolute_error(pred_np, target_np)
        torch_val = mae_torch(torch.tensor(pred_np), torch.tensor(target_np)).item()
        assert abs(np_val - torch_val) < 1e-6

    def test_relative_l2_consistency(self):
        """Verify numpy and torch relative L2 error agree on the same data."""
        pred_np = np.random.randn(100)
        target_np = np.random.randn(100) + 2
        np_val = compute_relative_l2_error(pred_np, target_np)
        torch_val = relative_l2_torch(
            torch.tensor(pred_np), torch.tensor(target_np)
        ).item()
        assert abs(np_val - torch_val) < 1e-5

    def test_r2_consistency(self):
        """Verify numpy and torch R2 score agree on the same data."""
        pred_np = np.random.randn(100)
        target_np = np.random.randn(100)
        np_val = compute_r2_score(pred_np, target_np)
        torch_val = r2_score_torch(
            torch.tensor(pred_np), torch.tensor(target_np)
        ).item()
        assert abs(np_val - torch_val) < 1e-5


class TestEdgeCases:
    """Edge cases for metrics."""

    def test_single_element(self):
        """Verify metrics handle single-element arrays correctly."""
        y = np.array([5.0])
        assert mean_absolute_error(y, y) == 0.0
        assert compute_r2_score(y, y) == 1.0

    def test_zero_target_relative_l2(self):
        """Verify relative L2 error returns a finite value when target is zero."""
        pred = np.array([1.0])
        target = np.array([0.0])
        val = compute_relative_l2_error(pred, target)
        assert np.isfinite(val)

    def test_constant_target_r2(self):
        """Verify R2 score is zero when target has zero variance."""
        target = np.ones(10)
        pred = np.ones(10) * 1.1
        r2 = compute_r2_score(pred, target)
        assert r2 == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
