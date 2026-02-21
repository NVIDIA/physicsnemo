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

import pytest
import torch

from physicsnemo.nn import Pade


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "numerator_order,denominator_order,expected_exponent",
    [
        (1, 2, -1),  # Should decay as 1/r for 2D-like behavior
        (1, 3, -2),  # Should decay as 1/r² for 3D-like behavior
        # (0, 1, -1),  # Constant numerator, linear denominator
        # (0, 2, -2),  # Constant numerator, quadratic denominator
        # (2, 3, -1),  # Quadratic numerator, cubic denominator
    ],
)
@pytest.mark.parametrize(
    "use_separate_mlps",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "share_denominator_across_channels",
    [
        True,
        False,
    ],
)
def test_pade_asymptotic_behavior(
    device: str,
    numerator_order: int,
    denominator_order: int,
    expected_exponent: int,
    use_separate_mlps: bool,
    share_denominator_across_channels: bool,
):
    """Test that Pade approximant has correct asymptotic decay behavior.

    For a Pade approximant with properly designed features that scale with distance r,
    the asymptotic behavior should be r^(numerator_order) / r^(denominator_order) = r^(numerator_order - denominator_order).
    """
    torch.manual_seed(42)

    # Create a simple Pade network with specified orders
    pade = Pade(
        in_features=1,
        hidden_features=[4, 4],
        out_features=1,
        numerator_order=numerator_order,
        denominator_order=denominator_order,
        use_separate_mlps=use_separate_mlps,
        share_denominator_across_channels=share_denominator_across_channels,
    ).to(device)

    # Test at various distances using log-spaced points for better coverage
    log_distances = torch.linspace(
        10, 20, 21, device=device
    )  # Same as test_field_kernel
    distances = torch.exp(log_distances)

    # Evaluate Pade approximant
    result = pade(distances[:, None])[:, 0]  # Shape: (21,)

    # Perform log-log regression to estimate power law exponent
    # log|y| = log(a) + b*log(r), where b is the exponent
    log_abs_values = torch.log(torch.abs(result))

    # Build design matrix for linear regression
    X = torch.stack(
        [
            torch.ones_like(log_distances),
            log_distances,
        ],
        dim=1,
    )
    y = log_abs_values

    # Solve least squares: coeffs = (X^T X)^(-1) X^T y
    coeffs = torch.linalg.lstsq(X, y).solution
    estimated_exponent = coeffs[1].item()

    # Check if estimated exponent matches expected
    exponent_error = abs(estimated_exponent - expected_exponent)
    tolerance = 0.01  # Use same tolerance as test_field_kernel

    assert exponent_error < tolerance, (
        f"Far-field decay exponent does not match expected value. "
        f"{expected_exponent=}, "
        f"{estimated_exponent=:.4f}, "
        f"{exponent_error=:.4f}. "
    )


if __name__ == "__main__":
    # Run the asymptotic behavior test
    print("Testing Pade asymptotic behavior with controlled initialization...")
    test_pade_asymptotic_behavior(1, 2, -1)
    test_pade_asymptotic_behavior(1, 3, -2)
    test_pade_asymptotic_behavior(0, 1, -1)
    test_pade_asymptotic_behavior(0, 2, -2)
    test_pade_asymptotic_behavior(2, 3, -1)
    print("All asymptotic tests passed!")
