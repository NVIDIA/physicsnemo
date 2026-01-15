# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
Test suite for Wigner D-matrix kernel functions.

Tests the warp implementations of wigner_d_l0 through wigner_d_l5 from
physicsnemo/experimental/nn/symmetry/kernels.py against the numpy reference
implementation in wigner_numpy.py.

Important: The warp functions compute only the REAL PART of the Wigner D-matrix:
    Re(D) = d * cos(m*alpha + m'*gamma)
while the numpy functions return the full complex D-matrix.
Tests compare warp output against numpy_result.real.
"""

import numpy as np
import pytest
import warp as wp

from physicsnemo.experimental.nn.symmetry.kernels import (
    wigner_d_l0,
    wigner_d_l1,
    wigner_d_l2,
    wigner_d_l3,
    wigner_d_l4,
    wigner_d_l5,
    wigner_d_matrix_shape,
)
from physicsnemo.experimental.nn.symmetry.wigner_numpy import wigner_d

# Initialize warp at module level
wp.init()


# =============================================================================
# Wrapper kernels for testing wigner_d_l* functions
# =============================================================================
# These are @wp.kernel wrappers that call the @wp.func decorated wigner_d_l*
# functions. The warp functions cannot be called directly from Python.


@wp.kernel
def _test_wigner_d_l0_kernel(
    alpha: float,
    beta: float,
    gamma: float,
    D_out: wp.array2d(dtype=float),
):
    """Wrapper kernel for testing wigner_d_l0."""
    wigner_d_l0(alpha, beta, gamma, D_out)


@wp.kernel
def _test_wigner_d_l1_kernel(
    alpha: float,
    beta: float,
    gamma: float,
    D_out: wp.array2d(dtype=float),
):
    """Wrapper kernel for testing wigner_d_l1."""
    wigner_d_l1(alpha, beta, gamma, D_out)


@wp.kernel
def _test_wigner_d_l2_kernel(
    alpha: float,
    beta: float,
    gamma: float,
    D_out: wp.array2d(dtype=float),
):
    """Wrapper kernel for testing wigner_d_l2."""
    wigner_d_l2(alpha, beta, gamma, D_out)


@wp.kernel
def _test_wigner_d_l3_kernel(
    alpha: float,
    beta: float,
    gamma: float,
    D_out: wp.array2d(dtype=float),
):
    """Wrapper kernel for testing wigner_d_l3."""
    wigner_d_l3(alpha, beta, gamma, D_out)


@wp.kernel
def _test_wigner_d_l4_kernel(
    alpha: float,
    beta: float,
    gamma: float,
    D_out: wp.array2d(dtype=float),
):
    """Wrapper kernel for testing wigner_d_l4."""
    wigner_d_l4(alpha, beta, gamma, D_out)


@wp.kernel
def _test_wigner_d_l5_kernel(
    alpha: float,
    beta: float,
    gamma: float,
    D_out: wp.array2d(dtype=float),
):
    """Wrapper kernel for testing wigner_d_l5."""
    wigner_d_l5(alpha, beta, gamma, D_out)


# Mapping from l value to wrapper kernel
_WIGNER_KERNELS = {
    0: _test_wigner_d_l0_kernel,
    1: _test_wigner_d_l1_kernel,
    2: _test_wigner_d_l2_kernel,
    3: _test_wigner_d_l3_kernel,
    4: _test_wigner_d_l4_kernel,
    5: _test_wigner_d_l5_kernel,
}


def compute_wigner_d_warp(
    l: int, alpha: float, beta: float, gamma: float, device: str
) -> np.ndarray:
    """Compute Wigner D-matrix using warp kernel.

    Parameters
    ----------
    l : int
        Angular momentum quantum number (0 <= l <= 5)
    alpha, beta, gamma : float
        Euler angles (z-y-z convention)
    device : str
        Device to run on ("cpu" or "cuda:0")

    Returns
    -------
    np.ndarray
        Real part of Wigner D-matrix as numpy array
    """
    wp_device = device if device == "cpu" else "cuda:0"
    shape = wigner_d_matrix_shape(l)

    # Create output array on device
    D_out = wp.zeros(shape, dtype=wp.float32, device=wp_device)

    # Get the appropriate kernel
    kernel = _WIGNER_KERNELS[l]

    # Launch kernel with dim=1 (single invocation)
    wp.launch(
        kernel,
        dim=1,
        inputs=[alpha, beta, gamma, D_out],
        device=wp_device,
    )

    # Synchronize and convert to numpy
    wp.synchronize_device(wp_device)
    return D_out.numpy()


# =============================================================================
# Test cases
# =============================================================================


@pytest.mark.parametrize("l", [0, 1, 2, 3, 4, 5])
def test_wigner_d_hardcoded_values(device, l):
    """Test wigner_d_l* against numpy reference with fixed angle triplets.

    Tests with fixed angle combinations:
    - (0, 0, 0): Identity case
    - (pi/4, pi/3, pi/6): Generic non-trivial angles
    - (pi/2, pi/2, pi/2): 90-degree rotations

    Parameters
    ----------
    device : str
        Pytest fixture providing device ("cpu" or "cuda:0")
    l : int
        Angular momentum quantum number
    """
    # Test angle triplets: (alpha, beta, gamma)
    test_angles = [
        (0.0, 0.0, 0.0),  # Identity
        (np.pi / 4, np.pi / 3, np.pi / 6),  # Generic angles
        (np.pi / 2, np.pi / 2, np.pi / 2),  # 90-degree rotations
    ]

    for alpha, beta, gamma in test_angles:
        # Compute with warp kernel
        D_warp = compute_wigner_d_warp(l, alpha, beta, gamma, device)

        # Compute with numpy reference (complex) and take real part
        D_numpy_complex = wigner_d(l, alpha, beta, gamma)
        D_numpy_real = D_numpy_complex.real

        # Compare with tolerance
        np.testing.assert_allclose(
            D_warp,
            D_numpy_real,
            atol=1e-5,
            err_msg=f"Mismatch for l={l}, angles=({alpha:.4f}, {beta:.4f}, {gamma:.4f})",
        )


@pytest.mark.parametrize("l", [0, 1, 2, 3, 4, 5])
def test_wigner_d_identity(device, l):
    """Test that D(0, 0, 0) equals identity matrix.

    At (alpha=0, beta=0, gamma=0), the Wigner D-matrix should be the identity
    matrix for any l value.

    Parameters
    ----------
    device : str
        Pytest fixture providing device ("cpu" or "cuda:0")
    l : int
        Angular momentum quantum number
    """
    alpha, beta, gamma = 0.0, 0.0, 0.0

    # Compute with warp kernel
    D_warp = compute_wigner_d_warp(l, alpha, beta, gamma, device)

    # Expected: identity matrix
    dim = 2 * l + 1
    expected = np.eye(dim, dtype=np.float32)

    np.testing.assert_allclose(
        D_warp,
        expected,
        atol=1e-5,
        err_msg=f"D(0,0,0) is not identity for l={l}",
    )


@pytest.mark.parametrize("l", [0, 1, 2, 3, 4, 5])
def test_wigner_d_beta_zero(device, l):
    """Test diagonal structure when beta=0.

    When beta=0, the small d-matrix is diagonal with d^l_{m,m'} = delta_{m,m'}.
    The full D-matrix becomes diagonal with elements exp(i*m*(alpha+gamma)),
    so the real part has diagonal elements cos(m*(alpha+gamma)).

    Parameters
    ----------
    device : str
        Pytest fixture providing device ("cpu" or "cuda:0")
    l : int
        Angular momentum quantum number
    """
    alpha = np.pi / 4
    beta = 0.0
    gamma = np.pi / 6

    # Compute with warp kernel
    D_warp = compute_wigner_d_warp(l, alpha, beta, gamma, device)

    # Build expected diagonal matrix
    # For beta=0, D^l_{m,m'} = delta_{m,m'} * exp(i*m*(alpha+gamma))
    # Real part: D[i,j] = delta_{i,j} * cos(m*(alpha+gamma))
    # where m = l - i (row i corresponds to m = l - i)
    dim = 2 * l + 1
    expected = np.zeros((dim, dim), dtype=np.float32)
    for i in range(dim):
        m = l - i  # m values go from l to -l
        expected[i, i] = np.cos(m * (alpha + gamma))

    np.testing.assert_allclose(
        D_warp,
        expected,
        atol=1e-5,
        err_msg=f"D(alpha, 0, gamma) not diagonal with cos(m*(alpha+gamma)) for l={l}",
    )


@pytest.mark.parametrize("l", [0, 1, 2, 3, 4, 5])
def test_wigner_d_random_angles(device, l):
    """Test wigner_d_l* against numpy reference with random angles.

    Tests with multiple random angle triplets to ensure broad coverage.

    Parameters
    ----------
    device : str
        Pytest fixture providing device ("cpu" or "cuda:0")
    l : int
        Angular momentum quantum number
    """
    # Generate random angles (use fixed seed for reproducibility via conftest)
    rng = np.random.default_rng(42)
    n_tests = 5

    for _ in range(n_tests):
        alpha = rng.uniform(0, 2 * np.pi)
        beta = rng.uniform(0, np.pi)
        gamma = rng.uniform(0, 2 * np.pi)

        # Compute with warp kernel
        D_warp = compute_wigner_d_warp(l, alpha, beta, gamma, device)

        # Compute with numpy reference (complex) and take real part
        D_numpy_complex = wigner_d(l, alpha, beta, gamma)
        D_numpy_real = D_numpy_complex.real

        # Compare with tolerance
        np.testing.assert_allclose(
            D_warp,
            D_numpy_real,
            atol=1e-5,
            err_msg=f"Mismatch for l={l}, angles=({alpha:.4f}, {beta:.4f}, {gamma:.4f})",
        )


@pytest.mark.parametrize("l", [0, 1, 2, 3, 4, 5])
def test_wigner_d_matrix_shape(device, l):
    """Test that output matrix has correct shape.

    The Wigner D-matrix for angular momentum l should be (2l+1) x (2l+1).

    Parameters
    ----------
    device : str
        Pytest fixture providing device ("cpu" or "cuda:0")
    l : int
        Angular momentum quantum number
    """
    D_warp = compute_wigner_d_warp(l, 0.0, 0.0, 0.0, device)

    expected_shape = (2 * l + 1, 2 * l + 1)
    assert D_warp.shape == expected_shape, (
        f"Shape mismatch for l={l}: {D_warp.shape} != {expected_shape}"
    )


@pytest.mark.parametrize("l", [0, 1, 2, 3, 4, 5])
def test_wigner_d_alpha_only(device, l):
    """Test behavior when only alpha is non-zero.

    When beta=0, gamma=0, the matrix should be diagonal with
    D[i,i] = cos(m*alpha) where m = l - i.

    Parameters
    ----------
    device : str
        Pytest fixture providing device ("cpu" or "cuda:0")
    l : int
        Angular momentum quantum number
    """
    alpha = np.pi / 3
    beta = 0.0
    gamma = 0.0

    D_warp = compute_wigner_d_warp(l, alpha, beta, gamma, device)

    # Build expected diagonal matrix
    dim = 2 * l + 1
    expected = np.zeros((dim, dim), dtype=np.float32)
    for i in range(dim):
        m = l - i
        expected[i, i] = np.cos(m * alpha)

    np.testing.assert_allclose(
        D_warp,
        expected,
        atol=1e-5,
        err_msg=f"D(alpha, 0, 0) failed for l={l}",
    )


@pytest.mark.parametrize("l", [0, 1, 2, 3, 4, 5])
def test_wigner_d_gamma_only(device, l):
    """Test behavior when only gamma is non-zero.

    When alpha=0, beta=0, the matrix should be diagonal with
    D[i,i] = cos(m*gamma) where m = l - i.

    Parameters
    ----------
    device : str
        Pytest fixture providing device ("cpu" or "cuda:0")
    l : int
        Angular momentum quantum number
    """
    alpha = 0.0
    beta = 0.0
    gamma = np.pi / 5

    D_warp = compute_wigner_d_warp(l, alpha, beta, gamma, device)

    # Build expected diagonal matrix
    dim = 2 * l + 1
    expected = np.zeros((dim, dim), dtype=np.float32)
    for i in range(dim):
        m = l - i
        expected[i, i] = np.cos(m * gamma)

    np.testing.assert_allclose(
        D_warp,
        expected,
        atol=1e-5,
        err_msg=f"D(0, 0, gamma) failed for l={l}",
    )


@pytest.mark.parametrize("l", [0, 1, 2, 3, 4, 5])
def test_wigner_d_beta_pi(device, l):
    """Test behavior when beta=pi.

    When beta=pi, the small d-matrix has specific structure that
    can be validated against the numpy reference.

    Parameters
    ----------
    device : str
        Pytest fixture providing device ("cpu" or "cuda:0")
    l : int
        Angular momentum quantum number
    """
    alpha = np.pi / 4
    beta = np.pi
    gamma = np.pi / 3

    D_warp = compute_wigner_d_warp(l, alpha, beta, gamma, device)

    # Compute with numpy reference
    D_numpy_complex = wigner_d(l, alpha, beta, gamma)
    D_numpy_real = D_numpy_complex.real

    np.testing.assert_allclose(
        D_warp,
        D_numpy_real,
        atol=1e-5,
        err_msg=f"D(alpha, pi, gamma) failed for l={l}",
    )
