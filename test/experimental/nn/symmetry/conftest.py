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

"""Shared pytest fixtures and utilities for symmetry module tests.

This module provides consolidated fixtures for dtype, device, and lmax/mmax
parameterization, as well as tolerance utilities for numerical comparisons
across different floating-point precisions.

Half-Precision Notes
--------------------
Half-precision dtypes (float16, bfloat16) have significant numerical limitations
for Wigner D-matrix computations. Based on empirical testing:

- float16 at lmax<=2: Orthogonality errors ~0.2-0.4%, usable with loose tolerances
- float16 at lmax>=3: Orthogonality errors ~3-8%, requires xfail for precision tests
- bfloat16: Consistently worse than float16, errors can reach 26% at lmax=4

Tests that require high numerical precision (orthogonality, equivariance) should
use ``pytest.xfail`` on a case-by-case basis for half-precision dtypes.
"""

from __future__ import annotations

import pytest
import torch

# =============================================================================
# Half-Precision Utilities
# =============================================================================


def is_half_precision(dtype: torch.dtype) -> bool:
    """Check if dtype is half-precision (float16 or bfloat16).

    Parameters
    ----------
    dtype : torch.dtype
        The data type to check.

    Returns
    -------
    bool
        True if dtype is float16 or bfloat16, False otherwise.
    """
    return dtype in (torch.float16, torch.bfloat16)


# =============================================================================
# Tolerance Helper for Multi-Precision Testing
# =============================================================================


def get_rtol_atol(dtype: torch.dtype, scale: float = 1.0) -> tuple[float, float]:
    """Return (rtol, atol) appropriate for the given torch dtype.

    Parameters
    ----------
    dtype : torch.dtype
        The data type to get tolerances for.
    scale : float, optional
        Scaling factor to apply to both rtol and atol. Use values > 1.0 to
        loosen tolerances for tests that can accept larger errors (e.g.,
        structural checks, batch consistency). Default is 1.0 (no scaling).

    Returns
    -------
    tuple[float, float]
        A tuple of (rtol, atol) values appropriate for the dtype, scaled by
        the given factor.

    Notes
    -----
    Tolerances are calibrated to accommodate the numerical requirements of
    various tests in the symmetry module, including:
    - Wigner D-matrix orthogonality tests
    - SO(3) equivariance verification
    - SO(2) rotation tests

    The base tolerances provide safety margins appropriate for each precision level:
    - float16:  ~3 decimal digits of precision
    - bfloat16: ~2-3 decimal digits of precision (wider dynamic range)
    - float32:  ~4-5 decimal digits for accumulated numerical operations
    - float64:  ~8-10 decimal digits for high-precision verification

    Examples
    --------
    >>> rtol, atol = get_rtol_atol(torch.float32)  # Standard tolerances
    >>> rtol, atol = get_rtol_atol(torch.float16, scale=10.0)  # 10x looser
    """
    if dtype == torch.float16:
        # float16 has limited precision (~3.3 decimal digits)
        rtol, atol = 5e-3, 5e-3
    elif dtype == torch.bfloat16:
        # bfloat16 has even less precision (~2.4 decimal digits)
        rtol, atol = 3e-2, 3e-2
    elif dtype == torch.float32:
        # float32 needs looser tolerances for equivariance tests with
        # accumulated numerical operations (matrix multiplications, rotations)
        rtol, atol = 1e-4, 1e-4
    elif dtype == torch.float64:
        # float64 needs slightly looser tolerances for accumulated numerical
        # operations in Wigner D-matrix and rotation calculations
        rtol, atol = 1e-6, 1e-6
    else:
        # Default fallback for other dtypes
        rtol, atol = 1e-5, 1e-5

    return rtol * scale, atol * scale


# =============================================================================
# Fixtures for parameterized dtype/device testing
# =============================================================================


@pytest.fixture(params=[torch.float16, torch.bfloat16, torch.float32, torch.float64])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Parameterized fixture for testing with different floating-point precisions.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest fixture request object.

    Returns
    -------
    torch.dtype
        The dtype to use for tensor operations.
        Includes: float16, bfloat16, float32, float64.
    """
    return request.param


@pytest.fixture(params=["cpu", "cuda"])
def device(request: pytest.FixtureRequest) -> torch.device:
    """Parameterized fixture for testing on CPU and GPU.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest fixture request object.

    Returns
    -------
    torch.device
        The device to use for tensor operations.

    Notes
    -----
    Automatically skips CUDA tests if CUDA is not available.
    """
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


@pytest.fixture(
    params=[
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 2),
        (3, 1),
        (3, 2),
        (3, 3),
        (4, 2),
        (4, 4),
    ]
)
def lmax_mmax(request: pytest.FixtureRequest) -> tuple[int, int]:
    """Parameterized fixture for testing with different lmax/mmax configurations.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest fixture request object.

    Returns
    -------
    tuple[int, int]
        Tuple of (lmax, mmax) values for spherical harmonic configurations.

    Notes
    -----
    Covers a wide range of configurations including edge cases:
    - (0, 0): Scalar only
    - (1, 0), (1, 1): Vector representations
    - Higher order combinations for comprehensive testing
    """
    return request.param
