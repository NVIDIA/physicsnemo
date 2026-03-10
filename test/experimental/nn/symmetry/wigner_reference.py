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
#
# This file contains code derived from `fairchem` found at
# https://github.com/facebookresearch/fairchem.
# Copyright (c) [2025] Meta, Inc. and its affiliates.
# Licensed under MIT License.

r"""
Reference values computed with SymPy for Wigner d-matrix and J-matrix testing.

These constants are for regression testing Wigner d-matrix implementations.

Convention
----------
The matrices use the following index convention:

- Row index i corresponds to m = l - i (m goes from l to -l as i goes from 0 to 2l)
- Column index j corresponds to m' = l - j
- D_MATRIX_L{l}_BETA_{name}[i][j] = d^l_{l-i, l-j}(beta)

This matches SymPy's wigner_d_small convention where wigner_d_small(l, beta)[l-mp, l-m]
gives the element d^l_{m,m'}(beta).

Wigner d-matrix Definition
--------------------------
The small Wigner d-matrix d^l(beta) is the rotation matrix for rotating spherical
harmonics by angle beta about the y-axis. For angular momentum l, the matrix has
dimension (2l+1) x (2l+1).

J-matrix Definition (Involution Matrix)
---------------------------------------
The J matrix is used in the factored Wigner D-matrix formula::

    D(alpha, beta, gamma) = Z(alpha) @ J @ Z(beta) @ J @ Z(gamma)

where Z(theta) is a diagonal matrix with exp(i*m*theta) on the diagonal.

The J matrix is defined as::

    J^l = diag((-1)^i) * d^l(pi/2)

Key property: J @ J = I (involution)

Symbolic Values at beta = pi/2
------------------------------
For l=1::

    d^1(pi/2) = | 1/2        sqrt(2)/2  1/2      |
               | -sqrt(2)/2  0          sqrt(2)/2 |
               | 1/2        -sqrt(2)/2  1/2      |

For l=2::

    d^2(pi/2) = | 1/4        1/2        sqrt(6)/4  1/2        1/4      |
               | -1/2       -1/2       0          1/2        1/2      |
               | sqrt(6)/4   0         -1/2       0          sqrt(6)/4 |
               | -1/2       1/2        0         -1/2        1/2      |
               | 1/4       -1/2        sqrt(6)/4 -1/2        1/4      |

For l=3 (partial)::

    d^3_{3,0}(pi/2) = -sqrt(5)/4 ~ -0.559017
    d^3_{0,0}(pi/2) = 0
    d^3_{2,1}(pi/2) = -sqrt(10)/8 ~ -0.395285

Generated via
-------------
The values were computed using SymPy's ``sympy.physics.wigner.wigner_d_small``
function with high precision (20+ significant figures).
"""

import torch

# =============================================================================
# Mathematical Constants (High Precision)
# =============================================================================
SQRT2_OVER_2 = 0.7071067811865476  # sqrt(2)/2
SQRT6_OVER_4 = 0.6123724356957945  # sqrt(6)/4
SQRT5_OVER_4 = 0.5590169943749474  # sqrt(5)/4
SQRT3_OVER_2 = 0.8660254037844387  # sqrt(3)/2
SQRT3_OVER_4 = 0.4330127018922193  # sqrt(3)/4
SQRT10_OVER_4 = 0.7905694150420949  # sqrt(10)/4
SQRT10_OVER_8 = 0.3952847075210474  # sqrt(10)/8
SQRT15_OVER_4 = 0.9682458365518543  # sqrt(15)/4
SQRT15_OVER_8 = 0.4841229182759271  # sqrt(15)/8


# =============================================================================
# Wigner d-matrices at beta = 0 (Identity)
# =============================================================================

D_MATRIX_L0_BETA_0 = torch.tensor(
    [
        [1.0],
    ],
    dtype=torch.float64,
)

D_MATRIX_L1_BETA_0 = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=torch.float64,
)

D_MATRIX_L2_BETA_0 = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.float64,
)

D_MATRIX_L3_BETA_0 = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.float64,
)


# =============================================================================
# Wigner d-matrices at beta = pi/6 (30 degrees)
# =============================================================================

D_MATRIX_L1_BETA_PI_6 = torch.tensor(
    [
        [0.9330127018922193, 0.35355339059327379, 0.066987298107780674],
        [-0.35355339059327379, 0.8660254037844386, 0.35355339059327379],
        [0.066987298107780674, -0.35355339059327379, 0.9330127018922193],
    ],
    dtype=torch.float64,
)

D_MATRIX_L2_BETA_PI_6 = torch.tensor(
    [
        [
            0.8705127018922193,
            0.46650635094610965,
            0.15309310892394862,
            0.033493649053890337,
            0.0044872981077806766,
        ],
        [
            -0.46650635094610965,
            0.6830127018922193,
            0.5303300858899106,
            0.18301270189221933,
            0.033493649053890337,
        ],
        [
            0.15309310892394862,
            -0.5303300858899106,
            0.625,
            0.5303300858899106,
            0.15309310892394862,
        ],
        [
            -0.033493649053890337,
            0.18301270189221933,
            -0.5303300858899106,
            0.6830127018922193,
            0.46650635094610965,
        ],
        [
            0.0044872981077806766,
            -0.033493649053890337,
            0.15309310892394862,
            -0.46650635094610965,
            0.8705127018922193,
        ],
    ],
    dtype=torch.float64,
)


# =============================================================================
# Wigner d-matrices at beta = pi/4 (45 degrees)
# =============================================================================

D_MATRIX_L1_BETA_PI_4 = torch.tensor(
    [
        [0.85355339059327373, 0.5, 0.14644660940672624],
        [-0.5, 0.70710678118654757, 0.5],
        [0.14644660940672624, -0.5, 0.85355339059327373],
    ],
    dtype=torch.float64,
)

D_MATRIX_L2_BETA_PI_4 = torch.tensor(
    [
        [
            0.72855339059327373,
            0.60355339059327373,
            0.30618621784789724,
            0.10355339059327376,
            0.021446609406726238,
        ],
        [
            -0.60355339059327373,
            0.35355339059327379,
            0.61237243569579447,
            0.35355339059327379,
            0.10355339059327376,
        ],
        [
            0.30618621784789724,
            -0.61237243569579447,
            0.25,
            0.61237243569579447,
            0.30618621784789724,
        ],
        [
            -0.10355339059327376,
            0.35355339059327379,
            -0.61237243569579447,
            0.35355339059327379,
            0.60355339059327373,
        ],
        [
            0.021446609406726238,
            -0.10355339059327376,
            0.30618621784789724,
            -0.60355339059327373,
            0.72855339059327373,
        ],
    ],
    dtype=torch.float64,
)


# =============================================================================
# Wigner d-matrices at beta = pi/3 (60 degrees)
# =============================================================================

D_MATRIX_L1_BETA_PI_3 = torch.tensor(
    [
        [0.75, 0.61237243569579447, 0.25],
        [-0.61237243569579447, 0.5, 0.61237243569579447],
        [0.25, -0.61237243569579447, 0.75],
    ],
    dtype=torch.float64,
)

D_MATRIX_L2_BETA_PI_3 = torch.tensor(
    [
        [0.5625, 0.649519052838329, 0.45927932677184591, 0.21650635094610965, 0.0625],
        [-0.649519052838329, 0.0, 0.5303300858899106, 0.5, 0.21650635094610965],
        [
            0.45927932677184591,
            -0.5303300858899106,
            -0.125,
            0.5303300858899106,
            0.45927932677184591,
        ],
        [-0.21650635094610965, 0.5, -0.5303300858899106, 0.0, 0.649519052838329],
        [0.0625, -0.21650635094610965, 0.45927932677184591, -0.649519052838329, 0.5625],
    ],
    dtype=torch.float64,
)


# =============================================================================
# Wigner d-matrices at beta = pi/2 (90 degrees) - Most important for J-matrix
# =============================================================================

D_MATRIX_L0_BETA_PI_2 = torch.tensor(
    [
        [1.0],
    ],
    dtype=torch.float64,
)

D_MATRIX_L1_BETA_PI_2 = torch.tensor(
    [
        [0.5, 0.70710678118654757, 0.5],
        [-0.70710678118654757, 0.0, 0.70710678118654757],
        [0.5, -0.70710678118654757, 0.5],
    ],
    dtype=torch.float64,
)

D_MATRIX_L2_BETA_PI_2 = torch.tensor(
    [
        [0.25, 0.5, 0.61237243569579447, 0.5, 0.25],
        [-0.5, -0.5, 0.0, 0.5, 0.5],
        [0.61237243569579447, 0.0, -0.5, 0.0, 0.61237243569579447],
        [-0.5, 0.5, 0.0, -0.5, 0.5],
        [0.25, -0.5, 0.61237243569579447, -0.5, 0.25],
    ],
    dtype=torch.float64,
)

D_MATRIX_L3_BETA_PI_2 = torch.tensor(
    [
        [
            0.125,
            0.30618621784789724,
            0.48412291827592713,
            0.55901699437494745,
            0.48412291827592713,
            0.30618621784789724,
            0.125,
        ],
        [
            -0.30618621784789724,
            -0.5,
            -0.39528470752104744,
            0.0,
            0.39528470752104744,
            0.5,
            0.30618621784789724,
        ],
        [
            0.48412291827592713,
            0.39528470752104744,
            -0.125,
            -0.4330127018922193,
            -0.125,
            0.39528470752104744,
            0.48412291827592713,
        ],
        [
            -0.55901699437494745,
            0.0,
            0.4330127018922193,
            0.0,
            -0.4330127018922193,
            0.0,
            0.55901699437494745,
        ],
        [
            0.48412291827592713,
            -0.39528470752104744,
            -0.125,
            0.4330127018922193,
            -0.125,
            -0.39528470752104744,
            0.48412291827592713,
        ],
        [
            -0.30618621784789724,
            0.5,
            -0.39528470752104744,
            0.0,
            0.39528470752104744,
            -0.5,
            0.30618621784789724,
        ],
        [
            0.125,
            -0.30618621784789724,
            0.48412291827592713,
            -0.55901699437494745,
            0.48412291827592713,
            -0.30618621784789724,
            0.125,
        ],
    ],
    dtype=torch.float64,
)


# =============================================================================
# Wigner d-matrices at beta = pi (180 degrees) - Anti-diagonal permutation with signs
# =============================================================================

D_MATRIX_L0_BETA_PI = torch.tensor(
    [
        [1.0],
    ],
    dtype=torch.float64,
)

D_MATRIX_L1_BETA_PI = torch.tensor(
    [
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=torch.float64,
)

D_MATRIX_L2_BETA_PI = torch.tensor(
    [
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=torch.float64,
)

D_MATRIX_L3_BETA_PI = torch.tensor(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=torch.float64,
)


# =============================================================================
# J-matrices (Involution matrices): J^l = diag((-1)^i) * d^l(pi/2)
# Property: J @ J = I
# =============================================================================

J_MATRIX_L0 = torch.tensor(
    [
        [1.0],
    ],
    dtype=torch.float64,
)

J_MATRIX_L1 = torch.tensor(
    [
        [0.5, 0.70710678118654757, 0.5],
        [0.70710678118654757, 0.0, -0.70710678118654757],
        [0.5, -0.70710678118654757, 0.5],
    ],
    dtype=torch.float64,
)

J_MATRIX_L2 = torch.tensor(
    [
        [0.25, 0.5, 0.61237243569579447, 0.5, 0.25],
        [0.5, 0.5, 0.0, -0.5, -0.5],
        [0.61237243569579447, 0.0, -0.5, 0.0, 0.61237243569579447],
        [0.5, -0.5, 0.0, 0.5, -0.5],
        [0.25, -0.5, 0.61237243569579447, -0.5, 0.25],
    ],
    dtype=torch.float64,
)

J_MATRIX_L3 = torch.tensor(
    [
        [
            0.125,
            0.30618621784789724,
            0.48412291827592713,
            0.55901699437494745,
            0.48412291827592713,
            0.30618621784789724,
            0.125,
        ],
        [
            0.30618621784789724,
            0.5,
            0.39528470752104744,
            0.0,
            -0.39528470752104744,
            -0.5,
            -0.30618621784789724,
        ],
        [
            0.48412291827592713,
            0.39528470752104744,
            -0.125,
            -0.4330127018922193,
            -0.125,
            0.39528470752104744,
            0.48412291827592713,
        ],
        [
            0.55901699437494745,
            0.0,
            -0.4330127018922193,
            0.0,
            0.4330127018922193,
            0.0,
            -0.55901699437494745,
        ],
        [
            0.48412291827592713,
            -0.39528470752104744,
            -0.125,
            0.4330127018922193,
            -0.125,
            -0.39528470752104744,
            0.48412291827592713,
        ],
        [
            0.30618621784789724,
            -0.5,
            0.39528470752104744,
            0.0,
            -0.39528470752104744,
            0.5,
            -0.30618621784789724,
        ],
        [
            0.125,
            -0.30618621784789724,
            0.48412291827592713,
            -0.55901699437494745,
            0.48412291827592713,
            -0.30618621784789724,
            0.125,
        ],
    ],
    dtype=torch.float64,
)


# =============================================================================
# Specific element values for edge case testing
# Format: D_L{l}_{m}_{mp}_BETA_{name}
# Note: For negative m or mp, we use N for "negative", e.g., D_L1_N1_0 = d^1_{-1,0}
# =============================================================================

# l=1 elements at beta=pi/2
D_L1_1_0_BETA_PI_2 = -0.70710678118654757  # d^1_{1,0}(pi/2) = -sqrt(2)/2
D_L1_0_1_BETA_PI_2 = 0.70710678118654757  # d^1_{0,1}(pi/2) = sqrt(2)/2
D_L1_N1_0_BETA_PI_2 = 0.70710678118654757  # d^1_{-1,0}(pi/2) = sqrt(2)/2

# l=2 elements at beta=pi/2
D_L2_2_0_BETA_PI_2 = 0.61237243569579447  # d^2_{2,0}(pi/2) = sqrt(6)/4
D_L2_0_2_BETA_PI_2 = 0.61237243569579447  # d^2_{0,2}(pi/2) = sqrt(6)/4
D_L2_0_0_BETA_PI_2 = -0.5  # d^2_{0,0}(pi/2) = -1/2
D_L2_1_1_BETA_PI_2 = -0.5  # d^2_{1,1}(pi/2) = -1/2
D_L2_N2_0_BETA_PI_2 = 0.61237243569579447  # d^2_{-2,0}(pi/2) = sqrt(6)/4

# l=3 elements at beta=pi/2
D_L3_3_0_BETA_PI_2 = -0.55901699437494745  # d^3_{3,0}(pi/2) = -sqrt(5)/4
D_L3_0_0_BETA_PI_2 = 0.0  # d^3_{0,0}(pi/2) = 0
D_L3_2_1_BETA_PI_2 = -0.39528470752104744  # d^3_{2,1}(pi/2) = -sqrt(10)/8
D_L3_1_0_BETA_PI_2 = 0.4330127018922193  # d^3_{1,0}(pi/2) = sqrt(3)/4


# =============================================================================
# Verification Functions (using PyTorch)
# =============================================================================


def verify_d_matrix_orthogonality(D: torch.Tensor, tol: float = 1e-10) -> bool:
    """
    Verify that D^T @ D = I (orthogonality).

    The Wigner d-matrix is a real orthogonal matrix.

    Parameters
    ----------
    D : torch.Tensor
        The d-matrix to verify orthogonality for.
    tol : float
        Tolerance for numerical comparison.

    Returns
    -------
    bool
        True if the matrix is orthogonal within tolerance.
    """
    DTD = D.T @ D
    identity = torch.eye(len(D), dtype=D.dtype, device=D.device)
    return torch.allclose(DTD, identity, atol=tol, rtol=0.0)


def verify_d_matrix_symmetry(ell: int, D: torch.Tensor, tol: float = 1e-10) -> bool:
    """
    Verify the symmetry relation: d^l_{m,m'}(beta) = (-1)^{m-m'} * d^l_{m',m}(beta).

    Parameters
    ----------
    ell : int
        Angular momentum quantum number.
    D : torch.Tensor
        The d-matrix D[i,j] = d^l_{l-i, l-j}(beta)
    tol : float
        Tolerance for numerical comparison.

    Returns
    -------
    bool
        True if the symmetry relation holds within tolerance.
    """
    size = 2 * ell + 1
    for i in range(size):
        for j in range(size):
            m = ell - i
            mp = ell - j
            sign = (-1) ** (m - mp)
            if not torch.isclose(
                D[i, j], torch.tensor(sign, dtype=D.dtype) * D[j, i], atol=tol
            ):
                return False
    return True


def verify_j_matrix_involution(J: torch.Tensor, tol: float = 1e-12) -> bool:
    """
    Verify that J @ J = I (involution property).

    The J matrix is an involution matrix used in the factored D-matrix formula.

    Parameters
    ----------
    J : torch.Tensor
        The J matrix to verify.
    tol : float
        Tolerance for numerical comparison.

    Returns
    -------
    bool
        True if J @ J = I within tolerance.
    """
    JJ = J @ J
    identity = torch.eye(len(J), dtype=J.dtype, device=J.device)
    return torch.allclose(JJ, identity, atol=tol, rtol=0.0)


def verify_j_matrix_from_d(
    ell: int, J: torch.Tensor, D_pi_2: torch.Tensor, tol: float = 1e-12
) -> bool:
    """
    Verify that J = diag((-1)^i) @ d(pi/2).

    Parameters
    ----------
    ell : int
        Angular momentum quantum number.
    J : torch.Tensor
        The J matrix to verify.
    D_pi_2 : torch.Tensor
        The d-matrix at beta=pi/2.
    tol : float
        Tolerance for numerical comparison.

    Returns
    -------
    bool
        True if J = diag((-1)^i) @ d(pi/2) within tolerance.
    """
    size = 2 * ell + 1
    signs = torch.tensor(
        [(-1.0) ** i for i in range(size)], dtype=J.dtype, device=J.device
    )
    J_computed = torch.diag(signs) @ D_pi_2
    return torch.allclose(J, J_computed, atol=tol, rtol=0.0)


# =============================================================================
# Lookup dictionaries for parameterized tests
# =============================================================================

D_MATRICES_BETA_0 = {
    0: D_MATRIX_L0_BETA_0,
    1: D_MATRIX_L1_BETA_0,
    2: D_MATRIX_L2_BETA_0,
    3: D_MATRIX_L3_BETA_0,
}

D_MATRICES_BETA_PI_2 = {
    0: D_MATRIX_L0_BETA_PI_2,
    1: D_MATRIX_L1_BETA_PI_2,
    2: D_MATRIX_L2_BETA_PI_2,
    3: D_MATRIX_L3_BETA_PI_2,
}

D_MATRICES_BETA_PI = {
    0: D_MATRIX_L0_BETA_PI,
    1: D_MATRIX_L1_BETA_PI,
    2: D_MATRIX_L2_BETA_PI,
    3: D_MATRIX_L3_BETA_PI,
}

J_MATRICES = {
    0: J_MATRIX_L0,
    1: J_MATRIX_L1,
    2: J_MATRIX_L2,
    3: J_MATRIX_L3,
}


if __name__ == "__main__":
    # Self-verification on import
    print("Verifying reference values...")

    # Verify orthogonality
    for ell in range(4):
        D = D_MATRICES_BETA_PI_2[ell]
        assert verify_d_matrix_orthogonality(D), f"Orthogonality failed for l={ell}"
        print(f"  l={ell}: d-matrix orthogonality OK")

    # Verify symmetry
    for ell in range(4):
        D = D_MATRICES_BETA_PI_2[ell]
        assert verify_d_matrix_symmetry(ell, D), f"Symmetry failed for l={ell}"
        print(f"  l={ell}: d-matrix symmetry OK")

    # Verify J involution
    for ell in range(4):
        J = J_MATRICES[ell]
        assert verify_j_matrix_involution(J), f"Involution failed for l={ell}"
        print(f"  l={ell}: J-matrix involution OK")

    # Verify J = diag @ d(pi/2)
    for ell in range(4):
        J = J_MATRICES[ell]
        D = D_MATRICES_BETA_PI_2[ell]
        assert verify_j_matrix_from_d(ell, J, D), f"J construction failed for l={ell}"
        print(f"  l={ell}: J-matrix construction OK")

    print("\nAll verifications passed!")
