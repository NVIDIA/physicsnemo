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

"""On-the-fly Wigner D-matrix computation for edge-aligned rotations.

This module provides the :class:`EdgeRotation` module for computing Wigner D-matrices
that rotate spherical harmonic coefficients between the global frame and edge-aligned
local frames. This enables SO(3) equivariance using only SO(2) convolutions.

The implementation computes J matrices on-the-fly at initialization and uses the factored formula:

.. math::

    D^l(\\alpha, \\beta, \\gamma) = Z(\\alpha) \\cdot J \\cdot Z(\\beta) \\cdot J \\cdot Z(\\gamma)

where :math:`Z(\\phi)` is the z-axis rotation matrix and :math:`J` is the transformation
matrix satisfying :math:`J^2 = I`.

The computation exploits the sparse structure of :math:`Z(\\phi)` matrices (only diagonal
and anti-diagonal elements are non-zero) to reduce from 4 matrix multiplications to 1,
plus efficient batched element-wise operations.

Key Components
--------------
:class:`EdgeRotation`
    Module that computes Wigner D-matrices from edge direction vectors.
:func:`edge_vectors_to_euler_angles`
    Convert edge direction vectors to Euler angles (ZYZ convention).
:func:`rotate_grid_coefficients`
    Apply Wigner D-matrix rotation to grid-layout spherical harmonic coefficients.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from jaxtyping import Bool, Float

from physicsnemo.nn import Module

__all__ = [
    "EdgeRotation",
    "edge_vectors_to_euler_angles",
    "rotate_grid_coefficients",
    "compute_wigner_d_matrices",
]

# Numerical stability constant
_EPS = 1e-7


# =============================================================================
# Numerically stable trigonometric functions with gradients
# =============================================================================


class _SafeAcos(torch.autograd.Function):
    """Safe arccos with stable gradients near +/- 1."""

    @staticmethod
    def forward(ctx, x):  # type: ignore[override]
        x_clamped = x.clamp(-1 + _EPS, 1 - _EPS)
        ctx.save_for_backward(x_clamped)
        return torch.acos(x.clamp(-1.0, 1.0))

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        (x_clamped,) = ctx.saved_tensors
        denom = torch.sqrt(1 - x_clamped.pow(2)).clamp(min=_EPS)
        return -grad_output / denom


class _SafeAtan2(torch.autograd.Function):
    """Safe atan2 with stable gradients."""

    @staticmethod
    def forward(ctx, y, x):  # type: ignore[override]
        ctx.save_for_backward(y, x)
        return torch.atan2(y, x)

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_output):  # type: ignore[override]
        y, x = ctx.saved_tensors
        denom = (x.pow(2) + y.pow(2)).clamp(min=_EPS)
        return (x / denom) * grad_output, (-y / denom) * grad_output


def _safe_acos(x: torch.Tensor) -> torch.Tensor:
    """Compute arccos with stable gradients near +/- 1."""
    result: torch.Tensor = _SafeAcos.apply(x)  # type: ignore[assignment]
    return result


def _safe_atan2(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute atan2 with stable gradients."""
    result: torch.Tensor = _SafeAtan2.apply(y, x)  # type: ignore[assignment]
    return result


# =============================================================================
# Edge vector to Euler angle conversion
# =============================================================================


def edge_vectors_to_euler_angles(
    edge_vecs: Float[torch.Tensor, "... 3"],
) -> tuple[
    Float[torch.Tensor, "..."],  # alpha
    Float[torch.Tensor, "..."],  # beta
    Float[torch.Tensor, "..."],  # gamma
]:
    """Convert edge direction vectors to Euler angles (ZYZ convention).

    Computes Euler angles that rotate the z-axis to align with the given
    edge direction. The gamma angle is always zero since edge rotations
    only require specifying a direction, not a roll.

    Parameters
    ----------
    edge_vecs : Float[torch.Tensor, "... 3"]
        Edge direction vectors (not necessarily normalized).
        Shape (..., 3) where last dimension is (x, y, z).

    Returns
    -------
    alpha : Float[torch.Tensor, "..."]
        Azimuthal angle in radians, range [−π, π].
    beta : Float[torch.Tensor, "..."]
        Polar angle in radians, range [0, π].
    gamma : Float[torch.Tensor, "..."]
        Roll angle in radians (always zero for edge rotations).

    Examples
    --------
    >>> import torch
    >>> edge = torch.tensor([[0.0, 1.0, 0.0]])  # y-direction
    >>> alpha, beta, gamma = edge_vectors_to_euler_angles(edge)
    >>> beta.item()  # Should be ~0 (pointing along y)
    0.0

    Notes
    -----
    The convention uses:

    - alpha is the longitude (atan2(x, z))
    - beta is the latitude (acos(y))
    - gamma is set to 0 (no roll for edge-aligned frames)
    """
    # Normalize edge vectors with numerical stability
    norm = torch.norm(edge_vecs, dim=-1, keepdim=True).clamp(min=_EPS)
    xyz = edge_vecs / norm

    # Clamp for numerical stability
    xyz = xyz.clamp(-1.0, 1.0)

    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    # Beta is the polar angle (latitude) from y-axis
    beta = _safe_acos(y)

    # Alpha is the azimuthal angle (longitude) in xz-plane
    alpha = _safe_atan2(x, z)

    # Gamma is zero for edge rotations (no roll)
    gamma = torch.zeros_like(alpha)

    return alpha, beta, gamma


# =============================================================================
# Small Wigner d-matrix computation (used to compute J matrices at init)
# =============================================================================


def _compute_d_matrix_l1(
    beta: torch.Tensor,
    c: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    """Compute d-matrix for l=1 using closed-form expressions."""
    cb = torch.cos(beta)
    sb = torch.sin(beta)
    sqrt2 = 2**0.5

    d = torch.zeros((*beta.shape, 3, 3), dtype=beta.dtype, device=beta.device)

    # Row 0: m=1
    d[..., 0, 0] = c * c
    d[..., 0, 1] = sqrt2 * sb / 2
    d[..., 0, 2] = s * s

    # Row 1: m=0
    d[..., 1, 0] = -sqrt2 * sb / 2
    d[..., 1, 1] = cb
    d[..., 1, 2] = sqrt2 * sb / 2

    # Row 2: m=-1
    d[..., 2, 0] = s * s
    d[..., 2, 1] = -sqrt2 * sb / 2
    d[..., 2, 2] = c * c

    return d


def _compute_d_matrix_l2(
    beta: torch.Tensor,
    c: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    """Compute d-matrix for l=2 using closed-form expressions."""
    cb = torch.cos(beta)
    sb = torch.sin(beta)
    sqrt6 = 6**0.5
    c2 = c * c
    c3 = c2 * c
    c4 = c2 * c2
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    sb2 = sb * sb
    cos2b = torch.cos(2 * beta)
    sin2b = torch.sin(2 * beta)

    d = torch.zeros((*beta.shape, 5, 5), dtype=beta.dtype, device=beta.device)

    # Row 0: m=2
    d[..., 0, 0] = c4
    d[..., 0, 1] = 2 * s * c3
    d[..., 0, 2] = sqrt6 * sb2 / 4
    d[..., 0, 3] = 2 * s3 * c
    d[..., 0, 4] = s4

    # Row 1: m=1
    d[..., 1, 0] = -2 * s * c3
    d[..., 1, 1] = cb / 2 + cos2b / 2
    d[..., 1, 2] = sqrt6 * sin2b / 4
    d[..., 1, 3] = cb / 2 - cos2b / 2
    d[..., 1, 4] = 2 * s3 * c

    # Row 2: m=0
    d[..., 2, 0] = sqrt6 * sb2 / 4
    d[..., 2, 1] = -sqrt6 * sin2b / 4
    d[..., 2, 2] = 1 - 3 * sb2 / 2
    d[..., 2, 3] = sqrt6 * sin2b / 4
    d[..., 2, 4] = sqrt6 * sb2 / 4

    # Row 3: m=-1
    d[..., 3, 0] = -2 * s3 * c
    d[..., 3, 1] = cb / 2 - cos2b / 2
    d[..., 3, 2] = -sqrt6 * sin2b / 4
    d[..., 3, 3] = cb / 2 + cos2b / 2
    d[..., 3, 4] = 2 * s * c3

    # Row 4: m=-2
    d[..., 4, 0] = s4
    d[..., 4, 1] = -2 * s3 * c
    d[..., 4, 2] = sqrt6 * sb2 / 4
    d[..., 4, 3] = -2 * s * c3
    d[..., 4, 4] = c4

    return d


def _compute_d_element(
    ell: int,
    m: int,
    mp: int,
    c: torch.Tensor,
    s: torch.Tensor,
    factorials: list,
) -> torch.Tensor:
    """Compute a single element of the d-matrix using Wigner formula."""
    # Prefactor
    prefactor = math.sqrt(
        factorials[ell + m]
        * factorials[ell - m]
        * factorials[ell + mp]
        * factorials[ell - mp]
    )

    # Sum over k
    k_min = max(0, m - mp)
    k_max = min(ell + m, ell - mp)

    result = torch.zeros_like(c)
    for k in range(k_min, k_max + 1):
        denom = (
            factorials[ell + m - k]
            * factorials[ell - mp - k]
            * factorials[k + mp - m]
            * factorials[k]
        )
        sign = (-1) ** (m - mp + k)
        exp_c = 2 * ell + mp - m - 2 * k
        exp_s = m - mp + 2 * k
        term = sign * prefactor / denom * (c**exp_c) * (s**exp_s)
        result = result + term

    return result


def _compute_d_matrix_from_lower(
    ell: int,
    beta: torch.Tensor,
    c: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    """Compute d^l for l > 2 using closed-form factorial expression."""
    dim = 2 * ell + 1
    d = torch.zeros((*beta.shape, dim, dim), dtype=beta.dtype, device=beta.device)

    # Precompute factorials
    factorials = [1.0]
    for i in range(1, 2 * ell + 2):
        factorials.append(factorials[-1] * i)

    for mi in range(dim):
        m = ell - mi  # m goes from l to -l
        for mpi in range(dim):
            mp = ell - mpi  # m' goes from l to -l
            d[..., mi, mpi] = _compute_d_element(ell, m, mp, c, s, factorials)

    return d


def _compute_d_matrix(
    ell: int,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Compute small Wigner d-matrix for angular momentum l and angle beta.

    Used internally to compute J matrices at initialization.
    """
    if ell == 0:
        return torch.ones((*beta.shape, 1, 1), dtype=beta.dtype, device=beta.device)

    c = torch.cos(beta / 2)
    s = torch.sin(beta / 2)

    if ell == 1:
        return _compute_d_matrix_l1(beta, c, s)
    elif ell == 2:
        return _compute_d_matrix_l2(beta, c, s)
    else:
        return _compute_d_matrix_from_lower(ell, beta, c, s)


# =============================================================================
# J matrix and Z-rotation matrix computation
# =============================================================================


def _compute_J_matrix(
    ell: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute the J matrix for angular momentum l.

    The J matrix relates z-axis and y-axis rotations:

    .. math::

        D^l(\\alpha, \\beta, \\gamma) = Z(\\alpha) \\cdot J \\cdot Z(\\beta) \\cdot J \\cdot Z(\\gamma)

    It is computed as:

    .. math::

        J^l = \\text{diag}((-1)^{l-m}) \\cdot d^l(\\pi/2)

    where m ranges from l to -l. This ensures J @ J = I (J is an involution).
    """
    target_device = device if device is not None else torch.device("cpu")

    pi_half = torch.tensor([torch.pi / 2], dtype=dtype, device=target_device)
    d_pi2 = _compute_d_matrix(ell, pi_half).squeeze(0)

    # Compute sign factors: (-1)^{l-m} = (-1)^i for row index i
    dim = 2 * ell + 1
    signs = torch.tensor(
        [(-1) ** i for i in range(dim)],
        dtype=dtype,
        device=target_device,
    )

    # Apply sign correction: J = diag(signs) @ d(pi/2)
    J = signs.unsqueeze(1) * d_pi2

    return J


# =============================================================================
# Grid-layout coefficient rotation utility
# =============================================================================


def _apply_rotation_to_grid(
    x: Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 channels"],
    D: Float[torch.Tensor, "batch full_dim full_dim"],
    lmax: int,
    mmax: int,
) -> Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 channels"]:
    """Apply a pre-computed Wigner D-matrix to grid-layout spherical harmonic coefficients.

    This implementation uses real arithmetic only, and the rotation is
    applied to both real and complex components.

    Parameters
    ----------
    x : Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 channels"]
        Input coefficients in grid layout.
    D : Float[torch.Tensor, "batch full_dim full_dim"]
        Pre-computed block-diagonal Wigner D-matrix from compute_wigner_d_matrices.
        Shape [batch, (lmax+1)^2, (lmax+1)^2] with real values.
    lmax : int
        Maximum spherical harmonic degree.
    mmax : int
        Maximum spherical harmonic order in the grid layout.

    Returns
    -------
    Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 channels"]
        Rotated coefficients in grid layout.

    Notes
    -----
    The grid layout stores only m >= 0 with real and imaginary parts separated:
    - x[:, l, m, 0, :] = real part of Y_l^m
    - x[:, l, m, 1, :] = imaginary part of Y_l^m

    For real spherical harmonics, the relation for negative m is:
    - Y_l^{-m} real part = (-1)^m * Y_l^m real part
    - Y_l^{-m} imag part = (-1)^{m+1} * Y_l^m imag part
    """
    batch_size = x.shape[0]
    device = x.device
    dtype = x.dtype
    channels = x.shape[4]

    # Initialize output
    out = torch.zeros_like(x)

    # Process each degree l separately
    offset = 0  # Track position in the full D-matrix
    for ell in range(lmax + 1):
        full_dim_l = 2 * ell + 1
        m_limit = min(mmax, ell)

        # Extract D^l block from the full block-diagonal D-matrix
        D_l = D[:, offset : offset + full_dim_l, offset : offset + full_dim_l]

        # Build full representation for this l: [batch, 2*l+1, 2, channels]
        # Ordering: m = l, l-1, ..., 0, ..., -l
        x_l_full = torch.zeros(
            batch_size, full_dim_l, 2, channels, dtype=dtype, device=device
        )

        # Fill positive m (and m=0) from grid
        for m in range(m_limit + 1):
            idx = (
                ell - m
            )  # Position in full ordering (m=l is at idx=0, m=0 is at idx=l)
            x_l_full[:, idx, 0, :] = x[:, ell, m, 0, :]  # real part
            x_l_full[:, idx, 1, :] = x[:, ell, m, 1, :]  # imag part

        # Fill negative m using Y_l^{-m} = (-1)^m * conj(Y_l^m)
        # For real SH: real(-m) = (-1)^m * real(m), imag(-m) = (-1)^{m+1} * imag(m)
        for m in range(1, m_limit + 1):
            idx_neg = ell + m  # Position for -m in full ordering
            sign_real = (-1) ** m
            sign_imag = (-1) ** (m + 1)
            x_l_full[:, idx_neg, 0, :] = sign_real * x[:, ell, m, 0, :]
            x_l_full[:, idx_neg, 1, :] = sign_imag * x[:, ell, m, 1, :]

        # Apply rotation using real arithmetic only
        # D_l is a real matrix: [batch, full_dim_l, full_dim_l]
        # x_l_full: [batch, full_dim_l, 2, channels]
        # We want: y[b, i, ri, c] = sum_j D[b, i, j] * x[b, j, ri, c]
        # Reshape for batch matrix multiplication: [batch, full_dim_l, 2*channels]
        x_l_flat = x_l_full.reshape(batch_size, full_dim_l, 2 * channels)

        # Apply D-matrix: simple batch matrix multiplication (no complex arithmetic!)
        y_l_flat = torch.bmm(D_l, x_l_flat)  # [batch, full_dim_l, 2*channels]

        # Reshape back: [batch, full_dim_l, 2, channels]
        y_l_full = y_l_flat.reshape(batch_size, full_dim_l, 2, channels)

        # Extract m >= 0 back to grid layout
        for m in range(m_limit + 1):
            idx = ell - m
            out[:, ell, m, 0, :] = y_l_full[:, idx, 0, :]  # real part
            out[:, ell, m, 1, :] = y_l_full[:, idx, 1, :]  # imag part

        # Move to next block
        offset += full_dim_l

    # Enforce m=0 reality constraint: imaginary part must be zero
    # This is a fundamental property of spherical harmonics Y_l^0
    out[:, :, 0, 1, :] = 0.0

    return out


def rotate_grid_coefficients(
    x: Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 channels"],
    rotation: (
        tuple[
            Float[torch.Tensor, "batch"] | float,
            Float[torch.Tensor, "batch"] | float,
            Float[torch.Tensor, "batch"] | float,
        ]
        | Float[torch.Tensor, "batch full_dim full_dim"]
    ),
) -> Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 channels"]:
    r"""Apply Wigner D-matrix rotation to grid-layout spherical harmonic coefficients.

    This function rotates spherical harmonic coefficients by applying the
    Wigner D-matrix corresponding to the given rotation. The rotation can be
    specified either as Euler angles or as a pre-computed D-matrix.

    Parameters
    ----------
    x : Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 channels"]
        Input coefficients in grid layout where:
        - batch: number of samples
        - lmax+1: degrees from l=0 to l=lmax
        - mmax+1: orders from m=0 to m=mmax (non-negative m only)
        - 2: real (index 0) and imaginary (index 1) components
        - channels: feature channels
    rotation : tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Float[torch.Tensor, "batch full_dim full_dim"]
        The rotation, specified as either:
        - A tuple of (alpha, beta, gamma) Euler angles in **radians** (ZYZ convention),
          each of shape [batch] or scalar. Valid ranges: α, γ ∈ [0, 2π), β ∈ [0, π].
        - A pre-computed block-diagonal Wigner D-matrix of shape
          [batch, (lmax+1)^2, (lmax+1)^2] from compute_wigner_d_matrices

    Returns
    -------
    Float[torch.Tensor, "batch lmax_plus_1 mmax_plus_1 2 channels"]
        Rotated coefficients in grid layout with the same shape as input.

    Examples
    --------
    Using Euler angles:

    >>> import torch
    >>> from physicsnemo.experimental.nn.symmetry.wigner import rotate_grid_coefficients
    >>> batch, lmax, mmax, channels = 4, 3, 2, 8
    >>> x = torch.randn(batch, lmax + 1, mmax + 1, 2, channels)
    >>> alpha = torch.rand(batch) * 2 * torch.pi
    >>> beta = torch.rand(batch) * torch.pi
    >>> gamma = torch.rand(batch) * 2 * torch.pi
    >>> x_rotated = rotate_grid_coefficients(x, (alpha, beta, gamma))
    >>> x_rotated.shape
    torch.Size([4, 4, 3, 2, 8])

    Using pre-computed D-matrix:

    >>> from physicsnemo.experimental.nn.symmetry.wigner import compute_wigner_d_matrices
    >>> D = compute_wigner_d_matrices(alpha, beta, gamma, lmax=3)
    >>> x_rotated = rotate_grid_coefficients(x, D)
    >>> x_rotated.shape
    torch.Size([4, 4, 3, 2, 8])

    Notes
    -----
    The Wigner D-matrix rotation is applied per-channel and per-degree l.
    Each degree l has its own (2l+1) x (2l+1) rotation matrix that mixes
    coefficients with different m values but the same l.
    """
    device = x.device
    dtype = x.dtype

    if not torch.compiler.is_compiling():
        if x.ndim != 5:
            raise ValueError(
                f"Expected 5D tensor [batch, lmax+1, mmax+1, 2, channels], got {x.ndim}D"
            )
        if x.shape[1] < x.shape[2]:
            raise ValueError(f"Expected lmax (dim 1) <= mmax (dim 2), got {x.shape}")
        if x.shape[3] != 2:
            raise ValueError(
                f"Expected dimension 3 to be 2 (real/imag), got {x.shape[3]}"
            )

    # infer values from tensor shape
    batch_size = x.shape[0]
    lmax = x.shape[1] - 1
    mmax = x.shape[2] - 1

    # Detect input type and compute D-matrix if needed
    if isinstance(rotation, tuple):
        # Euler angles provided - compute D-matrix
        alpha, beta, gamma = rotation

        # Handle scalar angles
        if isinstance(alpha, (int, float)):
            alpha = torch.full((batch_size,), float(alpha), dtype=dtype, device=device)
        if isinstance(beta, (int, float)):
            beta = torch.full((batch_size,), float(beta), dtype=dtype, device=device)
        if isinstance(gamma, (int, float)):
            gamma = torch.full((batch_size,), float(gamma), dtype=dtype, device=device)

        D = compute_wigner_d_matrices(alpha, beta, gamma, lmax)
    else:
        # Assume D-matrix provided directly
        D = rotation

        # Validate D-matrix shape
        if not torch.compiler.is_compiling():
            full_dim = (lmax + 1) ** 2
            if D.ndim != 3:
                raise ValueError(
                    f"Expected D-matrix to be 3D [batch, full_dim, full_dim], got {D.ndim}D with shape {tuple(D.shape)}"
                )
            if D.shape[0] != batch_size:
                raise ValueError(
                    f"D-matrix batch size {D.shape[0]} does not match input batch size {batch_size}"
                )
            if D.shape[1] != full_dim or D.shape[2] != full_dim:
                raise ValueError(
                    f"Expected D-matrix shape [batch, {full_dim}, {full_dim}], got [batch, {D.shape[1]}, {D.shape[2]}]"
                )

    # Apply the rotation using the helper function
    return _apply_rotation_to_grid(x, D, lmax, mmax)


# =============================================================================
# Standalone Wigner D-matrix computation
# =============================================================================


def compute_wigner_d_matrices(
    alpha: Float[torch.Tensor, "batch"],
    beta: Float[torch.Tensor, "batch"],
    gamma: Float[torch.Tensor, "batch"],
    lmax: int,
    J_matrices: list[torch.Tensor] | None = None,
) -> Float[torch.Tensor, "batch full_dim full_dim"]:
    r"""Compute block-diagonal Wigner D-matrices from Euler angles.

    This function computes Wigner D-matrices that describe rotations of spherical
    harmonic coefficients. The matrices are organized in a block-diagonal structure,
    where each block corresponds to a different angular momentum degree l.

    The computation uses the factored formula:

    .. math::

        D^l(\alpha, \beta, \gamma) = Z(\alpha) \cdot J \cdot Z(\beta) \cdot J \cdot Z(\gamma)

    where Z is the z-axis rotation matrix (diagonal with Z_mm = exp(i m φ)) and
    J is a precomputed involution matrix satisfying J^2 = I.

    Parameters
    ----------
    alpha : Float[torch.Tensor, "batch"]
        First Euler angle in radians (ZYZ convention), rotation about z-axis.
        Valid range: [0, 2π).
    beta : Float[torch.Tensor, "batch"]
        Second Euler angle in radians (ZYZ convention), rotation about y-axis.
        Valid range: [0, π].
    gamma : Float[torch.Tensor, "batch"]
        Third Euler angle in radians (ZYZ convention), rotation about z-axis.
        Valid range: [0, 2π).
    lmax : int
        Maximum spherical harmonic degree. Output will have shape
        [batch, (lmax+1)^2, (lmax+1)^2].
    J_matrices : list[torch.Tensor] | None, optional
        Pre-computed J matrices for each degree l (0 to lmax).
        If None, they will be computed on-the-fly.
        Each J_matrices[l] should have shape [2l+1, 2l+1].

    Returns
    -------
    Float[torch.Tensor, "batch full_dim full_dim"]
        Block-diagonal Wigner D-matrices where full_dim = (lmax+1)^2.
        Each batch element is an orthogonal matrix representing a rotation
        in the spherical harmonic basis.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.nn.symmetry.wigner import compute_wigner_d_matrices
    >>> batch_size = 4
    >>> lmax = 2
    >>> alpha = torch.randn(batch_size) * 2 * torch.pi
    >>> beta = torch.rand(batch_size) * torch.pi
    >>> gamma = torch.randn(batch_size) * 2 * torch.pi
    >>> D = compute_wigner_d_matrices(alpha, beta, gamma, lmax)
    >>> D.shape
    torch.Size([4, 9, 9])
    >>> # Verify orthogonality
    >>> I = torch.eye(9)
    >>> torch.allclose(D @ D.transpose(-2, -1), I.unsqueeze(0), atol=1e-5)
    True

    Notes
    -----
    The output matrices are orthogonal, meaning D @ D^T = I. The inverse
    rotation is simply the transpose.

    This function appears to be a lot more complex than it would be to
    avoid many small tensor allocations: the large tensors are allocated
    up front, and the operations are written to try and make use of them
    without re-allocating unnecessarily.
    """
    # Validate inputs
    if not torch.compiler.is_compiling():
        if alpha.ndim != 1:
            raise ValueError(
                f"Expected alpha to be 1D tensor, got {alpha.ndim}D with shape {tuple(alpha.shape)}"
            )
        if beta.ndim != 1:
            raise ValueError(
                f"Expected beta to be 1D tensor, got {beta.ndim}D with shape {tuple(beta.shape)}"
            )
        if gamma.ndim != 1:
            raise ValueError(
                f"Expected gamma to be 1D tensor, got {gamma.ndim}D with shape {tuple(gamma.shape)}"
            )
        if alpha.shape[0] != beta.shape[0] or alpha.shape[0] != gamma.shape[0]:
            raise ValueError(
                f"All Euler angles must have same batch size, got alpha={alpha.shape[0]}, "
                f"beta={beta.shape[0]}, gamma={gamma.shape[0]}"
            )
        if lmax < 0:
            raise ValueError(f"lmax must be non-negative, got {lmax}")

    batch_size = alpha.shape[0]
    device = alpha.device
    dtype = alpha.dtype
    full_dim = (lmax + 1) ** 2

    # Compute J matrices if not provided
    if J_matrices is None:
        J_matrices = []
        for ell in range(lmax + 1):
            J_l = _compute_J_matrix(ell, dtype=dtype, device=device)
            J_matrices.append(J_l)

    # Prepare padded J matrices for vectorized computation
    max_dim = 2 * lmax + 1
    num_blocks = lmax + 1

    # Pad J matrices to max_dim x max_dim
    J_padded = torch.zeros(num_blocks, max_dim, max_dim, dtype=dtype, device=device)
    for ell in range(num_blocks):
        dim_l = 2 * ell + 1
        J_l = J_matrices[ell].to(dtype=dtype, device=device)
        J_padded[ell, :dim_l, :dim_l] = J_l

    # Prepare m-values for each block
    m_vals_padded = torch.zeros(num_blocks, max_dim, dtype=dtype, device=device)
    m_flip_padded = torch.zeros(num_blocks, max_dim, dtype=dtype, device=device)
    for ell in range(num_blocks):
        dim_l = 2 * ell + 1
        m_vals_l = torch.arange(ell, -ell - 1, -1, dtype=dtype, device=device)
        m_vals_padded[ell, :dim_l] = m_vals_l
        m_flip_padded[ell, :dim_l] = -m_vals_l

    # Prepare flip indices
    flip_indices = torch.zeros(num_blocks, max_dim, dtype=torch.long, device=device)
    for ell in range(num_blocks):
        dim_l = 2 * ell + 1
        flip_indices[ell, :dim_l] = torch.arange(dim_l - 1, -1, -1, device=device)

    # Compute trigonometric values
    # alpha: (B,) -> (B, 1, 1) for broadcasting with m_vals: (num_blocks, max_dim)
    # Result: (B, num_blocks, max_dim)
    alpha_expanded = alpha.view(batch_size, 1, 1)
    beta_expanded = beta.view(batch_size, 1, 1)
    gamma_expanded = gamma.view(batch_size, 1, 1)

    cos_alpha = torch.cos(alpha_expanded * m_vals_padded)  # (B, num_blocks, max_dim)
    sin_alpha = torch.sin(alpha_expanded * m_vals_padded)
    cos_beta = torch.cos(beta_expanded * m_vals_padded)
    sin_beta_flip = torch.sin(beta_expanded * m_flip_padded)
    cos_gamma = torch.cos(gamma_expanded * m_vals_padded)
    sin_gamma_flip = torch.sin(gamma_expanded * m_flip_padded)

    # Prepare flipped J matrices for efficient computation
    ell_idx = torch.arange(num_blocks, device=device)
    J_flipped_rows = J_padded[
        ell_idx.view(-1, 1), flip_indices, :
    ]  # (num_blocks, max_dim, max_dim)

    J_flipped_cols = torch.gather(
        J_padded,
        dim=2,
        index=flip_indices.unsqueeze(1).expand(-1, max_dim, -1),
    )  # (num_blocks, max_dim, max_dim)

    # Step 1: A = Z(α) · J for all blocks
    # A[b, ell, i, k] = cos_alpha[b, ell, i] * J[ell, i, k] + sin_alpha[b, ell, i] * J[ell, flip[i], k]
    A = torch.einsum("bni,nik->bnik", cos_alpha, J_padded) + torch.einsum(
        "bni,nik->bnik", sin_alpha, J_flipped_rows
    )
    # A shape: (B, num_blocks, max_dim, max_dim)

    # Step 2: B = J · Z(γ) for all blocks
    # B[b, ell, k, j] = J[ell, k, j] * cos_gamma[b, ell, j] + J[ell, k, flip[j]] * sin_gamma_flip[b, ell, j]
    B = torch.einsum("nkj,bnj->bnkj", J_padded, cos_gamma) + torch.einsum(
        "nkj,bnj->bnkj", J_flipped_cols, sin_gamma_flip
    )
    # B shape: (B, num_blocks, max_dim, max_dim)

    # Step 3: AZ = A · Z(β) for all blocks (element-wise with flip)
    A_flipped = torch.gather(
        A,
        dim=3,
        index=flip_indices.view(1, num_blocks, 1, max_dim).expand(
            batch_size, -1, max_dim, -1
        ),
    )
    AZ = A * cos_beta.unsqueeze(2) + A_flipped * sin_beta_flip.unsqueeze(2)
    # AZ shape: (B, num_blocks, max_dim, max_dim)

    # Step 4: D = AZ @ B for all blocks using batched matmul
    # Reshape for bmm: (B * num_blocks, max_dim, max_dim)
    AZ_flat = AZ.reshape(batch_size * num_blocks, max_dim, max_dim)
    B_flat = B.reshape(batch_size * num_blocks, max_dim, max_dim)
    D_flat = torch.bmm(AZ_flat, B_flat)
    D_padded = D_flat.reshape(batch_size, num_blocks, max_dim, max_dim)
    # D_padded shape: (B, num_blocks, max_dim, max_dim)

    # Step 5: Scatter padded blocks into block-diagonal output
    # Precompute scatter indices
    row_indices = []
    col_indices = []
    block_indices = []
    local_row = []
    local_col = []

    offset = 0
    for ell in range(num_blocks):
        dim_l = 2 * ell + 1
        for i in range(dim_l):
            for j in range(dim_l):
                row_indices.append(offset + i)
                col_indices.append(offset + j)
                block_indices.append(ell)
                local_row.append(i)
                local_col.append(j)
        offset += dim_l

    scatter_row = torch.tensor(row_indices, dtype=torch.long, device=device)
    scatter_col = torch.tensor(col_indices, dtype=torch.long, device=device)
    scatter_block = torch.tensor(block_indices, dtype=torch.long, device=device)
    scatter_local_row = torch.tensor(local_row, dtype=torch.long, device=device)
    scatter_local_col = torch.tensor(local_col, dtype=torch.long, device=device)

    # Gather values from padded representation
    values = D_padded[:, scatter_block, scatter_local_row, scatter_local_col]
    # values: (B, num_elements)

    # Create output and scatter
    wigner = torch.zeros(batch_size, full_dim, full_dim, dtype=dtype, device=device)
    batch_idx = (
        torch.arange(batch_size, device=device).view(-1, 1).expand(-1, len(scatter_row))
    )
    wigner[batch_idx, scatter_row, scatter_col] = values

    return wigner


# =============================================================================
# EdgeRotation module
# =============================================================================


class EdgeRotation(Module):
    r"""Compute and apply Wigner D-matrices for edge rotations in equivariant networks.

    This module computes rotation matrices needed to transform spherical
    harmonic coefficients between the global frame and edge-aligned local frames,
    and can apply these rotations to embeddings.

    It uses Wigner D-matrices organized in a block-diagonal structure with
    optional reduction to lower orders for efficiency.

    The key formula is:

    .. math::

        D^l(\alpha, \beta, \gamma) = Z(\alpha) \cdot J \cdot Z(\beta) \cdot J \cdot Z(\gamma)

    where Z is the z-rotation matrix and J is a precomputed involution matrix.

    The module caches computed D-matrices for efficient reuse when the same
    rotation is applied to multiple tensors (e.g., in multi-layer networks).
    Call :meth:`get_wigner_matrices` to compute and cache D-matrices, then
    use :meth:`forward` to apply rotations. Use :meth:`clear_cache` to free
    the cached memory; this should be called after gradients are computed
    to free up the graph.

    Parameters
    ----------
    lmax : int
        Maximum angular momentum quantum number. The full representation will have
        dimension (lmax + 1)^2.
    mmax : int, optional
        Maximum order for the reduced representation. Orders |m| > mmax are
        excluded. If None, defaults to lmax (no reduction). Must satisfy mmax <= lmax.
    computation_dtype : torch.dtype, optional
        Optional dtype for internal Wigner D-matrix computations. If provided,
        angles and intermediate matrices will be promoted to this dtype during
        computation, then cast back to the input dtype. Useful for maintaining
        numerical precision with half-precision inputs. If None (default), uses
        the input tensor's dtype directly.

    Raises
    ------
    ValueError
        If mmax > lmax.
    RuntimeError
        If :meth:`forward` is called before :meth:`get_wigner_matrices`
        (no cached D-matrices available).

    Forward
    -------
    x : Float[torch.Tensor, "... full_dim channels"]
        Spherical harmonic coefficients to rotate. Requires D-matrices to be
        cached via :meth:`get_wigner_matrices` first.
    inverse : bool, default=False
        If True, apply inverse rotation (from edge frame to global frame).

    Outputs
    -------
    Float[torch.Tensor, "... reduced_dim channels"]
        Rotated coefficients. If inverse=True, output has full_dim instead.

    Examples
    --------
    Basic usage with caching:

    >>> import torch
    >>> edge_rot = EdgeRotation(lmax=2, mmax=1)
    >>> edge_vecs = torch.randn(4, 5, 3)  # 4 nodes, 5 neighbors each
    >>> x = torch.randn(4, 5, 9, 64)  # [nodes, neighbors, (lmax+1)^2, channels]
    >>>
    >>> # Compute and cache D-matrices
    >>> D = edge_rot.get_wigner_matrices(edge_vecs)
    >>> D.shape
    torch.Size([4, 5, 7, 9])
    >>>
    >>> # Apply rotation (uses cached D-matrices)
    >>> x_rotated = edge_rot(x)
    >>> x_rotated.shape
    torch.Size([4, 5, 7, 64])
    >>>
    >>> # Apply inverse rotation
    >>> x_back = edge_rot(x_rotated, inverse=True)
    >>> x_back.shape
    torch.Size([4, 5, 9, 64])
    >>>
    >>> # Reuse cached D-matrices for another tensor
    >>> y = torch.randn(4, 5, 9, 32)
    >>> y_rotated = edge_rot(y)  # No recomputation needed
    >>>
    >>> # Clear cache when done
    >>> edge_rot.clear_cache()

    Notes
    -----
    The inverse rotation is simply the transpose: ``D_inv = D.transpose(-2, -1)``
    since Wigner D-matrices are orthogonal.

    When mmax < lmax, the forward rotation reduces dimensionality from full_dim
    to reduced_dim. The inverse rotation reconstructs full_dim but cannot recover
    the discarded high-order components - this is inherently lossy.
    """

    def __init__(
        self,
        lmax: int,
        mmax: int | None = None,
        computation_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax if mmax is not None else lmax
        self.computation_dtype = computation_dtype

        if self.mmax > self.lmax:
            raise ValueError(
                f"mmax must be <= lmax, got mmax={self.mmax}, lmax={self.lmax}"
            )

        # Compute representation dimensions
        self._full_dim = (lmax + 1) ** 2
        self._reduced_dim = sum(
            min(2 * self.mmax + 1, 2 * ell + 1) for ell in range(lmax + 1)
        )

        # Compute and register J matrices as persistent buffers (kept for backward
        # compatibility with state_dict)
        for ell in range(lmax + 1):
            J_l = _compute_J_matrix(
                ell, dtype=torch.float32, device=torch.device("cpu")
            )
            self.register_buffer(f"_J_{ell}", J_l, persistent=True)

        # Create block-diagonal J matrix for efficient computation: shape (full_dim, full_dim)
        J_full = torch.zeros(self._full_dim, self._full_dim, dtype=torch.float32)
        offset = 0
        for ell in range(lmax + 1):
            dim_l = 2 * ell + 1
            J_l = _compute_J_matrix(
                ell, dtype=torch.float32, device=torch.device("cpu")
            )
            J_full[offset : offset + dim_l, offset : offset + dim_l] = J_l
            offset += dim_l
        self.register_buffer("_J_full", J_full, persistent=False)

        # Precompute all m-values for all l blocks: shape (full_dim,)
        # m_vals[offset:offset+dim_l] contains m values for block l
        all_m_vals = []
        all_m_flip = []
        for ell in range(lmax + 1):
            dim_l = 2 * ell + 1
            m_vals_l = torch.arange(ell, -ell - 1, -1, dtype=torch.float32)
            all_m_vals.append(m_vals_l)
            all_m_flip.append(-m_vals_l)

        self.register_buffer("_all_m_vals", torch.cat(all_m_vals), persistent=False)
        self.register_buffer("_all_m_flip", torch.cat(all_m_flip), persistent=False)

        # Precompute block offsets and dimensions for loop-free indexing
        block_offsets = []
        block_dims = []
        offset = 0
        for ell in range(lmax + 1):
            dim_l = 2 * ell + 1
            block_offsets.append(offset)
            block_dims.append(dim_l)
            offset += dim_l
        self.register_buffer(
            "_block_offsets",
            torch.tensor(block_offsets, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_block_dims", torch.tensor(block_dims, dtype=torch.long), persistent=False
        )

        # =====================================================================
        # Fully vectorized buffers: pad and batch across all l values
        # =====================================================================
        max_dim = 2 * lmax + 1
        self._max_dim = max_dim
        num_blocks = lmax + 1

        # Padded J matrices: (num_blocks, max_dim, max_dim)
        # Each J_l is padded with zeros to max_dim x max_dim
        J_padded = torch.zeros(num_blocks, max_dim, max_dim, dtype=torch.float32)
        for ell in range(num_blocks):
            dim_l = 2 * ell + 1
            J_l = _compute_J_matrix(
                ell, dtype=torch.float32, device=torch.device("cpu")
            )
            J_padded[ell, :dim_l, :dim_l] = J_l
        self.register_buffer("_J_padded", J_padded, persistent=False)

        # Padded m-values: (num_blocks, max_dim)
        # m_vals for block l are in positions 0:dim_l, rest is zero
        m_vals_padded = torch.zeros(num_blocks, max_dim, dtype=torch.float32)
        m_flip_padded = torch.zeros(num_blocks, max_dim, dtype=torch.float32)
        for ell in range(num_blocks):
            dim_l = 2 * ell + 1
            m_vals_l = torch.arange(ell, -ell - 1, -1, dtype=torch.float32)
            m_vals_padded[ell, :dim_l] = m_vals_l
            m_flip_padded[ell, :dim_l] = -m_vals_l
        self.register_buffer("_m_vals_padded", m_vals_padded, persistent=False)
        self.register_buffer("_m_flip_padded", m_flip_padded, persistent=False)

        # Flip indices for each block: (num_blocks, max_dim)
        # For block l with dim_l elements, flip_indices[l, i] = dim_l - 1 - i for i < dim_l
        # For padding positions, use 0 (will be masked out anyway)
        flip_indices = torch.zeros(num_blocks, max_dim, dtype=torch.long)
        for ell in range(num_blocks):
            dim_l = 2 * ell + 1
            flip_indices[ell, :dim_l] = torch.arange(dim_l - 1, -1, -1)
        self.register_buffer("_flip_indices", flip_indices, persistent=False)

        # Precompute scatter indices for placing padded blocks into block-diagonal output
        # Maps from padded representation to flat block-diagonal positions
        row_indices = []
        col_indices = []
        block_indices = []
        local_row = []
        local_col = []

        offset = 0
        for ell in range(num_blocks):
            dim_l = 2 * ell + 1
            for i in range(dim_l):
                for j in range(dim_l):
                    row_indices.append(offset + i)
                    col_indices.append(offset + j)
                    block_indices.append(ell)
                    local_row.append(i)
                    local_col.append(j)
            offset += dim_l

        self.register_buffer(
            "_scatter_row",
            torch.tensor(row_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_scatter_col",
            torch.tensor(col_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_scatter_block",
            torch.tensor(block_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_scatter_local_row",
            torch.tensor(local_row, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_scatter_local_col",
            torch.tensor(local_col, dtype=torch.long),
            persistent=False,
        )

        # Build index mapping for reduced extraction
        self._index_mapping: list[tuple[int, int, int, int]] = []
        reduced_offset = 0
        full_offset = 0
        for ell in range(lmax + 1):
            full_dim_l = 2 * ell + 1
            reduced_dim_l = min(2 * self.mmax + 1, full_dim_l)
            m_limit = min(self.mmax, ell)
            start_idx = ell - m_limit
            end_idx = ell + m_limit + 1

            self._index_mapping.append(
                (reduced_offset, full_offset, start_idx, end_idx)
            )

            reduced_offset += reduced_dim_l
            full_offset += full_dim_l

        # Cache for Wigner D-matrices (allocated lazily on first use)
        # Shape will be (num_nodes, max_neighbors, reduced_dim, full_dim)
        # The last two dimensions are fixed; batch dims resize as needed
        self._cached_wigner: torch.Tensor | None = None
        self._cache_batch_shape: tuple[int, ...] | None = (
            None  # (num_nodes, max_neighbors)
        )

    def _get_J_matrix(self, ell: int) -> torch.Tensor:
        """Get the J matrix for angular momentum l."""
        return getattr(self, f"_J_{ell}")

    def _compute_wigner_block_diagonal(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute full block-diagonal Wigner D-matrix.

        This method delegates to the standalone ``compute_wigner_d_matrices``
        function for code reuse. It passes the precomputed J matrices to avoid
        recomputing them on every call.
        """
        # Store original dtype for casting back
        original_dtype = alpha.dtype

        # Promote to computation dtype if specified
        if self.computation_dtype is not None:
            alpha = alpha.to(self.computation_dtype)
            beta = beta.to(self.computation_dtype)
            gamma = gamma.to(self.computation_dtype)

        # Collect J matrices from registered buffers
        J_matrices = []
        for ell in range(self.lmax + 1):
            J_l = self._get_J_matrix(ell)
            # Cast J matrices to match angle dtype
            if J_l.dtype != alpha.dtype:
                J_l = J_l.to(alpha.dtype)
            J_matrices.append(J_l)

        # Call the standalone function
        result = compute_wigner_d_matrices(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lmax=self.lmax,
            J_matrices=J_matrices,
        )

        # Cast back to original dtype if we promoted
        if self.computation_dtype is not None and result.dtype != original_dtype:
            result = result.to(original_dtype)

        return result

    def _extract_reduced(
        self,
        wigner_full: torch.Tensor,
    ) -> torch.Tensor:
        """Extract reduced representation from full block-diagonal matrix."""
        batch_size = wigner_full.shape[0]
        wigner_reduced = torch.zeros(
            batch_size,
            self._reduced_dim,
            self._full_dim,
            dtype=wigner_full.dtype,
            device=wigner_full.device,
        )

        for ell in range(self.lmax + 1):
            reduced_offset, full_offset, start_idx, end_idx = self._index_mapping[ell]
            dim_l = 2 * ell + 1
            reduced_dim_l = end_idx - start_idx

            wigner_reduced[
                :,
                reduced_offset : reduced_offset + reduced_dim_l,
                full_offset : full_offset + dim_l,
            ] = wigner_full[
                :,
                full_offset + start_idx : full_offset + end_idx,
                full_offset : full_offset + dim_l,
            ]

        return wigner_reduced

    def _get_identity_reduced(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Get identity matrix for reduced representation."""
        identity = torch.zeros(
            batch_size,
            self._reduced_dim,
            self._full_dim,
            dtype=dtype,
            device=device,
        )

        for ell in range(self.lmax + 1):
            reduced_offset, full_offset, start_idx, end_idx = self._index_mapping[ell]

            for i, full_i in enumerate(range(start_idx, end_idx)):
                identity[
                    :,
                    reduced_offset + i,
                    full_offset + full_i,
                ] = 1.0

        return identity

    def _apply_mask(
        self,
        wigner: torch.Tensor,
        mask: Bool[torch.Tensor, "num_nodes max_neighbors"],
    ) -> torch.Tensor:
        """Apply mask to replace invalid edges with identity."""
        num_nodes, max_neighbors = mask.shape
        identity = self._get_identity_reduced(
            num_nodes * max_neighbors,
            wigner.dtype,
            wigner.device,
        )
        identity = identity.reshape(
            num_nodes, max_neighbors, self._reduced_dim, self._full_dim
        )

        return torch.where(
            mask.unsqueeze(-1).unsqueeze(-1),
            wigner,
            identity,
        )

    def _compute_wigner_matrices(
        self,
        edge_vecs: Float[torch.Tensor, "num_nodes max_neighbors 3"],
        mask: Bool[torch.Tensor, "num_nodes max_neighbors"] | None = None,
    ) -> Float[torch.Tensor, "num_nodes max_neighbors reduced_dim full_dim"]:
        """Compute Wigner D-matrices for edge rotations (internal method).

        Parameters
        ----------
        edge_vecs : Float[torch.Tensor, "num_nodes max_neighbors 3"]
            Edge direction vectors. Shape (num_nodes, max_neighbors, 3).
        mask : Bool[torch.Tensor, "num_nodes max_neighbors"], optional
            Boolean mask for valid edges. If None, all edges are assumed valid.

        Returns
        -------
        Float[torch.Tensor, "num_nodes max_neighbors reduced_dim full_dim"]
            Wigner D-matrices in reduced representation.
        """
        # Validate input shape
        if not torch.compiler.is_compiling():
            if edge_vecs.ndim != 3 or edge_vecs.shape[-1] != 3:
                raise ValueError(
                    f"Expected edge_vecs shape (num_nodes, max_neighbors, 3), "
                    f"got {tuple(edge_vecs.shape)}"
                )

        num_nodes, max_neighbors = edge_vecs.shape[:2]

        # Flatten edge vectors to (batch, 3)
        edge_vecs_flat = edge_vecs.reshape(-1, 3)

        # Convert to Euler angles
        alpha, beta, gamma = edge_vectors_to_euler_angles(edge_vecs_flat)

        # Compute full block-diagonal Wigner matrices
        wigner_full = self._compute_wigner_block_diagonal(alpha, beta, gamma)

        # Extract reduced representation
        wigner_reduced = self._extract_reduced(wigner_full)

        # Reshape to (num_nodes, max_neighbors, reduced_dim, full_dim)
        wigner = wigner_reduced.reshape(
            num_nodes, max_neighbors, self._reduced_dim, self._full_dim
        )

        # Apply mask if provided
        if mask is not None:
            wigner = self._apply_mask(wigner, mask)

        return wigner

    def forward(
        self,
        x: Float[torch.Tensor, "... dim channels"],
        inverse: bool = False,
    ) -> Float[torch.Tensor, "... out_dim channels"]:
        # Check cache is populated
        if self._cached_wigner is None:
            raise RuntimeError(
                "No cached Wigner D-matrices. Call get_wigner_matrices(edge_vecs) first."
            )

        wigner = self._cached_wigner

        # Get shapes
        *batch_dims, dim_in, channels = x.shape
        *_, reduced_dim, full_dim = wigner.shape

        # Validate dimensions
        if not torch.compiler.is_compiling():
            if inverse:
                # For inverse rotation, input should have reduced_dim
                if dim_in != reduced_dim:
                    raise ValueError(
                        f"For inverse rotation, input dim {dim_in} doesn't match "
                        f"Wigner reduced_dim {reduced_dim}"
                    )
            else:
                # For forward rotation, input should have full_dim
                if dim_in != full_dim:
                    raise ValueError(
                        f"Input dim {dim_in} doesn't match Wigner full_dim {full_dim}"
                    )

        # Flatten batch dimensions for bmm
        batch_size = 1
        for d in batch_dims:
            batch_size *= d

        x_flat = x.reshape(batch_size, dim_in, channels)
        wigner_flat = wigner.reshape(batch_size, reduced_dim, full_dim)

        if inverse:
            # D^T @ x: transpose the D-matrix
            # wigner_flat: [batch, reduced_dim, full_dim]
            # For inverse: D^T: [batch, full_dim, reduced_dim]
            # Input x has shape [batch, reduced_dim, channels]
            # Output: [batch, full_dim, channels]
            wigner_T = wigner_flat.transpose(-2, -1)  # [batch, full_dim, reduced_dim]
            y_flat = torch.bmm(wigner_T, x_flat)
            out_dim = full_dim
        else:
            # D @ x: forward rotation
            # wigner: [batch, reduced_dim, full_dim]
            # x: [batch, full_dim, channels]
            # y: [batch, reduced_dim, channels]
            y_flat = torch.bmm(wigner_flat, x_flat)
            out_dim = reduced_dim

        # Reshape back to original batch dims
        return y_flat.reshape(*batch_dims, out_dim, channels)

    def get_wigner_matrices(
        self,
        edge_vecs: Float[torch.Tensor, "... 3"],
        mask: Bool[torch.Tensor, "..."] | None = None,
    ) -> Float[torch.Tensor, "... reduced_dim full_dim"]:
        """Compute Wigner D-matrices and cache for subsequent rotations.

        This method computes D-matrices from edge vectors and stores them
        in an internal cache. The cached matrices are used by :meth:`forward`
        to apply rotations without recomputation.

        Parameters
        ----------
        edge_vecs : Float[torch.Tensor, "... 3"]
            Edge direction vectors. The last dimension must be 3 (x, y, z).
            Typical shape: (num_nodes, max_neighbors, 3).
        mask : Bool[torch.Tensor, "..."], optional
            Boolean mask for valid edges. Shape should match edge_vecs
            without the last dimension. Invalid edges get identity matrices.

        Returns
        -------
        Float[torch.Tensor, "... reduced_dim full_dim"]
            Computed Wigner D-matrices, also stored in cache.

        Examples
        --------
        >>> edge_rot = EdgeRotation(lmax=2, mmax=1)
        >>> edge_vecs = torch.randn(4, 5, 3)
        >>> D = edge_rot.get_wigner_matrices(edge_vecs)
        >>> D.shape
        torch.Size([4, 5, 7, 9])
        """
        # Compute D-matrices
        wigner = self._compute_wigner_matrices(edge_vecs, mask)

        # Cache the result
        self._cached_wigner = wigner
        self._cache_batch_shape = wigner.shape[
            :-2
        ]  # All dims except (reduced_dim, full_dim)

        return wigner

    def clear_cache(self) -> None:
        """Clear cached Wigner D-matrices to free memory.

        Call this method when the cached D-matrices are no longer needed,
        such as between training batches with different graph structures.

        Examples
        --------
        >>> edge_rot = EdgeRotation(lmax=2)
        >>> edge_vecs = torch.randn(4, 5, 3)
        >>> _ = edge_rot.get_wigner_matrices(edge_vecs)
        >>> edge_rot.clear_cache()
        """
        self._cached_wigner = None
        self._cache_batch_shape = None
