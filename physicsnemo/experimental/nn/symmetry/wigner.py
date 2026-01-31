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

"""On-the-fly Wigner D-matrix computation for edge-aligned rotations.

This module provides the :class:`EdgeRotation` module for computing Wigner D-matrices
that rotate spherical harmonic coefficients between the global frame and edge-aligned
local frames. This enables SO(3) equivariance using only SO(2) convolutions.

The implementation computes J matrices on-the-fly at initialization (no precomputed
``Jd.pt`` files required) and uses the factored formula:

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
"""

from __future__ import annotations

import math

import torch
from torch import nn
from jaxtyping import Bool, Float

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
    tuple of (alpha, beta, gamma)
        Euler angles in radians. gamma is always 0 for edge rotations.

        - alpha : Float[torch.Tensor, "..."]
            Azimuthal angle (rotation about z-axis).
        - beta : Float[torch.Tensor, "..."]
            Polar angle (rotation about y-axis).
        - gamma : Float[torch.Tensor, "..."]
            Roll angle (always zero for edges).

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
# EdgeRotation module
# =============================================================================


class EdgeRotation(nn.Module):
    r"""Compute Wigner D-matrices for edge rotations in equivariant networks.

    This module computes the rotation matrices needed to transform spherical
    harmonic coefficients between the global frame and edge-aligned local frames.
    It uses Wigner D-matrices organized in a block-diagonal structure with
    optional reduction to lower orders for efficiency.

    The key formula is:

    .. math::

        D^l(\alpha, \beta, \gamma) = Z(\alpha) \cdot J \cdot Z(\beta) \cdot J \cdot Z(\gamma)

    where Z is the z-rotation matrix and J is a precomputed involution matrix.

    Parameters
    ----------
    lmax : int
        Maximum angular momentum quantum number. The full representation will have
        dimension (lmax + 1)^2.
    mmax : int, optional
        Maximum order for the reduced representation. Orders |m| > mmax are
        excluded. If None, defaults to lmax (no reduction). Must satisfy mmax <= lmax.

    Raises
    ------
    ValueError
        If mmax > lmax.

    Forward
    -------
    edge_vecs : Float[torch.Tensor, "num_nodes max_neighbors 3"]
        Edge direction vectors from nodes to their neighbors.
    mask : Bool[torch.Tensor, "num_nodes max_neighbors"], optional
        Boolean mask indicating valid edges. Invalid edges get identity matrices.

    Outputs
    -------
    Float[torch.Tensor, "num_nodes max_neighbors reduced_dim full_dim"]
        Wigner D-matrices in the reduced representation.

    Examples
    --------
    >>> import torch
    >>> edge_rot = EdgeRotation(lmax=2, mmax=1)
    >>> edge_vecs = torch.randn(4, 5, 3)  # 4 nodes, 5 neighbors each
    >>> D = edge_rot(edge_vecs)
    >>> D.shape
    torch.Size([4, 5, 7, 9])

    Notes
    -----
    The inverse rotation is simply the transpose: ``D_inv = D.transpose(-2, -1)``
    since Wigner D-matrices are orthogonal.
    """

    def __init__(
        self,
        lmax: int,
        mmax: int | None = None,
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax if mmax is not None else lmax

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

    def _get_J_matrix(self, ell: int) -> torch.Tensor:
        """Get the J matrix for angular momentum l."""
        return getattr(self, f"_J_{ell}")

    def _compute_wigner_block_diagonal(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute full block-diagonal Wigner D-matrix (fully vectorized).

        This implementation pads all blocks to max_dim and computes all l values
        simultaneously to minimize kernel launches and intermediate allocations.

        The key insight is that Z(φ) has only diagonal and anti-diagonal
        non-zero elements, so ZxDense and DensexZ can be computed in O(dim²)
        instead of O(dim³).
        """
        batch_size = alpha.shape[0]
        device = alpha.device
        dtype = alpha.dtype
        num_blocks = self.lmax + 1
        max_dim = self._max_dim

        # Get precomputed buffers
        J_padded = self._J_padded.to(
            dtype=dtype, device=device
        )  # (num_blocks, max_dim, max_dim)
        m_vals = self._m_vals_padded.to(
            dtype=dtype, device=device
        )  # (num_blocks, max_dim)
        m_flip = self._m_flip_padded.to(
            dtype=dtype, device=device
        )  # (num_blocks, max_dim)
        flip_idx = self._flip_indices.to(device=device)  # (num_blocks, max_dim)

        # Compute all trig values at once
        # alpha: (B,) -> (B, 1, 1) for broadcasting with m_vals: (num_blocks, max_dim)
        # Result: (B, num_blocks, max_dim)
        alpha_expanded = alpha.view(batch_size, 1, 1)
        beta_expanded = beta.view(batch_size, 1, 1)
        gamma_expanded = gamma.view(batch_size, 1, 1)

        cos_alpha = torch.cos(alpha_expanded * m_vals)  # (B, num_blocks, max_dim)
        sin_alpha = torch.sin(alpha_expanded * m_vals)
        cos_beta = torch.cos(beta_expanded * m_vals)
        sin_beta_flip = torch.sin(beta_expanded * m_flip)
        cos_gamma = torch.cos(gamma_expanded * m_vals)
        sin_gamma_flip = torch.sin(gamma_expanded * m_flip)

        # For gathering flipped rows/columns, we need advanced indexing
        # J_padded[ell, flip_idx[ell], :] gives the flipped rows for each block
        ell_idx = torch.arange(num_blocks, device=device)
        J_flipped_rows = J_padded[
            ell_idx.view(-1, 1), flip_idx, :
        ]  # (num_blocks, max_dim, max_dim)

        # For J_flipped_cols: J_padded[ell, :, flip_idx[ell, :]]
        # Use gather along dim=2
        J_flipped_cols = torch.gather(
            J_padded,
            dim=2,
            index=flip_idx.unsqueeze(1).expand(-1, max_dim, -1),
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
        # Need A[:, :, :, flip_idx] - gather along last dimension
        A_flipped = torch.gather(
            A,
            dim=3,
            index=flip_idx.view(1, num_blocks, 1, max_dim).expand(
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
        # Gather from D_padded and scatter to wigner using precomputed indices
        # D_padded: (B, num_blocks, max_dim, max_dim)
        # We want: wigner[b, row_idx, col_idx] = D_padded[b, block_idx, local_row, local_col]
        values = D_padded[
            :, self._scatter_block, self._scatter_local_row, self._scatter_local_col
        ]
        # values: (B, num_elements)

        # Create output and scatter using advanced indexing
        wigner = torch.zeros(
            batch_size, self._full_dim, self._full_dim, dtype=dtype, device=device
        )
        batch_idx = (
            torch.arange(batch_size, device=device)
            .view(-1, 1)
            .expand(-1, len(self._scatter_row))
        )
        wigner[batch_idx, self._scatter_row, self._scatter_col] = values

        return wigner

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

    def forward(
        self,
        edge_vecs: Float[torch.Tensor, "num_nodes max_neighbors 3"],
        mask: Bool[torch.Tensor, "num_nodes max_neighbors"] | None = None,
    ) -> Float[torch.Tensor, "num_nodes max_neighbors reduced_dim full_dim"]:
        """Compute Wigner D-matrices for edge rotations.

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
