# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Differentiable proper polar rotations for deformation solvers.

The closest proper rotation has an ordinary SVD forward, but differentiating
that SVD directly is undefined when singular values repeat or vanish. Planar
3D vertex neighborhoods have a rank-two covariance by construction, so this
module supplies the polar-factor derivative through a Sylvester equation.
"""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable


def _closest_proper_rotation(covariance: torch.Tensor) -> torch.Tensor:
    """Compute ``U diag(1, ..., det(U Vh)) Vh`` for square matrices."""

    u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
    unconstrained = u @ vh
    final_sign = torch.where(
        torch.linalg.det(unconstrained) < 0,
        -torch.ones_like(unconstrained[..., 0, 0]),
        torch.ones_like(unconstrained[..., 0, 0]),
    )
    correction = torch.cat(
        (torch.ones_like(u[..., 0, :-1]), final_sign.unsqueeze(-1)), dim=-1
    )
    return (u * correction.unsqueeze(-2)) @ vh


def _sylvester_pseudoinverse(
    symmetric_factor: torch.Tensor,
    right_hand_side: torch.Tensor,
) -> torch.Tensor:
    """Solve ``H Z + Z H = B`` with a spectral pair-sum pseudoinverse."""

    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_factor)
    pair_sums = eigenvalues.unsqueeze(-1) + eigenvalues.unsqueeze(-2)
    scale = eigenvalues.abs().amax(dim=-1, keepdim=True).unsqueeze(-1)
    threshold = torch.finfo(eigenvalues.dtype).eps * eigenvalues.shape[-1] * scale
    invertible = pair_sums.abs() > threshold
    safe_pair_sums = torch.where(invertible, pair_sums, torch.ones_like(pair_sums))
    reciprocal = torch.where(
        invertible, safe_pair_sums.reciprocal(), torch.zeros_like(pair_sums)
    )

    eigenbasis_rhs = eigenvectors.mT @ right_hand_side @ eigenvectors
    eigenbasis_solution = eigenbasis_rhs * reciprocal
    solution = eigenvectors @ eigenbasis_solution @ eigenvectors.mT
    # Roundoff in ``eigh`` can introduce a symmetric component even though the
    # right-hand side and exact solution are skew-symmetric.
    return 0.5 * (solution - solution.mT)


class _ProperRotation(torch.autograd.Function):
    """First-order custom autograd rule for the closest proper polar factor."""

    @staticmethod
    def forward(ctx, covariance: torch.Tensor) -> torch.Tensor:
        """Compute and save the factors needed by the polar derivative."""

        rotation = _closest_proper_rotation(covariance)
        symmetric_factor = rotation.mT @ covariance
        # The determinant correction can make this factor indefinite, but it
        # remains symmetric. Remove only floating-point antisymmetric noise.
        symmetric_factor = 0.5 * (symmetric_factor + symmetric_factor.mT)
        ctx.save_for_backward(rotation, symmetric_factor)
        return rotation

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_rotation: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the polar-factor VJP through a Sylvester solve."""

        rotation, symmetric_factor = ctx.saved_tensors
        cotangent_body = rotation.mT @ grad_rotation
        skew_right_hand_side = cotangent_body - cotangent_body.mT
        solution = _sylvester_pseudoinverse(symmetric_factor, skew_right_hand_side)
        return (rotation @ solution,)


def proper_rotation(covariance: torch.Tensor) -> torch.Tensor:
    r"""Return the closest proper rotation with a stable first-order VJP.

    For each square covariance matrix :math:`C`, this returns
    :math:`R = U D V^T` for an SVD :math:`C = U \Sigma V^T`, where
    :math:`D = \operatorname{diag}(1, \ldots, \det(UV^T))`. Consequently every
    returned matrix is orthogonal with determinant ``+1``.

    The backward uses :math:`H=R^T C`, :math:`A=R^T \bar R`, and solves

    .. math::

       H Z + Z H = A - A^T, \qquad \bar C = R Z.

    A spectral pair-sum pseudoinverse makes this derivative finite for the
    rank-two covariances produced by planar neighborhoods in 3D. For rank below
    two in 3D, the optimal rotation is generally non-unique; the pseudoinverse
    returns the minimum-norm subgradient associated with the SVD-selected
    rotation rather than a unique mathematical derivative.

    Parameters
    ----------
    covariance : torch.Tensor
        Square covariance matrices with shape ``(..., D, D)``, ``D >= 2``, and
        dtype ``torch.float32`` or ``torch.float64``.

    Returns
    -------
    torch.Tensor
        Proper rotation matrices with the same shape, dtype, and device.

    Notes
    -----
    Only first-order differentiation is supported. The operation is intended
    as internal machinery for local/global deformation solvers.
    """

    if covariance.ndim < 2:
        raise ValueError(
            f"covariance must have shape (..., D, D), got {tuple(covariance.shape)}"
        )
    if covariance.shape[-2] != covariance.shape[-1]:
        raise ValueError(
            f"covariance must contain square matrices, got {tuple(covariance.shape)}"
        )
    if covariance.shape[-1] < 2:
        raise ValueError("covariance matrix dimension must be at least 2")
    if covariance.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "covariance must have dtype torch.float32 or torch.float64, got "
            f"{covariance.dtype}"
        )
    return _ProperRotation.apply(covariance)


__all__ = ["proper_rotation"]
