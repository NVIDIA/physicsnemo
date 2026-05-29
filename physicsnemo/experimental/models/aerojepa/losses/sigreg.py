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

"""SIGReg latent regularizer and its token-shaped wrapper.

:class:`SIGReg` pushes a learned latent distribution toward an isotropic
Gaussian by comparing the empirical Fourier characteristic function of
random projections of the latent samples against the reference
:math:`\\exp(-t^2 / 2)` on a uniform knot grid.
:class:`TokenLatentSIGReg` accepts token-shaped features plus an optional
padding mask and delegates to ``SIGReg`` after flattening the valid rows.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .masking import reshape_token_features_for_sigreg


class SIGReg(nn.Module):
    r"""Sketch Isotropic Gaussian Regularizer.

    Given samples ``proj`` of shape ``(T, B, D)``, draws ``num_proj`` random
    unit-norm projection directions and computes an integrated squared
    difference between the empirical Fourier characteristic function of the
    projected samples and the reference Gaussian one
    (:math:`\exp(-t^2 / 2)`) over a uniform grid of ``knots`` evaluation
    points on ``[0, 3]``. The grid uses trapezoidal weights pre-multiplied
    by the reference window so the final statistic is a single scalar
    suitable for use as a regularization term.

    The buffers ``t``, ``phi``, and ``weights`` are non-learnable and
    saved with the module.

    Parameters
    ----------
    knots : int, optional
        Number of grid knots on ``[0, 3]``. Must be at least 2. Default 17.
    num_proj : int, optional
        Number of random projection directions sampled per forward call.
        Default 1024.

    Shape
    -----
    - Input ``proj``: ``(T, B, D)`` where ``T`` is the number of projection
      groups, ``B`` the batch / sample axis, and ``D`` the latent dim.
    - Output: scalar tensor (a 0-d ``torch.Tensor``).

    Raises
    ------
    ValueError
        If ``knots < 2`` at construction or if ``proj`` is not rank 3 at
        forward time.

    References
    ----------
    Maes, Le Lidec, Scieur, LeCun, Balestriero, "LeWorldModel: Stable
    End-to-End Joint-Embedding Predictive Architecture from Pixels", 2026.
    """

    def __init__(self, *, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        if int(knots) < 2:
            raise ValueError("SIGReg requires knots >= 2.")
        self.num_proj = int(num_proj)
        t = torch.linspace(0.0, 3.0, int(knots), dtype=torch.float32)
        dt = 3.0 / float(max(int(knots) - 1, 1))
        weights = torch.full((int(knots),), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.ndim != 3:
            raise ValueError(
                f"proj must have shape [T, B, D], got {tuple(proj.shape)}"
            )
        if int(proj.shape[-1]) == 0 or int(proj.shape[-2]) == 0:
            return proj.new_zeros(())
        A = torch.randn(
            int(proj.shape[-1]),
            self.num_proj,
            device=proj.device,
            dtype=proj.dtype,
        )
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
        x_t = (proj @ A).unsqueeze(-1) * self.t.to(
            device=proj.device, dtype=proj.dtype
        )
        err = (
            x_t.cos().mean(dim=-3)
            - self.phi.to(device=proj.device, dtype=proj.dtype)
        ).square()
        err = err + x_t.sin().mean(dim=-3).square()
        statistic = (
            err @ self.weights.to(device=proj.device, dtype=proj.dtype)
        ) * int(proj.shape[-2])
        return statistic.mean()


class TokenLatentSIGReg(nn.Module):
    r"""Apply SIGReg to token-shaped features with optional padding masks.

    A thin wrapper around :class:`SIGReg` that accepts the
    ``(B, N, D)`` (or ``(N, D)``) layout produced by AeroJEPA encoders,
    drops padded rows via the supplied boolean mask, reshapes the result
    into the ``(T=1, B', D)`` layout SIGReg expects, and returns a scalar
    regularization loss. Returns a zero scalar when no valid token
    survives masking.

    Parameters
    ----------
    knots : int, optional
        Forwarded to :class:`SIGReg`. Default 17.
    num_proj : int, optional
        Forwarded to :class:`SIGReg`. Default 1024.

    Shape
    -----
    - Input ``features``: ``(N, D)`` or ``(B, N, D)``.
    - Optional ``mask``: ``(B, N)`` boolean, required to be ``None`` for
      rank-2 ``features``.
    - Output: scalar tensor.
    """

    def __init__(self, *, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.regularizer = SIGReg(knots=int(knots), num_proj=int(num_proj))

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        proj = reshape_token_features_for_sigreg(features, mask)
        if int(proj.shape[1]) == 0:
            return features.new_zeros(())
        return self.regularizer(proj)
