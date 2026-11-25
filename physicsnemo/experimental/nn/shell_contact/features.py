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
"""Edge features of shell contact pairs.

The feature layout is shared by every contact consumer in PhysicsNeMo: eight
geometry channels, the relative displacement, distance and signed gap over a
feature scale, and a unit contact normal.
"""

from __future__ import annotations

import math

import torch

from .search import ShellContactPairs

CONTACT_FEATURE_DIM = 8
"""Width of the contact features."""

_EPS = 1.0e-8


def contact_edge_features(
    relative_displacement: torch.Tensor,
    signed_gap: torch.Tensor,
    feature_scale: float,
    contact_normal: torch.Tensor | None = None,
    normal_epsilon: float | None = None,
) -> torch.Tensor:
    r"""Encode displacement, distance, signed gap and contact normal of contact edges.

    Parameters
    ----------
    relative_displacement : torch.Tensor
        ``[E, 3]`` vector from the receiving node to its contact partner (the
        closest point on the partner surface), in physical units.
    signed_gap : torch.Tensor
        ``[E]`` or ``[E, 1]`` shell clearance of every edge, negative inside
        the thickness envelope.
    feature_scale : float
        Length that normalizes the first five channels.
    contact_normal : torch.Tensor, optional
        ``[E, 3]`` unit direction of the contact. When omitted it is the
        regularized direction of ``relative_displacement``.
    normal_epsilon : float, optional
        Regularization length of the direction: a positive value bounds the
        Jacobian of the normal by ``1 / normal_epsilon`` and maps a coincident
        pair to a zero direction instead of an arbitrary unit vector.

    Returns
    -------
    torch.Tensor
        ``[E, 8]`` features.
    """
    if not (math.isfinite(feature_scale) and feature_scale > 0.0):
        raise ValueError(
            f"feature_scale must be positive and finite; got {feature_scale}"
        )
    if relative_displacement.ndim != 2 or relative_displacement.shape[-1] != 3:
        raise ValueError("relative_displacement must have shape [E, 3]")
    if signed_gap.ndim == 1:
        signed_gap = signed_gap.unsqueeze(-1)
    if signed_gap.shape != (relative_displacement.shape[0], 1):
        raise ValueError("signed_gap must have shape [E] or [E, 1]")

    distance = torch.linalg.vector_norm(relative_displacement, dim=-1, keepdim=True)
    if normal_epsilon is not None and (
        not math.isfinite(normal_epsilon) or normal_epsilon <= 0
    ):
        raise ValueError("normal_epsilon must be positive and finite")
    if contact_normal is None:
        # A physical regularization length bounds the normal Jacobian by 1/eps.
        # At coincidence the direction is zero, not an arbitrary unit vector.
        denominator = (
            (
                relative_displacement.square().sum(-1, keepdim=True) + normal_epsilon**2
            ).sqrt()
            if normal_epsilon is not None
            else distance.clamp_min(_EPS)
        )
        contact_normal = relative_displacement / denominator
    if contact_normal.shape != relative_displacement.shape:
        raise ValueError("contact_normal must have shape [E, 3]")
    features = torch.cat(
        (
            relative_displacement / feature_scale,
            distance / feature_scale,
            signed_gap / feature_scale,
            contact_normal,
        ),
        dim=-1,
    )
    return features


def shell_contact_edge_features(
    pairs: ShellContactPairs,
    positions: torch.Tensor,
    *,
    feature_scale: float,
    normal_epsilon: float,
) -> torch.Tensor:
    r"""Live contact features of discovered shell contact pairs.

    The displacement channel points from the node to the live closest point
    on the facet; the signed gap is the pair's shell clearance, negative once
    the node has crossed the shell (``pairs.crossed``), and the contact normal
    flips with it so that it keeps pointing from the node towards the shell
    surface it is in contact with. Everything is differentiable in
    ``positions`` through the live projection.

    Parameters
    ----------
    pairs : ShellContactPairs
        Output of :func:`shell_contact_pairs` on ``positions``.
    positions : torch.Tensor
        ``[N, 3]`` the node positions the pairs were discovered on.
    feature_scale : float
        Length that normalizes the displacement, distance and gap channels.
    normal_epsilon : float
        Regularization length of the contact normal.

    Returns
    -------
    torch.Tensor
        ``[E, 8]`` features.
    """
    if not math.isfinite(normal_epsilon) or normal_epsilon <= 0:
        raise ValueError("normal_epsilon must be positive and finite")
    relative = pairs.closest - positions[pairs.query]
    normal = None
    if pairs.signed:
        # The reference side test signs the gap: a crossed pair's direction
        # flips so that it keeps pointing from the node towards the shell.
        side = torch.where(pairs.crossed, -1.0, 1.0).to(positions.dtype)
        normal = (
            relative
            / (relative.square().sum(-1, keepdim=True) + normal_epsilon**2).sqrt()
            * side[:, None]
        )
    return contact_edge_features(
        relative,
        pairs.gap,
        feature_scale,
        contact_normal=normal,
        normal_epsilon=normal_epsilon,
    )
