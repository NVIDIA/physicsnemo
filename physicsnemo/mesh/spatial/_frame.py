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

"""A canonical frame of a weighted point cloud, which moves with the cloud.

Coordinates in this frame are the same in every pose of the cloud, so a
space-filling curve or a tree built on them does not depend on the pose.
"""

import torch
from jaxtyping import Float

#: Below this weighted normalized third moment along a principal axis, the
#: axis's sign is a tie (a symmetric body) and is fixed by the lab axis it most
#: overlaps instead. Computed in float64, so a body's skewness is reproduced under a
#: rigid motion to 1e-12 and only a genuinely symmetric axis falls back.
CANONICAL_SKEW_TIE = 1e-9


def canonical_frame(
    points: Float[torch.Tensor, "n_points n_spatial_dims"],
    weights: Float[torch.Tensor, " n_points"],
) -> Float[torch.Tensor, "n_spatial_dims n_spatial_dims"]:
    r"""A frame that the point cloud carries with it.

    The axes are the principal axes of the weighted covariance of the points, in
    ascending variance. Each axis is signed so that the weighted third moment of
    the points along it is positive. Under a rotation, reflection, translation or
    rescale of the cloud, the frame moves with it, so the coordinates
    ``points @ frame`` are invariant. The exceptions are ties: equal variances, or
    a symmetric axis whose third moment vanishes (below
    :data:`CANONICAL_SKEW_TIE`), whose sign then follows the lab axis it overlaps
    most.

    Computed in float64 whatever the input precision, without gradients.

    Parameters
    ----------
    points : torch.Tensor
        Point coordinates, shape ``(n_points, n_spatial_dims)``.
    weights : torch.Tensor
        Nonnegative point weights, shape ``(n_points,)``, such as cell measures.
        Negative weights are treated as zero.

    Returns
    -------
    torch.Tensor
        The frame, shape ``(n_spatial_dims, n_spatial_dims)``, with the axes as
        columns, in the dtype of ``points``.
    """
    n_spatial_dims = points.shape[1]
    x = points.detach().double()
    w = weights.detach().double().clamp_min(0)
    w = w / w.sum().clamp_min(torch.finfo(torch.float64).tiny)
    x = x - (w[:, None] * x).sum(0)
    wx = w[:, None] * x
    ### Column sums, not a GEMM (see frame_coordinates).
    cov = torch.stack([(wx * x[:, j, None]).sum(0) for j in range(n_spatial_dims)], 1)
    _, axes = torch.linalg.eigh(cov)
    proj = frame_coordinates(x, axes)
    m2 = (w[:, None] * proj.square()).sum(0)
    m3 = (w[:, None] * proj.pow(3)).sum(0)
    skew = m3 / m2.clamp_min(torch.finfo(torch.float64).tiny).pow(1.5)
    lab = axes[
        axes.abs().argmax(dim=0), torch.arange(n_spatial_dims, device=points.device)
    ]
    sign = torch.where(
        skew.abs() > CANONICAL_SKEW_TIE, torch.sign(skew), torch.sign(lab)
    )
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return (axes * sign[None, :]).to(points.dtype)


def frame_coordinates(
    points: Float[torch.Tensor, "n_points n_spatial_dims"],
    frame: Float[torch.Tensor, "n_spatial_dims n_spatial_dims"],
) -> Float[torch.Tensor, "n_points n_spatial_dims"]:
    """The coordinates ``points @ frame``, computed as ``n_spatial_dims`` column products.

    A float64 GEMM this thin (or the ``(n_spatial_dims, n_points) @ (n_points,
    n_spatial_dims)`` of a covariance) runs far below memory bandwidth on GPUs
    with few float64 units: on GB300 at 17.7M points, 17.8 ms (34 ms) against
    0.9 ms (1.3 ms) as columns.

    Parameters
    ----------
    points : torch.Tensor
        Point coordinates, shape ``(n_points, n_spatial_dims)``.
    frame : torch.Tensor
        A frame with the axes as columns, shape ``(n_spatial_dims, n_spatial_dims)``,
        such as :func:`canonical_frame`'s.

    Returns
    -------
    torch.Tensor
        The coordinates along each axis, shape ``(n_points, n_spatial_dims)``.
    """
    out = points[:, :1] * frame[0]
    for k in range(1, points.shape[1]):
        out = out + points[:, k : k + 1] * frame[k]
    return out
