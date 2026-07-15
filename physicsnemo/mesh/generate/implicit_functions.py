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

"""Implicit-function (signed-distance) building blocks.

An *implicit function* here is any callable ``phi(x: (..., d)) -> (...)``
that is negative inside the domain, positive outside, and differentiable
almost everywhere under torch autograd. Exact signed-distance functions
give the best-conditioned projections, but any level-set function with a
nonvanishing gradient near its zero set works (the mesher's Newton
projection normalizes by ``|grad phi|^2``).

These helpers are conveniences for common shapes and CSG composition; user
domains can equally be lambdas, neural implicit fields, or interpolants.
Note the CSG combinators (`sdf_union`, `sdf_intersection`,
`sdf_difference`) return exact distances only away from the combined
surface's new edges -- the standard min/max approximation.
"""

from typing import Callable

import torch

__all__ = [
    "project_to_zero_set",
    "sdf_box",
    "sdf_difference",
    "sdf_intersection",
    "sdf_polygon_2d",
    "sdf_sphere",
    "sdf_union",
]

ImplicitFunction = Callable[[torch.Tensor], torch.Tensor]


def sdf_sphere(center, radius: float) -> ImplicitFunction:
    """Signed distance to a d-sphere (any dimension, from ``len(center)``)."""

    def phi(x):
        c = torch.as_tensor(center, dtype=x.dtype, device=x.device)
        return (x - c).norm(dim=-1) - radius

    return phi


def sdf_box(lo, hi) -> ImplicitFunction:
    """Exact signed distance to an axis-aligned box ``[lo, hi]``."""

    def phi(x):
        lo_t = torch.as_tensor(lo, dtype=x.dtype, device=x.device)
        hi_t = torch.as_tensor(hi, dtype=x.dtype, device=x.device)
        q = (x - 0.5 * (lo_t + hi_t)).abs() - 0.5 * (hi_t - lo_t)
        outside = q.clamp(min=0.0).norm(dim=-1)
        inside = q.max(dim=-1).values.clamp(max=0.0)
        return outside + inside

    return phi


def sdf_union(*phis: ImplicitFunction) -> ImplicitFunction:
    """Union of domains: pointwise minimum of the fields."""
    return lambda x: torch.stack([p(x) for p in phis], dim=0).min(dim=0).values


def sdf_intersection(*phis: ImplicitFunction) -> ImplicitFunction:
    """Intersection of domains: pointwise maximum of the fields."""
    return lambda x: torch.stack([p(x) for p in phis], dim=0).max(dim=0).values


def sdf_difference(
    phi_a: ImplicitFunction, phi_b: ImplicitFunction
) -> ImplicitFunction:
    """Domain of ``phi_a`` minus the domain of ``phi_b``."""
    return lambda x: torch.maximum(phi_a(x), -phi_b(x))


def sdf_polygon_2d(vertices, chunk: int = 65536) -> ImplicitFunction:
    """Exact signed distance to a simple 2D polygon.

    Parameters
    ----------
    vertices : array-like, shape (n_vertices, 2)
        Polygon loop (either orientation; sign comes from an even-odd
        crossing test, so the loop must be simple/non-self-intersecting).
    chunk : int, optional
        Query points are processed in chunks of this size to bound the
        ``(n_queries, n_segments)`` intermediate.
    """
    verts = torch.as_tensor(vertices, dtype=torch.float64)

    def phi(x):
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, 2)  # honor the (..., d) implicit-function contract
        v = verts.to(dtype=x.dtype, device=x.device)
        a = v
        b = torch.roll(v, -1, dims=0)
        out = torch.empty(x.shape[0], dtype=x.dtype, device=x.device)
        for s in range(0, x.shape[0], chunk):
            p = x[s : s + chunk]
            pa = p[:, None, :] - a[None, :, :]
            ba = (b - a)[None, :, :]
            # clamp_min: a zero-length segment (duplicate consecutive
            # vertex) would make t = 0/0 = NaN, and the min over segments
            # then poisons EVERY query; clamped, the degenerate segment
            # reduces to distance-to-point.
            t = ((pa * ba).sum(-1) / (ba * ba).sum(-1).clamp_min(1e-300)).clamp(
                0.0, 1.0
            )
            d2 = ((pa - t[..., None] * ba) ** 2).sum(-1)
            dist = d2.min(dim=1).values.clamp_min(1e-300).sqrt()
            ay, by = a[None, :, 1], b[None, :, 1]
            py = p[:, 1:2]
            straddle = (ay > py) != (by > py)
            t_cross = (py - ay) / (by - ay + 1e-300)
            x_cross = a[None, :, 0] + t_cross * (b[None, :, 0] - a[None, :, 0])
            crossings = (straddle & (x_cross > p[:, 0:1])).sum(dim=1)
            sign = torch.where(crossings % 2 == 1, -1.0, 1.0).to(x.dtype)
            out[s : s + chunk] = sign * dist
        return out.reshape(batch_shape)

    return phi


def project_to_zero_set(
    phi: ImplicitFunction, x: torch.Tensor, iters: int = 3
) -> torch.Tensor:
    """Newton-project points onto ``phi = 0`` along the autograd gradient.

    Robust to gradient kinks (corners of CSG shapes, polygon vertices,
    points landing exactly on the surface): non-finite steps are replaced
    by zero, so a point already on the zero set stays put.
    """
    for _ in range(iters):
        xg = x.detach().requires_grad_(True)
        f = phi(xg)
        if not f.requires_grad:  # phi ignores x (e.g. a constant field)
            return x.detach()
        (g,) = torch.autograd.grad(f.sum(), xg, allow_unused=True)
        if g is None:
            return x.detach()
        step = f.detach()[:, None] * g / (g * g).sum(-1, keepdim=True).clamp_min(1e-30)
        step = torch.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)
        x = (xg.detach() - step).detach()
    return x
