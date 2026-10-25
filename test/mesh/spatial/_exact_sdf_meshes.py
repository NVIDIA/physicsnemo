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

"""Meshes, maps and a call helper for the tests of
:func:`physicsnemo.mesh.spatial.exact_signed_distance`."""

import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.spatial import exact_signed_distance

#: Leg length of the simplices of :func:`soup`.
CELL_SIZE = 0.2


def exact_sdf(query_points, points, cells):
    """``exact_signed_distance`` on the mesh of ``points`` and ``cells``."""
    return exact_signed_distance(Mesh(points=points, cells=cells), query_points)


def cloud(n: int, d: int = 3, seed: int = 0, *, dtype=torch.float64):
    """A random anisotropic cloud: ``points (n, d)`` and unit ``normals (n, d)``."""
    g = torch.Generator().manual_seed(seed)
    scale = torch.arange(d, 0, -1, dtype=dtype)  # (d, d-1, ..., 1)
    pts = torch.randn(n, d, generator=g, dtype=dtype) * scale
    nrm = torch.nn.functional.normalize(
        torch.randn(n, d, generator=g, dtype=dtype), dim=-1
    )
    return pts, nrm


def soup(points, normals, size: float = CELL_SIZE) -> Mesh:
    """A ``Mesh[d-1, d]`` of one right-angled simplex of legs ``size`` per point,
    its vertex mean the point, in the plane orthogonal to its normal and oriented
    so that its cell normal is that normal."""
    n, d = points.shape
    nrm = torch.nn.functional.normalize(normals, dim=-1)
    g = torch.Generator().manual_seed(0)
    ### An orthonormal tangent basis per cell: QR of [n | random], columns 1..d-1.
    rnd = torch.randn(n, d, d - 1, generator=g, dtype=points.dtype)
    q, _ = torch.linalg.qr(torch.cat([nrm[..., None], rnd], -1))
    tangents = q[..., 1:]  # (n, d, d-1)
    offsets = torch.cat([torch.zeros(n, d, 1, dtype=points.dtype), tangents], -1) * size
    offsets = offsets - offsets.mean(-1, keepdim=True)  # vertex mean = the point
    verts = (points[:, :, None] + offsets).transpose(1, 2).reshape(n * d, d)
    cells = torch.arange(n * d).reshape(n, d)
    ### Orient every cell: swapping its last two vertices flips its normal.
    flip = (Mesh(points=verts, cells=cells).cell_normals * nrm).sum(-1) < 0
    cells[flip, -2:] = cells[flip, -2:].flip(-1)
    return Mesh(points=verts, cells=cells)


def transform(mesh: Mesh, R=None, shift=None, scale: float = 1.0) -> Mesh:
    """The mesh under the similarity ``x -> scale R x + shift``, with the cells
    re-oriented so their normals follow the map (under an improper map a cell's
    normal, which follows its vertex order, would come out as ``det(R) R n``)."""
    d = mesh.n_spatial_dims
    dtype, device = mesh.points.dtype, mesh.points.device
    R = (
        torch.eye(d, dtype=dtype, device=device)
        if R is None
        else R.to(dtype=dtype, device=device)
    )
    pts = scale * mesh.points @ R.T
    if shift is not None:
        pts = pts + shift.to(dtype=dtype, device=device)
    cells = mesh.cells.clone()
    wanted = mesh.cell_normals @ R.T
    flip = (Mesh(points=pts, cells=cells).cell_normals * wanted).sum(-1) < 0
    cells[flip, -2:] = cells[flip, -2:].flip(-1)
    return Mesh(points=pts, cells=cells)


def orthogonal(d: int, seed: int, improper: bool = False):
    """A random orthogonal map with ``det = -1`` when ``improper``, else ``+1``."""
    q, _ = torch.linalg.qr(
        torch.randn(
            d, d, dtype=torch.float64, generator=torch.Generator().manual_seed(seed)
        )
    )
    if (torch.det(q) < 0) != improper:
        q[:, 0] = -q[:, 0]
    return q
