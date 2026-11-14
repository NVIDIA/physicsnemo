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

"""Boundary meshes for the ISLA tests.

The model reads a boundary ``Mesh[d-1, d]`` through its cell centroids, cell
normals and effective cell measures, and an interior model measures its
prediction points' distance to the surface its cells make up. :func:`soup`
builds such a mesh from a prescribed cloud of centroids, unit normals and
measures: one right-angled simplex per cell, centred on its point in the plane
orthogonal to its normal, oriented so that the cell normal is that normal (to
roundoff) and sized so that its geometric measure is the prescribed measure, so
the soup is a whole boundary (its cells cover all the surface their measures
stand for) and the measures are stored as the effective cell measures as well.
It works in any spatial dimension (segments in two, triangles in three,
tetrahedra in four).
"""

import math

import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import EFFECTIVE_MEASURE_KEY

#: Leg length of the simplices of :func:`soup` when no measures are prescribed
#: (small against the clouds of the tests, large enough that the normals
#: recomputed from the vertices carry no appreciable roundoff).
CELL_SIZE = 0.2


def cloud(n: int, d: int = 3, seed: int = 0, *, dtype=torch.float64, scale=None):
    """A random anisotropic cloud: ``points (n, d)``, unit ``normals (n, d)``,
    positive ``measures (n,)`` and a unit global direction ``(d,)``."""
    g = torch.Generator().manual_seed(seed)
    if scale is None:
        scale = torch.arange(d, 0, -1, dtype=dtype)  # (d, d-1, ..., 1)
    else:
        scale = torch.as_tensor(scale, dtype=dtype)
    pts = torch.randn(n, d, generator=g, dtype=dtype) * scale
    nrm = torch.nn.functional.normalize(
        torch.randn(n, d, generator=g, dtype=dtype), dim=-1
    )
    w = torch.rand(n, generator=g, dtype=dtype) + 0.5
    drv = torch.nn.functional.normalize(
        torch.randn(d, generator=g, dtype=dtype), dim=-1
    )
    return pts, nrm, w, drv


def soup(
    points, normals, measures=None, cell_data=None, size: float = CELL_SIZE
) -> Mesh:
    """A ``Mesh[d-1, d]`` whose cell centroids are ``points``, cell normals
    ``normals`` and cell measures ``measures`` (geometric and effective; ``None``:
    every cell a simplex of legs ``size``). ``cell_data`` adds per-cell fields."""
    n, d = points.shape
    if measures is not None:
        ### a right-angled simplex of legs s has measure s^(d-1) / (d-1)!
        size = (math.factorial(d - 1) * measures.to(points.dtype)) ** (1.0 / (d - 1))
        size = size[:, None, None]
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
    data = dict(cell_data or {})
    if measures is not None:
        data[EFFECTIVE_MEASURE_KEY] = measures
    return Mesh(points=verts, cells=cells, cell_data=data)


def transform(mesh: Mesh, R=None, shift=None, scale: float = 1.0, specs=None) -> Mesh:
    """The mesh under the similarity ``x -> scale R x + shift``: the points are
    moved, the effective measures scale with ``scale**(d-1)``, and the cell
    fields declared in ``specs`` (name -> RankSpec) transform by their rank
    (rank 0 unchanged, rank 1 ``R v``, rank 2 ``R T R^T``)."""
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
    ### A cell normal follows the vertex ordering, so under an improper map it
    ### comes out as det(R) R n; the boundary's normal is the true vector R n, so
    ### re-orient the cells whose recomputed normal points the other way.
    cells = mesh.cells.clone()
    wanted = mesh.cell_normals @ R.T
    flip = (Mesh(points=pts, cells=cells).cell_normals * wanted).sum(-1) < 0
    cells[flip, -2:] = cells[flip, -2:].flip(-1)
    data = {}
    for key in mesh.cell_data.keys(include_nested=True, leaves_only=True):
        name = ".".join(key) if isinstance(key, tuple) else key
        t = mesh.cell_data[key]
        if name == EFFECTIVE_MEASURE_KEY:
            t = t * scale ** (d - 1)
        elif specs is not None:
            rank = specs[name].rank
            t = t if rank == 0 else (t @ R.T if rank == 1 else R @ t @ R.T)
        data[key] = t
    return Mesh(points=pts, cells=cells, cell_data=data)


def rescale_measures(mesh: Mesh, k: float) -> Mesh:
    """The same mesh with its effective measures multiplied by ``k`` (other cell
    data kept)."""
    data = mesh.cell_data.clone()
    data[EFFECTIVE_MEASURE_KEY] = data[EFFECTIVE_MEASURE_KEY] * k
    return mesh.with_data(cell_data=data)


def refine(mesh: Mesh) -> Mesh:
    """Every cell as two coincident half-measure copies: the same discrete
    boundary measure, re-represented."""
    w = mesh.cell_data[EFFECTIVE_MEASURE_KEY]
    return Mesh(
        points=mesh.points,
        cells=mesh.cells.repeat_interleave(2, 0),
        cell_data={EFFECTIVE_MEASURE_KEY: w.repeat_interleave(2) / 2},
    )


def points_mesh(points, point_data=None) -> Mesh:
    """A point-cloud Mesh of prediction points with optional per-point fields."""
    return Mesh(points=points, point_data={} if point_data is None else point_data)


def law(x, R, rank: int):
    """``R^{(x)rank} x`` on the trailing ``rank`` axes of an unbatched output ``x``."""
    y = x
    for k in range(rank):
        y = torch.movedim(
            torch.tensordot(y, R.to(x.dtype), dims=([1 + k], [1])), -1, 1 + k
        )
    return y


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


def outputs(mesh: Mesh) -> dict[str, torch.Tensor]:
    """The output Mesh's point_data as a flat ``{dotted name: tensor}`` mapping."""
    return {
        (".".join(k) if isinstance(k, tuple) else k): mesh.point_data[k]
        for k in mesh.point_data.keys(include_nested=True, leaves_only=True)
    }


def same(a, b, atol, rtol=0.0) -> bool:
    """Two output Meshes (or flat output dicts) agree field by field, to ``atol``
    absolute (the contracts' stated bounds are absolute: no relative term unless the
    caller asks for one)."""
    a = outputs(a) if isinstance(a, Mesh) else a
    b = outputs(b) if isinstance(b, Mesh) else b
    assert a.keys() == b.keys()
    return all(torch.allclose(a[k], b[k], atol=atol, rtol=rtol) for k in a)


def live(model, seed: int = 0):
    """``model`` with the output projection of every boundary read of the interior
    task (``out``, zero-initialized so that a fresh read adds nothing) given seeded
    random values (``randn * 0.1``), so an interior contract test exercises the read;
    returns the model. A surface model, which has no read, is returned as it is."""
    if getattr(model, "kernel_reads", None) is None:
        return model
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for read in model.kernel_reads:
            w = read.out.weight
            w.copy_(torch.randn(w.shape, generator=gen, dtype=w.dtype) * 0.1)
    return model
