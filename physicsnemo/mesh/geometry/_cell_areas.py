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

"""Cell area (n-simplex volume) computation for simplicial meshes.

Computes the volume of each n-simplex from its edge vectors using
dimension-specific closed-form expressions where possible:

- **Edges** (n=1): vector norm.
- **Triangles** (n=2): Lagrange identity (works in any spatial dimension).
- **Tetrahedra** (n=3): scalar triple product in 3-space, or Sarrus' rule
  on the 3x3 Gram matrix for higher spatial dimensions.
- **General** (n>=4): Gram determinant via ``torch.det``.

The closed-form branches use only multiply-add-sqrt operations, so they
support reduced-precision dtypes (bfloat16, float16) natively. The general
fallback disables ``torch.autocast`` to keep ``torch.matmul`` in the
native dtype, since ``torch.det`` dispatches to cuBLAS LU factorization
which does not support reduced-precision dtypes.
"""

import math

import torch
from jaxtyping import Float


def compute_cell_areas(
    relative_vectors: Float[torch.Tensor, "n_cells n_manifold_dims n_spatial_dims"],
) -> Float[torch.Tensor, " n_cells"]:
    r"""Compute volumes (areas) of n-simplices from edge vectors.

    Given the edge vectors ``e_i = v_{i+1} - v_0`` for each simplex, computes
    the n-dimensional volume:

    .. math::
        \text{vol} = \frac{1}{n!} \sqrt{\lvert \det(E E^T) \rvert}

    where :math:`E` is the matrix whose rows are the edge vectors. Specialized
    closed-form expressions are used for :math:`n \le 3` (see module docstring).

    Parameters
    ----------
    relative_vectors : torch.Tensor
        Edge vectors of shape ``(n_cells, n_manifold_dims, n_spatial_dims)``.
        Row ``i`` is the vector from vertex 0 to vertex ``i+1`` of each simplex.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(n_cells,)`` with the volume of each simplex.
        For 1-simplices this is edge length, for 2-simplices triangle area,
        for 3-simplices tetrahedral volume, etc.

    Examples
    --------
    >>> # Unit right triangle in 2D
    >>> vecs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    >>> compute_cell_areas(vecs)
    tensor([0.5000])

    >>> # Unit edge in 3D
    >>> vecs = torch.tensor([[[1.0, 0.0, 0.0]]])
    >>> compute_cell_areas(vecs)
    tensor([1.])

    >>> # Regular tetrahedron
    >>> vecs = torch.tensor([[[1.0, 0.0, 0.0],
    ...                       [0.5, 0.866025, 0.0],
    ...                       [0.5, 0.288675, 0.816497]]])
    >>> compute_cell_areas(vecs).item()  # doctest: +SKIP
    0.1178...
    """
    n_manifold_dims = relative_vectors.shape[-2]

    match n_manifold_dims:
        case 1:
            result = _edge_lengths(relative_vectors)
        case 2:
            result = _triangle_areas(relative_vectors)
        case 3:
            result = _tetrahedron_volumes(relative_vectors)
        case _:
            result = _gram_det_volumes(relative_vectors)

    # Lock the dtype contract: under CUDA ``torch.autocast`` (e.g. bf16),
    # reductions like ``aten::sum`` that the closed-form branches rely on are on
    # the fp32 cast list, so the result can silently come back as fp32 even when
    # ``relative_vectors`` is bf16.
    return result.to(relative_vectors.dtype)


# ---------------------------------------------------------------------------
# Specialized branches
# ---------------------------------------------------------------------------


def _edge_lengths(
    relative_vectors: Float[torch.Tensor, "n_cells 1 n_spatial_dims"],
) -> Float[torch.Tensor, " n_cells"]:
    """Edge length = ||e1||."""
    return relative_vectors[:, 0].norm(dim=-1)


def _triangle_areas(
    relative_vectors: Float[torch.Tensor, "n_cells 2 n_spatial_dims"],
) -> Float[torch.Tensor, " n_cells"]:
    r"""Triangle area via Lagrange's identity (any spatial dimension).

    .. math::
        A = \tfrac{1}{2}\sqrt{\|e_1\|^2 \|e_2\|^2 - (e_1 \cdot e_2)^2}

    This is equivalent to ``||e1 x e2|| / 2`` but generalises beyond 3-space.
    """
    e1, e2 = relative_vectors[:, 0], relative_vectors[:, 1]
    d11 = (e1 * e1).sum(-1)
    d22 = (e2 * e2).sum(-1)
    d12 = (e1 * e2).sum(-1)
    # clamp guards against tiny negative values from floating-point roundoff
    return (d11 * d22 - d12 * d12).clamp(min=0).sqrt() / 2


def _tetrahedron_volumes(
    relative_vectors: Float[torch.Tensor, "n_cells 3 n_spatial_dims"],
) -> Float[torch.Tensor, " n_cells"]:
    """Tetrahedral volume, dispatching on spatial dimension."""
    n_spatial_dims = relative_vectors.shape[-1]
    if n_spatial_dims == 3:
        return _tetrahedron_volumes_3d(relative_vectors)
    return _tetrahedron_volumes_general(relative_vectors)


def _tetrahedron_volumes_3d(
    relative_vectors: Float[torch.Tensor, "n_cells 3 3"],
) -> Float[torch.Tensor, " n_cells"]:
    r"""Tetrahedral volume via scalar triple product (3D only).

    .. math::
        V = \frac{1}{6} \lvert e_1 \cdot (e_2 \times e_3) \rvert
    """
    e1, e2, e3 = relative_vectors[:, 0], relative_vectors[:, 1], relative_vectors[:, 2]
    return (e1 * torch.linalg.cross(e2, e3)).sum(-1).abs() / 6


def _tetrahedron_volumes_general(
    relative_vectors: Float[torch.Tensor, "n_cells 3 n_spatial_dims"],
) -> Float[torch.Tensor, " n_cells"]:
    r"""Tetrahedral volume via Sarrus' rule on the 3x3 Gram matrix.

    Computes the 6 unique entries of the symmetric Gram matrix
    :math:`G_{ij} = e_i \cdot e_j` and evaluates its determinant with the
    closed-form 3x3 expansion. Works for any spatial dimension >= 3.
    """
    e1, e2, e3 = relative_vectors[:, 0], relative_vectors[:, 1], relative_vectors[:, 2]
    ### 6 unique dot products (G is symmetric)
    g11 = (e1 * e1).sum(-1)
    g22 = (e2 * e2).sum(-1)
    g33 = (e3 * e3).sum(-1)
    g12 = (e1 * e2).sum(-1)
    g13 = (e1 * e3).sum(-1)
    g23 = (e2 * e3).sum(-1)
    ### Sarrus' rule: det(G) expanded along first row
    det_G = (
        g11 * (g22 * g33 - g23 * g23)
        - g12 * (g12 * g33 - g23 * g13)
        + g13 * (g12 * g23 - g22 * g13)
    )
    return det_G.clamp(min=0).sqrt() / 6


def _gram_det_volumes(
    relative_vectors: Float[torch.Tensor, "n_cells n_manifold_dims n_spatial_dims"],
) -> Float[torch.Tensor, " n_cells"]:
    r"""General n-simplex volume via Gram determinant (n >= 4).

    Falls back to ``torch.matmul`` + ``torch.det`` for manifold dimensions
    that lack a closed-form specialization. Disables ``torch.autocast`` so
    that ``matmul`` operates in the native dtype of the input, because
    ``torch.det`` dispatches to cuBLAS LU factorization which does not
    support reduced-precision dtypes (bfloat16, float16).
    """
    with torch.autocast(device_type=relative_vectors.device.type, enabled=False):
        gram_matrix = torch.matmul(
            relative_vectors,
            relative_vectors.transpose(-2, -1),
        )
        n_manifold_dims = relative_vectors.shape[-2]
        factorial = math.factorial(n_manifold_dims)
        return gram_matrix.det().abs().sqrt() / factorial
