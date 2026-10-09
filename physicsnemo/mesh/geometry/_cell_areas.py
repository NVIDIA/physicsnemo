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
- **Triangles** (n=2): cross product in 3-space, or exterior product in
  other spatial dimensions, with a rescaled norm.
- **Tetrahedra** (n=3): scalar triple product in 3-space, or the norm of the
  3x3 minors of the edge vectors for higher spatial dimensions.
- **General** (n>=4): Gram determinant via ``torch.det``.

The closed-form branches use only multiply-add-sqrt operations, so they
support reduced-precision dtypes (bfloat16, float16) natively. The general
fallback disables ``torch.autocast`` to keep ``torch.matmul`` in the
native dtype, since ``torch.det`` dispatches to cuBLAS LU factorization
which does not support reduced-precision dtypes.
"""

import itertools
import math

import torch
from jaxtyping import Float

from physicsnemo.utils._small_linalg import small_det


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
    r"""Triangle area from the exterior product (any spatial dimension).

    .. math::
        A = \tfrac{1}{2}\sqrt{\sum_{i<j}(e_{1,i}e_{2,j}-e_{1,j}e_{2,i})^2}

    Direct minors avoid subtracting nearly equal squared dot products for
    thin triangles. Rescaling before the norm avoids squaring tiny or large
    area components; in 3D these are the usual cross product components.
    """
    e1, e2 = relative_vectors[:, 0], relative_vectors[:, 1]
    n_spatial_dims = relative_vectors.shape[-1]
    if n_spatial_dims == 2:
        return small_det(relative_vectors).abs() / 2
    if n_spatial_dims == 3:
        components = torch.linalg.cross(e1, e2)
    else:
        i, j = torch.triu_indices(
            n_spatial_dims, n_spatial_dims, offset=1, device=relative_vectors.device
        )
        components = e1[:, i] * e2[:, j] - e1[:, j] * e2[:, i]
    scale = components.abs().amax(dim=-1)
    scaled = components / scale.masked_fill(scale == 0, 1).unsqueeze(-1)
    return scaled.norm(dim=-1) * (scale / 2)


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
    return small_det(relative_vectors).abs() / 6


def _tetrahedron_volumes_general(
    relative_vectors: Float[torch.Tensor, "n_cells 3 n_spatial_dims"],
) -> Float[torch.Tensor, " n_cells"]:
    r"""Tetrahedral volume from the 3x3 minors of the edge vectors.

    By the Cauchy-Binet formula, :math:`\det(E E^T)` is the sum of the squared
    3x3 minors of :math:`E`, so the volume is their norm over 6. As for
    triangles, direct minors avoid the cancellation of the Gram determinant for
    thin tetrahedra, and rescaling before the norm avoids squaring tiny or large
    minors. Works for any spatial dimension >= 3.
    """
    n_spatial_dims = relative_vectors.shape[-1]
    # Column slices rather than an index tensor, which a CUDA H2D copy would need
    minors = torch.stack(
        [
            small_det(torch.stack([relative_vectors[..., c] for c in columns], dim=-1))
            for columns in itertools.combinations(range(n_spatial_dims), 3)
        ],
        dim=-1,
    )  # (n_cells, n_minors)
    scale = minors.abs().amax(dim=-1)
    scaled = minors / scale.masked_fill(scale == 0, 1).unsqueeze(-1)
    return scaled.norm(dim=-1) * (scale / 6)


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
