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

"""Cell normal computation for codimension-1 simplicial meshes.

Computes unit normal vectors for each cell using the generalized cross
product (Hodge star), with dimension-specific closed-form expressions
where possible:

- **Edges in 2D** (d=2): 90-degree counterclockwise rotation.
- **Triangles in 3D** (d=3): ``torch.linalg.cross``.
- **General** (d>=4): signed minor determinants of the edge-vector matrix.

The closed-form branches for d=2 and d=3 use only multiply-add
operations, so they support reduced-precision dtypes (bfloat16, float16)
natively. The general fallback disables ``torch.autocast`` to keep
``torch.det`` in the native dtype, since it dispatches to cuBLAS LU
factorization which does not support reduced-precision dtypes.
"""

import torch
import torch.nn.functional as F
from jaxtyping import Float


def compute_cell_normals(
    relative_vectors: Float[torch.Tensor, "n_cells n_manifold_dims n_spatial_dims"],
) -> Float[torch.Tensor, "n_cells n_spatial_dims"]:
    """Compute unit normal vectors for codimension-1 simplices.

    Given the edge vectors ``e_i = v_{i+1} - v_0`` for each simplex, computes
    the outward-pointing unit normal via the generalized cross product.
    The caller must ensure the codimension-1 constraint:
    ``n_manifold_dims == n_spatial_dims - 1``.

    Parameters
    ----------
    relative_vectors : Float[torch.Tensor, "n_cells n_manifold_dims n_spatial_dims"]
        Edge vectors. Row ``i`` is the vector from vertex 0 to vertex ``i+1``
        of each simplex. Must satisfy ``n_manifold_dims == n_spatial_dims - 1``.

    Returns
    -------
    Float[torch.Tensor, "n_cells n_spatial_dims"]
        Unit normal vectors. For degenerate cells (zero-area), the normal is
        a zero vector (from ``F.normalize``'s default behavior).

    Examples
    --------
    >>> # Edge in 2D: normal is 90-degree CCW rotation
    >>> vecs = torch.tensor([[[1.0, 0.0]]])
    >>> compute_cell_normals(vecs)
    tensor([[-0., 1.]])

    >>> # Triangle in XY-plane: normal is +Z
    >>> vecs = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    >>> compute_cell_normals(vecs)
    tensor([[0., 0., 1.]])
    """
    n_spatial_dims = relative_vectors.shape[-1]

    match n_spatial_dims:
        case 2:
            result = _normals_2d(relative_vectors)
        case 3:
            result = _normals_3d(relative_vectors)
        case _:
            result = _normals_general(relative_vectors)

    # Lock the dtype contract: under CUDA ``torch.autocast`` (e.g. bf16),
    # ``F.normalize`` calls ``aten::norm`` which is on the fp32 cast list, so
    # the closed-form branches can silently return fp32 even when
    # ``relative_vectors`` is bf16.
    return result.to(relative_vectors.dtype)


# ---------------------------------------------------------------------------
# Specialized branches
# ---------------------------------------------------------------------------


def _normals_2d(
    relative_vectors: Float[torch.Tensor, "n_cells 1 2"],
) -> Float[torch.Tensor, "n_cells 2"]:
    """Edge normals in 2D via 90-degree CCW rotation: (x, y) -> (-y, x)."""
    e = relative_vectors[:, 0]  # (n_cells, 2)
    normals = torch.stack([-e[:, 1], e[:, 0]], dim=-1)
    return F.normalize(normals, dim=-1)


def _normals_3d(
    relative_vectors: Float[torch.Tensor, "n_cells 2 3"],
) -> Float[torch.Tensor, "n_cells 3"]:
    """Triangle normals in 3D via cross product."""
    normals = torch.linalg.cross(relative_vectors[:, 0], relative_vectors[:, 1])
    return F.normalize(normals, dim=-1)


def _normals_general(
    relative_vectors: Float[torch.Tensor, "n_cells n_manifold_dims n_spatial_dims"],
) -> Float[torch.Tensor, "n_cells n_spatial_dims"]:
    r"""Normals in :math:`d \ge 4` via signed minor determinants (Hodge star).

    For :math:`n - 1` vectors in :math:`\mathbb{R}^n` (rows of :math:`E`),
    the normal components are:

    .. math::
        n_i = (-1)^{n - 1 + i} \det(E_{\setminus i})

    where :math:`E_{\setminus i}` is :math:`E` with column ``i`` removed.

    Disables ``torch.autocast`` because ``torch.det`` dispatches to cuBLAS
    LU factorization which does not support reduced-precision dtypes.
    """
    n_spatial_dims = relative_vectors.shape[-1]
    n_manifold_dims = relative_vectors.shape[-2]

    with torch.autocast(device_type=relative_vectors.device.type, enabled=False):
        normal_components: list[torch.Tensor] = []

        for i in range(n_spatial_dims):
            # (n-1)x(n-1) submatrix: remove column i
            # Uses slice concatenation to avoid aten.nonzero (torch.compile
            # graph break from dynamic shapes).
            submatrix = torch.cat(
                [relative_vectors[:, :, :i], relative_vectors[:, :, i + 1 :]],
                dim=-1,
            )
            det = submatrix.det()
            sign = (-1) ** (n_manifold_dims + i)
            normal_components.append(sign * det)

        normals = torch.stack(normal_components, dim=-1)

    return F.normalize(normals, dim=-1)
