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

r"""Dual mesh (circumcentric/Voronoi) volume computation and DEC dual operators.

This module provides the unified implementation of dual cell volumes (Voronoi regions),
circumcenters, and cotangent weights for n-dimensional simplicial meshes. These are
fundamental to both:

- Discrete Exterior Calculus (DEC) operators (Hodge star, Laplacian, etc.)
- Discrete differential geometry (curvature computations)

Dual 0-cell volumes follow Meyer et al. (2003), *Discrete Differential-Geometry
Operators for Triangulated 2-Manifolds*, for 2D manifolds, using the mixed
Voronoi area approach that handles both acute and obtuse triangles correctly.
For higher dimensions, barycentric approximation is used as rigorous circumcentric
dual volumes require well-centered meshes (Desbrun et al. 2005, *Discrete Exterior
Calculus*; Hirani 2003, *Discrete Exterior Calculus* (PhD thesis)).

Circumcenters and cotangent weights are computed using the perpendicular bisector
method and FEM stiffness matrix approach, respectively, following Desbrun et al.
(2005), *Discrete Exterior Calculus*, §3 (Primal Simplicial Complex and Dual
Cell Complex) and §9 (Divergence and Laplace–Beltrami).

References
----------
Meyer, M., Desbrun, M., Schröder, P., & Barr, A. H. (2003).
*Discrete Differential-Geometry Operators for Triangulated 2-Manifolds*.
In: Visualization and Mathematics III, pp. 35-57.

Desbrun, M., Hirani, A. N., Leok, M., & Marsden, J. E. (2005).
*Discrete Exterior Calculus*. arXiv:math/0508341v2.

Hirani, A. N. (2003). *Discrete Exterior Calculus*. PhD thesis, California
Institute of Technology.
"""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float, Int

from physicsnemo.mesh.utilities._gather import gather_rows
from physicsnemo.mesh.utilities._tolerances import safe_eps
from physicsnemo.utils._small_linalg import (
    best_base_first,
    normalized_gram_inverse,
    normalized_inverse,
    small_lstsq,
)

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _scatter_add_cell_contributions_to_vertices(
    dual_volumes: Float[torch.Tensor, " n_points"],
    cells: Int[torch.Tensor, "n_cells n_vertices_per_cell"],
    contributions: Float[torch.Tensor, "n_cells ..."],
) -> None:
    """Scatter cell volume contributions to all cell vertices (in place).

    Accepts either a uniform per-cell contribution (broadcast to all vertices)
    or distinct per-vertex contributions.

    Parameters
    ----------
    dual_volumes : torch.Tensor
        Accumulator for dual volumes, shape ``(n_points,)``. Modified in place.
    cells : torch.Tensor
        Cell connectivity, shape ``(n_cells, n_vertices_per_cell)``.
    contributions : torch.Tensor
        If 1-D ``(n_cells,)``: each cell contributes the same value to all
        its vertices (e.g. ``volume / n_verts``).
        If 2-D ``(n_cells, n_vertices_per_cell)``: per-vertex contributions
        (e.g. Meyer mixed Voronoi areas).

    Examples
    --------
    >>> import torch
    >>> dual_volumes = torch.zeros(4)
    >>> cells = torch.tensor([[0, 1, 2], [1, 2, 3]])
    >>> # Uniform: 1/3 of each triangle area to every vertex
    >>> _scatter_add_cell_contributions_to_vertices(
    ...     dual_volumes, cells, torch.tensor([0.5, 0.5]) / 3.0
    ... )
    >>> # Per-vertex: different contribution per corner
    >>> dual_volumes2 = torch.zeros(4)
    >>> _scatter_add_cell_contributions_to_vertices(
    ...     dual_volumes2, cells, torch.tensor([[0.1, 0.2, 0.2], [0.15, 0.15, 0.2]])
    ... )
    """
    if contributions.ndim not in (1, 2):
        raise ValueError(
            f"contributions must be 1D or 2D, got {contributions.ndim}D "
            f"with shape {tuple(contributions.shape)}"
        )
    if contributions.ndim == 1:
        contributions = contributions.unsqueeze(-1).expand_as(cells)
    dual_volumes.scatter_add_(0, cells.flatten(), contributions.reshape(-1))


def _compute_meyer_mixed_voronoi_areas(
    cell_vertices: Float[torch.Tensor, "n_cells 3 n_spatial_dims"],
    cell_areas: Float[torch.Tensor, " n_cells"],
) -> Float[torch.Tensor, " n_cells_times_3"]:
    r"""Compute per-(cell, local_vertex) mixed Voronoi areas.

    Implements the branchless mixed Voronoi area formula of Meyer et al. (2003),
    *Discrete Differential-Geometry Operators for Triangulated 2-Manifolds*, for
    triangular meshes, handling both acute and obtuse triangles correctly. For
    acute triangles, uses the circumcentric Voronoi formula (Meyer et al. 2003,
    Eq. 7). For obtuse triangles, uses the mixed area subdivision (Meyer et al.
    2003, Fig. 4).

    Parameters
    ----------
    cell_vertices : Float[torch.Tensor, "n_cells 3 n_spatial_dims"]
        Vertex positions for each triangle cell.
    cell_areas : Float[torch.Tensor, " n_cells"]
        Area of each triangle cell.

    Returns
    -------
    Float[torch.Tensor, " n_cells_times_3"]
        Per-(cell, local_vertex) Voronoi areas, shape ``(n_cells * 3,)``.
        Ordered as ``[cell0_v0, cell0_v1, cell0_v2, cell1_v0, ...]``, i.e.
        the flattened ``(n_cells, 3)`` tensor where column ``j`` corresponds
        to local vertex ``j``.

    References
    ----------
    Meyer, M., Desbrun, M., Schröder, P., & Barr, A. H. (2003).
    *Discrete Differential-Geometry Operators for Triangulated 2-Manifolds*.
    §3.3 (Equation 7) and §3.4 (Figure 4).
    """
    from physicsnemo.mesh.geometry._angles import stable_angle_between_vectors

    ### Edge vectors e_k = v_{k+1} - v_k, each computed once
    # The edges from vertex k are e_k (to the next vertex) and v_{k+2} - v_k,
    # which is exactly -e_{k+2} in IEEE arithmetic, with the same squared length.
    vertices = [cell_vertices[:, k, :] for k in range(3)]
    edges = [vertices[(k + 1) % 3] - vertices[k] for k in range(3)]
    edges_sq = [(edge**2).sum(dim=-1) for edge in edges]  # (n_cells,) each

    ### Angle at each local vertex: (3, n_cells), row j = angle at vertex j
    all_angles = torch.stack(
        [stable_angle_between_vectors(edges[k], -edges[(k + 2) % 3]) for k in range(3)]
    )

    # Check if triangle is obtuse (any angle > pi/2)
    is_obtuse = torch.any(all_angles > torch.pi / 2, dim=0)  # (n_cells,)

    ### Branchless computation of mixed Voronoi areas
    # Computes both acute (Eq. 7) and obtuse (Fig. 4) formulas for all cells,
    # then selects per-cell via torch.where. This avoids data-dependent branching
    # that would break torch.compile.
    eps = safe_eps(all_angles.dtype)
    # One contiguous row at a time: vectorized and scalar cos and sin can round
    # differently, so the elements keep their positions in each kernel call.
    cot = [
        torch.cos(angles) / torch.sin(angles).clamp(min=eps) for angles in all_angles
    ]  # (n_cells,) each

    voronoi_per_vertex = []
    for local_v_idx in range(3):
        next_idx = (local_v_idx + 1) % 3
        prev_idx = (local_v_idx + 2) % 3

        ### Voronoi contribution (Eq. 7) - computed for ALL cells
        # |edge to next|^2 * cot(angle at prev) + |edge to prev|^2 * cot(angle at next)
        voronoi_contribution = (
            edges_sq[local_v_idx] * cot[prev_idx] + edges_sq[prev_idx] * cot[next_idx]
        ) / 8.0  # (n_cells,)

        ### Mixed-area contribution (Figure 4) - computed for ALL cells
        is_obtuse_at_vertex = all_angles[local_v_idx] > torch.pi / 2
        mixed_contribution = torch.where(
            is_obtuse_at_vertex,
            cell_areas / 2.0,
            cell_areas / 4.0,
        )  # (n_cells,)

        ### Select per cell: Voronoi for acute, mixed for obtuse
        voronoi_per_vertex.append(
            torch.where(is_obtuse, mixed_contribution, voronoi_contribution)
        )

    return torch.stack(voronoi_per_vertex, dim=1).reshape(-1)  # (n_cells * 3,)


def compute_dual_volumes_0(mesh: "Mesh") -> Float[torch.Tensor, " n_points"]:
    r"""Compute circumcentric dual 0-cell volumes (Voronoi regions) at mesh vertices.

    This is the unified, mathematically rigorous implementation used by both DEC
    operators and curvature computations. It replaces the previous buggy
    ``compute_dual_volumes_0()`` in ``calculus/_circumcentric_dual.py`` which failed
    on obtuse triangles (giving up to 513% conservation error).

    The dual 0-cell (also called Voronoi cell or circumcentric dual) of a vertex
    is the region of points closer to that vertex than to any other. In DEC, these
    volumes appear in the Hodge star operator and normalization of the Laplacian.

    .. note::

        In the curvature/differential geometry literature, these are often
        called "Voronoi areas" (for 2D) or "Voronoi volumes". In DEC literature,
        they are called "dual 0-cell volumes" (denoted :math:`|{\star}v|`). These
        are identical concepts.

    Dimension-specific algorithms:

    **1D manifolds (edges)**: each vertex receives half the length of each
    incident edge,

    .. math::

        V(v) = \sum_{e \ni v} \tfrac{1}{2} |e|.

    **2D manifolds (triangles)**: uses the Meyer et al. (2003) mixed-area approach.

    - For acute triangles (all angles :math:`\le \pi/2`), uses the circumcentric
      Voronoi formula (Eq. 7),

      .. math::

          V(v) = \tfrac{1}{8} \sum
              \left(\|e_i\|^{2} \cot \alpha_i + \|e_j\|^{2} \cot \alpha_j\right),

      where :math:`e_i, e_j` are edges from :math:`v` and :math:`\alpha_i, \alpha_j`
      are the opposite angles.

    - For obtuse triangles, uses the mixed area subdivision (Figure 4): if obtuse
      at vertex :math:`v`, then :math:`V(v) = \operatorname{area}(T)/2`, otherwise
      :math:`V(v) = \operatorname{area}(T)/4`.

    This branch correctly handles **both** acute and obtuse triangles. The previous
    buggy implementation in ``_circumcentric_dual.py`` assumed circumcenters were
    always inside the triangle, which is only true for acute triangles. Together
    they ensure perfect tiling and optimal error bounds.

    **3D+ manifolds (tetrahedra, etc.)**: barycentric approximation,

    .. math::

        V(v) = \sum_{\sigma \ni v} \frac{|\sigma|}{n + 1},

    where :math:`n` is the manifold dimension (so :math:`n + 1` is the number
    of vertices per cell).

    Rigorous circumcentric dual volumes in 3D+ require "well-centered" meshes
    where all circumcenters lie inside their simplices (Desbrun et al. 2005,
    *Discrete Exterior Calculus*). Mixed volume formulas for obtuse tetrahedra
    do not exist in the literature.

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh.

    Returns
    -------
    Float[torch.Tensor, " n_points"]
        Dual 0-cell volume for each vertex, shape ``(n_points,)``.
        For isolated vertices, volume is 0. Satisfies the perfect-tiling
        identity :math:`\sum_v V(v) = |M|`.

    Raises
    ------
    NotImplementedError
        If ``n_manifold_dims > 3``.

    Notes
    -----
    Mathematical properties:

    1. Conservation: :math:`\sum_v |{\star}v| = |M|` (perfect tiling).
    2. Optimality: minimizes spatial averaging error (Meyer et al. 2003,
       *Discrete Differential-Geometry Operators for Triangulated 2-Manifolds*,
       §3.2).
    3. Gauss-Bonnet: enables
       :math:`\sum_i K_i \, |{\star}v_i| = 2 \pi \chi(M)` to hold exactly.

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
    >>> mesh = two_triangles_2d.load()
    >>> dual_vols = compute_dual_volumes_0(mesh)
    >>> # Use in Hodge star: (star f)(star v) = f(v) * dual_vols[v]
    >>> # Use in Laplacian: Lap_f(v) = (1 / dual_vols[v]) * sum w_ij (f_j - f_i)

    References
    ----------
    - Meyer et al. (2003), *Discrete Differential-Geometry Operators for
      Triangulated 2-Manifolds*, Equation 7 (circumcentric Voronoi, acute
      triangles) and Figure 4 (mixed area, obtuse triangles).
    - Desbrun et al. (2005), *Discrete Exterior Calculus*, Definition 3.7
      (circumcentric duality operator).
    - Hirani (2003), *Discrete Exterior Calculus* (PhD thesis), Definition 2.4.5
      (circumcentric dual cell).
    """
    device = mesh.points.device
    n_points = mesh.n_points
    n_manifold_dims = mesh.n_manifold_dims

    ### Initialize dual volumes
    dual_volumes = torch.zeros(n_points, dtype=mesh.points.dtype, device=device)

    ### Handle empty mesh
    if mesh.n_cells == 0:
        return dual_volumes

    ### Get cell volumes (reuse existing computation)
    cell_volumes = mesh.cell_areas  # (n_cells,) - "areas" is volumes in nD

    ### Dimension-specific computation
    if n_manifold_dims == 1:
        ### 1D: Each vertex gets half the length of each incident edge
        # This is exact for piecewise linear 1-manifolds
        _scatter_add_cell_contributions_to_vertices(
            dual_volumes, mesh.cells, cell_volumes / 2.0
        )

    elif n_manifold_dims == 2:
        ### 2D: Meyer et al. (2003) mixed Voronoi area (see docstring Notes)
        # (n_cells, 3, n_spatial_dims)
        cell_vertices = gather_rows(mesh.points, mesh.cells)
        voronoi_areas = _compute_meyer_mixed_voronoi_areas(
            cell_vertices, cell_volumes
        )  # (n_cells * 3,)

        ### Scatter per-vertex Voronoi areas to global dual volumes
        _scatter_add_cell_contributions_to_vertices(
            dual_volumes, mesh.cells, voronoi_areas.reshape(mesh.n_cells, 3)
        )

    elif n_manifold_dims >= 3:
        ### 3D and higher: barycentric subdivision (see docstring Notes)
        n_vertices_per_cell = n_manifold_dims + 1
        _scatter_add_cell_contributions_to_vertices(
            dual_volumes, mesh.cells, cell_volumes / n_vertices_per_cell
        )

    else:
        raise NotImplementedError(
            f"Dual volume computation not implemented for {n_manifold_dims=}. "
            f"Currently supported: 1D (edges), 2D (triangles), 3D+ (tetrahedra, etc.)."
        )

    return dual_volumes


def _compute_triangle_circumcenters_3d(vertices: torch.Tensor) -> torch.Tensor:
    """Compute 3D triangle circumcenters with the closed-form cross formula."""
    output_dtype = vertices.dtype
    vertices_64 = vertices.to(dtype=torch.float64)

    p0 = vertices_64[:, 0, :]
    edge_01 = vertices_64[:, 1, :] - p0
    edge_02 = vertices_64[:, 2, :] - p0

    normal = torch.linalg.cross(edge_01, edge_02, dim=-1)
    normal_norm_sq = (normal * normal).sum(dim=-1, keepdim=True)

    edge_01_sq = (edge_01 * edge_01).sum(dim=-1, keepdim=True)
    edge_02_sq = (edge_02 * edge_02).sum(dim=-1, keepdim=True)

    # Scale-free degeneracy test: |normal|^2 / (|e01|^2 |e02|^2) = sin^2 of the
    # angle at p0, against the (n * eps)^2 of normalized_gram_inverse.
    # Degenerate cells return their centroid, like the general path.
    is_degenerate = ~(
        normal_norm_sq
        > (2 * torch.finfo(torch.float64).eps) ** 2 * edge_01_sq * edge_02_sq
    )
    normal_norm_sq = torch.where(is_degenerate, 1.0, normal_norm_sq)

    offset = (
        edge_01_sq * torch.linalg.cross(edge_02, normal, dim=-1)
        + edge_02_sq * torch.linalg.cross(normal, edge_01, dim=-1)
    ) / (2.0 * normal_norm_sq)

    return torch.where(is_degenerate, vertices_64.mean(dim=1), p0 + offset).to(
        dtype=output_dtype
    )


def compute_circumcenters(
    vertices: Float[torch.Tensor, "n_cells n_vertices_per_cell n_spatial_dims"],
) -> Float[torch.Tensor, "n_cells n_spatial_dims"]:
    r"""Compute circumcenters of simplices using the perpendicular-bisector method.

    The circumcenter is the unique point equidistant from all vertices of the
    simplex; it lies at the intersection of the perpendicular-bisector hyperplanes.

    Parameters
    ----------
    vertices : Float[torch.Tensor, "n_cells n_vertices_per_cell n_spatial_dims"]
        Vertex positions for each cell (simplex).

    Returns
    -------
    Float[torch.Tensor, "n_cells n_spatial_dims"]
        Circumcenters, shape ``(n_cells, n_spatial_dims)``.

    Notes
    -----
    For a simplex with vertices :math:`v_0, v_1, \ldots, v_n`, the circumcenter
    :math:`c` satisfies

    .. math::

        \|c - v_0\|^2 = \|c - v_1\|^2 = \cdots = \|c - v_n\|^2.

    Substituting :math:`d = c - v_0` yields :math:`n` linear equations,

    .. math::

        2 (v_i - v_0) \cdot d = \|v_i - v_0\|^2 \quad \text{for } i = 1, \ldots, n,

    or in matrix form :math:`A d = b` with
    :math:`A = 2 [(v_1 - v_0)^\top, (v_2 - v_0)^\top, \ldots]^\top`
    and
    :math:`b = [\|v_1 - v_0\|^2, \|v_2 - v_0\|^2, \ldots]^\top`.

    Then :math:`c = v_0 + d`. Square systems (full-dimensional cells) use the
    closed form :math:`d = \tfrac{1}{2} E^{-1} [\|e_i\|^2]_i` with
    :math:`E = A / 2` (rows :math:`v_i - v_0`), with each edge scaled to unit
    length, so the closed form stays in floating-point range at any scale.
    The base vertex :math:`v_0` is the one whose edges are closest to
    orthogonal (see ``best_base_first``), so the result does not depend on
    the order of the vertices. Degenerate cells, whose unit edges have a
    numerically zero determinant, have no circumcenter and return their
    centroid. Under-determined systems
    (manifolds embedded in a higher-dimensional space) use the minimum-norm
    least-squares solution, which lies in the cell's affine hull.
    """
    n_cells, n_verts_per_cell, n_spatial_dims = vertices.shape
    n_manifold_dims = n_verts_per_cell - 1

    ### Handle low-dimensional special cases up front
    match (n_verts_per_cell, n_spatial_dims):
        case (1, _):
            # 0-simplex: circumcenter is the vertex itself
            return vertices.squeeze(1)
        case (2, _):
            # 1-simplex (edge): circumcenter is the midpoint. Avoids numerical
            # issues with underdetermined lstsq for edges in higher dimensions.
            return vertices.mean(dim=1)
        case (3, 3):
            # In float64, which is accurate from any base vertex
            return _compute_triangle_circumcenters_3d(vertices)

    if vertices.dtype in (torch.float16, torch.bfloat16):  # tolerances need float32
        return compute_circumcenters(vertices.float()).to(vertices.dtype)

    # The circumcenter does not depend on the order of the vertices, and the
    # edges from the best base vertex give the most accurate closed forms.
    vertices, _ = best_base_first(vertices)

    ### Build linear system A @ (c - v0) = b for each simplex
    v0 = vertices[:, 0, :]  # (n_cells, n_spatial_dims)
    relative_vecs = vertices[:, 1:, :] - v0.unsqueeze(1)
    # (n_cells, n_manifold_dims, n_spatial_dims)

    if n_manifold_dims == n_spatial_dims:
        ### Square system: closed form c - v0 = inv(E) b with b_i = |e_i|^2 / 2
        # Batched solve/lstsq factorize every tiny matrix separately. With the
        # edges scaled to unit length, E = D Ê, inv(E) b = inv(Ê) (|e_i| / 2)_i,
        # which stays in range for tiny or huge cells. Degenerate cells have no
        # circumcenter and return their centroid; normalized_inverse flags only
        # cells singular to working precision, so stretched and near-degenerate
        # cells keep their exact (possibly distant) circumcenter.
        E_hat_inv, edge_lengths, is_degenerate = normalized_inverse(relative_vecs)
        # Broadcast-and-sum: a batched matmul of millions of tiny matrices is slow.
        c_minus_v0 = (E_hat_inv * (0.5 * edge_lengths)[:, None, :]).sum(dim=-1)
        return torch.where(
            is_degenerate[:, None], vertices.mean(dim=1), v0 + c_minus_v0
        )

    ### Under-determined system (manifold embedded in higher-dim ambient space)
    # The minimum-norm solution lies in the cell's affine hull. small_lstsq
    # factorizes A itself; the closed-form Gram route, inv(E E^T), would square
    # the condition number of thin cells.
    A = 2 * relative_vecs
    rhs = (relative_vecs**2).sum(dim=-1).unsqueeze(-1)  # (n_cells, n_manifold_dims, 1)
    return v0 + small_lstsq(A, rhs).squeeze(-1)


def _rotated_pair_index(
    n_vertices: int, device: torch.device
) -> Int[torch.Tensor, "n_vertices n_pairs"]:
    """Where each local edge goes when a cell's vertices are rotated.

    Local edges are the vertex pairs ``(a, b)``, ``a < b``, in
    ``itertools.combinations`` order. Row ``base`` holds, for each of them, the
    position of the same edge among the edges of the vertices rotated by
    :func:`~physicsnemo.utils._small_linalg.best_base_first`, in which vertex
    ``a`` is at ``(a - base) % n_vertices``. Built on the device, so that no
    host-to-device copy synchronizes CUDA.
    """
    a, b = torch.triu_indices(n_vertices, n_vertices, offset=1, device=device)
    shift = torch.arange(n_vertices, device=device)[:, None]
    a, b = (a - shift) % n_vertices, (b - shift) % n_vertices
    low, high = torch.minimum(a, b), torch.maximum(a, b)
    # Position of the pair (low, high) in itertools.combinations order
    return low * (2 * n_vertices - low - 1) // 2 + (high - low - 1)


def compute_cotan_weights_fem(
    mesh: "Mesh",
) -> tuple[Float[torch.Tensor, " n_edges"], Int[torch.Tensor, "n_edges 2"]]:
    r"""Compute cotangent weights for all edges using the FEM stiffness matrix.

    This is the dimension-general approach that works for simplicial meshes of
    any manifold dimension (1D edges, 2D triangles, 3D tetrahedra, etc.). It
    derives the cotangent weights from the Finite Element Method (FEM) stiffness
    matrix with piecewise-linear basis functions.

    For an :math:`n`-simplex with vertices :math:`v_0, \ldots, v_n` and
    barycentric coordinate functions :math:`\lambda_i`, the stiffness matrix
    entry for edge :math:`(i, j)` is

    .. math::

        K_{ij} = |\sigma| \, \bigl(\nabla \lambda_i \cdot \nabla \lambda_j\bigr).

    The cotangent weight :math:`w_{ij} = -K_{ij}` is accumulated over all cells
    sharing the edge. This is mathematically equivalent to the classical
    cotangent formula in 2D, :math:`w_{ij} = \tfrac{1}{2}(\cot \alpha + \cot \beta)`.

    The gradient dot products are computed efficiently via the Gram matrix:

    .. math::

        E &= [v_1 - v_0, \ldots, v_n - v_0]   \quad (n \times d \text{ edge matrix}) \\
        G &= E E^\top                         \quad (n \times n \text{ Gram matrix}) \\
        \nabla \lambda_k \cdot \nabla \lambda_l &= (G^{-1})_{k-1,\, l-1}
            \quad \text{for } k, l \ge 1.

    For pairs involving vertex 0, the constraint
    :math:`\sum_i \nabla \lambda_i = 0` is used. Vertex 0 here is the base
    vertex whose edges are closest to orthogonal (see ``best_base_first``), so
    the weights do not depend on the order of the vertices. When the cells are
    full-dimensional (:math:`n = d`), :math:`E` is square and
    :math:`G^{-1} = E^{-\top} E^{-1}` comes from the inverse of :math:`E` itself:
    forming :math:`G` would square the condition number of thin cells.

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh of any manifold dimension.

    Returns
    -------
    tuple[Float[torch.Tensor, " n_edges"], Int[torch.Tensor, "n_edges 2"]]
        Tuple of ``(cotan_weights, unique_edges)``:

        - ``cotan_weights``: cotangent weight for each unique edge,
          shape ``(n_edges,)``.
        - ``unique_edges``: sorted edge vertex indices,
          shape ``(n_edges, 2)``.

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
    >>> mesh = two_triangles_2d.load()
    >>> weights, edges = compute_cotan_weights_fem(mesh)
    >>> # weights[i] is the cotangent weight for edges[i]
    """
    from physicsnemo.mesh.utilities._topology import extract_unique_edges

    device = mesh.points.device
    dtype = mesh.points.dtype
    # The degeneracy cutoff below needs at least float32 precision
    compute_dtype = torch.promote_types(dtype, torch.float32)
    n_cells = mesh.n_cells
    n_manifold_dims = mesh.n_manifold_dims

    ### Extract unique edges and the inverse mapping from candidate edges
    unique_edges, inverse_indices = extract_unique_edges(mesh)
    n_unique_edges = len(unique_edges)

    ### Handle empty mesh
    if n_cells == 0:
        return (
            torch.zeros(n_unique_edges, dtype=dtype, device=device),
            unique_edges,
        )

    ### Compute edge vectors from the best base vertex of each cell
    # The weights do not depend on the base vertex, but the closed forms below
    # do: from the tip of a needle, a well-shaped cell would look degenerate.
    # cell_vertices: (n_cells, n_verts_per_cell, n_spatial_dims), rotated so
    # that vertex i is vertex (i + base) % n_verts_per_cell of mesh.cells
    cell_vertices, base = best_base_first(
        gather_rows(mesh.points, mesh.cells).to(compute_dtype)
    )
    # E: (n_cells, n_manifold_dims, n_spatial_dims) - rows are e_k = v_k - v_0
    E = cell_vertices[:, 1:, :] - cell_vertices[:, :1, :]

    ### Invert the Gram matrix G = E E^T, with the edges scaled to unit length
    # With E = D Ê (D the diagonal of edge lengths), inv(G) = D^-1 inv(Ê Êᵀ) D^-1,
    # and the normalized inverse stays in range for tiny or huge cells.
    # Degenerate cells (collinear or coplanar vertices) have no inverse. Their
    # contribution is meant to vanish anyway, since their volume is ~0, so they
    # invert the identity instead. The cutoff, |det(Ê)| < 1e-6, is the one main
    # applied to G in units of the mean edge length, |det(G / g)| < 1e-12; unit
    # edges stop it from also flagging long, thin but well-shaped cells.
    # Written branchlessly so torch.compile can trace through without graph breaks.
    if n_manifold_dims == mesh.n_spatial_dims:
        ### Full-dimensional cells: invert E itself, inv(G) = inv(E)^T inv(E)
        # Forming G squares the condition number of E, which in float32 costs thin
        # cells most of their accuracy.
        E_hat_inv, edge_lengths, _ = normalized_inverse(E, tolerance=1e-6)
        # Broadcast-and-sum: a batched matmul of millions of tiny matrices is slow.
        G_hat_inv = (E_hat_inv.unsqueeze(-1) * E_hat_inv.unsqueeze(-2)).sum(dim=-3)
    else:
        G_hat_inv, edge_lengths, _ = normalized_gram_inverse(E, tolerance=1e-6)

    # G_inv: (n_cells, n_manifold_dims, n_manifold_dims)
    G_inv = G_hat_inv / (edge_lengths[:, :, None] * edge_lengths[:, None, :])

    ### Extract gradient dot products for each local edge pair (i, j), i < j
    # These are the upper-triangle entries of C = H @ G_inv @ H^T, where
    # H = [[-1,...,-1]; I_n] encodes grad lambda_0 = -sum(grad lambda_k for k>=1):
    #   C[0, j] = -sum_k G_inv[k, j-1]  and  C[i, j] = G_inv[i-1, j-1]  (i >= 1).
    # Reading them off directly avoids batched matmuls of millions of tiny
    # matrices, which are slow on GPUs.
    # Upper-triangle order is itertools.combinations order, which is the local edge
    # order that extract_candidate_facets produces. Building the indices on-device
    # avoids the host-to-device copy (and its synchronization) a Python list needs.
    pair_i, pair_j = torch.triu_indices(
        n_manifold_dims, n_manifold_dims, offset=1, device=device
    )

    # grad_dots: (n_cells, n_pairs) - one value per cell per local edge
    grad_dots = torch.cat([-G_inv.sum(dim=-2), G_inv[:, pair_i, pair_j]], dim=1)
    # Local edges in the vertex order of mesh.cells, not the rotated one. take
    # on the flat table: on CUDA, indexing its rows by millions of cells is 15x
    # slower.
    n_pairs = grad_dots.shape[1]
    rotated_pairs = (
        _rotated_pair_index(n_manifold_dims + 1, device)
        .reshape(-1)
        .take(base[:, None] * n_pairs + torch.arange(n_pairs, device=device))
    )
    grad_dots = grad_dots.gather(1, rotated_pairs)

    ### Compute cotangent weight contributions per cell per edge
    # w = -|sigma| * (grad lambda_i . grad lambda_j)
    cell_volumes = mesh.cell_areas.to(compute_dtype)  # (n_cells,)
    weights_per_cell = -cell_volumes[:, None] * grad_dots  # (n_cells, n_pairs)

    ### Accumulate contributions to unique edges via scatter_add
    cotan_weights = torch.zeros(n_unique_edges, dtype=compute_dtype, device=device)
    # inverse_indices maps each candidate edge to its unique edge index.
    # For 1D: shape (n_cells,); for nD>1: shape (n_cells * n_pairs,)
    # weights_per_cell.reshape(-1) aligns with inverse_indices in both cases.
    cotan_weights.scatter_add_(0, inverse_indices, weights_per_cell.reshape(-1))

    return cotan_weights.to(dtype), unique_edges


def compute_dual_volumes_1(
    mesh: "Mesh",
) -> tuple[Float[torch.Tensor, " n_edges"], Int[torch.Tensor, "n_edges 2"]]:
    r"""Compute dual 1-cell volumes (dual to edges).

    The dual 1-cell of an edge is the portion of the circumcentric dual mesh
    associated with that edge. For a 2D triangle mesh, it consists of segments
    from the edge midpoint to the circumcenters of adjacent triangles,

    .. math::

        |{\star}e| = |e| \, w_{ij},

    where :math:`w_{ij}` is the FEM cotangent weight for the edge. This
    relationship holds for any manifold dimension; the FEM stiffness matrix
    approach (see :func:`compute_cotan_weights_fem`) derives these weights from
    the gradient dot products of barycentric basis functions.

    Parameters
    ----------
    mesh : Mesh
        Input simplicial mesh of any manifold dimension.

    Returns
    -------
    tuple[Float[torch.Tensor, " n_edges"], Int[torch.Tensor, "n_edges 2"]]
        Tuple of ``(dual_volumes, edges)``:

        - ``dual_volumes``: dual 1-cell volume for each edge,
          shape ``(n_edges,)``. May be negative for edges in non-Delaunay
          configurations (obtuse angles exceeding :math:`\pi/2` at both
          adjacent cells).
        - ``edges``: canonically sorted edge connectivity,
          shape ``(n_edges, 2)``, with ``edges[:, 0] < edges[:, 1]``.

    Notes
    -----
    Negative dual volumes are geometrically meaningful: they indicate that the
    circumcentric dual edge crosses the primal edge. Clamping them to zero (as
    some implementations do) silently degrades accuracy on non-Delaunay meshes.
    """
    ### Derive cotangent weights from the FEM stiffness matrix (works for any dimension)
    cotan_weights, edges = compute_cotan_weights_fem(mesh)

    ### |star e| = w_ij * |e|
    edge_vectors = mesh.points[edges[:, 1]] - mesh.points[edges[:, 0]]
    edge_lengths = torch.norm(edge_vectors, dim=-1)
    dual_volumes_1 = cotan_weights * edge_lengths

    return dual_volumes_1, edges


def get_or_compute_dual_volumes_0(mesh: "Mesh") -> Float[torch.Tensor, " n_points"]:
    r"""Get cached dual 0-cell volumes or compute if not present.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.

    Returns
    -------
    Float[torch.Tensor, " n_points"]
        Dual volumes for vertices, shape ``(n_points,)``.
    """
    return mesh._cached(
        ("point", "dual_volumes_0"), lambda: compute_dual_volumes_0(mesh)
    )


def get_or_compute_circumcenters(
    mesh: "Mesh",
) -> Float[torch.Tensor, "n_cells n_spatial_dims"]:
    r"""Get cached circumcenters or compute if not present.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.

    Returns
    -------
    Float[torch.Tensor, "n_cells n_spatial_dims"]
        Circumcenters for all cells, shape ``(n_cells, n_spatial_dims)``.
    """
    return mesh._cached(
        ("cell", "circumcenters"),
        lambda: compute_circumcenters(gather_rows(mesh.points, mesh.cells)),
    )
