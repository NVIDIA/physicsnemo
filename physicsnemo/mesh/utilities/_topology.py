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

"""General mesh topology utilities."""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Int

from physicsnemo.mesh.boundaries._facet_extraction import extract_candidate_facets
from physicsnemo.mesh.utilities._duplicate_detection import (
    vectorized_connected_components,
)
from physicsnemo.utils._index_tuple_ops import unique_index_tuples

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


_TRIANGLE_HINGES_CACHE_KEY = "triangle_hinges"
_VALID_SIMPLEX_TOPOLOGY_CACHE_KEY = "valid_simplex_topology"
_DEFORMATION_ENERGY_CELLS_CACHE_KEY = "deformation_energy_cells_int64"
_CLOSED_ORIENTED_TRIANGLE_SURFACE_CACHE_KEY = "closed_oriented_triangle_surface"


def _validate_indexed_simplex_topology(mesh: "Mesh") -> None:
    """Validate cell indices required by deformation-energy kernels.

    ``Mesh`` validates the rank and integer-like dtype of connectivity, but it
    intentionally does not scan connectivity for bounds or repeated vertices
    during construction. Energy evaluation indexes every cell and assumes a
    nondegenerate combinatorial simplex, so the stricter scan is performed once
    and cached with the topology.
    """

    manifold_dims = mesh.n_manifold_dims
    if not 1 <= manifold_dims <= 3:
        raise ValueError(
            "deformation energies require simplices with manifold dimension "
            f"between 1 and 3, got {manifold_dims}"
        )

    cached = mesh._cache.get(("topology", _VALID_SIMPLEX_TOPOLOGY_CACHE_KEY), None)
    if cached is not None:
        return

    source_cells = mesh.cells
    if (
        source_cells.dtype == torch.bool
        or source_cells.dtype.is_floating_point
        or source_cells.dtype.is_complex
    ):
        raise TypeError(
            "mesh cells must have an integer dtype other than torch.bool for "
            f"deformation energies, got {source_cells.dtype}"
        )
    cells = (
        source_cells
        if source_cells.dtype == torch.int64
        else source_cells.to(dtype=torch.int64)
    )
    if cells.numel() != 0:
        minimum_index = int(cells.amin().item())
        maximum_index = int(cells.amax().item())
        if minimum_index < 0 or maximum_index >= mesh.n_points:
            raise ValueError(
                "mesh cells contain point indices outside the valid range "
                f"[0, {mesh.n_points}); got minimum {minimum_index} and "
                f"maximum {maximum_index}"
            )

        sorted_cells = torch.sort(cells, dim=1).values
        if bool((sorted_cells[:, 1:] == sorted_cells[:, :-1]).any().item()):
            raise ValueError(
                "mesh cells must contain distinct point indices; found a cell "
                "with a repeated vertex"
            )

    if cells is not source_cells:
        mesh._cache["topology", _DEFORMATION_ENERGY_CELLS_CACHE_KEY] = cells
    mesh._cache["topology", _VALID_SIMPLEX_TOPOLOGY_CACHE_KEY] = torch.ones(
        (), dtype=torch.bool, device=mesh.points.device
    )


def _deformation_energy_cells(mesh: "Mesh") -> Int[torch.Tensor, "n_cells n_vertices"]:
    """Return connectivity in the int64 format required by tensor kernels.

    ``Mesh`` accepts integer connectivity dtypes other than int64. Conversion
    is cached with topology so a non-int64 mesh does not allocate on every
    energy evaluation.
    """

    _validate_indexed_simplex_topology(mesh)
    if mesh.cells.dtype == torch.int64:
        return mesh.cells

    cached = mesh._cache.get(("topology", _DEFORMATION_ENERGY_CELLS_CACHE_KEY), None)
    if cached is None:
        cached = mesh.cells.to(dtype=torch.int64)
        mesh._cache["topology", _DEFORMATION_ENERGY_CELLS_CACHE_KEY] = cached
    return cached


def extract_triangle_hinges(
    mesh: "Mesh",
) -> Int[torch.Tensor, "n_hinges 4"]:
    r"""Extract and cache the interior hinges of a triangle mesh.

    Each returned hinge is ``(u, v, k, l)``. The shared edge is canonicalized
    so that ``u < v``. The tuple represents the artificial oriented faces
    ``(u, v, k)`` and ``(v, u, l)`` used by the signed-dihedral energy. This
    convention is independent of the input face winding. Boundary edges are
    omitted.

    Parameters
    ----------
    mesh : Mesh
        Triangle mesh. Spatial coordinates are not inspected.

    Returns
    -------
    hinges : torch.Tensor
        Interior hinge vertex indices with shape ``(n_hinges, 4)``.

    Raises
    ------
    ValueError
        If ``mesh`` is not a triangle mesh, contains invalid cell indices, or
        contains an edge incident to more than two cells.

    Notes
    -----
    The result depends only on connectivity and is stored in the mesh topology
    cache. Geometric transformations therefore reuse it.
    """

    cached_hinges = mesh._cache.get(("topology", _TRIANGLE_HINGES_CACHE_KEY), None)
    if cached_hinges is not None:
        return cached_hinges

    if mesh.n_manifold_dims != 2:
        raise ValueError(
            "triangle hinges require a triangle mesh with manifold dimension "
            f"2, got {mesh.n_manifold_dims}"
        )
    cells = _deformation_energy_cells(mesh)
    if cells.shape[0] == 0:
        hinges = cells.new_empty((0, 4))
    else:
        canonical_faces = torch.sort(cells, dim=1).values
        if torch.unique(canonical_faces, dim=0).shape[0] != cells.shape[0]:
            raise ValueError(
                "triangle hinges require unique triangles; found a duplicate face"
            )

        # Candidate order is cell-major. A stable sort makes the opposite
        # vertices deterministic within every canonical-edge group.
        starts = cells[:, (0, 1, 2)].reshape(-1)
        ends = cells[:, (1, 2, 0)].reshape(-1)
        opposite = cells[:, (2, 0, 1)].reshape(-1)
        candidate_edges = torch.sort(torch.stack((starts, ends), dim=1), dim=1).values

        edge_keys = candidate_edges[:, 0] * mesh.n_points + candidate_edges[:, 1]
        order = torch.argsort(edge_keys, stable=True)
        sorted_edges = candidate_edges[order]
        sorted_opposite = opposite[order]
        unique_edges, counts = torch.unique_consecutive(
            sorted_edges, dim=0, return_counts=True
        )

        if bool((counts > 2).any().item()):
            maximum_adjacency = int(counts.max().item())
            raise ValueError(
                "triangle hinges require a manifold surface: every edge may "
                "belong to at most two triangles, but an edge belongs to "
                f"{maximum_adjacency}"
            )

        group_starts = torch.cat(
            (counts.new_zeros(1), counts.cumsum(dim=0)[:-1]), dim=0
        )
        interior = counts == 2
        first = group_starts[interior]
        interior_edges = unique_edges[interior]
        hinges = torch.cat(
            (
                interior_edges,
                sorted_opposite[first, None],
                sorted_opposite[first + 1, None],
            ),
            dim=1,
        )

    mesh._cache["topology", _TRIANGLE_HINGES_CACHE_KEY] = hinges
    return hinges


def _validate_closed_oriented_triangle_surface(mesh: "Mesh") -> None:
    """Validate and cache the topology required by enclosed-volume energy.

    A valid surface is nonempty, triangular, embedded in three dimensions,
    edge-connected, contains no duplicate faces, and has exactly two
    oppositely directed halfedges for every undirected edge. Vertex
    neighborhoods and self-intersection are not tested.
    """

    if mesh.n_manifold_dims != 2 or mesh.n_spatial_dims != 3:
        raise ValueError(
            "closed-surface volume energy requires a triangle surface embedded "
            "in three dimensions"
        )
    cached = mesh._cache.get(
        ("topology", _CLOSED_ORIENTED_TRIANGLE_SURFACE_CACHE_KEY), None
    )
    if cached is not None:
        return
    if mesh.n_cells == 0:
        raise ValueError("closed-surface volume energy requires at least one triangle")

    cells = _deformation_energy_cells(mesh)
    canonical_faces = torch.sort(cells, dim=1).values
    _, face_counts = torch.unique(canonical_faces, dim=0, return_counts=True)
    if bool((face_counts > 1).any().item()):
        raise ValueError(
            "closed-surface volume energy requires unique triangles; found a "
            "duplicate face"
        )

    starts = cells[:, (0, 1, 2)].reshape(-1)
    ends = cells[:, (1, 2, 0)].reshape(-1)
    halfedges = torch.stack((starts, ends), dim=1)
    undirected_edges = torch.sort(halfedges, dim=1).values
    _, inverse, edge_counts = torch.unique(
        undirected_edges,
        dim=0,
        return_inverse=True,
        return_counts=True,
    )
    if bool((edge_counts != 2).any().item()):
        raise ValueError(
            "closed-surface volume energy requires an edge-closed surface: "
            "every undirected edge must belong to exactly two triangles"
        )

    direction = torch.where(
        starts == undirected_edges[:, 0],
        torch.ones_like(starts),
        -torch.ones_like(starts),
    )
    direction_sum = torch.zeros(
        edge_counts.shape, dtype=direction.dtype, device=direction.device
    )
    direction_sum.scatter_add_(0, inverse, direction)
    if bool((direction_sum != 0).any().item()):
        raise ValueError(
            "closed-surface volume energy requires consistently oriented "
            "triangles: the two halfedges on every edge must have opposite "
            "directions"
        )

    halfedge_faces = torch.arange(
        mesh.n_cells, dtype=torch.int64, device=cells.device
    ).repeat_interleave(3)
    edge_order = torch.argsort(inverse, stable=True)
    adjacent_faces = halfedge_faces[edge_order].reshape(-1, 2)
    component_labels = vectorized_connected_components(
        adjacent_faces,
        mesh.n_cells,
    )
    if torch.unique(component_labels).numel() != 1:
        raise ValueError(
            "closed-surface volume energy requires one edge-connected triangle "
            "surface; evaluate disconnected components separately"
        )

    mesh._cache["topology", _CLOSED_ORIENTED_TRIANGLE_SURFACE_CACHE_KEY] = torch.ones(
        (), dtype=torch.bool, device=mesh.points.device
    )


def extract_unique_edges(
    mesh: "Mesh",
) -> tuple[Int[torch.Tensor, "n_edges 2"], Int[torch.Tensor, " n_candidates"]]:
    """Extract all unique edges from the mesh.

    For 1D meshes (cells are edges), the cells are deduplicated directly.
    For higher-dimensional meshes, edges are extracted via
    :func:`extract_candidate_facets` at the appropriate codimension.

    Parameters
    ----------
    mesh : Mesh
        Input mesh to extract edges from.

    Returns
    -------
    unique_edges : torch.Tensor
        Unique edge vertex indices, shape (n_edges, 2), canonically sorted
        so that ``unique_edges[:, 0] < unique_edges[:, 1]``.
    inverse_indices : torch.Tensor
        Mapping from candidate edges to unique edge indices.
        For 1D meshes, shape is (n_cells,).
        For n-manifolds with n > 1, shape is
        (n_cells * n_edges_per_cell,), which can be reshaped to
        (n_cells, n_edges_per_cell).

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
    >>> triangle_mesh = two_triangles_2d.load()
    >>> edges, inverse = extract_unique_edges(triangle_mesh)
    >>> edges.shape[1]
    2
    """
    if mesh.n_manifold_dims == 1:
        ### 1D meshes: cells ARE edges - sort and deduplicate directly
        sorted_cells = torch.sort(mesh.cells, dim=1)[0]
        unique_edges, inverse_indices = unique_index_tuples(
            sorted_cells,
            index_bound=mesh.n_points,
            return_inverse=True,
        )
        return unique_edges, inverse_indices

    ### General case: extract edges as (n-1)-codimension facets of each cell
    candidate_edges, _parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=mesh.n_manifold_dims - 1,
    )
    unique_edges, inverse_indices = unique_index_tuples(
        candidate_edges,
        index_bound=mesh.n_points,
        return_inverse=True,
    )
    return unique_edges, inverse_indices
