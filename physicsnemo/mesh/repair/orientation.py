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

"""Fix cell orientation for consistent windings.

Ensures all cells in a mesh have consistent orientation so normals point
in the same general direction.
"""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Bool, Float, Int

from physicsnemo.mesh.boundaries._facet_extraction import extract_candidate_facets
from physicsnemo.mesh.utilities._duplicate_detection import (
    vectorized_connected_components,
)
from physicsnemo.utils._index_tuple_ops import unique_index_tuples

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _induced_facet_orientations(
    cells: Int[torch.Tensor, "n_cells n_vertices"],
) -> tuple[Bool[torch.Tensor, "n_cells n_vertices"], Bool[torch.Tensor, " n_cells"]]:
    """Orientation that each cell induces on each of its facets, as a parity.

    A cell's orientation is the parity of the permutation that sorts its
    vertices. Dropping vertex ``i`` leaves a facet whose induced orientation,
    relative to the facet's sorted vertices, is ``(-1)**i`` times the parity of
    sorting the remaining vertices. Removing a vertex of rank ``r`` from a
    sequence changes its inversion count by ``i + r`` modulo 2, so the induced
    parity is the cell's parity plus the rank of the dropped vertex.

    Parameters
    ----------
    cells : torch.Tensor
        Cell connectivity, shape ``(n_cells, n_vertices)``.

    Returns
    -------
    induced : torch.Tensor
        ``induced[c, j]`` is the induced parity (``True`` for odd) of facet
        ``j`` of cell ``c``, with facets in :func:`extract_candidate_facets`
        order, where facet ``j`` drops vertex ``n_vertices - 1 - j``.
    is_degenerate : torch.Tensor
        ``True`` for cells with a repeated vertex, whose orientation is undefined.
    """
    n_vertices = cells.shape[1]
    less = cells[:, :, None] < cells[:, None, :]  # less[c, i, j]: v_i < v_j
    ranks = less.sum(dim=1)  # number of vertices below each vertex
    later = torch.ones(
        n_vertices, n_vertices, dtype=torch.bool, device=cells.device
    ).triu(diagonal=1)
    inversions = (less.transpose(1, 2) & later).sum(dim=(1, 2))  # v_i > v_j, i < j
    is_degenerate = (cells[:, :, None] == cells[:, None, :]).sum(
        dim=(1, 2)
    ) > n_vertices
    dropped_vertex = torch.arange(n_vertices - 1, -1, -1, device=cells.device)
    induced = (inversions[:, None] + ranks[:, dropped_vertex]) % 2 == 1
    return induced, is_degenerate


def _facet_pairs(
    cells: Int[torch.Tensor, "n_cells n_vertices"], n_points: int
) -> tuple[
    Int[torch.Tensor, "n_candidates n_facet_vertices"],
    Int[torch.Tensor, " n_links"],
    Int[torch.Tensor, " n_links"],
    Bool[torch.Tensor, " n_links"],
    Bool[torch.Tensor, " n_cells"],
]:
    """The two cells of each facet that exactly two cells share.

    Cells with a repeated vertex have no orientation: their facets pair with
    none, and do not count toward the cells that share a facet.

    Parameters
    ----------
    cells : torch.Tensor
        Cell connectivity, shape ``(n_cells, n_vertices)``, ``n_vertices >= 2``.
    n_points : int
        Number of points.

    Returns
    -------
    candidate_facets : torch.Tensor
        The facets of :func:`extract_candidate_facets`, each with its vertices
        sorted: facet ``j`` of cell ``c`` is at position ``c * n_vertices + j``.
    first, second : torch.Tensor
        Positions in ``candidate_facets`` of the two candidates of each such
        facet. The other positions pair with themselves, which links nothing.
    opposite : torch.Tensor
        Whether the two cells induce the same orientation on their facet, so
        that they agree only if exactly one of them flips.
    is_degenerate : torch.Tensor
        ``True`` for the cells with a repeated vertex.
    """
    ### Orientation that each cell induces on each of its facets
    induced, is_degenerate = _induced_facet_orientations(cells)
    candidate_facets, parent_cells = extract_candidate_facets(cells)

    ### Pair the two candidates of every facet shared by exactly two cells
    # Facets of degenerate cells get the out-of-range vertex ``n_points``, so
    # they never count toward the cells that share a real facet.
    candidate_facets.masked_fill_(is_degenerate[parent_cells, None], n_points)
    _, facet_ids, facet_counts = unique_index_tuples(
        candidate_facets,
        index_bound=n_points + 1,
        return_inverse=True,
        return_counts=True,
    )
    order = torch.argsort(facet_ids)
    first, second = order[:-1], order[1:]
    is_pair = (
        (facet_ids[first] == facet_ids[second])
        & (facet_counts[facet_ids[first]] == 2)
        & ~is_degenerate[parent_cells[first]]
        & ~is_degenerate[parent_cells[second]]
    )
    induced = induced.reshape(-1)
    opposite = is_pair & (induced[first] == induced[second])
    second = torch.where(is_pair, second, first)
    return candidate_facets, first, second, opposite, is_degenerate


def _flip_labels(
    a: Int[torch.Tensor, " n_links"],
    b: Int[torch.Tensor, " n_links"],
    opposite: Bool[torch.Tensor, " n_links"],
    n_elements: int,
) -> tuple[Int[torch.Tensor, " n_elements"], Int[torch.Tensor, " n_elements"]]:
    """Union-find over the kept (``e``) and flipped (``e + n_elements``) copies.

    Each pair ``(a, b)`` links the copies that agree: the kept copies, and the
    flipped ones, unless ``opposite``, when each kept copy agrees with the
    other's flipped copy. Returns the lowest index in the component of each
    element's kept copy, and of its flipped copy.
    """
    agreeing = torch.where(opposite, b + n_elements, b)
    agreeing_flipped = torch.where(opposite, b, b + n_elements)
    links = torch.stack(
        [torch.cat([a, a + n_elements]), torch.cat([agreeing, agreeing_flipped])],
        dim=1,
    )
    labels = vectorized_connected_components(links, 2 * n_elements)
    return labels[:n_elements], labels[n_elements:]


def _flips_around_points(
    cells: Int[torch.Tensor, "n_cells n_vertices"], n_points: int
) -> Bool[torch.Tensor, "n_cells n_vertices"]:
    """Which cells to flip around each point, to orient its cells consistently.

    ``flips[c, i]`` is ``True`` where cell ``c`` must flip, around its vertex
    ``i``, to agree with the lowest-index cell around that point that it
    reaches through facets containing the point and shared by exactly two
    cells. This is :func:`fix_orientation` around each point instead of
    across the mesh: decided from connectivity alone, so at any angle between
    the cells, and possible on the Möbius strip, around each point of which
    the cells are orientable although the strip is not. Consistently wound
    cells get no flips, nor do cells that cannot be oriented together.

    Parameters
    ----------
    cells : torch.Tensor
        Cell connectivity, shape ``(n_cells, n_vertices)``.
    n_points : int
        Number of points.

    Returns
    -------
    torch.Tensor
        The flips, shape ``(n_cells, n_vertices)``, or ``(0, n_vertices)``
        where no cell flips.
    """
    n_cells, n_vertices = cells.shape
    no_flips = cells.new_zeros((0, n_vertices), dtype=torch.bool)
    if n_vertices < 2:  # points have no facets
        return no_flips

    ### Consistently wound: no two cells orient a facet alike, as one sort shows
    # The highest vertex v of each facet becomes 2 * v + the orientation induced
    induced, _ = _induced_facet_orientations(cells)
    oriented = extract_candidate_facets(cells)[0].long()
    oriented[:, -1] = 2 * oriented[:, -1] + induced.flatten()
    if len(unique_index_tuples(oriented, index_bound=2 * n_points)) == len(oriented):
        return no_flips
    del induced, oriented

    facets, first, second, opposite, _ = _facet_pairs(cells, n_points)
    if not opposite.any():  # alike only on facets of 3+ cells or degenerate cells
        return no_flips

    ### Pair the corners of the two cells at each point of their facet, as
    ### fix_orientation pairs the cells; corner i of cell c is c * n_vertices + i
    is_pair = first != second
    pair_cells = torch.stack([first, second], dim=1)[is_pair] // n_vertices
    points = facets[first[is_pair]]  # sorted, so the same for both cells of a pair
    del facets, first, second
    is_corner = cells[pair_cells][..., None, :] == points[:, None, :, None]
    corners = pair_cells[..., None] * n_vertices + is_corner.byte().argmax(-1)
    corners = corners.transpose(1, 2).reshape(-1, 2)  # (n_pairs * (n_vertices - 1), 2)
    del pair_cells, points, is_corner
    kept_label, flipped_label = _flip_labels(
        corners[:, 0],
        corners[:, 1],
        opposite[is_pair].repeat_interleave(n_vertices - 1),
        n_cells * n_vertices,
    )
    # As in fix_orientation: a corner flips when its flipped copy joined the
    # lowest corner of its component
    return (kept_label > flipped_label).reshape(n_cells, n_vertices)


def _signed_volumes(
    points: Float[torch.Tensor, "n_points n_dims"],
    cells: Int[torch.Tensor, "n_cells n_vertices"],
) -> Float[torch.Tensor, " n_cells"]:
    """Signed volumes of full-dimensional simplices, times ``n_dims!``.

    Up to three dimensions, closed forms replace batched LU factorizations,
    which are slow for many tiny matrices.
    """
    points = points.to(torch.promote_types(points.dtype, torch.float32))
    vertices = points[cells]
    edges = vertices[:, 1:] - vertices[:, :1]  # (n_cells, n_dims, n_dims)
    n_dims = edges.shape[-1]
    if n_dims == 1:
        return edges[:, 0, 0]
    if n_dims == 2:
        return edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0]
    if n_dims == 3:
        return (edges[:, 0] * torch.linalg.cross(edges[:, 1], edges[:, 2])).sum(-1)
    return torch.linalg.det(edges)


def fix_orientation(
    mesh: "Mesh",
) -> tuple["Mesh", dict[str, int]]:
    """Orient all cells consistently.

    Two cells sharing a facet are consistently oriented when they induce
    opposite orientations on it, so that the facet cancels in the boundary of
    the oriented manifold: two triangles traverse their shared edge in
    opposite directions, two tetrahedra wind their shared face oppositely, and
    of two segments meeting at a vertex, one ends where the other starts. This
    is decided from connectivity alone, so it holds at any dihedral angle and
    in any manifold and spatial dimension.

    Every connected component is oriented to agree with its lowest-index
    cell. If the cells fill their space (``n_manifold_dims == n_spatial_dims``,
    such as triangles in 2D or tetrahedra in 3D), a component whose total
    signed volume is then negative is reversed, so that volumes come out
    positive. A cell is flipped by swapping its last two vertices. The
    orientations are solved together by a union-find over two copies of every
    cell (as given and flipped), in which each shared facet links the copies
    that agree on it; a component is orientable exactly when the two copies of
    its cells stay apart. A non-orientable component, such as a Möbius strip,
    is left as given and counted.

    Only facets shared by exactly two cells relate their cells, so cells that
    meet only at non-manifold facets are oriented independently. Cells with a
    repeated vertex have no orientation: they are left as given and do not
    count toward the cells that share a facet.

    Parameters
    ----------
    mesh : Mesh
        Input mesh with at least two vertices per cell (``n_manifold_dims >= 1``).

    Returns
    -------
    tuple[Mesh, dict[str, int]]
        Tuple of (oriented_mesh, stats_dict) where stats_dict contains:

        - "n_cells_flipped": Number of cells that were flipped
        - "n_faces_flipped": Same as "n_cells_flipped", kept for compatibility
        - "n_components": Number of facet-connected components found
        - "largest_component_size": Size of largest component
        - "n_non_orientable_components": Number of components left as given
          because they cannot be oriented consistently

    Raises
    ------
    ValueError
        If the cells are points (``n_manifold_dims == 0``).

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
    >>> mesh = sphere_icosahedral.load(subdivisions=2)
    >>> mesh_oriented, stats = fix_orientation(mesh)
    >>> assert "n_cells_flipped" in stats
    """
    if mesh.n_manifold_dims < 1:
        raise ValueError(
            f"Orientation is only defined for cells with at least two vertices. "
            f"Got {mesh.n_manifold_dims=}."
        )

    if mesh.n_cells == 0:
        return mesh, {
            "n_cells_flipped": 0,
            "n_faces_flipped": 0,
            "n_components": 0,
            "largest_component_size": 0,
            "n_non_orientable_components": 0,
        }

    device = mesh.cells.device
    n_cells = mesh.n_cells
    cells = mesh.cells

    ### Steps 1-2: Pair the two candidates of every facet shared by exactly two cells
    _, first, second, opposite, is_degenerate = _facet_pairs(cells, mesh.n_points)

    ### Step 3: Union-find over the kept (c) and flipped (c + n_cells) copies
    # Facet j of cell c is candidate c * n_vertices + j
    n_vertices = cells.shape[1]
    kept_label, flipped_label = _flip_labels(
        first // n_vertices, second // n_vertices, opposite, n_cells
    )

    # Each copy's label is the lowest index in its component. The lowest cell of
    # a component keeps its orientation, so a cell must flip exactly when its
    # flipped copy joined that lowest cell; both copies share a label only in a
    # non-orientable component.
    should_flip = kept_label > flipped_label
    component = torch.minimum(kept_label, flipped_label)
    is_orientable = kept_label != flipped_label
    is_root = component == torch.arange(n_cells, device=device)

    ### Step 4: Reverse full-dimensional components of negative total volume
    if mesh.codimension == 0:
        volumes = _signed_volumes(mesh.points, cells)
        total_volumes = torch.zeros_like(volumes).index_add_(
            0, component, torch.where(should_flip, -volumes, volumes)
        )
        # Degenerate cells and non-orientable components stay as given
        is_reversed = (total_volumes[component] < 0) & is_orientable & ~is_degenerate
        should_flip = should_flip ^ is_reversed

    ### Component statistics, read with the flip count in one transfer
    component_sizes = torch.zeros(n_cells, dtype=torch.long, device=device)
    component_sizes.index_add_(0, component, torch.ones_like(component))
    n_flipped, n_components, largest_component_size, n_non_orientable = torch.stack(
        [
            should_flip.sum(),
            is_root.sum(),
            component_sizes.max(),
            (is_root & ~is_orientable).sum(),
        ]
    ).tolist()

    ### Step 5: Apply flips by swapping the last two vertices of each flipped cell
    if n_flipped > 0:
        swapped = torch.cat([cells[:, :-2], cells[:, -1:], cells[:, -2:-1]], dim=1)
        new_cells = torch.where(should_flip[:, None], swapped, cells)

        # Repair results historically own independent data tensors. Preserve
        # that behavior while routing connectivity cache handling through the
        # public API.
        oriented_mesh = mesh.with_cells(new_cells).with_data(
            point_data=mesh.point_data.clone(),
            cell_data=mesh.cell_data.clone(),
            global_data=mesh.global_data.clone(),
        )
    else:
        oriented_mesh = mesh

    stats = {
        "n_cells_flipped": n_flipped,
        "n_faces_flipped": n_flipped,
        "n_components": n_components,
        "largest_component_size": largest_component_size,
        "n_non_orientable_components": n_non_orientable,
    }

    return oriented_mesh, stats
