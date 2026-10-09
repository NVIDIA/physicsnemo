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
from jaxtyping import Bool, Int

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
    cell, which is never flipped. A cell is flipped by swapping its last two
    vertices. The orientations are solved together by a union-find over two
    copies of every cell (as given and flipped), in which each shared facet
    links the copies that agree on it; a component is orientable exactly when
    the two copies of its cells stay apart. A non-orientable component, such
    as a Möbius strip, is left as given and counted.

    Only facets shared by exactly two cells relate their cells, so cells that
    meet only at non-manifold facets are oriented independently. Cells with a
    repeated vertex have no orientation and are left as given.

    Parameters
    ----------
    mesh : Mesh
        Input mesh with at least two vertices per cell (``n_manifold_dims >= 1``).

    Returns
    -------
    tuple[Mesh, dict[str, int]]
        Tuple of (oriented_mesh, stats_dict) where stats_dict contains:

        - "n_faces_flipped": Number of cells that were flipped
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
    >>> assert "n_faces_flipped" in stats
    """
    if mesh.n_manifold_dims < 1:
        raise ValueError(
            f"Orientation is only defined for cells with at least two vertices. "
            f"Got {mesh.n_manifold_dims=}."
        )

    if mesh.n_cells == 0:
        return mesh, {
            "n_faces_flipped": 0,
            "n_components": 0,
            "largest_component_size": 0,
            "n_non_orientable_components": 0,
        }

    device = mesh.cells.device
    n_cells = mesh.n_cells
    cells = mesh.cells

    ### Step 1: Orientation that each cell induces on each of its facets
    induced, is_degenerate = _induced_facet_orientations(cells)
    candidate_facets, parent_cells = extract_candidate_facets(cells)

    ### Step 2: Pair the two candidates of every facet shared by exactly two cells
    _, facet_ids, facet_counts = unique_index_tuples(
        candidate_facets,
        index_bound=mesh.n_points,
        return_inverse=True,
        return_counts=True,
    )
    order = torch.argsort(facet_ids)
    first, second = order[:-1], order[1:]
    cell_a, cell_b = parent_cells[first], parent_cells[second]
    is_pair = (
        (facet_ids[first] == facet_ids[second])
        & (facet_counts[facet_ids[first]] == 2)
        & ~is_degenerate[cell_a]
        & ~is_degenerate[cell_b]
    )

    ### Step 3: Union-find over the kept (c) and flipped (c + n_cells) copies
    # A pair that induces the same orientation on its facet agrees only if
    # exactly one of its cells flips. Positions that are not pairs become
    # self-loops, which link nothing.
    induced = induced.reshape(-1)
    opposite = induced[first] == induced[second]
    agreeing = torch.where(opposite, cell_b + n_cells, cell_b)
    agreeing_flipped = torch.where(opposite, cell_b, cell_b + n_cells)
    links = torch.stack(
        [
            torch.cat([cell_a, cell_a + n_cells]),
            torch.cat(
                [
                    torch.where(is_pair, agreeing, cell_a),
                    torch.where(is_pair, agreeing_flipped, cell_a + n_cells),
                ]
            ),
        ],
        dim=1,
    )
    labels = vectorized_connected_components(links, 2 * n_cells)
    kept_label, flipped_label = labels[:n_cells], labels[n_cells:]

    # Each copy's label is the lowest index in its component. The lowest cell of
    # a component keeps its orientation, so a cell must flip exactly when its
    # flipped copy joined that lowest cell; both copies share a label only in a
    # non-orientable component.
    should_flip = kept_label > flipped_label
    component = torch.minimum(kept_label, flipped_label)
    is_root = component == torch.arange(n_cells, device=device)

    ### Component statistics, read with the flip count in one transfer
    component_sizes = torch.zeros(n_cells, dtype=torch.long, device=device)
    component_sizes.index_add_(0, component, torch.ones_like(component))
    n_flipped, n_components, largest_component_size, n_non_orientable = torch.stack(
        [
            should_flip.sum(),
            is_root.sum(),
            component_sizes.max(),
            (is_root & (kept_label == flipped_label)).sum(),
        ]
    ).tolist()

    ### Step 4: Apply flips by swapping the last two vertices of each flipped cell
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
        "n_faces_flipped": n_flipped,
        "n_components": n_components,
        "largest_component_size": largest_component_size,
        "n_non_orientable_components": n_non_orientable,
    }

    return oriented_mesh, stats
