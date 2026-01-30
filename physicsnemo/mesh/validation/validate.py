# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Mesh validation to detect common errors and degenerate cases.

Provides comprehensive validation of mesh integrity including topology,
geometry, and data consistency checks.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def validate_mesh(
    mesh: "Mesh",
    check_degenerate_cells: bool = True,
    check_duplicate_vertices: bool = True,
    check_inverted_cells: bool = False,  # Expensive, opt-in
    check_out_of_bounds: bool = True,
    check_manifoldness: bool = False,  # Only 2D, opt-in
    check_self_intersection: bool = False,  # Very expensive, opt-in
    tolerance: float = 1e-10,
    raise_on_error: bool = False,
) -> Mapping[str, bool | int | torch.Tensor]:
    """Validate mesh integrity and detect common errors.

    Performs a comprehensive set of checks to ensure mesh is well-formed
    and suitable for geometric computations.

    Args:
        mesh: Mesh to validate
        check_degenerate_cells: Check for zero/negative area cells
        check_duplicate_vertices: Check for coincident vertices within tolerance
        check_inverted_cells: Check for cells with negative orientation (expensive)
        check_out_of_bounds: Check that cell indices are valid
        check_manifoldness: Check manifold topology (2D only, expensive)
        check_self_intersection: Check for self-intersecting cells (very expensive)
        tolerance: Tolerance for geometric checks (areas, distances)
        raise_on_error: If True, raise ValueError on first error. If False,
            return dict with all validation results.

    Returns:
        Dictionary with validation results:
            - "valid": bool, True if all enabled checks passed
            - "n_degenerate_cells": int, number of degenerate cells found
            - "degenerate_cell_indices": Tensor of indices (if any found)
            - "n_duplicate_vertices": int, number of duplicate vertex pairs
            - "duplicate_vertex_pairs": Tensor of index pairs (if any found)
            - "n_out_of_bounds_cells": int, cells with invalid indices
            - "out_of_bounds_cell_indices": Tensor of cell indices (if any)
            - "n_inverted_cells": int (if check enabled)
            - "inverted_cell_indices": Tensor (if check enabled and any found)
            - "is_manifold": bool (if check enabled, 2D only)
            - "non_manifold_edges": Tensor of edge indices (if check enabled)

    Raises:
        ValueError: If raise_on_error=True and validation fails

    Example:
        >>> report = validate_mesh(mesh)
        >>> if not report["valid"]:
        >>>     print(f"Found {report['n_degenerate_cells']} degenerate cells")
        >>>     print(f"Indices: {report['degenerate_cell_indices']}")
    """
    results = {
        "valid": True,
    }

    ### Check for out-of-bounds indices FIRST (before any geometric computations)
    if check_out_of_bounds:
        if mesh.n_cells > 0:
            min_index = mesh.cells.min()
            max_index = mesh.cells.max()

            out_of_bounds_mask = (mesh.cells < 0) | (mesh.cells >= mesh.n_points)
            out_of_bounds_cells = torch.any(out_of_bounds_mask, dim=1)
            n_out_of_bounds = out_of_bounds_cells.sum().item()

            results["n_out_of_bounds_cells"] = n_out_of_bounds

            if n_out_of_bounds > 0:
                results["valid"] = False
                results["out_of_bounds_cell_indices"] = torch.where(
                    out_of_bounds_cells
                )[0]

                if raise_on_error:
                    raise ValueError(
                        f"Found {n_out_of_bounds} cells with out-of-bounds indices.\n"
                        f"Cell indices must be in range [0, {mesh.n_points}), "
                        f"but got {min_index.item()=} and {max_index.item()=}.\n"
                        f"Problem cells: {results['out_of_bounds_cell_indices'].tolist()[:10]}"
                    )
        else:
            results["n_out_of_bounds_cells"] = 0

    ### Early return if out-of-bounds indices found (can't compute geometry)
    if check_out_of_bounds and results.get("n_out_of_bounds_cells", 0) > 0:
        if raise_on_error:
            # Already raised above
            pass
        else:
            # Skip remaining geometric checks
            return results

    ### Check for duplicate vertices
    if check_duplicate_vertices:
        # Compute pairwise distances between all points (expensive for large meshes)
        # For efficiency, only check if mesh is small or use approximate method
        if mesh.n_points < 10000:  # Exact check for small meshes
            # Compute all pairwise distances
            diff = mesh.points.unsqueeze(0) - mesh.points.unsqueeze(1)  # (n, n, d)
            distances = torch.norm(diff, dim=-1)  # (n, n)

            # Find pairs with distance < tolerance (excluding diagonal)
            mask = distances < tolerance
            mask.fill_diagonal_(False)  # Exclude self-pairs

            duplicate_indices = torch.where(torch.triu(mask, diagonal=1))
            n_duplicates = len(duplicate_indices[0])

            results["n_duplicate_vertices"] = n_duplicates

            if n_duplicates > 0:
                results["valid"] = False
                results["duplicate_vertex_pairs"] = torch.stack(
                    duplicate_indices, dim=1
                )

                if raise_on_error:
                    raise ValueError(
                        f"Found {n_duplicates} pairs of duplicate vertices "
                        f"(within tolerance={tolerance}).\n"
                        f"First few pairs: {results['duplicate_vertex_pairs'][:5].tolist()}"
                    )
        else:
            # For large meshes, skip exact check (too expensive)
            # Could implement approximate duplicate detection with spatial hashing
            results["n_duplicate_vertices"] = -1  # Not checked

    ### Check for degenerate cells
    if check_degenerate_cells and mesh.n_cells > 0:
        # Compute cell areas
        areas = mesh.cell_areas

        # Find cells with area below tolerance
        degenerate_mask = areas < tolerance
        n_degenerate = degenerate_mask.sum().item()

        results["n_degenerate_cells"] = n_degenerate

        if n_degenerate > 0:
            results["valid"] = False
            results["degenerate_cell_indices"] = torch.where(degenerate_mask)[0]
            results["degenerate_cell_areas"] = areas[degenerate_mask]

            if raise_on_error:
                raise ValueError(
                    f"Found {n_degenerate} degenerate cells with area < {tolerance}.\n"
                    f"Problem cells: {results['degenerate_cell_indices'].tolist()[:10]}\n"
                    f"Areas: {results['degenerate_cell_areas'].tolist()[:10]}"
                )
    elif check_degenerate_cells:
        results["n_degenerate_cells"] = 0

    ### Check for inverted cells (cells with negative orientation)
    if check_inverted_cells and mesh.n_cells > 0:
        # For simplicial meshes, check if determinant is negative
        # This indicates inverted orientation

        if mesh.n_manifold_dims == mesh.n_spatial_dims:
            # Volume mesh: can compute signed volume
            cell_vertices = mesh.points[mesh.cells]  # (n_cells, n_verts, n_dims)

            # Compute signed volume using determinant
            # For n-simplex: V = (1/n!) * det([v1-v0, v2-v0, ..., vn-v0])
            relative_vectors = cell_vertices[:, 1:] - cell_vertices[:, [0]]

            # Compute determinant
            if mesh.n_manifold_dims == 3:
                # 3D case: determinant of 3x3 matrix
                det = torch.det(relative_vectors)  # (n_cells,)

                inverted_mask = det < 0
                n_inverted = inverted_mask.sum().item()

                results["n_inverted_cells"] = n_inverted

                if n_inverted > 0:
                    results["valid"] = False
                    results["inverted_cell_indices"] = torch.where(inverted_mask)[0]

                    if raise_on_error:
                        raise ValueError(
                            f"Found {n_inverted} inverted cells (negative orientation).\n"
                            f"Problem cells: {results['inverted_cell_indices'].tolist()[:10]}"
                        )
            else:
                # For other dimensions, orientation check is more complex
                results["n_inverted_cells"] = -1  # Not implemented
        else:
            # Codimension > 0: orientation not well-defined
            results["n_inverted_cells"] = -1  # Not applicable
    elif check_inverted_cells:
        results["n_inverted_cells"] = 0

    ### Check manifoldness (2D only)
    if check_manifoldness:
        if mesh.n_manifold_dims == 2 and mesh.n_spatial_dims >= 2:
            # Check that each edge is shared by at most 2 triangles
            from physicsnemo.mesh.boundaries import extract_candidate_facets

            # Extract all edges (with duplicates)
            edges_with_dupes, parent_cells = extract_candidate_facets(
                mesh.cells, manifold_codimension=1
            )

            # Sort edges to canonical form
            edges_sorted = torch.sort(edges_with_dupes, dim=1).values

            # Find unique edges and their counts
            unique_edges, inverse_indices, counts = torch.unique(
                edges_sorted, dim=0, return_inverse=True, return_counts=True
            )

            # Manifold edges should appear exactly 1 (boundary) or 2 (interior) times
            non_manifold_mask = counts > 2
            n_non_manifold = non_manifold_mask.sum().item()

            results["is_manifold"] = n_non_manifold == 0
            results["n_non_manifold_edges"] = n_non_manifold

            if n_non_manifold > 0:
                results["valid"] = False
                results["non_manifold_edges"] = unique_edges[non_manifold_mask]
                results["non_manifold_edge_counts"] = counts[non_manifold_mask]

                if raise_on_error:
                    raise ValueError(
                        f"Mesh is not manifold: {n_non_manifold} edges shared by >2 faces.\n"
                        f"First few problem edges: {results['non_manifold_edges'][:5].tolist()}"
                    )
        else:
            results["is_manifold"] = None  # Only defined for 2D manifolds
            results["n_non_manifold_edges"] = -1  # Not applicable

    ### Check for self-intersections (very expensive, opt-in only)
    if check_self_intersection:
        # This is very expensive: O(n^2) cell-cell intersection tests
        # For production use, would need BVH acceleration
        results["has_self_intersection"] = None  # Not implemented yet
        results["intersecting_cell_pairs"] = None

        # TODO: Implement BVH-accelerated self-intersection detection
        if raise_on_error:
            raise NotImplementedError(
                "Self-intersection checking not yet implemented.\n"
                "This is a very expensive operation requiring BVH acceleration."
            )

    return results


def check_duplicate_cell_vertices(mesh: "Mesh") -> tuple[int, torch.Tensor]:
    """Check for cells with duplicate vertices (degenerate simplices).

    A valid n-simplex must have n+1 distinct vertices. Cells with duplicate
    vertices are degenerate and should be removed.

    Args:
        mesh: Mesh to check

    Returns:
        Tuple of (n_invalid_cells, invalid_cell_indices)

    Example:
        >>> n_invalid, indices = check_duplicate_cell_vertices(mesh)
        >>> if n_invalid > 0:
        >>>     print(f"Found {n_invalid} cells with duplicate vertices")
        >>>     mesh = mesh.slice_cells(~torch.isin(torch.arange(mesh.n_cells), indices))
    """
    if mesh.n_cells == 0:
        return 0, torch.tensor([], dtype=torch.long, device=mesh.cells.device)

    # For each cell, check if all vertices are unique
    invalid_cells = []

    for i in range(mesh.n_cells):
        cell_verts = mesh.cells[i]
        unique_verts = torch.unique(cell_verts)

        if len(unique_verts) < len(cell_verts):
            invalid_cells.append(i)

    if len(invalid_cells) == 0:
        return 0, torch.tensor([], dtype=torch.long, device=mesh.cells.device)

    invalid_indices = torch.tensor(
        invalid_cells, dtype=torch.long, device=mesh.cells.device
    )
    return len(invalid_cells), invalid_indices
