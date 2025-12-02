"""Comprehensive mesh repair pipeline.

Combines multiple repair operations into a single convenient function.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def repair_mesh(
    mesh: "Mesh",
    remove_duplicates: bool = True,
    remove_degenerates: bool = True,
    remove_isolated: bool = True,
    fix_orientation: bool = False,  # Requires 3D, has loops
    fill_holes: bool = False,  # Expensive, opt-in
    make_manifold: bool = False,  # Changes topology, opt-in
    tolerance: float = 1e-6,
    max_hole_edges: int = 10,
) -> tuple["Mesh", dict[str, dict]]:
    """Apply multiple repair operations in sequence.

    Applies a series of mesh repair operations to clean up common problems.
    Operations are applied in a specific order to maximize effectiveness.

    Order of operations:
    1. Remove degenerate cells (zero area)
    2. Remove duplicate vertices
    3. Remove isolated vertices
    4. Fix orientation (if enabled)
    5. Fill holes (if enabled)
    6. Make manifold (if enabled)

    Args:
        mesh: Input mesh to repair
        remove_duplicates: Merge coincident vertices
        remove_degenerates: Remove zero-area cells and cells with duplicate vertices
        remove_isolated: Remove vertices not in any cell
        fix_orientation: Ensure consistent face normals (2D in 3D only)
        fill_holes: Close boundary loops (expensive)
        make_manifold: Split non-manifold edges (changes topology)
        tolerance: Distance/area tolerance for various checks
        max_hole_edges: Maximum hole size to fill

    Returns:
        Tuple of (repaired_mesh, all_stats) where all_stats is a dict mapping
        operation name to its individual stats dict

    Example:
        >>> mesh_clean, stats = repair_mesh(
        ...     mesh,
        ...     remove_duplicates=True,
        ...     remove_degenerates=True,
        ...     remove_isolated=True,
        ... )
        >>> print(f"Removed {stats['degenerates']['n_cells_original'] - stats['degenerates']['n_cells_final']} cells")
    """
    current_mesh = mesh
    all_stats = {}

    ### Operation 1: Remove degenerate cells
    if remove_degenerates:
        from physicsnemo.mesh.repair.degenerate_removal import (
            remove_degenerate_cells as remove_deg,
        )

        current_mesh, stats = remove_deg(current_mesh, area_tolerance=tolerance)
        all_stats["degenerates"] = stats

    ### Operation 2: Remove duplicate vertices
    if remove_duplicates:
        from physicsnemo.mesh.repair.duplicate_removal import (
            remove_duplicate_vertices as remove_dup,
        )

        current_mesh, stats = remove_dup(current_mesh, tolerance=tolerance)
        all_stats["duplicates"] = stats

    ### Operation 3: Remove isolated vertices
    if remove_isolated:
        from physicsnemo.mesh.repair.isolated_removal import (
            remove_isolated_vertices as remove_iso,
        )

        current_mesh, stats = remove_iso(current_mesh)
        all_stats["isolated"] = stats

    ### Operation 4: Fix orientation
    if fix_orientation:
        if mesh.n_manifold_dims == 2 and mesh.n_spatial_dims == 3:
            from physicsnemo.mesh.repair.orientation import (
                fix_orientation as fix_orient,
            )

            current_mesh, stats = fix_orient(current_mesh)
            all_stats["orientation"] = stats
        else:
            all_stats["orientation"] = {"skipped": "Only for 2D manifolds in 3D"}

    ### Operation 5: Fill holes
    if fill_holes:
        if mesh.n_manifold_dims == 2:
            from physicsnemo.mesh.repair.hole_filling import fill_holes as fill_h

            current_mesh, stats = fill_h(current_mesh, max_hole_edges=max_hole_edges)
            all_stats["holes"] = stats
        else:
            all_stats["holes"] = {"skipped": "Only for 2D manifolds"}

    ### Operation 6: Make manifold
    if make_manifold:
        # Non-manifold edge splitting is not yet implemented
        raise NotImplementedError(
            "Manifold repair (split_nonmanifold_edges) is not yet implemented.\n"
            "This operation would duplicate vertices at non-manifold edges to make "
            "the mesh manifold, but requires complex topology-preserving logic.\n"
            "Set make_manifold=False to skip this operation."
        )

    return current_mesh, all_stats
