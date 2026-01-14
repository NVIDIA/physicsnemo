"""Mesh statistics and summary information.

Computes global statistics about mesh properties including counts,
distributions, and quality summaries.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_mesh_statistics(
    mesh: "Mesh",
    tolerance: float = 1e-10,
) -> Mapping[str, int | float | tuple[float, float, float, float]]:
    """Compute summary statistics for mesh.

    Returns dictionary with mesh statistics:
    - n_points: Number of vertices
    - n_cells: Number of cells
    - n_manifold_dims: Manifold dimension
    - n_spatial_dims: Spatial dimension
    - n_degenerate_cells: Cells with area < tolerance
    - n_isolated_vertices: Vertices not in any cell
    - edge_length_stats: (min, mean, max, std) of edge lengths
    - cell_area_stats: (min, mean, max, std) of cell areas
    - aspect_ratio_stats: (min, mean, max, std) of aspect ratios
    - quality_score_stats: (min, mean, max, std) of quality scores

    Args:
        mesh: Mesh to analyze
        tolerance: Threshold for degenerate cell detection

    Returns:
        Dictionary with statistics

    Example:
        >>> stats = compute_mesh_statistics(mesh)
        >>> print(f"Mesh: {stats['n_points']} points, {stats['n_cells']} cells")
        >>> print(f"Edge lengths: {stats['edge_length_stats']}")
        >>> print(f"Quality: {stats['quality_score_stats']}")
    """
    stats = {
        "n_points": mesh.n_points,
        "n_cells": mesh.n_cells,
        "n_manifold_dims": mesh.n_manifold_dims,
        "n_spatial_dims": mesh.n_spatial_dims,
    }

    if mesh.n_cells == 0:
        # Empty mesh
        stats["n_degenerate_cells"] = 0
        stats["n_isolated_vertices"] = mesh.n_points
        stats["edge_length_stats"] = (0.0, 0.0, 0.0, 0.0)
        stats["cell_area_stats"] = (0.0, 0.0, 0.0, 0.0)
        return stats

    ### Count degenerate cells
    areas = mesh.cell_areas
    n_degenerate = (areas < tolerance).sum().item()
    stats["n_degenerate_cells"] = n_degenerate

    ### Count isolated vertices
    # Vertices that don't appear in any cell
    used_vertices = torch.unique(mesh.cells.flatten())
    n_used = len(used_vertices)
    stats["n_isolated_vertices"] = mesh.n_points - n_used

    ### Compute edge length statistics
    cell_vertices = mesh.points[mesh.cells]  # (n_cells, n_verts, n_dims)
    n_verts_per_cell = mesh.n_manifold_dims + 1

    edge_lengths_list = []
    for i in range(n_verts_per_cell):
        for j in range(i + 1, n_verts_per_cell):
            edge = cell_vertices[:, j] - cell_vertices[:, i]
            length = torch.norm(edge, dim=-1)
            edge_lengths_list.append(length)

    all_edge_lengths = torch.cat(edge_lengths_list, dim=0)

    stats["edge_length_stats"] = (
        all_edge_lengths.min().item(),
        all_edge_lengths.mean().item(),
        all_edge_lengths.max().item(),
        all_edge_lengths.std(correction=0).item(),
    )

    ### Compute cell area statistics
    stats["cell_area_stats"] = (
        areas.min().item(),
        areas.mean().item(),
        areas.max().item(),
        areas.std(correction=0).item(),
    )

    ### Compute quality metrics statistics
    try:
        from physicsnemo.mesh.validation.quality import compute_quality_metrics

        quality_metrics = compute_quality_metrics(mesh)

        if "aspect_ratio" in quality_metrics.keys():
            aspect_ratios = quality_metrics["aspect_ratio"]
            stats["aspect_ratio_stats"] = (
                aspect_ratios.min().item(),
                aspect_ratios.mean().item(),
                aspect_ratios.max().item(),
                aspect_ratios.std(correction=0).item(),
            )

        if "quality_score" in quality_metrics.keys():
            quality_scores = quality_metrics["quality_score"]
            stats["quality_score_stats"] = (
                quality_scores.min().item(),
                quality_scores.mean().item(),
                quality_scores.max().item(),
                quality_scores.std(correction=0).item(),
            )
    except Exception:
        # If quality computation fails, skip it
        pass

    return stats
