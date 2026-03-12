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

"""Mesh statistics and summary information.

Computes global statistics about mesh properties including counts,
distributions, and quality summaries.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.validation.quality import compute_quality_metrics

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

    Parameters
    ----------
    mesh : Mesh
        Mesh to analyze
    tolerance : float
        Threshold for degenerate cell detection

    Returns
    -------
    Mapping[str, int | float | tuple[float, float, float, float]]
        Dictionary with statistics

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
    >>> mesh = two_triangles_2d.load()
    >>> stats = compute_mesh_statistics(mesh)
    >>> assert "n_points" in stats and "n_cells" in stats
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

    ### Compute cell area statistics
    stats["cell_area_stats"] = (
        areas.min().item(),
        areas.mean().item(),
        areas.max().item(),
        areas.std(correction=0).item(),
    )

    ### Compute quality metrics (includes edge lengths internally)
    quality_metrics = compute_quality_metrics(mesh)

    ### Extract edge length statistics from quality metrics
    # compute_quality_metrics already computes min/max edge lengths per cell,
    # so we derive stats from those to avoid a redundant compute_cell_edge_lengths call.
    min_edge = quality_metrics["min_edge_length"]
    max_edge = quality_metrics["max_edge_length"]
    stats["edge_length_stats"] = (
        min_edge.min().item(),
        (min_edge.mean().item() + max_edge.mean().item()) / 2.0,
        max_edge.max().item(),
        max_edge.std(correction=0).item(),
    )

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

    return stats
