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

from physicsnemo.mesh.validation.quality import (
    _compute_quality_metrics,
    compute_cell_edge_lengths,
)

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
    - edge_length_stats: (min, mean, max, std) of the lengths of every edge of
      every cell, so an interior edge shared by two cells counts twice. On a
      closed surface, where every edge has two cells, these equal the
      statistics of the unique edges.
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

    areas = mesh.cell_areas

    ### Count the vertices used by any cell. Unlike torch.unique, a mask over the
    # points has a fixed size, so this needs no device synchronization. The
    # .long() is for int32 cells, since index_fill_ takes only int64 indices.
    n_used = (
        torch.zeros(mesh.n_points, dtype=torch.bool, device=mesh.cells.device)
        .index_fill_(0, mesh.cells.flatten().long(), True)
        .sum()
    )

    ### Lengths of every edge of every cell (a shared edge counts once per cell)
    # The quality metrics reuse them, so they are computed once.
    cell_edge_lengths = compute_cell_edge_lengths(mesh)
    quality_metrics = _compute_quality_metrics(mesh, cell_edge_lengths)
    if mesh.n_manifold_dims == 0:
        # A 0-simplex has no edges; its quality metrics report a NaN length.
        edge_lengths = quality_metrics["min_edge_length"]
    else:
        edge_lengths = cell_edge_lengths.flatten()
    optional_stats = [
        key
        for key in ("aspect_ratio", "quality_score")
        if key in quality_metrics.keys()
    ]

    def summarize(values: torch.Tensor) -> list[torch.Tensor]:
        return [values.min(), values.mean(), values.max(), values.std(correction=0)]

    ### Read every statistic in one device-to-host transfer (one sync).
    # float64 holds the counts and the float32 statistics exactly, so the values
    # equal those of separate .item() calls.
    scalars = [
        (areas < tolerance).sum(),
        n_used,
        *summarize(areas),
        *summarize(edge_lengths),
        *[s for key in optional_stats for s in summarize(quality_metrics[key])],
    ]
    values = torch.stack([s.to(torch.float64) for s in scalars]).tolist()

    stats["n_degenerate_cells"] = int(values[0])
    stats["n_isolated_vertices"] = mesh.n_points - int(values[1])
    stats["cell_area_stats"] = tuple(values[2:6])
    stats["edge_length_stats"] = tuple(values[6:10])
    for i, key in enumerate(optional_stats):
        stats[f"{key}_stats"] = tuple(values[10 + 4 * i : 14 + 4 * i])

    return stats
