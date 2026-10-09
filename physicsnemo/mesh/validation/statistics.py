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

    areas = mesh.cell_areas

    ### Count the vertices used by any cell: the distinct values of the sorted
    # connectivity. Like torch.unique, this counts out-of-range indices too, but
    # its output has a fixed size, so it needs no device synchronization.
    used_vertices = mesh.cells.flatten().sort().values
    n_used = 1 + (used_vertices[1:] != used_vertices[:-1]).sum()
    del used_vertices  # one index per cell vertex; free it before the quality metrics

    ### Compute quality metrics (includes edge lengths internally)
    # compute_quality_metrics already computes min/max edge lengths per cell,
    # so we derive stats from those to avoid a redundant compute_cell_edge_lengths call.
    quality_metrics = compute_quality_metrics(mesh)
    min_edge = quality_metrics["min_edge_length"]
    max_edge = quality_metrics["max_edge_length"]
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
        min_edge.min(),
        min_edge.mean(),
        max_edge.mean(),
        max_edge.max(),
        max_edge.std(correction=0),
        *[s for key in optional_stats for s in summarize(quality_metrics[key])],
    ]
    values = torch.stack([s.to(torch.float64) for s in scalars]).tolist()

    stats["n_degenerate_cells"] = int(values[0])
    stats["n_isolated_vertices"] = mesh.n_points - int(values[1])
    stats["cell_area_stats"] = tuple(values[2:6])
    edge_min, min_edge_mean, max_edge_mean, edge_max, edge_std = values[6:11]
    stats["edge_length_stats"] = (
        edge_min,
        (min_edge_mean + max_edge_mean) / 2.0,
        edge_max,
        edge_std,
    )
    for i, key in enumerate(optional_stats):
        stats[f"{key}_stats"] = tuple(values[11 + 4 * i : 15 + 4 * i])

    return stats
