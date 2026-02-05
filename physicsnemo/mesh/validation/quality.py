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

"""Quality metrics for mesh cells.

Computes geometric quality metrics for simplicial cells including aspect ratio,
skewness, and angles. Higher quality = better shaped cells.
"""

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from physicsnemo.mesh.curvature._utils import compute_triangle_angles

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def compute_quality_metrics(mesh: "Mesh") -> TensorDict:
    """Compute geometric quality metrics for all cells.

    Returns TensorDict with per-cell quality metrics:
    - aspect_ratio: max_edge / min_altitude (lower is better, 1.0 is equilateral)
    - min_angle: Minimum interior angle in radians
    - max_angle: Maximum interior angle in radians
    - edge_length_ratio: max_edge / min_edge (1.0 is equilateral)
    - quality_score: Combined metric in [0,1] (1.0 is perfect equilateral)

    Parameters
    ----------
    mesh : Mesh
        Mesh to analyze

    Returns
    -------
    TensorDict
        TensorDict of shape (n_cells,) with quality metrics

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
    >>> mesh = two_triangles_2d.load()
    >>> metrics = compute_quality_metrics(mesh)
    >>> assert "quality_score" in metrics.keys()
    """
    if mesh.n_cells == 0:
        return TensorDict(
            {},
            batch_size=torch.Size([0]),
            device=mesh.points.device,
        )

    device = mesh.points.device
    dtype = mesh.points.dtype

    ### Compute edge lengths for each cell
    cell_vertices = mesh.points[mesh.cells]  # (n_cells, n_verts, n_dims)
    n_cells = mesh.n_cells
    n_verts_per_cell = mesh.n_manifold_dims + 1

    # Compute all pairwise edge lengths within each cell (vectorized)
    # Generate all (i, j) pairs with i < j using upper triangular indices
    i_indices, j_indices = torch.triu_indices(
        n_verts_per_cell, n_verts_per_cell, offset=1, device=device
    )
    # Compute all edge vectors at once: (n_cells, n_edges, n_dims)
    edges = cell_vertices[:, j_indices] - cell_vertices[:, i_indices]
    edge_lengths = torch.linalg.vector_norm(edges, dim=-1)  # (n_cells, n_edges)

    max_edge = edge_lengths.max(dim=1).values
    min_edge = edge_lengths.min(dim=1).values

    edge_length_ratio = max_edge / (min_edge + 1e-10)

    ### Compute aspect ratio (approximation using area and edges)
    areas = mesh.cell_areas

    # For triangles: aspect_ratio ≈ max_edge / (4*area/perimeter)
    # For general: use max_edge / characteristic_length
    perimeter = edge_lengths.sum(dim=1)
    characteristic_length = areas * n_verts_per_cell / (perimeter + 1e-10)
    aspect_ratio = max_edge / (characteristic_length + 1e-10)

    ### Compute angles (for 2D manifolds - triangles)
    if mesh.n_manifold_dims == 2:
        # Compute all three angles per triangle
        angle0 = compute_triangle_angles(
            cell_vertices[:, 0],
            cell_vertices[:, 1],
            cell_vertices[:, 2],
        )
        angle1 = compute_triangle_angles(
            cell_vertices[:, 1],
            cell_vertices[:, 2],
            cell_vertices[:, 0],
        )
        angle2 = compute_triangle_angles(
            cell_vertices[:, 2],
            cell_vertices[:, 0],
            cell_vertices[:, 1],
        )

        all_angles = torch.stack([angle0, angle1, angle2], dim=1)
        min_angle = all_angles.min(dim=1).values
        max_angle = all_angles.max(dim=1).values
    else:
        # For non-triangular cells, angle computation is more complex
        min_angle = torch.full((n_cells,), float("nan"), dtype=dtype, device=device)
        max_angle = torch.full((n_cells,), float("nan"), dtype=dtype, device=device)

    ### Compute combined quality score
    # Perfect simplex has:
    # - edge_length_ratio = 1.0 (all edges equal)
    # - For triangles: all angles = π/3
    # - aspect_ratio = 1.0

    # Quality score combines multiple metrics
    # Each component in [0, 1] where 1 is perfect

    # Edge uniformity: 1 / edge_length_ratio (clamped)
    edge_uniformity = 1.0 / torch.clamp(edge_length_ratio, min=1.0, max=10.0)

    # Aspect ratio quality: 1 / aspect_ratio (clamped)
    aspect_quality = 1.0 / torch.clamp(aspect_ratio, min=1.0, max=10.0)

    # Angle quality (for triangles): min_angle / (π/3) and (π/3) / max_angle
    if mesh.n_manifold_dims == 2:
        ideal_angle = torch.pi / 3
        min_angle_quality = torch.clamp(min_angle / ideal_angle, max=1.0)
        max_angle_quality = torch.clamp(ideal_angle / max_angle, max=1.0)
        angle_quality = (min_angle_quality + max_angle_quality) / 2
    else:
        angle_quality = torch.ones((n_cells,), dtype=dtype, device=device)

    # Combined score (geometric mean)
    quality_score = (edge_uniformity * aspect_quality * angle_quality) ** (1 / 3)

    return TensorDict(
        {
            "aspect_ratio": aspect_ratio,
            "edge_length_ratio": edge_length_ratio,
            "min_angle": min_angle,
            "max_angle": max_angle,
            "min_edge_length": min_edge,
            "max_edge_length": max_edge,
            "quality_score": quality_score,
        },
        batch_size=torch.Size([n_cells]),
        device=device,
    )
