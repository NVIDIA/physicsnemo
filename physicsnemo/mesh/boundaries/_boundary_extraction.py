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

"""Boundary mesh extraction for simplicial meshes.

Extracts codimension-1 facets that appear in exactly one parent cell (the
boundary surface). This is a convenience wrapper around
:func:`~physicsnemo.mesh.boundaries._facet_extraction.extract_facet_mesh_data`
with ``target_counts="boundary"`` and ``manifold_codimension=1``.
"""

from typing import TYPE_CHECKING, Literal

import torch
from tensordict import TensorDict

from physicsnemo.mesh.boundaries._facet_extraction import extract_facet_mesh_data

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def extract_boundary_mesh_data(
    parent_mesh: "Mesh",
    data_source: Literal["points", "cells"] = "cells",
    data_aggregation: Literal["mean", "area_weighted", "inverse_distance"] = "mean",
) -> tuple[torch.Tensor, TensorDict]:
    """Extract boundary mesh data from parent mesh.

    Extracts only the codimension-1 facets that lie on the boundary (appear in
    exactly one parent cell). This produces the watertight boundary surface.

    Parameters
    ----------
    parent_mesh : Mesh
        The parent mesh to extract boundary from
    data_source : {"points", "cells"}, optional
        Whether to inherit data from "cells" or "points"
    data_aggregation : {"mean", "area_weighted", "inverse_distance"}, optional
        How to aggregate data (only applies when data_source="cells").
        For boundary facets each facet has exactly one parent cell, so
        aggregation only matters if the same boundary facet appears multiple
        times (which shouldn't happen in a valid mesh).

    Returns
    -------
    boundary_cells : torch.Tensor
        Connectivity for boundary mesh, shape (n_boundary_facets, n_vertices_per_facet)
    boundary_cell_data : TensorDict
        Aggregated TensorDict for boundary mesh cells

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.procedural import lumpy_ball
    >>> from physicsnemo.mesh import Mesh
    >>> # Extract surface of a volume mesh
    >>> vol_mesh = lumpy_ball.load(n_shells=2, subdivisions=1)
    >>> boundary_cells, boundary_data = extract_boundary_mesh_data(vol_mesh)
    >>> boundary_mesh = Mesh(points=vol_mesh.points, cells=boundary_cells, cell_data=boundary_data)
    >>> assert boundary_mesh.n_manifold_dims == 2  # Surface triangles
    """
    return extract_facet_mesh_data(
        parent_mesh,
        manifold_codimension=1,
        data_source=data_source,
        data_aggregation=data_aggregation,
        target_counts="boundary",
    )
