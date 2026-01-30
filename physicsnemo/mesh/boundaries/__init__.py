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

"""Boundary detection and facet extraction for simplicial meshes.

This module provides:
1. Boundary detection: identify vertices, edges, and cells on mesh boundaries
2. Facet extraction: extract lower-dimensional simplices from cells
3. Boundary mesh extraction: extract the watertight boundary surface
4. Topology checking: validate watertight and manifold properties
5. Mesh cleaning: repair common mesh issues
"""

from physicsnemo.mesh.boundaries._boundary_extraction import (
    extract_boundary_mesh_data,
)
from physicsnemo.mesh.boundaries._cleaning import (
    clean_mesh,
    merge_duplicate_points,
    remove_duplicate_cells,
    remove_unused_points,
)
from physicsnemo.mesh.boundaries._detection import (
    get_boundary_cells,
    get_boundary_edges,
    get_boundary_vertices,
)
from physicsnemo.mesh.boundaries._facet_extraction import (
    categorize_facets_by_count,
    compute_aggregation_weights,
    deduplicate_and_aggregate_facets,
    extract_candidate_facets,
    extract_facet_mesh_data,
)
from physicsnemo.mesh.boundaries._topology import (
    is_manifold,
    is_watertight,
)
