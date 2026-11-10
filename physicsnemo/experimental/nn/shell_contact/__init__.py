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
"""Shell contact as pure functions over tensors.

* :func:`shell_contact_pairs` discovers the active node/facet pairs of a shell
  mesh (Warp BVH broad phase, fused Warp or Torch narrow phase, optional
  reference side test, clearance and crossing sweep) with live geometry;
* :func:`shell_contact_edge_features` turns pairs into the contact features of
  the mesh-attention models, :func:`smooth_contact_cutoff` into their taper;
* :func:`shell_penetration_depths`, :func:`wall_penetration_depths` give
  per-node penetration fields for metrics;
* :func:`project_shell_penetrations` and :func:`clamp_to_wall` are the
  evaluation-time geometric corrections of a predicted shape.
"""

from ._closest_point import supported_quad_mask
from .features import (
    CONTACT_FEATURE_DIM,
    KINEMATIC_CONTACT_FEATURE_DIM,
    contact_edge_features,
    shell_contact_edge_features,
    smooth_contact_cutoff,
)
from .geometry import (
    crossing_flags,
    facet_side,
    project_onto_facets,
    reference_facet_projection,
)
from .penetration import (
    clamp_to_wall,
    node_maximum,
    project_shell_penetrations,
    shell_penetration_depths,
    wall_penetration_depths,
)
from .search import (
    ShellContactPairs,
    ShellContactSearch,
    candidate_activation_gaps,
    linear_closest_approach,
    shell_contact_pairs,
    uses_fused_narrow_phase,
)

__all__ = [
    "CONTACT_FEATURE_DIM",
    "KINEMATIC_CONTACT_FEATURE_DIM",
    "ShellContactPairs",
    "ShellContactSearch",
    "candidate_activation_gaps",
    "clamp_to_wall",
    "contact_edge_features",
    "crossing_flags",
    "facet_side",
    "linear_closest_approach",
    "node_maximum",
    "project_onto_facets",
    "project_shell_penetrations",
    "reference_facet_projection",
    "shell_contact_edge_features",
    "shell_contact_pairs",
    "shell_penetration_depths",
    "smooth_contact_cutoff",
    "supported_quad_mask",
    "uses_fused_narrow_phase",
    "wall_penetration_depths",
]
