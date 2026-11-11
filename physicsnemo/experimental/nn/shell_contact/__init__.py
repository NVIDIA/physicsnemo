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
  the mesh-attention models;
* :func:`shell_penetration_depths`, :func:`wall_penetration_depths` give
  per-node penetration fields for metrics;
* :func:`clamp_to_wall` caps a predicted shape at an analytic rigid wall.

The search and its narrow phase are functionals with a Torch and a Warp
implementation selected by a string of the settings; they are meant to be
brought under :class:`physicsnemo.core.function_spec.FunctionSpec` before
they graduate from the experimental namespace.
"""

from ._closest_point import TriangleProjection, supported_quad_mask
from .features import CONTACT_FEATURE_DIM, shell_contact_edge_features
from .geometry import project_onto_facets
from .penetration import (
    clamp_to_wall,
    shell_penetration_depths,
    wall_penetration_depths,
)
from .search import ShellContactPairs, ShellContactSearch, shell_contact_pairs

__all__ = [
    "CONTACT_FEATURE_DIM",
    "ShellContactPairs",
    "ShellContactSearch",
    "TriangleProjection",
    "clamp_to_wall",
    "project_onto_facets",
    "shell_contact_edge_features",
    "shell_contact_pairs",
    "shell_penetration_depths",
    "supported_quad_mask",
    "wall_penetration_depths",
]
