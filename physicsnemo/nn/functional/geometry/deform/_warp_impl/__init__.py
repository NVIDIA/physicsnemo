# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Warp backends for point deformation, surface search, and deformation energies."""

from .energy_op import (
    closed_surface_volume_contributions_warp,
    hinge_bending_terms_warp,
    simplex_inversion_terms_warp,
    simplex_measure_components_warp,
    simplex_stvk_terms_warp,
)
from .ffd_op import (
    ffd_field_warp_impl,
    ffd_points_warp,
)
from .op import (
    compact_shepard_field_warp_impl,
    morph_points_warp,
)
from .rbf_op import rbf_field_warp
from .shrinkwrap_op import (
    nearest_surface_faces_warp,
    nearest_surface_faces_warp_impl,
)
from .sobolev_op import (
    sobolev_deform_points_warp,
    sobolev_displacement_warp_backward_impl,
    sobolev_displacement_warp_impl,
)

__all__ = [
    "closed_surface_volume_contributions_warp",
    "compact_shepard_field_warp_impl",
    "ffd_field_warp_impl",
    "ffd_points_warp",
    "hinge_bending_terms_warp",
    "morph_points_warp",
    "nearest_surface_faces_warp",
    "nearest_surface_faces_warp_impl",
    "rbf_field_warp",
    "simplex_inversion_terms_warp",
    "simplex_measure_components_warp",
    "simplex_stvk_terms_warp",
    "sobolev_deform_points_warp",
    "sobolev_displacement_warp_backward_impl",
    "sobolev_displacement_warp_impl",
]
