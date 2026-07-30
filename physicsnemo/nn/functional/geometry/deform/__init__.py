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

"""Point-deformation functionals and reference-relative deformation energies."""

from .deform import DisplacePoints, MorphPoints, displace_points, morph_points
from .energy import (
    ClosedSurfaceVolumeEnergy,
    SimplexInversionEnergy,
    SimplexMeasureEnergy,
    SimplexStrainEnergy,
    SurfaceBendingEnergy,
    TotalMeasureEnergy,
    closed_surface_volume_energy,
    simplex_inversion_energy,
    simplex_measure_energy,
    simplex_strain_energy,
    surface_bending_energy,
    total_measure_energy,
)
from .ffd import FreeFormDeformPoints, free_form_deform_points
from .rbf import RadialBasisFunctionDeformPoints, radial_basis_function_deform_points

__all__ = [
    "ClosedSurfaceVolumeEnergy",
    "DisplacePoints",
    "FreeFormDeformPoints",
    "MorphPoints",
    "RadialBasisFunctionDeformPoints",
    "SimplexInversionEnergy",
    "SimplexMeasureEnergy",
    "SimplexStrainEnergy",
    "SurfaceBendingEnergy",
    "TotalMeasureEnergy",
    "closed_surface_volume_energy",
    "displace_points",
    "free_form_deform_points",
    "morph_points",
    "radial_basis_function_deform_points",
    "simplex_inversion_energy",
    "simplex_measure_energy",
    "simplex_strain_energy",
    "surface_bending_energy",
    "total_measure_energy",
]
