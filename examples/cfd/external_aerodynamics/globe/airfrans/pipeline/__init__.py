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

"""AirFRANS datapipe pipeline components."""

from .arrow_reader import AirFRANSArrowReader
from .mesh_utils import (
    CHORD,
    NU,
    RHO,
    compute_airfoil_normals_nearest,
    compute_gradients,
    compute_mesh_quantities,
)
from .transforms import (
    ComputeAirfoilNormals,
    ComputeForceCoefficients,
    ComputeFreestreamQuantities,
    ComputeGradients,
    NondimensionalizeFields,
    PatchNonPhysicalValues,
)
from .vtk_reader import AirFRANSVTKReader

__all__ = [
    "AirFRANSArrowReader",
    "AirFRANSVTKReader",
    "CHORD",
    "ComputeAirfoilNormals",
    "ComputeForceCoefficients",
    "ComputeFreestreamQuantities",
    "ComputeGradients",
    "NU",
    "NondimensionalizeFields",
    "PatchNonPhysicalValues",
    "RHO",
    "compute_airfoil_normals_nearest",
    "compute_gradients",
    "compute_mesh_quantities",
]
