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

"""Registry of FunctionSpec classes to benchmark with ASV."""

from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.nn.functional.derivatives import (
    MeshGreenGaussGradient,
    MeshlessFDDerivatives,
    MeshLSQGradient,
    RectilinearGridCurl,
    RectilinearGridDivergence,
    RectilinearGridGradient,
    RectilinearGridLaplacian,
    SpectralGridGradient,
    UniformGridCurl,
    UniformGridDivergence,
    UniformGridGradient,
    UniformGridLaplacian,
)
from physicsnemo.nn.functional.fourier_spectral import (
    IRFFT,
    IRFFT2,
    RFFT,
    RFFT2,
    Imag,
    Real,
    ViewAsComplex,
)
from physicsnemo.nn.functional.geometry import (
    DisplacePoints,
    FarthestPointSampling,
    FreeFormDeformPoints,
    MeshPoissonDiskSample,
    MeshToVoxelFraction,
    MorphPoints,
    RadialBasisFunctionDeformPoints,
    RayMeshIntersect,
    SignedDistanceField,
)
from physicsnemo.nn.functional.interpolation import (
    GridToPointInterpolation,
    PointToGridInterpolation,
)
from physicsnemo.nn.functional.neighbors import KNN, RadiusSearch
from physicsnemo.nn.functional.regularization_parameterization import (
    DropPath,
    WeightFact,
)

# FunctionSpec classes listed here must implement ``make_inputs_forward`` for ASV.
# ``make_inputs_backward`` is optional and only used when backward benchmarks run.
FUNCTIONAL_SPECS: tuple[type[FunctionSpec], ...] = (
    # Regularization / parameterization.
    DropPath,
    WeightFact,
    # Neighbor queries.
    KNN,
    RadiusSearch,
    # Derivatives.
    UniformGridGradient,
    RectilinearGridDivergence,
    RectilinearGridCurl,
    RectilinearGridLaplacian,
    RectilinearGridGradient,
    MeshLSQGradient,
    MeshGreenGaussGradient,
    SpectralGridGradient,
    MeshlessFDDerivatives,
    UniformGridDivergence,
    UniformGridCurl,
    UniformGridLaplacian,
    # Geometry.
    DisplacePoints,
    MorphPoints,
    RadialBasisFunctionDeformPoints,
    FreeFormDeformPoints,
    FarthestPointSampling,
    MeshPoissonDiskSample,
    MeshToVoxelFraction,
    RayMeshIntersect,
    SignedDistanceField,
    # Interpolation.
    GridToPointInterpolation,
    PointToGridInterpolation,
    # Fourier spectral.
    RFFT,
    RFFT2,
    IRFFT,
    IRFFT2,
    ViewAsComplex,
    Real,
    Imag,
)

# Remeshing uses a dedicated benchmark that measures the complete Mesh-level
# operation, including topology cleanup. The current benchmark measures
# synchronized CUDA execution; see ``benchmarks/physicsnemo/mesh/remeshing.py``.

__all__ = ["FUNCTIONAL_SPECS"]
