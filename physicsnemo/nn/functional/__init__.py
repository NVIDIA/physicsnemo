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

from .derivatives import (
    mesh_green_gauss_gradient,
    mesh_lsq_gradient,
    meshless_fd_derivatives,
    rectilinear_grid_curl,
    rectilinear_grid_divergence,
    rectilinear_grid_gradient,
    rectilinear_grid_laplacian,
    spectral_grid_gradient,
    uniform_grid_curl,
    uniform_grid_divergence,
    uniform_grid_gradient,
    uniform_grid_laplacian,
)
from .equivariant_ops import (
    legendre_polynomials,
    polar_and_dipole_basis,
    smooth_log,
    spherical_basis,
    vector_project,
)
from .fourier_spectral import imag, irfft, irfft2, real, rfft, rfft2, view_as_complex
from .geometry import (
    displace_points,
    farthest_point_sampling,
    free_form_deform_points,
    mesh_poisson_disk_sample,
    mesh_to_voxel_fraction,
    morph_points,
    ray_mesh_intersect,
    remeshing,
    signed_distance_field,
)
from .interpolation import (
    grid_to_point_interpolation,
    interpolation,
    point_to_grid_interpolation,
)
from .natten import na1d, na2d, na3d
from .neighbors import knn, radius_search
from .regularization_parameterization import drop_path, weight_fact
from .rendering import (
    isosurface_render,
    line_integral_convolution,
    mesh_raycast,
    point_cloud_render,
    scalar_field_to_rgba,
    vector_field_to_rgba,
    volume_render,
    wireframe_render,
)

__all__ = [
    "displace_points",
    "irfft",
    "irfft2",
    "drop_path",
    "farthest_point_sampling",
    "free_form_deform_points",
    "uniform_grid_curl",
    "uniform_grid_divergence",
    "uniform_grid_laplacian",
    "grid_to_point_interpolation",
    "imag",
    "interpolation",
    "knn",
    "isosurface_render",
    "legendre_polynomials",
    "line_integral_convolution",
    "mesh_green_gauss_gradient",
    "mesh_raycast",
    "meshless_fd_derivatives",
    "mesh_lsq_gradient",
    "mesh_poisson_disk_sample",
    "mesh_to_voxel_fraction",
    "morph_points",
    "na1d",
    "na2d",
    "na3d",
    "point_to_grid_interpolation",
    "polar_and_dipole_basis",
    "radius_search",
    "real",
    "ray_mesh_intersect",
    "remeshing",
    "rectilinear_grid_curl",
    "rectilinear_grid_divergence",
    "rectilinear_grid_gradient",
    "rectilinear_grid_laplacian",
    "rfft",
    "rfft2",
    "point_cloud_render",
    "scalar_field_to_rgba",
    "signed_distance_field",
    "smooth_log",
    "spectral_grid_gradient",
    "spherical_basis",
    "uniform_grid_gradient",
    "vector_field_to_rgba",
    "vector_project",
    "volume_render",
    "view_as_complex",
    "weight_fact",
    "wireframe_render",
]
