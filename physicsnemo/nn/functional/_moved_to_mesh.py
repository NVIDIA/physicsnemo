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

"""Deprecated ``physicsnemo.nn.functional`` names of the mesh functionals.

These functionals moved to ``physicsnemo.mesh.functional``. ``nn`` must not
import ``mesh``, so the old names resolve lazily, on first access. Remove this
module, the ``__getattr__`` hooks that call it, and its ``.importlinter``
exception when ``physicsnemo.mesh`` becomes its own library.
"""

import warnings

from physicsnemo.core.warnings import LegacyFeatureWarning

_DERIVATIVES = ("mesh_green_gauss_gradient", "mesh_lsq_gradient")
_GEOMETRY = (
    "closed_surface_volume_energy",
    "displace_points",
    "free_form_deform_points",
    "mesh_poisson_disk_sample",
    "mesh_to_voxel_fraction",
    "morph_points",
    "radial_basis_function_deform_points",
    "ray_mesh_intersect",
    "remeshing",
    "shrinkwrap_points",
    "simplex_inversion_energy",
    "simplex_measure_energy",
    "simplex_strain_energy",
    "sobolev_deform_points",
    "surface_bending_energy",
    "total_measure_energy",
)

# Old module -> the names it exported that now live in the module of the same
# path under physicsnemo.mesh.functional.
MOVED_TO_MESH = {
    "physicsnemo.nn.functional": _DERIVATIVES + _GEOMETRY,
    "physicsnemo.nn.functional.derivatives": _DERIVATIVES
    + ("MeshGreenGaussGradient", "MeshLSQGradient"),
    "physicsnemo.nn.functional.geometry": _GEOMETRY
    + (
        "ClosedSurfaceVolumeEnergy",
        "DisplacePoints",
        "FreeFormDeformPoints",
        "MeshPoissonDiskSample",
        "MeshToVoxelFraction",
        "MorphPoints",
        "RadialBasisFunctionDeformPoints",
        "RayMeshIntersect",
        "Remeshing",
        "ShrinkwrapPoints",
        "SimplexInversionEnergy",
        "SimplexMeasureEnergy",
        "SimplexStrainEnergy",
        "SobolevDeformPoints",
        "SurfaceBendingEnergy",
        "TotalMeasureEnergy",
    ),
}


def moved_to_mesh(module: str, name: str):
    """Return ``name`` from its new mesh module, with a deprecation warning.

    For a module ``__getattr__``: raises ``AttributeError`` for names that did
    not move.
    """
    if name not in MOVED_TO_MESH[module]:
        raise AttributeError(f"module {module!r} has no attribute {name!r}")

    from physicsnemo.mesh import functional

    new_module = {
        "physicsnemo.nn.functional": functional,
        "physicsnemo.nn.functional.derivatives": functional.derivatives,
        "physicsnemo.nn.functional.geometry": functional.geometry,
    }[module]
    warnings.warn(
        f"Importing {name!r} from {module!r} is deprecated. "
        f"Use 'from {new_module.__name__} import {name}' instead. "
        "This backward-compatibility shim will be removed in a future release.",
        LegacyFeatureWarning,
        stacklevel=3,
    )
    return getattr(new_module, name)
