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

These functionals moved into the ``physicsnemo.mesh`` modules that use them:
``mesh_lsq_gradient`` into ``physicsnemo.mesh.calculus``, the rest as private
kernels. ``nn`` must not import ``mesh``, so the old names resolve lazily, on
first access. Remove this module, the ``__getattr__`` hooks that call it, and
its ``.importlinter`` exception when ``physicsnemo.mesh`` becomes its own
library.
"""

import warnings

from physicsnemo.core.warnings import LegacyFeatureWarning

_DEFORM = "physicsnemo.mesh.transformations.deform"
_ENERGIES = (
    "closed_surface_volume_energy",
    "simplex_inversion_energy",
    "simplex_measure_energy",
    "simplex_strain_energy",
    "surface_bending_energy",
    "total_measure_energy",
)

# Each moved function -> its public replacement, or None if it has none.
_USE_INSTEAD = {
    "mesh_green_gauss_gradient": None,
    "mesh_lsq_gradient": "physicsnemo.mesh.calculus.mesh_lsq_gradient",
    "displace_points": f"{_DEFORM}.displace on a Mesh",
    "free_form_deform_points": f"{_DEFORM}.free_form_deform on a Mesh",
    "morph_points": f"{_DEFORM}.morph on a Mesh",
    "radial_basis_function_deform_points": (
        f"{_DEFORM}.radial_basis_function_deform on a Mesh"
    ),
    "shrinkwrap_points": f"{_DEFORM}.shrinkwrap on a Mesh",
    "sobolev_deform_points": f"{_DEFORM}.sobolev_deform on a Mesh",
    **{name: f"physicsnemo.mesh.deformation.{name} on a Mesh" for name in _ENERGIES},
    "remeshing": "physicsnemo.mesh.remeshing.remesh on a Mesh",
    "mesh_poisson_disk_sample": None,
    "mesh_to_voxel_fraction": None,
    "ray_mesh_intersect": None,
}
_DERIVATIVES = ("mesh_green_gauss_gradient", "mesh_lsq_gradient")
_GEOMETRY = tuple(name for name in _USE_INSTEAD if name not in _DERIVATIVES)

# Old module -> the names it exported that moved into physicsnemo.mesh.
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
    """Return ``name`` from the mesh module that now holds it, with a warning.

    For a module ``__getattr__``: raises ``AttributeError`` for names that did
    not move.
    """
    if name not in MOVED_TO_MESH[module]:
        raise AttributeError(f"module {module!r} has no attribute {name!r}")

    from physicsnemo.mesh.calculus import _green_gauss_gradient, _lsq_gradient
    from physicsnemo.mesh.deformation import _kernels as energies
    from physicsnemo.mesh.remeshing import _kernels as remeshing
    from physicsnemo.mesh.sampling import _poisson_disk
    from physicsnemo.mesh.spatial import _ray_mesh_intersect, _voxel_fraction
    from physicsnemo.mesh.transformations.deform import _kernels as deform

    homes = (
        _green_gauss_gradient,
        _lsq_gradient,
        deform,
        energies,
        remeshing,
        _poisson_disk,
        _ray_mesh_intersect,
        _voxel_fraction,
    )
    home = next(home for home in homes if name in home.__all__)
    use_instead = _USE_INSTEAD.get(name)
    warnings.warn(
        f"Importing {name!r} from {module!r} is deprecated and will stop working "
        "in a future release."
        + (f" Use {use_instead} instead." if use_instead else ""),
        LegacyFeatureWarning,
        stacklevel=3,
    )
    return getattr(home, name)
