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

"""
Shared mesh utilities for the AirFRANS datapipe.

Provides two approaches for computing mesh-dependent quantities:

1. **physicsnemo.Mesh-based** (``compute_gradients``, ``compute_airfoil_normals_nearest``):
   Uses ``compute_point_derivatives`` and ``torch.cdist`` nearest-point lookup,
   matching the canonical old dataloader. Used by the VTK reader which has
   access to both the internal and airfoil boundary meshes.

2. **PyVista-based** (``compute_mesh_quantities``): Falls back to raw PyVista
   ``compute_derivative`` and surface extraction. Used by the Arrow reader
   which only has the internal mesh (no separate airfoil boundary).

Also defines the physical constants shared across the pipeline.
"""

from __future__ import annotations

import logging

import numpy as np
import pyvista as pv
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus import compute_point_derivatives
from physicsnemo.mesh.io import from_pyvista
from physicsnemo.mesh.projections import project

logger = logging.getLogger(__name__)

# --- AirFRANS physical constants ---
# NOTE: RHO=1 is correct; in some places the AirFRANS authors incorrectly
# report their density as 1.204, but the OpenFOAM case files use 1. You can
# confirm this from the data: RHO=1 yields constant far-field total pressure
# (physically correct), but RHO=1.204 does not.
RHO = 1.0  # kg/m^3
NU = 1.56e-5  # m^2/s kinematic viscosity
CHORD = 1.0  # reference chord length


# ---------------------------------------------------------------------------
# Approach 1: physicsnemo.Mesh-based (matches old dataloader)
# ---------------------------------------------------------------------------


def compute_gradients(
    internal_pv: pv.UnstructuredGrid,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pressure gradient and velocity jacobian via physicsnemo Mesh calculus.

    Converts the PyVista mesh to a 2D physicsnemo Mesh, slices vector fields
    to 2D, then uses ``compute_point_derivatives`` (least-squares
    reconstruction) -- the same code path as the canonical old dataloader.

    Parameters
    ----------
    internal_pv : pv.UnstructuredGrid
        Internal volume mesh with point_data ``"p"`` and ``"U"``.

    Returns
    -------
    grad_p : torch.Tensor
        Pressure gradient, shape ``(N, 2)``.
    velocity_jacobian : torch.Tensor
        Velocity Jacobian, shape ``(N, 2, 2)``.
    """
    internal = project(from_pyvista(internal_pv), keep_dims=[0, 1])

    # Slice 3D vector fields to 2D (replicating transform_point_data=True)
    if "U" in internal.point_data.keys() and internal.point_data["U"].shape[-1] == 3:
        internal.point_data["U"] = internal.point_data["U"][:, :2]

    mesh_with_grads = compute_point_derivatives(mesh=internal, keys=["p", "U"])
    grad_p = mesh_with_grads.point_data["p_gradient"]
    velocity_jacobian = mesh_with_grads.point_data["U_gradient"]
    return grad_p, velocity_jacobian


def compute_airfoil_normals_nearest(
    internal_points: torch.Tensor,
    airfoil_pv: pv.PolyData,
    implicit_distance: torch.Tensor,
) -> torch.Tensor:
    """Compute airfoil normals via nearest-point lookup into the boundary mesh.

    Matches the canonical old dataloader: converts the airfoil boundary to a
    physicsnemo Mesh, uses its ``point_normals``, and assigns each internal
    point the normal of its nearest airfoil point via ``torch.cdist``.
    Points not on the airfoil surface (``implicit_distance != 0``) are NaN.

    Parameters
    ----------
    internal_points : torch.Tensor
        Internal mesh points, shape ``(N, 2)``.
    airfoil_pv : pv.PolyData
        Airfoil boundary mesh (PyVista PolyData).
    implicit_distance : torch.Tensor
        Signed distance to airfoil surface, shape ``(N,)``.

    Returns
    -------
    torch.Tensor
        Airfoil normals, shape ``(N, 2)``. NaN for off-surface points.
    """
    airfoil = project(
        from_pyvista(airfoil_pv, manifold_dim=1),
        keep_dims=[0, 1],
    )

    point_is_on_airfoil = implicit_distance == 0
    nearest_idx = torch.cdist(internal_points, airfoil.points).argmin(dim=1)
    normals = -1 * airfoil.point_normals[nearest_idx]
    normals[~point_is_on_airfoil] = torch.nan
    return normals


# ---------------------------------------------------------------------------
# Approach 2: PyVista-based fallback (for Arrow reader without airfoil mesh)
# ---------------------------------------------------------------------------


def compute_mesh_quantities(
    mesh: pv.UnstructuredGrid,
) -> dict[str, np.ndarray]:
    """Compute mesh-dependent derived quantities using PyVista directly.

    Fallback for the Arrow reader which lacks a separate airfoil boundary
    mesh. Uses PyVista's ``compute_derivative`` for gradients and surface
    extraction + ``compute_normals`` for airfoil normals.

    Parameters
    ----------
    mesh : pv.UnstructuredGrid
        PyVista mesh with point_data ``"p"``, ``"U"``, ``"implicit_distance"``.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys ``grad_p``, ``velocity_jacobian``,
        ``airfoil_normals``, each as float32 numpy arrays.
    """
    results: dict[str, np.ndarray] = {}
    n_points = mesh.n_points

    # --- Pressure gradient (computed on cells, interpolated to points) ---
    try:
        mesh_with_grad = mesh.compute_derivative(
            scalars="p", gradient=True, preference="cell"
        ).cell_data_to_point_data()
        grad_p = mesh_with_grad.point_data["gradient"][:, :2].astype(np.float32)
    except Exception:
        logger.warning("Failed to compute pressure gradient, filling with NaN")
        grad_p = np.full((n_points, 2), np.nan, dtype=np.float32)
    results["grad_p"] = grad_p

    # --- Velocity jacobian (computed on points) ---
    try:
        mesh_with_jac = mesh.compute_derivative(
            scalars="U", gradient="jacobian"
        )
        jac_raw = mesh_with_jac.point_data["jacobian"].reshape(-1, 3, 3)
        velocity_jacobian = jac_raw[:, :2, :2].astype(np.float32)
    except Exception:
        logger.warning("Failed to compute velocity jacobian, filling with NaN")
        velocity_jacobian = np.full((n_points, 2, 2), np.nan, dtype=np.float32)
    results["velocity_jacobian"] = velocity_jacobian

    # --- Airfoil surface normals ---
    sdf = mesh.point_data.get("implicit_distance")
    if sdf is not None:
        on_surface = sdf == 0
        n_surface = int(on_surface.sum())

        if n_surface > 0:
            surface_ids = np.where(on_surface)[0]
            surface_mesh = mesh.extract_points(surface_ids)

            if surface_mesh.n_points > 0:
                surface_mesh = surface_mesh.extract_surface()
                surface_mesh = surface_mesh.compute_normals(
                    cell_normals=False,
                    point_normals=True,
                    auto_orient_normals=True,
                )

                sampled = mesh.sample(
                    target=surface_mesh,
                    snap_to_closest_point=True,
                )
                normals_full = sampled["Normals"][:, :2].astype(np.float32)
                normals_full *= -1  # Orient outwards (into the fluid domain)
                normals_full[~on_surface] = np.nan
            else:
                normals_full = np.full((n_points, 2), np.nan, dtype=np.float32)
        else:
            normals_full = np.full((n_points, 2), np.nan, dtype=np.float32)
    else:
        normals_full = np.full((n_points, 2), np.nan, dtype=np.float32)
    results["airfoil_normals"] = normals_full

    return results
