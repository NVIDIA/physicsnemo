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
AirFRANS-specific transforms for the datapipe pipeline.

These transforms implement the physics-based preprocessing steps from the
original AirFRANS dataloader as modular, composable operations that run on
GPU tensors.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from physicsnemo.datapipes.transforms.base import Transform

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus import compute_point_derivatives

from .mesh_utils import CHORD, NU, RHO


class ComputeGradients(Transform):
    """Compute pressure gradient and velocity jacobian from mesh connectivity.

    Builds a physicsnemo ``Mesh`` from ``points`` and ``internal_cells``,
    then runs weighted least-squares gradient reconstruction via
    ``compute_point_derivatives``.

    Added keys:
        - ``grad_p``: pressure gradient, shape ``(N, 2)``
        - ``velocity_jacobian``: velocity Jacobian, shape ``(N, 2, 2)``
    """

    def __call__(self, data: TensorDict) -> TensorDict:
        points = data["points"]
        cells = data["internal_cells"]
        p = data["p"]
        U = data["U"]

        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={"p": p, "U": U},
        )
        mesh_with_grads = compute_point_derivatives(mesh, keys=["p", "U"])

        data["grad_p"] = mesh_with_grads.point_data["p_gradient"]
        data["velocity_jacobian"] = mesh_with_grads.point_data["U_gradient"]

        return data


class ComputeAirfoilNormals(Transform):
    """Compute airfoil surface normals via nearest-point lookup.

    Builds a 1D ``Mesh`` from ``airfoil_points`` and ``airfoil_cells``
    to obtain boundary point normals, then assigns each internal point
    the normal of its nearest airfoil point using ``torch.cdist``.
    Off-surface points (``implicit_distance != 0``) are set to NaN.

    Added keys:
        - ``airfoil_normals``: surface normals, shape ``(N, 2)``
    """

    def __call__(self, data: TensorDict) -> TensorDict:
        points = data["points"]
        implicit_distance = data["implicit_distance"]
        airfoil_pts = data["airfoil_points"]
        airfoil_cells = data["airfoil_cells"]

        if airfoil_pts.shape[0] == 0:
            data["airfoil_normals"] = torch.full_like(points, float("nan"))
            return data

        airfoil_mesh = Mesh(points=airfoil_pts, cells=airfoil_cells)

        point_is_on_airfoil = implicit_distance == 0
        nearest_idx = torch.cdist(points, airfoil_pts).argmin(dim=1)
        normals = -1 * airfoil_mesh.point_normals[nearest_idx]
        normals[~point_is_on_airfoil] = torch.nan

        data["airfoil_normals"] = normals

        return data


class ComputeFreestreamQuantities(Transform):
    """Compute freestream reference quantities from angle of attack and inlet velocity.

    Reads ``angle_of_attack`` and ``inlet_velocity`` from the TensorDict and
    adds derived freestream quantities needed for nondimensionalization.

    Added keys:
        - ``U_inf``: freestream velocity vector, shape ``(2,)``
        - ``U_inf_magnitude``: freestream speed scalar, shape ``(1,)``
        - ``q_inf``: dynamic pressure ``0.5 * rho * V^2``, shape ``(1,)``
        - ``delta_FS``: Blasius boundary layer length scale, shape ``(1,)``
    """

    def __call__(self, data: TensorDict) -> TensorDict:
        aoa = data["angle_of_attack"]  # (1,)
        V = data["inlet_velocity"]  # (1,)

        U_inf = torch.stack([torch.cos(aoa), torch.sin(aoa)], dim=-1).squeeze(0) * V
        U_inf_magnitude = V.squeeze(0)
        q_inf = 0.5 * RHO * U_inf_magnitude**2
        delta_FS = (NU / U_inf_magnitude * CHORD) ** 0.5

        data["U_inf"] = U_inf
        data["U_inf_magnitude"] = U_inf_magnitude.unsqueeze(0)
        data["q_inf"] = q_inf.unsqueeze(0)
        data["delta_FS"] = delta_FS.unsqueeze(0)

        return data


class NondimensionalizeFields(Transform):
    """Nondimensionalize raw field data using freestream reference quantities.

    Expects ``ComputeFreestreamQuantities`` to have already run, providing
    ``U_inf``, ``U_inf_magnitude``, and ``q_inf`` in the TensorDict.

    Added keys:
        - ``U/|U_inf|``: velocity normalized by freestream speed, shape ``(N, 2)``
        - ``DeltaU/|U_inf|``: velocity difference from freestream, shape ``(N, 2)``
        - ``C_p``: pressure coefficient, shape ``(N,)``
        - ``C_pt``: total pressure coefficient, shape ``(N,)``
        - ``ln(1+nut/nu)``: log turbulent viscosity ratio, shape ``(N,)``
        - ``grad_C_p_chord``: nondimensional pressure gradient, shape ``(N, 2)``
    """

    def __call__(self, data: TensorDict) -> TensorDict:
        U = data["U"]  # (N, 2)
        p = data["p"]  # (N,)
        nut = data["nut"]  # (N,)
        U_inf = data["U_inf"]  # (2,)
        U_inf_mag = data["U_inf_magnitude"]  # (1,)
        q_inf = data["q_inf"]  # (1,)

        U_over_U_inf = U / U_inf_mag
        delta_U_over_U_inf = (U - U_inf.unsqueeze(0)) / U_inf_mag

        q = q_inf * torch.sum(U_over_U_inf**2, dim=-1)  # local dynamic pressure
        C_p = p / q_inf
        C_pt = (p + q) / q_inf
        ln_nut = torch.log1p(nut / NU)

        data["U/|U_inf|"] = U_over_U_inf
        data["DeltaU/|U_inf|"] = delta_U_over_U_inf
        data["C_p"] = C_p
        data["C_pt"] = C_pt
        data["ln(1+nut/nu)"] = ln_nut

        # Nondimensionalize the pressure gradient
        if "grad_p" in data.keys():
            data["grad_C_p_chord"] = (data["grad_p"] / q_inf) * CHORD

        return data


class ComputeForceCoefficients(Transform):
    """Compute surface force coefficients from velocity gradients, normals, and pressure.

    Expects ``NondimensionalizeFields`` and mesh-derived ``velocity_jacobian``
    and ``airfoil_normals`` to be present. Force coefficients are only
    physically meaningful on the airfoil surface; off-surface points are NaN.

    Added keys:
        - ``C_F_shear``: wall shear force coefficient, shape ``(N, 2)``
        - ``C_F_pressure``: pressure force coefficient, shape ``(N, 2)``
        - ``C_F``: net force coefficient, shape ``(N, 2)``
    """

    def __call__(self, data: TensorDict) -> TensorDict:
        if "velocity_jacobian" not in data.keys() or "airfoil_normals" not in data.keys():
            return data

        vel_jac = data["velocity_jacobian"]  # (N, 2, 2)
        normals = data["airfoil_normals"]  # (N, 2)
        p = data["p"]  # (N,)
        q_inf = data["q_inf"]  # (1,)

        # Strain rate tensor: 0.5 * (grad_U + grad_U^T)
        strain_rate = 0.5 * (vel_jac + vel_jac.transpose(1, 2))
        wall_shear_stress = 2 * NU * strain_rate

        # Wall shear force = tau_ij * n_j
        wall_shear_force = torch.einsum("pij,pj->pi", wall_shear_stress, normals)
        pressure_force = -1 * p.unsqueeze(-1) * normals
        net_force = wall_shear_force + pressure_force

        data["C_F_shear"] = wall_shear_force / q_inf
        data["C_F_pressure"] = pressure_force / q_inf
        data["C_F"] = net_force / q_inf

        return data


class PatchNonPhysicalValues(Transform):
    """Mask non-physical values in the output fields by setting them to NaN.

    Points where the total pressure coefficient exceeds the threshold are
    considered non-physical (typically caused by numerical artifacts near
    the wall). All output fields at these points are set to NaN.

    Parameters
    ----------
    threshold : float
        Maximum allowed value for ``C_pt``. Points exceeding this are masked.
    output_keys : list[str] or None
        Keys to mask. If ``None``, masks all keys that were added by the
        nondimensionalization and force transforms.
    warn_fraction : float
        If more than this fraction of points are masked, log a warning.
    """

    DEFAULT_OUTPUT_KEYS = [
        "U/|U_inf|",
        "DeltaU/|U_inf|",
        "C_p",
        "C_pt",
        "ln(1+nut/nu)",
        "grad_C_p_chord",
        "C_F_shear",
        "C_F_pressure",
        "C_F",
    ]

    def __init__(
        self,
        threshold: float = 1.02,
        output_keys: list[str] | None = None,
        warn_fraction: float = 0.0001,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.output_keys = output_keys or self.DEFAULT_OUTPUT_KEYS
        self.warn_fraction = warn_fraction

    def __call__(self, data: TensorDict) -> TensorDict:
        if "C_pt" not in data.keys():
            return data

        C_pt = data["C_pt"]
        non_physical = C_pt > self.threshold
        fraction = non_physical.float().mean().item()

        if fraction > self.warn_fraction:
            import logging

            logging.getLogger(__name__).warning(
                "%.2f%% of points have non-physical C_pt > %.2f",
                fraction * 100,
                self.threshold,
            )

        for key in self.output_keys:
            if key in data.keys():
                tensor = data[key]
                if tensor.shape[0] == non_physical.shape[0]:
                    if tensor.ndim == 1:
                        tensor = tensor.clone()
                        tensor[non_physical] = float("nan")
                    else:
                        tensor = tensor.clone()
                        tensor[non_physical] = float("nan")
                    data[key] = tensor

        return data

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}"
