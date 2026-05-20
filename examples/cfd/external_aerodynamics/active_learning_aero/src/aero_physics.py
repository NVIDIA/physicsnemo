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

"""External-aerodynamics physics for the active-learning example.

This is the *domain-adapter* layer of the example: everything that ties
the AL recipe to external-aerodynamics surface-CFD lives here. To
adapt the recipe to a different problem (a different geometry, a
different quantity of interest, or a different non-dimensionalization),
replace this file and leave ``gp_utils.py`` / ``strategies.py`` /
``run_al.py`` unchanged.

Contents:

* Reference constants (``FRONTAL_AREA``, ``REFERENCE_VELOCITY``,
  ``REFERENCE_DENSITY``, ``DRAG_COEFF_SCALE``).
* ``compute_force_coefficients_torch`` — pressure / shear surface integral
  that converts a surface-field prediction into a force coefficient.
* ``compute_drag_target_from_batch`` — drag target extracted from a
  dataloader batch (used as the GP regression target).
* ``compute_drag_from_subsampled_outputs`` — Monte-Carlo drag estimate
  from subsampled GeoTransolver predictions, with the computational
  graph preserved so gradients flow back into the encoder.
"""

from __future__ import annotations

import torch

from physicsnemo.models.domino.utils import unstandardize

# ---------------------------------------------------------------------------
# Aerodynamic reference constants
# ---------------------------------------------------------------------------

FRONTAL_AREA = 1.85  # m²
REFERENCE_VELOCITY = 40.0  # m/s
REFERENCE_DENSITY = 1.225  # kg/m³
DRAG_COEFF_SCALE = 0.35  # GP target = Cd / DRAG_COEFF_SCALE

# Pre-computed Cd = 2 F / (rho * V^2 * A) prefactor; reused by drag-integration
# helpers below and constant across all calls.
_CD_PREFACTOR = 2.0 / (FRONTAL_AREA * REFERENCE_DENSITY * REFERENCE_VELOCITY**2)


# ---------------------------------------------------------------------------
# Force-coefficient computation
# ---------------------------------------------------------------------------


def compute_force_coefficients_torch(
    normals: torch.Tensor,
    area: torch.Tensor,
    coeff: float,
    p: torch.Tensor,
    wss: torch.Tensor,
    force_direction: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute force coefficients from surface pressure and wall shear stress.

    Parameters
    ----------
    normals : torch.Tensor
        Surface normals, shape ``(N, 3)``.
    area : torch.Tensor
        Cell areas, shape ``(N,)`` or ``(N, 1)``.
    coeff : float
        Reference coefficient ``2 / (A * rho * U²)``.
    p : torch.Tensor
        Surface pressure, shape ``(N,)``.
    wss : torch.Tensor
        Wall shear stress, shape ``(N, 3)``.
    force_direction : torch.Tensor | None
        Unit vector for force projection; defaults to ``[1, 0, 0]`` (drag).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(c_total, c_pressure, c_friction)`` — scalar tensors.
    """
    if force_direction is None:
        force_direction = torch.tensor(
            [1.0, 0.0, 0.0],
            device=normals.device,
            dtype=normals.dtype,
        )
    area = area.view(-1)
    n_dot_f = (normals * force_direction).sum(dim=-1)
    c_p = coeff * (n_dot_f * area * p).sum()
    wss_dot_f = (wss * force_direction).sum(dim=-1)
    c_f = -coeff * (wss_dot_f * area).sum()
    return c_p + c_f, c_p, c_f


def compute_drag_target_from_batch(
    batch: dict,
    surface_factors: dict,
    device: torch.device,
    drag_scale: float = DRAG_COEFF_SCALE,
) -> torch.Tensor:
    """Extract a GP-scaled drag target from a dataloader batch.

    Unnormalises predicted surface fields, integrates pressure and shear to
    obtain the drag coefficient Cd, then returns ``Cd / drag_scale`` as a
    ``(1,)`` tensor suitable for GP training.
    """
    if "fields_full" in batch:
        fields = batch["fields_full"]
    else:
        fields = batch["fields"]
    if isinstance(fields, list):
        fields = fields[0]

    fields_phys = unstandardize(fields, surface_factors["mean"], surface_factors["std"])
    fields_phys = fields_phys.squeeze(0)
    p = fields_phys[:, 0]
    wss = fields_phys[:, 1:4]

    normals = batch["surface_normals"].squeeze(0).to(device, dtype=fields_phys.dtype)
    area = batch["surface_areas"].squeeze(0).to(device, dtype=fields_phys.dtype)
    p, wss = p.to(device), wss.to(device)

    c_total, _, _ = compute_force_coefficients_torch(
        normals, area, _CD_PREFACTOR, p, wss
    )
    return (c_total / drag_scale).unsqueeze(0)


def compute_drag_from_subsampled_outputs(
    outputs: torch.Tensor,
    batch: dict,
    surface_factors: dict,
    device: torch.device,
    drag_scale: float = DRAG_COEFF_SCALE,
) -> torch.Tensor:
    """Monte-Carlo drag estimate from subsampled GeoTransolver predictions.

    Preserves the computational graph through *outputs* so gradients can
    flow back into the GeoTransolver.  Returns ``(1,)`` in GP-scaled space.
    """
    fields_phys = unstandardize(
        outputs,
        surface_factors["mean"],
        surface_factors["std"],
    ).squeeze(0)
    p = fields_phys[:, 0]
    wss = fields_phys[:, 1:4]

    normals = (
        batch["surface_normals_sub"].squeeze(0).to(device, dtype=fields_phys.dtype)
    )
    areas = batch["surface_areas_sub"].squeeze(0).to(device, dtype=fields_phys.dtype)

    n_full = batch["surface_areas"].squeeze(0).shape[0]
    n_sub = p.shape[0]
    scale = n_full / n_sub

    c_total, _, _ = compute_force_coefficients_torch(
        normals,
        areas,
        _CD_PREFACTOR * scale,
        p,
        wss,
    )
    return (c_total / drag_scale).unsqueeze(0)
