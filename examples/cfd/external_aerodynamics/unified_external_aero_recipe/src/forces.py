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
Integrated aerodynamic force and moment coefficients from surface fields.

Surface models in this recipe predict, per surface cell, a pressure
coefficient :math:`C_p = (p - p_\\infty)/q_\\infty` and a skin-friction
coefficient vector :math:`C_f = \\tau_w / q_\\infty` (the pipeline's
``NonDimensionalizeByMetadata`` produces exactly these). The net
aerodynamic load on the body follows from integrating the surface
traction over the vehicle:

.. math::
    \\mathbf{t} = -C_p\\,\\mathbf{n} + \\mathbf{C}_f

where :math:`\\mathbf{n}` is the outward unit normal. The force and
moment coefficient vectors are

.. math::
    \\mathbf{C}_F = \\frac{1}{A_\\text{ref}} \\int_S \\mathbf{t}\\,dA,
    \\qquad
    \\mathbf{C}_M = \\frac{1}{A_\\text{ref} L_\\text{ref}}
        \\int_S (\\mathbf{x} - \\mathbf{x}_\\text{ref}) \\times \\mathbf{t}\\,dA .

The surface integral is evaluated with the mesh quadrature utility
:meth:`physicsnemo.mesh.Mesh.integrate` (cell-data / P0 rule:
:math:`\\int_S f\\,dA = \\sum_c f_c\\,A_c`).

The coefficient vectors are then projected onto an orthonormal
(drag, lift, side) triad built from the per-sample freestream direction
and a global "up" reference, yielding the conventional scalars:

- ``CD`` (drag, along the freestream), ``CL`` (lift), ``CS`` (side force).
- ``CMR`` / ``CMP`` / ``CMY`` (roll / pitch / yaw moment about the
  drag / side / lift axes).

Conventions and assumptions:

- **Outward normals.** ``Mesh.cell_normals`` are unique only up to
  orientation (they follow the cell winding). Forces assume a
  consistently *outward*-oriented surface; if a mesh is wound inward the
  pressure contribution flips sign. Because predicted and reference
  coefficients are integrated identically, the *error* between them is
  unaffected by a global orientation flip.
- **Reference area / length** are physical-unit quantities supplied by
  the caller. ``CD``/``CL``/``CM`` magnitudes are only physically
  meaningful when ``reference_area`` matches the dataset's convention;
  the predicted-vs-reference comparison is independent of it (a common
  positive scale factor cancels in the relative error).
- **Geometry scale.** Pipeline geometry is non-dimensionalized by
  ``L_ref`` (coordinates divided by ``L_ref``). Pass ``length_scale =
  L_ref`` to integrate on a physical-scale surface; areas and moment
  arms are translation-invariant, so the lost ``CenterMesh`` offset does
  not affect forces (and only shifts the moment reference for moments).
- **Full surface resolution.** The quadrature covers exactly the cells
  present on the ``vehicle`` mesh. If the pipeline subsampled the
  surface (``sampling_resolution`` below the mesh's cell count), the
  integral covers only the kept cells and every coefficient shrinks by
  roughly the kept-to-total area fraction -- for predicted and reference
  values alike, so the *comparison* stays meaningful but the magnitudes
  do not. ``ForceContext.coefficients``'s 1:1 points/cells contract
  check cannot detect this (a subsampled surface still satisfies it);
  ``infer.py`` warns when a vehicle's cell count sits at the
  ``sampling_resolution`` cap.
"""

from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import Float
from nondim import NondimFieldType
from omegaconf import DictConfig
from tensordict import TensorDict

from physicsnemo.mesh import DomainMesh, Mesh

### Coefficient keys returned by :func:`force_moment_coefficients`, in a
### stable order so callers can build tables / accumulators against them.
COEFFICIENT_NAMES: tuple[str, ...] = ("CD", "CL", "CS", "CMR", "CMP", "CMY")


def build_axis_frame(
    flow_direction: Float[torch.Tensor, "3"],
    up_direction: Float[torch.Tensor, "3"],
    *,
    eps: float = 1e-8,
) -> tuple[
    Float[torch.Tensor, "3"], Float[torch.Tensor, "3"], Float[torch.Tensor, "3"]
]:
    r"""Build an orthonormal (drag, lift, side) triad.

    The drag axis is the freestream direction. The lift axis is the
    component of ``up_direction`` orthogonal to drag (re-normalized); if
    ``up_direction`` is (nearly) parallel to the freestream, a world axis
    least aligned with the freestream is substituted. The side axis
    completes a right-handed triad as ``lift x drag``.

    Args:
        flow_direction: Freestream velocity (need not be unit), shape ``(3,)``.
        up_direction: Global "up" reference (need not be unit), shape ``(3,)``.
        eps: Numerical floor for norms / degeneracy handling.

    Returns:
        ``(drag_hat, lift_hat, side_hat)`` unit vectors, each shape ``(3,)``.
    """
    drag = flow_direction / (flow_direction.norm() + eps)

    up = up_direction / (up_direction.norm() + eps)
    lift = up - (up @ drag) * drag
    if lift.norm() < eps:
        ### up ~ parallel to the freestream: pick the world axis least
        ### aligned with drag as a surrogate "up", then re-orthogonalize.
        world = torch.eye(3, dtype=drag.dtype, device=drag.device)
        surrogate = world[(world * drag).sum(-1).abs().argmin()]
        lift = surrogate - (surrogate @ drag) * drag
    lift = lift / (lift.norm() + eps)

    side = torch.linalg.cross(lift, drag)
    side = side / (side.norm() + eps)
    return drag, lift, side


def force_moment_coefficients(
    vehicle: Mesh,
    pressure_coeff: Float[torch.Tensor, "n_cells ..."],
    shear_coeff: Float[torch.Tensor, "n_cells 3"],
    *,
    flow_direction: Float[torch.Tensor, "3"],
    up_direction: Float[torch.Tensor, "3"],
    moment_center: Float[torch.Tensor, "3"],
    reference_area: float,
    reference_length: float,
    length_scale: float = 1.0,
) -> dict[str, float]:
    r"""Integrate per-cell surface coefficients into force/moment scalars.

    Args:
        vehicle: Triangulated surface mesh (codimension-1) carrying the
            body. Its cells define the quadrature; ``cell_normals`` and
            ``cell_areas`` are taken from this mesh.
        pressure_coeff: Per-cell pressure coefficient :math:`C_p`, shape
            ``(n_cells,)`` (a trailing singleton dim, e.g. ``(n_cells, 1)``,
            is flattened internally). Must align 1:1 with ``vehicle`` cells.
        shear_coeff: Per-cell skin-friction coefficient vector
            :math:`C_f`, shape ``(n_cells, 3)``.
        flow_direction: Freestream velocity for the sample, shape ``(3,)``
            (defines the drag axis).
        up_direction: Global "up" reference, shape ``(3,)`` (defines the
            lift axis, orthogonalized against drag).
        moment_center: Moment reference point :math:`x_\text{ref}`, shape
            ``(3,)``, in the same (centered, ``length_scale``-applied)
            frame as the mesh.
        reference_area: :math:`A_\text{ref}` in physical units.
        reference_length: :math:`L_\text{ref}` for the moment coefficient.
        length_scale: Factor applied to the mesh geometry before
            integration (pass the pipeline's ``L_ref`` to recover a
            physical-scale surface from non-dimensionalized coordinates).

    Returns:
        ``{name: float}`` for each of :data:`COEFFICIENT_NAMES`
        (``CD``, ``CL``, ``CS``, ``CMR``, ``CMP``, ``CMY``).
    """
    mesh = vehicle.scale(length_scale) if length_scale != 1.0 else vehicle
    device = mesh.points.device
    dtype = torch.float32

    normals = mesh.cell_normals.to(dtype)  # (C, 3) unit, outward (assumed)
    centroids = mesh.cell_centroids.to(dtype)  # (C, 3)

    cp = pressure_coeff.to(device=device, dtype=dtype).reshape(-1)  # (C,)
    cf = shear_coeff.to(device=device, dtype=dtype).reshape(cp.shape[0], -1)  # (C, 3)

    ### Surface traction coefficient t = -Cp n + Cf (force per area / q_inf).
    traction = -cp.unsqueeze(-1) * normals + cf  # (C, 3)

    ### Quadrature: Mesh.integrate sums field_c * area_c over the surface.
    force = mesh.integrate(traction, data_source="cells")  # (3,)
    arm = centroids - moment_center.to(device=device, dtype=dtype)  # (C, 3)
    moment = mesh.integrate(
        torch.linalg.cross(arm, traction), data_source="cells"
    )  # (3,)

    c_f = force / reference_area
    c_m = moment / (reference_area * reference_length)

    drag, lift, side = build_axis_frame(
        flow_direction.to(device=device, dtype=dtype),
        up_direction.to(device=device, dtype=dtype),
    )

    return {
        "CD": float((c_f @ drag).item()),
        "CL": float((c_f @ lift).item()),
        "CS": float((c_f @ side).item()),
        "CMR": float((c_m @ drag).item()),
        "CMP": float((c_m @ side).item()),
        "CMY": float((c_m @ lift).item()),
    }


def surface_force_fields(
    field_types: dict[str, NondimFieldType],
) -> tuple[str, str] | None:
    """Identify the (pressure, shear) field names for force integration.

    Returns the field names whose non-dim recipes are ``"pressure"`` and
    ``"stress"`` (i.e. :math:`C_p` and :math:`C_f`), or ``None`` if the
    dataset does not declare both -- e.g. volume runs, which carry no
    surface traction, so force integration auto-skips.
    """
    pressure = next((n for n, t in field_types.items() if t == "pressure"), None)
    shear = next((n for n, t in field_types.items() if t == "stress"), None)
    if pressure is None or shear is None:
        return None
    return pressure, shear


@dataclass(frozen=True, eq=False)
class ForceContext:
    """Resolved configuration for integrating surface force/moment coefficients.

    Built once per run via :meth:`from_config`; :meth:`coefficients`
    then integrates the predicted + reference coefficients for each
    surface sample. Auto-disables (``from_config`` returns ``None``) when
    force coefficients are turned off or the dataset has no Cp/Cf fields.
    """

    pressure_field: str
    shear_field: str
    reference_area: float
    reference_length: float | None
    moment_center: torch.Tensor
    up_direction: torch.Tensor

    @classmethod
    def from_config(
        cls,
        force_cfg: DictConfig | None,
        field_types: dict[str, NondimFieldType],
        device: torch.device | str,
    ) -> "ForceContext | None":
        """Build a context from the ``force_coefficients`` config block.

        Returns ``None`` when the block is absent, disabled, or the
        dataset declares no pressure + shear surface fields (so callers
        can simply skip force integration).
        """
        if force_cfg is None or not force_cfg.get("enabled", False):
            return None
        fields = surface_force_fields(field_types)
        if fields is None:
            return None
        pressure_field, shear_field = fields
        ref_len = force_cfg.get("reference_length", None)
        ### `list()` resolves a ListConfig (iteration resolves any
        ### interpolations) and passes a plain-list default through
        ### unchanged, so absent keys genuinely fall back to the defaults.
        return cls(
            pressure_field=pressure_field,
            shear_field=shear_field,
            reference_area=float(force_cfg.get("reference_area", 1.0)),
            reference_length=float(ref_len) if ref_len is not None else None,
            moment_center=torch.tensor(
                list(force_cfg.get("moment_center", [0.0, 0.0, 0.0])),
                dtype=torch.float32,
                device=device,
            ),
            up_direction=torch.tensor(
                list(force_cfg.get("up_direction", [0.0, 0.0, 1.0])),
                dtype=torch.float32,
                device=device,
            ),
        )

    def coefficients(
        self,
        domain: DomainMesh,
        pred_pts: TensorDict,
        true_pts: TensorDict,
        normalizer: Any | None,
    ) -> tuple[dict[str, float], dict[str, float]] | None:
        """Integrate predicted + reference coefficients for one surface sample.

        ``pred_pts`` / ``true_pts`` are per-point (training-space)
        prediction / target TensorDicts; they are un-normalized (when a
        *normalizer* is supplied) into Cp / Cf coefficient space -- always,
        independent of any physical re-dimensionalization -- before
        integration over the ``vehicle`` boundary. Returns ``None`` when
        the sample is not a surface case (no ``vehicle`` boundary, or the
        interior points do not map 1:1 to the vehicle cells).
        """
        if "vehicle" not in domain.boundary_names:
            return None
        vehicle = domain.boundaries["vehicle"]
        ### Surface contract: interior points are the vehicle cell
        ### centroids, so predictions align 1:1 with cells. A mismatch
        ### means this is not a surface case (e.g. a volume interior).
        if domain.interior.points.shape[0] != vehicle.n_cells:
            return None

        unnorm = normalizer.inverse_td if normalizer is not None else (lambda td: td)
        pred_coeff = unnorm(pred_pts.float())
        true_coeff = unnorm(true_pts.float())

        gd = domain.global_data
        length_scale = float(gd["L_ref"].item()) if "L_ref" in gd else 1.0
        ref_len = (
            self.reference_length if self.reference_length is not None else length_scale
        )
        common = dict(
            flow_direction=gd["U_inf"],
            up_direction=self.up_direction,
            moment_center=self.moment_center,
            reference_area=self.reference_area,
            reference_length=ref_len,
            length_scale=length_scale,
        )
        pred = force_moment_coefficients(
            vehicle,
            pred_coeff[self.pressure_field],
            pred_coeff[self.shear_field],
            **common,
        )
        true = force_moment_coefficients(
            vehicle,
            true_coeff[self.pressure_field],
            true_coeff[self.shear_field],
            **common,
        )
        return pred, true


class ForceAccumulator:
    """Running sums of per-sample force/moment coefficients for reporting.

    Accumulates predicted / reference means and per-coefficient MAE across
    samples. ``totals`` and ``count`` are public so an external all-reduce
    can fold them across ranks before :meth:`summary`.
    """

    def __init__(self) -> None:
        ### Pre-populate every coefficient key at zero so the key set is
        ### identical on every rank, whether or not that rank's shard
        ### contained a surface sample. ``infer._allreduce_sums`` folds
        ### ``totals`` into a single fixed-length tensor for the cross-rank
        ### all-reduce; a rank that never called ``update`` would otherwise
        ### pack a shorter tensor and deadlock (NCCL) / abort (gloo) the
        ### collective. ``count`` stays 0 on such a rank, so the reported
        ### means are unaffected.
        self.totals: dict[str, float] = {
            f"{name}_{stat}": 0.0
            for name in COEFFICIENT_NAMES
            for stat in ("pred", "true", "mae")
        }
        self.count: int = 0

    def update(self, pred: dict[str, float], true: dict[str, float]) -> None:
        """Fold one sample's predicted / reference coefficients into the sums."""
        for name in COEFFICIENT_NAMES:
            self.totals[f"{name}_pred"] += pred[name]
            self.totals[f"{name}_true"] += true[name]
            self.totals[f"{name}_mae"] += abs(pred[name] - true[name])
        self.count += 1

    def summary(self) -> tuple[list[list[str]], dict[str, dict[str, float]]]:
        """Return ``(table_rows, jsonl_dict)`` of mean pred / true / MAE per coeff."""
        denom = max(self.count, 1)
        rows: list[list[str]] = []
        summary: dict[str, dict[str, float]] = {}
        for name in COEFFICIENT_NAMES:
            mean_pred = self.totals[f"{name}_pred"] / denom
            mean_true = self.totals[f"{name}_true"] / denom
            mae = self.totals[f"{name}_mae"] / denom
            summary[name] = {"pred_mean": mean_pred, "true_mean": mean_true, "mae": mae}
            rows.append([name, f"{mean_pred:.5f}", f"{mean_true:.5f}", f"{mae:.5f}"])
        return rows, summary
