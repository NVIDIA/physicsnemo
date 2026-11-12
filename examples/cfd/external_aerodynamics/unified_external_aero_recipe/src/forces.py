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
    \\mathbf{t} = -C_p\\,\\mathbf{n} + s\\,\\mathbf{C}_f

where :math:`\\mathbf{n}` is the unit normal that points out of the body,
and :math:`s` is :math:`+1` when the dataset's wall shear stress is the
stress the fluid exerts on the body and :math:`-1` when it is the stress
the wall exerts on the fluid. Stored surfaces differ in both respects (see
"Normal orientation" and "Wall shear side" below). The force and
moment coefficient vectors are

.. math::
    \\mathbf{C}_F = \\frac{1}{A_\\text{ref}} \\int_S \\mathbf{t}\\,dA,
    \\qquad
    \\mathbf{C}_M = \\frac{1}{A_\\text{ref} L_\\text{ref}}
        \\int_S (\\mathbf{x} - \\mathbf{x}_\\text{ref}) \\times \\mathbf{t}\\,dA .

The surface integral is evaluated with the mesh quadrature utility
:meth:`physicsnemo.mesh.Mesh.integrate` (cell-data / P0 rule:
:math:`\\int_S f\\,dA = \\sum_c f_c\\,\\mu_c`, where :math:`\\mu_c` is the
complete effective measure returned by ``cell_measures(mesh)``).

The coefficient vectors are then projected onto an orthonormal
(drag, lift, side) triad built from the per-sample freestream direction
and a global "up" reference, yielding the conventional scalars:

- ``CD`` (drag, along the freestream), ``CL`` (lift), ``CS`` (side force).
- ``CMR`` / ``CMP`` / ``CMY`` (roll / pitch / yaw moment about the
  drag / side / lift axes).

Conventions and assumptions:

- **Normal orientation.** ``Mesh.cell_normals`` follow the cell winding,
  and stored surfaces are wound either way. A dataset YAML declares where
  its surface's normals point with the top-level key
  ``surface_normals_point``: ``outward`` (the default) or ``inward``.
  With the wrong orientation the pressure force has the wrong sign. The
  surface must be wound consistently.
- **Wall shear side.** Solvers do not agree on which side the wall shear
  stress acts on. OpenFOAM's ``wallShearStress`` is the stress the wall
  exerts on the fluid, so the friction force on the body is its negative.
  Other solvers write the stress the fluid exerts on the body. A dataset
  YAML declares its convention with the top-level key
  ``wall_shear_acts_on``: ``body`` (the default) or ``fluid``. With the
  wrong side, the friction force is subtracted where it must be added.
- **Checking a dataset.** A wrong declaration gives wrong absolute
  coefficients, and the predicted-vs-reference error does not reveal it,
  because both are integrated the same way. To find a dataset's
  conventions, call :func:`force_moment_coefficients` on one stored case
  with the defaults, once with zero shear and once with zero pressure.
  With zero shear, ``CD`` must be positive for a car, and ``CL`` for a
  wing at a positive angle of attack; if it is negative, the normals
  point ``inward``. With zero pressure, ``CD`` must be positive, because
  the friction force on a body points downstream; if it is negative, the
  wall shear stress acts on the ``fluid``.
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
- **Subsampled surfaces.** Cell subsampling, by the reader's
  ``subsample_n_cells`` or by ``SubsampleMesh``, multiplies each retained
  cell's effective measure by ``n_before / n_kept``. ``Mesh.integrate`` uses
  that complete measure to compensate for the retained-area shrinkage. This
  gives an unbiased Horvitz--Thompson estimate when every cell has inclusion
  probability ``n_kept / n_before`` and its field value and physical moment
  reference are fixed independently of the sample.
  The large-population ``poisson_gap`` sampler is approximate, so this
  guarantee does not apply to every ``SubsampleMesh`` path. Predictions
  that depend on the sampled geometry can introduce additional bias.
  Likewise, centering after reader-level subsampling changes the physical
  moment origin between samples; reweighting cannot correct that frame change.
  The reader keeps one contiguous (cyclic) block of cells in storage order.
  Storage order is often spatially coherent, so reader-subsampled estimates
  can be much noisier than those from ``SubsampleMesh``'s uniform random
  subset.
  Subsampled coefficients can have both sampling noise and bias and should be
  checked for convergence with surface resolution.
  ``ForceContext.coefficients``'s 1:1 points/cells contract check cannot
  tell an exact full-surface integral from such an estimate; ``infer.py``
  warns when a vehicle's cell count sits at the ``sampling_resolution``
  cap.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import torch
from forward_kwargs import walk_path
from jaxtyping import Float
from nondim import NondimFieldType
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from utils import by_source_fields

from physicsnemo.datapipes.keys import as_nested_key
from physicsnemo.mesh import DomainMesh, Mesh

### Coefficient keys returned by :func:`force_moment_coefficients`, in a
### stable order so callers can build tables / accumulators against them.
COEFFICIENT_NAMES: tuple[str, ...] = ("CD", "CL", "CS", "CMR", "CMP", "CMY")

### Conventions a dataset YAML declares for its stored surface (see the
### module docstring): where the cell normals point, and which side the wall
### shear stress acts on.
NormalsOrientation = Literal["outward", "inward"]
WallShearSide = Literal["body", "fluid"]


def _convention_sign(value: str, *, positive: str, negative: str, key: str) -> float:
    """Return ``+1.0`` or ``-1.0`` for a declared convention, or raise."""
    if value == positive:
        return 1.0
    if value == negative:
        return -1.0
    raise ValueError(f"{key} must be {positive!r} or {negative!r}, got {value!r}.")


def normal_sign(surface_normals_point: str) -> float:
    """Sign that turns the stored cell normals into outward normals.

    Args:
        surface_normals_point: ``"outward"`` or ``"inward"``.

    Returns:
        ``+1.0`` for ``"outward"`` and ``-1.0`` for ``"inward"``.

    Raises:
        ValueError: If ``surface_normals_point`` is neither value.
    """
    return _convention_sign(
        surface_normals_point,
        positive="outward",
        negative="inward",
        key="surface_normals_point",
    )


def friction_sign(wall_shear_acts_on: str) -> float:
    """Sign of the wall shear stress in the traction on the body.

    Args:
        wall_shear_acts_on: ``"body"`` if the wall shear stress is the stress
            the fluid exerts on the body, ``"fluid"`` if it is the stress the
            wall exerts on the fluid (OpenFOAM's ``wallShearStress``).

    Returns:
        ``+1.0`` for ``"body"`` and ``-1.0`` for ``"fluid"``.

    Raises:
        ValueError: If ``wall_shear_acts_on`` is neither value.
    """
    return _convention_sign(
        wall_shear_acts_on, positive="body", negative="fluid", key="wall_shear_acts_on"
    )


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
    surface_normals_point: NormalsOrientation = "outward",
    wall_shear_acts_on: WallShearSide = "body",
) -> dict[str, float]:
    r"""Integrate per-cell surface coefficients into force/moment scalars.

    Args:
        vehicle: Triangulated surface mesh (codimension-1) carrying the
            body. Its cells define the quadrature; ``cell_normals`` and
            the complete effective cell measures (including sampling corrections)
            are taken from this mesh.
        pressure_coeff: Per-cell pressure coefficient :math:`C_p`, shape
            ``(n_cells,)`` (a trailing singleton dim, e.g. ``(n_cells, 1)``,
            is flattened internally). Must align 1:1 with ``vehicle`` cells.
        shear_coeff: Per-cell skin-friction coefficient vector
            :math:`C_f`, shape ``(n_cells, 3)``, with the dataset's sign.
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
        surface_normals_point: Where the cell normals of ``vehicle`` point:
            ``"outward"`` (out of the body) or ``"inward"``.
        wall_shear_acts_on: Which side ``shear_coeff`` acts on: ``"body"``
            (the friction force on the body is :math:`+C_f`) or ``"fluid"``
            (it is :math:`-C_f`, as for OpenFOAM's ``wallShearStress``).

    Returns:
        ``{name: float}`` for each of :data:`COEFFICIENT_NAMES`
        (``CD``, ``CL``, ``CS``, ``CMR``, ``CMP``, ``CMY``).
    """
    mesh = vehicle.scale(length_scale) if length_scale != 1.0 else vehicle
    device = mesh.points.device
    dtype = torch.float32

    ### (C, 3) unit normals, pointing out of the body.
    normals = normal_sign(surface_normals_point) * mesh.cell_normals.to(dtype)
    centroids = mesh.cell_centroids.to(dtype)  # (C, 3)

    cp = pressure_coeff.to(device=device, dtype=dtype).reshape(-1)  # (C,)
    cf = shear_coeff.to(device=device, dtype=dtype).reshape(cp.shape[0], -1)  # (C, 3)

    ### Surface traction coefficient on the body, t = -Cp n +/- Cf (force per
    ### area / q_inf); the friction sign follows the dataset's convention.
    traction = -cp.unsqueeze(-1) * normals + friction_sign(wall_shear_acts_on) * cf

    ### Quadrature: Mesh.integrate sums field_c * effective_measure_c.
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

    ### Batched D2H: one .tolist() for all six coefficients.
    coeffs = torch.stack(
        [c_f @ drag, c_f @ lift, c_f @ side, c_m @ drag, c_m @ side, c_m @ lift]
    ).tolist()
    return dict(zip(("CD", "CL", "CS", "CMR", "CMP", "CMY"), coeffs))


def surface_force_fields(
    field_types: dict[str, NondimFieldType],
    target_sources: dict[str, str] | None = None,
) -> tuple[str, str] | None:
    """Identify the (pressure, shear) field names for force integration.

    Returns the field names whose non-dim recipes are ``"pressure"`` and
    ``"stress"`` (i.e. :math:`C_p` and :math:`C_f`), or ``None`` if the
    dataset does not declare both -- e.g. volume runs, which carry no
    surface traction, so force integration auto-skips. With
    *target_sources*, the first pair of target names whose source fields
    have those recipes and share a source mesh.
    """
    if target_sources is not None:
        split = {n: p.split(".point_data.", 1) for n, p in target_sources.items()}
        kind = {n: field_types.get(leaf) for n, (_, leaf) in split.items()}
        pairs = (
            (p, s)
            for p, (mesh_p, _) in split.items()
            for s, (mesh_s, _) in split.items()
            if kind[p] == "pressure" and kind[s] == "stress" and mesh_p == mesh_s
        )
        return next(pairs, None)
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
    surface_normals_point: NormalsOrientation = "outward"
    wall_shear_acts_on: WallShearSide = "body"

    def __post_init__(self) -> None:
        ### Fail at setup, not on the first sample, for a misspelled value.
        normal_sign(self.surface_normals_point)
        friction_sign(self.wall_shear_acts_on)

    @classmethod
    def from_config(
        cls,
        force_cfg: DictConfig | None,
        field_types: dict[str, NondimFieldType],
        device: torch.device | str,
        dataset_cfg: DictConfig | None = None,
        target_sources: dict[str, str] | None = None,
    ) -> "ForceContext | None":
        """Build a context from the ``force_coefficients`` config block.

        ``dataset_cfg`` is the dataset YAML. Its top-level keys
        ``surface_normals_point`` and ``wall_shear_acts_on`` declare the
        conventions of the stored surface (see the module docstring). A
        missing key, or no dataset config, means ``outward`` and ``body``.
        ``target_sources`` picks fields on one mesh (see
        :func:`surface_force_fields`).

        Returns ``None`` when the block is absent, disabled, or the
        dataset declares no pressure + shear surface fields (so callers
        can simply skip force integration).
        """
        if force_cfg is None or not force_cfg.get("enabled", False):
            return None
        fields = surface_force_fields(field_types, target_sources)
        if fields is None:
            return None
        pressure_field, shear_field = fields
        conventions = {
            key: default
            if dataset_cfg is None
            else OmegaConf.select(dataset_cfg, key, default=default)
            for key, default in (
                ("surface_normals_point", "outward"),
                ("wall_shear_acts_on", "body"),
            )
        }
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
            **conventions,
        )

    def coefficients(
        self,
        domain: DomainMesh,
        pred_pts: TensorDict,
        true_pts: TensorDict,
        normalizer: Any | None,
        target_sources: dict[str, str] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]] | None:
        """Integrate predicted + reference coefficients for one surface sample.

        ``pred_pts`` / ``true_pts`` are per-point (training-space)
        prediction / target TensorDicts; they are un-normalized (when a
        *normalizer* is supplied) into Cp / Cf coefficient space -- always,
        independent of any physical re-dimensionalization -- before
        integration over the ``vehicle`` boundary. Returns ``None`` when
        the sample is not a surface case (no ``vehicle`` boundary, or the
        interior points do not map 1:1 to the vehicle cells). With
        *target_sources*, the pressure and shear targets are integrated on
        the point set that holds them both, which must then sit at the
        vehicle cell centroids (else this warns once and returns ``None``).
        """
        if "vehicle" not in domain.boundary_names:
            return None
        vehicle = domain.boundaries["vehicle"]
        sources = target_sources or {}
        paths = {
            sources.get(n, "interior").split(".point_data.")[0]
            for n in (self.pressure_field, self.shear_field)
        }
        if len(paths) != 1:
            return None
        surface = walk_path(domain, paths.pop())
        ### Surface contract: the prediction points are the vehicle cell
        ### centroids, so predictions align 1:1 with cells. A mismatch
        ### means this is not a surface case (e.g. a volume interior).
        if surface.n_points != vehicle.n_cells:
            return None
        if target_sources is not None and not torch.allclose(
            surface.points, vehicle.cell_centroids
        ):
            warnings.warn(
                "Surface targets are not aligned with vehicle cells; "
                "skipping force integration",
                stacklevel=2,
            )
            return None

        unnorm = normalizer.inverse_td if normalizer is not None else (lambda td: td)
        pred_coeff = by_source_fields(pred_pts.float(), target_sources, unnorm)
        true_coeff = by_source_fields(true_pts.float(), target_sources, unnorm)

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
            surface_normals_point=self.surface_normals_point,
            wall_shear_acts_on=self.wall_shear_acts_on,
        )
        ### Field names come from the dataset YAML and may spell nested
        ### leaves ("solution.pressure"); resolve them to TensorDict keys.
        pressure_key = as_nested_key(self.pressure_field)
        shear_key = as_nested_key(self.shear_field)
        pred = force_moment_coefficients(
            vehicle,
            pred_coeff[pressure_key],
            pred_coeff[shear_key],
            **common,
        )
        true = force_moment_coefficients(
            vehicle,
            true_coeff[pressure_key],
            true_coeff[shear_key],
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
