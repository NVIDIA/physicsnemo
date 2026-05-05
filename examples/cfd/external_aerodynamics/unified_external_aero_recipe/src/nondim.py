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
Physics-based non-dimensionalization transform.

Recipe-local transform registered into the global datapipe component
registry so it can be referenced via ``${dp:NonDimensionalizeByMetadata}``
in Hydra YAML configs.

Import this module before Hydra instantiation to register the transform.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.mesh import DomainMesh, Mesh


def _get_mesh_section(mesh: Mesh, section: str) -> TensorDict:
    """Look up a Mesh data section by name."""
    if section == "point_data":
        return mesh.point_data
    if section == "cell_data":
        return mesh.cell_data
    if section == "global_data":
        return mesh.global_data
    raise ValueError(f"Unknown mesh section: {section!r}")


def _freestream_scales(
    global_data: TensorDict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Derive reference scales from freestream metadata (cast to float32 once).

    Returns ``(q_inf, p_inf, U_inf_mag, rho_inf, T_inf)`` where
    ``q_inf = 0.5 * rho_inf * |U_inf|^2``.  ``T_inf`` is ``None``
    when the metadata does not contain a freestream temperature (e.g.
    incompressible datasets).
    """
    U_inf = global_data["U_inf"].float()
    rho_inf = global_data["rho_inf"].float()
    p_inf = global_data["p_inf"].float()
    U_inf_mag_sq = (U_inf * U_inf).sum()
    q_inf = 0.5 * rho_inf * U_inf_mag_sq
    U_inf_mag = U_inf_mag_sq.sqrt()
    T_inf = global_data["T_inf"].float() if "T_inf" in global_data else None
    return q_inf, p_inf, U_inf_mag, rho_inf, T_inf


_FIELD_TYPES = frozenset(
    {"pressure", "stress", "velocity", "temperature", "density", "identity"}
)

# Number of tensor channels each field type occupies.
_FIELD_CHANNELS = {
    "pressure": 1,
    "stress": 3,
    "velocity": 3,
    "temperature": 1,
    "density": 1,
    "identity": 1,
}


def _nondim_field(
    val: torch.Tensor,
    ftype: str,
    q_inf: torch.Tensor,
    p_inf: torch.Tensor,
    U_inf_mag: torch.Tensor,
    *,
    rho_inf: torch.Tensor | None = None,
    T_inf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply forward non-dimensionalization to a single field."""
    if ftype == "identity":
        return val
    if ftype == "pressure":
        return (val - p_inf) / q_inf
    if ftype == "stress":
        return val / q_inf
    if ftype == "velocity":
        return val / U_inf_mag
    if ftype == "temperature":
        if T_inf is None:
            raise ValueError("T_inf required for temperature non-dimensionalization")
        return val / T_inf
    if ftype == "density":
        if rho_inf is None:
            raise ValueError("rho_inf required for density non-dimensionalization")
        return val / rho_inf
    raise ValueError(f"Unknown field type: {ftype!r}")


def _redim_field(
    val: torch.Tensor,
    ftype: str,
    q_inf: torch.Tensor,
    p_inf: torch.Tensor,
    U_inf_mag: torch.Tensor,
    *,
    rho_inf: torch.Tensor | None = None,
    T_inf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reverse non-dimensionalization for a single field."""
    if ftype == "identity":
        return val
    if ftype == "pressure":
        return val * q_inf + p_inf
    if ftype == "stress":
        return val * q_inf
    if ftype == "velocity":
        return val * U_inf_mag
    if ftype == "temperature":
        if T_inf is None:
            raise ValueError("T_inf required for temperature re-dimensionalization")
        return val * T_inf
    if ftype == "density":
        if rho_inf is None:
            raise ValueError("rho_inf required for density re-dimensionalization")
        return val * rho_inf
    raise ValueError(f"Unknown field type: {ftype!r}")


@register()
class NonDimensionalizeByMetadata(MeshTransform):
    r"""Non-dimensionalize fields and geometry using freestream conditions from ``global_data``.

    Expects ``U_inf``, ``rho_inf``, and ``p_inf`` to be present in
    ``global_data`` (injected by the dataset builder).  Computes
    the dynamic pressure ``q_inf = 0.5 * rho_inf * |U_inf|^2`` and
    applies standard non-dimensionalization formulas:

    - **pressure**: ``(p - p_inf) / q_inf`` (pressure coefficient Cp)
    - **stress**: ``tau / q_inf`` (skin-friction coefficient Cf)
    - **velocity**: ``U / |U_inf|``
    - **temperature**: ``T / T_inf`` (requires ``T_inf`` in ``global_data``)
    - **density**: ``rho / rho_inf``
    - **identity**: pass-through (no scaling applied)

    If ``L_ref`` is present in ``global_data``, mesh points are divided
    by it to produce non-dimensional coordinates: ``x* = x / L_ref``.
    This normalises point clouds and cell centroids computed downstream.

    Parameters
    ----------
    fields : dict[str, str]
        Mapping of ``{field_name: field_type}`` where *field_type* is one
        of ``"pressure"``, ``"stress"``, ``"velocity"``, ``"temperature"``,
        ``"density"``, or ``"identity"``.
    section : str
        Mesh data section containing the fields (``"point_data"`` or
        ``"cell_data"``).

    Example YAML::

        - _target_: ${dp:NonDimensionalizeByMetadata}
          fields:
            pMeanTrim: pressure
            wallShearStressMeanTrim: stress
          section: point_data
    """

    def __init__(
        self,
        fields: dict[str, str],
        section: str = "point_data",
    ) -> None:
        super().__init__()
        for name, ftype in fields.items():
            if ftype not in _FIELD_TYPES:
                raise ValueError(
                    f"Unknown field type {ftype!r} for {name!r}. "
                    f"Must be one of {sorted(_FIELD_TYPES)}."
                )
        self._fields = fields
        self._section = section

    def _transform_mesh(
        self,
        mesh: Mesh,
        field_fn,
        *,
        inverse: bool,
        scales: tuple | None = None,
        skip_missing: bool = False,
    ) -> Mesh:
        """Shared implementation for forward and inverse mesh transforms.

        Parameters
        ----------
        scales : tuple or None
            Pre-computed ``(q_inf, p_inf, U_inf_mag, rho_inf, T_inf, L_ref)``
            to use instead of deriving them from ``mesh.global_data``.
        skip_missing : bool
            If *True*, silently skip fields not present in the mesh section.
        """
        if scales is not None:
            q_inf, p_inf, U_inf_mag, rho_inf, T_inf, L_ref = scales
        else:
            gd = mesh.global_data
            q_inf, p_inf, U_inf_mag, rho_inf, T_inf = _freestream_scales(gd)
            L_ref = gd["L_ref"].float() if "L_ref" in gd else None

        td = _get_mesh_section(mesh, self._section)
        new_td = td.clone()

        for field_name, ftype in self._fields.items():
            if skip_missing and field_name not in new_td.keys():
                continue
            val = new_td[field_name].float()
            new_td[field_name] = field_fn(
                val,
                ftype,
                q_inf,
                p_inf,
                U_inf_mag,
                rho_inf=rho_inf,
                T_inf=T_inf,
            )

        points = mesh.points
        if L_ref is not None:
            points = points * L_ref if inverse else points / L_ref

        kwargs: dict = {
            "points": points,
            "cells": mesh.cells,
            "point_data": mesh.point_data,
            "cell_data": mesh.cell_data,
            "global_data": mesh.global_data,
        }
        kwargs[self._section] = new_td
        return Mesh(**kwargs)

    def __call__(self, mesh: Mesh) -> Mesh:
        return self._transform_mesh(mesh, _nondim_field, inverse=False)

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Non-dimensionalize a DomainMesh using domain-level ``global_data``.

        Freestream scales are read once from ``domain.global_data``
        (where the metadata injector placed them) and applied to the
        interior and every boundary mesh.  Fields that are not present
        on a particular sub-mesh (e.g. volume fields on a surface
        boundary) are silently skipped.
        """
        gd = domain.global_data
        q_inf, p_inf, U_inf_mag, rho_inf, T_inf = _freestream_scales(gd)
        L_ref = gd["L_ref"].float() if "L_ref" in gd else None
        scales = (q_inf, p_inf, U_inf_mag, rho_inf, T_inf, L_ref)

        return domain.apply_to_meshes(
            lambda m: self._transform_mesh(
                m,
                _nondim_field,
                inverse=False,
                scales=scales,
                skip_missing=True,
            )
        )

    def inverse(self, mesh: Mesh) -> Mesh:
        """Re-dimensionalize: reverse the non-dimensionalization.

        Uses the same ``global_data`` metadata (``U_inf``, ``rho_inf``,
        ``p_inf``, and optionally ``L_ref``) to convert non-dimensional
        fields and geometry back to physical units.

        Parameters
        ----------
        mesh : Mesh
            Mesh with non-dimensionalized fields and metadata in ``global_data``.

        Returns
        -------
        Mesh
            Mesh with re-dimensionalized fields.
        """
        return self._transform_mesh(mesh, _redim_field, inverse=True)

    def inverse_tensor(
        self,
        tensor: torch.Tensor,
        field_types: dict[str, str],
        q_inf: torch.Tensor,
        p_inf: torch.Tensor,
        U_inf_mag: torch.Tensor,
        *,
        rho_inf: torch.Tensor | None = None,
        T_inf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Re-dimensionalize a concatenated output tensor.

        Operates on model output tensors (shape ``(*, C)``) where channels
        are ordered according to *field_types*.  This is useful at inference
        time when you have a raw model prediction rather than a Mesh.

        Parameters
        ----------
        tensor : Tensor
            Shape ``(*, C)`` with channels ordered by *field_types*.
        field_types : dict[str, str]
            Ordered mapping of ``{field_name: nondim_type}`` where
            *nondim_type* is one of ``"pressure"``, ``"stress"``,
            ``"velocity"``, ``"temperature"``, ``"density"``, or
            ``"identity"``.
            Uses the model's output field names (e.g. after renaming),
            not the original mesh field names.
        q_inf, p_inf, U_inf_mag : Tensor
            Reference quantities (scalars or broadcastable).
        rho_inf : Tensor or None
            Freestream density.  Required when *field_types* contains
            ``"density"``.
        T_inf : Tensor or None
            Freestream temperature.  Required when *field_types* contains
            ``"temperature"``.

        Returns
        -------
        Tensor
            Same shape, with each field's channels re-dimensionalized.
        """
        out = tensor.clone()
        idx = 0
        for name, ftype in field_types.items():
            n = _FIELD_CHANNELS[ftype]
            out[..., idx : idx + n] = _redim_field(
                out[..., idx : idx + n],
                ftype,
                q_inf,
                p_inf,
                U_inf_mag,
                rho_inf=rho_inf,
                T_inf=T_inf,
            )
            idx += n
        return out

    def extra_repr(self) -> str:
        return f"fields={self._fields}, section={self._section}"
