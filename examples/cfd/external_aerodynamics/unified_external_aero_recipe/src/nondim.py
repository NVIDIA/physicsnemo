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
from physicsnemo.mesh import Mesh


def _get_mesh_section(mesh: Mesh, section: str) -> TensorDict:
    """Look up a Mesh data section by name."""
    if section == "point_data":
        return mesh.point_data
    if section == "cell_data":
        return mesh.cell_data
    if section == "global_data":
        return mesh.global_data
    raise ValueError(f"Unknown mesh section: {section!r}")


def _compute_q_inf(global_data: TensorDict) -> torch.Tensor:
    """Compute dynamic pressure q_inf = 0.5 * rho_inf * |U_inf|^2."""
    U_inf = global_data["U_inf"].float()
    rho_inf = global_data["rho_inf"].float()
    U_inf_mag_sq = (U_inf * U_inf).sum()
    return 0.5 * rho_inf * U_inf_mag_sq


_FIELD_TYPES = frozenset({"pressure", "stress", "velocity"})


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

    If ``L_ref`` is present in ``global_data``, mesh points are divided
    by it to produce non-dimensional coordinates: ``x* = x / L_ref``.
    This normalises point clouds and cell centroids computed downstream.

    Parameters
    ----------
    fields : dict[str, str]
        Mapping of ``{field_name: field_type}`` where *field_type* is one
        of ``"pressure"``, ``"stress"``, or ``"velocity"``.
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

    def __call__(self, mesh: Mesh) -> Mesh:
        gd = mesh.global_data
        q_inf = _compute_q_inf(gd)
        p_inf = gd["p_inf"].float()
        U_inf = gd["U_inf"].float()
        U_inf_mag = (U_inf * U_inf).sum().sqrt()

        td = _get_mesh_section(mesh, self._section)
        new_td = td.clone()

        for field_name, ftype in self._fields.items():
            val = new_td[field_name].float()
            if ftype == "pressure":
                new_td[field_name] = (val - p_inf) / q_inf
            elif ftype == "stress":
                new_td[field_name] = val / q_inf
            elif ftype == "velocity":
                new_td[field_name] = val / U_inf_mag

        points = mesh.points
        if "L_ref" in gd:
            points = points / gd["L_ref"].float()

        kwargs: dict = {
            "points": points,
            "cells": mesh.cells,
            "point_data": mesh.point_data,
            "cell_data": mesh.cell_data,
            "global_data": mesh.global_data,
        }
        kwargs[self._section] = new_td
        return Mesh(**kwargs)

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
        gd = mesh.global_data
        q_inf = _compute_q_inf(gd)
        p_inf = gd["p_inf"].float()
        U_inf = gd["U_inf"].float()
        U_inf_mag = (U_inf * U_inf).sum().sqrt()

        td = _get_mesh_section(mesh, self._section)
        new_td = td.clone()

        for field_name, ftype in self._fields.items():
            val = new_td[field_name].float()
            if ftype == "pressure":
                new_td[field_name] = val * q_inf + p_inf
            elif ftype == "stress":
                new_td[field_name] = val * q_inf
            elif ftype == "velocity":
                new_td[field_name] = val * U_inf_mag

        points = mesh.points
        if "L_ref" in gd:
            points = points * gd["L_ref"].float()

        kwargs: dict = {
            "points": points,
            "cells": mesh.cells,
            "point_data": mesh.point_data,
            "cell_data": mesh.cell_data,
            "global_data": mesh.global_data,
        }
        kwargs[self._section] = new_td
        return Mesh(**kwargs)

    def inverse_tensor(
        self,
        tensor: torch.Tensor,
        field_types: dict[str, str],
        q_inf: torch.Tensor,
        p_inf: torch.Tensor,
        U_inf_mag: torch.Tensor,
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
            *nondim_type* is one of ``"pressure"``, ``"stress"``, or
            ``"velocity"``.  Uses the model's output field names (e.g.
            after renaming), not the original mesh field names.
        q_inf, p_inf, U_inf_mag : Tensor
            Reference quantities (scalars or broadcastable).

        Returns
        -------
        Tensor
            Same shape, with each field's channels re-dimensionalized.
        """
        out = tensor.clone()
        idx = 0
        for name, ftype in field_types.items():
            if ftype == "pressure":
                out[..., idx] = out[..., idx] * q_inf + p_inf
                idx += 1
            elif ftype == "stress":
                out[..., idx : idx + 3] = out[..., idx : idx + 3] * q_inf
                idx += 3
            elif ftype == "velocity":
                out[..., idx : idx + 3] = out[..., idx : idx + 3] * U_inf_mag
                idx += 3
        return out

    def extra_repr(self) -> str:
        return f"fields={self._fields}, section={self._section}"
