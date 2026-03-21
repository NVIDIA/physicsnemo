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
Deterministic mesh transforms (Mesh -> Mesh) and terminal conversions.
"""

from __future__ import annotations

from typing import Literal

import torch
from tensordict import TensorDict

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.datapipes.transforms.subsample import poisson_sample_indices_fixed
from physicsnemo.mesh import DomainMesh, Mesh


@register()
class ScaleMesh(MeshTransform):
    r"""Scale mesh geometry (and optionally point/cell/global data) by a uniform factor."""

    def __init__(
        self,
        factor: float | torch.Tensor,
        transform_point_data: bool = False,
        transform_cell_data: bool = False,
        transform_global_data: bool = False,
    ) -> None:
        super().__init__()
        self.factor = factor
        self.transform_point_data = transform_point_data
        self.transform_cell_data = transform_cell_data
        self.transform_global_data = transform_global_data

    def __call__(self, mesh: Mesh) -> Mesh:
        return mesh.scale(
            self.factor,
            transform_point_data=self.transform_point_data,
            transform_cell_data=self.transform_cell_data,
            transform_global_data=self.transform_global_data,
        )

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        return domain.scale(
            self.factor,
            transform_point_data=self.transform_point_data,
            transform_cell_data=self.transform_cell_data,
            transform_global_data=self.transform_global_data,
        )

    def extra_repr(self) -> str:
        return f"factor={self.factor}"


@register()
class TranslateMesh(MeshTransform):
    r"""Translate mesh geometry by a vector."""

    def __init__(self, vector: torch.Tensor | list[float]) -> None:
        super().__init__()
        if not isinstance(vector, torch.Tensor):
            vector = torch.tensor(vector, dtype=torch.float32)
        self.vector = vector

    def __call__(self, mesh: Mesh) -> Mesh:
        return mesh.translate(self.vector.to(mesh.points.device))

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        return domain.translate(self.vector.to(domain.interior.points.device))

    def extra_repr(self) -> str:
        return f"vector={self.vector.tolist()}"


@register()
class RotateMesh(MeshTransform):
    r"""Rotate mesh geometry (and optionally point/cell/global data) about an axis."""

    def __init__(
        self,
        angle: float,
        axis: torch.Tensor | list | tuple | Literal["x", "y", "z"] | None = None,
        center: torch.Tensor | list | tuple | None = None,
        transform_point_data: bool = False,
        transform_cell_data: bool = False,
        transform_global_data: bool = False,
    ) -> None:
        super().__init__()
        self.angle = angle
        self.axis = axis
        self.center = center
        self.transform_point_data = transform_point_data
        self.transform_cell_data = transform_cell_data
        self.transform_global_data = transform_global_data

    def __call__(self, mesh: Mesh) -> Mesh:
        return mesh.rotate(
            self.angle,
            axis=self.axis,
            center=self.center,
            transform_point_data=self.transform_point_data,
            transform_cell_data=self.transform_cell_data,
            transform_global_data=self.transform_global_data,
        )

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        return domain.rotate(
            self.angle,
            axis=self.axis,
            center=self.center,
            transform_point_data=self.transform_point_data,
            transform_cell_data=self.transform_cell_data,
            transform_global_data=self.transform_global_data,
        )

    def extra_repr(self) -> str:
        parts = [f"angle={self.angle}"]
        if self.axis is not None:
            parts.append(f"axis={self.axis}")
        if self.center is not None:
            parts.append(f"center={self.center}")
        return ", ".join(parts)


@register()
class CenterMesh(MeshTransform):
    r"""Translate mesh so its center of mass is at the origin."""

    def __init__(self, use_area_weighting: bool = True) -> None:
        super().__init__()
        self.use_area_weighting = use_area_weighting

    def _compute_com(self, mesh: Mesh) -> torch.Tensor:
        """Compute center of mass for a single mesh."""
        if self.use_area_weighting and mesh.n_cells > 0:
            areas = mesh.cell_areas  # (n_cells,)
            centroids = mesh.cell_centroids  # (n_cells, n_spatial_dims)
            total_area = areas.sum()
            return (centroids * areas.unsqueeze(-1)).sum(dim=0) / total_area
        return mesh.points.mean(dim=0)

    def __call__(self, mesh: Mesh) -> Mesh:
        return mesh.translate(-self._compute_com(mesh))

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        com = self._compute_com(domain.interior)
        return domain.translate(-com)

    def extra_repr(self) -> str:
        return f"use_area_weighting={self.use_area_weighting}"


def _compact_points(mesh: Mesh) -> Mesh:
    """Remove unreferenced points and remap cell indices."""
    if mesh.n_cells == 0:
        return mesh
    referenced = torch.unique(mesh.cells)
    if referenced.numel() == mesh.n_points:
        return mesh
    new_points = mesh.points[referenced]
    remap = torch.empty(mesh.n_points, dtype=torch.long, device=mesh.cells.device)
    remap[referenced] = torch.arange(referenced.numel(), device=mesh.cells.device)
    new_cells = remap[mesh.cells]
    new_point_data = (
        mesh.point_data[referenced] if mesh.point_data.keys() else mesh.point_data
    )
    return Mesh(
        points=new_points,
        cells=new_cells,
        point_data=new_point_data,
        cell_data=mesh.cell_data,
        global_data=mesh.global_data,
    )


@register()
class SubsampleMesh(MeshTransform):
    r"""Subsample a mesh to a fixed number of cells and/or points."""

    def __init__(
        self,
        n_cells: int | None = None,
        n_points: int | None = None,
        compact: bool = True,
    ) -> None:
        super().__init__()
        if n_cells is None and n_points is None:
            raise ValueError("At least one of n_cells or n_points must be specified.")
        self.n_cells = n_cells
        self.n_points = n_points
        self.compact = compact

    def _random_indices(self, total: int, k: int, device: torch.device) -> torch.Tensor:
        if total <= k:
            return torch.arange(total, device=device)
        if total > 2**24:
            return poisson_sample_indices_fixed(total, k, device=device)
        return torch.randperm(total, device=device)[:k]

    def __call__(self, mesh: Mesh) -> Mesh:
        if self.n_cells is not None and mesh.n_cells > self.n_cells:
            indices = self._random_indices(
                mesh.n_cells, self.n_cells, mesh.cells.device
            )
            mesh = mesh.slice_cells(indices)
            if self.compact:
                mesh = _compact_points(mesh)

        if self.n_points is not None and mesh.n_points > self.n_points:
            indices = self._random_indices(
                mesh.n_points, self.n_points, mesh.points.device
            )
            mesh = mesh.slice_points(indices)

        return mesh

    def extra_repr(self) -> str:
        parts = []
        if self.n_cells is not None:
            parts.append(f"n_cells={self.n_cells}")
        if self.n_points is not None:
            parts.append(f"n_points={self.n_points}")
        return ", ".join(parts)


def _rename_td_keys(td: TensorDict, mapping: dict[str, str]) -> TensorDict:
    """Rename keys in a TensorDict, returning a new TensorDict."""
    out = td.clone()
    for old_key, new_key in mapping.items():
        if old_key in out.keys():
            out[new_key] = out.pop(old_key)
    return out


@register()
class DropMeshFields(MeshTransform):
    r"""Remove fields from a Mesh's point_data, cell_data, or global_data.

    Useful for dropping fields that would interfere with downstream
    transforms (e.g. removing a scalar ``TimeValue`` from ``global_data``
    before a rotation that expects all global fields to be 3-vectors).
    """

    def __init__(
        self,
        point_data: list[str] | None = None,
        cell_data: list[str] | None = None,
        global_data: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._point_data_keys = point_data or []
        self._cell_data_keys = cell_data or []
        self._global_data_keys = global_data or []

    def __call__(self, mesh: Mesh) -> Mesh:
        new_pd = mesh.point_data
        if self._point_data_keys:
            new_pd = new_pd.clone()
            for k in self._point_data_keys:
                if k in new_pd.keys():
                    del new_pd[k]

        new_cd = mesh.cell_data
        if self._cell_data_keys:
            new_cd = new_cd.clone()
            for k in self._cell_data_keys:
                if k in new_cd.keys():
                    del new_cd[k]

        new_gd = mesh.global_data
        if self._global_data_keys:
            new_gd = new_gd.clone()
            for k in self._global_data_keys:
                if k in new_gd.keys():
                    del new_gd[k]

        return Mesh(
            points=mesh.points,
            cells=mesh.cells,
            point_data=new_pd,
            cell_data=new_cd,
            global_data=new_gd,
        )

    def extra_repr(self) -> str:
        parts = []
        if self._point_data_keys:
            parts.append(f"point_data={self._point_data_keys}")
        if self._cell_data_keys:
            parts.append(f"cell_data={self._cell_data_keys}")
        if self._global_data_keys:
            parts.append(f"global_data={self._global_data_keys}")
        return ", ".join(parts)


@register()
class RenameMeshFields(MeshTransform):
    r"""Rename fields in a Mesh's point_data, cell_data, or global_data.

    Useful for harmonizing field names across datasets that store
    the same physical quantity under different keys (e.g.
    ``pMeanTrim`` vs ``pressure_average``).
    """

    def __init__(
        self,
        point_data: dict[str, str] | None = None,
        cell_data: dict[str, str] | None = None,
        global_data: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._point_data_map = point_data or {}
        self._cell_data_map = cell_data or {}
        self._global_data_map = global_data or {}

    def __call__(self, mesh: Mesh) -> Mesh:
        new_pd = (
            _rename_td_keys(mesh.point_data, self._point_data_map)
            if self._point_data_map
            else mesh.point_data
        )
        new_cd = (
            _rename_td_keys(mesh.cell_data, self._cell_data_map)
            if self._cell_data_map
            else mesh.cell_data
        )
        new_gd = (
            _rename_td_keys(mesh.global_data, self._global_data_map)
            if self._global_data_map
            else mesh.global_data
        )
        return Mesh(
            points=mesh.points,
            cells=mesh.cells,
            point_data=new_pd,
            cell_data=new_cd,
            global_data=new_gd,
        )

    def extra_repr(self) -> str:
        parts = []
        if self._point_data_map:
            parts.append(f"point_data={self._point_data_map}")
        if self._cell_data_map:
            parts.append(f"cell_data={self._cell_data_map}")
        if self._global_data_map:
            parts.append(f"global_data={self._global_data_map}")
        return ", ".join(parts)


@register()
class SetGlobalField(MeshTransform):
    r"""Inject constant tensor fields into a Mesh's global_data.

    Fields are set on every call, overwriting any existing field with
    the same key.  Tensors are moved to the mesh's device automatically.

    Typical use: inject a per-dataset inlet velocity vector so that
    downstream rotation transforms (with ``transform_global_data=True``)
    rotate it consistently with the mesh geometry.
    """

    def __init__(
        self,
        fields: dict[str, torch.Tensor | list[float]],
    ) -> None:
        super().__init__()
        self._fields: dict[str, torch.Tensor] = {}
        for k, v in fields.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, dtype=torch.float32)
            self._fields[k] = v

    def __call__(self, mesh: Mesh) -> Mesh:
        new_gd = mesh.global_data.clone()
        for k, v in self._fields.items():
            new_gd[k] = v.to(device=mesh.points.device, dtype=mesh.points.dtype)
        return Mesh(
            points=mesh.points,
            cells=mesh.cells,
            point_data=mesh.point_data,
            cell_data=mesh.cell_data,
            global_data=new_gd,
        )

    def extra_repr(self) -> str:
        shapes = {k: tuple(v.shape) for k, v in self._fields.items()}
        return f"fields={shapes}"


def _get_mesh_section(mesh: Mesh, section: str) -> TensorDict:
    """Look up a Mesh data section by name."""
    if section == "point_data":
        return mesh.point_data
    if section == "cell_data":
        return mesh.cell_data
    if section == "global_data":
        return mesh.global_data
    raise ValueError(f"Unknown mesh section: {section!r}")


@register()
class NonDimensionalizeFields(MeshTransform):
    r"""Non-dimensionalize fields by a dynamic pressure q derived per-sample.

    Computes q from the ratio of two co-located fields -- one dimensional,
    one already non-dimensional -- and divides target fields by q.  Uses
    the median ratio for robustness against surface points where the
    denominator passes through zero.

    Typical use: given ``pMeanTrim`` (Pa) and ``CpMeanTrim`` (dimensionless)
    in ``point_data``, compute ``q = median(pMeanTrim / CpMeanTrim)`` and
    divide ``wallShearStressMeanTrim`` by q to obtain the skin-friction
    coefficient Cf.
    """

    def __init__(
        self,
        dimensional_field: str,
        nondimensional_field: str,
        section: str = "point_data",
        target_fields: list[str] | None = None,
        target_section: str | None = None,
        min_denominator: float = 0.01,
    ) -> None:
        super().__init__()
        self._dim_field = dimensional_field
        self._nondim_field = nondimensional_field
        self._section = section
        self._target_fields = target_fields or []
        self._target_section = target_section or section
        self._min_denom = min_denominator

    def __call__(self, mesh: Mesh) -> Mesh:
        ref_td = _get_mesh_section(mesh, self._section)
        dim_vals = ref_td[self._dim_field].float().reshape(-1)
        nondim_vals = ref_td[self._nondim_field].float().reshape(-1)

        mask = nondim_vals.abs() > self._min_denom
        q = (dim_vals[mask] / nondim_vals[mask]).median()

        target_td = _get_mesh_section(mesh, self._target_section)
        new_td = target_td.clone()
        for field_name in self._target_fields:
            new_td[field_name] = new_td[field_name].float() / q

        kwargs: dict = {
            "points": mesh.points,
            "cells": mesh.cells,
            "point_data": mesh.point_data,
            "cell_data": mesh.cell_data,
            "global_data": mesh.global_data,
        }
        kwargs[self._target_section] = new_td
        return Mesh(**kwargs)

    def extra_repr(self) -> str:
        targets = [f"{self._target_section}.{f}" for f in self._target_fields]
        return (
            f"q={self._section}.{self._dim_field}"
            f"/{self._section}.{self._nondim_field}, "
            f"targets={targets}"
        )


def _mesh_to_tensordict(mesh: Mesh) -> TensorDict:
    """Convert a single Mesh into a flat TensorDict (no cache, no tensorclass)."""
    out: dict = {
        "points": mesh.points,
        "cells": mesh.cells,
    }
    if mesh.point_data.keys():
        out["point_data"] = mesh.point_data.clone()
    if mesh.cell_data.keys():
        out["cell_data"] = mesh.cell_data.clone()
    if mesh.global_data.keys():
        out["global_data"] = mesh.global_data.clone()
    return TensorDict(out, batch_size=[])


@register()
class MeshToTensorDict(MeshTransform):
    r"""Convert a Mesh or DomainMesh into a plain TensorDict.

    This is a terminal transform -- place it last in the transform chain.
    After conversion the data is no longer a Mesh and cannot be passed to
    other MeshTransform instances.

    For a single :class:`Mesh` the output layout is::

        TensorDict({
            "points":     (N_p, D_s),
            "cells":      (N_c, D_m+1),
            "point_data": TensorDict({field: tensor, ...}),
            "cell_data":  TensorDict({field: tensor, ...}),
            "global_data": TensorDict({field: tensor, ...}),
        })

    For a :class:`DomainMesh` the output layout is::

        TensorDict({
            "interior":   TensorDict({points, cells, ...}),
            "boundaries": TensorDict({
                "wall":  TensorDict({points, cells, ...}),
                ...
            }),
            "global_data": TensorDict({field: tensor, ...}),
        })
    """

    def __call__(self, mesh: Mesh) -> TensorDict:  # type: ignore[override]
        return _mesh_to_tensordict(mesh)

    def apply_to_domain(self, domain: DomainMesh) -> TensorDict:  # type: ignore[override]
        out: dict = {
            "interior": _mesh_to_tensordict(domain.interior),
        }
        if domain.n_boundaries > 0:
            out["boundaries"] = TensorDict(
                {
                    name: _mesh_to_tensordict(domain.boundaries[name])
                    for name in domain.boundary_names
                },
                batch_size=[],
            )
        if domain.global_data.keys():
            out["global_data"] = domain.global_data.clone()
        return TensorDict(out, batch_size=[])


def _resolve_td_path(td: TensorDict, dotted_key: str) -> torch.Tensor:
    """Resolve a dot-separated key path into a tensor from a TensorDict."""
    parts = dotted_key.split(".")
    current = td
    for part in parts:
        current = current[part]
    return current


@register()
class ComputeCellCentroids(MeshTransform):
    r"""Compute cell centroids from points and cells in a TensorDict.

    Placed after :class:`MeshToTensorDict`, this adds a ``cell_centroids``
    key of shape :math:`(N_c, D_s)` computed as the mean of each cell's
    vertex positions.  Requires ``points`` and ``cells`` to be present.
    """

    def __call__(self, td: TensorDict) -> TensorDict:  # type: ignore[override]
        points = td["points"]
        cells = td["cells"]
        centroids = points[cells].mean(dim=1)
        td = td.clone()
        td["cell_centroids"] = centroids
        return td


@register()
class RestructureTensorDict(MeshTransform):
    r"""Reorganize a flat TensorDict into named groups.

    Placed after :class:`MeshToTensorDict`, this transform picks fields
    from the flat layout and assembles them into a structured dict
    (e.g. separate ``input`` and ``output`` groups for model training).

    Each group is defined as ``{dest_key: source_path}`` where
    ``source_path`` uses dots for nesting (e.g. ``point_data.pressure``).

    Example YAML::

        - _target_: ${dp:RestructureTensorDict}
          groups:
            input:
              points: points
              inlet_velocity: global_data.inlet_velocity
            output:
              pressure: point_data.pressure
              wss: point_data.wss
    """

    def __init__(self, groups: dict[str, dict[str, str]]) -> None:
        super().__init__()
        self._groups = groups

    def __call__(self, td: TensorDict) -> TensorDict:  # type: ignore[override]
        out: dict = {}
        for group_name, mapping in self._groups.items():
            group: dict = {}
            for dest_key, source_path in mapping.items():
                group[dest_key] = _resolve_td_path(td, source_path)
            out[group_name] = TensorDict(group, batch_size=[])
        return TensorDict(out, batch_size=[])

    def extra_repr(self) -> str:
        lines = []
        for group, mapping in self._groups.items():
            sources = ", ".join(f"{k}<-{v}" for k, v in mapping.items())
            lines.append(f"{group}: {{{sources}}}")
        return "; ".join(lines)
