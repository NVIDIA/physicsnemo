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
from jaxtyping import Float, Int
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
        factor: float | Float[torch.Tensor, ""],
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
        """Apply uniform scaling to a :class:`DomainMesh`.

        Parameters
        ----------
        domain : DomainMesh
            Input domain mesh (interior + boundaries).

        Returns
        -------
        DomainMesh
            Scaled domain mesh.
        """
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

    def __init__(
        self, vector: Float[torch.Tensor, " spatial_dims"] | list[float]
    ) -> None:
        super().__init__()
        if not isinstance(vector, torch.Tensor):
            vector = torch.tensor(vector, dtype=torch.float32)
        self.vector = vector

    def __call__(self, mesh: Mesh) -> Mesh:
        return mesh.translate(self.vector.to(mesh.points.device))

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Apply translation to a :class:`DomainMesh`.

        Parameters
        ----------
        domain : DomainMesh
            Input domain mesh (interior + boundaries).

        Returns
        -------
        DomainMesh
            Translated domain mesh.
        """
        return domain.translate(self.vector.to(domain.interior.points.device))

    def extra_repr(self) -> str:
        return f"vector={self.vector.tolist()}"


@register()
class RotateMesh(MeshTransform):
    r"""Rotate mesh geometry (and optionally point/cell/global data) about an axis."""

    def __init__(
        self,
        angle: float,
        axis: Float[torch.Tensor, " spatial_dims"]
        | list
        | tuple
        | Literal["x", "y", "z"]
        | None = None,
        center: Float[torch.Tensor, " spatial_dims"] | list | tuple | None = None,
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
        """Apply rotation to a :class:`DomainMesh`.

        Parameters
        ----------
        domain : DomainMesh
            Input domain mesh (interior + boundaries).

        Returns
        -------
        DomainMesh
            Rotated domain mesh.
        """
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

    def _compute_com(self, mesh: Mesh) -> Float[torch.Tensor, " spatial_dims"]:
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
        """Translate a :class:`DomainMesh` so its interior center of mass is at the origin.

        The center of mass is computed from the interior mesh and the same
        translation is applied to all boundaries to keep them consistent.

        Parameters
        ----------
        domain : DomainMesh
            Input domain mesh (interior + boundaries).

        Returns
        -------
        DomainMesh
            Centered domain mesh.
        """
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
        self._generator: torch.Generator | None = None

    def _random_indices(
        self, total: int, k: int, device: torch.device
    ) -> Int[torch.Tensor, " k"]:
        if total <= k:
            return torch.arange(total, device=device)
        if total > 2**24:
            return poisson_sample_indices_fixed(
                total,
                k,
                device=device,
                generator=self._generator,
            )
        return torch.randperm(total, device=device, generator=self._generator)[:k]

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
        fields: dict[str, Float[torch.Tensor, " *shape"] | list[float]],
    ) -> None:
        super().__init__()
        self._fields: dict[str, Float[torch.Tensor, " *shape"]] = {}
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
class NormalizeMeshFields(MeshTransform):
    r"""Standardize mesh data fields with direction-preserving vector support.

    For **scalar** fields: ``(x - mean) / std``.

    For **vector** fields: ``(x - mean_vec) / std_shared`` where
    ``mean_vec`` is a per-component mean and ``std_shared`` is a single
    scalar applied uniformly to all components.  This preserves relative
    component magnitudes (and therefore vector direction) while bringing
    the overall field scale to O(1).

    Statistics may come from two sources (checked in order):

    1. **stats_file** — path to a ``.pt`` file mapping field names to
       dicts with keys ``type``, ``mean``, ``std``.
    2. **fields** — inline dict supplied directly in YAML.

    Example YAML (inline)::

        - _target_: ${dp:NormalizeMeshFields}
          section: point_data
          fields:
            pressure: {type: scalar, mean: -0.15, std: 0.45}
            wss: {type: vector, mean: [0.003, 0.0, 0.0], std: 0.005}

    Example YAML (from .pt file)::

        - _target_: ${dp:NormalizeMeshFields}
          section: point_data
          stats_file: /path/to/norm_stats.pt
    """

    def __init__(
        self,
        section: str = "point_data",
        fields: dict[str, dict] | None = None,
        stats_file: str | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self._section = section
        self._eps = eps

        if stats_file is not None:
            self._stats: dict[str, dict[str, Float[torch.Tensor, " *shape"] | str]] = (
                torch.load(stats_file, weights_only=True)
            )
        elif fields is not None:
            self._stats = {}
            for name, cfg in fields.items():
                self._stats[name] = {
                    "type": cfg["type"],
                    "mean": torch.as_tensor(cfg["mean"], dtype=torch.float32),
                    "std": torch.as_tensor(cfg["std"], dtype=torch.float32),
                }
        else:
            raise ValueError("Provide one of 'stats_file' or 'fields'")

    def __call__(self, mesh: Mesh) -> Mesh:
        td = _get_mesh_section(mesh, self._section)
        new_td = td.clone()

        for field_name, stats in self._stats.items():
            if field_name not in new_td.keys():
                continue
            val = new_td[field_name].float()
            mean = stats["mean"].to(dtype=val.dtype, device=val.device)
            std = stats["std"].to(dtype=val.dtype, device=val.device)
            new_td[field_name] = (val - mean) / (std + self._eps)

        kwargs: dict = {
            "points": mesh.points,
            "cells": mesh.cells,
            "point_data": mesh.point_data,
            "cell_data": mesh.cell_data,
            "global_data": mesh.global_data,
        }
        kwargs[self._section] = new_td
        return Mesh(**kwargs)

    def inverse_tensor(
        self,
        tensor: Float[torch.Tensor, "*batch channels"],
        target_config: dict[str, str],
        n_spatial_dims: int = 3,
    ) -> Float[torch.Tensor, "*batch channels"]:
        """Un-normalize a concatenated output tensor back to physical units.

        Fields present in ``target_config`` but absent from the stored
        normalization stats are passed through unchanged (their channels
        are skipped).  This allows partial normalization (e.g. only WSS)
        without requiring every field to have stats.

        Parameters
        ----------
        tensor : Tensor
            Shape ``(*, C)`` where channels are ordered according to
            *target_config*.
        target_config : dict[str, str]
            Ordered mapping of ``{field_name: field_type}`` matching the
            channel layout, e.g. ``{"pressure": "scalar", "wss": "vector"}``.
        n_spatial_dims : int, optional
            Dimensionality of vector fields. Default is 3.

        Returns
        -------
        Tensor
            Same shape, with each normalized field's channels un-normalized.
        """
        out = tensor.clone()
        idx = 0
        for name, ftype in target_config.items():
            dim = 1 if ftype == "scalar" else n_spatial_dims
            if name in self._stats:
                stats = self._stats[name]
                mean = stats["mean"].to(dtype=tensor.dtype, device=tensor.device)
                std = stats["std"].to(dtype=tensor.dtype, device=tensor.device)
                out[..., idx : idx + dim] = (
                    out[..., idx : idx + dim] * (std + self._eps) + mean
                )
            idx += dim
        return out

    @property
    def stats(self) -> dict:
        """Normalization statistics dict (for serialization)."""
        return self._stats

    def extra_repr(self) -> str:
        parts = []
        for name, s in self._stats.items():
            parts.append(f"{name}({s['type']}): mean={s['mean']}, std={s['std']}")
        return f"section={self._section}, " + ", ".join(parts)


@register()
class ComputeSurfaceNormals(MeshTransform):
    r"""Compute surface normal vectors and store them in point_data or cell_data.

    Uses the :class:`~physicsnemo.mesh.Mesh` built-in normal computation
    (cross product for triangles in 3D, angle-area weighted averaging for
    vertex normals).

    Place this transform **before** :class:`SubsampleMesh` so that the
    normals are subsampled along with the other fields.

    Parameters
    ----------
    store_as : {"cell_data", "point_data"}
        Where to store the computed normals.  ``"cell_data"`` stores one
        normal per cell (the face normal).  ``"point_data"`` stores one
        normal per vertex (angle-area weighted average of adjacent face
        normals).  Both modes require the mesh to have cells.
    field_name : str
        Key under which to store the normals.  Default ``"normals"``.
    """

    def __init__(
        self,
        store_as: Literal["cell_data", "point_data"] = "cell_data",
        field_name: str = "normals",
    ) -> None:
        super().__init__()
        if store_as not in ("cell_data", "point_data"):
            raise ValueError(
                f"store_as must be 'cell_data' or 'point_data', got {store_as!r}"
            )
        self.store_as = store_as
        self.field_name = field_name

    def __call__(self, mesh: Mesh) -> Mesh:
        if self.store_as == "cell_data":
            normals = mesh.cell_normals
            new_cd = mesh.cell_data.clone()
            new_cd[self.field_name] = normals
            return Mesh(
                points=mesh.points,
                cells=mesh.cells,
                point_data=mesh.point_data,
                cell_data=new_cd,
                global_data=mesh.global_data,
            )
        else:
            normals = mesh.point_normals
            new_pd = mesh.point_data.clone()
            new_pd[self.field_name] = normals
            return Mesh(
                points=mesh.points,
                cells=mesh.cells,
                point_data=new_pd,
                cell_data=mesh.cell_data,
                global_data=mesh.global_data,
            )

    def extra_repr(self) -> str:
        return f"store_as={self.store_as!r}, field_name={self.field_name!r}"


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
        """Convert a :class:`DomainMesh` into a nested :class:`TensorDict`.

        The output contains an ``"interior"`` key with the interior mesh
        converted via :func:`_mesh_to_tensordict`, an optional
        ``"boundaries"`` sub-dict keyed by boundary name, and an optional
        ``"global_data"`` entry.

        Parameters
        ----------
        domain : DomainMesh
            Input domain mesh (interior + boundaries).

        Returns
        -------
        TensorDict
            Nested TensorDict representation of the domain.
        """
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


def _resolve_td_path(td: TensorDict, dotted_key: str) -> Float[torch.Tensor, " *shape"]:
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
