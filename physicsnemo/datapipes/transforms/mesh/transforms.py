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
from physicsnemo.mesh import (
    MESH_FIELD_ASSOCIATIONS,
    DomainMesh,
    Mesh,
    MeshFieldAssociation,
)


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
    """Rename keys in a TensorDict, returning a new TensorDict.

    Uses :meth:`tensordict.TensorDict.rename_key_` for each entry --
    that's the named TensorDict API for the operation, equivalent to
    ``td[new] = td.pop(old)`` but explicit. Missing source keys are
    silently skipped.
    """
    out = td.clone()
    present = set(out.keys())
    for old_key, new_key in mapping.items():
        if old_key in present:
            out.rename_key_(old_key, new_key)
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
        ### ``TensorDict.exclude(*keys)`` is null-safe: it returns a
        ### fresh TD minus the named keys (silently tolerating missing
        ### ones) and is a no-op clone when the key list is empty.
        return Mesh(
            points=mesh.points,
            cells=mesh.cells,
            point_data=mesh.point_data.exclude(*self._point_data_keys),
            cell_data=mesh.cell_data.exclude(*self._cell_data_keys),
            global_data=mesh.global_data.exclude(*self._global_data_keys),
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
        ### Coerce + bundle into a single TensorDict so the per-sample
        ### path in __call__ can collapse to a batched ``td.to(...)`` plus
        ### ``new_gd.update(...)`` (no Python-level per-key loop).
        coerced: dict[str, torch.Tensor] = {}
        for k, v in fields.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, dtype=torch.float32)
            coerced[k] = v
        self._fields: TensorDict = TensorDict(coerced, batch_size=[])

    def __call__(self, mesh: Mesh) -> Mesh:
        new_gd = mesh.global_data.clone()
        new_gd.update(
            self._fields.to(device=mesh.points.device, dtype=mesh.points.dtype)
        )
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
          association: point_data
          fields:
            pressure: {type: scalar, mean: -0.15, std: 0.45}
            wss: {type: vector, mean: [0.003, 0.0, 0.0], std: 0.005}

    Example YAML (from .pt file)::

        - _target_: ${dp:NormalizeMeshFields}
          association: point_data
          stats_file: /path/to/norm_stats.pt
    """

    def __init__(
        self,
        association: MeshFieldAssociation = "point_data",
        fields: dict[str, dict] | None = None,
        stats_file: str | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if association not in MESH_FIELD_ASSOCIATIONS:
            raise ValueError(
                f"association must be one of {MESH_FIELD_ASSOCIATIONS!r}, "
                f"got {association!r}"
            )
        self._association = association
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
        ### Clone and z-score the targeted association's TensorDict in
        ### place; fields absent from `_stats` (or absent from the mesh)
        ### are left untouched.
        new_td = getattr(mesh, self._association).clone()
        for field_name, stats in self._stats.items():
            if field_name not in new_td.keys():
                continue
            val = new_td[field_name].float()
            mean = stats["mean"].to(dtype=val.dtype, device=val.device)
            std = stats["std"].to(dtype=val.dtype, device=val.device)
            new_td[field_name] = (val - mean) / (std + self._eps)

        ### `Mesh.copy` is a tensorclass-provided shallow copy: `points`,
        ### `cells`, the untouched associations, and the geometric `_cache`
        ### (centroids / areas / normals) are all shared with `mesh`, then
        ### `setattr` swaps in the freshly-cloned association.
        new_mesh = mesh.copy()  # ty: ignore[unresolved-attribute]
        setattr(new_mesh, self._association, new_td)
        return new_mesh

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

    def inverse_td(self, td: TensorDict) -> TensorDict:
        r"""Un-normalize a per-field :class:`~tensordict.TensorDict` back to physical units.

        Companion to :meth:`inverse_tensor` for the per-field
        TensorDict-keyed I/O flow used by recipes that consume named
        prediction fields directly (rather than a concatenated
        ``(*, C)`` tensor). Each leaf is independently un-normalized
        using the stored stats; leaves whose names are absent from the
        stats dict are passed through unchanged.

        Parameters
        ----------
        td : TensorDict
            Per-field TensorDict whose leaves are normalized predictions
            keyed by field name. Each leaf can be any shape -- mean and
            std are broadcast against it -- as long as the trailing
            dim(s) match the stats' shape.

        Returns
        -------
        TensorDict
            New TensorDict (same keys, batch_size, and device as *td*)
            whose leaves are in physical units.
        """

        ### ``named_apply`` walks every leaf and collects the returns
        ### into a fresh TD; leaves whose name is absent from
        ### ``self._stats`` pass through unchanged.
        def _inverse_field(name: str, val: torch.Tensor) -> torch.Tensor:
            stats = self._stats.get(name)
            if stats is None:
                return val
            mean = stats["mean"].to(dtype=val.dtype, device=val.device)
            std = stats["std"].to(dtype=val.dtype, device=val.device)
            return val * (std + self._eps) + mean

        ### ``named_apply`` is typed ``TensorDict | None`` for its
        ### in-place mode; the out-of-place path always returns a TD.
        return td.named_apply(_inverse_field)  # ty: ignore[invalid-return-type]

    @property
    def stats(self) -> dict:
        """Normalization statistics dict (for serialization)."""
        return self._stats

    def extra_repr(self) -> str:
        parts = []
        for name, s in self._stats.items():
            parts.append(f"{name}({s['type']}): mean={s['mean']}, std={s['std']}")
        return f"association={self._association!r}, " + ", ".join(parts)


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


@register()
class MeshToDomainMesh(MeshTransform):
    r"""Convert a :class:`Mesh` into a :class:`DomainMesh` with a prediction-vs-input split.

    The output ``DomainMesh`` follows a semantic contract that separates
    *where the predictions live* from *what the inputs are*:

    - ``interior``: a :class:`Mesh` (typically a point cloud
      :class:`Mesh[0, n_spatial_dims]`) whose ``points`` are the prediction
      locations and whose ``point_data`` carries the prediction targets.
    - ``boundaries``: a single named entry mapping ``boundary_name`` to the
      input mesh, with target fields removed (so consumers cannot accidentally
      read targets through the boundary).
    - ``global_data``: passed through unchanged from the input mesh.

    This transform is intended for the common case where a single :class:`Mesh`
    serves as both the input geometry and (after centroid sampling or vertex
    sampling) the prediction locations -- e.g. surface CFD datasets where the
    model is asked to predict ``C_p`` and ``C_f`` on the same surface that
    forms its boundary condition.

    Wide inputs / narrow outputs:

    - Accepts any :class:`Mesh` (any manifold dimension; any cell / point /
      global data layout).
    - Always produces a :class:`DomainMesh` with exactly one boundary entry.
    - For :class:`DomainMesh` inputs, :meth:`apply_to_domain` is an identity
      passthrough so this transform may sit harmlessly at the end of any
      pipeline.

    Parameters
    ----------
    cell_data_targets : list[str] or None, default ``None``
        Names of cell-centered fields on the input mesh to use as prediction
        targets. They are moved out of the boundary's ``cell_data`` and into
        ``interior.point_data``. Use with ``interior_points='cell_centroids'``.
        If ``None`` (and ``point_data_targets`` is also ``None``), no targets
        are placed on the interior.
    point_data_targets : list[str] or None, default ``None``
        Names of vertex-centered fields on the input mesh to use as prediction
        targets. They are moved out of the boundary's ``point_data`` and into
        ``interior.point_data``. Use with ``interior_points='vertices'``.
    interior_points : str, default ``'cell_centroids'``
        Where the prediction interior is located. One of:

        - ``'cell_centroids'``: the interior is a point cloud
          :class:`Mesh[0, n_spatial_dims]` at ``mesh.cell_centroids``. Requires
          ``mesh.n_cells > 0``.
        - ``'vertices'``: the interior is a point cloud
          :class:`Mesh[0, n_spatial_dims]` at ``mesh.points``.
    boundary_name : str, default ``'vehicle'``
        Key used in the output ``DomainMesh``'s ``boundaries`` dict for the
        input mesh. The default ``'vehicle'`` matches the curated DrivAerML
        and HighLiftAeroML ``.pdmsh`` files, which standardize on
        ``vehicle`` as the body-boundary key in both their surface and
        volume layouts; downstream code can then resolve
        ``boundaries.vehicle`` uniformly across domains. Override with a
        more descriptive name when working outside that convention (e.g.
        ``'wing'``, ``'turbine_blade'``).

    Raises
    ------
    NotImplementedError
        Raised at call time when the requested combination of arguments is
        not implemented in this version. v1 implements two diagonals:
        ``(cell_data_targets, interior_points='cell_centroids')`` and
        ``(point_data_targets, interior_points='vertices')``. Cross-corners
        (cell-centered targets at vertex-located interior, or vice versa)
        require cell-to-point / point-to-cell interpolation and are deferred.
        Also raised if ``interior_points='cell_centroids'`` and the input mesh
        has no cells.

    Examples
    --------
    Convert a triangulated surface mesh into a DomainMesh with predictions at
    cell centroids and the original mesh as the ``vehicle`` boundary:

    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> mesh = Mesh(
    ...     points=torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
    ...     cells=torch.tensor([[0, 1, 2]]),
    ...     cell_data={"C_p": torch.tensor([0.5]), "normals": torch.tensor([[0., 0., 1.]])},
    ... )
    >>> transform = MeshToDomainMesh(
    ...     cell_data_targets=["C_p"], interior_points="cell_centroids", boundary_name="vehicle",
    ... )
    >>> domain = transform(mesh)
    >>> domain.interior.n_points  # one centroid per input cell
    1
    >>> "C_p" in domain.interior.point_data.keys()
    True
    >>> "normals" in domain.boundaries["vehicle"].cell_data.keys()  # non-target stays on boundary
    True
    >>> "C_p" in domain.boundaries["vehicle"].cell_data.keys()  # target removed from boundary
    False
    """

    _IMPLEMENTED_INTERIOR_POINTS = ("cell_centroids", "vertices")

    def __init__(
        self,
        cell_data_targets: list[str] | None = None,
        point_data_targets: list[str] | None = None,
        interior_points: Literal["cell_centroids", "vertices"] = "cell_centroids",
        boundary_name: str = "vehicle",
    ) -> None:
        super().__init__()
        ### Defensive runtime check: YAML / Hydra inputs bypass static
        ### typing, so we still validate against the implemented tuple.
        if interior_points not in self._IMPLEMENTED_INTERIOR_POINTS:
            raise ValueError(
                f"interior_points must be one of "
                f"{self._IMPLEMENTED_INTERIOR_POINTS!r}, got {interior_points!r}"
            )
        self._cell_data_targets: list[str] = list(cell_data_targets or [])
        self._point_data_targets: list[str] = list(point_data_targets or [])
        self._interior_points = interior_points
        self._boundary_name = boundary_name

    def __call__(self, mesh: Mesh) -> DomainMesh:  # type: ignore[override]
        ### v1 supports two diagonal corners:
        ### (cell_data_targets, interior_points='cell_centroids')
        ### (point_data_targets, interior_points='vertices')
        ### Cross-corners require cell<->point interpolation and are deferred.
        if self._interior_points == "cell_centroids":
            if self._point_data_targets:
                raise NotImplementedError(
                    f"point_data_targets={self._point_data_targets!r} requires "
                    f"point-to-cell interpolation when interior_points='cell_centroids'; "
                    f"this combination is not implemented in v1. Use "
                    f"interior_points='vertices' for point-centered targets, or "
                    f"interpolate to cell_data upstream."
                )
            if mesh.n_cells == 0:
                raise NotImplementedError(
                    f"interior_points='cell_centroids' requires the input mesh to "
                    f"have at least one cell, but got n_cells={mesh.n_cells}. Use "
                    f"interior_points='vertices' for point-cloud inputs."
                )
            return self._call_cell_centroids(mesh)
        else:  # interior_points == 'vertices'
            if self._cell_data_targets:
                raise NotImplementedError(
                    f"cell_data_targets={self._cell_data_targets!r} requires "
                    f"cell-to-point interpolation when interior_points='vertices'; "
                    f"this combination is not implemented in v1. Use "
                    f"interior_points='cell_centroids' for cell-centered targets, "
                    f"or interpolate to point_data upstream."
                )
            return self._call_vertices(mesh)

    def _call_cell_centroids(self, mesh: Mesh) -> DomainMesh:
        ### Build the interior as a point cloud at cell centroids, with target
        ### cell_data fields moved into interior.point_data.
        interior_point_data = (
            mesh.cell_data.select(*self._cell_data_targets)
            if self._cell_data_targets
            else TensorDict({}, batch_size=[mesh.n_cells])
        )
        interior = Mesh(
            points=mesh.cell_centroids,
            point_data=interior_point_data,
        )
        ### Build the boundary by stripping target fields from cell_data.
        boundary_cell_data = (
            mesh.cell_data.exclude(*self._cell_data_targets)
            if self._cell_data_targets
            else mesh.cell_data
        )
        boundary = Mesh(
            points=mesh.points,
            cells=mesh.cells,
            point_data=mesh.point_data,
            cell_data=boundary_cell_data,
            global_data=mesh.global_data,
        )
        return DomainMesh(
            interior=interior,
            boundaries={self._boundary_name: boundary},
            global_data=mesh.global_data,
        )

    def _call_vertices(self, mesh: Mesh) -> DomainMesh:
        ### Build the interior as a point cloud at the input mesh's vertices,
        ### with target point_data fields moved into interior.point_data.
        interior_point_data = (
            mesh.point_data.select(*self._point_data_targets)
            if self._point_data_targets
            else TensorDict({}, batch_size=[mesh.n_points])
        )
        interior = Mesh(
            points=mesh.points,
            point_data=interior_point_data,
        )
        boundary_point_data = (
            mesh.point_data.exclude(*self._point_data_targets)
            if self._point_data_targets
            else mesh.point_data
        )
        boundary = Mesh(
            points=mesh.points,
            cells=mesh.cells,
            point_data=boundary_point_data,
            cell_data=mesh.cell_data,
            global_data=mesh.global_data,
        )
        return DomainMesh(
            interior=interior,
            boundaries={self._boundary_name: boundary},
            global_data=mesh.global_data,
        )

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:  # type: ignore[override]
        """Identity passthrough for :class:`DomainMesh` inputs.

        A ``DomainMesh`` already satisfies the prediction-vs-input contract,
        so this transform is a no-op when applied to one. This lets you place
        :class:`MeshToDomainMesh` at the end of any pipeline -- it converts
        single-Mesh outputs and leaves DomainMesh outputs alone.

        Parameters
        ----------
        domain : DomainMesh

        Returns
        -------
        DomainMesh
            The input, unchanged.
        """
        return domain

    def extra_repr(self) -> str:
        parts: list[str] = []
        if self._cell_data_targets:
            parts.append(f"cell_data_targets={self._cell_data_targets}")
        if self._point_data_targets:
            parts.append(f"point_data_targets={self._point_data_targets}")
        parts.append(f"interior_points={self._interior_points!r}")
        parts.append(f"boundary_name={self._boundary_name!r}")
        return ", ".join(parts)
