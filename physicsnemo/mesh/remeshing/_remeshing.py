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

"""Public Mesh API for Warp-accelerated surface remeshing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias

import torch
from tensordict import TensorDict

from physicsnemo.nn.functional.geometry.remeshing.remeshing import (
    _remeshing_with_mapping,
)

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh

PointDataKey: TypeAlias = str | tuple[str, ...]
PointDataSelection: TypeAlias = bool | PointDataKey | list[PointDataKey] | None
_SUPPORTED_FIELD_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}
_LINEAR_RESOLUTION_DENSITY_EXPONENT = 4.0


def _canonical_point_data_key(key: PointDataKey) -> PointDataKey:
    """Match TensorDict's canonical spelling for one-component key paths."""
    if isinstance(key, tuple) and len(key) == 1:
        return key[0]
    return key


def _point_data_keys(
    mesh: Mesh,
    selection: PointDataSelection,
) -> list[PointDataKey]:
    """Resolve a point-data transfer selection into unique leaf paths."""
    if selection is None or selection is False:
        return []
    available = list(mesh.point_data.keys(include_nested=True, leaves_only=True))
    if selection is True:
        return available

    if isinstance(selection, str):
        requested: list[PointDataKey] = [selection]
    elif isinstance(selection, tuple) and all(
        isinstance(part, str) for part in selection
    ):
        if not selection:
            raise ValueError("point_data key paths cannot be empty")
        requested = [selection]
    elif isinstance(selection, list):
        requested = list(selection)
    else:
        raise TypeError(
            "transfer_point_data must be a bool, point_data key/path, "
            "list of keys/paths, or None"
        )

    resolved: list[PointDataKey] = []
    for key in requested:
        if isinstance(key, str):
            normalized_key: PointDataKey = key
        elif (
            isinstance(key, tuple)
            and key
            and all(isinstance(part, str) for part in key)
        ):
            normalized_key = _canonical_point_data_key(key)
        else:
            raise TypeError(
                "each transfer_point_data entry must be a string or a "
                "nonempty tuple of strings"
            )
        if normalized_key not in available:
            raise KeyError(
                f"point_data field {normalized_key!r} was not found. "
                f"Available keys: {available}"
            )
        if normalized_key not in resolved:
            resolved.append(normalized_key)
    return resolved


def _resolve_resolution_field(
    mesh: Mesh,
    key: PointDataKey | None,
) -> torch.Tensor | None:
    """Resolve and validate an attached linear-resolution field."""
    if key is None:
        return None
    if not isinstance(key, (str, tuple)):
        raise TypeError("resolution_field must be a point_data key/path or None")
    if isinstance(key, tuple) and (
        not key or not all(isinstance(part, str) for part in key)
    ):
        raise TypeError("resolution_field paths must be nonempty tuples of strings")
    key = _canonical_point_data_key(key)
    available = list(mesh.point_data.keys(include_nested=True, leaves_only=True))
    if key not in available:
        raise KeyError(
            f"resolution_field {key!r} was not found in point_data. "
            f"Available keys: {available}"
        )
    resolution = mesh.point_data[key]
    if not isinstance(resolution, torch.Tensor):
        raise TypeError(f"resolution_field {key!r} must resolve to a torch.Tensor")
    if resolution.shape != (mesh.n_points,):
        raise ValueError(
            f"resolution_field {key!r} must have shape ({mesh.n_points},), "
            f"got {tuple(resolution.shape)}"
        )
    if resolution.dtype not in _SUPPORTED_FIELD_DTYPES:
        raise TypeError(
            f"resolution_field {key!r} must use a supported real "
            "floating-point dtype. Expected float16, bfloat16, float32, or "
            f"float64, got {resolution.dtype}"
        )
    if resolution.device != mesh.points.device:
        raise ValueError(
            f"resolution_field {key!r} and mesh points must be on the same device"
        )
    return resolution


def _validate_transfer_fields(
    mesh: Mesh,
    keys: Sequence[PointDataKey],
) -> None:
    """Validate selected fields before starting the remeshing operation."""
    for key in keys:
        values = mesh.point_data[key]
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"point_data field {key!r} must be a torch.Tensor")
        if values.dtype not in _SUPPORTED_FIELD_DTYPES:
            raise TypeError(
                f"point_data field {key!r} must use a supported real "
                "floating-point dtype for barycentric interpolation. "
                "Expected float16, bfloat16, float32, or float64, got "
                f"{values.dtype}"
            )
        if values.shape[0] != mesh.n_points:
            raise ValueError(
                f"point_data field {key!r} must start with "
                f"n_points={mesh.n_points}, got shape {tuple(values.shape)}"
            )
        if values.device != mesh.points.device:
            raise ValueError(
                f"point_data field {key!r} and mesh points must be on the same device"
            )


def _interpolate_point_data(
    mesh: Mesh,
    keys: Sequence[PointDataKey],
    source_faces: torch.Tensor,
    barycentric_coordinates: torch.Tensor,
) -> TensorDict:
    """Interpolate selected source fields at projected output vertices."""
    output = TensorDict(
        {},
        batch_size=[source_faces.shape[0]],
        device=mesh.points.device,
    )
    if not keys:
        return output

    if bool((source_faces < 0).any()):
        raise RuntimeError(
            "Warp could not locate a source triangle for every remeshed "
            "vertex, so point data cannot be transferred"
        )
    source_vertices = mesh.cells.to(torch.int64)[source_faces]
    weights_by_dtype: dict[torch.dtype, torch.Tensor] = {}
    for key in keys:
        values = mesh.point_data[key]
        accumulation_dtype = (
            torch.float32
            if values.dtype in (torch.float16, torch.bfloat16)
            else values.dtype
        )
        gathered = values[source_vertices].to(dtype=accumulation_dtype)
        weights = weights_by_dtype.get(accumulation_dtype)
        if weights is None:
            weights = barycentric_coordinates.to(dtype=accumulation_dtype)
            weights_by_dtype[accumulation_dtype] = weights
        weight_shape = (source_faces.shape[0], 3) + (1,) * (values.ndim - 1)
        weights = weights.reshape(weight_shape)
        interpolated = (gathered * weights).sum(dim=1).to(dtype=values.dtype)
        output.set(key, interpolated)
    return output


def remesh(
    mesh: Mesh,
    n_clusters: int,
    *,
    max_iterations: int = 4,
    transfer_point_data: PointDataSelection = False,
    resolution_field: PointDataKey | None = None,
) -> Mesh:
    """Remesh a triangle surface with point-data and resolution controls.

    Warp performs integration-mass-weighted centroidal clustering, projects
    cluster centers back to the source surface with a bounding volume
    hierarchy, and reconstructs compact triangle connectivity. A positive
    point-data field can specify relative local linear resolution.

    Parameters
    ----------
    mesh : Mesh
        Input triangle surface. Only 2D triangle manifolds embedded in 3D are
        supported.
    n_clusters : int
        Target output vertex count. Cleanup can produce slightly fewer vertices.
        Must be between 3 and the input point count, inclusive.
    max_iterations : int, optional
        Maximum centroid-relaxation iterations. Default is ``4``. Values must
        be non-negative.
    transfer_point_data : bool, str, tuple, list, or None, optional
        Point-data fields to interpolate onto the output vertices. ``False``
        or ``None`` transfers no fields. ``True`` transfers every point-data
        leaf. A string or tuple selects one key or nested key path. A list
        selects several keys or paths. Selected fields must contain real
        floating-point tensors with float16, bfloat16, float32, or float64
        dtype. Default is ``False``.
    resolution_field : str, tuple, or None, optional
        Key or nested key path for a positive scalar point-data field with
        shape ``(n_points,)``. Values specify relative linear resolution. A
        value twice another requests approximately half the local edge
        spacing. The fixed ``n_clusters`` budget and source geometry limit the
        realized spacing. The field must use float16, bfloat16, float32, or
        float64. Only relative values matter. Default is ``None`` for uniform
        remeshing.

    Returns
    -------
    Mesh
        Remeshed surface on the input device. Selected point data is
        barycentrically interpolated from the original source surface. Cell
        data and unselected point data are discarded. Global data is
        preserved.

    Raises
    ------
    TypeError
        If counts, tuning parameters, point coordinates, a field selection, or
        a selected field has an invalid type.
    ValueError
        If a count is out of range or geometry, connectivity, or a selected
        field is invalid.
    KeyError
        If a requested point-data key or path does not exist.
    NotImplementedError
        If ``mesh`` is not a 2D triangle surface embedded in 3D.
    ImportError
        If Warp is unavailable.
    RuntimeError
        If cleanup cannot reconstruct a nonempty manifold triangle surface or
        point-data transfer provenance is unavailable.

    Notes
    -----
    Remeshing, topology, projection choices, and resolution control are
    intentionally non-differentiable. Transferred fields remain differentiable
    with respect to their source values because the final barycentric
    interpolation uses PyTorch. Warp computes geometry in centered and scaled
    coordinates in float32, then restores the input point dtype and coordinate
    frame. For the 2D squared-distance CVT objective, the implementation
    converts linear resolution ``r`` to integration density ``r**4``. Ideal
    local point density therefore scales approximately as ``r**2``. These
    relationships guide allocation but do not guarantee exact edge lengths or
    local point counts. Because clustering uses spatial distance rather than
    mesh connectivity, sheets or thin features separated by less than the mean
    cluster spacing can be assigned to a common cluster and welded together.
    Projection can map distinct cluster centroids to the same surface position.
    Output vertices are compacted by connectivity but are not welded by
    position. Backend-specific tuning remains available through
    :func:`physicsnemo.nn.functional.remeshing`. These advanced parameters may
    change as the implementation evolves.
    """
    if mesh.n_manifold_dims != 2 or mesh.n_spatial_dims != 3:
        raise NotImplementedError(
            "remesh only supports 2D triangle surfaces embedded in 3D. Got "
            f"n_manifold_dims={mesh.n_manifold_dims} and "
            f"n_spatial_dims={mesh.n_spatial_dims}"
        )

    transfer_keys = _point_data_keys(mesh, transfer_point_data)
    _validate_transfer_fields(mesh, transfer_keys)
    linear_resolution = _resolve_resolution_field(mesh, resolution_field)
    (
        output_points,
        output_cells,
        source_faces,
        barycentric_coordinates,
    ) = _remeshing_with_mapping(
        mesh.points,
        mesh.cells,
        n_clusters,
        max_iterations=max_iterations,
        vertex_density=linear_resolution,
        vertex_density_exponent=_LINEAR_RESOLUTION_DENSITY_EXPONENT,
    )
    output_point_data = (
        _interpolate_point_data(
            mesh,
            transfer_keys,
            source_faces,
            barycentric_coordinates,
        )
        if transfer_keys
        else None
    )

    from physicsnemo.mesh.mesh import Mesh

    return Mesh(
        points=output_points,
        cells=output_cells,
        point_data=output_point_data,
        global_data=mesh.global_data.clone(),
    )


__all__ = ["remesh"]
