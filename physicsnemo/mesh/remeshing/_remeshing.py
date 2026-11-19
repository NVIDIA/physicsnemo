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
ResolutionField: TypeAlias = PointDataKey | torch.Tensor | None
# A linear-resolution multiplier is an inverse target edge length. For the
# squared-distance CVT objective on a 2D surface, its integration density is
# the fourth power of that multiplier.
_LINEAR_RESOLUTION_DENSITY_EXPONENT = 4.0
# Clustering integrates only at source vertices, so a large source triangle
# holds no cluster inside it. Source edges are split down to this fraction of
# the output's target edge, then until each cluster has this many source
# points, within this many source faces per cluster.
_SPLIT_EDGE_FRACTION = 0.5
_MIN_POINTS_PER_CLUSTER = 4
_MAX_FACES_PER_CLUSTER = 256


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
    field: ResolutionField,
) -> torch.Tensor | None:
    """Resolve and validate a direct or attached linear-resolution field."""
    if field is None:
        return None
    if isinstance(field, torch.Tensor):
        resolution = field
        description = "resolution_field"
    else:
        if not isinstance(field, (str, tuple)):
            raise TypeError(
                "resolution_field must be a torch.Tensor, point_data key/path, or None"
            )
        if isinstance(field, tuple) and (
            not field or not all(isinstance(part, str) for part in field)
        ):
            raise TypeError("resolution_field paths must be nonempty tuples of strings")
        key = _canonical_point_data_key(field)
        available = list(mesh.point_data.keys(include_nested=True, leaves_only=True))
        if key not in available:
            raise KeyError(
                f"resolution_field {key!r} was not found in point_data. "
                f"Available keys: {available}"
            )
        resolution = mesh.point_data[key]
        description = f"resolution_field {key!r}"
        if not isinstance(resolution, torch.Tensor):
            raise TypeError(f"{description} must resolve to a torch.Tensor")
    if resolution.shape != (mesh.n_points,):
        raise ValueError(
            f"{description} must have shape ({mesh.n_points},), "
            f"got {tuple(resolution.shape)}"
        )
    if not torch.is_floating_point(resolution):
        raise TypeError(
            f"{description} must use a real floating-point dtype, "
            f"got {resolution.dtype}"
        )
    if resolution.device != mesh.points.device:
        raise ValueError(f"{description} and mesh points must be on the same device")
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
        if not torch.is_floating_point(values):
            raise TypeError(
                f"point_data field {key!r} must use a real floating-point "
                "dtype for barycentric interpolation, got "
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
            torch.float32 if values.element_size() < 4 else values.dtype
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


def _widen(values: torch.Tensor) -> torch.Tensor:
    """Promote sub-32-bit floats to float32 for averaging."""
    return values.to(torch.float32) if values.element_size() < 4 else values


def _split_long_edges(
    points: torch.Tensor,
    cells: torch.Tensor,
    fields: list[torch.Tensor | TensorDict],
    n_clusters: int,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor | TensorDict]]:
    """Split long edges at their midpoints, the longest first.

    Each round splits the edges longer than half the longest, and the faces
    bisect their longest split edge until none is left, so the surface and
    every source vertex stay unchanged. ``fields`` hold per-point tensors or
    TensorDicts, averaged at the midpoints. Refinement stops when the next
    round would exceed the face budget.
    """
    vertices = points[cells].to(torch.float64)
    edge_vectors = vertices[:, 1:] - vertices[:, :1]
    doubled_area = (
        torch.linalg.cross(edge_vectors[:, 0], edge_vectors[:, 1]).norm(dim=1).sum()
    )
    # Edge of an equilateral triangle when the area is cut into the output's
    # roughly 2 * n_clusters faces
    target_edge = float((doubled_area / (3**0.5 * n_clusters)).sqrt())
    target = _SPLIT_EDGE_FRACTION * target_edge
    max_faces = max(cells.shape[0], _MAX_FACES_PER_CLUSTER * n_clusters)
    rotation = torch.arange(3, device=cells.device)
    while True:
        starts, ends = cells, cells.roll(-1, dims=1)
        keys = torch.minimum(starts, ends) << 32 | torch.maximum(starts, ends)
        edges = torch.unique(keys)
        lengths = (points[edges >> 32] - points[edges & 0xFFFFFFFF]).float().norm(dim=1)
        longest = float(lengths.max())
        if longest <= target and points.shape[0] >= (
            _MIN_POINTS_PER_CLUSTER * n_clusters
        ):
            break
        threshold = longest / 2 if longest <= target else max(target, longest / 2)
        split = edges[lengths > threshold]
        if split.numel() == 0 or cells.shape[0] + 3 * split.numel() > max_faces:
            break
        n_points = points.shape[0]
        a, b = split >> 32, split & 0xFFFFFFFF
        points = torch.cat([points, (points[a] + points[b]) / 2])
        fields = [torch.cat([field, (field[a] + field[b]) / 2]) for field in fields]
        while True:
            starts, ends = cells, cells.roll(-1, dims=1)
            keys = torch.minimum(starts, ends) << 32 | torch.maximum(starts, ends)
            index = torch.searchsorted(split, keys).clamp_max(split.numel() - 1)
            pending = split[index] == keys
            if not bool(pending.any()):
                break
            # Rotate the longest pending edge to edge 0 (a to b), then bisect
            # it into (a, m, c) and (m, b, c)
            lengths = (points[starts] - points[ends]).float().norm(dim=2)
            longest_edge = lengths.masked_fill(~pending, -1.0).argmax(1, keepdim=True)
            bisect = pending.any(dim=1)
            order = (rotation + longest_edge[bisect]) % 3
            a, b, c = cells[bisect].gather(1, order).T
            m = n_points + index[bisect].gather(1, order)[:, 0]
            cells = torch.cat(
                [cells[~bisect], torch.stack([a, m, c], 1), torch.stack([m, b, c], 1)]
            )
    return points, cells, fields


def _fill_holes(mesh: Mesh) -> Mesh:
    """Fill every hole, giving each new centroid its boundary's mean point data."""
    from physicsnemo.mesh.repair import fill_holes

    n_points = mesh.n_points
    mesh = fill_holes(mesh, max_hole_edges=n_points)[0]
    if mesh.n_points == n_points or len(mesh.point_data.keys()) == 0:
        return mesh
    # Each fan face holds one centroid and two boundary points
    is_new = mesh.cells >= n_points
    fans = is_new.any(dim=1)
    centroid = mesh.cells[fans].max(dim=1).values
    boundary = mesh.cells[fans][~is_new[fans]].reshape(-1, 2)
    counts = 2 * torch.bincount(centroid, minlength=mesh.n_points)

    def boundary_mean(values: torch.Tensor) -> torch.Tensor:
        wide = _widen(values)
        sums = torch.zeros_like(wide).index_add(0, centroid, wide[boundary].sum(1))
        means = sums / counts.reshape((-1,) + (1,) * (values.ndim - 1))
        return torch.cat([values[:n_points], means[n_points:].to(values.dtype)])

    return mesh.with_data(point_data=mesh.point_data.apply(boundary_mean))


def _orient_like(mesh: Mesh, vertex_normals: torch.Tensor) -> Mesh:
    """Wind each component consistently, then like most of ``vertex_normals``."""
    from physicsnemo.mesh.repair import fix_orientation
    from physicsnemo.mesh.repair.orientation import _facet_pairs
    from physicsnemo.mesh.utilities._duplicate_detection import (
        vectorized_connected_components,
    )

    mesh = fix_orientation(mesh)[0]
    _, first, second, _, _ = _facet_pairs(mesh.cells, mesh.n_points)
    component = vectorized_connected_components(
        torch.stack([first // 3, second // 3], dim=1), mesh.n_cells
    )
    # Points added by hole filling have no source normal
    vertex_normals = torch.nn.functional.pad(
        vertex_normals, (0, 0, 0, mesh.n_points - vertex_normals.shape[0])
    )
    vertices = mesh.points[mesh.cells].float()
    normals = torch.linalg.cross(
        vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0], dim=1
    )
    agreement = (normals * vertex_normals[mesh.cells].sum(dim=1)).sum(dim=1)
    votes = torch.zeros_like(agreement).index_add(0, component, agreement)
    flip = votes[component] < 0
    if not bool(flip.any()):
        return mesh
    return mesh.with_cells(
        torch.where(flip[:, None], mesh.cells[:, [0, 2, 1]], mesh.cells)
    )


def _close_and_orient(output: Mesh, source: Mesh, source_faces: torch.Tensor) -> Mesh:
    """Close a closed source's remesh and wind each component like the source.

    When the source is watertight, holes are filled, and faces on edges that
    no winding can make consistent, such as those of a fold, are replaced by
    hole fills. After :func:`~physicsnemo.mesh.repair.fix_orientation` winds
    each component consistently, a component is reversed when its normals
    mostly oppose those of the source faces its vertices were projected onto.
    """
    from physicsnemo.mesh.boundaries import is_watertight
    from physicsnemo.mesh.repair import remove_isolated_points, split_nonmanifold_edges

    source_vertices = source.points[source.cells[source_faces]].float()
    vertex_normals = torch.linalg.cross(
        source_vertices[:, 1] - source_vertices[:, 0],
        source_vertices[:, 2] - source_vertices[:, 0],
        dim=1,
    )
    if not is_watertight(source):
        return _orient_like(output, vertex_normals)
    output = _orient_like(_fill_holes(output), vertex_normals)
    # A fill can itself make a bad edge, so cut and fill up to three times
    for _ in range(3):
        # Edges in more than two faces, or run the same way by two
        cells = output.cells
        starts, ends = cells.T.reshape(-1), cells.roll(-1, dims=1).T.reshape(-1)
        keys = torch.minimum(starts, ends) * output.n_points + torch.maximum(
            starts, ends
        )
        _, edge_ids, counts = torch.unique(
            keys, return_inverse=True, return_counts=True
        )
        direction = torch.where(starts < ends, 1, -1)
        winding = torch.zeros_like(counts).index_add(0, edge_ids, direction)
        is_bad = (counts > 2) | ((counts == 2) & (winding != 0))
        bad_faces = is_bad[edge_ids].reshape(3, -1).any(dim=0)
        if not bool(bad_faces.any()):
            break
        output = _fill_holes(output.slice_cells(~bad_faces))
        output = _orient_like(output, vertex_normals)
    # Where two sheets still share an edge, give each its own copy of it.
    # Isolated points go last, so that vertex_normals stay aligned.
    output = split_nonmanifold_edges(output, drop_fins=False)[0]
    return remove_isolated_points(output)[0]


def remesh(
    mesh: Mesh,
    n_clusters: int,
    *,
    max_iterations: int = 4,
    transfer_point_data: PointDataSelection = False,
    resolution_field: ResolutionField = None,
) -> Mesh:
    """Remesh a triangle surface with point-data and resolution controls.

    Long source edges are first split at their midpoints, so that clusters
    can land inside large faces. Warp then performs integration-mass-weighted
    centroidal clustering, projects cluster centers back to the source surface
    with a bounding volume hierarchy, and reconstructs compact triangle
    connectivity. If the source is watertight, holes in the result are filled.
    Each connected component is then wound consistently and like the source,
    so a closed, outward-wound source gives a closed, outward-wound result. A
    direct positive tensor or an attached point-data field can specify
    relative local linear resolution.

    Parameters
    ----------
    mesh : Mesh
        Input triangle surface. Only 2D triangle manifolds embedded in 3D are
        supported.
    n_clusters : int
        Target output vertex count. Cleanup can produce slightly fewer
        vertices, and hole filling slightly more.
        Must be between 3 and the input point count, inclusive.
    max_iterations : int, optional
        Maximum centroid-relaxation iterations. Default is ``4``. Values must
        be non-negative.
    transfer_point_data : bool, str, tuple, list, or None, optional
        Point-data fields to interpolate onto the output vertices. ``False``
        or ``None`` transfers no fields. ``True`` transfers every point-data
        leaf. A string or tuple selects one key or nested key path. A list
        selects several keys or paths. Selected fields must contain real
        floating-point tensors. Default is ``False``.
    resolution_field : str, tuple, torch.Tensor, or None, optional
        Positive scalar tensor with shape ``(n_points,)``, or a key or nested
        key path resolving to one in ``mesh.point_data``. Values specify
        relative linear resolution. A value twice another requests
        approximately half the local edge spacing. The fixed ``n_clusters``
        budget and source geometry limit the realized spacing. The field must
        use a real floating-point dtype on the mesh device. Direct tensor
        entries correspond to ``mesh.points`` order and are not attached to or
        transferred with the output mesh. Only relative values matter. Default
        is ``None`` for uniform remeshing.

    Returns
    -------
    Mesh
        Remeshed surface on the input device. Selected point data is
        barycentrically interpolated from the original source surface. A
        vertex added by hole filling takes the mean of its hole's boundary.
        Cell data and unselected point data are discarded. Global data is
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
    Edge splitting continues until edges are at most half the target output
    edge and the source has four points per cluster, within 256 source faces
    per cluster.
    Output vertices are compacted by connectivity but are not welded by
    position. Backend-specific tuning remains available through
    :func:`physicsnemo.nn.functional.remeshing`. These advanced parameters may
    change as the implementation evolves.
    """
    from physicsnemo.mesh.calculus.measure import EFFECTIVE_MEASURE_KEY

    if (
        EFFECTIVE_MEASURE_KEY in mesh.cell_data
        or EFFECTIVE_MEASURE_KEY in mesh.point_data
    ):
        raise ValueError(
            "Remeshing explicit quadrature requires a conservative measure transfer; supply replacement measures instead of interpolating or dropping them"
        )

    if mesh.n_manifold_dims != 2 or mesh.n_spatial_dims != 3:
        raise NotImplementedError(
            "remesh only supports 2D triangle surfaces embedded in 3D. Got "
            f"n_manifold_dims={mesh.n_manifold_dims} and "
            f"n_spatial_dims={mesh.n_spatial_dims}"
        )

    transfer_keys = _point_data_keys(mesh, transfer_point_data)
    _validate_transfer_fields(mesh, transfer_keys)
    linear_resolution = _resolve_resolution_field(mesh, resolution_field)

    from physicsnemo.mesh.mesh import Mesh

    candidates = [(mesh, linear_resolution)]
    if (
        isinstance(n_clusters, int)
        and 3 <= n_clusters <= mesh.n_points
        and mesh.n_cells > 0
        and bool(torch.isfinite(mesh.points).all())
        and 0 <= int(mesh.cells.min())
        and int(mesh.cells.max()) < mesh.n_points
    ):
        # Invalid inputs skip the refinement, so that remeshing reports them.
        # Midpoint averages use at least float32, as the final interpolation.
        fields = [mesh.point_data.select(*transfer_keys), linear_resolution]
        fields = [
            field.apply(_widen) if isinstance(field, TensorDict) else _widen(field)
            for field in fields
            if field is not None
        ]
        points, cells, fields = _split_long_edges(
            mesh.points.detach(), mesh.cells.to(torch.int64), fields, n_clusters
        )
        refined = Mesh(points=points, cells=cells, point_data=fields[0])
        candidates.insert(0, (refined, fields[1] if len(fields) > 1 else None))
    for source, resolution in candidates:
        try:
            (
                output_points,
                output_cells,
                source_faces,
                barycentric_coordinates,
            ) = _remeshing_with_mapping(
                source.points,
                source.cells,
                n_clusters,
                max_iterations=max_iterations,
                vertex_density=resolution,
                vertex_density_exponent=_LINEAR_RESOLUTION_DENSITY_EXPONENT,
            )
            break
        except RuntimeError:
            # A strip thinner than the cluster spacing can collapse once
            # refined, so remesh the input as given
            if source is mesh:
                raise
    output_point_data = (
        _interpolate_point_data(
            source,
            transfer_keys,
            source_faces,
            barycentric_coordinates,
        )
        if transfer_keys
        else None
    )
    output = Mesh(
        points=output_points,
        cells=output_cells,
        point_data=output_point_data,
        global_data=mesh.global_data.clone(),
    )
    output = _close_and_orient(output, source, source_faces)
    for key in transfer_keys:
        output.point_data[key] = output.point_data[key].to(mesh.point_data[key].dtype)
    return output


__all__ = ["remesh"]
