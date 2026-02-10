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

"""Text rendering to mesh in various configurations.

Provides functions to convert text strings into meshes with different
dimensional configurations: 1D curves, 2D surfaces, 3D volumes, and boundaries.

Uses matplotlib's font rendering, Delaunay triangulation, and intelligent
hole detection (for letters like 'o', 'e', 'a') using the shoelace formula.

This module requires matplotlib to be installed.
"""

import torch

from physicsnemo.core.version_check import require_version_spec
from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.projections import embed, extrude


def _compute_polygon_signed_area(vertices) -> float:
    """Compute signed area using shoelace formula (positive=outer, negative=hole)."""
    import numpy as np

    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
    vertices = np.array(vertices)

    n = len(vertices)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]

    return -area * 0.5  # Negate for positive=outer convention


def _sample_curve_segment(p0, control_points, pn, num_samples: int):
    """Sample Bezier curve segment."""
    t = torch.linspace(0, 1, num_samples, dtype=p0.dtype, device=p0.device).unsqueeze(1)

    if len(control_points) == 1:
        # Quadratic Bezier
        p1 = control_points[0]
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * pn
    elif len(control_points) == 2:
        # Cubic Bezier
        p1, p2 = control_points
        return (
            (1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t**2 * p2
            + t**3 * pn
        )
    else:
        raise ValueError(
            f"Unsupported curve order with {len(control_points)} control points"
        )


def _text_to_path(text: str, font_size: float = 12.0, samples_per_unit: float = 10):
    """Convert text to sampled path with edges.

    Returns
    -------
    tuple
        Tuple of (points, edges, matplotlib Path object)
    """
    import importlib

    font_manager = importlib.import_module("matplotlib.font_manager")
    mpl_path = importlib.import_module("matplotlib.path")
    textpath = importlib.import_module("matplotlib.textpath")

    fp = font_manager.FontProperties(family="sans-serif", weight="bold")
    text_path = textpath.TextPath((0, 0), text, size=font_size, prop=fp)

    verts = torch.tensor(text_path.vertices.copy(), dtype=torch.float32)
    codes = torch.tensor(text_path.codes.copy(), dtype=torch.int64)

    all_points: list[torch.Tensor] = []
    all_edges: list[torch.Tensor] = []
    current_offset = 0
    path_points: list[torch.Tensor] = []

    i = 0
    while i < len(codes):
        code = codes[i].item()

        if code == mpl_path.Path.MOVETO:
            if path_points:
                path_points.append(path_points[0])
                n_edges = len(path_points) - 1
                edges = torch.stack(
                    [
                        torch.arange(n_edges, dtype=torch.int64) + current_offset,
                        torch.arange(n_edges, dtype=torch.int64) + current_offset + 1,
                    ],
                    dim=1,
                )
                all_edges.extend(edges)
                all_points.extend(path_points)
                current_offset += len(path_points)
            path_points = [verts[i]]
            i += 1
        elif code == mpl_path.Path.LINETO:
            path_points.append(verts[i])
            i += 1
        elif code == mpl_path.Path.CURVE3:
            dist = torch.norm(verts[i + 1] - path_points[-1]).item()
            num_samples = max(5, int(dist * samples_per_unit))
            sampled = _sample_curve_segment(
                path_points[-1], [verts[i]], verts[i + 1], num_samples
            )
            path_points.extend(sampled[1:])
            i += 2
        elif code == mpl_path.Path.CURVE4:
            dist = torch.norm(verts[i + 2] - path_points[-1]).item()
            num_samples = max(5, int(dist * samples_per_unit))
            sampled = _sample_curve_segment(
                path_points[-1], [verts[i], verts[i + 1]], verts[i + 2], num_samples
            )
            path_points.extend(sampled[1:])
            i += 3
        elif code == mpl_path.Path.CLOSEPOLY:
            if path_points:
                path_points.append(path_points[0])
                n_edges = len(path_points) - 1
                edges = torch.stack(
                    [
                        torch.arange(n_edges, dtype=torch.int64) + current_offset,
                        torch.arange(n_edges, dtype=torch.int64) + current_offset + 1,
                    ],
                    dim=1,
                )
                all_edges.extend(edges)
                all_points.extend(path_points)
                current_offset += len(path_points)
            path_points = []
            i += 1
        else:
            i += 1

    if path_points:
        path_points.append(path_points[0])
        n_edges = len(path_points) - 1
        edges = torch.stack(
            [
                torch.arange(n_edges, dtype=torch.int64) + current_offset,
                torch.arange(n_edges, dtype=torch.int64) + current_offset + 1,
            ],
            dim=1,
        )
        all_edges.extend(edges)
        all_points.extend(path_points)

    points = torch.stack(all_points, dim=0)
    edges = torch.stack(all_edges, dim=0)

    # Center
    center = points.mean(dim=0)
    points = points - center

    centered_vertices = text_path.vertices - center.cpu().numpy()
    text_path = mpl_path.Path(centered_vertices, text_path.codes)

    return points, edges, text_path


def _refine_edges(points: torch.Tensor, edges: torch.Tensor, max_length: float):
    """Subdivide long edges."""
    refined_points = [points]
    refined_edges = []
    next_idx = len(points)

    for edge in edges:
        p0_idx, p1_idx = edge[0].item(), edge[1].item()
        p0, p1 = points[p0_idx], points[p1_idx]
        edge_vec = p1 - p0
        edge_length = torch.norm(edge_vec).item()

        if edge_length <= max_length:
            refined_edges.append(edge)
        else:
            n_segments = int(torch.ceil(torch.tensor(edge_length / max_length)).item())
            prev_idx = p0_idx
            for j in range(1, n_segments):
                t = j / n_segments
                interp_point = p0 + t * edge_vec
                refined_points.append(interp_point.unsqueeze(0))
                refined_edges.append(
                    torch.tensor([prev_idx, next_idx], dtype=torch.int64)
                )
                prev_idx = next_idx
                next_idx += 1
            refined_edges.append(torch.tensor([prev_idx, p1_idx], dtype=torch.int64))

    return torch.cat(refined_points, dim=0), torch.stack(refined_edges, dim=0)


def _group_letters(text_path):
    """Group polygons into letters using signed area and containment."""
    import importlib

    import numpy as np

    mpl_path = importlib.import_module("matplotlib.path")

    path_codes = np.array(text_path.codes)
    closepoly_indices = np.where(path_codes == mpl_path.Path.CLOSEPOLY)[0]

    outers, holes = [], []
    start_idx = 0

    for close_idx in closepoly_indices:
        end_idx = close_idx + 1
        polygon_verts = text_path.vertices[start_idx:end_idx]
        signed_area = _compute_polygon_signed_area(polygon_verts)

        if signed_area > 0:
            outers.append((start_idx, end_idx))
        else:
            holes.append((start_idx, end_idx))

        start_idx = end_idx

    # Assign holes to parents via containment
    letter_groups = []
    for outer_start, outer_end in outers:
        if text_path.vertices is None or text_path.codes is None:
            continue
        outer_verts = text_path.vertices[outer_start:outer_end]
        outer_codes = text_path.codes[outer_start:outer_end]
        outer_path = mpl_path.Path(outer_verts, outer_codes)

        contained_holes = []
        for hole_start, hole_end in holes:
            hole_sample = text_path.vertices[hole_start]
            if outer_path.contains_point(hole_sample):
                contained_holes.append((hole_start, hole_end))

        letter_groups.append(
            {"outer": (outer_start, outer_end), "holes": contained_holes}
        )

    return letter_groups


def _winding_number(points: torch.Tensor, path) -> torch.Tensor:
    """Compute winding number for path containment test."""
    import importlib

    import numpy as np

    mpl_path = importlib.import_module("matplotlib.path")

    path_codes = np.array(path.codes)
    moveto_indices = np.where(path_codes == mpl_path.Path.MOVETO)[0]
    total_winding = torch.zeros(len(points), dtype=torch.float32, device=points.device)

    for i, start_idx in enumerate(moveto_indices):
        end_idx = (
            int(moveto_indices[i + 1])
            if i < len(moveto_indices) - 1
            else len(path_codes)
        )
        contour_verts = torch.tensor(
            path.vertices[start_idx:end_idx], dtype=torch.float32
        )
        winding_contour = torch.zeros(
            len(points), dtype=torch.float32, device=points.device
        )

        for j in range(len(contour_verts)):
            v0 = contour_verts[j]
            v1 = contour_verts[(j + 1) % len(contour_verts)]

            if v0[1] == v1[1]:
                continue

            y_low = torch.minimum(v0[1], v1[1])
            y_high = torch.maximum(v0[1], v1[1])
            y_in_range = (points[:, 1] >= y_low) & (points[:, 1] < y_high)

            t = (points[:, 1] - v0[1]) / (v1[1] - v0[1])
            x_intersect = v0[0] + t * (v1[0] - v0[0])
            crosses = y_in_range & (x_intersect > points[:, 0])
            direction = torch.sign(v1[1] - v0[1])
            winding_contour = winding_contour + crosses.float() * direction

        total_winding = total_winding + winding_contour

    return total_winding


def _get_letter_points(points, edges, text_path, polygon_ranges):
    """Get points belonging to a letter (outer + holes)."""
    import numpy as np

    letter_point_indices = []
    for start_idx, end_idx in polygon_ranges:
        polygon_verts = text_path.vertices[start_idx:end_idx]
        for i, point in enumerate(points):
            point_np = point.cpu().numpy()
            distances = np.linalg.norm(polygon_verts - point_np, axis=1)
            if np.min(distances) < 0.01:
                letter_point_indices.append(i)

    letter_point_set = set(letter_point_indices)
    for edge in edges:
        p0, p1 = edge[0].item(), edge[1].item()
        if p0 in letter_point_set or p1 in letter_point_set:
            letter_point_set.add(p0)
            letter_point_set.add(p1)

    return torch.tensor(sorted(letter_point_set), dtype=torch.long)


def _triangulate(points, edges, text_path):
    """Triangulate text letter-by-letter with hole support."""
    import importlib

    import numpy as np

    mpl_path = importlib.import_module("matplotlib.path")
    mpl_tri = importlib.import_module("matplotlib.tri")

    letter_groups = _group_letters(text_path)

    all_points_list = []
    all_triangles = []
    global_offset = 0

    for group in letter_groups:
        outer = group["outer"]
        holes = group["holes"]
        all_polygon_ranges = [outer] + holes

        letter_point_indices = _get_letter_points(
            points, edges, text_path, all_polygon_ranges
        )
        if len(letter_point_indices) < 3:
            continue

        letter_points = points[letter_point_indices]
        letter_points_np = letter_points.cpu().numpy()

        tri = mpl_tri.Triangulation(letter_points_np[:, 0], letter_points_np[:, 1])

        if text_path.vertices is None or text_path.codes is None:
            continue

        combined_verts = []
        combined_codes = []
        for start_idx, end_idx in all_polygon_ranges:
            combined_verts.append(text_path.vertices[start_idx:end_idx])
            combined_codes.append(text_path.codes[start_idx:end_idx])

        combined_verts = np.vstack(combined_verts)
        combined_codes = np.hstack(combined_codes)
        letter_path = mpl_path.Path(combined_verts, combined_codes)

        centroids_np = letter_points_np[tri.triangles].mean(axis=1)
        centroids_torch = torch.tensor(centroids_np, dtype=torch.float32)
        winding = _winding_number(centroids_torch, letter_path)
        inside_mask = winding != 0

        letter_triangles = tri.triangles[inside_mask.cpu().numpy()]
        letter_triangles_global = letter_triangles + global_offset

        if len(letter_triangles_global) > 0:
            all_triangles.append(letter_triangles_global)

        all_points_list.append(letter_points)
        global_offset += len(letter_points)

    all_points = torch.cat(all_points_list, dim=0) if all_points_list else points
    triangles = (
        torch.from_numpy(np.vstack(all_triangles)).long()
        if all_triangles
        else torch.empty((0, 3), dtype=torch.long)
    )

    return all_points, triangles


@require_version_spec("matplotlib")
def text_1d_2d(
    text: str = "physicsnemo.mesh",
    font_size: float = 12.0,
    samples_per_unit: float = 10,
    max_segment_length: float = 0.25,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Render text as 1D curve in 2D space (boundary path only).

    Converts text to a polyline mesh representing the outline of each letter.

    Parameters
    ----------
    text : str, optional
        Text string to render
    font_size : float, optional
        Font size in arbitrary units
    samples_per_unit : float, optional
        Density of curve sampling for Bezier curves
    max_segment_length : float, optional
        Maximum edge length after subdivision
    device : torch.device | str, optional
        Device for mesh tensors ('cpu', 'cuda', or torch.device)

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=1, n_spatial_dims=2 (polyline in 2D)

    Examples
    --------
    >>> mesh = text_1d_2d("Hello", font_size=10.0)
    >>> assert mesh.n_manifold_dims == 1
    >>> assert mesh.n_spatial_dims == 2
    """
    if isinstance(device, str):
        device = torch.device(device)

    points, edges, _ = _text_to_path(text, font_size, samples_per_unit)
    points_refined, edges_refined = _refine_edges(points, edges, max_segment_length)

    return Mesh(
        points=points_refined.to(device),
        cells=edges_refined.to(device),
    )


@require_version_spec("matplotlib")
def text_2d_2d(
    text: str = "physicsnemo.mesh",
    font_size: float = 12.0,
    samples_per_unit: float = 10,
    max_segment_length: float = 0.25,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Render text as 2D triangulated surface in 2D space (filled letters).

    Converts text to a filled mesh with proper hole handling for letters
    like 'o', 'e', 'a'. Uses Delaunay triangulation and shoelace formula
    for hole detection.

    Parameters
    ----------
    text : str, optional
        Text string to render
    font_size : float, optional
        Font size in arbitrary units
    samples_per_unit : float, optional
        Density of curve sampling for Bezier curves
    max_segment_length : float, optional
        Maximum edge length after subdivision
    device : torch.device | str, optional
        Device for mesh tensors ('cpu', 'cuda', or torch.device)

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=2 (filled text in 2D plane)

    Examples
    --------
    >>> mesh = text_2d_2d("Hello", font_size=10.0)
    >>> assert mesh.n_manifold_dims == 2
    >>> assert mesh.n_spatial_dims == 2
    """
    if isinstance(device, str):
        device = torch.device(device)

    points, edges, text_path = _text_to_path(text, font_size, samples_per_unit)
    points_refined, edges_refined = _refine_edges(points, edges, max_segment_length)
    points_filled, triangles = _triangulate(points_refined, edges_refined, text_path)

    return Mesh(
        points=points_filled.to(device),
        cells=triangles.to(device),
    )


@require_version_spec("matplotlib")
def text_3d_3d(
    text: str = "physicsnemo.mesh",
    font_size: float = 12.0,
    samples_per_unit: float = 10,
    max_segment_length: float = 0.25,
    extrusion_height: float = 2.0,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Render text as 3D tetrahedral volume (solid extruded text).

    Creates solid 3D text by triangulating in 2D, embedding to 3D, and
    extruding along the z-axis.

    Parameters
    ----------
    text : str, optional
        Text string to render
    font_size : float, optional
        Font size in arbitrary units
    samples_per_unit : float, optional
        Density of curve sampling for Bezier curves
    max_segment_length : float, optional
        Maximum edge length after subdivision
    extrusion_height : float, optional
        Height to extrude in z-direction
    device : torch.device | str, optional
        Device for mesh tensors ('cpu', 'cuda', or torch.device)

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=3, n_spatial_dims=3 (solid tetrahedral volume)

    Examples
    --------
    >>> mesh = text_3d_3d("Hello", font_size=10.0, extrusion_height=1.0)
    >>> assert mesh.n_manifold_dims == 3
    >>> assert mesh.n_spatial_dims == 3
    """
    if isinstance(device, str):
        device = torch.device(device)

    # Create 2D mesh
    mesh_2d = text_2d_2d(
        text, font_size, samples_per_unit, max_segment_length, device="cpu"
    )

    # Embed to 3D and extrude
    mesh_3d_surface = embed(mesh_2d, target_n_spatial_dims=3)
    volume = extrude(
        mesh_3d_surface,
        vector=torch.tensor(
            [0.0, 0.0, extrusion_height], device=mesh_3d_surface.points.device
        ),
    )

    # Move to target device
    if device != mesh_2d.points.device:
        volume = Mesh(
            points=volume.points.to(device),
            cells=volume.cells.to(device),
            point_data=volume.point_data,
            cell_data=volume.cell_data,
            global_data=volume.global_data,
        )

    return volume


@require_version_spec("matplotlib")
def text_2d_3d(
    text: str = "physicsnemo.mesh",
    font_size: float = 12.0,
    samples_per_unit: float = 10,
    max_segment_length: float = 0.25,
    extrusion_height: float = 2.0,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Render text as 2D boundary surface in 3D space (hollow extruded text).

    Creates the surface of 3D text by extracting the boundary from an
    extruded tetrahedral volume.

    Parameters
    ----------
    text : str, optional
        Text string to render
    font_size : float, optional
        Font size in arbitrary units
    samples_per_unit : float, optional
        Density of curve sampling for Bezier curves
    max_segment_length : float, optional
        Maximum edge length after subdivision
    extrusion_height : float, optional
        Height to extrude in z-direction
    device : torch.device | str, optional
        Device for mesh tensors ('cpu', 'cuda', or torch.device)

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=3 (triangulated surface in 3D)

    Examples
    --------
    >>> mesh = text_2d_3d("Hello", font_size=10.0, extrusion_height=1.0)
    >>> assert mesh.n_manifold_dims == 2
    >>> assert mesh.n_spatial_dims == 3
    """
    volume = text_3d_3d(
        text, font_size, samples_per_unit, max_segment_length, extrusion_height, device
    )
    return volume.get_boundary_mesh(data_source="cells")
