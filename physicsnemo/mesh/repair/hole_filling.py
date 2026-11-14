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

"""Fill holes in triangle meshes.

Detects boundary loops (cycles of boundary edges) and closes each loop
independently: planar loops by ear clipping in their plane, others by a fan
from a centroid vertex.
"""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float, Int

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _trace_boundary_loops(
    boundary_edges: Int[torch.Tensor, "n_boundary_edges 2"],
) -> tuple[list[torch.Tensor], int]:
    """Split a set of boundary edges into simple cycles.

    Walks unused edges from vertex to vertex on CPU. When the walk returns to
    a vertex already on its path, the cycle from that vertex is cut off as one
    loop and the walk goes on, so a vertex of boundary degree above 2 (two
    holes touching at a vertex, chained slits) splits its edges between
    several loops. Edges of walks that end without closing (at a vertex with
    no unused edge) belong to no loop.

    Parameters
    ----------
    boundary_edges : torch.Tensor
        Boundary edges, shape (n_boundary_edges, 2). Each row is [v0, v1].

    Returns
    -------
    tuple[list[torch.Tensor], int]
        The loops and the number of edges in no loop. Each loop holds the
        vertex indices of a simple cycle in traversal order (on the same device
        as ``boundary_edges``); each consecutive pair shares a boundary edge,
        and the last connects back to the first.
    """
    if len(boundary_edges) == 0:
        return [], 0

    device = boundary_edges.device

    ### Remap sparse vertex indices to a compact 0..n_boundary_verts-1 range
    unique_verts, inverse = torch.unique(boundary_edges, return_inverse=True)
    n_boundary_verts = len(unique_verts)

    ### Group the edge ends by vertex: the incident edges and their other ends
    ends = inverse.reshape(-1)
    order = torch.argsort(ends, stable=True)
    offsets = torch.zeros(n_boundary_verts + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.bincount(ends, minlength=n_boundary_verts).cumsum(0)

    ### Convert to Python lists for scalar-access walks (a few bulk transfers).
    incident = (order // 2).tolist()
    other = inverse.flip(1).reshape(-1)[order].tolist()
    offsets = offsets.tolist()
    unique_verts_cpu = unique_verts.cpu()

    ### Walk loops using pure-Python indexing.
    # This walk is sequential by nature: each step depends on the path so far,
    # so it cannot be vectorized. Boundary loops are small in practice, so the
    # O(N) CPU walk is not a bottleneck.
    used = [False] * len(boundary_edges)
    next_slot = offsets[:-1]
    loops: list[torch.Tensor] = []
    n_edges_untraced = 0

    for start in range(n_boundary_verts):
        path = [start]
        position = {start: 0}
        while True:
            current = path[-1]
            slot = next_slot[current]
            while slot < offsets[current + 1] and used[incident[slot]]:
                slot += 1
            next_slot[current] = slot
            if slot == offsets[current + 1]:
                break
            used[incident[slot]] = True
            nxt = other[slot]
            if nxt in position:
                # Closed a cycle: cut it off as a loop and walk on from its start
                cut = position[nxt]
                loop_compact = torch.tensor(path[cut:], dtype=torch.long)
                loops.append(unique_verts_cpu[loop_compact].to(device))
                for v in path[cut + 1 :]:
                    del position[v]
                del path[cut + 1 :]
            else:
                position[nxt] = len(path)
                path.append(nxt)
        n_edges_untraced += len(path) - 1

    return loops, n_edges_untraced


# Not physicsnemo.mesh.tessellation.triangulate: it fans convex polygons from vertex 0,
# giving zero-area triangles at the collinear vertices that rims beside T-junction seams have.
def _ear_clip(
    xy: Float[torch.Tensor, "n_loop 2"],
) -> list[tuple[int, int, int]] | None:
    """Ear-clip a counterclockwise simple polygon into counterclockwise triangles.

    Returns triangles of local vertex indices, or ``None`` when no ear is found
    (the polygon is not simple).
    """
    pts = xy.tolist()

    def cross(i: int, j: int, k: int) -> float:
        (xi, yi), (xj, yj), (xk, yk) = pts[i], pts[j], pts[k]
        return (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)

    remaining = list(range(len(pts)))
    triangles = []
    while len(remaining) > 3:
        n = len(remaining)
        for k in range(n):
            a, b, c = remaining[k - 1], remaining[k], remaining[(k + 1) % n]
            if cross(a, b, c) <= 0:
                continue  # reflex or degenerate corner
            if any(
                cross(a, b, m) >= 0 and cross(b, c, m) >= 0 and cross(c, a, m) >= 0
                for m in remaining
                if m not in (a, b, c)
            ):
                continue  # another vertex lies in the candidate ear
            triangles.append((a, b, c))
            del remaining[k]
            break
        else:
            return None
    triangles.append((remaining[0], remaining[1], remaining[2]))
    return triangles


def fill_holes(
    mesh: "Mesh",
    max_hole_edges: int = 10,
    tolerance: float | None = None,
) -> tuple["Mesh", dict[str, int]]:
    """Fill holes bounded by boundary loops (2D manifolds only).

    Splits the boundary edges into simple cycles (a vertex shared by several
    holes splits its edges between them) and closes each loop independently,
    wound against the boundary half-edges of the faces around it so that the
    fill keeps the surface's orientation. A loop that lies in a plane (within
    a tolerance) is ear-clipped in that plane, which adds no points; any other
    loop gets a fan from a centroid vertex inserted at its center. A loop of
    zero area (a slit, which :func:`stitch_t_junctions` closes) is left open.

    In ``point_data`` and ``cell_data``, the new centroid points and fill faces
    hold NaN for floating-point fields and 0 for all other fields.

    Parameters
    ----------
    mesh : Mesh
        Input mesh (must be a 2D manifold, i.e., a triangle mesh).
    max_hole_edges : int
        Maximum number of edges in a hole to fill. Holes larger than this
        are left open. This prevents accidentally filling large openings
        that may be intentional geometry.
    tolerance : float, optional
        Distance within which a loop counts as planar, and, times its
        perimeter, the area below which it counts as a slit. Defaults to
        ``max(1e-6, 100 * eps)`` times the bounding-box diagonal, where
        ``eps`` is the machine epsilon of the point dtype.

    Returns
    -------
    tuple[Mesh, dict[str, int]]
        Tuple of (filled_mesh, stats_dict) where stats_dict contains:

        - ``"n_holes_detected"``: Total number of boundary loops found.
        - ``"n_holes_filled"``: Number of holes actually filled.
        - ``"n_holes_skipped"``: Number of holes skipped (more than
          max_hole_edges edges).
        - ``"n_holes_zero_area"``: Number of loops of zero area left open.
        - ``"n_edges_untraced"``: Number of boundary edges in no loop.
        - ``"n_faces_added"``: Total number of new triangular faces added.
        - ``"n_points_added"``: Total number of new centroid points added.

    Raises
    ------
    ValueError
        If mesh is not a 2D manifold.

    Example
    -------
    >>> from physicsnemo.mesh.primitives.surfaces import cylinder_open
    >>> mesh = cylinder_open.load()
    >>> mesh_filled, stats = fill_holes(mesh, max_hole_edges=40)
    >>> assert stats["n_holes_detected"] >= 0
    """
    if mesh.n_manifold_dims != 2:
        raise ValueError(
            f"Hole filling only implemented for 2D manifolds (triangle meshes). "
            f"Got {mesh.n_manifold_dims=}."
        )

    from physicsnemo.mesh.repair._stitching import _open_half_edges
    from physicsnemo.mesh.utilities._tolerances import default_length_tolerance

    device = mesh.points.device
    stats = {
        "n_holes_detected": 0,
        "n_holes_filled": 0,
        "n_holes_skipped": 0,
        "n_holes_zero_area": 0,
        "n_edges_untraced": 0,
        "n_faces_added": 0,
        "n_points_added": 0,
    }

    ### Step 1: Find boundary half-edges, directed as their owner faces traverse them
    boundary_edges, _, _ = _open_half_edges(mesh.cells, mesh.n_points)

    ### Step 2: Trace boundary edges into simple loops
    loops, stats["n_edges_untraced"] = _trace_boundary_loops(boundary_edges)
    stats["n_holes_detected"] = len(loops)
    if len(loops) == 0:
        return mesh, stats

    ### Step 3: Wind each loop against its owner faces' half-edges
    lengths = torch.tensor([len(loop) for loop in loops], device=device)
    flat = torch.cat(loops)
    loop_id = torch.repeat_interleave(torch.arange(len(loops), device=device), lengths)
    loop_start = torch.cumsum(lengths, 0) - lengths
    position = torch.arange(len(flat), device=device) - loop_start[loop_id]
    flat_next = flat[loop_start[loop_id] + (position + 1) % lengths[loop_id]]
    n = mesh.n_points
    is_forward = torch.isin(
        flat * n + flat_next, boundary_edges[:, 0] * n + boundary_edges[:, 1]
    )
    n_forward = torch.zeros(len(loops), dtype=torch.long, device=device)
    n_forward.scatter_add_(0, loop_id, is_forward.long())
    reverse = (2 * n_forward > lengths).tolist()
    loops = [loop.flip(0) if rev else loop for loop, rev in zip(loops, reverse)]

    ### Step 4: Fill each loop that is small enough
    if tolerance is None:
        tolerance = default_length_tolerance(mesh.points)
    new_points_list: list[torch.Tensor] = []
    new_faces_list: list[torch.Tensor] = []
    next_point_idx = mesh.n_points

    for loop_tensor in loops:
        if len(loop_tensor) > max_hole_edges:
            stats["n_holes_skipped"] += 1
            continue

        ### Vector area (Newell normal) and planarity of the loop, in 3D
        loop_points = mesh.points[loop_tensor]
        centroid = loop_points.mean(dim=0)
        rel = torch.nn.functional.pad(
            loop_points - centroid, (0, 3 - mesh.n_spatial_dims)
        )
        area_vector = 0.5 * torch.linalg.cross(rel, rel.roll(-1, dims=0)).sum(dim=0)
        area = area_vector.norm()
        perimeter = (rel - rel.roll(-1, dims=0)).norm(dim=-1).sum()
        if area <= tolerance * perimeter:
            stats["n_holes_zero_area"] += 1
            continue
        normal = area_vector / area

        ### Planar loop: ear-clip in its plane, in which it is counterclockwise
        triangles = None
        if (rel @ normal).abs().max() <= tolerance:
            axis = torch.zeros_like(normal)
            axis[normal.abs().argmin()] = 1.0
            u = torch.linalg.cross(normal, axis)
            u = u / u.norm()
            v = torch.linalg.cross(normal, u)
            triangles = _ear_clip(torch.stack([rel @ u, rel @ v], dim=1))

        if triangles is not None:
            local = torch.tensor(triangles, dtype=torch.long, device=device)
            new_faces_list.append(loop_tensor[local])
        else:
            ### Fan from the centroid: each consecutive pair + centroid
            loop_next = loop_tensor.roll(-1)
            centroid_col = torch.full_like(loop_tensor, next_point_idx)
            new_faces_list.append(
                torch.stack([loop_tensor, loop_next, centroid_col], dim=1)
            )
            new_points_list.append(centroid.unsqueeze(0))
            next_point_idx += 1
        stats["n_holes_filled"] += 1

    ### Step 5: Assemble the filled mesh
    if stats["n_holes_filled"] == 0:
        return mesh, stats

    all_new_points = torch.cat(  # (n_centroids, n_spatial_dims)
        [mesh.points[:0], *new_points_list], dim=0
    )
    all_new_faces = torch.cat(new_faces_list, dim=0)  # (n_new_faces, 3)
    n_new_points = all_new_points.shape[0]
    n_new_faces = all_new_faces.shape[0]

    new_points = torch.cat([mesh.points, all_new_points], dim=0)
    new_cells = torch.cat([mesh.cells, all_new_faces], dim=0)

    ### Step 6: Extend point_data and cell_data for the new elements
    def extend_point_data(tensor: torch.Tensor) -> torch.Tensor:
        """Extend a point_data tensor with NaN (float) or 0 (int) for new centroids."""
        if tensor.shape[0] != mesh.n_points:
            return tensor
        fill = float("nan") if tensor.dtype.is_floating_point else 0
        pad_shape = (n_new_points, *tensor.shape[1:])
        pad = torch.full(pad_shape, fill, dtype=tensor.dtype, device=device)
        return torch.cat([tensor, pad], dim=0)

    def extend_cell_data(tensor: torch.Tensor) -> torch.Tensor:
        """Extend a cell_data tensor with NaN (float) or 0 (int) for new faces."""
        if tensor.shape[0] != mesh.n_cells:
            return tensor
        fill = float("nan") if tensor.dtype.is_floating_point else 0
        pad_shape = (n_new_faces, *tensor.shape[1:])
        pad = torch.full(pad_shape, fill, dtype=tensor.dtype, device=device)
        return torch.cat([tensor, pad], dim=0)

    new_point_data = mesh.point_data.apply(
        extend_point_data, batch_size=torch.Size([new_points.shape[0]])
    )
    new_cell_data = mesh.cell_data.apply(
        extend_cell_data, batch_size=torch.Size([new_cells.shape[0]])
    )

    from physicsnemo.mesh.mesh import Mesh

    filled_mesh = Mesh(
        points=new_points,
        cells=new_cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
        global_data=mesh.global_data.clone(),
    )

    stats["n_faces_added"] = n_new_faces
    stats["n_points_added"] = n_new_points

    return filled_mesh, stats
