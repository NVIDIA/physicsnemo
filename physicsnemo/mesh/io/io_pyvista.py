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

from typing import Literal

import numpy as np
import torch

from physicsnemo.core.version_check import require_version_spec
from physicsnemo.mesh.mesh import Mesh


@require_version_spec("pyvista")
def from_pyvista(
    pyvista_mesh: "pyvista.PolyData | pyvista.UnstructuredGrid | pyvista.PointSet",  # noqa: F821
    manifold_dim: int | Literal["auto"] = "auto",
) -> Mesh:
    """Convert a PyVista mesh to a physicsnemo.mesh Mesh.

    Parameters
    ----------
    pyvista_mesh : pv.PolyData or pv.UnstructuredGrid or pv.PointSet
        Input PyVista mesh (PolyData, UnstructuredGrid, or PointSet).
    manifold_dim : int or {"auto"}
        Manifold dimension (0, 1, 2, or 3), or "auto" to detect automatically.
        - 0: Point cloud (vertices only)
        - 1: Line mesh (edge cells)
        - 2: Surface mesh (triangular cells)
        - 3: Volume mesh (tetrahedral cells)

    Returns
    -------
    Mesh
        Mesh object with converted geometry and data (on CPU).

    Raises
    ------
    ValueError
        If manifold dimension cannot be determined or is invalid.
    ImportError
        If pyvista is not installed.
    """
    import importlib

    pv = importlib.import_module("pyvista")

    ### Determine the manifold dimension
    if manifold_dim == "auto":
        # Handle PointSet (always 0D)
        if isinstance(pyvista_mesh, pv.PointSet) and not isinstance(
            pyvista_mesh, (pv.PolyData, pv.UnstructuredGrid)
        ):
            manifold_dim = 0
        else:
            # Get counts of different geometry types
            n_lines = _get_count_safely(pyvista_mesh, "n_lines")
            n_verts = _get_count_safely(pyvista_mesh, "n_verts")

            # For faces, need to handle PolyData vs UnstructuredGrid differently
            if isinstance(pyvista_mesh, pv.PolyData):
                # For PolyData, n_cells includes verts, lines, and faces
                # We need to distinguish between them
                # Faces are present when n_cells > n_verts + n_lines
                n_cells_total = _get_count_safely(pyvista_mesh, "n_cells")
                n_faces = max(0, n_cells_total - n_verts - n_lines)
            else:
                # For UnstructuredGrid, check cells_dict for 2D cells
                cells_dict = getattr(pyvista_mesh, "cells_dict", {})
                n_faces = sum(
                    len(cells)
                    for cell_type, cells in cells_dict.items()
                    if cell_type
                    in [pv.CellType.TRIANGLE, pv.CellType.QUAD, pv.CellType.POLYGON]
                )

            # Check for 3D volume cells
            cells_dict = getattr(pyvista_mesh, "cells_dict", {})
            volume_cell_types = [
                pv.CellType.TETRA,
                pv.CellType.HEXAHEDRON,
                pv.CellType.WEDGE,
                pv.CellType.PYRAMID,
                pv.CellType.VOXEL,
            ]
            n_volume_cells = sum(
                len(cells)
                for cell_type, cells in cells_dict.items()
                if cell_type in volume_cell_types
            )

            # Determine dimension based on what's present (highest dimension wins)
            if n_volume_cells > 0:
                manifold_dim = 3
            elif n_faces > 0:
                if n_lines > 0:
                    raise ValueError(
                        f"Cannot automatically determine manifold dimension.\n"
                        f"Mesh has both lines and faces: {n_lines=}, {n_faces=}.\n"
                        f"Please specify manifold_dim explicitly."
                    )
                manifold_dim = 2
            elif n_lines > 0:
                manifold_dim = 1
            else:
                # Only vertices or nothing
                manifold_dim = 0

    ### Validate manifold dimension
    if manifold_dim not in {0, 1, 2, 3}:
        raise ValueError(
            f"Invalid {manifold_dim=}. Must be one of {{0, 1, 2, 3}} or 'auto'."
        )

    ### Preprocess mesh based on manifold dimension
    if manifold_dim == 2:
        # Ensure all faces are triangles
        if not pyvista_mesh.is_all_triangles:
            pyvista_mesh = pyvista_mesh.triangulate()

    elif manifold_dim == 3:
        if not hasattr(pyvista_mesh, "cells_dict"):
            raise ValueError(
                f"Expected a `cells_dict` attribute for 3D meshes (typically pv.UnstructuredGrid), "
                f"but did not find one. For reference, got {type(pyvista_mesh)=}."
            )

        def is_all_tetra(pv_mesh) -> bool:
            """Check if mesh contains only tetrahedral cells."""
            return list(pv_mesh.cells_dict.keys()) == [pv.CellType.TETRA]

        if not is_all_tetra(pyvista_mesh):
            pyvista_mesh = pyvista_mesh.tessellate(max_n_subdivide=1)

        if not is_all_tetra(pyvista_mesh):
            cell_type_names = "\n".join(
                f"- {pv.CellType(id)}" for id in pyvista_mesh.cells_dict.keys()
            )
            raise ValueError(
                f"Expected all cells to be tetrahedra after tessellation, but got:\n{cell_type_names}"
            )

    ### Extract and convert geometry
    # Points
    points = torch.from_numpy(pyvista_mesh.points).float()

    # Cells
    if manifold_dim == 0:
        # Point cloud - no connectivity
        cells = torch.empty((0, 1), dtype=torch.long)

    elif manifold_dim == 1:
        # Lines - extract from PyVista lines format
        # PyVista stores lines as [n0, i0, i1, ..., i_{n0-1}, n1, j0, j1, ...]
        # where n is the number of points in each polyline
        # For a manifold 1D mesh, we convert polylines to line segments
        lines_raw = pyvista_mesh.lines
        if lines_raw is None or len(lines_raw) == 0:
            cells = torch.empty((0, 2), dtype=torch.long)
        else:
            # Parse the lines array and convert to line segments
            cells_list = []
            i = 0
            while i < len(lines_raw):
                n_points = lines_raw[i]
                point_ids = lines_raw[i + 1 : i + 1 + n_points]

                # Convert polyline to line segments (consecutive pairs)
                cells_list.extend(
                    [
                        [point_ids[j], point_ids[j + 1]]
                        for j in range(len(point_ids) - 1)
                    ]
                )

                i += n_points + 1

            if cells_list:
                cells = torch.from_numpy(np.array(cells_list)).long()
            else:
                cells = torch.empty((0, 2), dtype=torch.long)

    elif manifold_dim == 2:
        # Triangular cells - use regular_faces property
        # After triangulation, regular_faces returns n_cells × 3 array
        regular_faces = pyvista_mesh.regular_faces
        cells = torch.from_numpy(regular_faces).long()

    elif manifold_dim == 3:
        # Tetrahedral cells - extract from cells
        # After tessellation, all cells should be tetrahedra
        cells_dict = pyvista_mesh.cells_dict
        if pv.CellType.TETRA not in cells_dict:
            raise ValueError(
                f"Expected tetrahedral cells after tessellation, but got {list(cells_dict.keys())}"
            )
        tetra_cells = cells_dict[pv.CellType.TETRA]
        cells = torch.from_numpy(tetra_cells).long()

    ### Return Mesh object
    return Mesh(
        points=points,
        cells=cells,
        point_data=pyvista_mesh.point_data,  # type: ignore[arg-type]
        cell_data=pyvista_mesh.cell_data,  # type: ignore[arg-type]
        global_data=pyvista_mesh.field_data,  # type: ignore[arg-type]
    )


@require_version_spec("pyvista")
def to_pyvista(
    mesh: Mesh,
) -> "pyvista.PolyData | pyvista.UnstructuredGrid | pyvista.PointSet":  # noqa: F821
    """Convert a physicsnemo.mesh Mesh to a PyVista mesh.

    Parameters
    ----------
    mesh : Mesh
        Input physicsnemo.mesh Mesh object.

    Returns
    -------
    pv.PolyData or pv.UnstructuredGrid or pv.PointSet
        PyVista mesh (PointSet for 0D, PolyData for 1D/2D, UnstructuredGrid for 3D).

    Raises
    ------
    ValueError
        If manifold dimension is not supported.
    ImportError
        If pyvista is not installed.
    """
    import importlib

    pv = importlib.import_module("pyvista")

    ### Convert points to numpy and pad to 3D if needed (PyVista requires 3D points)
    points_np = mesh.points.cpu().numpy()

    if mesh.n_spatial_dims < 3:
        # Pad with zeros to make 3D
        padding_width = 3 - mesh.n_spatial_dims
        points_np = np.pad(
            points_np,
            ((0, 0), (0, padding_width)),
            mode="constant",
            constant_values=0.0,
        )

    ### Convert based on manifold dimension
    if mesh.n_manifold_dims == 0:
        # Point cloud - create PointSet
        pv_mesh = pv.PointSet(points_np)

    elif mesh.n_manifold_dims == 1:
        # Line mesh - create PolyData with lines
        # Convert line segments to PyVista format: [n_points, id0, id1, ...]
        cells_np = mesh.cells.cpu().numpy()

        if mesh.n_cells == 0:
            # Empty lines
            pv_mesh = pv.PolyData(points_np)
        else:
            # Each line segment has 2 points
            # PyVista format: [2, i0, i1, 2, j0, j1, ...]
            lines_list = []
            for cell in cells_np:
                lines_list.append(2)  # Number of points in this line
                lines_list.extend(cell)
            lines_array = np.array(lines_list, dtype=np.int64)

            pv_mesh = pv.PolyData(points_np, lines=lines_array)

    elif mesh.n_manifold_dims == 2:
        # Surface mesh - create PolyData with triangular cells
        cells_np = mesh.cells.cpu().numpy()

        if mesh.n_cells == 0:
            # Empty cells
            pv_mesh = pv.PolyData(points_np)
        else:
            # PyVista format for cells: [3, i0, i1, i2, 3, j0, j1, j2, ...]
            cells_list = []
            for cell in cells_np:
                cells_list.append(3)  # Number of points in this triangle
                cells_list.extend(cell)
            cells_array = np.array(cells_list, dtype=np.int64)

            pv_mesh = pv.PolyData(points_np, faces=cells_array)

    elif mesh.n_manifold_dims == 3:
        # Volume mesh - create UnstructuredGrid with tetrahedral cells
        cells_np = mesh.cells.cpu().numpy()

        if mesh.n_cells == 0:
            # Empty cells - create UnstructuredGrid with no cells
            cells = np.array([], dtype=np.int64)
            celltypes = np.array([], dtype=np.uint8)
            pv_mesh = pv.UnstructuredGrid(cells, celltypes, points_np)
        else:
            # PyVista format for cells: [4, i0, i1, i2, i3, 4, j0, j1, j2, j3, ...]
            cells_list = []
            for cell in cells_np:
                cells_list.append(4)  # Number of points in this tetrahedron
                cells_list.extend(cell)
            cells_array = np.array(cells_list, dtype=np.int64)

            # All cells are tetrahedra
            celltypes = np.full(mesh.n_cells, pv.CellType.TETRA, dtype=np.uint8)

            pv_mesh = pv.UnstructuredGrid(cells_array, celltypes, points_np)

    else:
        raise ValueError(f"Unsupported {mesh.n_manifold_dims=}. Must be 0, 1, 2, or 3.")

    ### Convert data dictionaries
    for k, v in mesh.point_data.items(include_nested=True, leaves_only=True):
        pv_mesh.point_data[str(k)] = v.cpu().numpy()

    for k, v in mesh.cell_data.items(include_nested=True, leaves_only=True):
        pv_mesh.cell_data[str(k)] = v.cpu().numpy()

    for k, v in mesh.global_data.items(include_nested=True, leaves_only=True):
        pv_mesh.field_data[str(k)] = v.cpu().numpy()

    return pv_mesh


def _get_count_safely(obj, attr: str) -> int:
    """Safely get count from an attribute, returning 0 if it doesn't exist or is None.

    Parameters
    ----------
    obj : object
        Object to get attribute from.
    attr : str
        Name of the attribute.

    Returns
    -------
    int
        Count value, or 0 if attribute doesn't exist or is None.
    """
    try:
        value = getattr(obj, attr, None)
        if value is None:
            return 0
        if hasattr(value, "__len__"):
            return len(value)
        return int(value) if isinstance(value, (int, float)) else 0
    except (AttributeError, TypeError):
        return 0
