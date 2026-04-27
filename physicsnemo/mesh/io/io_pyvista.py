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

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from physicsnemo.core.version_check import require_version_spec
from physicsnemo.mesh.mesh import Mesh

if TYPE_CHECKING:
    import pyvista


def _vtk_data_to_tensor_dict(data) -> dict[str, torch.Tensor]:  # noqa: ANN001
    """Convert a PyVista/VTK data container to a plain tensor dictionary."""
    tensor_data: dict[str, torch.Tensor] = {}
    for key, value in dict(data).items():
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
            continue
        tensor_data[str(key)] = torch.as_tensor(array)
    return tensor_data


@require_version_spec("pyvista")
def from_pyvista(
    pyvista_mesh: "pyvista.PolyData | pyvista.UnstructuredGrid | pyvista.PointSet",
    manifold_dim: int | Literal["auto"] = "auto",
    *,
    point_source: Literal["vertices", "cell_centroids"] = "vertices",
    warn_on_lost_data: bool = True,
    force_copy: bool = False,
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
        When ``point_source="cell_centroids"``, only 0 and 1 are valid
        (defaulting to 0 for "auto").
    point_source : {"vertices", "cell_centroids"}
        Controls what becomes the Mesh points:
        - ``"vertices"`` (default): Mesh vertices become points, ``point_data``
          is preserved. ``manifold_dim`` controls cell topology as usual.
        - ``"cell_centroids"``: Cell centroids become points, ``cell_data``
          is mapped to ``point_data``. With ``manifold_dim=0`` the result is
          a point cloud; with ``manifold_dim=1`` the result is a dual graph
          whose edges connect cells that share a face in the original mesh.
          This mode avoids expensive tetrahedralization and is suitable for
          large polyhedral meshes.
    warn_on_lost_data : bool
        If True, emit a ``UserWarning`` when the conversion discards non-empty
        data arrays. Cell data is lost when ``point_source="vertices"`` and
        ``manifold_dim`` is lower than the detected mesh dimension. Point data
        is lost when ``point_source="cell_centroids"``.
    force_copy : bool
        If True, copy point and cell arrays so the returned Mesh owns its
        memory independently of the source PyVista mesh.  When False
        (default), the returned tensors may share memory with the source
        for efficiency; mutating the Mesh's ``points`` or ``cells`` could
        then also modify the PyVista mesh.

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

    ### Validate point_source
    if point_source not in {"vertices", "cell_centroids"}:
        raise ValueError(
            f"Invalid {point_source=!r}. Must be 'vertices' or 'cell_centroids'."
        )

    ### Handle cell_centroids path (completely separate flow)
    if point_source == "cell_centroids":
        return _from_pyvista_cell_centroids(
            pyvista_mesh, manifold_dim, warn_on_lost_data
        )

    ### Determine native mesh dimension (used for auto-detection, data-loss
    ### warnings, and deciding whether cell_data can be passed through).
    native_dim = _detect_native_dim(pyvista_mesh, pv)

    if manifold_dim == "auto":
        if isinstance(pyvista_mesh, pv.PointSet) and not isinstance(
            pyvista_mesh, (pv.PolyData, pv.UnstructuredGrid)
        ):
            manifold_dim = 0
        else:
            manifold_dim = native_dim
            # PolyData can mix verts, lines, and faces in a single mesh.
            # Reject cases where both lines and surface cells coexist,
            # since the intended dimension is ambiguous.
            if manifold_dim == 2:
                n_lines = _get_count_safely(pyvista_mesh, "n_lines")
                if n_lines > 0:
                    raise ValueError(
                        f"Cannot automatically determine manifold dimension.\n"
                        f"Mesh has both lines and faces: {n_lines=}.\n"
                        f"Please specify manifold_dim explicitly."
                    )

    ### Validate manifold dimension
    if manifold_dim not in {0, 1, 2, 3}:
        raise ValueError(
            f"Invalid {manifold_dim=}. Must be one of {{0, 1, 2, 3}} or 'auto'."
        )

    ### Warn about data that will be dropped
    if warn_on_lost_data:
        _warn_on_data_loss(
            pyvista_mesh,
            point_source="vertices",
            manifold_dim=manifold_dim,
            detected_dim=native_dim,
        )

    ### Preprocess mesh based on manifold dimension
    if manifold_dim == 2:
        if isinstance(pyvista_mesh, pv.PolyData):
            all_triangles = pyvista_mesh.is_all_triangles
        elif isinstance(pyvista_mesh, pv.UnstructuredGrid):
            all_triangles = bool((pyvista_mesh.celltypes == pv.CellType.TRIANGLE).all())
        else:
            raise NotImplementedError(
                f"Only PolyData and UnstructuredGrid are supported for manifold dimension 2, got {type(pyvista_mesh)=}."
            )
        if not all_triangles:
            pyvista_mesh = pyvista_mesh.triangulate()

    elif manifold_dim == 3:
        if not isinstance(pyvista_mesh, pv.UnstructuredGrid):
            raise ValueError(
                f"Expected an UnstructuredGrid with volume cells for 3D meshes, "
                f"but got {type(pyvista_mesh)=}."
            )

        def is_all_tetra(pv_mesh) -> bool:
            """Check if mesh contains only tetrahedral cells."""
            return bool((pv_mesh.celltypes == pv.CellType.TETRA).all())

        if not is_all_tetra(pyvista_mesh):
            pyvista_mesh = pyvista_mesh.triangulate()

        if not is_all_tetra(pyvista_mesh):
            unique_types = np.unique(pyvista_mesh.celltypes)
            cell_type_names = "\n".join(f"- {pv.CellType(t)}" for t in unique_types)
            raise ValueError(
                f"Expected all cells to be tetrahedra after triangulation, but got:\n{cell_type_names}"
            )

    ### Extract and convert geometry
    def _maybe_copy(arr: np.ndarray) -> np.ndarray:
        return arr.copy() if force_copy else arr

    # Points
    points = torch.from_numpy(_maybe_copy(pyvista_mesh.points)).float()

    # Cells
    if manifold_dim == 0:
        cells = None  # Mesh constructor creates empty cells

    elif manifold_dim == 1:
        # Lines - extract from PyVista lines format.
        # If the mesh has no native lines (e.g., a 3D volume mesh with
        # manifold_dim=1 requested explicitly), extract all unique edges
        # from the mesh topology to build a vertex graph.
        lines_raw = getattr(pyvista_mesh, "lines", None)
        if (lines_raw is None or len(lines_raw) == 0) and pyvista_mesh.n_cells > 0:
            edges_mesh = pyvista_mesh.extract_all_edges()
            lines_raw = edges_mesh.lines

        if lines_raw is None or len(lines_raw) == 0:
            cells = torch.empty((0, 2), dtype=torch.long)
        else:
            lines_array = np.asarray(lines_raw)

            # Fast path: check if all line segments have uniform vertex count
            # (common case — all edges have 2 vertices, stride = 3)
            first_count = int(lines_array[0])
            stride = first_count + 1
            is_uniform = len(lines_array) % stride == 0 and len(lines_array) >= stride
            if is_uniform:
                n_segments = len(lines_array) // stride
                reshaped = lines_array.reshape(n_segments, stride)
                is_uniform = bool((reshaped[:, 0] == first_count).all())

            if is_uniform:
                # Vectorized path: reshape and extract vertex columns
                point_ids = reshaped[:, 1:]  # (n_segments, first_count)

                # Convert polylines to consecutive line segments
                if first_count == 2:
                    # Already line segments — use directly
                    cells = torch.from_numpy(point_ids.copy()).long()
                else:
                    # Polylines with >2 vertices: create consecutive pairs
                    seg_starts = point_ids[:, :-1].reshape(-1)
                    seg_ends = point_ids[:, 1:].reshape(-1)
                    cells = torch.stack(
                        [
                            torch.from_numpy(seg_starts.copy()),
                            torch.from_numpy(seg_ends.copy()),
                        ],
                        dim=1,
                    ).long()
            else:
                # Fallback: Python loop for non-uniform segment sizes
                cells_list = []
                i = 0
                while i < len(lines_array):
                    n_pts = int(lines_array[i])
                    point_ids = lines_array[i + 1 : i + 1 + n_pts]

                    # Convert polyline to line segments (consecutive pairs)
                    cells_list.extend(
                        [
                            [point_ids[j], point_ids[j + 1]]
                            for j in range(len(point_ids) - 1)
                        ]
                    )

                    i += n_pts + 1

                if cells_list:
                    cells = torch.from_numpy(np.array(cells_list)).long()
                else:
                    cells = torch.empty((0, 2), dtype=torch.long)

    elif manifold_dim == 2:
        # After triangulation, extract the (n_cells, 3) connectivity array
        if isinstance(pyvista_mesh, pv.PolyData):
            tri_faces = _maybe_copy(pyvista_mesh.regular_faces)
        elif isinstance(pyvista_mesh, pv.UnstructuredGrid):
            tri_faces = pyvista_mesh.cells_dict[np.uint8(pv.CellType.TRIANGLE)]
        else:
            raise NotImplementedError(
                f"Only PolyData and UnstructuredGrid are supported for manifold dimension 2, got {type(pyvista_mesh)=}."
            )
        cells = torch.from_numpy(tri_faces).long()

    elif manifold_dim == 3:
        # Tetrahedral cells - extract from cells
        # After triangulation, all cells should be tetrahedra
        cells_dict = pyvista_mesh.cells_dict
        if pv.CellType.TETRA not in cells_dict:
            raise ValueError(
                f"Expected tetrahedral cells after triangulation, but got {list(cells_dict.keys())}"
            )
        tetra_cells = cells_dict[np.uint8(pv.CellType.TETRA)]
        cells = torch.from_numpy(tetra_cells).long()

    ### Return Mesh object
    # Cell data can only be passed through when the output cells have a
    # 1:many relationship with input cells (e.g., VTK's triangulate
    # replicates cell_data to child cells).  This fails when manifold_dim
    # is lower than native_dim (different cell topology), when
    # manifold_dim is 0 (output has no cells at all), or when a topology
    # transformation (polyline splitting, edge extraction) changed the
    # cell count.
    n_output_cells = 0 if cells is None else cells.shape[0]
    pass_cell_data = (
        manifold_dim > 0
        and manifold_dim >= native_dim
        and n_output_cells == pyvista_mesh.n_cells
    )
    return Mesh(
        points=points,
        cells=cells,
        point_data=_vtk_data_to_tensor_dict(pyvista_mesh.point_data),
        cell_data=_vtk_data_to_tensor_dict(pyvista_mesh.cell_data)
        if pass_cell_data
        else {},
        global_data=_vtk_data_to_tensor_dict(pyvista_mesh.field_data),
    )


@require_version_spec("pyvista")
def to_pyvista(
    mesh: Mesh,
) -> "pyvista.PolyData | pyvista.UnstructuredGrid | pyvista.PointSet":
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
    points_np = mesh.points.float().cpu().numpy()

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
        pv_mesh = pv.PointSet(points_np)

    elif mesh.n_manifold_dims == 1:
        cells_np = mesh.cells.cpu().numpy()
        if mesh.n_cells == 0:
            pv_mesh = pv.PolyData(points_np)
        else:
            pv_mesh = pv.PolyData(points_np, lines=_to_vtk_cell_array(cells_np))

    elif mesh.n_manifold_dims == 2:
        cells_np = mesh.cells.cpu().numpy()
        if mesh.n_cells == 0:
            pv_mesh = pv.PolyData(points_np)
        else:
            pv_mesh = pv.PolyData.from_regular_faces(points_np, cells_np)

    elif mesh.n_manifold_dims == 3:
        cells_np = mesh.cells.cpu().numpy()
        if mesh.n_cells == 0:
            pv_mesh = pv.UnstructuredGrid(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.uint8),
                points_np,
            )
        else:
            celltypes = np.full(mesh.n_cells, pv.CellType.TETRA, dtype=np.uint8)
            pv_mesh = pv.UnstructuredGrid(
                _to_vtk_cell_array(cells_np), celltypes, points_np
            )

    else:
        raise ValueError(f"Unsupported {mesh.n_manifold_dims=}. Must be 0, 1, 2, or 3.")

    ### Copy data to PyVista (flatten high-rank tensors for VTK compatibility)
    for source, target in [
        (mesh.point_data, pv_mesh.point_data),
        (mesh.cell_data, pv_mesh.cell_data),
        (mesh.global_data, pv_mesh.field_data),
    ]:
        for k, v in source.items(include_nested=True, leaves_only=True):
            arr = v.float().cpu().numpy()
            target[str(k)] = arr.reshape(arr.shape[0], -1) if arr.ndim > 2 else arr

    return pv_mesh


def _from_pyvista_cell_centroids(
    pyvista_mesh: "pyvista.PolyData | pyvista.UnstructuredGrid",
    manifold_dim: int | Literal["auto"],
    warn_on_lost_data: bool,
) -> Mesh:
    """Build a Mesh from cell centroids, mapping cell_data to point_data.

    Parameters
    ----------
    pyvista_mesh : pv.PolyData or pv.UnstructuredGrid
        Input PyVista mesh.
    manifold_dim : int or {"auto"}
        0 for a point cloud, 1 for a dual graph (cell-neighbor edges).
        "auto" resolves to 0.
    warn_on_lost_data : bool
        Emit a warning if non-empty point_data will be discarded.

    Returns
    -------
    Mesh
        Mesh whose points are the cell centroids.
    """
    if manifold_dim == "auto":
        manifold_dim = 0
    if manifold_dim not in {0, 1}:
        raise ValueError(
            f"point_source='cell_centroids' only supports manifold_dim in {{0, 1}}, "
            f"got {manifold_dim=}."
        )

    if warn_on_lost_data:
        _warn_on_data_loss(
            pyvista_mesh,
            point_source="cell_centroids",
            manifold_dim=manifold_dim,
            detected_dim=None,
        )

    ### Compute cell centroids (fast C++ filter, works for all cell types)
    centroids_np = pyvista_mesh.cell_centers().points
    points = torch.from_numpy(centroids_np.copy()).float()

    ### Build cells
    if manifold_dim == 0:
        cells = None  # Mesh constructor creates empty cells
    else:
        # Dual graph: edges connect cells that share a face.
        cells = _build_dual_graph_edges(pyvista_mesh)

    return Mesh(
        points=points,
        cells=cells,
        point_data=_vtk_data_to_tensor_dict(pyvista_mesh.cell_data),
        global_data=_vtk_data_to_tensor_dict(pyvista_mesh.field_data),
    )


def _to_vtk_cell_array(cells_np: np.ndarray) -> np.ndarray:
    """Prepend per-cell vertex counts to a regular connectivity array.

    Converts an ``(n_cells, n_verts_per_cell)`` array into the flat
    VTK cell-array format ``[n_verts, v0, v1, ..., n_verts, v0, ...]``.

    Parameters
    ----------
    cells_np : np.ndarray
        Shape ``(n_cells, n_verts_per_cell)``.

    Returns
    -------
    np.ndarray
        Flattened 1-D array of dtype ``int64``.
    """
    n_verts = cells_np.shape[1]
    return np.column_stack(
        [np.full(len(cells_np), n_verts, dtype=np.int64), cells_np]
    ).ravel()


@require_version_spec("vtk")
def _build_dual_graph_edges(
    pyvista_mesh: "pyvista.PolyData | pyvista.UnstructuredGrid",
) -> torch.Tensor:
    """Build (n_edges, 2) tensor of cell-neighbor pairs sharing a face.

    Iterates over every cell and its faces, using VTK's cell links for
    O(1) per-face neighbor lookups.  VTK objects are reused across
    iterations and results are written directly to chunked numpy buffers
    to minimize Python-level overhead (~10x faster than the equivalent
    PyVista ``cell_neighbors`` wrapper).  The overall cost is one pass
    over all cells and their faces; for very large meshes (>10M cells)
    this may still take minutes.

    Parameters
    ----------
    pyvista_mesh : pv.PolyData or pv.UnstructuredGrid
        Input mesh with cell connectivity.

    Returns
    -------
    torch.Tensor
        Shape ``(n_edges, 2)`` with dtype ``torch.long``.
    """
    import importlib

    _vtk = importlib.import_module("vtk")

    pyvista_mesh.BuildLinks()
    n_cells = pyvista_mesh.n_cells

    if n_cells == 0:
        return torch.empty((0, 2), dtype=torch.long)

    face_pt_ids = _vtk.vtkIdList()
    nbr_ids = _vtk.vtkIdList()

    # Collect upper-triangular neighbor pairs into chunked numpy buffers.
    _CHUNK = 1 << 20
    chunks: list[np.ndarray] = []
    buf = np.empty((_CHUNK, 2), dtype=np.int64)
    idx = 0

    for i in range(n_cells):
        cell = pyvista_mesh.GetCell(i)
        for f in range(cell.GetNumberOfFaces()):
            face = cell.GetFace(f)
            face_pt_ids.Reset()
            for p in range(face.GetNumberOfPoints()):
                face_pt_ids.InsertNextId(face.GetPointId(p))

            nbr_ids.Reset()
            pyvista_mesh.GetCellNeighbors(i, face_pt_ids, nbr_ids)

            for k in range(nbr_ids.GetNumberOfIds()):
                j = nbr_ids.GetId(k)
                if j > i:
                    buf[idx, 0] = i
                    buf[idx, 1] = j
                    idx += 1
                    if idx == _CHUNK:
                        chunks.append(buf.copy())
                        idx = 0

    if idx > 0:
        chunks.append(buf[:idx].copy())

    if not chunks:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.from_numpy(np.concatenate(chunks, axis=0))


def _detect_native_dim(pyvista_mesh, pv) -> int:  # noqa: ANN001
    """Determine the native manifold dimension of a PyVista mesh.

    This is a lightweight check (no cell processing) used for data-loss
    warnings and for deciding whether cell_data can be passed through.

    Parameters
    ----------
    pyvista_mesh : PyVista mesh
        Input mesh.
    pv : module
        The lazily-imported pyvista module.

    Returns
    -------
    int
        0, 1, 2, or 3.
    """
    if pyvista_mesh.n_cells == 0:
        return 0
    if hasattr(pyvista_mesh, "celltypes"):
        ct = pyvista_mesh.celltypes
        volume_types = [
            pv.CellType.TETRA,
            pv.CellType.HEXAHEDRON,
            pv.CellType.WEDGE,
            pv.CellType.PYRAMID,
            pv.CellType.VOXEL,
            pv.CellType.POLYHEDRON,
        ]
        if np.isin(ct, volume_types).any():
            return 3
        surface_types = [pv.CellType.TRIANGLE, pv.CellType.QUAD, pv.CellType.POLYGON]
        if np.isin(ct, surface_types).any():
            return 2
        line_types = [pv.CellType.LINE, pv.CellType.POLY_LINE]
        if np.isin(ct, line_types).any():
            return 1
        return 0
    # PolyData without celltypes - check for faces vs lines
    n_lines = _get_count_safely(pyvista_mesh, "n_lines")
    n_cells = _get_count_safely(pyvista_mesh, "n_cells")
    n_verts = _get_count_safely(pyvista_mesh, "n_verts")
    if n_cells > n_verts + n_lines:
        return 2
    if n_lines > 0:
        return 1
    return 0


def _warn_on_data_loss(
    pyvista_mesh: "pyvista.PolyData | pyvista.UnstructuredGrid | pyvista.PointSet",
    point_source: str,
    manifold_dim: int,
    detected_dim: int | None,
) -> None:
    """Emit UserWarning if non-empty data arrays will be discarded.

    Parameters
    ----------
    pyvista_mesh : PyVista mesh
        The input mesh (before any preprocessing).
    point_source : str
        ``"vertices"`` or ``"cell_centroids"``.
    manifold_dim : int
        The resolved (non-"auto") target manifold dimension.
    detected_dim : int or None
        The auto-detected manifold dimension of the original mesh.
        ``None`` when called from the cell_centroids path.
    """
    ### Case 1: point_data lost when using cell centroids
    if point_source == "cell_centroids":
        pd_keys = list(pyvista_mesh.point_data.keys())
        if pd_keys:
            warnings.warn(
                f"point_source='cell_centroids' discards {len(pd_keys)} point_data "
                f"field(s) from the input mesh: {pd_keys}. "
                f"Use point_source='vertices' to preserve point_data, "
                f"or set warn_on_lost_data=False to silence this warning.",
                UserWarning,
                stacklevel=3,
            )

    ### Case 2: cell_data lost when extracting lower-dimensional representation
    if point_source == "vertices" and detected_dim is not None:
        if manifold_dim < detected_dim:
            cd_keys = list(pyvista_mesh.cell_data.keys())
            if cd_keys:
                warnings.warn(
                    f"manifold_dim={manifold_dim} (< detected {detected_dim}) with "
                    f"point_source='vertices' discards {len(cd_keys)} cell_data "
                    f"field(s) from the input mesh: {cd_keys}. "
                    f"Use point_source='cell_centroids' to preserve cell_data "
                    f"as point_data, or set warn_on_lost_data=False to silence "
                    f"this warning.",
                    UserWarning,
                    stacklevel=3,
                )


def _get_count_safely(obj, attr: str) -> int:
    """Return an integer-valued attribute, or 0 if it doesn't exist.

    Parameters
    ----------
    obj : object
        Object to get attribute from.
    attr : str
        Name of the attribute (e.g. ``"n_lines"``, ``"n_verts"``).

    Returns
    -------
    int
        Attribute value cast to int, or 0 if absent/None.
    """
    value = getattr(obj, attr, None)
    return int(value) if value is not None else 0
