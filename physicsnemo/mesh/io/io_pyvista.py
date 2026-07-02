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
from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from jaxtyping import Int

from physicsnemo.core.version_check import OptionalImport, require_version_spec
from physicsnemo.mesh.mesh import Mesh

### Optional dependencies. Construction does not import the package; the
### nicely-formatted ``ImportError`` (with the ``[mesh-extras]`` install hint)
### fires only on first attribute access on ``pv`` / ``vtk``. The
### ``@require_version_spec`` decorators on the public entry points raise
### that same error proactively, before any function-body work happens.
if TYPE_CHECKING:
    import pyvista as pv
    import vtk
else:
    pv = OptionalImport("pyvista")
    vtk = OptionalImport("vtk")


def _vtk_data_to_tensor_dict(data) -> dict[str, torch.Tensor]:  # noqa: ANN001
    """Convert a PyVista/VTK data container to a plain tensor dictionary."""
    tensor_data: dict[str, torch.Tensor] = {}
    for key, value in dict(data).items():
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
            continue
        tensor_data[str(key)] = torch.as_tensor(array)
    return tensor_data


def _tensor_to_vtk_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor data without narrowing dtypes supported by PyVista."""
    tensor = tensor.detach().cpu()
    # VTK has no native real type below float32. PyVista represents complex
    # values with two real components, but likewise only supports complex64
    # and complex128 inputs.
    if tensor.is_floating_point() and tensor.element_size() < 4:
        tensor = tensor.to(dtype=torch.float32)
    elif tensor.is_complex() and tensor.element_size() < 8:
        tensor = tensor.to(dtype=torch.complex64)
    return tensor.resolve_conj().resolve_neg().numpy()


@require_version_spec("pyvista")
def from_pyvista(
    pyvista_mesh: "pv.PolyData | pv.UnstructuredGrid | pv.PointSet",
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
          whose edges connect cells that share a facet (an edge for surface
          meshes, a face for volume meshes) in the original mesh. This mode
          avoids expensive tetrahedralization and is suitable for large
          polyhedral meshes.
    warn_on_lost_data : bool
        If True, emit a ``UserWarning`` when the conversion discards non-empty
        data arrays. Cell data is lost when native cells of other dimensions
        are discarded, and all cell data is lost for a 0D output. Point data is
        lost when ``point_source="cell_centroids"``.
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
    native_dims = _detect_native_dims(pyvista_mesh)
    native_dim = max(native_dims)

    if manifold_dim == "auto":
        if isinstance(pyvista_mesh, pv.PointSet) and not isinstance(
            pyvista_mesh, (pv.PolyData, pv.UnstructuredGrid)
        ):
            manifold_dim = 0
        else:
            manifold_dim = native_dim
            # Choosing one dimension from a mixed-dimensional VTK dataset
            # silently discards native cells and their data. Require an
            # explicit target dimension instead.
            if len(native_dims) > 1:
                raise ValueError(
                    "Cannot automatically determine manifold dimension.\n"
                    f"Mesh contains native cells with dimensions {sorted(native_dims)}.\n"
                    "Please specify manifold_dim explicitly."
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
            detected_dims=native_dims,
        )

    def _maybe_copy(arr: np.ndarray) -> np.ndarray:
        return arr.copy() if force_copy else arr

    def _linearize_native_cells(cell_type: int) -> tuple[np.ndarray, dict]:
        """Select one native dimension, linearize it, and retain point IDs."""
        if isinstance(pyvista_mesh, pv.PolyData):
            grid = pyvista_mesh.cast_to_unstructured_grid()
        elif isinstance(pyvista_mesh, pv.UnstructuredGrid):
            grid = pyvista_mesh
        else:
            raise NotImplementedError(
                "Native-cell conversion requires PolyData or UnstructuredGrid, "
                f"got {type(pyvista_mesh)=}."
            )

        target_dim = int(vtk.vtkCellTypeUtilities.GetDimension(int(cell_type)))
        native_cell_types = grid.celltypes
        target_types = [
            native_type
            for native_type in np.unique(native_cell_types)
            if vtk.vtkCellTypeUtilities.GetDimension(int(native_type)) == target_dim
        ]
        target_mask = np.isin(native_cell_types, target_types)
        if not target_mask.any():
            raise ValueError(
                f"Input mesh has no cells that can be converted to manifold_dim={manifold_dim}."
            )

        # Triangulating a mixed-dimensional grid can needlessly expand millions
        # of unrelated volume cells. Extract only the requested native
        # dimension first. Extraction compacts points, so carry VTK's original
        # point IDs through triangulation and map connectivity back below.
        remap_to_original_points = not target_mask.all()
        if not remap_to_original_points:
            selected = grid
        else:
            selected = grid.extract_cells(
                target_mask, pass_cell_ids=False, pass_point_ids=True
            )

        if bool((selected.celltypes == cell_type).all()):
            linearized = selected
        else:
            linearized = selected.triangulate()

        output_mask = linearized.celltypes == cell_type
        if not output_mask.any():
            raise ValueError(
                f"Input mesh has no cells that can be converted to manifold_dim={manifold_dim}."
            )
        if not output_mask.all():
            remaining_types = np.unique(linearized.celltypes[~output_mask])
            cell_type_names = ", ".join(
                str(pv.CellType(native_type)) for native_type in remaining_types
            )
            raise ValueError(
                f"Could not linearize every manifold_dim={manifold_dim} cell; "
                f"remaining cell types: {cell_type_names}."
            )

        connectivity = linearized.cells_dict[np.uint8(cell_type)]
        if remap_to_original_points:
            original_point_ids = np.asarray(
                linearized.point_data["vtkOriginalPointIds"]
            )
            connectivity = original_point_ids[connectivity]
        data = {
            key: np.asarray(value) for key, value in dict(linearized.cell_data).items()
        }
        return connectivity, data

    cell_data: dict[str, torch.Tensor] = {}
    if manifold_dim == 0:
        cells = None  # Mesh constructor creates empty cells

    elif manifold_dim == 1:
        if pyvista_mesh.n_cells == 0:
            cells = torch.empty((0, 2), dtype=torch.long)
        elif 1 in native_dims:
            # Fast paths avoid a VTK filter for already-linear line meshes.
            if isinstance(pyvista_mesh, pv.UnstructuredGrid) and bool(
                (pyvista_mesh.celltypes == pv.CellType.LINE).all()
            ):
                line_cells = pyvista_mesh.cells_dict[np.uint8(pv.CellType.LINE)]
                cell_data = _vtk_data_to_tensor_dict(pyvista_mesh.cell_data)
            elif isinstance(pyvista_mesh, pv.PolyData) and native_dims == {1}:
                lines = np.asarray(pyvista_mesh.lines)
                first_count = int(lines[0])
                stride = first_count + 1
                is_uniform = (
                    first_count >= 2
                    and len(lines) >= stride
                    and len(lines) % stride == 0
                )
                if is_uniform:
                    reshaped = lines.reshape(-1, stride)
                    is_uniform = bool((reshaped[:, 0] == first_count).all())

                if is_uniform:
                    point_ids = reshaped[:, 1:]
                    if first_count == 2:
                        line_cells = point_ids
                        selected_data = pyvista_mesh.cell_data
                    else:
                        line_cells = np.stack(
                            (point_ids[:, :-1], point_ids[:, 1:]), axis=-1
                        ).reshape(-1, 2)
                        selected_data = {
                            key: np.repeat(np.asarray(value), first_count - 1, axis=0)
                            for key, value in dict(pyvista_mesh.cell_data).items()
                        }
                    cell_data = _vtk_data_to_tensor_dict(selected_data)
                else:
                    line_cells, selected_data = _linearize_native_cells(
                        pv.CellType.LINE
                    )
                    cell_data = _vtk_data_to_tensor_dict(selected_data)
            else:
                line_cells, selected_data = _linearize_native_cells(pv.CellType.LINE)
                cell_data = _vtk_data_to_tensor_dict(selected_data)
            cells = torch.from_numpy(_maybe_copy(line_cells)).long()
        else:
            # No native lines: derive the unique edge graph from higher-
            # dimensional topology. Cell data has no unambiguous mapping to
            # shared edges, so it is intentionally dropped below.
            edge_grid = pyvista_mesh.extract_all_edges().cast_to_unstructured_grid()
            edge_cells = edge_grid.cells_dict.get(
                np.uint8(pv.CellType.LINE), np.empty((0, 2), dtype=np.int64)
            )
            cells = torch.from_numpy(edge_cells.copy()).long()

    elif manifold_dim == 2:
        if 2 not in native_dims:
            raise ValueError("Input mesh does not contain native 2D cells.")
        if isinstance(pyvista_mesh, pv.PolyData) and pyvista_mesh.is_all_triangles:
            tri_faces = pyvista_mesh.regular_faces
            cell_data = _vtk_data_to_tensor_dict(pyvista_mesh.cell_data)
        elif isinstance(pyvista_mesh, pv.PolyData) and native_dims == {2}:
            # Retain PyVista's direct PolyData fast path for ordinary polygon
            # surfaces. Casting first to an UnstructuredGrid roughly doubles
            # conversion time on large all-quad meshes.
            triangulated = pyvista_mesh.triangulate()
            if not triangulated.is_all_triangles:
                raise ValueError("Could not triangulate every native 2D cell.")
            tri_faces = triangulated.regular_faces
            cell_data = _vtk_data_to_tensor_dict(triangulated.cell_data)
        elif isinstance(pyvista_mesh, pv.UnstructuredGrid) and bool(
            (pyvista_mesh.celltypes == pv.CellType.TRIANGLE).all()
        ):
            tri_faces = pyvista_mesh.cells_dict[np.uint8(pv.CellType.TRIANGLE)]
            cell_data = _vtk_data_to_tensor_dict(pyvista_mesh.cell_data)
        else:
            tri_faces, selected_data = _linearize_native_cells(pv.CellType.TRIANGLE)
            cell_data = _vtk_data_to_tensor_dict(selected_data)
        cells = torch.from_numpy(_maybe_copy(tri_faces)).long()

    elif manifold_dim == 3:
        if not isinstance(pyvista_mesh, pv.UnstructuredGrid):
            raise ValueError(
                "Expected an UnstructuredGrid with volume cells for 3D meshes, "
                f"but got {type(pyvista_mesh)=}."
            )
        if 3 not in native_dims:
            raise ValueError("Input mesh does not contain native 3D cells.")
        if bool((pyvista_mesh.celltypes == pv.CellType.TETRA).all()):
            tetra_cells = pyvista_mesh.cells_dict[np.uint8(pv.CellType.TETRA)]
            cell_data = _vtk_data_to_tensor_dict(pyvista_mesh.cell_data)
        else:
            tetra_cells, selected_data = _linearize_native_cells(pv.CellType.TETRA)
            cell_data = _vtk_data_to_tensor_dict(selected_data)
        cells = torch.from_numpy(_maybe_copy(tetra_cells)).long()

    points = torch.from_numpy(_maybe_copy(pyvista_mesh.points)).float()
    return Mesh(
        points=points,
        cells=cells,
        point_data=_vtk_data_to_tensor_dict(pyvista_mesh.point_data),
        cell_data=cell_data,
        global_data=_vtk_data_to_tensor_dict(pyvista_mesh.field_data),
    )


@require_version_spec("pyvista")
def to_pyvista(
    mesh: Mesh,
) -> "pv.PolyData | pv.UnstructuredGrid | pv.PointSet":
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
    ### Convert points to numpy and pad to 3D if needed (PyVista requires 3D points)
    # .detach() first so a grad-tracked mesh can still be exported (.numpy() would
    # otherwise raise on a tensor that requires grad).
    points_np = mesh.points.detach().float().cpu().numpy()

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
            arr = _tensor_to_vtk_numpy(v)
            target[str(k)] = arr.reshape(arr.shape[0], -1) if arr.ndim > 2 else arr

    return pv_mesh


def _from_pyvista_cell_centroids(
    pyvista_mesh: "pv.PolyData | pv.UnstructuredGrid",
    manifold_dim: int | Literal["auto"],
    warn_on_lost_data: bool,
) -> Mesh:
    """Build a Mesh from cell centroids, mapping cell_data to point_data.

    Parameters
    ----------
    pyvista_mesh : pv.PolyData or pv.UnstructuredGrid
        Input PyVista mesh.
    manifold_dim : int or {"auto"}
        0 for a point cloud, 1 for a dual graph (edges between cells that
        share a (d-1)-facet). "auto" resolves to 0.
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
            detected_dims=None,
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


def _cell_facet_point_ids(cell: "vtk.vtkCell") -> Iterator[list[int]]:
    """Yield the point-id lists of a cell's (d-1)-facets (dimension-generic).

    A volume cell's facets are its 2D faces, a surface cell's facets are its
    edges (1-faces), and a line cell's facets are its endpoint vertices. Two
    cells are adjacent across a shared facet, so these are precisely the facets
    that define dual-graph edges in any dimension.

    Parameters
    ----------
    cell : vtk.vtkCell
        A VTK cell.

    Yields
    ------
    list[int]
        Point ids of one (d-1)-facet. Nothing is yielded for 0D cells
        (isolated points have no facets, hence no adjacency).

    Notes
    -----
    Facets are yielded in VTK's canonical per-cell-type order, and the point
    ids within each facet follow VTK's canonical winding; both are
    deterministic. The sole consumer, :func:`_build_dual_graph_edges`, passes
    these ids to ``vtkDataSet.GetCellNeighbors``, which matches cells
    containing the full point *set* and is therefore insensitive to facet
    ordering and to point order within a facet.
    """
    # VTK cell dimensions are bounded to {0, 1, 2, 3}, so matching the
    # exact dimension is equivalent to the previous ``dim >= 3`` guard.
    match cell.GetCellDimension():
        case 3:  # Volume cell: facets are its 2D faces.
            subcells = (cell.GetFace(f) for f in range(cell.GetNumberOfFaces()))
        case 2:  # Surface cell: facets are its edges (1-faces).
            subcells = (cell.GetEdge(e) for e in range(cell.GetNumberOfEdges()))
        case 1:  # Line cell: facets are its two endpoint vertices (0-faces).
            for p in range(cell.GetNumberOfPoints()):
                yield [cell.GetPointId(p)]
            return
        case _:  # 0D (or anything unexpected): isolated points have no facets.
            return
    for sub in subcells:
        yield [sub.GetPointId(p) for p in range(sub.GetNumberOfPoints())]


@require_version_spec("vtk")
def _build_dual_graph_edges(
    pyvista_mesh: "pv.PolyData | pv.UnstructuredGrid",
) -> Int[torch.Tensor, "n_edges 2"]:
    """Build (n_edges, 2) tensor of cell-neighbor pairs sharing a (d-1)-facet.

    Two cells are adjacent (joined by a dual-graph edge) when they share a
    facet: a 2D face for volume cells, an edge for surface cells, or a vertex
    for line cells (see :func:`_cell_facet_point_ids`).  Iterates over every
    cell and its facets, using VTK's cell links for O(1) per-facet neighbor
    lookups.  VTK objects are reused across iterations and results are written
    directly to chunked numpy buffers to minimize Python-level overhead
    (~10x faster than the equivalent PyVista ``cell_neighbors`` wrapper).  The
    overall cost is one pass over all cells and their facets; for very large
    meshes (>10M cells) this may still take minutes.  A fully vectorized
    facet-hashing pass (sorting each cell's facets and matching duplicates) is
    ~6-10x faster again, but only for homogeneous, manifold meshes; the VTK
    ``GetCellNeighbors`` path is kept here because it also handles mixed cell
    types, polyhedra, and non-manifold facets generically.

    Parameters
    ----------
    pyvista_mesh : pv.PolyData or pv.UnstructuredGrid
        Input mesh with cell connectivity.

    Returns
    -------
    torch.Tensor
        Shape ``(n_edges, 2)`` with dtype ``torch.long``.
    """
    pyvista_mesh.BuildLinks()
    n_cells = pyvista_mesh.n_cells

    if n_cells == 0:
        return torch.empty((0, 2), dtype=torch.long)

    facet_pt_ids = vtk.vtkIdList()
    nbr_ids = vtk.vtkIdList()

    # Collect upper-triangular neighbor pairs into chunked numpy buffers.
    _CHUNK = 1 << 20
    chunks: list[np.ndarray] = []
    buf = np.empty((_CHUNK, 2), dtype=np.int64)
    idx = 0

    for i in range(n_cells):
        cell = pyvista_mesh.GetCell(i)
        for facet_ids in _cell_facet_point_ids(cell):
            facet_pt_ids.Reset()
            for point_id in facet_ids:
                facet_pt_ids.InsertNextId(point_id)

            nbr_ids.Reset()
            pyvista_mesh.GetCellNeighbors(i, facet_pt_ids, nbr_ids)

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


def _detect_native_dims(
    pyvista_mesh: "pv.PolyData | pv.UnstructuredGrid | pv.PointSet",
) -> set[int]:
    """Return every topological dimension represented by native VTK cells."""
    if pyvista_mesh.n_cells == 0:
        return {0}
    if hasattr(pyvista_mesh, "celltypes"):
        return {
            int(vtk.vtkCellTypeUtilities.GetDimension(int(cell_type)))
            for cell_type in np.unique(pyvista_mesh.celltypes)
        }

    # PolyData does not expose ``celltypes``; its cell arrays are separated by
    # dimension, so counts provide the same information without cell traversal.
    n_lines = _get_count_safely(pyvista_mesh, "n_lines")
    n_cells = _get_count_safely(pyvista_mesh, "n_cells")
    n_verts = _get_count_safely(pyvista_mesh, "n_verts")
    dimensions: set[int] = set()
    if n_verts > 0:
        dimensions.add(0)
    if n_lines > 0:
        dimensions.add(1)
    if n_cells > n_verts + n_lines:
        dimensions.add(2)
    return dimensions or {0}


def _warn_on_data_loss(
    pyvista_mesh: "pv.PolyData | pv.UnstructuredGrid | pv.PointSet",
    point_source: str,
    manifold_dim: int,
    detected_dims: set[int] | None,
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
    detected_dims : set[int] or None
        Native manifold dimensions present in the original mesh.
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

    ### Case 2: cell_data lost when discarding native cells
    if point_source == "vertices" and detected_dims is not None:
        # A 0D Mesh is represented as a point cloud with no cells, so even
        # native VTK vertex-cell data has nowhere to attach.
        dropped_dims = (
            detected_dims if manifold_dim == 0 else detected_dims - {manifold_dim}
        )
        cd_keys = list(pyvista_mesh.cell_data.keys())
        if dropped_dims and cd_keys:
            warnings.warn(
                f"Selecting manifold_dim={manifold_dim} from native dimensions "
                f"{sorted(detected_dims)} cannot preserve cell_data values attached "
                f"to dimensions {sorted(dropped_dims)} for fields {cd_keys}. "
                f"Use point_source='cell_centroids' to preserve all cell_data "
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
