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
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch
from jaxtyping import Int
from tensordict import TensorDict

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

_PARENT_CELL_ID_KEY = "__physicsnemo_parent_cell_id"
_ORIGINAL_POINT_ID_KEY = "vtkOriginalPointIds"


def _vtk_data_to_tensor_dict(
    data: "pv.DataSetAttributes",
    force_copy: bool = False,
    indices: np.ndarray | None = None,
) -> TensorDict:
    """Convert a PyVista/VTK data container to a TensorDict.

    The returned TensorDict has no batch dimensions; ``Mesh.__post_init__``
    assigns the batch_size appropriate to the container it lands in.
    """
    tensor_data: dict[str, torch.Tensor] = {}
    for key, value in dict(data).items():
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
            continue
        if indices is not None:
            array = array[indices]
        elif force_copy:
            array = array.copy()
        tensor_data[str(key)] = torch.as_tensor(array)
    return TensorDict(tensor_data, device="cpu")


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


def _geometry_to_vtk_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert coordinates using PhysicsNeMo's PyVista dtype policy."""
    tensor = tensor.detach()
    if tensor.dtype not in (torch.float32, torch.float64):
        # PyVista/VTK can store some additional coordinate dtypes, but
        # PhysicsNeMo has historically exported integer, reduced-precision,
        # and complex geometry as float32. Keep that compatibility policy.
        tensor = tensor.float()
    return tensor.cpu().resolve_conj().resolve_neg().numpy()


def _vtk_cell_type_name(cell_type: int) -> str:
    """Return a symbolic PyVista cell-type name, including for unknown IDs."""
    try:
        return pv.CellType(int(cell_type)).name
    except ValueError:
        return f"UNKNOWN_CELL_TYPE_{int(cell_type)}"


def _linear_cell_specs() -> dict[int, tuple[int, int | None, int | None]]:
    """Map supported cells to ``(dimension, exact_arity, minimum_arity)``."""
    return {
        int(pv.CellType.EMPTY_CELL): (0, 0, None),
        int(pv.CellType.VERTEX): (0, 1, None),
        int(pv.CellType.POLY_VERTEX): (0, None, 1),
        int(pv.CellType.LINE): (1, 2, None),
        int(pv.CellType.POLY_LINE): (1, None, 2),
        int(pv.CellType.TRIANGLE): (2, 3, None),
        int(pv.CellType.TRIANGLE_STRIP): (2, None, 3),
        int(pv.CellType.POLYGON): (2, None, 3),
        int(pv.CellType.PIXEL): (2, 4, None),
        int(pv.CellType.QUAD): (2, 4, None),
        int(pv.CellType.TETRA): (3, 4, None),
        int(pv.CellType.VOXEL): (3, 8, None),
        int(pv.CellType.HEXAHEDRON): (3, 8, None),
        int(pv.CellType.WEDGE): (3, 6, None),
        int(pv.CellType.PYRAMID): (3, 5, None),
        int(pv.CellType.PENTAGONAL_PRISM): (3, 10, None),
        int(pv.CellType.HEXAGONAL_PRISM): (3, 12, None),
        int(pv.CellType.POLYHEDRON): (3, None, 4),
    }


def _fixed_higher_order_cell_specs() -> dict[int, tuple[int, int | None, int | None]]:
    """Map fixed-size nonlinear cells supported by VTK centroid filtering."""
    return {
        int(pv.CellType.QUADRATIC_EDGE): (1, 3, None),
        int(pv.CellType.QUADRATIC_TRIANGLE): (2, 6, None),
        int(pv.CellType.QUADRATIC_QUAD): (2, 8, None),
        int(pv.CellType.QUADRATIC_TETRA): (3, 10, None),
        int(pv.CellType.QUADRATIC_HEXAHEDRON): (3, 20, None),
        int(pv.CellType.QUADRATIC_WEDGE): (3, 15, None),
        int(pv.CellType.QUADRATIC_PYRAMID): (3, 13, None),
        int(pv.CellType.BIQUADRATIC_QUAD): (2, 9, None),
        int(pv.CellType.TRIQUADRATIC_HEXAHEDRON): (3, 27, None),
        int(pv.CellType.TRIQUADRATIC_PYRAMID): (3, 19, None),
        int(pv.CellType.QUADRATIC_LINEAR_QUAD): (2, 6, None),
        int(pv.CellType.QUADRATIC_LINEAR_WEDGE): (3, 12, None),
        int(pv.CellType.BIQUADRATIC_QUADRATIC_WEDGE): (3, 18, None),
        int(pv.CellType.BIQUADRATIC_QUADRATIC_HEXAHEDRON): (3, 24, None),
        int(pv.CellType.BIQUADRATIC_TRIANGLE): (2, 7, None),
        int(pv.CellType.CUBIC_LINE): (1, 4, None),
    }


def _unsupported_cell_types(
    cell_types: np.ndarray,
    cell_specs: dict[int, tuple[int, int | None, int | None]],
) -> list[int]:
    """Return sorted cell types absent from a supplied specification."""
    return sorted(
        {int(cell_type) for cell_type in cell_types if int(cell_type) not in cell_specs}
    )


def _validate_vtk_attribute_lengths(
    pyvista_mesh: "pv.PolyData | pv.UnstructuredGrid | pv.PointSet",
) -> None:
    """Validate VTK point/cell tuple counts before invoking any VTK filter.

    VTK filters assume that every attached attribute has one tuple per entity.
    Some malformed arrays bypass PyVista's normal assignment validation when
    added through VTK directly; passing them into a filter can truncate data or
    crash inside VTK.
    """
    associations = (
        ("point_data", pyvista_mesh.GetPointData(), pyvista_mesh.n_points),
        ("cell_data", pyvista_mesh.GetCellData(), pyvista_mesh.n_cells),
    )
    for association, attributes, expected in associations:
        for array_index in range(attributes.GetNumberOfArrays()):
            array = attributes.GetAbstractArray(array_index)
            actual = int(array.GetNumberOfTuples())
            if actual == expected:
                continue
            key = array.GetName() or f"<unnamed array {array_index}>"
            raise ValueError(
                f"Invalid {association} array {key!r}: expected {expected} "
                f"tuples, got {actual}."
            )


def _validate_vtk_cell_array_structure(
    cell_array: "vtk.vtkCellArray | None",
    association: str,
    expected_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a VTK cell array before reading offsets or connectivity."""
    if cell_array is None:
        if expected_cells == 0:
            return np.array([0], dtype=np.int64), np.empty(0, dtype=np.int64)
        raise ValueError(
            f"Invalid {association}: missing cell array for {expected_cells} cells."
        )
    if not bool(cell_array.IsValid()):
        raise ValueError(
            f"Invalid {association}: VTK cell array is not structurally valid."
        )

    offsets = np.asarray(cell_array.GetOffsetsArray())
    connectivity = np.asarray(cell_array.GetConnectivityArray())
    n_cells = int(cell_array.GetNumberOfCells())
    if n_cells != expected_cells or len(offsets) != expected_cells + 1:
        raise ValueError(
            f"Invalid {association}: expected {expected_cells} cells and "
            f"{expected_cells + 1} offsets, got {n_cells} cells and "
            f"{len(offsets)} offsets."
        )
    return offsets, connectivity


def _first_invalid_point_id_index(
    connectivity: np.ndarray,
    n_points: int,
) -> int | None:
    """Return the first out-of-bounds connectivity index, if one exists."""
    if len(connectivity) == 0:
        return None
    if int(connectivity.min()) >= 0 and int(connectivity.max()) < n_points:
        return None

    # Allocate an elementwise mask only on the malformed error path.
    invalid = (connectivity < 0) | (connectivity >= n_points)
    return int(np.flatnonzero(invalid)[0])


def _line_segments_from_vtk_cell_array(
    offsets: np.ndarray,
    connectivity: np.ndarray,
) -> np.ndarray:
    """Expand VTK line and polyline cells into consecutive two-point segments."""
    arities = np.diff(offsets)
    if len(arities) == 0:
        return np.empty((0, 2), dtype=np.int64)
    if int(arities.min()) == 2 and int(arities.max()) == 2:
        return connectivity.reshape(-1, 2).copy()

    # Every connectivity position starts a segment except each cell's final
    # point, which must not connect to the next cell's first point.
    is_segment_start = np.ones(len(connectivity) - 1, dtype=np.bool_)
    is_segment_start[offsets[1:-1] - 1] = False
    segment_starts = np.flatnonzero(is_segment_start)
    return np.column_stack(
        [connectivity[segment_starts], connectivity[segment_starts + 1]]
    )


def _validate_polydata_topology(
    pyvista_mesh: "pv.PolyData",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Validate every PolyData cell array without invoking VTK filters."""
    specifications = (
        (
            "verts",
            pyvista_mesh.GetVerts,
            1,
            pyvista_mesh.GetNumberOfVerts(),
        ),
        (
            "lines",
            pyvista_mesh.GetLines,
            2,
            pyvista_mesh.GetNumberOfLines(),
        ),
        (
            "faces",
            pyvista_mesh.GetPolys,
            3,
            pyvista_mesh.GetNumberOfPolys(),
        ),
        (
            "strips",
            pyvista_mesh.GetStrips,
            3,
            pyvista_mesh.GetNumberOfStrips(),
        ),
    )
    topology = {}
    for association, get_cell_array, minimum_arity, expected_cells in specifications:
        offsets, connectivity = _validate_vtk_cell_array_structure(
            get_cell_array(),
            f"PolyData {association}",
            int(expected_cells),
        )
        arities = np.diff(offsets)
        invalid_arity_indices = np.flatnonzero(arities < minimum_arity)
        if len(invalid_arity_indices) > 0:
            cell_index = int(invalid_arity_indices[0])
            raise ValueError(
                f"Invalid PolyData {association} cell at index {cell_index}: "
                f"expected at least {minimum_arity} points, "
                f"got {int(arities[cell_index])}."
            )

        connectivity_index = _first_invalid_point_id_index(
            connectivity,
            pyvista_mesh.n_points,
        )
        if connectivity_index is not None:
            cell_index = int(
                np.searchsorted(offsets[1:], connectivity_index, side="right")
            )
            point_id = int(connectivity[connectivity_index])
            raise ValueError(
                f"Invalid point ID {point_id} in PolyData {association} cell at "
                f"index {cell_index}: valid point IDs are in "
                f"[0, {pyvista_mesh.n_points})."
            )
        topology[association] = (offsets, connectivity)
    return topology


def _validate_unstructured_connectivity_bounds(
    pyvista_mesh: "pv.UnstructuredGrid",
    cell_types: np.ndarray,
) -> None:
    """Reject invalid point IDs without allocating a full-size success mask."""
    offsets, connectivity = _validate_vtk_cell_array_structure(
        pyvista_mesh.GetCells(),
        "UnstructuredGrid cells",
        pyvista_mesh.n_cells,
    )
    connectivity_index = _first_invalid_point_id_index(
        connectivity,
        pyvista_mesh.n_points,
    )
    if connectivity_index is None:
        return
    cell_index = int(np.searchsorted(offsets[1:], connectivity_index, side="right"))
    point_id = int(connectivity[connectivity_index])
    cell_type_name = _vtk_cell_type_name(int(cell_types[cell_index]))
    raise ValueError(
        f"Invalid point ID {point_id} in VTK {cell_type_name} cell at index "
        f"{cell_index}: valid point IDs are in [0, {pyvista_mesh.n_points})."
    )


def _homogeneous_simplex_dimension(
    pyvista_mesh: "pv.UnstructuredGrid",
) -> int | None:
    """Validate and identify an all-LINE/TRIANGLE/TETRA grid in O(1) memory."""
    cell_types = np.asarray(pyvista_mesh.celltypes)
    if len(cell_types) == 0:
        return None
    cell_type = int(cell_types[0])
    simplex_dimensions = {
        int(pv.CellType.LINE): 1,
        int(pv.CellType.TRIANGLE): 2,
        int(pv.CellType.TETRA): 3,
    }
    if cell_type not in simplex_dimensions:
        return None
    if int(cell_types.min()) != cell_type or int(cell_types.max()) != cell_type:
        return None

    _validate_unstructured_connectivity_bounds(pyvista_mesh, cell_types)
    expected_arity = int(_linear_cell_specs()[cell_type][1] or 0)
    actual_arity = int(pyvista_mesh.GetCells().IsHomogeneous())
    if actual_arity != expected_arity:
        arities = np.diff(np.asarray(pyvista_mesh.offset))
        invalid_cell_index = int(np.flatnonzero(arities != expected_arity)[0])
        cell_type_name = _vtk_cell_type_name(cell_type)
        raise ValueError(
            f"Invalid VTK {cell_type_name} cell at index {invalid_cell_index}: "
            f"expected exactly {expected_arity} points, "
            f"got {int(arities[invalid_cell_index])}."
        )
    return simplex_dimensions[cell_type]


def _validate_polyhedron_face_complex(
    parent_id: int,
    parent_point_ids: np.ndarray,
    parent_face_ids: np.ndarray,
    face_offsets: np.ndarray,
    face_point_ids: np.ndarray,
) -> None:
    """Validate one polyhedron's parent/face topology as a closed shell."""
    unique_parent_point_ids, parent_point_counts = np.unique(
        parent_point_ids,
        return_counts=True,
    )
    repeated_parent_points = unique_parent_point_ids[parent_point_counts > 1]

    referenced_faces: list[np.ndarray] = []
    seen_face_point_sets: dict[tuple[int, ...], int] = {}
    edge_to_local_faces: dict[tuple[int, int], list[int]] = {}
    for local_face_index, face_id_value in enumerate(parent_face_ids):
        face_id = int(face_id_value)
        point_ids = face_point_ids[face_offsets[face_id] : face_offsets[face_id + 1]]
        unique_point_ids, point_counts = np.unique(point_ids, return_counts=True)
        repeated_point_ids = unique_point_ids[point_counts > 1]
        if len(repeated_point_ids) > 0:
            raise ValueError(
                f"Invalid POLYHEDRON face {face_id} referenced by cell at index "
                f"{parent_id}: face repeats point ID {int(repeated_point_ids[0])}."
            )

        face_point_set = tuple(sorted(map(int, point_ids)))
        previous_face_id = seen_face_point_sets.get(face_point_set)
        if previous_face_id is not None:
            raise ValueError(
                f"Invalid POLYHEDRON cell at index {parent_id}: duplicate faces "
                f"{previous_face_id} and {face_id} use the same point IDs."
            )
        seen_face_point_sets[face_point_set] = face_id
        referenced_faces.append(point_ids)

        for start, end in zip(point_ids, np.roll(point_ids, -1)):
            edge = tuple(sorted((int(start), int(end))))
            edge_to_local_faces.setdefault(edge, []).append(local_face_index)

    referenced_face_point_ids = np.unique(np.concatenate(referenced_faces))
    if not np.array_equal(unique_parent_point_ids, referenced_face_point_ids):
        missing_from_faces = np.setdiff1d(
            unique_parent_point_ids,
            referenced_face_point_ids,
        )
        missing_from_parent = np.setdiff1d(
            referenced_face_point_ids,
            unique_parent_point_ids,
        )
        raise ValueError(
            f"Invalid POLYHEDRON cell at index {parent_id}: parent "
            "connectivity and referenced faces use different point IDs; "
            f"missing from faces {missing_from_faces.tolist()}, missing "
            f"from parent connectivity {missing_from_parent.tolist()}."
        )
    if len(repeated_parent_points) > 0:
        raise ValueError(
            f"Invalid POLYHEDRON cell at index {parent_id}: parent connectivity "
            f"repeats point ID {int(repeated_parent_points[0])}."
        )

    for edge, incident_faces in edge_to_local_faces.items():
        if len(incident_faces) != 2:
            raise ValueError(
                f"Invalid POLYHEDRON cell at index {parent_id}: edge "
                f"{list(edge)} must belong to exactly 2 faces in a closed shell, "
                f"got {len(incident_faces)}."
            )

    face_adjacency = [set() for _ in parent_face_ids]
    for first_face, second_face in edge_to_local_faces.values():
        face_adjacency[first_face].add(second_face)
        face_adjacency[second_face].add(first_face)
    connected_faces = {0}
    pending_faces = [0]
    while pending_faces:
        face_index = pending_faces.pop()
        new_neighbors = face_adjacency[face_index] - connected_faces
        connected_faces.update(new_neighbors)
        pending_faces.extend(new_neighbors)
    if len(connected_faces) != len(parent_face_ids):
        disconnected_face_ids = [
            int(parent_face_ids[index])
            for index in range(len(parent_face_ids))
            if index not in connected_faces
        ]
        raise ValueError(
            f"Invalid POLYHEDRON cell at index {parent_id}: referenced faces "
            "must form one connected closed shell; disconnected face IDs "
            f"include {disconnected_face_ids}."
        )


def _gather_vtk_rows(
    offsets: np.ndarray,
    values: np.ndarray,
    row_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather ragged VTK rows without a Python loop.

    Returns the gathered values, their zero-based output-row IDs, and an
    offsets array for the gathered rows.
    """
    row_lengths = offsets[row_ids + 1] - offsets[row_ids]
    gathered_offsets = np.empty(len(row_ids) + 1, dtype=np.int64)
    gathered_offsets[0] = 0
    np.cumsum(row_lengths, out=gathered_offsets[1:])
    n_values = int(gathered_offsets[-1])
    if n_values == 0:
        return (
            np.empty(0, dtype=values.dtype),
            np.empty(0, dtype=np.int64),
            gathered_offsets,
        )

    output_row_ids = np.repeat(
        np.arange(len(row_ids), dtype=np.int64),
        row_lengths,
    )
    value_indices = (
        np.repeat(offsets[row_ids], row_lengths)
        + np.arange(n_values, dtype=np.int64)
        - np.repeat(gathered_offsets[:-1], row_lengths)
    )
    return values[value_indices], output_row_ids, gathered_offsets


def _unique_owner_values(
    owner_ids: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort owner/value pairs and identify their first occurrences."""
    order = np.lexsort((values, owner_ids))
    sorted_owners = owner_ids[order]
    sorted_values = values[order]
    is_first = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        is_first[1:] = (sorted_owners[1:] != sorted_owners[:-1]) | (
            sorted_values[1:] != sorted_values[:-1]
        )
    return sorted_owners, sorted_values, is_first


def _polyhedron_face_point_offsets(
    parent_ids: np.ndarray,
    location_offsets: np.ndarray,
    face_ids: np.ndarray,
    face_offsets: np.ndarray,
) -> np.ndarray:
    """Return cumulative referenced face-point counts for selected parents."""
    count_chunk_size = 256
    face_point_offsets = np.empty(len(parent_ids) + 1, dtype=np.int64)
    face_point_offsets[0] = 0
    for count_start in range(0, len(parent_ids), count_chunk_size):
        count_end = min(count_start + count_chunk_size, len(parent_ids))
        chunk_face_ids, _, chunk_face_offsets = _gather_vtk_rows(
            location_offsets,
            face_ids,
            parent_ids[count_start:count_end],
        )
        chunk_face_lengths = (
            face_offsets[chunk_face_ids + 1] - face_offsets[chunk_face_ids]
        )
        face_point_offsets[count_start + 1 : count_end + 1] = np.add.reduceat(
            chunk_face_lengths,
            chunk_face_offsets[:-1],
        )
    np.cumsum(face_point_offsets[1:], out=face_point_offsets[1:])
    return face_point_offsets


def _polyhedron_validation_batches(
    parent_ids: np.ndarray,
    location_offsets: np.ndarray,
    face_ids: np.ndarray,
    face_offsets: np.ndarray,
    *,
    max_cells: int = 4096,
    max_face_points: int = 1 << 18,
) -> Iterator[np.ndarray]:
    """Batch structurally valid parents by cell and referenced-point counts."""
    face_point_offsets = _polyhedron_face_point_offsets(
        parent_ids,
        location_offsets,
        face_ids,
        face_offsets,
    )

    batch_start = 0
    while batch_start < len(parent_ids):
        point_limited_end = int(
            np.searchsorted(
                face_point_offsets,
                face_point_offsets[batch_start] + max_face_points,
                side="right",
            )
            - 1
        )
        batch_end = max(
            batch_start + 1,
            min(
                batch_start + max_cells,
                point_limited_end,
                len(parent_ids),
            ),
        )
        yield parent_ids[batch_start:batch_end]
        batch_start = batch_end


def _validate_polyhedron_face_complexes(
    parent_ids: np.ndarray,
    cell_offsets: np.ndarray,
    cell_point_ids: np.ndarray,
    location_offsets: np.ndarray,
    face_ids: np.ndarray,
    face_offsets: np.ndarray,
    face_point_ids: np.ndarray,
) -> None:
    """Batch-check prevalidated polyhedron arrays with exact diagnostics.

    Every parent is expected to reference at least four faces, and every face
    is expected to contain at least three valid point IDs. The auxiliary-array
    validator enforces those structural preconditions before calling here.
    """
    # Bound temporary sort buffers by both cells and gathered face-point IDs.
    # A single unusually large cell remains indivisible and forms its own batch.
    for batch_parent_ids in _polyhedron_validation_batches(
        parent_ids,
        location_offsets,
        face_ids,
        face_offsets,
    ):
        n_parents = len(batch_parent_ids)
        candidate_owners = np.zeros(n_parents, dtype=bool)

        parent_points, point_owners, _ = _gather_vtk_rows(
            cell_offsets,
            cell_point_ids,
            batch_parent_ids,
        )
        parent_face_ids, face_owners, _ = _gather_vtk_rows(
            location_offsets,
            face_ids,
            batch_parent_ids,
        )
        face_points, point_face_ids, gathered_face_offsets = _gather_vtk_rows(
            face_offsets,
            face_point_ids,
            parent_face_ids,
        )
        face_point_owners = face_owners[point_face_ids]

        # Repeated points in either parent connectivity or an individual face.
        sorted_owners, sorted_points, is_first = _unique_owner_values(
            point_owners,
            parent_points,
        )
        candidate_owners[sorted_owners[~is_first]] = True
        sorted_face_ids, _, is_first_face_point = _unique_owner_values(
            point_face_ids, face_points
        )
        candidate_owners[face_owners[sorted_face_ids[~is_first_face_point]]] = True

        # Compare the parent and referenced-face point sets exactly. Each
        # source is unique before merging, so unmatched pairs occur once and
        # matching pairs occur twice.
        unique_parent_owners = sorted_owners[is_first]
        unique_parent_points = sorted_points[is_first]
        face_set_owners, face_set_points, is_first_face_set_point = (
            _unique_owner_values(face_point_owners, face_points)
        )
        unique_face_owners = face_set_owners[is_first_face_set_point]
        unique_face_points = face_set_points[is_first_face_set_point]
        set_owners = np.concatenate((unique_parent_owners, unique_face_owners))
        set_points = np.concatenate((unique_parent_points, unique_face_points))
        set_order = np.lexsort((set_points, set_owners))
        set_owners = set_owners[set_order]
        set_points = set_points[set_order]
        matched = np.zeros(len(set_order), dtype=bool)
        if len(set_order) > 1:
            equal_neighbors = (set_owners[1:] == set_owners[:-1]) & (
                set_points[1:] == set_points[:-1]
            )
            matched[:-1] |= equal_neighbors
            matched[1:] |= equal_neighbors
        candidate_owners[set_owners[~matched]] = True

        # Identical point sets imply identical integer signatures. Signature
        # collisions only add a scalar diagnostic check; they cannot hide an
        # invalid duplicate face.
        face_lengths = np.diff(gathered_face_offsets)
        face_starts = gathered_face_offsets[:-1]
        face_sums = np.add.reduceat(face_points, face_starts)
        face_square_sums = np.add.reduceat(face_points * face_points, face_starts)
        face_xors = np.bitwise_xor.reduceat(face_points, face_starts)
        signature_order = np.lexsort(
            (
                face_xors,
                face_square_sums,
                face_sums,
                face_lengths,
                face_owners,
            )
        )
        if len(signature_order) > 1:
            previous = signature_order[:-1]
            current = signature_order[1:]
            duplicate_signatures = (
                (face_owners[current] == face_owners[previous])
                & (face_lengths[current] == face_lengths[previous])
                & (face_sums[current] == face_sums[previous])
                & (face_square_sums[current] == face_square_sums[previous])
                & (face_xors[current] == face_xors[previous])
            )
            candidate_owners[face_owners[current[duplicate_signatures]]] = True

        # Every undirected edge in a closed shell has exactly two incident
        # faces. Retain the corresponding face pairs for connectivity below.
        edge_ends = np.roll(face_points, -1)
        edge_ends[gathered_face_offsets[1:] - 1] = face_points[
            gathered_face_offsets[:-1]
        ]
        edge_lows = np.minimum(face_points, edge_ends)
        edge_highs = np.maximum(face_points, edge_ends)
        edge_order = np.lexsort((edge_highs, edge_lows, face_point_owners))
        ordered_edge_owners = face_point_owners[edge_order]
        ordered_edge_lows = edge_lows[edge_order]
        ordered_edge_highs = edge_highs[edge_order]
        edge_group_starts = np.flatnonzero(
            np.r_[
                True,
                (ordered_edge_owners[1:] != ordered_edge_owners[:-1])
                | (ordered_edge_lows[1:] != ordered_edge_lows[:-1])
                | (ordered_edge_highs[1:] != ordered_edge_highs[:-1]),
            ]
        )
        edge_group_lengths = np.diff(np.r_[edge_group_starts, len(edge_order)])
        invalid_edge_groups = edge_group_lengths != 2
        candidate_owners[
            ordered_edge_owners[edge_group_starts[invalid_edge_groups]]
        ] = True

        paired_edge_starts = edge_group_starts[~invalid_edge_groups]
        ordered_point_face_ids = point_face_ids[edge_order]
        first_faces = ordered_point_face_ids[paired_edge_starts]
        second_faces = ordered_point_face_ids[paired_edge_starts + 1]

        # Propagate component labels in parallel. Unusually deep face graphs
        # conservatively fall back to the scalar checker if 32 rounds do not
        # converge; non-convergence can only create a false-positive candidate.
        component_labels = np.arange(len(parent_face_ids), dtype=np.int64)
        for _ in range(32):
            old_labels = component_labels
            pair_labels = np.minimum(
                component_labels[first_faces],
                component_labels[second_faces],
            )
            component_labels = component_labels.copy()
            np.minimum.at(component_labels, first_faces, pair_labels)
            np.minimum.at(component_labels, second_faces, pair_labels)
            component_labels = component_labels[component_labels]
            if np.array_equal(component_labels, old_labels):
                break
        minimum_labels = np.full(n_parents, len(parent_face_ids), dtype=np.int64)
        maximum_labels = np.full(n_parents, -1, dtype=np.int64)
        np.minimum.at(minimum_labels, face_owners, component_labels)
        np.maximum.at(maximum_labels, face_owners, component_labels)
        candidate_owners[minimum_labels != maximum_labels] = True

        # The scalar implementation is retained as the authoritative error
        # reporter and confirms the deliberately conservative candidates.
        for owner in np.flatnonzero(candidate_owners):
            parent_id = int(batch_parent_ids[owner])
            _validate_polyhedron_face_complex(
                parent_id,
                cell_point_ids[cell_offsets[parent_id] : cell_offsets[parent_id + 1]],
                face_ids[location_offsets[parent_id] : location_offsets[parent_id + 1]],
                face_offsets,
                face_point_ids,
            )


def _validate_polyhedron_auxiliary_arrays(
    pyvista_mesh: "pv.UnstructuredGrid",
    cell_types: np.ndarray,
    *,
    validate_face_complexes: bool = True,
    face_complex_parent_ids: np.ndarray | None = None,
) -> None:
    """Validate polyhedron arrays and, when requested, selected face complexes."""
    n_polyhedra = int((cell_types == pv.CellType.POLYHEDRON).sum())
    if n_polyhedra == 0:
        return

    faces = pyvista_mesh.GetPolyhedronFaces()
    face_locations = pyvista_mesh.GetPolyhedronFaceLocations()
    if faces is None:
        raise ValueError("Invalid POLYHEDRON faces: missing face array.")
    face_offsets, face_point_ids = _validate_vtk_cell_array_structure(
        faces,
        "POLYHEDRON faces",
        int(faces.GetNumberOfCells()),
    )
    location_offsets, face_ids = _validate_vtk_cell_array_structure(
        face_locations,
        "POLYHEDRON face locations",
        pyvista_mesh.n_cells,
    )

    face_arities = np.diff(face_offsets)
    if len(face_arities) > 0 and int(face_arities.min()) < 3:
        face_index = int(np.flatnonzero(face_arities < 3)[0])
        raise ValueError(
            f"Invalid POLYHEDRON face {face_index}: expected at least 3 "
            f"points, got {int(face_arities[face_index])}."
        )
    if len(face_point_ids) > 0 and (
        int(face_point_ids.min()) < 0
        or int(face_point_ids.max()) >= pyvista_mesh.n_points
    ):
        bad_id = int(
            face_point_ids[
                (face_point_ids < 0) | (face_point_ids >= pyvista_mesh.n_points)
            ][0]
        )
        raise ValueError(
            f"Invalid POLYHEDRON face point ID {bad_id}: valid point IDs are "
            f"in [0, {pyvista_mesh.n_points})."
        )

    location_arities = np.diff(location_offsets)
    polyhedron_mask = cell_types == pv.CellType.POLYHEDRON
    invalid_polyhedra = np.flatnonzero(polyhedron_mask & (location_arities < 4))
    invalid_other_cells = np.flatnonzero(~polyhedron_mask & (location_arities != 0))
    if len(invalid_polyhedra) > 0 or len(invalid_other_cells) > 0:
        raise ValueError(
            "Invalid POLYHEDRON face locations: polyhedron parents must "
            "reference at least 4 faces and other cell types must reference 0."
        )
    n_faces = int(faces.GetNumberOfCells())
    if len(face_ids) > 0 and (
        int(face_ids.min()) < 0 or int(face_ids.max()) >= n_faces
    ):
        bad_face_id = int(face_ids[(face_ids < 0) | (face_ids >= n_faces)][0])
        raise ValueError(
            f"Invalid POLYHEDRON face-location reference {bad_face_id}: valid "
            f"face IDs are in [0, {n_faces})."
        )

    if not validate_face_complexes:
        return
    if face_complex_parent_ids is None:
        face_complex_parent_ids = np.flatnonzero(polyhedron_mask)
    if len(face_complex_parent_ids) == 0:
        return

    cell_offsets = np.asarray(pyvista_mesh.offset)
    cell_point_ids = np.asarray(pyvista_mesh.GetCells().GetConnectivityArray())
    _validate_polyhedron_face_complexes(
        np.asarray(face_complex_parent_ids, dtype=np.int64),
        cell_offsets,
        cell_point_ids,
        location_offsets,
        face_ids,
        face_offsets,
        face_point_ids,
    )


def _polyhedron_is_convex(
    polyhedron: "vtk.vtkPolyhedron",
    source_point_dtype: np.dtype,
    *,
    trust_vtk_result: bool,
) -> bool:
    """Return whether every polyhedron face defines a supporting plane."""
    # VTK's C++ predicate is fast and reliable as of VTK 9.6. Earlier versions
    # can report false for valid prism cells, so retain a geometric fallback.
    vtk_is_convex = bool(polyhedron.IsConvex())
    if vtk_is_convex or trust_vtk_result:
        return vtk_is_convex

    point_values = np.asarray(polyhedron.GetPoints().GetData())
    point_dtype = (
        source_point_dtype
        if np.issubdtype(source_point_dtype, np.floating)
        else np.dtype(np.float64)
    )
    points = point_values.astype(np.float64, copy=False)
    extent_scale = float(np.linalg.norm(np.ptp(points, axis=0)))
    if extent_scale == 0.0:
        return False
    coordinate_scale = max(
        float(np.max(np.abs(points), initial=0.0)),
        extent_scale,
    )
    coordinate_spacing = abs(
        float(np.spacing(np.asarray(coordinate_scale, dtype=point_dtype)))
    )
    distance_tolerance = 4.0 * max(
        coordinate_spacing,
        np.finfo(point_dtype).eps * extent_scale,
    )
    normal_tolerance = 64.0 * np.finfo(np.float64).eps * extent_scale * extent_scale

    ### A convex cell lies entirely within one closed half-space of each face.
    ### Face winding is irrelevant because either half-space orientation works.
    for face_index in range(polyhedron.GetNumberOfFaces()):
        face = polyhedron.GetFace(face_index)
        face_points = np.asarray(face.GetPoints().GetData()).astype(
            np.float64,
            copy=False,
        )
        face_edges = face_points[1:] - face_points[0]  # (F - 1, 3)
        candidate_normals = np.cross(face_edges[:-1], face_edges[1:])
        candidate_norms = np.linalg.norm(candidate_normals, axis=1)
        valid_normal_indices = np.flatnonzero(candidate_norms > normal_tolerance)
        if len(valid_normal_indices) == 0:
            return False
        normal_index = int(valid_normal_indices[0])
        unit_normal = candidate_normals[normal_index] / candidate_norms[normal_index]
        signed_distances = (points - face_points[0]) @ unit_normal  # (P,)
        if bool((signed_distances > distance_tolerance).any()) and bool(
            (signed_distances < -distance_tolerance).any()
        ):
            return False
    return True


def _validate_selected_polyhedra_for_tetrahedralization(
    pyvista_mesh: "pv.UnstructuredGrid",
    cell_types: np.ndarray,
    selected_parent_ids: np.ndarray,
) -> None:
    """Validate selected polyhedra before VTK tetrahedralization."""
    polyhedron_parent_ids = selected_parent_ids[
        cell_types[selected_parent_ids] == pv.CellType.POLYHEDRON
    ]
    if len(polyhedron_parent_ids) == 0:
        return

    source_point_dtype = pyvista_mesh.points.dtype
    vtk_version = vtk.vtkVersion()
    trust_vtk_result = (
        vtk_version.GetVTKMajorVersion(),
        vtk_version.GetVTKMinorVersion(),
    ) >= (9, 6)

    for parent_id in polyhedron_parent_ids:
        polyhedron = vtk.vtkPolyhedron.SafeDownCast(
            pyvista_mesh.GetCell(int(parent_id))
        )
        if polyhedron is None:
            raise ValueError(
                f"Invalid VTK POLYHEDRON cell at index {int(parent_id)}: "
                "cell type did not resolve to vtkPolyhedron."
            )
        if _polyhedron_is_convex(
            polyhedron,
            source_point_dtype,
            trust_vtk_result=trust_vtk_result,
        ):
            continue
        raise ValueError(
            f"Unsupported non-convex VTK POLYHEDRON cell at index "
            f"{int(parent_id)}: VTK tetrahedralization can alter concave geometry."
        )


def _validate_cells_against_specs(
    pyvista_mesh: "pv.UnstructuredGrid",
    cell_types: np.ndarray,
    unique_cell_types: np.ndarray,
    cell_specs: dict[int, tuple[int, int | None, int | None]],
    *,
    validate_polyhedron_topology: bool = True,
) -> None:
    """Validate bounds and listed arities without rejecting unlisted cells."""
    _validate_unstructured_connectivity_bounds(pyvista_mesh, cell_types)
    if validate_polyhedron_topology:
        _validate_polyhedron_auxiliary_arrays(pyvista_mesh, cell_types)
    if len(cell_types) == 0:
        return

    arities = np.diff(np.asarray(pyvista_mesh.offset))
    for cell_type_value in unique_cell_types:
        cell_type = int(cell_type_value)
        spec = cell_specs.get(cell_type)
        if spec is None:
            continue
        cell_type_name = _vtk_cell_type_name(cell_type)
        _, expected_arity, minimum_arity = spec
        if expected_arity is not None:
            invalid_arity_indices = np.flatnonzero(
                (cell_types == cell_type) & (arities != expected_arity)
            )
            expectation = f"exactly {expected_arity}"
        else:
            if minimum_arity is None:
                raise RuntimeError(f"Missing arity specification for {cell_type_name}.")
            invalid_arity_indices = np.flatnonzero(
                (cell_types == cell_type) & (arities < minimum_arity)
            )
            expectation = f"at least {minimum_arity}"
        if len(invalid_arity_indices) > 0:
            cell_index = int(invalid_arity_indices[0])
            raise ValueError(
                f"Invalid VTK {cell_type_name} cell at index {cell_index}: "
                f"expected {expectation} points, got {int(arities[cell_index])}."
            )


def _unstructured_cell_dimensions(
    pyvista_mesh: "pv.UnstructuredGrid",
) -> np.ndarray:
    """Validate supported linear topology and return each cell dimension."""
    cell_types = np.asarray(pyvista_mesh.celltypes)
    if len(cell_types) == 0:
        return np.empty(0, dtype=np.uint8)
    unique_cell_types, inverse = np.unique(
        cell_types,
        return_inverse=True,
    )
    linear_specs = _linear_cell_specs()
    _validate_cells_against_specs(
        pyvista_mesh,
        cell_types,
        unique_cell_types,
        linear_specs,
        validate_polyhedron_topology=False,
    )

    ### Reject every topology family outside the explicit linear allowlist.
    unsupported_types = _unsupported_cell_types(unique_cell_types, linear_specs)
    if unsupported_types:
        names = ", ".join(_vtk_cell_type_name(t) for t in unsupported_types)
        raise ValueError(
            f"Unsupported VTK cell type(s) {names}: PhysicsNeMo does not "
            "provide trusted globally conforming topology for these cell "
            "families; globally conforming higher-order tessellation is "
            "deferred."
        )

    type_dimensions = np.array(
        [linear_specs[int(cell_type)][0] for cell_type in unique_cell_types],
        dtype=np.uint8,
    )
    return type_dimensions[inverse]


def _triangulate_with_parent_ids(
    pyvista_mesh: "pv.PolyData | pv.UnstructuredGrid",
    parent_ids: np.ndarray,
) -> tuple["pv.PolyData | pv.UnstructuredGrid", np.ndarray]:
    """Triangulate cells while retaining one source parent ID per output cell."""
    # A shallow copy gives the filter an isolated attribute container while
    # retaining zero-copy geometry. Carry only the provenance that is consumed.
    working = pyvista_mesh.copy(deep=False)
    original_point_ids = working.point_data.get(_ORIGINAL_POINT_ID_KEY)
    working.point_data.clear()
    if original_point_ids is not None:
        working.point_data[_ORIGINAL_POINT_ID_KEY] = original_point_ids
    working.cell_data.clear()
    working.field_data.clear()
    working.cell_data[_PARENT_CELL_ID_KEY] = parent_ids

    triangulated = working.triangulate()
    if _PARENT_CELL_ID_KEY in triangulated.cell_data:
        output_parent_ids = np.asarray(
            triangulated.cell_data[_PARENT_CELL_ID_KEY],
            dtype=np.int64,
        ).copy()
        del triangulated.cell_data[_PARENT_CELL_ID_KEY]
    else:
        output_parent_ids = np.empty(0, dtype=np.int64)

    if len(output_parent_ids) != triangulated.n_cells:
        raise ValueError(
            "VTK simplex conversion did not preserve one provenance value per "
            f"output cell: expected {triangulated.n_cells}, got "
            f"{len(output_parent_ids)}."
        )
    unexpected_parent_ids = np.setdiff1d(
        np.unique(output_parent_ids),
        parent_ids,
        assume_unique=True,
    )
    if len(unexpected_parent_ids) > 0:
        raise ValueError(
            "VTK simplex conversion produced unknown parent IDs: "
            f"{unexpected_parent_ids.tolist()}."
        )
    return triangulated, output_parent_ids


def _polydata_surface_triangles(
    pyvista_mesh: "pv.PolyData",
    topology: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Triangulate polygon faces and strips with exact source parent IDs."""
    n_verts = pyvista_mesh.GetNumberOfVerts()
    n_lines = pyvista_mesh.GetNumberOfLines()
    n_faces = pyvista_mesh.GetNumberOfPolys()
    n_strips = pyvista_mesh.GetNumberOfStrips()
    if n_faces + n_strips == 0:
        raise ValueError("PolyData has no native surface cells for manifold_dim=2.")

    face_offsets, face_connectivity = topology["faces"]
    all_faces_triangles = bool((np.diff(face_offsets) == 3).all())
    if n_verts == 0 and n_lines == 0 and n_strips == 0 and all_faces_triangles:
        return pyvista_mesh.regular_faces, None

    first_face_parent = n_verts + n_lines
    triangle_parts: list[np.ndarray] = []
    parent_parts: list[np.ndarray] = []

    if n_faces > 0:
        face_parent_ids = np.arange(
            first_face_parent,
            first_face_parent + n_faces,
            dtype=np.int64,
        )
        if all_faces_triangles:
            triangle_parts.append(face_connectivity.reshape(-1, 3))
            parent_parts.append(face_parent_ids)
        else:
            faces = pv.PolyData(
                pyvista_mesh.points,
                faces=pyvista_mesh.faces,
                force_float=False,
            )
            triangulated_faces, face_output_parent_ids = _triangulate_with_parent_ids(
                faces, face_parent_ids
            )
            if not triangulated_faces.is_all_triangles:
                raise ValueError("VTK triangulation left non-triangle PolyData faces.")
            triangle_parts.append(triangulated_faces.regular_faces)
            parent_parts.append(face_output_parent_ids)

    if n_strips > 0:
        strip_parent_ids = np.arange(
            first_face_parent + n_faces,
            first_face_parent + n_faces + n_strips,
            dtype=np.int64,
        )
        strips = pv.PolyData(
            pyvista_mesh.points,
            strips=pyvista_mesh.strips,
            force_float=False,
        )
        triangulated_strips, strip_output_parent_ids = _triangulate_with_parent_ids(
            strips,
            strip_parent_ids,
        )
        if not triangulated_strips.is_all_triangles:
            raise ValueError("VTK triangulation left non-triangle PolyData strips.")
        triangle_parts.append(triangulated_strips.regular_faces)
        parent_parts.append(strip_output_parent_ids)

    triangles = (
        triangle_parts[0]
        if len(triangle_parts) == 1
        else np.concatenate(triangle_parts, axis=0)
    )
    output_parent_ids = (
        parent_parts[0] if len(parent_parts) == 1 else np.concatenate(parent_parts)
    )
    selected_surface_parents = np.arange(
        first_face_parent,
        first_face_parent + n_faces + n_strips,
        dtype=np.int64,
    )
    missing_parent_ids = np.setdiff1d(
        selected_surface_parents,
        np.unique(output_parent_ids),
        assume_unique=True,
    )
    if len(missing_parent_ids) > 0:
        raise ValueError(
            "VTK triangulation dropped selected PolyData surface parents "
            f"{missing_parent_ids.tolist()}."
        )
    return triangles, output_parent_ids


def _select_and_linearize_unstructured_grid(
    pyvista_mesh: "pv.UnstructuredGrid",
    cell_dimensions: np.ndarray,
    target_dim: int,
) -> tuple["pv.UnstructuredGrid", np.ndarray | None, np.ndarray | None]:
    """Select one native dimension and convert its cells to simplices.

    Point and parent-cell ID maps are returned only when output connectivity or
    cell data must be mapped back to the source grid.
    """
    cell_types = np.asarray(pyvista_mesh.celltypes)
    selected_parent_ids = np.flatnonzero(cell_dimensions == target_dim).astype(np.int64)
    if len(selected_parent_ids) == 0:
        available_dimensions = sorted(set(map(int, cell_dimensions)))
        raise ValueError(
            f"UnstructuredGrid has no cells with manifold dimension {target_dim}; "
            f"available dimensions are {available_dimensions}."
        )
    selected_polyhedron_ids = selected_parent_ids[
        cell_types[selected_parent_ids] == pv.CellType.POLYHEDRON
    ]
    _validate_polyhedron_auxiliary_arrays(
        pyvista_mesh,
        cell_types,
        validate_face_complexes=len(selected_polyhedron_ids) > 0,
        face_complex_parent_ids=selected_polyhedron_ids,
    )
    if target_dim == 3:
        _validate_selected_polyhedra_for_tetrahedralization(
            pyvista_mesh,
            cell_types,
            selected_parent_ids,
        )

    ### Extract only the requested native dimension. PyVista compacts points,
    ### while vtkOriginalPointIds records how to restore source connectivity.
    selected_all_cells = len(selected_parent_ids) == pyvista_mesh.n_cells
    if selected_all_cells:
        selected = pyvista_mesh
    else:
        extraction_source = pyvista_mesh.copy(deep=False)
        extraction_source.point_data.clear()
        extraction_source.cell_data.clear()
        extraction_source.field_data.clear()
        try:
            selected = extraction_source.extract_cells(
                selected_parent_ids,
                pass_cell_ids=False,
                pass_point_ids=True,
            )
        except TypeError as error:
            # PyVista 0.46 does not expose the pass_*_ids keywords and always
            # adds synthetic ID arrays.
            if "pass_cell_ids" not in str(error) and "pass_point_ids" not in str(error):
                raise
            # PyVista 0.46 always adds synthetic ID fields to the extraction
            # source; that mutation is already isolated on the shallow copy.
            selected = extraction_source.extract_cells(selected_parent_ids)

    simplex_type = {
        1: pv.CellType.LINE,
        2: pv.CellType.TRIANGLE,
        3: pv.CellType.TETRA,
    }[target_dim]
    if bool((selected.celltypes == simplex_type).all()):
        original_point_ids = (
            None
            if selected_all_cells
            else np.asarray(selected.point_data[_ORIGINAL_POINT_ID_KEY]).copy()
        )
        output_parent_ids = None if selected_all_cells else selected_parent_ids
        return selected, original_point_ids, output_parent_ids

    linearized_mesh, output_parent_ids = _triangulate_with_parent_ids(
        selected,
        selected_parent_ids,
    )
    linearized = cast("pv.UnstructuredGrid", linearized_mesh)

    ### Every selected parent must generate at least one output simplex.
    produced_parent_ids = np.unique(output_parent_ids)
    missing_parent_ids = np.setdiff1d(
        selected_parent_ids,
        produced_parent_ids,
        assume_unique=True,
    )
    if len(missing_parent_ids) > 0:
        missing_details = ", ".join(
            f"parent {int(parent_id)} "
            f"({_vtk_cell_type_name(int(cell_types[parent_id]))})"
            for parent_id in missing_parent_ids
        )
        raise ValueError(
            f"VTK simplex conversion dropped selected parent cells: {missing_details}."
        )

    ### Fail before connectivity extraction if VTK left non-simplex cells.
    unexpected_types = np.unique(
        linearized.celltypes[linearized.celltypes != simplex_type]
    )
    if linearized.n_cells == 0 or len(unexpected_types) > 0:
        output_names = (
            ", ".join(_vtk_cell_type_name(int(t)) for t in unexpected_types)
            or "no output cells"
        )
        raise ValueError(
            f"Could not linearize manifold dimension {target_dim} to "
            f"{simplex_type.name}; VTK returned {output_names}."
        )

    original_point_ids = (
        None
        if selected_all_cells
        else np.asarray(linearized.point_data[_ORIGINAL_POINT_ID_KEY]).copy()
    )
    return linearized, original_point_ids, output_parent_ids


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

        For an ``UnstructuredGrid``, explicit 1D conversion selects native line
        cells when present; otherwise it derives the unique edge graph from
        higher-dimensional cells. Explicit 2D and 3D conversion selects native
        cells of that dimension and raises ``ValueError`` when none exist.

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
        data arrays. Cell-data values are lost when
        ``point_source="vertices"`` drops cells from unselected native
        dimensions. Point data is lost when ``point_source="cell_centroids"``.
    force_copy : bool
        If True, copy geometry and attached data arrays so the returned Mesh
        owns its memory independently of the source PyVista mesh. When False
        (default), returned tensors may share memory with the source for
        efficiency.

    Returns
    -------
    Mesh
        Mesh object with converted geometry and data (on CPU).

    Raises
    ------
    ValueError
        If manifold dimension cannot be determined or is invalid, or if
        consumed topology is malformed or unsupported.
    ImportError
        If pyvista is not installed.

    Notes
    -----
    Point coordinates with a ``float32`` or ``float64`` dtype retain that
    dtype. Other coordinate dtypes are converted to ``float32``. Retaining
    ``float64`` doubles coordinate storage relative to ``float32``, and
    downstream geometric calculations generally remain in ``float64``. To
    normalize the returned mesh and its floating data to ``float32``, use
    ``from_pyvista(...).to(torch.float32)``.

    Topology conversion is limited to explicitly supported linear VTK cell
    families. Higher-order, control-net, parametric, abstract, and generic
    convex-point-set cells are rejected because globally conforming
    tessellation is deferred. Explicit ``manifold_dim=0`` with
    ``point_source="vertices"`` preserves input points without inspecting
    ``UnstructuredGrid`` connectivity. Centroid point-cloud filtering
    additionally accepts fixed-size quadratic and cubic cells, which do not
    require tessellation, and rejects ``EMPTY_CELL`` parents because VTK omits
    their centers.

    Polyhedron auxiliary-array structure is checked before any VTK topology
    filter runs. Closed-shell face complexes are checked in bounded batches
    only for polyhedra consumed by the requested conversion. Cell-centroid and
    derived-edge conversions consume every input cell and therefore check
    every polyhedron face complex.
    """
    ### Validate point_source
    if point_source not in {"vertices", "cell_centroids"}:
        raise ValueError(
            f"Invalid {point_source=!r}. Must be 'vertices' or 'cell_centroids'."
        )
    if manifold_dim not in {"auto", 0, 1, 2, 3}:
        raise ValueError(
            f"Invalid {manifold_dim=}. Must be one of {{0, 1, 2, 3}} or 'auto'."
        )

    # VTK filters assume valid attribute tuple counts and may crash otherwise.
    _validate_vtk_attribute_lengths(pyvista_mesh)
    uses_unstructured_topology = point_source == "cell_centroids" or manifold_dim != 0
    if isinstance(pyvista_mesh, pv.UnstructuredGrid) and uses_unstructured_topology:
        _validate_vtk_cell_array_structure(
            pyvista_mesh.GetCells(),
            "UnstructuredGrid cells",
            pyvista_mesh.n_cells,
        )
        cell_types = np.asarray(pyvista_mesh.celltypes)
        if cell_types.ndim != 1:
            raise ValueError(
                "Invalid UnstructuredGrid cell types: expected a "
                f"one-dimensional array, got shape {cell_types.shape}."
            )
        n_cell_types = len(cell_types)
        if n_cell_types != pyvista_mesh.n_cells:
            raise ValueError(
                "Invalid UnstructuredGrid cell types: expected "
                f"{pyvista_mesh.n_cells}, got {n_cell_types}."
            )
    polydata_topology = (
        _validate_polydata_topology(pyvista_mesh)
        if isinstance(pyvista_mesh, pv.PolyData)
        else None
    )

    ### Handle cell_centroids path (completely separate flow)
    if point_source == "cell_centroids":
        if isinstance(pyvista_mesh, pv.UnstructuredGrid):
            cell_types = np.asarray(pyvista_mesh.celltypes)
            unique_cell_types = np.unique(cell_types)
            centroid_cell_specs = _linear_cell_specs()
            centroid_scope = "linear dual-graph cell families"
            if manifold_dim in {"auto", 0}:
                centroid_cell_specs |= _fixed_higher_order_cell_specs()
                centroid_scope = (
                    "linear and fixed-size quadratic or cubic cell families"
                )
            _validate_cells_against_specs(
                pyvista_mesh,
                cell_types,
                unique_cell_types,
                centroid_cell_specs,
            )
            unsupported_types = _unsupported_cell_types(
                unique_cell_types,
                centroid_cell_specs,
            )
            if unsupported_types:
                names = ", ".join(
                    _vtk_cell_type_name(cell_type) for cell_type in unsupported_types
                )
                raise ValueError(
                    f"Unsupported VTK cell type(s) {names}: centroid filtering "
                    f"is only validated for {centroid_scope}."
                )
            empty_cell_indices = np.flatnonzero(cell_types == pv.CellType.EMPTY_CELL)
            if len(empty_cell_indices) > 0:
                raise ValueError(
                    f"VTK {pv.CellType.EMPTY_CELL.name} parents at indices "
                    f"{empty_cell_indices.tolist()} cannot produce cell centers; "
                    "omitting them would break cell_data alignment."
                )
        return _from_pyvista_cell_centroids(
            pyvista_mesh, manifold_dim, warn_on_lost_data, force_copy
        )

    ### Determine native mesh dimension (used for auto-detection, data-loss
    ### warnings, and deciding whether cell_data can be passed through).
    source_pyvista_mesh = pyvista_mesh
    native_cell_dimensions = np.empty(0, dtype=np.uint8)
    homogeneous_simplex_dim = None
    if manifold_dim == 0:
        # Point-cloud conversion consumes only points and point-associated data.
        native_dimensions = set()
        native_dim = 0
    elif isinstance(pyvista_mesh, pv.UnstructuredGrid):
        homogeneous_simplex_dim = _homogeneous_simplex_dimension(pyvista_mesh)
        if homogeneous_simplex_dim is not None:
            native_dimensions = {homogeneous_simplex_dim}
            native_dim = homogeneous_simplex_dim
        else:
            native_cell_dimensions = _unstructured_cell_dimensions(pyvista_mesh)
            native_dimensions = set(map(int, native_cell_dimensions)) or {0}
            native_dim = (
                int(native_cell_dimensions.max())
                if len(native_cell_dimensions) > 0
                else 0
            )
    else:
        native_dimensions = _detect_native_dimensions(pyvista_mesh)
        native_dim = max(native_dimensions)

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

    ### Preprocess mesh based on manifold dimension
    original_point_ids = None
    output_parent_ids = None
    polydata_tri_faces = None
    selected_unstructured_cells = False
    is_unstructured = isinstance(pyvista_mesh, pv.UnstructuredGrid)
    homogeneous_simplex_selected = bool(
        is_unstructured
        and homogeneous_simplex_dim is not None
        and manifold_dim == homogeneous_simplex_dim
    )
    has_native_1d_cells = bool(manifold_dim == 1 and 1 in native_dimensions)
    if homogeneous_simplex_selected:
        selected_unstructured_cells = True

    elif (
        is_unstructured
        and manifold_dim in {1, 2, 3}
        and (manifold_dim != 1 or has_native_1d_cells)
    ):
        if homogeneous_simplex_dim is not None:
            raise ValueError(
                f"UnstructuredGrid has no cells with manifold dimension "
                f"{manifold_dim}; available dimensions are "
                f"[{homogeneous_simplex_dim}]."
            )
        (
            pyvista_mesh,
            original_point_ids,
            output_parent_ids,
        ) = _select_and_linearize_unstructured_grid(
            pyvista_mesh, native_cell_dimensions, manifold_dim
        )
        selected_unstructured_cells = True

    elif manifold_dim == 2:
        if not isinstance(pyvista_mesh, pv.PolyData):
            raise NotImplementedError(
                f"Only PolyData and UnstructuredGrid are supported for manifold dimension 2, got {type(pyvista_mesh)=}."
            )
        if polydata_topology is None:
            raise RuntimeError("PolyData topology metadata was not initialized.")
        polydata_tri_faces, output_parent_ids = _polydata_surface_triangles(
            pyvista_mesh,
            polydata_topology,
        )

    elif manifold_dim == 3:
        raise ValueError(
            f"Expected an UnstructuredGrid with volume cells for 3D meshes, "
            f"but got {type(pyvista_mesh)=}."
        )

    ### Extract and convert geometry
    def _maybe_copy(arr: np.ndarray) -> np.ndarray:
        return arr.copy() if force_copy else arr

    geometry_source = (
        source_pyvista_mesh
        if original_point_ids is not None or output_parent_ids is not None
        else pyvista_mesh
    )

    # Preserve float32/float64 coordinates. Convert other coordinate dtypes to
    # float32, matching PhysicsNeMo's prior geometry contract.
    points = torch.from_numpy(_maybe_copy(geometry_source.points))
    if not points.is_floating_point() or points.element_size() < 4:
        points = points.float()

    # Cells
    if manifold_dim == 0:
        cells = None  # Mesh constructor creates empty cells

    elif manifold_dim == 1:
        line_offsets = None
        line_connectivity = None
        native_polydata_lines = False
        if selected_unstructured_cells:
            line_offsets, line_connectivity = _validate_vtk_cell_array_structure(
                pyvista_mesh.GetCells(),
                "linearized line cells",
                pyvista_mesh.n_cells,
            )
        elif (
            isinstance(pyvista_mesh, pv.PolyData)
            and pyvista_mesh.GetNumberOfLines() > 0
        ):
            if polydata_topology is None:
                raise RuntimeError("PolyData topology metadata was not initialized.")
            line_offsets, line_connectivity = polydata_topology["lines"]
            native_polydata_lines = True
        elif pyvista_mesh.n_cells > 0:
            # If no native lines exist, derive the unique edge graph from the
            # higher-dimensional topology.
            if isinstance(pyvista_mesh, pv.UnstructuredGrid):
                _validate_polyhedron_auxiliary_arrays(
                    pyvista_mesh,
                    np.asarray(pyvista_mesh.celltypes),
                )
            edges_mesh = pyvista_mesh.extract_all_edges()
            line_offsets, line_connectivity = _validate_vtk_cell_array_structure(
                edges_mesh.GetLines(),
                "extracted edge lines",
                edges_mesh.GetNumberOfLines(),
            )

        if line_offsets is None or line_connectivity is None:
            cells = torch.empty((0, 2), dtype=torch.long)
        else:
            line_segments = _line_segments_from_vtk_cell_array(
                line_offsets,
                line_connectivity,
            )
            cells = torch.from_numpy(line_segments).long()

            if native_polydata_lines:
                line_arities = np.diff(line_offsets)
                n_line_parents = pyvista_mesh.GetNumberOfLines()
                all_two_point_lines = bool(
                    len(line_arities) > 0
                    and int(line_arities.min()) == 2
                    and int(line_arities.max()) == 2
                )
                identity_parent_map = bool(
                    pyvista_mesh.GetNumberOfVerts() == 0
                    and pyvista_mesh.GetNumberOfPolys() == 0
                    and pyvista_mesh.GetNumberOfStrips() == 0
                    and n_line_parents == pyvista_mesh.n_cells
                    and n_line_parents > 0
                    and all_two_point_lines
                )
                if not identity_parent_map:
                    first_line_parent = pyvista_mesh.GetNumberOfVerts()
                    line_parent_ids = (
                        np.arange(n_line_parents, dtype=np.int64) + first_line_parent
                    )
                    output_parent_ids = np.repeat(
                        line_parent_ids,
                        line_arities - 1,
                    )

    elif manifold_dim == 2:
        # After triangulation, extract the (n_cells, 3) connectivity array
        if isinstance(pyvista_mesh, pv.PolyData):
            if polydata_tri_faces is None:
                raise RuntimeError("PolyData surface triangles were not initialized.")
            tri_faces = _maybe_copy(polydata_tri_faces)
        elif isinstance(pyvista_mesh, pv.UnstructuredGrid):
            # cells_dict materializes independent regular connectivity arrays.
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
            cell_type_names = ", ".join(
                _vtk_cell_type_name(int(cell_type)) for cell_type in cells_dict
            )
            raise ValueError(
                "Expected TETRA cells after triangulation, but got "
                f"{cell_type_names or 'no cells'}."
            )
        # cells_dict materializes independent regular connectivity arrays.
        tetra_cells = cells_dict[np.uint8(pv.CellType.TETRA)]
        cells = torch.from_numpy(tetra_cells).long()

    ### Restore source point IDs after dimension selection compacted the grid.
    if original_point_ids is not None and cells is not None:
        point_id_map = torch.from_numpy(original_point_ids).long()
        cells = point_id_map[cells]

    ### Warn only after target selection and all topology filters succeeded.
    if warn_on_lost_data:
        _warn_on_data_loss(
            source_pyvista_mesh,
            point_source="vertices",
            manifold_dim=manifold_dim,
            detected_dims=native_dimensions,
            warning_stacklevel=4,
        )

    ### Return Mesh object
    # Identity outputs can share aligned data directly. Every selected or split
    # output instead indexes the original data through its source-parent map.
    n_output_cells = 0 if cells is None else cells.shape[0]
    pass_cell_data = (
        manifold_dim > 0
        and n_output_cells == pyvista_mesh.n_cells
        and (selected_unstructured_cells or manifold_dim >= native_dim)
    )
    if output_parent_ids is not None:
        output_cell_data = _vtk_data_to_tensor_dict(
            source_pyvista_mesh.cell_data,
            force_copy,
            indices=output_parent_ids,
        )
    elif pass_cell_data:
        output_cell_data = _vtk_data_to_tensor_dict(
            pyvista_mesh.cell_data,
            force_copy,
        )
    else:
        output_cell_data = {}

    return Mesh(
        points=points,
        cells=cells,
        point_data=_vtk_data_to_tensor_dict(geometry_source.point_data, force_copy),
        cell_data=output_cell_data,
        global_data=_vtk_data_to_tensor_dict(
            source_pyvista_mesh.field_data, force_copy
        ),
    )


@require_version_spec("pyvista")
def to_pyvista(
    mesh: Mesh,
    *,
    force_copy: bool = False,
) -> "pv.PolyData | pv.UnstructuredGrid | pv.PointSet":
    """Convert a physicsnemo.mesh Mesh to a PyVista mesh.

    Parameters
    ----------
    mesh : Mesh
        Input physicsnemo.mesh Mesh object.
    force_copy : bool
        If True, copy geometry and attached data arrays so the returned
        PyVista object cannot mutate the source Mesh through shared CPU
        storage. When False (default), arrays may share storage for efficiency.

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

    Notes
    -----
    ``float32`` and ``float64`` point coordinates are exported without
    narrowing; other coordinate dtypes are converted to ``float32``. To
    normalize a mesh and all its floating data before export, use
    ``to_pyvista(mesh.to(torch.float32))``. Retaining ``float64`` coordinates
    doubles their storage relative to ``float32`` and may keep downstream
    PyVista computations in double precision.
    """
    ### Convert points to numpy and pad to 3D if needed (PyVista requires 3D points)
    # .detach() first so a grad-tracked mesh can still be exported (.numpy() would
    # otherwise raise on a tensor that requires grad).
    points_np = _geometry_to_vtk_numpy(mesh.points)

    if mesh.n_spatial_dims < 3:
        # Pad with zeros to make 3D. np.pad already returns independent storage.
        padding_width = 3 - mesh.n_spatial_dims
        points_np = np.pad(
            points_np,
            ((0, 0), (0, padding_width)),
            mode="constant",
            constant_values=0.0,
        )
    elif force_copy:
        points_np = points_np.copy()

    ### Convert based on manifold dimension
    if mesh.n_manifold_dims == 0:
        pv_mesh = pv.PointSet(points_np)

    elif mesh.n_manifold_dims == 1:
        cells_np = mesh.cells.cpu().numpy()
        if mesh.n_cells == 0:
            pv_mesh = pv.PolyData(points_np)
        else:
            # _to_vtk_cell_array returns independent VTK-format connectivity.
            pv_mesh = pv.PolyData(points_np, lines=_to_vtk_cell_array(cells_np))

    elif mesh.n_manifold_dims == 2:
        cells_np = mesh.cells.cpu().numpy()
        if mesh.n_cells == 0:
            pv_mesh = pv.PolyData(points_np)
        else:
            if force_copy:
                cells_np = cells_np.copy()
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
            # _to_vtk_cell_array returns independent VTK-format connectivity.
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
            arr = arr.reshape(arr.shape[0], -1) if arr.ndim > 2 else arr
            target[str(k)] = arr.copy() if force_copy else arr

    return pv_mesh


def _from_pyvista_cell_centroids(
    pyvista_mesh: "pv.PolyData | pv.UnstructuredGrid",
    manifold_dim: int | Literal["auto"],
    warn_on_lost_data: bool,
    force_copy: bool,
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
    force_copy : bool
        Copy attached data arrays instead of sharing their storage.

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
            warning_stacklevel=5,
        )

    ### Compute cell centroids with VTK's fast C++ filter.
    centroids_np = pyvista_mesh.cell_centers().points
    points = torch.from_numpy(centroids_np.copy())
    if not points.is_floating_point() or points.element_size() < 4:
        points = points.float()

    ### Build cells
    if manifold_dim == 0:
        cells = None  # Mesh constructor creates empty cells
    else:
        # Dual graph: edges connect cells that share a face.
        cells = _build_dual_graph_edges(pyvista_mesh)

    return Mesh(
        points=points,
        cells=cells,
        point_data=_vtk_data_to_tensor_dict(pyvista_mesh.cell_data, force_copy),
        global_data=_vtk_data_to_tensor_dict(pyvista_mesh.field_data, force_copy),
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


def _detect_native_dimensions(
    pyvista_mesh: "pv.PolyData | pv.PointSet",
) -> set[int]:
    """Return native dimensions represented by a non-UnstructuredGrid dataset.

    Parameters
    ----------
    pyvista_mesh : pyvista.PolyData or pyvista.PointSet
        Input mesh.

    Returns
    -------
    set[int]
        Non-empty subset of ``{0, 1, 2}``.
    """
    if pyvista_mesh.n_cells == 0:
        return {0}
    n_lines = _get_count_safely(pyvista_mesh, "n_lines")
    n_cells = _get_count_safely(pyvista_mesh, "n_cells")
    n_verts = _get_count_safely(pyvista_mesh, "n_verts")
    dimensions = set()
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
    warning_stacklevel: int,
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
        Native manifold dimensions represented by the original mesh.
        ``None`` when called from the cell_centroids path.
    warning_stacklevel : int
        Stack level that resolves to the public caller.
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
                stacklevel=warning_stacklevel,
            )

    ### Case 2: cell_data tuples lost when selecting one native dimension.
    if (
        point_source == "vertices"
        and detected_dims is not None
        and pyvista_mesh.n_cells > 0
    ):
        preserved_dims = (
            {manifold_dim}
            if manifold_dim > 0 and manifold_dim in detected_dims
            else set()
        )
        dropped_dims = sorted(detected_dims - preserved_dims)
        cd_keys = list(pyvista_mesh.cell_data.keys())
        drops_all_parents = manifold_dim == 0 and pyvista_mesh.n_cells > 0
        if (dropped_dims or drops_all_parents) and cd_keys:
            dropped_description = (
                f"native dimensions {dropped_dims}"
                if dropped_dims
                else "all uninterpreted topology dimensions"
            )
            warnings.warn(
                f"manifold_dim={manifold_dim} with point_source='vertices' "
                f"drops parent cells from {dropped_description} and discards "
                f"their cell_data values in {len(cd_keys)} field(s): "
                f"{cd_keys}. Handle those parent values before conversion, "
                "or set warn_on_lost_data=False to silence this warning.",
                UserWarning,
                stacklevel=warning_stacklevel,
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
