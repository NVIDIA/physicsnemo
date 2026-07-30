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
        if force_copy:
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
    if len(offsets) == 0 or int(offsets[0]) != 0:
        raise ValueError(f"Invalid {association}: offsets must start at 0.")
    if int(offsets[-1]) != len(connectivity):
        raise ValueError(
            f"Invalid {association}: final offset {int(offsets[-1])} does not "
            f"equal connectivity length {len(connectivity)}."
        )
    if len(offsets) > 1 and bool((np.diff(offsets) < 0).any()):
        raise ValueError(f"Invalid {association}: offsets must be monotonic.")
    return offsets, connectivity


def _parse_polydata_cell_stream(
    stream: np.ndarray,
    association: str,
    minimum_arity: int,
    expected_cells: int,
    n_points: int,
) -> int | np.ndarray:
    """Validate a count-prefixed PolyData cell stream and return arities."""
    stream = np.asarray(stream)
    if len(stream) == 0:
        if expected_cells != 0:
            raise ValueError(
                f"Invalid PolyData {association} stream: expected "
                f"{expected_cells} cells, got 0."
            )
        return 0

    first_count = int(stream[0])
    stride = first_count + 1
    if first_count >= minimum_arity and len(stream) % stride == 0:
        reshaped = stream.reshape(-1, stride)
        if bool((reshaped[:, 0] == first_count).all()):
            if len(reshaped) != expected_cells:
                raise ValueError(
                    f"Invalid PolyData {association} stream: expected "
                    f"{expected_cells} cells, parsed {len(reshaped)}."
                )
            point_ids = reshaped[:, 1:]
            if len(point_ids) > 0 and (
                int(point_ids.min()) < 0 or int(point_ids.max()) >= n_points
            ):
                bad_id = int(point_ids[(point_ids < 0) | (point_ids >= n_points)][0])
                raise ValueError(
                    f"Invalid point ID {bad_id} in PolyData {association}: "
                    f"valid point IDs are in [0, {n_points})."
                )
            return first_count

    counts: list[int] = []
    offset = 0
    while offset < len(stream):
        count = int(stream[offset])
        cell_index = len(counts)
        if count < minimum_arity:
            raise ValueError(
                f"Invalid PolyData {association} cell at index {cell_index}: "
                f"expected at least {minimum_arity} points, got {count}."
            )
        end = offset + count + 1
        if end > len(stream):
            raise ValueError(
                f"Invalid PolyData {association} stream at cell {cell_index}: "
                f"declared {count} points but only {len(stream) - offset - 1} "
                "remain."
            )
        point_ids = stream[offset + 1 : end]
        if int(point_ids.min()) < 0 or int(point_ids.max()) >= n_points:
            bad_id = int(point_ids[(point_ids < 0) | (point_ids >= n_points)][0])
            raise ValueError(
                f"Invalid point ID {bad_id} in PolyData {association} cell at "
                f"index {cell_index}: valid point IDs are in [0, {n_points})."
            )
        counts.append(count)
        offset = end

    if offset != len(stream) or len(counts) != expected_cells:
        raise ValueError(
            f"Invalid PolyData {association} stream: expected {expected_cells} "
            f"cells, parsed {len(counts)}, ending at {offset} of {len(stream)}."
        )
    return np.asarray(counts, dtype=np.int64)


def _validate_polydata_topology(
    pyvista_mesh: "pv.PolyData",
) -> dict[str, int | np.ndarray]:
    """Validate every PolyData cell stream without invoking VTK filters."""
    specifications = (
        (
            "verts",
            pyvista_mesh.GetVerts(),
            pyvista_mesh.verts,
            1,
            pyvista_mesh.GetNumberOfVerts(),
        ),
        (
            "lines",
            pyvista_mesh.GetLines(),
            pyvista_mesh.lines,
            2,
            pyvista_mesh.GetNumberOfLines(),
        ),
        (
            "faces",
            pyvista_mesh.GetPolys(),
            pyvista_mesh.faces,
            3,
            pyvista_mesh.GetNumberOfPolys(),
        ),
        (
            "strips",
            pyvista_mesh.GetStrips(),
            pyvista_mesh.strips,
            3,
            pyvista_mesh.GetNumberOfStrips(),
        ),
    )
    counts = {}
    for (
        association,
        cell_array,
        stream,
        minimum_arity,
        expected_cells,
    ) in specifications:
        _validate_vtk_cell_array_structure(
            cell_array,
            f"PolyData {association}",
            int(expected_cells),
        )
        counts[association] = _parse_polydata_cell_stream(
            stream,
            association,
            minimum_arity,
            int(expected_cells),
            pyvista_mesh.n_points,
        )
    return counts


def _validate_unstructured_connectivity_bounds(
    pyvista_mesh: "pv.UnstructuredGrid",
    cell_types: np.ndarray,
) -> None:
    """Reject invalid point IDs without allocating a full-size success mask."""
    _, connectivity = _validate_vtk_cell_array_structure(
        pyvista_mesh.GetCells(),
        "UnstructuredGrid cells",
        pyvista_mesh.n_cells,
    )
    if len(connectivity) == 0:
        return
    minimum_id = int(connectivity.min())
    maximum_id = int(connectivity.max())
    if minimum_id >= 0 and maximum_id < pyvista_mesh.n_points:
        return

    # Allocate an elementwise mask only on the malformed error path.
    invalid_connectivity = (connectivity < 0) | (connectivity >= pyvista_mesh.n_points)
    connectivity_index = int(np.flatnonzero(invalid_connectivity)[0])
    offsets = np.asarray(pyvista_mesh.offset)
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


def _validate_polyhedron_auxiliary_arrays(
    pyvista_mesh: "pv.UnstructuredGrid",
    cell_types: np.ndarray,
) -> None:
    """Validate VTK polyhedron face and face-location arrays."""
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


def _validate_supported_linear_cells(
    pyvista_mesh: "pv.UnstructuredGrid",
    cell_types: np.ndarray,
    unique_cell_types: np.ndarray,
) -> None:
    """Validate bounds and arities for supported linear cells only."""
    _validate_unstructured_connectivity_bounds(pyvista_mesh, cell_types)
    _validate_polyhedron_auxiliary_arrays(pyvista_mesh, cell_types)
    if len(cell_types) == 0:
        return

    arities = np.diff(np.asarray(pyvista_mesh.offset))
    linear_specs = _linear_cell_specs()
    for cell_type_value in unique_cell_types:
        cell_type = int(cell_type_value)
        spec = linear_specs.get(cell_type)
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
    _validate_supported_linear_cells(
        pyvista_mesh,
        cell_types,
        unique_cell_types,
    )

    ### Reject every topology family outside the explicit linear allowlist.
    linear_specs = _linear_cell_specs()
    unsupported_types = [
        int(cell_type)
        for cell_type in unique_cell_types
        if int(cell_type) not in linear_specs
    ]
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


def _select_and_linearize_unstructured_grid(
    pyvista_mesh: "pv.UnstructuredGrid",
    cell_dimensions: np.ndarray,
    target_dim: int,
) -> tuple["pv.UnstructuredGrid", np.ndarray | None]:
    """Select one native dimension and convert its cells to simplices.

    The returned point-ID map is present only when cell extraction compacted
    the point array.
    """
    cell_types = np.asarray(pyvista_mesh.celltypes)
    selected_parent_ids = np.flatnonzero(cell_dimensions == target_dim).astype(np.int64)
    if len(selected_parent_ids) == 0:
        available_dimensions = sorted(set(map(int, cell_dimensions)))
        raise ValueError(
            f"UnstructuredGrid has no cells with manifold dimension {target_dim}; "
            f"available dimensions are {available_dimensions}."
        )

    ### Extract only the requested native dimension. PyVista compacts points,
    ### while vtkOriginalPointIds records how to restore source connectivity.
    selected_all_cells = len(selected_parent_ids) == pyvista_mesh.n_cells
    if selected_all_cells:
        selected = pyvista_mesh
    else:
        original_cell_id_key = "vtkOriginalCellIds"
        user_original_cell_ids = (
            np.asarray(pyvista_mesh.cell_data[original_cell_id_key])[
                selected_parent_ids
            ].copy()
            if original_cell_id_key in pyvista_mesh.cell_data
            else None
        )
        try:
            selected = pyvista_mesh.extract_cells(
                selected_parent_ids,
                pass_cell_ids=False,
                pass_point_ids=True,
            )
        except TypeError as error:
            # PyVista 0.46 does not expose the pass_*_ids keywords and always
            # adds both synthetic arrays. Remove its cell IDs while restoring
            # any user field that occupied the same name.
            if "pass_cell_ids" not in str(error) and "pass_point_ids" not in str(error):
                raise
            # PyVista 0.46 writes synthetic ID fields onto the input dataset.
            # Isolate that mutation to a shallow copy with separate attribute
            # containers while retaining zero-copy geometry/data buffers.
            legacy_source = pyvista_mesh.copy(deep=False)
            selected = legacy_source.extract_cells(selected_parent_ids)
        if original_cell_id_key in selected.cell_data:
            del selected.cell_data[original_cell_id_key]
        if user_original_cell_ids is not None:
            selected.cell_data[original_cell_id_key] = user_original_cell_ids

    simplex_type = {
        1: pv.CellType.LINE,
        2: pv.CellType.TRIANGLE,
        3: pv.CellType.TETRA,
    }[target_dim]
    if bool((selected.celltypes == simplex_type).all()):
        original_point_ids = (
            None
            if selected_all_cells
            else np.asarray(selected.point_data["vtkOriginalPointIds"]).copy()
        )
        return selected, original_point_ids

    ### Add collision-safe parent provenance to a non-mutating shallow copy.
    working = selected.copy(deep=False)
    provenance_key = "__physicsnemo_parent_cell_id"
    suffix = 0
    while provenance_key in working.cell_data:
        suffix += 1
        provenance_key = f"__physicsnemo_parent_cell_id_{suffix}"
    working.cell_data[provenance_key] = selected_parent_ids

    linearized = working.triangulate()
    if provenance_key in linearized.cell_data:
        output_parent_ids = np.asarray(
            linearized.cell_data[provenance_key], dtype=np.int64
        ).copy()
        del linearized.cell_data[provenance_key]
    else:
        output_parent_ids = np.empty(0, dtype=np.int64)

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
    if len(output_parent_ids) != linearized.n_cells:
        raise ValueError(
            "VTK simplex conversion did not preserve one provenance value per "
            f"output cell: expected {linearized.n_cells}, got "
            f"{len(output_parent_ids)}."
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
        else np.asarray(linearized.point_data["vtkOriginalPointIds"]).copy()
    )
    return linearized, original_point_ids


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
        If manifold dimension cannot be determined or is invalid.
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
    ``point_source="vertices"`` may still preserve their points without
    interpreting cell topology. Centroid filtering follows the same strict
    linear allowlist and additionally rejects ``EMPTY_CELL`` parents because
    VTK omits their centers.
    """
    ### Validate point_source
    if point_source not in {"vertices", "cell_centroids"}:
        raise ValueError(
            f"Invalid {point_source=!r}. Must be 'vertices' or 'cell_centroids'."
        )

    # VTK filters assume valid attribute tuple counts and may crash otherwise.
    _validate_vtk_attribute_lengths(pyvista_mesh)
    if isinstance(pyvista_mesh, pv.UnstructuredGrid):
        _validate_vtk_cell_array_structure(
            pyvista_mesh.GetCells(),
            "UnstructuredGrid cells",
            pyvista_mesh.n_cells,
        )
    polydata_cell_counts = (
        _validate_polydata_topology(pyvista_mesh)
        if isinstance(pyvista_mesh, pv.PolyData)
        else None
    )

    ### Handle cell_centroids path (completely separate flow)
    if point_source == "cell_centroids":
        if isinstance(pyvista_mesh, pv.UnstructuredGrid):
            cell_types = np.asarray(pyvista_mesh.celltypes)
            unique_cell_types = np.unique(cell_types)
            _validate_supported_linear_cells(
                pyvista_mesh,
                cell_types,
                unique_cell_types,
            )
            unsupported_types = sorted(
                {
                    int(cell_type)
                    for cell_type in unique_cell_types
                    if int(cell_type) not in _linear_cell_specs()
                }
            )
            if unsupported_types:
                names = ", ".join(
                    _vtk_cell_type_name(cell_type) for cell_type in unsupported_types
                )
                raise ValueError(
                    f"Unsupported VTK cell type(s) {names}: centroid filtering "
                    "is outside this safe-linear conversion scope."
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
    if isinstance(pyvista_mesh, pv.UnstructuredGrid):
        homogeneous_simplex_dim = _homogeneous_simplex_dimension(pyvista_mesh)
        if homogeneous_simplex_dim is not None:
            native_dimensions = {homogeneous_simplex_dim}
            native_dim = homogeneous_simplex_dim
        else:
            unique_cell_types = set(map(int, np.unique(pyvista_mesh.celltypes)))
            has_unsupported_topology = not unique_cell_types.issubset(
                _linear_cell_specs()
            )
            if manifold_dim == 0 and has_unsupported_topology:
                # Point-cloud conversion does not inspect cell connectivity.
                native_dimensions = set()
                native_dim = 0
            else:
                native_cell_dimensions = _unstructured_cell_dimensions(pyvista_mesh)
                native_dimensions = set(map(int, native_cell_dimensions)) or {0}
                native_dim = (
                    int(native_cell_dimensions.max())
                    if len(native_cell_dimensions) > 0
                    else 0
                )
    else:
        native_dim = _detect_native_dim(pyvista_mesh)
        native_dimensions = {native_dim}
        if isinstance(pyvista_mesh, pv.PolyData):
            n_verts = _get_count_safely(pyvista_mesh, "n_verts")
            n_lines = _get_count_safely(pyvista_mesh, "n_lines")
            native_dimensions = set()
            if n_verts > 0:
                native_dimensions.add(0)
            if n_lines > 0:
                native_dimensions.add(1)
            if pyvista_mesh.n_cells > n_verts + n_lines:
                native_dimensions.add(2)
            if not native_dimensions:
                native_dimensions.add(0)

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

    ### Preprocess mesh based on manifold dimension
    original_point_ids = None
    polydata_surface_parent_ids = None
    selected_unstructured_cells = False
    is_unstructured = isinstance(pyvista_mesh, pv.UnstructuredGrid)
    homogeneous_simplex_selected = bool(
        is_unstructured
        and homogeneous_simplex_dim is not None
        and manifold_dim == homogeneous_simplex_dim
    )
    has_native_1d_cells = bool(
        manifold_dim == 1
        and (
            homogeneous_simplex_dim == 1
            or (len(native_cell_dimensions) > 0 and (native_cell_dimensions == 1).any())
        )
    )
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
        pyvista_mesh, original_point_ids = _select_and_linearize_unstructured_grid(
            pyvista_mesh,
            native_cell_dimensions,
            manifold_dim,
        )
        selected_unstructured_cells = True

    elif manifold_dim == 2:
        if not isinstance(pyvista_mesh, pv.PolyData):
            raise NotImplementedError(
                f"Only PolyData and UnstructuredGrid are supported for manifold dimension 2, got {type(pyvista_mesh)=}."
            )
        if not pyvista_mesh.is_all_triangles:
            if polydata_cell_counts is None:
                raise RuntimeError("PolyData topology metadata was not initialized.")
            n_verts = pyvista_mesh.GetNumberOfVerts()
            n_lines = pyvista_mesh.GetNumberOfLines()
            n_faces = pyvista_mesh.GetNumberOfPolys()
            n_strips = pyvista_mesh.GetNumberOfStrips()
            if n_faces + n_strips == 0:
                raise ValueError(
                    "PolyData has no native surface cells for manifold_dim=2."
                )
            first_face_parent = n_verts + n_lines
            selected_surface_parents = np.concatenate(
                [
                    np.arange(
                        first_face_parent,
                        first_face_parent + n_faces,
                        dtype=np.int64,
                    ),
                    np.arange(
                        first_face_parent + n_faces,
                        first_face_parent + n_faces + n_strips,
                        dtype=np.int64,
                    ),
                ]
            )

            surface = pv.PolyData(
                pyvista_mesh.points,
                faces=pyvista_mesh.faces,
                strips=pyvista_mesh.strips,
            )
            provenance_key = "__physicsnemo_parent_cell_id"
            suffix = 0
            while provenance_key in pyvista_mesh.cell_data:
                suffix += 1
                provenance_key = f"__physicsnemo_parent_cell_id_{suffix}"
            surface.cell_data[provenance_key] = selected_surface_parents
            triangulated = surface.triangulate()
            output_face_counts = _parse_polydata_cell_stream(
                triangulated.faces,
                "faces",
                3,
                triangulated.GetNumberOfPolys(),
                triangulated.n_points,
            )
            all_output_triangles = (
                output_face_counts == 3
                if isinstance(output_face_counts, int)
                else bool((output_face_counts == 3).all())
            )
            if not all_output_triangles:
                raise ValueError("VTK triangulation left non-triangle PolyData faces.")
            polydata_surface_parent_ids = np.asarray(
                triangulated.cell_data[provenance_key],
                dtype=np.int64,
            ).copy()
            del triangulated.cell_data[provenance_key]
            missing_parent_ids = np.setdiff1d(
                selected_surface_parents,
                np.unique(polydata_surface_parent_ids),
                assume_unique=True,
            )
            if len(missing_parent_ids) > 0:
                raise ValueError(
                    "VTK triangulation dropped selected PolyData surface "
                    f"parents {missing_parent_ids.tolist()}."
                )
            pyvista_mesh = triangulated

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
        if original_point_ids is not None or polydata_surface_parent_ids is not None
        else pyvista_mesh
    )

    # Preserve float32/float64 coordinates. Convert other coordinate dtypes to
    # float32, matching PhysicsNeMo's prior geometry contract.
    points = torch.from_numpy(_maybe_copy(geometry_source.points))
    if not points.is_floating_point() or points.element_size() < 4:
        points = points.float()

    # Cells
    polydata_line_parent_ids = None
    if manifold_dim == 0:
        cells = None  # Mesh constructor creates empty cells

    elif manifold_dim == 1:
        # Lines - extract from PyVista lines format.
        # If the mesh has no native lines (e.g., a 3D volume mesh with
        # manifold_dim=1 requested explicitly), extract all unique edges
        # from the mesh topology to build a vertex graph.
        if selected_unstructured_cells:
            # Once linearized, the legacy cell array has exactly the same
            # count-prefixed layout as ``PolyData.lines``.
            lines_raw = pyvista_mesh.cells
            native_polydata_lines = False
        else:
            lines_raw = getattr(pyvista_mesh, "lines", None)
            native_polydata_lines = bool(
                isinstance(pyvista_mesh, pv.PolyData)
                and lines_raw is not None
                and len(lines_raw) > 0
            )

        if (lines_raw is None or len(lines_raw) == 0) and pyvista_mesh.n_cells > 0:
            edges_mesh = pyvista_mesh.extract_all_edges()
            lines_raw = edges_mesh.lines
            native_polydata_lines = False

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

            if native_polydata_lines:
                if polydata_cell_counts is None:
                    raise RuntimeError(
                        "PolyData topology metadata was not initialized."
                    )
                line_point_counts = polydata_cell_counts["lines"]
                n_line_parents = pyvista_mesh.GetNumberOfLines()
                all_two_point_lines = (
                    line_point_counts == 2
                    if isinstance(line_point_counts, int)
                    else bool(
                        len(line_point_counts) > 0
                        and int(line_point_counts.min()) == 2
                        and int(line_point_counts.max()) == 2
                    )
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
                    polydata_line_parent_ids = np.repeat(
                        line_parent_ids,
                        line_point_counts - 1,
                    )

    elif manifold_dim == 2:
        # After triangulation, extract the (n_cells, 3) connectivity array
        if isinstance(pyvista_mesh, pv.PolyData):
            tri_faces = _maybe_copy(pyvista_mesh.regular_faces)
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
    # Cell data can only be passed through when the output cells have a
    # 1:many relationship with input cells (e.g., VTK's triangulate
    # replicates cell_data to child cells). Explicit UnstructuredGrid
    # selection preserves only the selected parents; other lower-dimensional
    # transformations and 0D output cannot pass cell data through.
    n_output_cells = 0 if cells is None else cells.shape[0]
    pass_cell_data = (
        manifold_dim > 0
        and n_output_cells == pyvista_mesh.n_cells
        and (selected_unstructured_cells or manifold_dim >= native_dim)
    )
    if polydata_surface_parent_ids is not None:
        output_cell_data = _vtk_data_to_tensor_dict(
            source_pyvista_mesh.cell_data,
            force_copy,
            indices=polydata_surface_parent_ids,
        )
    elif polydata_line_parent_ids is not None:
        output_cell_data = _vtk_data_to_tensor_dict(
            source_pyvista_mesh.cell_data,
            force_copy,
            indices=polydata_line_parent_ids,
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

    ### Compute cell centroids (fast C++ filter, works for all cell types)
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


def _detect_native_dim(
    pyvista_mesh: "pv.PolyData | pv.PointSet",
) -> int:
    """Determine the native dimension of a non-UnstructuredGrid dataset.

    Parameters
    ----------
    pyvista_mesh : pyvista.PolyData or pyvista.PointSet
        Input mesh.

    Returns
    -------
    int
        0, 1, 2, or 3.
    """
    if pyvista_mesh.n_cells == 0:
        return 0
    n_lines = _get_count_safely(pyvista_mesh, "n_lines")
    n_cells = _get_count_safely(pyvista_mesh, "n_cells")
    n_verts = _get_count_safely(pyvista_mesh, "n_verts")
    if n_cells > n_verts + n_lines:
        return 2
    if n_lines > 0:
        return 1
    return 0


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
            if isinstance(pyvista_mesh, pv.UnstructuredGrid):
                unique_cell_types = np.unique(pyvista_mesh.celltypes)
                has_empty_parent = bool(pv.CellType.EMPTY_CELL in unique_cell_types)
                linear_cell_types = _linear_cell_specs()
                unsupported_parent_types = sorted(
                    int(cell_type)
                    for cell_type in unique_cell_types
                    if int(cell_type) not in linear_cell_types
                )
            else:
                has_empty_parent = False
                unsupported_parent_types = []
            if has_empty_parent:
                remediation = (
                    "EMPTY_CELL parent cell_data cannot be preserved because "
                    "point-cloud output has no cells and centroid mode rejects "
                    "EMPTY_CELL; handle or remove those parent tuples before "
                    "conversion."
                )
            elif unsupported_parent_types:
                names = ", ".join(
                    _vtk_cell_type_name(cell_type)
                    for cell_type in unsupported_parent_types
                )
                remediation = (
                    f"Parent cell_data for unsupported topology {names} cannot "
                    "be preserved by this point-cloud conversion, and centroid "
                    "mode is outside the safe-linear scope; handle those tuples "
                    "before conversion."
                )
            else:
                remediation = (
                    "Use point_source='cell_centroids' to preserve all "
                    "cell_data as point_data, or set warn_on_lost_data=False to "
                    "silence this warning."
                )
            warnings.warn(
                f"manifold_dim={manifold_dim} with point_source='vertices' "
                f"drops parent cells from {dropped_description} and discards "
                f"their cell_data values in {len(cd_keys)} field(s): "
                f"{cd_keys}. {remediation}",
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
