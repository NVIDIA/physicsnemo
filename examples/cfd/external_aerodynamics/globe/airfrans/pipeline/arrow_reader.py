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
AirFRANS Arrow Reader - Reads samples from the PLAID/HuggingFace AirfRANS dataset.

Each sample is a pickled CGNS tree containing a 2D unstructured mesh with
vertex/cell fields from RANS simulations around airfoils. The reader
performs I/O only: deserializes the pickle, extracts raw field data, and
emits mesh connectivity tensors. Derived quantities (gradients, normals)
are computed by downstream transforms.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import types
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyvista as pv
import torch

from physicsnemo.datapipes.readers.base import Reader

logger = logging.getLogger(__name__)


def _ensure_plaid_stub() -> None:
    """Register a stub ``plaid.containers.sample.Sample`` class so that
    ``pickle.loads`` can deserialize PLAID binary blobs without having the
    ``plaid`` package installed."""
    if "plaid" not in sys.modules:
        plaid = types.ModuleType("plaid")
        plaid_containers = types.ModuleType("plaid.containers")
        plaid_containers_sample = types.ModuleType("plaid.containers.sample")

        class Sample:
            pass

        plaid_containers_sample.Sample = Sample
        plaid.containers = plaid_containers
        plaid_containers.sample = plaid_containers_sample
        sys.modules["plaid"] = plaid
        sys.modules["plaid.containers"] = plaid_containers
        sys.modules["plaid.containers.sample"] = plaid_containers_sample


def _parse_cgns_tree(sample_blob: bytes) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Deserialize a PLAID binary blob and extract arrays from its CGNS tree.

    Returns
    -------
    arrays : dict[str, np.ndarray]
        Vertex fields, coordinates, connectivity, and cell fields.
    scalars : dict[str, float]
        Global scalars (angle_of_attack, inlet_velocity, C_D, C_L).
    """
    _ensure_plaid_stub()
    sample = pickle.loads(sample_blob)  # noqa: S301
    d = sample.__dict__["__dict__"]

    scalars: dict[str, float] = dict(d["_scalars"])

    cgns_tree = d["_meshes"][0.0]
    base = cgns_tree[2][1]
    zone = base[2][1]

    grid_coords = zone[2][1]
    elements = zone[2][2]
    vertex_fields = zone[2][3]
    cell_fields = zone[2][4]

    arrays: dict[str, np.ndarray] = {}

    arrays["x"] = np.asarray(grid_coords[2][0][1], dtype=np.float32)
    arrays["y"] = np.asarray(grid_coords[2][1][1], dtype=np.float32)

    conn_raw = np.asarray(elements[2][1][1])
    arrays["connectivity"] = conn_raw - 1
    elem_type = int(elements[1][0])
    arrays["element_type"] = np.array([elem_type], dtype=np.int32)
    elem_range = np.asarray(elements[2][0][1])
    arrays["n_elements"] = np.array(
        [elem_range[1] - elem_range[0] + 1], dtype=np.int64
    )

    for child in vertex_fields[2]:
        name, value, _, ntype = child
        if ntype == "DataArray_t" and name not in ("vtkOriginalPointIds",):
            arrays[f"vertex_{name}"] = np.asarray(value, dtype=np.float32)

    for child in cell_fields[2]:
        name, value, _, ntype = child
        if ntype == "DataArray_t" and name not in (
            "vtkOriginalCellIds",
            "cell_ids",
        ):
            arrays[f"cell_{name}"] = np.asarray(value, dtype=np.float32)

    return arrays, scalars


def _build_pyvista_mesh(arrays: dict[str, np.ndarray]) -> pv.UnstructuredGrid:
    """Construct a PyVista UnstructuredGrid from parsed CGNS arrays."""
    x = arrays["x"]
    y = arrays["y"]
    n_points = len(x)
    points_3d = np.column_stack([x, y, np.zeros(n_points, dtype=np.float32)])

    conn = arrays["connectivity"]
    n_elems = int(arrays["n_elements"][0])
    elem_type = int(arrays["element_type"][0])

    if elem_type == 7:
        nodes_per_elem = 4
        vtk_cell_type = 9  # VTK_QUAD
    elif elem_type == 5:
        nodes_per_elem = 3
        vtk_cell_type = 5  # VTK_TRIANGLE
    else:
        raise ValueError(f"Unsupported CGNS element type: {elem_type}")

    conn_reshaped = conn.reshape(n_elems, nodes_per_elem)

    vtk_conn = np.column_stack(
        [np.full(n_elems, nodes_per_elem, dtype=np.int64), conn_reshaped]
    ).ravel()
    cell_types = np.full(n_elems, vtk_cell_type, dtype=np.uint8)

    mesh = pv.UnstructuredGrid(vtk_conn, cell_types, points_3d)

    for key, val in arrays.items():
        if key.startswith("vertex_") and len(val) == n_points:
            mesh.point_data[key.removeprefix("vertex_")] = val

    for key, val in arrays.items():
        if key.startswith("cell_") and len(val) == n_elems:
            mesh.cell_data[key.removeprefix("cell_")] = val

    if "vertex_Ux" in arrays and "vertex_Uy" in arrays:
        mesh.point_data["U"] = np.column_stack(
            [arrays["vertex_Ux"], arrays["vertex_Uy"], np.zeros(n_points, dtype=np.float32)]
        )
    if "cell_Ux" in arrays and "cell_Uy" in arrays:
        mesh.cell_data["U"] = np.column_stack(
            [arrays["cell_Ux"], arrays["cell_Uy"], np.zeros(n_elems, dtype=np.float32)]
        )

    return mesh


class AirFRANSArrowReader(Reader):
    """Reader for the PLAID/HuggingFace AirfRANS dataset (I/O only).

    Loads samples from the Arrow dataset on disk, deserializes each sample's
    pickled CGNS tree, and extracts raw field data plus mesh connectivity
    tensors. Derived quantities are computed by downstream transforms.

    Parameters
    ----------
    dataset_path : str or Path
        Path to the HuggingFace dataset saved to disk (the directory
        containing ``dataset_dict.json``).
    task : str
        AirFRANS task name. One of ``"full"``, ``"scarce"``,
        ``"reynolds"``, ``"aoa"``.
    split : str
        Dataset split. One of ``"train"``, ``"test"``.
    pin_memory : bool
        If ``True``, place tensors in pinned memory for faster GPU transfer.
    include_index_in_metadata : bool
        If ``True``, include the sample index in the metadata dict.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        task: Literal["full", "scarce", "reynolds", "aoa"] = "full",
        split: Literal["train", "test"] = "train",
        pin_memory: bool = False,
        include_index_in_metadata: bool = True,
    ) -> None:
        super().__init__(
            pin_memory=pin_memory,
            include_index_in_metadata=include_index_in_metadata,
        )

        self.dataset_path = Path(dataset_path)
        self.task = task
        self.split = split

        from datasets import load_from_disk

        full_ds = load_from_disk(str(self.dataset_path))
        self._dataset = full_ds["all_samples"]

        info_path = self.dataset_path / "all_samples" / "dataset_info.json"
        with open(info_path) as f:
            info = json.load(f)

        split_info = info.get("description", {}).get("split", {})
        effective_task = "full" if (task == "scarce" and split == "test") else task
        split_key = f"{effective_task}_{split}"

        if split_key not in split_info:
            available = sorted(split_info.keys())
            raise ValueError(
                f"Split '{split_key}' not found. Available: {available}"
            )

        self._indices: list[int] = split_info[split_key]
        logger.info(
            "AirFRANSArrowReader: task=%s, split=%s -> %d samples",
            task,
            split,
            len(self._indices),
        )

    def __len__(self) -> int:
        return len(self._indices)

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        dataset_index = self._indices[index]

        blob = self._dataset[dataset_index]["sample"]
        arrays, scalars = _parse_cgns_tree(blob)

        mesh = _build_pyvista_mesh(arrays)

        # Triangulate and extract internal cells
        if not (set(mesh.cells_dict.keys()) == {pv.CellType.TRIANGLE}):
            mesh = mesh.triangulate()
        internal_cells = mesh.cells_dict[pv.CellType.TRIANGLE]

        # Extract airfoil boundary from internal mesh
        sdf = mesh.point_data.get("implicit_distance")
        if sdf is not None:
            on_surface = sdf == 0
            surface_ids = np.where(on_surface)[0]
            if len(surface_ids) > 0:
                surface_mesh = mesh.extract_points(surface_ids).extract_surface()
                airfoil_pts = surface_mesh.points[:, :2].astype(np.float32)
                lines = np.asarray(surface_mesh.lines)
                if len(lines) > 0:
                    stride = int(lines[0]) + 1
                    n_seg = len(lines) // stride
                    airfoil_cells = lines.reshape(n_seg, stride)[:, 1:].astype(np.int64)
                else:
                    airfoil_cells = np.empty((0, 2), dtype=np.int64)
            else:
                airfoil_pts = np.empty((0, 2), dtype=np.float32)
                airfoil_cells = np.empty((0, 2), dtype=np.int64)
        else:
            airfoil_pts = np.empty((0, 2), dtype=np.float32)
            airfoil_cells = np.empty((0, 2), dtype=np.int64)

        return {
            "points": torch.stack(
                [
                    torch.from_numpy(arrays["x"]),
                    torch.from_numpy(arrays["y"]),
                ],
                dim=-1,
            ),
            "U": torch.stack(
                [
                    torch.from_numpy(arrays["vertex_Ux"]),
                    torch.from_numpy(arrays["vertex_Uy"]),
                ],
                dim=-1,
            ),
            "p": torch.from_numpy(arrays["vertex_p"]),
            "nut": torch.from_numpy(arrays["vertex_nut"]),
            "implicit_distance": torch.from_numpy(
                arrays["vertex_implicit_distance"]
            ),
            "angle_of_attack": torch.tensor(
                [scalars["angle_of_attack"]], dtype=torch.float32
            ),
            "inlet_velocity": torch.tensor(
                [scalars["inlet_velocity"]], dtype=torch.float32
            ),
            "C_D": torch.tensor([scalars["C_D"]], dtype=torch.float32),
            "C_L": torch.tensor([scalars["C_L"]], dtype=torch.float32),
            "internal_cells": torch.from_numpy(internal_cells).long(),
            "airfoil_points": torch.from_numpy(airfoil_pts),
            "airfoil_cells": torch.from_numpy(airfoil_cells).long(),
        }

    def _get_sample_metadata(self, index: int) -> dict[str, Any]:
        return {
            "dataset_index": self._indices[index],
            "task": self.task,
            "split": self.split,
        }

    def _get_field_names(self) -> list[str]:
        return [
            "points",
            "U",
            "p",
            "nut",
            "implicit_distance",
            "angle_of_attack",
            "inlet_velocity",
            "C_D",
            "C_L",
            "internal_cells",
            "airfoil_points",
            "airfoil_cells",
        ]

    @property
    def _supports_coordinated_subsampling(self) -> bool:
        return False

    def __repr__(self) -> str:
        return (
            f"AirFRANSArrowReader(path={self.dataset_path}, "
            f"task={self.task}, split={self.split}, "
            f"len={len(self)})"
        )
