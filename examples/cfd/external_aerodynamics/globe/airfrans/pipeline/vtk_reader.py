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
AirFRANS VTK Reader - Reads samples from VTU/VTP mesh files on disk.

Each sample is a directory containing three VTK files exported from OpenFOAM:
  - ``{name}_freestream.vtp``: freestream boundary mesh
  - ``{name}_aerofoil.vtp``: airfoil boundary mesh
  - ``{name}_internal.vtu``: internal volume mesh

The reader performs I/O only: loads VTK files, extracts raw simulation fields,
and emits mesh connectivity tensors. Derived quantities (gradients, normals)
are computed by downstream transforms for clean separation and profiling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyvista as pv
import torch

from physicsnemo.datapipes.readers.base import Reader

logger = logging.getLogger(__name__)


class AirFRANSVTKReader(Reader):
    """Reader for AirFRANS samples stored as VTU/VTP files on disk.

    Expects a ``data_dir`` containing sample subdirectories, each with three
    VTK files following the AirFRANS naming convention. A ``manifest.json``
    at the root of ``data_dir`` defines the task/split membership.

    Parameters
    ----------
    data_dir : str or Path
        Root directory containing ``manifest.json`` and sample subdirectories.
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
        data_dir: str | Path,
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

        self.data_dir = Path(data_dir)
        self.task = task
        self.split = split

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json not found in {self.data_dir}. "
                "This file defines train/test splits for each task."
            )

        manifest = json.loads(manifest_path.read_text())
        effective_task = "full" if (task == "scarce" and split == "test") else task
        split_key = f"{effective_task}_{split}"

        if split_key not in manifest:
            available = sorted(manifest.keys())
            raise ValueError(
                f"Split '{split_key}' not found in manifest.json. "
                f"Available: {available}"
            )

        self._sample_paths: list[Path] = [
            self.data_dir / name for name in manifest[split_key]
        ]

        missing = [p for p in self._sample_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} sample directories listed in manifest.json "
                f"do not exist. First few: {missing[:5]}"
            )

        logger.info(
            "AirFRANSVTKReader: task=%s, split=%s -> %d samples",
            task,
            split,
            len(self._sample_paths),
        )

    def __len__(self) -> int:
        return len(self._sample_paths)

    @staticmethod
    def _load_pyvista_meshes(sample_path: Path) -> dict[str, pv.DataSet]:
        """Load the three VTK mesh files for a single AirFRANS sample."""
        base = sample_path.name
        mesh_paths = {
            "freestream": sample_path / f"{base}_freestream.vtp",
            "airfoil": sample_path / f"{base}_aerofoil.vtp",
            "internal": sample_path / f"{base}_internal.vtu",
        }
        for f in mesh_paths.values():
            if not f.exists():
                raise FileNotFoundError(f"Missing required file: {f}")

        return {k: pv.read(v) for k, v in mesh_paths.items()}

    @staticmethod
    def _extract_cells_2d(mesh: pv.UnstructuredGrid) -> np.ndarray:
        """Extract triangulated cell connectivity from an UnstructuredGrid."""
        if not (set(mesh.cells_dict.keys()) == {pv.CellType.TRIANGLE}):
            mesh = mesh.triangulate()
        return mesh.cells_dict[pv.CellType.TRIANGLE]

    @staticmethod
    def _extract_line_cells(mesh: pv.PolyData) -> np.ndarray:
        """Extract edge connectivity from a PolyData boundary mesh."""
        lines = np.asarray(mesh.lines)
        if len(lines) == 0:
            return np.empty((0, 2), dtype=np.int64)
        stride = int(lines[0]) + 1
        n_segments = len(lines) // stride
        return lines.reshape(n_segments, stride)[:, 1:].astype(np.int64)

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        sample_path = self._sample_paths[index]
        meshes = self._load_pyvista_meshes(sample_path)

        internal = meshes["internal"]
        freestream = meshes["freestream"]
        airfoil = meshes["airfoil"]

        if "U" in freestream.point_data:
            U_fs = freestream.point_data["U"][:, :2].astype(np.float32)
        elif "U" in freestream.cell_data:
            U_fs = freestream.cell_data["U"][:, :2].astype(np.float32)
        else:
            raise KeyError(
                f"Freestream mesh at {sample_path} has neither point_data "
                "nor cell_data 'U'."
            )
        U_inf = U_fs.mean(axis=0)
        inlet_velocity = float(np.linalg.norm(U_inf))
        angle_of_attack = float(np.arctan2(U_inf[1], U_inf[0]))

        def get_2d(array: np.ndarray) -> np.ndarray:
            if array.ndim == 1:
                return array.astype(np.float32)
            return array[:, :2].astype(np.float32)

        points = get_2d(internal.points)
        U = get_2d(internal.point_data["U"])
        p = internal.point_data["p"].astype(np.float32)
        nut = internal.point_data["nut"].astype(np.float32)
        implicit_distance = internal.point_data["implicit_distance"].astype(
            np.float32
        )

        internal_cells = self._extract_cells_2d(internal)
        airfoil_pts = get_2d(airfoil.points)
        airfoil_cells = self._extract_line_cells(airfoil)

        return {
            "points": torch.from_numpy(points),
            "U": torch.from_numpy(U),
            "p": torch.from_numpy(p),
            "nut": torch.from_numpy(nut),
            "implicit_distance": torch.from_numpy(implicit_distance),
            "angle_of_attack": torch.tensor(
                [angle_of_attack], dtype=torch.float32
            ),
            "inlet_velocity": torch.tensor(
                [inlet_velocity], dtype=torch.float32
            ),
            "C_D": torch.tensor([float("nan")], dtype=torch.float32),
            "C_L": torch.tensor([float("nan")], dtype=torch.float32),
            "internal_cells": torch.from_numpy(internal_cells).long(),
            "airfoil_points": torch.from_numpy(airfoil_pts),
            "airfoil_cells": torch.from_numpy(airfoil_cells).long(),
        }

    def _get_sample_metadata(self, index: int) -> dict[str, Any]:
        return {
            "sample_name": self._sample_paths[index].name,
            "sample_path": str(self._sample_paths[index]),
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
            f"AirFRANSVTKReader(data_dir={self.data_dir}, "
            f"task={self.task}, split={self.split}, "
            f"len={len(self)})"
        )
