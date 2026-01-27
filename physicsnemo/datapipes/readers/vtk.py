# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
VTKReader - Read data from VTK format files (.stl, .vtp, .vtu).

Supports reading mesh data from directories containing VTK files using PyVista.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from physicsnemo.core.version_check import check_version_spec
from physicsnemo.datapipes.readers.base import Reader
from physicsnemo.datapipes.registry import register

# Check if pyvista is available
PYVISTA_AVAILABLE = check_version_spec("pyvista", hard_fail=False)

if PYVISTA_AVAILABLE:
    pv = importlib.import_module("pyvista")


@register()
class VTKReader(Reader):
    r"""
    Read samples from VTK format files (.stl, .vtp, .vtu).

    This reader loads mesh data from directories where each subdirectory contains
    VTK files representing one sample. Supports STL (surface meshes), VTP
    (PolyData), and VTU (UnstructuredGrid) formats.

    Requires PyVista to be installed. If PyVista is not available, attempting
    to instantiate this reader will raise an ImportError with installation
    instructions.

    Examples
    --------
    >>> # Directory structure:
    >>> # data/
    >>> #   sample_0/
    >>> #     geometry.stl
    >>> #     surface.vtp
    >>> #   sample_1/
    >>> #     geometry.stl
    >>> #     surface.vtp
    >>> #   ...
    >>>
    >>> reader = VTKReader(  # doctest: +SKIP
    ...     "data/",
    ...     keys_to_read=["stl_coordinates", "stl_faces", "surface_normals"],
    ... )
    >>> data, metadata = reader[0]  # Returns (TensorDict, dict) tuple  # doctest: +SKIP
    >>> print(data["stl_coordinates"].shape)  # (N, 3)  # doctest: +SKIP

    Available Keys:
        From .stl files:
            - ``stl_coordinates``: Vertex coordinates, shape :math:`(N, 3)`
            - ``stl_faces``: Face indices (flattened), shape :math:`(M*3,)`
            - ``stl_centers``: Face centers, shape :math:`(M, 3)`
            - ``surface_normals``: Face normals, shape :math:`(M, 3)`

        From .vtp files:
            - ``surface_mesh_centers``: Cell centers
            - ``surface_normals``: Cell normals
            - ``surface_mesh_sizes``: Cell areas
            - Additional fields from the VTP file

    Note:
        VTK files are typically small enough to fit in memory, so coordinated
        subsampling is not supported. Use transforms for downsampling if needed.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        keys_to_read: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
        pin_memory: bool = False,
        include_index_in_metadata: bool = True,
    ) -> None:
        """
        Initialize the VTK reader.

        Parameters
        ----------
        path : str or Path
            Path to directory containing subdirectories with VTK files.
        keys_to_read : list[str], optional
            List of keys to extract from VTK files.
            If None, extracts all available data.
        exclude_patterns : list[str], optional
            List of filename patterns to exclude (e.g., ["single_solid"]).
        pin_memory : bool, default=False
            If True, place tensors in pinned memory for faster GPU transfer.
        include_index_in_metadata : bool, default=True
            If True, include sample index in metadata.

        Raises
        ------
        ImportError
            If PyVista is not installed.
        FileNotFoundError
            If path doesn't exist.
        ValueError
            If no valid VTK directories found.
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError(
                "PyVista is required for VTKReader but is not installed.\n"
                "Install it with: pip install pyvista\n"
                "See https://docs.pyvista.org/getting-started/installation.html "
                "for more information."
            )

        super().__init__(
            pin_memory=pin_memory,
            include_index_in_metadata=include_index_in_metadata,
        )

        self.path = Path(path)
        self.keys_to_read = keys_to_read
        self.exclude_patterns = exclude_patterns or ["single_solid"]

        if not self.path.exists():
            raise FileNotFoundError(f"Path not found: {self.path}")

        if not self.path.is_dir():
            raise ValueError(f"Path must be a directory: {self.path}")

        # Find all subdirectories containing VTK files
        self._directories = []
        for subdir in self.path.iterdir():
            if subdir.is_dir() and self._is_vtk_directory(subdir):
                self._directories.append(subdir)

        self._directories = sorted(self._directories)

        if not self._directories:
            raise ValueError(
                f"No directories containing VTK files found in {self.path}"
            )

        self._length = len(self._directories)

        # Supported file keys mapped to file extensions
        self._stl_keys = {
            "stl_coordinates",
            "stl_centers",
            "stl_faces",
            "stl_areas",
            "surface_normals",
        }
        self._vtp_keys = {
            "surface_mesh_centers",
            "surface_normals",
            "surface_mesh_sizes",
            "CpMeanTrim",
            "pMeanTrim",
            "wallShearStressMeanTrim",
        }
        self._vtu_keys = {
            "volume_mesh_centers",
            "volume_fields",
        }

    def _is_vtk_directory(self, directory: Path) -> bool:
        """Check if a directory contains VTK files."""
        vtk_extensions = {".stl", ".vtp", ".vtu", ".vtk"}
        for file in directory.iterdir():
            if file.suffix in vtk_extensions:
                return True
        return False

    def _get_file_by_extension(self, directory: Path, extension: str) -> Optional[Path]:
        """Get the first file with the given extension, excluding patterns."""
        for file in directory.iterdir():
            if file.suffix == extension:
                # Check if any exclude pattern is in the filename
                if not any(pattern in file.name for pattern in self.exclude_patterns):
                    return file
        return None

    def _read_stl_data(self, stl_path: Path) -> dict[str, torch.Tensor]:
        """Read data from an STL file."""
        mesh = pv.read(stl_path)

        data = {}

        # Extract faces (reshape from flat array to triangles)
        faces = mesh.faces.reshape(-1, 4)
        faces = faces[:, 1:]  # Remove the first column (always 3 for triangles)
        data["stl_faces"] = torch.from_numpy(faces.flatten())

        # Extract coordinates
        data["stl_coordinates"] = torch.from_numpy(mesh.points)

        # Extract normals
        data["surface_normals"] = torch.from_numpy(mesh.cell_normals)

        # Compute face centers (for stl_centers)
        # Each face has 3 vertices, compute the mean
        vertices = mesh.points
        face_indices = faces
        face_centers = vertices[face_indices].mean(axis=1)
        data["stl_centers"] = torch.from_numpy(face_centers)

        # Compute face areas (for stl_areas)
        # Area of triangle = 0.5 * ||cross(v1-v0, v2-v0)||
        v0 = vertices[face_indices[:, 0]]
        v1 = vertices[face_indices[:, 1]]
        v2 = vertices[face_indices[:, 2]]
        cross_prod = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
        data["stl_areas"] = torch.from_numpy(areas)

        return data

    def _read_vtp_data(self, vtp_path: Path) -> dict[str, torch.Tensor]:
        """Read data from a VTP file."""
        # VTP reading is not yet implemented in the original cae_dataset.py
        # Placeholder for future implementation
        raise NotImplementedError(
            "VTP file reading is not yet implemented. "
            "This will be added in a future update."
        )

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        """Load a single sample from a VTK directory."""
        directory = self._directories[index]

        result = {}

        # Determine which file types to read based on requested keys
        need_stl = self.keys_to_read is None or any(
            key in self._stl_keys for key in self.keys_to_read
        )
        need_vtp = self.keys_to_read is not None and any(
            key in self._vtp_keys for key in self.keys_to_read
        )
        need_vtu = self.keys_to_read is not None and any(
            key in self._vtu_keys for key in self.keys_to_read
        )

        # Read STL data if needed
        if need_stl:
            stl_path = self._get_file_by_extension(directory, ".stl")
            if stl_path:
                stl_data = self._read_stl_data(stl_path)
                result.update(stl_data)

        # Read VTP data if needed
        if need_vtp:
            vtp_path = self._get_file_by_extension(directory, ".vtp")
            if vtp_path:
                vtp_data = self._read_vtp_data(vtp_path)
                result.update(vtp_data)

        # Read VTU data if needed
        if need_vtu:
            raise NotImplementedError("VTU file reading is not yet implemented.")

        # Filter to requested keys if specified
        if self.keys_to_read is not None:
            result = {k: v for k, v in result.items() if k in self.keys_to_read}

        return result

    def __len__(self) -> int:
        """Return number of samples."""
        return self._length

    def _get_field_names(self) -> list[str]:
        """Return field names."""
        if self.keys_to_read is not None:
            return self.keys_to_read

        # Load first sample to discover available keys
        if len(self) == 0:
            return []

        sample = self._load_sample(0)
        return list(sample.keys())

    def _get_sample_metadata(self, index: int) -> dict[str, Any]:
        """Return metadata for a sample including source directory info."""
        return {
            "source_file": str(self._directories[index]),
            "source_filename": self._directories[index].name,
        }

    @property
    def _supports_coordinated_subsampling(self) -> bool:
        """VTK files don't support coordinated subsampling."""
        return False

    def __repr__(self) -> str:
        return f"VTKReader(path={self.path}, len={len(self)}, keys={self.keys_to_read})"
