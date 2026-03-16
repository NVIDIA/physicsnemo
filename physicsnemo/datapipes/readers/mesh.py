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
Mesh readers - Load physicsnemo Mesh from physicsnemo mesh format (e.g. .pt).

MeshReader returns (Mesh, metadata) per sample.
MultiMeshReader returns (TensorDict[str, Mesh], metadata) per sample.
Both use Mesh.load(path) directly; no conversion from other formats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

from tensordict import TensorDict

from physicsnemo.datapipes.registry import register
from physicsnemo.mesh import Mesh

logger = logging.getLogger(__name__)

# Default extension for physicsnemo mesh format (tensordict/tensorclass layout).
# Do not hardcode elsewhere so format can evolve.
DEFAULT_MESH_EXTENSION = ".pt"


@register()
class MeshReader:
    r"""
    Read single-mesh samples from directories of physicsnemo mesh files.

    Each sample is one Mesh. Returns (Mesh, metadata) per index.
    Uses Mesh.load(path) for physicsnemo mesh format (currently .pt).
    """

    def __init__(
        self,
        path: Path | str,
        *,
        pattern: str = f"**/*{DEFAULT_MESH_EXTENSION}",
        include_index_in_metadata: bool = True,
    ) -> None:
        """
        Initialize the mesh reader.

        Parameters
        ----------
        path : Path or str
            Root directory containing mesh files (e.g. .pt directories).
        pattern : str, optional
            Glob pattern for mesh paths under ``path``. Default matches ``**/*.pt``.
        include_index_in_metadata : bool, default=True
            If True, include sample index in metadata.
        """
        self._root = Path(path)
        self._pattern = pattern
        self.include_index_in_metadata = include_index_in_metadata

        if not self._root.exists():
            raise FileNotFoundError(f"Path not found: {self._root}")
        if not self._root.is_dir():
            raise ValueError(f"Path must be a directory: {self._root}")

        self._paths = sorted(self._root.glob(pattern))
        if not self._paths:
            raise ValueError(
                f"No paths matching {pattern!r} found in {self._root}"
            )
        self._length = len(self._paths)

    def _load_sample(self, index: int) -> Mesh:
        """Load a single Mesh from disk."""
        mesh_path = self._paths[index]
        return Mesh.load(mesh_path)

    def _get_sample_metadata(self, index: int) -> dict[str, Any]:
        """Return metadata for the sample (e.g. source path)."""
        return {"source_path": str(self._paths[index])}

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[Mesh, dict[str, Any]]:
        mesh = self._load_sample(index)
        metadata = self._get_sample_metadata(index)
        if self.include_index_in_metadata:
            metadata["index"] = index
        return mesh, metadata

    def __iter__(self) -> Iterator[tuple[Mesh, dict[str, Any]]]:
        for i in range(len(self)):
            try:
                yield self[i]
            except Exception as e:
                logger.error("Sample %s failed: %s", i, e)
                raise RuntimeError(f"Sample {i} failed: {e}") from e

    def __repr__(self) -> str:
        return f"MeshReader(path={self._root!r}, len={len(self)})"


@register()
class MultiMeshReader:
    r"""
    Read multi-mesh samples: one sample per subdirectory, each with multiple mesh files.

    Each sample is a TensorDict[str, Mesh]. Returns (TensorDict[str, Mesh], metadata).
    Uses Mesh.load(path) for physicsnemo mesh format (currently .pt).
    """

    def __init__(
        self,
        path: Path | str,
        *,
        mesh_pattern: str = f"*{DEFAULT_MESH_EXTENSION}",
        include_index_in_metadata: bool = True,
    ) -> None:
        """
        Initialize the multi-mesh reader.

        Parameters
        ----------
        path : Path or str
            Root directory; each direct subdirectory is one sample.
        mesh_pattern : str, optional
            Glob pattern for mesh files inside each sample subdirectory.
            Default matches ``*.pt``.
        include_index_in_metadata : bool, default=True
            If True, include sample index in metadata.
        """
        self._root = Path(path)
        self._mesh_pattern = mesh_pattern
        self.include_index_in_metadata = include_index_in_metadata

        if not self._root.exists():
            raise FileNotFoundError(f"Path not found: {self._root}")
        if not self._root.is_dir():
            raise ValueError(f"Path must be a directory: {self._root}")

        self._sample_dirs = sorted(
            d for d in self._root.iterdir() if d.is_dir()
        )
        if not self._sample_dirs:
            raise ValueError(
                f"No subdirectories found in {self._root}"
            )
        self._length = len(self._sample_dirs)

    def _load_sample(self, index: int) -> TensorDict:
        """Load all meshes in the sample subdirectory as TensorDict[str, Mesh]."""
        sample_dir = self._sample_dirs[index]
        mesh_paths = sorted(sample_dir.glob(self._mesh_pattern))
        if not mesh_paths:
            raise ValueError(
                f"No mesh files matching {self._mesh_pattern!r} in {sample_dir}"
            )
        out = {}
        for p in mesh_paths:
            # Use stem (filename without extension) as key
            key = p.stem
            out[key] = Mesh.load(p)
        return TensorDict(out, batch_size=[])

    def _get_sample_metadata(self, index: int) -> dict[str, Any]:
        """Return metadata for the sample (e.g. source dir and mesh names)."""
        sample_dir = self._sample_dirs[index]
        mesh_paths = sorted(sample_dir.glob(self._mesh_pattern))
        return {
            "source_dir": str(sample_dir),
            "mesh_names": [p.stem for p in mesh_paths],
        }

    def __len__(self) -> int:
        return self._length

    def __getitem__(
        self, index: int
    ) -> tuple[TensorDict, dict[str, Any]]:
        data = self._load_sample(index)
        metadata = self._get_sample_metadata(index)
        if self.include_index_in_metadata:
            metadata["index"] = index
        return data, metadata

    def __iter__(self) -> Iterator[tuple[TensorDict, dict[str, Any]]]:
        for i in range(len(self)):
            try:
                yield self[i]
            except Exception as e:
                logger.error("Sample %s failed: %s", i, e)
                raise RuntimeError(f"Sample {i} failed: {e}") from e

    def __repr__(self) -> str:
        return f"MultiMeshReader(path={self._root!r}, len={len(self)})"
