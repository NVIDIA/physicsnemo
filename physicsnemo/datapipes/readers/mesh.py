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
Mesh readers - Load physicsnemo Mesh / DomainMesh from physicsnemo mesh format (.pmsh / .pdmsh).

MeshReader returns (Mesh, metadata) per sample.
DomainMeshReader returns (DomainMesh, metadata) per sample.
Both use tensorclass .load(path) directly; no conversion from other formats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import torch

from physicsnemo.datapipes.registry import register
from physicsnemo.mesh import DomainMesh, Mesh

logger = logging.getLogger(__name__)

# Default extensions for physicsnemo mesh formats (tensordict/tensorclass layout).
# Do not hardcode elsewhere so format can evolve.
DEFAULT_MESH_EXTENSION = ".pmsh"
DEFAULT_DOMAIN_MESH_EXTENSION = ".pdmsh"


def _contiguous_block_slice(
    total: int,
    k: int,
    generator: torch.Generator | None = None,
) -> slice:
    """Return a random contiguous ``slice`` of length *k* within ``[0, total)``.

    A contiguous slice produces sequential I/O on memmap-backed tensors,
    which is orders of magnitude faster than scattered fancy-indexing.
    For best results the on-disk point order should be pre-shuffled so
    that a contiguous block is spatially representative.
    """
    if total <= k:
        return slice(0, total)
    start = torch.randint(0, total - k + 1, (1,), generator=generator).item()
    return slice(start, start + k)


def _subsample_mesh_points(
    mesh: Mesh,
    n_points: int,
    generator: torch.Generator | None = None,
) -> Mesh:
    """Subsample a Mesh to *n_points* via a contiguous block read.

    Uses a contiguous slice for sequential I/O on memmap-backed data.
    For point clouds (``n_cells == 0``) this avoids the heavy
    cell-remapping logic in :meth:`Mesh.slice_points` which allocates
    two *N*-element intermediate tensors.  For meshes with cells it
    falls back to ``slice_points``.
    """
    if mesh.n_points <= n_points:
        return mesh
    sl = _contiguous_block_slice(mesh.n_points, n_points, generator=generator)
    if mesh.n_cells == 0:
        return Mesh(
            points=mesh.points[sl],
            cells=mesh.cells,
            point_data=mesh.point_data[sl],
            cell_data=mesh.cell_data,
            global_data=mesh.global_data,
        )
    return mesh.slice_points(torch.arange(sl.start, sl.stop, device=mesh.points.device))


def _subsample_mesh_cells(
    mesh: Mesh,
    n_cells: int,
    generator: torch.Generator | None = None,
) -> Mesh:
    """Subsample a Mesh to *n_cells* via a contiguous block read on cells.

    Preserves cell topology: each selected cell retains its full vertex
    connectivity.  Unreferenced points are compacted out.  Uses
    ``_contiguous_block_slice`` for sequential I/O on memmap-backed
    cell tensors.

    Use this instead of :func:`_subsample_mesh_points` when the mesh
    has cell connectivity (triangulated surfaces, volume meshes) and
    downstream transforms or outputs depend on cell topology (e.g.
    surface normals, cell centroids, cell_data fields).
    """
    if mesh.n_cells <= n_cells:
        return mesh
    sl = _contiguous_block_slice(mesh.n_cells, n_cells, generator=generator)
    mesh = mesh.slice_cells(sl)
    # Compact: drop vertices not referenced by any surviving cell
    referenced = torch.unique(mesh.cells)
    if referenced.numel() < mesh.n_points:
        mesh = mesh.slice_points(referenced)
    return mesh


def _subsample_mesh(
    mesh: Mesh,
    n_cells: int | None = None,
    n_points: int | None = None,
    generator: torch.Generator | None = None,
) -> Mesh:
    """Apply cell and/or point subsampling to a single Mesh.

    Cells are subsampled first (preserving topology) so that the
    subsequent point subsample operates on the already-reduced mesh.
    """
    if n_cells is not None:
        mesh = _subsample_mesh_cells(mesh, n_cells, generator=generator)
    if n_points is not None:
        mesh = _subsample_mesh_points(mesh, n_points, generator=generator)
    return mesh


@register()
class MeshReader:
    r"""
    Read single-mesh samples from directories of physicsnemo mesh files.

    Each sample is one Mesh. Returns (Mesh, metadata) per index.
    Uses Mesh.load(path) for physicsnemo mesh format (.pmsh).
    """

    def __init__(
        self,
        path: Path | str,
        *,
        pattern: str = f"**/*{DEFAULT_MESH_EXTENSION}",
        pin_memory: bool = False,
        include_index_in_metadata: bool = True,
        subsample_n_points: int | None = None,
        subsample_n_cells: int | None = None,
    ) -> None:
        """
        Initialize the mesh reader.

        Parameters
        ----------
        path : Path or str
            Root directory containing mesh files (e.g. .pmsh directories).
        pattern : str, optional
            Glob pattern for mesh paths under ``path``. Default matches ``**/*.pmsh``.
        pin_memory : bool, default=False
            If True, place tensors in pinned (page-locked) memory for faster
            async CPU→GPU transfers.
        include_index_in_metadata : bool, default=True
            If True, include sample index in metadata.
        subsample_n_points : int, optional
            If set, subsample the mesh to this many points *before*
            ``pin_memory``.  Uses contiguous block reads for sequential
            I/O on memmap-backed data.  Appropriate for point clouds
            or meshes where cell topology is not needed downstream.
            For best results, pre-shuffle the on-disk point order so
            that a contiguous block is spatially representative.
        subsample_n_cells : int, optional
            If set, subsample the mesh to this many cells *before*
            ``pin_memory``.  Uses contiguous block reads on the cell
            tensor for sequential I/O, then compacts unreferenced
            vertices.  Preserves cell topology and is the correct
            choice for triangulated surface meshes where downstream
            transforms depend on cells (e.g. surface normals, cell
            centroids, cell_data fields).  Applied before
            ``subsample_n_points`` when both are set.
        """
        self._root = Path(path)
        self._pattern = pattern
        self.pin_memory = pin_memory
        self.include_index_in_metadata = include_index_in_metadata
        self.subsample_n_points = subsample_n_points
        self.subsample_n_cells = subsample_n_cells
        self._subsample_generator: torch.Generator | None = None

        if not self._root.exists():
            raise FileNotFoundError(f"Path not found: {self._root}")
        if not self._root.is_dir():
            raise ValueError(f"Path must be a directory: {self._root}")

        self._paths = sorted(self._root.glob(pattern))
        if not self._paths:
            raise ValueError(f"No paths matching {pattern!r} found in {self._root}")

    def _load_sample(self, index: int) -> Mesh:
        """Load a single Mesh from disk."""
        mesh_path = self._paths[index]
        return Mesh.load(mesh_path)

    def _get_sample_metadata(self, index: int) -> dict[str, Any]:
        """Return metadata for the sample (e.g. source path)."""
        return {"source_path": str(self._paths[index])}

    def __len__(self) -> int:
        return len(self._paths)

    def set_generator(self, generator: torch.Generator) -> None:
        """Assign a ``torch.Generator`` for reproducible subsampling.

        Called by :class:`MeshDataset` when the DataLoader provides a
        seed.  Replaces any previously assigned generator.

        Parameters
        ----------
        generator : torch.Generator
            Generator to use for contiguous block selection.
        """
        self._subsample_generator = generator

    def set_epoch(self, epoch: int) -> None:
        """Reseed the subsample RNG for a new epoch.

        Produces a different (but deterministic) sequence of contiguous
        blocks each epoch when a generator has been assigned via
        :meth:`set_generator`.
        """
        if self._subsample_generator is not None:
            self._subsample_generator.manual_seed(
                self._subsample_generator.initial_seed() + epoch
            )

    def __getitem__(self, index: int) -> tuple[Mesh, dict[str, Any]]:
        mesh = self._load_sample(index)

        mesh = _subsample_mesh(
            mesh,
            self.subsample_n_cells,
            self.subsample_n_points,
            generator=self._subsample_generator,
        )

        if self.pin_memory:
            mesh = mesh.pin_memory()

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
class DomainMeshReader:
    r"""
    Read DomainMesh samples from a directory of physicsnemo mesh files.

    Each sample is one DomainMesh (interior + named boundaries + global_data).
    Returns (DomainMesh, metadata) per index.
    Uses DomainMesh.load(path) for physicsnemo mesh format (.pdmsh).
    """

    def __init__(
        self,
        path: Path | str,
        *,
        pattern: str = f"**/*{DEFAULT_DOMAIN_MESH_EXTENSION}",
        pin_memory: bool = False,
        include_index_in_metadata: bool = True,
        subsample_n_points: int | None = None,
        subsample_n_cells: int | None = None,
        extra_boundaries: dict[str, dict] | None = None,
    ) -> None:
        """
        Initialize the domain mesh reader.

        Parameters
        ----------
        path : Path or str
            Root directory containing DomainMesh files (e.g. .pdmsh archives).
        pattern : str, optional
            Glob pattern for DomainMesh paths under ``path``.
            Default matches ``**/*.pdmsh``.
        pin_memory : bool, default=False
            If True, place tensors in pinned (page-locked) memory for faster
            async CPU→GPU transfers.
        include_index_in_metadata : bool, default=True
            If True, include sample index in metadata.
        subsample_n_points : int, optional
            If set, subsample the interior and each boundary mesh to
            at most this many points *before* ``pin_memory``.  Uses
            contiguous block reads for sequential I/O on memmap-backed
            data.  Appropriate for point clouds or meshes where cell
            topology is not needed downstream.  For best results,
            pre-shuffle the on-disk point order so that a contiguous
            block is spatially representative.
        subsample_n_cells : int, optional
            If set, subsample the interior and each boundary mesh to
            at most this many cells *before* ``pin_memory``.  Uses
            contiguous block reads on cell tensors for sequential I/O,
            then compacts unreferenced vertices.  Preserves cell
            topology and is the correct choice when downstream
            transforms depend on cells.  Applied before
            ``subsample_n_points`` when both are set.
        extra_boundaries : dict[str, dict] or None, optional
            Load additional sibling meshes as extra boundaries on each
            sample.  Each key is the boundary name to assign; each value
            is a dict with a ``"pattern"`` key giving a glob pattern
            (relative to the sample's parent directory) to find the mesh
            file.  These meshes are loaded at full resolution and are
            **not** subsampled, making them suitable for geometric
            queries like SDF computation.

            Example::

                extra_boundaries:
                  stl_geometry:
                    pattern: "*_single_solid.stl.pmsh"
        """
        self._root = Path(path)
        self._pattern = pattern
        self.pin_memory = pin_memory
        self.include_index_in_metadata = include_index_in_metadata
        self.subsample_n_points = subsample_n_points
        self.subsample_n_cells = subsample_n_cells
        self._subsample_generator: torch.Generator | None = None
        self._extra_boundaries = extra_boundaries or {}

        if not self._root.exists():
            raise FileNotFoundError(f"Path not found: {self._root}")
        if not self._root.is_dir():
            raise ValueError(f"Path must be a directory: {self._root}")

        self._paths = sorted(self._root.glob(pattern))
        if not self._paths:
            raise ValueError(f"No paths matching {pattern!r} found in {self._root}")

    def _load_sample(self, index: int) -> DomainMesh:
        """Load a single DomainMesh from disk."""
        return DomainMesh.load(self._paths[index])

    def __len__(self) -> int:
        return len(self._paths)

    def set_generator(self, generator: torch.Generator) -> None:
        """Assign a ``torch.Generator`` for reproducible subsampling.

        Called by :class:`MeshDataset` when the DataLoader provides a
        seed.  Replaces any previously assigned generator.

        Parameters
        ----------
        generator : torch.Generator
            Generator to use for contiguous block selection.
        """
        self._subsample_generator = generator

    def set_epoch(self, epoch: int) -> None:
        """Reseed the subsample RNG for a new epoch.

        Produces a different (but deterministic) sequence of contiguous
        blocks each epoch when a generator has been assigned via
        :meth:`set_generator`.
        """
        if self._subsample_generator is not None:
            self._subsample_generator.manual_seed(
                self._subsample_generator.initial_seed() + epoch
            )

    def __getitem__(self, index: int) -> tuple[DomainMesh, dict[str, Any]]:
        dm = self._load_sample(index)

        if self.subsample_n_cells is not None or self.subsample_n_points is not None:
            sub_kw = dict(
                n_cells=self.subsample_n_cells,
                n_points=self.subsample_n_points,
                generator=self._subsample_generator,
            )
            dm = DomainMesh(
                interior=_subsample_mesh(dm.interior, **sub_kw),
                boundaries={
                    name: _subsample_mesh(dm.boundaries[name], **sub_kw)
                    for name in dm.boundary_names
                },
                global_data=dm.global_data,
            )

        # Load extra boundary meshes (full resolution, no subsampling).
        if self._extra_boundaries:
            dm = self._load_extra_boundaries(dm, index)

        if self.pin_memory:
            dm = dm.pin_memory()

        metadata: dict[str, Any] = {
            "source_path": str(self._paths[index]),
            "boundary_names": dm.boundary_names,
        }
        if self.include_index_in_metadata:
            metadata["index"] = index
        return dm, metadata

    def _load_extra_boundaries(self, dm: DomainMesh, index: int) -> DomainMesh:
        """Find and load sibling meshes as additional boundaries.

        Extra boundaries are loaded at full resolution (no subsampling)
        so they are suitable for geometric queries like SDF computation.
        """
        case_dir = Path(self._paths[index]).parent
        new_boundaries = dict(dm.boundaries)

        for bnd_name, bnd_cfg in self._extra_boundaries.items():
            glob_pattern = bnd_cfg["pattern"]
            matches = sorted(case_dir.glob(glob_pattern))
            if not matches:
                raise FileNotFoundError(
                    f"No mesh matching {glob_pattern!r} found in "
                    f"{case_dir} for extra boundary {bnd_name!r}"
                )
            if len(matches) > 1:
                logger.warning(
                    "Multiple meshes found for extra boundary %r in %s "
                    "matching %r; using %s",
                    bnd_name,
                    case_dir,
                    glob_pattern,
                    matches[0],
                )
            new_boundaries[bnd_name] = Mesh.load(matches[0])

        return DomainMesh(
            interior=dm.interior,
            boundaries=new_boundaries,
            global_data=dm.global_data,
        )

    def __iter__(self) -> Iterator[tuple[DomainMesh, dict[str, Any]]]:
        for i in range(len(self)):
            try:
                yield self[i]
            except Exception as e:
                logger.error("Sample %s failed: %s", i, e)
                raise RuntimeError(f"Sample {i} failed: {e}") from e

    def __repr__(self) -> str:
        return f"DomainMeshReader(path={self._root!r}, len={len(self)})"
