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

import glob as _glob
import logging
from pathlib import Path
from typing import Any, Iterator

import torch

from physicsnemo.datapipes._indexing import _cyclic_block_indices
from physicsnemo.datapipes._rng import spawn_generator
from physicsnemo.datapipes.registry import register
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import compose_measure_weights

logger = logging.getLogger(__name__)

# Default extensions for physicsnemo mesh formats (tensordict/tensorclass layout).
# Do not hardcode elsewhere so format can evolve.
DEFAULT_MESH_EXTENSION = ".pmsh"
DEFAULT_DOMAIN_MESH_EXTENSION = ".pdmsh"


def _subsample_mesh_points(
    mesh: Mesh,
    n_points: int,
    generator: torch.Generator | None = None,
) -> Mesh:
    """Subsample a Mesh to *n_points* via a cyclic contiguous block read.

    Uses one or two contiguous runs for page-sequential I/O on memmap-backed
    data while giving every point the same inclusion probability.
    For point clouds (``n_cells == 0``) this avoids the heavy
    cell-remapping logic in :meth:`Mesh.slice_points` which allocates
    two *N*-element intermediate tensors.  For meshes with cells it
    falls back to ``slice_points``.

    Unlike :func:`_subsample_mesh_cells`, this does NOT maintain
    measure weights: dropping points removes cells implicitly, with
    no per-cell inclusion probability to invert.  Prefer cell
    subsampling when downstream code integrates over the mesh.
    """
    if mesh.n_points <= n_points:
        return mesh
    indices = _cyclic_block_indices(
        mesh.n_points,
        n_points,
        generator=generator,
        device=mesh.points.device,
    )
    if mesh.n_cells == 0:
        return Mesh(
            points=mesh.points[indices],
            cells=mesh.cells,
            point_data=mesh.point_data[indices],
            cell_data=mesh.cell_data,
            global_data=mesh.global_data,
        )
    return mesh.slice_points(indices)


def _subsample_mesh_cells(
    mesh: Mesh,
    n_cells: int,
    generator: torch.Generator | None = None,
) -> Mesh:
    """Subsample a Mesh to *n_cells* via a cyclic contiguous block read on cells.

    Preserves cell topology: each selected cell retains its full vertex
    connectivity.  Unreferenced points are compacted out.  Uses
    :func:`_cyclic_block_indices` for (page-)sequential I/O on
    memmap-backed cell tensors.

    Preserves the mesh's integration measure: every cell's inclusion
    probability is exactly ``k/N``, and the retained cells' measure
    weights (see :mod:`physicsnemo.mesh.calculus.measure`) are multiplied by
    ``N/k``, composing with any weights from earlier sampling stages.
    Consumers of the effective cell measure (see
    :mod:`physicsnemo.mesh.calculus.measure`) then see an unbiased estimate
    of the full-mesh measure rather than the ~``k/N`` retained fraction.

    Use this instead of :func:`_subsample_mesh_points` when the mesh
    has cell connectivity (triangulated surfaces, volume meshes) and
    downstream transforms or outputs depend on cell topology (e.g.
    surface normals, cell centroids, cell_data fields).
    """
    n_total = mesh.n_cells
    if n_total <= n_cells:
        return mesh
    indices = _cyclic_block_indices(
        n_total,
        n_cells,
        generator=generator,
        device=mesh.cells.device,
    )
    mesh = mesh.slice_cells(indices)
    # Compact: drop vertices not referenced by any surviving cell
    referenced = torch.unique(mesh.cells)
    if referenced.numel() < mesh.n_points:
        mesh = mesh.slice_points(referenced)
    ### Compose the Horvitz-Thompson weight for this sampling stage.
    ### slice_cells/slice_points returned fresh TensorDicts, so the
    ### in-place update cannot leak into the memmap-backed source.
    compose_measure_weights(mesh, n_total / n_cells)
    return mesh


def _indices_to_runs(indices: torch.Tensor) -> list[tuple[int, int]]:
    """Convert cyclic-block indices (1-2 ascending contiguous runs) to runs."""
    breaks = torch.nonzero(indices[1:] != indices[:-1] + 1).flatten()
    starts = [0] + [int(b) + 1 for b in breaks]
    ends = [int(b) + 1 for b in breaks] + [len(indices)]
    return [(int(indices[s]), int(indices[e - 1]) + 1) for s, e in zip(starts, ends)]


def _zarr_mesh_subsampled(
    group,
    n_cells: int | None,
    n_points: int | None,
    generator: torch.Generator | None,
) -> Mesh:
    """Partial-read a zarr mesh group: fetch only the subsample window.

    Reproduces :func:`_subsample_mesh` semantics (cyclic contiguous blocks,
    vertex compaction, Horvitz-Thompson measure weights) while reading only
    the selected rows from the store instead of materializing the full mesh.
    """
    from physicsnemo.mesh.io import io_zarr as _ioz

    total_cells = group["cells"].shape[0] if "cells" in group else 0
    total_points = group["points"].shape[0]

    if total_cells > 0 and n_cells is not None and total_cells > n_cells:
        indices = _cyclic_block_indices(total_cells, n_cells, generator=generator)
        runs = _indices_to_runs(indices)
        cells = _ioz._read_rows(group["cells"], runs)
        # Compact: gather only referenced vertices; remap connectivity to the
        # sorted-unique order, matching slice_cells + slice_points.
        referenced, inverse = torch.unique(cells, return_inverse=True)
        cells = inverse.reshape(cells.shape)
        ref_np = referenced.numpy()
        mesh = Mesh(
            points=_ioz._read_index(group["points"], ref_np),
            cells=cells,
            point_data=_ioz._read_tree(
                group, "point_data", leaf_reader=lambda a: _ioz._read_index(a, ref_np)
            ),
            cell_data=_ioz._read_tree(
                group, "cell_data", leaf_reader=lambda a: _ioz._read_rows(a, runs)
            ),
            global_data=_ioz._read_tree(group, "global_data"),
        )
        compose_measure_weights(mesh, total_cells / n_cells)
        if n_points is not None:
            mesh = _subsample_mesh_points(mesh, n_points, generator=generator)
        return mesh

    if total_cells == 0 and n_points is not None and total_points > n_points:
        indices = _cyclic_block_indices(total_points, n_points, generator=generator)
        runs = _indices_to_runs(indices)
        return Mesh(
            points=_ioz._read_rows(group["points"], runs),
            point_data=_ioz._read_tree(
                group, "point_data", leaf_reader=lambda a: _ioz._read_rows(a, runs)
            ),
            cell_data=_ioz._read_tree(group, "cell_data"),
            global_data=_ioz._read_tree(group, "global_data"),
        )

    # No subsampling applies (small mesh, or unsupported combination):
    # eager full read keeps semantics identical to the memmap path.
    return _ioz._mesh_from_group(group, None)


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
            ``pin_memory``.  Uses cyclic contiguous block reads for
            page-sequential I/O on memmap-backed data, with uniform point
            inclusion probability.  Appropriate for point clouds
            or meshes where cell topology is not needed downstream.
            For best results, pre-shuffle the on-disk point order so
            that a contiguous block is spatially representative.
        subsample_n_cells : int, optional
            If set, subsample the mesh to this many cells *before*
            ``pin_memory``.  Uses cyclic contiguous block reads on the
            cell tensor for sequential I/O, then compacts unreferenced
            vertices.  Preserves cell topology and is the correct
            choice for triangulated surface meshes where downstream
            transforms depend on cells (e.g. surface normals, cell
            centroids, cell_data fields).  Records the inverse inclusion
            probability as measure weights, preserving the integration
            measure (see :mod:`physicsnemo.mesh.calculus.measure`).  Applied before
            ``subsample_n_points`` when both are set.
        """
        self._root = Path(path)
        self._pattern = pattern
        self.pin_memory = pin_memory
        self.include_index_in_metadata = include_index_in_metadata
        self.subsample_n_points = subsample_n_points
        self.subsample_n_cells = subsample_n_cells
        # Base seed + epoch for deterministic per-index RNG (see
        # :meth:`set_generator`). ``None`` means unseeded.
        self._seed_base: int | None = None
        self._epoch: int = 0

        if not self._root.exists():
            raise FileNotFoundError(f"Path not found: {self._root}")
        if not self._root.is_dir():
            raise ValueError(f"Path must be a directory: {self._root}")

        # glob.glob instead of Path.glob: the latter re-stats each entry and
        # silently drops entries under Lustre metadata-server load.
        self._paths = sorted(
            Path(p) for p in _glob.glob(str(self._root / pattern), recursive=True)
        )
        if not self._paths:
            raise ValueError(f"No paths matching {pattern!r} found in {self._root}")

    def _load_sample(self, index: int) -> Mesh:
        """Load a single Mesh from disk."""
        mesh_path = self._paths[index]
        if (mesh_path / "zarr.json").exists():
            from physicsnemo.mesh.io import from_zarr, io_zarr

            if (
                self.subsample_n_cells is not None
                or self.subsample_n_points is not None
            ):
                # Push the subsample into the read: fetch only the selected
                # window from the store. The generator derivation matches
                # __getitem__, so the draw is identical to subsampling after
                # an eager load (whose subsample then no-ops).
                # Cache opened store handles: re-opening walks the store's
                # group-metadata chain, and on networked filesystems every
                # uncached lookup is a metadata-server round-trip per draw.
                cache = getattr(self, "_zarr_groups", None)
                if cache is None:
                    cache = self._zarr_groups = {}
                group = cache.get(mesh_path)
                if group is None:
                    group = cache[mesh_path] = io_zarr._open_group(mesh_path)
                generator = (
                    None
                    if self._seed_base is None
                    else spawn_generator(self._seed_base, self._epoch, index)
                )
                return _zarr_mesh_subsampled(
                    group,
                    self.subsample_n_cells,
                    self.subsample_n_points,
                    generator,
                )
            return from_zarr(mesh_path)
        return Mesh.load(mesh_path)

    def _get_sample_metadata(self, index: int) -> dict[str, Any]:
        """Return metadata for the sample (e.g. source path)."""
        return {"source_path": str(self._paths[index])}

    def __len__(self) -> int:
        return len(self._paths)

    def set_generator(self, generator: torch.Generator) -> None:
        """Assign a base seed for reproducible, order-independent subsampling.

        Called by :class:`MeshDataset` when the DataLoader provides a
        seed.  Stores ``generator.initial_seed()`` as the base seed; each
        sample then derives its own generator from
        ``(base_seed, epoch, index)``, so subsampling is reproducible
        regardless of read order or worker thread.

        Parameters
        ----------
        generator : torch.Generator
            Generator whose ``initial_seed()`` seeds all per-sample RNG.
        """
        self._seed_base = generator.initial_seed()

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to vary per-sample RNG deterministically.

        The epoch is folded into each sample's derived seed, producing a
        different (but deterministic) sequence of contiguous blocks each
        epoch when a base seed has been assigned via :meth:`set_generator`.
        """
        self._epoch = epoch

    def __getitem__(self, index: int) -> tuple[Mesh, dict[str, Any]]:
        mesh = self._load_sample(index)

        generator = (
            None
            if self._seed_base is None
            else spawn_generator(self._seed_base, self._epoch, index)
        )
        mesh = _subsample_mesh(
            mesh,
            self.subsample_n_cells,
            self.subsample_n_points,
            generator=generator,
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
        drop_interior_cells: bool = False,
        drop_in_file_boundaries: bool = False,
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
            cyclic contiguous block reads for page-sequential I/O on
            memmap-backed data, with uniform point inclusion probability.
            Appropriate for point clouds or meshes where cell topology is
            not needed downstream.  For best results,
            pre-shuffle the on-disk point order so that a contiguous
            block is spatially representative.
        subsample_n_cells : int, optional
            If set, subsample the interior and each boundary mesh to
            at most this many cells *before* ``pin_memory``.  Uses
            cyclic contiguous block reads on cell tensors for
            sequential I/O, then compacts unreferenced vertices.
            Preserves cell topology and is the correct choice when
            downstream transforms depend on cells.  Records the
            inverse inclusion probability as measure weights, preserving
            the integration measure (see
            :mod:`physicsnemo.mesh.calculus.measure`).  Applied
            before
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
        drop_interior_cells : bool, default=False
            If True, discard the interior mesh's cell connectivity (and
            cell_data) immediately after load, turning it into a point
            cloud.  This makes ``subsample_n_points`` take the cheap
            contiguous-block path instead of the expensive
            ``slice_points`` remap (which allocates an ``n_points`` map
            and scatter-reads the full cell array from the memmap).  Use
            for point-based models that consume only ``interior.points``
            and ``interior.point_data`` (e.g. GeoTransolver volume) and
            never the interior tet/cell topology.  Boundaries are
            unaffected, so surface normals etc. still work.
        drop_in_file_boundaries : bool, default=False
            If True, discard the boundaries stored *in* the DomainMesh
            file immediately after load (before subsampling and pinning).
            ``extra_boundaries`` are added afterwards and are therefore
            unaffected.  Use when the model consumes only the interior
            (plus any ``extra_boundaries``) and never the in-file
            boundaries -- e.g. a volume pipeline whose SDF comes from an
            injected STL, where the in-file car-surface boundary would
            otherwise be subsampled (an expensive ``slice_points`` remap,
            GIL-held, that blocks worker-thread overlap) and pinned every
            sample for nothing.
        """
        self._root = Path(path)
        self._pattern = pattern
        self.pin_memory = pin_memory
        self.include_index_in_metadata = include_index_in_metadata
        self.drop_interior_cells = drop_interior_cells
        self.drop_in_file_boundaries = drop_in_file_boundaries
        self.subsample_n_points = subsample_n_points
        self.subsample_n_cells = subsample_n_cells
        # Base seed + epoch for deterministic per-index RNG (see
        # :meth:`set_generator`). ``None`` means unseeded.
        self._seed_base: int | None = None
        self._epoch: int = 0
        self._extra_boundaries = extra_boundaries or {}

        if not self._root.exists():
            raise FileNotFoundError(f"Path not found: {self._root}")
        if not self._root.is_dir():
            raise ValueError(f"Path must be a directory: {self._root}")

        # glob.glob instead of Path.glob: the latter re-stats each entry and
        # silently drops entries under Lustre metadata-server load.
        self._paths = sorted(
            Path(p) for p in _glob.glob(str(self._root / pattern), recursive=True)
        )
        if not self._paths:
            raise ValueError(f"No paths matching {pattern!r} found in {self._root}")

    def _load_sample(self, index: int) -> DomainMesh:
        """Load a single DomainMesh from disk."""
        path = self._paths[index]
        if (path / "zarr.json").exists():
            from physicsnemo.mesh.io import from_zarr, io_zarr

            if (
                self.subsample_n_cells is not None
                or self.subsample_n_points is not None
            ):
                # Push the subsample into the read (window reads per
                # sub-mesh); drop flags are honored at read time so skipped
                # data is never fetched. Generator derivation and sub-mesh
                # order match __getitem__, whose subsample then no-ops.
                generator = (
                    None
                    if self._seed_base is None
                    else spawn_generator(self._seed_base, self._epoch, index)
                )
                cache = getattr(self, "_zarr_groups", None)
                if cache is None:
                    cache = self._zarr_groups = {}
                root = cache.get(path)
                if root is None:
                    root = cache[path] = io_zarr._open_group(path)
                interior = _zarr_mesh_subsampled(
                    root["interior"],
                    self.subsample_n_cells,
                    self.subsample_n_points,
                    generator,
                )
                boundaries = {}
                if not self.drop_in_file_boundaries and "boundaries" in root:
                    boundaries = {
                        name: _zarr_mesh_subsampled(
                            grp,
                            self.subsample_n_cells,
                            self.subsample_n_points,
                            generator,
                        )
                        for name, grp in root["boundaries"].groups()
                    }
                return DomainMesh(
                    interior=interior,
                    boundaries=boundaries,
                    global_data=io_zarr._read_tree(root, "global_data"),
                )
            return from_zarr(path)
        return DomainMesh.load(path)

    def __len__(self) -> int:
        return len(self._paths)

    def set_generator(self, generator: torch.Generator) -> None:
        """Assign a base seed for reproducible, order-independent subsampling.

        Called by :class:`MeshDataset` when the DataLoader provides a
        seed.  Stores ``generator.initial_seed()`` as the base seed; each
        sample then derives its own generator from
        ``(base_seed, epoch, index)``, so subsampling is reproducible
        regardless of read order or worker thread.

        Parameters
        ----------
        generator : torch.Generator
            Generator whose ``initial_seed()`` seeds all per-sample RNG.
        """
        self._seed_base = generator.initial_seed()

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to vary per-sample RNG deterministically.

        The epoch is folded into each sample's derived seed, producing a
        different (but deterministic) sequence of contiguous blocks each
        epoch when a base seed has been assigned via :meth:`set_generator`.
        """
        self._epoch = epoch

    def __getitem__(self, index: int) -> tuple[DomainMesh, dict[str, Any]]:
        dm = self._load_sample(index)

        # Trim unused data before subsample/pin. Both references are lazy (no
        # memmap materialization here):
        #  - drop_interior_cells: turn the interior into a point cloud so its
        #    point subsample takes the cheap contiguous-block path instead of a
        #    full slice_points remap + scattered reads.
        #  - drop_in_file_boundaries: skip the in-file boundaries entirely so we
        #    don't subsample (an expensive, GIL-held slice_points remap that
        #    starves worker-thread overlap) or pin a surface the model ignores.
        if (self.drop_interior_cells and dm.interior.n_cells > 0) or (
            self.drop_in_file_boundaries and len(dm.boundary_names) > 0
        ):
            interior = dm.interior
            if self.drop_interior_cells and interior.n_cells > 0:
                interior = Mesh(
                    points=interior.points,
                    point_data=interior.point_data,
                    global_data=interior.global_data,
                )
            boundaries = {} if self.drop_in_file_boundaries else dm.boundaries
            dm = DomainMesh(
                interior=interior,
                boundaries=boundaries,
                global_data=dm.global_data,
            )

        if self.subsample_n_cells is not None or self.subsample_n_points is not None:
            generator = (
                None
                if self._seed_base is None
                else spawn_generator(self._seed_base, self._epoch, index)
            )
            sub_kw = dict(
                n_cells=self.subsample_n_cells,
                n_points=self.subsample_n_points,
                generator=generator,
            )
            interior = _subsample_mesh(dm.interior, **sub_kw)
            boundaries = {
                name: _subsample_mesh(dm.boundaries[name], **sub_kw)
                for name in dm.boundary_names
            }
            dm = DomainMesh(
                interior=interior,
                boundaries=boundaries,
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
            if (matches[0] / "zarr.json").exists():
                from physicsnemo.mesh.io import from_zarr

                new_boundaries[bnd_name] = from_zarr(matches[0])
            else:
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
