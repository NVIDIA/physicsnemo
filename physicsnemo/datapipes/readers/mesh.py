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
import os
from pathlib import Path
from typing import Any, Iterator

import torch

from physicsnemo.datapipes._indexing import _cyclic_block_indices
from physicsnemo.datapipes._rng import spawn_generator
from physicsnemo.datapipes.caching import DatasetCache, cached_or_load
from physicsnemo.datapipes.registry import register
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import compose_measure_weights

logger = logging.getLogger(__name__)

# Default extensions for physicsnemo mesh formats (tensordict/tensorclass layout).
# Do not hardcode elsewhere so format can evolve.
DEFAULT_MESH_EXTENSION = ".pmsh"
DEFAULT_DOMAIN_MESH_EXTENSION = ".pdmsh"


def _pread_leaf(
    t: torch.Tensor,
    runs: list[tuple[int, int]] | None = None,
    *,
    pin_memory: bool = False,
) -> torch.Tensor | None:
    """Materialize a memory-mapped tensor by reading it, not by faulting it in.

    Reads the whole tensor, or only the rows covered by *runs*
    (``(start, length)`` pairs), into a fresh buffer with
    :func:`os.preadv`.  The result is identical to indexing the memory map
    and copying, but the kernel fills the destination directly: no page
    faults, one call per run instead of one fault per page, and no mapped
    pages left charged to the process.

    Returns ``None`` when *t* is not a whole, contiguous, memory-mapped
    tensor, or the platform has no ``preadv``, leaving the caller to use
    the ordinary path.
    """
    filename = getattr(t, "filename", None)
    if filename is None or getattr(t, "index", None) is not None:
        return None
    if not hasattr(os, "preadv") or not t.is_contiguous() or t.numel() == 0:
        return None
    ### A view into a larger mapping would need the parent's layout to locate
    ### its bytes on disk; only a whole tensor maps 1:1 onto its backing file.
    if t.storage_offset() != 0 or t.untyped_storage().nbytes() != t.nbytes:
        return None

    if runs is None:
        ranges = [(0, t.nbytes)]
        shape = t.shape
    else:
        if t.ndim == 0:
            return None
        row_bytes = t.nbytes // t.shape[0]
        ranges = [(start * row_bytes, length * row_bytes) for start, length in runs]
        shape = torch.Size((sum(length for _, length in runs), *t.shape[1:]))

    buf = torch.empty(
        sum(nbytes for _, nbytes in ranges), dtype=torch.uint8, pin_memory=pin_memory
    )
    # os.preadv needs the buffer protocol, which Tensor does not implement.
    view = buf.numpy()
    fd = os.open(filename, os.O_RDONLY)
    try:
        pos = 0
        for offset, nbytes in ranges:
            if os.preadv(fd, [view[pos : pos + nbytes]], offset) != nbytes:
                raise OSError(
                    f"short read of {nbytes} bytes at offset {offset} in {filename}"
                )
            pos += nbytes
    finally:
        os.close(fd)
    return buf.view(t.dtype).reshape(shape)


def _contiguous_runs(indices: torch.Tensor) -> list[tuple[int, int]] | None:
    """Rewrite row *indices* as ``(start, length)`` runs of consecutive rows.

    Each run is one byte range, so selecting a block becomes one or two
    reads instead of a gather over the mapping.
    :func:`_cyclic_block_indices` produces exactly that shape: one
    ascending run, or two when the block wraps past the end of the array.

    Returns ``None`` when the indices would need more than two runs,
    which tells the caller to fall back to ordinary indexing.  The runs
    are read off the tensor rather than recomputed from the block's start
    and length, so the bytes read stay in step with the indices even if
    the sampler changes.
    """
    if indices.ndim != 1 or indices.numel() == 0:
        return None
    breaks = (indices.diff() != 1).nonzero().flatten()
    if breaks.numel() > 1:
        return None
    return [
        (int(run[0]), run.numel())
        for run in indices.tensor_split((breaks + 1).tolist())
    ]


def _read_rows(
    t: torch.Tensor,
    indices: torch.Tensor,
    runs: list[tuple[int, int]] | None,
    pin_memory: bool,
) -> torch.Tensor:
    """Rows *indices* of *t*, read with ``pread`` when *t* is memory-mapped."""
    if runs is not None:
        out = _pread_leaf(t, runs, pin_memory=pin_memory)
        if out is not None:
            return out
    return t[indices]


def _pin_memory(md: Mesh | DomainMesh) -> Mesh | DomainMesh:
    """``pin_memory()`` that reads memory-mapped leaves instead of faulting them.

    Every leaf is copied either way; this only changes how the bytes reach
    the pinned buffer.
    """

    def pin(t: torch.Tensor) -> torch.Tensor:
        out = _pread_leaf(t, pin_memory=True)
        return t.pin_memory() if out is None else out

    return md.apply(pin)


def _subsample_mesh_points(
    mesh: Mesh,
    n_points: int,
    generator: torch.Generator | None = None,
    *,
    pin_memory: bool = False,
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

    ``pin_memory`` reads the selected rows of memmap-backed data straight
    into pinned memory (see :func:`_pread_leaf`), sparing the caller's
    later ``pin_memory()`` a second copy.
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
        runs = _contiguous_runs(indices)
        return Mesh(
            points=_read_rows(mesh.points, indices, runs, pin_memory),
            cells=mesh.cells,
            point_data=mesh.point_data.apply(
                lambda t: _read_rows(t, indices, runs, pin_memory),
                batch_size=indices.shape,
            ),
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


def _subsample_mesh(
    mesh: Mesh,
    n_cells: int | None = None,
    n_points: int | None = None,
    generator: torch.Generator | None = None,
    *,
    pin_memory: bool = False,
) -> Mesh:
    """Apply cell and/or point subsampling to a single Mesh.

    Cells are subsampled first (preserving topology) so that the
    subsequent point subsample operates on the already-reduced mesh.
    """
    if n_cells is not None:
        mesh = _subsample_mesh_cells(mesh, n_cells, generator=generator)
    if n_points is not None:
        mesh = _subsample_mesh_points(
            mesh, n_points, generator=generator, pin_memory=pin_memory
        )
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
        cache: DatasetCache | None = None,
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
            async CPU→GPU transfers.  Memmap-backed data is read into that
            buffer rather than faulted in through the mapping, so the
            process keeps no mapped pages of the samples it has read.
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
        cache : DatasetCache, optional
            Optional :class:`~physicsnemo.datapipes.caching.DatasetCache`.
            Each sample load becomes a tree-backed cache entry: the loaded
            mesh (memmap-backed, so only small metadata is RAM-resident) is
            kept in the cache's RAM tier, and its many small files
            (``meta.json``, tiny ``global_data`` memmaps) are mirrored to
            the cache's disk tier -- removing the per-sample metadata storm
            on network filesystems. ``Mesh.load`` remains the only decoder;
            failures fall back to loading from the source. The same
            instance may be shared across readers.
        """
        self._root = Path(path)
        self._pattern = pattern
        self.pin_memory = pin_memory
        self.include_index_in_metadata = include_index_in_metadata
        self.subsample_n_points = subsample_n_points
        self.subsample_n_cells = subsample_n_cells
        self._cache = cache
        # Base seed + epoch for deterministic per-index RNG (see
        # :meth:`set_generator`). ``None`` means unseeded.
        self._seed_base: int | None = None
        self._epoch: int = 0

        if not self._root.exists():
            raise FileNotFoundError(f"Path not found: {self._root}")
        if not self._root.is_dir():
            raise ValueError(f"Path must be a directory: {self._root}")

        self._paths = sorted(self._root.glob(pattern))
        if not self._paths:
            raise ValueError(f"No paths matching {pattern!r} found in {self._root}")

    def _load_sample(self, index: int) -> Mesh:
        """Load a single Mesh from disk (through the cache when configured)."""
        mesh_path = self._paths[index]
        return self._cached("mesh/v1", mesh_path, Mesh.load, src=mesh_path)

    def _cached(
        self,
        kind: str,
        path: Path,
        loader: Any,
        *,
        src: Path | None = None,
    ) -> Any:
        """Route a small, immutable read through the optional cache.

        Same contract as :meth:`Reader._cached` (this class predates the
        ``Reader`` ABC and does not subclass it).
        """
        return cached_or_load(self._cache, kind, path, loader, src=src)

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
            pin_memory=self.pin_memory,
        )

        if self.pin_memory:
            mesh = _pin_memory(mesh)

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
        cache: DatasetCache | None = None,
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
            async CPU→GPU transfers.  Memmap-backed data is read into that
            buffer rather than faulted in through the mapping, so the
            process keeps no mapped pages of the samples it has read.
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
        cache : DatasetCache, optional
            Optional :class:`~physicsnemo.datapipes.caching.DatasetCache`.
            Each sample load becomes a tree-backed cache entry: the loaded
            DomainMesh (memmap-backed, so only small metadata is
            RAM-resident) is kept in the cache's RAM tier, and its many
            small files (per-node ``meta.json``, tiny ``global_data``
            memmaps) are mirrored to the cache's disk tier -- removing the
            per-sample metadata storm on network filesystems.
            ``extra_boundaries`` glob resolutions and their meshes are also
            cached. ``DomainMesh.load`` remains the only decoder; failures
            fall back to loading from the source. The same instance may be
            shared across readers.
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
        self._cache = cache

        if not self._root.exists():
            raise FileNotFoundError(f"Path not found: {self._root}")
        if not self._root.is_dir():
            raise ValueError(f"Path must be a directory: {self._root}")

        self._paths = sorted(self._root.glob(pattern))
        if not self._paths:
            raise ValueError(f"No paths matching {pattern!r} found in {self._root}")

    def _load_sample(self, index: int) -> DomainMesh:
        """Load a single DomainMesh from disk (through the cache when configured)."""
        dm_path = self._paths[index]
        return self._cached("domain-mesh/v1", dm_path, DomainMesh.load, src=dm_path)

    def _cached(
        self,
        kind: str,
        path: Path,
        loader: Any,
        *,
        src: Path | None = None,
    ) -> Any:
        """Route a small, immutable read through the optional cache.

        Same contract as :meth:`Reader._cached` (this class predates the
        ``Reader`` ABC and does not subclass it).
        """
        return cached_or_load(self._cache, kind, path, loader, src=src)

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
                pin_memory=self.pin_memory,
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
            dm = _pin_memory(dm)

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
            matches = self._glob_extra_boundary(case_dir, glob_pattern)
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
            bnd_path = Path(matches[0])
            new_boundaries[bnd_name] = self._cached(
                "mesh/v1", bnd_path, Mesh.load, src=bnd_path
            )

        return DomainMesh(
            interior=dm.interior,
            boundaries=new_boundaries,
            global_data=dm.global_data,
        )

    def _glob_extra_boundary(self, case_dir: Path, glob_pattern: str) -> list[str]:
        """Resolve an extra-boundary glob (a readdir per sample, cached)."""

        def _do_glob() -> list[str]:
            return [str(p) for p in sorted(case_dir.glob(glob_pattern))]

        if self._cache is None:
            return _do_glob()
        # Direct key: the identity must include the pattern, not just the dir.
        return self._cache.get_or_load(
            ("glob/v1", f"{case_dir.resolve()}::{glob_pattern}"), _do_glob
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
