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

"""Reader of the full-vehicle Toyota Yaris crash dataset: Zarr stores, shared mesh.

Every case is one Zarr store with the same 1.48M-node reference mesh. Nodes
listed in an outlier mask (rigid-wall nodes and fly-away nodes) are dropped
from inputs, targets and neighborhoods; positions and thickness are read per
case and per frame, so no trajectory is preloaded. The reader also exposes the
structural connectivity (shell edges, optionally the solid-element edges) and
the shell facets of the contact surface, both renumbered to the kept nodes.
The datapipe built on it is :mod:`datapipes.px_yaris_dataset`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import torch
import zarr


def file_sha256(path):
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# The Yaris store layout; every array is checked where the recipe reads it.
STORE_LAYOUT = (
    "nodes/pos [T, N, 3], nodes/thickness [N], edges [E, 2], shell/connectivity "
    "[S, 4] for contact and solid/connectivity [H, 8] with include_solid_edges"
)


def store_array(store, key, path):
    """``store[key]`` with a layout error naming the missing array."""
    if key not in store:
        raise ValueError(
            f"{path} is not a Yaris store: it lacks {key}; the reader expects "
            + STORE_LAYOUT
        )
    return store[key]


def load_splits(path):
    """Case names per split from the dataset creator's CSV (names are identity)."""
    splits = {key: [] for key in ("train", "validation", "test")}
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if "split" not in row or "sample_name" not in row:
                raise ValueError(
                    f"{path}: the split manifest needs the columns split,sample_name"
                )
            split, name = row["split"].strip(), row["sample_name"].strip()
            name = name.removesuffix(".zarr")
            if split not in splits or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError(f"Invalid split/sample name: {split!r}/{name!r}")
            if name in seen:
                raise ValueError(f"Duplicate/cross-split sample name: {name}")
            seen.add(name)
            splits[split].append(name)
    if any(not values for values in splits.values()):
        raise ValueError("Explicit nonempty train, validation and test splits required")
    return splits


_HEX_EDGE_PAIRS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def solid_element_edges(store, total_nodes):
    """Distinct edges ``[E, 2]`` of the store's hexahedral ``solid/connectivity``.

    Collapsed hexes (wedges and tets written with repeated node ids) lose their
    degenerate edges. Returns an empty array when the store has no solid table.
    """
    try:
        connectivity = np.asarray(store["solid/connectivity"][:], dtype=np.int64)
    except KeyError:
        return np.empty((0, 2), dtype=np.int64)
    if connectivity.ndim != 2 or connectivity.shape[1] != 8:
        raise ValueError("solid/connectivity must be [num_solids, 8]")
    if connectivity.size and (
        connectivity.min() < 0 or connectivity.max() >= total_nodes
    ):
        raise ValueError("solid/connectivity references nodes outside the mesh")
    edges = np.concatenate(
        [
            np.stack((connectivity[:, a], connectivity[:, b]), axis=1)
            for a, b in _HEX_EDGE_PAIRS
        ]
    )
    edges = edges[edges[:, 0] != edges[:, 1]]
    return np.unique(np.sort(edges, axis=1), axis=0) if edges.size else edges


def connect_isolated_nodes(edges, positions, k):
    """Add Euclidean links from edge-less nodes to their ``k`` nearest nodes.

    ``edges`` is ``[E, 2]`` over ``len(positions)`` nodes; the result is again
    unique and sorted. Nodes that already have an edge are untouched.
    """
    from scipy.spatial import cKDTree

    n = int(len(positions))
    degree = (
        np.bincount(edges.reshape(-1), minlength=n) if edges.size else np.zeros(n, int)
    )
    isolated = np.flatnonzero(degree == 0)
    if isolated.size == 0:
        return edges
    k = min(int(k), n - 1)
    _, neighbors = cKDTree(np.asarray(positions, dtype=np.float64)).query(
        np.asarray(positions, dtype=np.float64)[isolated], k=k + 1
    )
    neighbors = np.asarray(neighbors).reshape(len(isolated), -1)
    source = np.repeat(isolated, neighbors.shape[1])
    target = neighbors.reshape(-1)
    links = np.stack((source, target), axis=1)
    links = links[links[:, 0] != links[:, 1]]
    merged = np.concatenate((edges.reshape(-1, 2), links))
    return np.unique(np.sort(merged, axis=1), axis=0)


class YarisFullReader:
    """Read selected frames of the shared-mesh Yaris stores, read-only."""

    def __init__(
        self,
        manifest_path,
        outlier_mask_path=None,
        stride=5,
        raw_dt=1e-3,
        max_frames=None,
        include_solid_edges=False,
        connect_isolated_k=0,
    ):
        # Structural connectivity for geodesic neighborhoods: the store's shell
        # edges, optionally the 12 edges of every solid (hex) element, and a
        # Euclidean fallback that links nodes left without any edge (beam,
        # mass or unsupported element nodes) to their nearest kept neighbors.
        self.include_solid_edges = bool(include_solid_edges)
        if isinstance(connect_isolated_k, bool) or int(connect_isolated_k) < 0:
            raise ValueError("connect_isolated_k must be a nonnegative integer")
        self.connect_isolated_k = int(connect_isolated_k)
        self.manifest_path = str(Path(manifest_path).resolve())
        self.splits = load_splits(self.manifest_path)
        self.manifest_signature = hashlib.sha256(
            json.dumps(self.splits, sort_keys=True).encode()
        ).hexdigest()
        self.stride, self.raw_dt = int(stride), float(raw_dt)
        if self.stride < 1 or not math.isfinite(self.raw_dt) or self.raw_dt <= 0:
            raise ValueError("Positive finite raw_dt and stride required")
        if max_frames is not None and (int(max_frames) != max_frames or max_frames < 6):
            raise ValueError("max_frames must be null or an integer >= 6")
        self.max_frames = None if max_frames is None else int(max_frames)
        self.outlier_mask_path = (
            None
            if outlier_mask_path is None
            else str(Path(outlier_mask_path).resolve())
        )
        self.exclude = None
        self.outlier_signature = "none"
        if self.outlier_mask_path is not None:
            with np.load(self.outlier_mask_path) as data:
                exclude = np.asarray(data["exclude"]).astype(bool).reshape(-1)
            if not exclude.any():
                raise ValueError("Outlier mask excludes no node; pass null instead")
            self.exclude = exclude
            self.outlier_signature = hashlib.sha256(exclude.tobytes()).hexdigest()
        self.total_nodes = None
        self.keep_ids = None
        self.reference = None  # physical [n_keep, 3] float32 after outlier removal
        self.checked = set()

    @property
    def dt(self):
        """Frame spacing of the strided sequence (``raw_dt`` times the stride)."""
        return self.raw_dt * self.stride

    def paths(self, directory, split, num_samples):
        """Return the first ``num_samples`` case paths of ``split``."""
        names = self.splits[split]
        if int(num_samples) > len(names):
            raise ValueError(
                f"{split} has {len(names)} CSV cases but {num_samples} were requested"
            )
        if int(num_samples) < len(names):
            logging.getLogger(__name__).warning(
                "%s uses the first %d of %d CSV cases", split, num_samples, len(names)
            )
        names = names[: int(num_samples)]
        paths = [Path(directory).resolve() / (name + ".zarr") for name in names]
        missing = [str(path) for path in paths if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing manifest stores: {missing}")
        return paths

    def _establish_reference(self, store, path):
        pos = store["nodes/pos"]
        total = int(pos.shape[1])
        if self.exclude is None:
            self.exclude = np.zeros(total, dtype=bool)
        if self.exclude.shape != (total,):
            raise ValueError(
                f"Outlier mask has {self.exclude.shape[0]} entries for {total} nodes"
            )
        self.total_nodes = total
        self.keep_ids = np.flatnonzero(~self.exclude).astype(np.int64)
        if len(self.keep_ids) == 0:
            raise ValueError("Outlier mask removes every node")
        reference = np.asarray(pos[0], dtype=np.float32)
        if not np.isfinite(reference).all():
            raise ValueError(f"Non-finite reference geometry in {path}")
        self.reference = np.ascontiguousarray(reference[self.keep_ids])
        self._reference_full = reference

    def open(self, path):
        """Open one Zarr case store and check its mesh against the reference."""
        store = zarr.open_group(str(path), mode="r")
        pos = store_array(store, "nodes/pos", path)
        if len(pos.shape) != 3 or pos.shape[2] != 3:
            raise ValueError(f"Unexpected position shape {pos.shape} in {path}")
        if self.reference is None:
            self._establish_reference(store, path)
        if str(path) not in self.checked:
            if pos.shape[1] != self.total_nodes or not np.array_equal(
                np.asarray(pos[0], dtype=np.float32), self._reference_full
            ):
                raise ValueError(f"Reference geometry/order differs in {path}")
            self.checked.add(str(path))
        return store

    def frames(self, store):
        """Frame indices of ``store``: every ``stride``-th, up to ``max_frames``."""
        frames = np.arange(0, store["nodes/pos"].shape[0], self.stride, dtype=np.int64)
        return frames if self.max_frames is None else frames[: self.max_frames]

    def coordinates(self, store, frames):
        """Physical positions ``[F, n_keep, 3]`` float32 for the selected frames."""
        pos = store["nodes/pos"]
        values = np.stack(
            [np.asarray(pos[int(t)], dtype=np.float32)[self.keep_ids] for t in frames]
        )
        if not np.isfinite(values).all():
            raise ValueError("Non-finite trajectory coordinates")
        return values

    def thickness(self, store):
        """Shell thickness per kept node, validated as finite and nonnegative."""
        values = np.asarray(
            store_array(store, "nodes/thickness", store.store.root)[:], dtype=np.float32
        ).reshape(-1)
        if values.shape != (self.total_nodes,):
            raise ValueError("Thickness/node count mismatch")
        values = values[self.keep_ids]
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Thickness must be finite and nonnegative")
        return values

    def structural_edges(self, store):
        """Reference connectivity restricted and renumbered to kept nodes, ``[2, E]``.

        Shell edges from the store, plus (``include_solid_edges``) the edges of
        the hexahedral ``solid/connectivity`` table (collapsed hexes contribute
        their distinct edges), plus (``connect_isolated_k > 0``) Euclidean links
        from every node left without an edge to its nearest kept neighbors.
        Duplicates are removed; direction is irrelevant downstream.
        """
        edges = np.asarray(
            store_array(store, "edges", store.store.root)[:], dtype=np.int64
        )
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges must be [E, 2]")
        if edges.size and (edges.min() < 0 or edges.max() >= self.total_nodes):
            raise ValueError("edges reference nodes outside the mesh")
        if self.include_solid_edges:
            edges = np.concatenate(
                (edges, solid_element_edges(store, self.total_nodes))
            )
        remap = np.full(self.total_nodes, -1, dtype=np.int64)
        remap[self.keep_ids] = np.arange(len(self.keep_ids), dtype=np.int64)
        edges = remap[edges]
        edges = edges[(edges >= 0).all(axis=1) & (edges[:, 0] != edges[:, 1])]
        edges = np.unique(np.sort(edges, axis=1), axis=0) if edges.size else edges
        if self.connect_isolated_k > 0:
            edges = connect_isolated_nodes(
                edges, self.reference, self.connect_isolated_k
            )
        return torch.from_numpy(np.ascontiguousarray(edges.T))

    def contact_faces(self, store):
        """Shell facets ``[F, 4]`` (long, CPU) on kept nodes; triangles as ``[a,b,c,c]``.

        Faces touching an excluded node are dropped, duplicates removed, and
        quads that the surface-contact geometry cannot represent at the
        reference (folded, overlapping or degenerate centre fan) are split
        into two triangles so the surface stays covered.
        """
        from physicsnemo.experimental.nn.shell_contact import supported_quad_mask

        try:
            cells = np.asarray(store["shell/connectivity"][:], dtype=np.int64)
        except KeyError as error:
            raise ValueError(
                "surface contact needs shell/connectivity in the store"
            ) from error
        if cells.ndim != 2 or cells.shape[1] != 4:
            raise ValueError("shell/connectivity must be [num_shells, 4]")
        if cells.size and (cells.min() < 0 or cells.max() >= self.total_nodes):
            raise ValueError("shell/connectivity references nodes outside the mesh")
        remap = np.full(self.total_nodes, -1, dtype=np.int64)
        remap[self.keep_ids] = np.arange(len(self.keep_ids), dtype=np.int64)
        faces = remap[cells]
        faces = faces[(faces >= 0).all(axis=1)]
        # Triangles are encoded with a repeated last id; any other repetition
        # is a degenerate shell and is dropped.
        tri = faces[:, 2] == faces[:, 3]
        sorted_first3 = np.sort(faces[:, :3], axis=1)
        ok3 = (sorted_first3[:, 1:] != sorted_first3[:, :-1]).all(axis=1)
        sorted4 = np.sort(faces, axis=1)
        ok4 = (sorted4[:, 1:] != sorted4[:, :-1]).all(axis=1)
        faces = faces[np.where(tri, ok3, ok4)]
        tri = faces[:, 2] == faces[:, 3]
        canonical = np.where(
            tri[:, None],
            np.sort(
                np.concatenate((faces[:, :3], np.full((len(faces), 1), -1)), axis=1),
                axis=1,
            ),
            np.sort(faces, axis=1),
        )
        _, first = np.unique(canonical, axis=0, return_index=True)
        faces = faces[np.sort(first)]
        tri = faces[:, 2] == faces[:, 3]
        quads = faces[~tri]
        if len(quads):
            reference = torch.from_numpy(self.reference)
            supported = supported_quad_mask(reference[torch.from_numpy(quads)]).numpy()
            bad = quads[~supported]
            split = (
                np.concatenate(
                    (
                        np.stack((bad[:, 0], bad[:, 1], bad[:, 2], bad[:, 2]), axis=1),
                        np.stack((bad[:, 0], bad[:, 2], bad[:, 3], bad[:, 3]), axis=1),
                    )
                )
                if len(bad)
                else np.empty((0, 4), dtype=np.int64)
            )
            faces = np.concatenate((faces[tri], quads[supported], split))
        return torch.from_numpy(np.ascontiguousarray(faces))
