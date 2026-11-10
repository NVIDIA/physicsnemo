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
"""Lazy full-resolution Yaris data for the DeFormer recipes.

Every case is one Zarr store with the same 1.48M-node reference mesh. This
datapipe keeps the mesh at full resolution: no coarsening, no prolongation.
Nodes listed in an outlier mask (rigid-wall nodes and fly-away nodes) are
dropped from inputs, targets and neighborhoods. The static kNN neighborhoods
and their standardized features are built once from the shared reference
geometry; positions and thickness are read per case and per frame, so no
trajectory is preloaded.

Normalization follows the crash datapipe: per-axis position mean/std over the
training cases (equal case weight), velocities and accelerations in
normalized coordinates, and a per-channel thickness standardization.

Optional graph attachments for the corrector and the evaluation metrics: the
shell facets, nodal thickness and one-ring exclusions of the contact search
(``contact_surface``), the structural mesh edges with standardized reference
features (``mesh_edges``) and the rigid wall plane (``rigid_wall_position``).
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import zarr
from datapipe import EPS, KNN_EDGE_STATS_FILE, SimSample
from torch.utils.data import Dataset

STATS_FILE = "yaris_full_stats.json"
NORMALIZATION = "yaris-full-train-equal-case-axis-std-v1"


def file_sha256(path):
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    """Write ``value`` as JSON to ``path`` via a temporary file and atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


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


def _case_moments(args):
    """Per-case moments (equal case weight) for position, velocity, acceleration."""
    path, keep_ids, stride, max_frames, dt = args
    store = zarr.open_group(str(path), mode="r")
    pos = store["nodes/pos"]
    frames = np.arange(0, pos.shape[0], stride, dtype=np.int64)
    if max_frames is not None:
        frames = frames[:max_frames]
    if len(frames) < 6:
        raise ValueError(f"{path}: training needs two observed plus four future frames")
    local = {name: np.zeros((2, 3)) for name in ("pos", "vel", "acc")}
    before = previous = None
    for frame in frames:
        now = np.asarray(pos[int(frame)], dtype=np.float64)[keep_ids]
        if not np.isfinite(now).all():
            raise ValueError(f"{path}: non-finite coordinates at frame {frame}")
        local["pos"] += np.stack((now.mean(0), np.square(now).mean(0))) / len(frames)
        if previous is not None:
            vel = (now - previous) / dt
            local["vel"] += np.stack((vel.mean(0), np.square(vel).mean(0))) / (
                len(frames) - 1
            )
        if before is not None:
            acc = (now - 2 * previous + before) / dt**2
            local["acc"] += np.stack((acc.mean(0), np.square(acc).mean(0))) / (
                len(frames) - 2
            )
        before, previous = previous, now
    thickness = np.asarray(store["nodes/thickness"][:], dtype=np.float64)[keep_ids]
    feature = np.array([thickness.mean(), np.square(thickness).mean()])
    return local, feature, len(frames)


def compute_frozen_stats(reader, train_paths, workers=None):
    """Stream the TRAIN cases once (parallel over cases) on kept nodes only."""
    if reader.keep_ids is None:
        reader.open(train_paths[0])
    jobs = [
        (str(path), reader.keep_ids, reader.stride, reader.max_frames, reader.dt)
        for path in train_paths
    ]
    workers = max(1, min(int(workers or 1), os.cpu_count() or 1, len(jobs)))
    results = None
    if workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_case_moments, jobs))
        except (OSError, PermissionError) as error:  # sandboxed hosts: no pool
            logging.getLogger(__name__).warning(
                "Process pool unavailable (%s); computing statistics serially", error
            )
    if results is None:
        results = [_case_moments(job) for job in jobs]
    moment = {name: np.zeros((2, 3)) for name in ("pos", "vel", "acc")}
    feature = np.zeros(2)
    frame_counts = []
    for local, local_feature, count in results:
        for key in moment:
            moment[key] += local[key] / len(results)
        feature += local_feature / len(results)
        frame_counts.append(int(count))
    pos_mean = moment["pos"][0]
    pos_std = np.sqrt(np.maximum(moment["pos"][1] - pos_mean**2, 0) + EPS)
    node = dict(pos_mean=pos_mean, pos_std=pos_std)
    for key in ("vel", "acc"):
        mean = moment[key][0] / pos_std
        var = np.maximum(moment[key][1] - moment[key][0] ** 2, 0) / pos_std**2
        node[f"norm_{key}_mean"], node[f"norm_{key}_std"] = mean, np.sqrt(var + EPS)
    return dict(
        node={k: v.tolist() for k, v in node.items()},
        feature=dict(
            feature_mean=[feature[0]],
            feature_std=[float(np.sqrt(max(feature[1] - feature[0] ** 2, 0) + EPS))],
        ),
        frames_per_case=frame_counts,
    )


class YarisFullGraphDataset(Dataset):
    """Full-resolution Yaris cases with static kNN neighborhoods.

    Parameters
    ----------
    reader : YarisFullReader
        Store reader (split manifest, outlier mask, stride).
    data_dir : str
        Directory of the ``<case>.zarr`` stores.
    num_samples : int
        Number of cases of the split to use (the first ones of the manifest).
    num_steps : int
        Frame bound per case (``training.num_time_steps``); cases may be shorter.
    split : {"train", "validation", "test"}
        Split of the manifest.
    name : str
        Dataset name for logging.
    stats_dir : str
        Directory of the frozen normalization statistics and caches.
    stats_mode : {"compute", "load"}
        Compute the statistics from the training split (rank zero) or load them.
    logger : logging.Logger, optional
        Logger for startup messages.
    sample_type : {"one_time_step", "all_time_steps"}
        One (case, future frame) pair per access (time-conditional training)
        or the complete trajectory (validation and inference).
    dt : float
        Frame interval in seconds (must equal ``stride * raw_dt`` of the reader).
    static_features, dynamic_features, dynamic_targets, global_features,
    global_features_filepath
        Feature selection of the generic crash datapipe; the Yaris recipe is
        displacement-only with a thickness feature.
    local_knn_k, local_knn_metric, local_knn_cache_dir
        Static reference neighbourhoods of the DeFormer's local attention.
    stats_workers : int, optional
        Processes used to compute the frozen statistics.
    contact_surface : bool
        Attach the shell facets, each design's nodal thickness and the one-ring
        exclusions (``graph.contact_faces``, ``shell_thickness``,
        ``contact_surface_exclusions``) for the contact search of the corrector
        and the penetration metrics.
    rigid_wall_position : float, optional
        Analytic rigid wall plane ``x = position`` in millimetres
        (``graph.rigid_wall``; Yaris: 100), for the wall clamp and the wall
        penetration metrics; also attaches each design's nodal thickness.
    mesh_edges : bool
        Attach the structural shell connectivity in both directions with its
        standardized reference features (``graph.mesh_edge_index``,
        ``mesh_edge_attr``, ``mesh_edge_mean``, ``mesh_edge_std``) for the
        MeshGraphNet corrector; ``edge_index`` itself stays as the local
        operator expects it (empty under the Euclidean metric).
    """

    def __init__(
        self,
        reader,
        data_dir,
        num_samples,
        num_steps=31,
        split="train",
        name="yaris",
        stats_dir="stats",
        stats_mode="load",
        logger=None,
        sample_type="one_time_step",
        dt=0.005,
        static_features=("thickness",),
        dynamic_features=(),
        dynamic_targets=(),
        global_features=None,
        global_features_filepath=None,
        local_knn_k=32,
        local_knn_metric="euclidean",
        local_knn_cache_dir=None,
        stats_workers=None,
        contact_surface=False,
        rigid_wall_position=None,
        mesh_edges=False,
    ):
        self.reader, self.name, self.split = reader, name, split
        self.mesh_edges = bool(mesh_edges)
        self.contact_surface = bool(contact_surface)
        self.rigid_wall_position = (
            None if rigid_wall_position is None else float(rigid_wall_position)
        )
        self.logger = logger or logging.getLogger(__name__)
        if (
            list(static_features) != ["thickness"]
            or list(dynamic_features)
            or list(dynamic_targets)
            or global_features
            or global_features_filepath
        ):
            raise ValueError(
                "The full-mesh Yaris recipe is displacement-only with a thickness feature"
            )
        if not math.isclose(float(dt), reader.dt, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Datapipe dt must equal stride * raw frame dt")
        if sample_type not in ("all_time_steps", "one_time_step"):
            raise ValueError("sample_type must be one_time_step or all_time_steps")
        if split != "train" and sample_type != "all_time_steps":
            raise ValueError("Validation/test must cover complete selected histories")
        if stats_mode not in ("compute", "load"):
            raise ValueError("stats_mode must be compute or load")
        if local_knn_metric not in ("geodesic", "euclidean"):
            raise ValueError("local_knn_metric must be geodesic or euclidean")
        if isinstance(local_knn_k, bool) or int(local_knn_k) <= 0:
            raise ValueError("local_knn_k must be a positive integer")
        self.dt, self.sample_type = float(dt), sample_type
        self.num_steps = int(num_steps)
        self.local_knn_k, self.local_knn_metric = int(local_knn_k), local_knn_metric
        self.paths = reader.paths(data_dir, split, num_samples)
        self.num_samples = len(self.paths)
        self.reader.open(self.paths[0])  # establishes the shared reference geometry
        self.num_nodes = int(len(self.reader.keep_ids))
        # one_time_step enumerates every (case, future frame) pair; the time label
        # k / (num_steps - 1) is normalized by the configured frame bound so the
        # same label means the same physical time in every case.
        self.case_future_frames = None
        self.case_offsets = None
        if sample_type == "one_time_step":
            counts = []
            for path in self.paths:
                frames = reader.frames(reader.open(path))
                if len(frames) > self.num_steps or len(frames) < 2:
                    raise ValueError(
                        f"{path.stem}: {len(frames)} selected frames outside [2, {self.num_steps}]"
                    )
                counts.append(len(frames) - 1)
            self.case_future_frames = counts
            offsets, total = [], 0
            for count in counts:
                offsets.append(total)
                total += count
            self.case_offsets = offsets
            self.num_pairs = total

        self._stats_dir = Path(stats_dir)
        self._stats_dir.mkdir(parents=True, exist_ok=True)
        identity = dict(
            schema=1,
            split_manifest_signature=reader.manifest_signature,
            outlier_signature=reader.outlier_signature,
            kept_nodes=self.num_nodes,
            stride=reader.stride,
            raw_dt=reader.raw_dt,
            max_frames=reader.max_frames,
            normalization=NORMALIZATION,
        )
        stats_path = self._stats_dir / STATS_FILE
        if stats_mode == "compute" and split == "train" and not stats_path.exists():
            started = time.monotonic()
            train_paths = reader.paths(data_dir, "train", len(reader.splits["train"]))
            values = compute_frozen_stats(reader, train_paths, workers=stats_workers)
            atomic_json(
                stats_path,
                dict(
                    identity=identity,
                    values=values,
                    train_cases=[path.stem for path in train_paths],
                    data_root=str(Path(data_dir).resolve()),
                ),
            )
            self.logger.info(
                f"Yaris frozen statistics computed in {time.monotonic() - started:.0f}s"
            )
        if not stats_path.exists():
            raise FileNotFoundError(
                f"{stats_path} not found; build the training split first"
            )
        saved = json.loads(stats_path.read_text())
        if saved["identity"] != identity:
            raise ValueError(
                "Frozen normalization identity changed: "
                f"saved={saved['identity']} current={identity}"
            )
        values = saved["values"]
        for group in ("node", "feature"):
            for key, value in values[group].items():
                array = np.asarray(value, dtype=np.float64)
                if not np.isfinite(array).all() or (
                    key.endswith("std") and (array <= 0).any()
                ):
                    raise ValueError(f"Invalid frozen normalization value {key}")
        self.node_stats = {
            k: torch.as_tensor(v, dtype=torch.float32)
            for k, v in values["node"].items()
        }
        self.feature_stats = {
            k: torch.as_tensor(v, dtype=torch.float32)
            for k, v in values["feature"].items()
        }
        self.edge_stats = {}
        self.global_stats = {
            "global_mean": torch.zeros(0, dtype=torch.float32),
            "global_std": torch.ones(0, dtype=torch.float32),
        }
        self.global_features_keys = []
        self.graph = self._build_graph(local_knn_cache_dir, stats_mode)

    # ----- graph -----
    def _node_feature_tensor(self, store):
        """Standardized thickness, ``[N, 1]``."""
        thickness = torch.from_numpy(self.reader.thickness(store))
        return (thickness[:, None] - self.feature_stats["feature_mean"]) / (
            self.feature_stats["feature_std"] + EPS
        )

    def _build_graph(self, cache_dir, stats_mode):
        """The shared graph: kNN neighbourhoods with standardized features and the
        optional mesh edges, contact topology and rigid wall."""
        from local_neighborhoods import (
            LocalNeighborhoodCache,
            LocalNeighborhoodData,
            knn_edge_features,
            mesh_edge_features,
            with_self_slot,
        )

        reference = torch.from_numpy(self.reader.reference.copy())
        if self.local_knn_metric == "geodesic":
            edge_index = self.reader.structural_edges(self.reader.open(self.paths[0]))
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        cache = LocalNeighborhoodCache(cache_dir, logger=self.logger)
        neighborhood = cache.get(
            reference, edge_index, self.local_knn_k, self.local_knn_metric
        )
        knn_index, knn_mask = with_self_slot(neighborhood)
        normalized_reference = (reference - self.node_stats["pos_mean"]) / (
            self.node_stats["pos_std"] + EPS
        )
        knn_edge_attr = knn_edge_features(normalized_reference, knn_index)
        knn_stats_path = self._stats_dir / KNN_EDGE_STATS_FILE
        if stats_mode == "compute" and self.split == "train":
            valid = knn_mask.clone()
            valid[:, 0] = False
            values = knn_edge_attr[valid].to(torch.float64)
            if values.numel() == 0:
                raise ValueError("no valid kNN neighbors to compute statistics")
            mean = values.mean(dim=0)
            variance = torch.clamp((values * values).mean(dim=0) - mean * mean, min=0)
            self.knn_edge_stats = {
                "knn_edge_mean": mean.to(torch.float32),
                "knn_edge_std": torch.sqrt(variance + EPS).to(torch.float32),
            }
            atomic_json(
                knn_stats_path,
                {k: v.tolist() for k, v in self.knn_edge_stats.items()},
            )
        elif knn_stats_path.exists():
            self.knn_edge_stats = {
                k: torch.as_tensor(v, dtype=torch.float32)
                for k, v in json.loads(knn_stats_path.read_text()).items()
            }
        else:
            raise FileNotFoundError(
                f"{knn_stats_path} not found; build the training split with the "
                "same local_knn settings first"
            )
        if self.contact_surface:
            from surface_topology import SurfaceContactData

            graph = SurfaceContactData(
                edge_index=edge_index,
                edge_attr=torch.zeros((edge_index.shape[1], 4), dtype=torch.float32),
                num_nodes=self.num_nodes,
            )
        else:
            graph = LocalNeighborhoodData(
                edge_index=edge_index,
                edge_attr=torch.zeros((edge_index.shape[1], 4), dtype=torch.float32),
                num_nodes=self.num_nodes,
            )
        graph.knn_index = knn_index
        graph.knn_mask = knn_mask
        graph.knn_edge_attr = (
            knn_edge_attr - self.knn_edge_stats["knn_edge_mean"].view(1, 1, -1)
        ) / (self.knn_edge_stats["knn_edge_std"].view(1, 1, -1) + EPS)
        graph.knn_metric = self.local_knn_metric
        if self.mesh_edges:
            # Shell (plus optional solid) connectivity in both directions with its
            # reference displacement and length, standardized over the edge set
            # (the mesh is shared by every design, so the statistics are too).
            undirected = (
                edge_index
                if edge_index.shape[1]
                else self.reader.structural_edges(self.reader.open(self.paths[0]))
            )
            mesh_edge_index = torch.cat((undirected, undirected.flip(0)), dim=1)
            attr = mesh_edge_features(normalized_reference, mesh_edge_index)
            values = attr.to(torch.float64)
            mean = values.mean(dim=0)
            variance = torch.clamp((values * values).mean(dim=0) - mean * mean, min=0)
            graph.mesh_edge_index = mesh_edge_index
            graph.mesh_edge_mean = mean.to(torch.float32)
            graph.mesh_edge_std = torch.sqrt(variance + EPS).to(torch.float32)
            graph.mesh_edge_attr = (attr - graph.mesh_edge_mean) / (
                graph.mesh_edge_std + EPS
            )
        if self.contact_surface:
            # Facets and exclusions are mesh topology, shared by every design;
            # the nodal thickness is the design variable and is attached per
            # case by _case_graph.
            store = self.reader.open(self.paths[0])
            graph.contact_faces = self.reader.contact_faces(store)
            graph.contact_surface_exclusions = self._contact_exclusions(
                graph.contact_faces
            )
        if self.rigid_wall_position is not None:
            graph.rigid_wall = torch.tensor(
                [self.rigid_wall_position], dtype=torch.float32
            )
        return graph

    def _contact_exclusions(self, faces):
        """One-ring node/facet exclusions, cached next to the statistics."""
        from surface_topology import surface_one_ring_exclusions

        digest = hashlib.sha256(faces.numpy().tobytes()).hexdigest()[:16]
        path = self._stats_dir / f"contact_one_ring_{digest}.pt"
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True)
        started = time.monotonic()
        exclusions = surface_one_ring_exclusions(faces, self.num_nodes)
        tmp = path.with_suffix(".tmp")
        torch.save(exclusions, tmp)
        os.replace(tmp, path)
        self.logger.info(
            f"Yaris contact exclusions: {exclusions.shape[1]} node-facet pairs for "
            f"{len(faces)} facets in {time.monotonic() - started:.0f}s"
        )
        return exclusions

    # ----- Dataset protocol -----
    def _case_graph(self, store):
        """The shared graph with this design's physical thickness ``[N]`` attached.

        The thickness varies between designs, so it is read per case and set on
        a shallow copy of the shared graph (``Data.__copy__`` copies the
        attribute store, not the tensors) for the contact search of the
        corrector and the penetration metrics.
        """
        if not (self.contact_surface or self.rigid_wall_position is not None):
            return self.graph
        graph = copy.copy(self.graph)
        graph.shell_thickness = torch.from_numpy(self.reader.thickness(store))
        return graph

    def __len__(self):
        if self.sample_type == "one_time_step":
            return self.num_pairs
        return self.num_samples

    def _pair(self, index):
        """Map a flat one_time_step index to (case index, future frame offset k)."""
        import bisect

        case = bisect.bisect_right(self.case_offsets, index) - 1
        return case, index - self.case_offsets[case]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.sample_type == "one_time_step":
            return self._time_step_sample(index)
        path = self.paths[index]
        store = self.reader.open(path)
        frames = self.reader.frames(store)
        if len(frames) > self.num_steps or len(frames) < 2:
            raise ValueError(
                f"{path.stem}: {len(frames)} selected frames outside the configured "
                f"[2, {self.num_steps}] range"
            )
        pos = torch.from_numpy(self.reader.coordinates(store, frames))
        pos = (pos - self.node_stats["pos_mean"]) / (self.node_stats["pos_std"] + EPS)
        features = self._node_feature_tensor(store)
        # The initial frame is observed; every later frame is a target.
        sample = SimSample(
            node_features=dict(coords=pos[0], features=features),
            node_target=pos[1:].transpose(0, 1).contiguous(),
            graph=self._case_graph(store),
            global_features=None,
        )
        sample.yaris_case_name = path.stem
        sample.yaris_target_frames = frames[1:].tolist()
        return sample

    def _time_step_sample(self, index):
        """Initial frame plus one future frame, labelled by normalized time."""
        case, k = self._pair(index)
        path = self.paths[case]
        store = self.reader.open(path)
        frames = self.reader.frames(store)
        pos = torch.from_numpy(
            self.reader.coordinates(store, [frames[0], frames[1 + k]])
        )
        pos = (pos - self.node_stats["pos_mean"]) / (self.node_stats["pos_std"] + EPS)
        node_features = dict(
            coords=pos[0],
            features=self._node_feature_tensor(store),
            time=torch.tensor(k / (self.num_steps - 1), dtype=torch.float32),
        )
        sample = SimSample(
            node_features=node_features,
            node_target=pos[1].contiguous(),
            graph=self._case_graph(store),
            global_features=None,
        )
        sample.yaris_case_name = path.stem
        sample.yaris_target_frames = [int(frames[1 + k])]
        return sample
