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
"""Lazy full-resolution Yaris data for the kNN-attention DeFormer.

Every case is one Zarr store with the same 1.48M-node reference mesh. This
datapipe keeps the mesh at full resolution: no coarse bank, no prolongation.
Nodes listed in an outlier mask (rigid-wall nodes and fly-away nodes) are
dropped from inputs, targets and neighborhoods. The static kNN neighborhoods
and their standardized features are built once from the shared reference
geometry; positions and thickness are read per case and per window, so no
trajectory is preloaded.

Normalization follows the crash datapipe: per-axis position mean/std over the
training cases (equal case weight), velocities and accelerations in
normalized coordinates, and a per-channel thickness standardization.
"""

from __future__ import annotations

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


def load_splits(path):
    """Case names per split from the dataset creator's CSV (names are identity)."""
    splits = {key: [] for key in ("train", "validation", "test")}
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            split, name = row["split"].strip(), row["sample_name"].strip()
            name = name.removesuffix(".zarr")
            if split not in splits or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError(f"Invalid split/sample name: {split!r}/{name!r}")
            if name in seen:
                raise ValueError(f"Duplicate/cross-split sample name: {name}")
            if name.endswith("100_GR252303"):
                raise ValueError(
                    "Known mislabelled duplicate 100_GR252303 must be excluded"
                )
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
        pos = store["nodes/pos"]
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
        values = np.asarray(store["nodes/thickness"][:], dtype=np.float32).reshape(-1)
        if values.shape != (self.total_nodes,):
            raise ValueError("Thickness/node count mismatch")
        values = values[self.keep_ids]
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Thickness must be finite and nonnegative")
        return values

    def part_ids(self, store):
        """Part id per kept node (``nodes/part_id``), int64 ``[n_keep]``."""
        try:
            values = np.asarray(store["nodes/part_id"][:]).reshape(-1)
        except KeyError as error:
            raise ValueError("part tokens need nodes/part_id in the store") from error
        if values.shape != (self.total_nodes,):
            raise ValueError("part_id/node count mismatch")
        return values[self.keep_ids].astype(np.int64)

    def structural_edges(self, store):
        """Reference connectivity restricted and renumbered to kept nodes, ``[2, E]``.

        Shell edges from the store, plus (``include_solid_edges``) the edges of
        the hexahedral ``solid/connectivity`` table (collapsed hexes contribute
        their distinct edges), plus (``connect_isolated_k > 0``) Euclidean links
        from every node left without an edge to its nearest kept neighbors.
        Duplicates are removed; direction is irrelevant downstream.
        """
        edges = np.asarray(store["edges"][:], dtype=np.int64)
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
        from physicsnemo.nn.functional.neighbors.surface_contact import (
            supported_quad_mask,
        )

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

    def design_variables(self, store):
        """The 8 thickness-group deviations and a known-flag, ``[9]`` float32.

        Deviations are fractional part-group thickness changes within about
        +-0.25; they are scaled by 4 to order one. Six of the 99 Yaris runs
        carry no design record: they get zeros and flag 0.
        """
        attrs = dict(store.attrs)
        keys = [f"bc_thickness_group_{i}_deviation" for i in range(1, 9)]
        values = None
        if all(key in attrs for key in keys):
            values = [float(attrs[key]) for key in keys]
        else:
            try:
                values = [
                    float(
                        np.asarray(store[f"boundary_conditions/{key[3:]}"][:]).reshape(
                            -1
                        )[0]
                    )
                    for key in keys
                ]
            except KeyError:
                values = None
        if values is None or not all(math.isfinite(v) for v in values):
            return np.zeros(9, dtype=np.float32)
        return np.asarray([4.0 * v for v in values] + [1.0], dtype=np.float32)


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
    """Full-resolution Yaris cases with static kNN neighborhoods; one window per access."""

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
        sample_type="random_time_window",
        initial_history_steps=2,
        rollout_window_steps=4,
        dt=0.005,
        static_features=("thickness",),
        dynamic_features=(),
        dynamic_targets=(),
        global_features=None,
        global_features_filepath=None,
        local_knn_k=32,
        local_knn_metric="euclidean",
        local_knn_cache_dir=None,
        window_seed=42,
        windows_per_sample=1,
        stats_workers=None,
        design_variables=False,
        contact_surface=False,
        rigid_wall_position=None,
        thickness_deviation=False,
        mixup_alpha=0.0,
        mixup_prob=0.5,
        mean_displacement_baseline=False,
        delta_reference=False,
        delta_val_references=3,
        delta_train_nearest_prob=0.5,
        delta_train_nearest_k=5,
        part_tokens=False,
    ):
        self.reader, self.name, self.split = reader, name, split
        # Part-pooled design tokens: one token per part (reference centroid,
        # log size, mean of the node feature columns) under
        # sample.global_features["parts"], consumed by the model's design encoder.
        self.part_tokens = bool(part_tokens)
        # Delta-to-reference: every sample pairs the case with a TRAINING case
        # (the reference) at the same future frame; the node features carry
        # both thickness fields and the reference's displacement is served as
        # the output baseline, so the network predicts the response to the
        # design change. Validation averages the predictions obtained from
        # the nearest training designs (thickness-field distance).
        self.delta_reference = bool(delta_reference)
        if (
            isinstance(delta_val_references, bool)
            or int(delta_val_references) < 1
            or not 0.0 <= float(delta_train_nearest_prob) <= 1.0
            or isinstance(delta_train_nearest_k, bool)
            or int(delta_train_nearest_k) < 1
        ):
            raise ValueError(
                "delta_val_references and delta_train_nearest_k must be positive "
                "integers and delta_train_nearest_prob in [0, 1]"
            )
        self.delta_val_references = int(delta_val_references)
        self.delta_train_nearest_prob = float(delta_train_nearest_prob)
        self.delta_train_nearest_k = int(delta_train_nearest_k)
        if self.delta_reference and (mean_displacement_baseline or mixup_alpha > 0):
            raise ValueError(
                "delta_reference excludes mean_displacement_baseline and mixup"
            )
        if self.delta_reference and (
            sample_type not in ("one_time_step", "all_time_steps")
            or int(initial_history_steps) != 1
        ):
            raise ValueError("delta_reference is for the time-conditional sample types")
        # Output baseline: the mean normalized displacement of the TRAIN cases
        # at each future frame, served as node_features["baseline"] (one frame)
        # or ["baseline_all"] (every validated frame); the model adds it to its
        # prediction so the network carries a zero-mean residual.
        self.mean_displacement_baseline = bool(mean_displacement_baseline)
        # Design mixup (train, one_time_step): with probability mixup_prob a
        # sample blends two training cases at the same future frame, lambda ~
        # Beta(alpha, alpha), over the thickness features, the design token and
        # the target positions (the reference geometry is shared).
        if (
            not math.isfinite(mixup_alpha)
            or mixup_alpha < 0
            or not 0.0 <= float(mixup_prob) <= 1.0
        ):
            raise ValueError("mixup_alpha must be >= 0 and mixup_prob in [0, 1]")
        self.mixup_alpha, self.mixup_prob = float(mixup_alpha), float(mixup_prob)
        # The 8 thickness-group deviations (+ known flag) as a global token.
        self.design_variables = bool(design_variables)
        # Mesh-derived design signal: each node's thickness minus its mean over
        # the TRAIN cases, scaled by the fleet deviation std (a second feature
        # column). General for shared-mesh datasets; needs no design metadata.
        self.thickness_deviation = bool(thickness_deviation)
        # Surface contact topology (shell facets, physical thickness, one-ring
        # exclusions) and the analytic rigid wall ``x = rigid_wall_position``.
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
        if sample_type == "one_time_step":
            if int(initial_history_steps) != 1:
                raise ValueError("one_time_step predicts from the initial frame only")
        elif sample_type == "random_time_window" and int(initial_history_steps) != 2:
            raise ValueError("Two observed frames initialize the velocity")
        elif int(initial_history_steps) not in (1, 2):
            raise ValueError("initial_history_steps must be 1 or 2")
        self.initial_history_steps = int(initial_history_steps)
        if sample_type not in ("random_time_window", "all_time_steps", "one_time_step"):
            raise ValueError(
                "sample_type must be random_time_window, one_time_step or all_time_steps"
            )
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
        self.rollout_window_steps = int(rollout_window_steps)
        if not 1 <= self.rollout_window_steps <= self.num_steps - 2:
            raise ValueError("rollout_window_steps must be in [1, num_steps - 2]")
        self.windows_per_sample = int(windows_per_sample)
        if self.windows_per_sample < 1:
            raise ValueError("windows_per_sample must be positive")
        self.window_seed = None if window_seed is None else int(window_seed)
        self.local_knn_k, self.local_knn_metric = int(local_knn_k), local_knn_metric
        self.epoch = 0
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
        n_global = 9 if self.design_variables else 0
        self.global_stats = {
            "global_mean": torch.zeros(n_global, dtype=torch.float32),
            "global_std": torch.ones(n_global, dtype=torch.float32),
        }
        self.global_features_keys = ["design"] if self.design_variables else []
        self.thickness_fleet_mean, self.thickness_fleet_dev_std = (
            self._thickness_fleet(data_dir) if self.thickness_deviation else (None, 0.0)
        )
        self.mean_displacement = (
            self._mean_displacement(data_dir, stats_mode)
            if self.mean_displacement_baseline
            else None
        )
        self.delta_neighbours = (
            self._delta_neighbours(data_dir, stats_mode)
            if self.delta_reference
            else None
        )
        if self.part_tokens:
            ids = reader.part_ids(reader.open(self.paths[0]))
            unique, inverse = np.unique(ids, return_inverse=True)
            self.part_inverse = torch.from_numpy(inverse.astype(np.int64).reshape(-1))
            self.part_count = int(len(unique))
            counts = torch.bincount(self.part_inverse, minlength=self.part_count)
            self.part_counts = counts.to(torch.float32).clamp_min(1.0)
            reference = (
                torch.from_numpy(reader.reference) - self.node_stats["pos_mean"]
            ) / (self.node_stats["pos_std"] + EPS)
            centroid = (
                torch.zeros((self.part_count, 3)).index_add_(
                    0, self.part_inverse, reference
                )
                / self.part_counts[:, None]
            )
            log_size = torch.log(self.part_counts) / math.log(max(self.num_nodes, 2))
            self.part_static = torch.cat((centroid, log_size[:, None]), dim=1)
        self.graph = self._build_graph(local_knn_cache_dir, stats_mode)

    # ----- shared reference structure -----
    def _thickness_fleet(self, data_dir):
        """Per-node mean thickness over the TRAIN cases and the deviation scale.

        Cached in the statistics directory under the split/outlier identity.
        The scale is the root mean square of the per-node deviation over all
        training cases and kept nodes (nodes whose thickness never changes
        contribute zeros, so group nodes come out several scales wide).
        """
        digest = hashlib.sha256(
            f"{self.reader.manifest_signature}:{self.reader.outlier_signature}".encode()
        ).hexdigest()[:16]
        path = self._stats_dir / f"thickness_fleet_{digest}.npz"
        if path.exists():
            data = np.load(path)
            return torch.from_numpy(data["mean"].astype(np.float32)), float(
                data["dev_std"]
            )
        started = time.monotonic()
        train_paths = self.reader.paths(
            data_dir, "train", len(self.reader.splits["train"])
        )
        total = square = None
        for case in train_paths:
            values = self.reader.thickness(self.reader.open(case)).astype(np.float64)
            total = values if total is None else total + values
            square = values**2 if square is None else square + values**2
        mean = total / len(train_paths)
        variance = np.maximum(square / len(train_paths) - mean**2, 0.0)
        dev_std = float(np.sqrt(variance.mean()))
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, mean=mean.astype(np.float32), dev_std=dev_std)
        os.replace(tmp, path)
        self.logger.info(
            f"Yaris thickness fleet mean over {len(train_paths)} training cases "
            f"(deviation scale {dev_std:.4f} mm) in {time.monotonic() - started:.0f}s"
        )
        return torch.from_numpy(mean.astype(np.float32)), dev_std

    def _mean_displacement(self, data_dir, stats_mode):
        """Mean normalized displacement of the TRAIN cases per future frame.

        ``[num_steps - 1, N, 3]`` float32, frame ``1 + k`` minus frame 0 over the
        position scale, averaged over the training cases that have that frame.
        Computed by the CPU preparation (``stats_mode="compute"``) and cached in
        the statistics directory; training and validation load it.
        """
        reader = self.reader
        digest = hashlib.sha256(
            f"{reader.manifest_signature}:{reader.outlier_signature}:{reader.stride}:"
            f"{reader.max_frames}:{self.num_steps}".encode()
        ).hexdigest()[:16]
        path = self._stats_dir / f"mean_displacement_{digest}.npy"
        if path.exists():
            return torch.from_numpy(np.load(path))
        if stats_mode != "compute":
            raise FileNotFoundError(
                f"{path} not found; run yaris_full_prepare.py with "
                "datapipe.mean_displacement_baseline=true first"
            )
        started = time.monotonic()
        train_paths = reader.paths(data_dir, "train", len(reader.splits["train"]))
        steps = self.num_steps - 1
        total = np.zeros((steps, self.num_nodes, 3), dtype=np.float64)
        count = np.zeros(steps, dtype=np.int64)
        for case in train_paths:
            store = reader.open(case)
            frames = reader.frames(store)
            n = min(len(frames) - 1, steps)
            coords = reader.coordinates(store, frames[: n + 1]).astype(np.float64)
            total[:n] += coords[1:] - coords[:1]
            count[:n] += 1
        scale = self.node_stats["pos_std"].numpy().reshape(1, 1, 3) + EPS
        mean = (total / np.maximum(count, 1)[:, None, None] / scale).astype(np.float32)
        tmp = path.with_name(path.stem + ".tmp.npy")
        np.save(tmp, mean)
        os.replace(tmp, path)
        self.logger.info(
            f"Yaris mean displacement baseline over {len(train_paths)} training cases "
            f"and {steps} future frames in {time.monotonic() - started:.0f}s"
        )
        return torch.from_numpy(mean)

    def _delta_neighbours(self, data_dir, stats_mode):
        """Training references per case, nearest first by thickness-field distance.

        Returns ``{"train_cases": [...], "train_frames": {name: frames},
        "nearest": {case: [train names]}}`` for every case of every split,
        excluding the case itself. Cached in the statistics directory (built by
        the CPU preparation with ``stats_mode="compute"``).
        """
        reader = self.reader
        digest = hashlib.sha256(
            f"{reader.manifest_signature}:{reader.outlier_signature}:{reader.stride}:"
            f"{reader.max_frames}".encode()
        ).hexdigest()[:16]
        path = self._stats_dir / f"delta_neighbours_{digest}.json"
        if path.exists():
            data = json.loads(path.read_text())
            data["train_paths"] = {
                case.stem: case
                for case in reader.paths(data_dir, "train", len(reader.splits["train"]))
            }
            return data
        if stats_mode != "compute":
            raise FileNotFoundError(
                f"{path} not found; run yaris_full_prepare.py with "
                "datapipe.delta_reference=true first"
            )
        started = time.monotonic()
        train_paths = reader.paths(data_dir, "train", len(reader.splits["train"]))
        train_names = [case.stem for case in train_paths]
        train_thickness = np.stack(
            [
                reader.thickness(reader.open(case)).astype(np.float32)
                for case in train_paths
            ]
        )
        train_frames = {
            case.stem: int(len(reader.frames(reader.open(case))))
            for case in train_paths
        }
        nearest = {}
        for split, names in reader.splits.items():
            for case in reader.paths(data_dir, split, len(names)):
                thickness = reader.thickness(reader.open(case)).astype(np.float32)
                distance = np.sqrt(np.square(train_thickness - thickness[None]).sum(1))
                order = [
                    train_names[i]
                    for i in np.argsort(distance, kind="stable")
                    if train_names[i] != case.stem
                ]
                nearest[case.stem] = order
        data = dict(train_cases=train_names, train_frames=train_frames, nearest=nearest)
        atomic_json(path, data)
        self.logger.info(
            f"Yaris delta references ranked for {len(nearest)} cases against "
            f"{len(train_names)} training designs in {time.monotonic() - started:.0f}s"
        )
        data["train_paths"] = {case.stem: case for case in train_paths}
        return data

    def _reference_displacement(self, name, frame_indices):
        """Features of training case ``name`` and its normalized displacement at
        the given target frame indices (frame ``1 + k`` minus frame 0): ``[N, 3]``
        for one index, ``[N, T, 3]`` for several."""
        store = self.reader.open(self.delta_neighbours["train_paths"][name])
        frames = self.reader.frames(store)
        selected = [frames[0]] + [frames[1 + k] for k in frame_indices]
        pos = torch.from_numpy(self.reader.coordinates(store, selected))
        displacement = (pos[1:] - pos[:1]) / (self.node_stats["pos_std"] + EPS)
        features = self._node_feature_tensor(store)
        if len(frame_indices) == 1:
            return features, displacement[0].contiguous()
        return features, displacement.permute(1, 0, 2).contiguous()

    def _delta_train_reference(self, index, case_name, k):
        """Training reference for sample ``index``: nearest-k with probability
        ``delta_train_nearest_prob``, else uniform over the training designs."""
        frames = self.delta_neighbours["train_frames"]
        ranked = [
            name
            for name in self.delta_neighbours["nearest"][case_name]
            if frames[name] >= k + 2
        ]
        if not ranked:
            raise ValueError(
                f"no training reference with frame {1 + k} for {case_name}"
            )
        key = f"crash-delta-v1:{self.window_seed}:{self.epoch}:{index}".encode()
        seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
        rng = np.random.default_rng(seed)
        pool = (
            ranked[: self.delta_train_nearest_k]
            if rng.random() < self.delta_train_nearest_prob
            else ranked
        )
        return pool[int(rng.integers(len(pool)))]

    def _node_feature_tensor(self, store):
        """Standardized thickness, plus the fleet deviation column when enabled."""
        thickness = torch.from_numpy(self.reader.thickness(store))
        features = (thickness[:, None] - self.feature_stats["feature_mean"]) / (
            self.feature_stats["feature_std"] + EPS
        )
        if self.thickness_deviation:
            deviation = (thickness - self.thickness_fleet_mean) / (
                self.thickness_fleet_dev_std + EPS
            )
            features = torch.cat((features, deviation[:, None].to(features.dtype)), 1)
        return features

    def _build_graph(self, cache_dir, stats_mode):
        from local_neighborhoods import (
            LocalNeighborhoodCache,
            LocalNeighborhoodData,
            knn_edge_features,
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
        if self.contact_surface or self.rigid_wall_position is not None:
            store = self.reader.open(self.paths[0])
            graph.shell_thickness = torch.from_numpy(self.reader.thickness(store))
        if self.contact_surface:
            graph.contact_faces = self.reader.contact_faces(store)
            graph.contact_surface_exclusions = self._contact_exclusions(
                graph.contact_faces
            )
            graph.penetration_reference_positions = reference.clone()
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
    def __len__(self):
        if self.sample_type == "one_time_step":
            return self.num_pairs
        return self.num_samples * self.windows_per_sample

    def _pair(self, index):
        """Map a flat one_time_step index to (case index, future frame offset k)."""
        import bisect

        case = bisect.bisect_right(self.case_offsets, index) - 1
        return case, index - self.case_offsets[case]

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _window_start(self, sample_idx, num_frames):
        """Same stateless draw as the crash datapipe (seed, epoch, sample slot)."""
        window = self.rollout_window_steps + 2
        upper = num_frames - window  # inclusive upper bound for the first frame
        if self.window_seed is None:
            return int(torch.randint(0, upper + 1, (1,)).item())
        key = f"crash-window-v2:{self.window_seed}:{self.epoch}:{sample_idx}".encode()
        seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
        generator = torch.Generator().manual_seed(seed)
        return int(torch.randint(0, upper + 1, (1,), generator=generator).item())

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.sample_type == "one_time_step":
            return self._time_step_sample(index)
        path = self.paths[index % self.num_samples]
        store = self.reader.open(path)
        frames = self.reader.frames(store)
        minimum = (
            self.rollout_window_steps + 2
            if self.sample_type == "random_time_window"
            else self.initial_history_steps + 1
        )
        if len(frames) > self.num_steps or len(frames) < minimum:
            raise ValueError(
                f"{path.stem}: {len(frames)} selected frames outside the configured "
                f"[{minimum}, {self.num_steps}] range"
            )
        if self.split == "train" and self.sample_type == "random_time_window":
            start = self._window_start(index, len(frames))
            frames = frames[start : start + self.rollout_window_steps + 2]
        pos = torch.from_numpy(self.reader.coordinates(store, frames))
        pos = (pos - self.node_stats["pos_mean"]) / (self.node_stats["pos_std"] + EPS)
        features = self._node_feature_tensor(store)
        # One observed frame (time-conditional evaluation) or two (velocity
        # initialization of the autoregressive rollout); targets follow.
        history = self.initial_history_steps
        node_features = dict(coords=pos[history - 1], features=features)
        if history == 2:
            node_features["previous_coords"] = pos[0]
        if self.mean_displacement is not None and history == 1:
            steps = int(pos.shape[0] - history)
            node_features["baseline_all"] = (
                self.mean_displacement[:steps].permute(1, 0, 2).contiguous()
            )
        references, extra_references = [], []
        if self.delta_reference and history == 1:
            steps = int(pos.shape[0] - history)
            train_frames = self.delta_neighbours["train_frames"]
            references = [
                name
                for name in self.delta_neighbours["nearest"][path.stem]
                if train_frames[name] >= steps + 1
            ][: self.delta_val_references]
            if not references:
                raise ValueError(
                    f"no training reference covers the {steps} frames of {path.stem}"
                )
            for position, name in enumerate(references):
                ref_features, ref_displacement = self._reference_displacement(
                    name, list(range(steps))
                )
                paired = torch.cat((features, ref_features), dim=1)
                if position == 0:
                    node_features["features"] = paired
                    node_features["baseline_all"] = ref_displacement
                else:
                    extra_references.append((paired, ref_displacement))
        sample = SimSample(
            node_features=node_features,
            node_target=pos[history:].transpose(0, 1).contiguous(),
            graph=self.graph,
            global_features=self._global_features(store),
        )
        sample.yaris_case_name = path.stem
        sample.yaris_target_frames = frames[history:].tolist()
        sample.yaris_delta_references = references
        sample.delta_references = extra_references
        return self._attach_part_tokens(sample)

    def _global_features(self, store):
        if not self.design_variables:
            return None
        return {"design": torch.from_numpy(self.reader.design_variables(store))}

    def _part_token_tensor(self, features):
        """``[P, 4 + F]``: reference centroid, log size, per-part feature means."""
        means = (
            torch.zeros((self.part_count, features.shape[1])).index_add_(
                0, self.part_inverse, features.float()
            )
            / self.part_counts[:, None]
        )
        return torch.cat((self.part_static, means), dim=1)

    def _attach_part_tokens(self, sample):
        if self.part_tokens:
            tokens = self._part_token_tensor(sample.node_features["features"])
            sample.global_features = {**(sample.global_features or {}), "parts": tokens}
        return sample

    def _mixup_draw(self, index, case, k):
        """(partner case, lambda) or None; stateless in (seed, epoch, index)."""
        if self.split != "train" or self.mixup_alpha <= 0 or self.mixup_prob <= 0:
            return None
        key = f"crash-mixup-v1:{self.window_seed}:{self.epoch}:{index}".encode()
        seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
        rng = np.random.default_rng(seed)
        if rng.random() >= self.mixup_prob:
            return None
        candidates = [
            j
            for j, count in enumerate(self.case_future_frames)
            if j != case and count > k
        ]
        if not candidates:
            return None
        partner = int(candidates[int(rng.integers(len(candidates)))])
        lam = float(rng.beta(self.mixup_alpha, self.mixup_alpha))
        return partner, lam

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
        features = self._node_feature_tensor(store)
        target = pos[1].contiguous()
        global_features = self._global_features(store)
        name = path.stem
        mix = self._mixup_draw(index, case, k)
        if mix is not None:
            partner, lam = mix
            other_path = self.paths[partner]
            other = self.reader.open(other_path)
            other_frames = self.reader.frames(other)
            other_pos = torch.from_numpy(
                self.reader.coordinates(other, [other_frames[1 + k]])
            )[0]
            other_pos = (other_pos - self.node_stats["pos_mean"]) / (
                self.node_stats["pos_std"] + EPS
            )
            target = lam * target + (1.0 - lam) * other_pos
            features = lam * features + (1.0 - lam) * self._node_feature_tensor(other)
            other_global = self._global_features(other)
            if global_features is not None and other_global is not None:
                global_features = {
                    key: lam * value + (1.0 - lam) * other_global[key]
                    for key, value in global_features.items()
                }
            name = f"{path.stem}+{other_path.stem}@{lam:.3f}"
        node_features = dict(
            coords=pos[0],
            features=features,
            time=torch.tensor(k / (self.num_steps - 1), dtype=torch.float32),
        )
        if self.mean_displacement is not None:
            node_features["baseline"] = self.mean_displacement[k]
        reference = None
        if self.delta_reference:
            reference = self._delta_train_reference(index, path.stem, k)
            ref_features, ref_displacement = self._reference_displacement(
                reference, [k]
            )
            node_features["features"] = torch.cat((features, ref_features), dim=1)
            node_features["baseline"] = ref_displacement
        sample = SimSample(
            node_features=node_features,
            node_target=target.contiguous(),
            graph=self.graph,
            global_features=global_features,
        )
        sample.yaris_delta_reference = reference
        sample.yaris_case_name = name
        sample.yaris_target_frames = [int(frames[1 + k])]
        sample.yaris_mixup = mix
        return self._attach_part_tokens(sample)
