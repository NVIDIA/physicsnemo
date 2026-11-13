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
"""Static reference kNN neighborhoods for DeFormer's kNN attention.

The datapipe computes, once per distinct initial mesh, each node's ``k``
nearest reference neighbors (geodesic along structural edges or Euclidean),
prepends the node itself as slot zero, and attaches per-slot reference
displacement/distance features. Neighborhoods describe the undeformed material
configuration and never change during a rollout. Selection is discrete; the
features are standardized with train-only statistics like structural edges.
"""

import fcntl
import hashlib
import inspect
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from physicsnemo.experimental.models.deformer import (  # noqa: F401 (re-export)
    LocalNeighborhoodData,
)
from physicsnemo.experimental.nn import geodesic_knn as _geodesic_knn_module
from physicsnemo.experimental.nn.geodesic_knn import (
    ReferenceNeighborhood,
    geodesic_knn,
)
from physicsnemo.nn.functional import knn

KNN_METRICS = ("geodesic", "euclidean")


@torch.no_grad()
def euclidean_knn(reference_positions: torch.Tensor, k: int) -> ReferenceNeighborhood:
    """Return the ``k`` nearest other nodes by straight-line reference distance.

    The query is the library's :func:`physicsnemo.nn.functional.knn` on its SciPy
    KD-tree backend (pinned, so cached neighborhoods do not depend on which
    backends are installed), asked for ``k + 1`` candidates per node in float64.
    This wrapper removes the node itself, orders the candidates by
    ``(distance, node ID)`` and pads short rows like the geodesic query, so both
    metrics share one contract. Coincident duplicate nodes are ordinary
    neighbors at zero distance. Which of several candidates tied exactly at the
    ``k`` boundary is returned is the KD-tree's choice, not the smaller node ID.
    """
    if (
        not isinstance(reference_positions, torch.Tensor)
        or reference_positions.device.type != "cpu"
        or reference_positions.dtype not in (torch.float32, torch.float64)
        or reference_positions.ndim != 2
        or reference_positions.shape[1] != 3
        or not torch.isfinite(reference_positions).all()
    ):
        raise ValueError("reference_positions must be finite CPU float [N,3]")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    xyz = reference_positions.detach().to(torch.float64)
    n = xyz.shape[0]
    own = np.arange(n, dtype=np.int64)[:, None]
    out_index = np.repeat(own, k, axis=1)
    out_distance = np.zeros((n, k), dtype=np.float32)
    out_mask = np.zeros((n, k), dtype=bool)
    if n > 1:
        queries = min(k + 1, n)
        try:
            index, distance = knn(xyz, xyz, queries, implementation="scipy")
        except ImportError as error:
            raise ImportError(
                "Euclidean reference neighborhoods need SciPy (pip install scipy)"
            ) from error
        index = index.numpy().astype(np.int64).reshape(n, queries)
        distance = distance.numpy().astype(np.float64).reshape(n, queries)
        # The query itself (if returned) last, then (distance, node ID).
        order = np.lexsort((index, distance, index == own), axis=1)
        index = np.take_along_axis(index, order, axis=1)[:, :k]
        distance = np.take_along_axis(distance, order, axis=1)[:, :k]
        valid = (index != own) & (index < n) & np.isfinite(distance)
        # Valid candidates first (stable), the padding after them.
        compact = np.argsort(~valid, axis=1, kind="stable")
        index = np.take_along_axis(index, compact, axis=1)
        distance = np.take_along_axis(distance, compact, axis=1)
        valid = np.take_along_axis(valid, compact, axis=1)
        width = index.shape[1]
        out_index[:, :width] = np.where(valid, index, own)
        out_distance[:, :width] = np.where(valid, distance, 0.0).astype(np.float32)
        out_mask[:, :width] = valid
    return ReferenceNeighborhood(
        torch.from_numpy(out_index),
        torch.from_numpy(out_distance),
        torch.from_numpy(out_mask),
    )


# ``LocalNeighborhoodData`` (PyG batching with per-graph offsets of ``knn_index``)
# ships with the library model and is re-exported above for the datapipes.


def with_self_slot(neighborhood: ReferenceNeighborhood):
    """Return ``[N, k+1]`` index/mask with the node itself in slot zero."""
    own = torch.arange(neighborhood.index.shape[0], dtype=torch.long)[:, None]
    index = torch.cat((own, neighborhood.index.to(torch.long)), dim=1)
    mask = torch.cat((torch.ones_like(own, dtype=torch.bool), neighborhood.mask), 1)
    return index, mask


def knn_edge_features(pos: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Neighbor-minus-query displacement and distance, ``[N, K, 4]``.

    Matches the structural edge convention (source minus destination, where the
    query node receives messages). Slot zero (the node itself) is all zeros
    before standardization. ``pos`` is the same normalized initial geometry that
    structural edge features use.
    """
    pos = torch.as_tensor(pos, dtype=torch.float32)
    if pos.ndim != 2 or pos.shape[1] != 3 or index.ndim != 2:
        raise ValueError("pos must be [N,3] and index [N,K]")
    displacement = pos[index] - pos[:, None, :]
    distance = torch.linalg.norm(displacement, dim=-1, keepdim=True)
    return torch.cat((displacement, distance), dim=-1)


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


class LocalNeighborhoodCache:
    """Content-addressed, atomic CPU cache for immutable reference neighborhoods.

    The key covers the physical initial coordinates, the structural edge list
    (geodesic only), ``k``, the metric, the cache schema and the source hash of
    both neighborhood queries (this module and the library's geodesic module). Entries are checksummed and validated on load; corrupt entries
    fail closed. Process-local hits return the same tensors, and distributed
    startup warms the cache once on rank zero.
    """

    def __init__(self, directory=None, *, logger=None, timeout_seconds=3600):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Cache lock timeout must be finite and positive")
        self.directory = None if directory is None else Path(directory)
        self.logger = logger
        self.timeout_seconds = timeout_seconds
        self.memory = {}
        self.counts = dict(memory_hits=0, disk_hits=0, computed=0)
        source = Path(inspect.getfile(_geodesic_knn_module)).read_bytes()
        source += Path(__file__).read_bytes()
        self.algorithm = hashlib.sha256(source).hexdigest()

    def get(self, reference_positions, edge_index, k, metric) -> ReferenceNeighborhood:
        """Load or compute the ``k`` nearest reference neighbors per node."""
        if metric not in KNN_METRICS:
            raise ValueError(f"metric must be one of {KNN_METRICS}; got {metric!r}")
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        positions = reference_positions.detach().cpu().contiguous().to(torch.float32)
        edges = edge_index.detach().cpu().contiguous().to(torch.long)
        inputs = [
            {
                "name": "reference_positions",
                "shape": list(positions.shape),
                "sha256": _digest(positions),
            }
        ]
        if metric == "geodesic":
            inputs.append(
                {
                    "name": "edge_index",
                    "shape": list(edges.shape),
                    "sha256": _digest(edges),
                }
            )
        metadata = {
            "schema": 1,
            "algorithm": self.algorithm,
            "k": k,
            "metric": metric,
            "inputs": inputs,
        }
        key = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if key in self.memory:
            self.counts["memory_hits"] += 1
            return self.memory[key]
        n = positions.shape[0]

        def compute():
            if self.logger:
                self.logger.info(
                    f"Static kNN compute start key={key} metric={metric} k={k} nodes={n}"
                )
            started = time.monotonic()
            if metric == "geodesic":
                value = geodesic_knn(positions, edges, k)
            else:
                value = euclidean_knn(positions, k)
            self.counts["computed"] += 1
            if self.logger:
                self.logger.info(
                    f"Static kNN computed in {time.monotonic() - started:.1f}s key={key}"
                )
            return value

        def valid_entry(entry):
            index, distance, mask = entry["index"], entry["distance"], entry["mask"]
            if entry["metadata"] != metadata or entry["sha256"] != _digest(index):
                return False
            if (
                index.dtype != torch.int64
                or distance.dtype != torch.float32
                or mask.dtype != torch.bool
            ):
                return False
            if (
                index.shape != (n, k)
                or distance.shape != (n, k)
                or mask.shape != (n, k)
            ):
                return False
            if index.numel() and ((index < 0).any() or (index >= n).any()):
                return False
            return bool(torch.isfinite(distance).all()) and bool((distance >= 0).all())

        if self.directory is None:
            value = compute()
        else:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{key}.pt"
            with (self.directory / f"{key}.lock").open("a+b") as lock:
                deadline = time.monotonic() + self.timeout_seconds
                while True:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"Timed out waiting for kNN cache {key}")
                        time.sleep(0.1)
                if path.exists():
                    try:
                        entry = torch.load(path, map_location="cpu", weights_only=True)
                        if not valid_entry(entry):
                            raise ValueError(
                                "metadata, checksum or tensor validation failed"
                            )
                        value = ReferenceNeighborhood(
                            entry["index"], entry["distance"], entry["mask"]
                        )
                    except Exception as error:
                        raise RuntimeError(f"Invalid kNN cache entry {path}") from error
                    self.counts["disk_hits"] += 1
                else:
                    value = compute()
                    entry = {
                        "metadata": metadata,
                        "index": value.index,
                        "distance": value.distance,
                        "mask": value.mask,
                        "sha256": _digest(value.index),
                    }
                    fd, temporary = tempfile.mkstemp(
                        prefix=f".{key}.", dir=self.directory
                    )
                    try:
                        with os.fdopen(fd, "wb") as stream:
                            torch.save(entry, stream)
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(temporary, path)
                    finally:
                        if os.path.exists(temporary):
                            os.unlink(temporary)
        self.memory[key] = value
        if self.logger:
            self.logger.info(
                f"Static kNN ready key={key} valid={int(value.mask.sum())} counts={self.counts}"
            )
        return value
