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

"""Content-addressed, atomic CPU cache for immutable reference contact masks."""

import fcntl
import hashlib
import inspect
import json
import math
import os
import tempfile
import time
from pathlib import Path

import torch

from physicsnemo.nn.functional.neighbors.reference_geodesic import (
    reference_geodesic_exclusions,
)


def _digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


class SurfaceGeodesicCache:
    """Cache by physical reference geometry, thickness, topology AND all options.

    Published entries are checksummed and validated, never silently repaired.
    Atomic replacement prevents partial reads. Advisory locks serialize writers
    and release automatically if a process exits; lock files are not deleted.
    The functional's entire source file is fingerprinted for invalidation.
    """

    def __init__(self, directory=None, *, logger=None, timeout_seconds=3600):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Cache lock timeout must be finite and positive")
        self.directory = None if directory is None else Path(directory)
        self.logger = logger
        self.timeout_seconds = timeout_seconds
        self.memory = {}
        self.counts = dict(memory_hits=0, disk_hits=0, computed=0)
        source = Path(
            inspect.getfile(inspect.unwrap(reference_geodesic_exclusions))
        ).read_bytes()
        self.algorithm = hashlib.sha256(source).hexdigest()

    def get(self, positions, faces, thickness, options):
        """Load or compute a validated exclusion mask keyed by all physical inputs."""
        inputs = [t.detach().cpu().contiguous() for t in (positions, faces, thickness)]
        metadata = {
            "schema": 1,
            "algorithm": self.algorithm,
            "options": dict(options),
            "inputs": [
                {"shape": list(t.shape), "dtype": str(t.dtype), "sha256": _digest(t)}
                for t in inputs
            ],
        }
        key = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if key in self.memory:
            self.counts["memory_hits"] += 1
            return self.memory[key]
        p, f, t = inputs

        def compute():
            if self.logger:
                self.logger.info(f"Geodesic cache compute start key={key}")
            value = reference_geodesic_exclusions(
                p, f, t * 0.5, t[f].amax(1) * 0.5, **options
            )
            self.counts["computed"] += 1
            return value

        if self.directory is None:
            value = compute()
        else:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{key}.pt"
            # Persistent lock inode: unlinking it would permit concurrent locks
            # on different inodes. No global RNG is used for locks or temp names.
            with (self.directory / f"{key}.lock").open("a+b") as lock:
                deadline = time.monotonic() + self.timeout_seconds
                while True:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"Timed out waiting for geodesic cache {key}"
                            )
                        time.sleep(0.1)
                if path.exists():
                    try:
                        entry = torch.load(path, map_location="cpu", weights_only=True)
                        value = entry["pairs"]
                        valid = (
                            entry["metadata"] == metadata
                            and value.dtype == torch.int64
                            and value.ndim == 2
                            and value.shape[0] == 2
                            and value.shape[1] <= options["max_pairs"]
                            and entry["sha256"] == _digest(value)
                            and (value >= 0).all().item()
                            and (value[0] < len(p)).all().item()
                            and (value[1] < len(f)).all().item()
                        )
                        if valid and value.shape[1] > 1:
                            order = value[0] * len(f) + value[1]
                            valid = (order[1:] > order[:-1]).all().item()
                        if not valid:
                            raise ValueError(
                                "metadata, checksum or pair validation failed"
                            )
                    except Exception as error:
                        raise RuntimeError(
                            f"Invalid geodesic cache entry {path}"
                        ) from error
                    self.counts["disk_hits"] += 1
                else:
                    value = compute()
                    entry = {
                        "metadata": metadata,
                        "pairs": value,
                        "sha256": _digest(value),
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
                f"Geodesic cache ready key={key} pairs={value.shape[1]} counts={self.counts}"
            )
        return value
