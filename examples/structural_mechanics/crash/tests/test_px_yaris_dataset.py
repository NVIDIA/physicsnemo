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

"""The full-vehicle datapipe on tiny synthetic stores: per-design thickness."""

import csv
import os
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("torch_geometric")
pytest.importorskip("scipy")  # Euclidean neighbourhoods and one-ring exclusions
zarr = pytest.importorskip("zarr")

THIS_DIR = os.path.dirname(__file__)
CRASH_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if CRASH_DIR not in sys.path:
    sys.path.insert(0, CRASH_DIR)

from datapipes.px_yaris_dataset import YarisFullGraphDataset  # noqa: E402
from readers.px_yaris_reader import YarisFullReader  # noqa: E402

NX, NY, FRAMES = 6, 3, 8
CASES = {"train": ["case_a", "case_b"], "validation": ["case_c"], "test": ["case_d"]}


def _write_stores(work, frames_per_case=None):
    """Four stores sharing the reference mesh, each with its own thickness."""
    rng = np.random.default_rng(0)
    n = NX * NY

    def idx(i, j):
        return i * NY + j

    base = (
        np.stack(np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij"), -1)
        .reshape(n, 2)
        .astype(np.float32)
        * 10.0
    )
    reference = np.zeros((n, 3), np.float32)
    reference[:, :2] = base
    quads = np.array(
        [
            [idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
            for i in range(NX - 1)
            for j in range(NY - 1)
        ],
        dtype=np.int64,
    )
    edge_set = set()
    for quad in quads:
        for a, b in zip(quad, np.roll(quad, -1)):
            edge_set.add((int(min(a, b)), int(max(a, b))))
    edges = np.array(sorted(edge_set), dtype=np.int64)

    def put(group, key, data):
        sub, _, name = key.rpartition("/")
        target = group.require_group(sub) if sub else group
        array = target.create_array(name, shape=data.shape, dtype=data.dtype)
        array[...] = data

    data_dir = work / "stores"
    data_dir.mkdir(parents=True)
    thickness = {}
    for cases in CASES.values():
        for name in cases:
            frames = (frames_per_case or {}).get(name, FRAMES)
            group = zarr.open_group(str(data_dir / f"{name}.zarr"), mode="w")
            pos = np.repeat(reference[None], frames, axis=0).copy()
            for t in range(1, frames):  # frame 0 is the shared reference
                pos[t, :, 2] = 0.5 * t + rng.normal(scale=0.1, size=n)
                pos[t, :, :2] += rng.normal(scale=0.2, size=(n, 2))
            put(group, "nodes/pos", pos.astype(np.float32))
            values = rng.uniform(1.0, 3.0, size=n).astype(np.float32)
            thickness[name] = torch.from_numpy(values)
            put(group, "nodes/thickness", values)
            put(group, "edges", edges)
            put(group, "shell/connectivity", quads)
    manifest = work / "splits.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "sample_name"])
        for split, cases in CASES.items():
            for name in cases:
                writer.writerow([split, name])
    return data_dir, manifest, thickness


def _dataset(
    work, data_dir, manifest, split, sample_type, stats_mode, num_steps=FRAMES
):
    reader = YarisFullReader(manifest, None, stride=1, raw_dt=0.005)
    return YarisFullGraphDataset(
        reader,
        data_dir,
        num_samples=len(CASES[split]),
        num_steps=num_steps,
        split=split,
        stats_dir=work / "stats",
        stats_mode=stats_mode,
        sample_type=sample_type,
        dt=0.005,
        local_knn_k=4,
        local_knn_metric="euclidean",
        local_knn_cache_dir=work / "knn",
        stats_workers=1,
        contact_surface=True,
        rigid_wall_position=100.0,
        mesh_edges=True,
    )


def test_each_sample_carries_its_own_design_thickness(tmp_path):
    """Designs share the reference mesh but not their shell thickness."""
    data_dir, manifest, thickness = _write_stores(tmp_path)
    train = _dataset(tmp_path, data_dir, manifest, "train", "one_time_step", "compute")
    validation = _dataset(
        tmp_path, data_dir, manifest, "validation", "all_time_steps", "load"
    )
    seen = set()
    for dataset in (train, validation):
        # the shared graph never carries one design's thickness
        assert getattr(dataset.graph, "shell_thickness", None) is None
        for i in range(len(dataset)):
            sample = dataset[i]
            own = thickness[sample.yaris_case_name]
            assert torch.equal(sample.graph.shell_thickness, own), (
                sample.yaris_case_name
            )
            assert sample.graph.contact_faces is not None
            assert float(sample.graph.rigid_wall.reshape(-1)[0]) == 100.0
            seen.add(sample.yaris_case_name)
    assert seen == {"case_a", "case_b", "case_c"}
    # a second design's thickness never leaks into the first one's sample
    first = thickness["case_a"]
    assert not torch.equal(first, thickness["case_b"])


def test_too_long_validation_case_fails_at_construction(tmp_path):
    """``all_time_steps`` checks the frame counts at startup, like the train split."""
    data_dir, manifest, _ = _write_stores(
        tmp_path, frames_per_case={"case_c": FRAMES + 3}
    )
    _dataset(tmp_path, data_dir, manifest, "train", "one_time_step", "compute")
    with pytest.raises(ValueError, match="case_c"):
        _dataset(tmp_path, data_dir, manifest, "validation", "all_time_steps", "load")
