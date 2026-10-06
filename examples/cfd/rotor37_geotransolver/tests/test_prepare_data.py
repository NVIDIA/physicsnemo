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

"""Check source decoding, splitting and training-only preparation."""

import json
import pickle

import numpy as np
import prepare_data
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from basis import fit_transforms
from conftest import synthetic_case
from prepare_data import (
    CONDITION_NAMES,
    FIELD_NAMES,
    GLOBAL_NAMES,
    NORMAL_NAMES,
    decode_sample,
    make_splits,
)


def source_payload(arrays):
    """Encode arrays in the CGNS tree layout of the source dataset."""

    def node(name, data=None, children=()):
        return [name, data, list(children), "DataArray_t"]

    point_data = [node("GridLocation", np.frombuffer(b"Vertex", dtype="S1"))]
    point_data += [
        node(name, arrays["normals"][:, axis].astype(np.float64))
        for axis, name in enumerate(NORMAL_NAMES)
    ]
    scalars = dict(zip(CONDITION_NAMES, map(float, arrays["conditions"])))
    if "fields" in arrays:
        point_data += [
            node(name, arrays["fields"][:, channel].astype(np.float64))
            for channel, name in enumerate(FIELD_NAMES)
        ]
        scalars.update(zip(GLOBAL_NAMES, map(float, arrays["globals"])))
    coordinates = node(
        "GridCoordinates",
        children=[
            node(f"Coordinate{axis}", arrays["points"][:, index].astype(np.float64))
            for index, axis in enumerate("XYZ")
        ],
    )
    elements = node(
        "Elements_QUAD_4",
        np.array([7, 0]),
        [node("ElementConnectivity", arrays["quads"].reshape(-1) + 1)],
    )
    zone = node(
        "Zone", children=[coordinates, elements, node("PointData", children=point_data)]
    )
    tree = node("CGNSTree", children=[node("Base_2_3", children=[zone])])
    return pickle.dumps({"meshes": {0.0: tree}, "scalars": scalars})


@pytest.fixture
def case():
    """Return one synthetic labeled case."""
    return synthetic_case(np.random.default_rng(3))


def test_decoding_preserves_vertices_and_connectivity(case):
    """Decoded arrays keep the source vertex order and zero-based quads."""
    decoded = decode_sample(source_payload(case), labeled=True)
    for name in ("points", "normals", "fields", "conditions", "globals"):
        np.testing.assert_array_equal(decoded[name], case[name])
    np.testing.assert_array_equal(decoded["quads"], case["quads"])


def test_splits_are_reproducible_and_disjoint():
    """Splitting the labeled cases depends only on the seed."""
    official = {"train_1000": list(range(1000)), "test": list(range(1000, 1200))}
    splits = make_splits(official)
    assert splits == make_splits(official)
    assert splits != make_splits(official, seed=43)
    sizes = {name: len(ids) for name, ids in splits.items()}
    assert sizes == {"train": 800, "validation": 100, "test": 100, "official_test": 200}
    labeled = set().union(*(splits[name] for name in ("train", "validation", "test")))
    assert labeled == set(official["train_1000"])
    with pytest.raises(ValueError, match="distinct"):
        make_splits({"train_1000": [0, 1, 2], "test": [2]})


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    """Provide a downloaded source dataset with distinct held-out outliers."""
    official = {"train_1000": list(range(10)), "test": [10, 11]}
    splits = make_splits(official)
    generator = np.random.default_rng(5)
    cases = []
    for sample_id in range(12):
        arrays = synthetic_case(generator, labeled=sample_id < 10)
        if sample_id not in splits["train"] and sample_id < 10:
            arrays["fields"] *= np.array([1.0, 50.0, 50.0], dtype=np.float32)
        cases.append(arrays)
    raw = tmp_path / "raw"
    (raw / "data").mkdir(parents=True)
    pq.write_table(
        pa.table({"sample": [source_payload(arrays) for arrays in cases]}),
        raw / "data" / "all_samples-00000.parquet",
    )
    card = {"dataset_info": {"description": {"split": official}}}
    (raw / "README.md").write_text(f"---\n{json.dumps(card)}\n---\n")
    monkeypatch.setattr(
        prepare_data,
        "fit_transforms",
        lambda path: fit_transforms(path, geometry_rank=2),
    )
    return tmp_path, splits, cases


def test_preparation_uses_only_training_cases(snapshot):
    """Statistics and bases ignore validation and test targets."""
    root, splits, cases = snapshot
    output = prepare_data.prepare(root)
    stats = json.loads((output / "stats.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    training = np.concatenate([cases[i]["fields"] for i in splits["train"]]).astype(
        np.float64
    )
    np.testing.assert_allclose(stats["fields"]["mean"], training.mean(axis=0))
    np.testing.assert_allclose(stats["fields"]["std"], training.std(axis=0))
    assert stats["sample_ids"] == splits["train"]
    assert manifest["source"]["repository"] == prepare_data.DATASET_ID
    assert manifest["splits"] == splits
    with np.load(output / "basis.npz") as basis:
        assert basis["training_sample_ids"].tolist() == splits["train"]
    with np.load(output / "samples" / "000011.npz") as sample:
        assert "fields" not in sample.files
