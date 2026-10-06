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

"""Shared fixtures with a small synthetic prepared dataset."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

RECIPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RECIPE))

from basis import fit_transforms  # noqa: E402
from prepare_data import make_splits  # noqa: E402

GRID = 5
GAS_CONSTANT = 287.0


def grid_quads(size=GRID):
    """Return the quadrilaterals of a structured ``size`` by ``size`` grid."""
    return np.array(
        [
            [
                row * size + column,
                row * size + column + 1,
                (row + 1) * size + column + 1,
                (row + 1) * size + column,
            ]
            for row in range(size - 1)
            for column in range(size - 1)
        ],
        dtype=np.int32,
    )


def synthetic_case(generator, labeled=True):
    """Return one case whose fields depend smoothly on its inputs."""
    u, v = (values.ravel() for values in np.meshgrid(*2 * [np.linspace(0, 1, GRID)]))
    shape = generator.normal(size=4) * 0.05
    conditions = np.array(
        [800 + 50 * generator.random(), 1e5 + 1e4 * generator.random()]
    )
    height = shape[0] * np.sin(np.pi * u) + shape[1] * v**2
    points = np.stack((u + shape[2] * v, v + shape[3] * u**2, height), axis=-1)
    normals = np.stack(
        (-shape[0] * np.pi * np.cos(np.pi * u), -2 * shape[1] * v, np.ones_like(u)),
        axis=-1,
    )
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    arrays = {
        "points": points.astype(np.float32),
        "normals": normals.astype(np.float32),
        "quads": grid_quads(),
        "conditions": conditions.astype(np.float32),
    }
    if labeled:
        front = np.tanh(10 * (u - 0.5 - shape[0]))
        pressure = conditions[1] * (1 + 0.2 * height + 0.05 * front)
        temperature = 288 * (1 + 0.01 * conditions[0] / 800 + 0.02 * points[:, 0])
        density = pressure / (GAS_CONSTANT * temperature)
        arrays["fields"] = np.stack((density, pressure, temperature), axis=-1).astype(
            np.float32
        )
        arrays["globals"] = np.array(
            [20 + shape[0], 1.5 + conditions[0] / 1e4, 0.9 - shape[1]], dtype=np.float32
        )
    return arrays


def write_prepared(directory, cases=16, official_test=2, seed=0, geometry_rank=2):
    """Write a prepared dataset with fitted statistics and bases."""
    generator = np.random.default_rng(seed)
    official = {
        "train_1000": list(range(cases)),
        "test": list(range(cases, cases + official_test)),
    }
    splits = make_splits(
        official, seed=42, validation_fraction=0.125, test_fraction=0.125
    )
    (directory / "samples").mkdir(parents=True)
    samples = {}
    for sample_id in range(cases + official_test):
        arrays = synthetic_case(generator, labeled=sample_id < cases)
        path = f"samples/{sample_id:06d}.npz"
        np.savez(directory / path, **arrays)
        samples[str(sample_id)] = {"path": path, "labeled": sample_id < cases}
    manifest = {
        "source": {"repository": "synthetic", "license": "none"},
        "split_seed": 42,
        "splits": splits,
        "samples": samples,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    fit_transforms(directory, geometry_rank=geometry_rank)
    return directory


@pytest.fixture(scope="session")
def prepared(tmp_path_factory):
    """Provide a read-only prepared dataset shared across tests."""
    return write_prepared(tmp_path_factory.mktemp("rotor37") / "processed")


def run_script(script, overrides, cwd):
    """Run a recipe entry point on the CPU and return its completed process."""
    result = subprocess.run(  # noqa: S603 fixed interpreter, recipe script and overrides
        [sys.executable, str(RECIPE / script), *overrides],
        cwd=cwd,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=600,
    )
    return result


def small_overrides(prepared, output_dir, model="geotransolver", epochs=2):
    """Return Hydra overrides for a small model trained on the fixture."""
    with np.load(prepared / "basis.npz") as basis:
        outputs = 3 + len(basis["pressure_modes"]) + len(basis["temperature_modes"])
    networks = {
        "geotransolver": [
            "model.functional_dim=6",
            "model.global_dim=6",
            f"model.out_dim={outputs}",
            "model.n_layers=1",
            "model.n_hidden=16",
            "model.n_head=2",
            "model.slice_num=4",
        ],
        "mlp": [
            "model.in_features=6",
            f"model.out_features={outputs}",
            "model.layer_size=16",
            "model.num_layers=2",
        ],
    }
    return [
        f"model={model}",
        f"output_dir={output_dir}",
        f"data.data_dir={prepared}",
        "data.geometry_points=8",
        *networks[model],
        f"training.num_epochs={epochs}",
        "training.batch_size=4",
        "training.validation_interval=1",
        "training.checkpoint_interval=1",
        "num_threads=1",
    ]
