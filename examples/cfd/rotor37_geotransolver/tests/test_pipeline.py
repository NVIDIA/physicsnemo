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

"""Train and evaluate a small model through the recipe entry points."""

import csv
import json

import numpy as np
import pytest
from conftest import GAS_CONSTANT, run_script, small_overrides


@pytest.mark.parametrize("model", ["geotransolver", "mlp"])
def test_training_and_evaluation(prepared, tmp_path, model):
    """A run trains, saves its final checkpoint and scores every validation case."""
    run = tmp_path / "run"
    manifest = json.loads((prepared / "manifest.json").read_text())
    validation = manifest["splits"]["validation"]
    overrides = small_overrides(prepared, run, model)
    for script, extra in (
        ("train.py", []),
        ("evaluate.py", [f"evaluation.export_samples=[{validation[0]}]"]),
    ):
        result = run_script(script, overrides + extra, tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
    assert len(list((run / "checkpoints").glob("*.0.2.mdlus"))) == 1
    output = run / "evaluation" / "validation"
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["epoch"] == 2 and metrics["cases"] == len(validation)
    with (output / "cases.csv").open() as stream:
        assert [int(row["sample_id"]) for row in csv.DictReader(stream)] == validation
    with np.load(output / "predictions" / f"sample_{validation[0]:06d}.npz") as export:
        density, pressure, temperature = export["field_prediction"].T.astype(float)
        np.testing.assert_allclose(
            pressure, GAS_CONSTANT * density * temperature, rtol=1e-6
        )
