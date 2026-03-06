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

# %%
import json
import logging
import os
from pathlib import Path

import torch
import yaml
from dataset import AirFRANSDataSet
from utilities import disable_autotune_printing

from physicsnemo.experimental.models.globe.model import GLOBE
from physicsnemo.utils.logging import PythonLogger

disable_autotune_printing()
torch._logging.set_logs(graph_breaks=True, recompiles=True)
logging.basicConfig(level=logging.INFO)
logger = PythonLogger("globe.airfrans.inference")

# %%
# Resolve output directory: GLOBE_OUTPUT_DIR env var, or the most-recently-modified
# subdirectory under output/.
_output_root = Path(__file__).parent / "output"
if _env_dir := os.environ.get("GLOBE_OUTPUT_DIR"):
    output_dir = Path(_env_dir)
else:
    output_subdirs = [d for d in _output_root.iterdir() if d.is_dir()]
    if not output_subdirs:
        raise FileNotFoundError(f"No output directories found in {_output_root}")
    output_dir = max(output_subdirs, key=lambda d: d.stat().st_mtime)
logger.info(f"Using output directory: {output_dir}")

if not (_data_env := os.environ.get("AIRFRANS_DATA_DIR")):
    raise ValueError(
        "AIRFRANS_DATA_DIR environment variable is not set. "
        "Pass it explicitly or use run.sh which sets it automatically."
    )
data_dir = Path(_data_env)
manifest = json.loads((data_dir / "manifest.json").read_text())
sample_paths: dict[str, list[Path]] = {
    "train": [data_dir / f for f in manifest["full_train"]],
    "test": [data_dir / f for f in manifest["full_test"]],
}
sample_path = sample_paths["test"][0]

# %%
device = torch.device("cuda")
torch.set_float32_matmul_precision("high")

### [Datasets with cached preprocessing]
sample = AirFRANSDataSet.preprocess(sample_path).to(device)

### [Model]
hyperparameters = yaml.safe_load((output_dir / "hyperparameters.yaml").read_text())
model = GLOBE(
    **hyperparameters["model"],
).to(device)

best_model_path = output_dir / "best_model.mdlus"
if not best_model_path.exists():
    raise RuntimeError(f"No best model found at {best_model_path}")
logger.info(f"Loading best model from {best_model_path.name!r}...")
model.load(best_model_path)

# model = torch.compile(model)

# %%
with torch.no_grad():
    model.eval()
    pred_mesh = model(**sample.model_input_kwargs, chunk_size=128)

# %%
AirFRANSDataSet.postprocess(
    pred_mesh=pred_mesh.to(device="cpu"),
    true_mesh=sample.interior_mesh.to(device="cpu"),
)
