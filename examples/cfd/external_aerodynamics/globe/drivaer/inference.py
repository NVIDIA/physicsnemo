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

"""Single-sample inference for the GLOBE DrivAerML case study.

Loads a trained GLOBE model, runs inference on a test sample, computes
integrated force coefficients, and produces 3D surface visualizations.
"""

# %%
import logging
import os
from pathlib import Path

import torch
import yaml
from dataset import DrivAerMLDataSet, postprocess, visualize_comparison
from physicsnemo.experimental.utils import disable_autotune_printing

from physicsnemo.experimental.models.globe.model import GLOBE
from physicsnemo.utils.logging import PythonLogger

disable_autotune_printing()
torch._logging.set_logs(graph_breaks=True, recompiles=True)
logging.basicConfig(level=logging.INFO)
logger = PythonLogger("globe.drivaer.inference")

# %%
### Resolve output directory
_output_root = Path(__file__).parent / "output"
if _env_dir := os.environ.get("GLOBE_OUTPUT_DIR"):
    output_dir = Path(_env_dir)
else:
    output_subdirs = [d for d in _output_root.iterdir() if d.is_dir()]
    if not output_subdirs:
        raise FileNotFoundError(f"No output directories found in {_output_root}")
    output_dir = max(output_subdirs, key=lambda d: d.stat().st_mtime)
logger.info(f"Using output directory: {output_dir}")

### Resolve data directory and sample
if not (_data_env := os.environ.get("DRIVAER_DATA_DIR")):
    raise ValueError(
        "DRIVAER_DATA_DIR environment variable is not set. "
        "Pass it explicitly or use run.sh which sets it automatically."
    )
data_dir = Path(_data_env)
sample_paths: dict[str, list[Path]] = {
    split: DrivAerMLDataSet.get_split_paths(data_dir, split)
    for split in ("train", "validation")
}
sample_path = sample_paths["validation"][0]

# %%
device = torch.device("cuda")
torch.set_float32_matmul_precision("high")

### Load hyperparameters and model
hyperparameters = yaml.safe_load((output_dir / "hyperparameters.yaml").read_text())

### Load and prepare sample
sample = DrivAerMLDataSet.load_single_sample(
    sample_path,
    n_faces_per_boundary=hyperparameters.get("n_faces_per_boundary", 20_000),
    device=device,
)

model = GLOBE(**hyperparameters["model"]).to(device)

best_model_path = output_dir / "best_model.mdlus"
if not best_model_path.exists():
    raise RuntimeError(f"No best model found at {best_model_path}")
logger.info(f"Loading best model from {best_model_path.name!r}...")
model.load(best_model_path)

# %%
### Run inference
with torch.no_grad():
    model.eval()
    pred_mesh = model(**sample.model_input_kwargs)

# %%
combined = postprocess(
    pred_mesh=pred_mesh.to(device="cpu"),
    sample=sample.to(device="cpu"),
)

### Visualize predictions vs ground truth
save_path = output_dir / f"inference_{sample_path.name}.png"
visualize_comparison(combined, save_path=save_path)

### Log force coefficients
for src in ("pred", "true"):
    coeffs = combined.global_data[src].to_dict()  # ty: ignore[unresolved-attribute]
    logger.info(
        f"Force coefficients ({src}):"
        + "".join(f"\n  {k}: {coeffs[k]:.5f}" for k in ("Cd", "Cl", "Cs"))
    )
