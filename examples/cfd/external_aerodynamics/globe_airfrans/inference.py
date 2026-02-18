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
from pathlib import Path

import torch
import yaml

from config import get_data_dir
from dataset import AirFRANSDataSet
from utilities import (
    get_latest_checkpoint_path,
    to,
    disable_autotune_printing,
)
from physicsnemo.models.globe.model import GLOBE

disable_autotune_printing()  # Silences the verbose output of `torch.compile(..., mode="max-autotune")`.
torch._logging.set_logs(graph_breaks=True, recompiles=True)

# %%
output_dir = (
    Path(__file__).parent / "output" / "run_35_full"
)
data_dir = get_data_dir()
manifest = json.loads((data_dir / "manifest.json").read_text())
train_sample_paths = [data_dir / f for f in manifest["full_train"]]
valid_sample_paths = [data_dir / f for f in manifest["full_test"]]
sample_path = valid_sample_paths[0]

# %%
device = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(0.99)
torch.set_float32_matmul_precision("high")

### [Datasets with cached preprocessing]
input_dict, _ = AirFRANSDataSet.preprocess(sample_path)
input_dict = to(input_dict, device=device)

### [Model]
hyperparameters = yaml.safe_load((output_dir / "hyperparameters.yaml").read_text())
model = GLOBE(  # ty: ignore[missing-argument]
    **hyperparameters["model"],
).to(device)

previous_checkpoint: Path | None = get_latest_checkpoint_path(output_dir=output_dir)
if previous_checkpoint:
    print(f"Loading checkpoint {previous_checkpoint.name!r}...")
    model.load_state_dict(
        torch.load(previous_checkpoint, map_location=device)["model_state_dict"]
    )
else:
    raise RuntimeError("No checkpoint found in output directory!")

# model = torch.compile(model)

# %%
with torch.no_grad():
    model.eval()
    pred_results = model(
        prediction_points=input_dict["prediction_points"],
        boundary_meshes=input_dict["boundary_meshes"],
        reference_lengths=input_dict["reference_lengths"],
        global_scalars=input_dict["global_scalars"],
        global_vectors=input_dict["global_vectors"],
        chunk_size=128,
        verbose=False,
    )

# %%
AirFRANSDataSet.postprocess(
    to(pred_results, device=torch.device("cpu"), dtype=torch.float64),
    sample_path,
    fields_to_plot="true",
)
