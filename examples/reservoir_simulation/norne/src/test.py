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

"""
@Modified by: clement etienam
"""

# 🛠 Standard Library
import pickle
import logging
import gzip

# 🔥 Torch & PhyNeMo
import torch
from hydra.utils import to_absolute_path

# 📦 Local Modules
from utils.model_utils import create_fno_model


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger("Forward problem")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logging()

# ── device ────────────────────────────────────────────────────────────────────
cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {cuda}")

# ── build model ───────────────────────────────────────────────────────────────
input_keys_peacemann_wwpr = ["X"]
output_keys_peacemann_wwpr = ["Y"]

surrogate_peacemann_wwpr = create_fno_model(
    90,
    66,
    len(output_keys_peacemann_wwpr),
    cuda,
    num_fno_modes=32,
    latent_channels=64,
    decoder_layer_size=64,
    padding=54,  # was 20 — gives length 10+54=64, FFT half = 32
    num_fno_layers=5,
    decoder_layers=4,
    dimension=1,
).to(cuda)
surrogate_peacemann_wwpr.eval()
logger.info("Model built successfully")

# ── test tensor: (B=4, C=9, T=20) ────────────────────────────────────────────
B, C, T = 4, 90, 10
test_input = torch.randn(B, C, T, dtype=torch.float32).to(cuda)
logger.info(f"Input  shape : {test_input.shape}")

# ── inference ─────────────────────────────────────────────────────────────────
with torch.no_grad():
    try:
        test_output = surrogate_peacemann_wwpr(test_input)

        # handle dict or tensor output
        if isinstance(test_output, dict):
            out_tensor = test_output
        else:
            out_tensor = test_output

        logger.info(f"Output shape : {out_tensor.shape}")
        logger.info(
            f"Output min={out_tensor.min().item():.4f}  "
            f"max={out_tensor.max().item():.4f}  "
            f"mean={out_tensor.mean().item():.4f}"
        )
        logger.info("✔ Inference test PASSED")

    except Exception as e:
        logger.error(f"✘ Inference test FAILED: {e}")
        raise

with gzip.open(to_absolute_path("../data/time_train.pkl.gz"), "rb") as f1:
    time_physics = pickle.load(f1)

print(f"type:  {type(time_physics)}")

# If it's a numpy array or torch tensor:
if hasattr(time_physics, "shape"):
    print(f"shape: {time_physics.shape}")
    print(f"dtype: {time_physics.dtype}")
    print(f"size:  {time_physics.size if hasattr(time_physics, 'size') else 'N/A'}")

# If it's a dict (common for saved training data):
if isinstance(time_physics, dict):
    print(f"keys:  {list(time_physics.keys())}")
    for k, v in time_physics.items():
        print(
            f"  {k}: type={type(v).__name__}, "
            f"shape={getattr(v, 'shape', 'N/A')}, "
            f"dtype={getattr(v, 'dtype', 'N/A')}"
        )

# If it's a list:
if isinstance(time_physics, (list, tuple)):
    print(f"length: {len(time_physics)}")
    if len(time_physics) > 0:
        print(
            f"first element: type={type(time_physics[0]).__name__}, "
            f"shape={getattr(time_physics[0], 'shape', 'N/A')}"
        )
