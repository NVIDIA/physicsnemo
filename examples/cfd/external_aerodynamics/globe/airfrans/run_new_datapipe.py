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
AirFRANS Datapipe - Hydra entrypoint.

Builds the full AirFRANS data pipeline from YAML configuration using
``hydra.utils.instantiate``. Supports two interchangeable readers:

- **arrow** (default): Reads from the PLAID/HuggingFace Arrow dataset.
- **vtk**: Reads from VTU/VTP mesh files on disk.

Both readers produce an identical TensorDict schema so the same shared
transform pipeline works with either.

Usage
-----
    # Arrow reader (default)
    python run.py

    # VTK reader
    python run.py reader=vtk data_dir=/path/to/vtk/samples

    # Override task/split
    python run.py task=reynolds split=test
"""

from __future__ import annotations

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from physicsnemo.datapipes import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="./conf",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    logger.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg))

    logger.info("Building AirFRANS dataset...")
    dataset: Dataset = hydra.utils.instantiate(cfg.dataset)

    logger.info("Dataset has %d samples", len(dataset))
    logger.info("Field names: %s", dataset.field_names)

    logger.info("Loading first sample...")
    data, metadata = dataset[0]

    logger.info("Metadata: %s", metadata)
    logger.info("Data keys and shapes:")
    for key in sorted(data.keys()):
        tensor = data[key]
        finite = tensor[~torch.isnan(tensor)]
        lo = finite.min().item() if finite.numel() > 0 else float("nan")
        hi = finite.max().item() if finite.numel() > 0 else float("nan")
        logger.info(
            "  %-25s shape=%-20s dtype=%s  range=[%.4f, %.4f]",
            key,
            str(tuple(tensor.shape)),
            tensor.dtype,
            lo,
            hi,
        )

    dataset.close()


if __name__ == "__main__":
    main()
