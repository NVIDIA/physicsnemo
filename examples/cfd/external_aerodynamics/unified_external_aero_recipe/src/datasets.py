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
Dataset factory functions for external aerodynamics surface mesh pipelines.

Builds MeshDataset instances from Hydra-instantiable YAML configs.
Each config's ``pipeline:`` block declares a ``reader:`` and ``transforms:``
list with ``_target_: ${dp:ComponentName}`` entries, instantiated via
``hydra.utils.instantiate()``.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import physicsnemo.datapipes  # noqa: F401  (registers ${dp:...} resolvers)
from physicsnemo.datapipes import MeshDataset, MultiDataset


def load_dataset_config(yaml_path: str | Path) -> DictConfig:
    """Load a dataset YAML config and return an OmegaConf DictConfig."""
    return OmegaConf.load(yaml_path)


def build_surface_dataset(cfg: DictConfig) -> MeshDataset:
    """Build a single MeshDataset from a Hydra-style pipeline config.

    Parameters
    ----------
    cfg : DictConfig
        Dataset config with a ``pipeline:`` block containing ``reader:``
        and ``transforms:`` entries.

    Returns
    -------
    MeshDataset
    """
    reader = hydra.utils.instantiate(cfg.pipeline.reader)
    transforms = None
    if "transforms" in cfg.pipeline and cfg.pipeline.transforms:
        transforms = [hydra.utils.instantiate(t) for t in cfg.pipeline.transforms]
    return MeshDataset(reader, transforms=transforms)


def build_multi_surface_dataset(*cfgs: DictConfig) -> MultiDataset:
    """Build a MultiDataset from multiple Hydra-style pipeline configs.

    Parameters
    ----------
    *cfgs : DictConfig
        One config per dataset, each with a ``pipeline:`` block.

    Returns
    -------
    MultiDataset
    """
    datasets = [build_surface_dataset(c) for c in cfgs]
    return MultiDataset(*datasets, output_strict=False)
