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

import nondim  # noqa: F401  (registers InjectMetadata, NonDimensionalizeByMetadata)


def load_dataset_config(yaml_path: str | Path) -> DictConfig:
    """Load a dataset YAML config and return an OmegaConf DictConfig."""
    return OmegaConf.load(yaml_path)


_PATH_KEYS = {"stats_parquet", "stats_file"}
_INJECT_METADATA_SUFFIX = "InjectMetadata"


def _resolve_transform_paths(t_cfg: DictConfig, base_dir: Path) -> DictConfig:
    """Resolve relative file paths in a transform config against *base_dir*.

    Transforms like ``NormalizeMeshFields`` accept ``stats_parquet`` or
    ``stats_file`` parameters that may be relative.  When Hydra changes
    the working directory these would break, so we resolve them to
    absolute paths before instantiation.
    """
    for key in _PATH_KEYS:
        val = OmegaConf.select(t_cfg, key, default=None)
        if val is not None and not Path(val).is_absolute():
            resolved = base_dir / val
            if resolved.exists():
                t_cfg = OmegaConf.merge(t_cfg, {key: str(resolved)})
    return t_cfg


def _inject_metadata_into_transform(
    t_cfg: DictConfig,
    metadata: dict,
) -> DictConfig:
    """If *t_cfg* targets ``InjectMetadata`` and has no explicit ``metadata``
    parameter, merge in the dataset-level metadata from the YAML config."""
    target = OmegaConf.select(t_cfg, "_target_", default="")
    if target.endswith(_INJECT_METADATA_SUFFIX):
        if OmegaConf.select(t_cfg, "metadata", default=None) is None:
            t_cfg = OmegaConf.merge(t_cfg, {"metadata": metadata})
    return t_cfg


def build_surface_dataset(cfg: DictConfig, base_dir: Path | None = None) -> MeshDataset:
    """Build a single MeshDataset from a Hydra-style pipeline config.

    Parameters
    ----------
    cfg : DictConfig
        Dataset config with a ``pipeline:`` block containing ``reader:``
        and ``transforms:`` entries.  An optional top-level ``metadata:``
        block is automatically injected into any ``InjectMetadata``
        transform that does not already specify its own ``metadata``
        parameter.
    base_dir : Path, optional
        Root directory for resolving relative paths in transform configs
        (e.g. ``stats_parquet``).  Defaults to the recipe root
        (two levels above this file).

    Returns
    -------
    MeshDataset
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    metadata = OmegaConf.to_container(
        OmegaConf.select(cfg, "metadata", default=OmegaConf.create({})),
        resolve=True,
    )

    reader = hydra.utils.instantiate(cfg.pipeline.reader)
    transforms = None
    if "transforms" in cfg.pipeline and cfg.pipeline.transforms:
        resolved = []
        for t in cfg.pipeline.transforms:
            t = _resolve_transform_paths(t, base_dir)
            if metadata:
                t = _inject_metadata_into_transform(t, metadata)
            resolved.append(hydra.utils.instantiate(t))
        transforms = resolved
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
