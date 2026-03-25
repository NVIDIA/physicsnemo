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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

import physicsnemo.datapipes  # noqa: F401  (registers ${dp:...} resolvers)
from physicsnemo.datapipes import MeshDataset, MultiDataset
from physicsnemo.mesh import Mesh

import nondim  # noqa: F401  (registers NonDimensionalizeByMetadata)


def load_dataset_config(yaml_path: str | Path) -> DictConfig:
    """Load a dataset YAML config and return an OmegaConf DictConfig."""
    return OmegaConf.load(yaml_path)


_PATH_KEYS = {"stats_file"}
_CENTER_MESH_SUFFIX = "CenterMesh"


def _resolve_transform_paths(t_cfg: DictConfig, base_dir: Path) -> DictConfig:
    """Resolve relative file paths in a transform config against *base_dir*.

    Transforms like ``NormalizeMeshFields`` accept ``stats_file``
    parameters that may be relative.  When Hydra changes the working
    directory these would break, so we resolve them to absolute paths
    before instantiation.
    """
    for key in _PATH_KEYS:
        val = OmegaConf.select(t_cfg, key, default=None)
        if val is not None and not Path(val).is_absolute():
            resolved = base_dir / val
            if resolved.exists():
                t_cfg = OmegaConf.merge(t_cfg, {key: str(resolved)})
    return t_cfg


def _make_metadata_injector(metadata: dict):
    """Create a callable that injects dataset metadata into ``mesh.global_data``.

    This replaces the former ``InjectMetadata`` transform class.  The
    returned callable is prepended to the transform list so that
    downstream transforms like ``NonDimensionalizeByMetadata`` can read
    freestream quantities from ``global_data``.
    """
    fields: dict[str, torch.Tensor] = {}
    for k, v in metadata.items():
        if isinstance(v, torch.Tensor):
            fields[k] = v.float()
        elif isinstance(v, (list, tuple)):
            fields[k] = torch.tensor(v, dtype=torch.float32)
        else:
            fields[k] = torch.tensor(v, dtype=torch.float32)

    def inject(mesh: Mesh) -> Mesh:
        new_gd = mesh.global_data.clone()
        for k, v in fields.items():
            new_gd[k] = v.to(device=mesh.points.device, dtype=mesh.points.dtype)
        return Mesh(
            points=mesh.points,
            cells=mesh.cells,
            point_data=mesh.point_data,
            cell_data=mesh.cell_data,
            global_data=new_gd,
        )

    return inject


def build_surface_dataset(
    cfg: DictConfig,
    base_dir: Path | None = None,
    augment: bool = False,
    device: str | torch.device | None = "auto",
    num_workers: int = 1,
    pin_memory: bool = False,
) -> MeshDataset:
    """Build a single MeshDataset from a Hydra-style pipeline config.

    Parameters
    ----------
    cfg : DictConfig
        Dataset config with a ``pipeline:`` block containing ``reader:``
        and ``transforms:`` entries.  An optional ``pipeline.augmentations``
        list defines stochastic augmentation transforms (e.g.
        ``RandomRotateMesh``, ``RandomTranslateMesh``) that are inserted
        after ``CenterMesh`` when *augment* is ``True``.  If a top-level
        ``metadata:`` block is present, its values are injected into
        ``mesh.global_data`` as the first transform step.
    base_dir : Path, optional
        Root directory for resolving relative paths in transform configs
        (e.g. ``stats_file``).  Defaults to the recipe root
        (two levels above this file).
    augment : bool, optional
        When ``True``, ``pipeline.augmentations`` transforms are inserted
        into the pipeline after ``CenterMesh``.  Should be ``False`` for
        validation / test datasets.  Default ``False``.
    device : str or torch.device, optional
        Device to transfer mesh data to before transforms.  When ``None``,
        data stays on CPU.
    num_workers : int, default=1
        Number of worker threads for the MeshDataset prefetch pool.
    pin_memory : bool, default=False
        If True, the reader places tensors in pinned (page-locked) memory
        for faster async CPU-to-GPU transfers.

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

    reader = hydra.utils.instantiate(cfg.pipeline.reader, pin_memory=pin_memory)
    resolved = []

    # Inject dataset metadata into global_data as the first transform
    if metadata:
        resolved.append(_make_metadata_injector(metadata))

    if "transforms" in cfg.pipeline and cfg.pipeline.transforms:
        for t in cfg.pipeline.transforms:
            t = _resolve_transform_paths(t, base_dir)
            resolved.append(hydra.utils.instantiate(t))

        if augment and "augmentations" in cfg.pipeline and cfg.pipeline.augmentations:
            aug = [hydra.utils.instantiate(a) for a in cfg.pipeline.augmentations]
            # +1 for the metadata injector prepended above
            offset = 1 if metadata else 0
            insert_idx = next(
                (
                    offset + i + 1
                    for i, t_cfg in enumerate(cfg.pipeline.transforms)
                    if t_cfg.get("_target_", "").endswith(_CENTER_MESH_SUFFIX)
                ),
                len(resolved),
            )
            resolved[insert_idx:insert_idx] = aug

    transforms = resolved if resolved else None
    return MeshDataset(
        reader, transforms=transforms, device=device, num_workers=num_workers
    )


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
