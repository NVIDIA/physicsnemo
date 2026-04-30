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
Dataset factory functions for external aerodynamics mesh pipelines.

Builds MeshDataset instances from Hydra-instantiable YAML configs.
Each config's ``pipeline:`` block declares a ``reader:`` and ``transforms:``
list with ``_target_: ${dp:ComponentName}`` entries, instantiated via
``hydra.utils.instantiate()``.

The main builder (``build_surface_dataset``) is fully generic and works
for both surface and volume mesh configs -- the distinction is purely in
the YAML transform chain.  ``build_dataset`` is provided as a
mesh-type-agnostic alias.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import Sampler
import hydra
from omegaconf import DictConfig, OmegaConf

import physicsnemo.datapipes  # noqa: F401  (registers ${dp:...} resolvers)
from physicsnemo.datapipes import MeshDataset, MultiDataset
from physicsnemo.mesh import DomainMesh, Mesh

import nondim  # noqa: F401  (registers NonDimensionalizeByMetadata)
import sdf  # noqa: F401  (registers ComputeSDFFromBoundary, DropBoundary)


def load_dataset_config(yaml_path: str | Path) -> DictConfig:
    """Load a dataset YAML config and return an OmegaConf DictConfig.

    The returned config is merged with ``dataset_paths.yaml`` (looked up in
    the same directory as *yaml_path*, then one level up) so that dataset
    YAMLs can use ``${dataset_paths.<key>}`` interpolation for root paths.
    """
    yaml_path = Path(yaml_path)
    paths_file = yaml_path.parent / "dataset_paths.yaml"
    paths = OmegaConf.load(paths_file) if paths_file.exists() else OmegaConf.create()
    cfg = OmegaConf.load(yaml_path)
    return OmegaConf.merge({"dataset_paths": paths}, cfg)


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
    """Create a callable that injects dataset metadata into ``global_data``.

    Handles both :class:`Mesh` (single-mesh) and :class:`DomainMesh`
    (domain with interior + boundaries).  For ``DomainMesh``, metadata
    is merged into the domain-level ``global_data`` so that
    domain-aware transforms like ``NonDimensionalizeByMetadata`` can
    read freestream quantities from ``domain.global_data``.

    The returned callable is prepended to the transform list so that
    downstream transforms can read freestream quantities from
    ``global_data``.
    """
    fields: dict[str, torch.Tensor] = {}
    for k, v in metadata.items():
        if isinstance(v, torch.Tensor):
            fields[k] = v.float()
        elif isinstance(v, (list, tuple)):
            fields[k] = torch.tensor(v, dtype=torch.float32)
        else:
            fields[k] = torch.tensor(v, dtype=torch.float32)

    def inject(data):
        if isinstance(data, DomainMesh):
            new_gd = data.global_data.clone()
            device = data.interior.points.device
            dtype = data.interior.points.dtype
            for k, v in fields.items():
                new_gd[k] = v.to(device=device, dtype=dtype)
            return DomainMesh(
                interior=data.interior,
                boundaries=data.boundaries,
                global_data=new_gd,
            )
        # Single Mesh path
        new_gd = data.global_data.clone()
        for k, v in fields.items():
            new_gd[k] = v.to(device=data.points.device, dtype=data.points.dtype)
        return Mesh(
            points=data.points,
            cells=data.cells,
            point_data=data.point_data,
            cell_data=data.cell_data,
            global_data=new_gd,
        )

    # MeshDataset calls t.apply_to_domain(data) for DomainMesh inputs,
    # so the plain function needs this attribute.
    inject.apply_to_domain = inject
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


# Mesh-type-agnostic alias -- build_surface_dataset is fully generic.
build_dataset = build_surface_dataset


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


# ---------------------------------------------------------------------------
# Manifest-based split support
# ---------------------------------------------------------------------------


def load_manifest(path: str | Path, *, split: str | None = None) -> list[str]:
    """Load a split manifest file.

    Supports three formats:

    - **JSON dict** (with *split*): a dict of ``{split_name: [paths, ...]}``.
      The *split* key selects which list to return.  This is the format
      used by ``PhysicsNeMo-HighLiftAeroML/manifest.json``.
    - **JSON list** (without *split*): a flat list of sub-path strings.
    - **Text** (without *split*): one sub-path per line (blank lines and
      ``#`` comments are stripped).

    Parameters
    ----------
    path : str or Path
        Path to the manifest file.
    split : str, optional
        Key to extract from a JSON dict manifest (e.g.
        ``"single_aoa_4_train"``).  Required when the manifest is a dict,
        ignored for flat list / text manifests.

    Returns
    -------
    list[str]
        Sorted list of sub-path strings.

    Raises
    ------
    KeyError
        If *split* is given but not found in the manifest dict.
    ValueError
        If the manifest format doesn't match expectations.
    """
    p = Path(path)
    text = p.read_text()
    # Try JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if split is None:
                raise ValueError(
                    f"Manifest {p.name} is a JSON dict; "
                    f"a 'split' key is required. "
                    f"Available keys: {list(data.keys())[:10]}"
                )
            if split not in data:
                raise KeyError(
                    f"Split {split!r} not found in manifest. "
                    f"Available: {list(data.keys())}"
                )
            entries = data[split]
        elif isinstance(data, list):
            entries = data
        else:
            raise ValueError(
                f"Manifest JSON must be a list or dict, got {type(data).__name__}"
            )
        return sorted(str(e) for e in entries)
    except json.JSONDecodeError:
        pass
    # Fall back to one-per-line text
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.append(line)
    return sorted(entries)


def resolve_manifest_indices(
    reader,
    manifest_entries: list[str],
) -> list[int]:
    """Map manifest sub-paths to reader sample indices.

    Each manifest entry is matched against the reader's discovered paths.
    A reader path matches if any of its parent directories (relative to
    the reader root) equals the manifest entry.

    Parameters
    ----------
    reader : MeshReader or DomainMeshReader
        An instantiated reader with ``_root`` and ``_paths`` attributes.
    manifest_entries : list[str]
        Sub-path strings from the manifest (e.g. ``["run_1", "run_5"]``).

    Returns
    -------
    list[int]
        Sorted list of reader indices whose paths match the manifest.

    Raises
    ------
    ValueError
        If no reader paths match any manifest entry.
    """
    entry_set = set(manifest_entries)
    indices = []
    for idx, full_path in enumerate(reader._paths):
        try:
            rel = full_path.relative_to(reader._root)
        except ValueError:
            continue
        # Check if any parent component matches a manifest entry
        # e.g. rel = "run_1/domain_1.pmsh" -> parts = ("run_1", "domain_1.pmsh")
        for part in rel.parts[:-1]:
            if part in entry_set:
                indices.append(idx)
                break
        else:
            # Also check if the immediate parent dir name matches
            if rel.parent.name in entry_set:
                indices.append(idx)
    if not indices:
        raise ValueError(
            f"No reader paths matched manifest entries. "
            f"Reader root: {reader._root}, "
            f"sample entries: {list(entry_set)[:5]}"
        )
    return sorted(indices)


class ManifestSampler(Sampler[int]):
    """Sampler that restricts iteration to a subset of dataset indices.

    Supports shuffling with epoch-aware seeding and distributed sharding.

    Parameters
    ----------
    indices : list[int]
        Dataset indices that belong to this split.
    shuffle : bool
        Whether to shuffle indices each epoch.
    seed : int
        Base random seed for reproducible shuffling.
    rank : int
        Current process rank (for distributed sharding). 0 for single-GPU.
    world_size : int
        Total number of processes. 1 for single-GPU.
    drop_last : bool
        If True, drop tail indices so every rank gets the same count.
    """

    def __init__(
        self,
        indices: list[int],
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
    ) -> None:
        self._indices = list(indices)
        self._shuffle = shuffle
        self._seed = seed
        self._rank = rank
        self._world_size = world_size
        self._drop_last = drop_last
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling."""
        self._epoch = epoch

    def __len__(self) -> int:
        n = len(self._indices)
        if self._world_size > 1:
            if self._drop_last:
                n = n // self._world_size
            else:
                n = math.ceil(n / self._world_size)
        return n

    def __iter__(self) -> Iterator[int]:
        indices = list(self._indices)
        if self._shuffle:
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in perm]

        if self._world_size > 1:
            if self._drop_last:
                # Truncate so every rank gets the same count
                n_keep = (len(indices) // self._world_size) * self._world_size
                indices = indices[:n_keep]
            else:
                # Pad to make evenly divisible
                padding = math.ceil(
                    len(indices) / self._world_size
                ) * self._world_size - len(indices)
                indices += indices[:padding]
            # Shard
            indices = indices[self._rank :: self._world_size]

        return iter(indices)
