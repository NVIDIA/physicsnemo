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

The single builder ``build_dataset`` is mesh-type-agnostic: it works
identically for surface and volume mesh configs because the distinction
is entirely in the YAML transform chain (volume YAMLs already produce a
``DomainMesh`` natively via ``DomainMeshReader``; surface YAMLs append a
``MeshToDomainMesh`` terminal transform to reach the same shape).
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

### Make this folder importable by its bare module names (`nondim`, `sdf`)
### regardless of whether the caller invoked `python src/train.py` (which
### already adds `src/` to sys.path[0]) or imported `datasets` from a
### different working directory. Guarded so repeated imports are no-ops.
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch.utils.data import Sampler

import physicsnemo.datapipes  # noqa: F401  (registers ${dp:...} resolvers)
from physicsnemo.datapipes import MeshDataset
from physicsnemo.datapipes.transforms.mesh import MeshTransform, NormalizeMeshFields
from physicsnemo.mesh import DomainMesh, Mesh

import nondim  # noqa: F401  (registers NonDimensionalizeByMetadata)
import sdf  # noqa: F401  (registers ComputeSDFFromBoundary, DropBoundary)
from metrics import MetricName
from utils import FieldType

### Module-level logger used for warnings emitted from helpers below
### (e.g. ``validate_dataset_consistency``). Goes through the same Python
### logging pipeline as the recipe's ``PythonLogger``, so it shows up in
### whatever handlers the training script has configured.
_LOGGER = logging.getLogger("training.datasets")


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


### Transform-config kwargs whose values are file paths and should be
### resolved against the recipe root before Hydra instantiates the
### transform. Add new entries here when introducing transforms with
### path arguments (e.g. an ``index_file`` for a future indexing
### transform). Matched by exact key name; values that are already
### absolute paths or that don't exist on disk are passed through
### unchanged so the underlying transform owns the error message.
_PATH_KEYS = {"stats_file"}

### Match `.CenterMesh` (with leading dot) on the fully-qualified
### `_target_` so a hypothetical sibling like `CenterMeshAdjusted`
### doesn't silently capture the augmentation insertion point.
_CENTER_MESH_TARGET_SUFFIX = ".CenterMesh"
_MESH_TO_DOMAIN_MESH_SUFFIX = "MeshToDomainMesh"


def _resolve_transform_paths(t_cfg: DictConfig, base_dir: Path) -> DictConfig:
    """Resolve relative file paths in a transform config against *base_dir*.

    Walks the keys in :data:`_PATH_KEYS` and, for each that resolves to
    a relative existing path under *base_dir*, rewrites it to an absolute
    path so Hydra's working-directory change doesn't break the
    instantiation. Transforms like ``NormalizeMeshFields``'s
    ``stats_file`` parameter rely on this.
    """
    for key in _PATH_KEYS:
        val = OmegaConf.select(t_cfg, key, default=None)
        if val is not None and not Path(val).is_absolute():
            resolved = base_dir / val
            if resolved.exists():
                t_cfg = OmegaConf.merge(t_cfg, {key: str(resolved)})
    return t_cfg


def _maybe_inject_targets(t_cfg: DictConfig, target_names: list[str]) -> DictConfig:
    """Auto-inject target names into a ``MeshToDomainMesh`` transform.

    The dataset YAML's ``targets:`` block is the single source of truth for
    target field names. Repeating the names inside the ``MeshToDomainMesh``
    transform is redundant and error-prone, so when the user omits them we
    fill them in here based on the transform's ``interior_points`` strategy:

    - ``interior_points='cell_centroids'`` (default) uses ``cell_data_targets``.
    - ``interior_points='vertices'`` uses ``point_data_targets``.

    No-op for transforms that aren't ``MeshToDomainMesh`` or that already
    specify target names explicitly.
    """
    target = OmegaConf.select(t_cfg, "_target_", default="") or ""
    if not target.endswith(_MESH_TO_DOMAIN_MESH_SUFFIX):
        return t_cfg

    interior = OmegaConf.select(t_cfg, "interior_points", default="cell_centroids")
    if interior == "cell_centroids":
        if OmegaConf.select(t_cfg, "cell_data_targets", default=None) is None:
            t_cfg = OmegaConf.merge(
                t_cfg, OmegaConf.create({"cell_data_targets": list(target_names)})
            )
    elif interior == "vertices":
        if OmegaConf.select(t_cfg, "point_data_targets", default=None) is None:
            t_cfg = OmegaConf.merge(
                t_cfg, OmegaConf.create({"point_data_targets": list(target_names)})
            )
    return t_cfg


class _InjectMetadata(MeshTransform):
    """Inject dataset-wide metadata fields into a sample's ``global_data``.

    The dataset YAML's ``metadata:`` block carries freestream conditions
    (``U_inf``, ``rho_inf``, ``p_inf``, ``L_ref``, ...) that downstream
    transforms like ``NonDimensionalizeByMetadata`` need to consume from
    ``global_data``. This transform writes those values into the sample's
    ``global_data`` (matching the sample's device/dtype) on every call.

    For :class:`DomainMesh` inputs the values land in the domain-level
    ``global_data`` (so they're visible to domain-aware transforms);
    for :class:`Mesh` inputs they land in the mesh's ``global_data``.
    Subclasses :class:`MeshTransform` so the recipe's dataset builder
    can hand it to :class:`MeshDataset` alongside any other transform.
    """

    def __init__(self, metadata: dict[str, Any]) -> None:
        super().__init__()
        ### Coerce every field to a 0-D / 1-D float32 tensor up front and
        ### pack the bundle into a single ``TensorDict``. Per-sample work
        ### then collapses to one batched ``td.to(device, dtype)`` plus
        ### one ``new_gd.update(...)`` -- no Python-level loop.
        coerced: dict[str, torch.Tensor] = {}
        for k, v in metadata.items():
            if isinstance(v, torch.Tensor):
                coerced[k] = v.float()
            else:
                ### Accepts list / tuple / scalar uniformly.
                coerced[k] = torch.as_tensor(v, dtype=torch.float32)
        self._fields: TensorDict = TensorDict(coerced, batch_size=[])

    def __call__(self, mesh: Mesh) -> Mesh:
        new_gd = mesh.global_data.clone()
        new_gd.update(
            self._fields.to(device=mesh.points.device, dtype=mesh.points.dtype)
        )
        return Mesh(
            points=mesh.points,
            cells=mesh.cells,
            point_data=mesh.point_data,
            cell_data=mesh.cell_data,
            global_data=new_gd,
        )

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Inject metadata into ``domain.global_data``."""
        interior_pts = domain.interior.points
        new_gd = domain.global_data.clone()
        new_gd.update(
            self._fields.to(device=interior_pts.device, dtype=interior_pts.dtype)
        )
        return DomainMesh(
            interior=domain.interior,
            boundaries=domain.boundaries,
            global_data=new_gd,
        )

    def extra_repr(self) -> str:
        return f"fields={sorted(self._fields.keys())}"


def find_normalizer(
    datasets: list[MeshDataset],
) -> NormalizeMeshFields | None:
    """Return the first :class:`NormalizeMeshFields` found across *datasets* pipelines.

    Used at checkpoint-save time to persist normalization stats alongside
    the model weights so inference can re-apply the inverse. Returns
    ``None`` when no dataset has a ``NormalizeMeshFields`` transform.
    """
    for ds in datasets:
        for t in getattr(ds, "transforms", []):
            if isinstance(t, NormalizeMeshFields):
                return t
    return None


def validate_dataset_consistency(
    ds_key: str,
    ds_targets: dict[str, FieldType],
    ds_metrics: list[MetricName],
    ds_metadata: dict[str, Any],
    first_targets: dict[str, FieldType],
    first_metrics: list[MetricName],
    first_metadata: dict[str, Any],
) -> None:
    """Reject ``targets:`` mismatch across multi-dataset training; warn on softer drift.

    ``targets:`` is the loss / metric contract; mismatched names or types
    silently produces wrong per-field losses (only the first dataset's
    keys are honored downstream). ``metrics:`` and ``metadata:`` are
    softer -- the recipe still uses the first dataset's view -- but
    drift is almost always a config bug, so we warn loudly.
    """
    if ds_targets != first_targets:
        raise ValueError(
            f"Dataset {ds_key!r} declares targets={ds_targets!r}, "
            f"which does not match the first dataset's targets="
            f"{first_targets!r}. All datasets in `cfg.data` must "
            f"declare the same `targets:` block (same names, same "
            f"types, same iteration order)."
        )
    if ds_metrics != first_metrics:
        _LOGGER.warning(
            f"Dataset {ds_key!r} declares metrics={ds_metrics!r}, "
            f"which differs from the first dataset's metrics="
            f"{first_metrics!r}. Using the first dataset's metrics."
        )
    if ds_metadata != first_metadata:
        _LOGGER.warning(
            f"Dataset {ds_key!r} has metadata that differs from the "
            f"first dataset's; using the first dataset's metadata "
            f"for the loss / non-dim back-conversion. Per-dataset "
            f"metadata is still injected into each sample's "
            f"global_data by the dataset builder."
        )


def resolve_manifest_spec(ds_yaml: DictConfig, ds_cfg_block: DictConfig) -> dict | None:
    """Resolve a `data.<key>` block's manifest config; return ``None`` for directory mode.

    Two manifest styles are recognised:

    - **Style A (separate files):** ``train_manifest`` / ``val_manifest``
      point at flat lists of run subpaths.
    - **Style B (single dict manifest):** ``manifest`` + ``train_split`` /
      ``val_split`` keys into a JSON dict. If ``manifest`` is omitted, we
      look for a sibling ``manifest.json`` next to the dataset YAML's
      ``train_datadir``.

    Returns a flat dict with both styles' fields present (extras are
    ``None``); returns ``None`` if neither style is configured.
    """
    train_manifest = ds_cfg_block.get("train_manifest", None)
    val_manifest = ds_cfg_block.get("val_manifest", None)
    manifest = ds_cfg_block.get("manifest", None)
    train_split = ds_cfg_block.get("train_split", None)
    val_split = ds_cfg_block.get("val_split", None)

    ### Auto-derive manifest path from `train_datadir/manifest.json` when
    ### the user gave a split key but no explicit manifest path.
    if manifest is None and train_split is not None:
        train_datadir = OmegaConf.select(ds_yaml, "train_datadir", default=None)
        if train_datadir:
            derived = Path(str(train_datadir)) / "manifest.json"
            if derived.exists():
                manifest = str(derived)

    has_manifest = train_manifest is not None or (
        manifest is not None and train_split is not None
    )
    if not has_manifest:
        return None
    return {
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "manifest": manifest,
        "train_split": train_split,
        "val_split": val_split,
    }


def build_dataset(
    cfg: DictConfig,
    base_dir: Path | None = None,
    augment: bool = False,
    device: str | torch.device | None = "auto",
    num_workers: int = 1,
    pin_memory: bool = False,
) -> MeshDataset:
    """Build a single MeshDataset from a Hydra-style pipeline config.

    Args:
        cfg: Dataset config with a ``pipeline:`` block containing
            ``reader:`` and ``transforms:`` entries. An optional
            ``pipeline.augmentations`` list defines stochastic
            augmentation transforms (e.g. ``RandomRotateMesh``,
            ``RandomTranslateMesh``) that are inserted after
            ``CenterMesh`` when *augment* is ``True``. If a top-level
            ``metadata:`` block is present, its values are injected into
            ``mesh.global_data`` as the first transform step.
        base_dir: Root directory for resolving relative paths in
            transform configs (e.g. ``stats_file``). Defaults to the
            recipe root (two levels above this file).
        augment: When ``True``, ``pipeline.augmentations`` transforms
            are inserted into the pipeline after ``CenterMesh``. Should
            be ``False`` for validation / test datasets.
        device: Device to transfer mesh data to before transforms. When
            ``None``, data stays on CPU.
        num_workers: Number of worker threads for the MeshDataset
            prefetch pool.
        pin_memory: If True, the reader places tensors in pinned
            (page-locked) memory for faster async CPU-to-GPU transfers.

    Returns:
        Configured ``MeshDataset`` ready to be wrapped in a DataLoader.
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
        resolved.append(_InjectMetadata(metadata))

    target_names = list(
        OmegaConf.to_container(
            OmegaConf.select(cfg, "targets", default=OmegaConf.create({})),
            resolve=True,
        )
        or {}
    )

    if "transforms" in cfg.pipeline and cfg.pipeline.transforms:
        for t in cfg.pipeline.transforms:
            t = _resolve_transform_paths(t, base_dir)
            t = _maybe_inject_targets(t, target_names)
            resolved.append(hydra.utils.instantiate(t))

        if augment and "augmentations" in cfg.pipeline and cfg.pipeline.augmentations:
            aug = [hydra.utils.instantiate(a) for a in cfg.pipeline.augmentations]
            # +1 for the metadata injector prepended above
            offset = 1 if metadata else 0
            insert_idx = next(
                (
                    offset + i + 1
                    for i, t_cfg in enumerate(cfg.pipeline.transforms)
                    if t_cfg.get("_target_", "").endswith(_CENTER_MESH_TARGET_SUFFIX)
                ),
                len(resolved),
            )
            resolved[insert_idx:insert_idx] = aug

    transforms = resolved if resolved else None
    return MeshDataset(
        reader, transforms=transforms, device=device, num_workers=num_workers
    )


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

    Args:
        path: Path to the manifest file.
        split: Key to extract from a JSON dict manifest (e.g.
            ``"single_aoa_4_train"``). Required when the manifest is a
            dict; ignored for flat list / text manifests.

    Returns:
        Sorted list of sub-path strings.

    Raises:
        KeyError: If *split* is given but not found in the manifest dict.
        ValueError: If the manifest format doesn't match expectations.
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

    Args:
        reader: An instantiated reader (``MeshReader`` or
            ``DomainMeshReader``) with ``_root`` and ``_paths`` attributes.
        manifest_entries: Sub-path strings from the manifest
            (e.g. ``["run_1", "run_5"]``).

    Returns:
        Sorted list of reader indices whose paths match the manifest.

    Raises:
        ValueError: If no reader paths match any manifest entry.
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

    Args:
        indices: Dataset indices that belong to this split.
        shuffle: Whether to shuffle indices each epoch.
        seed: Base random seed for reproducible shuffling.
        rank: Current process rank (for distributed sharding). 0 for
            single-GPU.
        world_size: Total number of processes. 1 for single-GPU.
        drop_last: If True, drop tail indices so every rank gets the
            same count.
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
