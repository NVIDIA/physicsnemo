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

"""DataPool + small data-setup helpers for multi-class DrivAerStar AL.

Wraps the transolver datapipe with index tracking to support the
physicsnemo active learning ``DataPool`` protocol.  Each sample is
tagged with its vehicle class (F/N/E, SE/SF, …) for composition
analysis.  Supports loading from pre-built JSON manifests for
reproducibility.

Also exposes two thin factory helpers shared by the AL pipeline,
the from-scratch ceiling trainer, and the inference script:

* ``build_surface_factors`` — build per-channel ``{mean, std}``
  factors for ``surface_fields``, either by physics
  non-dimensionalisation (Cp / Cf) from freestream metadata or by
  loading ``surface_fields_normalization.npz`` from disk.
* ``build_pool`` — construct an ``AeroDataPool`` with every sample
  in the pool marked as "in training" (i.e. iterable).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

import omegaconf
from omegaconf import DictConfig, OmegaConf
from physicsnemo.datapipes.cae.transolver_datapipe import create_transolver_dataset


def load_manifests(
    manifest_dir: str | Path,
) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, str]]:
    """Load test/pool splits from JSON manifests.

    Parameters
    ----------
    manifest_dir : str | Path
        Directory containing ``manifest_class_*.json`` files written by
        ``create_manifests.py``.

    Returns
    -------
    pool_by_class : dict[str, list[int]]
        Class label -> list of local indices forming the AL pool.
    test_by_class : dict[str, list[int]]
        Class label -> list of local indices forming the held-out test set.
    paths_by_class : dict[str, str]
        Class label -> path to the zarr ``val`` directory for that class.
    """
    manifest_dir = Path(manifest_dir)
    pool_by_class: dict[str, list[int]] = {}
    test_by_class: dict[str, list[int]] = {}
    paths_by_class: dict[str, str] = {}

    for manifest_file in sorted(manifest_dir.glob("manifest_class_*.json")):
        with open(manifest_file) as f:
            m = json.load(f)
        cls = m["class"]
        pool_by_class[cls] = m["pool_indices"]
        test_by_class[cls] = m["test_indices"]
        paths_by_class[cls] = m["zarr_path"]

    return pool_by_class, test_by_class, paths_by_class


class AeroDataPool(Dataset):
    """Pool of DrivAerStar samples with index-based training set tracking.

    Concatenates samples from multiple class directories (Fastback,
    Notchback, Estateback) into a single flat index space and tracks
    which indices are currently in the training set.

    Parameters
    ----------
    data_cfg : omegaconf.DictConfig
        Data config (from the geotransolver_surface_gp yaml).
    class_paths : dict[str, str]
        Mapping from class label (e.g. "F", "N", "E") to the zarr
        val directory path for that class.
    surface_factors : dict
        Normalization factors (mean/std tensors).
    local_indices_by_class : dict[str, list[int]] | None
        If provided, restricts the addressable samples per class to
        these local dataset indices (from manifests).  If None, all
        samples in each class are addressable.
    train_indices : torch.LongTensor | None
        Initial training (flat) indices.  If None, starts empty.
    """

    def __init__(
        self,
        data_cfg: omegaconf.DictConfig,
        class_paths: dict[str, str],
        surface_factors: dict,
        local_indices_by_class: dict[str, list[int]] | None = None,
        train_indices: torch.LongTensor | None = None,
    ) -> None:
        super().__init__()
        self._raw_datasets: list = []
        self._datapipes: list = []
        self._class_labels: list[str] = []
        self._class_offsets: list[int] = []
        self._flat_to_local: list[tuple[int, int]] = []

        offset = 0
        for cls_label, path in class_paths.items():
            cfg_copy = omegaconf.OmegaConf.create(
                omegaconf.OmegaConf.to_container(data_cfg, resolve=True)
            )
            cfg_copy.val.data_path = path
            datapipe = create_transolver_dataset(
                cfg_copy,
                phase="val",
                surface_factors=surface_factors,
                volume_factors=None,
            )
            ds_idx = len(self._raw_datasets)
            self._raw_datasets.append(datapipe.dataset)
            self._datapipes.append(datapipe)
            self._class_offsets.append(offset)

            if (
                local_indices_by_class is not None
                and cls_label in local_indices_by_class
            ):
                local_idxs = local_indices_by_class[cls_label]
            else:
                local_idxs = list(range(len(datapipe.dataset)))

            for li in local_idxs:
                self._flat_to_local.append((ds_idx, li))
                self._class_labels.append(cls_label)
            offset += len(local_idxs)

        self._total_samples = offset
        self.train_indices = (
            train_indices if train_indices is not None else torch.LongTensor([])
        )

    @property
    def total_samples(self) -> int:
        """Total number of samples across all underlying datasets."""
        return self._total_samples

    @property
    def class_labels(self) -> list[str]:
        """Per-sample class label list, indexed by flat sample index."""
        return self._class_labels

    def class_of(self, flat_idx: int) -> str:
        """Return the class label for a given flat sample index."""
        return self._class_labels[flat_idx]

    def _get_preprocessed(self, flat_idx: int) -> dict:
        """Fetch a raw sample by flat index and run the datapipe preprocessing."""
        ds_idx, local_idx = self._flat_to_local[flat_idx]
        raw_sample = self._raw_datasets[ds_idx][local_idx]
        return self._datapipes[ds_idx](raw_sample)

    def prefetch(self, flat_idx: int) -> None:
        """Asynchronously schedule a read for the sample at ``flat_idx``.

        Backed by :py:meth:`physicsnemo.datapipes.cae.cae_dataset.CAEDataset.preload`
        which uses an in-process ``ThreadPoolExecutor``.  Calling this before
        ``__getitem__`` lets file I/O overlap with the previous step's
        GPU compute.  Idempotent: a re-prefetch of an in-flight index is a
        no-op.  The eventual ``__getitem__`` will consume the preloaded
        result if it has landed, or block on the future otherwise.

        This stays in-process on purpose: per-class ``CAEDataset`` instances
        hold zarr handles and the datapipe holds GPU-resident
        ``surface_factors``; neither is safe to pickle across DataLoader
        worker subprocess boundaries.
        """
        if not (0 <= flat_idx < self._total_samples):
            return
        ds_idx, local_idx = self._flat_to_local[flat_idx]
        preload = getattr(self._raw_datasets[ds_idx], "preload", None)
        if preload is not None:
            preload(local_idx)

    def unlabeled_indices(self) -> torch.LongTensor:
        """Return flat indices not yet in the training set."""
        all_idx = torch.arange(self._total_samples)
        mask = ~torch.isin(all_idx, self.train_indices)
        return all_idx[mask]

    def __len__(self) -> int:
        return len(self.train_indices)

    def __getitem__(self, index: int) -> dict:
        flat_idx = self.train_indices[index].item()
        return self._get_preprocessed(flat_idx)

    def get_by_flat_idx(self, flat_idx: int) -> dict:
        """Access a sample by its flat (pool-wide) index, bypassing train_indices."""
        return self._get_preprocessed(flat_idx)

    def __iter__(self) -> Iterator[dict]:
        for i in range(len(self)):
            yield self[i]

    def append(self, item: int) -> None:
        """Add a flat index to the training set."""
        self.train_indices = torch.cat([self.train_indices, torch.LongTensor([item])])

    def set_indices(self, indices: list[int]) -> None:
        """Directly set the training indices (for DDP sampler compatibility)."""
        self.train_indices = torch.LongTensor(indices)


# ---------------------------------------------------------------------------
# Factory helpers shared by run_al / train_ceiling / infer_aero
# ---------------------------------------------------------------------------


def build_surface_factors(
    cfg: DictConfig, device: torch.device, logger
) -> dict[str, torch.Tensor]:
    """Build per-channel ``{mean, std}`` factors for ``surface_fields``.

    Two modes, selected by ``cfg.data.physics_nondim.enabled``:

    * **Physics non-dim (Cp / Cf):** factors computed in memory from freestream
      metadata.  Combined with the dataloader's existing ``mean_std_scaling``,
      ``(p - p_inf) / q_inf`` lands as Cp and ``wss / (q_inf * wss_factor)``
      lands as Cf divided by an extra std factor that matches the unified
      recipe (``shift_suv_fastback.yaml`` uses ``std=0.00183``).  Geometry
      non-dim is orthogonal and set via ``data.reference_scale=[L_ref]*3``.
    * **File mode (default):** load ``mean`` and ``std`` from
      ``surface_fields_normalization.npz`` in ``cfg.data.normalization_dir``
      (the legacy Fastback path).
    """
    pn = OmegaConf.select(cfg, "data.physics_nondim", default=None)
    if pn is not None and bool(OmegaConf.select(pn, "enabled", default=False)):
        U_inf = float(OmegaConf.select(pn, "U_inf", default=30.0))
        rho_inf = float(OmegaConf.select(pn, "rho_inf", default=1.225))
        p_inf = float(OmegaConf.select(pn, "p_inf", default=0.0))
        wss_factor = float(OmegaConf.select(pn, "wss_factor", default=0.00183))
        q_inf = 0.5 * rho_inf * U_inf * U_inf
        mean = torch.tensor([p_inf, 0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        wss_std = q_inf * wss_factor
        std = torch.tensor(
            [q_inf, wss_std, wss_std, wss_std], device=device, dtype=torch.float32
        )
        logger.info(
            f"Surface factors: physics non-dim (U_inf={U_inf}, rho_inf={rho_inf}, "
            f"p_inf={p_inf}, q_inf={q_inf:.4f}, wss_factor={wss_factor}) -> "
            f"mean={mean.tolist()}, std={std.tolist()}"
        )
        return {"mean": mean, "std": std}

    norm_dir = getattr(cfg.data, "normalization_dir", ".")
    norm_file = str(Path(norm_dir) / "surface_fields_normalization.npz")
    norm_data = np.load(norm_file)
    factors = {
        "mean": torch.from_numpy(norm_data["mean"]).to(device),
        "std": torch.from_numpy(norm_data["std"]).to(device),
    }
    logger.info(
        f"Surface factors: loaded from {norm_file} -> "
        f"mean={factors['mean'].tolist()}, std={factors['std'].tolist()}"
    )
    return factors


def build_pool(
    data_cfg: DictConfig,
    paths_by_class: dict[str, str],
    indices_by_class: dict[str, list[int]],
    surface_factors: dict,
) -> AeroDataPool:
    """Build an ``AeroDataPool`` with all listed indices marked as 'in training'.

    For both the train and val pools we want every sample iterable, so we
    populate ``train_indices`` with the full flat range up-front.
    """
    pool = AeroDataPool(
        data_cfg=data_cfg,
        class_paths=paths_by_class,
        surface_factors=surface_factors,
        local_indices_by_class=indices_by_class,
        train_indices=torch.LongTensor([]),
    )
    pool.train_indices = torch.arange(pool.total_samples).long()
    return pool
