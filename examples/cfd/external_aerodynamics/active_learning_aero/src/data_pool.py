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

"""DataPool for multi-class DrivAerStar active learning.

Wraps the transolver datapipe with index tracking to support the
physicsnemo active learning ``DataPool`` protocol.  Each sample is
tagged with its vehicle class (F/N/E) for composition analysis.
Supports loading from pre-built JSON manifests for reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset

import omegaconf
from physicsnemo.datapipes.cae.transolver_datapipe import create_transolver_dataset


def load_manifests(
    manifest_dir: str | Path,
) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, str]]:
    """Load test/pool splits from JSON manifests.

    Returns
    -------
    pool_by_class : dict mapping class label -> list of local indices for AL pool
    test_by_class : dict mapping class label -> list of local indices for test
    paths_by_class : dict mapping class label -> zarr path
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

            if local_indices_by_class is not None and cls_label in local_indices_by_class:
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
        return self._total_samples

    @property
    def class_labels(self) -> list[str]:
        return self._class_labels

    def class_of(self, flat_idx: int) -> str:
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
        self.train_indices = torch.cat(
            [self.train_indices, torch.LongTensor([item])]
        )

    def set_indices(self, indices: list[int]) -> None:
        """Directly set the training indices (for DDP sampler compatibility)."""
        self.train_indices = torch.LongTensor(indices)
