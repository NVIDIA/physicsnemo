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

from abc import ABC, abstractmethod
from functools import cache
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset


class CachedPreprocessingDataset(Dataset, ABC):
    """Dataset that lazily preprocesses samples and caches results to disk/RAM.

    Subclasses implement the ``preprocess`` static method to define how raw
    samples are transformed. On first access the result is computed and
    (optionally) saved to *cache_dir* as a ``.pt`` file keyed by the sample
    directory name. Subsequent accesses load the cached result directly.

    Args:
        sample_paths: Paths to individual samples in the dataset.
        cache_dir: Directory for disk caching. ``None`` disables disk caching.
        use_ram_caching: If True, wraps ``__getitem__`` with
            ``functools.cache`` for in-memory caching (increases memory usage).

    Raises:
        FileNotFoundError: If any *sample_paths* entry does not exist on disk.
    """

    def __init__(
        self,
        sample_paths: Sequence[Path | str],
        cache_dir: Path | str | None = None,
        use_ram_caching: bool = False,
    ):
        self.sample_paths = [Path(path) for path in sample_paths]
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.use_ram_caching = use_ram_caching

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        nonexistent_sample_paths = [
            path for path in self.sample_paths if not path.exists()
        ]
        if nonexistent_sample_paths:
            raise FileNotFoundError(
                "The following sample paths were given, but do not exist:\n"
                f"{nonexistent_sample_paths}"
            )

        if self.use_ram_caching:
            self.__getitem__ = cache(self.__getitem__)  # ty: ignore[invalid-assignment]

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index) -> Any:  # ty: ignore[invalid-method-override]
        sample_path = self.sample_paths[index]

        if self.cache_dir is not None:
            cache_path = (self.cache_dir / sample_path.name).with_suffix(".pt")
            if cache_path.exists():
                return torch.load(cache_path, weights_only=False)

        sample = self.preprocess(sample_path=sample_path)

        if self.cache_dir is not None:
            torch.save(sample, cache_path)

        return sample

    @staticmethod
    @abstractmethod
    def preprocess(sample_path: Path) -> Any:
        """Transform a raw sample at *sample_path* into the desired format.

        The returned object must be compatible with ``torch.save``/``torch.load``
        (e.g. tensors, tensor containers, or picklable Python objects), since
        disk caching serializes it as a ``.pt`` file.
        """
