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

"""GPU prefetch wrapper for PyTorch DataLoaders.

Overlaps host-to-device transfers with GPU compute by using a
dedicated CUDA stream.  Batches arrive on-device; the training
loop no longer needs explicit ``.to(device)`` calls.

Usage
-----
>>> loader = DataLoader(dataset, batch_size=4)
>>> prefetched = GPUPrefetcher(loader, device="cuda:0")
>>> for inputs, targets in prefetched:
...     # inputs and targets are already on GPU
...     pred = model(inputs)
"""

from typing import Iterator, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader


class GPUPrefetcher:
    """Prefetches the next batch to GPU while the current batch trains.

    Parameters
    ----------
    loader : DataLoader
        Source dataloader (CPU-side).
    device : str or torch.device
        Target GPU device.
    """

    def __init__(
        self,
        loader: DataLoader,
        device: Union[str, torch.device] = "cuda",
    ):
        self.loader = loader
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)

    def __iter__(self) -> Iterator[Tuple[Tensor, ...]]:
        """Yield batches that are already on *device*."""
        it = iter(self.loader)

        try:
            batch = next(it)
        except StopIteration:
            return

        batch = self._transfer(batch)

        for next_batch in it:
            with torch.cuda.stream(self.stream):
                next_batch = self._transfer(next_batch)

            yield batch
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            batch = next_batch

        yield batch

    def __len__(self) -> int:
        """Number of batches (delegated to the wrapped loader)."""
        return len(self.loader)

    # ------------------------------------------------------------------
    # Forwarded attributes so the prefetcher is a drop-in for DataLoader
    # ------------------------------------------------------------------

    @property
    def dataset(self):
        """Underlying dataset."""
        return self.loader.dataset

    @property
    def sampler(self):
        """Underlying sampler."""
        return self.loader.sampler

    def _transfer(self, batch: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        return tuple(t.to(self.device, non_blocking=True) for t in batch)
