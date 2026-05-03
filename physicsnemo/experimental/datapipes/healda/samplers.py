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
"""Stateful distributed sampler with checkpoint support.

``RestartableDistributedSampler`` is a distributed sampler that tracks its
iteration state so that training can be resumed from a checkpoint without
replaying already-seen samples.
"""

import torch
import torch.utils.data


class RestartableDistributedSampler(torch.utils.data.Sampler):
    """A stateful distributed sampler that automatically loops over the dataset.

    Each epoch generates a shared random permutation across ranks, then
    partitions it by stride so every sample is visited exactly once per epoch.
    The sampler tracks its position within the permutation so that
    ``restart()`` can resume from an exact checkpoint.

    Args:
        dataset: Map-style dataset (used only for ``len``).
        rank: This worker's global rank.
        num_replicas: Total number of data-parallel workers.
        shuffle: Whether to shuffle (always True in practice).
        drop_last: Drop remainder so all ranks get the same count.
        seed: Base random seed.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        rank=0,
        num_replicas=1,
        shuffle=True,
        drop_last=True,
        seed=42,
    ):
        super().__init__()
        self.iteration = 0
        self.epoch = 0
        self.len = len(dataset)
        self.seed = seed
        self.permutation = None
        self.rank = rank
        self.num_replicas = num_replicas

    def __len__(self):
        return self.len // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.iteration = 0
        rng = torch.Generator().manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(self.len, generator=rng)

        rem = self.len % self.num_replicas
        if rem > 0:
            permutation = permutation[:-rem]
        self.permutation = permutation[self.rank :: self.num_replicas]

    def restart(self, epoch, iteration, seed=None):
        """Resume from a checkpoint."""
        self.seed = seed or self.seed
        self.set_epoch(epoch)
        self.iteration = iteration

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= len(self):
            self.set_epoch(self.epoch + 1)
            raise StopIteration()

        idx = self.permutation[self.iteration].item()
        self.iteration += 1
        return idx
