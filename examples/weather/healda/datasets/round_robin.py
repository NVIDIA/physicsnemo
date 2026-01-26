# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch


class RoundRobinLoader(torch.utils.data.IterableDataset):
    """Round-robin interleaving of multiple map-style dataloaders.

    This allows converting map-style datasets to iterable-style. It loads data
    from the dataloaders in a round-robin manner, removing exhausted dataloaders
    from rotation until ALL dataloaders are exhausted.

    Args:
        dataloaders: List of DataLoader instances to interleave
    """

    def __init__(self, dataloaders: list[torch.utils.data.DataLoader]):
        super().__init__()
        self.dataloaders = dataloaders

    def __len__(self):
        return sum(len(dl) for dl in self.dataloaders)

    def __iter__(self):
        iterators = [iter(dl) for dl in self.dataloaders]
        active_indices = list(range(len(self.dataloaders)))

        while active_indices:
            for idx in list(active_indices):
                try:
                    yield next(iterators[idx])
                except StopIteration:
                    # Remove exhausted dataloader from rotation
                    active_indices.remove(idx)
