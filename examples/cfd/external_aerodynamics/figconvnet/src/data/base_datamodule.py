# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import webdataset as wds

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.distributed import DistributedManager


class BaseDataModule:
    """Base data module."""

    @property
    def train_dataset(self) -> Dataset:
        raise NotImplementedError

    @property
    def val_dataset(self) -> Dataset:
        raise NotImplementedError

    @property
    def test_dataset(self) -> Dataset:
        raise NotImplementedError

    def _create_dataloader(self, dataset: Dataset, **kwargs) -> DataLoader:
        collate_fn = getattr(self, "collate_fn", None)

        shuffle = kwargs.pop("shuffle", False)
        sampler = kwargs.pop("sampler", None)
        if sampler is None and DistributedManager().distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)

        return DataLoader(
            dataset,
            collate_fn=collate_fn,
            sampler=sampler,
            shuffle=None if sampler is not None else shuffle,
            **kwargs,
        )

    def train_dataloader(self, **kwargs) -> DataLoader:
        return self._create_dataloader(self.train_dataset, **kwargs)

    def val_dataloader(self, **kwargs) -> DataLoader:
        return self._create_dataloader(self.val_dataset, **kwargs)

    def test_dataloader(self, **kwargs) -> DataLoader:
        return self._create_dataloader(self.test_dataset, **kwargs)

    @staticmethod
    def set_epoch(dataloader: DataLoader, epoch: int):
        """Sets the epoch in the dataloader.

        In some cases, such as in distributed case with DistributedSampler,
        the sampler requires setting an epoch number to properly shuffle entries.
        """
        try:
            dataloader.sampler.set_epoch(epoch)
        except AttributeError:
            pass


class WebdatasetDataModule(BaseDataModule):
    """
    Base class for webdataset
    """

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def val_dataset(self):
        return self._val_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    def _create_dataloader(self, dataset: wds.DataPipeline, **kwargs) -> wds.WebLoader:
        # Handle shuffling and batching.
        stages = []
        if (buf_size := kwargs.pop("shuffle_buffer_size", 0)) or kwargs.pop(
            "shuffle", False
        ):
            stages.append(wds.shuffle(buf_size if buf_size > 0 else 100))

        batch_size = kwargs.pop("batch_size", 1)
        stages.append(
            wds.batched(batch_size, collation_fn=torch.utils.data.default_collate)
        )

        # Create dataloader from the pipeline.
        # Use `compose` to avoid changing the original dataset.
        return wds.WebLoader(
            dataset.compose(*stages),
            batch_size=None,
            shuffle=False,
            **kwargs,
        )
