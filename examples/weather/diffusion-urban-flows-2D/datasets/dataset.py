# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Dataset initialization and dataloader creation utilities."""

from typing import Iterable, Tuple, Union
import copy
import torch

from torch.utils.data import distributed

from modulus.distributed import DistributedManager

from . import base, uflow2d


# this maps all known dataset types to the corresponding init function
known_datasets = {"uflow2d": uflow2d.OneObs2D}


def init_train_valid_datasets_from_config(
    dataset_cfg: dict,
    dataloader_cfg: Union[dict, None] = None,
    batch_size: int = 1,
    seed: int = 0,
    validation_dataset_cfg: Union[dict, None] = None,
    train_test_split: bool = True,
) -> Tuple[
    base.DownscalingDataset,
    Iterable,
    Union[base.DownscalingDataset, None],
    Union[Iterable, None],
]:
    """
    A wrapper function for managing the train-test split for the CWB dataset.

    Parameters:
    - dataset_cfg (dict): Configuration for the dataset.
    - dataloader_cfg (dict, optional): Configuration for the dataloader. Defaults to None.
    - batch_size (int): The number of samples in each batch of data. Defaults to 1.
    - seed (int): The random seed for dataset shuffling. Defaults to 0.
    - train_test_split (bool): A flag to determine whether to create a validation dataset. Defaults to True.

    Returns:
    - Tuple[base.DownscalingDataset, Iterable, Optional[base.DownscalingDataset], Optional[Iterable]]: A tuple containing the training dataset and iterator, and optionally the validation dataset and iterator if train_test_split is True.
    """

    config = copy.deepcopy(dataset_cfg)
    (dataset, dataset_iter) = init_dataset_from_config(
        config, dataloader_cfg, batch_size=batch_size, seed=seed
    )
    if train_test_split:
        valid_dataset_cfg = copy.deepcopy(config)
        if validation_dataset_cfg:
            valid_dataset_cfg.update(validation_dataset_cfg)
        (valid_dataset, valid_dataset_iter) = init_dataset_from_config(
            valid_dataset_cfg, dataloader_cfg, batch_size=batch_size, seed=seed
        )
    else:
        valid_dataset = valid_dataset_iter = None

    return dataset, dataset_iter, valid_dataset, valid_dataset_iter


def init_dataset_from_config(
    dataset_cfg: dict,
    dataloader_cfg: Union[dict, None] = None,
    batch_size: int = 1,
    seed: int = 0,
) -> Tuple[base.DownscalingDataset, Iterable]:
    """Initialize a dataset and dataloader from configuration dictionaries.

    Parameters
    ----------
    dataset_cfg : dict
        Configuration dictionary for the dataset.
    dataloader_cfg : dict, optional
        Configuration dictionary for the dataloader.
    batch_size : int, optional
        Number of samples per batch. Default is 1.
    seed : int, optional
        Random seed for data shuffling. Default is 0.

    Returns
    -------
    tuple
        A tuple containing (dataset_object, dataset_iterator).
    """
    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_type = dataset_cfg.pop("type", "cwb")
    if "train_test_split" in dataset_cfg:
        # handled by init_train_valid_datasets_from_config
        del dataset_cfg["train_test_split"]
    dataset_init_func = known_datasets[dataset_type]

    dataset_obj = dataset_init_func(**dataset_cfg)
    if dataloader_cfg is None:
        dataloader_cfg = {}

    dist = DistributedManager()
    #    dataset_sampler = InfiniteSampler(
    #        dataset=dataset_obj, rank=dist.rank, num_replicas=dist.world_size, seed=seed, shuffle=True,
    #    )
    dataset_sampler = distributed.DistributedSampler(
        dataset=dataset_obj,
        rank=dist.rank,
        num_replicas=dist.world_size,
        seed=seed,
        shuffle=True,
    )

    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_size,
            worker_init_fn=None,
            **dataloader_cfg,
        )
    )

    return (dataset_obj, dataset_iterator)


def get_dataset_and_dataloader_from_config(
    dataset_cfg: dict,
    dataloader_cfg: Union[dict, None] = None,
    batch_size: int = 1,
    seed: int = 0,
    shuffle=False,
    dist=None,
    Train=False,
):
    """Create a dataset and dataloader from configuration dictionaries.

    Parameters
    ----------
    dataset_cfg : dict
        Configuration dictionary for the dataset.
    dataloader_cfg : dict, optional
        Configuration dictionary for the dataloader.
    batch_size : int, optional
        Number of samples per batch. Default is 1.
    seed : int, optional
        Random seed for data shuffling. Default is 0.
    shuffle : bool, optional
        Whether to shuffle the data. Default is False.
    dist : DistributedManager, optional
        Distributed training manager instance.
    Train : bool, optional
        Whether this is for training mode. Default is False.

    Returns
    -------
    tuple
        A tuple containing (dataset_object, dataloader).
    """

    assert dist is not None

    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_type = dataset_cfg.pop("type", "none")

    assert dataset_type in known_datasets
    dataset_init_func = known_datasets[dataset_type]

    dataset_obj = dataset_init_func(**dataset_cfg)
    if dataloader_cfg is None:
        dataloader_cfg = {}

    if Train:
        dist = DistributedManager()

    dataset_sampler = distributed.DistributedSampler(
        dataset=dataset_obj,
        rank=dist.rank,
        num_replicas=dist.world_size,
        seed=seed,
        shuffle=shuffle,
    )

    dataset_loader = torch.utils.data.DataLoader(
        dataset=dataset_obj,
        sampler=dataset_sampler,
        batch_size=batch_size,
        worker_init_fn=None,
        **dataloader_cfg,
    )

    return (dataset_obj, dataset_loader)
