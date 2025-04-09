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

from typing import Iterable, Tuple, Union
import copy
import torch
import importlib.util
from pathlib import Path

from physicsnemo.utils.generative import InfiniteSampler
from physicsnemo.distributed import DistributedManager

from . import base, cwb, hrrrmini, gefs_hrrr


# this maps all known dataset types to the corresponding init function
known_datasets = {
    "cwb": cwb.get_zarr_dataset,
    "hrrr_mini": hrrrmini.HRRRMiniDataset,
    "gefs_hrrr": gefs_hrrr.HrrrForecastGEFSDataset,
}


def register_dataset(dataset_spec: str) -> None:
    """
    Register a new dataset class from a file path specification.

    Parameters
    ----------
    dataset_spec : str
        String specification in the format "path_to_file.py::dataset_class"

    Raises
    ------
    ValueError
        If the dataset_spec format is invalid or if the file doesn't exist
    ImportError
        If the dataset class cannot be imported
    """
    if dataset_spec in known_datasets:
        return  # Dataset already registered
    try:
        file_path, class_name = dataset_spec.split("::")
    except ValueError:
        raise ValueError(
            "Invalid dataset specification. Expected format: "
            "'path_to_file.py::dataset_class'"
        )
    if class_name in known_datasets:
        return  # Dataset already registered

    # Convert to Path and validate
    file_path = Path(file_path)
    if not file_path.exists():
        raise ValueError(f"Dataset file not found: {file_path}")
    if not file_path.suffix == ".py":
        raise ValueError(f"Dataset file must be a Python file: {file_path}")

    # Import the module and get the class
    spec = importlib.util.spec_from_file_location(file_path.stem, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        dataset_class = getattr(module, class_name)
    except AttributeError:
        raise ImportError(f"Could not find dataset class '{class_name}' in {file_path}")

    # Register the dataset
    known_datasets[dataset_spec] = dataset_class
    return


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
    dataset_sampler = InfiniteSampler(
        dataset=dataset_obj, rank=dist.rank, num_replicas=dist.world_size, seed=seed
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
