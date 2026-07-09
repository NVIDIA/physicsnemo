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

"""Shared fixtures/helpers for the test_healpix*.py suite (classic/Zarr,
coupled/uncoupled).

These describe the same on-disk NFS test dataset and the same
scaling/coupling configuration shapes across all four test_healpix*.py
modules in this package, so they're centralized here instead of being
duplicated (with minor, incidental drift) in each file.
"""

from dataclasses import dataclass

import pytest


@pytest.fixture
def data_dir(nfs_data_dir):
    """Directory of the classic (non-Zarr) prebuilt test dataset."""
    return nfs_data_dir.joinpath("datasets/healpix")


@pytest.fixture
def dataset_name():
    return "healpix"


@pytest.fixture
def create_path(nfs_data_dir):
    return nfs_data_dir.joinpath("datasets/healpix/merge")


@pytest.fixture
def dataset_path(nfs_data_dir):
    """Path to the Zarr-backed version of the same test dataset."""
    return nfs_data_dir.joinpath("datasets/healpix/healpix.zarr")


@pytest.fixture
def splits():
    """Date ranges that fall within the small test dataset's actual time
    range, suitable for exercising real dataloader construction (as opposed
    to just checking that arbitrary split values are accepted).
    """
    return {
        "train_date_start": "1979-01-01",
        "train_date_end": "1979-01-01T21:00",
        "val_date_start": "1979-01-02",
        "val_date_end": "1979-01-02T09:00",
        "test_date_start": "1979-01-02T12:00",
        "test_date_end": "1979-01-02T18:00",
    }


@pytest.fixture
def scaling_dict():
    omegaconf = pytest.importorskip("omegaconf")
    scaling = {
        "t2m0": {"mean": 287.8665771484375, "std": 14.86227798461914},
        "t850": {"mean": 281.2710266113281, "std": 12.04991626739502},
        "tau300-700": {"mean": 61902.72265625, "std": 2559.8408203125},
        "tcwv0": {"mean": 24.034976959228516, "std": 16.411935806274414},
        "z1000": {"mean": 952.1435546875, "std": 895.7516479492188},
        "z1000-12h": {"mean": 952.1435546875, "std": 895.7516479492188},
        "z250": {"mean": 101186.28125, "std": 5551.77978515625},
        "z500": {"mean": 55625.9609375, "std": 2681.712890625},
        "lsm": {"mean": 0, "std": 1},
        "z": {"mean": 0, "std": 1},
        "tp6": {"mean": 1, "std": 0, "log_epsilon": 1e-6},
        "extra": {"mean": 0, "std": 0},  # doesn't appear in test dataset
    }
    return omegaconf.DictConfig(scaling)


@pytest.fixture
def scaling_double_dict():
    omegaconf = pytest.importorskip("omegaconf")
    scaling = {
        "t2m0": {"mean": 0, "std": 2},
        "t850": {"mean": 0, "std": 2},
        "tau300-700": {"mean": 0, "std": 2},
        "tcwv0": {"mean": 0, "std": 2},
        "z1000": {"mean": 0, "std": 2},
        "z1000-12h": {"mean": 0, "std": 2},
        "z250": {"mean": 0, "std": 2},
        "z500": {"mean": 0, "std": 2},
        "tp6": {"mean": 0, "std": 2, "log_epsilon": 1e-6},
        "lsm": {"mean": 0, "std": 2},
        "z": {"mean": 0, "std": 2},
        "extra": {"mean": 0, "std": 2},  # doesn't appear in test dataset
    }
    return omegaconf.DictConfig(scaling)


@dataclass
class coupler_helper:
    """Stand-in for a coupled module, exposing just what setup_coupling()
    needs from it (output_variables, time_step)."""

    output_variables: list
    time_step: str


@pytest.fixture
def constant_coupler_config():
    """A single-variable ConstantCoupler coupling config, in the list-of-dict
    shape consumed by Coupled*Dataset(Zarr)/*DataModule(Zarr) `couplings=`."""
    return [
        {
            "coupler": "ConstantCoupler",
            "params": {
                "batch_size": 1,
                "variables": ["z250"],
                "input_times": ["0h"],
                "input_time_dim": 1,
                "output_time_dim": 1,
                "presteps": 0,
                "prepared_coupled_data": True,
            },
        }
    ]


@pytest.fixture
def average_coupler_config():
    """A single-variable TrailingAverageCoupler coupling config, in the same
    shape as `constant_coupler_config` but for a different coupler class."""
    return [
        {
            "coupler": "TrailingAverageCoupler",
            "params": {
                "batch_size": 1,
                "variables": ["z250"],
                "input_times": ["6h"],
                "averaging_window": "6h",
                "input_time_dim": 1,
                "output_time_dim": 1,
                "presteps": 0,
                "prepared_coupled_data": True,
            },
        }
    ]


def assert_shard_dataloaders(timeseries_dm):
    """Shared assertions for `*DataModule(Zarr).{train,val,test}_dataloader`:
    a single shard should get no distributed sampler, while multiple shards
    should. Used identically across the classic/Zarr, coupled/uncoupled data
    module tests.
    """
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    train_dataloader, train_sampler = timeseries_dm.train_dataloader(num_shards=1)
    assert train_sampler is None
    assert isinstance(train_dataloader, DataLoader)

    val_dataloader, val_sampler = timeseries_dm.val_dataloader(num_shards=1)
    assert val_sampler is None
    assert isinstance(val_dataloader, DataLoader)

    test_dataloader, test_sampler = timeseries_dm.test_dataloader(num_shards=1)
    assert test_sampler is None
    assert isinstance(test_dataloader, DataLoader)

    train_dataloader, train_sampler = timeseries_dm.train_dataloader(num_shards=2)
    assert isinstance(train_sampler, DistributedSampler)
    assert isinstance(train_dataloader, DataLoader)

    val_dataloader, val_sampler = timeseries_dm.val_dataloader(num_shards=2)
    assert isinstance(val_sampler, DistributedSampler)
    assert isinstance(val_dataloader, DataLoader)

    test_dataloader, test_sampler = timeseries_dm.test_dataloader(num_shards=2)
    assert isinstance(test_sampler, DistributedSampler)
    assert isinstance(test_dataloader, DataLoader)
