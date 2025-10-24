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


import pytest
from pytest_utils import import_or_fail


@import_or_fail("h5py")
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_dataloader_setup(device, pytestconfig):
    from physicsnemo.datapipes.climate import (
        SyntheticWeatherDataLoader,
        SyntheticWeatherDataset,
    )

    dataloader = SyntheticWeatherDataLoader(
        channels=[0, 1, 2, 3],
        num_samples_per_year=12,
        num_steps=5,
        device=device,
        grid_size=(8, 8),
        batch_size=3,
        shuffle=True,
        num_workers=2,
    )
    """Test the SyntheticWeatherDataLoader setup including batch size and shuffle configuration."""
    assert dataloader.batch_size == 3
    assert dataloader.num_workers == 2
    assert isinstance(dataloader.dataset, SyntheticWeatherDataset)


@import_or_fail("h5py")
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_dataloader_iteration(device, pytestconfig):
    """Test the iteration over batches in the DataLoader."""

    from physicsnemo.datapipes.climate import (
        SyntheticWeatherDataLoader,
    )

    dataloader = SyntheticWeatherDataLoader(
        channels=[0, 1],
        num_samples_per_year=30,
        num_steps=4,
        device=device,
        grid_size=(24, 24),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    for batch in dataloader:
        assert isinstance(batch, list)
        sample = batch[0]
        assert "invar" in sample
        assert "outvar" in sample
        assert sample["invar"].shape == (dataloader.batch_size, 2, 24, 24)
        assert sample["outvar"].shape == (dataloader.batch_size, 4, 2, 24, 24)
        assert sample["invar"].device.type == device
        assert sample["outvar"].device.type == device
        break  # Only test one batch for quick testing


@import_or_fail("h5py")
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_dataloader_length(device, pytestconfig):
    """Test the length of the DataLoader to ensure it is correct based on the dataset and batch size."""

    from physicsnemo.datapipes.climate import (
        SyntheticWeatherDataLoader,
    )

    dataloader = SyntheticWeatherDataLoader(
        channels=[0, 1, 2],
        num_samples_per_year=30,
        num_steps=2,
        device=device,
        grid_size=(10, 10),
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )
    expected_length = (30 - 2) // 4  # dataset length divided by batch size
    assert len(dataloader) == expected_length
