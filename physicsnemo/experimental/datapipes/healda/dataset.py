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
"""Map-style dataset combining ERA5 state with observations.

``ObsERA5Dataset`` is the primary dataset class for HealDA training.  It loads
ERA5 analysis state from an xarray DataArray, observations from an
``ObsLoader`` (e.g. ``UFSUnifiedLoader``), and applies a ``Transform``
(e.g. ``ERA5ObsTransform``) to produce training batches.

Temporal windowing, model-parallel rank slicing, and train/test splitting are
handled internally via ``FrameIndexGenerator`` and ``MultiCoordIndex``.

Example usage::

    from physicsnemo.experimental.datapipes.healda.dataset import ObsERA5Dataset
    from physicsnemo.experimental.datapipes.healda.loaders.ufs_obs import UFSUnifiedLoader
    from physicsnemo.experimental.datapipes.healda.transforms.era5_obs import ERA5ObsTransform
    from physicsnemo.experimental.datapipes.healda.configs.variable_configs import VARIABLE_CONFIGS

    obs_loader = UFSUnifiedLoader(
        data_path="/path/to/obs",
        sensors=["atms", "mhs", "conv"],
        obs_context_hours=(-21, 3),
    )
    transform = ERA5ObsTransform(variable_config=VARIABLE_CONFIGS["era5"])

    dataset = ObsERA5Dataset(
        era5_data=era5_xarray["data"],
        obs_loader=obs_loader,
        transform=transform,
        variable_config=VARIABLE_CONFIGS["era5"],
        split="train",
    )
"""

from __future__ import annotations

import asyncio
from typing import Union

import numpy as np
import pandas as pd
import torch

from physicsnemo.experimental.datapipes.healda.indexing import get_flat_indexer
from physicsnemo.experimental.datapipes.healda.loaders.era5 import get_batch_info
from physicsnemo.experimental.datapipes.healda.protocols import ObsLoader, Transform
from physicsnemo.experimental.datapipes.healda.time_utils import as_cftime
from physicsnemo.experimental.datapipes.healda.types import VariableConfig

# HEALPix level-6 pixel count: 12 * 4^6
NPIX_HPX6 = 12 * 4**6

# Default year held out for evaluation
DEFAULT_TEST_YEAR = 2022

# ERA5 time range available in the zarr store
ERA5_TIME_START = "2000-01-01 00:00:00"
ERA5_TIME_END = "2023-10-31 23:00:00"


class ObsERA5Dataset(torch.utils.data.Dataset):
    """Map-style dataset loading ERA5 state + observations.

    Args:
        era5_data: xarray DataArray with dimensions ``(time, variable, pixel)``
            containing the ERA5 state.  Must have a ``"time"`` coordinate.
        obs_loader: Any object implementing the ``ObsLoader`` protocol
            (``async def sel_time(times) -> dict``).
        transform: Any object implementing the ``Transform`` protocol
            (``def transform(times, frames) -> dict``).
        variable_config: ``VariableConfig`` describing the variables and levels.
        split: ``"train"`` (year != ``DEFAULT_TEST_YEAR``),
            ``"test"`` (year == ``DEFAULT_TEST_YEAR``), ``""`` (all),
            or a list of years to include.
        time_length: Number of frames per training window.
        frame_step: Step size between frames (default 1).
        model_rank: Model-parallel rank for time slicing.
        model_world_size: Total model-parallel world size.
    """

    def __init__(
        self,
        era5_data,
        obs_loader: ObsLoader,
        transform: Transform,
        variable_config: VariableConfig,
        *,
        split: Union[str, list[int]] = "",
        time_length: int = 1,
        frame_step: int = 1,
        model_rank: int = 0,
        model_world_size: int = 1,
    ):
        self.variable_config = variable_config
        self.batch_info = get_batch_info(variable_config, time_step=6)

        # Accept either xr.DataArray or xr.Dataset["data"]
        era5 = era5_data
        era5 = era5.sel(time=slice(ERA5_TIME_START, ERA5_TIME_END))
        time = pd.to_datetime(era5["time"].values)

        mask = self._create_time_mask(time, split)
        self._era5 = era5.isel(time=mask)
        self._obs_loader = obs_loader
        self.npix = NPIX_HPX6

        self.time_length = time_length
        self._indexer = get_flat_indexer(
            self._era5,
            [],
            "time",
            time_length=time_length,
            frame_step=frame_step,
            model_rank=model_rank,
            model_world_size=model_world_size,
        )
        self.transform = transform

    @staticmethod
    def _create_time_mask(
        time: pd.DatetimeIndex, split: Union[str, list[int]]
    ) -> np.ndarray:
        """Create a boolean mask for filtering times based on split."""
        if isinstance(split, str):
            mask = {
                "train": time.year != DEFAULT_TEST_YEAR,
                "test": time.year == DEFAULT_TEST_YEAR,
                "": np.ones_like(time, dtype=np.bool_),
            }[split]
        else:
            mask = time.year.isin(split)
        return mask

    def __len__(self):
        return len(self._indexer)

    @property
    def times(self):
        """All available times in the dataset."""
        return pd.to_datetime(self._era5["time"].values)

    def _get_state(self, i):
        coords = self._indexer[i]
        state = self._era5.sel(variable=self.batch_info.channels).isel(coords)
        return state.values

    def _get_times(self, i):
        coords = self._indexer[i]
        state = self._era5.isel(coords)
        return pd.to_datetime(state.time)

    def _get_obs(self, i):
        time = self._get_times(i)
        return asyncio.run(self._obs_loader.sel_time(time))["obs"]

    def get(self, i):
        """Load state + obs for a single sample index.

        Returns:
            ``(times, frames)`` where ``times`` is a list of ``cftime`` objects
            and ``frames`` is a list of dicts with ``"state"`` and ``"obs"``.
        """
        times = [as_cftime(t) for t in self._get_times(i)]
        state = self._get_state(i)
        obs = self._get_obs(i)
        time_per_rank = state.shape[0]
        objs = [{"state": state[t], "obs": obs[t]} for t in range(time_per_rank)]
        return times, objs

    def __getitems__(self, indexes):
        """Batched access — called by DataLoader with batched sampler.

        Loads all samples, then applies the transform to produce the batch dict.
        """
        times, objs = zip(*[self.get(i) for i in indexes])
        return self.transform.transform(times, objs)


def identity_collate(obj):
    """Identity collate function for use with ``ObsERA5Dataset``.

    Since ``__getitems__`` already returns an assembled batch dict, no
    collation is needed.  Pass this as ``collate_fn`` to the DataLoader::

        DataLoader(dataset, sampler=sampler, collate_fn=identity_collate)
    """
    return obj
