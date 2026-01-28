# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
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

import dataclasses
import datetime
from typing import Literal, Optional

import config.environment as config
import numpy as np
import pandas as pd
import torch

from datasets.analysis_loaders import (
    ERA5Loader,
    get_batch_info,
)
from datasets.base import (
    DatasetMetadata,
    TimeUnit,
    VariableConfig,
)
from datasets.filter_times import get_chunk_aligned_times
from datasets.merged_dataset import TimeMergedDataset, TimeMergedMapStyle
from datasets.obs_loader import UFSUnifiedLoader
from datasets.variable_configs import VARIABLE_CONFIGS
from physicsnemo.models.healda import ObsConfig

OBS_INTERVALS = [[-48, -24], [-24, 0], [-21, 3], [-18, 6], [-15, 9]]

_default_config = VARIABLE_CONFIGS["default"]

DATASET_METADATA: dict[str, DatasetMetadata] = {
    "ufs": DatasetMetadata(
        name="ufs",
        start="1994-01-01 00:00:00",  # actual is 1993-12-31 18:00:00, but align to 00/12
        end="2023-10-13 03:00:00",
        time_step=6,
        time_unit=TimeUnit.HOUR,
    ),
    "ufs_obs": DatasetMetadata(
        name="ufs_obs",
        start="2000-01-01 00:00:00",  # actual is 1994-01-01-00:00:00
        end="2023-12-31 18:00:00",
        time_step=6,  # can get 3hr spacing
        time_unit=TimeUnit.HOUR,
    ),
    "era5": DatasetMetadata(
        name="era5",
        start="2000",
        end="2023-10-31 23:00:00",
        time_step=6,
        time_unit=TimeUnit.HOUR,
    ),
}


def frame_dropout(x, p_dropout):
    gate = torch.rand_like(x[..., :1, :1]) < p_dropout
    return x * gate.to(x)


def shift_time(x, shift):
    if shift < 0:
        raise ValueError(f"shift must be non-negative, got {shift}")
    t_dim = -2
    x = x.roll(shift, dims=t_dim)
    x[..., :shift, :] = 0.0
    return x


def _compute_frame_step(
    dataset_spacing: datetime.timedelta, time_step: int, time_length: int
) -> int:
    if time_length == 1:
        return 1

    model_resolution_timedelta = datetime.timedelta(hours=time_step)
    return model_resolution_timedelta // dataset_spacing


def get_label_from_obs_context_hours(obs_context_hours):
    """Map observation context window to a label index for conditioning."""
    if isinstance(obs_context_hours, np.ndarray):
        obs_interval = obs_context_hours.tolist()
    else:
        obs_interval = list(obs_context_hours)

    if obs_interval in OBS_INTERVALS:
        label = OBS_INTERVALS.index(obs_interval)
    else:
        label = 0
    return label


@dataclasses.dataclass
class NullTransform:
    """Placeholder transform."""

    dataset: Literal["era5", "ufs"] = "era5"
    variable_config: VariableConfig = _default_config


def get_sensors_for_config(config: ObsConfig):
    """Return list of sensor names enabled by the ObsConfig."""
    sensors = ["atms", "mhs", "amsua", "amsub"]
    if config.use_infrared:
        sensors.append("iasi")

    if config.use_conv:
        sensors.append("conv")
    return sensors


def _get_ufs_obs_loaders(
    obs_config: ObsConfig,
):
    if obs_config.innovation_type != "none":
        raise ValueError(
            f"innovation_type must be 'none' for UFS obs loaders, "
            f"got '{obs_config.innovation_type}'"
        )

    return [
        UFSUnifiedLoader(
            config.UFS_OBS_PATH,
            sensors=get_sensors_for_config(obs_config),
            obs_context_hours=(obs_config.context_start, obs_config.context_end),
            normalization="zscore",
            filesystem_type="s3"
            if config.UFS_OBS_PATH.startswith("s3://")
            else "local",
            remote_name=config.UFS_OBS_PROFILE,
            drop_obs_channel_ids=obs_config.drop_obs_channel_ids,
            conv_uv_in_situ_only=obs_config.conv_uv_in_situ_only,
            conv_gps_level1_only=obs_config.conv_gps_level1_only,
        )
    ]


def _get_splits(
    dataset: str,
    obs_config: Optional[ObsConfig] = None,
    start_year: Optional[int] = None,
):
    metadata = DATASET_METADATA[dataset]
    valid_times = pd.date_range(metadata.start, metadata.end, freq=metadata.freq)

    if obs_config is not None and obs_config.use_obs:
        obs_metadata = DATASET_METADATA["ufs_obs"]

        if obs_config.innovation_type != "none":
            # ufs obs anl files are missing for these dates
            dropouts = [
                ("2018-12-19", "2020-07-10"),
                ("2022-05-05", "2022-10-01"),
            ]
        else:
            dropouts = []

        aligned_times = get_chunk_aligned_times(
            base_metadata=metadata,
            obs_metadata=obs_metadata,
            dropouts=dropouts,
            chunk_size=24,
        )

        valid_times = aligned_times

    train_times = valid_times[valid_times.year < 2022]
    test_times = valid_times[valid_times.year >= 2022]

    if start_year is not None:
        train_times = train_times[train_times.year >= start_year]

    return {"train": train_times, "test": test_times, "": valid_times}


def get_dataset(
    *,
    dataset: Literal["era5", "ufs"] = "era5",
    split: str = "",
    transform=None,
    variable_config=None,
    rank: int = 0,
    world_size: int = 1,
    model_rank: int = 0,
    model_world_size: int = 1,
    obs_config: Optional[ObsConfig] = None,
    infinite: bool = False,
    shuffle: bool = True,
    chunk_size: int = 8,
    time_step: int = 1,  # in hours
    time_length: int = 1,
    window_stride: int = 1,
    map_style: bool = False,
    batch_transform=None,
    start_year: Optional[int] = None,
) -> torch.utils.data.Dataset:
    """Build dataset for DA training or inference"""
    variable_config = variable_config or VARIABLE_CONFIGS[dataset]

    obs_input = obs_config is not None and obs_config.use_obs

    loaders = []

    if dataset == "era5":
        loaders.append(ERA5Loader(variable_config))
    elif dataset == "ufs":
        raise ValueError(
            "Training with ufs analysis as a target is no longer supported."
        )

    if obs_input:
        loaders.extend(_get_ufs_obs_loaders(obs_config))

    # if transform.background_source == "da":
    #     loaders.append(
    #         PthFileDataset(
    #             config.ERA5_DA_BACKGROUND_PATH,
    #         )
    #     )
    # elif transform.background_source is not None:
    #     raise ValueError(f"Invalid background source: {transform.background_source}")

    # Get the appropriate loaders for the dataset
    times = _get_splits(dataset, obs_config, start_year=start_year)[split]
    if times.size == 0:
        raise RuntimeError("No times are selected.")

    # Compute frame step
    dataset_key = "ufs_obs" if obs_input else dataset
    metadata = DATASET_METADATA[dataset_key]
    meta_time_step = metadata.time_step
    dataset_spacing = metadata.time_unit.to_timedelta(meta_time_step)
    frame_step = _compute_frame_step(dataset_spacing, time_step, time_length)

    # Force map_style for multi-frame validation/inference
    map_style = map_style or (time_length > 1 and split != "train")

    # Create and return the dataset
    if map_style:
        # Used for video validation/inference
        ds = TimeMergedMapStyle(
            times,
            time_loaders=loaders,
            frame_step=frame_step,
            time_length=time_length,
            cache_chunk_size=chunk_size,
            batch_transform=batch_transform,
            transform=transform,
            model_rank=model_rank,
            model_world_size=model_world_size,
        )
    else:
        ds = TimeMergedDataset(
            times,
            time_loaders=loaders,
            # transform=transform.transform,
            transform=transform,
            rank=rank,
            world_size=world_size,
            infinite=infinite,
            shuffle=shuffle,
            chunk_size=chunk_size,
            frame_step=frame_step,
            time_length=time_length,
            window_stride=window_stride,
        )

    ds.batch_info = get_batch_info(
        # config=transform.variable_config,
        config=variable_config,
        time_step=time_step,
        time_unit=TimeUnit.HOUR,
        # background_source=transform.background_source,
    )
    ds.calendar = "standard"
    ds.time_units = "seconds since 1970-1-1 0:0:0"
    return ds
