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

import dataclasses
import warnings

import numpy as np
import pandas as pd
import torch
import zarr
from datasets.dataset import (
    VARIABLE_CONFIGS,
    get_dataset as get_dataset_ufs,
    get_sensors_for_config,
)
from datasets.transform import TransformV2
from utils import distributed as dist
from utils.dataclass_parser import Help, a
from physicsnemo.experimental.models.healda import HealDA
from config.model_config import ObsConfig


warnings.filterwarnings("ignore", message="Cannot do a zero-copy NCHW to NHWC.")


# Copied from training loop
def _to_batch(x, device, non_blocking=True):
    if isinstance(x, dict):
        return {
            k: _to_batch(v, device, non_blocking=non_blocking) for k, v in x.items()
        }
    elif isinstance(x, list):
        return [_to_batch(i, device, non_blocking=non_blocking) for i in x]
    elif torch.is_tensor(x):
        if torch.is_floating_point(x):
            x = x.float()
        return x.to(device, non_blocking=non_blocking)
    elif hasattr(x, "to") and callable(getattr(x, "to")):
        # custom object with a 'to' method
        return x.to(device, non_blocking=non_blocking)
    else:
        raise NotImplementedError(x)


@dataclasses.dataclass
class DAConfig:
    checkpoint_path: a[str, Help("Path to the .mdlus checkpoint file")]
    output_path: a[str, Help("Output zarr file path")] = "healda_analysis.zarr"
    dataset: a[str, Help("Dataset to use (ufs or era5)")] = "era5"
    num_samples: a[int, Help("Number of samples (-1 for all)")] = 32
    time_frequency: a[str, Help("Spacing to sample times from")] = "6h"
    use_infrared: a[bool, Help("Use infrared observations")] = False
    use_conv: a[bool, Help("Use conventional observations")] = True
    context_start: a[int, Help("Obs window start (hours before analysis)")] = -21
    context_end: a[int, Help("Obs window end (hours after analysis)")] = 3
    batch_gpu: a[int, Help("Batch size per GPU")] = 8
    z06_18_inits: a[bool, Help("Use 06z/18z inits instead of 00z/12z")] = False
    use_analysis: a[
        bool,
        Help(
            "Save out ground truth target dataset instead of predicted HealDA analysis"
        ),
    ] = False
    conv_uv_in_situ_only: a[bool, Help("Exclude satellite UV (keep in-situ)")] = False
    conv_gps_level1_only: a[bool, Help("Exclude GPS T/Q (keep bending angle)")] = False
    use_class_labels: a[bool, Help("Use class labels for conditioning")] = False
    split: a[str, Help("Test (2022) or Train (2021)")] = "test"


def scoring_times(
    z06_z18_inits: bool, time_frequency, split: str = "test"
) -> pd.DatetimeIndex:
    year = 2022 if split == "test" else 2021
    start_date = f"{year}-01-01-00" if not z06_z18_inits else f"{year}-01-01-06"
    return pd.date_range(start_date, f"{year}-12-31-12", freq=time_frequency)


class DAModel:
    def __init__(self, args: DAConfig):
        self.args = args

        model = HealDA.from_checkpoint(args.checkpoint_path)
        model.cuda().eval()

        self.model = model
        self.split = args.split
        # Inference time obs config for the dataset.
        # channel/emb dims are for the model and do not matter here
        self.obs_config = ObsConfig(
            use_obs=True,  # Always use observations for DA
            context_start=args.context_start,
            context_end=args.context_end,
            use_infrared=args.use_infrared,
            use_conv=args.use_conv,
            conv_uv_in_situ_only=args.conv_uv_in_situ_only,
            conv_gps_level1_only=args.conv_gps_level1_only,
        )
        self._batch_info = None
        self.use_class_labels = args.use_class_labels

        self.variable_config = (
            VARIABLE_CONFIGS["default"]
            if args.dataset == "ufs"
            else VARIABLE_CONFIGS["era5"]
        )
        self.sensor_names = list(self.model.obs_embedder.sensor_names)
        configured_sensors = get_sensors_for_config(self.obs_config)
        if configured_sensors != self.sensor_names:
            raise ValueError(
                "Observation config sensors do not match the checkpoint sensor order. "
                f"Configured sensors: {configured_sensors}. "
                f"Checkpoint sensors: {self.sensor_names}."
            )
        self.transform = TransformV2(
            variable_config=self.variable_config,
            sensors=self.sensor_names,
        )

    def get_dataset(
        self,
        split="test",
        time_length: int = 1,
        time_step: int = 12,
        map_style: bool = True,
        chunk_size: int = 0,
    ):
        args = self.args

        dist.print0(f"Loading {args.dataset} dataset...")

        ds = get_dataset_ufs(
            dataset=args.dataset,
            batch_transform=self.transform.transform,
            split=split,
            shuffle=False,
            obs_config=self.obs_config,
            map_style=map_style,
            time_length=time_length,
            time_step=time_step,
            chunk_size=chunk_size,
        )
        self._batch_info = ds.batch_info
        return ds

    @property
    def batch_info(self):
        if self._batch_info is None:
            self.get_dataset(split=self.split)
        return self._batch_info

    @property
    def device(self):
        return next(self.model.parameters()).device

    def get_state(self, batch):
        if isinstance(batch.get("obs"), tuple):
            batch = self.transform.device_transform(batch, self.device)
        else:
            batch = _to_batch(batch, self.device)
        target = batch["target"]
        b = target.shape[0]
        noise_labels = torch.zeros([b], device=target.device)
        class_labels = batch["labels"]

        if not self.use_class_labels:
            class_labels = torch.empty([b, 0], device=target.device)

        condition = batch["condition"]
        obs = batch["obs"]
        timestamp = batch["timestamp"]
        second_of_day = batch["second_of_day"]
        day_of_year = batch["day_of_year"]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            return {
                "timestamp": timestamp,
                "second_of_day": second_of_day,
                "day_of_year": day_of_year,
                "target": self.model(
                    condition,
                    noise_labels,
                    **obs,
                    second_of_day=second_of_day,
                    day_of_year=day_of_year,
                    class_labels=class_labels,
                ),
            }


def write_to_zarr(group, channels, index, data):
    """Write denormalized predictions to zarr arrays.

    Args:
        group: Zarr group to write to
        channels: List of channel names
        index: Time indices to write to
        data: Data array with shape (batch, channels, 1, cells)
    """
    for c in range(len(channels)):
        name = channels[c]
        array = data[:, c, 0, :]
        group[name][index] = array


def setup_zarr_output(
    output_path, channels, num_times, batch_size, subsampled_times=None
):
    """Setup zarr output structure for inference results.

    Args:
        output_path: Path to output zarr file
        channels: List of channel names
        num_times: Number of time steps
        batch_size: Batch size for chunking
        subsampled_times: Optional pandas DatetimeIndex for time coordinate

    Returns:
        Opened zarr group (mode='w')
    """

    group = zarr.open_group(output_path, mode="w")

    # Create data arrays for each channel
    for field in channels:
        group.create_array(
            field,
            shape=(num_times, 49152),
            chunks=(batch_size, 49152),
            fill_value=float("NaN"),
            dimension_names=("time", "cells"),
            dtype="f",
            compressors=[],
        )

    # Create time coordinate if provided
    if subsampled_times is not None:
        times_array = subsampled_times.to_numpy()
        time_v = group.create_array(
            "time",
            dtype=np.int64,
            shape=times_array.shape,
            chunks=times_array.shape,
            dimension_names=["time"],
        )
        time_v[:] = times_array.astype("datetime64[s]").astype(np.int64)
        time_v.attrs["units"] = "seconds since 1970-01-01 00:00:00"
        time_v.attrs["calendar"] = "standard"

    zarr.consolidate_metadata(group.store)
    return group
