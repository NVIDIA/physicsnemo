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

import dataclasses
import enum
import warnings
from typing import TypedDict

import pandas as pd
import torch
from datasets.dataset import VARIABLE_CONFIGS
from datasets.dataset import (
    get_dataset as get_dataset_ufs,
)
from datasets.transform import TransformV2
from torch.utils.data import Dataset
from utils import distributed as dist
from utils.checkpointing import Checkpoint
from utils.dataclass_parser import Help, a

from physicsnemo.models.healda import ObsConfig, UnifiedObservation


class Rolling(Dataset):
    """Returns window_size consecutive frames from dataset."""

    def __init__(self, dataset, window_size, stride=1, step=1):
        self.dataset = dataset
        self.window_size = window_size
        self.stride = stride
        self.step = step
        self.max_start = len(dataset) - (window_size - 1) * step
        if self.max_start <= 0:
            raise ValueError("Dataset too small for given window size and step.")

    def __len__(self):
        return (self.max_start + self.stride - 1) // self.stride

    @property
    def times(self):
        return self.dataset.times[: self.max_start : self.stride]

    def __getitem__(self, idx):
        start = idx * self.stride
        indices = [start + i * self.step for i in range(self.window_size)]
        return [self.dataset[i] for i in indices]


class Batch(TypedDict):
    """Input batch structure of DA model"""

    target: torch.Tensor
    condition: torch.Tensor
    second_of_day: torch.Tensor
    day_of_year: torch.Tensor
    labels: torch.Tensor
    timestamp: torch.Tensor
    unified_obs: UnifiedObservation


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


class InnovationType(enum.Enum):
    """Observation-minus-background (innovation) type"""

    NONE = "none"
    ADJUSTED = "adjusted"
    UNADJUSTED = "unadjusted"


class SaveMode(enum.Enum):
    """Controls which data to save during inference."""

    DATA = "data"  # Save only ground truth
    INFERENCE = "inference"  # Save only model outputs
    ALL = "all"  # Save both


@dataclasses.dataclass
class DAConfig:
    checkpoint_path: a[str, Help("Path to the checkpoint file")]
    output_path: a[str, Help("Output NetCDF file path")] = "da_regression_output"
    dataset: a[str, Help("Dataset to use (ufs or era5)")] = "ufs"
    innovation_type: InnovationType = InnovationType.NONE
    num_samples: a[int, Help("Number of samples to generate")] = 32
    time_frequency: a[str, Help("Spacing to sample times from")] = "12h"
    use_infrared: a[bool, Help("Use infrared")] = False
    use_conv: a[bool, Help("Use conv")] = True
    context_start: a[int, Help("Context start (hours)")] = -21
    context_end: a[int, Help("Context end (hours)")] = 3
    batch_gpu: a[int, Help("Batch size")] = 8
    z06_18_inits: a[bool, Help("Use 06z and 18z inits instead of 00z/12z")] = False
    save_mode: SaveMode = SaveMode.ALL
    use_analysis: bool = False
    conv_uv_in_situ_only: bool = False
    conv_gps_level1_only: bool = False
    post_process_to_fcn3: bool = False
    blend: float | None = None  # weight of forecast defaults to 0
    use_class_labels: bool = False
    use_12hr_residual_stats: bool = False
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
        with Checkpoint(args.checkpoint_path, mode="r") as ckpt:
            model = ckpt.read_model(map_location="cuda")
        model.eval().cuda()

        self.model = model
        self.split = args.split
        # Inference time obs config for the dataset.
        # channel/emb dims are for the model and do not matter here
        self.obs_config = ObsConfig(
            use_obs=True,  # Always use observations for DA
            innovation_type=args.innovation_type.value,
            context_start=args.context_start,
            context_end=args.context_end,
            use_infrared=args.use_infrared,
            use_conv=args.use_conv,
            conv_uv_in_situ_only=args.conv_uv_in_situ_only,
            conv_gps_level1_only=args.conv_gps_level1_only,
        )
        self._batch_info = None
        self.use_class_labels = args.use_class_labels
        self.use_12hr_residual_stats = args.use_12hr_residual_stats

        self.variable_config = (
            VARIABLE_CONFIGS["default"]
            if args.dataset == "ufs"
            else VARIABLE_CONFIGS["era5"]
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

        transform = TransformV2(
            variable_config=self.variable_config,
        )

        ds = get_dataset_ufs(
            dataset=args.dataset,
            batch_transform=transform.transform,
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
        batch = _to_batch(batch, self.device)
        target = batch["target"]
        b, c, t, x = target.shape
        noise_labels = torch.zeros([b], device=target.device)
        class_labels = batch["labels"]

        if not self.use_class_labels:
            class_labels = torch.empty([b, 0], device=target.device)

        condition = batch["condition"]
        obs = batch["unified_obs"]
        # TODO generalize this logic in train_regression.py:_step it is like this
        # time_step = self.time_step * 3600
        time_step = 0
        timestamp = (
            batch["timestamp"].unsqueeze(1)
            + torch.arange(t, device=target.device) * time_step
        )
        second_of_day = batch["second_of_day"]
        day_of_year = batch["day_of_year"]

        # Get predictions
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return {
                "timestamp": timestamp,
                "second_of_day": second_of_day,
                "day_of_year": day_of_year,
                "target": self.model(
                    condition,
                    noise_labels=noise_labels,
                    class_labels=class_labels,
                    second_of_day=second_of_day,
                    day_of_year=day_of_year,
                    unified_obs=obs,
                    timestamp=timestamp,
                ).out,
            }


def time_length(batch):
    return batch["target"].shape[2]


def find_matching_indices(targets, available):
    indices = available.get_indexer(targets)
    valid = indices != -1
    return indices[valid], available[indices[valid]]


def enumerate_to_dict(array):
    return {int(val): i for i, val in enumerate(array)}


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
    import numpy as np
    import zarr

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


def post_process_to_fcn3(da_zarr_path: str, batch_info, pixel_order=None):
    """Convert HPX zarr to FCN3-compatible lat-lon format.

    Saves a fcn3 compatible zarr file, renaming and regridding fields. Saves to
    original path with "_fcn3" suffix.

    Args:
        da_zarr_path: Path to input HPX zarr file
        batch_info: BatchInfo object with channel names and normalization
        pixel_order: HEALPix pixel ordering (default: HEALPIX_PAD_XY)

    Returns:
        Path to output fcn3 zarr file
    """
    import earth2grid
    import numpy as np
    import torch
    import xarray as xr
    import zarr
    from earth2grid import healpix

    if pixel_order is None:
        pixel_order = earth2grid.healpix.HEALPIX_PAD_XY

    da = xr.open_zarr(da_zarr_path)

    out_map = {
        "tcwv": "tcwv",
        "tas": "t2m",
        "uas": "u10m",
        "vas": "v10m",
        "100u": "u100m",
        "100v": "v100m",
        "pres_msl": "msl",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- rename and convert to torch ----
    field_dict = {}
    for c, name in enumerate(batch_info.channels):
        arr_torch = torch.from_numpy(da[name].values.astype(np.float32)).to(device)
        if name in out_map:
            out_name = out_map[name]
        else:
            out_name = name.lower()
        field_dict[out_name] = arr_torch

    # ---- HPX -> lat-lon regridder ----
    hpx_grid = healpix.Grid(level=6, pixel_order=pixel_order)
    ll_grid = earth2grid.latlon.equiangular_lat_lon_grid(nlat=721, nlon=1440)
    regridder = earth2grid.get_regridder(hpx_grid, ll_grid).to(torch.float32).to(device)

    # ---- out zarr ----
    out_store = da_zarr_path.rstrip("/").split(".zarr")[0] + "_fcn3.zarr"
    g = zarr.open_group(out_store, mode="w")
    print(f"out_store: {out_store}")

    # get times from da
    times_array = da["time"].values
    time_v = g.create_array(
        "time",
        dtype=np.int64,
        shape=times_array.shape,
        chunks=times_array.shape,
        dimension_names=["time"],
    )
    time_v[:] = times_array.astype("datetime64[s]").astype(np.int64)
    time_v.attrs["units"] = "seconds since 1970-01-01 00:00:00"
    time_v.attrs["calendar"] = "standard"

    lat_arr = np.asarray(ll_grid.lat).squeeze()
    if lat_arr.ndim != 1:  # handles (721,1)
        lat_arr = lat_arr[:, 0]

    lon_arr = np.asarray(ll_grid.lon).squeeze()
    if lon_arr.ndim != 1:  # handles (1,1440)
        lon_arr = lon_arr[0, :]

    lat_v = g.create_array(
        "lat",
        dtype=np.float32,
        shape=lat_arr.shape,
        chunks=lat_arr.shape,
        dimension_names=["lat"],
    )
    lat_v[:] = lat_arr

    lon_v = g.create_array(
        "lon",
        dtype=np.float32,
        shape=lon_arr.shape,
        chunks=lon_arr.shape,
        dimension_names=["lon"],
    )
    lon_v[:] = lon_arr

    src_dims = da["tcwv"].dims

    for name, arr in field_dict.items():
        # flatten all leading dims, regrid each slice
        lead_shape = arr.shape[:-1]
        out = regridder(arr).cpu().numpy()

        dim_names = tuple(src_dims[:-1]) + ("lat", "lon")
        chunks = (*((1,) * len(lead_shape)), 721, 1440)

        g.create_array(
            name,
            shape=out.shape,
            dtype="f",
            chunks=chunks,
            fill_value=np.float32("nan"),
            dimension_names=dim_names,
            compressors=None,
            overwrite=True,
        )[:] = out

    zarr.consolidate_metadata(g.store)

    return out_store
