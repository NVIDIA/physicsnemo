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
import functools
import warnings

import cftime
import earth2grid
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch

from datasets import features, static_data
from datasets.analysis_loaders import (
    get_batch_info,
)
from datasets.base import (
    VariableConfig,
)
from datasets.variable_configs import VARIABLE_CONFIGS
from physicsnemo.models.healda import profiling, types

warnings.filterwarnings(
    "ignore",
    message="The given NumPy array is not writable, and PyTorch does not support non-writable tensors",
)

# Column names required by the encode function
ENCODE_REQUIRED_COLUMNS = [
    "Latitude",
    "Longitude",
    "Absolute_Obs_Time",
    "Platform_ID",
    "Observation_Type",
    "Observation",
    "Global_Channel_ID",
    "Sat_Zenith_Angle",
    "Sol_Zenith_Angle",
    "sensor_id",
    "local_channel_id",
]

# Optional column names that are checked for existence in encode function
ENCODE_OPTIONAL_COLUMNS = [
    "Height",
    "Pressure",
    "Scan_Angle",
]

# All column names (required + optional)
ENCODE_ALL_COLUMNS = ENCODE_REQUIRED_COLUMNS + ENCODE_OPTIONAL_COLUMNS


def _cftime_to_timestamp(time: cftime.DatetimeGregorian) -> float:
    return datetime.datetime(
        *cftime.to_tuple(time), tzinfo=datetime.timezone.utc
    ).timestamp()


def _reorder_nest_to_hpxpad(x):
    x = torch.as_tensor(x)
    src_order = earth2grid.healpix.NEST
    dst_order = earth2grid.healpix.HEALPIX_PAD_XY
    return earth2grid.healpix.reorder(x, src_order, dst_order)


def _compute_second_of_day(time: cftime.datetime):
    day_start = time.replace(hour=0, minute=0, second=0)
    return (time - day_start) / datetime.timedelta(seconds=1)


def _compute_day_of_year(time: cftime.datetime):
    day_start = time.replace(hour=0, minute=0, second=0)
    year_start = day_start.replace(month=1, day=1)
    return (time - year_start) / datetime.timedelta(seconds=86400)


def _compute_timestamp(time: cftime.datetime):
    return int(_cftime_to_timestamp(time))


def _get_static_condition(HPX_LEVEL, variable_config) -> torch.Tensor:
    lfrac = static_data.load_lfrac(HPX_LEVEL)
    orography = static_data.load_orography()
    # insert land mask
    orog_scale, orog_mean = 627.3885284872, 232.56013904090733
    lfrac_scale, lfrac_mean = 0.4695501683565522, 0.3410480857539571
    data = {
        "orog": (orography - orog_mean) / orog_scale,
        "lfrac": (lfrac - lfrac_mean) / lfrac_scale,
    }
    arrays = [torch.as_tensor(data[name]) for name in variable_config.variables_static]
    array = torch.stack(arrays).float()  # c x
    return array.unsqueeze(1)


@dataclasses.dataclass
class TransformV2:
    """Batch transform for normalizing state data and preparing observations for training.

    Two-stage pipeline:
        1. ``transform(times, frames)`` - CPU preprocessing, returns intermediate dict
        2. ``device_transform(batch, device)`` - GPU transfer and featurization

    Stage 1 - ``transform()`` returns dict with:
        - ``target``: Normalized state tensor (B, C, T, X).
        - ``unified_obs``: Tuple of (obs_tensors, offsets_3d, sensor_id_to_local).
        - ``condition``: Static conditioning features (1, C_cond, X).
        - ``second_of_day``, ``day_of_year``, ``timestamp``: Time encodings (B, T).

    Intermediate ``unified_obs`` tuple structure:
        - ``obs_tensors``: Dict of 1D tensors (N_obs,) - latitude, longitude,
          observation, global_channel_id, sensor_id, platform_id, etc.
        - ``offsets_3d``: Shape (S, B, T) cumulative end indices per sensor/batch/time.
        - ``sensor_id_to_local``: Maps sensor_id to local index in offsets_3d.

    Stage 2 - ``device_transform()`` converts ``unified_obs`` to ``UnifiedObservation``:
        - Moves tensors to GPU
        - Computes ``float_metadata`` via ``features.compute_unified_metadata()``
          (encodes lat/lon, time deltas, zenith angles, etc.)
        - Builds ``int_metadata`` tensor (N_obs, 6) with columns:
          [sensor_id, hpx_pixel, local_channel, platform_id, obs_type, global_channel]
        - Returns ``types.UnifiedObservation`` dataclass ready for model input.

    Sensor grouping: Observations sorted by sensor_id with offsets_3d enabling
    efficient (sensor, batch, time) slicing (see ``split_by_sensor`` in
    ``physicsnemo.models.healda.types``).
    """

    variable_config: VariableConfig = VARIABLE_CONFIGS["era5"]
    hpx_level: int = 10  # pixel level of the observations
    hpx_level_condition: int = 6

    def __post_init__(self):
        batch_info = get_batch_info(self.variable_config)

        self.mean = np.array(batch_info.center)[:, None]
        self.std = np.array(batch_info.scales)[:, None]

    @functools.cached_property
    def _grid(self):
        return earth2grid.healpix.Grid(
            self.hpx_level, pixel_order=earth2grid.healpix.NEST
        )

    @staticmethod
    def _sort_by_record_batch(table: pa.Table, column_name: str) -> pa.Table:
        """
        Sort PyArrow table by grouping record batches by a column value.
        Assumes all rows in the batch have the same value for the column name.
        """
        record_batches_order = []
        for batch in table.to_batches():
            if batch.num_rows == 0:
                continue
            group_value = batch[column_name][0]
            record_batches_order.append((group_value, batch))

        # in empty case, from_batches will raise an error
        if not record_batches_order:
            return table

        record_batches_order.sort(key=lambda x: x[0].as_py())
        return pa.Table.from_batches([batch for _, batch in record_batches_order])

    @staticmethod
    def _append_batch_time_info_chunked(
        table: pa.Table, b: int, t: int, timestamp: int
    ) -> pa.Table:
        """
        Add batch/time indices and target time while maintaining original chunking.
        """
        b_idx_type = pa.int16()
        t_idx_type = pa.int16()
        time_type = pa.int64()

        ref_col = table.column(0)

        b_idx_chunks = []
        t_idx_chunks = []
        time_chunks = []

        for chunk in ref_col.chunks:
            L = len(chunk)
            if L == 0:
                b_idx_chunks.append(pa.array([], type=b_idx_type))
                t_idx_chunks.append(pa.array([], type=t_idx_type))
                time_chunks.append(pa.array([], type=time_type))
                continue

            # directly creating pa array of int16 not supported so use np first
            b_idx_arr = np.full(L, b, dtype=np.int16)
            t_idx_arr = np.full(L, t, dtype=np.int16)
            times_arr = np.full(L, timestamp, dtype=np.int64)

            b_idx_chunks.append(pa.array(b_idx_arr, type=b_idx_type))
            t_idx_chunks.append(pa.array(t_idx_arr, type=t_idx_type))
            time_chunks.append(pa.array(times_arr, type=time_type))

        b_chunked = pa.chunked_array(b_idx_chunks, type=b_idx_type)
        t_chunked = pa.chunked_array(t_idx_chunks, type=t_idx_type)
        time_chunked = pa.chunked_array(time_chunks, type=time_type)

        out = table.append_column("batch_idx", b_chunked)
        out = out.append_column("time_idx", t_chunked)
        out = out.append_column("target_time", time_chunked)
        return out

    @staticmethod
    def _build_observation_offsets_3d(obs_table: pa.Table, frame_times):
        B, T = len(frame_times), len(frame_times[0])

        sensor_ids = set()
        counts_map = {}  # sensor_id -> (b, t) array of num obs in that sensor, batch, time

        for batch in obs_table.to_batches():
            if batch.num_rows == 0:
                continue

            s_id = int(batch["sensor_id"][0].as_py())
            b_id = int(batch["batch_idx"][0].as_py())
            t_id = int(batch["time_idx"][0].as_py())
            n = batch.num_rows

            sensor_ids.add(s_id)

            if s_id not in counts_map:
                counts_map[s_id] = torch.zeros((B, T), dtype=torch.int32)

            counts_map[s_id][b_id, t_id] += n

        # build per-sensor cumulative ends in row-major (b,t) order
        # use only active sensors and maintain map, as all possible sensor ids unknown here
        active_sensor_ids = sorted(sensor_ids)
        S = len(active_sensor_ids)

        # Handle empty case: no observations in entire batch
        if not sensor_ids:
            offsets_3d = torch.zeros((0, B, T), dtype=torch.int32)
            sensor_id_to_local = torch.zeros((0,), dtype=torch.int32)
            return offsets_3d, sensor_id_to_local

        max_sensor_id = max(sensor_ids)

        offsets_3d = torch.zeros((S, B, T), dtype=torch.int32)

        prev_count = 0
        for s_local, s_id in enumerate(active_sensor_ids):
            counts_bt = counts_map[s_id]  # [B,T]
            flat_counts = counts_bt.reshape(-1)  # len = B*T
            flat_cumsum = torch.cumsum(flat_counts, dim=0)  # cumulative ends
            offsets_3d[s_local] = prev_count + flat_cumsum.reshape(B, T)
            prev_count += flat_cumsum[-1].item()

        # Create sensor_id -> local_idx map
        sensor_id_to_local = torch.full((max_sensor_id + 1,), -1, dtype=torch.int32)
        for local_idx, sensor_id in enumerate(active_sensor_ids):
            sensor_id_to_local[sensor_id] = local_idx

        return offsets_3d, sensor_id_to_local

    @profiling.nvtx
    def _process_obs(self, target_times: list[list[cftime.datetime]], frames):
        # Add batch and time indices to each table before concatenation
        all_obs_with_indices = []
        for b_idx, sample_frames in enumerate(frames):
            for t_idx, frame_dict in enumerate(sample_frames):
                table = frame_dict["obs_v2"]
                table_with_indices = self._append_batch_time_info_chunked(
                    table,
                    b_idx,
                    t_idx,
                    _compute_timestamp(target_times[b_idx][t_idx]),
                )
                all_obs_with_indices.append(table_with_indices)

        obs = pa.concat_tables(all_obs_with_indices)

        obs = self._sort_by_record_batch(obs, "sensor_id")

        offsets_3d, sensor_id_to_local = self._build_observation_offsets_3d(
            obs, target_times
        )

        # Extract columns into dictionary of torch tensors
        obs_tensors = {}

        # Required columns mapping
        required_columns = {
            "latitude": "Latitude",
            "longitude": "Longitude",
            "observation": "Observation",
            "global_channel_id": "Global_Channel_ID",
            "sat_zenith_angle": "Sat_Zenith_Angle",
            "sol_zenith_angle": "Sol_Zenith_Angle",
            "sensor_id": "sensor_id",
            "local_channel_id": "local_channel_id",
            "height": "Height",
            "pressure": "Pressure",
            "scan_angle": "Scan_Angle",
        }

        # Process required columns
        for tensor_key, column_name in required_columns.items():
            obs_tensors[tensor_key] = torch.from_numpy(obs[column_name].to_numpy())

        arr = obs["Absolute_Obs_Time"].to_numpy().astype("datetime64[ns]", copy=False)
        obs_tensors["absolute_obs_time"] = torch.from_numpy(arr.view(np.int64))
        obs_tensors["target_time_sec"] = torch.from_numpy(obs["target_time"].to_numpy())

        platform_id = pc.fill_null(obs["Platform_ID"], 0)
        obs_tensors["platform_id"] = torch.from_numpy(platform_id.to_numpy())

        obs_type = pc.fill_null(obs["Observation_Type"], 0)
        obs_tensors["observation_type"] = torch.from_numpy(obs_type.to_numpy())

        return (
            obs_tensors,
            offsets_3d,
            sensor_id_to_local,
        )

    def _get_target(self, frames) -> torch.Tensor:
        all_state = [f["state"] for sample in frames for f in sample]
        batch_size = len(frames)
        state = np.stack(all_state)
        state = state.reshape((batch_size, -1) + state.shape[1:])
        state = (state - self.mean) / self.std
        target = torch.from_numpy(state)
        b, t, c, x = range(4)
        out = target.permute(b, c, t, x)
        return _reorder_nest_to_hpxpad(out)

    @functools.cached_property
    def _static_condition(self):
        condition = _get_static_condition(
            self.hpx_level_condition, self.variable_config
        )
        condition = condition.unsqueeze(0)
        return _reorder_nest_to_hpxpad(condition)

    @profiling.nvtx
    def transform(self, times, frames):
        """
        frames: [[{state: (c, x), obs_v2: Obs}]]
        times: [[cftime]]
        """
        out = {}

        def _apply_time_func(func):
            return torch.from_numpy(np.vectorize(func)(times))

        if "obs_v2" in frames[0][0].keys():
            out["unified_obs"] = self._process_obs(times, frames)
        out["target"] = self._get_target(frames).float()
        out["second_of_day"] = _apply_time_func(_compute_second_of_day).float()
        out["day_of_year"] = _apply_time_func(_compute_day_of_year).float()
        out["timestamp"] = _apply_time_func(_compute_timestamp)
        out["condition"] = self._static_condition.float()
        out["labels"] = torch.empty([len(frames), 0])
        return out

    @profiling.nvtx
    def device_transform(self, batch, device):
        """Transforms to the output of .transform that can occur on gpu

        Typically used with the prefetch_map in the main training process.
        """
        batch = batch.copy()
        out = {}

        for key in batch:
            if key == "unified_obs":
                obs_tensors, offsets, sensor_id_to_local = batch["unified_obs"]
                out[key] = self._device_transform_unified_obs(
                    obs_tensors, offsets, sensor_id_to_local, device
                )
            else:
                out[key] = batch[key].to(device, non_blocking=True)
        return out

    @profiling.nvtx
    def _device_transform_unified_obs(
        self, obs_tensors, offsets, sensor_id_to_local, device
    ):
        # Move all tensors to device efficiently
        def _to_device(tensor, non_blocking=True):
            if isinstance(tensor, torch.Tensor):
                return tensor.to(device, non_blocking=non_blocking)
            else:
                return torch.from_numpy(tensor).to(device, non_blocking=non_blocking)

        obs_tensors = {key: _to_device(val) for key, val in obs_tensors.items()}

        obs_time_ns = obs_tensors["absolute_obs_time"]
        lat_tensor = obs_tensors["latitude"]
        lon_tensor = obs_tensors["longitude"]
        height_tensor = obs_tensors["height"]
        pressure_tensor = obs_tensors["pressure"]
        scan_angle_tensor = obs_tensors["scan_angle"]
        sat_zenith_tensor = obs_tensors["sat_zenith_angle"]
        sol_zenith_tensor = obs_tensors["sol_zenith_angle"]
        platform_id_tensor = obs_tensors["platform_id"].int()
        obs_type_tensor = obs_tensors["observation_type"].int()
        sensor_id_tensor = obs_tensors["sensor_id"].int()
        pix = self._grid.ang2pix(lon_tensor, lat_tensor).int()
        local_channel_id_tensor = obs_tensors["local_channel_id"].int()
        global_channel_id_tensor = obs_tensors["global_channel_id"].int()
        observation_tensor = obs_tensors["observation"]

        # Compute metadata
        meta = features.compute_unified_metadata(
            obs_tensors["target_time_sec"],
            time=obs_time_ns,
            lat=lat_tensor,
            lon=lon_tensor,
            height=height_tensor,
            pressure=pressure_tensor,
            scan_angle=scan_angle_tensor,
            sat_zenith_angle=sat_zenith_tensor,
            sol_zenith_angle=sol_zenith_tensor,
        )

        # Build index tensor
        index = torch.stack(
            [
                sensor_id_tensor,
                pix,
                local_channel_id_tensor,
                platform_id_tensor,
                obs_type_tensor,
                global_channel_id_tensor,
            ],
            dim=1,  # Stack along dimension 1 to get (n_obs, 6) instead of (6, n_obs)
        )

        # Create UnifiedObservation object
        out = types.UnifiedObservation(
            obs=observation_tensor,
            time=obs_time_ns,
            float_metadata=meta,
            int_metadata=index,
            hpx_level=self.hpx_level,
            offsets=_to_device(offsets),
            sensor_id_to_local=_to_device(sensor_id_to_local),
        )
        return out


def collate(obj):
    return obj
