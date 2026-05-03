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
"""Two-stage transform for ERA5 state + observation data.

``ERA5ObsTransform`` implements both the ``Transform`` and ``DeviceTransform``
protocols:

- **Stage 1** (``transform``): CPU-side preprocessing in DataLoader workers.
  Normalizes state, encodes observations to tensors, computes time encodings.

- **Stage 2** (``device_transform``): GPU-side transfer and featurization in
  the ``prefetch_map`` background thread.  Moves tensors to device, computes
  observation metadata features, creates ``UnifiedObservation``.
"""

import dataclasses
import functools
import warnings

import numpy as np
import torch

from physicsnemo.core.version_check import OptionalImport

earth2grid = OptionalImport("earth2grid")
pa = OptionalImport("pyarrow")
pc = OptionalImport("pyarrow.compute")

from physicsnemo.experimental.datapipes.healda.configs.sensors import (
    NPLATFORMS,
    PLATFORM_NAME_TO_ID,
    SENSOR_CONFIGS,
    SENSOR_NAME_TO_ID,
)
from physicsnemo.experimental.datapipes.healda.configs.static_data import load_lfrac, load_orography
from physicsnemo.experimental.datapipes.healda.configs.variable_configs import VARIABLE_CONFIGS
from physicsnemo.experimental.datapipes.healda.loaders.era5 import get_batch_info
from physicsnemo.experimental.datapipes.healda.time_utils import (
    compute_day_of_year,
    compute_second_of_day,
    compute_timestamp,
)
from physicsnemo.experimental.datapipes.healda.transforms import obs_features, obs_features_ext
from physicsnemo.experimental.datapipes.healda.types import UnifiedObservation, VariableConfig

warnings.filterwarnings(
    "ignore",
    message="The given NumPy array is not writable, and PyTorch does not support non-writable tensors",
)


def _reorder_nest_to_hpxpad(x):
    x = torch.as_tensor(x)
    src_order = earth2grid.healpix.NEST
    dst_order = earth2grid.healpix.HEALPIX_PAD_XY
    return earth2grid.healpix.reorder(x, src_order, dst_order)


def _get_static_condition(HPX_LEVEL, variable_config) -> torch.Tensor:
    lfrac = load_lfrac(HPX_LEVEL)
    orography = load_orography()
    # Precomputed global mean/std over the UFS HEALPix level-6 grid (2000–2023 ERA5).
    orog_scale, orog_mean = 627.3885284872, 232.56013904090733
    lfrac_scale, lfrac_mean = 0.4695501683565522, 0.3410480857539571
    data = {
        "orog": (orography - orog_mean) / orog_scale,
        "lfrac": (lfrac - lfrac_mean) / lfrac_scale,
    }
    arrays = [torch.as_tensor(data[name]) for name in variable_config.variables_static]
    array = torch.stack(arrays).float()  # c x
    return array.unsqueeze(1)


def _map_platform_to_local(
    platform: torch.Tensor,
    lengths: torch.Tensor,
    ordered_sensor_ids: torch.Tensor,
    platform_luts: dict[int, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    local_platform = torch.zeros_like(platform)
    prev_end = 0
    for s_local, sensor_id in enumerate(ordered_sensor_ids.tolist()):
        count = int(lengths[s_local].sum().item())
        end = prev_end + count
        if end <= prev_end:
            continue
        lut = platform_luts.get(sensor_id)
        if lut is None:
            raise ValueError(f"Missing platform lookup table for sensor_id={sensor_id}")
        lut = lut.to(device)
        sensor_platform = platform[prev_end:end].long().clamp_(0, lut.shape[0] - 1)
        local_platform[prev_end:end] = lut[sensor_platform]
        prev_end = end
    return local_platform


@dataclasses.dataclass
class ERA5ObsTransform:
    """Two-stage batch transform for ERA5 state + observation data.

    Implements both ``Transform`` and ``DeviceTransform`` protocols.

    Args:
        variable_config: Which variables and levels to normalize.
        hpx_level: HEALPix level for observation pixel lookup.
        hpx_level_condition: HEALPix level for static conditioning data.
        extended_features: Whether to use extended (30-feature) observation
            encoding instead of the standard 28-feature encoding.
        sensors: Ordered list of sensor names (keys of ``SENSOR_CONFIGS``,
            e.g. ``["atms", "mhs", "conv"]``).  Controls observation
            grouping in the ``lengths`` tensor and per-sensor platform ID
            remapping.  Must match the sensors passed to the obs loader.
    """

    variable_config: VariableConfig = VARIABLE_CONFIGS["era5"]
    hpx_level: int = 10
    hpx_level_condition: int = 6
    extended_features: bool = False
    sensors: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        batch_info = get_batch_info(self.variable_config)
        self.mean = np.array(batch_info.center)[:, None]
        self.std = np.array(batch_info.scales)[:, None]

    @functools.cached_property
    def _grid(self):
        return earth2grid.healpix.Grid(
            self.hpx_level, pixel_order=earth2grid.healpix.HEALPIX_PAD_XY
        )

    @functools.cached_property
    def _ordered_sensor_ids(self) -> torch.Tensor:
        if not self.sensors:
            return torch.zeros((0,), dtype=torch.int32)
        return torch.tensor(
            [SENSOR_NAME_TO_ID[sensor_name] for sensor_name in self.sensors],
            dtype=torch.int32,
        )

    @functools.cached_property
    def _platform_luts(self) -> dict[int, torch.Tensor]:
        luts: dict[int, torch.Tensor] = {}
        for sensor_name in self.sensors:
            sensor_id = SENSOR_NAME_TO_ID[sensor_name]
            platform_ids = [
                PLATFORM_NAME_TO_ID[p] for p in SENSOR_CONFIGS[sensor_name].platforms
            ]
            lut = torch.zeros(NPLATFORMS, dtype=torch.long)
            for local_platform_id, global_platform_id in enumerate(platform_ids):
                lut[global_platform_id] = local_platform_id
            luts[sensor_id] = lut
        return luts

    # ------------------------------------------------------------------
    # Obs processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_by_record_batch(table: pa.Table, column_name: str) -> pa.Table:
        record_batches_order = []
        for batch in table.to_batches():
            if batch.num_rows == 0:
                continue
            group_value = batch[column_name][0]
            record_batches_order.append((group_value, batch))

        if not record_batches_order:
            return table

        record_batches_order.sort(key=lambda x: x[0].as_py())
        return pa.Table.from_batches([batch for _, batch in record_batches_order])

    @staticmethod
    def _append_batch_time_info_chunked(
        table: pa.Table, b: int, t: int, timestamp: int
    ) -> pa.Table:
        b_idx_type = pa.int16()
        t_idx_type = pa.int16()
        time_type = pa.int64()
        ref_col = table.column(0)

        b_idx_chunks, t_idx_chunks, time_chunks = [], [], []
        for chunk in ref_col.chunks:
            L = len(chunk)
            if L == 0:
                b_idx_chunks.append(pa.array([], type=b_idx_type))
                t_idx_chunks.append(pa.array([], type=t_idx_type))
                time_chunks.append(pa.array([], type=time_type))
                continue

            b_idx_chunks.append(pa.array(np.full(L, b, dtype=np.int16), type=b_idx_type))
            t_idx_chunks.append(pa.array(np.full(L, t, dtype=np.int16), type=t_idx_type))
            time_chunks.append(pa.array(np.full(L, timestamp, dtype=np.int64), type=time_type))

        out = table.append_column("batch_idx", pa.chunked_array(b_idx_chunks, type=b_idx_type))
        out = out.append_column("time_idx", pa.chunked_array(t_idx_chunks, type=t_idx_type))
        out = out.append_column("target_time", pa.chunked_array(time_chunks, type=time_type))
        return out

    @staticmethod
    def _build_observation_lengths_3d(
        obs_table: pa.Table, frame_times, ordered_sensor_ids: torch.Tensor
    ):
        B, T = len(frame_times), len(frame_times[0])
        counts_map = {}

        for batch in obs_table.to_batches():
            if batch.num_rows == 0:
                continue
            s_id = int(batch["sensor_id"][0].as_py())
            b_id = int(batch["batch_idx"][0].as_py())
            t_id = int(batch["time_idx"][0].as_py())
            n = batch.num_rows
            if s_id not in counts_map:
                counts_map[s_id] = torch.zeros((B, T), dtype=torch.int32)
            counts_map[s_id][b_id, t_id] += n

        S = int(ordered_sensor_ids.numel())
        if S == 0:
            return torch.zeros((0, B, T), dtype=torch.int32)

        lengths_3d = torch.zeros((S, B, T), dtype=torch.int32)
        for s_local, s_id in enumerate(ordered_sensor_ids.tolist()):
            if s_id in counts_map:
                lengths_3d[s_local] = counts_map[s_id]

        return lengths_3d

    def _process_obs(self, target_times, frames):
        if not self.sensors:
            raise ValueError("ERA5ObsTransform requires configured sensors.")

        all_obs_with_indices = []
        for b_idx, sample_frames in enumerate(frames):
            for t_idx, frame_dict in enumerate(sample_frames):
                table = frame_dict["obs"]
                table_with_indices = self._append_batch_time_info_chunked(
                    table, b_idx, t_idx,
                    compute_timestamp(target_times[b_idx][t_idx]),
                )
                all_obs_with_indices.append(table_with_indices)

        obs = pa.concat_tables(all_obs_with_indices)
        obs = self._sort_by_record_batch(obs, "sensor_id")

        lengths_3d = self._build_observation_lengths_3d(
            obs, target_times, self._ordered_sensor_ids
        )

        obs_tensors = {}
        required_columns = {
            "latitude": "Latitude",
            "longitude": "Longitude",
            "observation": "Observation",
            "global_channel_id": "Global_Channel_ID",
            "sat_zenith_angle": "Sat_Zenith_Angle",
            "sol_zenith_angle": "Sol_Zenith_Angle",
            "local_channel_id": "local_channel_id",
            "height": "Height",
            "pressure": "Pressure",
            "scan_angle": "Scan_Angle",
        }

        for tensor_key, column_name in required_columns.items():
            obs_tensors[tensor_key] = torch.from_numpy(obs[column_name].to_numpy())

        arr = obs["Absolute_Obs_Time"].to_numpy().astype("datetime64[ns]", copy=False)
        obs_tensors["absolute_obs_time"] = torch.from_numpy(arr.view(np.int64))
        obs_tensors["target_time_sec"] = torch.from_numpy(obs["target_time"].to_numpy())

        platform_id = pc.fill_null(obs["Platform_ID"], 0)
        obs_tensors["platform_id"] = torch.from_numpy(platform_id.to_numpy())

        obs_type = pc.fill_null(obs["Observation_Type"], 0)
        obs_tensors["observation_type"] = torch.from_numpy(obs_type.to_numpy())

        return (obs_tensors, lengths_3d)

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

    # ------------------------------------------------------------------
    # Stage 1: CPU transform (DataLoader workers)
    # ------------------------------------------------------------------

    def transform(self, times, frames):
        """CPU-side batch transform.

        Args:
            times: ``list[list[cftime]]`` shaped ``(batch, time_per_sample)``.
            frames: ``list[list[dict]]`` shaped ``(batch, time_per_sample)``.

        Returns:
            Batch dict with ``target``, ``unified_obs``, ``condition``, and
            time encodings.
        """
        out = {}

        def _apply_time_func(func):
            return torch.from_numpy(np.vectorize(func)(times))

        if "obs" in frames[0][0].keys():
            out["unified_obs"] = self._process_obs(times, frames)
        out["target"] = self._get_target(frames).float()
        out["second_of_day"] = _apply_time_func(compute_second_of_day).float()
        out["day_of_year"] = _apply_time_func(compute_day_of_year).float()
        out["timestamp"] = _apply_time_func(compute_timestamp)
        b, _, t, _ = out["target"].shape
        condition = self._static_condition.float()
        if condition.shape[0] not in (1, b):
            raise ValueError(
                f"condition batch dim {condition.shape[0]} must be 1 or target batch {b}"
            )
        if condition.shape[2] not in (1, t):
            raise ValueError(
                f"condition time dim {condition.shape[2]} must be 1 or target time {t}"
            )
        if condition.shape[0] == 1 and b > 1:
            condition = condition.expand(b, -1, -1, -1)
        if condition.shape[2] == 1 and t > 1:
            condition = condition.expand(-1, -1, t, -1)
        out["condition"] = condition.clone()
        out["labels"] = torch.empty([len(frames), 0])
        return out

    # ------------------------------------------------------------------
    # Stage 2: GPU transform (prefetch_map background thread)
    # ------------------------------------------------------------------

    def device_transform(self, batch, device):
        """GPU-side transform: move to device and compute observation features.

        Args:
            batch: Output of ``transform()``.
            device: Target ``torch.device``.

        Returns:
            Batch dict with all tensors on device and ``unified_obs`` as
            ``UnifiedObservation``.
        """
        batch = batch.copy()
        out = {}
        for key in batch:
            if key == "unified_obs":
                obs_tensors, lengths = batch["unified_obs"]
                out[key] = self._device_transform_unified_obs(
                    obs_tensors, lengths, device
                )
            else:
                out[key] = batch[key].to(device, non_blocking=True)
        return out

    def _device_transform_unified_obs(self, obs_tensors, lengths, device):
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
        pix = self._grid.ang2pix(lon_tensor, lat_tensor).int()
        local_channel_id_tensor = obs_tensors["local_channel_id"].int()
        global_channel_id_tensor = obs_tensors["global_channel_id"].int()
        observation_tensor = obs_tensors["observation"]

        if self.extended_features:
            meta = obs_features_ext.compute_unified_metadata(
                obs_tensors["target_time_sec"],
                time=obs_time_ns,
                lon=lon_tensor,
                lat=lat_tensor,
                height=height_tensor,
                pressure=pressure_tensor,
                scan_angle=scan_angle_tensor,
                sat_zenith_angle=sat_zenith_tensor,
                sol_zenith_angle=sol_zenith_tensor,
            )
        else:
            meta = obs_features.compute_unified_metadata(
                obs_tensors["target_time_sec"],
                time=obs_time_ns,
                lon=lon_tensor,
                height=height_tensor,
                pressure=pressure_tensor,
                scan_angle=scan_angle_tensor,
                sat_zenith_angle=sat_zenith_tensor,
                sol_zenith_angle=sol_zenith_tensor,
            )

        lengths = _to_device(lengths)
        local_platform = _map_platform_to_local(
            platform=platform_id_tensor,
            lengths=lengths,
            ordered_sensor_ids=self._ordered_sensor_ids.to(device),
            platform_luts=self._platform_luts,
            device=device,
        )

        return UnifiedObservation(
            obs=observation_tensor,
            time=obs_time_ns,
            float_metadata=meta,
            pix=pix,
            local_channel=local_channel_id_tensor,
            local_platform=local_platform,
            obs_type=obs_type_tensor,
            global_channel=global_channel_id_tensor,
            global_platform=platform_id_tensor,
            hpx_level=self.hpx_level,
            lengths=lengths,
        )
