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
"""ERA5 analysis loader and normalization statistics.

Provides ``get_batch_info`` for normalization constants (mean/std per channel)
and ERA5-specific variable name mapping from ECMWF conventions to the standard
names used internally.
"""

import os
import pathlib
from typing import Optional

import numpy as np
import pandas as pd

from physicsnemo.core.version_check import OptionalImport

xarray = OptionalImport("xarray")

from physicsnemo.experimental.datapipes.healda.configs.sensors import STATS_DIR_ENV
from physicsnemo.experimental.datapipes.healda.loaders.zarr_loader import NO_LEVEL, ZarrLoader
from physicsnemo.experimental.datapipes.healda.types import BatchInfo, TimeUnit, VariableConfig

__all__ = ["ERA5Loader", "get_batch_info"]

SST_LAND_FILL_VALUE = 290
HPX_LEVEL = 6


class ERA5Loader:
    """Load ERA5 reanalysis state via async zarr I/O.

    Wraps ``ZarrLoader`` with ERA5-specific variable naming conventions
    and returns ``{"state": ndarray, "label": list[int]}``.

    Args:
        variable_config: Describes which 2D/3D variables and levels to load.
        era5_zarr_path: Path to ERA5 zarr store. If *None*, reads from
            ``ERA5_74VAR`` environment variable.
    """

    def __init__(self, variable_config: VariableConfig, era5_zarr_path: str | None = None):
        self.variable_config = variable_config
        variables_2d = [
            "sstk", "ci", "msl", "10u", "10v", "2t", "tcwv", "100u", "100v",
        ]
        path = era5_zarr_path or os.environ.get("ERA5_74VAR", os.environ.get("V6_ERA5_ZARR", ""))
        self._loader = ZarrLoader(
            path=path,
            variables_3d=["u", "v", "t", "z", "q"],
            variables_2d=variables_2d,
            level_coord_name="levels",
            levels=variable_config.levels,
        )

    async def sel_time(self, times):
        data = await self._loader.sel_time(times)
        self._convert_to_standard(data)
        shape = (len(times), 4**HPX_LEVEL * 12)
        state = _collect_fields(
            _get_index(self.variable_config), data, shape=shape
        )
        state = np.moveaxis(state, 0, 1)  # c t x -> t c x
        return {
            "state": state,
            "label": [1] * len(times),  # 1 = era5 label index
        }

    def _convert_to_standard(self, data):
        if ("sstk", NO_LEVEL) in data:
            sstk = data[("sstk", NO_LEVEL)]
            if not np.ma.isMaskedArray(sstk):
                sstk = np.ma.masked_invalid(sstk)
            data[("sstk", NO_LEVEL)] = sstk.filled(SST_LAND_FILL_VALUE)

        if ("ci", NO_LEVEL) in data:
            ci = data[("ci", NO_LEVEL)]
            if not np.ma.isMaskedArray(ci):
                ci = np.ma.masked_invalid(ci)
            data[("ci", NO_LEVEL)] = ci.filled(0)

        if ("tp", NO_LEVEL) in data:
            water_density = 1000
            seconds_per_hour = 3600
            data[("tp", NO_LEVEL)] = (
                data[("tp", NO_LEVEL)] * water_density / seconds_per_hour
            )

        fields_out_map = {
            "tclw": "cllvi", "tciw": "clivi", "2t": "tas", "10u": "uas",
            "10v": "vas", "100u": "100u", "100v": "100v", "msl": "pres_msl",
            "tp": "pr", "sstk": "sst", "ci": "sic", "tcwv": "prw",
            "u": "U", "v": "V", "t": "T", "z": "Z", "q": "Q",
        }
        for key, value in list(data.items()):
            match key:
                case (name, level):
                    if name in fields_out_map:
                        data[(fields_out_map[name], level)] = value


# ---------------------------------------------------------------------------
# Normalization statistics
# ---------------------------------------------------------------------------


def get_batch_info(
    config: VariableConfig,
    time_step: int = 1,
    time_unit: TimeUnit = TimeUnit.HOUR,
) -> BatchInfo:
    return BatchInfo(
        channels=[_encode_channel(tup) for tup in _get_index(config).tolist()],
        scales=_get_std(config),
        center=_get_mean(config),
        time_step=time_step,
        time_unit=time_unit,
    )


def _get_index(config: VariableConfig):
    return pd.MultiIndex.from_tuples(
        [(v, level) for v in config.variables_3d for level in config.levels]
        + [(v, NO_LEVEL) for v in config.variables_2d],
        names=["variable", "level"],
    )


def _collect_fields(
    index,
    data: dict[tuple[str, int | None], np.ndarray],
    shape,
    prefix: Optional[str] = None,
) -> np.ndarray:
    out = np.full(
        shape=(index.size,) + shape,
        dtype=np.float32,
        fill_value=np.nan,
    )
    for i, (var, lev) in enumerate(index):
        key = (prefix, var, lev) if prefix is not None else (var, lev)
        if key in data:
            out[i] = data[key]
    return out


def _get_mean(config: VariableConfig) -> np.ndarray:
    return _get_nearest_stats(config)["mean"].values


def _get_std(config: VariableConfig) -> np.ndarray:
    return _get_nearest_stats(config)["std"].values


def _encode_channel(channel) -> str:
    name, level = channel
    if level != NO_LEVEL:
        return f"{name}{level}"
    else:
        return name


def _load_raw_stats(config: VariableConfig) -> pd.DataFrame:
    if config.name == "ufs":
        file_name = "ufs_v0_stats.csv"
    elif config.name == "era5":
        file_name = "era5_13_levels_stats.csv"
    else:
        raise ValueError(f"Unknown dataset: {config.name}")
    stats_dir = os.environ.get(STATS_DIR_ENV)
    if not stats_dir:
        raise RuntimeError(
            f"{STATS_DIR_ENV} is not set; point it at a directory containing "
            f"{file_name} (see examples/weather/healda/configs)."
        )
    path = pathlib.Path(stats_dir) / file_name
    return pd.read_csv(path).set_index(["variable", "level"])


def _get_nearest_stats(config: VariableConfig):
    raw = _load_raw_stats(config)
    idx = _get_index(config)

    mapped_idx = []
    for var, level in idx:
        if level != NO_LEVEL:
            available = raw.loc[var].index.values
            nearest = available[np.abs(available - level).argmin()]
            mapped_idx.append((var, nearest))
        else:
            mapped_idx.append((var, level))

    mapped_idx = pd.MultiIndex.from_tuples(mapped_idx, names=["variable", "level"])
    return raw.loc[mapped_idx]


def open_era5_xarray(path: str | None = None, **kwargs) -> "xarray.Dataset":
    """Open the ERA5 74-variable zarr dataset as xarray.

    Args:
        path: Zarr store path. If *None*, reads ``ERA5_74VAR`` env var.
    """
    path = path or os.environ["ERA5_74VAR"]
    return xarray.open_zarr(path, **kwargs)
