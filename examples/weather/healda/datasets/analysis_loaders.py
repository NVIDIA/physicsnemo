# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Optional

import numpy as np
import pandas as pd
import pathlib

from healda.datasets import catalog
from healda.datasets.base import BatchInfo, TimeUnit, VariableConfig
from healda.datasets.zarr_loader import NO_LEVEL, ZarrLoader
import zarr.api.asynchronous


__all__ = ["ERA5Loader", "get_batch_info"]

SST_LAND_FILL_VALUE = 290
MONTHLY_SST = "monthly_sst"

HPX_LEVEL = 6

LABELS = [
    "icon",
    "era5",
    "ufs",
]


class PackedZarrLoader:
    def __init__(self, entry, array_name, label: int):
        self.time = entry.to_xarray(chunks=None).indexes["time"]
        self._store = entry.to_zarr().store
        self._array = None
        self._name = array_name
        self._label = label

    async def sel_time(self, times):
        if self._array is None:
            group = await zarr.api.asynchronous.open_group(
                self._store,
                use_consolidated=True,
                mode="r",
            )
            self._array = await group.get(self._name)

        t = self.time.get_indexer(times)
        if any(t == -1):
            raise KeyError(t[t == -1])

        state = await self._array.get_orthogonal_selection(t)
        return {
            "state": state,
            "label": [self._label] * len(times),
        }


class ERA5Loader:
    def __init__(self, variable_config: VariableConfig):
        self.variable_config = variable_config
        variables_2d = [
            "sstk",
            "ci",
            "msl",
            "10u",
            "10v",
            "2t",
            "tcwv",
            "100u",
            "100v",
        ]
        # add tp, handling missing data from 2022-2023, deal with surface pressure
        entry = catalog.era5_hpx6()
        self._loader = ZarrLoader(
            path=entry.to_store(),
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
        )  # c t x
        state = np.moveaxis(state, 0, 1)  # t c x
        return {
            "state": state,
            "label": [LABELS.index("era5")] * len(times),
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

        # era5 precip is in liquid water equivalent accumulated over 1 hour (m)
        # icon is in mass flux units (kg / s / m^2)
        # unit conversion: tp / 3600 * density water = tp / 3600 * 1000
        if ("tp", NO_LEVEL) in data:
            water_density = 1000
            seconds_per_hour = 3600
            data[("tp", NO_LEVEL)] = (
                data[("tp", NO_LEVEL)] * water_density / seconds_per_hour
            )

        fields_out_map = {
            # mapping of ecmwf name to icon name
            "tclw": "cllvi",
            "tciw": "clivi",
            "2t": "tas",
            "10u": "uas",
            "10v": "vas",
            "100u": "100u",
            "100v": "100v",
            "msl": "pres_msl",
            "tp": "pr",
            "sstk": "sst",
            "ci": "sic",
            "tcwv": "prw",
            "u": "U",
            "v": "V",
            "t": "T",
            "z": "Z",
            "q": "Q",
            "tosbcs": MONTHLY_SST,
        }
        for key, value in list(data.items()):
            match key:
                case (name, level):
                    if name in fields_out_map:
                        data[(fields_out_map[name], level)] = value


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
    mean = _get_nearest_stats(config)["mean"].values
    return mean


def _get_std(config: VariableConfig) -> np.ndarray:
    std = _get_nearest_stats(config)["std"].values
    return std


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
    path = pathlib.Path(__file__).parent / file_name
    return pd.read_csv(path).set_index(["variable", "level"])


# def get_sst_stats(config: VariableConfig = _default_config):
#     df = _load_raw_stats(config)
#     row = df.loc[("sst", NO_LEVEL)]
#     return row["mean"].item(), row["std"].item()


def _get_nearest_stats(config: VariableConfig):
    # To handle float levels, gets nearest level
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
