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
import asyncio
import urllib.parse

import cftime
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import zarr.storage
from zarr.core.sync import sync

NO_LEVEL = -1


def _is_local(path):
    url = urllib.parse.urlparse(path)
    return url.scheme == ""


async def _getitem(array, index):
    return await array.get_orthogonal_selection(index)


async def _getitem_static(array, num_times: int):
    """Return the static field broadcasted to the number of times in the chunk"""
    field = await array.getitem((slice(None),) * array.ndim)
    field = field[None, ...]
    return np.broadcast_to(field, (num_times, *field.shape[1:]))


class ZarrLoader:
    """Load 2d and 3d data from a zarr dataset"""

    def __init__(
        self,
        *,
        path: zarr.storage.StoreLike,
        variables_3d,
        variables_2d,
        levels,
        level_coord_name: str = "",
        storage_options=None,
        time_sel_method: str | None = None,
        variables_static: list[str] = [],
    ):
        """
        Args:
            time_sel_method: passed to pd.Index.get_indexer(method=)
        """
        self.time_sel_method = time_sel_method
        self.variables_2d = variables_2d
        self.variables_3d = variables_3d
        self.levels = levels
        self.variables_static = variables_static
        if isinstance(path, str):
            if _is_local(path):
                storage_options = None
            self.group = sync(
                zarr.api.asynchronous.open_group(
                    path,
                    storage_options=storage_options,
                    use_consolidated=True,
                    mode="r",
                )
            )
        else:
            self.group = sync(
                zarr.api.asynchronous.open_group(
                    path,
                    storage_options=storage_options,
                    use_consolidated=True,
                    mode="r",
                )
            )

        if self.variables_3d:
            self.inds = sync(self._get_vertical_indices(level_coord_name, levels))

        self._arrays = {}
        self._has_time = self.variables_3d or self.variables_2d
        if self._has_time:
            time_num, self.units, self.calendar = sync(self._get_time())
            if np.issubdtype(time_num.dtype, np.datetime64):
                self.times = pd.DatetimeIndex(time_num)
            else:
                self.times = xr.CFTimeIndex(
                    cftime.num2date(time_num, units=self.units, calendar=self.calendar)
                )

    async def sel_time(self, times) -> dict[tuple[str, int], np.ndarray]:
        """

        Returns:
            dict of output data:
                keys are like (name, level), level == -1 for 2d variables

        """
        if self._has_time:
            index_in_loader = self.times.get_indexer(times, method=self.time_sel_method)
            if (index_in_loader == -1).any():
                raise KeyError("Index not found.")
        else:
            index_in_loader = np.arange(len(times))
        arr = await self._get(index_in_loader)

        return arr

    async def _get_time(self):
        time = await self.group.get("time")
        time_data = await time.getitem(slice(None))
        return time_data, time.attrs.get("units"), time.attrs.get("calendar")

    async def _get_vertical_indices(self, coord_name, levels):
        levels_var = await self.group.get(coord_name)
        levels_arr = await levels_var.getitem(slice(None))
        return pd.Index(levels_arr).get_indexer(levels, method="nearest")

    async def _get_array(self, name):
        if name not in self._arrays:
            self._arrays[name] = await self.group.get(name)
        return self._arrays[name]

    async def _get(self, t) -> dict[tuple[str, int | None], np.ndarray]:
        tasks = []
        keys = []

        for name in self.variables_3d:
            arr = await self._get_array(name)
            if arr is None:
                raise KeyError(name)
            for level, k in zip(self.levels, self.inds):
                key = (name, level)
                # NOTE creating a length 1 list for this indexer avoids an zarr
                # bug, when using a scalar value (k_indexer = 1)
                #
                # ValueError: could not broadcast input array from shape (2,1,49152) into shape (2,49152)
                #
                # not sure when this bug appeared
                # but it's failing with zarr 3.1.13 on dfw
                k_indexer = [k]
                value = _getitem(arr, (t, k_indexer))
                tasks.append(value)
                keys.append(key)

        for name in self.variables_2d:
            arr = await self._get_array(name)
            if arr is None:
                raise KeyError(name)
            key = (name, NO_LEVEL)
            value = _getitem(arr, (t,))
            tasks.append(value)
            keys.append(key)

        for name in self.variables_static:
            arr = await self._get_array(name)
            if arr is None:
                raise KeyError(name)
            key = (name, NO_LEVEL)
            value = _getitem_static(arr, len(t))
            tasks.append(value)
            keys.append(key)

        arrays = await asyncio.gather(*tasks)
        # squeeze out the dimenions added to workaround the zarr bug. See NOTE above.
        out = {}
        for key, array in zip(keys, arrays):
            name, _ = key
            if name in self.variables_3d:
                out[key] = np.squeeze(array, 1)
            else:
                out[key] = array

        return out
