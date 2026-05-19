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
"""Async zarr loader for 2D and 3D atmospheric variables.

``ZarrLoader`` provides concurrent I/O for multiple variables and levels
using ``asyncio.gather``.  It is used by ``ERA5Loader`` to read ERA5 data
from zarr stores.
"""

from __future__ import annotations

import asyncio
import urllib.parse

import cftime
import numpy as np
import pandas as pd

from physicsnemo.core.version_check import OptionalImport

xr = OptionalImport("xarray")
zarr = OptionalImport("zarr")
_zarr_sync = OptionalImport("zarr.core.sync")

NO_LEVEL = -1  # sentinel for 2D (surface) variables that lack a pressure level


def _is_local(path):
    url = urllib.parse.urlparse(path)
    return url.scheme == ""


async def _getitem(array, index):
    return await array.get_orthogonal_selection(index)


async def _getitem_static(array, num_times: int):
    field = await array.getitem((slice(None),) * array.ndim)
    field = field[None, ...]
    return np.broadcast_to(field, (num_times, *field.shape[1:]))


class ZarrLoader:
    """Load 2D and 3D variables from a zarr dataset with async I/O.

    Args:
        path: Zarr store path (local or remote).
        variables_3d: List of 3D variable names.
        variables_2d: List of 2D variable names.
        levels: Pressure levels to extract.
        level_coord_name: Name of the vertical coordinate in the zarr store.
        storage_options: fsspec storage options for remote stores.
        time_sel_method: Passed to ``pd.Index.get_indexer(method=)``.
        variables_static: List of static (time-invariant) variable names.
    """

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
        self.time_sel_method = time_sel_method
        self.variables_2d = variables_2d
        self.variables_3d = variables_3d
        self.levels = levels
        self.variables_static = variables_static

        if isinstance(path, str) and _is_local(path):
            storage_options = None

        self.group = _zarr_sync.sync(
            zarr.api.asynchronous.open_group(
                path,
                storage_options=storage_options,
                use_consolidated=True,
                mode="r",
            )
        )

        if self.variables_3d:
            self.inds = _zarr_sync.sync(self._get_vertical_indices(level_coord_name, levels))

        self._arrays = {}
        self._has_time = bool(self.variables_3d or self.variables_2d)
        if self._has_time:
            time_num, self.units, self.calendar = _zarr_sync.sync(self._get_time())
            if np.issubdtype(time_num.dtype, np.datetime64):
                self.times = pd.DatetimeIndex(time_num)
            else:
                self.times = xr.CFTimeIndex(
                    cftime.num2date(time_num, units=self.units, calendar=self.calendar)
                )

    async def sel_time(self, times) -> dict[tuple[str, int], np.ndarray]:
        """Load data for the given times.

        Returns:
            Dict with keys ``(variable_name, level)`` where ``level == -1``
            for 2D variables.
        """
        if self._has_time:
            index_in_loader = self.times.get_indexer(
                times, method=self.time_sel_method
            )
            if (index_in_loader == -1).any():
                raise KeyError("Index not found.")
        else:
            index_in_loader = np.arange(len(times))
        return await self._get(index_in_loader)

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
        out = {}
        for key, array in zip(keys, arrays):
            name, _ = key
            if name in self.variables_3d:
                out[key] = np.squeeze(array, 1)
            else:
                out[key] = array

        return out
