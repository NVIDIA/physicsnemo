# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import sys

import dask.diagnostics
import xarray

base = sys.argv[1:-1]
out = sys.argv[-1]

with dask.diagnostics.ProgressBar():
    t = xarray.open_mfdataset(
        base,
        group="prediction",
        concat_dim="ensemble",
        combine="nested",
        chunks={"time": 1, "ensemble": 10},
    )
    t.to_zarr(out, group="prediction")

    t = xarray.open_dataset(base[0], group="input", chunks={"time": 1})
    t.to_zarr(out, group="input", mode="a")

    t = xarray.open_dataset(base[0], group="truth", chunks={"time": 1})
    t.to_zarr(out, group="truth", mode="a")

    t = xarray.open_dataset(base[0])
    t.to_zarr(out, mode="a")
