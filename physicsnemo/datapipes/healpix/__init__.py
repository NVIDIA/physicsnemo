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

"""
Healpix DataPipes.

This package contains the Healpix DataPipes for loading and processing healpix data.
It supports loading data from xarray Datasets or zarr Groups, and coupling it with a model.

The main classes are:
- TimeSeriesDataModule: A DataModule for loading and processing time series data.
- CoupledTimeSeriesDataModule: A DataModule for loading and processing coupled time series data.
- TimeSeriesDataset: A dataset for loading and processing time series data.
- CoupledTimeSeriesDataset: A dataset for loading and processing coupled time series data.
- Zarr versions of the above classes.
- ConstantCoupler: A coupler that uses a constant value for the coupled data.
- TrailingAverageCoupler: A coupler that uses a trailing average for the coupled data.
"""

from .coupledtimeseries_dataset import CoupledTimeSeriesDataset
from .coupledtimeseries_dataset_zarr import CoupledTimeSeriesDatasetZarr
from .couplers import ConstantCoupler, TrailingAverageCoupler
from .data_modules import CoupledTimeSeriesDataModule, TimeSeriesDataModule
from .data_modules_zarr import CoupledTimeSeriesDataModuleZarr, TimeSeriesDataModuleZarr
from .timeseries_dataset import TimeSeriesDataset
from .timeseries_dataset_zarr import TimeSeriesDatasetZarr

__all__ = [
    "TimeSeriesDataModule",
    "CoupledTimeSeriesDataModule",
    "TimeSeriesDatasetZarr",
    "CoupledTimeSeriesDatasetZarr",
    "TimeSeriesDataset",
    "CoupledTimeSeriesDataset",
    "TimeSeriesDataModuleZarr",
    "CoupledTimeSeriesDataModuleZarr",
    "ConstantCoupler",
    "TrailingAverageCoupler",
]
