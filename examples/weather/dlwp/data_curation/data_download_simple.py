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

import os
import cdsapi


def download_data(var, year, save_path):
    c = cdsapi.Client()
    if var[0] == "single_level":
        config = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": var[1],
            "year": year,
            "month": "01",
            "day": ["01", "02", "03"],
            "time": ["00:00", "06:00", "12:00", "18:00"],
        }
        c.retrieve(
            "reanalysis-era5-single-levels",
            config,
            save_path,
        )
    elif var[0] == "pressure_level":
        config = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": var[1],
            "pressure_level": var[2],
            "year": year,
            "month": "01",
            "day": ["01", "02", "03"],
            "time": ["00:00", "06:00", "12:00", "18:00"],
        }
        c.retrieve(
            "reanalysis-era5-pressure-levels",
            config,
            save_path,
        )


var_list = [
    ("pressure_level", "temperature", "850"),
    ("pressure_level", "geopotential", "1000"),
    ("pressure_level", "geopotential", "700"),
    ("pressure_level", "geopotential", "500"),
    ("pressure_level", "geopotential", "300"),
    ("single_level", "total_column_water"),
    ("single_level", "2m_temperature"),
]

for i, var in enumerate(var_list):
    if not os.path.exists("data/train_temp/"):
        os.makedirs("data/train_temp/")
    download_data(var, "1979", "./data/train_temp/1979_" + str(i) + ".nc")

for i, var in enumerate(var_list):
    if not os.path.exists("data/test_temp/"):
        os.makedirs("data/test_temp/")
    download_data(var, "2017", "./data/test_temp/2017_" + str(i) + ".nc")

for i, var in enumerate(var_list):
    if not os.path.exists("data/out_of_sample_temp/"):
        os.makedirs("data/out_of_sample_temp/")
    download_data(var, "2018", "./data/out_of_sample_temp/2018_" + str(i) + ".nc")
