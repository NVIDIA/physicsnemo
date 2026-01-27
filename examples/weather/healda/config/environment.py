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
import os

import dotenv

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

non_config = dir()

CACHE_DIR = os.path.expanduser("~/.cache/healda")

###############
# ERA5 inputs #
###############
V6_ERA5_ZARR = os.getenv("V6_ERA5_ZARR", "")
V6_ERA5_ZARR_PROFILE = os.getenv("V6_ERA5_ZARR_PROFILE", "")

# Climatology processed from WeatherBench2
ERA5_CLIMATOLOGY_ZARR = os.getenv("ERA5_CLIMATOLOGY_ZARR", "")

########
# UFS #
########
UFS_HPX6_ZARR = os.getenv("UFS_HPX6_ZARR", "")
UFS_LAND_DATA_ZARR = os.getenv("UFS_LAND_DATA_ZARR", "")
UFS_LAND_DATA_PROFILE = os.getenv("UFS_LAND_DATA_PROFILE", "")
UFS_ZARR_PROFILE = os.getenv("UFS_ZARR_PROFILE", "")
UFS_OBS_PATH = os.getenv("UFS_OBS_PATH", "")
UFS_OBS_PROFILE = os.getenv("UFS_OBS_PROFILE", "")
# project file
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "")
DATA_ROOT = os.getenv("DATA_ROOT", os.path.join(PROJECT_ROOT, "datasets"))
CHECKPOINT_ROOT = os.getenv(
    "CHECKPOINT_ROOT", os.path.join(PROJECT_ROOT, "training-runs")
)


_config_vars = dict(vars())


def print_config():
    print("Environment settings:")
    print("-" * 80)
    for v in _config_vars:
        if v == "non_config":
            continue

        if v in non_config:
            continue

        value = _config_vars[v]
        print(f"{v}={value}")
    print("-" * 80)
