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
from healda.datasets.base import VariableConfig

VARIABLE_CONFIGS = {}
VARIABLE_CONFIGS["default"] = VariableConfig(
    name="ufs",
    # Data is on model pressure levels, not absolute pressure levels.
    # For simplicity, we use the nearest model levels to the desired levels.
    levels=[1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    variables_3d=["Q", "U", "V", "T", "Z"],
    variables_2d=[
        "tas",
        "uas",
        "vas",
        "rlut",
        "rsut",
        "pressfc",
        "pr",
        "rsds",
        "sst",
        "sic",
        "hfls",
        "huss",
    ],
    variables_static=["orog", "lfrac"],
)
VARIABLE_CONFIGS["era5"] = VariableConfig(
    name="era5",
    levels=[1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    variables_3d=["U", "V", "T", "Z", "Q"],
    variables_2d=[
        "tcwv",
        "tas",
        "uas",
        "vas",
        "100u",
        "100v",
        "pres_msl",
        "sst",
        "sic",
    ],
    variables_static=["orog", "lfrac"],
)
VARIABLE_CONFIGS["gfs"] = VariableConfig(
    name="gfs",
    levels=[1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    variables_3d=["U", "V", "T", "Z", "Q"],
    variables_2d=[
        "tcwv",
        "tas",
        "uas",
        "vas",
        "100u",
        "100v",
        "pres_msl",
        "sp",  # Surface pressure
    ],
    variables_static=["orog", "lfrac"],
)
