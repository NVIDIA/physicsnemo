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
"""Variable configurations for supported datasets."""

from physicsnemo.experimental.datapipes.healda.types import VariableConfig

VARIABLE_CONFIGS = {}

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