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
# ruff: noqa: S101
import pandas as pd
from datasets.obs_time_range_loader import Loader, _get_file_names


def test_file_name():
    assert _get_file_names(
        "a",
        ["atms"],
        pd.Timestamp(2000, 1, 1, 1),
        pd.Timestamp(2000, 1, 1, 23),
    ) == ["a/atms/20000101/0.parquet", "a/atms/20000102/0.parquet"]


def test_loader_get_empty():
    columns = ("Latitude",)
    loader = Loader(sensors=["amsua", "atms"], columns=columns)
    out = loader._get_empty()
    assert tuple(f.name for f in out.schema) == columns
