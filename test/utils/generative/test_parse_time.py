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

import datetime

import pytest
import yaml
from pytest_utils import import_or_fail

cftime = pytest.importorskip("cftime")

# ruff: noqa: S101  # TODo remove exception


def test_datetime_yaml():
    """test parse time"""
    dt = datetime.datetime(2011, 1, 1)
    s = dt.isoformat()
    loaded = yaml.safe_load(s)
    assert dt == loaded


@import_or_fail("cftime")
def test_convert_to_cftime(pytestconfig):
    """test parse time"""

    from physicsnemo.utils.diffusion import convert_datetime_to_cftime

    dt = datetime.datetime(2011, 1, 1)
    expected = cftime.DatetimeGregorian(2011, 1, 1)
    assert convert_datetime_to_cftime(dt) == expected
