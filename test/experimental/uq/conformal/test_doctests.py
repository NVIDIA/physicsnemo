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

"""Run the package doctests, which CI's doctest pass skips for ``experimental``."""

import doctest
import importlib

import pytest

MODULES = ["scores"]


@pytest.mark.parametrize("name", MODULES)
def test_module_doctests(name):
    module = importlib.import_module(f"physicsnemo.experimental.uq.conformal.{name}")
    result = doctest.testmod(module)
    assert result.attempted > 0 and result.failed == 0
