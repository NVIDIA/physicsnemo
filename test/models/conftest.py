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
import sys

import pytest

LAYER_NORM_PATH = "physicsnemo.models.layers.layer_norm"

"""
Physicsnemo layernorm module makes a decision, at import time, on whether or not
to use transformer engine.  It can be overridden by setting the
PHYSICSNEMO_FORCE_TE environment variable.

But, transformer engine is not available on CPU, so we need to cleanup the
module after the tests are run.

This fixture runs automatically before every test in the models directory.
It deletes the layer_norm module from sys.modules, which forces a fresh import
of the module at each test.

"""


@pytest.fixture(scope="module", autouse=True)
def cleanup_layer_norm_module():
    yield
    if LAYER_NORM_PATH in sys.modules:
        del sys.modules[LAYER_NORM_PATH]


@pytest.fixture
def set_physicsnemo_force_te(monkeypatch, device):
    if device == "cpu":
        te_force_val = False
    else:
        te_force_val = os.environ.get("PHYSICSNEMO_FORCE_TE", "")
    monkeypatch.setenv("PHYSICSNEMO_FORCE_TE", str(te_force_val))
