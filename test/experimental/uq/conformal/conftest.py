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

"""Fixtures for the conformal-UQ suite (helpers live in ``_helpers.py``)."""

import pytest
import torch


@pytest.fixture
def fake_multi_rank(monkeypatch):
    """Return an ``activate()`` that fakes a two-rank ``torch.distributed`` group.

    Deferred so tests can build fitted objects before the fake group exists.
    """

    def activate():
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    return activate


@pytest.fixture
def default_dtype():
    """Restore torch's default dtype after a test that changes it."""
    original = torch.get_default_dtype()
    yield original
    torch.set_default_dtype(original)
