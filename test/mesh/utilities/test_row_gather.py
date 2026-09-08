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

"""Tests for the memmap-friendly row gather."""

import pytest
import torch

from physicsnemo.mesh.utilities import _row_gather as rg


@pytest.mark.parametrize("cap", [0, 1 << 40])  # plain gather / range-read branch
@pytest.mark.parametrize(
    "idx",
    [
        torch.tensor([3, 7, 8, 500, 999]),
        torch.tensor([999, 3, 8, 3, 500]),  # unsorted, duplicated
        torch.tensor([0]),
        torch.tensor([], dtype=torch.long),
    ],
)
def test_gather_rows_matches_fancy_index(monkeypatch, cap, idx):
    """Both branches return exactly ``t[idx]`` as a fresh tensor."""
    monkeypatch.setattr(rg, "RANGE_READ_MAX_BYTES", cap)
    for t in (torch.randn(1000, 3), torch.arange(1000), torch.randn(1000, 2, 2)):
        out = rg.gather_rows(t, idx)
        assert torch.equal(out, t[idx])
        assert out.shape == t[idx].shape
        if idx.numel():
            assert out.data_ptr() != t.data_ptr()
