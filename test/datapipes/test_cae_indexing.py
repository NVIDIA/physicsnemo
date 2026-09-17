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

"""Tests for legacy CAE reader indexing."""

import numpy as np

from physicsnemo.datapipes.cae.cae_dataset import NpzFileReader


def test_volume_subsampling_wraps_within_requested_range(monkeypatch):
    reader = NpzFileReader([], {})
    monkeypatch.setattr(np.random, "randint", lambda _: 8)

    indices = reader.select_random_sections_from_slice(5, 15, 4)

    np.testing.assert_array_equal(indices, np.array([13, 14, 5, 6]))
