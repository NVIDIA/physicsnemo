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

"""PCA feature names match independently known axes without reordering values."""

import torch

from physicsnemo.experimental.uq.ood.geometry import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    extract_features,
    feature_hash,
)
from physicsnemo.mesh import Mesh


def test_pca_names_match_known_rotated_box_axes(device):
    """Distinct box extents yield known, non-coordinate-aligned principal axes."""
    axes = (
        torch.tensor(
            [[1, 2, -2], [2, 1, 2], [2, -2, -1]], dtype=torch.float64, device=device
        )
        / 3
    )
    vertices = (
        torch.cartesian_prod(
            torch.tensor([-3.0, 3.0], device=device, dtype=torch.float64),
            torch.tensor([-2.0, 2.0], device=device, dtype=torch.float64),
            torch.tensor([-1.0, 1.0], device=device, dtype=torch.float64),
        )
        @ axes.T
    )
    cells = torch.tensor(
        [
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
        ],
        device=device,
    )
    result = extract_features(Mesh(points=vertices, cells=cells), return_tensor=True)
    for axis in (1, 2):
        for coordinate, name in enumerate("xyz"):
            index = FEATURE_NAMES.index(f"pca_axis{axis}_{name}")
            torch.testing.assert_close(result[index], axes[coordinate, axis - 1])
    # Preserve the historical numerical order used by fitted density artifacts.
    expected = torch.tensor([1, 2, 2, 1, 2, -2], dtype=result.dtype, device=device) / 3
    torch.testing.assert_close(result[3:9], expected)
    legacy_names = FEATURE_NAMES.copy()
    legacy_names[3:9] = [f"pca_axis{a}_{c}" for a in (1, 2) for c in "xyz"]
    assert FEATURE_VERSION == "v1.1"
    assert feature_hash(FEATURE_NAMES) != feature_hash(legacy_names)
