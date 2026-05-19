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

"""Tests for match_points (near-exact vertex matching via KNN + tolerance)."""

import torch

from physicsnemo.mesh.sampling import match_points


class TestMatchPoints:
    """Verify match_points finds coincident vertices and respects tolerance."""

    def test_exact_matches(self):
        """All source points have an exact counterpart in target."""
        target = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        src_idx, tgt_idx = match_points(source, target, tolerance=1e-8)
        assert src_idx.tolist() == [0, 1]
        assert tgt_idx.tolist() == [1, 2]

    def test_no_matches_tight_tolerance(self):
        """No matches when tolerance is smaller than the closest distance."""
        a = torch.tensor([[0.0, 0.0]])
        b = torch.tensor([[0.1, 0.0]])
        src_idx, tgt_idx = match_points(a, b, tolerance=0.01)
        assert len(src_idx) == 0
        assert len(tgt_idx) == 0

    def test_partial_matches(self):
        """Only some source points are within tolerance of a target."""
        target = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
        source = torch.tensor([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
        src_idx, tgt_idx = match_points(source, target, tolerance=1e-6)
        assert src_idx.tolist() == [0, 2]
        assert tgt_idx.tolist() == [0, 1]

    def test_tolerance_boundary(self):
        """A point exactly at the tolerance distance is included."""
        target = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        source = torch.tensor([[0.5, 0.0]], dtype=torch.float64)
        src_idx, _ = match_points(source, target, tolerance=0.5)
        assert len(src_idx) == 1
        src_idx, _ = match_points(source, target, tolerance=0.49)
        assert len(src_idx) == 0

    def test_3d_points(self):
        """Works with 3D point clouds."""
        target = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        source = torch.tensor([[1.0, 2.0, 3.0]])
        src_idx, tgt_idx = match_points(source, target, tolerance=1e-6)
        assert src_idx.tolist() == [0]
        assert tgt_idx.tolist() == [0]

    def test_duplicate_source_points(self):
        """Multiple source points matching the same target point."""
        target = torch.tensor([[0.0, 0.0]])
        source = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        src_idx, tgt_idx = match_points(source, target, tolerance=1e-6)
        assert src_idx.tolist() == [0, 1]
        assert tgt_idx.tolist() == [0, 0]
