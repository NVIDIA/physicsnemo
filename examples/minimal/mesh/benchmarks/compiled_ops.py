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

"""``torch.compile``-wrapped PhysicsNeMo-Mesh operations for benchmarking.

Each compiled function is a thin wrapper around the corresponding function in
:mod:`raw_ops`, with ``torch.compile`` applied.  See ``raw_ops`` for the
actual implementations and docstrings.
"""

import torch

from . import raw_ops

cell_normals = torch.compile(raw_ops.cell_normals)
gaussian_curvature = torch.compile(raw_ops.gaussian_curvature)
gradient = torch.compile(raw_ops.gradient)
subdivide = torch.compile(raw_ops.subdivide)
p2p_neighbors = torch.compile(raw_ops.p2p_neighbors)
c2c_neighbors = torch.compile(raw_ops.c2c_neighbors)
sample_points = torch.compile(raw_ops.sample_points)
sample_points_area_weighted = torch.compile(raw_ops.sample_points_area_weighted)
smooth = torch.compile(raw_ops.smooth)
transforms = torch.compile(raw_ops.transforms)
