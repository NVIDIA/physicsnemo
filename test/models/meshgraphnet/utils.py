# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

import torch


def get_random_graph(
    num_nodes: int, min_degree: int, max_degree: int
) -> torch.Tensor:  # pragma: no cover
    """utility function which creates a random CSC-graph structure
    defined by an offsets and indices buffer based on a given number of
    nodes, and minimum and maximum node degree.
    """
    offsets = torch.empty(num_nodes + 1, dtype=torch.int64)
    offsets[0] = 0
    offsets[1:] = torch.randint(
        min_degree, max_degree + 1, (num_nodes,), dtype=torch.int64
    )
    offsets = offsets.cumsum(dim=0)
    num_indices = offsets[-1].item()
    indices = torch.randint(0, num_nodes, (num_indices,), dtype=torch.int64)

    return offsets, indices
