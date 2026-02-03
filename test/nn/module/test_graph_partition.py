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

import pytest
import torch


@pytest.fixture
def global_graph():
    """test fixture: simple graph with a degree of 2 per node"""
    num_src_nodes = 8
    num_dst_nodes = 4
    offsets = torch.arange(num_dst_nodes + 1, dtype=torch.int64) * 2
    indices = torch.arange(num_src_nodes, dtype=torch.int64)

    return (offsets, indices, num_src_nodes, num_dst_nodes)


@pytest.fixture
def global_graph_square():
    """test fixture: simple non-bipartie graph with a degree of 2 per node"""
    # num_src_nodes = 4
    # num_dst_nodes = 4
    # num_edges = 8
    offsets = torch.tensor([0, 2, 4, 6, 8], dtype=torch.int64)
    indices = torch.tensor([0, 3, 2, 1, 1, 0, 1, 2], dtype=torch.int64)

    return (offsets, indices, 4, 4)


def assert_partitions_are_equal(a, b):
    """test utility: check if a matches b"""
    attributes = [
        "partition_size",
        "partition_rank",
        "device",
        "num_local_src_nodes",
        "num_local_dst_nodes",
        "num_local_indices",
        "sizes",
        "num_src_nodes_in_each_partition",
        "num_dst_nodes_in_each_partition",
        "num_indices_in_each_partition",
    ]
    torch_attributes = [
        "local_offsets",
        "local_indices",
        "scatter_indices",
        "map_partitioned_src_ids_to_global",
        "map_partitioned_dst_ids_to_global",
        "map_partitioned_edge_ids_to_global",
    ]

    for attr in attributes:
        val_a, val_b = getattr(a, attr), getattr(b, attr)
        error_msg = f"{attr} does not match, got {val_a} and {val_b}"
        assert val_a == val_b, error_msg

    for attr in torch_attributes:
        val_a, val_b = getattr(a, attr), getattr(b, attr)
        error_msg = f"{attr} does not match, got {val_a} and {val_b}"
        if isinstance(val_a, list):
            assert isinstance(val_b, list), error_msg
            assert len(val_a) == len(val_b), error_msg
            for i in range(len(val_a)):
                assert torch.allclose(val_a[i], val_b[i]), error_msg
        else:
            assert torch.allclose(val_a, val_b), error_msg


if __name__ == "__main__":
    pytest.main([__file__])
