# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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
from pytest_utils import import_or_fail


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


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_gp_mapping(global_graph, device, pytestconfig):

    from physicsnemo.models.gnn_layers import (
        GraphPartition,
        partition_graph_with_id_mapping,
    )

    offsets, indices, num_src_nodes, num_dst_nodes = global_graph
    partition_size = 4
    partition_rank = 0

    mapping_src_ids_to_ranks = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    mapping_dst_ids_to_ranks = torch.tensor([0, 1, 2, 3])

    pg = partition_graph_with_id_mapping(
        offsets,
        indices,
        mapping_src_ids_to_ranks,
        mapping_dst_ids_to_ranks,
        partition_size,
        partition_rank,
        device,
    )

    pg_expected = GraphPartition(
        partition_size=4,
        partition_rank=0,
        device=device,
        local_offsets=torch.tensor([0, 2]),
        local_indices=torch.tensor([0, 1]),
        num_local_src_nodes=2,
        num_local_dst_nodes=1,
        num_local_indices=2,
        map_partitioned_src_ids_to_global=torch.tensor([0, 4]),
        map_partitioned_dst_ids_to_global=torch.tensor([0]),
        map_partitioned_edge_ids_to_global=torch.tensor([0, 1]),
        sizes=[[1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, 1]],
        scatter_indices=[
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
        ],
        num_src_nodes_in_each_partition=[2, 2, 2, 2],
        num_dst_nodes_in_each_partition=[1, 1, 1, 1],
        num_indices_in_each_partition=[2, 2, 2, 2],
    ).to(device=device)

    assert_partitions_are_equal(pg, pg_expected)


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_gp_nodewise(global_graph, device, pytestconfig):

    from physicsnemo.models.gnn_layers import (
        GraphPartition,
        partition_graph_nodewise,
    )

    offsets, indices, num_src_nodes, num_dst_nodes = global_graph
    partition_size = 4
    partition_rank = 0

    pg = partition_graph_nodewise(
        offsets,
        indices,
        partition_size,
        partition_rank,
        device,
    )

    pg_expected = GraphPartition(
        partition_size=4,
        partition_rank=0,
        device=device,
        local_offsets=torch.tensor([0, 2]),
        local_indices=torch.tensor([0, 1]),
        num_local_src_nodes=2,
        num_local_dst_nodes=1,
        num_local_indices=2,
        map_partitioned_src_ids_to_global=torch.tensor([0, 1]),
        map_partitioned_dst_ids_to_global=torch.tensor([0]),
        map_partitioned_edge_ids_to_global=torch.tensor([0, 1]),
        sizes=[[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]],
        scatter_indices=[
            torch.tensor([0, 1]),
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
        ],
        num_src_nodes_in_each_partition=[2, 2, 2, 2],
        num_dst_nodes_in_each_partition=[1, 1, 1, 1],
        num_indices_in_each_partition=[2, 2, 2, 2],
    ).to(device=device)

    assert_partitions_are_equal(pg, pg_expected)


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_gp_matrixdecomp(global_graph_square, device, pytestconfig):

    from physicsnemo.models.gnn_layers import (
        GraphPartition,
        partition_graph_nodewise,
    )

    offsets, indices, num_src_nodes, num_dst_nodes = global_graph_square
    partition_size = 4
    partition_rank = 0

    pg = partition_graph_nodewise(
        offsets, indices, partition_size, partition_rank, device, matrix_decomp=True
    )

    pg_expected = GraphPartition(
        partition_size=4,
        partition_rank=0,
        device=device,
        local_offsets=torch.tensor([0, 2]),
        local_indices=torch.tensor([0, 1]),
        num_local_src_nodes=2,
        num_local_dst_nodes=1,
        num_local_indices=2,
        map_partitioned_src_ids_to_global=torch.tensor([0, 3]),
        map_partitioned_dst_ids_to_global=torch.tensor([0]),
        map_partitioned_edge_ids_to_global=torch.tensor([0, 1]),
        sizes=[[1, 0, 1, 0], [0, 1, 1, 1], [0, 1, 0, 1], [1, 0, 0, 0]],
        scatter_indices=[
            torch.tensor([0]),
            torch.tensor([], dtype=torch.int64),
            torch.tensor([0]),
            torch.tensor([], dtype=torch.int64),
        ],
        num_src_nodes_in_each_partition=[2, 2, 2, 2],
        num_dst_nodes_in_each_partition=[1, 1, 1, 1],
        num_indices_in_each_partition=[2, 2, 2, 2],
    ).to(device=device)

    assert_partitions_are_equal(pg, pg_expected)


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_gp_coordinate_bbox(global_graph, device, pytestconfig):

    from physicsnemo.models.gnn_layers import (
        GraphPartition,
        partition_graph_by_coordinate_bbox,
    )

    offsets, indices, num_src_nodes, num_dst_nodes = global_graph
    partition_size = 4
    partition_rank = 0
    coordinate_separators_min = [[0, 0], [None, 0], [None, None], [0, None]]
    coordinate_separators_max = [[None, None], [0, None], [0, 0], [None, 0]]
    device = "cuda:0"
    src_coordinates = torch.FloatTensor(
        [
            [-1.0, 1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0],
            [-2.0, 2.0],
            [2.0, 2.0],
            [-2.0, -2.0],
            [2.0, -2.0],
        ]
    )
    dst_coordinates = torch.FloatTensor(
        [
            [-1.0, 1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0],
        ]
    )
    pg = partition_graph_by_coordinate_bbox(
        offsets,
        indices,
        src_coordinates,
        dst_coordinates,
        coordinate_separators_min,
        coordinate_separators_max,
        partition_size,
        partition_rank,
        device,
    )

    pg_expected = GraphPartition(
        partition_size=4,
        partition_rank=0,
        device=device,
        local_offsets=torch.tensor([0, 2]),
        local_indices=torch.tensor([0, 1]),
        num_local_src_nodes=2,
        num_local_dst_nodes=1,
        num_local_indices=2,
        map_partitioned_src_ids_to_global=torch.tensor([1, 5]),
        map_partitioned_dst_ids_to_global=torch.tensor([1]),
        map_partitioned_edge_ids_to_global=torch.tensor([2, 3]),
        sizes=[[0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]],
        scatter_indices=[
            torch.tensor([], dtype=torch.int64),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([], dtype=torch.int64),
        ],
        num_src_nodes_in_each_partition=[2, 2, 2, 2],
        num_dst_nodes_in_each_partition=[1, 1, 1, 1],
        num_indices_in_each_partition=[2, 2, 2, 2],
    ).to(device=device)

    assert_partitions_are_equal(pg, pg_expected)


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_gp_coordinate_bbox_lat_long(global_graph, device, pytestconfig):

    from physicsnemo.models.gnn_layers import (
        GraphPartition,
        partition_graph_by_coordinate_bbox,
    )

    offsets, indices, num_src_nodes, num_dst_nodes = global_graph
    src_lat = torch.FloatTensor([-75, -60, -45, -30, 30, 45, 60, 75]).view(-1, 1)
    dst_lat = torch.FloatTensor([-60, -30, 30, 30]).view(-1, 1)
    src_long = torch.FloatTensor([-135, -135, 135, 135, -45, -45, 45, 45]).view(-1, 1)
    dst_long = torch.FloatTensor([-135, 135, -45, 45]).view(-1, 1)
    src_coordinates = torch.cat([src_lat, src_long], dim=1)
    dst_coordinates = torch.cat([dst_lat, dst_long], dim=1)
    coordinate_separators_min = [
        [-90, -180],
        [-90, 0],
        [0, -180],
        [0, 0],
    ]
    coordinate_separators_max = [
        [0, 0],
        [0, 180],
        [90, 0],
        [90, 180],
    ]
    partition_size = 4
    partition_rank = 0
    device = "cuda:0"
    pg = partition_graph_by_coordinate_bbox(
        offsets,
        indices,
        src_coordinates,
        dst_coordinates,
        coordinate_separators_min,
        coordinate_separators_max,
        partition_size,
        partition_rank,
        device,
    )
    pg_expected = GraphPartition(
        partition_size=4,
        partition_rank=0,
        device=device,
        local_offsets=torch.tensor([0, 2]),
        local_indices=torch.tensor([0, 1]),
        num_local_src_nodes=2,
        num_local_dst_nodes=1,
        num_local_indices=2,
        map_partitioned_src_ids_to_global=torch.tensor([0, 1]),
        map_partitioned_dst_ids_to_global=torch.tensor([0]),
        map_partitioned_edge_ids_to_global=torch.tensor([0, 1]),
        sizes=[[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]],
        scatter_indices=[
            torch.tensor([0, 1]),
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
        ],
        num_src_nodes_in_each_partition=[2, 2, 2, 2],
        num_dst_nodes_in_each_partition=[1, 1, 1, 1],
        num_indices_in_each_partition=[2, 2, 2, 2],
    ).to(device=device)

    assert_partitions_are_equal(pg, pg_expected)


if __name__ == "__main__":
    pytest.main([__file__])
