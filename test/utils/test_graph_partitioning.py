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

import torch

try:
    import dgl
except ImportError:
    pass

try:
    import torch_geometric as pyg
except ImportError:
    pass

from test.conftest import requires_module


def create_simple_graph():
    """Creates a simple 4x4 grid graph for testing."""
    # Graph structure:
    # ┌───┬───┬───┐
    # │   │   │   │
    # ├───┼───┼───┤
    # │   │   │   │
    # ├───┼───┼───┤
    # │   │   │   │
    # └───┴───┴───┘
    # Corresponding node ids:
    #  0  1  2  3
    #  4  5  6  7
    #  8  9 10 11
    # 12 13 14 15
    #
    size = 4
    num_nodes = size * size

    # Generate edges for grid connectivity.
    edges = []
    for i in range(size):
        for j in range(size):
            node_id = i * size + j
            # Connect to right neighbor.
            if j < size - 1:
                edges.append([node_id, node_id + 1])
            # Connect to bottom neighbor.
            if i < size - 1:
                edges.append([node_id, node_id + size])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # Make bidirectional.
    edge_index = pyg.utils.to_undirected(edge_index)
    # Add self loops.
    edge_index, _ = pyg.utils.add_self_loops(edge_index)

    # Create node features (coordinates and some dummy features).
    node_coords = torch.zeros(num_nodes, 2)
    for i in range(size):
        for j in range(size):
            node_id = i * size + j
            node_coords[node_id] = torch.tensor([i, j], dtype=torch.float32)

    node_features = torch.randn(num_nodes, 5)  # 5 dummy features
    edge_features = torch.randn(edge_index.size(1), 3)  # 3 edge features

    return edge_index, node_coords, node_features, edge_features


@requires_module(["dgl", "torch_geometric", "pyg_lib"])
def test_graph_partitioning_comparison(pytestconfig):
    """Compares DGL metis_partition with PyG ClusterData partitioning.

    No halo regions are used in this test.
    """

    torch.manual_seed(42)

    # Create a simple test graph.
    edge_index, node_coords, node_features, edge_features = create_simple_graph()
    num_nodes = node_coords.size(0)
    num_partitions = 4

    # Create DGL graph.
    dgl_graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    dgl_graph.ndata["coordinates"] = node_coords
    dgl_graph.ndata["features"] = node_features
    dgl_graph.edata["features"] = edge_features

    # Create PyG data.
    pyg_data = pyg.data.Data(
        edge_index=edge_index,
        coordinates=node_coords,
        x=node_features,
        edge_attr=edge_features,
    )

    # Test DGL partitioning.
    dgl_partitions = dgl.metis_partition(
        dgl_graph, k=num_partitions, extra_cached_hops=0, reshuffle=True
    )

    # Test PyG partitioning.
    cluster_data = pyg.loader.ClusterData(pyg_data, num_parts=num_partitions)

    # Compare basic properties.
    print(f"DGL created {len(dgl_partitions)} partitions")
    print(f"PyG created {cluster_data.num_parts} partitions")

    # Check that we get the expected number of partitions.
    assert len(dgl_partitions) == num_partitions
    assert cluster_data.num_parts == num_partitions

    # Count total nodes across all DGL partitions.
    total_dgl_nodes = sum(subgraph.num_nodes() for subgraph in dgl_partitions.values())
    print(f"Total nodes in DGL partitions: {total_dgl_nodes}")

    part_meta = cluster_data.partition
    # Count total nodes across all PyG partitions.
    total_pyg_nodes = sum(
        len(part_meta.node_perm[part_meta.partptr[i] : part_meta.partptr[i + 1]])
        for i in range(cluster_data.num_parts)
    )
    print(f"Total nodes in PyG partitions: {total_pyg_nodes}")

    # Due to overlapping nodes in halo regions, total might be >= original.
    assert total_dgl_nodes >= num_nodes
    assert total_pyg_nodes >= num_nodes

    # Check that node features and edges are preserved in DGL partitions.
    for subgraph in dgl_partitions.values():
        assert dgl_graph.ndata["coordinates"][subgraph.ndata[dgl.NID]].shape == (4, 2)
        assert dgl_graph.ndata["features"][subgraph.ndata[dgl.NID]].shape == (4, 5)
        assert subgraph.num_edges() == 12

    # Check that node features and edges are preserved in PyG partitions.
    for i in range(cluster_data.num_parts):
        start_idx = part_meta.partptr[i]
        end_idx = part_meta.partptr[i + 1]
        partition_nodes = part_meta.node_perm[start_idx:end_idx]
        assert pyg_data.coordinates[partition_nodes].shape == (4, 2)
        assert pyg_data.x[partition_nodes].shape == (4, 5)
        # ClusterData provides a way access sub-graph data.
        subgraph = cluster_data[i]
        assert (subgraph.x == pyg_data.x[partition_nodes]).all()


@requires_module(["dgl", "torch_geometric", "pyg_lib"])
def test_graph_partitioning_comparison_with_halo(pytestconfig):
    """Compares DGL metis_partition with PyG ClusterData partitioning.

    Halo regions are used in this test.
    """

    halo_size = 1
    torch.manual_seed(42)

    # Create a simple test graph.
    edge_index, node_coords, node_features, edge_features = create_simple_graph()
    num_nodes = node_coords.size(0)
    num_partitions = 4

    # Create DGL graph.
    dgl_graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    dgl_graph.ndata["coordinates"] = node_coords
    dgl_graph.ndata["features"] = node_features
    dgl_graph.edata["features"] = edge_features

    # Create PyG data.
    pyg_data = pyg.data.Data(
        edge_index=edge_index,
        coordinates=node_coords,
        x=node_features,
        edge_attr=edge_features,
    )

    # Test DGL partitioning.
    dgl_partitions = dgl.metis_partition(
        dgl_graph, k=num_partitions, extra_cached_hops=halo_size, reshuffle=True
    )

    # Test PyG partitioning.
    cluster_data = pyg.loader.ClusterData(pyg_data, num_parts=num_partitions)
    part_meta = cluster_data.partition

    # Compare basic properties.
    print(f"DGL created {len(dgl_partitions)} partitions")
    print(f"PyG created {cluster_data.num_parts} partitions")

    # Create partitions with halo regions.
    for part_idx in range(cluster_data.num_parts):
        part_inner_node_ids = part_meta.node_perm[
            part_meta.partptr[part_idx] : part_meta.partptr[part_idx + 1]
        ]
        part_node_ids_with_halo, edge_index, mapping, edge_mask = (
            pyg.utils.k_hop_subgraph(
                part_inner_node_ids,
                num_hops=halo_size,
                edge_index=pyg_data.edge_index,
                num_nodes=pyg_data.num_nodes,
                relabel_nodes=True,
            )
        )
        # Check partition 0 for specific values.
        # The partition itself consists of nodes 2, 3, 6, 7 (see create_simple_graph).
        # The halo region consists of nodes 1, 5, 10, 11.
        # The edge_index is the edge_index of the subgraph, which includes the halo region.
        # The mapping is the mapping of the nodes in the partition to the nodes in the subgraph.
        # The edge_mask is the mask of the edges in the subgraph.
        if part_idx == 0:
            assert (part_inner_node_ids == torch.tensor([2, 3, 6, 7])).all()
            assert (
                part_node_ids_with_halo == torch.tensor([1, 2, 3, 5, 6, 7, 10, 11])
            ).all()
            assert edge_index.shape == (2, 28)
            assert (mapping == torch.tensor([1, 2, 4, 5])).all()
            assert edge_mask.sum() == 28
