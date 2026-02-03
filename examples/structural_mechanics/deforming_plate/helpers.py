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

from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.nn.functional import radius_search


def add_world_edges(graph, world_edge_radius=0.03, edge_stats_path="edge_stats.json"):
    """
    Adds world edges to the graph.
    """
    graph = graph.clone()
    device = graph.x.device
    # Get the edge stats
    edge_stats = load_json(edge_stats_path)
    edge_mean = edge_stats["edge_mean"].to(device)
    edge_std = edge_stats["edge_std"].to(device)

    # Get the mesh edge index
    mesh_src, mesh_dst = graph.edge_index
    mesh_edges = set(
        (int(src), int(dst)) for src, dst in zip(mesh_src.tolist(), mesh_dst.tolist())
    )

    # Get the world edge index
    world_pos = graph.world_pos
    edge_index = radius_search(
        world_pos,
        world_pos,
        radius=world_edge_radius,
        return_dists=False,
        return_points=False,
    )

    # Filter out self-loops
    filter = edge_index[0] != edge_index[1]
    filtered_edge_index = edge_index[:, filter]

    # Exclude existing edges
    candidate_edges = set(
        (int(src), int(dst))
        for src, dst in zip(
            filtered_edge_index[0].tolist(), filtered_edge_index[1].tolist()
        )
    )
    world_edges = torch.tensor(
        [list(edge) for edge in candidate_edges if edge not in mesh_edges],
        dtype=torch.int32,
        device=device,
    ).T  # shape: (2, num_world_edges)

    if world_edges.size(0) == 0:
        raise ValueError("No world edges to add. Try increasing the world edge radius.")

    # Compute edge features for new edges
    world_src, world_dst = world_edges
    world_disp = world_pos[world_src] - world_pos[world_dst]
    world_disp_norm = torch.linalg.norm(world_disp, dim=-1, keepdim=True)
    world_edge_features = torch.cat([world_disp, world_disp_norm], dim=1)
    world_edge_features = (world_edge_features - edge_mean) / edge_std

    # Concatenate the new features to the existing ones and assign
    # world_edge_features = torch.tensor(world_edge_features, dtype=mesh_edge_features.dtype, device=mesh_edge_features.device)

    # Compute the mesh edge features based on world pos
    row, col = graph.edge_index
    disp = world_pos[row] - world_pos[col]
    disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
    mesh_edges_world_pos = torch.cat((disp, disp_norm), dim=1)
    mesh_edges_world_pos = (mesh_edges_world_pos - edge_mean) / edge_std
    mesh_edge_features = torch.cat([graph.edge_attr, mesh_edges_world_pos], dim=1)

    # Duplicate world edge features because graph is homogeneous
    world_edge_features = world_edge_features.repeat(1, 2)

    # Add new edges to the graph
    graph.edge_index = torch.cat([graph.edge_index, world_edges], dim=1)

    all_edge_features = torch.cat([mesh_edge_features, world_edge_features], dim=0)
    graph.edge_attr = all_edge_features

    return graph, mesh_edge_features, world_edge_features
