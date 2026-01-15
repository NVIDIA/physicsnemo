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

import os

import pytest
import torch

from physicsnemo.distributed import DistributedManager


def get_random_graph(device):
    """test utility: create random graph for this test"""
    num_src_nodes = 4321
    num_dst_nodes = 1234
    min_degree = 2
    max_degree = 8
    degree = torch.randint(
        min_degree, max_degree + 1, (num_dst_nodes,), device=device, dtype=torch.int64
    )
    degree = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=device),
            degree,
        ]
    )
    offsets = torch.cumsum(degree, dim=0)
    indices = torch.randint(
        0,
        num_src_nodes,
        (offsets[-1],),
        device=device,
        dtype=torch.int64,
    )
    if max(indices).item() != num_src_nodes - 1:
        indices[-1] = num_src_nodes - 1
    return (offsets, indices, num_src_nodes, num_dst_nodes)


def get_random_feat(num_src_nodes, num_dst_nodes, num_indices, num_channels, device):
    """test utility: create random node and edge features for given graph"""
    src_feat = (
        10
        * torch.rand((num_src_nodes, num_channels), dtype=torch.float32, device=device)
        + 16
    )
    dst_feat = (
        10
        * torch.rand((num_dst_nodes, num_channels), dtype=torch.float32, device=device)
        + 8
    )
    edge_feat = (
        10 * torch.rand((num_indices, num_channels), dtype=torch.float32, device=device)
        + 4
    )
    src_feat.requires_grad_(True)
    dst_feat.requires_grad_(True)
    edge_feat.requires_grad_(True)
    return src_feat, dst_feat, edge_feat


def get_random_lat_lon_coordinates(num_src_nodes, num_dst_nodes, device):
    """test utility: create random lat-lon coordintes for given graph nodes"""
    x_src = torch.rand(
        (num_src_nodes, 2),
        dtype=torch.float32,
        device=device,
    )
    x_src[:, 0] = x_src[:, 0] * 180 - 90
    x_src[:, 1] = x_src[:, 1] * 360 - 180

    x_dst = torch.rand(
        (num_dst_nodes, 2),
        dtype=torch.float32,
        device=device,
    )
    x_dst[:, 0] = x_dst[:, 0] * 180 - 90
    x_dst[:, 1] = x_dst[:, 1] * 360 - 180

    return x_src, x_dst


def broadcast(src: torch.Tensor, ref: torch.Tensor, dim: int) -> torch.Tensor:
    """helper function for scatter_reduce"""
    size = ((1,) * dim) + (-1,) + ((1,) * (ref.dim() - dim - 1))
    return src.view(size).expand_as(ref)


def scatter_reduce(
    feat: torch.Tensor,
    offsets: torch.Tensor,
    indices: torch.Tensor,
):
    """helper function for message-passing without further dependencies
    or graph format conversions"""
    num_dst = offsets.size(0) - 1

    src = feat[indices]
    degree = offsets[1:] - offsets[0:-1]
    ids = torch.arange(0, num_dst, dtype=torch.int64, device=feat.device)
    index = ids.repeat_interleave(degree)
    index = broadcast(index, src, 0).to(torch.int64)
    size = list(src.size())
    size[0] = num_dst

    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_reduce_(0, index, src, reduce="sum", include_self=False)


def run_test_distributed_graph(
    rank: int,
    world_size: int,
    partition_scheme: str,
    use_torchrun: bool = False,
):
    from physicsnemo.models.graphcast.graph_cast_net import (
        get_lat_lon_partition_separators,
    )
    from physicsnemo.nn.gnn_layers import (
        DistributedGraph,
        partition_graph_by_coordinate_bbox,
    )

    print(f"Rank {rank} checking in")

    os.environ["RANK"] = f"{rank}"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12355)
    os.environ["LOCAL_RANK"] = f"{rank % torch.cuda.device_count()}"

    DistributedManager.initialize()

    manager = DistributedManager()
    assert manager.is_initialized() and manager._distributed

    print(f"Rank {rank} device: {manager.device}")

    # actual test
    num_channels = 64
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    offsets, indices, num_src_nodes, num_dst_nodes = get_random_graph(manager.device)
    num_indices = indices.numel()
    src_feat, dst_feat, edge_feat = get_random_feat(
        num_src_nodes, num_dst_nodes, num_indices, num_channels, manager.device
    )

    if partition_scheme == "lat_lon_bbox":
        x_src, x_dst = get_random_lat_lon_coordinates(
            num_src_nodes, num_dst_nodes, manager.device
        )
        min_seps, max_seps = get_lat_lon_partition_separators(manager.world_size)
        graph_partition = partition_graph_by_coordinate_bbox(
            offsets,
            indices,
            x_src,
            x_dst,
            min_seps,
            max_seps,
            partition_size=manager.world_size,
            partition_rank=manager.rank,
            device=manager.device,
        ).to(device=manager.device)

    else:
        graph_partition = None

    dist_graph = DistributedGraph(
        offsets,
        indices,
        partition_size=manager.world_size,
        graph_partition=graph_partition,
    )

    # src-feat
    for scatter_features in [False, True]:
        for get_on_all_ranks in [False, True]:
            src_feat.grad = None
            local_src_feat = dist_graph.get_src_node_features_in_partition(
                src_feat,
                scatter_features=scatter_features,
            )
            global_src_feat = dist_graph.get_global_src_node_features(
                local_src_feat,
                get_on_all_ranks=get_on_all_ranks,
            )
            loss = global_src_feat.sum()
            if get_on_all_ranks:
                loss = loss / dist_graph.partition_size
            loss.backward()

            if get_on_all_ranks or (dist_graph.partition_rank == 0):
                assert torch.allclose(global_src_feat, src_feat)

            if scatter_features:
                if dist_graph.partition_rank == 0:
                    assert torch.allclose(src_feat.grad, torch.ones_like(src_feat))
            else:
                with torch.no_grad():
                    grad_local = dist_graph.get_src_node_features_in_partition(
                        src_feat.grad,
                        scatter_features=False,
                    )
                    assert torch.allclose(grad_local, torch.ones_like(local_src_feat))

            # test "local-graph" by comparing against simpel reduction on "global graph"
            global_src_feat = src_feat.detach().clone()
            global_src_feat.requires_grad_(True)
            src_feat.grad = None
            local_src_feat = dist_graph.get_src_node_features_in_partition(
                src_feat,
                scatter_features=scatter_features,
            )

            global_agg = scatter_reduce(global_src_feat, offsets, indices)

            local_src_feat = dist_graph.get_src_node_features_in_local_graph(
                local_src_feat
            )

            local_agg = scatter_reduce(
                local_src_feat,
                dist_graph.graph_partition.local_offsets,
                dist_graph.graph_partition.local_indices,
            )
            local_agg = dist_graph.get_global_dst_node_features(
                local_agg,
                get_on_all_ranks=get_on_all_ranks,
            )
            if get_on_all_ranks or (dist_graph.partition_rank == 0):
                assert torch.allclose(local_agg, global_agg)

            local_loss = local_agg.sum()
            if get_on_all_ranks:
                local_loss = local_loss / dist_graph.partition_size
            global_loss = global_agg.sum()
            local_loss.backward()
            global_loss.backward()

            if scatter_features:
                if dist_graph.partition_rank == 0:
                    assert torch.allclose(src_feat.grad, global_src_feat.grad)
            else:
                with torch.no_grad():
                    grad_local = dist_graph.get_src_node_features_in_partition(
                        src_feat.grad,
                        scatter_features=False,
                    )
                    grad_global_local = dist_graph.get_src_node_features_in_partition(
                        global_src_feat.grad,
                        scatter_features=False,
                    )
                    assert torch.allclose(grad_local, grad_global_local)

    # dst-feat
    for scatter_features in [False, True]:
        for get_on_all_ranks in [False, True]:
            dst_feat.grad = None
            local_dst_feat = dist_graph.get_dst_node_features_in_partition(
                dst_feat,
                scatter_features=scatter_features,
            )
            global_dst_feat = dist_graph.get_global_dst_node_features(
                local_dst_feat,
                get_on_all_ranks=get_on_all_ranks,
            )

            loss = global_dst_feat.sum()
            if get_on_all_ranks:
                loss = loss / dist_graph.partition_size
            loss.backward()

            if get_on_all_ranks:
                assert torch.allclose(global_dst_feat, dst_feat)
            else:
                if dist_graph.partition_rank == 0:
                    assert torch.allclose(global_dst_feat, dst_feat)

            if scatter_features:
                if dist_graph.partition_rank == 0:
                    assert torch.allclose(dst_feat.grad, torch.ones_like(dst_feat))
            else:
                grad = dst_feat.grad
                with torch.no_grad():
                    grad_local = dist_graph.get_dst_node_features_in_partition(
                        grad,
                        scatter_features=False,
                    )
                    assert torch.allclose(grad_local, torch.ones_like(local_dst_feat))

    # edge-feat
    for scatter_features in [False, True]:
        for get_on_all_ranks in [False, True]:
            edge_feat.grad = None
            local_edge_feat = dist_graph.get_edge_features_in_partition(
                edge_feat,
                scatter_features=scatter_features,
            )
            global_edge_feat = dist_graph.get_global_edge_features(
                local_edge_feat,
                get_on_all_ranks=get_on_all_ranks,
            )

            loss = global_edge_feat.sum()
            if get_on_all_ranks:
                loss = loss / dist_graph.partition_size
            loss.backward()

            if get_on_all_ranks:
                assert torch.allclose(global_edge_feat, edge_feat)
            else:
                if dist_graph.partition_rank == 0:
                    assert torch.allclose(global_edge_feat, edge_feat)

            if scatter_features:
                if dist_graph.partition_rank == 0:
                    assert torch.allclose(edge_feat.grad, torch.ones_like(edge_feat))
            else:
                grad = edge_feat.grad
                with torch.no_grad():
                    grad_local = dist_graph.get_edge_features_in_partition(
                        edge_feat.grad,
                        scatter_features=False,
                    )
                    assert torch.allclose(grad_local, torch.ones_like(local_edge_feat))


@pytest.mark.multigpu_dynamic
@pytest.mark.parametrize("partition_scheme", ["lat_lon_bbox", "default"])
def test_distributed_graph(partition_scheme, pytestconfig):
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"
    world_size = 2  # num_gpus

    torch.multiprocessing.set_start_method("spawn", force=True)

    torch.multiprocessing.spawn(
        run_test_distributed_graph,
        args=(
            world_size,
            partition_scheme,
        ),
        nprocs=world_size,
        join=True,
        daemon=True,
    )


if __name__ == "__main__":
    # to be launched with torchrun
    DistributedManager.initialize()
    run_test_distributed_graph(-1, -1, "lat_lon_bbox", True)
    run_test_distributed_graph(-1, -1, "default", True)
    DistributedManager.cleanup()
