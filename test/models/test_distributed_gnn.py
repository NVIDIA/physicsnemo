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
"""MeshGraphNet and GraphCast on a graph split across GPUs match a single GPU."""

import copy
import os

import pytest
import torch

from physicsnemo.distributed import DistributedManager, mark_module_as_shared
from test.conftest import requires_module


def _init_distributed(rank, world_size):
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="localhost",
        MASTER_PORT="12355",
        LOCAL_RANK=str(rank % torch.cuda.device_count()),
    )
    DistributedManager.initialize()
    DistributedManager.create_process_subgroup(name="graph_partition", size=world_size)
    return DistributedManager()


def _assert_param_grads_close(model, model_dist, atol, rtol):
    for (name, p), p_dist in zip(model.named_parameters(), model_dist.parameters()):
        torch.testing.assert_close(
            p_dist.grad, p.grad, atol=atol, rtol=rtol, msg=lambda m: f"{name}: {m}"
        )


def run_test_distributed_meshgraphnet(rank, world_size, partition_scheme):
    from torch_geometric.data import Data

    from physicsnemo.models.meshgraphnet import MeshGraphNet
    from physicsnemo.nn.module.gnn_layers import (
        DistributedGraph,
        edge_index_to_csc,
        partition_graph_by_coordinate_bbox,
        partition_graph_with_id_mapping,
    )

    manager = _init_distributed(rank, world_size)
    device, n = manager.device, 256
    torch.manual_seed(753)
    model = MeshGraphNet(
        input_dim_nodes=32,
        input_dim_edges=32,
        output_dim=32,
        processor_size=2,
        hidden_dim_node_encoder=32,
        hidden_dim_edge_encoder=32,
        hidden_dim_node_decoder=32,
    ).to(device)
    model_dist = mark_module_as_shared(copy.deepcopy(model), "graph_partition")

    # 3 to 6 incoming edges per node, in random order to exercise edge_perm, and
    # no outgoing edges from the last 5 nodes
    dst = torch.repeat_interleave(torch.randint(3, 7, (n,)))
    src = torch.randint(0, n - 5, dst.shape)
    edge_index = torch.stack([src, dst])[:, torch.randperm(dst.numel())]
    nfeat = torch.randn(n, 32, device=device, requires_grad=True)
    efeat = torch.randn(edge_index.size(1), 32, device=device)

    offsets, indices, edge_perm = edge_index_to_csc(edge_index, n)
    group_name, process_group, partition = "graph_partition", None, None
    if partition_scheme == "mapping":
        ranks = torch.randint(0, world_size, (n,))
        partition = partition_graph_with_id_mapping(
            offsets, indices, ranks, ranks, world_size, rank, device
        )
    elif partition_scheme == "coordinate_bbox":
        x = torch.rand(n, 1)
        seps = [[r / world_size] for r in range(1, world_size)]
        mins, maxs = [[None]] + seps, seps + [[None]]
        partition = partition_graph_by_coordinate_bbox(
            offsets, indices, x, x, mins, maxs, world_size, rank, device
        )
    else:  # default partitioning, with the process group passed directly
        group_name, process_group = None, manager.group("graph_partition")
    graph = DistributedGraph(
        offsets,
        indices,
        world_size,
        group_name,
        partition,
        process_group,
        num_src_nodes=n,
    )

    out = model(nfeat, efeat, Data(edge_index=edge_index, num_nodes=n).to(device))
    out.sum().backward()
    nfeat_dist = graph.get_src_node_features_in_partition(nfeat.detach())
    nfeat_dist.requires_grad_(True)
    efeat_dist = graph.get_edge_features_in_partition(efeat[edge_perm])
    out_dist = model_dist(nfeat_dist, efeat_dist, graph)
    out_dist.sum().backward()
    if partition_scheme == "process_group":
        # without the node count, source and destination partitions differ
        graph_short = DistributedGraph(
            offsets, indices, world_size, None, None, process_group
        )
        with pytest.raises(ValueError, match="same partition"):
            model_dist(nfeat_dist, efeat_dist, graph_short)

    out_expected = graph.get_dst_node_features_in_partition(out.detach())
    torch.testing.assert_close(out_dist, out_expected, atol=1e-4, rtol=1e-5)
    grad_expected = graph.get_src_node_features_in_partition(nfeat.grad)
    torch.testing.assert_close(nfeat_dist.grad, grad_expected, atol=1e-2, rtol=1e-4)
    _assert_param_grads_close(model, model_dist, atol=1e-2, rtol=1e-4)
    DistributedManager.cleanup()


def run_test_distributed_graphcast(
    rank, world_size, dtype, do_concat_trick, do_checkpointing, use_lat_lon_partitioning
):
    from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet

    manager = _init_distributed(rank, world_size)
    # a coarse grid on a fine mesh, so m2g gives the last mesh nodes no edges
    kwargs = dict(
        mesh_level=5,
        input_res=(10, 20),
        input_dim_grid_nodes=34,
        input_dim_mesh_nodes=3,
        input_dim_edges=4,
        output_dim_grid_nodes=34,
        processor_layers=4,
        hidden_dim=32,
        do_concat_trick=do_concat_trick,
    )
    torch.manual_seed(42)
    model = GraphCastNet(**kwargs).to(device=manager.device, dtype=dtype)
    model_dist = GraphCastNet(
        partition_size=world_size,
        partition_group_name="graph_partition",
        expect_partitioned_input=True,
        produce_aggregated_output=False,
        use_lat_lon_partitioning=use_lat_lon_partitioning,
        **kwargs,
    ).to(device=manager.device, dtype=dtype)
    model_dist.load_state_dict(model.state_dict())
    mark_module_as_shared(model_dist, "graph_partition")
    if do_checkpointing:
        model.set_checkpoint_model(True)
        model_dist.set_checkpoint_model(True)

    x = torch.randn(1, 34, 10, 20, device=manager.device, dtype=dtype)
    out = model(x)
    out.sum().backward()
    x_dist = model_dist.g2m_graph.get_src_node_features_in_partition(x[0].flatten(1).T)
    out_dist = model_dist(x_dist)
    out_dist.sum().backward()

    out_dist = model_dist.m2g_graph.get_global_dst_node_features(out_dist.detach())
    tols = {
        torch.float32: (1e-5, 1e-6, 1e-4, 1e-4),
        torch.float16: (0.25, 1e-2, 0.25, 1e-2),
    }
    atol, rtol, atol_w, rtol_w = tols[dtype]
    torch.testing.assert_close(out_dist, out[0].flatten(1).T, atol=atol, rtol=rtol)
    _assert_param_grads_close(model, model_dist, atol=atol_w, rtol=rtol_w)
    DistributedManager.cleanup()


def _spawn(fn, world_size, *args):
    assert world_size >= 2, "Not enough GPUs available for test"
    torch.multiprocessing.spawn(fn, args=(world_size, *args), nprocs=world_size)


@requires_module(["torch_geometric", "torch_scatter"])
@pytest.mark.multigpu_dynamic
@pytest.mark.parametrize(
    "partition_scheme", ["mapping", "coordinate_bbox", "process_group"]
)
def test_distributed_meshgraphnet(partition_scheme):
    world_size = torch.cuda.device_count()
    _spawn(run_test_distributed_meshgraphnet, world_size, partition_scheme)


@requires_module(["torch_geometric", "torch_sparse"])
@pytest.mark.multigpu_dynamic
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("do_concat_trick", [False, True])
@pytest.mark.parametrize("do_checkpointing", [False, True])
@pytest.mark.parametrize("use_lat_lon_partitioning", [False, True])
def test_distributed_graphcast(
    dtype, do_concat_trick, do_checkpointing, use_lat_lon_partitioning
):
    # the test graph is too small to split across more than 4 GPUs
    world_size = min(4, torch.cuda.device_count())
    flags = (do_concat_trick, do_checkpointing, use_lat_lon_partitioning)
    _spawn(run_test_distributed_graphcast, world_size, dtype, *flags)
