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
# ruff: noqa: E402
import os
import sys

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import numpy as np
import pytest
import torch
from meshgraphnet.utils import get_random_graph
from pytest_utils import import_or_fail

from physicsnemo.distributed import DistributedManager, mark_module_as_shared

torch.backends.cuda.matmul.allow_tf32 = False


def run_test_distributed_meshgraphnet(rank, world_size, dtype, partition_scheme):
    from physicsnemo.models.gnn_layers import (
        partition_graph_by_coordinate_bbox,
        partition_graph_nodewise,
        partition_graph_with_id_mapping,
    )
    from physicsnemo.models.gnn_layers.utils import CuGraphCSC
    from physicsnemo.models.meshgraphnet.meshgraphnet import MeshGraphNet

    os.environ["RANK"] = f"{rank}"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12355)

    DistributedManager.initialize()
    DistributedManager.create_process_subgroup(
        name="graph_partition",
        size=world_size,
    )
    manager = DistributedManager()
    assert manager.is_initialized() and manager._distributed

    model_kwds = {
        "input_dim_nodes": 32,
        "input_dim_edges": 32,
        "output_dim": 32,
        "processor_size": 2,
        "num_layers_node_processor": 2,
        "num_layers_edge_processor": 2,
        "hidden_dim_node_encoder": 32,
        "num_layers_node_encoder": 2,
        "hidden_dim_edge_encoder": 32,
        "num_layers_edge_encoder": 2,
        "hidden_dim_node_decoder": 32,
        "num_layers_node_decoder": 2,
    }

    # initialize single GPU model for reference
    torch.cuda.manual_seed(753)
    torch.manual_seed(753)
    np.random.seed(753)
    model_single_gpu = MeshGraphNet(**model_kwds).to(device=manager.device, dtype=dtype)

    # initialze distributed model with the same seeds
    torch.cuda.manual_seed(753)
    torch.manual_seed(753)
    np.random.seed(753)

    model_multi_gpu = MeshGraphNet(**model_kwds).to(device=manager.device, dtype=dtype)
    mark_module_as_shared(model_multi_gpu, "graph_partition")

    # initialize data
    torch.cuda.manual_seed(753)
    torch.manual_seed(753)
    np.random.seed(753)
    num_nodes = 256

    nfeat_single_gpu = torch.randn((num_nodes, model_kwds["input_dim_nodes"]))
    nfeat_single_gpu = nfeat_single_gpu.to(device=manager.device, dtype=dtype)
    nfeat_single_gpu = nfeat_single_gpu.requires_grad_(True)

    offsets, indices = get_random_graph(
        num_nodes=num_nodes,
        min_degree=3,
        max_degree=6,
    )

    efeat_single_gpu = torch.randn((indices.numel(), model_kwds["input_dim_edges"]))
    efeat_single_gpu = efeat_single_gpu.to(device=manager.device, dtype=dtype)
    efeat_single_gpu = efeat_single_gpu.requires_grad_(True)

    graph_single_gpu = CuGraphCSC(
        offsets.to(manager.device),
        indices.to(manager.device),
        num_nodes,
        num_nodes,
    )

    if partition_scheme == "nodewise":
        # nodewise should be default
        graph_partition = partition_graph_nodewise(
            offsets,
            indices,
            world_size,
            manager.rank,
            manager.device,
        )

    elif partition_scheme == "none":
        # check default which should be nodewise
        graph_partition = None

    elif partition_scheme == "coordinate_bbox":
        src_coordinates = torch.rand((num_nodes, 1), device=offsets.device)
        dst_coordinates = src_coordinates

        step_size = 1.0 / (world_size + 1)
        coordinate_separators_min = [[step_size * p] for p in range(world_size)]
        coordinate_separators_max = [[step_size * (p + 1)] for p in range(world_size)]
        coordinate_separators_min[0] = [None]
        coordinate_separators_max[-1] = [None]

        graph_partition = partition_graph_by_coordinate_bbox(
            offsets,
            indices,
            src_coordinates,
            dst_coordinates,
            coordinate_separators_min,
            coordinate_separators_max,
            world_size,
            manager.rank,
            manager.device,
        )

    elif partition_scheme == "mapping":
        mapping_src_ids_to_ranks = torch.randint(
            0, world_size, (num_nodes,), device=offsets.device
        )
        mapping_dst_ids_to_ranks = mapping_src_ids_to_ranks

        graph_partition = partition_graph_with_id_mapping(
            offsets,
            indices,
            mapping_src_ids_to_ranks,
            mapping_dst_ids_to_ranks,
            world_size,
            manager.rank,
            manager.device,
        )

    else:
        assert False  # only schemes above are supported

    graph_multi_gpu = CuGraphCSC(
        offsets.to(manager.device),
        indices.to(manager.device),
        num_nodes,
        num_nodes,
        partition_size=world_size,
        partition_group_name="graph_partition",
        graph_partition=graph_partition,
    )

    nfeat_multi_gpu = nfeat_single_gpu.detach().clone()
    nfeat_multi_gpu = graph_multi_gpu.get_src_node_features_in_partition(
        nfeat_multi_gpu
    ).requires_grad_(True)
    efeat_multi_gpu = efeat_single_gpu.detach().clone()
    efeat_multi_gpu = graph_multi_gpu.get_edge_features_in_partition(
        efeat_multi_gpu
    ).requires_grad_(True)

    # zero grads
    for param in model_single_gpu.parameters():
        param.grad = None
    for param in model_multi_gpu.parameters():
        param.grad = None

    # forward + backward passes
    out_single_gpu = model_single_gpu(
        nfeat_single_gpu, efeat_single_gpu, graph_single_gpu
    )
    loss = out_single_gpu.sum()
    loss.backward()

    out_multi_gpu = model_multi_gpu(nfeat_multi_gpu, efeat_multi_gpu, graph_multi_gpu)
    loss = out_multi_gpu.sum()
    loss.backward()

    # numeric tolerances based on dtype
    tolerances = {
        torch.float32: (1e-4, 1e-5),
        torch.float16: (1e-2, 1e-3),
    }
    tolerances_weight = {
        torch.float32: (1e-2, 1e-4),
        torch.float16: (1e-1, 1e-2),
    }
    atol, rtol = tolerances[dtype]
    atol_w, rtol_w = tolerances_weight[dtype]

    # compare forward
    out_single_gpu_dist = graph_multi_gpu.get_src_node_features_in_partition(
        out_single_gpu
    )
    diff = out_single_gpu_dist - out_multi_gpu
    diff = torch.abs(diff)
    mask = diff > atol
    assert torch.allclose(
        out_single_gpu_dist, out_multi_gpu, atol=atol, rtol=rtol
    ), f"{mask.sum()} elements have diff > {atol} \n {out_single_gpu_dist[mask]} \n {out_multi_gpu[mask]}"

    # compare data gradient
    nfeat_grad_single_gpu_dist = graph_multi_gpu.get_src_node_features_in_partition(
        nfeat_single_gpu.grad
    )
    diff = nfeat_grad_single_gpu_dist - nfeat_multi_gpu.grad
    diff = torch.abs(diff)
    mask = diff > atol
    assert torch.allclose(
        nfeat_multi_gpu.grad, nfeat_grad_single_gpu_dist, atol=atol_w, rtol=rtol_w
    ), f"{mask.sum()} elements have diff > {atol} \n {nfeat_grad_single_gpu_dist[mask]} \n {nfeat_multi_gpu.grad[mask]}"

    # compare model gradients (ensure correctness of backward)
    model_multi_gpu_parameters = list(model_multi_gpu.parameters())
    for param_idx, param in enumerate(model_single_gpu.parameters()):
        diff = param.grad - model_multi_gpu_parameters[param_idx].grad
        diff = torch.abs(diff)
        mask = diff > atol_w
        assert torch.allclose(
            param.grad,
            model_multi_gpu_parameters[param_idx].grad,
            atol=atol_w,
            rtol=rtol_w,
        ), f"{mask.sum()} for param[{param_idx}].grad elements have diff > {atol_w} with a avg. diff of {diff[mask].mean().item()} ({diff.mean().item()} overall)"

    # cleanup distributed
    del os.environ["RANK"]
    del os.environ["WORLD_SIZE"]
    del os.environ["MASTER_ADDR"]
    del os.environ["MASTER_PORT"]

    DistributedManager.cleanup()


@import_or_fail("dgl")
@pytest.mark.multigpu
@pytest.mark.parametrize(
    "partition_scheme", ["mapping", "nodewise", "coordinate_bbox", "none"]
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_distributed_meshgraphnet(dtype, partition_scheme, pytestconfig):
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"
    world_size = 2

    torch.multiprocessing.set_start_method("spawn", force=True)

    torch.multiprocessing.spawn(
        run_test_distributed_meshgraphnet,
        args=(world_size, dtype, partition_scheme),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    pytest.main([__file__])
