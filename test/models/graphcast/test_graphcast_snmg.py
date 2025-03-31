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

import pytest
import torch
from graphcast.utils import create_random_input, fix_random_seeds
from pytest_utils import import_or_fail

from physicsnemo.distributed import DistributedManager, mark_module_as_shared
from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet

torch.backends.cuda.matmul.allow_tf32 = False


def run_test_distributed_graphcast(
    rank: int,
    world_size: int,
    dtype: torch.dtype,
    do_concat_trick: bool,
    do_checkpointing: bool,
    use_lat_lon_partitioning: bool,
):
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

    res_h = 16
    res_w = 32

    model_kwds = {
        "multimesh_level": 2,
        "input_res": (res_h, res_w),
        "input_dim_grid_nodes": 34,
        "input_dim_mesh_nodes": 3,
        "input_dim_edges": 4,
        "output_dim_grid_nodes": 34,
        "processor_layers": 4,
        "hidden_dim": 32,
        "do_concat_trick": do_concat_trick,
        "use_cugraphops_encoder": True,
        "use_cugraphops_processor": True,
        "use_cugraphops_decoder": True,
    }

    device = manager.device

    # initialize single GPU model for reference
    fix_random_seeds(42)
    model_single_gpu = GraphCastNet(partition_size=1, **model_kwds).to(
        device=device, dtype=dtype
    )
    if do_checkpointing:
        model_single_gpu.set_checkpoint_model(True)

    # initialze distributed model with the same seeds
    fix_random_seeds(42)
    model_multi_gpu = GraphCastNet(
        partition_size=world_size,
        partition_group_name="graph_partition",
        expect_partitioned_input=True,
        produce_aggregated_output=False,
        use_lat_lon_partitioning=use_lat_lon_partitioning,
        **model_kwds,
    ).to(device=device, dtype=dtype)
    if do_checkpointing:
        model_multi_gpu.set_checkpoint_model(True)

    mark_module_as_shared(model_multi_gpu, "graph_partition")

    # initialize data
    x_single_gpu = create_random_input(
        input_res=model_kwds["input_res"], dim=model_kwds["input_dim_grid_nodes"]
    ).to(device=device, dtype=dtype)
    x_multi_gpu = x_single_gpu.detach().clone()
    x_multi_gpu = (
        x_multi_gpu[0].view(model_multi_gpu.input_dim_grid_nodes, -1).permute(1, 0)
    )
    x_multi_gpu = model_multi_gpu.g2m_graph.get_src_node_features_in_partition(
        x_multi_gpu
    ).requires_grad_(True)

    # forward + backward passes
    out_single_gpu = model_single_gpu(x_single_gpu)
    loss = out_single_gpu.sum()
    loss.backward()

    out_multi_gpu = model_multi_gpu(x_multi_gpu)
    loss = out_multi_gpu.sum()
    loss.backward()

    # numeric tolerances based on dtype
    tolerances = {
        torch.float32: (1e-5, 1e-6),
        torch.float16: (0.25, 1e-2),
    }
    tolerances_weight = {
        torch.float32: (1e-4, 1e-5),
        torch.float16: (0.25, 1e-2),
    }
    atol, rtol = tolerances[dtype]
    atol_w, rtol_w = tolerances_weight[dtype]

    # compare forward, now fully materialize out_multi_gpu to faciliate comparison
    out_multi_gpu = model_multi_gpu.m2g_graph.get_global_dst_node_features(
        out_multi_gpu
    )
    out_multi_gpu = out_multi_gpu.permute(1, 0).view(out_single_gpu.shape)
    diff = out_single_gpu - out_multi_gpu
    diff = torch.abs(diff)
    mask = diff > atol
    assert torch.allclose(
        out_single_gpu, out_multi_gpu, atol=atol, rtol=rtol
    ), f"{mask.sum()} elements have diff > {atol} \n {out_single_gpu[mask]} \n {out_multi_gpu[mask]}"

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
        ), f"{mask.sum()} elements have diff > {atol_w} \n {param.grad[mask]} \n {model_multi_gpu_parameters[param_idx].grad[mask]}"

    # cleanup distributed
    DistributedManager.cleanup()
    del os.environ["RANK"]
    del os.environ["WORLD_SIZE"]
    del os.environ["MASTER_ADDR"]
    del os.environ["MASTER_PORT"]


@import_or_fail("dgl")
@pytest.mark.multigpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("do_concat_trick", [False, True])
@pytest.mark.parametrize("do_checkpointing", [False, True])
@pytest.mark.parametrize("use_lat_lon_partitioning", [False, True])
def test_distributed_graphcast(
    dtype, do_concat_trick, do_checkpointing, use_lat_lon_partitioning, pytestconfig
):
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"
    world_size = min(
        4, num_gpus
    )  # test-graph is otherwise too small for distribution across more GPUs

    torch.multiprocessing.set_start_method("spawn", force=True)

    torch.multiprocessing.spawn(
        run_test_distributed_graphcast,
        args=(
            world_size,
            dtype,
            do_concat_trick,
            do_checkpointing,
            use_lat_lon_partitioning,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    pytest.main([__file__])
