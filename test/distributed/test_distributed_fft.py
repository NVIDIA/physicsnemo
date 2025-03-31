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
import torch.distributed as dist
from distributed_utils_for_testing import modify_environment

from physicsnemo.distributed import DistributedManager
from physicsnemo.distributed.fft import DistributedRFFT2


def distributed_setup(rank, model_parallel_size, verbose):

    DistributedManager._shared_state = {}

    # Setup distributed process config
    DistributedManager.initialize()

    # Create model parallel process group
    DistributedManager.create_process_subgroup(
        "model_parallel", int(model_parallel_size), verbose=verbose
    )
    # Create data parallel process group for DDP allreduce
    DistributedManager.create_orthogonal_process_group(
        "data_parallel", "model_parallel", verbose=verbose
    )
    # Create spatial parallel process group
    DistributedManager.create_process_subgroup(
        "spatial_parallel",
        int(model_parallel_size),
        group_name="model_parallel",
        verbose=verbose,
    )
    if verbose:
        print(DistributedManager())


def global_rfft2(inp, dim, norm, s=None):
    """Wrapper to compute single GPU 2D FFT"""
    if s is not None:
        x = torch.fft.rfft(inp, dim=dim[-1], norm=norm, s=s[-1])
        x = torch.fft.fft(x, dim=dim[0], norm=norm, s=s[0])
    else:
        x = torch.fft.rfft(inp, dim=dim[-1], norm=norm)
        x = torch.fft.fft(x, dim=dim[0], norm=norm)
    return x


def run_distributed_fft(rank, model_parallel_size, verbose):
    with modify_environment(
        RANK=f"{rank}",
        WORLD_SIZE=f"{model_parallel_size}",
        MASTER_ADDR="localhost",
        MASTER_PORT=str(12355),
        LOCAL_RANK=f"{rank % torch.cuda.device_count()}",
    ):
        # Setup DistributedManager
        distributed_setup(rank, model_parallel_size, verbose)

        B = 2  # batch size
        C = 10  # channels
        H = 720  # height
        W = 1440  # width

        input_split_dim = -1  # dimension to split inputs
        output_split_dim = -2  # dimension to split inputs

        manager = DistributedManager()

        if verbose and manager.rank == 0:
            print(
                "Running FFT for "
                f"({B}, {C}, {H}, {W}) on {manager.group_size(name='spatial_parallel')}"
                " ranks"
            )

        # Set random seed for reproducible tests
        torch.cuda.manual_seed(13)

        # Create inputs
        global_input = torch.rand(
            (B, C, H, W), dtype=torch.float32, device=manager.device, requires_grad=True
        )

        if manager.distributed:
            # Broadcast global input from rank 0 to all other ranks
            dist.broadcast(
                global_input, src=0, group=manager.group(name="spatial_parallel")
            )
        torch.cuda.synchronize()

        # Split global input to get each rank's local input
        with torch.no_grad():
            split_size = global_input.shape[input_split_dim] // manager.group_size(
                name="spatial_parallel"
            )
            tmp = torch.split(global_input, split_size, dim=input_split_dim)[
                manager.group_rank(name="spatial_parallel")
            ].contiguous()
            local_input = torch.empty_like(tmp, requires_grad=True)
            local_input.copy_(tmp)
        torch.cuda.synchronize()

        local_output = DistributedRFFT2.apply(
            local_input, (None, None), (-2, -1), "ortho"
        )
        dist.barrier()

        global_output = global_rfft2(global_input, dim=(-2, -1), norm="ortho")

        # Split global fft and get local shard
        with torch.no_grad():
            split_size = global_output.shape[output_split_dim] // manager.group_size(
                name="spatial_parallel"
            )
            split_global_output = torch.split(
                global_output, split_size, dim=output_split_dim
            )[manager.group_rank(name="spatial_parallel")].contiguous()

        if verbose:
            print(f"local_output.shape = {local_output.shape}")
            print(f"global_output.shape = {global_output.shape}")
            print(f"split_global_output.shape = {split_global_output.shape}")

        # Ensure that distributed FFT matches single GPU
        assert torch.allclose(
            local_output, split_global_output, rtol=1e-3, atol=1e-3
        ), "Distributed FFT does not match single GPU version!"

        # Now test backward pass
        # Create input gradients
        global_output_grads = torch.rand_like(global_output).contiguous()

        # Global gradients
        global_output.backward(global_output_grads)
        global_input_grads = global_input.grad.clone().contiguous()

        if manager.distributed:
            # Broadcast global input from rank 0 to all other ranks
            global_output_grads_tmp = torch.view_as_real(global_output_grads)
            dist.broadcast(
                global_output_grads_tmp,
                src=0,
                group=manager.group(name="spatial_parallel"),
            )
            global_output_grads = torch.view_as_complex(global_output_grads_tmp)
        torch.cuda.synchronize()

        # Split global grads and get local shard
        with torch.no_grad():
            split_size = global_output_grads.shape[
                output_split_dim
            ] // manager.group_size(name="spatial_parallel")
            split_global_output_grads = torch.split(
                global_output_grads, split_size, dim=output_split_dim
            )[manager.group_rank(name="spatial_parallel")].contiguous()

        # Distributed gradients
        local_output.backward(split_global_output_grads)
        local_input_grads = local_input.grad.clone()

        # Split global input grads and get local shard
        with torch.no_grad():
            split_size = global_input_grads.shape[
                input_split_dim
            ] // manager.group_size(name="spatial_parallel")
            split_global_input_grads = torch.split(
                global_input_grads, split_size, dim=input_split_dim
            )[manager.group_rank(name="spatial_parallel")].contiguous()

        if verbose:
            print(f"global_output_grads.shape = {global_output_grads.shape}")
            print(
                f"split_global_output_grads.shape = {split_global_output_grads.shape}"
            )
            print(f"local_input_grads.shape = {local_input_grads.shape}")
            print(f"global_input_grads.shape = {global_input_grads.shape}")
            print(f"split_global_input_grads.shape = {split_global_input_grads.shape}")

        # Ensure that distributed FFT backward matches single GPU
        assert torch.allclose(
            local_input_grads, split_global_input_grads, rtol=1e-3, atol=1e-3
        ), "Distributed FFT backward does not match single GPU version!"


@pytest.mark.multigpu
def test_distributed_fft():
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"
    model_parallel_size = 2
    verbose = False  # Change to True for debug

    torch.multiprocessing.set_start_method("spawn", force=True)

    torch.multiprocessing.spawn(
        run_distributed_fft,
        args=(model_parallel_size, verbose),
        nprocs=model_parallel_size,
    )


if __name__ == "__main__":
    pytest.main([__file__])
