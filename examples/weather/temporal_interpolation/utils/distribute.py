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

from physicsnemo.distributed import DistributedManager
import torch


def distribute_model(
    model: torch.nn.Module,
) -> tuple[torch.nn.Module, DistributedManager]:
    """
    Initialize DistributedManager and distribute model to multiple processes with DDP.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be distributed across multiple processes.

    Returns
    -------
    tuple[torch.nn.Module, DistributedManager]
        A tuple containing:
        - model : torch.nn.Module
            The model, wrapped with DistributedDataParallel if needed.
        - dist : DistributedManager
            The initialized DistributedManager instance.
    """
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()

    dist = DistributedManager()
    model = model.to(dist.device)

    if dist.world_size > 1:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    return (model, dist)
