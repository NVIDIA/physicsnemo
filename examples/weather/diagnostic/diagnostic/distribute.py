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

from typing import Tuple

import torch

from physicsnemo import Module
from physicsnemo.distributed import DistributedManager


def distribute_model(model: Module) -> Tuple[Module, DistributedManager]:
    """Distribute model using DDP.

    Parameters
    ----------
    model: physicsnemo.Module
        The model to be distributed

    Returns
    -------
    (model: physicsnemo.Module, dist: physicsnemo.distributed.DistributedManager)
        A tuple of the local copy of the distributed model and the
        DistributedManager object.
    """

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
