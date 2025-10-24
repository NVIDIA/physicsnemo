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

from datetime import datetime

import torch

from physicsnemo.distributed import DistributedManager


def create_ddp_group_tag(group_name: str = None) -> str:
    """Creates a common group tag for logging

    For some reason this does not work with multi-node. Seems theres a bug in PyTorch
    when one uses a distributed util before DDP

    Parameters
    ----------
    group_name : str, optional
        Optional group name prefix. If None will use ``"DDP_Group_"``, by default None

    Returns
    -------
    str
        Group tag
    """
    dist = DistributedManager()
    if dist.rank == 0:
        # Store time stamp as int tensor for broadcasting
        def tint(x):
            return int(datetime.now().strftime(f"%{x}"))

        time_index = torch.IntTensor(
            [tint(x) for x in ["m", "d", "y", "H", "M", "S"]]
        ).to(dist.device)
    else:
        time_index = torch.IntTensor([0, 0, 0, 0, 0, 0]).to(dist.device)

    if torch.distributed.is_available():
        # Broadcast group ID to all processes
        torch.distributed.broadcast(time_index, src=0)

    time_string = f"{time_index[0]}/{time_index[1]}/{time_index[2]}_\
        {time_index[3]}-{time_index[4]}-{time_index[5]}"

    if group_name is None:
        group_name = "DDP_Group"
    return group_name + "_" + time_string
