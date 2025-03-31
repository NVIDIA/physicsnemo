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


import os
import torch
from . import training_stats
from torch.distributed import is_initialized

# ----------------------------------------------------------------------------


def init():
    """
    Initialize the distributed training environment.
    """
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"

    backend = "gloo" if os.name == "nt" else "nccl"
    torch.distributed.init_process_group(backend=backend, init_method="env://")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    sync_device = torch.device("cuda") if get_world_size() > 1 else None
    training_stats.init_multiprocessing(rank=get_rank(), sync_device=sync_device)


# ----------------------------------------------------------------------------


def get_rank():
    """
    Get the rank of the current process.
    """
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


# ----------------------------------------------------------------------------


def get_world_size():
    """
    Get the number of processes in the current distributed group.
    """
    return (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )


# ----------------------------------------------------------------------------


def should_stop():
    return False


# ----------------------------------------------------------------------------


def update_progress(cur, total):
    """
    Update the progress of the current process.
    """
    _ = cur, total


# ----------------------------------------------------------------------------


def print0(*args, **kwargs):
    """
    Print a message from the root process.
    """
    if get_rank() == 0:
        print(*args, **kwargs)


# ----------------------------------------------------------------------------
