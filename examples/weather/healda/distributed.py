# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import torch
from healda import training_stats
import datetime
# ----------------------------------------------------------------------------


def init(timeout_infinite=False):
    if "WORLD_SIZE" not in os.environ:
        if "SLURM_NTASKS" in os.environ:
            os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS", "1")
        else:
            os.environ["WORLD_SIZE"] = "1"
    if "MASTER_ADDR" not in os.environ:
        if (
            int(os.environ["WORLD_SIZE"]) > 1
            and "SLURM_LAUNCH_NODE_IPADDR" in os.environ
        ):
            os.environ["MASTER_ADDR"] = os.environ.get(
                "SLURM_LAUNCH_NODE_IPADDR", "localhost"
            )
        else:
            os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if "RANK" not in os.environ:
        if "SLURM_PROCID" in os.environ:
            os.environ["RANK"] = os.environ.get("SLURM_PROCID", "0")
        else:
            os.environ["RANK"] = "0"
    if "LOCAL_RANK" not in os.environ:
        if "SLURM_LOCALID" in os.environ:
            os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")
        else:
            os.environ["LOCAL_RANK"] = "0"

    backend = "gloo" if os.name == "nt" else "nccl"
    if timeout_infinite:
        timeout = datetime.timedelta(days=365)
    else:
        timeout = None

    device_id = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device_id)
    torch.distributed.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timeout,
        device_id=torch.device("cuda", index=device_id),
    )

    sync_device = torch.device("cuda") if get_world_size() > 1 else None
    training_stats.init_multiprocessing(rank=get_rank(), sync_device=sync_device)


# ----------------------------------------------------------------------------


def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


# ----------------------------------------------------------------------------


def get_world_size():
    return (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )


# ----------------------------------------------------------------------------


def should_stop():
    return False


# ----------------------------------------------------------------------------


def update_progress(cur, total):
    _ = cur, total


# ----------------------------------------------------------------------------


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


# ----------------------------------------------------------------------------
