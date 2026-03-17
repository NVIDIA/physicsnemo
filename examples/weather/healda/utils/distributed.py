# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
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
from training import training_stats

from physicsnemo.distributed import DistributedManager

# ----------------------------------------------------------------------------


def init():
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

    DistributedManager.initialize()
    manager = DistributedManager()

    sync_device = manager.device if manager.world_size > 1 else None
    training_stats.init_multiprocessing(rank=manager.rank, sync_device=sync_device)


# ----------------------------------------------------------------------------


def get_rank():
    """Return current process rank, or 0 if not distributed."""
    if not DistributedManager.is_initialized():
        return 0
    return DistributedManager().rank


# ----------------------------------------------------------------------------


def get_world_size():
    """Return world size, or 1 if not distributed."""
    if not DistributedManager.is_initialized():
        return 1
    return DistributedManager().world_size


# ----------------------------------------------------------------------------


def should_stop():
    return False


# ----------------------------------------------------------------------------


def update_progress(cur, total):
    """Progress callback stub (no-op by default)."""
    _ = cur, total


# ----------------------------------------------------------------------------


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


# ----------------------------------------------------------------------------
