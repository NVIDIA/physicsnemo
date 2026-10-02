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

"""Coordinate slow CPU dataset initialization without outstanding NCCL work."""

import math
from datetime import timedelta

import torch.distributed as dist


def initialize_datasets(build, timeout_seconds=3600):
    """Run ``build(stats_mode)`` on every rank, publishing rank-zero data first.

    ``build`` must initialize BOTH training and validation datasets. Rank zero
    alone publishes normalization files and warms shared geometry caches. All
    ranks finish before returning to any GPU collectives. CPU failure messages
    are propagated collectively; a process crash is bounded by the Gloo timeout.
    This does not increase the training process group's NCCL timeout.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Dataset startup timeout must be finite and positive")
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return build("compute")
    group = dist.new_group(backend="gloo", timeout=timedelta(seconds=timeout_seconds))
    result = None
    try:
        status = [None]
        if dist.get_rank() == 0:
            try:
                result = build("compute")
            except Exception as error:
                status[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(status, src=0, group=group)
        if status[0] is not None:
            raise RuntimeError(f"Rank-zero dataset startup failed: {status[0]}")
        failure = None
        if dist.get_rank() != 0:
            try:
                result = build("load")
            except Exception as error:
                failure = f"rank {dist.get_rank()}: {type(error).__name__}: {error}"
        failures = [None] * dist.get_world_size()
        dist.all_gather_object(failures, failure, group=group)
        if any(failures):
            raise RuntimeError(f"Dataset startup failed: {failures}")
        return result
    finally:
        dist.destroy_process_group(group)
