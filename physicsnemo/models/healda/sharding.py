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
"""
Distributed utilities for working with [b t x c] shaped data

there are two states
    - t-sharded (the input)
    - x-sharded (used for temporal attention)
"""

import einops
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_to_all_single

DATA_DIM = 0
MODEL_DIM = 1


def shard_x(tensor, group):
    """unshard t and shard x across ranks

    Args:
        tensor: (b, t, n x, c) sharded in t.
        group: model parallel group

    """
    n = dist.get_world_size(group)
    tensor = einops.rearrange(tensor, "b t (n x) c -> n b t x c", n=n)
    tensor = tensor.contiguous()
    output = torch.empty_like(tensor)
    output = all_to_all_single(output, tensor, group=group)
    output = einops.rearrange(output, "n b t x c -> b (n t) x c")
    return output


def shard_t(tensor, group):
    """unshard x and shard t across ranks

    Args:
        tensor: (b, n t, x, c) sharded in t.
        group: model parallel group
    Returns
        (b, t, n x, c)

    """
    n = dist.get_world_size(group)
    tensor = einops.rearrange(tensor, "b (n t) x c -> n b t x c", n=n)
    tensor = tensor.contiguous()
    output = torch.empty_like(tensor)
    output = all_to_all_single(output, tensor, group=group)
    output = einops.rearrange(output, "n b t x c -> b t (n x) c")
    return output
