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

from typing import List

import time
import numpy as np
from jaxtyping import Float

import torch
from torch import Tensor
import torch.distributed as dist


def all_gather_size(
    tensor: Float[Tensor, "*"] = None, flatten: bool = True
) -> List[int]:
    """
    All gather tensor size
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if tensor is not None:
        tensor_size = [s for s in tensor.size()]
    else:
        tensor_size = [0]
    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, tensor_size)
    return output


def all_gather_tensor(
    tensor: Float[Tensor, "*"], flatten: bool
) -> List[Float[Tensor, "*"]]:
    """
    All gather tensor with different sizes
    """
    assert tensor is not None
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)

    # if backend nccl
    if dist.get_backend() == "nccl":
        tensor = tensor.cuda()

    if flatten:
        tensor = tensor.flatten()
    all_sizes = all_gather_size(tensor)
    # create empty tensors
    output = [
        torch.empty(tuple(size), dtype=tensor.dtype, device=tensor.device)
        for size in all_sizes
    ]
    dist.all_gather(output, tensor)
    return output


class AverageMeter:
    """Average meter."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """update"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterDict:
    """Average Meter with dictionary values."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = {}
        self.avg = {}
        self.sum = {}
        self.count = {}
        self.max = {}
        self._private_attributes = {}

    def all_gather_attributes(self):
        """
        All gather private attributes to a tensor
        """
        # Check if dist is initialized
        for k, v in self._private_attributes.items():
            if isinstance(v, list) and isinstance(v[0], torch.Tensor):
                v = torch.cat(v)
                self._private_attributes[k] = v

            if dist.is_initialized():
                self._private_attributes[k] = torch.cat(
                    all_gather_tensor(v.cuda(), flatten=True)
                ).cpu()

        if dist.is_initialized():
            # Merge sum, count, max and compute avg
            for k in self.sum.keys():
                self.sum[k] = (
                    torch.cat(all_gather_tensor(self.sum[k], flatten=True))
                    .cpu()
                    .sum()
                    .item()
                )
                self.count[k] = (
                    torch.cat(all_gather_tensor(self.count[k], flatten=True))
                    .cpu()
                    .sum()
                    .item()
                )
                self.max[k] = (
                    torch.cat(all_gather_tensor(self.max[k], flatten=True))
                    .cpu()
                    .max()
                    .item()
                )
                self.avg[k] = self.sum[k] / self.count[k]

    def update(self, val, n=1):
        """update"""
        for k, v in val.items():
            if isinstance(k, str) and k[0] == "_":
                # Add to private attributes
                if k not in self._private_attributes:
                    self._private_attributes[k] = [v]
                else:
                    self._private_attributes[k].append(v)
                # Skip updating the meter
                continue

            if k not in self.val:
                self.val[k] = 0
                self.sum[k] = 0
                self.count[k] = 0
                self.max[k] = -np.inf
            self.val[k] = v
            self.sum[k] += v * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]
            self.max[k] = max(v, self.max[k])


class Timer:
    """Timer."""

    def __init__(self):
        self.tot_time = 0
        self.num_calls = 0

    def tic(self):
        self.tic_time = time.time()

    def toc(self):
        diff = time.time() - self.tic_time
        self.tot_time += diff
        self.num_calls += 1
        return diff

    @property
    def average_time(self):
        return self.tot_time / self.num_calls
