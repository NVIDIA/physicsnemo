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

import torch

from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.distributed.utils import (
    _reduce,
    _split,
    all_gather_v_wrapper,
    compute_split_shapes,
)


class _CopyToParallelRegion(torch.autograd.Function):
    """Pass the input to the parallel region"""

    @staticmethod
    def symbolic(graph, input_, group_):  # pragma: no cover
        return input_

    @staticmethod
    def forward(ctx, input_, group_):  # pragma: no cover
        ctx.group = group_
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return _reduce(grad_output, group=DistributedManager().group(ctx.group)), None


class _ReduceFromParallelRegion(torch.autograd.Function):
    """All-reduce the input from the parallel region"""

    @staticmethod
    def symbolic(graph, input_, group_):  # pragma: no cover
        return _reduce(input_, group=DistributedManager().group(group_))

    @staticmethod
    def forward(ctx, input_, group_):  # pragma: no cover
        return _reduce(input_, group=DistributedManager().group(group_))

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return grad_output, None


class _ScatterToParallelRegion(torch.autograd.Function):
    """Split the input and keep only the chunk corresponding to the rank."""

    @staticmethod
    def symbolic(graph, input_, dim_, group_):  # pragma: no cover
        return _split(input_, dim_, group=DistributedManager().group(group_))

    @staticmethod
    def forward(ctx, input_, dim_, group_):  # pragma: no cover
        ctx.dim = dim_
        ctx.group = group_
        ctx.split_shapes = compute_split_shapes(
            input_.shape[dim_], DistributedManager().group_size(group_)
        )
        return _split(input_, dim_, group=DistributedManager().group(group_))

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return (
            all_gather_v_wrapper(
                grad_output,
                ctx.split_shapes,
                ctx.dim,
                group=DistributedManager().group(ctx.group),
            ),
            None,
            None,
        )


class _GatherFromParallelRegion(torch.autograd.Function):
    """Gather the input from parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_, dim_, group_, shapes_):  # pragma: no cover
        return all_gather_v_wrapper(
            input_, shapes_, dim_, group=DistributedManager().group(group_)
        )

    @staticmethod
    def forward(ctx, input_, dim_, shapes_, group_):  # pragma: no cover
        ctx.dim = dim_
        ctx.group = group_
        return all_gather_v_wrapper(
            input_, shapes_, dim_, group=DistributedManager().group(group_)
        )

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return (
            _split(grad_output, ctx.dim, group=DistributedManager().group(ctx.group)),
            None,
            None,
            None,
        )


# -----------------
# Helper functions.
# -----------------
def copy_to_parallel_region(input, group):  # pragma: no cover
    """Copy input"""
    return _CopyToParallelRegion.apply(input, group)


def reduce_from_parallel_region(input, group):  # pragma: no cover
    """All-reduce the input from the matmul parallel region."""
    return _ReduceFromParallelRegion.apply(input, group)


def scatter_to_parallel_region(input, dim, group):  # pragma: no cover
    """Split the input and keep only the corresponding chuck to the rank."""
    return _ScatterToParallelRegion.apply(input, dim, group)


def gather_from_parallel_region(input, dim, shapes, group):  # pragma: no cover
    """Gather the input from matmul parallel region and concatenate."""
    return _GatherFromParallelRegion.apply(input, dim, shapes, group)
