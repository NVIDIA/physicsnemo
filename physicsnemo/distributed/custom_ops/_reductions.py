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

from typing import Iterable

import torch

from physicsnemo.utils.version_check import check_module_requirements

check_module_requirements("physicsnemo.distributed.shard_tensor")

from torch.distributed.tensor.placement_types import (  # noqa: E402
    Partial,
    Shard,
)

from physicsnemo.distributed.shard_tensor import ShardTensor  # noqa: E402

aten = torch.ops.aten


class ShardedMean(torch.autograd.Function):
    """
    This is a custom mean operation that takes into account the fact that the
    sharded tensor may be unevenly sharded.

    The strategy is to do a weighted mean, where the weight is the size of the
    shard in the dimension being reduced.

    """

    @staticmethod
    def forward(ctx, *args, **kwargs):
        # The difference in the sharded mean is that we need to keep track
        # of the portion of the global tensor held by the local shard.

        def mean_args(
            input, dim=None, keepdim=False, dtype=None, out=None, *args, **kwargs
        ):
            # Sanitize the arguments:

            return input, dim, keepdim, dtype, out

        input, dim, keepdim, dtype, out = mean_args(*args, **kwargs)

        weight = 1.0

        if dim is None:

            dim = range(len(input.shape))

        # Convert dim to a tuple if it's not already iterable
        if not isinstance(dim, Iterable):
            dim = (dim,)

        denom = 1.0
        local_shape = input._local_tensor.shape
        # For each dimension being reduced, multiply weight by local size
        # and track global size for denominator
        for d in dim:
            if d < 0:
                d += input.ndim
            weight *= local_shape[d]
            denom *= input.shape[d]

        # Compute the weight:
        weight = weight / denom

        local_input = input._local_tensor

        # Now we can do the mean:
        local_mean = aten.mean(local_input, dim=dim, keepdim=keepdim, dtype=dtype)

        # If dim is None, placements will be partial across all mesh dims.
        if dim is None:
            placements = [Partial("sum") for _ in range(input.ndim)]
        else:
            # dim is not none, but make sure we only put partial on the dims we're reducing on.
            placements = []
            for i_p, p in enumerate(input._spec.placements):
                if isinstance(p, Shard) and p.dim in dim:
                    placements.append(Partial("sum"))
                else:
                    placements.append(p)

        # Create a new ShardTensor with the same mesh and right placements
        local_mean = ShardTensor.from_local(
            weight * local_mean,  # Scale by weight to account for local size
            input.device_mesh,
            placements,
        )

        # print(f"Local mean: {local_mean}")
        return local_mean

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None, None


def sharded_mean_wrapper(*args, **kwargs):
    return ShardedMean.apply(*args, **kwargs)


mean_ops = [
    aten.mean.default,
    aten.mean.dim,
    aten.mean.dtype_out,
    aten.mean.names_dim,
    aten.mean.names_out,
    aten.mean.op,
    aten.mean.out,
]
for op in mean_ops:
    ShardTensor.register_function_handler(op, sharded_mean_wrapper)
