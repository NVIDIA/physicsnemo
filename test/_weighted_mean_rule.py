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

"""Independent weighted rule shared by accumulator and distributed tests."""

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor, Partial, Replicate


def _any_invalid(values, predicate):
    """Reduce a scalar validation flag, without making sharded boolean masks.

    Boolean-valued operations do not yet preserve every uneven ShardTensor
    layout. Evaluate the predicate on the local tensor and sum one error flag
    across the mesh. Empty shards contribute zero; every rank gets the same
    decision. This communication validates inputs, not statistical state.
    """
    mesh = getattr(values, "device_mesh", None)
    local = values.to_local() if mesh is not None else values
    flag = predicate(local).any().to(torch.int64)
    if mesh is not None:
        flag = DTensor.from_local(flag, mesh, [Partial()] * mesh.ndim)
        flag = flag.redistribute(placements=[Replicate()] * mesh.ndim).to_local()
    return bool(flag.item())


@dataclass(frozen=True)
class WeightedMeanRule:
    """Test-only custom rule exercising auxiliary inputs and state merging."""

    def summarize(self, values, *, weights, dim, reduce_sum):
        """Validate weights and reduce both sufficient statistics."""
        if _any_invalid(weights, lambda w: ~torch.isfinite(w) | (w < 0)):
            raise ValueError("Weights must be finite and nonnegative")
        return {
            "weighted_sum": reduce_sum(values * weights, dim),
            "weight_sum": reduce_sum(weights, dim),
        }

    def merge(self, left, right):
        """Add sufficient statistics, rather than averaging batch means."""
        return {key: left[key] + right[key] for key in left}

    def result(self, state, *, reduce_sum):
        """Divide after reductions, requiring positive finite total weights."""
        weight_sum = state["weight_sum"]
        invalid_weights = _any_invalid(
            weight_sum, lambda w: (w <= 0) | ~torch.isfinite(w)
        )
        invalid_sum = _any_invalid(state["weighted_sum"], lambda x: ~torch.isfinite(x))
        if invalid_weights or invalid_sum:
            raise ValueError(
                "Each output needs a positive finite total weight and finite weighted sum"
            )
        return state["weighted_sum"] / weight_sum
