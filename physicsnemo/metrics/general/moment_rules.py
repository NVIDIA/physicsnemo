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

"""Mergeable rules used by tensor-mode mean and variance."""

from dataclasses import dataclass

import torch


def _check_correction(correction):
    if (
        isinstance(correction, bool)
        or not isinstance(correction, int)
        or correction < 0
    ):
        raise ValueError("correction must be a nonnegative integer")


@dataclass(frozen=True)
class MeanRule:
    """Mean represented by a count and mean, with stable pairwise merging."""

    def summarize(self, values, *, dim, reduce_sum):
        """Produce a batch's count and mean; an empty batch has a zero mean."""
        n = 1 if dim is None else values.shape[dim]
        total = reduce_sum(values, dim)
        return {"n": n, "mean": total / n if n else total}

    def merge(self, left, right):
        """Combine two nonempty states without modifying either input."""
        n = left["n"] + right["n"]
        mean = left["mean"] + (right["mean"] - left["mean"]) * (right["n"] / n)
        return {"n": n, "mean": mean}

    def result(self, state, *, reduce_sum):
        """Return the stored mean; the accumulator gives callers an owned copy."""
        return state["mean"]


@dataclass(frozen=True)
class VarianceRule:
    """Count, mean and centered squared deviations using Chan's merge formula."""

    def summarize(self, values, *, dim, reduce_sum):
        """Compute centered deviations only after completing the batch mean."""
        n = 1 if dim is None else values.shape[dim]
        if dim is None:
            return {"n": n, "mean": values.clone(), "m2": torch.zeros_like(values)}
        total = reduce_sum(values, dim, keepdim=True)
        mean = total / n if n else total
        return {
            "n": n,
            "mean": mean.sum(dim=dim),
            "m2": reduce_sum((values - mean).square(), dim),
        }

    def merge(self, left, right):
        """Merge nonempty moment states with a stable between-batch correction."""
        n = left["n"] + right["n"]
        delta = right["mean"] - left["mean"]
        return {
            "n": n,
            "mean": left["mean"] + delta * (right["n"] / n),
            "m2": left["m2"]
            + right["m2"]
            + delta.square() * (left["n"] * right["n"] / n),
        }

    def result(self, state, *, reduce_sum, correction=1, statistic="variance"):
        """Read variance with a correction, or the mean of the same observations."""
        if statistic == "mean":
            return state["mean"]
        if statistic != "variance":
            raise ValueError("statistic must be 'mean' or 'variance'")
        _check_correction(correction)
        if state["n"] <= correction:
            raise ValueError("Insufficient observations for the requested correction")
        return state["m2"] / (state["n"] - correction)
