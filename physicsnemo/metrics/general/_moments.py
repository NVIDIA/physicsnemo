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

"""Detached streaming moments implemented with tensor operations.

Distributed layouts, rather than a metric-owned process group, determine which
axes communicate. This private implementation backs the opt-in tensor mode of
Mean and Variance.
"""

import torch
from torch.distributed.tensor import Replicate


def _complete(value):
    """Finish pending reductions without gathering unreduced spatial shards."""
    placements = getattr(value, "placements", ())
    if any(p.is_partial() for p in placements):
        return value.redistribute(
            placements=[Replicate() if p.is_partial() else p for p in placements]
        )
    return value


def _check_compatible(left, right):
    if (
        type(left) is not type(right)
        or left.shape != right.shape
        or left.device != right.device
        or left.dtype != right.dtype
    ):
        raise ValueError(
            "Statistics require matching tensor type, output shape, device, and dtype"
        )
    if getattr(left, "placements", ()) != getattr(right, "placements", ()):
        raise ValueError("Statistics require aligned distributed placements")
    if getattr(left, "device_mesh", None) != getattr(right, "device_mesh", None):
        raise ValueError("Statistics require the same device mesh")
    # ShardTensor supports non-chunked partitions. Equal global shapes and
    # placements alone do not establish correspondence of output elements.
    left_shapes = getattr(getattr(left, "_spec", None), "sharding_shapes", None)
    right_shapes = getattr(getattr(right, "_spec", None), "sharding_shapes", None)
    if left_shapes is not None and left_shapes() != right_shapes():
        raise ValueError("Statistics require aligned shard boundaries")


class _StreamingMoments:
    def __init__(self, *, variance, like=None, dtype=None):
        if dtype not in (None, torch.float32, torch.float64):
            raise ValueError("Accumulation dtype must be float32 or float64")
        self.variance = variance
        self.dtype = dtype
        self._explicit_dtype = dtype is not None
        self.count = 0
        self.mean = None
        self.m2 = None
        self.template = None
        if like is not None:
            self.template = torch.zeros_like(self._prepare(like))
            self.dtype = self.template.dtype

    def _prepare(self, value):
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError("Expected a real floating-point tensor")
        dtype = self.dtype or (
            torch.float64 if value.dtype == torch.float64 else torch.float32
        )
        if (
            not self._explicit_dtype
            and self.dtype == torch.float32
            and value.dtype == torch.float64
        ):
            raise ValueError(
                "Accumulator dtype is torch.float32; use dtype=torch.float64 to retain float64 inputs"
            )
        return _complete(value.detach().to(dtype=dtype))

    def update(self, value, dim):
        value = self._prepare(value)
        if dim is None:
            n, mean = 1, value.clone()
            m2 = torch.zeros_like(mean) if self.variance else None
        else:
            if isinstance(dim, bool) or not isinstance(dim, int):
                raise TypeError("dim must be an integer observation axis or None")
            if not -value.ndim <= dim < value.ndim:
                raise IndexError("Observation axis is out of range")
            dim %= value.ndim
            n = value.shape[dim]
            mean_keepdim = _complete(value.sum(dim=dim, keepdim=True))
            if n:
                mean_keepdim = mean_keepdim / n
            # A singleton sum preserves arbitrary ShardTensor partition metadata;
            # generic squeeze currently falls back to chunk-based DTensor metadata.
            mean = mean_keepdim.sum(dim=dim)
            m2 = (
                _complete((value - mean_keepdim).square().sum(dim=dim))
                if self.variance
                else None
            )
        self._append(n, mean, m2)
        # Commit inferred precision only after validation and accumulation succeed.
        self.dtype = value.dtype

    def _append(self, n, mean, m2):
        template = self.mean if self.mean is not None else self.template
        if template is not None:
            _check_compatible(template, mean)
        if n == 0:
            if self.mean is None and self.template is None:
                self.template = torch.zeros_like(mean)
            return
        if self.count == 0:
            self.mean = mean.clone()
            self.m2 = m2.clone() if self.variance else None
        else:
            delta = mean - self.mean
            total = self.count + n
            if self.variance:
                self.m2 = self.m2 + m2 + delta.square() * (self.count * n / total)
            self.mean = self.mean + delta * (n / total)
        self.count += n
        self.template = None

    def combine(self, other):
        if self.variance != other.variance or (
            self.dtype is not None
            and other.dtype is not None
            and self.dtype != other.dtype
        ):
            raise ValueError(
                "Can only combine the same statistic and settled accumulation dtype"
            )
        if self.dtype is None:
            self.dtype = other.dtype
        if other.count:
            self._append(other.count, other.mean, other.m2)
        elif other.template is not None:
            self._append(0, other.template, None)

    def reset(self):
        """Retain dtype policy and layout while discarding observations."""
        if self.mean is not None:
            self.template = torch.zeros_like(self.mean)
        self.mean = self.m2 = None
        self.count = 0

    def read_mean(self):
        if not self.count:
            raise ValueError("Cannot read an empty accumulator")
        return self.mean.clone()

    def read_variance(self, correction):
        _check_correction(correction)
        if self.count <= correction:
            raise ValueError("Insufficient observations for the requested correction")
        return self.m2 / (self.count - correction)


def _check_correction(correction):
    if (
        isinstance(correction, bool)
        or not isinstance(correction, int)
        or correction < 0
    ):
        raise ValueError("correction must be a nonnegative integer")
