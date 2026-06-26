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

"""Warp kernels for CSR segmented softmax."""

import warp as wp


@wp.kernel
def segment_softmax_forward_kernel(
    logits: wp.array2d(dtype=wp.float32),
    offsets: wp.array(dtype=wp.int64),
    out: wp.array2d(dtype=wp.float32),
):
    segment, channel = wp.tid()
    start = int(offsets[segment])
    end = int(offsets[segment + 1])
    if end <= start:
        return

    max_value = float(-3.4028234663852886e38)
    i = start
    while i < end:
        value = logits[i, channel]
        if value > max_value:
            max_value = value
        i += 1

    denom = float(0.0)
    i = start
    while i < end:
        denom += wp.exp(logits[i, channel] - max_value)
        i += 1

    i = start
    while i < end:
        out[i, channel] = wp.exp(logits[i, channel] - max_value) / denom
        i += 1


@wp.kernel
def segment_softmax_backward_kernel(
    grad_out: wp.array2d(dtype=wp.float32),
    softmax_out: wp.array2d(dtype=wp.float32),
    offsets: wp.array(dtype=wp.int64),
    grad_logits: wp.array2d(dtype=wp.float32),
):
    segment, channel = wp.tid()
    start = int(offsets[segment])
    end = int(offsets[segment + 1])
    if end <= start:
        return

    dot = float(0.0)
    i = start
    while i < end:
        dot += grad_out[i, channel] * softmax_out[i, channel]
        i += 1

    i = start
    while i < end:
        y = softmax_out[i, channel]
        grad_logits[i, channel] = y * (grad_out[i, channel] - dot)
        i += 1
