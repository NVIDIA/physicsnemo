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

"""Tests for physicsnemo.nn.module.running_norm (RunningNorm)."""

import math

import pytest
import torch

from physicsnemo.nn.module.running_norm import RunningNorm


def test_eval_before_train_is_noop():
    norm = RunningNorm(shape=(1,))
    norm.eval()
    x = torch.randn(16)
    assert torch.equal(norm(x), x)


def test_first_batch_all_zero_input_does_not_poison_state():
    # An all-zero slice carries no magnitude signal. The first batch should
    # leave the running estimate at its default (identity) rather than
    # collapsing to log(0) = -inf.
    norm = RunningNorm(shape=(1,), initialization_behavior="first_batch")
    norm.train()
    out = norm(torch.zeros(16))
    assert torch.isfinite(out).all()
    assert norm.ln_running_mean.item() == 0.0


def test_all_zero_slice_leaves_running_estimate_unchanged():
    norm = RunningNorm(shape=(1,), initialization_behavior="first_batch", momentum=0.9)
    norm.train()
    norm(torch.full((16,), 2.0))  # estimate becomes ln(2)
    estimate_before = norm.ln_running_mean.item()
    assert estimate_before == pytest.approx(math.log(2.0))

    out = norm(torch.zeros(16))
    assert torch.isfinite(out).all()
    assert norm.ln_running_mean.item() == pytest.approx(estimate_before, rel=1e-5)


def test_reduction_ignores_zeros():
    # Zero entries are excluded from mean(abs(x)), so the estimate after
    # [1, 2, 0] is 1.5 rather than 1.0.
    norm = RunningNorm(shape=(1,), initialization_behavior="first_batch")
    norm.train()
    norm(torch.tensor([1.0, 2.0, 0.0]))
    assert norm.ln_running_mean.item() == pytest.approx(math.log(1.5))


def test_output_normalized_by_running_estimate():
    norm = RunningNorm(shape=(1,), initialization_behavior="first_batch")
    norm.train()
    norm(torch.full((32,), 5.0))
    out = norm(torch.full((32,), 5.0))
    assert torch.allclose(out, torch.ones(32), atol=1e-3)
