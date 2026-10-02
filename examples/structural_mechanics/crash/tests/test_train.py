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

import os
import sys

import pytest
import torch

THIS_DIR = os.path.dirname(__file__)
CRASH_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if CRASH_DIR not in sys.path:
    sys.path.insert(0, CRASH_DIR)

from train import (  # noqa: E402
    LinearTeacherForcingSchedule,
    ValidationEarlyStopping,
    average_model_gradients,
    execution_end_epoch,
    masked_acceleration_mse,
)


def test_execution_limit_does_not_change_training_target():
    """Verify execution limit does not change training target."""
    assert execution_end_epoch(0, 500, None) == 500
    assert execution_end_epoch(0, 500, 10) == 10
    assert execution_end_epoch(10, 500, 1) == 11
    assert execution_end_epoch(495, 500, 10) == 500
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="max_epochs_this_run"):
            execution_end_epoch(0, 500, invalid)


def test_validation_early_stopping_resets_on_improvement():
    """Verify validation early stopping resets on improvement."""
    controller = ValidationEarlyStopping(patience=2, min_delta=0.01)

    assert controller.update(1.0) == (False, True)
    assert controller.update(0.995) == (False, False)
    assert controller.update(0.8) == (False, True)
    assert controller.stale_evaluations == 0
    assert controller.update(0.795) == (False, False)
    assert controller.update(0.79) == (True, False)


def test_validation_early_stopping_disabled_and_validates_settings():
    """Verify validation early stopping disabled and validates settings."""
    controller = ValidationEarlyStopping(patience=0)

    assert controller.update(float("nan")) == (False, False)
    assert controller.update(float("inf")) == (False, False)
    assert controller.stale_evaluations == 2

    with pytest.raises(ValueError, match="patience"):
        ValidationEarlyStopping(patience=-1)
    with pytest.raises(ValueError, match="min_delta"):
        ValidationEarlyStopping(min_delta=-1.0)


def test_linear_teacher_forcing_schedule():
    """Verify linear teacher forcing schedule."""
    schedule = LinearTeacherForcingSchedule(
        warmup_epochs=2,
        decay_epochs=4,
        start_probability=1.0,
        end_probability=0.0,
    )

    assert schedule.probability(0) == 1.0
    assert schedule.probability(2) == 1.0
    assert schedule.probability(4) == 0.5
    assert schedule.probability(6) == 0.0
    assert schedule.probability(20) == 0.0

    with pytest.raises(ValueError, match="warmup_epochs"):
        LinearTeacherForcingSchedule(warmup_epochs=-1)
    with pytest.raises(ValueError, match="start_probability"):
        LinearTeacherForcingSchedule(start_probability=1.1)


def test_masked_acceleration_mse_uses_only_reference_state_transitions():
    """Verify masked acceleration MSE uses only reference state transitions."""
    prediction = torch.tensor([1.0, 10.0, 3.0]).view(1, 3, 1).expand(-1, -1, 3)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([True, False, True])

    loss = masked_acceleration_mse(prediction, target, mask)

    torch.testing.assert_close(loss, torch.tensor(5.0))

    large_fp16_prediction = torch.ones(30_000, 1, 3, dtype=torch.float16)
    large_fp16_loss = masked_acceleration_mse(
        large_fp16_prediction,
        torch.zeros_like(large_fp16_prediction),
        torch.tensor([True]),
    )
    assert torch.isfinite(large_fp16_loss)
    torch.testing.assert_close(large_fp16_loss, torch.tensor(1.0))


def test_average_model_gradients_reduces_after_backward(monkeypatch):
    """Verify average model gradients reduces after backward."""
    model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.LayerNorm(1))
    original_gradients = []
    for index, parameter in enumerate(model.parameters(), start=1):
        parameter.grad = torch.full_like(parameter, float(index))
        original_gradients.append(parameter.grad.clone())

    def fake_all_reduce(flat, op):
        assert op == torch.distributed.ReduceOp.SUM
        flat.add_(2.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    average_model_gradients(model, world_size=2)

    for gradient, original in zip(
        (parameter.grad for parameter in model.parameters()), original_gradients
    ):
        torch.testing.assert_close(gradient, (original + 2.0) / 2.0)
