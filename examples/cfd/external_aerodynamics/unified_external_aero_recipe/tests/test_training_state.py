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

"""Independent tests for EMA restart, skipped AMP updates, and clipping."""

import pytest
import torch
from training_state import ModelEMA, optimizer_update

from physicsnemo.utils import load_checkpoint, save_checkpoint


def test_ema_values_buffers_and_exception_restore():
    """Average weights, copy buffers, and restore raw storage after errors."""
    model = torch.nn.Linear(1, 1, bias=False)
    model.register_buffer("counter", torch.tensor(1))
    with torch.no_grad():
        model.weight.fill_(2)
    ema = ModelEMA(model, 0.75)
    with torch.no_grad():
        model.weight.fill_(10)
        model.counter.fill_(7)
    ema.update()
    assert ema.updates == 1
    torch.testing.assert_close(ema.parameters["weight"], torch.tensor([[4.0]]))
    original_pointer = model.weight.data_ptr()
    with pytest.raises(RuntimeError, match="validation failed"):
        with ema.average_parameters():
            torch.testing.assert_close(model.weight, torch.tensor([[4.0]]))
            assert model.counter.item() == 7
            assert model.weight.data_ptr() == original_pointer
            raise RuntimeError("validation failed")
    torch.testing.assert_close(model.weight, torch.tensor([[10.0]]))
    assert model.weight.data_ptr() == original_pointer


def test_ema_accumulates_reduced_precision_parameters_in_float32():
    """Retain small EMA increments when model weights use reduced precision."""
    model = torch.nn.Linear(1, 1, bias=False).bfloat16()
    with torch.no_grad():
        model.weight.fill_(1)
    ema = ModelEMA(model, 0.9999)
    with torch.no_grad():
        model.weight.fill_(2)
    ema.update()
    assert ema.parameters["weight"].dtype == torch.float32
    assert 1.0 < ema.parameters["weight"].item() < 1.001


def test_amp_clips_unscaled_gradient_and_skips_ema_on_overflow():
    """Clip unscaled gradients and leave weights and EMA unchanged on overflow."""
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    scaler = torch.amp.GradScaler("cpu", init_scale=8, growth_interval=10)
    ema = ModelEMA(model, 0.5)
    updated = optimizer_update(
        model.weight.sum() * 4,
        optimizer,
        model.parameters(),
        scaler=scaler,
        grad_clip_norm=1,
        ema=ema,
    )
    assert updated
    torch.testing.assert_close(model.weight, torch.tensor([[1.5]]))
    torch.testing.assert_close(ema.parameters["weight"], torch.tensor([[1.75]]))
    previous = model.weight.detach().clone()
    scale = scaler.get_scale()
    updated = optimizer_update(
        model.weight.sum() * float("inf"),
        optimizer,
        model.parameters(),
        scaler=scaler,
        grad_clip_norm=1,
        ema=ema,
    )
    assert not updated
    assert scaler.get_scale() < scale
    assert ema.updates == 1
    torch.testing.assert_close(model.weight, previous)
    torch.testing.assert_close(ema.parameters["weight"], torch.tensor([[1.75]]))


def test_ema_checkpoint_continuation_matches_uninterrupted(tmp_path):
    """Resume raw parameters, optimizer momentum, and EMA exactly."""
    original = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(original.parameters(), lr=0.1, momentum=0.9)
    ema = ModelEMA(original, 0.9)
    x = torch.tensor([[1.0, -2.0]])
    for _ in range(3):
        optimizer_update(
            original(x).square().sum(), optimizer, original.parameters(), ema=ema
        )
    save_checkpoint(
        tmp_path,
        models=original,
        optimizer=optimizer,
        epoch=3,
        metadata={"training_ema": ema.state_dict()},
    )
    resumed = torch.nn.Linear(2, 1)
    resumed_optimizer = torch.optim.SGD(resumed.parameters(), lr=0.1, momentum=0.9)
    metadata = {}
    epoch = load_checkpoint(
        tmp_path,
        models=resumed,
        optimizer=resumed_optimizer,
        device="cpu",
        metadata_dict=metadata,
    )
    assert epoch == 3
    resumed_ema = ModelEMA(resumed, 0.9)
    resumed_ema.load_state_dict(metadata["training_ema"])
    for model, opt, average in (
        (original, optimizer, ema),
        (resumed, resumed_optimizer, resumed_ema),
    ):
        optimizer_update(model(x).square().sum(), opt, model.parameters(), ema=average)
    assert ema.updates == resumed_ema.updates == 4
    for name, value in ema.parameters.items():
        torch.testing.assert_close(value, resumed_ema.parameters[name], rtol=0, atol=0)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("decay", [-1, 1, float("nan"), True])
def test_invalid_ema_decay(decay):
    """Reject invalid EMA decay values before creating accumulators."""
    with pytest.raises(ValueError, match="decay"):
        ModelEMA(torch.nn.Linear(1, 1), decay)


def test_ema_rejects_changed_decay_and_shape():
    """Reject incompatible EMA checkpoint decay or parameter dimensions."""
    ema = ModelEMA(torch.nn.Linear(2, 1), 0.9)
    with pytest.raises(ValueError, match="decay"):
        ModelEMA(torch.nn.Linear(2, 1), 0.99).load_state_dict(ema.state_dict())
    with pytest.raises(ValueError, match="shape"):
        ModelEMA(torch.nn.Linear(3, 1), 0.9).load_state_dict(ema.state_dict())


@pytest.mark.parametrize("clip", [0, -1, float("inf"), float("nan"), True])
def test_invalid_gradient_clip(clip):
    """Reject invalid clipping thresholds before updating parameters."""
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="grad_clip_norm"):
        optimizer_update(
            model.weight.sum(), optimizer, model.parameters(), grad_clip_norm=clip
        )


def test_ema_export_loads_as_inference_checkpoint_without_mutating_raw(tmp_path):
    """Export averaged inference weights while preserving raw training weights."""
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2)
    ema = ModelEMA(model, 0.5)
    with torch.no_grad():
        model.weight.fill_(4)
    ema.update()
    with ema.average_parameters():
        save_checkpoint(tmp_path, models=model, epoch=5)
    restored = torch.nn.Linear(1, 1, bias=False)
    assert load_checkpoint(tmp_path, models=restored, device="cpu") == 5
    torch.testing.assert_close(restored.weight, torch.tensor([[3.0]]))
    torch.testing.assert_close(model.weight, torch.tensor([[4.0]]))
