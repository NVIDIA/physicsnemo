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

"""Hand-derived updates and interrupted training checks for optimization."""

import copy
from functools import partial

import pytest
import torch
from optimization import Lion, WarmupCosine


def test_lion_hand_computed_updates():
    """Match two hand-calculated Lion parameter and momentum updates."""
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    opt = Lion([p], lr=0.1, weight_decay=0.2)
    p.grad = torch.tensor([2.0, -3.0], dtype=p.dtype)
    opt.step()
    torch.testing.assert_close(p, torch.tensor([0.88, 2.06], dtype=p.dtype))
    p.grad = torch.tensor([-0.01, 0.01], dtype=p.dtype)
    opt.step()
    torch.testing.assert_close(p, torch.tensor([0.7624, 2.1188], dtype=p.dtype))
    torch.testing.assert_close(
        opt.state[p]["momentum"], torch.tensor([0.0197, -0.0296], dtype=p.dtype)
    )


@pytest.mark.parametrize("warmup", [0, 4])
def test_schedule_endpoints_and_resume(warmup):
    """Preserve scheduled learning rates and updates across checkpoint resume."""
    p = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    opt = Lion([p], lr=0.1)
    scheduler = WarmupCosine(opt, total_steps=12, warmup_steps=warmup, min_lr=0.001)
    rates = []
    for step in range(12):
        rates.append(opt.param_groups[0]["lr"])
        p.grad = torch.tensor([(-1.0) ** step, 0.3])
        opt.step()
        scheduler.step()
        if step == 5:
            checkpoint = (
                p.detach().clone(),
                copy.deepcopy(opt.state_dict()),
                copy.deepcopy(scheduler.state_dict()),
            )
    assert rates[0] == pytest.approx(0.1 / warmup if warmup else 0.1)
    assert rates[warmup - 1 if warmup else 0] == pytest.approx(0.1)
    assert rates[-1] == pytest.approx(0.001)
    resumed = torch.nn.Parameter(checkpoint[0])
    opt2 = Lion([resumed], lr=0.1)
    scheduler2 = WarmupCosine(opt2, total_steps=12, warmup_steps=warmup, min_lr=0.001)
    opt2.load_state_dict(checkpoint[1])
    scheduler2.load_state_dict(checkpoint[2])
    for step in range(6, 12):
        assert opt2.param_groups[0]["lr"] == rates[step]
        resumed.grad = torch.tensor([(-1.0) ** step, 0.3])
        opt2.step()
        scheduler2.step()
    torch.testing.assert_close(p, resumed, rtol=0, atol=0)
    torch.testing.assert_close(
        opt.state[p]["momentum"], opt2.state[resumed]["momentum"], rtol=0, atol=0
    )


def test_optimizer_factory_explicit_and_legacy():
    """Select an explicit optimizer while preserving the legacy default."""
    from omegaconf import OmegaConf
    from utils import build_optimizer

    model = torch.nn.Linear(2, 1)
    cfg = OmegaConf.create(
        {"training": {"optimizer": {"_target_": "optimization.Lion", "lr": 5e-5}}}
    )
    assert isinstance(build_optimizer(model, cfg), Lion)
    cfg.training.optimizer = {"lr": 1e-4}
    legacy = build_optimizer(model, cfg)
    assert not isinstance(legacy, Lion)


@pytest.mark.parametrize(
    "target,shape,with_bias,backend",
    [
        ("optimization.Lion", (2,), False, "eager"),
        ("torch.optim.AdamW", (2,), False, "eager"),
        (None, (2,), False, "eager"),
        (None, (2, 2), False, "eager"),
        (None, (2, 2), True, "eager"),
        ("optimization.Lion", (2,), False, "inductor"),
    ],
)
def test_compiled_optimizer_scheduler_updates_match_eager(
    monkeypatch, target, shape, with_bias, backend
):
    """Construct a real scheduler around compiled steps and preserve its updates."""
    from omegaconf import OmegaConf
    from utils import build_optimizer

    # Eager covers scheduler wrapping cheaply; real Inductor catches stale
    # scalar learning rates that Dynamo's eager backend cannot expose.
    monkeypatch.setattr(torch, "compile", partial(torch.compile, backend=backend))
    config = {"lr": 0.1, "weight_decay": 0.01}
    if target is not None:
        config["_target_"] = target
    cfg = OmegaConf.create({"training": {"optimizer": config}})
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(shape)))
    if with_bias:
        model.register_parameter("bias", torch.nn.Parameter(torch.ones(2)))
    reference = copy.deepcopy(model)
    optimizer = build_optimizer(model, cfg, compile_optimizer=True)
    reference_optimizer = build_optimizer(reference, cfg)
    scheduler = WarmupCosine(optimizer, total_steps=10, warmup_steps=2)
    reference_scheduler = WarmupCosine(
        reference_optimizer, total_steps=10, warmup_steps=2
    )
    for _ in range(4):
        for current, opt, schedule in (
            (model, optimizer, scheduler),
            (reference, reference_optimizer, reference_scheduler),
        ):
            opt.zero_grad()
            sum(p.square().sum() for p in current.parameters()).backward()
            opt.step()
            schedule.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)
        assert scheduler.get_last_lr() == reference_scheduler.get_last_lr()
    assert scheduler.last_epoch == 4
    assert not torch.equal(model.weight, torch.ones_like(model.weight))
