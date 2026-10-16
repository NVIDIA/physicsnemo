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

r"""Diffusion updates through the optional general-purpose training loop."""

import subprocess
import sys

import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("nvalchemi")

from nvalchemi.training import TrainingStrategy

from physicsnemo.core import Module
from physicsnemo.diffusion import DiffusionExtraTimes
from physicsnemo.diffusion.distillation.strategy import DistillationStrategy

from .helpers import StructuredDiffusionModel, extract_teacher_features


class NativeLoss:
    def __init__(self, models):
        self.models = models
        self.calls = []

    def __call__(self, x0, *, phase, step, t=None, **kwargs):
        self.calls.append((phase, step))
        role = "student" if phase == "student" else "fake_score"
        t = torch.zeros(x0.shape[0]) if t is None else t
        prediction = self.models[role](x0, t)[0]
        return {"total_loss": prediction.square().mean()}


def make_strategy(*, alternating=False, **kwargs):
    roles = ("student", "fake_score") if alternating else ("student",)
    models = {name: StructuredDiffusionModel() for name in roles}
    loss = NativeLoss(models)
    configs = {
        name: [
            {
                "optimizer_cls": torch.optim.SGD,
                "optimizer_kwargs": {"lr": 0.01},
                "scheduler_cls": torch.optim.lr_scheduler.StepLR,
                "scheduler_kwargs": {"step_size": 1, "gamma": 0.5},
            }
        ]
        for name in roles
    }
    if "num_epochs" not in kwargs and "num_steps" not in kwargs:
        kwargs["num_steps"] = 6
    strategy = DistillationStrategy(
        models=models, loss_fn=loss, optimizer_configs=configs, **kwargs
    )
    return strategy, loss


@pytest.mark.parametrize("structured", (False, True))
def test_native_strategy_one_call_and_direct_update(structured):
    def batch_arguments(models, batch):
        return {
            "x0": batch["x0"],
            "t": (
                batch["t"],
                DiffusionExtraTimes(r=batch["r"], batch_size=batch.batch_size),
            ),
        }

    strategy, loss = make_strategy(
        **({"training_fn": batch_arguments} if structured else {})
    )
    assert isinstance(strategy, TrainingStrategy)
    model = strategy.models["student"]
    direct = StructuredDiffusionModel()
    direct.load_state_dict(model.state_dict())
    optimizer = torch.optim.SGD(direct.parameters(), lr=0.01)
    x = torch.randn(2, 3, 5)
    times = (
        (
            torch.tensor([0.3, 0.7]),
            DiffusionExtraTimes(r=torch.tensor([0.1, 0.2]), batch_size=[2]),
        )
        if structured
        else None
    )
    reference = NativeLoss({"student": direct})(x, t=times, phase="student", step=0)[
        "total_loss"
    ]
    reference.backward()
    optimizer.step()
    batch = TensorDict({"x0": x}, batch_size=[2])
    if times is not None:
        batch["t"], batch["r"] = times[0], times[1].r
    strategy.train_batch(batch)
    assert loss.calls == [("student", 0)]
    assert strategy.step_count == 1
    torch.testing.assert_close(model.state_dict(), direct.state_dict())


def test_strategy_phase_and_inactive_scheduler():
    strategy, loss = make_strategy(alternating=True)
    batch = TensorDict({"x0": torch.randn(2, 3, 5)}, batch_size=[2])
    fake = strategy.models["fake_score"].gain.detach().clone()
    strategy.train_batch(batch)
    torch.testing.assert_close(strategy.models["fake_score"].gain, fake)
    student = strategy.models["student"].gain.detach().clone()
    strategy.train_batch(batch)
    torch.testing.assert_close(strategy.models["student"].gain, student)
    for _ in range(4):
        strategy.train_batch(batch)
    assert loss.calls == [
        ("student" if i % 5 == 0 else "auxiliary", i) for i in range(6)
    ]
    assert [scheduler.last_epoch for scheduler in strategy._lr_schedulers] == [2, 4]


def test_strategy_custom_phase():
    strategy, loss = make_strategy(
        alternating=True,
        phase_fn=lambda step: "auxiliary" if step % 2 == 0 else "student",
    )
    batch = TensorDict({"x0": torch.randn(2, 3, 5)}, batch_size=[2])
    strategy.run([batch, batch])
    assert loss.calls == [
        ("auxiliary" if i % 2 == 0 else "student", i) for i in range(6)
    ]


def test_checkpoint_hook_sees_completed_joint_state(tmp_path):
    from nvalchemi.training._stages import TrainingStage

    class SaveHook:
        stage = TrainingStage.AFTER_BATCH
        frequency = 1

        def __call__(self, ctx, stage):
            if ctx.batch_count == 2:
                ctx.workflow.save_checkpoint(tmp_path, epoch=1)

    strategy, _ = make_strategy(
        alternating=True, accumulation_steps=2, hooks=[SaveHook()]
    )
    batch = TensorDict({"x0": torch.randn(2, 3, 5)}, batch_size=[2])
    strategy.train_batch(batch)
    strategy.train_batch(batch)
    assert (tmp_path / "step-1" / "state").exists()


def test_strategy_resume(tmp_path):
    strategy, _ = make_strategy(alternating=True)
    batch = TensorDict({"x0": torch.randn(2, 3, 5)}, batch_size=[2])
    strategy.train_batch(batch)
    strategy.save_checkpoint(tmp_path, epoch=1, include_rng=True)
    strategy.train_batch(batch)
    fresh, loss = make_strategy(alternating=True)
    fresh.load_checkpoint(tmp_path, epoch=1, restore_rng=True)
    fresh.train_batch(batch)
    assert loss.calls == [("auxiliary", 1)]
    for name in strategy.models:
        torch.testing.assert_close(
            fresh.models[name].state_dict(), strategy.models[name].state_dict()
        )
    for expected, actual in zip(strategy._optimizers, fresh._optimizers):
        torch.testing.assert_close(expected.state_dict(), actual.state_dict())


def test_strategy_accumulation_phase_and_checkpoint(tmp_path):
    strategy, loss = make_strategy(alternating=True, accumulation_steps=2)
    batch = TensorDict({"x0": torch.randn(2, 3, 5)}, batch_size=[2])
    before = strategy.models["student"].gain.detach().clone()
    strategy.train_batch(batch)
    assert strategy.step_count == 0
    torch.testing.assert_close(strategy.models["student"].gain, before)
    with pytest.raises(ValueError, match="accumulation"):
        strategy.save_checkpoint(tmp_path, epoch=0)
    strategy.train_batch(batch)
    strategy.train_batch(batch)
    strategy.train_batch(batch)
    assert strategy.step_count == 2
    assert loss.calls == [
        ("student", 0),
        ("student", 0),
        ("auxiliary", 1),
        ("auxiliary", 1),
    ]


def test_strategy_scaler_skip_keeps_phase():
    strategy, loss = make_strategy(alternating=True, precision=torch.float16)
    before = strategy.models["student"].gain.detach().clone()
    strategy.train_batch(
        TensorDict({"x0": torch.full((2, 3, 5), float("inf"))}, batch_size=[2])
    )
    assert strategy.step_count == 0
    assert strategy.batch_count == 1
    torch.testing.assert_close(strategy.models["student"].gain, before)
    strategy.train_batch(TensorDict({"x0": torch.randn(2, 3, 5)}, batch_size=[2]))
    assert loss.calls == [("student", 0), ("student", 0)]
    assert strategy.step_count == 1


def test_strategy_scaler_resume(tmp_path):
    strategy, _ = make_strategy(precision=torch.float16)
    batch = TensorDict({"x0": torch.randn(2, 3, 5)}, [2])
    strategy.train_batch(batch)
    assert strategy.state_dict()["scalers"]
    strategy.save_checkpoint(tmp_path, epoch=1)
    fresh, _ = make_strategy(precision=torch.float16)
    fresh.load_checkpoint(tmp_path, epoch=1)
    assert fresh.state_dict()["scalers"] == strategy.state_dict()["scalers"]
    fresh.train_batch(batch)
    strategy.train_batch(batch)
    torch.testing.assert_close(
        fresh.models["student"].state_dict(), strategy.models["student"].state_dict()
    )


def test_native_setup_wrapper_binding(tmp_path):
    from nvalchemi.training._stages import TrainingStage

    class WrapHook:
        stage = TrainingStage.SETUP
        frequency = 1

        def __call__(self, ctx, stage):
            if not isinstance(ctx.models["student"], CountingWrapper):
                ctx.models["student"] = CountingWrapper(ctx.models["student"])

    strategy, loss = make_strategy(
        hooks=[WrapHook()],
        bind_models_fn=lambda loss, models: setattr(loss, "models", models),
    )
    strategy.train_batch(TensorDict({"x0": torch.randn(2, 3, 5)}, [2]))
    assert loss.models["student"].calls == 1
    strategy.save_checkpoint(tmp_path, epoch=1)
    fresh, _ = make_strategy(
        hooks=[WrapHook()],
        bind_models_fn=lambda loss, models: setattr(loss, "models", models),
    )
    fresh.load_checkpoint(tmp_path, epoch=1)
    assert isinstance(fresh.models["student"], CountingWrapper)
    torch.testing.assert_close(
        fresh.models["student"].state_dict(), strategy.models["student"].state_dict()
    )


def test_accumulation_epoch_duration():
    strategy, loss = make_strategy(num_epochs=1, accumulation_steps=2)
    batch = TensorDict({"x0": torch.randn(2, 3, 5)}, [2])
    strategy.run([batch] * 4)
    assert len(loss.calls) == 4
    assert strategy.epoch_count == 1
    assert strategy.step_count == 2


class CountingWrapper(Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return self.model(*args, **kwargs)


@pytest.mark.parametrize("method", ("sCM", "MeanFlow", "DMD2", "f-distill"))
def test_strategy_binds_existing_execution_wrapper(method):
    pytest.importorskip("fastgen")
    from physicsnemo.diffusion.distillation import DistillationLoss
    from physicsnemo.diffusion.noise_schedulers import RectifiedFlowNoiseScheduler

    from .helpers import FeatureDiscriminator

    model = StructuredDiffusionModel()
    roles = {"student": model}
    options = {}
    if method in ("DMD2", "f-distill"):
        roles.update(
            teacher=StructuredDiffusionModel(), fake_score=StructuredDiffusionModel()
        )
        auxiliary = {"fake_score": roles["fake_score"]}
        if method == "f-distill":
            roles["discriminator"] = FeatureDiscriminator()
            auxiliary["discriminator"] = roles["discriminator"]
        options = {
            "teacher": roles["teacher"],
            "auxiliary_models": auxiliary,
            "model_options": {
                "teacher": {
                    "feature_names": ("spatial",),
                    "feature_fn": extract_teacher_features,
                }
            },
        }
    loss = DistillationLoss(
        model, RectifiedFlowNoiseScheduler(), method=method, **options
    )
    wrapper = CountingWrapper(model)
    roles["student"] = wrapper
    strategy = DistillationStrategy(
        models=roles,
        loss_fn=loss,
        optimizer_configs={
            "student": [
                {"optimizer_cls": torch.optim.SGD, "optimizer_kwargs": {"lr": 0.01}}
            ]
        },
        num_steps=1,
    )
    strategy.train_batch(TensorDict({"x0": torch.randn(2, 3, 5)}, batch_size=[2]))
    assert wrapper.calls > 0
    assert strategy._optimizers[0].param_groups[0]["params"][0] is model.gain


def test_strategy_compiled_and_mixed_modes():
    strategy, loss = make_strategy()
    model = strategy.models["student"]
    model.eval()
    wrapper = torch.compile(model, backend="eager", fullgraph=True)
    strategy.models["student"] = wrapper
    strategy.bind_models_fn = lambda loss, models: setattr(loss, "models", models)
    strategy.train_batch(TensorDict({"x0": torch.randn(2, 3, 5)}, batch_size=[2]))
    assert loss.models["student"] is wrapper
    assert not model.training


def test_strategy_ema_after_successful_updates():
    strategy, _ = make_strategy()
    target = StructuredDiffusionModel()
    before = target.gain.detach().clone()
    strategy.models["ema"] = target
    strategy.ema_models = {"student": "ema"}
    strategy.ema_decay = 0.5
    strategy.bind_models_fn = lambda loss, models: setattr(loss, "models", models)
    strategy.train_batch(TensorDict({"x0": torch.randn(2, 3, 5)}, batch_size=[2]))
    torch.testing.assert_close(
        target.gain, (before + strategy.models["student"].gain) / 2
    )


def test_joint_phase_scaler_skips_all_roles():
    strategy, _ = make_strategy(
        alternating=True,
        precision=torch.float16,
        phase_fn=lambda step: "joint",
        phase_roles={"joint": ("student", "fake_score")},
    )
    student, fake = (strategy.models[name] for name in ("student", "fake_score"))
    strategy.loss_fn = lambda **kwargs: {
        "total_loss": student.gain.square() + fake.gain * float("inf")
    }
    before = student.gain.detach().clone()
    strategy.train_batch(TensorDict({"x0": torch.randn(2, 3, 5)}, batch_size=[2]))
    torch.testing.assert_close(student.gain, before)
    assert strategy.step_count == 0


def test_optional_strategy_import():
    program = """
import importlib.metadata
import sys
original = importlib.metadata.version
def version(name):
    if name == 'nvalchemi-toolkit':
        raise importlib.metadata.PackageNotFoundError(name)
    return original(name)
importlib.metadata.version = version
from physicsnemo.diffusion.distillation import BaseDistillationLoss, DistillationStrategy
assert not any(name.startswith('nvalchemi') for name in sys.modules)
try:
    DistillationStrategy()
except ImportError as exc:
    assert 'diffusion-strategy' in str(exc)
else:
    raise AssertionError('Expected actionable delayed error')
"""
    subprocess.run(  # noqa: S603
        [sys.executable, "-c", program], check=True, capture_output=True, text=True
    )


@pytest.mark.parametrize("steps", (1, 2))
def test_strategy_f_distill_resume(tmp_path, steps):
    pytest.importorskip("fastgen")
    from physicsnemo.diffusion.distillation import DistillationLoss
    from physicsnemo.diffusion.noise_schedulers import RectifiedFlowNoiseScheduler

    from .helpers import FeatureDiscriminator

    def build():
        models = {
            role: StructuredDiffusionModel()
            for role in ("student", "teacher", "fake_score", "ema")
        }
        models["discriminator"] = FeatureDiscriminator()
        loss = DistillationLoss(
            models["student"],
            RectifiedFlowNoiseScheduler(),
            method="f-distill",
            teacher=models["teacher"],
            auxiliary_models={
                name: models[name] for name in ("fake_score", "discriminator")
            },
            model_options={
                "teacher": {
                    "feature_names": ("spatial",),
                    "feature_fn": extract_teacher_features,
                }
            },
        )
        return DistillationStrategy(
            models=models,
            loss_fn=loss,
            num_steps=6,
            phase_fn=lambda step: "student" if step % 2 else "auxiliary",
            ema_models={"student": "ema"},
            optimizer_configs={
                role: [
                    {
                        "optimizer_cls": torch.optim.Adam,
                        "optimizer_kwargs": {"lr": 0.001},
                    }
                ]
                for role in ("student", "fake_score", "discriminator")
            },
        )

    strategy = build()
    batch = TensorDict({"x0": torch.randn(2, 3, 5)}, batch_size=[2])
    for _ in range(steps):
        strategy.train_batch(batch)
    strategy.save_checkpoint(tmp_path, epoch=steps, include_rng=True)
    strategy.train_batch(batch)
    fresh = build()
    fresh.load_checkpoint(tmp_path, epoch=steps, restore_rng=True)
    fresh.train_batch(batch)
    for name in strategy.models:
        torch.testing.assert_close(
            fresh.models[name].state_dict(), strategy.models[name].state_dict()
        )
    for left, right in zip(strategy._optimizers, fresh._optimizers):
        torch.testing.assert_close(left.state_dict(), right.state_dict())
    torch.testing.assert_close(
        strategy.loss_fn.state_dict(), fresh.loss_fn.state_dict()
    )
    torch.testing.assert_close(strategy._last_loss, fresh._last_loss)
