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

r"""Roundtrip named training roles through ordinary PhysicsNeMo checkpoints."""

import random

import numpy as np
import pytest
import torch

from physicsnemo.diffusion.distillation.checkpoint import CheckpointStateBundle

from .helpers import StructuredDiffusionModel


def test_bundle_roundtrip(tmp_path):
    models = {
        name: StructuredDiffusionModel() for name in ("student", "fake_score", "ema")
    }
    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=0.01)
        for name, model in models.items()
        if name != "ema"
    }
    schedulers = {
        name: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        for name, optimizer in optimizers.items()
    }
    for name, optimizer in optimizers.items():
        models[name].gain.square().backward()
        optimizer.step()
        schedulers[name].step()
    bundle = CheckpointStateBundle(models, optimizers=optimizers, schedulers=schedulers)
    bundle.save_checkpoint(tmp_path, epoch=4, progress={"step": 4}, include_rng=True)
    expected_rng = (torch.rand(3), np.random.rand(3), random.random())
    restored = {name: StructuredDiffusionModel() for name in models}
    restored_opts = {
        name: torch.optim.Adam(restored[name].parameters(), lr=0.1)
        for name in optimizers
    }
    restored_sched = {
        name: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        for name, optimizer in restored_opts.items()
    }
    fresh = CheckpointStateBundle(
        restored, optimizers=restored_opts, schedulers=restored_sched
    )
    progress = fresh.load_checkpoint(tmp_path, epoch=4, restore_rng=True)
    assert progress == {"step": 4}
    torch.testing.assert_close(torch.rand(3), expected_rng[0])
    np.testing.assert_array_equal(np.random.rand(3), expected_rng[1])
    assert random.random() == expected_rng[2]
    for name in models:
        torch.testing.assert_close(restored[name].gain, models[name].gain)
    for name in optimizers:
        torch.testing.assert_close(
            restored_opts[name].state_dict(), optimizers[name].state_dict()
        )
        assert restored_sched[name].state_dict() == schedulers[name].state_dict()


def test_bundle_role_mismatch_before_loading(tmp_path):
    original = StructuredDiffusionModel()
    CheckpointStateBundle({"student": original}).save_checkpoint(tmp_path, epoch=1)
    other = StructuredDiffusionModel()
    with torch.no_grad():
        other.gain.fill_(1.5)
    with pytest.raises(ValueError, match="roles"):
        CheckpointStateBundle({"fake_score": other}).load_checkpoint(tmp_path, epoch=1)
    assert other.gain == 1.5


def test_bundle_missing_checkpoint(tmp_path):
    bundle = CheckpointStateBundle({"student": StructuredDiffusionModel()})
    with pytest.raises(FileNotFoundError):
        bundle.load_checkpoint(tmp_path, epoch=3)


def test_bundle_validates_optimizer_association():
    student, other = StructuredDiffusionModel(), StructuredDiffusionModel()
    with pytest.raises(ValueError, match="parameters"):
        CheckpointStateBundle(
            {"student": student},
            optimizers={"student": torch.optim.Adam(other.parameters())},
        )


def test_bundle_rejects_overwrite(tmp_path):
    bundle = CheckpointStateBundle({"student": StructuredDiffusionModel()})
    bundle.save_checkpoint(tmp_path, epoch=0)
    with pytest.raises(FileExistsError):
        bundle.save_checkpoint(tmp_path, epoch=0)


def test_bundle_partial_snapshot_does_not_load_models(tmp_path):
    model = StructuredDiffusionModel()
    bundle = CheckpointStateBundle({"student": model})
    bundle.save_checkpoint(tmp_path, epoch=0)
    checkpoint = next((tmp_path / "step-0" / "models").rglob("*.pt"))
    checkpoint.unlink()
    with torch.no_grad():
        model.gain.fill_(3.0)
    with pytest.raises(FileNotFoundError):
        bundle.load_checkpoint(tmp_path, epoch=0)
    assert model.gain == 3.0


def test_bundle_requires_captured_rng(tmp_path):
    bundle = CheckpointStateBundle({"student": StructuredDiffusionModel()})
    bundle.save_checkpoint(tmp_path, epoch=0)
    with pytest.raises(ValueError, match="random states"):
        bundle.load_checkpoint(tmp_path, epoch=0, restore_rng=True)


def test_bundle_corrupt_metadata_reports_validation_error(tmp_path):
    bundle = CheckpointStateBundle({"student": StructuredDiffusionModel()})
    bundle.save_checkpoint(tmp_path, epoch=0)
    checkpoint = next((tmp_path / "step-0" / "state").rglob("*.pt"))
    checkpoint.write_bytes(checkpoint.read_bytes()[:32])
    with pytest.raises(ValueError, match="Invalid checkpoint bundle"):
        bundle.load_checkpoint(tmp_path, epoch=0)


@pytest.mark.parametrize("phase", ("student", "auxiliary"))
def test_alternating_resume(tmp_path, phase):
    pytest.importorskip("fastgen")
    from physicsnemo.diffusion.distillation import DistillationLoss
    from physicsnemo.diffusion.noise_schedulers import RectifiedFlowNoiseScheduler

    from .helpers import FeatureDiscriminator, extract_teacher_features

    def setup():
        roles = {
            name: StructuredDiffusionModel()
            for name in ("student", "teacher", "fake_score")
        }
        roles["discriminator"] = FeatureDiscriminator()
        loss_fn = DistillationLoss(
            roles["student"],
            RectifiedFlowNoiseScheduler(),
            method="f-distill",
            teacher=roles["teacher"],
            auxiliary_models={
                name: roles[name] for name in ("fake_score", "discriminator")
            },
            model_options={
                "teacher": {
                    "feature_names": ("spatial",),
                    "feature_fn": extract_teacher_features,
                }
            },
        )
        optimizers = {
            name: torch.optim.Adam(model.parameters(), lr=0.01)
            for name, model in roles.items()
            if name != "teacher"
        }
        bundle = CheckpointStateBundle(
            roles, optimizers=optimizers, stateful={"loss": loss_fn}
        )
        return roles, optimizers, loss_fn, bundle

    def update(roles, optimizers, loss_fn, phase, x):
        active = ("student",) if phase == "student" else ("fake_score", "discriminator")
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(x, phase=phase)["total_loss"]
        loss.backward()
        for name in active:
            optimizers[name].step()
        return loss.detach()

    roles, optimizers, loss_fn, bundle = setup()
    x = torch.randn(2, 3, 5)
    for initial in ("student", "auxiliary"):
        update(roles, optimizers, loss_fn, initial, x)
    bundle.save_checkpoint(tmp_path, epoch=2, progress={"step": 2}, include_rng=True)
    expected = update(roles, optimizers, loss_fn, phase, x)
    fresh, fresh_opts, fresh_loss, fresh_bundle = setup()
    assert fresh_bundle.load_checkpoint(tmp_path, epoch=2, restore_rng=True) == {
        "step": 2
    }
    actual = update(fresh, fresh_opts, fresh_loss, phase, x)
    torch.testing.assert_close(actual, expected)
    for name, model in roles.items():
        torch.testing.assert_close(fresh[name].state_dict(), model.state_dict())
    for name, optimizer in optimizers.items():
        torch.testing.assert_close(
            fresh_opts[name].state_dict(), optimizer.state_dict()
        )
    torch.testing.assert_close(fresh_loss.state_dict(), loss_fn.state_dict())
