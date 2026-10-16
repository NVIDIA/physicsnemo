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

r"""Check explicitly selected distribution-matching objectives and role ownership."""

import pytest
import torch

pytest.importorskip("fastgen")

from physicsnemo.diffusion.distillation import (
    DistillationLoss,
    WeightedDistillationLoss,
)
from physicsnemo.diffusion.noise_schedulers import (
    EDMNoiseScheduler,
    RectifiedFlowNoiseScheduler,
)

from .helpers import (
    FeatureDiscriminator,
    GaussianDenoiser,
    StochasticDiffusionModel,
    StructuredDiffusionModel,
    extract_teacher_features,
)


class TestDMD2:
    @pytest.mark.parametrize("empty", (False, True))
    def test_masked_nan_prediction(self, empty):
        student, teacher, fake = (StructuredDiffusionModel() for _ in range(3))
        with torch.no_grad():
            fake.gain.fill_(0.3)
        loss_fn = WeightedDistillationLoss(
            student,
            RectifiedFlowNoiseScheduler(),
            method="DMD2",
            teacher=teacher,
            auxiliary_models={"fake_score": fake},
        )
        x = torch.randn(2, 3, 5)
        weight = torch.zeros_like(x) if empty else torch.ones_like(x)
        weight[:, 1] = 0

        def corrupt(module, inputs, output):
            return torch.where(weight > 0, output[0], float("nan")), output[1]

        hook = student.register_forward_hook(corrupt)
        try:
            losses = loss_fn(
                x, torch.tensor([0.3, 0.7]), phase="student", weight=weight
            )
            assert torch.isfinite(losses["total_loss"])
            assert losses["gan_loss_gen"] == 0
            gradient = torch.autograd.grad(losses["total_loss"], student.gain)[0]
            assert torch.isfinite(gradient)
            if empty:
                assert losses["total_loss"] == 0
                assert gradient == 0
        finally:
            hook.remove()

    @pytest.mark.parametrize("phase", ("student", "auxiliary"))
    def test_masked_statistics(self, phase):
        student, teacher, fake = (StructuredDiffusionModel() for _ in range(3))
        with torch.no_grad():
            fake.gain.fill_(0.3)
        objective = WeightedDistillationLoss(
            student,
            RectifiedFlowNoiseScheduler(),
            method="DMD2",
            teacher=teacher,
            auxiliary_models={"fake_score": fake},
        )
        x, t = torch.randn(2, 3, 5), torch.tensor([0.3, 0.7])
        z, noise = torch.randn_like(x), torch.randn_like(x)
        weight = torch.ones_like(x)
        weight[:, 1] = 0
        corrupted = z.clone()
        corrupted[:, 1] = 1e15
        options = dict(
            t_student=torch.full_like(t, 0.9), noise=noise, weight=weight, phase=phase
        )
        expected = objective(x, t, input_student=z, **options)["total_loss"]
        actual = objective(x, t, input_student=corrupted, **options)["total_loss"]
        torch.testing.assert_close(actual, expected)
        parameter = student.gain if phase == "student" else fake.gain
        torch.testing.assert_close(
            torch.autograd.grad(actual, parameter)[0],
            torch.autograd.grad(expected, parameter)[0],
        )

    def test_explicit_phase(self):
        student, teacher, fake = (StructuredDiffusionModel() for _ in range(3))
        loss_fn = DistillationLoss(
            student,
            RectifiedFlowNoiseScheduler(),
            method="DMD2",
            teacher=teacher,
            auxiliary_models={"fake_score": fake},
        )
        x = torch.randn(2, 3, 5)
        with pytest.raises(ValueError, match="phase"):
            loss_fn(x)
        for phase, target in (("student", student), ("auxiliary", fake)):
            for model in (student, teacher, fake):
                model.zero_grad(set_to_none=True)
            loss = loss_fn(x, torch.tensor([0.3, 0.7]), phase=phase, step=3)
            loss["total_loss"].backward()
            assert target.gain.grad is not None
            for model in (student, teacher, fake):
                assert model.gain.requires_grad
                if model is not target:
                    assert model.gain.grad is None

    @pytest.mark.parametrize("phase", ("student", "auxiliary"))
    def test_unit_and_zero_weights(self, phase):
        student, teacher, fake = (StructuredDiffusionModel() for _ in range(3))
        kwargs = dict(
            method="DMD2", teacher=teacher, auxiliary_models={"fake_score": fake}
        )
        source = RectifiedFlowNoiseScheduler()
        plain = DistillationLoss(student, source, **kwargs)
        weighted = WeightedDistillationLoss(student, source, **kwargs)
        x, t = torch.randn(2, 3, 5), torch.tensor([0.3, 0.7])
        target = student if phase == "student" else fake
        torch.manual_seed(52)
        reference = plain(x, t, phase=phase)["total_loss"]
        reference_grad = torch.autograd.grad(reference, target.gain)[0]
        torch.manual_seed(52)
        actual = weighted(x, t, phase=phase, weight=torch.ones_like(x))["total_loss"]
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(
            torch.autograd.grad(actual, target.gain)[0], reference_grad
        )
        empty = weighted(x, t, phase=phase, weight=torch.zeros_like(x))["total_loss"]
        torch.testing.assert_close(empty, torch.zeros_like(empty))
        torch.testing.assert_close(
            torch.autograd.grad(empty, target.gain)[0], torch.zeros_like(target.gain)
        )

    @pytest.mark.parametrize("phase", ("student", "auxiliary"))
    def test_backend_parity(self, phase):
        from fastgen.methods.distribution_matching.dmd2 import DMD2Model

        student, teacher, fake = (StructuredDiffusionModel() for _ in range(3))
        with torch.no_grad():
            fake.gain.fill_(0.25)
        loss_fn = DistillationLoss(
            student,
            RectifiedFlowNoiseScheduler(),
            method="DMD2",
            teacher=teacher,
            auxiliary_models={"fake_score": fake},
            model_options={
                "student": {"scheduler_options": {"time_precision": torch.float32}}
            },
        )
        impl = loss_fn._implementation
        x = torch.randn(2, 3, 5)
        t = torch.tensor([0.3, 0.7], dtype=impl.net.noise_scheduler.t_precision)
        z, eps = torch.randn_like(x), torch.randn_like(x)
        t_student = torch.full_like(t, 0.9)
        actual = loss_fn(
            x, t, phase=phase, input_student=z, t_student=t_student, noise=eps
        )["total_loss"]
        impl.device = x.device
        if phase == "student":
            expected, _ = DMD2Model._student_update_step(
                impl, z, t_student, t, eps, {"real": x}
            )
        else:
            expected, _ = DMD2Model._fake_score_discriminator_update_step(
                impl, z, t_student, t, eps, x
            )
        target = student if phase == "student" else fake
        torch.testing.assert_close(actual, expected["total_loss"])
        torch.testing.assert_close(
            torch.autograd.grad(actual, target.gain)[0],
            torch.autograd.grad(expected["total_loss"], target.gain)[0],
        )

    def test_diagnostics_rng(self):
        loss_fn = DistillationLoss(
            StructuredDiffusionModel(),
            RectifiedFlowNoiseScheduler(),
            method="DMD2",
            teacher=StructuredDiffusionModel(),
            auxiliary_models={"fake_score": StructuredDiffusionModel()},
        )
        x = torch.randn(2, 3, 5)
        torch.manual_seed(61)
        expected = loss_fn(x, phase="student")
        state = torch.get_rng_state()
        torch.manual_seed(61)
        actual, outputs = loss_fn(x, phase="student", return_outputs=True)
        torch.testing.assert_close(actual["total_loss"], expected["total_loss"])
        assert torch.equal(state, torch.get_rng_state())
        assert all(not value.requires_grad for value in outputs.values())


@pytest.mark.parametrize("method", ("DMD2", "f-distill"))
class TestAdversarial:
    def test_feature_callback_required(self, method):
        with pytest.raises(ValueError, match="feature_fn"):
            DistillationLoss(
                StructuredDiffusionModel(),
                RectifiedFlowNoiseScheduler(),
                method=method,
                teacher=StructuredDiffusionModel(),
                auxiliary_models={
                    "fake_score": StructuredDiffusionModel(),
                    "discriminator": FeatureDiscriminator(),
                },
                model_options={"teacher": {"feature_names": ("spatial",)}},
            )

    def test_prediction_required(self, method):
        teacher = StructuredDiffusionModel()
        hook = teacher.register_forward_hook(
            lambda module, args, output: (None, output[1])
        )
        loss_fn = DistillationLoss(
            StructuredDiffusionModel(),
            RectifiedFlowNoiseScheduler(),
            method=method,
            teacher=teacher,
            auxiliary_models={
                "fake_score": StructuredDiffusionModel(),
                "discriminator": FeatureDiscriminator(),
            },
            model_options={
                "teacher": {
                    "feature_names": ("spatial",),
                    "feature_fn": extract_teacher_features,
                }
            },
        )
        with pytest.raises(TypeError, match="prediction tensor"):
            loss_fn(torch.randn(2, 3, 5), phase="student")
        hook.remove()

    @pytest.mark.parametrize("phase", ("student", "auxiliary"))
    def test_weighted_logits(self, method, phase):
        from tensordict import TensorDict

        student, teacher, fake = (StructuredDiffusionModel() for _ in range(3))
        discriminator = FeatureDiscriminator()
        kwargs = dict(
            method=method,
            teacher=teacher,
            auxiliary_models={"fake_score": fake, "discriminator": discriminator},
            model_options={
                "teacher": {
                    "feature_names": ("spatial",),
                    "feature_fn": extract_teacher_features,
                }
            },
            method_options={"gan_use_same_t_noise": True, "gan_r1_reg_weight": 0.5},
        )
        source = RectifiedFlowNoiseScheduler()
        plain = DistillationLoss(student, source, **kwargs)
        weighted = WeightedDistillationLoss(student, source, **kwargs)

        def adversarial(x, time, condition, weight):
            valid = weight > 0
            features = torch.where(valid, x, 0)
            logits = discriminator(TensorDict({"spatial": features}, [x.shape[0]]))
            return logits, valid.flatten(1).any(1, keepdim=True).to(x.dtype)

        x, t = torch.randn(2, 3, 5), torch.tensor([0.3, 0.7])
        torch.manual_seed(88)
        expected = plain(x, t, phase=phase)["total_loss"]
        torch.manual_seed(88)
        actual = weighted(
            x, t, phase=phase, weight=torch.ones_like(x), adversarial_fn=adversarial
        )["total_loss"]
        torch.testing.assert_close(actual, expected)
        targets = (
            (student.gain,) if phase == "student" else (fake.gain, discriminator.gain)
        )
        for got, reference in zip(
            torch.autograd.grad(actual, targets), torch.autograd.grad(expected, targets)
        ):
            torch.testing.assert_close(got, reference)
        state = weighted.state_dict()
        empty = weighted(
            x, t, phase=phase, weight=torch.zeros_like(x), adversarial_fn=adversarial
        )["total_loss"]
        torch.testing.assert_close(empty, torch.zeros_like(empty))
        for gradient in torch.autograd.grad(empty, targets):
            torch.testing.assert_close(gradient, torch.zeros_like(gradient))
        if state:
            torch.testing.assert_close(weighted.state_dict()["bins"], state["bins"])

    @pytest.mark.parametrize("phase", ("student", "auxiliary"))
    def test_backend_gan_parity(self, method, phase):
        from types import SimpleNamespace

        from fastgen.methods.distribution_matching.dmd2 import DMD2Model
        from fastgen.methods.distribution_matching.f_distill import FdistillModel

        student, teacher, fake = (StochasticDiffusionModel() for _ in range(3))
        discriminator = FeatureDiscriminator()
        objective = DistillationLoss(
            student,
            RectifiedFlowNoiseScheduler(),
            method=method,
            teacher=teacher,
            auxiliary_models={"fake_score": fake, "discriminator": discriminator},
            model_options={
                "student": {"scheduler_options": {"time_precision": torch.float32}},
                "teacher": {
                    "feature_names": ("spatial",),
                    "feature_fn": extract_teacher_features,
                },
            },
            method_options={"gan_use_same_t_noise": True, "gan_r1_reg_weight": 0.5},
        )
        impl = objective._implementation
        x, t = torch.randn(2, 3, 5), torch.tensor([0.3, 0.7])
        z, eps, ts = torch.randn_like(x), torch.randn_like(x), torch.full_like(t, 0.9)
        torch.manual_seed(89)
        actual = objective(x, t, phase=phase, input_student=z, t_student=ts, noise=eps)[
            "total_loss"
        ]
        state = torch.get_rng_state()
        if method == "f-distill":
            impl.config.f_distill = SimpleNamespace(**vars(impl._ratio_options))
            impl.bin_num = impl._ratio_options.bin_num
            impl.net.bins = torch.ones(impl.bin_num)
            impl._get_f_div_weighting_h = lambda logits, time: (
                FdistillModel._get_f_div_weighting_h(impl, logits, time)
            )
        torch.manual_seed(89)
        if phase == "student":
            backend = FdistillModel if method == "f-distill" else DMD2Model
            expected, _ = backend._student_update_step(impl, z, ts, t, eps, {"real": x})
        else:
            expected, _ = DMD2Model._fake_score_discriminator_update_step(
                impl, z, ts, t, eps, x
            )
        torch.testing.assert_close(actual, expected["total_loss"])
        assert torch.equal(torch.get_rng_state(), state)
        targets = (
            (student.gain,) if phase == "student" else (fake.gain, discriminator.gain)
        )
        for got, reference in zip(
            torch.autograd.grad(actual, targets),
            torch.autograd.grad(expected["total_loss"], targets),
        ):
            torch.testing.assert_close(got, reference)

    def test_teacher_called_once(self, method):
        teacher = StructuredDiffusionModel()
        calls = []
        hook = teacher.register_forward_hook(lambda *args: calls.append(1))
        loss_fn = DistillationLoss(
            StructuredDiffusionModel(),
            RectifiedFlowNoiseScheduler(),
            method=method,
            teacher=teacher,
            auxiliary_models={
                "fake_score": StructuredDiffusionModel(),
                "discriminator": FeatureDiscriminator(),
            },
            model_options={
                "teacher": {
                    "feature_names": ("spatial",),
                    "feature_fn": extract_teacher_features,
                }
            },
        )
        loss_fn(torch.randn(2, 3, 5), phase="student")
        hook.remove()
        assert len(calls) == 1

    @pytest.mark.parametrize("phase", ("student", "auxiliary"))
    def test_role_gradients(self, method, phase):
        student, teacher, fake = (StructuredDiffusionModel() for _ in range(3))
        discriminator = FeatureDiscriminator()
        loss_fn = DistillationLoss(
            student,
            RectifiedFlowNoiseScheduler(),
            method=method,
            teacher=teacher,
            auxiliary_models={"fake_score": fake, "discriminator": discriminator},
            model_options={
                "teacher": {
                    "feature_names": ("spatial",),
                    "feature_fn": extract_teacher_features,
                }
            },
            method_options={"gan_r1_reg_weight": 0.5},
        )
        x = torch.randn(2, 3, 5)
        loss_fn(x, torch.tensor([0.3, 0.7]), phase=phase)["total_loss"].backward()
        assert (student.gain.grad is not None) == (phase == "student")
        assert teacher.gain.grad is None
        assert (fake.gain.grad is not None) == (phase == "auxiliary")
        assert (discriminator.gain.grad is not None) == (phase == "auxiliary")
        assert all(
            model.gain.requires_grad
            for model in (student, teacher, fake, discriminator)
        )

    def test_mask_capability_required(self, method):
        loss_fn = WeightedDistillationLoss(
            StructuredDiffusionModel(),
            RectifiedFlowNoiseScheduler(),
            method=method,
            teacher=StructuredDiffusionModel(),
            auxiliary_models={
                "fake_score": StructuredDiffusionModel(),
                "discriminator": FeatureDiscriminator(),
            },
            model_options={
                "teacher": {
                    "feature_names": ("spatial",),
                    "feature_fn": extract_teacher_features,
                }
            },
        )
        x = torch.randn(2, 3, 5)
        rng = torch.get_rng_state()
        with pytest.raises(ValueError, match="adversarial_fn"):
            loss_fn(x, phase="student", weight=torch.ones_like(x))
        assert torch.equal(torch.get_rng_state(), rng)


class TestRatioState:
    @pytest.mark.parametrize(
        "divergence", ("rkl", "kl", "js", "sf", "neyman", "sh", "jf")
    )
    def test_divergences(self, divergence):
        loss_fn = DistillationLoss(
            StructuredDiffusionModel(),
            RectifiedFlowNoiseScheduler(),
            method="f-distill",
            teacher=StructuredDiffusionModel(),
            auxiliary_models={
                "fake_score": StructuredDiffusionModel(),
                "discriminator": FeatureDiscriminator(),
            },
            model_options={
                "teacher": {
                    "feature_names": ("spatial",),
                    "feature_fn": extract_teacher_features,
                }
            },
            method_options={"f_distill": {"f_div": divergence}},
        )
        loss = loss_fn(torch.randn(2, 3, 5), phase="student")["total_loss"]
        assert torch.isfinite(loss)
        loss.backward()

    def test_update_and_restore(self):
        options = dict(
            method="f-distill",
            teacher=StructuredDiffusionModel(),
            auxiliary_models={
                "fake_score": StructuredDiffusionModel(),
                "discriminator": FeatureDiscriminator(),
            },
            model_options={
                "teacher": {
                    "feature_names": ("spatial",),
                    "feature_fn": extract_teacher_features,
                }
            },
        )
        loss_fn = DistillationLoss(
            StructuredDiffusionModel(), RectifiedFlowNoiseScheduler(), **options
        )
        x, t = torch.randn(2, 3, 5), torch.tensor([0.3, 0.7])
        before = loss_fn.state_dict()
        loss_fn(x, t, phase="student", update_state=False)
        torch.testing.assert_close(loss_fn.state_dict()["bins"], before["bins"])
        loss_fn(x, t, phase="student")
        state = loss_fn.state_dict()
        assert not torch.equal(state["bins"], before["bins"])
        fresh = DistillationLoss(
            StructuredDiffusionModel(), RectifiedFlowNoiseScheduler(), **options
        )
        fresh.load_state_dict(state)
        torch.manual_seed(74)
        expected = loss_fn(x, t, phase="student")["total_loss"]
        torch.manual_seed(74)
        actual = fresh(x, t, phase="student")["total_loss"]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            fresh.state_dict()["bins"], loss_fn.state_dict()["bins"]
        )


class TestDistributionIntegration:
    def test_custom_role_coordinates(self):
        source, teacher_source = RectifiedFlowNoiseScheduler(), EDMNoiseScheduler()
        student, fake = StructuredDiffusionModel(), StructuredDiffusionModel()
        common = dict(method="DMD2", auxiliary_models={"fake_score": fake})
        native = DistillationLoss(
            student, source, teacher=GaussianDenoiser(source), **common
        )
        converted = DistillationLoss(
            student,
            source,
            teacher=GaussianDenoiser(teacher_source),
            **common,
            model_options={"teacher": {"noise_scheduler": teacher_source}},
        )
        x, t = torch.randn(2, 3, 5), torch.tensor([0.3, 0.7])
        torch.manual_seed(94)
        expected = native(x, t, phase="student")["total_loss"]
        torch.manual_seed(94)
        actual = converted(x, t, phase="student")["total_loss"]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, student.gain)[0],
            torch.autograd.grad(expected, student.gain)[0],
        )

    @pytest.mark.parametrize("phase", ("student", "auxiliary"))
    def test_user_compiled_roles(self, phase):
        student, teacher, fake = (StructuredDiffusionModel() for _ in range(3))
        source = RectifiedFlowNoiseScheduler()
        eager = DistillationLoss(
            student,
            source,
            method="DMD2",
            teacher=teacher,
            auxiliary_models={"fake_score": fake},
        )
        compiled = DistillationLoss(
            torch.compile(student, backend="eager", fullgraph=True),
            source,
            method="DMD2",
            teacher=torch.compile(teacher, backend="eager", fullgraph=True),
            auxiliary_models={
                "fake_score": torch.compile(fake, backend="eager", fullgraph=True)
            },
        )
        for _ in range(2):
            x, t = torch.randn(2, 3, 5), torch.tensor([0.3, 0.7])
            torch.manual_seed(95)
            expected = eager(x, t, phase=phase)["total_loss"]
            torch.manual_seed(95)
            actual = compiled(x, t, phase=phase)["total_loss"]
            torch.testing.assert_close(actual, expected)
            target = student.gain if phase == "student" else fake.gain
            torch.testing.assert_close(
                torch.autograd.grad(actual, target)[0],
                torch.autograd.grad(expected, target)[0],
            )

    def test_exception_restores_freeze_mask(self):
        student, teacher, fake = (StructuredDiffusionModel() for _ in range(3))
        discriminator = FeatureDiscriminator()
        teacher.requires_grad_(False)
        loss_fn = WeightedDistillationLoss(
            student,
            RectifiedFlowNoiseScheduler(),
            method="DMD2",
            teacher=teacher,
            auxiliary_models={"fake_score": fake, "discriminator": discriminator},
            model_options={
                "teacher": {
                    "feature_names": ("spatial",),
                    "feature_fn": extract_teacher_features,
                }
            },
        )

        def fail(*args):
            raise RuntimeError("test callback failure")

        x = torch.randn(2, 3, 5)
        with pytest.raises(RuntimeError, match="test callback failure"):
            loss_fn(x, phase="student", weight=torch.ones_like(x), adversarial_fn=fail)
        assert not teacher.gain.requires_grad
        assert all(model.gain.requires_grad for model in (student, fake, discriminator))
        assert loss_fn._implementation._call_weight is None
