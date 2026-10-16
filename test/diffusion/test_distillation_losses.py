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

r"""Exercise losses without a trainer, implicit model setup, or optimizer ownership."""

import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("fastgen")

from physicsnemo.diffusion import (
    DiffusionExtraTimes,
)
from physicsnemo.diffusion.distillation import (
    BaseDistillationLoss,
    DistillationLoss,
    WeightedDistillationLoss,
)
from physicsnemo.diffusion.noise_schedulers import (
    EDMNoiseScheduler,
    RectifiedFlowNoiseScheduler,
    TrigFlowNoiseScheduler,
)

from .helpers import (
    AttentionDiffusionModel,
    CustomGaussianScheduler,
    GaussianDenoiser,
    StochasticDiffusionModel,
    StructuredDiffusionModel,
    load_or_create_reference,
)

CONFIGS = (("sCM", TrigFlowNoiseScheduler), ("MeanFlow", RectifiedFlowNoiseScheduler))
IDS = ("scm", "meanflow")


@pytest.mark.parametrize("method,scheduler_type", CONFIGS, ids=IDS)
class TestConsistency:
    def test_teacher_coordinates(self, method, scheduler_type):
        source = scheduler_type()
        teacher_source = EDMNoiseScheduler()
        student = StructuredDiffusionModel()
        expected_fn = DistillationLoss(
            student, source, teacher=GaussianDenoiser(source), method=method
        )
        actual_fn = DistillationLoss(
            student,
            source,
            teacher=GaussianDenoiser(teacher_source),
            method=method,
            model_options={"teacher": {"noise_scheduler": teacher_source}},
        )
        x = torch.randn(2, 3, 5)
        time = (
            (
                torch.tensor([0.3, 0.7]),
                DiffusionExtraTimes(r=torch.tensor([0.1, 0.2]), batch_size=[2]),
            )
            if method == "MeanFlow"
            else torch.tensor([0.3, 0.7])
        )
        torch.manual_seed(45)
        expected = expected_fn(x, time, step=1000)["total_loss"]
        torch.manual_seed(45)
        actual = actual_fn(x, time, step=1000)["total_loss"]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, student.gain)[0],
            torch.autograd.grad(expected, student.gain)[0],
        )

    def test_time_callback(self, method, scheduler_type):
        objective = DistillationLoss(
            StructuredDiffusionModel(), scheduler_type(), method=method
        )
        x = torch.randn(2, 3, 5)
        time = (
            (
                torch.tensor([0.3, 0.7]),
                DiffusionExtraTimes(r=torch.tensor([0.1, 0.2]), batch_size=[2]),
            )
            if method == "MeanFlow"
            else torch.tensor([0.3, 0.7])
        )
        torch.manual_seed(21)
        expected = objective(x, time, step=1000)["total_loss"]
        torch.manual_seed(21)
        actual = objective(x, step=1000, sample_t_fn=lambda n, **kwargs: time)[
            "total_loss"
        ]
        torch.testing.assert_close(actual, expected)
        torch.manual_seed(21)
        torch.testing.assert_close(
            objective(x, time, step=1000)["total_loss"], expected
        )

    def test_masked_statistics(self, method, scheduler_type):
        student = StructuredDiffusionModel()
        objective = WeightedDistillationLoss(student, scheduler_type(), method=method)
        x = torch.randn(2, 3, 5)
        time = (
            (
                torch.tensor([0.3, 0.7]),
                DiffusionExtraTimes(r=torch.tensor([0.1, 0.2]), batch_size=[2]),
            )
            if method == "MeanFlow"
            else torch.tensor([0.3, 0.7])
        )
        weight = torch.ones_like(x)
        weight[:, 1] = 0
        corrupted = x.clone()
        corrupted[:, 1] = 1e15
        torch.manual_seed(43)
        expected = objective(x, time, step=1000, weight=weight)["total_loss"]
        torch.manual_seed(43)
        actual = objective(corrupted, time, step=1000, weight=weight)["total_loss"]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, student.gain)[0],
            torch.autograd.grad(expected, student.gain)[0],
        )

    def test_unit_weights(self, method, scheduler_type):
        student = StructuredDiffusionModel()
        teacher = StructuredDiffusionModel()
        plain = DistillationLoss(
            student, scheduler_type(), teacher=teacher, method=method
        )
        weighted = WeightedDistillationLoss(
            student, scheduler_type(), teacher=teacher, method=method
        )
        assert isinstance(plain, BaseDistillationLoss)
        x = torch.randn(2, 3, 5)
        times = (
            (
                torch.tensor([0.3, 0.7]),
                DiffusionExtraTimes(r=torch.tensor([0.1, 0.2]), batch_size=[2]),
            )
            if method == "MeanFlow"
            else torch.tensor([0.3, 0.7])
        )
        torch.manual_seed(41)
        expected = plain(x, times, step=1000)["total_loss"]
        torch.manual_seed(41)
        actual = weighted(x, times, step=1000, weight=torch.ones_like(x))["total_loss"]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, student.gain)[0],
            torch.autograd.grad(expected, student.gain)[0],
        )
        assert teacher.gain.grad is None

    def test_outputs(self, method, scheduler_type):
        loss = DistillationLoss(
            StructuredDiffusionModel(), scheduler_type(), method=method
        )
        x = torch.randn(2, 3, 5)
        torch.manual_seed(19)
        expected = loss(x, step=1000)
        state = torch.random.get_rng_state()
        torch.manual_seed(19)
        actual, output = loss(x, step=1000, return_outputs=True)
        torch.testing.assert_close(actual["total_loss"], expected["total_loss"])
        torch.testing.assert_close(torch.random.get_rng_state(), state)
        assert output
        assert all(not value.requires_grad for value in output.values())

    @pytest.mark.parametrize("logvar", (0.0, float("nan"), -float("inf")))
    def test_empty_mask(self, method, scheduler_type, logvar):
        student = StructuredDiffusionModel(logvar=logvar)
        loss = WeightedDistillationLoss(student, scheduler_type(), method=method)
        x = torch.randn(2, 3, 5)
        result = loss(x, step=1000, weight=torch.zeros_like(x))["total_loss"]
        assert result.item() == 0
        (gradient,) = torch.autograd.grad(result, student.gain)
        assert gradient.item() == 0
        if method == "sCM":
            result = loss(x, step=1000, weight=torch.zeros_like(x))["total_loss"]
            (gradient,) = torch.autograd.grad(result, student.logvar)
            assert gradient.item() == 0


@pytest.mark.parametrize("method,scheduler_type", CONFIGS, ids=IDS)
class TestNonRegression:
    def test_training(self, method, scheduler_type, tolerances):
        student = StructuredDiffusionModel()
        loss_fn = WeightedDistillationLoss(student, scheduler_type(), method=method)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
        torch.manual_seed(61)
        x = torch.randn(2, 3, 5)
        weight = torch.tensor([1.0, 0.0, 0.5])[None, :, None]
        time = (
            (
                torch.tensor([0.3, 0.7]),
                DiffusionExtraTimes(r=torch.tensor([0.1, 0.2]), batch_size=[2]),
            )
            if method == "MeanFlow"
            else torch.tensor([0.3, 0.7])
        )
        actual = {}
        for step in range(2):
            optimizer.zero_grad()
            loss = loss_fn(x, time, step=1000 + step, weight=weight)["total_loss"]
            loss.backward()
            actual[f"loss_{step}"] = loss.detach()
            actual[f"gradient_{step}"] = student.gain.grad.detach().clone()
            optimizer.step()
            actual[f"parameter_{step}"] = student.gain.detach().clone()
        reference = load_or_create_reference(
            f"distillation_{method.lower()}_weighted.pth", lambda: actual
        )
        for key, value in actual.items():
            torch.testing.assert_close(value, reference[key], **tolerances)

    def test_checkpoint(self, method, scheduler_type, tmp_path):
        student = StructuredDiffusionModel()
        loss_fn = DistillationLoss(student, scheduler_type(), method=method)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
        x = torch.randn(2, 3, 5)
        loss_fn(x, step=1000)["total_loss"].backward()
        optimizer.step()
        path = tmp_path / "student.mdlus"
        student.save(str(path))
        restored = StructuredDiffusionModel.from_checkpoint(str(path))
        restored_loss = DistillationLoss(restored, scheduler_type(), method=method)
        restored_loss.load_state_dict(loss_fn.state_dict())
        torch.manual_seed(67)
        expected = loss_fn(x, step=1001)["total_loss"]
        torch.manual_seed(67)
        actual = restored_loss(x, step=1001)["total_loss"]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, restored.gain)[0],
            torch.autograd.grad(expected, student.gain)[0],
        )


class TestDiagnostics:
    def test_missing_destination_time(self):
        objective = DistillationLoss(
            StructuredDiffusionModel(), RectifiedFlowNoiseScheduler(), method="MeanFlow"
        )
        times = (torch.ones(2), DiffusionExtraTimes(batch_size=[2]))
        with pytest.raises(ValueError, match="MeanFlow.*r"):
            objective(torch.randn(2, 3), times)

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown"):
            DistillationLoss(
                StructuredDiffusionModel(),
                RectifiedFlowNoiseScheduler(),
                method="not-a-method",
            )

    def test_invalid_weight(self):
        loss = WeightedDistillationLoss(
            StructuredDiffusionModel(), RectifiedFlowNoiseScheduler(), method="MeanFlow"
        )
        with pytest.raises(ValueError, match="nonnegative"):
            loss(torch.ones(2, 3), weight=torch.tensor(-1.0))


@pytest.mark.parametrize("method,scheduler_type", CONFIGS, ids=IDS)
class TestCompile:
    def test_attention(self, method, scheduler_type):
        student = AttentionDiffusionModel()
        objective = WeightedDistillationLoss(student, scheduler_type(), method=method)
        x = torch.randn(2, 2, 4, 8)
        time = (
            (
                torch.tensor([0.3, 0.7]),
                DiffusionExtraTimes(r=torch.tensor([0.1, 0.2]), batch_size=[2]),
            )
            if method == "MeanFlow"
            else torch.tensor([0.3, 0.7])
        )

        def fn(x):
            return objective(x, time, step=1000)["total_loss"]

        torch.manual_seed(57)
        expected = fn(x)
        compiled = torch.compile(fn, fullgraph=True, backend="aot_eager")
        torch.manual_seed(57)
        actual = compiled(x)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, student.gain)[0],
            torch.autograd.grad(expected, student.gain)[0],
        )

    def test_call(self, method, scheduler_type):
        student = StructuredDiffusionModel()
        objective = WeightedDistillationLoss(student, scheduler_type(), method=method)
        x = torch.randn(2, 3, 5)
        time = (
            (
                torch.tensor([0.3, 0.7]),
                DiffusionExtraTimes(r=torch.tensor([0.1, 0.2]), batch_size=[2]),
            )
            if method == "MeanFlow"
            else torch.tensor([0.3, 0.7])
        )
        weight = torch.tensor([1.0, 0.0, 0.5])[None, :, None]

        def fn(x):
            return objective(x, time, step=1000, weight=weight)["total_loss"]

        torch.manual_seed(17)
        expected = fn(x)
        compiled = torch.compile(fn, fullgraph=True, backend="aot_eager")
        torch.manual_seed(17)
        actual = compiled(x)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, student.gain)[0],
            torch.autograd.grad(expected, student.gain)[0],
        )
        torch.manual_seed(17)
        with torch._dynamo.config.patch(error_on_recompile=True):
            torch.testing.assert_close(compiled(x), expected)


class TestGradientFlow:
    @pytest.mark.parametrize("loss_type", ("l2", "opt_grad"))
    def test_input_gradients(self, loss_type):
        student = StructuredDiffusionModel()
        objective = DistillationLoss(
            student,
            RectifiedFlowNoiseScheduler(),
            method="MeanFlow",
            method_options={"loss_type": loss_type},
        )
        x = torch.randn(2, 3, 5, requires_grad=True)
        time = (
            torch.tensor([0.3, 0.7]),
            DiffusionExtraTimes(r=torch.tensor([0.1, 0.2]), batch_size=[2]),
        )
        result = objective(x, time, step=1000)["total_loss"]
        gradient = torch.autograd.grad(result, x)[0]
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0

    @pytest.mark.parametrize("training", (False, True), ids=("eval", "mixed"))
    def test_guidance_preserves_modes(self, training):
        student = StochasticDiffusionModel().train(training)
        student.dropout.eval()
        before = [module.training for module in student.modules()]
        objective = DistillationLoss(
            student,
            RectifiedFlowNoiseScheduler(),
            method="MeanFlow",
            method_options={"guidance_scale": 2.0},
        )
        x = torch.randn(2, 3, 5)
        condition = TensorDict({"shift": torch.ones(2, 1, 1)}, [2])
        negative = TensorDict({"shift": torch.zeros(2, 1, 1)}, [2])
        result = objective(x, condition=condition, neg_condition=negative)["total_loss"]
        torch.autograd.grad(result, student.gain)
        assert [module.training for module in student.modules()] == before

    def test_guidance_native_times(self):
        student, teacher = StructuredDiffusionModel(), StructuredDiffusionModel()
        x = torch.randn(2, 3, 5)
        time = (
            torch.tensor([2.0, 3.0]),
            DiffusionExtraTimes(r=torch.tensor([0.4, 0.7]), batch_size=[2]),
        )
        condition = TensorDict({"shift": torch.ones(2, 1, 1)}, [2])
        negative = TensorDict({"shift": torch.zeros(2, 1, 1)}, [2])
        results = []
        for scale in (1.0, 2.0):
            objective = DistillationLoss(
                student,
                EDMNoiseScheduler(),
                teacher=teacher,
                method="MeanFlow",
                method_options={"guidance_scale": scale},
                model_options={"teacher": {"prediction_type": "flow"}},
            )
            torch.manual_seed(123)
            results.append(
                objective(x, time, condition, neg_condition=negative)["total_loss"]
            )
        assert not torch.equal(results[0], results[1])

    def test_custom_scheduler_without_bounds(self):
        student = StructuredDiffusionModel()
        objective = DistillationLoss(
            student, CustomGaussianScheduler(), method="MeanFlow"
        )
        times = (
            torch.tensor([0.3, 0.7]),
            DiffusionExtraTimes(r=torch.tensor([-0.1, 0.2]), batch_size=[2]),
        )
        result = objective(torch.randn(2, 3), sample_t_fn=lambda n, **kwargs: times)[
            "total_loss"
        ]
        (gradient,) = torch.autograd.grad(result, student.gain)
        assert torch.isfinite(result)
        assert torch.isfinite(gradient)

    @pytest.mark.parametrize(
        "model_type",
        (StructuredDiffusionModel, StochasticDiffusionModel),
        ids=("deterministic", "dropout"),
    )
    def test_meanflow_backend_parity(self, model_type):
        from fastgen.methods.consistency_model.mean_flow import MeanFlowModel

        student = model_type()
        objective = DistillationLoss(
            student, RectifiedFlowNoiseScheduler(), method="MeanFlow"
        )
        x = torch.randn(2, 3, 5)
        t, r = torch.tensor([0.3, 0.7]), torch.tensor([0.1, 0.2])
        time = (t, DiffusionExtraTimes(r=r, batch_size=[2]))
        torch.manual_seed(123)
        actual = objective(x, time)["total_loss"]
        actual_rng = torch.get_rng_state()
        binding = objective._implementation
        torch.manual_seed(123)
        z = torch.randn_like(x)
        xt = binding.net.noise_scheduler.forward_process(x, z, t)
        _, velocity, _ = binding._get_velocity(x, z, t)
        tangent = binding._jvp(xt, t, r, velocity)
        prediction = binding.net(xt, t, r=r, fwd_pred_type="flow")
        reference = MeanFlowModel._mf_pred_to_loss(
            binding, prediction, tangent, xt, velocity, t, r, 0
        )[0].mean()
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(actual_rng, torch.get_rng_state())
        torch.testing.assert_close(
            torch.autograd.grad(actual, student.gain)[0],
            torch.autograd.grad(reference, student.gain)[0],
        )
