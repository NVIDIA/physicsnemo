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

r"""Check primary tensors and extra fields through diffusion wrappers."""

import pytest
import torch
from jaxtyping import Float

from physicsnemo.diffusion import DiffusionExtraOutputs, DiffusionExtraTimes
from physicsnemo.diffusion.preconditioners import (
    EDMPreconditioner,
    IDDPMPreconditioner,
    TrigFlowPreconditioner,
    VEPreconditioner,
    VPPreconditioner,
)
from physicsnemo.diffusion.utils.model_wrappers import ConcatConditionWrapper

from .helpers import CustomExtraTimes, StructuredDiffusionModel

CONFIGS = (
    (EDMPreconditioner, {}),
    (VEPreconditioner, {}),
    (VPPreconditioner, {}),
    (IDDPMPreconditioner, {"M": 32}),
    (TrigFlowPreconditioner, {"sigma_data": 1.4}),
)
IDS = ("edm", "ve", "vp", "iddpm", "trigflow")

TIME_FIELDS = (None, (), ("r",), ("r1",), ("r2",), ("r1", "r2"))
TIME_IDS = ("tensor", "empty", "r", "r1", "r2", "r1_r2")


def _make_time(
    t: Float[torch.Tensor, " B"], fields: tuple[str, ...] | None
) -> Float[torch.Tensor, " B"] | tuple[Float[torch.Tensor, " B"], CustomExtraTimes]:
    r"""Pair the primary time with the selected extra fields."""
    if fields is None:
        return t
    return t, CustomExtraTimes(
        **{name: t / (index + 2) for index, name in enumerate(fields)},
        batch_size=t.shape,
    )


@pytest.mark.parametrize("preconditioner,kwargs", CONFIGS, ids=IDS)
class TestConsistency:
    @pytest.mark.parametrize("fields", TIME_FIELDS, ids=TIME_IDS)
    def test_tuple_protocol(self, preconditioner, kwargs, device, fields):
        model = preconditioner(StructuredDiffusionModel(), **kwargs).to(device)
        x = torch.randn(2, 3, 5, 7, device=device)
        t = torch.tensor([0.4, 0.8], device=device)
        time = _make_time(t, fields)
        calls = []
        handle = model.model.register_forward_hook(
            lambda module, inputs, output: calls.append((inputs, output))
        )
        prediction, outputs = model(x, time, offset=0.2)
        handle.remove()
        inputs, raw = calls[0]
        sigma = model.sigma(t).reshape(2, 1, 1, 1)  # (B, 1, 1, 1)
        c_in, c_noise, c_out, c_skip = model.compute_coefficients(sigma)
        torch.testing.assert_close(inputs[0], c_in * x)
        encoded_t = inputs[1][0] if fields is not None else inputs[1]
        torch.testing.assert_close(encoded_t, c_noise.flatten())
        if fields is not None:
            assert isinstance(inputs[1][1], CustomExtraTimes)
            for index, name in enumerate(fields):
                original = t / (index + 2)
                sigma_extra = model.sigma(original).reshape(2, 1, 1, 1)
                encoded = model.compute_coefficients(sigma_extra)[1].flatten()
                torch.testing.assert_close(getattr(inputs[1][1], name), encoded)
                torch.testing.assert_close(getattr(time[1], name), original)
            for name in {"r", "r1", "r2"}.difference(fields):
                assert getattr(inputs[1][1], name) is None
        torch.testing.assert_close(prediction, c_skip * x + c_out * raw[0])
        assert outputs is raw[1]
        torch.testing.assert_close(outputs.logvar, model.model.gain * encoded_t)

    def test_tensor_output(self, preconditioner, kwargs, device):
        model = preconditioner(StructuredDiffusionModel(), **kwargs).to(device)
        plain = preconditioner(StructuredDiffusionModel("tensor"), **kwargs).to(device)
        x = torch.randn(2, 3, device=device)
        t = torch.tensor([0.4, 0.8], device=device)
        prediction, _ = model(x, t)
        torch.testing.assert_close(prediction, plain(x, t))


class TestConstructor:
    def test_optional_outputs(self):
        extras = DiffusionExtraOutputs(batch_size=[2])
        assert extras.logvar is None

    def test_conditioning_wrapper(self):
        model = StructuredDiffusionModel()
        wrapper = ConcatConditionWrapper(model)
        x = torch.randn(2, 3, 5, 7)
        t = torch.tensor([0.4, 0.8])
        time = (t, DiffusionExtraTimes(r=t / 2, batch_size=[2]))
        condition = torch.ones(2, 1, 5, 7)
        prediction, extras = wrapper(x, time, condition)
        expected, expected_extras = model(torch.cat((x, condition), dim=1), time)
        torch.testing.assert_close(prediction, expected)
        torch.testing.assert_close(extras.logvar, expected_extras.logvar)


@pytest.mark.parametrize("preconditioner,kwargs", CONFIGS, ids=IDS)
class TestGradientFlow:
    @pytest.mark.parametrize("fields", TIME_FIELDS, ids=TIME_IDS)
    def test_numerical_gradients(self, preconditioner, kwargs, fields):
        model = preconditioner(StructuredDiffusionModel(), **kwargs).double()
        x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        t = torch.tensor([0.4, 0.8], dtype=torch.float64, requires_grad=True)

        def forward(x, t):
            prediction, extras = model(x, _make_time(t, fields))
            return prediction, extras.logvar

        assert torch.autograd.gradcheck(forward, (x, t))


@pytest.mark.parametrize("preconditioner,kwargs", CONFIGS, ids=IDS)
class TestCompile:
    @pytest.mark.parametrize("fields", TIME_FIELDS, ids=TIME_IDS)
    def test_forward_backward(self, preconditioner, kwargs, device, fields):
        model = preconditioner(StructuredDiffusionModel(), **kwargs).to(device)
        x = torch.randn(2, 3, 5, 7, device=device, requires_grad=True)
        t = torch.tensor([0.4, 0.8], device=device, requires_grad=True)
        time = _make_time(t, fields)
        expected, expected_extras = model(x, time)
        expected_grads = torch.autograd.grad(
            expected.sum() + expected_extras.logvar.sum(),
            (x, t, model.model.gain),
            retain_graph=True,
        )
        compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
        result, extras = compiled(x, time)
        with torch._dynamo.config.patch(error_on_recompile=True):
            repeated, _ = compiled(x, time)
        torch.testing.assert_close(repeated, result)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(extras.logvar, expected_extras.logvar)
        gradients = torch.autograd.grad(
            result.sum() + extras.logvar.sum(), (x, t, model.model.gain)
        )
        for actual, reference in zip(gradients, expected_grads):
            torch.testing.assert_close(actual, reference)

    def test_conditioning_wrapper(self, preconditioner, kwargs, device):
        wrapper = ConcatConditionWrapper(
            preconditioner(StructuredDiffusionModel(), **kwargs)
        ).to(device)
        x = torch.randn(2, 3, device=device, requires_grad=True)
        t = torch.tensor([0.4, 0.8], device=device)
        time = (t, DiffusionExtraTimes(r=t / 2, batch_size=[2]))
        expected, _ = wrapper(x, time)
        compiled = torch.compile(wrapper, backend="aot_eager", fullgraph=True)
        actual, _ = compiled(x, time)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), x)[0],
            torch.autograd.grad(expected.sum(), x)[0],
        )


class TestDiagnostics:
    @pytest.mark.parametrize(
        "t,r,batch_size",
        (
            (torch.ones(2, 1), torch.ones(2), [2]),
            (torch.ones(2), torch.ones(2), []),
        ),
    )
    def test_invalid_time(self, t, r, batch_size):
        model = EDMPreconditioner(StructuredDiffusionModel())
        with pytest.raises(ValueError, match="shape|batch"):
            model(
                torch.ones(2, 3), (t, DiffusionExtraTimes(r=r, batch_size=batch_size))
            )


@pytest.mark.parametrize("preconditioner,kwargs", CONFIGS, ids=IDS)
class TestNonRegression:
    @pytest.mark.parametrize("fields", TIME_FIELDS, ids=TIME_IDS)
    def test_checkpoint(self, preconditioner, kwargs, tmp_path, fields):
        model = preconditioner(StructuredDiffusionModel(), **kwargs)
        x = torch.randn(2, 3)
        times = _make_time(torch.tensor([0.4, 0.8]), fields)
        expected, expected_extras = model(x, times)
        checkpoint = tmp_path / "model.pt"
        torch.save(model.state_dict(), checkpoint)
        restored = preconditioner(StructuredDiffusionModel(), **kwargs)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        actual, extras = restored(x, times)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(extras.logvar, expected_extras.logvar)
