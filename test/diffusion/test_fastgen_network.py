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

r"""Check private network translation without changing model ownership."""

import pytest
import torch

pytest.importorskip("fastgen")

from jaxtyping import Float, Shaped
from tensordict import TensorDict

from physicsnemo.core import Module
from physicsnemo.diffusion import (
    DiffusionExtraOutputs,
    DiffusionExtraTimes,
)
from physicsnemo.diffusion._fastgen import _FastGenNetwork
from physicsnemo.diffusion.noise_schedulers import RectifiedFlowNoiseScheduler
from physicsnemo.diffusion.noise_schedulers._fastgen import _FastGenNoiseScheduler


class StructuredDiffusionModel(Module):
    r"""Linear model with optional destination times and learned loss weights."""

    def __init__(self, output: str = "tuple", logvar: float | None = None):
        super().__init__()
        self.output = output
        self.gain = torch.nn.Parameter(torch.tensor(0.75))
        self.logvar = (
            None if logvar is None else torch.nn.Parameter(torch.tensor(logvar))
        )

    def forward(
        self,
        x: Float[torch.Tensor, " B *dims"],
        t: Float[torch.Tensor, " B"]
        | tuple[Float[torch.Tensor, " B"], Shaped[DiffusionExtraTimes, " B"]],
        condition: TensorDict | None = None,
        *,
        offset: float = 0.0,
    ) -> (
        Float[torch.Tensor, " B *dims"]
        | tuple[Float[torch.Tensor, " B *dims"], Shaped[DiffusionExtraOutputs, " B"]]
    ):
        r"""Return a prediction, optionally paired with its loss weight."""
        t, extra_times = t if isinstance(t, tuple) else (t, None)
        t_bc = t.reshape(-1, *([1] * (x.ndim - 1)))  # (B, 1, ...)
        prediction = self.gain * x + t_bc + offset
        if extra_times is not None:
            for time in extra_times.to_tensordict().values():
                prediction = prediction + time.reshape(-1, *([1] * (x.ndim - 1)))
        if condition is not None:
            prediction = prediction + condition["shift"]
        if self.output == "tensor":
            return prediction
        if self.output == "empty":
            return prediction, DiffusionExtraOutputs(batch_size=[x.shape[0]])
        return prediction, DiffusionExtraOutputs(
            logvar=(
                self.gain * t if self.logvar is None else self.logvar.expand(x.shape[0])
            ),
            batch_size=[x.shape[0]],
        )


CONFIGS = (
    (False, "x0"),
    (True, "epsilon"),
    (True, "flow"),
    (False, "score"),
)
IDS = ("x0", "epsilon", "flow", "score")


@pytest.mark.parametrize("with_destination,prediction_type", CONFIGS, ids=IDS)
class TestConsistency:
    def test_forward(self, with_destination, prediction_type):
        source = RectifiedFlowNoiseScheduler()
        model = StructuredDiffusionModel()
        adapter = _FastGenNetwork(
            model,
            _FastGenNoiseScheduler(source),
            prediction_type=prediction_type,
        )
        x = torch.randn(2, 3, 5)
        t = torch.tensor([0.2, 0.7])
        r = t / 2 if with_destination else None
        time = (t, DiffusionExtraTimes(r=r, batch_size=[2])) if r is not None else t
        raw = model(x, time)
        prediction, logvar = adapter(x, t, r=r, return_logvar=True)
        expected = raw[0]
        if prediction_type == "score":
            expected = source.score_to_epsilon(expected, t)
        torch.testing.assert_close(prediction, expected)
        torch.testing.assert_close(logvar[:, 0], raw[1].logvar)

    def test_reset(self, with_destination, prediction_type):
        model = StructuredDiffusionModel()
        schedule = _FastGenNoiseScheduler(RectifiedFlowNoiseScheduler())
        adapter = _FastGenNetwork(model, schedule, prediction_type=prediction_type)
        parameters = list(model.parameters())
        assert list(adapter.parameters()) == parameters
        before = [p.detach().clone() for p in parameters]
        adapter.reset_parameters()
        assert adapter.noise_scheduler is schedule
        for old, current in zip(before, model.parameters()):
            torch.testing.assert_close(old, current)


@pytest.mark.parametrize("with_destination,prediction_type", CONFIGS, ids=IDS)
class TestGradientFlow:
    def test_inputs(self, with_destination, prediction_type):
        model = StructuredDiffusionModel().double().requires_grad_(False)
        adapter = _FastGenNetwork(
            model,
            _FastGenNoiseScheduler(RectifiedFlowNoiseScheduler()),
            prediction_type=prediction_type,
        )
        x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        t = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda x, t: adapter(
                x, t, r=t / 2 if with_destination else None, fwd_pred_type="x0"
            ),
            (x, t),
        )


@pytest.mark.parametrize("with_destination,prediction_type", CONFIGS, ids=IDS)
class TestCompile:
    def test_forward(self, with_destination, prediction_type):
        model = StructuredDiffusionModel()
        adapter = _FastGenNetwork(
            model,
            _FastGenNoiseScheduler(RectifiedFlowNoiseScheduler()),
            prediction_type=prediction_type,
        )
        x = torch.randn(2, 3, requires_grad=True)
        t = torch.tensor([0.2, 0.7], requires_grad=True)
        r = t.detach().clone().requires_grad_() if with_destination else None
        compiled = torch.compile(adapter, fullgraph=True, backend="aot_eager")
        expected = adapter(x, t, r=r)
        actual = compiled(x, t, r=r)
        with torch._dynamo.config.patch(error_on_recompile=True):
            torch.testing.assert_close(compiled(x, t, r=r), expected)
        torch.testing.assert_close(actual, expected)
        inputs = (x, t) if r is None else (x, t, r)
        for a, b in zip(
            torch.autograd.grad(actual.sum(), inputs),
            torch.autograd.grad(expected.sum(), inputs),
        ):
            torch.testing.assert_close(a, b)


class TestDiagnostics:
    def test_average_flow(self):
        adapter = _FastGenNetwork(
            StructuredDiffusionModel(),
            _FastGenNoiseScheduler(RectifiedFlowNoiseScheduler()),
            prediction_type="flow",
            prediction_semantics="average",
        )
        with pytest.raises(ValueError, match="average"):
            adapter(
                torch.ones(2, 3), torch.ones(2), r=torch.zeros(2), fwd_pred_type="x0"
            )

    @pytest.mark.parametrize("output", ("tensor", "empty"))
    def test_required_logvar(self, output):
        adapter = _FastGenNetwork(
            StructuredDiffusionModel(output),
            _FastGenNoiseScheduler(RectifiedFlowNoiseScheduler()),
            prediction_type="x0",
        )
        with pytest.raises(ValueError, match="logvar"):
            adapter(torch.ones(2, 3), torch.ones(2), return_logvar=True)
