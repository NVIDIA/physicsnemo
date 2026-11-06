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

r"""Keep custom losses and public imports independent of optional backends."""

import subprocess
import sys

import torch

from physicsnemo.diffusion.distillation import BaseDistillationLoss


class TestConsistency:
    def test_native_loss(self):
        def native_loss(x0, *, condition=None, **kwargs):
            return {"total_loss": x0.square().mean()}

        assert isinstance(native_loss, BaseDistillationLoss)
        x = torch.randn(2, 3, requires_grad=True)
        (gradient,) = torch.autograd.grad(native_loss(x)["total_loss"], x)
        torch.testing.assert_close(gradient, 2 * x / x.numel())

    def test_optional_imports(self):
        program = """
import importlib.metadata
import sys
import warnings
import torch

original = importlib.metadata.version
def version(name):
    if name in ("fastgen", "nvalchemi"):
        raise importlib.metadata.PackageNotFoundError(name)
    return original(name)
importlib.metadata.version = version

with warnings.catch_warnings(record=True) as emitted:
    warnings.simplefilter("always")
    from physicsnemo.diffusion.distillation import BaseDistillationLoss, DistillationLoss
assert any("alpha" in str(item.message) for item in emitted)
assert not any(name.startswith(("fastgen", "nvalchemi")) for name in sys.modules)

from physicsnemo.diffusion.noise_schedulers import RectifiedFlowNoiseScheduler
try:
    DistillationLoss(torch.nn.Linear(3, 3), RectifiedFlowNoiseScheduler(), method="MeanFlow")
except ImportError as error:
    assert "FastGen" in str(error)
else:
    raise AssertionError("Missing optional backend did not raise")
"""
        subprocess.run(  # noqa: S603
            [sys.executable, "-c", program], check=True, capture_output=True, text=True
        )
