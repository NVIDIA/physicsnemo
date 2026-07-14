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

import pytest
import torch

from physicsnemo.nn import shrink_and_perturb_


class _ScalarParamNet(torch.nn.Module):
    """Module with a scalar (numel==1) parameter, e.g. a learnable temperature."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 8)
        self.scale = torch.nn.Parameter(torch.tensor(0.07))


def test_shrink_and_perturb_shrink_only(device):
    """perturb=0 shrinks every param exactly; returns the same module."""
    model = torch.nn.Linear(8, 8).to(device)
    orig = {n: p.detach().clone() for n, p in model.named_parameters()}

    ret = shrink_and_perturb_(model, shrink=0.4, perturb=0.0)

    assert ret is model
    for n, p in model.named_parameters():
        assert torch.allclose(p, 0.4 * orig[n])


def test_shrink_and_perturb_noop(device):
    """shrink=1, perturb=0 leaves every param untouched."""
    model = torch.nn.Linear(8, 8).to(device)
    orig = {n: p.detach().clone() for n, p in model.named_parameters()}

    shrink_and_perturb_(model, shrink=1.0, perturb=0.0)

    for n, p in model.named_parameters():
        assert torch.equal(p, orig[n])


def test_shrink_and_perturb_callable_noise_exact(device):
    """Callable noise reproduces theta <- shrink*theta + perturb*eps exactly."""
    model = torch.nn.Linear(4, 4).to(device)
    orig = {n: p.detach().clone() for n, p in model.named_parameters()}
    eps = {n: torch.ones_like(p) for n, p in model.named_parameters()}

    shrink_and_perturb_(
        model, shrink=0.5, perturb=0.2, noise=lambda p: torch.ones_like(p)
    )

    for n, p in model.named_parameters():
        assert torch.allclose(p, 0.5 * orig[n] + 0.2 * eps[n])


def test_scaled_normal_magnitude(device):
    """scaled_normal noise std tracks perturb * std(theta) per tensor."""
    model = torch.nn.Linear(256, 256).to(device)
    with torch.no_grad():
        model.weight.normal_(0.0, 3.0)
    w_std = model.weight.std().item()

    gen = torch.Generator(device=device).manual_seed(0)
    # shrink=0 isolates the noise term: result == perturb * std(theta) * z.
    shrink_and_perturb_(
        model,
        shrink=0.0,
        perturb=0.1,
        noise="scaled_normal",
        include=lambda n, p: n == "weight",
        generator=gen,
    )

    expected = 0.1 * w_std
    assert abs(model.weight.std().item() - expected) / expected < 0.1


@pytest.mark.parametrize("noise", ["scaled_normal", "normal"])
def test_gaussian_noise_reproducible(device, noise):
    """Same generator seed -> identical result; different seed -> different."""

    def run(seed):
        torch.manual_seed(0)  # identical init across runs
        model = torch.nn.Linear(8, 8).to(device)
        gen = torch.Generator(device=device).manual_seed(seed)
        shrink_and_perturb_(model, noise=noise, generator=gen)
        return {n: p.detach().clone() for n, p in model.named_parameters()}

    a, b, c = run(123), run(123), run(999)
    for n in a:
        assert torch.equal(a[n], b[n])
        assert not torch.equal(a[n], c[n])


def test_include_leaves_excluded_unchanged(device):
    """Excluded params are bit-identical (not even shrunk); included change."""
    model = torch.nn.Linear(8, 8).to(device)
    w0 = model.weight.detach().clone()
    b0 = model.bias.detach().clone()

    shrink_and_perturb_(
        model, shrink=0.5, perturb=0.1, include=lambda n, p: n == "weight"
    )

    assert torch.equal(model.bias, b0)
    assert not torch.equal(model.weight, w0)


def test_buffers_untouched(device):
    """Buffers (batch-norm running stats) are never modified."""
    model = torch.nn.BatchNorm1d(8).to(device)
    with torch.no_grad():
        model.running_mean.fill_(2.0)
        model.running_var.fill_(3.0)
    rm = model.running_mean.detach().clone()
    rv = model.running_var.detach().clone()

    shrink_and_perturb_(model, shrink=0.5, perturb=0.1)

    assert torch.equal(model.running_mean, rm)
    assert torch.equal(model.running_var, rv)
    # weight (all ones -> std 0 -> no noise) is still shrunk.
    assert torch.allclose(model.weight, torch.full_like(model.weight, 0.5))


def test_scalar_param_shrink_only(device):
    """A scalar (numel==1) param is shrunk without NaN under scaled_normal."""
    model = _ScalarParamNet().to(device)
    scale0 = model.scale.detach().clone()

    shrink_and_perturb_(model, shrink=0.5, perturb=0.1)  # default scaled_normal

    for p in model.parameters():
        assert torch.isfinite(p).all()
    # scalar has no defined spread -> shrink only, no noise.
    assert torch.allclose(model.scale, 0.5 * scale0)


def test_callable_wrong_shape_raises(device):
    """A noise callable returning a mismatched shape is rejected, not broadcast."""
    model = torch.nn.Linear(4, 4).to(device)
    with pytest.raises(ValueError):
        shrink_and_perturb_(model, noise=lambda p: torch.tensor(1.0, device=p.device))


def test_invalid_arguments(device):
    model = torch.nn.Linear(4, 4).to(device)
    with pytest.raises(ValueError):
        shrink_and_perturb_(model, shrink=-0.1)
    with pytest.raises(ValueError):
        shrink_and_perturb_(model, perturb=-1.0)
    with pytest.raises(ValueError):
        shrink_and_perturb_(model, noise="bogus")
