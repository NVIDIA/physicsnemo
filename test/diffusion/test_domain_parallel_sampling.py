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

"""Tests for domain-parallel diffusion sampling utilities.

Non-distributed tests verify the no-op path of ``_maybe_replicate_timesteps``
and the plain-tensor path through ``sample()``.

Distributed tests (``@pytest.mark.multigpu_static``) verify that
``DomainParallelNoiseScheduler.timesteps`` and
``DomainParallelNoiseScheduler.init_latents`` produce correctly distributed
tensors, and that ``sample()`` auto-replicates timesteps when ``xN`` is a
``ShardTensor``.
"""

import pytest
import torch

from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
from physicsnemo.diffusion.samplers import sample
from physicsnemo.diffusion.samplers.samplers import _maybe_replicate_timesteps


def denoiser(x, t):
    return x / (1 + t.view(-1, *([1] * (x.ndim - 1))) ** 2)


# =====================================================================
# Non-distributed tests (plain tensors)
# =====================================================================


def test_maybe_replicate_timesteps_noop_plain_tensors():
    """_maybe_replicate_timesteps is a no-op when xN has no device_mesh."""
    t_steps = torch.linspace(80, 0, 19)
    xN = torch.randn(2, 3, 8, 8)
    result = _maybe_replicate_timesteps(t_steps, xN)
    assert result is t_steps


def test_sample_plain_tensors():
    """sample() works end-to-end with plain tensors (no mesh)."""
    scheduler = EDMNoiseScheduler()
    xN = torch.randn(2, 3, 8, 8) * 80
    x0 = sample(denoiser, xN, scheduler, num_steps=5, solver="euler")
    assert x0.shape == (2, 3, 8, 8)


# =====================================================================
# Distributed tests (require multigpu_static with >=2 GPUs)
# =====================================================================


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
def test_wrapper_timesteps_replicated(distributed_mesh):
    """DomainParallelNoiseScheduler.timesteps returns a replicated ShardTensor."""
    from physicsnemo.diffusion.noise_schedulers import DomainParallelNoiseScheduler

    scheduler = EDMNoiseScheduler()
    wrapper = DomainParallelNoiseScheduler(scheduler, distributed_mesh)

    t_steps = wrapper.timesteps(10, device="cuda")

    assert hasattr(t_steps, "device_mesh"), "timesteps should be a distributed tensor"
    assert t_steps.shape == (11,)
    full = t_steps.full_tensor()
    ref = scheduler.timesteps(10, device="cuda")
    assert torch.allclose(full, ref)


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
def test_wrapper_init_latents_sharded(distributed_mesh):
    """DomainParallelNoiseScheduler.init_latents returns a sharded tensor."""
    from physicsnemo.diffusion.noise_schedulers import DomainParallelNoiseScheduler

    scheduler = EDMNoiseScheduler()
    wrapper = DomainParallelNoiseScheduler(
        scheduler,
        distributed_mesh,
        shard_dim=2,
    )

    tN = torch.tensor([80.0, 80.0], device="cuda")
    xN = wrapper.init_latents((3, 16, 16), tN, device="cuda")

    assert hasattr(xN, "device_mesh"), "init_latents should be a distributed tensor"
    assert xN.shape == (2, 3, 16, 16)


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
def test_sample_auto_replicates_timesteps(distributed_mesh):
    """sample() auto-replicates plain timesteps when xN is a ShardTensor."""
    from physicsnemo.diffusion.noise_schedulers import DomainParallelNoiseScheduler

    scheduler = EDMNoiseScheduler()
    wrapper = DomainParallelNoiseScheduler(
        scheduler,
        distributed_mesh,
        shard_dim=2,
    )

    tN = torch.tensor([80.0, 80.0], device="cuda")
    xN = wrapper.init_latents((3, 16, 16), tN, device="cuda")

    x0 = sample(denoiser, xN, scheduler, num_steps=3, solver="euler")
    assert x0.shape == (2, 3, 16, 16)


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
def test_sample_with_wrapper_timesteps(distributed_mesh):
    """sample() works when both xN and time_steps come from the wrapper."""
    from physicsnemo.diffusion.noise_schedulers import DomainParallelNoiseScheduler

    scheduler = EDMNoiseScheduler()
    wrapper = DomainParallelNoiseScheduler(
        scheduler,
        distributed_mesh,
        shard_dim=2,
    )

    t_steps = wrapper.timesteps(5, device="cuda")
    tN = t_steps[0].expand(2)
    xN = wrapper.init_latents((3, 16, 16), tN, device="cuda")

    x0 = sample(
        denoiser,
        xN,
        scheduler,
        num_steps=5,
        solver="euler",
        time_steps=t_steps,
    )
    assert x0.shape == (2, 3, 16, 16)
