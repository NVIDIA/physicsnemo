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

"""Multi-GPU tests for EDMPreconditioner with FSDP and ShardTensor.

Tests verify the interaction between:

- ``EDMPreconditioner`` with scalar and per-channel ``sigma_data``
- FSDP wrapping (which converts registered buffers to replicated DTensors)
- ``ShardTensor`` inputs for domain-parallel spatial sharding
- ``MSEDSMLoss`` with ``DomainParallelNoiseScheduler``
- ``sample()`` with domain-parallel scheduler

The distribution patterns follow the following expected pattern:

1. A 2-D ``(ddp, domain)`` device mesh is created.
2. The model is wrapped with FSDP (``NO_SHARD`` strategy) on the **ddp**
   sub-mesh, which converts registered buffers (e.g. ``sigma_data``) to
   replicated DTensors on that sub-mesh.
3. Spatial data is distributed as ``ShardTensor`` on the **domain** sub-mesh
   (sharded along height).
4. The noise scheduler is wrapped with ``DomainParallelNoiseScheduler`` on
   the **domain** sub-mesh to broadcast sampled times, shard initial
   latents, and promote alpha/sigma coefficients for ``add_noise``
   compatibility.
5. ``_ensure_plain_tensor`` + ``_replicate_on_mesh`` in the preconditioner
   unwrap FSDP DTensor coefficients (living on the ddp mesh) and
   re-promote them to replicated DTensors on the data mesh (domain)
   for type-compatible arithmetic.

Two mesh topologies are tested:

- **FSDP-only** — ``(world_size, 1)``: all ranks are data-parallel, no
  domain parallelism.  Exercises FSDP DTensor unwrapping with plain tensor
  inputs.
- **FSDP + ShardTensor** — ``(1, world_size)`` (pure domain-parallel) and
  ``(world_size/2, 2)`` (combined ddp + domain, requires >= 4 GPUs).
  Exercises the full unwrap/re-promote path across distinct mesh
  dimensions.

Distributed tests require ``@pytest.mark.multigpu_static`` (>= 2 GPUs).
"""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.core import Module
from physicsnemo.diffusion.metrics.losses import MSEDSMLoss
from physicsnemo.diffusion.noise_schedulers import (
    DomainParallelNoiseScheduler,
    EDMNoiseScheduler,
)
from physicsnemo.diffusion.preconditioners import EDMPreconditioner
from physicsnemo.diffusion.samplers import sample
from physicsnemo.domain_parallel.shard_tensor import scatter_tensor

# =====================================================================
# Test model and constants
# =====================================================================

_C, _H, _W = 3, 16, 16
_B = 2
_SHARD_DIM = 2


class _SimpleConvModel(Module):
    """Minimal conv model for distributed preconditioner testing.

    Uses ``kernel_size=1`` to avoid halo-exchange requirements when the
    input is sharded along a spatial dimension.
    """

    def __init__(self, channels: int = _C):
        super().__init__()
        self.channels = channels
        self.net = torch.nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, t, condition=None, **kwargs):
        out = self.net(x)
        t_bc = t.view(-1, *([1] * (x.ndim - 1)))
        return out + t_bc


# =====================================================================
# Helpers
# =====================================================================


def _make_model_deterministic(channels=_C, seed=42):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    model = _SimpleConvModel(channels=channels)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype))
    return model


def _make_preconditioner(sigma_data, seed=42):
    model = _make_model_deterministic(channels=_C, seed=seed)
    return EDMPreconditioner(model, sigma_data=sigma_data)


def _make_2d_mesh(ddp_size, domain_size):
    """Create a 2-D ``(ddp, domain)`` device mesh"""
    return dist.init_device_mesh(
        "cuda", (ddp_size, domain_size), mesh_dim_names=("ddp", "domain")
    )


def _wrap_fsdp(module, ddp_mesh):
    """Wrap *module* with FSDP ``NO_SHARD`` on *ddp_mesh*"""
    return FSDP(
        module,
        device_mesh=ddp_mesh,
        use_orig_params=False,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        sync_module_states=True,
    )


def _scatter(tensor, domain_mesh, shard_dim=_SHARD_DIM):
    """Scatter *tensor* from rank 0 onto *domain_mesh* as a ``ShardTensor``."""
    src = dist.get_global_rank(domain_mesh.get_group(), 0)
    placement = (
        Shard(shard_dim)
        if tensor.ndim >= 3 and tensor.shape[shard_dim] > 1
        else Replicate()
    )
    return scatter_tensor(
        tensor,
        src,
        domain_mesh,
        placements=(placement,),
        global_shape=tensor.shape,
        dtype=tensor.dtype,
    )


def _make_inputs(device="cuda", seed=42):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x = torch.randn(_B, _C, _H, _W, generator=gen).to(device)
    t = (torch.rand(_B, generator=gen) * 0.5 + 0.4).to(device)
    return x, t


def _make_dp_scheduler(sigma_data, domain_mesh):
    scheduler = EDMNoiseScheduler(sigma_data=sigma_data)
    return DomainParallelNoiseScheduler(scheduler, domain_mesh, shard_dim=_SHARD_DIM)


def _dp_mesh_from_config(config_name):
    """Build a 2-D mesh for the requested domain-parallel configuration.

    ``"domain_only"`` puts all ranks in the domain dimension (ddp=1).
    ``"ddp_and_domain"`` splits ranks evenly between ddp and domain
    (requires >= 4 GPUs with world_size divisible by 2).
    """
    ws = dist.get_world_size()
    if config_name == "domain_only":
        return _make_2d_mesh(1, ws)
    if ws < 4 or ws % 2 != 0:
        pytest.skip(f"Combined ddp+domain needs >= 4 GPUs divisible by 2 (have {ws})")
    return _make_2d_mesh(ws // 2, 2)


# Reusable parametrize decorators
_sigma_data_params = pytest.mark.parametrize(
    "sigma_data",
    [0.5, [0.3, 0.5, 0.7]],
    ids=["scalar_sigma", "per_channel_sigma"],
)

_dp_configs = pytest.mark.parametrize(
    "dp_config",
    ["domain_only", "ddp_and_domain"],
    ids=["domain_only", "ddp_and_domain"],
)


# =====================================================================
# FSDP-only (no ShardTensor) — mesh (world_size, 1)
#
# All ranks are data-parallel; no domain parallelism.  FSDP converts
# registered buffers (sigma_data) to DTensors on the ddp sub-mesh.
# The preconditioner must unwrap those DTensors back to plain tensors
# via _ensure_plain_tensor before element-wise arithmetic with plain
# tensor inputs.
# =====================================================================


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
@_sigma_data_params
def test_fsdp_only_forward(sigma_data):
    """FSDP-wrapped EDMPreconditioner with plain tensors matches non-distributed reference."""
    mesh = _make_2d_mesh(dist.get_world_size(), 1)

    precond_ref = _make_preconditioner(sigma_data).cuda()
    x, t = _make_inputs()
    with torch.no_grad():
        ref_out = precond_ref(x, t)

    precond = _make_preconditioner(sigma_data).cuda()
    precond = _wrap_fsdp(precond, mesh["ddp"])

    with torch.no_grad():
        out = precond(x, t)

    torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
@_sigma_data_params
def test_fsdp_only_gradient_flow(sigma_data):
    """Gradients flow through FSDP-wrapped EDMPreconditioner with plain tensors."""
    mesh = _make_2d_mesh(dist.get_world_size(), 1)
    precond = _make_preconditioner(sigma_data).cuda()
    precond = _wrap_fsdp(precond, mesh["ddp"])

    x, t = _make_inputs()
    out = precond(x, t)
    out.sum().backward()

    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in precond.parameters()
    )
    assert has_grad


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
@_sigma_data_params
def test_fsdp_only_loss(sigma_data):
    """MSEDSMLoss with FSDP-wrapped preconditioner and plain scheduler (no ShardTensor)."""
    mesh = _make_2d_mesh(dist.get_world_size(), 1)
    precond = _make_preconditioner(sigma_data).cuda()
    precond = _wrap_fsdp(precond, mesh["ddp"])
    scheduler = EDMNoiseScheduler(sigma_data=sigma_data)

    loss_fn = MSEDSMLoss(precond, scheduler)
    x0, _ = _make_inputs()
    loss = loss_fn(x0)

    assert loss.shape == ()
    assert torch.isfinite(loss)


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
@_sigma_data_params
def test_fsdp_only_loss_backward(sigma_data):
    """Gradients flow through MSEDSMLoss with FSDP-wrapped preconditioner (no ShardTensor)."""
    mesh = _make_2d_mesh(dist.get_world_size(), 1)
    precond = _make_preconditioner(sigma_data).cuda()
    precond = _wrap_fsdp(precond, mesh["ddp"])
    scheduler = EDMNoiseScheduler(sigma_data=sigma_data)

    loss_fn = MSEDSMLoss(precond, scheduler)
    x0, _ = _make_inputs()
    loss = loss_fn(x0)
    loss.backward()

    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in precond.parameters()
    )
    assert has_grad


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
@_sigma_data_params
def test_fsdp_only_training_step(sigma_data):
    """End-to-end training step with FSDP (no domain parallelism)."""
    mesh = _make_2d_mesh(dist.get_world_size(), 1)
    precond = _make_preconditioner(sigma_data).cuda()
    precond = _wrap_fsdp(precond, mesh["ddp"])
    scheduler = EDMNoiseScheduler(sigma_data=sigma_data)

    loss_fn = MSEDSMLoss(precond, scheduler)
    optimizer = torch.optim.Adam(precond.parameters(), lr=1e-3)

    x0, _ = _make_inputs()
    initial_params = [p.clone() for p in precond.parameters()]

    optimizer.zero_grad()
    loss = loss_fn(x0)
    loss.backward()
    optimizer.step()

    params_changed = any(
        not torch.equal(p_old, p_new)
        for p_old, p_new in zip(initial_params, precond.parameters())
    )
    assert params_changed, "Parameters were not updated after optimizer step"


# =====================================================================
# FSDP + ShardTensor — 2-D mesh (ddp, domain)
#
# The model is FSDP-wrapped on mesh["ddp"], data is scattered as
# ShardTensor on mesh["domain"], and the noise scheduler is wrapped
# with DomainParallelNoiseScheduler on mesh["domain"].
#
# This exercises the full _ensure_plain_tensor + _replicate_on_mesh
# path: FSDP creates DTensor coefficients on the ddp sub-mesh, which
# must be unwrapped and re-promoted to Replicate DTensors on the
# domain sub-mesh for type-compatible arithmetic with ShardTensor data.
#
# Two mesh configurations are parametrized:
# - domain_only: (1, world_size) — pure domain parallelism (>= 2 GPUs)
# - ddp_and_domain: (world_size/2, 2) — combined (>= 4 GPUs)
# =====================================================================


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
@_sigma_data_params
@_dp_configs
def test_dp_preconditioner_forward(sigma_data, dp_config):
    """FSDP + ShardTensor forward matches non-distributed reference."""
    mesh = _dp_mesh_from_config(dp_config)

    precond_ref = _make_preconditioner(sigma_data).cuda()
    x, t = _make_inputs()
    with torch.no_grad():
        ref_out = precond_ref(x, t)

    precond = _make_preconditioner(sigma_data).cuda()
    precond = _wrap_fsdp(precond, mesh["ddp"])
    x_shard = _scatter(x, mesh["domain"])

    with torch.no_grad():
        out = precond(x_shard, t)

    assert out.shape == (_B, _C, _H, _W)
    full_out = out.full_tensor()
    torch.testing.assert_close(full_out, ref_out, atol=1e-5, rtol=1e-5)


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
@_sigma_data_params
@_dp_configs
def test_dp_preconditioner_gradient_flow(sigma_data, dp_config):
    """Gradients flow through FSDP + ShardTensor preconditioner."""
    mesh = _dp_mesh_from_config(dp_config)
    precond = _make_preconditioner(sigma_data).cuda()
    precond = _wrap_fsdp(precond, mesh["ddp"])

    x, t = _make_inputs()
    x_shard = _scatter(x, mesh["domain"])

    out = precond(x_shard, t)
    out.sum().backward()

    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in precond.parameters()
    )
    assert has_grad


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
@_sigma_data_params
@_dp_configs
def test_dp_preconditioner_loss(sigma_data, dp_config):
    """MSEDSMLoss with FSDP on ddp mesh + domain-parallel scheduler on domain mesh."""
    mesh = _dp_mesh_from_config(dp_config)
    precond = _make_preconditioner(sigma_data).cuda()
    precond = _wrap_fsdp(precond, mesh["ddp"])
    dp_scheduler = _make_dp_scheduler(sigma_data, mesh["domain"])

    loss_fn = MSEDSMLoss(precond, dp_scheduler)

    x0, _ = _make_inputs()
    x0_shard = _scatter(x0, mesh["domain"])

    loss = loss_fn(x0_shard)

    assert loss.shape == ()
    assert torch.isfinite(loss)


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
@_sigma_data_params
@_dp_configs
def test_dp_preconditioner_loss_backward(sigma_data, dp_config):
    """Gradients flow through MSEDSMLoss with FSDP + domain-parallel scheduler."""
    mesh = _dp_mesh_from_config(dp_config)
    precond = _make_preconditioner(sigma_data).cuda()
    precond = _wrap_fsdp(precond, mesh["ddp"])
    dp_scheduler = _make_dp_scheduler(sigma_data, mesh["domain"])

    loss_fn = MSEDSMLoss(precond, dp_scheduler)

    x0, _ = _make_inputs()
    x0_shard = _scatter(x0, mesh["domain"])

    loss = loss_fn(x0_shard)
    loss.backward()

    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in precond.parameters()
    )
    assert has_grad


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
@_sigma_data_params
@_dp_configs
def test_dp_preconditioner_sampling(sigma_data, dp_config):
    """sample() with FSDP on ddp mesh + domain-parallel scheduler on domain mesh."""
    mesh = _dp_mesh_from_config(dp_config)
    precond = _make_preconditioner(sigma_data).cuda()
    precond = _wrap_fsdp(precond, mesh["ddp"])
    dp_scheduler = _make_dp_scheduler(sigma_data, mesh["domain"])

    denoiser = dp_scheduler.get_denoiser(x0_predictor=precond)

    tN = torch.tensor([80.0] * _B, device="cuda")
    xN = dp_scheduler.init_latents((_C, _H, _W), tN, device="cuda")

    with torch.no_grad():
        x0 = sample(denoiser, xN, dp_scheduler, num_steps=3, solver="euler")

    assert x0.shape == (_B, _C, _H, _W)
    full_x0 = x0.full_tensor()
    assert torch.isfinite(full_x0).all()


# =====================================================================
# Noise scheduler: add_noise + loss_weight through domain-parallel wrapper
# =====================================================================


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
@_sigma_data_params
@_dp_configs
def test_dp_scheduler_add_noise(sigma_data, dp_config):
    """EDMNoiseScheduler.add_noise works with ShardTensor clean data."""
    mesh = _dp_mesh_from_config(dp_config)
    dp_scheduler = _make_dp_scheduler(sigma_data, mesh["domain"])

    x0, _ = _make_inputs()
    x0_shard = _scatter(x0, mesh["domain"])

    t = dp_scheduler.sample_time(_B, device="cuda")
    x_t = dp_scheduler.add_noise(x0_shard, t)

    assert x_t.shape == (_B, _C, _H, _W)


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
@_sigma_data_params
@_dp_configs
def test_dp_scheduler_loss_weight(sigma_data, dp_config):
    """loss_weight through domain-parallel wrapper matches inner scheduler."""
    mesh = _dp_mesh_from_config(dp_config)
    scheduler = EDMNoiseScheduler(sigma_data=sigma_data)
    dp_scheduler = DomainParallelNoiseScheduler(
        scheduler, mesh["domain"], shard_dim=_SHARD_DIM
    )

    t = dp_scheduler.sample_time(_B, device="cuda")
    w = dp_scheduler.loss_weight(t)
    w_ref = scheduler.loss_weight(t)

    torch.testing.assert_close(w, w_ref)

    if isinstance(sigma_data, list):
        assert w.shape == (_B, _C)
    else:
        assert w.shape == (_B,)


# =====================================================================
# Full training step: end-to-end with FSDP, ShardTensor, and loss
# =====================================================================


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
@_sigma_data_params
@_dp_configs
def test_dp_full_training_step(sigma_data, dp_config):
    """End-to-end training step: forward + loss + backward + optimizer step.

    Mirrors a typical training loop: FSDP model on the ddp sub-mesh,
    domain-parallel scheduler on the domain sub-mesh, ShardTensor data,
    and an optimizer step.
    """
    mesh = _dp_mesh_from_config(dp_config)
    precond = _make_preconditioner(sigma_data).cuda()
    precond = _wrap_fsdp(precond, mesh["ddp"])
    dp_scheduler = _make_dp_scheduler(sigma_data, mesh["domain"])
    loss_fn = MSEDSMLoss(precond, dp_scheduler)
    optimizer = torch.optim.Adam(precond.parameters(), lr=1e-3)

    x0, _ = _make_inputs()
    x0_shard = _scatter(x0, mesh["domain"])

    initial_params = [p.clone() for p in precond.parameters()]

    optimizer.zero_grad()
    loss = loss_fn(x0_shard)
    loss.backward()
    optimizer.step()

    params_changed = any(
        not torch.equal(p_old, p_new)
        for p_old, p_new in zip(initial_params, precond.parameters())
    )
    assert params_changed, "Parameters were not updated after optimizer step"
