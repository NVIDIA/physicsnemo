# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Endpoint regularizer: physical units, piecewise gradients and causal allowance."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from penetration_loss import penetration_weight, rollout_penetration_loss

from physicsnemo.nn.functional.neighbors.surface_penetration import (
    surface_penetration_loss,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def scene(device="cpu", quad=False):
    x = torch.tensor(
        [[0.0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0], [1.3, 0.7, 0.4]],
        dtype=torch.float64,
        device=device,
    )
    f = torch.tensor([[0, 1, 2, 3] if quad else [0, 1, 2, 2]], device=device)
    h = torch.ones(5, dtype=x.dtype, device=device)
    return x, f, h


def loss(x, f, h, **kwargs):
    return surface_penetration_loss(x, f, h, implementation="torch", **kwargs)


@pytest.mark.parametrize("quad", [False, True])
def test_exact_value_query_and_face_gradients(device, quad):
    x, f, h = scene(device, quad)
    x.requires_grad_()
    r = loss(x, f, h, tolerance=0.1)
    torch.testing.assert_close(r.loss, x.new_tensor(0.5**2 / 5))
    assert r.active_nodes == 1
    r.loss.backward()
    assert x.grad[4, 2] < 0 and x.grad[:4, 2].sum() > 0
    torch.testing.assert_close(x.grad.sum(0), x.new_zeros(3), atol=1e-12, rtol=0)
    assert torch.autograd.gradcheck(
        lambda z: loss(z, f, h, tolerance=0.1).loss, (x,), atol=1e-5
    )


def test_initial_overlap_allowance_is_pair_specific_and_frozen(device):
    x, f, h = scene(device)
    initial = x.clone().requires_grad_()
    r = loss(x.requires_grad_(), f, h, reference_positions=initial, tolerance=0.05)
    assert r.loss == 0
    r.loss.backward()
    assert initial.grad is None and x.grad.abs().sum() == 0
    x = x.detach().clone()
    x[4, 2] = 0.2
    r = loss(x, f, h, reference_positions=initial, tolerance=0.05)
    torch.testing.assert_close(r.loss, x.new_tensor(0.15**2 / 5))
    x[4, 2] = 0.8
    assert loss(x, f, h, reference_positions=initial, tolerance=0.05).loss == 0
    # An initially separated pair gets no allowance from some OTHER face.
    initial = initial.detach().clone()
    initial[4, 2] = 2
    x[4, 2] = 0.2
    torch.testing.assert_close(
        loss(x, f, h, reference_positions=initial, tolerance=0.05).loss,
        x.new_tensor(0.75**2 / 5),
    )


@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
def test_units_translation_and_orientation(device, scale):
    x, f, h = scene(device, True)
    expected = loss(x, f, h, tolerance=0.1).loss
    actual = loss(
        x * scale + x.new_tensor([12.0, -5.0, 100.0]) * scale,
        f.flip(1),
        h * scale,
        tolerance=0.1 * scale,
        length_scale=scale,
    ).loss
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)


def test_incidence_exclusions_batch_and_empty(device):
    x, f, h = scene(device)
    for kwargs in (
        dict(excluded_pairs=torch.tensor([[4], [0]], device=device)),
        dict(batch=torch.tensor([0, 0, 0, 0, 1], device=device)),
    ):
        assert loss(x, f, h, **kwargs).loss == 0
    r = loss(x.requires_grad_(), f[:0], h)
    r.loss.backward()
    assert x.grad.abs().sum() == 0
    assert loss(x[:0], f[:0], h[:0]).loss == 0
    # All nodes on a face are incident and must be excluded.
    assert loss(x[:3], f, h[:3]).loss == 0


def test_pagination_duplicates_and_max_reduction(device):
    x, f, h = scene(device)
    x.requires_grad_()
    expected = loss(x, f, h).loss
    gradient = torch.autograd.grad(expected, x)[0]
    for page in (1, 2, 65536):
        r = loss(x, f.repeat(5, 1), h, pair_chunk_size=page)
        torch.testing.assert_close(r.loss, expected)
        torch.testing.assert_close(torch.autograd.grad(r.loss, x)[0], gradient)


def test_degenerate_face_finite_and_two_sided(device):
    x, f, h = scene(device)
    positive = loss(x, f, h).loss
    x[4, 2] *= -1
    torch.testing.assert_close(loss(x, f, h).loss, positive)
    x[:4] = 0
    x[4] = x.new_tensor([0.1, 0, 0.3])
    x.requires_grad_()
    r = loss(x, f, h)
    r.loss.backward()
    assert torch.isfinite(r.loss) and torch.isfinite(x.grad).all()


def test_tunneling_is_not_hidden_by_claiming_endpoint_ccd(device):
    x, f, h = scene(device)
    x[4, 2] = 2.0
    initial = x.clone()
    x[4, 2] = -2.0
    assert loss(x, f, h, reference_positions=initial).loss == 0
    # A full crossing can have zero endpoint overlap. CCD must remain separate.


def test_bptt_four_steps_and_denormalization(device):
    x, f, h = scene(device)
    x = x.float()
    h = h.float()
    initial = x.clone()
    initial[4, 2] = 2.0
    mean = x.new_tensor([200.0, -300.0, 50.0])
    std = x.new_tensor([100.0, 200.0, 50.0])
    velocity = x.new_zeros(x.shape)
    velocity[4, 2] = -0.05
    velocity.requires_grad_()
    z = (x - mean) / std
    values = []
    for _ in range(4):
        z = z + velocity / std
        values.append(z)
    trajectory = torch.stack(values, 1)
    trajectory.retain_grad()
    graph = SimpleNamespace(
        contact_faces=f, shell_thickness=h, penetration_reference_positions=initial
    )
    cfg = dict(tolerance_mm=0.1, length_scale_mm=1.0, implementation="torch")
    value, diag = rollout_penetration_loss(
        trajectory, graph, dict(node=dict(pos_mean=mean, pos_std=std)), cfg
    )
    expected = torch.stack(
        [
            loss(
                x + (i + 1) * velocity, f, h, reference_positions=initial, tolerance=0.1
            ).loss
            for i in range(4)
        ]
    ).mean()
    torch.testing.assert_close(value, expected, atol=2e-6, rtol=1e-4)
    value.backward()
    assert (trajectory.grad[4, :, 2].abs() > 0).all() and velocity.grad[4, 2] < 0
    assert diag["penetration_active_nodes"] == 1


def test_weight_schedule_and_invalid_inputs():
    cfg = dict(max_weight=0.01, warmup_epochs=50, ramp_epochs=50)
    assert penetration_weight(49, **cfg) == 0
    assert penetration_weight(50, **cfg) == pytest.approx(0.0002)
    assert penetration_weight(99, **cfg) == pytest.approx(0.01)
    assert penetration_weight(499, **cfg) == pytest.approx(0.01)
    for bad in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            penetration_weight(0, **dict(cfg, max_weight=bad))
    x, f, h = scene()
    for kwargs in (
        dict(tolerance=-1),
        dict(length_scale=0),
        dict(tolerance=float("nan")),
    ):
        with pytest.raises(ValueError):
            loss(x, f, h, **kwargs)


@pytest.mark.parametrize("quad", [False, True])
def test_warp_torch_parity_and_deterministic_backward(quad):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for backend parity")
    x, f, h = scene("cuda", quad)
    x.requires_grad_()
    prior = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        expected = loss(x, f, h, tolerance=0.1)
        actual = surface_penetration_loss(x, f, h, tolerance=0.1, pair_chunk_size=1)
        torch.testing.assert_close(actual.loss, expected.loss)
        torch.testing.assert_close(
            torch.autograd.grad(actual.loss, x)[0],
            torch.autograd.grad(expected.loss, x)[0],
        )
    finally:
        torch.use_deterministic_algorithms(prior)
