# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for corresponding-point Procrustes registration."""

import importlib
import inspect
from typing import Literal, get_type_hints

import pytest
import torch

import physicsnemo.nn.functional as functional
from benchmarks.physicsnemo.nn.functional._spec_utils import (
    build_benchmark_plan,
    case_by_index,
    case_labels,
)
from benchmarks.physicsnemo.nn.functional.registry import FUNCTIONAL_SPECS
from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.nn.functional import procrustes
from physicsnemo.nn.functional.geometry import Procrustes
from test.conftest import requires_module
from test.nn.functional._parity_utils import clone_case


def _proper_rotation(
    batch_shape: tuple[int, ...],
    num_dims: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """Create deterministic proper orthogonal matrices."""

    generator = torch.Generator(device=device).manual_seed(seed)
    matrix = torch.randn(
        (*batch_shape, num_dims, num_dims),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    rotation, _ = torch.linalg.qr(matrix)
    determinant = torch.linalg.det(rotation)
    correction = torch.cat(
        (
            torch.ones((*batch_shape, num_dims - 1), device=device, dtype=dtype),
            determinant.sign().unsqueeze(-1),
        ),
        dim=-1,
    )
    return rotation * correction.unsqueeze(-2)


def _apply_transform(
    points: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Apply the row-vector transform returned by ``procrustes``."""

    return (
        scale[..., None, None] * (points @ rotation.transpose(-2, -1))
        + translation[..., None, :]
    )


def _output_probes(
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor], seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create independent upstream cotangents for every returned tensor."""

    generator = torch.Generator(device=output[0].device).manual_seed(seed)
    return tuple(
        torch.randn(
            value.shape,
            generator=generator,
            device=value.device,
            dtype=value.dtype,
        )
        for value in output
    )


def _run_with_probes(
    source_value: torch.Tensor,
    target_value: torch.Tensor,
    probes: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    implementation: str,
    *,
    scale: bool = True,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    """Evaluate one backend and pull back independent output probes."""

    source = source_value.detach().clone().requires_grad_(True)
    target = target_value.detach().clone().requires_grad_(True)
    output = procrustes(
        source,
        target,
        scale=scale,
        implementation=implementation,
    )
    loss = sum((value * probe).sum() for value, probe in zip(output, probes))
    gradients = torch.autograd.grad(loss, (source, target))
    return output, gradients


def _trim_benchmark_case(
    args: tuple[torch.Tensor, torch.Tensor],
    *,
    max_batch: int = 2,
    max_points: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce benchmark-sized clouds while preserving their input contract."""

    trimmed = []
    for value in args:
        if value.ndim == 2:
            sliced = value[:max_points]
        else:
            sliced = value[:max_batch, :max_points]
        cloned = sliced.detach().clone()
        cloned.requires_grad_(value.requires_grad)
        trimmed.append(cloned)
    return tuple(trimmed)


def _identity_backend_output(source: torch.Tensor):
    """Return a correctly shaped identity transform for dispatch spies."""

    batch_size, _, num_dims = source.shape
    rotation = torch.eye(num_dims, device=source.device, dtype=source.dtype).expand(
        batch_size, num_dims, num_dims
    )
    translation = torch.zeros(
        (batch_size, num_dims), device=source.device, dtype=source.dtype
    )
    scale = torch.ones(batch_size, device=source.device, dtype=source.dtype)
    return rotation, translation, scale


def test_public_exports_and_function_spec():
    assert procrustes.__name__ == "procrustes"
    assert procrustes.__module__ == (
        "physicsnemo.nn.functional.geometry.deform.procrustes"
    )
    assert issubclass(Procrustes, FunctionSpec)
    assert Procrustes in FUNCTIONAL_SPECS
    assert functional.procrustes is procrustes
    assert not hasattr(functional, "Procrustes")
    assert list(inspect.signature(procrustes).parameters) == [
        "source",
        "target",
        "scale",
        "implementation",
    ]
    assert get_type_hints(procrustes)["implementation"] == (
        Literal["torch", "warp"] | None
    )
    assert Procrustes.implementations() == ("warp", "torch")


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("num_dims", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("estimate_scale", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_procrustes_exact_transform(
    device,
    implementation,
    num_dims,
    dtype,
    estimate_scale,
    batched,
):
    device = torch.device(device)
    batch_shape = (2,) if batched else ()
    generator = torch.Generator(device=device).manual_seed(100 + num_dims)
    source = torch.randn(
        (*batch_shape, 24, num_dims),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    expected_rotation = _proper_rotation(
        batch_shape,
        num_dims,
        device=device,
        dtype=dtype,
        seed=200 + num_dims,
    )
    expected_translation = torch.randn(
        (*batch_shape, num_dims),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    if estimate_scale:
        expected_scale = (
            torch.linspace(0.8, 1.4, 2, device=device, dtype=dtype)
            if batched
            else source.new_tensor(1.4)
        )
    else:
        expected_scale = torch.ones(batch_shape, device=device, dtype=dtype)
    target = _apply_transform(
        source,
        expected_rotation,
        expected_translation,
        expected_scale,
    )

    rotation, translation, scale = procrustes(
        source,
        target,
        scale=estimate_scale,
        implementation=implementation,
    )
    aligned = _apply_transform(source, rotation, translation, scale)

    atol, rtol = (5.0e-5, 5.0e-5) if dtype == torch.float32 else (2.0e-9, 2.0e-9)
    torch.testing.assert_close(rotation, expected_rotation, atol=atol, rtol=rtol)
    torch.testing.assert_close(translation, expected_translation, atol=atol, rtol=rtol)
    torch.testing.assert_close(scale, expected_scale, atol=atol, rtol=rtol)
    torch.testing.assert_close(aligned, target, atol=atol, rtol=rtol)


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_procrustes_returns_proper_rotation_for_reflection(device, implementation):
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(401)
    source = torch.randn(32, 2, generator=generator, device=device, dtype=torch.float64)
    reflection = source.new_tensor([[-1.0, 0.0], [0.0, 1.0]])
    target = source @ reflection.T

    rotation, translation, scale = procrustes(
        source, target, implementation=implementation
    )
    aligned = _apply_transform(source, rotation, translation, scale)

    torch.testing.assert_close(
        torch.linalg.det(rotation), source.new_tensor(1.0), atol=1e-10, rtol=1e-10
    )
    assert scale >= 0
    assert not torch.allclose(aligned, target, atol=1e-4, rtol=1e-4)


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_procrustes_noncontiguous_inputs(device, implementation):
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(501)
    source = torch.randn(
        2, 3, 17, generator=generator, device=device, dtype=torch.float64
    ).transpose(1, 2)
    target = torch.randn(
        2, 3, 17, generator=generator, device=device, dtype=torch.float64
    ).transpose(1, 2)
    assert not source.is_contiguous()
    assert not target.is_contiguous()

    actual = procrustes(source, target, implementation=implementation)
    reference = procrustes(
        source.contiguous(), target.contiguous(), implementation=implementation
    )
    Procrustes.compare_forward(actual, reference)


def test_procrustes_validation(device):
    device = torch.device(device)
    source = torch.randn(8, 3, device=device)
    target = torch.randn(8, 3, device=device)

    with pytest.raises(TypeError, match="source must be"):
        procrustes([[0.0]], target, implementation="torch")
    with pytest.raises(TypeError, match="target must be"):
        procrustes(source, [[0.0]], implementation="torch")
    with pytest.raises(TypeError, match="scale must be a bool"):
        procrustes(source, target, scale=1, implementation="torch")
    with pytest.raises(ValueError, match="must have shape"):
        procrustes(
            torch.randn(3, device=device),
            torch.randn(3, device=device),
            implementation="torch",
        )
    with pytest.raises(ValueError, match="must have shape"):
        procrustes(
            torch.randn(1, 2, 8, 3, device=device),
            torch.randn(1, 2, 8, 3, device=device),
            implementation="torch",
        )
    with pytest.raises(ValueError, match="identical shapes"):
        procrustes(source, target[:-1], implementation="torch")
    with pytest.raises(TypeError, match="same dtype"):
        procrustes(source.float(), target.double(), implementation="torch")
    with pytest.raises(TypeError, match="float32 or torch.float64"):
        procrustes(source.half(), target.half(), implementation="torch")
    with pytest.raises(ValueError, match="dimension must be 1, 2, or 3"):
        procrustes(
            torch.randn(8, 4, device=device),
            torch.randn(8, 4, device=device),
            implementation="torch",
        )
    with pytest.raises(ValueError, match="requires at least"):
        procrustes(
            torch.randn(2, 3, device=device),
            torch.randn(2, 3, device=device),
            implementation="torch",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_procrustes_rejects_mixed_devices():
    with pytest.raises(ValueError, match="same device"):
        procrustes(
            torch.randn(8, 3),
            torch.randn(8, 3, device="cuda"),
            implementation="torch",
        )


@requires_module("warp")
@pytest.mark.parametrize("num_dims", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_procrustes_torch_warp_forward_and_gradient_parity(device, num_dims, dtype):
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(601 + num_dims)
    source = torch.randn(
        2, 13, num_dims, generator=generator, device=device, dtype=dtype
    )
    rotation = _proper_rotation(
        (2,), num_dims, device=device, dtype=dtype, seed=701 + num_dims
    )
    translation = torch.randn(
        2, num_dims, generator=generator, device=device, dtype=dtype
    )
    scale = torch.tensor([0.9, 1.3], device=device, dtype=dtype)
    noise = 0.02 * torch.randn(
        source.shape, generator=generator, device=device, dtype=dtype
    )
    target = _apply_transform(source, rotation, translation, scale) + noise

    reference_output = procrustes(source, target, implementation="torch")
    probes = _output_probes(reference_output, seed=801 + num_dims)
    torch_result = _run_with_probes(source, target, probes, "torch")
    warp_result = _run_with_probes(source, target, probes, "warp")

    Procrustes.compare_forward(warp_result[0], torch_result[0])
    for actual, expected in zip(warp_result[1], torch_result[1]):
        Procrustes.compare_backward(actual, expected)


@requires_module("warp")
@pytest.mark.parametrize(
    ("num_dims", "batch_size", "seed"),
    [(2, 4, 633), (3, 8, 425)],
)
def test_procrustes_warp_float32_svd_regression(device, num_dims, batch_size, seed):
    """Guard against inaccurate native float32 Warp small-matrix SVDs."""

    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    source = torch.randn(batch_size, 17, num_dims, generator=generator, device=device)
    target = torch.randn(batch_size, 17, num_dims, generator=generator, device=device)

    reference_output = procrustes(source, target, implementation="torch")
    probes = _output_probes(reference_output, seed=seed + 1)
    torch_result = _run_with_probes(source, target, probes, "torch")
    warp_result = _run_with_probes(source, target, probes, "warp")
    Procrustes.compare_forward(warp_result[0], torch_result[0])
    for actual, expected in zip(warp_result[1], torch_result[1], strict=True):
        Procrustes.compare_backward(actual, expected)

    direction_source = torch.randn(source.shape, generator=generator, device=device)
    direction_target = torch.randn(target.shape, generator=generator, device=device)
    direction_source /= direction_source.norm()
    direction_target /= direction_target.norm()
    directional_autograd = sum(
        (gradient * direction).sum()
        for gradient, direction in zip(
            warp_result[1],
            (direction_source, direction_target),
            strict=True,
        )
    )

    def probed_warp_loss(source_value, target_value):
        output = procrustes(source_value, target_value, implementation="warp")
        return sum(
            (value * probe).sum() for value, probe in zip(output, probes, strict=True)
        )

    step = 1.0e-2
    directional_finite_difference = (
        probed_warp_loss(
            source + step * direction_source,
            target + step * direction_target,
        )
        - probed_warp_loss(
            source - step * direction_source,
            target - step * direction_target,
        )
    ) / (2.0 * step)
    torch.testing.assert_close(
        directional_autograd,
        directional_finite_difference,
        atol=3.0e-4,
        rtol=3.0e-3,
    )


@requires_module("warp")
def test_procrustes_warp_float32_elongated_cloud_regression(device):
    device = torch.device(device)
    amplitudes = torch.tensor([1.0, 1.0e-4, 0.0], device=device)
    source = torch.cat((torch.diag(amplitudes), -torch.diag(amplitudes)))
    expected_rotation = _proper_rotation(
        (), 3, device=device, dtype=source.dtype, seed=427
    )
    target = source @ expected_rotation.T

    rotation, translation, scale = procrustes(
        source,
        target,
        scale=False,
        implementation="warp",
    )
    aligned = _apply_transform(source, rotation, translation, scale)

    torch.testing.assert_close(rotation, expected_rotation, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(aligned, target, atol=2e-6, rtol=2e-5)


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_procrustes_gradcheck(implementation):
    generator = torch.Generator().manual_seed(850)
    source = torch.randn(
        7, 3, generator=generator, dtype=torch.float64, requires_grad=True
    )
    target = torch.randn(
        7, 3, generator=generator, dtype=torch.float64, requires_grad=True
    )

    assert torch.autograd.gradcheck(
        lambda source_, target_: procrustes(
            source_, target_, implementation=implementation
        ),
        (source, target),
        atol=1.0e-5,
        rtol=1.0e-4,
        fast_mode=True,
    )


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_procrustes_reflection_correction_gradcheck(implementation):
    generator = torch.Generator().manual_seed(851)
    source = torch.randn(
        9, 3, generator=generator, dtype=torch.float64, requires_grad=True
    )
    reflection = torch.diag(source.new_tensor([-1.0, 1.0, 1.0]))
    translation = source.new_tensor([0.2, -0.1, 0.3])
    target = (source.detach() @ reflection.T + translation).requires_grad_()

    source_centered = source.detach() - source.detach().mean(dim=0)
    target_centered = target.detach() - target.detach().mean(dim=0)
    assert torch.linalg.det(source_centered.T @ target_centered) < 0
    assert torch.autograd.gradcheck(
        lambda source_, target_: procrustes(
            source_, target_, implementation=implementation
        ),
        (source, target),
        atol=1.0e-5,
        rtol=1.0e-4,
        fast_mode=True,
    )


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("geometry", ["repeated-spectrum", "rank-d-minus-one"])
def test_procrustes_well_defined_degenerate_gradients(implementation, geometry):
    """Exercise the cases handled explicitly by the polar-factor VJP."""

    if geometry == "repeated-spectrum":
        basis = torch.eye(3, dtype=torch.float64)
        source_value = torch.cat((basis, -basis), dim=0)
        target_value = source_value.clone()
    else:
        generator = torch.Generator().manual_seed(861)
        planar = torch.randn(12, 2, generator=generator, dtype=torch.float64)
        source_value = torch.cat((planar, torch.zeros(12, 1)), dim=-1)
        rotation = _proper_rotation(
            (),
            3,
            device=torch.device("cpu"),
            dtype=torch.float64,
            seed=862,
        )
        translation = source_value.new_tensor([0.2, -0.3, 0.4])
        target_value = _apply_transform(
            source_value,
            rotation,
            translation,
            source_value.new_tensor(1.2),
        )

    source = source_value.requires_grad_(True)
    target = target_value.requires_grad_(True)
    output = procrustes(source, target, implementation=implementation)
    probes = _output_probes(output, seed=863)
    gradients = torch.autograd.grad(
        sum((value * probe).sum() for value, probe in zip(output, probes)),
        (source, target),
    )

    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_procrustes_rejects_second_derivatives(implementation):
    generator = torch.Generator().manual_seed(854)
    source = torch.randn(
        7, 3, generator=generator, dtype=torch.float64, requires_grad=True
    )
    target = torch.randn(
        7, 3, generator=generator, dtype=torch.float64, requires_grad=True
    )
    output = procrustes(source, target, implementation=implementation)
    probes = _output_probes(output, seed=855)
    loss = sum(
        (value * probe).sum() for value, probe in zip(output, probes, strict=True)
    )

    if implementation == "torch":
        with pytest.raises(RuntimeError, match="first-order reverse-mode"):
            torch.autograd.grad(loss, (source, target), create_graph=True)
    else:
        gradients = torch.autograd.grad(loss, (source, target), create_graph=True)
        with pytest.raises(RuntimeError, match="no autograd formula"):
            torch.autograd.grad(
                sum(gradient.sum() for gradient in gradients),
                (source, target),
            )


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("magnitude", [1.0e-20, 1.0e20])
def test_procrustes_float32_extreme_magnitudes(device, implementation, magnitude):
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(856)
    source = (
        magnitude * torch.randn(32, 3, generator=generator, device=device)
    ).requires_grad_()
    expected_rotation = _proper_rotation(
        (), 3, device=device, dtype=source.dtype, seed=857
    )
    expected_translation = magnitude * source.new_tensor([0.2, -0.3, 0.4])
    expected_scale = source.new_tensor(1.25)
    target = _apply_transform(
        source.detach(),
        expected_rotation,
        expected_translation,
        expected_scale,
    ).requires_grad_()

    rotation, translation, scale = procrustes(
        source, target, implementation=implementation
    )
    aligned = _apply_transform(source, rotation, translation, scale)
    torch.testing.assert_close(rotation, expected_rotation, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(scale, expected_scale, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(
        aligned / magnitude,
        target / magnitude,
        atol=3e-5,
        rtol=3e-5,
    )

    probe = source.new_tensor([[0.2, -0.7, 0.3], [0.5, 0.1, -0.4], [-0.6, 0.8, 0.9]])
    loss = (rotation * probe).sum() + translation.sum() / magnitude + scale
    gradients = torch.autograd.grad(loss, (source, target))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("expected_scale", [0.0, 2.0])
def test_procrustes_float32_scale_restoration_extreme_ratio(
    device, implementation, expected_scale
):
    """A finite fitted scale survives an overflowing raw magnitude ratio."""

    device = torch.device(device)
    correlated_direction = torch.tensor(
        [-1.0, 1.0, 0.0, 0.0], device=device, dtype=torch.float32
    )
    orthogonal_direction = torch.tensor(
        [0.0, 0.0, -1.0, 1.0], device=device, dtype=torch.float32
    )
    source = (1.0e-20 * correlated_direction).unsqueeze(-1)
    target = (1.0e20 * orthogonal_direction).unsqueeze(-1) + expected_scale * source

    rotation, translation, scale = procrustes(
        source,
        target,
        implementation=implementation,
    )

    assert rotation.dtype == source.dtype
    assert translation.dtype == source.dtype
    assert scale.dtype == source.dtype
    assert torch.isfinite(rotation).all()
    assert torch.isfinite(translation).all()
    assert torch.isfinite(scale).all()
    torch.testing.assert_close(
        rotation,
        torch.ones_like(rotation),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        translation,
        torch.zeros_like(translation),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        scale,
        source.new_tensor(expected_scale),
        atol=2.0e-5,
        rtol=2.0e-5,
    )


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_procrustes_float64_zero_covariance_extreme_ratio(device, implementation):
    """Zero covariance stays zero when the raw float64 ratio overflows."""

    device = torch.device(device)
    source = (
        1.0e-200
        * torch.tensor([-1.0, 1.0, 0.0, 0.0], device=device, dtype=torch.float64)
    ).unsqueeze(-1)
    target = (
        1.0e200
        * torch.tensor([0.0, 0.0, -1.0, 1.0], device=device, dtype=torch.float64)
    ).unsqueeze(-1)

    rotation, translation, scale = procrustes(
        source,
        target,
        implementation=implementation,
    )

    assert torch.isfinite(rotation).all()
    assert torch.isfinite(translation).all()
    assert torch.isfinite(scale).all()
    torch.testing.assert_close(scale, torch.zeros_like(scale), atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        translation,
        torch.zeros_like(translation),
        atol=0.0,
        rtol=0.0,
    )


def test_procrustes_default_cpu_dispatch(monkeypatch):
    module = importlib.import_module(Procrustes.__module__)
    source = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    target = source.clone()
    calls = []

    def torch_spy(source_b3, *_args):
        calls.append("torch")
        return _identity_backend_output(source_b3)

    def warp_spy(source_b3, *_args):
        calls.append("warp")
        return _identity_backend_output(source_b3)

    monkeypatch.setattr(module, "procrustes_torch", torch_spy)
    monkeypatch.setattr(module, "procrustes_warp", warp_spy)

    rotation, translation, scale = procrustes(source, target)
    assert calls == ["torch"]
    torch.testing.assert_close(rotation, torch.eye(2))
    torch.testing.assert_close(translation, torch.zeros(2))
    torch.testing.assert_close(scale, torch.ones(()))


@requires_module("warp")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_procrustes_cuda_dispatch_and_fallback(monkeypatch):
    module = importlib.import_module(Procrustes.__module__)
    device = torch.device("cuda")
    source = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device=device)
    target = source.clone()
    calls = []

    def torch_spy(source_b3, *_args):
        calls.append("torch")
        return _identity_backend_output(source_b3)

    def warp_spy(source_b3, *_args):
        calls.append("warp")
        return _identity_backend_output(source_b3)

    monkeypatch.setattr(module, "procrustes_torch", torch_spy)
    monkeypatch.setattr(module, "procrustes_warp", warp_spy)

    warp_impl = Procrustes._get_impls()["warp"]
    assert warp_impl.available
    procrustes(source, target)
    assert calls == ["warp"]

    calls.clear()
    unavailable_warp = type(warp_impl)(
        name=warp_impl.name,
        func=warp_impl.func,
        required_imports=warp_impl.required_imports,
        rank=warp_impl.rank,
        baseline=warp_impl.baseline,
        available=False,
    )
    monkeypatch.setitem(Procrustes._get_impls(), "warp", unavailable_warp)
    FunctionSpec._fallback_warned.discard(Procrustes._class_key())
    with pytest.warns(RuntimeWarning, match="falling back to implementation"):
        procrustes(source, target)
    assert calls == ["torch"]
    FunctionSpec._fallback_warned.discard(Procrustes._class_key())


@requires_module("warp")
@pytest.mark.parametrize("num_dims", [1, 2, 3])
def test_procrustes_warp_custom_op_opcheck(num_dims):
    from physicsnemo.nn.functional.geometry.deform._warp_impl import (
        procrustes_rotation_warp_impl,
    )

    generator = torch.Generator().manual_seed(901 + num_dims)
    covariance = torch.randn(
        2, num_dims, num_dims, generator=generator, dtype=torch.float64
    )
    covariance = (covariance + 2.0 * torch.eye(num_dims)).requires_grad_(True)
    torch.library.opcheck(procrustes_rotation_warp_impl, args=(covariance,))


@requires_module("warp")
@pytest.mark.parametrize("num_dims", [2, 3])
def test_procrustes_warp_custom_op_rejects_unpromoted_float32(num_dims):
    from physicsnemo.nn.functional.geometry.deform._warp_impl import (
        procrustes_rotation_warp_impl,
    )

    covariance = torch.randn(2, num_dims, num_dims, dtype=torch.float32)
    with pytest.raises(TypeError, match="requires float64 covariance"):
        procrustes_rotation_warp_impl(covariance)


@requires_module("warp")
@pytest.mark.parametrize("num_dims", [1, 2, 3])
def test_procrustes_warp_custom_op_fake(num_dims):
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    from physicsnemo.nn.functional.geometry.deform._warp_impl import (
        procrustes_rotation_warp_impl,
    )

    with FakeTensorMode():
        covariance = torch.empty(2, num_dims, num_dims, dtype=torch.float64)
        rotation, symmetric_factor = procrustes_rotation_warp_impl(covariance)

    assert isinstance(rotation, FakeTensor)
    assert isinstance(symmetric_factor, FakeTensor)
    assert rotation.shape == covariance.shape
    assert symmetric_factor.shape == covariance.shape


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_procrustes_torch_compile_fullgraph_forward_backward(
    device, implementation, dtype
):
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(1001)
    source_value = torch.randn(
        2, 17, 3, generator=generator, device=device, dtype=dtype
    )
    target_value = torch.randn(
        2, 17, 3, generator=generator, device=device, dtype=dtype
    )

    def operation(source, target):
        return procrustes(source, target, implementation=implementation)

    eager_source = source_value.clone().requires_grad_(True)
    eager_target = target_value.clone().requires_grad_(True)
    eager_output = operation(eager_source, eager_target)
    probes = _output_probes(eager_output, seed=1002)
    eager_loss = sum(
        (value * probe).sum() for value, probe in zip(eager_output, probes)
    )
    eager_gradients = torch.autograd.grad(eager_loss, (eager_source, eager_target))

    compiled = torch.compile(operation, fullgraph=True)
    compiled_source = source_value.clone().requires_grad_(True)
    compiled_target = target_value.clone().requires_grad_(True)
    compiled_output = compiled(compiled_source, compiled_target)
    compiled_loss = sum(
        (value * probe).sum() for value, probe in zip(compiled_output, probes)
    )
    compiled_gradients = torch.autograd.grad(
        compiled_loss, (compiled_source, compiled_target)
    )

    Procrustes.compare_forward(compiled_output, eager_output)
    for actual, expected in zip(compiled_gradients, eager_gradients):
        Procrustes.compare_backward(actual, expected)


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp", None])
def test_procrustes_cuda_graph_capture(device, implementation):
    device = torch.device(device)
    if device.type != "cuda":
        pytest.skip("CUDA Graph capture requires CUDA")

    generator = torch.Generator(device=device).manual_seed(1101)
    source = torch.randn(2, 24, 3, generator=generator, device=device)
    target = torch.randn(2, 24, 3, generator=generator, device=device)

    expected = procrustes(source, target, implementation=implementation)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = procrustes(source, target, implementation=implementation)
    graph.replay()
    torch.cuda.synchronize(device)
    Procrustes.compare_forward(captured, expected)


@requires_module("warp")
def test_procrustes_benchmark_generators_and_backend_parity(device):
    device = torch.device(device)

    forward_labels = []
    for label, args, kwargs in Procrustes.make_inputs_forward(device=device):
        forward_labels.append(label)
        reduced_args = _trim_benchmark_case(args)
        torch_args, torch_kwargs = clone_case(reduced_args, kwargs)
        warp_args, warp_kwargs = clone_case(reduced_args, kwargs)
        output_torch = Procrustes.dispatch(
            *torch_args, implementation="torch", **torch_kwargs
        )
        output_warp = Procrustes.dispatch(
            *warp_args, implementation="warp", **warp_kwargs
        )
        Procrustes.compare_forward(output_warp, output_torch)

    assert forward_labels == [case[0] for case in Procrustes._FORWARD_BENCHMARK_CASES]

    backward_labels = []
    for case_index, (label, args, kwargs) in enumerate(
        Procrustes.make_inputs_backward(device=device)
    ):
        backward_labels.append(label)
        reduced_args = _trim_benchmark_case(args)
        torch_args, torch_kwargs = clone_case(reduced_args, kwargs)
        warp_args, warp_kwargs = clone_case(reduced_args, kwargs)
        output_torch = Procrustes.dispatch(
            *torch_args, implementation="torch", **torch_kwargs
        )
        probes = _output_probes(output_torch, seed=1201 + case_index)
        output_warp = Procrustes.dispatch(
            *warp_args, implementation="warp", **warp_kwargs
        )
        Procrustes.compare_forward(output_warp, output_torch)
        loss_torch = sum(
            (value * probe).sum() for value, probe in zip(output_torch, probes)
        )
        loss_warp = sum(
            (value * probe).sum() for value, probe in zip(output_warp, probes)
        )
        gradients_torch = torch.autograd.grad(loss_torch, torch_args)
        gradients_warp = torch.autograd.grad(loss_warp, warp_args)
        for actual, expected in zip(gradients_warp, gradients_torch):
            Procrustes.compare_backward(actual, expected)

    assert backward_labels == [case[0] for case in Procrustes._BACKWARD_BENCHMARK_CASES]


@requires_module("warp")
@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_procrustes_benchmark_planner_contract(phase):
    expected_labels = [
        case[0]
        for case in (
            Procrustes._FORWARD_BENCHMARK_CASES
            if phase == "forward"
            else Procrustes._BACKWARD_BENCHMARK_CASES
        )
    ]
    assert case_labels(Procrustes, phase, "cpu") == expected_labels

    for case_index, expected_label in enumerate(expected_labels):
        label, _, _ = case_by_index(Procrustes, phase, case_index, "cpu")
        assert label == expected_label
    with pytest.raises(IndexError, match="out of range"):
        case_by_index(Procrustes, phase, len(expected_labels), "cpu")

    keys, key_to_spec = build_benchmark_plan(
        device="cpu",
        phases=[phase],
        selected_specs=[Procrustes],
    )
    expected_implementations = set(Procrustes.available_implementations())
    assert len(keys) == len(expected_labels) * len(expected_implementations)
    assert {key[2] for key in keys} == expected_implementations
    assert all(key_to_spec[key] is Procrustes for key in keys)
