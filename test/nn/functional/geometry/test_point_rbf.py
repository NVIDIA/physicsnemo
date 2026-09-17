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

"""Tests for thin-plate-spline radial-basis point deformation."""

import inspect
import math
from typing import Literal, get_type_hints

import pytest
import torch

from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.nn.functional import radial_basis_function_deform_points
from physicsnemo.nn.functional.geometry import RadialBasisFunctionDeformPoints
from physicsnemo.nn.functional.geometry.deform import rbf as rbf_module
from test.conftest import requires_module
from test.nn.functional._parity_utils import clone_case


def _controls(num_dims: int, *, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Return an affinely spanning, well-separated control layout."""

    values = {
        1: [[-1.3], [-0.25], [0.65], [1.7]],
        2: [
            [-1.2, -0.7],
            [0.9, -0.8],
            [-0.75, 1.15],
            [0.7, 0.95],
            [0.15, 0.2],
        ],
        3: [
            [-1.1, -0.8, -0.6],
            [1.0, -0.7, -0.5],
            [-0.65, 1.1, -0.4],
            [-0.55, -0.45, 1.2],
            [0.85, 0.9, 0.75],
            [0.2, -0.15, 0.35],
        ],
        4: [
            [-1.0, -1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0, 1.0],
            [0.5, 0.25, -0.5, 0.75],
        ],
    }[num_dims]
    return torch.tensor(values, dtype=dtype, device=device)


def _trim_benchmark_case(args, kwargs, max_points=24, max_controls=12):
    """Retain every benchmark branch without running benchmark-scale arrays."""

    points, controls, control_displacements = args
    num_controls = min(controls.shape[-2], max_controls)
    trimmed_args = (
        points[..., :max_points, :],
        controls[..., :num_controls, :],
        control_displacements[..., :num_controls, :],
    )
    trimmed_kwargs = dict(kwargs)
    point_weights = trimmed_kwargs["point_weights"]
    if point_weights is not None:
        trimmed_kwargs["point_weights"] = point_weights[..., :max_points]
    return trimmed_args, trimmed_kwargs


def _differentiable_tensors(args, kwargs):
    """Collect differentiable inputs in a stable order for parity checks."""

    values = (*args, kwargs["point_weights"])
    return tuple(
        value
        for value in values
        if isinstance(value, torch.Tensor) and value.requires_grad
    )


def test_public_exports_signature_and_function_spec_contract():
    assert (
        radial_basis_function_deform_points.__name__
        == "radial_basis_function_deform_points"
    )
    assert radial_basis_function_deform_points.__module__ == (
        "physicsnemo.nn.functional.geometry.deform.rbf"
    )
    assert issubclass(RadialBasisFunctionDeformPoints, FunctionSpec)
    assert list(inspect.signature(radial_basis_function_deform_points).parameters) == [
        "points",
        "control_points",
        "control_displacements",
        "kernel",
        "polynomial",
        "smoothing",
        "point_weights",
        "implementation",
    ]
    assert set(RadialBasisFunctionDeformPoints.implementations()) == {"torch", "warp"}
    assert get_type_hints(radial_basis_function_deform_points)["implementation"] == (
        Literal["torch", "warp"] | None
    )


@pytest.mark.parametrize("num_dims", [1, 2, 3, 4])
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_exact_interpolation_at_controls(num_dims, implementation):
    if implementation == "warp":
        pytest.importorskip("warp")
    controls = _controls(num_dims)
    generator = torch.Generator().manual_seed(1700 + num_dims)
    displacements = 0.2 * torch.randn(
        controls.shape, generator=generator, dtype=controls.dtype
    )

    output = radial_basis_function_deform_points(
        controls,
        controls,
        displacements,
        implementation=implementation,
    )
    torch.testing.assert_close(
        output,
        controls + displacements,
        atol=2.0e-11,
        rtol=2.0e-11,
    )


def test_torch_query_chunking_preserves_outputs_and_gradients(monkeypatch):
    from physicsnemo.nn.functional.geometry.deform import _rbf_torch_impl

    dtype = torch.float64
    generator = torch.Generator().manual_seed(2011)
    base_inputs = (
        torch.randn((19, 3), generator=generator, dtype=dtype),
        _controls(3, dtype=dtype),
        0.1 * torch.randn((6, 3), generator=generator, dtype=dtype),
    )

    def run(byte_budget):
        monkeypatch.setattr(
            _rbf_torch_impl,
            "_RBF_PAIRWISE_TEMPORARY_BYTE_BUDGET",
            byte_budget,
        )
        inputs = tuple(tensor.clone().requires_grad_() for tensor in base_inputs)
        output = radial_basis_function_deform_points(*inputs, implementation="torch")
        gradients = torch.autograd.grad(output.square().sum(), inputs)
        return output, gradients

    chunked_output, chunked_gradients = run(512)
    dense_output, dense_gradients = run(1 << 40)
    torch.testing.assert_close(chunked_output, dense_output)
    for actual, expected in zip(chunked_gradients, dense_gradients, strict=True):
        torch.testing.assert_close(actual, expected)


def test_thin_plate_spline_value_at_off_control_query():
    dtype = torch.float64
    controls = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=dtype,
    )
    radial_coefficients = torch.tensor(
        [[1.0, -0.5], [-1.0, 0.5], [-1.0, 0.5], [1.0, -0.5]],
        dtype=dtype,
    )
    affine_coefficients = torch.tensor(
        [[0.2, -0.1], [0.3, 0.05], [-0.15, 0.25]],
        dtype=dtype,
    )

    def field_at(point):
        value = affine_coefficients[0].clone()
        value += point[0] * affine_coefficients[1]
        value += point[1] * affine_coefficients[2]
        for control, coefficient in zip(controls, radial_coefficients, strict=True):
            delta = point - control
            radius = math.hypot(float(delta[0]), float(delta[1]))
            phi = 0.0 if radius == 0.0 else radius * radius * math.log(radius)
            value += phi * coefficient
        return value

    control_displacements = torch.stack([field_at(control) for control in controls])
    query = torch.tensor([[0.25, 0.4]], dtype=dtype)
    expected = query + field_at(query[0])
    actual = radial_basis_function_deform_points(
        query,
        controls,
        control_displacements,
        implementation="torch",
    )
    torch.testing.assert_close(actual, expected, atol=2.0e-12, rtol=2.0e-12)


@requires_module("warp")
def test_warp_exact_coincidence_gradients_match_torch(device):
    device = torch.device(device)
    dtype = torch.float64
    base_controls = _controls(2, dtype=dtype, device=device)[:4]
    base_displacements = torch.tensor(
        [[0.08, -0.04], [-0.03, 0.09], [0.05, 0.02], [-0.06, -0.01]],
        dtype=dtype,
        device=device,
    )
    cotangent = torch.tensor(
        [[0.2, -0.5], [-0.3, 0.1], [0.7, -0.4], [-0.2, 0.6]],
        dtype=dtype,
        device=device,
    )

    def run(implementation):
        points = base_controls.clone().requires_grad_()
        controls = base_controls.clone().requires_grad_()
        displacements = base_displacements.clone().requires_grad_()
        output = radial_basis_function_deform_points(
            points,
            controls,
            displacements,
            implementation=implementation,
        )
        gradients = torch.autograd.grad(
            (output * cotangent).sum(),
            (points, controls, displacements),
        )
        return output, gradients

    expected_output, expected_gradients = run("torch")
    actual_output, actual_gradients = run("warp")
    torch.testing.assert_close(actual_output, expected_output, atol=2e-10, rtol=2e-10)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize(("num_dims", "batch_size"), [(1, 1), (2, 2), (3, 2), (4, 1)])
def test_affine_reproduction_is_dimensionally_generic_and_batched(num_dims, batch_size):
    dtype = torch.float64
    base_controls = _controls(num_dims, dtype=dtype)
    generator = torch.Generator().manual_seed(2100 + num_dims)
    points = torch.randn((batch_size, 7, num_dims), generator=generator, dtype=dtype)
    controls = base_controls.unsqueeze(0).repeat(batch_size, 1, 1)
    controls = controls + 0.1 * torch.arange(batch_size, dtype=dtype).view(-1, 1, 1)
    linear = 0.15 * torch.randn(
        (batch_size, num_dims, num_dims), generator=generator, dtype=dtype
    )
    offset = 0.1 * torch.randn(
        (batch_size, 1, num_dims), generator=generator, dtype=dtype
    )
    control_displacements = torch.bmm(controls, linear) + offset
    expected_displacements = torch.bmm(points, linear) + offset

    output = radial_basis_function_deform_points(
        points,
        controls,
        control_displacements,
        implementation="torch",
    )

    assert output.shape == points.shape
    torch.testing.assert_close(
        output,
        points + expected_displacements,
        atol=2.0e-10,
        rtol=2.0e-10,
    )


def test_boolean_signed_and_amplifying_point_weights():
    dtype = torch.float64
    controls = _controls(2, dtype=dtype)
    displacements = torch.tensor(
        [[0.1, -0.2], [0.3, 0.05], [-0.15, 0.2], [0.05, -0.1], [0.2, 0.15]],
        dtype=dtype,
    )
    points = torch.tensor(
        [[-0.8, -0.4], [0.0, 0.1], [0.6, 0.75], [1.2, -0.25]],
        dtype=dtype,
    )
    unweighted = radial_basis_function_deform_points(
        points, controls, displacements, implementation="torch"
    )
    field = unweighted - points

    weights = torch.tensor([0.0, 0.5, -1.0, 2.0], dtype=dtype, requires_grad=True)
    weighted = radial_basis_function_deform_points(
        points,
        controls,
        displacements,
        point_weights=weights,
        implementation="torch",
    )
    torch.testing.assert_close(weighted, points + weights.unsqueeze(-1) * field)
    weighted.sum().backward()
    torch.testing.assert_close(weights.grad, field.sum(dim=-1))

    mask = torch.tensor([True, False, True, False])
    masked = radial_basis_function_deform_points(
        points,
        controls,
        displacements,
        point_weights=mask,
        implementation="torch",
    )
    torch.testing.assert_close(masked, torch.where(mask[:, None], unweighted, points))


def test_smoothing_relaxes_interpolation():
    controls = _controls(2)
    generator = torch.Generator().manual_seed(2231)
    displacements = torch.randn(
        controls.shape, generator=generator, dtype=controls.dtype
    )
    exact = radial_basis_function_deform_points(
        controls,
        controls,
        displacements,
        smoothing=0.0,
        implementation="torch",
    )
    smoothed = radial_basis_function_deform_points(
        controls,
        controls,
        displacements,
        smoothing=0.2,
        implementation="torch",
    )

    torch.testing.assert_close(exact, controls + displacements)
    assert torch.isfinite(smoothed).all()
    assert not torch.allclose(
        smoothed, controls + displacements, atol=1.0e-5, rtol=1.0e-5
    )


def test_no_polynomial_tail_still_interpolates_when_kernel_is_invertible():
    controls = _controls(2)
    displacements = torch.tensor(
        [[0.2, -0.1], [-0.15, 0.3], [0.05, 0.2], [0.1, -0.2], [-0.25, 0.1]],
        dtype=controls.dtype,
    )
    output = radial_basis_function_deform_points(
        controls,
        controls,
        displacements,
        polynomial=False,
        implementation="torch",
    )
    torch.testing.assert_close(
        output,
        controls + displacements,
        atol=2.0e-11,
        rtol=2.0e-11,
    )


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_empty_controls_are_identity(implementation):
    if implementation == "warp":
        pytest.importorskip("warp")
    points = torch.tensor(
        [[float("inf"), 1.0], [float("-inf"), 2.0], [float("nan"), 3.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    controls = torch.empty((0, 2), dtype=points.dtype, requires_grad=True)
    displacements = torch.empty_like(controls, requires_grad=True)
    point_weights = torch.tensor(
        [float("inf"), float("inf"), float("nan")],
        dtype=points.dtype,
        requires_grad=True,
    )
    output = radial_basis_function_deform_points(
        points,
        controls,
        displacements,
        point_weights=point_weights,
        implementation=implementation,
    )
    torch.testing.assert_close(output, points, equal_nan=True)
    gradients = torch.autograd.grad(
        output,
        (points, controls, displacements, point_weights),
        grad_outputs=torch.ones_like(output),
    )
    torch.testing.assert_close(gradients[0], torch.ones_like(points))
    torch.testing.assert_close(gradients[1], torch.zeros_like(controls))
    torch.testing.assert_close(gradients[2], torch.zeros_like(displacements))
    torch.testing.assert_close(gradients[3], torch.zeros_like(point_weights))


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (
            lambda: radial_basis_function_deform_points(
                [],
                torch.zeros(3, 2),
                torch.zeros(3, 2),
                implementation="torch",
            ),
            TypeError,
            "points must be a torch.Tensor, got list",
        ),
        (
            lambda: radial_basis_function_deform_points(
                torch.zeros(2, 2),
                torch.zeros(3, 2),
                [],
                implementation="torch",
            ),
            TypeError,
            "control_displacements must be a torch.Tensor, got list",
        ),
        (
            lambda: radial_basis_function_deform_points(
                torch.zeros(2, 2),
                torch.zeros(3, 2),
                torch.zeros(3, 2),
                point_weights=[],
                implementation="torch",
            ),
            TypeError,
            "point_weights must be a torch.Tensor or None, got list",
        ),
        (
            lambda: radial_basis_function_deform_points(
                torch.zeros(2, 2, dtype=torch.float16),
                torch.zeros(3, 2, dtype=torch.float16),
                torch.zeros(3, 2, dtype=torch.float16),
                implementation="torch",
            ),
            TypeError,
            "float32 or torch.float64",
        ),
        (
            lambda: radial_basis_function_deform_points(
                torch.zeros(2, 2),
                torch.zeros(3, 2),
                torch.zeros(4, 2),
                implementation="torch",
            ),
            ValueError,
            "identical shapes",
        ),
        (
            lambda: radial_basis_function_deform_points(
                torch.zeros(2, 2),
                torch.zeros(3, 2),
                torch.zeros(3, 2),
                kernel="tps",
                implementation="torch",
            ),
            ValueError,
            "kernel must be 'thin_plate_spline'",
        ),
        (
            lambda: radial_basis_function_deform_points(
                torch.zeros(2, 2),
                torch.zeros(3, 2),
                torch.zeros(3, 2),
                polynomial=1,
                implementation="torch",
            ),
            TypeError,
            "polynomial must be a bool",
        ),
        (
            lambda: radial_basis_function_deform_points(
                torch.zeros(2, 2),
                torch.zeros(3, 2),
                torch.zeros(3, 2),
                smoothing=True,
                implementation="torch",
            ),
            TypeError,
            "nonnegative finite Python real scalar",
        ),
        (
            lambda: radial_basis_function_deform_points(
                torch.zeros(2, 2),
                torch.zeros(3, 2),
                torch.zeros(3, 2),
                smoothing=-0.1,
                implementation="torch",
            ),
            ValueError,
            "nonnegative",
        ),
        (
            lambda: radial_basis_function_deform_points(
                torch.zeros(2, 2),
                torch.zeros(3, 2),
                torch.zeros(3, 2),
                smoothing=float("nan"),
                implementation="torch",
            ),
            ValueError,
            "finite",
        ),
        (
            lambda: radial_basis_function_deform_points(
                torch.zeros(2, 2),
                torch.zeros(3, 2),
                torch.zeros(3, 2),
                smoothing=10**10000,
                implementation="torch",
            ),
            ValueError,
            "finite in the control dtype",
        ),
        (
            lambda: radial_basis_function_deform_points(
                torch.zeros(2, 2),
                torch.zeros(2, 2),
                torch.zeros(2, 2),
                implementation="torch",
            ),
            ValueError,
            "at least D \\+ 1 controls",
        ),
        (
            lambda: radial_basis_function_deform_points(
                torch.zeros(2, 2),
                torch.zeros(3, 2),
                torch.zeros(3, 2),
                point_weights=torch.ones(2, 1),
                implementation="torch",
            ),
            ValueError,
            "point_weights must have shape",
        ),
        (
            lambda: radial_basis_function_deform_points(
                torch.zeros(1, 2, 2),
                torch.zeros(2, 3, 2),
                torch.zeros(2, 3, 2),
                implementation="torch",
            ),
            ValueError,
            "aligned batch sizes",
        ),
    ],
)
def test_validation(call, error, match):
    with pytest.raises(error, match=match):
        call()


def test_affinely_degenerate_controls_surface_checked_solve_diagnostic():
    controls = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.25, 0.0]], dtype=torch.float64)
    displacements = torch.tensor(
        [[0.1, 0.0], [0.0, 0.2], [-0.1, 0.1]], dtype=torch.float64
    )
    with pytest.raises(RuntimeError, match="singular"):
        radial_basis_function_deform_points(
            controls,
            controls,
            displacements,
            implementation="torch",
        )


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp", None])
def test_radial_basis_function_deform_points_rejects_cuda_graph_capture(
    device, implementation
):
    device = torch.device(device)
    if device.type != "cuda":
        pytest.skip("CUDA Graph capture requires CUDA")

    points = torch.rand((8, 3), device=device)
    controls = _controls(3, dtype=points.dtype, device=device)
    displacements = 0.1 * torch.rand_like(controls)
    point_weights = torch.rand(8, device=device)

    # Warm allocator and backend state before capture so the failure identifies
    # the checked coefficient solve rather than lazy initialization.
    radial_basis_function_deform_points(
        points,
        controls,
        displacements,
        point_weights=point_weights,
        implementation=implementation,
    )
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="not supported during CUDA Graph capture"):
        with torch.cuda.graph(graph):
            radial_basis_function_deform_points(
                points,
                controls,
                displacements,
                point_weights=point_weights,
                implementation=implementation,
            )


@pytest.mark.parametrize("polynomial", [False, True])
def test_inductor_rejects_singular_systems_like_eager(polynomial):
    if polynomial:
        controls = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.25, 0.0]])
    else:
        controls = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    displacements = torch.ones_like(controls)

    def singular_system(c, d):
        return radial_basis_function_deform_points(
            c,
            c,
            d,
            polynomial=polynomial,
            implementation="torch",
        )

    with pytest.raises(RuntimeError, match="singular"):
        singular_system(controls, displacements)
    compiled = torch.compile(singular_system, fullgraph=True, backend="inductor")
    with pytest.raises(RuntimeError, match="singular"):
        compiled(controls, displacements)


def _gradcheck_inputs():
    dtype = torch.float64
    points = torch.tensor(
        [[-0.55, -0.25], [0.1, 0.35], [0.75, -0.15]],
        dtype=dtype,
        requires_grad=True,
    )
    controls = _controls(2, dtype=dtype)[:4].clone().requires_grad_()
    displacements = torch.tensor(
        [[0.08, -0.04], [-0.03, 0.09], [0.05, 0.02], [-0.06, -0.01]],
        dtype=dtype,
        requires_grad=True,
    )
    weights = torch.tensor([0.7, -0.4, 1.1], dtype=dtype, requires_grad=True)
    return points, controls, displacements, weights


def test_torch_gradcheck_and_gradgradcheck():
    def operation(points, controls, displacements, weights):
        return radial_basis_function_deform_points(
            points,
            controls,
            displacements,
            smoothing=0.05,
            point_weights=weights,
            implementation="torch",
        )

    assert torch.autograd.gradcheck(
        operation,
        _gradcheck_inputs(),
        eps=1.0e-6,
        atol=3.0e-5,
        rtol=3.0e-4,
        fast_mode=True,
    )
    assert torch.autograd.gradgradcheck(
        operation,
        _gradcheck_inputs(),
        eps=1.0e-6,
        atol=8.0e-5,
        rtol=8.0e-4,
        fast_mode=True,
    )


def _run_with_gradients(implementation, device, dtype):
    points = torch.tensor(
        [[-0.55, -0.25], [0.1, 0.35], [0.75, -0.15]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    controls = _controls(2, dtype=dtype, device=device)[:4].clone().requires_grad_()
    displacements = torch.tensor(
        [[0.08, -0.04], [-0.03, 0.09], [0.05, 0.02], [-0.06, -0.01]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    weights = torch.tensor(
        [0.7, -0.4, 1.1], dtype=dtype, device=device, requires_grad=True
    )
    output = radial_basis_function_deform_points(
        points,
        controls,
        displacements,
        smoothing=0.05,
        point_weights=weights,
        implementation=implementation,
    )
    cotangent = torch.tensor(
        [[0.2, -0.5], [-0.3, 0.1], [0.7, -0.4]], dtype=dtype, device=device
    )
    gradients = torch.autograd.grad(
        (output * cotangent).sum(),
        (points, controls, displacements, weights),
    )
    return output, gradients


@requires_module("warp")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_torch_warp_forward_and_first_gradient_parity(device, dtype):
    device = torch.device(device)
    torch_result = _run_with_gradients("torch", device, dtype)
    warp_result = _run_with_gradients("warp", device, dtype)

    if dtype == torch.float32:
        atol, rtol = 2.0e-4, 2.0e-4
    else:
        atol, rtol = 2.0e-9, 2.0e-8
    torch.testing.assert_close(warp_result[0], torch_result[0], atol=atol, rtol=rtol)
    for actual, expected in zip(warp_result[1], torch_result[1], strict=True):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@requires_module("warp")
@pytest.mark.parametrize("polynomial", [False, True])
def test_warp_multiblock_backward_matches_torch(device, polynomial):
    device = torch.device(device)
    dtype = torch.float64
    generator = torch.Generator(device=device).manual_seed(2417)
    points = torch.randn((2, 513, 2), generator=generator, dtype=dtype, device=device)
    controls = _controls(2, dtype=dtype, device=device).unsqueeze(0).repeat(2, 1, 1)
    controls[1] += 0.1
    displacements = 0.1 * torch.randn(
        controls.shape, generator=generator, dtype=dtype, device=device
    )
    point_weights = torch.rand(
        points.shape[:-1], generator=generator, dtype=dtype, device=device
    )
    cotangent = torch.randn(
        points.shape, generator=generator, dtype=dtype, device=device
    )

    def run(implementation):
        inputs = tuple(
            tensor.clone().requires_grad_()
            for tensor in (points, controls, displacements, point_weights)
        )
        output = radial_basis_function_deform_points(
            inputs[0],
            inputs[1],
            inputs[2],
            polynomial=polynomial,
            smoothing=0.02,
            point_weights=inputs[3],
            implementation=implementation,
        )
        gradients = torch.autograd.grad((output * cotangent).sum(), inputs)
        return output, gradients

    expected_output, expected_gradients = run("torch")
    actual_output, actual_gradients = run("warp")
    torch.testing.assert_close(actual_output, expected_output, atol=2e-9, rtol=2e-9)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-8)


def test_default_dispatch_selects_device_backend(device, monkeypatch):
    device = torch.device(device)
    points = torch.tensor([[0.2, 0.1], [0.7, 0.0]], device=device)
    controls = torch.tensor([[0.0, 0.0], [1.1, 0.0], [0.0, 1.2]], device=device)
    displacements = torch.tensor([[0.0, 0.1], [0.2, -0.1], [-0.1, 0.05]], device=device)
    calls = []

    def torch_spy(normalized_points, *_args):
        calls.append("torch")
        return torch.zeros_like(normalized_points)

    def warp_spy(normalized_points, *_args):
        calls.append("warp")
        return torch.zeros_like(normalized_points)

    monkeypatch.setattr(rbf_module, "rbf_field_torch", torch_spy)
    monkeypatch.setattr(rbf_module, "rbf_field_warp", warp_spy)

    warp_impl = RadialBasisFunctionDeformPoints._get_impls()["warp"]
    expected = "warp" if device.type == "cuda" and warp_impl.available else "torch"
    output = radial_basis_function_deform_points(points, controls, displacements)
    assert calls == [expected]
    torch.testing.assert_close(output, points)


@pytest.mark.parametrize("implementation", ["torch", "warp", None])
def test_torch_compile_fullgraph_smoke(implementation):
    if implementation == "warp":
        pytest.importorskip("warp")
    dtype = torch.float64
    points = torch.tensor([[-0.4, -0.2], [0.25, 0.35]], dtype=dtype)
    controls = _controls(2, dtype=dtype)[:4]
    displacements = torch.tensor(
        [[0.08, -0.04], [-0.03, 0.09], [0.05, 0.02], [-0.06, -0.01]],
        dtype=dtype,
    )
    weights = torch.tensor([0.7, 1.1], dtype=dtype)

    def operation(p, c, d, w):
        return radial_basis_function_deform_points(
            p,
            c,
            d,
            smoothing=0.05,
            point_weights=w,
            implementation=implementation,
        )

    eager = operation(points, controls, displacements, weights)
    compiled = torch.compile(operation, fullgraph=True, backend="eager")
    torch.testing.assert_close(
        compiled(points, controls, displacements, weights), eager
    )


def test_torch_compile_fullgraph_dynamic_query_and_control_counts():
    def operation(points, controls, displacements):
        return radial_basis_function_deform_points(
            points,
            controls,
            displacements,
            smoothing=0.01,
            implementation="torch",
        )

    compiled = torch.compile(operation, fullgraph=True, dynamic=True, backend="eager")
    generator = torch.Generator().manual_seed(2519)
    base_controls = _controls(2, dtype=torch.float32)
    for num_points, num_controls in ((3, 3), (7, 4), (11, 5)):
        points = torch.randn((num_points, 2), generator=generator)
        controls = base_controls[:num_controls]
        displacements = 0.1 * torch.randn(
            controls.shape, generator=generator, dtype=controls.dtype
        )
        torch.testing.assert_close(
            compiled(points, controls, displacements),
            operation(points, controls, displacements),
        )


@requires_module("warp")
def test_inductor_warp_forward_and_backward(device):
    device = torch.device(device)
    dtype = torch.float32
    base_inputs = (
        torch.tensor([[-0.4, -0.2], [0.25, 0.35]], dtype=dtype, device=device),
        _controls(2, dtype=dtype, device=device)[:4],
        torch.tensor(
            [[0.08, -0.04], [-0.03, 0.09], [0.05, 0.02], [-0.06, -0.01]],
            dtype=dtype,
            device=device,
        ),
    )

    def operation(p, c, d):
        return radial_basis_function_deform_points(
            p,
            c,
            d,
            smoothing=0.05,
            implementation="warp",
        )

    def run(function):
        inputs = tuple(tensor.clone().requires_grad_() for tensor in base_inputs)
        output = function(*inputs)
        gradients = torch.autograd.grad(output.square().sum(), inputs)
        return output, gradients

    expected_output, expected_gradients = run(operation)
    compiled = torch.compile(operation, fullgraph=True, backend="inductor")
    actual_output, actual_gradients = run(compiled)
    torch.testing.assert_close(
        actual_output,
        expected_output,
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=2.0e-4, rtol=2.0e-4)


@pytest.mark.parametrize(
    ("smoothing", "match"),
    [
        (-0.1, "smoothing must be nonnegative"),
        (float("nan"), "smoothing must be finite"),
        (float("inf"), "smoothing must be finite"),
    ],
)
def test_torch_compile_rejects_invalid_static_smoothing_like_eager(smoothing, match):
    points = torch.tensor([[-0.4, -0.2], [0.25, 0.35]])
    controls = _controls(2, dtype=points.dtype)[:4]
    displacements = torch.zeros_like(controls)

    def invalid_smoothing(p, c, d):
        return radial_basis_function_deform_points(
            p,
            c,
            d,
            smoothing=smoothing,
            implementation="torch",
        )

    with pytest.raises(ValueError, match=match):
        invalid_smoothing(points, controls, displacements)
    with pytest.raises(ValueError, match=match):
        torch.compile(invalid_smoothing, backend="eager")(
            points, controls, displacements
        )


def test_torch_compile_rejects_invalid_dynamic_smoothing_without_recompiling():
    points = torch.tensor([[-0.4, -0.2], [0.25, 0.35]])
    controls = _controls(2, dtype=points.dtype)[:4]
    displacements = torch.zeros_like(controls)
    compiled_graphs = []

    def operation(p, c, d, smoothing):
        return radial_basis_function_deform_points(
            p,
            c,
            d,
            smoothing=smoothing,
            implementation="torch",
        )

    def counting_backend(graph_module, _example_inputs):
        compiled_graphs.append(graph_module)
        return graph_module.forward

    compiled = torch.compile(
        operation,
        fullgraph=True,
        dynamic=True,
        backend=counting_backend,
    )
    for smoothing in (0.1, 0.2, 0.0):
        torch.testing.assert_close(
            compiled(points, controls, displacements, smoothing),
            operation(points, controls, displacements, smoothing),
        )
    for smoothing in (-0.1, float("inf"), float("nan")):
        with pytest.raises(RuntimeError):
            compiled(points, controls, displacements, smoothing)
    assert len(compiled_graphs) == 1


@requires_module("warp")
def test_benchmark_forward_generator_contract(device):
    device = torch.device(device)
    labels = []
    for label, args, kwargs in RadialBasisFunctionDeformPoints.make_inputs_forward(
        device=device
    ):
        labels.append(label)
        reduced_args, reduced_kwargs = _trim_benchmark_case(args, kwargs)
        torch_args, torch_kwargs = clone_case(reduced_args, reduced_kwargs)
        warp_args, warp_kwargs = clone_case(reduced_args, reduced_kwargs)
        torch_output = RadialBasisFunctionDeformPoints.dispatch(
            *torch_args, implementation="torch", **torch_kwargs
        )
        warp_output = RadialBasisFunctionDeformPoints.dispatch(
            *warp_args, implementation="warp", **warp_kwargs
        )
        RadialBasisFunctionDeformPoints.compare_forward(warp_output, torch_output)

    assert labels == [
        case[0] for case in RadialBasisFunctionDeformPoints._FORWARD_BENCHMARK_CASES
    ]


@requires_module("warp")
def test_benchmark_backward_generator_contract(device):
    device = torch.device(device)
    labels = []
    for label, args, kwargs in RadialBasisFunctionDeformPoints.make_inputs_backward(
        device=device
    ):
        labels.append(label)
        reduced_args, reduced_kwargs = _trim_benchmark_case(args, kwargs)
        torch_args, torch_kwargs = clone_case(reduced_args, reduced_kwargs)
        warp_args, warp_kwargs = clone_case(reduced_args, reduced_kwargs)
        torch_output = RadialBasisFunctionDeformPoints.dispatch(
            *torch_args, implementation="torch", **torch_kwargs
        )
        warp_output = RadialBasisFunctionDeformPoints.dispatch(
            *warp_args, implementation="warp", **warp_kwargs
        )
        RadialBasisFunctionDeformPoints.compare_forward(warp_output, torch_output)

        torch_tensors = _differentiable_tensors(torch_args, torch_kwargs)
        warp_tensors = _differentiable_tensors(warp_args, warp_kwargs)
        torch_gradients = torch.autograd.grad(
            torch_output.square().mean(), torch_tensors
        )
        warp_gradients = torch.autograd.grad(warp_output.square().mean(), warp_tensors)
        for actual, expected in zip(warp_gradients, torch_gradients, strict=True):
            RadialBasisFunctionDeformPoints.compare_backward(actual, expected)

    assert labels == [
        case[0] for case in RadialBasisFunctionDeformPoints._BACKWARD_BENCHMARK_CASES
    ]
