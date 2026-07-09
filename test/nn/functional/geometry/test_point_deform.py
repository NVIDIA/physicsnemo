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

"""Tests for dense displacement and compact Shepard point deformation."""

import inspect
from typing import Literal, get_type_hints

import pytest
import torch

import physicsnemo.nn.functional as functional
from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.nn.functional import displace_points, morph_points
from physicsnemo.nn.functional.geometry import DisplacePoints, MorphPoints
from physicsnemo.nn.functional.geometry.deform import deform as deform_module
from test.conftest import requires_module
from test.nn.functional._parity_utils import clone_case


def _single_handle_fraction(q: torch.Tensor) -> torch.Tensor:
    """Independent scalar form of the compact Shepard interpolation fraction."""

    phi = (1 - q).pow(4) * (4 * q + 1)
    influence = phi / q.square()
    return influence / (1 + influence)


def _trim_morph_benchmark_case(args, kwargs, max_points=32, max_controls=16):
    """Keep benchmark coverage representative without running benchmark sizes."""

    points, controls, control_displacements = args
    num_controls = min(controls.shape[-2], max_controls)
    trimmed_args = (
        points[..., :max_points, :],
        controls[..., :num_controls, :],
        control_displacements[..., :num_controls, :],
    )
    trimmed_kwargs = dict(kwargs)
    radius = trimmed_kwargs["radius"]
    if isinstance(radius, torch.Tensor):
        trimmed_kwargs["radius"] = radius[..., :num_controls]
    point_weights = trimmed_kwargs["point_weights"]
    if isinstance(point_weights, torch.Tensor):
        trimmed_kwargs["point_weights"] = point_weights[..., :max_points]
    return trimmed_args, trimmed_kwargs


def _differentiable_case_tensors(args, kwargs):
    """Return differentiable tensors in a stable positional/keyword order."""

    values = (*args, kwargs["radius"], kwargs["point_weights"])
    return tuple(
        value
        for value in values
        if isinstance(value, torch.Tensor) and value.requires_grad
    )


def test_public_exports_and_function_specs():
    assert displace_points.__name__ == "displace_points"
    assert morph_points.__name__ == "morph_points"
    assert displace_points.__module__ == (
        "physicsnemo.nn.functional.geometry.deform.deform"
    )
    assert morph_points.__module__ == (
        "physicsnemo.nn.functional.geometry.deform.deform"
    )
    assert issubclass(DisplacePoints, FunctionSpec)
    assert issubclass(MorphPoints, FunctionSpec)
    assert not hasattr(functional, "DisplacePoints")
    assert not hasattr(functional, "MorphPoints")
    assert list(inspect.signature(displace_points).parameters) == [
        "points",
        "displacement",
        "point_weights",
        "implementation",
    ]
    assert list(inspect.signature(morph_points).parameters) == [
        "points",
        "control_points",
        "control_displacements",
        "radius",
        "point_weights",
        "kernel",
        "implementation",
    ]
    assert DisplacePoints.implementations() == ("torch",)
    assert set(MorphPoints.implementations()) == {"torch", "warp"}
    assert get_type_hints(displace_points)["implementation"] == Literal["torch"] | None
    assert get_type_hints(morph_points)["implementation"] == (
        Literal["torch", "warp"] | None
    )


def test_displace_benchmark_cases_and_hooks(device):
    """Every registered benchmark case must be callable and comparable."""

    device = torch.device(device)
    forward_labels = []
    for label, args, kwargs in DisplacePoints.make_inputs_forward(device=device):
        forward_labels.append(label)
        points, displacement = args
        point_weights = kwargs["point_weights"]
        scale = 1 if point_weights is None else point_weights.unsqueeze(-1)
        expected = points + scale * displacement
        output = DisplacePoints.dispatch(*args, implementation="torch", **kwargs)
        DisplacePoints.compare_forward(output, expected)

    assert forward_labels == [
        case[0] for case in DisplacePoints._FORWARD_BENCHMARK_CASES
    ]

    backward_labels = []
    for label, args, kwargs in DisplacePoints.make_inputs_backward(device=device):
        backward_labels.append(label)
        points, displacement = args
        point_weights = kwargs["point_weights"]
        output = DisplacePoints.dispatch(*args, implementation="torch", **kwargs)
        output.sum().backward()

        assert displacement.grad is not None
        DisplacePoints.compare_backward(
            displacement.grad, point_weights.detach().unsqueeze(-1).expand_as(points)
        )
        if points.requires_grad:
            assert points.grad is not None
            DisplacePoints.compare_backward(points.grad, torch.ones_like(points))
        if point_weights.requires_grad:
            assert point_weights.grad is not None
            DisplacePoints.compare_backward(
                point_weights.grad, displacement.detach().sum(dim=-1)
            )

    assert backward_labels == [
        case[0] for case in DisplacePoints._BACKWARD_BENCHMARK_CASES
    ]


@requires_module("warp")
def test_morph_benchmark_forward_cases_and_hooks(device):
    """Run every forward generator branch through reduced Torch/Warp parity cases."""

    device = torch.device(device)
    labels = []
    for label, args, kwargs in MorphPoints.make_inputs_forward(device=device):
        labels.append(label)
        reduced_args, reduced_kwargs = _trim_morph_benchmark_case(args, kwargs)
        args_torch, kwargs_torch = clone_case(reduced_args, reduced_kwargs)
        args_warp, kwargs_warp = clone_case(reduced_args, reduced_kwargs)

        output_torch = MorphPoints.dispatch(
            *args_torch, implementation="torch", **kwargs_torch
        )
        output_warp = MorphPoints.dispatch(
            *args_warp, implementation="warp", **kwargs_warp
        )
        MorphPoints.compare_forward(output_warp, output_torch)

        if label == "exact-handles-n2048-c16-d3":
            points, controls, control_displacements = args_torch
            num_controls = controls.shape[-2]
            distances = torch.cdist(controls, controls)
            diagonal = torch.eye(num_controls, dtype=torch.bool, device=controls.device)
            # At least one coincident query is also in another control's support,
            # so this assertion exercises the exact-handle override, not an
            # isolated-control special case.
            assert ((distances < kwargs_torch["radius"]) & ~diagonal).any()
            expected = points[..., :num_controls, :] + control_displacements
            torch.testing.assert_close(output_torch[..., :num_controls, :], expected)
            torch.testing.assert_close(output_warp[..., :num_controls, :], expected)

    assert labels == [case[0] for case in MorphPoints._FORWARD_BENCHMARK_CASES]


@requires_module("warp")
def test_morph_benchmark_backward_cases_and_hooks(device):
    """Run every backward generator branch through reduced Torch/Warp parity cases."""

    device = torch.device(device)
    labels = []
    for label, args, kwargs in MorphPoints.make_inputs_backward(device=device):
        labels.append(label)
        reduced_args, reduced_kwargs = _trim_morph_benchmark_case(args, kwargs)
        args_torch, kwargs_torch = clone_case(reduced_args, reduced_kwargs)
        args_warp, kwargs_warp = clone_case(reduced_args, reduced_kwargs)

        output_torch = MorphPoints.dispatch(
            *args_torch, implementation="torch", **kwargs_torch
        )
        output_warp = MorphPoints.dispatch(
            *args_warp, implementation="warp", **kwargs_warp
        )
        MorphPoints.compare_forward(output_warp, output_torch)

        tensors_torch = _differentiable_case_tensors(args_torch, kwargs_torch)
        tensors_warp = _differentiable_case_tensors(args_warp, kwargs_warp)
        gradients_torch = torch.autograd.grad(
            output_torch.square().mean(), tensors_torch
        )
        gradients_warp = torch.autograd.grad(output_warp.square().mean(), tensors_warp)
        for gradient_warp, gradient_torch in zip(
            gradients_warp, gradients_torch, strict=True
        ):
            MorphPoints.compare_backward(gradient_warp, gradient_torch)

    assert labels == [case[0] for case in MorphPoints._BACKWARD_BENCHMARK_CASES]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_displace_hard_and_soft_point_weights(dtype):
    points = torch.tensor([[1.0, 2.0], [-1.0, 3.0], [4.0, -2.0]], dtype=dtype)
    displacement = torch.tensor([[2.0, -4.0], [3.0, 5.0], [-1.0, 2.0]], dtype=dtype)

    hard = displace_points(
        points,
        -0.5 * displacement,
        point_weights=torch.tensor([True, False, True]),
        implementation="torch",
    )
    torch.testing.assert_close(
        hard,
        points + torch.tensor([[-1.0, 2.0], [0.0, 0.0], [0.5, -1.0]], dtype=dtype),
    )

    point_weights = torch.tensor([0.25, -1.0, 2.0], dtype=dtype, requires_grad=True)
    soft = displace_points(
        points,
        displacement,
        point_weights=point_weights,
        implementation="torch",
    )
    torch.testing.assert_close(
        soft, points + point_weights.unsqueeze(-1) * displacement
    )
    soft.sum().backward()
    torch.testing.assert_close(point_weights.grad, displacement.sum(dim=-1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_displace_float_weight_forward_is_grad_mode_independent(dtype, device):
    device = torch.device(device)
    points = torch.randn((32, 3), dtype=dtype, device=device, requires_grad=True)
    displacement = torch.randn_like(points, requires_grad=True)
    point_weights = torch.randn(32, dtype=dtype, device=device, requires_grad=True)

    with_grad = displace_points(
        points,
        displacement,
        point_weights=point_weights,
        implementation="torch",
    )
    with torch.no_grad():
        without_grad = displace_points(
            points,
            displacement,
            point_weights=point_weights,
            implementation="torch",
        )

    torch.testing.assert_close(with_grad, without_grad, rtol=0.0, atol=0.0)


def test_morph_kernel_default_and_validation():
    points = torch.tensor([[0.25, 0.0]])
    controls = torch.tensor([[0.0, 0.0]])
    displacements = torch.tensor([[0.0, 1.0]])

    default = morph_points(
        points, controls, displacements, radius=1.0, implementation="torch"
    )
    explicit = morph_points(
        points,
        controls,
        displacements,
        radius=1.0,
        kernel="wendland_c2",
        implementation="torch",
    )
    torch.testing.assert_close(explicit, default)

    with pytest.raises(ValueError, match="kernel.*wendland_c2"):
        morph_points(
            points,
            controls,
            displacements,
            radius=1.0,
            kernel="gaussian",
            implementation="torch",
        )


def test_single_handle_exact_fade_and_support_boundary():
    dtype = torch.float64
    points = torch.tensor(
        [[0.0, 0.0], [0.25, 0.0], [0.5, 0.0], [1.0, 0.0], [1.2, 0.0]],
        dtype=dtype,
    )
    controls = torch.tensor([[0.0, 0.0]], dtype=dtype)
    control_displacements = torch.tensor([[0.0, 2.0]], dtype=dtype)

    output = morph_points(
        points,
        controls,
        control_displacements,
        radius=1.0,
        implementation="torch",
    )
    expected_y = torch.zeros(5, dtype=dtype)
    expected_y[0] = 2.0
    q = torch.tensor([0.25, 0.5], dtype=dtype)
    expected_y[1:3] = 2.0 * _single_handle_fraction(q)

    torch.testing.assert_close(output[:, 0], points[:, 0])
    torch.testing.assert_close(output[:, 1], expected_y)
    assert output[0, 1] > output[1, 1] > output[2, 1] > output[3, 1]
    assert output[3, 1].item() == 0.0
    assert output[4, 1].item() == 0.0


def test_overlapping_controls_use_stationary_background():
    dtype = torch.float64
    points = torch.tensor([[0.5, 0.0]], dtype=dtype)
    controls = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
    displacements = torch.tensor([[0.0, 1.0], [0.0, 3.0]], dtype=dtype)
    output = morph_points(
        points, controls, displacements, radius=1.0, implementation="torch"
    )
    q = torch.tensor(0.5, dtype=dtype)
    phi = (1 - q).pow(4) * (4 * q + 1)
    influence = phi / q.square()
    expected_y = influence * 4 / (1 + 2 * influence)
    expected = torch.stack((torch.tensor(0.5, dtype=dtype), expected_y)).unsqueeze(0)
    torch.testing.assert_close(output, expected)


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_single_handle_analytical_first_derivatives(implementation):
    dtype = torch.float64
    radius = torch.tensor([1.5], dtype=dtype, requires_grad=True)
    q = 0.4
    points = torch.tensor([[q * radius.item(), 0.0]], dtype=dtype, requires_grad=True)
    controls = torch.tensor([[0.0, 0.0]], dtype=dtype, requires_grad=True)
    displacements = torch.tensor([[0.0, 2.0]], dtype=dtype, requires_grad=True)
    point_weights = torch.tensor([1.3], dtype=dtype, requires_grad=True)

    output = morph_points(
        points,
        controls,
        displacements,
        radius=radius,
        point_weights=point_weights,
        implementation=implementation,
    )
    gradients = torch.autograd.grad(
        output[0, 1],
        (points, controls, displacements, radius, point_weights),
    )

    phi = (1 - q) ** 4 * (4 * q + 1)
    phi_prime = -20 * q * (1 - q) ** 3
    fraction = phi / (q**2 + phi)
    fraction_prime = (phi_prime * q**2 - 2 * q * phi) / (q**2 + phi) ** 2
    scale = point_weights.item() * displacements[0, 1].item()
    expected = (
        torch.tensor([[scale * fraction_prime / radius.item(), 1.0]], dtype=dtype),
        torch.tensor([[-scale * fraction_prime / radius.item(), 0.0]], dtype=dtype),
        torch.tensor([[0.0, point_weights.item() * fraction]], dtype=dtype),
        torch.tensor([scale * fraction_prime * (-q / radius.item())], dtype=dtype),
        torch.tensor([displacements[0, 1].item() * fraction], dtype=dtype),
    )
    for actual, reference in zip(gradients, expected):
        torch.testing.assert_close(actual, reference, atol=2e-10, rtol=2e-10)


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_near_handle_and_support_boundary_have_defined_subgradients(implementation):
    dtype = torch.float64
    points = torch.tensor([[1.0e-10, 0.0], [1.0, 0.0]], dtype=dtype, requires_grad=True)
    controls = torch.tensor([[0.0, 0.0]], dtype=dtype, requires_grad=True)
    displacements = torch.tensor([[0.0, 1.0]], dtype=dtype, requires_grad=True)
    radius = torch.tensor([1.0], dtype=dtype, requires_grad=True)
    output = morph_points(
        points,
        controls,
        displacements,
        radius=radius,
        implementation=implementation,
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[1], points[1])

    boundary_gradients = torch.autograd.grad(
        output[1, 1], (points, controls, displacements, radius)
    )
    assert torch.isfinite(
        torch.cat([gradient.reshape(-1) for gradient in boundary_gradients])
    ).all()
    torch.testing.assert_close(
        boundary_gradients[0], torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)
    )
    torch.testing.assert_close(boundary_gradients[1], torch.zeros_like(controls))
    torch.testing.assert_close(boundary_gradients[2], torch.zeros_like(displacements))
    torch.testing.assert_close(boundary_gradients[3], torch.zeros_like(radius))


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_zero_controls_report_zero_gradients(implementation):
    dtype = torch.float64
    points = torch.tensor([[0.2, -0.1]], dtype=dtype, requires_grad=True)
    controls = torch.empty((0, 2), dtype=dtype, requires_grad=True)
    displacements = torch.empty((0, 2), dtype=dtype, requires_grad=True)
    radius = torch.empty(0, dtype=dtype, requires_grad=True)
    point_weights = torch.tensor([1.2], dtype=dtype, requires_grad=True)
    output = morph_points(
        points,
        controls,
        displacements,
        radius=radius,
        point_weights=point_weights,
        implementation=implementation,
    )
    gradients = torch.autograd.grad(
        output.sum(),
        (points, controls, displacements, radius, point_weights),
    )
    torch.testing.assert_close(gradients[0], torch.ones_like(points))
    for gradient in gradients[1:]:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))

    # Radius is unused when there are no controls, so a zero Python radius is a
    # valid identity operation rather than an input-value error.
    identity = morph_points(
        points.detach(),
        controls.detach(),
        displacements.detach(),
        radius=0.0,
        implementation=implementation,
    )
    torch.testing.assert_close(identity, points.detach())


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_exact_duplicate_handles_mean_and_defined_gradients(implementation):
    dtype = torch.float64
    points = torch.tensor([[0.0, 0.0]], dtype=dtype, requires_grad=True)
    controls = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=dtype, requires_grad=True)
    displacements = torch.tensor(
        [[2.0, 1.0], [4.0, 5.0]], dtype=dtype, requires_grad=True
    )
    radius = torch.tensor([0.5, 1.5], dtype=dtype, requires_grad=True)

    output = morph_points(
        points,
        controls,
        displacements,
        radius=radius,
        implementation=implementation,
    )
    torch.testing.assert_close(output, torch.tensor([[3.0, 3.0]], dtype=dtype))

    output[:, 1].sum().backward()
    # The morph field has zero geometry/radius subgradients at coincidences;
    # the identity x term still contributes to the query gradient.
    torch.testing.assert_close(points.grad, torch.tensor([[0.0, 1.0]], dtype=dtype))
    torch.testing.assert_close(controls.grad, torch.zeros_like(controls))
    torch.testing.assert_close(radius.grad, torch.zeros_like(radius))
    torch.testing.assert_close(
        displacements.grad,
        torch.tensor([[0.0, 0.5], [0.0, 0.5]], dtype=dtype),
    )


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_batched_aligned_controls_per_handle_radii_and_point_weights(implementation):
    points = torch.tensor([[[0.0, 0.0], [2.0, 0.0]], [[10.0, 0.0], [12.0, 0.0]]])
    controls = torch.tensor([[[0.0, 0.0]], [[10.0, 0.0]]])
    displacements = torch.tensor([[[0.0, 1.0]], [[0.0, -2.0]]])
    radii = torch.tensor([[1.0], [3.0]])
    point_weights = torch.tensor([[True, True], [False, True]])

    output = morph_points(
        points,
        controls,
        displacements,
        radius=radii,
        point_weights=point_weights,
        implementation=implementation,
    )
    torch.testing.assert_close(output[0, 0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(output[0, 1], points[0, 1])
    torch.testing.assert_close(output[1, 0], points[1, 0])
    assert output[1, 1, 1] < 0.0


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_empty_and_noncontiguous_inputs_preserve_shape_dtype_and_device(
    implementation,
):
    base = torch.arange(24, dtype=torch.float64).reshape(3, 8)
    points = base[:, ::2]
    displacement = torch.ones_like(base)[:, ::2]
    assert not points.is_contiguous()

    displaced = displace_points(points, 0.25 * displacement, implementation="torch")
    torch.testing.assert_close(displaced, points + 0.25)
    assert displaced.shape == points.shape
    assert displaced.dtype == points.dtype
    assert displaced.device == points.device

    empty_controls = torch.empty((0, points.shape[-1]), dtype=points.dtype)
    morphed = morph_points(
        points,
        empty_controls,
        empty_controls.clone(),
        radius=torch.empty(0, dtype=points.dtype),
        implementation=implementation,
    )
    torch.testing.assert_close(morphed, points)

    empty_points = torch.empty((0, 4), dtype=points.dtype)
    empty_output = morph_points(
        empty_points,
        torch.zeros((1, 4), dtype=points.dtype),
        torch.ones((1, 4), dtype=points.dtype),
        radius=1.0,
        implementation=implementation,
    )
    assert empty_output.shape == (0, 4)


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (
            lambda: displace_points(
                torch.zeros(2, 2, dtype=torch.float16),
                torch.zeros(2, 2, dtype=torch.float16),
                implementation="torch",
            ),
            TypeError,
            "float32 or torch.float64",
        ),
        (
            lambda: displace_points(
                torch.zeros(2, 2), torch.zeros(3, 2), implementation="torch"
            ),
            ValueError,
            "identical shapes",
        ),
        (
            lambda: displace_points(
                torch.zeros(2, 2),
                torch.zeros(2, 2),
                point_weights=torch.ones(2, 1),
                implementation="torch",
            ),
            ValueError,
            "point_weights must have shape",
        ),
        (
            lambda: displace_points(
                torch.zeros(2, 2),
                torch.zeros(2, 2),
                point_weights=torch.ones(2, dtype=torch.float64),
                implementation="torch",
            ),
            TypeError,
            "same dtype",
        ),
        (
            lambda: morph_points(
                torch.zeros(2, 2),
                torch.zeros(1, 2),
                torch.zeros(1, 2),
                radius=0.0,
                implementation="torch",
            ),
            ValueError,
            "strictly positive",
        ),
        (
            lambda: morph_points(
                torch.zeros(2, 2),
                torch.zeros(1, 2),
                torch.zeros(1, 2),
                radius=float("inf"),
                implementation="torch",
            ),
            ValueError,
            "finite",
        ),
        (
            lambda: morph_points(
                torch.zeros(2, 2),
                torch.zeros(1, 2),
                torch.zeros(1, 2),
                radius=1.0e100,
                implementation="torch",
            ),
            ValueError,
            "finite in the control dtype",
        ),
        (
            lambda: morph_points(
                torch.zeros(2, 2),
                torch.zeros(1, 2),
                torch.zeros(1, 2),
                radius=1.0e-50,
                implementation="torch",
            ),
            ValueError,
            "positive in the control dtype",
        ),
        (
            lambda: morph_points(
                torch.zeros(2, 2),
                torch.zeros(1, 2),
                torch.zeros(1, 2),
                radius=torch.ones(1, 1),
                implementation="torch",
            ),
            ValueError,
            "scalar or shape",
        ),
        (
            lambda: morph_points(
                torch.zeros(2, 2),
                torch.zeros(1, 2, dtype=torch.float64),
                torch.zeros(1, 2, dtype=torch.float64),
                radius=1.0,
                implementation="torch",
            ),
            TypeError,
            "same dtype",
        ),
        (
            lambda: morph_points(
                torch.zeros(2, 2, 2),
                torch.zeros(1, 1, 2),
                torch.zeros(1, 1, 2),
                radius=1.0,
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


def test_displace_rejects_removed_warp_backend():
    points = torch.zeros(2, 2)
    with pytest.raises(KeyError, match="No implementation named 'warp'"):
        displace_points(points, torch.zeros_like(points), implementation="warp")


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp", None])
def test_morph_points_is_cuda_graph_capture_safe(device, implementation):
    device = torch.device(device)
    if device.type != "cuda":
        pytest.skip("CUDA Graph capture requires CUDA")

    points = torch.rand((8, 3), device=device)
    controls = torch.rand((2, 3), device=device)
    displacements = torch.rand((2, 3), device=device)
    radius = torch.ones(2, device=device)
    point_weights = torch.ones(8, device=device)

    # Warm allocations and backend kernels before capture.
    expected = morph_points(
        points,
        controls,
        displacements,
        radius=radius,
        point_weights=point_weights,
        implementation=implementation,
    )
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = morph_points(
            points,
            controls,
            displacements,
            radius=radius,
            point_weights=point_weights,
            implementation=implementation,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(captured, expected)


def test_morph_torch_double_gradcheck():
    dtype = torch.float64
    points = torch.tensor([[0.2, 0.1], [0.65, -0.15]], dtype=dtype, requires_grad=True)
    controls = torch.tensor([[-0.1, 0.0], [0.9, 0.1]], dtype=dtype, requires_grad=True)
    displacements = torch.tensor(
        [[0.2, -0.3], [-0.1, 0.25]], dtype=dtype, requires_grad=True
    )
    radius = torch.tensor([1.3, 1.1], dtype=dtype, requires_grad=True)
    point_weights = torch.tensor([0.7, -0.4], dtype=dtype, requires_grad=True)

    def operation(p, c, d, r, w):
        return morph_points(
            p,
            c,
            d,
            radius=r,
            point_weights=w,
            implementation="torch",
        )

    assert torch.autograd.gradcheck(
        operation,
        (points, controls, displacements, radius, point_weights),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
    )


def test_torch_chunk_checkpoint_bounds_retained_autograd_storage(monkeypatch):
    """Chunk checkpointing must discard pairwise activations before backward."""
    import gc
    import weakref

    from physicsnemo.nn.functional.geometry.deform import _torch_impl

    # Force several small query/control blocks without making the test expensive.
    monkeypatch.setattr(_torch_impl, "_PAIRWISE_TEMPORARY_BYTE_BUDGET", 32 * 1024)

    def run_and_measure():
        generator = torch.Generator().manual_seed(1234)
        points = torch.randn((1, 128, 3), generator=generator, requires_grad=True)
        controls = torch.randn((1, 16, 3), generator=generator, requires_grad=True)
        displacements = torch.randn((1, 16, 3), generator=generator, requires_grad=True)
        radius = torch.ones((1, 16), requires_grad=True)
        saved = []

        def pack(tensor):
            saved.append(weakref.ref(tensor))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = _torch_impl.compact_shepard_field_torch(
                points, controls, displacements, radius
            )
        gc.collect()
        storages = {
            (
                tensor.untyped_storage().data_ptr(),
                tensor.untyped_storage().nbytes(),
            ): tensor.untyped_storage().nbytes()
            for reference in saved
            if (tensor := reference()) is not None
        }
        # Keep the output graph alive until after storage accounting.
        assert output.grad_fn is not None
        gradients = torch.autograd.grad(
            output.square().sum(), (points, controls, displacements, radius)
        )
        return sum(storages.values()), output.detach(), gradients

    checkpointed_bytes, checkpointed_output, checkpointed_gradients = run_and_measure()

    def run_without_checkpoint(function, *args, **kwargs):
        return function(*args)

    monkeypatch.setattr(_torch_impl, "checkpoint", run_without_checkpoint)
    uncheckpointed_bytes, eager_output, eager_gradients = run_and_measure()

    # This compares retained tensor storage, not allocator timing or RSS. Leave
    # ample margin so changes in small bookkeeping tensors do not make it flaky.
    assert checkpointed_bytes * 10 < uncheckpointed_bytes
    torch.testing.assert_close(checkpointed_output, eager_output)
    for checkpointed_gradient, eager_gradient in zip(
        checkpointed_gradients, eager_gradients
    ):
        torch.testing.assert_close(checkpointed_gradient, eager_gradient)


def test_torch_control_chunked_gradients_match_unchunked_on_ties(monkeypatch):
    """Control chunking must not change gradients on tied minimum distances.

    The first query point is exactly equidistant from the first two controls.
    With one-control chunks that tie straddles a chunk boundary, where the
    reference selection in ``_compact_shepard_query_chunk`` uses strict ``<``
    but the differentiable running minimum uses ``torch.minimum``, whose
    backward splits the gradient 50/50 across exact ties. That routes half of
    the reference-distance gradient to a control the downstream compensation
    algebra assumes has no reference role, corrupting the point, control, and
    radius gradients while forward values and displacement gradients still
    match. Chunking is a byte-budget implementation detail and must be
    gradient-invariant.
    """

    from physicsnemo.nn.functional.geometry.deform import _torch_impl

    dtype = torch.float64
    points = torch.tensor([[0.0, 0.0], [0.1, 0.2]], dtype=dtype)
    controls = torch.tensor([[0.3, 0.4], [-0.3, 0.4], [0.9, -0.7]], dtype=dtype)
    displacements = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=dtype)
    radius = torch.ones(3, dtype=dtype)

    def gradients():
        inputs = tuple(
            tensor.clone().requires_grad_()
            for tensor in (points, controls, displacements, radius)
        )
        output = morph_points(
            inputs[0],
            inputs[1],
            inputs[2],
            radius=inputs[3],
            implementation="torch",
        )
        return torch.autograd.grad(output.sum(), inputs)

    unchunked = gradients()
    monkeypatch.setattr(_torch_impl, "_PAIRWISE_TEMPORARY_BYTE_BUDGET", 1)
    chunked = gradients()

    for chunked_gradient, unchunked_gradient in zip(chunked, unchunked):
        torch.testing.assert_close(chunked_gradient, unchunked_gradient)


def test_compiled_torch_training_checkpoints_pairwise_activations():
    """The symbolic compile path must not retain the full pairwise graph."""

    import gc
    import weakref

    from physicsnemo.nn.functional.geometry.deform._torch_impl import (
        compact_shepard_field_torch,
    )

    def operation(points, controls, displacements, radius):
        return compact_shepard_field_torch(points, controls, displacements, radius)

    # Checkpoint tags are consumed by AOTAutograd. Dynamo's debug ``eager``
    # backend did not honor them before PyTorch 2.12.
    compiled = torch.compile(operation, fullgraph=True, backend="aot_eager")

    def make_inputs():
        generator = torch.Generator().manual_seed(4321)
        return (
            torch.randn((1, 128, 3), generator=generator, requires_grad=True),
            torch.randn((1, 16, 3), generator=generator, requires_grad=True),
            torch.randn((1, 16, 3), generator=generator, requires_grad=True),
            torch.ones((1, 16), requires_grad=True),
        )

    warm_inputs = make_inputs()
    compiled(*warm_inputs).sum().backward()
    del warm_inputs
    gc.collect()

    saved: list[weakref.ReferenceType[torch.Tensor]] = []

    def pack(tensor):
        saved.append(weakref.ref(tensor))
        return tensor

    inputs = make_inputs()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = compiled(*inputs)
    gc.collect()
    storages = {
        (tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes()): (
            tensor.untyped_storage().nbytes()
        )
        for reference in saved
        if (tensor := reference()) is not None
    }
    # Inputs plus checkpoint metadata are a few KiB. Retaining one full
    # pairwise graph for this case is roughly 0.5 MiB.
    assert sum(storages.values()) < 32 * 1024
    gradients = torch.autograd.grad(output.square().sum(), inputs)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@requires_module("warp")
def test_morph_warp_double_gradcheck():
    dtype = torch.float64
    inputs = (
        torch.tensor([[0.17, 0.11], [0.66, -0.13]], dtype=dtype, requires_grad=True),
        torch.tensor([[-0.1, -0.05], [0.91, 0.08]], dtype=dtype, requires_grad=True),
        torch.tensor([[0.12, 0.3], [-0.2, 0.09]], dtype=dtype, requires_grad=True),
        torch.tensor([0.83, 1.07], dtype=dtype, requires_grad=True),
        torch.tensor([0.8, -0.4], dtype=dtype, requires_grad=True),
    )

    def operation(p, c, d, r, w):
        return morph_points(
            p,
            c,
            d,
            radius=r,
            point_weights=w,
            implementation="warp",
        )

    assert torch.autograd.gradcheck(operation, inputs, eps=1e-6, atol=3e-5, rtol=3e-4)


def _run_morph_with_gradients(implementation, device, dtype):
    points = torch.tensor(
        [[0.2, 0.1], [0.55, -0.15], [1.15, 0.2]], device=device, dtype=dtype
    ).requires_grad_()
    controls = torch.tensor(
        [[-0.1, 0.0], [0.9, 0.1]], device=device, dtype=dtype
    ).requires_grad_()
    displacements = torch.tensor(
        [[0.2, -0.3], [-0.1, 0.25]], device=device, dtype=dtype
    ).requires_grad_()
    radius = torch.tensor([1.3, 1.1], device=device, dtype=dtype, requires_grad=True)
    point_weights = torch.tensor(
        [0.7, -0.4, 1.2], device=device, dtype=dtype, requires_grad=True
    )
    output = morph_points(
        points,
        controls,
        displacements,
        radius=radius,
        point_weights=point_weights,
        implementation=implementation,
    )
    cotangent = torch.tensor(
        [[0.2, -0.5], [-0.3, 0.1], [0.7, -0.4]], device=device, dtype=dtype
    )
    gradients = torch.autograd.grad(
        (output * cotangent).sum(),
        (points, controls, displacements, radius, point_weights),
    )
    return output, gradients


@requires_module("warp")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_morph_torch_warp_forward_and_first_gradient_parity(device, dtype):
    device = torch.device(device)
    morph_torch = _run_morph_with_gradients("torch", device, dtype)
    morph_warp = _run_morph_with_gradients("warp", device, dtype)

    if dtype == torch.float32:
        atol, rtol = 4e-5, 4e-5
    else:
        atol, rtol = 2e-9, 2e-8
    torch.testing.assert_close(morph_warp[0], morph_torch[0], atol=atol, rtol=rtol)
    for actual, expected in zip(morph_warp[1], morph_torch[1]):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_default_dispatch_selects_device_backend(device, monkeypatch):
    device = torch.device(device)
    points = torch.tensor([[0.2, 0.1], [0.7, 0.0]], device=device)
    controls = torch.tensor([[0.0, 0.0]], device=device)
    displacement = torch.tensor([[0.0, 1.0]], device=device)
    calls = []

    def torch_spy(normalized_points, *_args):
        calls.append("torch")
        return normalized_points

    def warp_spy(normalized_points, *_args):
        calls.append("warp")
        return normalized_points

    # Patch the names resolved by the registered methods, rather than the source
    # implementation modules, to assert which custom dispatch branch ran.
    monkeypatch.setattr(deform_module, "morph_points_torch", torch_spy)
    monkeypatch.setattr(deform_module, "morph_points_warp", warp_spy)

    warp_impl = MorphPoints._get_impls()["warp"]
    expected = "warp" if device.type == "cuda" and warp_impl.available else "torch"
    automatic = morph_points(points, controls, displacement, radius=1.0)
    assert calls == [expected]
    torch.testing.assert_close(automatic, points)

    if device.type == "cuda" and warp_impl.available:
        # CUDA must still fall back to Torch, with the standard one-time
        # warning, if the optional backend is unavailable.
        calls.clear()
        unavailable_warp = type(warp_impl)(
            name=warp_impl.name,
            func=warp_impl.func,
            required_imports=warp_impl.required_imports,
            rank=warp_impl.rank,
            baseline=warp_impl.baseline,
            available=False,
        )
        monkeypatch.setitem(MorphPoints._get_impls(), "warp", unavailable_warp)
        FunctionSpec._fallback_warned.discard(MorphPoints._class_key())
        with pytest.warns(RuntimeWarning, match="falling back to implementation"):
            automatic = morph_points(points, controls, displacement, radius=1.0)
        assert calls == ["torch"]
        torch.testing.assert_close(automatic, points)


@requires_module("warp")
def test_warp_custom_ops_opcheck():
    from physicsnemo.nn.functional.geometry.deform._warp_impl import (
        compact_shepard_field_warp_impl,
    )

    dtype = torch.float64
    points = torch.tensor([[[0.2, 0.1], [0.7, -0.2]]], dtype=dtype, requires_grad=True)
    controls = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=dtype, requires_grad=True)
    displacements = torch.tensor(
        [[[0.0, 0.3], [0.2, -0.1]]], dtype=dtype, requires_grad=True
    )
    radius = torch.tensor([[0.9, 1.1]], dtype=dtype, requires_grad=True)
    torch.library.opcheck(
        compact_shepard_field_warp_impl,
        args=(points, controls, displacements, radius),
    )


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp", None])
def test_torch_compile_fullgraph(implementation):
    points = torch.tensor([[0.2, 0.1], [0.6, -0.1]])
    controls = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    displacements = torch.tensor([[0.0, 0.3], [0.2, -0.1]])
    radius = torch.tensor([0.8, 1.1])
    point_weights = torch.tensor([0.5, 1.2])

    def operation(p, c, d, r, w):
        dense = displace_points(p, d, point_weights=w, implementation="torch")
        sparse = morph_points(
            p,
            c,
            d,
            radius=r,
            point_weights=w,
            implementation=implementation,
        )
        return dense, sparse

    eager = operation(points, controls, displacements, radius, point_weights)
    compiled = torch.compile(operation, fullgraph=True)(
        points, controls, displacements, radius, point_weights
    )
    for actual, reference in zip(compiled, eager):
        torch.testing.assert_close(actual, reference)


def test_torch_compile_fullgraph_dynamic_shapes():
    """Symbolic query and control counts use the vectorized compile path."""

    def operation(points, controls, displacements, radius):
        return morph_points(
            points,
            controls,
            displacements,
            radius=radius,
            implementation="torch",
        )

    compiled = torch.compile(operation, fullgraph=True, dynamic=True, backend="eager")
    generator = torch.Generator().manual_seed(3141)
    for num_points, num_controls in ((4, 2), (7, 3)):
        points = torch.randn((num_points, 3), generator=generator)
        controls = torch.randn((num_controls, 3), generator=generator)
        displacements = torch.randn((num_controls, 3), generator=generator)
        radius = torch.ones(num_controls)
        torch.testing.assert_close(
            compiled(points, controls, displacements, radius),
            operation(points, controls, displacements, radius),
        )


def test_torch_compile_fullgraph_dynamic_python_scalars_and_defaults():
    """Valid Python scalar options remain traceable in a full graph."""

    def defaults(points, controls, displacements):
        return morph_points(
            points,
            controls,
            displacements,
            radius=1.0,
            implementation="torch",
        )

    def runtime_scalar(points, controls, displacements, radius):
        return morph_points(
            points,
            controls,
            displacements,
            radius=radius,
            implementation="torch",
        )

    points = torch.randn((4, 3))
    controls = torch.randn((2, 3))
    displacements = torch.randn((2, 3))
    compiled_graphs = []

    def counting_backend(graph_module, _example_inputs):
        compiled_graphs.append(graph_module)
        return graph_module.forward

    compiled_defaults = torch.compile(
        defaults, fullgraph=True, dynamic=True, backend="eager"
    )
    compiled_scalar = torch.compile(
        runtime_scalar, fullgraph=True, dynamic=True, backend=counting_backend
    )
    torch.testing.assert_close(
        compiled_defaults(points, controls, displacements),
        defaults(points, controls, displacements),
    )
    # Multiple values force Dynamo to generalize the Python float to SymFloat;
    # the symbolic path must remain a single full graph.
    for radius in (1.0, 2.0, 0.5):
        torch.testing.assert_close(
            compiled_scalar(points, controls, displacements, radius),
            runtime_scalar(points, controls, displacements, radius),
        )

    # Value checks are intentionally skipped once a call-time scalar has been
    # generalized to SymFloat. Invalid symbolic values must follow the same
    # unvalidated numerical path as tensor radii rather than graph-breaking.
    for radius in (0.0, -1.0):
        torch.testing.assert_close(
            compiled_scalar(points, controls, displacements, radius),
            morph_points(
                points,
                controls,
                displacements,
                radius=torch.tensor(radius),
                implementation="torch",
            ),
        )
    assert len(compiled_graphs) == 1


@pytest.mark.parametrize(
    ("radius", "match"),
    [
        (0.0, "radius must be strictly positive"),
        (-1.0, "radius must be strictly positive"),
        (float("nan"), "radius must be finite"),
        (float("inf"), "radius must be finite"),
    ],
)
def test_torch_compile_rejects_invalid_python_scalars_like_eager(radius, match):
    """Compiled execution must reject the same invalid Python scalars as eager.

    Python radius literals must not bypass finite and strictly-positive checks
    merely because Dynamo is tracing the call.
    """

    points = torch.randn((5, 2))
    controls = torch.randn((3, 2))
    displacements = torch.randn((3, 2))

    def invalid_radius(p, c, d):
        return morph_points(p, c, d, radius=radius, implementation="torch")

    with pytest.raises(ValueError, match=match):
        invalid_radius(points, controls, displacements)
    with pytest.raises(ValueError, match=match):
        torch.compile(invalid_radius, backend="eager")(points, controls, displacements)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bool_mask_suppresses_nonfinite_dense_displacements(dtype, device):
    device = torch.device(device)
    points = torch.tensor(
        [[1.0, 2.0], [-1.0, 0.5], [3.0, -2.0]], device=device, dtype=dtype
    ).requires_grad_()
    displacement = torch.tensor(
        [[float("nan"), float("inf")], [2.0, -3.0], [-float("inf"), float("nan")]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    mask = torch.tensor([False, True, False], device=device)

    output = displace_points(
        points,
        displacement,
        point_weights=mask,
        implementation="torch",
    )
    expected = points.detach().clone()
    expected[1] += torch.tensor([2.0, -3.0], device=device, dtype=dtype)
    torch.testing.assert_close(output, expected)
    gradients = torch.autograd.grad(output.sum(), (points, displacement))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    torch.testing.assert_close(
        gradients[1][~mask], torch.zeros((2, 2), device=device, dtype=dtype)
    )


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize(
    ("dtype", "scale"),
    [(torch.float32, 1.0e30), (torch.float64, 1.0e200)],
)
def test_large_finite_coordinates_and_radius_have_finite_gradients(
    implementation, dtype, scale, device
):
    device = torch.device(device)
    points = torch.tensor(
        [[scale, 0.0]], device=device, dtype=dtype, requires_grad=True
    )
    controls = torch.tensor(
        [[0.0, 0.0]], device=device, dtype=dtype, requires_grad=True
    )
    displacements = torch.tensor(
        [[0.2, -0.3]], device=device, dtype=dtype, requires_grad=True
    )
    radius = torch.tensor([2 * scale], device=device, dtype=dtype, requires_grad=True)

    output = morph_points(
        points,
        controls,
        displacements,
        radius=radius,
        implementation=implementation,
    )
    gradients = torch.autograd.grad(
        output.sum(), (points, controls, displacements, radius)
    )
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize(
    ("dtype", "epsilon"),
    [(torch.float32, 1.0e-20), (torch.float64, 1.0e-160)],
)
def test_tied_ultra_near_axis_aligned_handles_have_parallel_finite_gradients(
    implementation, dtype, epsilon, device
):
    device = torch.device(device)
    points = torch.tensor([[0.0, 0.0]], device=device, dtype=dtype, requires_grad=True)
    controls = torch.tensor(
        [[-epsilon, 0.0], [epsilon, 0.0]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    displacements = torch.tensor(
        [[epsilon, 0.0], [-epsilon, 0.0]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    radius = torch.ones(2, device=device, dtype=dtype, requires_grad=True)

    output = morph_points(
        points,
        controls,
        displacements,
        radius=radius,
        implementation=implementation,
    )
    gradients = torch.autograd.grad(
        output[0, 0], (points, controls, displacements, radius)
    )
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    # With every input and cotangent on the x axis, vector gradients must stay
    # parallel to that axis even when the two minimum-distance handles tie.
    for gradient in gradients[:3]:
        torch.testing.assert_close(gradient[..., 1], torch.zeros_like(gradient[..., 1]))


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_subnormal_float64_radius_has_finite_nonzero_radius_gradient(
    implementation, device
):
    device = torch.device(device)
    dtype = torch.float64
    points = torch.tensor(
        [[5.0e-311, 0.0]], device=device, dtype=dtype, requires_grad=True
    )
    controls = torch.tensor(
        [[0.0, 0.0]], device=device, dtype=dtype, requires_grad=True
    )
    displacements = torch.tensor(
        [[1.0e-310, 0.0]], device=device, dtype=dtype, requires_grad=True
    )
    radius = torch.tensor([1.0e-310], device=device, dtype=dtype, requires_grad=True)

    output = morph_points(
        points,
        controls,
        displacements,
        radius=radius,
        implementation=implementation,
    )
    (radius_gradient,) = torch.autograd.grad(output[0, 0], (radius,))
    assert torch.isfinite(output).all()
    assert torch.isfinite(radius_gradient).all()
    assert torch.count_nonzero(radius_gradient) == 1


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_handle_extreme_radius_has_zero_finite_geometry_gradients(
    implementation, dtype, device
):
    device = torch.device(device)
    points = torch.tensor([[1.0, -2.0]], device=device, dtype=dtype, requires_grad=True)
    controls = points.detach().clone().requires_grad_()
    displacements = torch.tensor(
        [[0.25, 0.5]], device=device, dtype=dtype, requires_grad=True
    )
    radius = torch.tensor(
        [torch.finfo(dtype).max], device=device, dtype=dtype, requires_grad=True
    )

    output = morph_points(
        points,
        controls,
        displacements,
        radius=radius,
        implementation=implementation,
    )
    point_gradient, control_gradient, radius_gradient = torch.autograd.grad(
        output.sum(), (points, controls, radius)
    )
    torch.testing.assert_close(point_gradient, torch.ones_like(points))
    torch.testing.assert_close(control_gradient, torch.zeros_like(controls))
    torch.testing.assert_close(radius_gradient, torch.zeros_like(radius))
    assert torch.isfinite(output).all()
    assert torch.isfinite(point_gradient).all()
    assert torch.isfinite(control_gradient).all()
    assert torch.isfinite(radius_gradient).all()


@requires_module("warp")
def test_infinite_tensor_radius_agrees_across_backends():
    """Backends must agree when an unvalidated tensor radius contains ``inf``.

    Tensor radius values are documented as unvalidated, so ``radius=inf``
    must at least behave consistently. The infinite-radius limit of the
    compact Shepard field applies the full control displacement everywhere
    (``q -> 0``), which is what the Warp backend and the Torch backend at any
    large finite radius already produce; the Torch backend must not jump to
    zero influence exactly at ``inf``.
    """

    dtype = torch.float64
    points = torch.tensor([[0.5, 0.0], [3.0, 4.0]], dtype=dtype)
    controls = torch.tensor([[0.0, 0.0]], dtype=dtype)
    displacements = torch.tensor([[0.0, 1.0]], dtype=dtype)

    def run(radius_value, implementation):
        return morph_points(
            points,
            controls,
            displacements,
            radius=torch.tensor([radius_value], dtype=dtype),
            implementation=implementation,
        )

    torch_inf = run(torch.inf, "torch")
    warp_inf = run(torch.inf, "warp")
    torch.testing.assert_close(torch_inf, warp_inf)
    torch.testing.assert_close(torch_inf, run(1e300, "torch"))


@requires_module("warp")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_infinite_radius_with_overflowing_difference_agrees_across_backends(dtype):
    """An infinite radius overrides the finite-radius overflow sentinel."""

    coordinate = 0.75 * torch.finfo(dtype).max
    points = torch.tensor([[coordinate, 0.0]], dtype=dtype)
    controls = torch.tensor([[-coordinate, 0.0]], dtype=dtype)
    displacements = torch.tensor([[0.0, 1.0]], dtype=dtype)
    radius = torch.tensor([torch.inf], dtype=dtype)

    def run(implementation):
        inputs = tuple(
            tensor.detach().clone().requires_grad_()
            for tensor in (points, controls, displacements, radius)
        )
        output = morph_points(
            inputs[0],
            inputs[1],
            inputs[2],
            radius=inputs[3],
            implementation=implementation,
        )
        return output, torch.autograd.grad(output.sum(), inputs)

    torch_output, torch_gradients = run("torch")
    warp_output, warp_gradients = run("warp")

    torch.testing.assert_close(torch_output, warp_output)
    torch.testing.assert_close(torch_output, points + displacements)
    for torch_gradient, warp_gradient in zip(torch_gradients, warp_gradients):
        torch.testing.assert_close(torch_gradient, warp_gradient)


@requires_module("warp")
@pytest.mark.parametrize("morph_implementation", ["torch", "warp"])
def test_public_api_fake_tensor_propagation_with_tensor_options(
    morph_implementation,
):
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    with FakeTensorMode():
        points = torch.empty((4, 3), dtype=torch.float64)
        displacement = torch.empty_like(points)
        controls = torch.empty((2, 3), dtype=torch.float64)
        control_displacements = torch.empty_like(controls)
        radius = torch.empty(2, dtype=torch.float64)
        point_weights = torch.empty(4, dtype=torch.float64)

        dense = displace_points(
            points,
            displacement,
            point_weights=point_weights,
            implementation="torch",
        )
        sparse = morph_points(
            points,
            controls,
            control_displacements,
            radius=radius,
            point_weights=point_weights,
            implementation=morph_implementation,
        )

    for output in (dense, sparse):
        assert isinstance(output, FakeTensor)
        assert output.shape == points.shape
        assert output.dtype == points.dtype
        assert output.device == points.device


@requires_module("warp")
def test_warp_raw_custom_op_fake_strides_match_noncontiguous_real_outputs(device):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from physicsnemo.nn.functional.geometry.deform._warp_impl import (
        compact_shepard_field_warp_impl,
    )

    device = torch.device(device)
    points = torch.rand((1, 3, 5), device=device).transpose(1, 2)
    controls = torch.rand((1, 3, 2), device=device).transpose(1, 2)
    control_displacements = torch.rand_like(controls)
    radius = torch.ones((1, 2), device=device)

    real_sparse = compact_shepard_field_warp_impl(
        points, controls, control_displacements, radius
    )

    with FakeTensorMode() as mode:
        fake_sparse = compact_shepard_field_warp_impl(
            mode.from_tensor(points),
            mode.from_tensor(controls),
            mode.from_tensor(control_displacements),
            mode.from_tensor(radius),
        )

    assert [tensor.stride() for tensor in fake_sparse] == [
        tensor.stride() for tensor in real_sparse
    ]


@requires_module("warp")
def test_warp_no_grad_uses_forward_only_field(monkeypatch, device):
    from physicsnemo.nn.functional.geometry.deform._warp_impl import op as warp_op

    device = torch.device(device)
    points = torch.tensor(
        [[[0.2, 0.1], [0.7, -0.1]]], device=device, requires_grad=True
    )
    controls = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], device=device)
    displacements = torch.tensor([[[0.0, 0.3], [0.2, -0.1]]], device=device)
    radius = torch.tensor([[0.8, 1.1]], device=device)

    expected = warp_op.morph_points_warp(points, controls, displacements, radius, None)

    def fail_full_field(*args, **kwargs):
        raise AssertionError("backward auxiliaries must not be built in no-grad mode")

    monkeypatch.setattr(warp_op, "compact_shepard_field_warp_impl", fail_full_field)
    with torch.no_grad():
        actual = warp_op.morph_points_warp(
            points, controls, displacements, radius, None
        )
    torch.testing.assert_close(actual, expected)


@requires_module("warp")
def test_warp_zero_controls_bypass_launches_and_keep_zero_gradients(
    monkeypatch, device
):
    from physicsnemo.nn.functional.geometry.deform._warp_impl import op as warp_op

    device = torch.device(device)
    points = torch.randn((1, 4, 3), device=device, requires_grad=True)
    controls = torch.empty((1, 0, 3), device=device, requires_grad=True)
    displacements = torch.empty_like(controls, requires_grad=True)
    radius = torch.empty((1, 0), device=device, requires_grad=True)
    point_weights = torch.rand((1, 4), device=device, requires_grad=True)

    def fail_launch(*args, **kwargs):
        raise AssertionError("zero controls must bypass Warp custom ops")

    monkeypatch.setattr(warp_op, "compact_shepard_field_warp_impl", fail_launch)
    monkeypatch.setattr(
        warp_op, "compact_shepard_field_warp_forward_only_impl", fail_launch
    )
    output = warp_op.morph_points_warp(
        points, controls, displacements, radius, point_weights
    )
    gradients = torch.autograd.grad(
        output.sum(),
        (points, controls, displacements, radius, point_weights),
    )

    torch.testing.assert_close(output, points)
    torch.testing.assert_close(gradients[0], torch.ones_like(points))
    for gradient in gradients[1:]:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))


@requires_module("warp")
def test_warp_control_displacement_only_pullback_returns_requested_gradient(device):
    from physicsnemo.nn.functional.geometry.deform._warp_impl import op as warp_op

    device = torch.device(device)
    points = torch.rand((1, 5, 3), device=device)
    controls = torch.rand((1, 2, 3), device=device)
    control_displacements = torch.rand_like(controls)
    radius = torch.ones((1, 2), device=device)
    grad_output = torch.rand_like(points)

    field, min_q, denominator, exact_count, reference_index, correction = (
        warp_op.compact_shepard_field_warp_impl(
            points, controls, control_displacements, radius, False
        )
    )
    assert correction.numel() == 0
    sparse_gradients = warp_op.compact_shepard_field_warp_backward_impl(
        grad_output,
        points,
        controls,
        None,
        radius,
        min_q,
        denominator,
        exact_count,
        reference_index,
        None,
        False,
        False,
        True,
        False,
    )
    assert sparse_gradients[:2] == (None, None)
    assert sparse_gradients[2] is not None
    assert sparse_gradients[3] is None
    assert field.shape == points.shape
