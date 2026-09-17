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

"""Tests for lattice free-form point deformation."""

import importlib
import inspect
from typing import Literal, get_type_hints

import pytest
import torch

import physicsnemo.nn.functional as functional
from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.nn.functional import free_form_deform_points
from physicsnemo.nn.functional.geometry import FreeFormDeformPoints
from physicsnemo.nn.functional.geometry.deform import ffd as ffd_module
from test.conftest import requires_module
from test.nn.functional._parity_utils import clone_case

_INTERPOLATING_BASES = ("linear", "cubic_hermite", "quintic_hermite")
_BASES = ("bernstein", "bspline", *_INTERPOLATING_BASES)
_AFFINE_EXACT_BASES = ("bernstein", "bspline", "linear")
_EXPECTED_FFD_BASIS = Literal[
    "bernstein", "bspline", "linear", "cubic_hermite", "quintic_hermite"
]


def _lattice_nodes(resolution, origin, extent, basis, dtype, device):
    """Return world-space lattice nodes that reproduce affine fields.

    Bernstein interpolation reproduces an affine map sampled at the uniform
    lattice nodes. A uniform cubic B-spline reproduces it when sampled at the
    Greville abscissae ``(i - 1) / (n - 3)``.
    """

    axes = []
    for size, origin_d, extent_d in zip(resolution, origin, extent):
        if basis != "bspline":
            positions = torch.linspace(0.0, 1.0, size, dtype=dtype, device=device)
        else:
            positions = (torch.arange(size, dtype=dtype, device=device) - 1) / (
                size - 3
            )
        axes.append(origin_d + extent_d * positions)
    grids = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(grids, dim=-1)


def _trim_ffd_benchmark_case(args, kwargs, max_points=32):
    """Keep benchmark coverage representative without running benchmark sizes."""

    points, control_displacements = args
    trimmed_args = (points[..., :max_points, :], control_displacements)
    trimmed_kwargs = dict(kwargs)
    point_weights = trimmed_kwargs["point_weights"]
    if isinstance(point_weights, torch.Tensor):
        trimmed_kwargs["point_weights"] = point_weights[..., :max_points]
    return trimmed_args, trimmed_kwargs


def _differentiable_case_tensors(args, kwargs):
    """Return differentiable tensors in a stable positional/keyword order."""

    values = (*args, kwargs["point_weights"])
    return tuple(
        value
        for value in values
        if isinstance(value, torch.Tensor) and value.requires_grad
    )


def test_public_exports_and_function_specs():
    geometry = importlib.import_module("physicsnemo.nn.functional.geometry")
    deform = importlib.import_module("physicsnemo.nn.functional.geometry.deform")

    assert functional.free_form_deform_points is free_form_deform_points
    assert geometry.free_form_deform_points is free_form_deform_points
    assert deform.free_form_deform_points is free_form_deform_points
    assert geometry.FreeFormDeformPoints is FreeFormDeformPoints
    assert deform.FreeFormDeformPoints is FreeFormDeformPoints
    assert free_form_deform_points.__name__ == "free_form_deform_points"
    assert (
        free_form_deform_points.__module__
        == "physicsnemo.nn.functional.geometry.deform.ffd"
    )
    assert issubclass(FreeFormDeformPoints, FunctionSpec)
    assert not hasattr(functional, "FreeFormDeformPoints")
    for module in (functional, geometry, deform):
        assert "free_form_deform_points" in module.__all__
        assert "ffd_points" not in module.__all__
        assert not hasattr(module, "ffd_points")
    assert not hasattr(geometry, "FFDPoints")
    assert not hasattr(deform, "FFDPoints")
    assert list(inspect.signature(free_form_deform_points).parameters) == [
        "points",
        "control_displacements",
        "origin",
        "extent",
        "basis",
        "point_weights",
        "implementation",
    ]
    assert set(FreeFormDeformPoints.implementations()) == {"torch", "warp"}
    assert get_type_hints(free_form_deform_points)["implementation"] == (
        Literal["torch", "warp"] | None
    )
    assert get_type_hints(free_form_deform_points)["basis"] == _EXPECTED_FFD_BASIS


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("basis", _BASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ffd_zero_displacements_is_identity(device, implementation, basis, dtype):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    for resolution in ((4, 5), (4, 4, 5)):
        num_dims = len(resolution)
        generator = torch.Generator(device=device).manual_seed(11)
        points = torch.rand(
            (64, num_dims), generator=generator, device=device, dtype=dtype
        )
        control_displacements = torch.zeros(
            (*resolution, num_dims), device=device, dtype=dtype
        )
        output = free_form_deform_points(
            points,
            control_displacements,
            origin=[0.0] * num_dims,
            extent=[1.0] * num_dims,
            basis=basis,
            implementation=implementation,
        )
        assert torch.equal(output, points)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("basis", _BASES)
def test_ffd_constant_displacement_is_exact_translation(device, implementation, basis):
    """Partition of unity: a constant lattice translates every inside point."""

    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    dtype = torch.float64
    generator = torch.Generator(device=device).manual_seed(13)
    points = torch.rand((128, 3), generator=generator, device=device, dtype=dtype)
    # Include both box corners.
    points[0] = 0.0
    points[1] = 1.0
    translation = torch.tensor([0.1, -0.2, 0.3], device=device, dtype=dtype)
    control_displacements = translation.expand(5, 4, 6, 3).clone()
    output = free_form_deform_points(
        points,
        control_displacements,
        origin=[0.0, 0.0, 0.0],
        extent=[1.0, 1.0, 1.0],
        basis=basis,
        implementation=implementation,
    )
    torch.testing.assert_close(output, points + translation)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("basis", _AFFINE_EXACT_BASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ffd_linear_precision(device, implementation, basis, dtype):
    """Sampling an affine map on the lattice reproduces the affine map exactly."""

    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    resolution = (4, 6, 5)
    origin = (-0.5, 0.25, 0.0)
    extent = (2.0, 1.5, 1.0)
    generator = torch.Generator(device=device).manual_seed(17)
    linear = 0.1 * torch.randn((3, 3), generator=generator, device=device, dtype=dtype)
    offset = 0.1 * torch.randn((3,), generator=generator, device=device, dtype=dtype)
    nodes = _lattice_nodes(resolution, origin, extent, basis, dtype, device)
    control_displacements = nodes @ linear.mT + offset
    points = torch.rand(
        (256, 3), generator=generator, device=device, dtype=dtype
    ) * torch.tensor(extent, device=device, dtype=dtype) + torch.tensor(
        origin, device=device, dtype=dtype
    )
    output = free_form_deform_points(
        points,
        control_displacements,
        origin=list(origin),
        extent=list(extent),
        basis=basis,
        implementation=implementation,
    )
    expected = points + points @ linear.mT + offset
    if dtype == torch.float32:
        torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)
    else:
        torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("basis", _INTERPOLATING_BASES)
@pytest.mark.parametrize("num_dims", [1, 2, 3])
def test_interpolating_bases_reproduce_every_lattice_node(
    device, implementation, basis, num_dims
):
    """Local interpolating bases reproduce every control displacement exactly."""

    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    dtype = torch.float64
    resolution = tuple(range(3, 3 + num_dims))
    axes = [
        torch.linspace(0.0, 1.0, size, device=device, dtype=dtype)
        for size in resolution
    ]
    points = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(
        -1, num_dims
    )
    generator = torch.Generator(device=device).manual_seed(18 + num_dims)
    control_displacements = 0.1 * torch.randn(
        (*resolution, num_dims), generator=generator, device=device, dtype=dtype
    )

    output = free_form_deform_points(
        points,
        control_displacements,
        origin=[0.0] * num_dims,
        extent=[1.0] * num_dims,
        basis=basis,
        implementation=implementation,
    )

    torch.testing.assert_close(
        output, points + control_displacements.reshape(-1, num_dims)
    )


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("basis", _INTERPOLATING_BASES)
def test_interpolating_bases_match_piecewise_one_dimensional_oracle(
    device, implementation, basis
):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    dtype = torch.float64
    points = torch.tensor(
        [[0.125], [0.375], [0.625], [0.875]], device=device, dtype=dtype
    )
    control_displacements = torch.tensor(
        [[0.0], [2.0], [-1.0]], device=device, dtype=dtype
    )
    scaled = points[:, 0] * 2
    cell = scaled.floor().clamp_max(1).to(torch.long)
    t = scaled - cell
    if basis == "linear":
        upper = t
    elif basis == "cubic_hermite":
        upper = t * t * (3 - 2 * t)
    else:
        upper = t * t * t * (t * (6 * t - 15) + 10)
    field = (1 - upper) * control_displacements[
        cell, 0
    ] + upper * control_displacements[cell + 1, 0]

    output = free_form_deform_points(
        points,
        control_displacements,
        origin=[0.0],
        extent=[1.0],
        basis=basis,
        implementation=implementation,
    )

    torch.testing.assert_close(output[:, 0], points[:, 0] + field)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("basis", _INTERPOLATING_BASES)
def test_interpolating_bases_have_two_node_per_axis_local_support(
    device, implementation, basis
):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    points = torch.tensor([[0.8, 0.8]], device=device)
    control_displacements = torch.zeros((4, 4, 2), device=device)
    control_displacements[0, 0] = torch.tensor([1.0, -1.0], device=device)

    output = free_form_deform_points(
        points,
        control_displacements,
        origin=[0.0, 0.0],
        extent=[1.0, 1.0],
        basis=basis,
        implementation=implementation,
    )

    assert torch.equal(output, points)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("num_dims", [1, 2, 3])
def test_linear_upper_boundary_uses_final_cell_value_and_gradient(
    device, implementation, num_dims
):
    """Inclusive upper faces retain the final interior linear stencil."""

    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    dtype = torch.float32
    origin = torch.tensor(
        [-1.0 - axis for axis in range(num_dims)], device=device, dtype=dtype
    )
    maximum = torch.tensor(
        [2.0 + axis for axis in range(num_dims)], device=device, dtype=dtype
    )
    extent = maximum - origin
    axes = [
        torch.linspace(origin[d], maximum[d], 3, device=device, dtype=dtype)
        for d in range(num_dims)
    ]
    nodes = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    matrix = 0.05 * torch.arange(
        1, num_dims * num_dims + 1, device=device, dtype=dtype
    ).reshape(num_dims, num_dims)
    bias = torch.linspace(0.02, 0.02 * num_dims, num_dims, device=device, dtype=dtype)
    control_displacements = nodes @ matrix.mT + bias

    points = ((origin + maximum) / 2).expand(num_dims + 1, num_dims).clone()
    points[:-1].diagonal().copy_(maximum)
    points[-1].copy_(maximum)
    points.requires_grad_(True)

    output = free_form_deform_points(
        points,
        control_displacements,
        origin=origin,
        extent=extent,
        basis="linear",
        implementation=implementation,
    )
    expected = points.detach() + points.detach() @ matrix.mT + bias
    torch.testing.assert_close(output, expected, atol=3.0e-5, rtol=3.0e-5)

    output.sum().backward()
    expected_gradient = torch.ones_like(points) + matrix.sum(dim=0)
    torch.testing.assert_close(points.grad, expected_gradient, atol=3.0e-5, rtol=3.0e-5)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_linear_large_world_origin_preserves_float64_coordinates(
    device, implementation
):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    points = torch.tensor(
        [[1.0e8 + 0.25], [1.0e8 + 0.75]], device=device, dtype=torch.float64
    )
    control_displacements = torch.tensor(
        [[0.0], [0.5], [1.0]], device=device, dtype=torch.float64
    )

    output = free_form_deform_points(
        points,
        control_displacements,
        origin=[1.0e8],
        extent=[1.0],
        basis="linear",
        implementation=implementation,
    )

    expected_displacement = torch.tensor(
        [[0.25], [0.75]], device=device, dtype=torch.float64
    )
    torch.testing.assert_close(
        output - points, expected_displacement, atol=1.0e-6, rtol=1.0e-6
    )


@pytest.mark.parametrize(
    ("basis", "continuity_order"),
    [("cubic_hermite", 1), ("quintic_hermite", 2)],
)
def test_hermite_interpolating_bases_have_documented_knot_continuity(
    basis, continuity_order
):
    """One-sided derivatives agree through the advertised continuity order."""

    controls = torch.tensor([[0.0], [1.0], [-0.5]], dtype=torch.float64)

    def field_derivatives(coordinate):
        point = torch.tensor([[coordinate]], dtype=torch.float64, requires_grad=True)
        field = (
            free_form_deform_points(
                point,
                controls,
                origin=[0.0],
                extent=[1.0],
                basis=basis,
                implementation="torch",
            )
            - point
        ).sum()
        derivatives = []
        value = field
        for _ in range(continuity_order):
            (value,) = torch.autograd.grad(value, point, create_graph=True)
            derivatives.append(value)
        return derivatives

    epsilon = 1.0e-7
    left = field_derivatives(0.5 - epsilon)
    right = field_derivatives(0.5 + epsilon)
    for left_derivative, right_derivative in zip(left, right, strict=True):
        torch.testing.assert_close(
            left_derivative, right_derivative, atol=2.0e-4, rtol=0.0
        )


@requires_module("warp")
def test_ffd_high_degree_bernstein_warp_stays_finite(device):
    """High-degree Bernstein evaluation remains finite in float32."""

    device = torch.device(device)
    resolution = 133
    points = torch.tensor(
        [[0.0], [0.5], [0.999], [1.0]], device=device, dtype=torch.float32
    )
    control_displacements = (
        0.1 * torch.sin(torch.arange(resolution, device=device, dtype=torch.float32))
    ).reshape(resolution, 1)
    cotangent = torch.tensor([[0.2], [-0.3], [0.5], [0.7]], device=device)

    results = []
    for implementation in ("torch", "warp"):
        points_i = points.clone().requires_grad_()
        control_i = control_displacements.clone().requires_grad_()
        output = free_form_deform_points(
            points_i,
            control_i,
            origin=[0.0],
            extent=[1.0],
            basis="bernstein",
            implementation=implementation,
        )
        gradients = torch.autograd.grad(
            (output * cotangent).sum(), (points_i, control_i)
        )
        assert torch.isfinite(output).all()
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        results.append((output, gradients))

    torch.testing.assert_close(results[1][0], results[0][0], atol=1e-5, rtol=1e-5)
    for actual, expected in zip(results[1][1], results[0][1], strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_ffd_bernstein_degree_one_is_multilinear_interpolation(device, implementation):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    dtype = torch.float64
    generator = torch.Generator(device=device).manual_seed(19)
    corners = 0.2 * torch.randn(
        (2, 2, 2, 3), generator=generator, device=device, dtype=dtype
    )
    points = torch.rand((64, 3), generator=generator, device=device, dtype=dtype)
    output = free_form_deform_points(
        points,
        corners,
        origin=[0.0, 0.0, 0.0],
        extent=[1.0, 1.0, 1.0],
        basis="bernstein",
        implementation=implementation,
    )
    expected = points.clone()
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                weight = (
                    (points[:, 0] if i else 1 - points[:, 0])
                    * (points[:, 1] if j else 1 - points[:, 1])
                    * (points[:, 2] if k else 1 - points[:, 2])
                )
                expected = expected + weight.unsqueeze(-1) * corners[i, j, k]
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("basis", _BASES)
def test_ffd_outside_points_are_identity_with_zero_gradients(
    device, implementation, basis
):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    dtype = torch.float64
    points = torch.tensor(
        [[0.5, 0.5], [1.5, 0.5], [-1.0e30, 3.0], [0.25, -0.001]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    generator = torch.Generator(device=device).manual_seed(23)
    control_displacements = 0.3 * torch.randn(
        (4, 4, 2), generator=generator, device=device, dtype=dtype
    )
    control_displacements.requires_grad_(True)
    output = free_form_deform_points(
        points,
        control_displacements,
        origin=[0.0, 0.0],
        extent=[1.0, 1.0],
        basis=basis,
        implementation=implementation,
    )
    assert torch.equal(output[1:], points.detach()[1:])
    assert not torch.equal(output[0], points.detach()[0])

    outside_loss = output[1:].sum()
    grad_points, grad_lattice = torch.autograd.grad(
        outside_loss, (points, control_displacements), retain_graph=True
    )
    # Outside points carry only the identity gradient and never reach the
    # lattice.
    torch.testing.assert_close(grad_points[1:], torch.ones_like(grad_points[1:]))
    assert torch.equal(grad_points[0], torch.zeros_like(grad_points[0]))
    assert torch.equal(grad_lattice, torch.zeros_like(grad_lattice))

    inside_loss = output[0].sum()
    grad_points, grad_lattice = torch.autograd.grad(
        inside_loss, (points, control_displacements)
    )
    assert torch.isfinite(grad_points).all()
    assert torch.isfinite(grad_lattice).all()
    assert grad_lattice.abs().sum() > 0


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize(
    ("basis", "resolution", "zero_layers"),
    [
        ("bernstein", 5, 1),
        ("bspline", 8, 3),
        ("linear", 5, 1),
        ("cubic_hermite", 5, 1),
        ("quintic_hermite", 5, 1),
    ],
)
def test_ffd_zero_boundary_layers_match_fixed_exterior(
    device, implementation, basis, resolution, zero_layers
):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    dtype = torch.float64
    points = torch.tensor([[-0.1], [0.0], [1.0], [1.1]], device=device, dtype=dtype)
    control_displacements = torch.linspace(
        -0.2, 0.3, resolution, device=device, dtype=dtype
    ).unsqueeze(-1)
    control_displacements[:zero_layers] = 0
    control_displacements[-zero_layers:] = 0

    output = free_form_deform_points(
        points,
        control_displacements,
        origin=[0.0],
        extent=[1.0],
        basis=basis,
        implementation=implementation,
    )

    torch.testing.assert_close(output, points)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_ffd_nan_points_stay_nan_without_poisoning_gradients(device, implementation):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    points = torch.tensor(
        [[0.5, 0.5], [float("nan"), 0.5]], device=device, dtype=torch.float64
    )
    control_displacements = 0.1 * torch.ones((4, 4, 2), device=device).double()
    control_displacements.requires_grad_(True)
    output = free_form_deform_points(
        points,
        control_displacements,
        origin=[0.0, 0.0],
        extent=[1.0, 1.0],
        basis="bspline",
        implementation=implementation,
    )
    assert torch.isnan(output[1, 0])
    output[0].sum().backward()
    assert torch.isfinite(control_displacements.grad).all()


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_ffd_hard_and_soft_point_weights(device, implementation):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    dtype = torch.float64
    generator = torch.Generator(device=device).manual_seed(29)
    points = torch.rand((16, 2), generator=generator, device=device, dtype=dtype)
    control_displacements = 0.2 * torch.randn(
        (4, 4, 2), generator=generator, device=device, dtype=dtype
    )
    kwargs = {
        "origin": [0.0, 0.0],
        "extent": [1.0, 1.0],
        "basis": "bspline",
        "implementation": implementation,
    }
    unweighted = free_form_deform_points(points, control_displacements, **kwargs)
    field = unweighted - points

    mask = torch.rand((16,), generator=generator, device=device) > 0.5
    hard = free_form_deform_points(
        points, control_displacements, point_weights=mask, **kwargs
    )
    assert torch.equal(hard[~mask], points[~mask])
    torch.testing.assert_close(hard[mask], unweighted[mask])

    weights = torch.rand((16,), generator=generator, device=device, dtype=dtype)
    soft = free_form_deform_points(
        points, control_displacements, point_weights=weights, **kwargs
    )
    torch.testing.assert_close(soft, points + weights.unsqueeze(-1) * field)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("basis", _BASES)
def test_ffd_batched_matches_unbatched(device, implementation, basis):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    dtype = torch.float64
    generator = torch.Generator(device=device).manual_seed(31)
    points = torch.rand((2, 64, 3), generator=generator, device=device, dtype=dtype)
    control_displacements = 0.1 * torch.randn(
        (2, 4, 5, 4, 3), generator=generator, device=device, dtype=dtype
    )
    origin = torch.tensor(
        [[0.0, 0.0, 0.0], [0.25, -0.5, 0.1]], device=device, dtype=dtype
    )
    extent = torch.tensor(
        [[1.0, 1.0, 1.0], [0.5, 2.0, 0.8]], device=device, dtype=dtype
    )
    batched = free_form_deform_points(
        points,
        control_displacements,
        origin=origin,
        extent=extent,
        basis=basis,
        implementation=implementation,
    )
    for b in range(2):
        single = free_form_deform_points(
            points[b],
            control_displacements[b],
            origin=origin[b],
            extent=extent[b],
            basis=basis,
            implementation=implementation,
        )
        torch.testing.assert_close(batched[b], single)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_ffd_one_dimensional_lattice(device, implementation):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    dtype = torch.float64
    points = torch.linspace(0.05, 0.95, 12, device=device, dtype=dtype).unsqueeze(-1)
    control_displacements = torch.tensor(
        [[0.0], [0.1], [0.1], [0.0]], device=device, dtype=dtype
    )
    output = free_form_deform_points(
        points,
        control_displacements,
        origin=[0.0],
        extent=[1.0],
        basis="bernstein",
        implementation=implementation,
    )
    assert output.shape == points.shape
    assert (output[:, 0] >= points[:, 0]).all()


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_ffd_empty_points(device, implementation):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    control_displacements = torch.rand((4, 4, 4, 3), device=device, requires_grad=True)
    for shape in ((0, 3), (2, 0, 3), (0, 2, 3)):
        points = torch.empty(shape, device=device)
        lattice = (
            control_displacements
            if len(shape) == 2
            else control_displacements.expand(shape[0], 4, 4, 4, 3)
        )
        output = free_form_deform_points(
            points,
            lattice,
            origin=[0.0, 0.0, 0.0],
            extent=[1.0, 1.0, 1.0],
            implementation=implementation,
        )
        assert output.shape == points.shape
        output.sum().backward()
        assert torch.equal(
            control_displacements.grad, torch.zeros_like(control_displacements)
        )
        control_displacements.grad = None


@pytest.mark.parametrize("basis", _BASES)
def test_ffd_torch_double_gradcheck(basis):
    dtype = torch.float64
    points = torch.tensor(
        [[0.21, 0.13], [0.56, 0.84], [0.87, 0.42]], dtype=dtype, requires_grad=True
    )
    control_displacements = 0.1 * torch.sin(
        torch.arange(4 * 5 * 2, dtype=dtype)
    ).reshape(4, 5, 2)
    control_displacements.requires_grad_(True)
    point_weights = torch.tensor([0.7, -0.4, 1.2], dtype=dtype, requires_grad=True)

    def operation(p, c, w):
        return free_form_deform_points(
            p,
            c,
            origin=[0.0, 0.0],
            extent=[1.0, 1.0],
            basis=basis,
            point_weights=w,
            implementation="torch",
        )

    assert torch.autograd.gradcheck(
        operation,
        (points, control_displacements, point_weights),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
    )


@requires_module("warp")
@pytest.mark.parametrize("basis", _BASES)
def test_ffd_warp_double_gradcheck(basis):
    dtype = torch.float64
    points = torch.tensor(
        [[0.17, 0.11], [0.66, 0.87], [0.91, 0.33]], dtype=dtype, requires_grad=True
    )
    control_displacements = 0.1 * torch.cos(
        torch.arange(4 * 5 * 2, dtype=dtype)
    ).reshape(4, 5, 2)
    control_displacements.requires_grad_(True)
    point_weights = torch.tensor([0.8, -0.4, 1.1], dtype=dtype, requires_grad=True)

    def operation(p, c, w):
        return free_form_deform_points(
            p,
            c,
            origin=[0.0, 0.0],
            extent=[1.0, 1.0],
            basis=basis,
            point_weights=w,
            implementation="warp",
        )

    assert torch.autograd.gradcheck(
        operation,
        (points, control_displacements, point_weights),
        eps=1e-6,
        atol=3e-5,
        rtol=3e-4,
    )


def _run_ffd_with_gradients(implementation, device, dtype, basis):
    points = torch.tensor(
        [[0.2, 0.1], [0.55, 0.85], [1.45, 0.2]], device=device, dtype=dtype
    ).requires_grad_()
    control_displacements = (
        0.1
        * torch.sin(torch.arange(4 * 5 * 2, device=device, dtype=dtype)).reshape(
            4, 5, 2
        )
    ).requires_grad_()
    point_weights = torch.tensor(
        [0.7, -0.4, 1.2], device=device, dtype=dtype, requires_grad=True
    )
    output = free_form_deform_points(
        points,
        control_displacements,
        origin=[0.0, 0.0],
        extent=[1.0, 1.0],
        basis=basis,
        point_weights=point_weights,
        implementation=implementation,
    )
    cotangent = torch.tensor(
        [[0.2, -0.5], [-0.3, 0.1], [0.7, -0.4]], device=device, dtype=dtype
    )
    gradients = torch.autograd.grad(
        (output * cotangent).sum(),
        (points, control_displacements, point_weights),
    )
    return output, gradients


@pytest.mark.parametrize("basis", _BASES)
def test_ffd_torch_chunked_checkpoint_matches_unchunked(device, basis, monkeypatch):
    """One-point checkpointed chunks match unchunked outputs and gradients."""

    from physicsnemo.nn.functional.geometry.deform import _torch_impl

    device = torch.device(device)
    unchunked = _run_ffd_with_gradients("torch", device, torch.float64, basis)
    real_checkpoint = _torch_impl.checkpoint
    checkpoint_call_sizes = []

    def checkpoint_spy(function, *args, **kwargs):
        checkpoint_call_sizes.append(args[0].shape[1])
        return real_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(_torch_impl, "_FFD_TEMPORARY_BYTE_BUDGET", 1)
    monkeypatch.setattr(_torch_impl, "checkpoint", checkpoint_spy)
    chunked = _run_ffd_with_gradients("torch", device, torch.float64, basis)

    assert checkpoint_call_sizes == [1, 1, 1]
    torch.testing.assert_close(chunked[0], unchunked[0])
    for chunked_gradient, unchunked_gradient in zip(
        chunked[1], unchunked[1], strict=True
    ):
        torch.testing.assert_close(chunked_gradient, unchunked_gradient)


@requires_module("warp")
@pytest.mark.parametrize("basis", _BASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ffd_torch_warp_forward_and_first_gradient_parity(device, dtype, basis):
    device = torch.device(device)
    ffd_torch = _run_ffd_with_gradients("torch", device, dtype, basis)
    ffd_warp = _run_ffd_with_gradients("warp", device, dtype, basis)

    if dtype == torch.float32:
        atol, rtol = 4e-5, 4e-5
    else:
        atol, rtol = 2e-9, 2e-8
    torch.testing.assert_close(ffd_warp[0], ffd_torch[0], atol=atol, rtol=rtol)
    for actual, expected in zip(ffd_warp[1], ffd_torch[1]):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_default_dispatch_selects_device_backend(device, monkeypatch):
    device = torch.device(device)
    points = torch.tensor([[0.2, 0.1], [0.7, 0.0]], device=device)
    control_displacements = torch.zeros((4, 4, 2), device=device)
    calls = []

    def torch_spy(normalized_points, *_args):
        calls.append("torch")
        return normalized_points

    def warp_spy(normalized_points, *_args):
        calls.append("warp")
        return normalized_points

    # Patch the names resolved by registered methods to observe the selected
    # branch. Patching the source modules would not intercept dispatch.
    monkeypatch.setattr(ffd_module, "ffd_points_torch", torch_spy)
    monkeypatch.setattr(ffd_module, "ffd_points_warp", warp_spy)

    warp_impl = FreeFormDeformPoints._get_impls()["warp"]
    expected = "warp" if device.type == "cuda" and warp_impl.available else "torch"
    automatic = free_form_deform_points(
        points, control_displacements, origin=[0.0, 0.0], extent=[1.0, 1.0]
    )
    assert calls == [expected]
    torch.testing.assert_close(automatic, points)

    if device.type == "cuda" and warp_impl.available:
        # When Warp is unavailable, CUDA falls back to Torch with the standard
        # one-time warning.
        calls.clear()
        unavailable_warp = type(warp_impl)(
            name=warp_impl.name,
            func=warp_impl.func,
            required_imports=warp_impl.required_imports,
            rank=warp_impl.rank,
            baseline=warp_impl.baseline,
            available=False,
        )
        monkeypatch.setitem(FreeFormDeformPoints._get_impls(), "warp", unavailable_warp)
        FunctionSpec._fallback_warned.discard(FreeFormDeformPoints._class_key())
        with pytest.warns(RuntimeWarning, match="falling back to implementation"):
            automatic = free_form_deform_points(
                points, control_displacements, origin=[0.0, 0.0], extent=[1.0, 1.0]
            )
        assert calls == ["torch"]
        torch.testing.assert_close(automatic, points)


@requires_module("warp")
def test_warp_custom_ops_opcheck():
    from physicsnemo.nn.functional.geometry.deform._warp_impl import (
        ffd_field_warp_impl,
    )

    dtype = torch.float64
    points = torch.tensor([[[0.2, 0.1], [0.7, 0.6]]], dtype=dtype, requires_grad=True)
    control_displacements = 0.1 * torch.sin(torch.arange(4 * 4 * 2, dtype=dtype))
    control_displacements = control_displacements.reshape(1, 16, 2).requires_grad_(True)
    origin = torch.zeros((1, 2), dtype=dtype)
    extent = torch.ones((1, 2), dtype=dtype)
    for basis in _BASES:
        torch.library.opcheck(
            ffd_field_warp_impl,
            args=(points, control_displacements, origin, extent, [4, 4], basis),
        )


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp", None])
@pytest.mark.parametrize("basis", _BASES)
def test_torch_compile_fullgraph(implementation, basis):
    points = torch.tensor([[0.2, 0.1], [0.6, 0.9]])
    control_displacements = 0.1 * torch.sin(torch.arange(4 * 5 * 2.0)).reshape(4, 5, 2)
    point_weights = torch.tensor([0.5, 1.2])

    def operation(p, c, w):
        return free_form_deform_points(
            p,
            c,
            origin=[0.0, 0.0],
            extent=[1.0, 1.0],
            basis=basis,
            point_weights=w,
            implementation=implementation,
        )

    eager = operation(points, control_displacements, point_weights)
    compiled = torch.compile(operation, fullgraph=True)(
        points, control_displacements, point_weights
    )
    torch.testing.assert_close(compiled, eager)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_torch_compile_fullgraph_backward(implementation):
    """Compiled backward matches eager execution for Torch and Warp."""

    if implementation == "warp":
        pytest.importorskip("warp")

    def make_inputs():
        points = torch.tensor([[0.2, 0.1], [0.6, 0.9]], requires_grad=True)
        control_displacements = (
            0.1 * torch.sin(torch.arange(4 * 5 * 2.0)).reshape(4, 5, 2)
        ).requires_grad_()
        point_weights = torch.tensor([0.5, 1.2], requires_grad=True)
        return points, control_displacements, point_weights

    def operation(p, c, w):
        return free_form_deform_points(
            p,
            c,
            origin=[0.0, 0.0],
            extent=[1.0, 1.0],
            basis="bspline",
            point_weights=w,
            implementation=implementation,
        )

    eager_inputs = make_inputs()
    eager_output = operation(*eager_inputs)
    eager_gradients = torch.autograd.grad(eager_output.square().sum(), eager_inputs)

    compiled_inputs = make_inputs()
    compiled_output = torch.compile(operation, fullgraph=True)(*compiled_inputs)
    compiled_gradients = torch.autograd.grad(
        compiled_output.square().sum(), compiled_inputs
    )

    torch.testing.assert_close(compiled_output, eager_output)
    for compiled_gradient, eager_gradient in zip(
        compiled_gradients, eager_gradients, strict=True
    ):
        torch.testing.assert_close(compiled_gradient, eager_gradient)


@pytest.mark.parametrize("basis", _BASES)
def test_torch_compile_fullgraph_dynamic_shapes(basis):
    """Symbolic query counts use the vectorized compile path."""

    def operation(points, control_displacements):
        return free_form_deform_points(
            points,
            control_displacements,
            origin=[0.0, 0.0, 0.0],
            extent=[1.0, 1.0, 1.0],
            basis=basis,
            implementation="torch",
        )

    compiled = torch.compile(operation, fullgraph=True, dynamic=True, backend="eager")
    generator = torch.Generator().manual_seed(3141)
    control_displacements = 0.1 * torch.randn((4, 4, 4, 3), generator=generator)
    for num_points in (4, 7):
        points = torch.rand((num_points, 3), generator=generator)
        torch.testing.assert_close(
            compiled(points, control_displacements),
            operation(points, control_displacements),
        )


def test_torch_compile_dynamic_lattice_box_sequences():
    """Changing Python box values remain valid under ``torch.compile``."""

    points = torch.tensor([[0.2, 0.1], [0.6, 0.9]])
    control_displacements = 0.1 * torch.sin(torch.arange(4 * 5 * 2.0)).reshape(4, 5, 2)

    def operation(points, control_displacements, origin, extent):
        return free_form_deform_points(
            points,
            control_displacements,
            origin=origin,
            extent=extent,
            implementation="torch",
        )

    compiled = torch.compile(operation, fullgraph=True, dynamic=True, backend="eager")
    for origin, extent in (
        ([0.0, 0.0], [1.0, 1.0]),
        ([-0.2, -0.1], [1.2, 1.3]),
    ):
        torch.testing.assert_close(
            compiled(points, control_displacements, origin, extent),
            operation(points, control_displacements, origin, extent),
        )


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp", None])
def test_ffd_points_is_cuda_graph_capture_safe(device, implementation):
    device = torch.device(device)
    if device.type != "cuda":
        pytest.skip("CUDA Graph capture requires CUDA")

    points = torch.rand((8, 3), device=device)
    control_displacements = 0.1 * torch.rand((4, 4, 4, 3), device=device)
    origin = torch.zeros(3, device=device)
    extent = torch.ones(3, device=device)
    point_weights = torch.ones(8, device=device)

    # Warm allocations, backend kernels, and the cached lattice-resolution
    # tensor before capture.
    expected = free_form_deform_points(
        points,
        control_displacements,
        origin=origin,
        extent=extent,
        point_weights=point_weights,
        implementation=implementation,
    )
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = free_form_deform_points(
            points,
            control_displacements,
            origin=origin,
            extent=extent,
            point_weights=point_weights,
            implementation=implementation,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(captured, expected)


@requires_module("warp")
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_public_api_fake_tensor_propagation(implementation):
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    with FakeTensorMode():
        points = torch.empty((4, 3), dtype=torch.float64)
        control_displacements = torch.empty((4, 5, 4, 3), dtype=torch.float64)
        point_weights = torch.empty(4, dtype=torch.float64)
        output = free_form_deform_points(
            points,
            control_displacements,
            origin=[0.0, 0.0, 0.0],
            extent=[1.0, 1.0, 1.0],
            point_weights=point_weights,
            implementation=implementation,
        )

    assert isinstance(output, FakeTensor)
    assert output.shape == points.shape
    assert output.dtype == points.dtype
    assert output.device == points.device


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (
            lambda: free_form_deform_points(
                torch.zeros(2, 2),
                torch.zeros(4, 4, 2),
                origin=[0.0, 0.0],
                extent=[1.0, 1.0],
                basis="catmull_rom",
                implementation="torch",
            ),
            ValueError,
            "basis must be one of",
        ),
        (
            lambda: free_form_deform_points(
                torch.zeros(2, 2),
                torch.zeros(1, 4, 2),
                origin=[0.0, 0.0],
                extent=[1.0, 1.0],
                implementation="torch",
            ),
            ValueError,
            "requires at least 2 lattice nodes",
        ),
        (
            lambda: free_form_deform_points(
                torch.zeros(2, 2),
                torch.zeros(1, 4, 2),
                origin=[0.0, 0.0],
                extent=[1.0, 1.0],
                basis="cubic_hermite",
                implementation="torch",
            ),
            ValueError,
            "basis 'cubic_hermite' requires at least 2 lattice nodes",
        ),
        (
            lambda: free_form_deform_points(
                torch.zeros(2, 2),
                torch.zeros(4, 3, 2),
                origin=[0.0, 0.0],
                extent=[1.0, 1.0],
                basis="bspline",
                implementation="torch",
            ),
            ValueError,
            "requires at least 4 lattice nodes",
        ),
        (
            lambda: free_form_deform_points(
                torch.zeros(2, 3),
                torch.zeros(4, 4, 3),
                origin=[0.0, 0.0, 0.0],
                extent=[1.0, 1.0, 1.0],
                implementation="torch",
            ),
            ValueError,
            r"shape \(n_1, ..., n_D, D\)",
        ),
        (
            lambda: free_form_deform_points(
                torch.zeros(1, 2, 2),
                torch.zeros(4, 4, 2),
                origin=[0.0, 0.0],
                extent=[1.0, 1.0],
                implementation="torch",
            ),
            ValueError,
            "both be unbatched or both be batched",
        ),
        (
            lambda: free_form_deform_points(
                torch.zeros(2, 2, 2),
                torch.zeros(3, 4, 4, 2),
                origin=[0.0, 0.0],
                extent=[1.0, 1.0],
                implementation="torch",
            ),
            ValueError,
            "aligned batch",
        ),
        (
            lambda: free_form_deform_points(
                torch.zeros(2, 3),
                torch.zeros(4, 4, 4, 2),
                origin=[0.0, 0.0, 0.0],
                extent=[1.0, 1.0, 1.0],
                implementation="torch",
            ),
            ValueError,
            "components per lattice node",
        ),
        (
            lambda: free_form_deform_points(
                torch.zeros(2, 2),
                torch.zeros(4, 4, 2),
                origin=[0.0],
                extent=[1.0, 1.0],
                implementation="torch",
            ),
            TypeError,
            "origin must contain exactly 2 real values",
        ),
        (
            lambda: free_form_deform_points(
                torch.zeros(2, 2),
                torch.zeros(4, 4, 2),
                origin=[0.0, 0.0],
                extent=[1.0, 0.0],
                implementation="torch",
            ),
            ValueError,
            "extent values must be strictly positive",
        ),
        (
            lambda: free_form_deform_points(
                torch.zeros(2, 2),
                torch.zeros(4, 4, 2),
                origin=[0.0, 0.0],
                extent=[1.0, float("inf")],
                implementation="torch",
            ),
            ValueError,
            "extent values must be finite",
        ),
        (
            lambda: free_form_deform_points(
                torch.zeros(2, 2),
                torch.zeros(4, 4, 2),
                origin=[0.0, 0.0],
                extent=[1.0e100, 1.0],
                implementation="torch",
            ),
            ValueError,
            "extent values must be finite in the points dtype",
        ),
        (
            lambda: free_form_deform_points(
                torch.zeros(2, 2),
                torch.zeros(4, 4, 2),
                origin=[0.0, 0.0],
                extent=[1.0e-100, 1.0],
                implementation="torch",
            ),
            ValueError,
            "extent values must be strictly positive in the points dtype",
        ),
        (
            lambda: free_form_deform_points(
                torch.zeros(2, 2),
                torch.zeros(4, 4, 2),
                origin=torch.zeros(2, requires_grad=True),
                extent=[1.0, 1.0],
                implementation="torch",
            ),
            ValueError,
            "non-differentiable lattice configuration",
        ),
        (
            lambda: free_form_deform_points(
                torch.zeros(2, 2),
                torch.zeros(4, 4, 2),
                origin=torch.zeros(3),
                extent=[1.0, 1.0],
                implementation="torch",
            ),
            ValueError,
            r"origin must have shape \(2,\)",
        ),
        (
            lambda: free_form_deform_points(
                torch.zeros(2, 2),
                torch.zeros(4, 4, 2),
                origin=torch.zeros(2, dtype=torch.float64),
                extent=[1.0, 1.0],
                implementation="torch",
            ),
            TypeError,
            "same dtype",
        ),
        (
            lambda: free_form_deform_points(
                torch.zeros(2, 2),
                torch.zeros(4, 4, 2, dtype=torch.float64),
                origin=[0.0, 0.0],
                extent=[1.0, 1.0],
                implementation="torch",
            ),
            TypeError,
            "same dtype",
        ),
        (
            lambda: free_form_deform_points(
                torch.zeros(2, 2),
                torch.zeros(4, 4, 2),
                origin=[0.0, 0.0],
                extent=[1.0, 1.0],
                point_weights=torch.zeros(3),
                implementation="torch",
            ),
            ValueError,
            "point_weights must have shape",
        ),
    ],
)
def test_validation(call, error, match):
    with pytest.raises(error, match=match):
        call()


@requires_module("warp")
def test_ffd_benchmark_forward_cases_and_hooks(device):
    """Run every forward generator branch through reduced Torch/Warp parity cases."""

    device = torch.device(device)
    labels = []
    for label, args, kwargs in FreeFormDeformPoints.make_inputs_forward(device=device):
        labels.append(label)
        reduced_args, reduced_kwargs = _trim_ffd_benchmark_case(args, kwargs)
        args_torch, kwargs_torch = clone_case(reduced_args, reduced_kwargs)
        args_warp, kwargs_warp = clone_case(reduced_args, reduced_kwargs)

        output_torch = FreeFormDeformPoints.dispatch(
            *args_torch, implementation="torch", **kwargs_torch
        )
        output_warp = FreeFormDeformPoints.dispatch(
            *args_warp, implementation="warp", **kwargs_warp
        )
        FreeFormDeformPoints.compare_forward(output_warp, output_torch)

    assert labels == [case[0] for case in FreeFormDeformPoints._FORWARD_BENCHMARK_CASES]


@requires_module("warp")
def test_ffd_benchmark_backward_cases_and_hooks(device):
    """Run every backward generator branch through reduced Torch/Warp parity cases."""

    device = torch.device(device)
    labels = []
    for label, args, kwargs in FreeFormDeformPoints.make_inputs_backward(device=device):
        labels.append(label)
        reduced_args, reduced_kwargs = _trim_ffd_benchmark_case(args, kwargs)
        args_torch, kwargs_torch = clone_case(reduced_args, reduced_kwargs)
        args_warp, kwargs_warp = clone_case(reduced_args, reduced_kwargs)

        output_torch = FreeFormDeformPoints.dispatch(
            *args_torch, implementation="torch", **kwargs_torch
        )
        output_warp = FreeFormDeformPoints.dispatch(
            *args_warp, implementation="warp", **kwargs_warp
        )
        FreeFormDeformPoints.compare_forward(output_warp, output_torch)

        tensors_torch = _differentiable_case_tensors(args_torch, kwargs_torch)
        tensors_warp = _differentiable_case_tensors(args_warp, kwargs_warp)
        gradients_torch = torch.autograd.grad(
            output_torch.square().mean(), tensors_torch
        )
        gradients_warp = torch.autograd.grad(output_warp.square().mean(), tensors_warp)
        for gradient_warp, gradient_torch in zip(
            gradients_warp, gradients_torch, strict=True
        ):
            FreeFormDeformPoints.compare_backward(gradient_warp, gradient_torch)

    assert labels == [
        case[0] for case in FreeFormDeformPoints._BACKWARD_BENCHMARK_CASES
    ]
