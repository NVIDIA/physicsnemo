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

"""Tests for differentiable Sobolev point deformation."""

import importlib
import inspect
import warnings

import pytest
import torch

import physicsnemo.nn.functional as functional
from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.nn.functional import sobolev_deform_points
from physicsnemo.nn.functional.geometry import SobolevDeformPoints


def _dense_helmholtz_reference(
    points: torch.Tensor,
    cells: torch.Tensor,
    displacement: torch.Tensor,
    length_scale: float,
    fixed_points: torch.Tensor | None = None,
) -> torch.Tensor:
    """Solve the uniform-mass P1 system with dense linear algebra."""

    num_points = points.shape[0]
    mass = points.new_zeros(num_points)
    stiffness = points.new_zeros((num_points, num_points))

    for cell in cells:
        vertices = points[cell]
        manifold_dims = cell.numel() - 1
        edges = (vertices[1:] - vertices[0]).mT
        metric = edges.mT @ edges
        volume = torch.linalg.det(metric).sqrt() / float(
            torch.lgamma(points.new_tensor(manifold_dims + 1)).exp()
        )

        gradients_tail = edges @ torch.linalg.inv(metric)
        gradients = torch.cat(
            (-gradients_tail.sum(dim=1, keepdim=True), gradients_tail), dim=1
        ).mT
        local_stiffness = volume * (gradients @ gradients.mT)

        mass.index_add_(
            0,
            cell,
            torch.full_like(cell, volume / cell.numel(), dtype=points.dtype),
        )
        rows = cell[:, None].expand(-1, cell.numel()).reshape(-1)
        cols = cell[None, :].expand(cell.numel(), -1).reshape(-1)
        stiffness.index_put_((rows, cols), local_stiffness.reshape(-1), accumulate=True)

    isolated = mass == 0
    mean_mass = mass[~isolated].mean()
    mass = torch.full_like(mass, mean_mass)
    matrix = torch.diag(mass) + length_scale**2 * stiffness
    right_hand_side = mass[:, None] * displacement
    fixed = (
        torch.zeros(num_points, dtype=torch.bool, device=points.device)
        if fixed_points is None
        else fixed_points
    )
    free = ~(fixed | isolated)

    filtered = torch.zeros_like(displacement)
    if bool(free.any()):
        free_matrix = matrix[free][:, free]
        filtered[free] = torch.linalg.solve(free_matrix, right_hand_side[free])
    filtered[isolated & ~fixed] = displacement[isolated & ~fixed]
    return points + filtered


def test_public_exports_and_signature():
    geometry = importlib.import_module("physicsnemo.nn.functional.geometry")
    deform = importlib.import_module("physicsnemo.nn.functional.geometry.deform")

    assert functional.sobolev_deform_points is sobolev_deform_points
    assert geometry.sobolev_deform_points is sobolev_deform_points
    assert deform.sobolev_deform_points is sobolev_deform_points
    assert geometry.SobolevDeformPoints is SobolevDeformPoints
    assert deform.SobolevDeformPoints is SobolevDeformPoints
    assert issubclass(SobolevDeformPoints, FunctionSpec)
    assert SobolevDeformPoints.implementations() == ("warp", "torch")
    assert not hasattr(functional, "SobolevDeformPoints")
    for module in (functional, geometry, deform):
        assert "sobolev_deform_points" in module.__all__
    assert "SobolevDeformPoints" in geometry.__all__
    assert "SobolevDeformPoints" in deform.__all__

    signature = inspect.signature(sobolev_deform_points)
    assert list(signature.parameters) == [
        "points",
        "cells",
        "displacement",
        "length_scale",
        "fixed_points",
        "max_iterations",
        "tolerance",
        "implementation",
    ]
    for name in (
        "length_scale",
        "fixed_points",
        "max_iterations",
        "tolerance",
        "implementation",
    ):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["fixed_points"].default is None
    assert signature.parameters["max_iterations"].default == 128
    assert signature.parameters["tolerance"].default is None
    assert signature.parameters["implementation"].default is None
    assert "warp" in str(signature.parameters["implementation"].annotation)


def test_default_dispatch_selects_device_backend(device, monkeypatch):
    sobolev_module = importlib.import_module(
        "physicsnemo.nn.functional.geometry.deform.sobolev"
    )
    device = torch.device(device)
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device=device)
    cells = torch.tensor([[0, 1]], device=device)
    displacement = torch.zeros_like(points)
    calls = []

    def torch_spy(normalized_points, *_args, **_kwargs):
        calls.append("torch")
        return normalized_points

    def warp_spy(normalized_points, *_args, **_kwargs):
        calls.append("warp")
        return normalized_points

    monkeypatch.setattr(
        sobolev_module,
        "sobolev_deform_points_torch",
        torch_spy,
    )
    monkeypatch.setattr(
        sobolev_module,
        "sobolev_deform_points_warp",
        warp_spy,
    )

    warp_impl = SobolevDeformPoints._get_impls()["warp"]
    expected = "warp" if device.type == "cuda" and warp_impl.available else "torch"
    output = sobolev_deform_points(
        points,
        cells,
        displacement,
        length_scale=0.2,
    )
    assert calls == [expected]
    torch.testing.assert_close(output, points)

    if device.type == "cuda" and warp_impl.available:
        calls.clear()
        higher_dimensional_points = torch.zeros((5, 4), device=device)
        higher_dimensional_cells = torch.arange(5, device=device).reshape(1, 5)
        output = sobolev_deform_points(
            higher_dimensional_points,
            higher_dimensional_cells,
            torch.zeros_like(higher_dimensional_points),
            length_scale=0.2,
        )
        assert calls == ["torch"]
        torch.testing.assert_close(output, higher_dimensional_points)

        calls.clear()
        unavailable_warp = type(warp_impl)(
            name=warp_impl.name,
            func=warp_impl.func,
            required_imports=warp_impl.required_imports,
            rank=warp_impl.rank,
            baseline=warp_impl.baseline,
            available=False,
        )
        monkeypatch.setitem(
            SobolevDeformPoints._get_impls(),
            "warp",
            unavailable_warp,
        )
        FunctionSpec._fallback_warned.discard(SobolevDeformPoints._class_key())
        with pytest.warns(RuntimeWarning, match="falling back to implementation"):
            output = sobolev_deform_points(
                points,
                cells,
                displacement,
                length_scale=0.2,
            )
        assert calls == ["torch"]
        torch.testing.assert_close(output, points)


def test_single_segment_has_analytic_solution():
    points = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    cells = torch.tensor([[0, 1]])
    displacement = torch.tensor([[1.0], [0.0]], dtype=torch.float64)

    output = sobolev_deform_points(
        points,
        cells,
        displacement,
        length_scale=1.0,
        max_iterations=16,
        tolerance=1.0e-14,
        implementation="torch",
    )

    expected_displacement = torch.tensor([[0.6], [0.4]], dtype=torch.float64)
    torch.testing.assert_close(output, points + expected_displacement)


def test_zero_displacement_has_sobolev_filtered_adjoint():
    points = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    cells = torch.tensor([[0, 1]])
    displacement = torch.zeros_like(points, requires_grad=True)
    output_adjoint = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)

    output = sobolev_deform_points(
        points,
        cells,
        displacement,
        length_scale=0.5,
        max_iterations=16,
        tolerance=1.0e-14,
        implementation="torch",
    )
    displacement_adjoint = torch.autograd.grad(
        output,
        displacement,
        output_adjoint,
    )[0]

    torch.testing.assert_close(
        displacement_adjoint,
        0.5 * output_adjoint,
        atol=1.0e-13,
        rtol=1.0e-13,
    )


def test_nonuniform_mesh_preserves_constant_displacement_adjoint():
    points = torch.tensor([[0.0], [0.001], [1.0]], dtype=torch.float64)
    cells = torch.tensor([[0, 1], [1, 2]])
    displacement = torch.zeros_like(points, requires_grad=True)

    output = sobolev_deform_points(
        points,
        cells,
        displacement,
        length_scale=0.2,
        max_iterations=32,
        tolerance=1.0e-14,
        implementation="torch",
    )
    displacement_adjoint = torch.autograd.grad(
        output.sum(),
        displacement,
    )[0]

    torch.testing.assert_close(
        displacement_adjoint,
        torch.ones_like(displacement_adjoint),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_zero_length_scale_is_exact_dense_deformation_with_identity_gradients():
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    cells = torch.tensor([[0, 1, 2]])
    displacement = torch.tensor(
        [[0.2, -0.4], [0.3, 0.5], [-0.1, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    weights = torch.tensor([[1.0, -2.0], [0.5, 3.0], [-4.0, 0.25]], dtype=torch.float64)

    output = sobolev_deform_points(
        points,
        cells,
        displacement,
        length_scale=0.0,
        implementation="torch",
    )
    point_gradient, displacement_gradient = torch.autograd.grad(
        (output * weights).sum(), (points, displacement)
    )

    assert torch.equal(output, points + displacement)
    torch.testing.assert_close(point_gradient, weights, rtol=0.0, atol=0.0)
    torch.testing.assert_close(displacement_gradient, weights, rtol=0.0, atol=0.0)


def test_natural_boundaries_preserve_constant_fields_per_component():
    points = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
            [5.0, 0.0],
            [9.0, 2.0],
        ],
        dtype=torch.float64,
    )
    cells = torch.tensor([[0, 1], [2, 3]])
    displacement = torch.tensor(
        [
            [0.5, -0.25],
            [0.5, -0.25],
            [-1.0, 0.75],
            [-1.0, 0.75],
            [2.0, 3.0],
        ],
        dtype=torch.float64,
    )

    output = sobolev_deform_points(
        points,
        cells,
        displacement,
        length_scale=2.0,
        max_iterations=32,
        tolerance=1.0e-14,
        implementation="torch",
    )

    torch.testing.assert_close(output, points + displacement)


def test_fixed_and_isolated_points_follow_boundary_contract():
    points = torch.tensor([[0.0], [1.0], [4.0]], dtype=torch.float64)
    cells = torch.tensor([[0, 1]])
    displacement = torch.ones_like(points)
    fixed_points = torch.tensor([True, False, True])

    output = sobolev_deform_points(
        points,
        cells,
        displacement,
        length_scale=1.0,
        fixed_points=fixed_points,
        max_iterations=16,
        tolerance=1.0e-14,
        implementation="torch",
    )

    expected = points + torch.tensor([[0.0], [1.0 / 3.0], [0.0]])
    torch.testing.assert_close(output, expected)


def test_matches_dense_uniform_mass_p1_reference():
    points = torch.tensor(
        [
            [0.0, 0.0],
            [1.2, 0.1],
            [1.0, 1.1],
            [-0.1, 0.8],
            [3.0, 4.0],
        ],
        dtype=torch.float64,
    )
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]])
    displacement = torch.tensor(
        [
            [0.5, -0.2],
            [-0.3, 0.7],
            [0.9, 0.4],
            [-0.8, -0.1],
            [1.5, -2.0],
        ],
        dtype=torch.float64,
    )
    fixed_points = torch.tensor([False, True, False, False, False])
    length_scale = 0.45

    output = sobolev_deform_points(
        points,
        cells,
        displacement,
        length_scale=length_scale,
        fixed_points=fixed_points,
        max_iterations=64,
        tolerance=1.0e-14,
        implementation="torch",
    )
    expected = _dense_helmholtz_reference(
        points, cells, displacement, length_scale, fixed_points
    )

    torch.testing.assert_close(output, expected, atol=2.0e-12, rtol=2.0e-12)


def test_tetrahedron_matches_dense_uniform_mass_p1_reference():
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.1, 0.0],
            [0.2, 1.0, 0.1],
            [0.1, 0.2, 0.9],
        ],
        dtype=torch.float64,
    )
    cells = torch.tensor([[0, 1, 2, 3]])
    displacement = torch.tensor(
        [
            [0.2, -0.1, 0.3],
            [-0.3, 0.4, 0.1],
            [0.5, 0.2, -0.2],
            [-0.1, 0.3, 0.4],
        ],
        dtype=torch.float64,
    )
    fixed_points = torch.tensor([True, False, False, False])
    length_scale = 0.3

    output = sobolev_deform_points(
        points,
        cells,
        displacement,
        length_scale=length_scale,
        fixed_points=fixed_points,
        max_iterations=32,
        tolerance=1.0e-14,
        implementation="torch",
    )
    expected = _dense_helmholtz_reference(
        points,
        cells,
        displacement,
        length_scale,
        fixed_points,
    )

    torch.testing.assert_close(output, expected, atol=2.0e-12, rtol=2.0e-12)


def test_length_scale_uses_mesh_coordinate_units():
    points = torch.tensor(
        [[0.0, 0.0], [1.1, 0.1], [0.2, 0.9]],
        dtype=torch.float64,
    )
    cells = torch.tensor([[0, 1, 2]])
    displacement = torch.tensor(
        [[0.1, -0.2], [0.3, 0.05], [-0.15, 0.25]],
        dtype=torch.float64,
    )
    length_scale = 0.35
    output = sobolev_deform_points(
        points,
        cells,
        displacement,
        length_scale=length_scale,
        max_iterations=64,
        tolerance=1.0e-14,
        implementation="torch",
    )

    for scale in (0.01, 10.0):
        scaled_output = sobolev_deform_points(
            scale * points,
            cells,
            scale * displacement,
            length_scale=scale * length_scale,
            max_iterations=64,
            tolerance=1.0e-14,
            implementation="torch",
        )
        torch.testing.assert_close(
            scaled_output,
            scale * output,
            atol=1.0e-12,
            rtol=1.0e-11,
        )


def test_batched_shared_topology_matches_independent_calls(device):
    device = torch.device(device)
    points = torch.tensor(
        [[0.0, 0.0], [1.1, 0.0], [0.8, 0.9], [-0.1, 0.7]],
        device=device,
    )
    points = torch.stack((points, 1.7 * points + 0.25))
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]], device=device)
    displacement = torch.tensor(
        [
            [[0.1, -0.2], [0.3, 0.1], [-0.15, 0.25], [0.05, -0.1]],
            [[-0.2, 0.4], [0.1, -0.3], [0.25, 0.05], [-0.1, 0.2]],
        ],
        device=device,
    )
    fixed_points = torch.tensor(
        [[True, False, False, False], [False, False, True, False]],
        device=device,
    )

    batched = sobolev_deform_points(
        points,
        cells,
        displacement,
        length_scale=0.3,
        fixed_points=fixed_points,
        max_iterations=64,
        implementation="torch",
    )
    independent = torch.stack(
        tuple(
            sobolev_deform_points(
                points[index],
                cells,
                displacement[index],
                length_scale=0.3,
                fixed_points=fixed_points[index],
                max_iterations=64,
                implementation="torch",
            )
            for index in range(2)
        )
    )

    torch.testing.assert_close(batched, independent, atol=2.0e-5, rtol=2.0e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("simplex", ["segment", "triangle", "tetrahedron"])
@pytest.mark.parametrize("use_fixed_points", [False, True])
def test_warp_matches_torch_forward_and_first_gradients(
    device,
    dtype,
    simplex,
    use_fixed_points,
):
    """Warp matches the Torch P1 solve and its implicit adjoint."""

    device = torch.device(device)
    if device.type != "cuda":
        pytest.skip("the Warp Sobolev backend requires CUDA")
    pytest.importorskip("warp")

    if simplex == "segment":
        base_points = [
            [0.0, 0.0],
            [0.3, 0.1],
            [1.1, -0.05],
            [1.8, 0.4],
            [2.5, -0.2],
        ]
        cells = [[0, 1], [1, 2], [2, 3]]
    elif simplex == "triangle":
        base_points = [
            [0.0, 0.0, 0.0],
            [1.2, 0.1, 0.2],
            [0.8, 1.0, -0.1],
            [-0.2, 0.7, 0.3],
            [2.0, 1.8, -0.4],
        ]
        cells = [[0, 1, 2], [0, 2, 3]]
    else:
        base_points = [
            [0.0, 0.0, 0.0],
            [1.1, 0.1, 0.0],
            [0.2, 0.9, 0.1],
            [0.1, 0.2, 1.2],
            [2.0, -0.5, 0.7],
        ]
        cells = [[0, 1, 2, 3]]

    points_0 = torch.tensor(base_points, device=device, dtype=dtype)
    points = torch.stack((points_0, 1.35 * points_0 + 0.2))
    cells_t = torch.tensor(cells, device=device, dtype=torch.long)
    values = torch.arange(points.numel(), device=device, dtype=dtype).reshape_as(points)
    displacement = 0.15 * torch.sin(0.7 * values + 0.3)
    cotangent = torch.cos(0.4 * values - 0.2)
    fixed_points = None
    if use_fixed_points:
        fixed_points = torch.tensor(
            [[True, False, False, False, False], [False, False, True, False, True]],
            device=device,
        )

    def evaluate(implementation):
        point_values = points.detach().clone().requires_grad_(True)
        displacement_values = displacement.detach().clone().requires_grad_(True)
        output = sobolev_deform_points(
            point_values,
            cells_t,
            displacement_values,
            length_scale=0.35,
            fixed_points=fixed_points,
            max_iterations=128,
            tolerance=2.0e-6 if dtype == torch.float32 else 1.0e-11,
            implementation=implementation,
        )
        gradients = torch.autograd.grad(
            (output * cotangent).sum(),
            (point_values, displacement_values),
        )
        return output, gradients

    expected_output, expected_gradients = evaluate("torch")
    actual_output, actual_gradients = evaluate("warp")
    if dtype == torch.float32:
        atol, rtol = 6.0e-5, 6.0e-5
    else:
        atol, rtol = 2.0e-9, 2.0e-8
    torch.testing.assert_close(
        actual_output,
        expected_output,
        atol=atol,
        rtol=rtol,
    )
    for actual, expected in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_warp_restarts_from_true_residual_in_float32(device):
    """Residual replacement reaches the requested float32 tolerance."""

    device = torch.device(device)
    if device.type != "cuda":
        pytest.skip("the Warp Sobolev backend requires CUDA")
    pytest.importorskip("warp")

    num_points = 130
    x = torch.linspace(0, 3, num_points - 1, device=device)
    points = torch.cat((x, x.new_tensor([9.0]))).reshape(1, num_points, 1)
    points.requires_grad_()
    cells = torch.stack(
        (
            torch.arange(num_points - 2, device=device),
            torch.arange(1, num_points - 1, device=device),
        ),
        dim=-1,
    )
    generator = torch.Generator(device=device).manual_seed(7301)
    displacement = 0.1 * torch.randn(
        points.shape,
        generator=generator,
        device=device,
    )
    displacement.requires_grad_()
    fixed_points = torch.zeros((1, num_points), device=device, dtype=torch.bool)
    fixed_points[:, ::17] = True

    output = sobolev_deform_points(
        points,
        cells,
        displacement,
        length_scale=0.18,
        fixed_points=fixed_points,
        max_iterations=512,
        tolerance=1.0e-6,
        implementation="warp",
    )
    gradients = torch.autograd.grad(
        output.square().mean(),
        (points, displacement),
    )

    assert torch.isfinite(output).all()
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_warp_relative_tolerance_handles_tiny_fields(device):
    device = torch.device(device)
    if device.type != "cuda":
        pytest.skip("the Warp Sobolev backend requires CUDA")
    pytest.importorskip("warp")

    points = torch.linspace(0, 1, 9, dtype=torch.float64, device=device)
    points = points.reshape(-1, 1).requires_grad_()
    cells = torch.stack(
        (torch.arange(8, device=device), torch.arange(1, 9, device=device)),
        dim=-1,
    )
    displacement = 1.0e-35 * torch.sin(7 * points.detach())
    displacement.requires_grad_()
    output = sobolev_deform_points(
        points,
        cells,
        displacement,
        length_scale=0.2,
        max_iterations=128,
        tolerance=1.0e-10,
        implementation="warp",
    )
    output_adjoint = torch.full_like(output, 1.0e-35)
    gradients = torch.autograd.grad(
        output,
        (points, displacement),
        output_adjoint,
    )

    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize(
    ("implementation", "device_name"),
    [("torch", "cpu"), ("warp", "cuda")],
)
def test_empty_topology_is_dense_deformation(implementation, device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("the Warp Sobolev backend requires CUDA")
    device = torch.device(device_name)
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 2.0]],
        device=device,
        requires_grad=True,
    )
    displacement = torch.tensor(
        [[0.2, -0.1], [-0.3, 0.4]],
        device=device,
        requires_grad=True,
    )
    cells = torch.empty((0, 2), dtype=torch.long, device=device)

    output = sobolev_deform_points(
        points,
        cells,
        displacement,
        length_scale=10.0,
        implementation=implementation,
    )
    point_gradient, displacement_gradient = torch.autograd.grad(
        output.sum(),
        (points, displacement),
    )

    assert torch.equal(output, points + displacement)
    assert torch.equal(point_gradient, torch.ones_like(points))
    assert torch.equal(displacement_gradient, torch.ones_like(displacement))


def test_gradcheck_with_respect_to_points_and_displacement():
    points = torch.tensor(
        [[0.0, 0.0], [1.1, 0.1], [0.2, 0.9]],
        dtype=torch.float64,
        requires_grad=True,
    )
    cells = torch.tensor([[0, 1, 2]])
    displacement = torch.tensor(
        [[0.1, -0.2], [0.3, 0.05], [-0.15, 0.25]],
        dtype=torch.float64,
        requires_grad=True,
    )

    def operation(
        point_values: torch.Tensor, displacement_values: torch.Tensor
    ) -> torch.Tensor:
        return sobolev_deform_points(
            point_values,
            cells,
            displacement_values,
            length_scale=0.35,
            max_iterations=64,
            tolerance=1.0e-14,
            implementation="torch",
        )

    assert torch.autograd.gradcheck(
        operation,
        (points, displacement),
        eps=1.0e-6,
        atol=2.0e-5,
        rtol=2.0e-4,
    )


@pytest.mark.parametrize(
    ("implementation", "device_name"),
    [("torch", "cpu"), ("warp", "cuda")],
)
def test_torch_compile_fullgraph_forward_and_backward(implementation, device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("the Warp Sobolev backend requires CUDA")
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device_name)
    cells = torch.tensor([[0, 1], [1, 2]], device=device)
    base_points = torch.tensor([[0.0], [0.5], [1.0]], device=device)
    base_displacement = torch.tensor([[0.1], [-0.2], [0.3]], device=device)

    def operation(points: torch.Tensor, displacement: torch.Tensor):
        return sobolev_deform_points(
            points,
            cells,
            displacement,
            length_scale=0.2,
            max_iterations=16,
            implementation=implementation,
        )

    def evaluate(function):
        points = base_points.clone().requires_grad_(True)
        displacement = base_displacement.clone().requires_grad_(True)
        output = function(points, displacement)
        gradients = torch.autograd.grad(output.square().sum(), (points, displacement))
        return output, gradients

    expected_output, expected_gradients = evaluate(operation)
    compiled = torch.compile(operation, fullgraph=True, backend="aot_eager")
    actual_output, actual_gradients = evaluate(compiled)

    torch.testing.assert_close(actual_output, expected_output)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected)


def test_torch_compile_one_fullgraph_handles_dynamic_meshes_and_scalars():
    compiled_graphs = []

    def operation(points, cells, displacement, length_scale, tolerance):
        return sobolev_deform_points(
            points,
            cells,
            displacement,
            length_scale=length_scale,
            max_iterations=32,
            tolerance=tolerance,
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
    generator = torch.Generator().manual_seed(4319)
    for num_points, num_cells, length_scale, tolerance in (
        (5, 3, 0.04, 1.0e-5),
        (8, 5, 0.07, 2.0e-5),
        (11, 10, 0.05, 5.0e-6),
    ):
        x = torch.linspace(0.0, 1.0, num_points)
        points = torch.stack((x, 0.1 * x.square()), dim=-1)
        cells = torch.stack(
            (torch.arange(num_cells), torch.arange(1, num_cells + 1)),
            dim=-1,
        )
        displacement = 0.1 * torch.randn(
            points.shape,
            generator=generator,
        )

        torch.testing.assert_close(
            compiled(
                points,
                cells,
                displacement,
                length_scale,
                tolerance,
            ),
            operation(
                points,
                cells,
                displacement,
                length_scale,
                tolerance,
            ),
        )

    assert len(compiled_graphs) == 1


def test_torch_compile_runtime_zero_length_uses_exact_dense_branch():
    compiled_graphs = []

    def operation(points, cells, displacement, fixed_points, length_scale):
        return sobolev_deform_points(
            points,
            cells,
            displacement,
            length_scale=length_scale,
            fixed_points=fixed_points,
            max_iterations=16,
            tolerance=1.0e-12,
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
    cells = torch.tensor([[0, 1], [1, 2]])
    fixed_points = torch.tensor([False, True, False])
    valid_points = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
    displacement = torch.tensor([[0.2], [-0.3], [0.4]], dtype=torch.float64)

    for length_scale in (0.2, 0.3):
        compiled(
            valid_points.clone().requires_grad_(),
            cells,
            displacement.clone().requires_grad_(),
            fixed_points,
            length_scale,
        )
    assert len(compiled_graphs) == 1

    points = torch.zeros_like(valid_points, requires_grad=True)
    displacement = displacement.clone().requires_grad_()
    output = compiled(
        points,
        cells,
        displacement,
        fixed_points,
        0.0,
    )
    weighted_displacement = torch.where(
        (~fixed_points).unsqueeze(-1),
        displacement,
        torch.zeros_like(displacement),
    )
    assert torch.equal(output, points + weighted_displacement)
    assert len(compiled_graphs) == 2

    loss = output.square().sum()
    first_gradients = torch.autograd.grad(
        loss,
        (points, displacement),
        create_graph=True,
    )
    second_loss = sum(gradient.square().sum() for gradient in first_gradients)
    second_gradients = torch.autograd.grad(
        second_loss,
        (points, displacement),
    )

    reference_points = points.detach().clone().requires_grad_()
    reference_displacement = displacement.detach().clone().requires_grad_()
    reference_output = reference_points + torch.where(
        (~fixed_points).unsqueeze(-1),
        reference_displacement,
        torch.zeros_like(reference_displacement),
    )
    reference_first_gradients = torch.autograd.grad(
        reference_output.square().sum(),
        (reference_points, reference_displacement),
        create_graph=True,
    )
    reference_second_loss = sum(
        gradient.square().sum() for gradient in reference_first_gradients
    )
    reference_second_gradients = torch.autograd.grad(
        reference_second_loss,
        (reference_points, reference_displacement),
    )

    for actual, expected in zip(
        first_gradients,
        reference_first_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(
        second_gradients,
        reference_second_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("implementation", "device_name"),
    [("torch", "cpu"), ("warp", "cuda")],
)
def test_public_api_propagates_fake_tensors(implementation, device_name):
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("the Warp Sobolev backend requires CUDA")
    with FakeTensorMode():
        points = torch.empty(
            (2, 5, 2),
            dtype=torch.float64,
            device=device_name,
        )
        cells = torch.empty((4, 2), dtype=torch.long, device=device_name)
        displacement = torch.empty_like(points)
        fixed_points = torch.empty(
            (2, 5),
            dtype=torch.bool,
            device=device_name,
        )
        output = sobolev_deform_points(
            points,
            cells,
            displacement,
            length_scale=0.2,
            fixed_points=fixed_points,
            max_iterations=8,
            tolerance=1.0e-6,
            implementation=implementation,
        )

    assert isinstance(output, FakeTensor)
    assert output.shape == points.shape
    assert output.dtype == points.dtype
    assert output.device == points.device


@pytest.mark.parametrize("implementation", ["torch", "warp", None])
def test_sobolev_deform_points_rejects_cuda_graph_capture(device, implementation):
    device = torch.device(device)
    if device.type != "cuda":
        pytest.skip("CUDA Graph capture requires CUDA")

    x = torch.linspace(0.0, 1.0, 8, device=device)
    points = torch.stack((x, torch.zeros_like(x)), dim=-1)
    cells = torch.stack(
        (torch.arange(7, device=device), torch.arange(1, 8, device=device)),
        dim=-1,
    )
    displacement = torch.stack(
        (torch.zeros_like(x), 0.05 * torch.sin(4 * torch.pi * x)),
        dim=-1,
    )

    sobolev_deform_points(
        points,
        cells,
        displacement,
        length_scale=0.1,
        max_iterations=32,
        implementation=implementation,
    )
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The CUDA Graph is empty.*",
            category=UserWarning,
        )
        with pytest.raises(
            RuntimeError, match="not supported during CUDA Graph capture"
        ):
            with torch.cuda.graph(graph):
                sobolev_deform_points(
                    points,
                    cells,
                    displacement,
                    length_scale=0.1,
                    max_iterations=32,
                    implementation=implementation,
                )


def test_benchmark_generators_execute_registered_cases(device):
    device = torch.device(device)
    forward_labels = []
    for label, args, kwargs in SobolevDeformPoints.make_inputs_forward(device=device):
        forward_labels.append(label)
        output = SobolevDeformPoints.dispatch(
            *args,
            implementation="torch",
            **kwargs,
        )
        assert output.shape == args[0].shape
        assert output.dtype == args[0].dtype
        assert output.device == device
        assert torch.isfinite(output).all()
        SobolevDeformPoints.compare_forward(output, output.detach())

    assert forward_labels == [
        "small-n1024-d2",
        "medium-b4-n4096-d2",
        "float64-n2048-d2",
    ]

    backward_labels = []
    for label, args, kwargs in SobolevDeformPoints.make_inputs_backward(device=device):
        backward_labels.append(label)
        output = SobolevDeformPoints.dispatch(
            *args,
            implementation="torch",
            **kwargs,
        )
        gradients = torch.autograd.grad(
            output.square().mean(),
            (args[0], args[2]),
        )
        for gradient in gradients:
            assert gradient.shape == args[0].shape
            assert torch.isfinite(gradient).all()
            SobolevDeformPoints.compare_backward(gradient, gradient.detach())

    assert backward_labels == ["medium-n2048-d2"]


def test_nonconverged_solve_raises_instead_of_using_inexact_implicit_gradient():
    points = torch.tensor([[0.0], [0.1], [0.4], [1.0]], dtype=torch.float64)
    cells = torch.tensor([[0, 1], [1, 2], [2, 3]])
    displacement = torch.tensor([[0.3], [-0.2], [0.4], [-0.1]], dtype=torch.float64)

    with pytest.raises(RuntimeError, match="PCG did not converge"):
        sobolev_deform_points(
            points,
            cells,
            displacement,
            length_scale=1.0,
            max_iterations=1,
            tolerance=1.0e-14,
            implementation="torch",
        )


def test_validates_solver_and_boundary_inputs():
    points = torch.tensor([[0.0], [1.0]])
    cells = torch.tensor([[0, 1]])
    displacement = torch.zeros_like(points)

    with pytest.raises((TypeError, ValueError), match="displacement"):
        sobolev_deform_points(
            points,
            cells,
            torch.zeros(3, 1),
            length_scale=1.0,
            implementation="torch",
        )
    with pytest.raises((TypeError, ValueError), match="cells"):
        sobolev_deform_points(
            points,
            cells.to(torch.float32),
            displacement,
            length_scale=1.0,
            implementation="torch",
        )
    with pytest.raises((TypeError, ValueError), match="length_scale"):
        sobolev_deform_points(
            points,
            cells,
            displacement,
            length_scale=-1.0,
            implementation="torch",
        )
    with pytest.raises((TypeError, ValueError), match="fixed_points"):
        sobolev_deform_points(
            points,
            cells,
            displacement,
            length_scale=1.0,
            fixed_points=torch.ones(2),
            implementation="torch",
        )
    with pytest.raises((TypeError, ValueError), match="max_iterations"):
        sobolev_deform_points(
            points,
            cells,
            displacement,
            length_scale=1.0,
            max_iterations=0,
            implementation="torch",
        )
    with pytest.raises((TypeError, ValueError), match="tolerance"):
        sobolev_deform_points(
            points,
            cells,
            displacement,
            length_scale=1.0,
            tolerance=0.0,
            implementation="torch",
        )
    with pytest.raises(KeyError, match="implementation"):
        sobolev_deform_points(
            points,
            cells,
            displacement,
            length_scale=1.0,
            implementation="missing",
        )
    with pytest.raises(ValueError, match="outside"):
        sobolev_deform_points(
            points,
            torch.tensor([[0, 2]]),
            displacement,
            length_scale=1.0,
            implementation="torch",
        )
    with pytest.raises(ValueError, match="nondegenerate"):
        sobolev_deform_points(
            torch.tensor([[0.0], [0.0]]),
            cells,
            displacement,
            length_scale=1.0,
            implementation="torch",
        )


def test_warp_backend_rejects_cpu_tensors():
    pytest.importorskip("warp")
    points = torch.tensor([[0.0], [1.0]])
    cells = torch.tensor([[0, 1]])

    with pytest.raises(ValueError, match="requires CUDA"):
        sobolev_deform_points(
            points,
            cells,
            torch.zeros_like(points),
            length_scale=0.2,
            implementation="warp",
        )


def test_warp_backend_rejects_higher_dimensional_simplices(device):
    device = torch.device(device)
    if device.type != "cuda":
        pytest.skip("the Warp Sobolev backend requires CUDA")
    pytest.importorskip("warp")
    points = torch.eye(5, 4, device=device)
    cells = torch.arange(5, device=device).reshape(1, 5)

    with pytest.raises(ValueError, match="segment, triangle, and tetrahedron"):
        sobolev_deform_points(
            points,
            cells,
            torch.zeros_like(points),
            length_scale=0.2,
            implementation="warp",
        )
