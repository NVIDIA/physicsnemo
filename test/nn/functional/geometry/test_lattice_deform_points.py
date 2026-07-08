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

import importlib

import pytest
import torch

import physicsnemo.nn.functional as functional
from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.nn.functional import lattice_deform_points
from physicsnemo.nn.functional.geometry import LatticeDeformPoints
from physicsnemo.nn.functional.geometry import (
    lattice_deform_points as geometry_lattice_deform_points,
)
from test.conftest import requires_module
from test.nn.functional._parity_utils import clone_case

lattice_torch_impl = importlib.import_module(
    "physicsnemo.nn.functional.geometry.lattice_deform_points._torch_impl"
)
lattice_warp_impl = importlib.import_module(
    "physicsnemo.nn.functional.geometry.lattice_deform_points._warp_impl"
)


def _constant_controls(
    translation: torch.Tensor,
    lattice_size: int = 3,
) -> torch.Tensor:
    """Create a channel-first constant displacement lattice."""
    dim = translation.shape[0]
    view_shape = (dim,) + (1,) * dim
    return (
        translation.reshape(view_shape).expand((dim,) + (lattice_size,) * dim).clone()
    )


def _make_small_parity_case(
    device: torch.device | str,
    *,
    dim: int,
    interpolation_type: str,
    requires_grad: bool,
) -> tuple[tuple[object, ...], dict[str, object]]:
    """Build deterministic interior points, controls, and weights for parity."""
    device = torch.device(device)
    parameter = torch.arange(17, device=device, dtype=torch.float32) / 17
    points = torch.stack(
        [
            0.12 + 0.76 * torch.remainder((2 * axis + 1) * parameter + 0.13, 1.0)
            for axis in range(dim)
        ],
        dim=-1,
    )
    control_shape = (dim,) + (4,) * dim
    control_values = torch.linspace(
        -0.2,
        0.3,
        dim * 4**dim,
        device=device,
    )
    controls = torch.sin(control_values.reshape(control_shape) * torch.pi)
    weights = torch.linspace(0.2, 1.2, 17, device=device)
    if requires_grad:
        points.requires_grad_(True)
        controls.requires_grad_(True)
        weights.requires_grad_(True)
    return (
        (points, controls, [(0.0, 1.0)] * dim),
        {
            "interpolation_type": interpolation_type,
            "point_weights": weights,
        },
    )


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize(
    "interpolation_type", ["linear", "smooth_step_1", "smooth_step_2"]
)
def test_lattice_deform_points_torch(
    device: str,
    dim: int,
    interpolation_type: str,
):
    """Zero controls are identity and constant controls translate in 1D--3D."""
    dtype = torch.float32
    bounds = [(-1.0 - axis, 2.0 + axis) for axis in range(dim)]
    lower = torch.tensor([bound[0] for bound in bounds], device=device, dtype=dtype)
    upper = torch.tensor([bound[1] for bound in bounds], device=device, dtype=dtype)
    points = torch.stack([lower, 0.25 * lower + 0.75 * upper, upper])
    original_points = points.clone()

    zero_controls = torch.zeros((dim,) + (3,) * dim, device=device, dtype=dtype)
    identity = lattice_deform_points(
        points,
        zero_controls,
        bounds,
        interpolation_type=interpolation_type,
        implementation="torch",
    )
    torch.testing.assert_close(identity, points)

    translation = torch.arange(1, dim + 1, device=device, dtype=dtype) / 10
    controls = _constant_controls(translation)
    original_controls = controls.clone()
    translated = lattice_deform_points(
        points,
        controls,
        bounds,
        interpolation_type=interpolation_type,
        implementation="torch",
    )

    torch.testing.assert_close(translated, points + translation)
    torch.testing.assert_close(points, original_points)
    torch.testing.assert_close(controls, original_controls)
    assert translated.shape == points.shape
    assert translated.dtype == points.dtype
    assert translated.device == points.device


@requires_module("warp")
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize(
    "interpolation_type", ["linear", "smooth_step_1", "smooth_step_2"]
)
def test_lattice_deform_points_warp(
    device: str,
    dim: int,
    interpolation_type: str,
):
    """Warp independently satisfies identity and constant-translation oracles."""
    dtype = torch.float32
    bounds = [(-1.0 - axis, 2.0 + axis) for axis in range(dim)]
    lower = torch.tensor([bound[0] for bound in bounds], device=device, dtype=dtype)
    upper = torch.tensor([bound[1] for bound in bounds], device=device, dtype=dtype)
    points = torch.stack([lower, 0.25 * lower + 0.75 * upper, upper])

    zero_controls = torch.zeros((dim,) + (3,) * dim, device=device, dtype=dtype)
    identity = LatticeDeformPoints.dispatch(
        points,
        zero_controls,
        bounds,
        interpolation_type=interpolation_type,
        implementation="warp",
    )
    torch.testing.assert_close(identity, points)

    translation = torch.arange(1, dim + 1, device=device, dtype=dtype) / 10
    controls = _constant_controls(translation)
    translated = LatticeDeformPoints.dispatch(
        points,
        controls,
        bounds,
        interpolation_type=interpolation_type,
        implementation="warp",
    )
    torch.testing.assert_close(translated, points + translation)


def test_lattice_deform_points_linear_reproduces_affine_displacement(device: str):
    """Linear lattice interpolation exactly reproduces an affine vector field."""
    dtype = torch.float32
    bounds = [(-2.0, 1.0), (1.0, 5.0), (-1.0, 0.5)]
    lattice_shape = (4, 3, 5)
    axes = [
        torch.linspace(lower, upper, size, device=device, dtype=dtype)
        for (lower, upper), size in zip(bounds, lattice_shape, strict=True)
    ]
    coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)
    matrix = torch.tensor(
        [[0.10, -0.05, 0.02], [0.03, 0.20, -0.04], [-0.08, 0.01, 0.15]],
        device=device,
        dtype=dtype,
    )
    bias = torch.tensor([0.2, -0.1, 0.05], device=device, dtype=dtype)
    controls = torch.einsum("ij,j...->i...", matrix, coordinates) + bias.reshape(
        3, 1, 1, 1
    )
    points = torch.tensor(
        [[-1.55, 1.8, -0.65], [-0.25, 3.4, -0.10], [0.65, 4.2, 0.30]],
        device=device,
        dtype=dtype,
    )

    actual = lattice_deform_points(
        points,
        controls,
        bounds,
        interpolation_type="linear",
        implementation="torch",
    )
    expected = points + points @ matrix.T + bias
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_lattice_deform_points_linear_upper_boundary_gradient(
    device: str,
    implementation: str,
    dim: int,
):
    """Upper-bound points use the final interior linear stencil and gradient."""
    if implementation == "warp" and (
        "warp" not in LatticeDeformPoints.available_implementations()
    ):
        pytest.skip("Warp is not available")

    dtype = torch.float32
    bounds = [(-1.0 - axis, 2.0 + axis) for axis in range(dim)]
    axes = [
        torch.linspace(lower, upper, 3, device=device, dtype=dtype)
        for lower, upper in bounds
    ]
    coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)
    matrix = 0.05 * torch.arange(
        1,
        dim * dim + 1,
        device=device,
        dtype=dtype,
    ).reshape(dim, dim)
    bias = torch.linspace(0.02, 0.02 * dim, dim, device=device, dtype=dtype)
    controls = torch.einsum("ij,j...->i...", matrix, coordinates) + bias.reshape(
        (dim,) + (1,) * dim
    )

    lower = torch.tensor([bound[0] for bound in bounds], device=device, dtype=dtype)
    upper = torch.tensor([bound[1] for bound in bounds], device=device, dtype=dtype)
    points = ((lower + upper) / 2).expand(dim + 1, dim).clone()
    points[:-1].diagonal().copy_(upper)
    points[-1].copy_(upper)
    points.requires_grad_(True)

    output = lattice_deform_points(
        points,
        controls,
        bounds,
        interpolation_type="linear",
        implementation=implementation,
    )
    expected = points.detach() + points.detach() @ matrix.T + bias
    torch.testing.assert_close(output, expected, atol=3e-5, rtol=3e-5)

    output.sum().backward()
    expected_gradient = torch.ones_like(points) + matrix.sum(dim=0)
    torch.testing.assert_close(points.grad, expected_gradient, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_lattice_deform_points_large_world_offset_uses_stable_unit_coordinates(
    device: str,
    implementation: str,
):
    """Large float64 origins do not collapse delegated lattice metadata."""
    if implementation == "warp" and (
        "warp" not in LatticeDeformPoints.available_implementations()
    ):
        pytest.skip("Warp is not available")

    bounds = [(1.0e8, 1.0e8 + 1.0)]
    points = torch.tensor(
        [[1.0e8 + 0.25], [1.0e8 + 0.75]], device=device, dtype=torch.float64
    )
    controls = torch.tensor([[0.0, 0.5, 1.0]], device=device, dtype=torch.float64)
    expected_displacement = torch.tensor(
        [[0.25], [0.75]], device=device, dtype=torch.float64
    )

    actual = lattice_deform_points(
        points,
        controls,
        bounds,
        interpolation_type="linear",
        implementation=implementation,
    )
    torch.testing.assert_close(
        actual - points, expected_displacement, atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize(
    "interpolation_type", ["linear", "smooth_step_1", "smooth_step_2"]
)
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_lattice_deform_points_control_values_are_exact_at_lattice_nodes(
    device: str,
    interpolation_type: str,
    implementation: str,
):
    """Every supported basis interpolates its control values at grid nodes."""
    if implementation == "warp" and (
        "warp" not in LatticeDeformPoints.available_implementations()
    ):
        pytest.skip("Warp is not available")

    dtype = torch.float32
    bounds = [(-1.0, 2.0), (3.0, 5.0)]
    axes = [
        torch.linspace(-1.0, 2.0, 3, device=device, dtype=dtype),
        torch.linspace(3.0, 5.0, 4, device=device, dtype=dtype),
    ]
    meshgrid = torch.meshgrid(*axes, indexing="ij")
    points = torch.stack(meshgrid, dim=-1).reshape(-1, 2)
    controls = torch.arange(2 * 3 * 4, device=device, dtype=dtype).reshape(2, 3, 4)
    expected_displacements = controls.movedim(0, -1).reshape(-1, 2)

    actual = lattice_deform_points(
        points,
        controls,
        bounds,
        interpolation_type=interpolation_type,
        implementation=implementation,
    )
    torch.testing.assert_close(actual - points, expected_displacements)


def test_lattice_deform_points_point_weights_scale_and_mask_displacement(device: str):
    """Weights may fix, reverse, or amplify individual point displacements."""
    points = torch.tensor(
        [[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]], device=device, dtype=torch.float32
    )
    translation = torch.tensor([0.4, -0.2], device=device)
    controls = _constant_controls(translation, lattice_size=2)
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    weights = torch.tensor([0.0, -0.5, 2.0], device=device)

    weighted = lattice_deform_points(
        points,
        controls,
        bounds,
        point_weights=weights,
        implementation="torch",
    )
    torch.testing.assert_close(weighted, points + weights[:, None] * translation)

    mask = torch.tensor([False, True, False], device=device)
    masked = lattice_deform_points(
        points,
        controls,
        bounds,
        point_weights=mask,
        implementation="torch",
    )
    torch.testing.assert_close(masked, points + mask[:, None] * translation)


def test_lattice_deform_points_analytic_backward_for_points_controls_and_weights(
    device: str,
):
    """Gradients propagate through query points, controls, and point weights."""
    dtype = torch.float32
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    axis = torch.linspace(0.0, 1.0, 3, device=device, dtype=dtype)
    coordinates = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=0)
    matrix = torch.tensor([[0.20, -0.10], [0.05, 0.30]], device=device, dtype=dtype)
    bias = torch.tensor([0.10, -0.04], device=device, dtype=dtype)
    controls = (
        torch.einsum("ij,j...->i...", matrix, coordinates) + bias.reshape(2, 1, 1)
    ).requires_grad_(True)
    points = torch.tensor(
        [[0.15, 0.25], [0.45, 0.65], [0.80, 0.35]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    weights = torch.tensor(
        [0.25, 0.75, 1.50], device=device, dtype=dtype, requires_grad=True
    )

    output = lattice_deform_points(
        points,
        controls,
        bounds,
        interpolation_type="linear",
        point_weights=weights,
        implementation="torch",
    )
    output.sum().backward()

    displacement = points.detach() @ matrix.T + bias
    expected_point_grad = torch.ones_like(points) + weights.detach()[
        :, None
    ] * matrix.sum(dim=0)
    expected_weight_grad = displacement.sum(dim=-1)
    torch.testing.assert_close(points.grad, expected_point_grad, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(weights.grad, expected_weight_grad, atol=2e-5, rtol=2e-5)
    assert controls.grad is not None
    assert torch.isfinite(controls.grad).all()
    torch.testing.assert_close(
        controls.grad.flatten(1).sum(dim=1),
        weights.detach().sum().expand(2),
        atol=2e-5,
        rtol=2e-5,
    )


@requires_module("warp")
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize(
    "interpolation_type", ["linear", "smooth_step_1", "smooth_step_2"]
)
def test_lattice_deform_points_backend_forward_parity(
    device: str,
    dim: int,
    interpolation_type: str,
):
    """Torch and Warp forward results agree on independent input clones."""
    args, kwargs = _make_small_parity_case(
        device,
        dim=dim,
        interpolation_type=interpolation_type,
        requires_grad=False,
    )
    args_torch, kwargs_torch = clone_case(args, kwargs)
    args_warp, kwargs_warp = clone_case(args, kwargs)

    output_torch = LatticeDeformPoints.dispatch(
        *args_torch,
        implementation="torch",
        **kwargs_torch,
    )
    output_warp = LatticeDeformPoints.dispatch(
        *args_warp,
        implementation="warp",
        **kwargs_warp,
    )
    LatticeDeformPoints.compare_forward(output_warp, output_torch)


@requires_module("warp")
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize(
    "interpolation_type", ["linear", "smooth_step_1", "smooth_step_2"]
)
def test_lattice_deform_points_backend_backward_parity(
    device: str,
    dim: int,
    interpolation_type: str,
):
    """Point, control, and weight gradients agree across backends."""
    args, kwargs = _make_small_parity_case(
        device,
        dim=dim,
        interpolation_type=interpolation_type,
        requires_grad=True,
    )
    args_torch, kwargs_torch = clone_case(args, kwargs)
    args_warp, kwargs_warp = clone_case(args, kwargs)
    points_torch, controls_torch, _ = args_torch
    points_warp, controls_warp, _ = args_warp
    weights_torch = kwargs_torch["point_weights"]
    weights_warp = kwargs_warp["point_weights"]

    output_torch = LatticeDeformPoints.dispatch(
        *args_torch,
        implementation="torch",
        **kwargs_torch,
    )
    output_warp = LatticeDeformPoints.dispatch(
        *args_warp,
        implementation="warp",
        **kwargs_warp,
    )
    LatticeDeformPoints.compare_forward(output_warp, output_torch)

    grad_output = torch.linspace(
        0.5,
        1.5,
        output_torch.numel(),
        device=output_torch.device,
        dtype=output_torch.dtype,
    ).reshape_as(output_torch)
    gradients_torch = torch.autograd.grad(
        output_torch,
        (points_torch, controls_torch, weights_torch),
        grad_outputs=grad_output,
    )
    gradients_warp = torch.autograd.grad(
        output_warp,
        (points_warp, controls_warp, weights_warp),
        grad_outputs=grad_output.clone(),
    )
    for gradient_warp, gradient_torch in zip(
        gradients_warp, gradients_torch, strict=True
    ):
        assert gradient_warp is not None
        assert gradient_torch is not None
        LatticeDeformPoints.compare_backward(gradient_warp, gradient_torch)


def test_lattice_deform_points_error_handling(device: str):
    """Invalid point and control tensors fail before backend dispatch."""
    points = torch.tensor([[0.25, 0.75]], device=device)
    controls = torch.zeros(2, 2, 2, device=device)
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    with pytest.raises(TypeError, match="points must be a torch.Tensor"):
        lattice_deform_points([[0.25, 0.75]], controls, bounds, implementation="torch")
    with pytest.raises(TypeError, match="control_displacements must be a torch.Tensor"):
        lattice_deform_points(points, [[[0.0]]], bounds, implementation="torch")
    with pytest.raises(ValueError, match="points must have shape"):
        lattice_deform_points(
            points.unsqueeze(0), controls, bounds, implementation="torch"
        )
    with pytest.raises(TypeError, match="floating-point dtype"):
        lattice_deform_points(
            points.to(torch.int64), controls, bounds, implementation="torch"
        )
    with pytest.raises(TypeError, match="supports float32 and float64"):
        lattice_deform_points(
            points.to(torch.float16),
            controls.to(torch.float16),
            bounds,
            implementation="torch",
        )
    with pytest.raises(ValueError, match="supports 1D, 2D, or 3D"):
        lattice_deform_points(
            torch.zeros(1, 4, device=device),
            torch.zeros(4, 2, 2, 2, 2, device=device),
            [(0.0, 1.0)] * 4,
            implementation="torch",
        )
    with pytest.raises(ValueError, match="expected rank"):
        lattice_deform_points(
            points,
            torch.zeros(2, 2, device=device),
            bounds,
            implementation="torch",
        )
    with pytest.raises(ValueError, match="one channel"):
        lattice_deform_points(
            points,
            torch.zeros(3, 2, 2, device=device),
            bounds,
            implementation="torch",
        )
    with pytest.raises(TypeError, match="control_displacements must have a floating"):
        lattice_deform_points(
            points,
            controls.to(torch.int64),
            bounds,
            implementation="torch",
        )
    with pytest.raises(TypeError, match="same dtype"):
        lattice_deform_points(
            points, controls.to(torch.float64), bounds, implementation="torch"
        )


def test_lattice_deform_points_bounds_error_handling(device: str):
    """Malformed lattice bounds and resolutions produce actionable errors."""
    points = torch.tensor([[0.25, 0.75]], device=device)
    controls = torch.zeros(2, 2, 2, device=device)
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    with pytest.raises(ValueError, match="one .* pair per dimension"):
        lattice_deform_points(points, controls, [(0.0, 1.0)], implementation="torch")
    with pytest.raises(TypeError, match="lattice_bounds must be a sequence"):
        lattice_deform_points(points, controls, 1.0, implementation="torch")
    with pytest.raises(TypeError, match="lattice_bounds must be a sequence"):
        lattice_deform_points(points, controls, "01", implementation="torch")
    with pytest.raises(TypeError, match=r"lattice_bounds\[0\].*pair"):
        lattice_deform_points(
            points,
            controls,
            [0.0, (0.0, 1.0)],
            implementation="torch",
        )
    with pytest.raises(TypeError, match=r"lattice_bounds\[0\].*pair"):
        lattice_deform_points(
            points,
            controls,
            ["01", (0.0, 1.0)],
            implementation="torch",
        )
    with pytest.raises(ValueError, match="must be a .* pair"):
        lattice_deform_points(
            points,
            controls,
            [(0.0,), (0.0, 1.0)],
            implementation="torch",
        )
    with pytest.raises(TypeError, match="numeric lower and upper"):
        lattice_deform_points(
            points,
            controls,
            [("lower", 1.0), (0.0, 1.0)],
            implementation="torch",
        )
    with pytest.raises(ValueError, match="must be finite"):
        lattice_deform_points(
            points,
            controls,
            [(0.0, float("inf")), (0.0, 1.0)],
            implementation="torch",
        )
    with pytest.raises(ValueError, match="lower < upper"):
        lattice_deform_points(
            points,
            controls,
            [(1.0, 0.0), (0.0, 1.0)],
            implementation="torch",
        )
    with pytest.raises(ValueError, match="at least two control points"):
        lattice_deform_points(
            points,
            torch.zeros(2, 1, 2, device=device),
            bounds,
            implementation="torch",
        )


def test_lattice_deform_points_interpolation_error_handling(device: str):
    """Unsupported interpolation selectors are rejected before dispatch."""
    points = torch.tensor([[0.25, 0.75]], device=device)
    controls = torch.zeros(2, 2, 2, device=device)
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    with pytest.raises(ValueError, match="interpolation_type"):
        lattice_deform_points(
            points,
            controls,
            bounds,
            interpolation_type="gaussian",
            implementation="torch",
        )
    with pytest.raises(TypeError, match="interpolation_type must be a string"):
        lattice_deform_points(
            points,
            controls,
            bounds,
            interpolation_type=None,
            implementation="torch",
        )


def test_lattice_deform_points_weight_error_handling(device: str):
    """Point weights must match the point count, dtype, and device."""
    points = torch.tensor([[0.25, 0.75]], device=device)
    controls = torch.zeros(2, 2, 2, device=device)
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    with pytest.raises(TypeError, match="point_weights must be a torch.Tensor"):
        lattice_deform_points(
            points,
            controls,
            bounds,
            point_weights=[1.0],
            implementation="torch",
        )
    with pytest.raises(ValueError, match="point_weights must have shape"):
        lattice_deform_points(
            points,
            controls,
            bounds,
            point_weights=torch.ones(1, 1, device=device),
            implementation="torch",
        )
    with pytest.raises(TypeError, match="point_weights must be bool"):
        lattice_deform_points(
            points,
            controls,
            bounds,
            point_weights=torch.ones(1, dtype=torch.int64, device=device),
            implementation="torch",
        )


def test_lattice_deform_points_bounds_dtype_limits(device: str):
    """Bounds must remain distinguishable and finite in the point dtype."""
    with pytest.raises(ValueError, match="not distinguishable"):
        lattice_deform_points(
            torch.tensor([[1.0e8]], device=device),
            torch.zeros(1, 2, device=device),
            [(1.0e8, 1.0e8 + 1.0)],
            implementation="torch",
        )
    with pytest.raises(ValueError, match="finite positive span"):
        lattice_deform_points(
            torch.tensor([[0.0]], device=device),
            torch.zeros(1, 2, device=device),
            [(-3.0e38, 3.0e38)],
            implementation="torch",
        )
    with pytest.raises(ValueError, match="finite positive span"):
        lattice_deform_points(
            torch.tensor([[0.0]], device=device, dtype=torch.float64),
            torch.zeros(1, 2, device=device, dtype=torch.float64),
            [(-1.0e308, 1.0e308)],
            implementation="torch",
        )


def test_lattice_deform_points_device_error_handling(device: str):
    """Controls and point weights must share the point device."""
    points = torch.tensor([[0.25, 0.75]], device=device)
    controls = torch.zeros(2, 2, 2, device=device)
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    if points.device.type == "cuda":
        mismatch_device = torch.device("cpu")
    elif torch.cuda.is_available():
        mismatch_device = torch.device("cuda")
    else:
        mismatch_device = None
    if mismatch_device is not None:
        with pytest.raises(ValueError, match="same device"):
            lattice_deform_points(
                points,
                controls.to(mismatch_device),
                bounds,
                implementation="torch",
            )
        with pytest.raises(ValueError, match="same device"):
            lattice_deform_points(
                points,
                controls,
                bounds,
                point_weights=torch.ones(1, device=mismatch_device),
                implementation="torch",
            )


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_lattice_deform_points_empty_points(device: str, implementation: str):
    """Empty inputs retain finite zero-valued autograd edges without a launch."""
    if implementation == "warp" and (
        "warp" not in LatticeDeformPoints.available_implementations()
    ):
        pytest.skip("Warp is not available")
    points = torch.empty(0, 2, device=device, requires_grad=True)
    controls = torch.zeros(2, 2, 2, device=device)
    controls[0, 0, 0] = float("inf")
    controls.requires_grad_()
    weights = torch.empty(0, device=device, requires_grad=True)
    output = lattice_deform_points(
        points,
        controls,
        [(0.0, 1.0), (0.0, 1.0)],
        point_weights=weights,
        implementation=implementation,
    )
    assert output.shape == (0, 2)
    gradients = torch.autograd.grad(output.sum(), (points, controls, weights))
    for gradient, value in zip(gradients, (points, controls, weights), strict=True):
        torch.testing.assert_close(gradient, torch.zeros_like(value))


@pytest.mark.parametrize(
    ("adapter_module", "adapter", "expected_implementation"),
    [
        (lattice_torch_impl, lattice_torch_impl.lattice_deform_points_torch, "torch"),
        (lattice_warp_impl, lattice_warp_impl.lattice_deform_points_warp, "warp"),
    ],
)
def test_lattice_deform_points_adapter_selects_child_backend(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
    adapter_module,
    adapter,
    expected_implementation: str,
):
    """Each adapter explicitly selects its corresponding interpolation backend."""
    selected_implementations = []

    def fake_interpolation(
        query_points: torch.Tensor,
        context_grid: torch.Tensor,
        _grid,
        **kwargs,
    ) -> torch.Tensor:
        selected_implementations.append(kwargs["implementation"])
        return query_points.new_zeros((query_points.shape[0], context_grid.shape[0]))

    monkeypatch.setattr(
        adapter_module,
        "grid_to_point_interpolation",
        fake_interpolation,
    )
    points = torch.tensor([[0.25, 0.75]], device=device)
    controls = torch.zeros(2, 2, 2, device=device)
    output = adapter(points, controls, [(0.0, 1.0), (0.0, 1.0)])

    assert selected_implementations == [expected_implementation]
    torch.testing.assert_close(output, points)


def test_lattice_deform_points_torch_compile(device: str):
    """The Torch path supports full-graph AOT capture and backward."""
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    def deform(points: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        return lattice_deform_points(
            points,
            controls,
            bounds,
            implementation="torch",
        )

    compiled_deform = torch.compile(deform, backend="aot_eager", fullgraph=True)
    base_points = torch.tensor(
        [[0.25, 0.75]],
        device=device,
    )
    base_controls = torch.linspace(-0.1, 0.1, 8, device=device).reshape(2, 2, 2)
    eager_points = base_points.clone().requires_grad_(True)
    eager_controls = base_controls.clone().requires_grad_(True)
    compiled_points = base_points.clone().requires_grad_(True)
    compiled_controls = base_controls.clone().requires_grad_(True)

    eager_output = deform(eager_points, eager_controls)
    compiled_output = compiled_deform(compiled_points, compiled_controls)
    torch.testing.assert_close(compiled_output, eager_output)

    grad_output = torch.tensor([[0.7, -1.3]], device=device)
    eager_gradients = torch.autograd.grad(
        eager_output,
        (eager_points, eager_controls),
        grad_outputs=grad_output,
    )
    compiled_gradients = torch.autograd.grad(
        compiled_output,
        (compiled_points, compiled_controls),
        grad_outputs=grad_output,
    )
    for compiled_gradient, eager_gradient in zip(
        compiled_gradients,
        eager_gradients,
        strict=True,
    ):
        torch.testing.assert_close(compiled_gradient, eager_gradient)


@requires_module("warp")
def test_lattice_deform_points_warp_compile(device: str):
    """The composite Warp forward remains a full-graph custom-op boundary."""
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    def deform(points: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        return lattice_deform_points(
            points,
            controls,
            bounds,
            implementation="warp",
        )

    points = torch.tensor([[0.25, 0.75]], device=device)
    controls = torch.zeros(2, 2, 2, device=device)
    eager = deform(points, controls)
    compiled_deform = torch.compile(deform, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled_deform(points, controls), eager)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_lattice_deform_points_supported_dtypes_are_preserved(
    device: str,
    dtype: torch.dtype,
    implementation: str,
):
    """Both backends preserve the two supported public dtypes."""
    if implementation == "warp" and (
        "warp" not in LatticeDeformPoints.available_implementations()
    ):
        pytest.skip("Warp is not available")
    points = torch.tensor([[0.25, 0.75]], device=device, dtype=dtype)
    translation = torch.tensor([0.1, -0.2], device=device, dtype=dtype)
    controls = _constant_controls(translation, lattice_size=2)
    output = lattice_deform_points(
        points,
        controls,
        [(0.0, 1.0), (0.0, 1.0)],
        implementation=implementation,
    )
    assert output.dtype == dtype
    torch.testing.assert_close(output, points + translation)


def test_lattice_deform_points_torch_dtype_is_independent_of_default_dtype(
    device: str,
):
    """Torch metadata creation cannot promote explicitly float32 inputs."""
    original_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        points = torch.tensor([[0.25, 0.75]], device=device, dtype=torch.float32)
        translation = torch.tensor([0.1, -0.2], device=device, dtype=torch.float32)
        controls = _constant_controls(translation, lattice_size=2)
        output = lattice_deform_points(
            points,
            controls,
            [(0.0, 1.0), (0.0, 1.0)],
            implementation="torch",
        )
    finally:
        torch.set_default_dtype(original_default_dtype)

    assert output.dtype == torch.float32
    torch.testing.assert_close(output, points + translation)


def test_lattice_deform_points_make_inputs_forward(device: str):
    """Every forward benchmark case is labeled and structurally valid."""
    cases = list(LatticeDeformPoints.make_inputs_forward(device=device))
    assert len(cases) == 3
    for label, args, kwargs in cases:
        points, controls, bounds = args
        weights = kwargs["point_weights"]

        assert isinstance(label, str)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        assert points.ndim == 2
        assert controls.ndim == points.shape[-1] + 1
        assert len(bounds) == points.shape[-1]
        assert weights.shape == (points.shape[0],)
        assert kwargs["interpolation_type"] == "smooth_step_2"
        assert not points.requires_grad
        assert not controls.requires_grad
        assert not weights.requires_grad

    _, args, kwargs = cases[0]
    points = args[0]

    output = LatticeDeformPoints.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    assert output.shape == points.shape
    assert torch.isfinite(output).all()


def test_lattice_deform_points_make_inputs_backward(device: str):
    """Every backward case carries structurally valid differentiable leaves."""
    cases = list(LatticeDeformPoints.make_inputs_backward(device=device))
    assert len(cases) == 3
    for label, args, kwargs in cases:
        points, controls, bounds = args
        weights = kwargs["point_weights"]

        assert isinstance(label, str)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        assert len(bounds) == points.shape[-1]
        assert points.requires_grad
        assert controls.requires_grad
        assert weights.requires_grad

    _, args, kwargs = cases[0]
    points, controls, _ = args
    weights = kwargs["point_weights"]

    output = LatticeDeformPoints.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    output.square().mean().backward()
    for value in (points, controls, weights):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


def test_lattice_deform_points_benchmark_inputs_are_deterministic(device: str):
    """Repeated benchmark generation produces identical labeled cases."""
    for make_inputs in (
        LatticeDeformPoints.make_inputs_forward,
        LatticeDeformPoints.make_inputs_backward,
    ):
        first_cases = list(make_inputs(device=device))
        repeated_cases = list(make_inputs(device=device))
        for first, repeated in zip(first_cases, repeated_cases, strict=True):
            first_label, first_args, first_kwargs = first
            repeated_label, repeated_args, repeated_kwargs = repeated
            assert first_label == repeated_label
            assert first_args[2] == repeated_args[2]
            for first_tensor, repeated_tensor in zip(
                (*first_args[:2], first_kwargs["point_weights"]),
                (*repeated_args[:2], repeated_kwargs["point_weights"]),
                strict=True,
            ):
                torch.testing.assert_close(first_tensor, repeated_tensor)


def test_lattice_deform_points_compare_forward_contract(device: str):
    """The forward comparison hook accepts matches and rejects large errors."""
    reference = torch.randn(7, 2, device=device)
    LatticeDeformPoints.compare_forward(reference.clone(), reference)
    with pytest.raises(AssertionError):
        LatticeDeformPoints.compare_forward(reference + 1.0, reference)


def test_lattice_deform_points_compare_backward_contract(device: str):
    """The backward comparison hook accepts matches and rejects large errors."""
    reference = torch.randn(7, 2, device=device)
    LatticeDeformPoints.compare_backward(reference.clone(), reference)
    with pytest.raises(AssertionError):
        LatticeDeformPoints.compare_backward(reference + 1.0, reference)


def test_lattice_deform_points_public_exports():
    """Top-level and geometry exports resolve to the same public function."""
    assert lattice_deform_points is geometry_lattice_deform_points
    assert lattice_deform_points.__name__ == "lattice_deform_points"
    assert lattice_deform_points.__module__ == (
        "physicsnemo.nn.functional.geometry.lattice_deform_points.lattice_deform_points"
    )
    assert issubclass(LatticeDeformPoints, FunctionSpec)
    assert not hasattr(functional, "LatticeDeformPoints")
    assert LatticeDeformPoints.implementations() == ("warp", "torch")


def test_lattice_deform_points_default_dispatch(device: str):
    """Normal rank-based dispatch produces the documented smooth-step result."""
    points = torch.tensor([[0.25]], device=device)
    controls = torch.tensor([[0.0, 1.0]], device=device)
    output = lattice_deform_points(points, controls, [(0.0, 1.0)])
    torch.testing.assert_close(output, torch.tensor([[0.353515625]], device=device))
