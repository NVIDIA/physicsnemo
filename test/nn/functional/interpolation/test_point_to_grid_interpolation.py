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

from contextlib import nullcontext

import pytest
import torch

from physicsnemo.nn.functional import point_to_grid_interpolation
from physicsnemo.nn.functional.interpolation import (
    GridToPointInterpolation,
    PointToGridInterpolation,
)
from test.conftest import requires_module
from test.nn.functional._parity_utils import assert_optional_match, clone_case

_INTERPOLATION_TYPES = (
    "nearest_neighbor",
    "linear",
    "smooth_step_1",
    "smooth_step_2",
    "gaussian",
)


# Build a deterministic interpolation setup for adjoint and parity tests.
def _build_reference_problem(device: torch.device | str, dims: int = 3):
    device = torch.device(device)
    grid = [(-1.0, 2.0, 24)] * dims

    # Build smooth query points in domain interior.
    num_points = 96
    query_points = torch.stack(
        [torch.linspace(-0.2, 1.2, num_points, device=device) for _ in range(dims)],
        axis=-1,
    ).to(torch.float32)

    # Build deterministic point values with two channels.
    point_values = torch.stack(
        (
            torch.sin(query_points.sum(dim=-1)),
            torch.cos(query_points.prod(dim=-1)),
        ),
        dim=-1,
    ).to(torch.float32)

    # Build a deterministic context grid used for adjoint property tests.
    linspace = [torch.linspace(x[0], x[1], x[2], device=device) for x in grid]
    mesh_grid = torch.meshgrid(linspace, indexing="ij")
    mesh_grid = torch.stack(mesh_grid, dim=0)
    context_grid = torch.zeros(
        2, *mesh_grid.shape[1:], device=device, dtype=torch.float32
    )
    context_grid[0] = torch.sin(mesh_grid.sum(dim=0))
    context_grid[1] = torch.cos(mesh_grid.prod(dim=0))

    return query_points, point_values, grid, context_grid


# Validate the torch backend using the adjoint property with grid-to-point.
@pytest.mark.parametrize("mem_speed_trade", [True, False])
@pytest.mark.parametrize("interpolation_type", _INTERPOLATION_TYPES)
def test_point_to_grid_interpolation_torch(
    device: str,
    mem_speed_trade: bool,
    interpolation_type: str,
):
    query_points, point_values, grid, context_grid = _build_reference_problem(device)

    sampled = GridToPointInterpolation.dispatch(
        query_points,
        context_grid,
        grid,
        interpolation_type=interpolation_type,
        mem_speed_trade=mem_speed_trade,
        implementation="torch",
    )
    lhs = (sampled * point_values).sum()

    scattered = PointToGridInterpolation.dispatch(
        query_points,
        point_values,
        grid,
        interpolation_type=interpolation_type,
        mem_speed_trade=mem_speed_trade,
        implementation="torch",
    )
    rhs = (context_grid * scattered).sum()

    torch.testing.assert_close(lhs, rhs, atol=5e-5, rtol=1e-4)


# Validate the warp backend on the same adjoint identity.
@requires_module("warp")
@pytest.mark.parametrize("mem_speed_trade", [True, False])
@pytest.mark.parametrize("interpolation_type", _INTERPOLATION_TYPES)
def test_point_to_grid_interpolation_warp(
    device: str,
    mem_speed_trade: bool,
    interpolation_type: str,
):
    query_points, point_values, grid, context_grid = _build_reference_problem(device)

    warning_context = (
        pytest.warns(
            UserWarning,
            match="ignores mem_speed_trade and always runs the same kernel path",
        )
        if not mem_speed_trade
        else nullcontext()
    )
    with warning_context:
        sampled = GridToPointInterpolation.dispatch(
            query_points,
            context_grid,
            grid,
            interpolation_type=interpolation_type,
            mem_speed_trade=mem_speed_trade,
            implementation="warp",
        )
        scattered = PointToGridInterpolation.dispatch(
            query_points,
            point_values,
            grid,
            interpolation_type=interpolation_type,
            mem_speed_trade=mem_speed_trade,
            implementation="warp",
        )

    lhs = (sampled * point_values).sum()
    rhs = (context_grid * scattered).sum()
    torch.testing.assert_close(lhs, rhs, atol=5e-5, rtol=1e-4)


# Validate API-level alias behavior and input error handling.
def test_point_to_grid_interpolation_error_handling(device: str):
    query_points, point_values, grid, _ = _build_reference_problem(device, dims=2)

    # Check top-level functional export path.
    output = point_to_grid_interpolation(
        query_points,
        point_values,
        grid,
        interpolation_type="linear",
        mem_speed_trade=True,
    )
    assert output.shape == (point_values.shape[1], grid[0][2], grid[1][2])

    # Check that the public API defaults to warp when available.
    if "warp" in PointToGridInterpolation.available_implementations():
        warp_output = PointToGridInterpolation.dispatch(
            query_points,
            point_values,
            grid,
            interpolation_type="linear",
            mem_speed_trade=True,
            implementation="warp",
        )
        PointToGridInterpolation.compare_forward(output, warp_output)

    # Unsupported dimensionality.
    with pytest.raises(ValueError, match="supports 1-3D grids"):
        PointToGridInterpolation.dispatch(
            torch.randn(16, 4, device=device, dtype=torch.float32),
            torch.randn(16, 2, device=device, dtype=torch.float32),
            [(-1.0, 1.0, 8)] * 4,
            interpolation_type="linear",
            mem_speed_trade=True,
            implementation="torch",
        )

    # Query shape mismatch for grid dimensionality.
    with pytest.raises(ValueError, match="must have shape"):
        PointToGridInterpolation.dispatch(
            torch.randn(16, 3, device=device, dtype=torch.float32),
            torch.randn(16, 2, device=device, dtype=torch.float32),
            [(-1.0, 1.0, 8)] * 2,
            interpolation_type="linear",
            mem_speed_trade=True,
            implementation="torch",
        )

    # point_values shape mismatch.
    with pytest.raises(ValueError, match="point_values must have shape"):
        PointToGridInterpolation.dispatch(
            query_points,
            torch.randn(4, 4, 4, device=device, dtype=torch.float32),
            grid,
            interpolation_type="linear",
            mem_speed_trade=True,
            implementation="torch",
        )

    # Leading dimension mismatch.
    with pytest.raises(ValueError, match="same leading dimension"):
        PointToGridInterpolation.dispatch(
            query_points,
            point_values[:-1],
            grid,
            interpolation_type="linear",
            mem_speed_trade=True,
            implementation="torch",
        )

    # Query-point gradients are supported and should execute without error.
    query_with_grad = query_points.detach().clone().requires_grad_(True)
    values_with_grad = point_values.detach().clone().requires_grad_(True)
    grad_output = torch.randn(
        values_with_grad.shape[1], grid[0][2], grid[1][2], device=device
    )
    output = PointToGridInterpolation.dispatch(
        query_with_grad,
        values_with_grad,
        grid,
        interpolation_type="linear",
        mem_speed_trade=True,
        implementation="torch",
    )
    output.backward(grad_output)
    assert query_with_grad.grad is not None
    assert values_with_grad.grad is not None

    # dtype checks.
    with pytest.raises(TypeError, match="query_points must be float32"):
        PointToGridInterpolation.dispatch(
            query_points.to(torch.float64),
            point_values,
            grid,
            interpolation_type="linear",
            mem_speed_trade=True,
            implementation="torch",
        )
    with pytest.raises(TypeError, match="point_values must be float32"):
        PointToGridInterpolation.dispatch(
            query_points,
            point_values.to(torch.float64),
            grid,
            interpolation_type="linear",
            mem_speed_trade=True,
            implementation="torch",
        )


# Compare torch and warp forward outputs on benchmark representative inputs.
@requires_module("warp")
def test_point_to_grid_interpolation_backend_forward_parity(device: str):
    for _label, args, kwargs in PointToGridInterpolation.make_inputs_forward(
        device=device
    ):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        out_torch = PointToGridInterpolation.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        out_warp = PointToGridInterpolation.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        PointToGridInterpolation.compare_forward(out_warp, out_torch)


# Compare torch and warp backward gradients on benchmark representative inputs.
@requires_module("warp")
def test_point_to_grid_interpolation_backend_backward_parity(device: str):
    for label, args, kwargs in PointToGridInterpolation.make_inputs_backward(
        device=device
    ):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        query_torch, values_torch, _ = args_torch
        query_warp, values_warp, _ = args_warp

        out_torch = PointToGridInterpolation.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        out_warp = PointToGridInterpolation.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        PointToGridInterpolation.compare_forward(out_warp, out_torch)

        grad_out = torch.randn_like(out_torch)
        out_torch.backward(grad_out)
        out_warp.backward(grad_out)

        assert_optional_match(
            query_warp.grad,
            query_torch.grad,
            PointToGridInterpolation.compare_backward,
            mismatch_message=(
                f"query gradient mismatch for case '{label}' on device '{device}'"
            ),
        )
        assert_optional_match(
            values_warp.grad,
            values_torch.grad,
            PointToGridInterpolation.compare_backward,
            mismatch_message=(
                f"point-values gradient mismatch for case '{label}' on device '{device}'"
            ),
        )


# Compare one-grad-input backward behavior for query-only and point-only cases.
@requires_module("warp")
def test_point_to_grid_interpolation_backend_backward_optional_grads(device: str):
    for query_requires_grad, values_requires_grad in ((True, False), (False, True)):
        for label, args, kwargs in PointToGridInterpolation.make_inputs_backward(
            device=device
        ):
            args_torch, kwargs_torch = clone_case(args, kwargs)
            args_warp, kwargs_warp = clone_case(args, kwargs)

            query_torch, values_torch, _ = args_torch
            query_warp, values_warp, _ = args_warp
            query_torch.requires_grad_(query_requires_grad)
            values_torch.requires_grad_(values_requires_grad)
            query_warp.requires_grad_(query_requires_grad)
            values_warp.requires_grad_(values_requires_grad)

            out_torch = PointToGridInterpolation.dispatch(
                *args_torch,
                implementation="torch",
                **kwargs_torch,
            )
            out_warp = PointToGridInterpolation.dispatch(
                *args_warp,
                implementation="warp",
                **kwargs_warp,
            )
            PointToGridInterpolation.compare_forward(out_warp, out_torch)

            if (
                query_requires_grad
                and not values_requires_grad
                and kwargs_torch["interpolation_type"] == "nearest_neighbor"
            ):
                assert not out_torch.requires_grad
                assert out_warp.requires_grad
                continue

            if not out_torch.requires_grad and not out_warp.requires_grad:
                assert query_torch.grad is None
                assert query_warp.grad is None
                assert values_torch.grad is None
                assert values_warp.grad is None
                continue

            grad_out = torch.randn_like(out_torch)
            out_torch.backward(grad_out)
            out_warp.backward(grad_out)

            assert_optional_match(
                query_warp.grad,
                query_torch.grad,
                PointToGridInterpolation.compare_backward,
                mismatch_message=(
                    f"query gradient mismatch (None handling) for case '{label}' "
                    f"on device '{device}'"
                ),
            )
            assert_optional_match(
                values_warp.grad,
                values_torch.grad,
                PointToGridInterpolation.compare_backward,
                mismatch_message=(
                    f"point-values gradient mismatch (None handling) for case '{label}' "
                    f"on device '{device}'"
                ),
            )


# Validate benchmark input generation contract for forward interpolation cases.
def test_point_to_grid_interpolation_make_inputs_forward(device: str):
    label, args, kwargs = next(
        iter(PointToGridInterpolation.make_inputs_forward(device=device))
    )
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    output = PointToGridInterpolation.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    assert output.ndim >= 2
    assert output.shape[0] == args[1].shape[1]


# Validate benchmark input generation contract for backward interpolation cases.
def test_point_to_grid_interpolation_make_inputs_backward(device: str):
    label, args, kwargs = next(
        iter(PointToGridInterpolation.make_inputs_backward(device=device))
    )
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    query_points, point_values, _ = args
    assert query_points.requires_grad
    assert point_values.requires_grad
    if not point_values.is_leaf:
        point_values.retain_grad()

    output = PointToGridInterpolation.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    if output.requires_grad:
        output.sum().backward()
        # Some interpolation modes may not backpropagate through query points.
        assert (query_points.grad is not None) or (point_values.grad is not None)
