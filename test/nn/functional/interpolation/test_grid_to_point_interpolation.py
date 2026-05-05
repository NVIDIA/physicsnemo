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

from physicsnemo.nn.functional import (
    grid_to_point_interpolation,
)
from physicsnemo.nn.functional import (
    interpolation as deprecated_interpolation,
)
from physicsnemo.nn.functional.interpolation.grid_to_point_interpolation import (
    GridToPointInterpolation,
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


# Build a simple analytic interpolation problem used by backend and API tests.
def _build_reference_problem(device: torch.device | str):
    device = torch.device(device)
    grid = [(-1.0, 2.0, 30), (-1.0, 2.0, 30), (-1.0, 2.0, 30)]
    linspace = [torch.linspace(x[0], x[1], x[2], device=device) for x in grid]
    mesh_grid = torch.meshgrid(linspace, indexing="ij")
    mesh_grid = torch.stack(mesh_grid, dim=0)
    context_grid = torch.sin(
        mesh_grid[0:1, :, :] + mesh_grid[1:2, :, :] ** 2 + mesh_grid[2:3, :, :] ** 3
    )
    num_points = 100
    query_points = torch.stack(
        [
            torch.linspace(0.0, 1.0, num_points, device=device),
            torch.linspace(0.0, 1.0, num_points, device=device),
            torch.linspace(0.0, 1.0, num_points, device=device),
        ],
        axis=-1,
    )
    reference = torch.sin(
        query_points[:, 0:1] + query_points[:, 1:2] ** 2 + query_points[:, 2:3] ** 3
    )
    return query_points, context_grid, grid, reference, num_points


# Validate the torch backend on an analytic interpolation setup.
@pytest.mark.parametrize("mem_speed_trade", [True, False])
@pytest.mark.parametrize("interpolation_type", _INTERPOLATION_TYPES)
def test_grid_to_point_interpolation_torch(
    device: str,
    mem_speed_trade: bool,
    interpolation_type: str,
):
    query_points, context_grid, grid, reference, num_points = _build_reference_problem(
        device
    )
    output = GridToPointInterpolation.dispatch(
        query_points,
        context_grid,
        grid,
        interpolation_type=interpolation_type,
        mem_speed_trade=mem_speed_trade,
        implementation="torch",
    )
    error = torch.linalg.norm((output - reference) / num_points)
    assert float(error) < 1e-2


# Validate the warp backend on the same analytic interpolation setup.
@requires_module("warp")
@pytest.mark.parametrize("mem_speed_trade", [True, False])
@pytest.mark.parametrize("interpolation_type", _INTERPOLATION_TYPES)
def test_grid_to_point_interpolation_warp(
    device: str,
    mem_speed_trade: bool,
    interpolation_type: str,
):
    query_points, context_grid, grid, reference, num_points = _build_reference_problem(
        device
    )
    warning_context = (
        pytest.warns(
            UserWarning,
            match="ignores mem_speed_trade and always runs the same kernel path",
        )
        if not mem_speed_trade
        else nullcontext()
    )
    with warning_context:
        output = GridToPointInterpolation.dispatch(
            query_points,
            context_grid,
            grid,
            interpolation_type=interpolation_type,
            mem_speed_trade=mem_speed_trade,
            implementation="warp",
        )
    error = torch.linalg.norm((output - reference) / num_points)
    assert float(error) < 1e-2


# Validate deprecated alias and input/error handling paths.
def test_grid_to_point_interpolation_error_handling(device: str):
    grid = [(-1.0, 1.0, 16)]
    query_points = torch.linspace(0.0, 1.0, 8, device=device).unsqueeze(-1)
    context_grid = torch.sin(torch.linspace(-1.0, 1.0, 16, device=device).unsqueeze(0))

    # Check deprecated alias warning and behavior.
    with pytest.warns(DeprecationWarning, match="`interpolation` is deprecated"):
        old_output = deprecated_interpolation(
            query_points,
            context_grid,
            grid=grid,
            interpolation_type="linear",
            mem_speed_trade=True,
        )

    new_output = grid_to_point_interpolation(
        query_points,
        context_grid,
        grid=grid,
        interpolation_type="linear",
        mem_speed_trade=True,
        implementation="torch",
    )
    GridToPointInterpolation.compare_forward(old_output, new_output)

    # Deprecated alias should still allow explicit backend overrides.
    if "warp" in GridToPointInterpolation.available_implementations():
        with pytest.warns(DeprecationWarning, match="`interpolation` is deprecated"):
            old_warp_output = deprecated_interpolation(
                query_points,
                context_grid,
                grid=grid,
                interpolation_type="linear",
                mem_speed_trade=True,
                implementation="warp",
            )
        warp_output = GridToPointInterpolation.dispatch(
            query_points,
            context_grid,
            grid,
            interpolation_type="linear",
            mem_speed_trade=True,
            implementation="warp",
        )
        GridToPointInterpolation.compare_forward(old_warp_output, warp_output)

    # Check that the non-deprecated API defaults to warp when available.
    if "warp" in GridToPointInterpolation.available_implementations():
        default_output = grid_to_point_interpolation(
            query_points,
            context_grid,
            grid=grid,
            interpolation_type="linear",
            mem_speed_trade=True,
        )
        warp_output = GridToPointInterpolation.dispatch(
            query_points,
            context_grid,
            grid,
            interpolation_type="linear",
            mem_speed_trade=True,
            implementation="warp",
        )
        GridToPointInterpolation.compare_forward(default_output, warp_output)

    # Check torch validation paths.
    query_points, context_grid, grid, _, _ = _build_reference_problem(device)
    with pytest.raises(RuntimeError, match="not supported"):
        GridToPointInterpolation.dispatch(
            query_points,
            context_grid,
            grid,
            interpolation_type="invalid_type",
            mem_speed_trade=True,
            implementation="torch",
        )

    # Check warp-only validation paths when warp is available.
    if "warp" not in GridToPointInterpolation.available_implementations():
        return

    # Check warp mem_speed_trade warning behavior.
    query_points, context_grid, grid, _, _ = _build_reference_problem(device)
    with pytest.warns(
        UserWarning,
        match="ignores mem_speed_trade and always runs the same kernel path",
    ):
        GridToPointInterpolation.dispatch(
            query_points,
            context_grid,
            grid,
            interpolation_type="linear",
            mem_speed_trade=False,
            implementation="warp",
        )

    query_points, context_grid, grid, _, _ = _build_reference_problem(device)
    with pytest.raises(ValueError, match="must be one of"):
        GridToPointInterpolation.dispatch(
            query_points,
            context_grid,
            grid,
            interpolation_type="invalid_type",
            mem_speed_trade=True,
            implementation="warp",
        )

    grid = [(-1.0, 1.0, 16), (-1.0, 1.0, 16)]
    context_grid = torch.randn(1, 16, 16, device=device)
    query_points = torch.randn(32, 3, device=device)
    with pytest.raises(ValueError, match="last dimension 2"):
        GridToPointInterpolation.dispatch(
            query_points,
            context_grid,
            grid,
            interpolation_type="linear",
            mem_speed_trade=True,
            implementation="warp",
        )

    from physicsnemo.nn.functional.interpolation.grid_to_point_interpolation._warp_impl.op import (  # noqa: PLC0415
        interpolation_impl,
    )

    query_points = torch.randn(8, 1, device=device)
    context_grid = torch.randn(1, 16, device=device)
    with pytest.raises(ValueError, match="supports 1-3D grids"):
        interpolation_impl(
            query_points=query_points,
            context_grid=context_grid,
            grid_meta=torch.empty((0, 3), dtype=torch.float32),
            interp_id=1,
            mem_speed_trade=True,
        )

    grid_4d = [(-1.0, 1.0, 8)] * 4
    query_points_4d = torch.randn(16, 4, device=device)
    context_grid_4d = torch.randn(1, 8, 8, 8, 8, device=device)
    with pytest.raises(ValueError, match="supports 1-3D grids"):
        GridToPointInterpolation.dispatch(
            query_points_4d,
            context_grid_4d,
            grid_4d,
            interpolation_type="linear",
            mem_speed_trade=True,
            implementation="warp",
        )

    # Check non-contiguous input handling for warp launches.
    query_points, context_grid, grid, _, _ = _build_reference_problem(device)
    query_points = query_points.transpose(0, 1).contiguous().transpose(0, 1)
    context_grid = context_grid.permute(0, 3, 2, 1)
    assert not query_points.is_contiguous()
    assert not context_grid.is_contiguous()
    args_torch = (
        query_points.detach().clone().requires_grad_(True),
        context_grid.detach().clone().requires_grad_(True),
        grid,
    )
    args_warp = (
        query_points.detach().clone().requires_grad_(True),
        context_grid.detach().clone().requires_grad_(True),
        grid,
    )
    kwargs = {"interpolation_type": "smooth_step_2", "mem_speed_trade": True}
    out_torch = GridToPointInterpolation.dispatch(
        *args_torch,
        implementation="torch",
        **kwargs,
    )
    out_warp = GridToPointInterpolation.dispatch(
        *args_warp,
        implementation="warp",
        **kwargs,
    )
    GridToPointInterpolation.compare_forward(out_warp, out_torch)
    grad_out = torch.randn_like(out_torch)
    out_torch.backward(grad_out)
    out_warp.backward(grad_out)
    GridToPointInterpolation.compare_backward(args_warp[0].grad, args_torch[0].grad)
    GridToPointInterpolation.compare_backward(args_warp[1].grad, args_torch[1].grad)

    # Check backward parity for one-grad-input cases.
    for query_requires_grad, grid_requires_grad in ((True, False), (False, True)):
        for label, args, kwargs in GridToPointInterpolation.make_inputs_backward(
            device=device
        ):
            args_torch, kwargs_torch = clone_case(args, kwargs)
            args_warp, kwargs_warp = clone_case(args, kwargs)
            query_torch, grid_torch, _ = args_torch
            query_warp, grid_warp, _ = args_warp
            query_torch.requires_grad_(query_requires_grad)
            grid_torch.requires_grad_(grid_requires_grad)
            query_warp.requires_grad_(query_requires_grad)
            grid_warp.requires_grad_(grid_requires_grad)
            out_torch = GridToPointInterpolation.dispatch(
                *args_torch,
                implementation="torch",
                **kwargs_torch,
            )
            out_warp = GridToPointInterpolation.dispatch(
                *args_warp,
                implementation="warp",
                **kwargs_warp,
            )
            GridToPointInterpolation.compare_forward(out_warp, out_torch)
            if (
                query_requires_grad
                and not grid_requires_grad
                and kwargs_torch["interpolation_type"] == "nearest_neighbor"
            ):
                assert not out_torch.requires_grad
                assert out_warp.requires_grad
                continue
            if not out_torch.requires_grad and not out_warp.requires_grad:
                assert query_torch.grad is None
                assert query_warp.grad is None
                assert grid_torch.grad is None
                assert grid_warp.grad is None
                continue
            grad_out = torch.randn_like(out_torch)
            out_torch.backward(grad_out)
            out_warp.backward(grad_out)
            assert_optional_match(
                query_warp.grad,
                query_torch.grad,
                GridToPointInterpolation.compare_backward,
                mismatch_message=(
                    f"query gradient mismatch (None handling) for case '{label}' "
                    f"on device '{device}'"
                ),
            )
            assert_optional_match(
                grid_warp.grad,
                grid_torch.grad,
                GridToPointInterpolation.compare_backward,
                mismatch_message=(
                    f"context-grid gradient mismatch (None handling) for case '{label}' "
                    f"on device '{device}'"
                ),
            )

    # Check context grid spatial size validation path.
    grid = [(-1.0, 1.0, 16), (-1.0, 1.0, 16)]
    context_grid = torch.randn(1, 15, 16, device=device)
    query_points = torch.randn(32, 2, device=device)
    with pytest.raises(ValueError, match="context_grid shape must match grid sizes"):
        GridToPointInterpolation.dispatch(
            query_points,
            context_grid,
            grid,
            interpolation_type="linear",
            mem_speed_trade=True,
            implementation="warp",
        )


# Compare torch and warp forward outputs on benchmark representative inputs.
@requires_module("warp")
def test_grid_to_point_interpolation_backend_forward_parity(device: str):
    for _label, args, kwargs in GridToPointInterpolation.make_inputs_forward(
        device=device
    ):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)
        out_torch = GridToPointInterpolation.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        out_warp = GridToPointInterpolation.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        GridToPointInterpolation.compare_forward(out_warp, out_torch)


# Compare torch and warp backward gradients on benchmark representative inputs.
@requires_module("warp")
def test_grid_to_point_interpolation_backend_backward_parity(device: str):
    for label, args, kwargs in GridToPointInterpolation.make_inputs_backward(
        device=device
    ):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        query_torch, grid_torch, _ = args_torch
        query_warp, grid_warp, _ = args_warp

        out_torch = GridToPointInterpolation.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        out_warp = GridToPointInterpolation.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        GridToPointInterpolation.compare_forward(out_warp, out_torch)

        if not out_torch.requires_grad and not out_warp.requires_grad:
            assert query_torch.grad is None
            assert query_warp.grad is None
            assert grid_torch.grad is None
            assert grid_warp.grad is None
            continue

        grad_out = torch.randn_like(out_torch)
        out_torch.backward(grad_out)
        out_warp.backward(grad_out)

        assert_optional_match(
            query_warp.grad,
            query_torch.grad,
            GridToPointInterpolation.compare_backward,
            mismatch_message=(
                f"query gradient mismatch (None handling) for case '{label}' "
                f"on device '{device}'"
            ),
        )
        assert_optional_match(
            grid_warp.grad,
            grid_torch.grad,
            GridToPointInterpolation.compare_backward,
            mismatch_message=(
                f"context-grid gradient mismatch (None handling) for case '{label}' "
                f"on device '{device}'"
            ),
        )


# Validate benchmark input generation contract for forward interpolation cases.
def test_grid_to_point_interpolation_make_inputs_forward(device: str):
    label, args, kwargs = next(
        iter(GridToPointInterpolation.make_inputs_forward(device=device))
    )
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    output = GridToPointInterpolation.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    assert output.ndim == 2
    assert output.shape[0] == args[0].shape[0]


# Validate benchmark input generation contract for backward interpolation cases.
def test_grid_to_point_interpolation_make_inputs_backward(device: str):
    label, args, kwargs = next(
        iter(GridToPointInterpolation.make_inputs_backward(device=device))
    )
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    query_points, context_grid, _ = args
    assert query_points.requires_grad
    assert context_grid.requires_grad

    output = GridToPointInterpolation.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    if output.requires_grad:
        output.sum().backward()
        # Some interpolation modes (for example nearest-neighbor) are not
        # differentiable w.r.t. query points, so only require at least one
        # gradient-carrying input to receive gradients.
        assert (query_points.grad is not None) or (context_grid.grad is not None)
