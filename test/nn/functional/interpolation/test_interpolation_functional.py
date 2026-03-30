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

import pytest
import torch

from physicsnemo.nn.functional import interpolation
from physicsnemo.nn.functional.interpolation import Interpolation
from test.conftest import requires_module
from test.nn.functional._parity_utils import assert_optional_match, clone_case


# Validate torch backend wrapper path against direct dispatch.
def test_interpolation_torch_wrapper(device: str):
    _, args, kwargs = next(iter(Interpolation.make_inputs_forward(device=device)))
    output = interpolation(*args, implementation="torch", **kwargs)
    reference = Interpolation.dispatch(*args, implementation="torch", **kwargs)
    Interpolation.compare_forward(output, reference)


# Validate benchmark input generation contract for forward inputs.
def test_interpolation_make_inputs_forward(device: str):
    label, args, kwargs = next(iter(Interpolation.make_inputs_forward(device=device)))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    output = Interpolation.dispatch(*args, implementation="torch", **kwargs)
    assert output.ndim == 2


# Validate benchmark input generation contract for backward inputs.
def test_interpolation_make_inputs_backward(device: str):
    label, args, kwargs = next(iter(Interpolation.make_inputs_backward(device=device)))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    query_points, context_grid, _ = args
    assert query_points.requires_grad
    assert context_grid.requires_grad

    output = Interpolation.dispatch(*args, implementation="torch", **kwargs)
    output.sum().backward()

    # Query gradients can be implementation-dependent for this interpolation op.
    assert query_points.grad is None or query_points.grad.shape == query_points.shape
    assert context_grid.grad is not None


# Validate warp and torch parity for forward outputs.
@requires_module("warp")
def test_interpolation_backend_forward_parity(device: str):
    for case_index, (label, args, kwargs) in enumerate(
        Interpolation.make_inputs_forward(device=device)
    ):
        if case_index >= 3:
            break

        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)
        output_torch = Interpolation.dispatch(
            *args_torch, implementation="torch", **kwargs_torch
        )
        output_warp = Interpolation.dispatch(
            *args_warp, implementation="warp", **kwargs_warp
        )
        Interpolation.compare_forward(output_warp, output_torch)


# Validate warp and torch parity for backward gradients.
@requires_module("warp")
def test_interpolation_backend_backward_parity(device: str):
    for case_index, (label, args, kwargs) in enumerate(
        Interpolation.make_inputs_backward(device=device)
    ):
        if case_index >= 3:
            break

        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        query_torch = args_torch[0]
        grid_torch = args_torch[1]
        query_warp = args_warp[0]
        grid_warp = args_warp[1]

        output_torch = Interpolation.dispatch(
            *args_torch, implementation="torch", **kwargs_torch
        )
        output_warp = Interpolation.dispatch(
            *args_warp, implementation="warp", **kwargs_warp
        )
        Interpolation.compare_forward(output_warp, output_torch)

        grad_output = torch.randn_like(output_torch)
        output_torch.backward(grad_output)
        output_warp.backward(grad_output)

        assert_optional_match(
            query_warp.grad,
            query_torch.grad,
            Interpolation.compare_backward,
            mismatch_message=f"query gradient mismatch for case '{label}'",
        )
        assert_optional_match(
            grid_warp.grad,
            grid_torch.grad,
            Interpolation.compare_backward,
            mismatch_message=f"context-grid gradient mismatch for case '{label}'",
        )


# Validate compare-forward hook contract for interpolation.
def test_interpolation_compare_forward_contract(device: str):
    _, args, kwargs = next(iter(Interpolation.make_inputs_forward(device=device)))
    output = Interpolation.dispatch(*args, implementation="torch", **kwargs)
    reference = output.detach().clone()
    Interpolation.compare_forward(output, reference)


# Validate compare-backward hook contract for interpolation.
def test_interpolation_compare_backward_contract(device: str):
    _, args, kwargs = next(iter(Interpolation.make_inputs_backward(device=device)))
    query_points, context_grid, _ = args

    output = Interpolation.dispatch(*args, implementation="torch", **kwargs)
    output.sum().backward()

    assert context_grid.grad is not None
    Interpolation.compare_backward(
        context_grid.grad, context_grid.grad.detach().clone()
    )
    if query_points.grad is not None:
        Interpolation.compare_backward(
            query_points.grad, query_points.grad.detach().clone()
        )


# Validate interpolation API error handling paths.
def test_interpolation_error_handling(device: str):
    _, args, kwargs = next(iter(Interpolation.make_inputs_forward(device=device)))
    invalid_kwargs = dict(kwargs)
    invalid_kwargs["interpolation_type"] = "not-a-valid-interpolation-type"

    with pytest.raises(RuntimeError, match="not supported"):
        interpolation(*args, implementation="torch", **invalid_kwargs)
