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

"""Tests for differentiable local/global template fitting."""

from __future__ import annotations

import inspect
from typing import Literal

import pytest
import torch

import physicsnemo.nn.functional as functional
from benchmarks.physicsnemo.nn.functional.registry import FUNCTIONAL_SPECS
from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.nn.functional.geometry.deform import fit as fit_module
from physicsnemo.nn.functional.geometry.deform._fit_torch_impl import (
    _cg_solve_device,
    _cg_solve_eager,
    _implicit_spd_solve,
    _unique_edges,
)
from physicsnemo.nn.functional.geometry.deform.fit import (
    FitTemplatePoints,
    fit_template_points,
)
from physicsnemo.nn.functional.geometry.point_to_mesh_distance import (
    point_to_mesh_distance,
)


def _small_template(dtype: torch.dtype = torch.float64):
    points = torch.tensor(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )
    triangles = torch.tensor(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=torch.int64,
    )
    return points, triangles


def _nonrigid_target(dtype: torch.dtype = torch.float64):
    points = torch.tensor(
        [
            [-0.45, -0.45, 0.25],
            [0.45, -0.45, 0.25],
            [0.45, 0.45, 0.55],
            [-0.45, 0.45, 0.55],
            [0.0, 0.0, 0.38],
        ],
        dtype=dtype,
    )
    triangles = torch.tensor(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=torch.int64,
    )
    return points, triangles


def _edge_length_distortion(
    reference: torch.Tensor,
    deformed: torch.Tensor,
    triangles: torch.Tensor,
) -> torch.Tensor:
    edges = _unique_edges(triangles)
    reference_lengths = torch.linalg.vector_norm(
        reference[edges[:, 0]] - reference[edges[:, 1]], dim=-1
    )
    deformed_lengths = torch.linalg.vector_norm(
        deformed[edges[:, 0]] - deformed[edges[:, 1]], dim=-1
    )
    return ((deformed_lengths / reference_lengths - 1.0) ** 2).mean()


def _stable_backend_case(dtype: torch.dtype = torch.float32):
    """Return a case whose queries project inside one unambiguous target face."""

    template = torch.tensor(
        [[-0.2, -0.2, 0.0], [0.2, -0.2, 0.0], [0.0, 0.2, 0.0]],
        dtype=dtype,
    )
    target = torch.tensor(
        [[-2.0, -2.0, 0.3], [2.0, -2.0, 0.3], [0.0, 2.0, 0.3]],
        dtype=dtype,
    )
    triangles = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    return template, triangles, target, triangles.clone()


def test_fit_error_decreases():
    template, template_triangles = _small_template()
    target, target_triangles = _nonrigid_target()
    initial_distance, _ = point_to_mesh_distance(
        target, target_triangles, template, squared=True
    )

    fitted = fit_template_points(
        template,
        template_triangles,
        target,
        target_triangles,
        fit_weight=1.0,
        arap_weight=0.2,
        steps=8,
        cg_tolerance=1.0e-12,
    )
    final_distance, _ = point_to_mesh_distance(
        target, target_triangles, fitted, squared=True
    )

    assert final_distance.mean() < 0.4 * initial_distance.mean()


def test_arap_reduces_edge_length_distortion():
    template, template_triangles = _small_template()
    target, target_triangles = _nonrigid_target()
    unconstrained = fit_template_points(
        template,
        template_triangles,
        target,
        target_triangles,
        arap_weight=0.0,
        steps=8,
        cg_tolerance=1.0e-12,
    )
    regularized = fit_template_points(
        template,
        template_triangles,
        target,
        target_triangles,
        arap_weight=1.0,
        steps=8,
        cg_tolerance=1.0e-12,
    )

    unconstrained_distortion = _edge_length_distortion(
        template, unconstrained, template_triangles
    )
    regularized_distortion = _edge_length_distortion(
        template, regularized, template_triangles
    )
    assert regularized_distortion < 0.1 * unconstrained_distortion


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_rigid_translation_is_recovered(dtype):
    template, triangles = _small_template(dtype)
    translation = template.new_tensor((0.0, 0.0, 0.3))
    target = template + translation
    tolerance = 1.0e-6 if dtype == torch.float32 else 1.0e-12

    fitted = fit_template_points(
        template,
        triangles,
        target,
        triangles,
        fit_weight=1.0,
        arap_weight=2.0,
        steps=2,
        cg_tolerance=tolerance,
    )

    comparison_tolerance = 50.0 * torch.finfo(dtype).eps
    torch.testing.assert_close(
        fitted,
        target,
        atol=comparison_tolerance,
        rtol=comparison_tolerance,
    )


def test_gradients_are_finite_for_template_and_target():
    template, triangles = _small_template()
    target, target_triangles = _nonrigid_target()
    template.requires_grad_()
    target.requires_grad_()

    fitted = fit_template_points(
        template,
        triangles,
        target,
        target_triangles,
        steps=3,
        cg_tolerance=1.0e-12,
    )
    probe = torch.linspace(-0.7, 0.9, fitted.numel()).reshape_as(fitted)
    template_gradient, target_gradient = torch.autograd.grad(
        (fitted * probe).sum(), (template, target)
    )

    assert torch.isfinite(template_gradient).all()
    assert torch.isfinite(target_gradient).all()
    assert torch.count_nonzero(template_gradient) > 0
    assert torch.count_nonzero(target_gradient) > 0


def test_directional_gradient_matches_finite_difference_inside_target_face():
    dtype = torch.float64
    template = torch.tensor(
        [[-0.2, -0.2, 0.0], [0.2, -0.2, 0.0], [0.0, 0.2, 0.0]],
        dtype=dtype,
        requires_grad=True,
    )
    target = torch.tensor(
        [[-2.0, -2.0, 0.3], [2.0, -2.0, 0.3], [0.0, 2.0, 0.3]],
        dtype=dtype,
        requires_grad=True,
    )
    triangles = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    probe = torch.tensor(
        [[0.2, -0.3, 0.7], [0.4, 0.1, -0.2], [-0.5, 0.6, 0.3]],
        dtype=dtype,
    )
    template_direction = torch.tensor(
        [[0.2, -0.1, 0.05], [-0.1, 0.15, -0.02], [0.07, -0.04, 0.1]],
        dtype=dtype,
    )
    target_direction = torch.tensor(
        [[0.1, 0.02, -0.03], [-0.05, 0.07, 0.04], [0.03, -0.08, 0.02]],
        dtype=dtype,
    )

    def objective(template_value, target_value):
        fitted = fit_template_points(
            template_value,
            triangles,
            target_value,
            triangles,
            fit_weight=1.0,
            arap_weight=0.4,
            steps=4,
            cg_tolerance=1.0e-13,
            cg_max_iterations=64,
        )
        return (fitted * probe).sum()

    value = objective(template, target)
    template_gradient, target_gradient = torch.autograd.grad(value, (template, target))
    analytic = (template_gradient * template_direction).sum() + (
        target_gradient * target_direction
    ).sum()

    epsilon = 1.0e-6
    with torch.no_grad():
        positive = objective(
            template + epsilon * template_direction,
            target + epsilon * target_direction,
        )
        negative = objective(
            template - epsilon * template_direction,
            target - epsilon * target_direction,
        )
    finite_difference = (positive - negative) / (2.0 * epsilon)
    torch.testing.assert_close(analytic, finite_difference, atol=2.0e-7, rtol=2.0e-5)


def test_matrix_free_cg_matches_dense_system_and_adjoint():
    _, triangles = _small_template()
    edges = _unique_edges(triangles)
    generator = torch.Generator().manual_seed(42)
    right_hand_side = torch.randn((5, 3), generator=generator, dtype=torch.float64)
    right_hand_side.requires_grad_()
    fit_weight = 1.3
    arap_weight = 0.7

    matrix_free = _implicit_spd_solve(
        right_hand_side,
        edges,
        fit_weight,
        arap_weight,
        1.0e-13,
        64,
    )
    laplacian = torch.zeros((5, 5), dtype=torch.float64)
    for start, end in edges.tolist():
        laplacian[start, start] += 1.0
        laplacian[end, end] += 1.0
        laplacian[start, end] -= 1.0
        laplacian[end, start] -= 1.0
    system = (
        fit_weight * torch.eye(5, dtype=torch.float64) + 2.0 * arap_weight * laplacian
    )
    dense = torch.linalg.solve(system, right_hand_side)
    torch.testing.assert_close(matrix_free, dense, atol=1.0e-11, rtol=1.0e-11)

    probe = torch.randn((5, 3), generator=generator, dtype=torch.float64)
    matrix_free_gradient = torch.autograd.grad(
        (matrix_free * probe).sum(), right_hand_side
    )[0]
    dense_gradient = torch.linalg.solve(system.T, probe)
    torch.testing.assert_close(
        matrix_free_gradient, dense_gradient, atol=1.0e-11, rtol=1.0e-11
    )


def test_device_controlled_cg_matches_eager_and_dense():
    _, triangles = _small_template()
    edges = _unique_edges(triangles)
    generator = torch.Generator().manual_seed(73)
    right_hand_side = torch.randn((5, 3), generator=generator, dtype=torch.float64)
    fit_weight = 1.3
    arap_weight = 0.7

    eager = _cg_solve_eager(
        right_hand_side,
        edges,
        fit_weight,
        arap_weight,
        1.0e-12,
        64,
    )
    device_controlled = _cg_solve_device(
        right_hand_side,
        edges,
        fit_weight,
        arap_weight,
        1.0e-12,
        64,
    )

    laplacian = torch.zeros((5, 5), dtype=torch.float64)
    for start, end in edges.tolist():
        laplacian[start, start] += 1.0
        laplacian[end, end] += 1.0
        laplacian[start, end] -= 1.0
        laplacian[end, start] -= 1.0
    system = (
        fit_weight * torch.eye(5, dtype=torch.float64) + 2.0 * arap_weight * laplacian
    )
    dense = torch.linalg.solve(system, right_hand_side)

    torch.testing.assert_close(device_controlled, eager, atol=1.0e-11, rtol=1.0e-11)
    torch.testing.assert_close(device_controlled, dense, atol=1.0e-11, rtol=1.0e-11)


def test_device_controlled_cg_handles_zero_rhs():
    _, triangles = _small_template()
    edges = _unique_edges(triangles)
    right_hand_side = torch.zeros((5, 3), dtype=torch.float32)
    solution = _cg_solve_device(
        right_hand_side,
        edges,
        1.0,
        0.2,
        1.0e-6,
        32,
    )
    torch.testing.assert_close(solution, torch.zeros_like(solution))


def test_device_controlled_cg_matches_eager_initial_tolerance_exit():
    _, triangles = _small_template()
    edges = _unique_edges(triangles)
    right_hand_side = torch.ones((5, 3), dtype=torch.float32)
    eager = _cg_solve_eager(right_hand_side, edges, 1.0, 0.2, 1.0, 32)
    device_controlled = _cg_solve_device(
        right_hand_side,
        edges,
        1.0,
        0.2,
        1.0,
        32,
    )
    torch.testing.assert_close(eager, torch.zeros_like(eager))
    torch.testing.assert_close(device_controlled, eager)


def test_device_controlled_cg_reports_nonconvergence():
    _, triangles = _small_template()
    edges = _unique_edges(triangles)
    right_hand_side = torch.arange(15, dtype=torch.float64).reshape(5, 3) + 1.0
    with pytest.raises(RuntimeError, match="did not converge within 1 iterations"):
        _cg_solve_device(
            right_hand_side,
            edges,
            1.0,
            0.7,
            1.0e-14,
            1,
        )


def test_device_controlled_cg_fullgraph_compile_and_no_host_scalar_source():
    _, triangles = _small_template(torch.float32)
    edges = _unique_edges(triangles)
    right_hand_side = torch.arange(15, dtype=torch.float32).reshape(5, 3) + 1.0

    def solve(values):
        return _cg_solve_device(values, edges, 1.0, 0.2, 1.0e-6, 32)

    compiled = torch.compile(solve, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(right_hand_side), solve(right_hand_side))

    def implicit_solve(values):
        return _implicit_spd_solve(values, edges, 1.0, 0.2, 1.0e-6, 32)

    compiled_implicit = torch.compile(
        implicit_solve,
        backend="eager",
        fullgraph=True,
    )
    torch.testing.assert_close(
        compiled_implicit(right_hand_side),
        implicit_solve(right_hand_side),
    )

    source = inspect.getsource(_cg_solve_device)
    assert ".item(" not in source
    assert "float(" not in source
    assert "bool(" not in source


def test_full_fit_torch_compile_fullgraph_forward_and_backward():
    template, template_triangles, target, target_triangles = _stable_backend_case()

    def apply(template_values, target_values):
        return fit_template_points(
            template_values,
            template_triangles,
            target_values,
            target_triangles,
            arap_weight=0.4,
            steps=2,
            cg_tolerance=1.0e-6,
            implementation="torch",
        )

    def evaluate(operation):
        template_values = template.clone().requires_grad_()
        target_values = target.clone().requires_grad_()
        output = operation(template_values, target_values)
        gradients = torch.autograd.grad(
            output.square().sum(),
            (template_values, target_values),
        )
        return output.detach(), gradients

    eager_output, eager_gradients = evaluate(apply)
    compiled_output, compiled_gradients = evaluate(torch.compile(apply, fullgraph=True))

    torch.testing.assert_close(compiled_output, eager_output, atol=2.0e-6, rtol=2.0e-6)
    for compiled_gradient, eager_gradient in zip(
        compiled_gradients,
        eager_gradients,
        strict=True,
    ):
        assert torch.isfinite(compiled_gradient).all()
        torch.testing.assert_close(
            compiled_gradient,
            eager_gradient,
            atol=2.0e-5,
            rtol=2.0e-5,
        )


@pytest.mark.parametrize(
    ("keyword", "value", "exception"),
    [
        ("fit_weight", 0.0, ValueError),
        ("fit_weight", float("inf"), ValueError),
        ("arap_weight", -0.1, ValueError),
        ("steps", -1, ValueError),
        ("steps", 1.5, TypeError),
        ("cg_tolerance", 0.0, ValueError),
        ("cg_max_iterations", 0, ValueError),
    ],
)
def test_solver_option_validation(keyword, value, exception):
    template, triangles = _small_template()
    target, target_triangles = _nonrigid_target()
    with pytest.raises(exception):
        fit_template_points(
            template,
            triangles,
            target,
            target_triangles,
            **{keyword: value},
        )


def test_input_validation():
    template, triangles = _small_template()
    target, target_triangles = _nonrigid_target()

    with pytest.raises(ValueError, match="template_points must have shape"):
        fit_template_points(template.unsqueeze(0), triangles, target, target_triangles)
    with pytest.raises(TypeError, match="same dtype"):
        fit_template_points(template.float(), triangles, target, target_triangles)
    with pytest.raises(TypeError, match="template_triangles must have dtype"):
        fit_template_points(template, triangles.float(), target, target_triangles)
    invalid_triangles = triangles.clone()
    invalid_triangles[0, 1] = invalid_triangles[0, 0]
    with pytest.raises(RuntimeError, match="repeated indices"):
        fit_template_points(template, invalid_triangles, target, target_triangles)
    out_of_range = triangles.clone()
    out_of_range[0, 0] = template.shape[0]
    with pytest.raises(RuntimeError, match="outside"):
        fit_template_points(template, out_of_range, target, target_triangles)
    nonfinite = template.clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="finite coordinates"):
        fit_template_points(nonfinite, triangles, target, target_triangles)


def test_public_signature_and_function_spec():
    assert functional.fit_template_points is fit_template_points
    assert issubclass(FitTemplatePoints, FunctionSpec)
    assert FitTemplatePoints in FUNCTIONAL_SPECS
    assert not hasattr(functional, "FitTemplatePoints")
    assert FitTemplatePoints.implementations() == ("warp", "torch")
    signature = inspect.signature(fit_template_points)
    assert tuple(signature.parameters) == (
        "template_points",
        "template_triangles",
        "target_points",
        "target_triangles",
        "fit_weight",
        "arap_weight",
        "steps",
        "cg_tolerance",
        "cg_max_iterations",
        "implementation",
    )
    assert signature.parameters["implementation"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["implementation"].annotation == (
        Literal["torch", "warp"] | None
    )


def test_default_dispatch_selects_device_backend(device, monkeypatch):
    device = torch.device(device)
    template, template_triangles, target, target_triangles = _stable_backend_case()
    template = template.to(device)
    template_triangles = template_triangles.to(device)
    target = target.to(device)
    target_triangles = target_triangles.to(device)
    calls = []

    def solver_spy(template_values, *_args, point_implementation):
        calls.append(point_implementation)
        return template_values

    # Both registered implementations resolve this name in the public module;
    # the selected correspondence backend is passed into the shared solver.
    monkeypatch.setattr(fit_module, "fit_template_points_torch", solver_spy)

    warp_impl = FitTemplatePoints._get_impls()["warp"]
    expected = "warp" if device.type == "cuda" and warp_impl.available else "torch"
    if device.type == "cuda" and not warp_impl.available:
        FunctionSpec._fallback_warned.discard(FitTemplatePoints._class_key())
        with pytest.warns(RuntimeWarning, match="falling back to implementation"):
            automatic = fit_template_points(
                template,
                template_triangles,
                target,
                target_triangles,
            )
    else:
        automatic = fit_template_points(
            template,
            template_triangles,
            target,
            target_triangles,
        )

    assert calls == [expected]
    torch.testing.assert_close(automatic, template)


def test_warp_forward_and_gradient_parity_on_cpu():
    template, template_triangles, target, target_triangles = _stable_backend_case()
    probe = torch.tensor([[0.2, -0.3, 0.7], [0.4, 0.1, -0.2], [-0.5, 0.6, 0.3]])

    def evaluate(implementation: str):
        template_values = template.clone().requires_grad_()
        target_values = target.clone().requires_grad_()
        output = fit_template_points(
            template_values,
            template_triangles,
            target_values,
            target_triangles,
            arap_weight=0.4,
            steps=3,
            cg_tolerance=1.0e-6,
            implementation=implementation,
        )
        gradients = torch.autograd.grad(
            (output * probe).sum(), (template_values, target_values)
        )
        return output, gradients

    torch_output, torch_gradients = evaluate("torch")
    warp_output, warp_gradients = evaluate("warp")
    FitTemplatePoints.compare_forward(warp_output, torch_output)
    for actual, expected in zip(warp_gradients, torch_gradients, strict=True):
        FitTemplatePoints.compare_backward(actual, expected)


def test_explicit_warp_rejects_float64_without_downcasting():
    template, template_triangles, target, target_triangles = _stable_backend_case(
        torch.float64
    )
    with pytest.raises(TypeError, match="only torch.float32"):
        fit_template_points(
            template,
            template_triangles,
            target,
            target_triangles,
            implementation="warp",
        )


def test_function_spec_benchmark_case_runs_with_both_backends_on_cpu():
    label, args, kwargs = next(iter(FitTemplatePoints.make_inputs_forward("cpu")))
    assert label
    torch_output = FitTemplatePoints.dispatch(*args, **kwargs, implementation="torch")
    warp_output = FitTemplatePoints.dispatch(*args, **kwargs, implementation="warp")
    FitTemplatePoints.compare_forward(warp_output, torch_output)


def test_zero_steps_returns_distinct_identity_tensor():
    template, triangles = _small_template()
    target, target_triangles = _nonrigid_target()
    fitted = fit_template_points(template, triangles, target, target_triangles, steps=0)
    torch.testing.assert_close(fitted, template)
    assert fitted.data_ptr() != template.data_ptr()
