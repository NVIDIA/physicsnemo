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

"""Tests for guarded, normal-directed mesh updates."""

import typing

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import tetrahedron_surface
from physicsnemo.mesh.transformations import (
    NormalUpdateDiagnostics,
    guarded_normal_update,
)


def _square_surface(
    *, reverse_winding: bool = False, dtype=torch.float32, device="cpu"
) -> Mesh:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]], device=device)
    if reverse_winding:
        cells = cells.flip(-1)
    return Mesh(points=points, cells=cells)


def _fan_surface(*, device="cpu") -> Mesh:
    points = torch.tensor(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        device=device,
    )
    cells = torch.tensor([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], device=device)
    return Mesh(points=points, cells=cells)


def _backtracking_curve(*, device="cpu") -> Mesh:
    # Area-weighting the two incident edge normals gives point 1 a +y normal.
    points = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], device=device)
    cells = torch.tensor([[0, 1], [1, 2]], device=device)
    return Mesh(points=points, cells=cells)


def test_mesh_method_type_hints_resolve_at_runtime():
    hints = typing.get_type_hints(Mesh.guarded_normal_update)
    assert hints["return"] == tuple[Mesh, NormalUpdateDiagnostics]


def test_projection_uses_signed_direction_and_is_winding_invariant():
    direction = torch.tensor([2.0, -3.0, 4.0]).expand(4, -1)
    expected_delta = torch.tensor([0.0, 0.0, 1.0]).expand(4, -1)

    outputs = []
    for reverse_winding in (False, True):
        mesh = _square_surface(reverse_winding=reverse_winding)
        updated, diagnostics = guarded_normal_update(
            mesh,
            0.25 * direction,
            smoothing_iterations=0,
            max_backtracks=0,
        )

        assert diagnostics.accepted
        torch.testing.assert_close(updated.points - mesh.points, expected_delta)
        torch.testing.assert_close(mesh.points[:, 2], torch.zeros(4))
        assert diagnostics.proposed_max_step == pytest.approx(1.0)
        assert diagnostics.applied_max_step == pytest.approx(1.0)
        outputs.append(updated.points)

    torch.testing.assert_close(outputs[0], outputs[1])

    # The caller includes sign and step magnitude in the supplied direction.
    mesh = _square_surface()
    objective_gradient = torch.tensor([0.0, 0.0, 1.0]).expand(4, -1)
    updated, _ = guarded_normal_update(
        mesh,
        -0.1 * objective_gradient,
        smoothing_iterations=0,
        max_backtracks=0,
    )
    assert updated.points[:, 2].sum() < mesh.points[:, 2].sum()


def test_scalar_smoothing_reduces_roughness_and_preserves_constants():
    mesh = _fan_surface()
    spike = torch.zeros_like(mesh.points)
    spike[4, 2] = 1.0

    updated, diagnostics = guarded_normal_update(
        mesh,
        spike,
        smoothing_iterations=1,
        smoothing_relaxation=0.2,
        max_backtracks=0,
    )

    assert diagnostics.accepted
    smoothed = (updated.points - mesh.points)[:, 2]
    roughness_before = ((spike[4, 2] - spike[:4, 2]) ** 2).sum()
    roughness_after = ((smoothed[4] - smoothed[:4]) ** 2).sum()
    assert roughness_after < roughness_before
    assert 0.0 < smoothed[4] < 1.0
    assert torch.all(smoothed[:4] > 0.0)

    constant = torch.tensor([0.0, 0.0, 2.0]).expand_as(mesh.points)
    translated, diagnostics = guarded_normal_update(
        mesh,
        constant,
        smoothing_iterations=3,
        smoothing_relaxation=0.2,
        max_backtracks=0,
    )
    assert diagnostics.accepted
    torch.testing.assert_close(translated.points - mesh.points, constant)


def test_smoothing_on_curved_surface_remains_normal_directed(device):
    mesh = tetrahedron_surface.load(device=device)
    direction = torch.tensor(
        [
            [1.0, 2.0, -1.0],
            [-2.0, 1.0, 0.5],
            [0.25, -1.0, 2.0],
            [1.5, 0.5, -0.25],
        ],
        device=device,
    )

    updated, diagnostics = guarded_normal_update(
        mesh,
        0.02 * direction,
        smoothing_iterations=2,
        smoothing_relaxation=0.2,
        max_backtracks=0,
    )

    assert diagnostics.accepted
    delta = updated.points - mesh.points
    normals = mesh.point_normals
    normal_component = (delta * normals).sum(-1, keepdim=True) * normals
    torch.testing.assert_close(delta, normal_component, atol=1e-6, rtol=1e-5)


def test_weights_are_applied_before_one_global_clip_scale():
    mesh = _square_surface()
    direction = torch.zeros_like(mesh.points)
    direction[:, 2] = torch.tensor([100.0, 1.0, 2.0, 4.0])
    weights = torch.tensor([0.0, 1.0, 1.0, 0.25])

    updated, diagnostics = guarded_normal_update(
        mesh,
        direction,
        point_weights=weights,
        smoothing_iterations=0,
        max_step=1.0,
        max_backtracks=0,
    )

    assert diagnostics.accepted
    expected = torch.tensor([0.0, 0.5, 1.0, 0.5])
    torch.testing.assert_close((updated.points - mesh.points)[:, 2], expected)
    assert diagnostics.clip_scale == pytest.approx(0.5)
    assert diagnostics.proposed_max_step == pytest.approx(2.0)
    assert diagnostics.applied_max_step == pytest.approx(1.0)


def test_backtracking_rejects_flip_then_collapse_before_accepting(device):
    mesh = _backtracking_curve(device=device)
    direction = torch.zeros_like(mesh.points)
    direction[1, 1] = -2.0

    updated, diagnostics = guarded_normal_update(
        mesh,
        direction,
        smoothing_iterations=0,
        backtracking_factor=0.5,
        max_backtracks=2,
    )

    assert diagnostics.accepted
    assert diagnostics.n_backtracks == 2
    assert diagnostics.backtracking_scale == pytest.approx(0.25)
    assert diagnostics.clip_scale == pytest.approx(1.0)
    assert diagnostics.proposed_max_step == pytest.approx(2.0)
    assert diagnostics.applied_max_step == pytest.approx(0.5)
    assert diagnostics.validation is not None
    assert diagnostics.validation["valid"]
    for key in (
        "minimum_relative_cell_measure",
        "minimum_cell_normal_alignment",
    ):
        value = diagnostics.validation[key]
        assert isinstance(value, torch.Tensor)
        assert value.device.type == device
        assert not value.requires_grad
    torch.testing.assert_close(
        updated.points[1], torch.tensor([0.0, 0.5], device=device)
    )


def test_failed_backtracking_returns_the_unchanged_mesh():
    mesh = _backtracking_curve()
    direction = torch.zeros_like(mesh.points)
    direction[1, 1] = -2.0

    updated, diagnostics = guarded_normal_update(
        mesh,
        direction,
        smoothing_iterations=0,
        backtracking_factor=0.5,
        max_backtracks=1,
    )

    assert not diagnostics.accepted
    assert diagnostics.n_backtracks == 1
    assert diagnostics.backtracking_scale == pytest.approx(0.0)
    assert diagnostics.applied_max_step == pytest.approx(0.0)
    assert diagnostics.validation is not None
    assert not diagnostics.validation["valid"]
    assert updated is mesh
    torch.testing.assert_close(updated.points, mesh.points)


def test_autograd_survives_validation_and_conditional_backtracking(device):
    mesh = _square_surface(dtype=torch.float64, device=device)
    direction = torch.zeros_like(mesh.points)
    direction[:, 2] = torch.tensor(
        [1.0, 2.0, 3.0, 4.0], dtype=torch.float64, device=device
    )
    direction.requires_grad_()
    step = torch.tensor(0.2, dtype=torch.float64, device=device, requires_grad=True)

    updated, diagnostics = guarded_normal_update(
        mesh,
        step * direction,
        smoothing_iterations=0,
        max_backtracks=0,
    )
    assert diagnostics.accepted
    updated.points[:, 2].sum().backward()

    expected_direction_grad = torch.zeros_like(direction)
    expected_direction_grad[:, 2] = 0.2
    torch.testing.assert_close(direction.grad, expected_direction_grad)
    torch.testing.assert_close(
        step.grad, torch.tensor(10.0, dtype=torch.float64, device=device)
    )

    curve = _backtracking_curve(device=device)
    backtracked_direction = torch.zeros_like(curve.points)
    backtracked_direction[1, 1] = -2.0
    backtracked_direction.requires_grad_()
    backtracked, diagnostics = guarded_normal_update(
        curve,
        backtracked_direction,
        smoothing_iterations=0,
        max_backtracks=2,
    )
    assert diagnostics.accepted
    backtracked.points[1, 1].backward()
    torch.testing.assert_close(
        backtracked_direction.grad[1, 1], torch.tensor(0.25, device=device)
    )


def test_autograd_survives_global_step_clipping():
    mesh = _square_surface(dtype=torch.float64)
    direction = torch.zeros_like(mesh.points)
    direction[:, 2] = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    direction.requires_grad_()

    updated, diagnostics = guarded_normal_update(
        mesh,
        direction,
        smoothing_iterations=0,
        max_step=0.5,
        max_backtracks=0,
    )
    assert diagnostics.accepted
    updated.points[:, 2].sum().backward()

    expected = torch.zeros_like(direction)
    expected[:, 2] = torch.tensor([0.125, 0.125, 0.125, -0.1875], dtype=torch.float64)
    torch.testing.assert_close(direction.grad, expected)


def test_mesh_method_resolves_point_data_direction_and_weights():
    mesh = _square_surface()
    mesh.point_data["shape_direction"] = torch.tensor([0.0, 0.0, 0.5]).expand_as(
        mesh.points
    )
    mesh.point_data["design_weight"] = torch.tensor([0.0, 0.5, 1.0, 1.0])
    mesh.cell_data["label"] = torch.tensor([3.0, 4.0])
    mesh.global_data["case"] = torch.tensor(7.0)

    updated, diagnostics = mesh.guarded_normal_update(
        "shape_direction",
        point_weights="design_weight",
        smoothing_iterations=0,
        max_backtracks=0,
    )

    assert diagnostics.accepted
    assert updated is not mesh
    torch.testing.assert_close(mesh.points[:, 2], torch.zeros(4))
    torch.testing.assert_close(updated.cells, mesh.cells)
    torch.testing.assert_close(
        updated.point_data["design_weight"], mesh.point_data["design_weight"]
    )
    torch.testing.assert_close(updated.cell_data["label"], mesh.cell_data["label"])
    torch.testing.assert_close(updated.global_data["case"], mesh.global_data["case"])
    torch.testing.assert_close(
        (updated.points - mesh.points)[:, 2],
        torch.tensor([0.0, 0.25, 0.5, 0.5]),
    )
    assert diagnostics.n_masked_points == 1


def test_bool_point_mask_freezes_selected_points():
    mesh = _square_surface()
    direction = torch.tensor([0.0, 0.0, 0.25]).expand_as(mesh.points)
    mask = torch.tensor([False, True, False, True])

    updated, diagnostics = guarded_normal_update(
        mesh,
        direction,
        point_weights=mask,
        smoothing_iterations=0,
        max_backtracks=0,
    )

    assert diagnostics.accepted
    assert diagnostics.n_masked_points == 2
    torch.testing.assert_close(
        (updated.points - mesh.points)[:, 2],
        torch.tensor([0.0, 0.25, 0.0, 0.25]),
    )


def test_recomputes_geometry_after_an_objective_graph_is_consumed(device):
    mesh = tetrahedron_surface.load(device=device)
    mesh.points.requires_grad_()
    axis = torch.tensor([0.3, -0.2, 0.7], device=device)
    objective = (mesh.cell_areas * (mesh.cell_normals @ axis).square()).sum()
    direction = -0.01 * torch.autograd.grad(objective, mesh.points)[0].detach()

    updated, diagnostics = guarded_normal_update(
        mesh,
        direction,
        smoothing_iterations=2,
        max_step=0.01,
    )
    assert diagnostics.accepted

    updated.points.square().sum().backward()
    assert mesh.points.grad is not None
    assert torch.isfinite(mesh.points.grad).all()


def test_rejects_orientation_manifold_and_point_normal_defects():
    points = _square_surface().points
    inconsistent = Mesh(
        points=points,
        cells=torch.tensor([[0, 1, 2], [0, 3, 2]]),
    )
    with pytest.raises(ValueError, match="consistently oriented cells"):
        guarded_normal_update(inconsistent, torch.zeros_like(points))

    inconsistent_curve = Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
        cells=torch.tensor([[0, 1], [2, 1]]),
    )
    with pytest.raises(ValueError, match="consistently oriented cells"):
        guarded_normal_update(
            inconsistent_curve, torch.zeros_like(inconsistent_curve.points)
        )

    cancelling_curve = Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        cells=torch.tensor([[0, 1], [1, 0]]),
    )
    with pytest.raises(ValueError, match="finite, nonzero point normals"):
        guarded_normal_update(
            cancelling_curve, torch.zeros_like(cancelling_curve.points)
        )

    nearly_cancelling_curve = Mesh(
        points=torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0e-13]], dtype=torch.float64
        ),
        cells=torch.tensor([[0, 1], [1, 2]]),
    )
    direction = torch.zeros_like(nearly_cancelling_curve.points)
    direction[1, 0] = 0.01
    updated, diagnostics = guarded_normal_update(
        nearly_cancelling_curve,
        direction,
        smoothing_iterations=0,
        max_backtracks=0,
    )
    assert diagnostics.accepted
    torch.testing.assert_close(
        updated.points[1] - nearly_cancelling_curve.points[1],
        torch.tensor([0.01, 0.0], dtype=torch.float64),
    )

    branched_curve = Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        cells=torch.tensor([[0, 1], [2, 0], [0, 3]]),
    )
    with pytest.raises(ValueError, match="locally manifold source topology"):
        guarded_normal_update(branched_curve, torch.zeros_like(branched_curve.points))

    inconsistent_hypersurface = Mesh(
        points=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        ),
        cells=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4]]),
    )
    with pytest.raises(ValueError, match="consistently oriented cells"):
        guarded_normal_update(
            inconsistent_hypersurface,
            torch.zeros_like(inconsistent_hypersurface.points),
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rejects_unsupported_geometry_dtype(dtype):
    mesh = _square_surface(dtype=dtype)
    with pytest.raises(TypeError, match="torch.float32 or torch.float64"):
        guarded_normal_update(mesh, torch.zeros_like(mesh.points))


def test_rejects_zero_dimensional_codimension_one_mesh():
    mesh = Mesh(
        points=torch.tensor([[0.0], [1.0]]),
        cells=torch.tensor([[0], [1]]),
    )
    with pytest.raises(ValueError, match="positive-dimensional"):
        guarded_normal_update(mesh, torch.zeros_like(mesh.points))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_microscale_surface_normals_are_renormalized(dtype):
    scale = 1.0e-6
    mesh = Mesh(
        points=scale * _square_surface(dtype=dtype).points,
        cells=torch.tensor([[0, 1, 2], [0, 2, 3]]),
    )
    direction = torch.tensor([0.0, 0.0, 1.0], dtype=dtype).expand_as(mesh.points)

    updated, diagnostics = guarded_normal_update(
        mesh,
        0.1 * scale * direction,
        smoothing_iterations=0,
        max_backtracks=0,
    )

    assert diagnostics.accepted
    torch.testing.assert_close(
        updated.points - mesh.points,
        0.1 * scale * direction,
    )


@pytest.mark.parametrize(
    ("kwargs", "error_type", "match"),
    [
        ({"smoothing_iterations": True}, TypeError, "must be an integer"),
        ({"smoothing_iterations": -1}, ValueError, "must be >= 0"),
        ({"smoothing_relaxation": float("nan")}, ValueError, "must be finite"),
        ({"max_step": 0.0}, ValueError, "must be positive and finite"),
        ({"backtracking_factor": 1.0}, ValueError, "strictly between"),
        ({"max_backtracks": True}, TypeError, "must be an integer"),
        ({"validation_tolerance": 0.0}, ValueError, "positive and finite"),
        ({"min_cell_measure_ratio": 1.1}, ValueError, r"must be in \[0, 1\]"),
    ],
)
def test_validates_scalar_parameters(kwargs, error_type, match):
    mesh = _square_surface()
    with pytest.raises(error_type, match=match):
        guarded_normal_update(mesh, torch.zeros_like(mesh.points), **kwargs)


def test_validates_direction_weights_and_point_data_resolution():
    mesh = _square_surface()

    with pytest.raises(ValueError, match="same shape"):
        guarded_normal_update(mesh, torch.zeros(mesh.n_points, 2))
    with pytest.raises(TypeError, match="same dtype"):
        guarded_normal_update(mesh, torch.zeros_like(mesh.points, dtype=torch.float64))
    with pytest.raises(KeyError, match="not found"):
        guarded_normal_update(mesh, "missing")
    with pytest.raises(ValueError, match="point_weights must have shape"):
        guarded_normal_update(
            mesh,
            torch.zeros_like(mesh.points),
            point_weights=torch.ones(mesh.n_points, 1),
        )
    with pytest.raises(TypeError, match="bool dtype or the same dtype"):
        guarded_normal_update(
            mesh,
            torch.zeros_like(mesh.points),
            point_weights=torch.ones(mesh.n_points, dtype=torch.int64),
        )
    nonfinite_weights = torch.ones(mesh.n_points)
    nonfinite_weights[0] = torch.inf
    with pytest.raises(ValueError, match="point_weights must contain only finite"):
        guarded_normal_update(
            mesh,
            torch.zeros_like(mesh.points),
            point_weights=nonfinite_weights,
        )


def test_rejects_nonfinite_direction_invalid_source_and_non_surface_mesh():
    mesh = _square_surface()
    nonfinite = torch.zeros_like(mesh.points)
    nonfinite[0, 2] = torch.nan
    with pytest.raises(ValueError, match="direction must contain only finite"):
        guarded_normal_update(mesh, nonfinite)

    degenerate = Mesh(
        points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        cells=torch.tensor([[0, 1, 2]]),
    )
    with pytest.raises(ValueError, match="requires locally valid source geometry"):
        guarded_normal_update(degenerate, torch.zeros_like(degenerate.points))

    planar = Mesh(
        points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        cells=torch.tensor([[0, 1, 2]]),
    )
    with pytest.raises(ValueError, match="codimension-one"):
        guarded_normal_update(planar, torch.zeros_like(planar.points))
