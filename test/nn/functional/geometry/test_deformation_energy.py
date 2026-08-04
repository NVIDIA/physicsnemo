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

"""Tests for differentiable reference-relative deformation energies."""

import inspect
import math
from collections.abc import Callable
from contextlib import contextmanager
from typing import Literal, get_type_hints

import pytest
import torch

import physicsnemo.nn.functional as functional
from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.nn.functional import (
    closed_surface_volume_energy,
    radial_basis_function_deform_points,
    simplex_inversion_energy,
    simplex_measure_energy,
    simplex_strain_energy,
    surface_bending_energy,
    total_measure_energy,
)
from physicsnemo.nn.functional.geometry import (
    ClosedSurfaceVolumeEnergy,
    SimplexInversionEnergy,
    SimplexMeasureEnergy,
    SimplexStrainEnergy,
    SurfaceBendingEnergy,
    TotalMeasureEnergy,
)
from physicsnemo.nn.functional.geometry.deform import energy as energy_module
from test.conftest import requires_module

LOCAL_ENERGY_FUNCTIONS = (
    simplex_strain_energy,
    simplex_measure_energy,
    simplex_inversion_energy,
)

ENERGY_SPECS = (
    SimplexStrainEnergy,
    SimplexMeasureEnergy,
    TotalMeasureEnergy,
    SimplexInversionEnergy,
    SurfaceBendingEnergy,
    ClosedSurfaceVolumeEnergy,
)


def _canonical_simplex(
    manifold_dim: int,
    spatial_dim: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a unit-coordinate simplex and its one-row connectivity."""

    points = torch.zeros((manifold_dim + 1, spatial_dim), dtype=dtype, device=device)
    for axis in range(manifold_dim):
        points[axis + 1, axis] = 1.0
    cells = torch.arange(manifold_dim + 1, dtype=torch.int64, device=device).unsqueeze(
        0
    )
    return points, cells


def _two_disjoint_triangles(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    points = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )
    cells = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64, device=device)
    return points, cells


def _hinge(
    angle: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an oriented unit hinge folded through ``angle`` radians."""

    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -math.cos(angle), -math.sin(angle)],
        ],
        dtype=dtype,
        device=device,
    )
    hinges = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64, device=device)
    return points, hinges


def _tetrahedron_surface(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a unit-coordinate tetrahedron with outward-oriented faces."""

    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )
    triangles = torch.tensor(
        [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]],
        dtype=torch.int64,
        device=device,
    )
    return points, triangles


def _rotation_3d(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    axis = torch.tensor([0.3, -0.4, 0.5], dtype=dtype, device=device)
    axis = axis / torch.linalg.vector_norm(axis)
    x, y, z = axis.unbind()
    skew = torch.stack(
        (
            torch.stack((x * 0, -z, y)),
            torch.stack((z, y * 0, -x)),
            torch.stack((-y, x, z * 0)),
        )
    )
    angle = torch.tensor(0.73, dtype=dtype, device=device)
    identity = torch.eye(3, dtype=dtype, device=device)
    return identity + torch.sin(angle) * skew + (1 - torch.cos(angle)) * (skew @ skew)


def test_public_exports_signatures_and_function_spec_contract():
    functions_and_classes = (
        (simplex_strain_energy, SimplexStrainEnergy),
        (simplex_measure_energy, SimplexMeasureEnergy),
        (total_measure_energy, TotalMeasureEnergy),
        (simplex_inversion_energy, SimplexInversionEnergy),
        (surface_bending_energy, SurfaceBendingEnergy),
        (closed_surface_volume_energy, ClosedSurfaceVolumeEnergy),
    )
    expected_parameters = {
        simplex_strain_energy: [
            "points",
            "reference_points",
            "cells",
            "lame_lambda",
            "shear_modulus",
            "reduction",
            "implementation",
        ],
        simplex_measure_energy: [
            "points",
            "reference_points",
            "cells",
            "target_ratio",
            "reduction",
            "implementation",
        ],
        total_measure_energy: [
            "points",
            "reference_points",
            "cells",
            "target_ratio",
            "reduction",
            "implementation",
        ],
        simplex_inversion_energy: [
            "points",
            "reference_points",
            "cells",
            "minimum_jacobian",
            "reduction",
            "implementation",
        ],
        surface_bending_energy: [
            "points",
            "reference_points",
            "hinges",
            "reduction",
            "implementation",
        ],
        closed_surface_volume_energy: [
            "points",
            "reference_points",
            "triangles",
            "target_ratio",
            "reduction",
            "implementation",
        ],
    }

    for energy_function, energy_class in functions_and_classes:
        assert getattr(functional, energy_function.__name__) is energy_function
        assert not hasattr(functional, energy_class.__name__)
        assert issubclass(energy_class, FunctionSpec)
        assert (
            list(inspect.signature(energy_function).parameters)
            == expected_parameters[energy_function]
        )
        assert set(energy_class.implementations()) == {"torch", "warp"}
        assert get_type_hints(energy_function)["implementation"] == (
            Literal["torch", "warp"] | None
        )


def test_benchmark_generators_and_comparison_hooks(device):
    device = torch.device(device)
    for energy_class in ENERGY_SPECS:
        forward_labels = []
        for label, args, kwargs in energy_class.make_inputs_forward(device):
            forward_labels.append(label)
            output = energy_class.dispatch(*args, **kwargs)
            assert torch.isfinite(output).all()
            energy_class.compare_forward(output, output.detach().clone())
        assert forward_labels
        assert len(set(forward_labels)) == len(forward_labels)

        backward_labels = []
        for label, args, kwargs in energy_class.make_inputs_backward(device):
            backward_labels.append(label)
            output = energy_class.dispatch(*args, **kwargs)
            output.sum().backward()
            for coordinates in args[:2]:
                assert coordinates.grad is not None
                assert torch.isfinite(coordinates.grad).all()
                energy_class.compare_backward(
                    coordinates.grad, coordinates.grad.detach().clone()
                )
        assert backward_labels
        assert len(set(backward_labels)) == len(backward_labels)


def test_default_dispatch_selects_device_backend(device, monkeypatch):
    device = torch.device(device)
    points, cells = _canonical_simplex(2, 2, dtype=torch.float32, device=device)
    calls = []

    def torch_spy(normalized_points, _reference, topology, *_args):
        calls.append("torch")
        return torch.zeros(
            (normalized_points.shape[0], topology.shape[0]),
            dtype=normalized_points.dtype,
            device=normalized_points.device,
        )

    def warp_spy(normalized_points, _reference, topology, *_args):
        calls.append("warp")
        return torch.zeros(
            (normalized_points.shape[0], topology.shape[0]),
            dtype=normalized_points.dtype,
            device=normalized_points.device,
        )

    monkeypatch.setattr(energy_module, "simplex_stvk_terms_torch", torch_spy)
    monkeypatch.setattr(energy_module, "simplex_stvk_terms_warp", warp_spy)

    warp_impl = SimplexStrainEnergy._get_impls()["warp"]
    expected = "warp" if device.type == "cuda" and warp_impl.available else "torch"
    output = simplex_strain_energy(points, points, cells)
    assert calls == [expected]
    torch.testing.assert_close(output, torch.zeros_like(output))

    if device.type == "cuda":
        calls.clear()
        points_4d, cells_4d = _canonical_simplex(
            2, 4, dtype=torch.float32, device=device
        )
        output = simplex_strain_energy(points_4d, points_4d, cells_4d)
        assert calls == ["torch"]
        torch.testing.assert_close(output, torch.zeros_like(output))

        if warp_impl.available:
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
                SimplexStrainEnergy._get_impls(), "warp", unavailable_warp
            )
            FunctionSpec._fallback_warned.discard(SimplexStrainEnergy._class_key())
            with pytest.warns(RuntimeWarning, match="falling back to implementation"):
                output = simplex_strain_energy(points, points, cells)
            assert calls == ["torch"]
            torch.testing.assert_close(output, torch.zeros_like(output))


def test_rigid_motion_has_zero_reference_relative_energy():
    reference, triangles = _tetrahedron_surface()
    rotation = _rotation_3d()
    points = reference @ rotation.T + torch.tensor([1.2, -0.7, 0.4])
    cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    hinges = torch.tensor(
        [
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2],
            [1, 2, 0, 3],
            [1, 3, 2, 0],
            [2, 3, 0, 1],
        ],
        dtype=torch.int64,
    )

    outputs = (
        simplex_strain_energy(points, reference, cells, implementation="torch"),
        simplex_measure_energy(points, reference, cells, implementation="torch"),
        total_measure_energy(points, reference, cells, implementation="torch"),
        simplex_inversion_energy(points, reference, cells, implementation="torch"),
        surface_bending_energy(points, reference, hinges, implementation="torch"),
        closed_surface_volume_energy(
            points, reference, triangles, implementation="torch"
        ),
    )
    for output in outputs:
        torch.testing.assert_close(output, torch.zeros_like(output), atol=2e-28, rtol=0)


@pytest.mark.parametrize(
    ("manifold_dim", "spatial_dim"),
    [(1, 1), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 3), (3, 4)],
)
def test_uniform_scaling_has_analytic_strain_and_measure_energies(
    manifold_dim, spatial_dim
):
    reference, cells = _canonical_simplex(manifold_dim, spatial_dim)
    scale = 1.35
    points = scale * reference
    reference_measure = 1 / math.factorial(manifold_dim)
    ratio = scale**manifold_dim
    target_ratio = 0.82
    strain = 0.5 * (scale * scale - 1)
    expected_strain = reference_measure * (
        manifold_dim * strain * strain + 0.5 * (manifold_dim * strain) ** 2
    )
    expected_measure = 0.5 * reference_measure * (ratio - target_ratio) ** 2

    torch.testing.assert_close(
        simplex_strain_energy(points, reference, cells, implementation="torch"),
        torch.tensor(expected_strain, dtype=reference.dtype),
    )
    torch.testing.assert_close(
        simplex_measure_energy(
            points,
            reference,
            cells,
            target_ratio=target_ratio,
            implementation="torch",
        ),
        torch.tensor(expected_measure, dtype=reference.dtype),
    )
    torch.testing.assert_close(
        total_measure_energy(
            points,
            reference,
            cells,
            target_ratio=target_ratio,
            implementation="torch",
        ),
        torch.tensor(expected_measure, dtype=reference.dtype),
    )


def test_slender_float32_triangle_has_analytic_energies():
    height = 3.0e-4
    reference = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, height]], dtype=torch.float32
    )
    points = reference.clone()
    points[-1, 1] *= 1.1
    cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    reference_area = 0.5 * height
    normal_strain = 0.5 * (1.1**2 - 1.0)

    expected_strain = reference_area * (normal_strain**2 + 0.5 * normal_strain**2)
    expected_measure = 0.5 * reference_area * (1.1 - 1.0) ** 2
    expected_inversion = 0.5 * reference_area * (1.2 - 1.1) ** 2

    torch.testing.assert_close(
        simplex_strain_energy(points, reference, cells, implementation="torch"),
        torch.tensor(expected_strain, dtype=reference.dtype),
        atol=2.0e-11,
        rtol=2.0e-5,
    )
    torch.testing.assert_close(
        simplex_measure_energy(points, reference, cells, implementation="torch"),
        torch.tensor(expected_measure, dtype=reference.dtype),
        atol=2.0e-11,
        rtol=2.0e-5,
    )
    torch.testing.assert_close(
        simplex_inversion_energy(
            points,
            reference,
            cells,
            minimum_jacobian=1.2,
            implementation="torch",
        ),
        torch.tensor(expected_inversion, dtype=reference.dtype),
        atol=2.0e-11,
        rtol=2.0e-5,
    )


def test_embedded_triangle_transverse_deformation_has_analytic_energy():
    reference, cells = _canonical_simplex(2, 3)
    height = 0.6
    points = reference.clone()
    points[2, 2] = height
    transverse_strain = 0.5 * height**2
    expected_strain = 0.5 * (transverse_strain**2 + 0.5 * transverse_strain**2)
    area_ratio = math.sqrt(1.0 + height**2)
    expected_measure = 0.25 * (area_ratio - 1.0) ** 2

    torch.testing.assert_close(
        simplex_strain_energy(points, reference, cells, implementation="torch"),
        torch.tensor(expected_strain, dtype=reference.dtype),
    )
    torch.testing.assert_close(
        simplex_measure_energy(points, reference, cells, implementation="torch"),
        torch.tensor(expected_measure, dtype=reference.dtype),
    )
    torch.testing.assert_close(
        total_measure_energy(points, reference, cells, implementation="torch"),
        torch.tensor(expected_measure, dtype=reference.dtype),
    )


def test_reflection_is_strain_blind_but_signed_energies_detect_it():
    reference, cells = _canonical_simplex(2, 2)
    points = reference.clone()
    points[:, 0].neg_()

    strain = simplex_strain_energy(points, reference, cells, implementation="torch")
    measure = simplex_measure_energy(points, reference, cells, implementation="torch")
    total_measure = total_measure_energy(
        points, reference, cells, implementation="torch"
    )
    inversion = simplex_inversion_energy(
        points,
        reference,
        cells,
        minimum_jacobian=0.1,
        implementation="torch",
    )

    torch.testing.assert_close(strain, torch.zeros_like(strain))
    # V0=1/2 and J=-1: 0.5 * V0 * (J-1)^2 = 1.
    torch.testing.assert_close(measure, torch.tensor(1.0, dtype=reference.dtype))
    torch.testing.assert_close(total_measure, torch.tensor(1.0, dtype=reference.dtype))
    # 0.5 * V0 * max(0.1-J, 0)^2 = 0.3025.
    torch.testing.assert_close(inversion, torch.tensor(0.3025, dtype=reference.dtype))


def test_physically_stable_negative_lame_lambda_is_accepted():
    reference, cells = _canonical_simplex(2, 2)
    scale = 1.2
    strain = 0.5 * (scale * scale - 1)
    lame_lambda = -0.5
    shear_modulus = 1.0
    expected = 0.5 * (
        shear_modulus * 2 * strain * strain + 0.5 * lame_lambda * (2 * strain) ** 2
    )

    actual = simplex_strain_energy(
        scale * reference,
        reference,
        cells,
        lame_lambda=lame_lambda,
        shear_modulus=shear_modulus,
        implementation="torch",
    )
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=reference.dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stvk_rejects_unrepresentable_combined_lame_coefficient(dtype):
    reference, cells = _canonical_simplex(1, 1, dtype=dtype)
    maximum = torch.finfo(dtype).max

    with pytest.raises(ValueError, match="must be finite"):
        simplex_strain_energy(
            reference,
            reference,
            cells,
            lame_lambda=maximum,
            shear_modulus=maximum,
            implementation="torch",
        )


@pytest.mark.parametrize("volumetric_modulus", [0.0, math.ulp(2.0 / 3.0)])
def test_stvk_large_dilation_remains_nonnegative_at_stability_boundary(
    volumetric_modulus,
):
    reference, cells = _canonical_simplex(3, 3)
    scale = 1.0e4
    lame_lambda = volumetric_modulus - 2.0 / 3.0
    strain = 0.5 * (scale * scale - 1.0)
    trace = 3.0 * strain
    expected = (1.0 / 6.0) * 0.5 * volumetric_modulus * trace * trace

    actual = simplex_strain_energy(
        scale * reference,
        reference,
        cells,
        lame_lambda=lame_lambda,
        shear_modulus=1.0,
        implementation="torch",
    )

    assert actual >= 0.0
    torch.testing.assert_close(
        actual,
        torch.tensor(expected, dtype=reference.dtype),
        atol=1.0e-12,
        rtol=2.0e-15,
    )


@pytest.mark.cuda
@requires_module("warp")
@pytest.mark.parametrize("volumetric_modulus", [0.0, math.ulp(2.0 / 3.0)])
def test_warp_stvk_large_dilation_boundary_gradients_match_torch(
    volumetric_modulus,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Warp stability-boundary coverage")
    reference, cells = _canonical_simplex(3, 3, device="cuda:0")
    points = 1.0e4 * reference
    lame_lambda = volumetric_modulus - 2.0 / 3.0
    kwargs = {"lame_lambda": lame_lambda, "shear_modulus": 1.0}

    expected_output, expected_gradients = _run_with_gradients(
        simplex_strain_energy,
        points,
        reference,
        cells,
        implementation="torch",
        kwargs=kwargs,
    )
    actual_output, actual_gradients = _run_with_gradients(
        simplex_strain_energy,
        points,
        reference,
        cells,
        implementation="warp",
        kwargs=kwargs,
    )

    assert expected_output >= 0.0
    assert actual_output >= 0.0
    torch.testing.assert_close(actual_output, expected_output, atol=1.0e-12, rtol=2e-15)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=1.0e-9, rtol=2.0e-12)


def test_local_measure_penalty_distinguishes_globally_canceling_changes():
    reference, cells = _two_disjoint_triangles()
    delta = 0.36
    points = reference.clone()
    points[:3] *= math.sqrt(1 + delta)
    center = torch.tensor([2.0, 0.0], dtype=reference.dtype)
    points[3:] = center + math.sqrt(1 - delta) * (points[3:] - center)

    local = simplex_measure_energy(
        points, reference, cells, reduction="none", implementation="torch"
    )
    total = total_measure_energy(points, reference, cells, implementation="torch")

    torch.testing.assert_close(
        local,
        torch.full((2,), 0.25 * delta * delta, dtype=reference.dtype),
    )
    torch.testing.assert_close(total, torch.zeros_like(total), atol=2e-31, rtol=0)


@pytest.mark.parametrize(("manifold_dim", "spatial_dim"), [(1, 2), (2, 3)])
def test_embedded_measure_collapse_uses_finite_zero_current_subgradient(
    manifold_dim, spatial_dim
):
    reference, cells = _canonical_simplex(manifold_dim, spatial_dim)
    points = reference.clone()
    points[-1] = points[0]
    points.requires_grad_()
    reference.requires_grad_()

    energy = simplex_measure_energy(points, reference, cells, implementation="torch")
    grad_points, grad_reference = torch.autograd.grad(energy, (points, reference))

    expected = 0.5 / math.factorial(manifold_dim)
    torch.testing.assert_close(energy, torch.tensor(expected, dtype=reference.dtype))
    assert torch.isfinite(grad_points).all()
    assert torch.isfinite(grad_reference).all()
    torch.testing.assert_close(grad_points, torch.zeros_like(grad_points))


def test_unit_hinge_fold_has_analytic_bending_energy():
    reference, hinges = _hinge(0.0)
    angle = 0.47
    points, _ = _hinge(angle)

    actual = surface_bending_energy(
        points, reference, hinges, reduction="none", implementation="torch"
    )
    expected = torch.tensor([0.5 * angle * angle], dtype=reference.dtype)
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-15)


def test_hinge_bending_wraps_across_signed_angle_branch_cut():
    epsilon = 0.03
    reference, hinges = _hinge(math.pi - epsilon)
    points, _ = _hinge(-math.pi + epsilon)

    actual = surface_bending_energy(
        points, reference, hinges, reduction="none", implementation="torch"
    )
    expected = torch.tensor([0.5 * (2.0 * epsilon) ** 2], dtype=reference.dtype)
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-15)


def test_flat_reference_hinge_has_finite_zero_energy_and_gradients():
    points, hinges = _hinge(0.0)
    points = points.requires_grad_()
    reference = points.detach().clone().requires_grad_()
    energy = surface_bending_energy(points, reference, hinges, implementation="torch")
    gradients = torch.autograd.grad(energy, (points, reference))

    torch.testing.assert_close(energy, torch.zeros_like(energy))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize("scale", [1.0e-12, 1.0e10])
def test_float32_hinge_bending_is_scale_invariant(scale):
    reference, hinges = _hinge(0.17, dtype=torch.float32)
    points, _ = _hinge(0.51, dtype=torch.float32)
    baseline, baseline_gradients = _run_with_gradients(
        surface_bending_energy,
        points,
        reference,
        hinges,
        implementation="torch",
    )
    actual, actual_gradients = _run_with_gradients(
        surface_bending_energy,
        scale * points,
        scale * reference,
        hinges,
        implementation="torch",
    )

    torch.testing.assert_close(actual, baseline, atol=2.0e-7, rtol=2.0e-6)
    for actual_gradient, baseline_gradient in zip(
        actual_gradients, baseline_gradients, strict=True
    ):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(
            scale * actual_gradient,
            baseline_gradient,
            atol=3.0e-6,
            rtol=3.0e-5,
        )


def test_closed_tetrahedron_volume_has_analytic_energy():
    reference, triangles = _tetrahedron_surface()
    scale = 1.4
    target_ratio = 0.82
    points = scale * reference
    expected = 0.5 * (1 / 6) * (scale**3 - target_ratio) ** 2

    actual = closed_surface_volume_energy(
        points,
        reference,
        triangles,
        target_ratio=target_ratio,
        implementation="torch",
    )
    torch.testing.assert_close(
        actual, torch.tensor(expected, dtype=reference.dtype), atol=2e-15, rtol=2e-15
    )


def test_closed_volume_nonzero_energy_is_independent_of_global_winding():
    reference, outward = _tetrahedron_surface()
    inward = outward[:, (0, 2, 1)]
    points = reference.clone()
    points[1:] += torch.tensor(
        [[0.08, 0.02, -0.01], [-0.03, 0.07, 0.04], [0.01, -0.02, -0.06]],
        dtype=reference.dtype,
    )

    outward_points = points.clone().requires_grad_()
    outward_reference = reference.clone().requires_grad_()
    outward_energy = closed_surface_volume_energy(
        outward_points,
        outward_reference,
        outward,
        target_ratio=0.9,
        implementation="torch",
    )
    outward_gradients = torch.autograd.grad(
        outward_energy, (outward_points, outward_reference)
    )

    inward_points = points.clone().requires_grad_()
    inward_reference = reference.clone().requires_grad_()
    inward_energy = closed_surface_volume_energy(
        inward_points,
        inward_reference,
        inward,
        target_ratio=0.9,
        implementation="torch",
    )
    inward_gradients = torch.autograd.grad(
        inward_energy, (inward_points, inward_reference)
    )

    assert outward_energy > 0
    torch.testing.assert_close(inward_energy, outward_energy)
    for actual, expected in zip(inward_gradients, outward_gradients, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("dtype", "offset"),
    [(torch.float32, 1.0e3), (torch.float64, 1.0e9)],
)
def test_closed_volume_is_stable_under_large_translation_and_unused_points(
    dtype, offset
):
    reference, triangles = _tetrahedron_surface(dtype=dtype)
    points = 1.25 * reference
    expected = closed_surface_volume_energy(
        points, reference, triangles, implementation="torch"
    )

    unused = reference.new_full((1, 3), torch.nan)
    translated_reference = torch.cat((unused, reference + offset), dim=0)
    translated_points = torch.cat((unused, points + offset), dim=0)
    actual = closed_surface_volume_energy(
        translated_points,
        translated_reference,
        triangles + 1,
        implementation="torch",
    )

    atol, rtol = (2.0e-6, 2.0e-6) if dtype == torch.float32 else (2.0e-12, 2.0e-12)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_cell_order_and_orientation_preserving_permutations_do_not_change_energy():
    reference, cells = _two_disjoint_triangles()
    points = reference.clone()
    points[1] += torch.tensor([0.18, 0.11], dtype=points.dtype)
    points[5] += torch.tensor([-0.07, 0.14], dtype=points.dtype)
    permuted = cells.flip(0)[:, [1, 2, 0]]

    for energy_function in LOCAL_ENERGY_FUNCTIONS:
        expected = energy_function(
            points, reference, cells, reduction="sum", implementation="torch"
        )
        actual = energy_function(
            points, reference, permuted, reduction="sum", implementation="torch"
        )
        torch.testing.assert_close(actual, expected)


def test_batched_and_unbatched_reduction_contracts():
    reference, cells = _two_disjoint_triangles()
    first = reference.clone()
    first[1, 0] = 1.2
    second = reference.clone()
    second[4, 1] = 0.25
    points = torch.stack((first, second))
    references = reference.expand(2, -1, -1).clone()

    for energy_function in LOCAL_ENERGY_FUNCTIONS:
        terms = energy_function(
            points, references, cells, reduction="none", implementation="torch"
        )
        assert terms.shape == (2, 2)
        torch.testing.assert_close(
            energy_function(
                points, references, cells, reduction="sum", implementation="torch"
            ),
            terms.sum(),
        )
        torch.testing.assert_close(
            energy_function(
                points, references, cells, reduction="mean", implementation="torch"
            ),
            terms.mean(),
        )
        assert energy_function(
            first, reference, cells, reduction="none", implementation="torch"
        ).shape == (2,)

    totals = total_measure_energy(
        points, references, cells, reduction="none", implementation="torch"
    )
    assert totals.shape == (2,)
    torch.testing.assert_close(
        total_measure_energy(
            points, references, cells, reduction="sum", implementation="torch"
        ),
        totals.sum(),
    )
    torch.testing.assert_close(
        total_measure_energy(
            points, references, cells, reduction="mean", implementation="torch"
        ),
        totals.mean(),
    )
    assert (
        total_measure_energy(
            first, reference, cells, reduction="none", implementation="torch"
        ).shape
        == ()
    )


def test_batched_bending_and_closed_volume_reductions():
    reference_hinge, hinges = _hinge(0.23)
    folded_a, _ = _hinge(0.47)
    folded_b, _ = _hinge(-0.16)
    hinge_points = torch.stack((folded_a, folded_b))
    hinge_references = reference_hinge.expand(2, -1, -1).clone()
    bending_terms = surface_bending_energy(
        hinge_points,
        hinge_references,
        hinges,
        reduction="none",
        implementation="torch",
    )
    assert bending_terms.shape == (2, 1)
    torch.testing.assert_close(
        surface_bending_energy(
            hinge_points,
            hinge_references,
            hinges,
            reduction="mean",
            implementation="torch",
        ),
        bending_terms.mean(),
    )

    reference_tetra, triangles = _tetrahedron_surface()
    tetra_points = torch.stack((1.1 * reference_tetra, 0.8 * reference_tetra))
    tetra_references = reference_tetra.expand(2, -1, -1).clone()
    volume_terms = closed_surface_volume_energy(
        tetra_points,
        tetra_references,
        triangles,
        reduction="none",
        implementation="torch",
    )
    assert volume_terms.shape == (2,)
    torch.testing.assert_close(
        closed_surface_volume_energy(
            tetra_points,
            tetra_references,
            triangles,
            reduction="sum",
            implementation="torch",
        ),
        volume_terms.sum(),
    )
    torch.testing.assert_close(
        closed_surface_volume_energy(
            tetra_points,
            tetra_references,
            triangles,
            reduction="mean",
            implementation="torch",
        ),
        volume_terms.mean(),
    )


@pytest.mark.parametrize("batch_size", [None, 3])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_empty_local_topology_has_explicit_zero_contract(batch_size, reduction):
    reference, _ = _canonical_simplex(2, 2)
    if batch_size is not None:
        reference = reference.expand(batch_size, -1, -1).clone()
    points = reference.clone().requires_grad_()
    reference = reference.clone().requires_grad_()
    empty_cells = torch.empty((0, 3), dtype=torch.int64)

    for energy_function in LOCAL_ENERGY_FUNCTIONS:
        output = energy_function(
            points,
            reference,
            empty_cells,
            reduction=reduction,
            implementation="torch",
        )
        expected_shape = (
            ((0,) if batch_size is None else (batch_size, 0))
            if reduction == "none"
            else ()
        )
        assert output.shape == expected_shape
        assert output.numel() == 0 or torch.count_nonzero(output) == 0
        if output.numel() != 0:
            gradients = torch.autograd.grad(
                output, (points, reference), retain_graph=True
            )
            for gradient in gradients:
                torch.testing.assert_close(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize("batch_size", [None, 2])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_empty_hinges_have_explicit_zero_contract(batch_size, reduction):
    reference, _ = _hinge(0.0)
    if batch_size is not None:
        reference = reference.expand(batch_size, -1, -1).clone()
    points = reference.clone().requires_grad_()
    reference = reference.clone().requires_grad_()
    empty_hinges = torch.empty((0, 4), dtype=torch.int64)

    output = surface_bending_energy(
        points,
        reference,
        empty_hinges,
        reduction=reduction,
        implementation="torch",
    )
    expected_shape = (
        ((0,) if batch_size is None else (batch_size, 0)) if reduction == "none" else ()
    )
    assert output.shape == expected_shape
    assert output.numel() == 0 or torch.count_nonzero(output) == 0
    if output.numel() != 0:
        gradients = torch.autograd.grad(output, (points, reference))
        for gradient in gradients:
            torch.testing.assert_close(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize(
    ("energy_function", "coordinate_dimension", "vertices_per_primitive"),
    [
        (simplex_strain_energy, 2, 3),
        (simplex_measure_energy, 2, 3),
        (simplex_inversion_energy, 2, 3),
        (surface_bending_energy, 3, 4),
    ],
)
def test_zero_points_and_empty_topology_have_connected_zero_backward(
    energy_function,
    coordinate_dimension,
    vertices_per_primitive,
):
    points = torch.empty(
        (0, coordinate_dimension), dtype=torch.float64, requires_grad=True
    )
    reference = points.detach().clone().requires_grad_()
    topology = torch.empty((0, vertices_per_primitive), dtype=torch.int64)

    energy = energy_function(points, reference, topology, implementation="torch")
    gradients = torch.autograd.grad(energy, (points, reference))

    torch.testing.assert_close(energy, torch.zeros_like(energy))
    for gradient in gradients:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize(
    ("energy_function", "coordinate_dimension", "vertices_per_primitive"),
    [
        (simplex_strain_energy, 2, 3),
        (simplex_measure_energy, 2, 3),
        (total_measure_energy, 2, 3),
        (simplex_inversion_energy, 2, 3),
        (surface_bending_energy, 3, 4),
        (closed_surface_volume_energy, 3, 3),
    ],
)
def test_zero_points_and_nonempty_topology_produce_nan(
    energy_function,
    coordinate_dimension,
    vertices_per_primitive,
):
    points = torch.empty((0, coordinate_dimension), dtype=torch.float64)
    topology = torch.zeros((1, vertices_per_primitive), dtype=torch.int64)
    energy = energy_function(points, points, topology, implementation="torch")
    assert torch.isnan(energy)


def _gradcheck_cases() -> list[
    tuple[
        str,
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]
]:
    triangle, cells = _canonical_simplex(2, 2)
    deformed_triangle = triangle.clone()
    deformed_triangle[1] = torch.tensor([1.13, 0.17], dtype=triangle.dtype)
    deformed_triangle[2] = torch.tensor([-0.09, 0.79], dtype=triangle.dtype)

    reference_hinge, hinges = _hinge(0.21)
    deformed_hinge, _ = _hinge(0.53)
    deformed_hinge[2] += torch.tensor([0.08, -0.04, 0.11])

    reference_tetra, triangles = _tetrahedron_surface()
    deformed_tetra = reference_tetra.clone()
    deformed_tetra[1:] += torch.tensor(
        [[0.08, 0.03, -0.02], [-0.04, 0.07, 0.05], [0.02, -0.03, -0.09]],
        dtype=reference_tetra.dtype,
    )

    return [
        (
            "strain",
            lambda p, r: simplex_strain_energy(p, r, cells, implementation="torch"),
            deformed_triangle,
            triangle,
        ),
        (
            "local_measure",
            lambda p, r: simplex_measure_energy(
                p, r, cells, target_ratio=0.93, implementation="torch"
            ),
            deformed_triangle,
            triangle,
        ),
        (
            "total_measure",
            lambda p, r: total_measure_energy(
                p, r, cells, target_ratio=0.93, implementation="torch"
            ),
            deformed_triangle,
            triangle,
        ),
        (
            "inversion",
            lambda p, r: simplex_inversion_energy(
                p, r, cells, minimum_jacobian=1.2, implementation="torch"
            ),
            deformed_triangle,
            triangle,
        ),
        (
            "bending",
            lambda p, r: surface_bending_energy(p, r, hinges, implementation="torch"),
            deformed_hinge,
            reference_hinge,
        ),
        (
            "closed_volume",
            lambda p, r: closed_surface_volume_energy(
                p, r, triangles, target_ratio=0.9, implementation="torch"
            ),
            deformed_tetra,
            reference_tetra,
        ),
    ]


@pytest.mark.parametrize("case_index", range(6))
def test_torch_gradcheck_and_gradgradcheck_current_and_reference(case_index):
    _label, operation, points, reference = _gradcheck_cases()[case_index]
    inputs = (
        points.clone().requires_grad_(),
        reference.clone().requires_grad_(),
    )
    assert torch.autograd.gradcheck(
        operation,
        inputs,
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
        check_undefined_grad=False,
    )
    assert torch.autograd.gradgradcheck(
        operation,
        inputs,
        eps=1e-6,
        atol=4e-5,
        rtol=4e-4,
        check_undefined_grad=False,
    )


def _run_with_gradients(
    energy_function,
    points,
    reference,
    topology,
    *,
    implementation,
    kwargs=None,
):
    kwargs = {} if kwargs is None else kwargs
    points = points.detach().clone().requires_grad_()
    reference = reference.detach().clone().requires_grad_()
    output = energy_function(
        points,
        reference,
        topology,
        reduction="none",
        implementation=implementation,
        **kwargs,
    )
    cotangent = torch.linspace(
        0.7, 1.3, max(output.numel(), 1), dtype=output.dtype, device=output.device
    ).reshape(output.shape)
    objective = output if output.ndim == 0 else (output * cotangent).sum()
    gradients = torch.autograd.grad(objective, (points, reference))
    return output, gradients


@pytest.mark.cuda
@requires_module("warp")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_warp_matches_torch_forward_and_current_reference_gradients(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Warp GPU parity")
    device = torch.device("cuda:0")

    reference_triangle, cells = _two_disjoint_triangles(dtype=dtype, device=device)
    points_triangle = reference_triangle.clone()
    points_triangle[1] += torch.tensor([0.14, 0.06], dtype=dtype, device=device)
    points_triangle[5] += torch.tensor([-0.05, 0.11], dtype=dtype, device=device)
    reference_hinge, hinges = _hinge(0.17, dtype=dtype, device=device)
    points_hinge, _ = _hinge(0.51, dtype=dtype, device=device)
    reference_tetra, triangles = _tetrahedron_surface(dtype=dtype, device=device)
    points_tetra = reference_tetra.clone()
    points_tetra[1:] += torch.tensor(
        [[0.07, 0.02, -0.01], [-0.03, 0.06, 0.04], [0.01, -0.02, -0.08]],
        dtype=dtype,
        device=device,
    )

    cases = (
        (simplex_strain_energy, points_triangle, reference_triangle, cells, {}),
        (
            simplex_measure_energy,
            points_triangle,
            reference_triangle,
            cells,
            {"target_ratio": 0.94},
        ),
        (
            total_measure_energy,
            points_triangle,
            reference_triangle,
            cells,
            {"target_ratio": 0.94},
        ),
        (
            simplex_inversion_energy,
            points_triangle,
            reference_triangle,
            cells,
            {"minimum_jacobian": 1.2},
        ),
        (surface_bending_energy, points_hinge, reference_hinge, hinges, {}),
        (
            closed_surface_volume_energy,
            points_tetra,
            reference_tetra,
            triangles,
            {"target_ratio": 0.9},
        ),
    )

    atol, rtol = (2e-5, 3e-5) if dtype == torch.float32 else (2e-9, 2e-8)
    for energy_function, points, reference, topology, kwargs in cases:
        expected_output, expected_gradients = _run_with_gradients(
            energy_function,
            points,
            reference,
            topology,
            implementation="torch",
            kwargs=kwargs,
        )
        actual_output, actual_gradients = _run_with_gradients(
            energy_function,
            points,
            reference,
            topology,
            implementation="warp",
            kwargs=kwargs,
        )
        torch.testing.assert_close(actual_output, expected_output, atol=atol, rtol=rtol)
        for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.cuda
@requires_module("warp")
@pytest.mark.parametrize("scale", [1.0e-12, 1.0e10])
def test_warp_float32_hinge_scale_extremes_match_torch(scale):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Warp hinge-scale parity")
    device = torch.device("cuda:0")
    reference, hinges = _hinge(0.17, dtype=torch.float32, device=device)
    points, _ = _hinge(0.51, dtype=torch.float32, device=device)
    points = scale * points
    reference = scale * reference

    expected, expected_gradients = _run_with_gradients(
        surface_bending_energy,
        points,
        reference,
        hinges,
        implementation="torch",
    )
    actual, actual_gradients = _run_with_gradients(
        surface_bending_energy,
        points,
        reference,
        hinges,
        implementation="warp",
    )

    torch.testing.assert_close(actual, expected, atol=2.0e-7, rtol=2.0e-6)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(
            scale * actual_gradient,
            scale * expected_gradient,
            atol=3.0e-6,
            rtol=3.0e-5,
        )


@pytest.mark.cuda
@requires_module("warp")
@pytest.mark.parametrize("invalid_index", [-1, 10])
def test_out_of_bounds_topology_agrees_across_backends(invalid_index):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Warp invalid-topology parity")
    device = torch.device("cuda:0")

    reference, cells = _canonical_simplex(2, 2, device=device)
    points = reference.clone()
    points[2, 1] = 1.2
    invalid_cells = torch.cat(
        (
            cells,
            torch.tensor([[invalid_index, 1, 2]], dtype=torch.int64, device=device),
        )
    )
    reference_hinge, hinges = _hinge(0.0, device=device)
    points_hinge, _ = _hinge(0.3, device=device)
    invalid_hinges = torch.cat(
        (
            hinges,
            torch.tensor([[invalid_index, 1, 2, 0]], dtype=torch.int64, device=device),
        )
    )
    reference_surface, triangles = _tetrahedron_surface(device=device)
    points_surface = 1.1 * reference_surface
    invalid_triangles = triangles.clone()
    invalid_triangles[1, 1] = invalid_index

    cases = (
        (
            simplex_strain_energy,
            points,
            reference,
            invalid_cells,
            {"reduction": "none"},
        ),
        (
            simplex_measure_energy,
            points,
            reference,
            invalid_cells,
            {"reduction": "none"},
        ),
        (total_measure_energy, points, reference, invalid_cells, {}),
        (
            simplex_inversion_energy,
            points,
            reference,
            invalid_cells,
            {"reduction": "none"},
        ),
        (
            surface_bending_energy,
            points_hinge,
            reference_hinge,
            invalid_hinges,
            {"reduction": "none"},
        ),
        (
            closed_surface_volume_energy,
            points_surface,
            reference_surface,
            invalid_triangles,
            {},
        ),
    )
    for energy_function, current, rest, topology, kwargs in cases:
        expected = energy_function(
            current,
            rest,
            topology,
            implementation="torch",
            **kwargs,
        )
        actual = energy_function(
            current,
            rest,
            topology,
            implementation="warp",
            **kwargs,
        )
        torch.testing.assert_close(torch.isnan(actual), torch.isnan(expected))
        finite = torch.isfinite(expected)
        torch.testing.assert_close(actual[finite], expected[finite])


@pytest.mark.cuda
@requires_module("warp")
def test_warp_batched_measure_backward_and_selective_autograd_match_torch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Warp batched/selective autograd parity")
    device = torch.device("cuda:0")
    reference, cells = _two_disjoint_triangles(dtype=torch.float64, device=device)
    references = reference.unsqueeze(0).expand(2, -1, -1).clone()
    first = reference.clone()
    first[1] += first.new_tensor([0.12, 0.03])
    second = reference.clone()
    second[5] += second.new_tensor([-0.06, 0.09])
    points = torch.stack((first, second))

    expected_output, expected_gradients = _run_with_gradients(
        simplex_measure_energy,
        points,
        references,
        cells,
        implementation="torch",
        kwargs={"target_ratio": 0.93},
    )
    actual_output, actual_gradients = _run_with_gradients(
        simplex_measure_energy,
        points,
        references,
        cells,
        implementation="warp",
        kwargs={"target_ratio": 0.93},
    )
    torch.testing.assert_close(actual_output, expected_output, atol=2e-10, rtol=2e-9)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-8)

    for differentiate_current in (True, False):
        current = points.detach().clone().requires_grad_(differentiate_current)
        rest = references.detach().clone().requires_grad_(not differentiate_current)
        variable = current if differentiate_current else rest
        output = simplex_strain_energy(current, rest, cells, implementation="warp")
        (gradient,) = torch.autograd.grad(output, (variable,))
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


@pytest.mark.cuda
@requires_module("warp")
def test_warp_empty_local_topology_has_connected_zero_backward():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Warp empty-launch coverage")
    points = torch.zeros(
        (3, 2), dtype=torch.float64, device="cuda:0", requires_grad=True
    )
    reference = points.detach().clone().requires_grad_()
    cells = torch.empty((0, 3), dtype=torch.int64, device=points.device)

    energy = simplex_measure_energy(points, reference, cells, implementation="warp")
    gradients = torch.autograd.grad(energy, (points, reference))

    torch.testing.assert_close(energy, torch.zeros_like(energy))
    for gradient in gradients:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))


@pytest.mark.cuda
@requires_module("warp")
@pytest.mark.parametrize(
    ("energy_function", "coordinate_dimension", "vertices_per_primitive"),
    [
        (simplex_strain_energy, 2, 3),
        (simplex_measure_energy, 2, 3),
        (simplex_inversion_energy, 2, 3),
        (surface_bending_energy, 3, 4),
    ],
)
def test_warp_zero_points_and_empty_topology_have_connected_zero_backward(
    energy_function,
    coordinate_dimension,
    vertices_per_primitive,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Warp zero-point empty-topology coverage")
    device = torch.device("cuda:0")
    points = torch.empty(
        (0, coordinate_dimension),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    reference = points.detach().clone().requires_grad_()
    topology = torch.empty(
        (0, vertices_per_primitive), dtype=torch.int64, device=device
    )

    energy = energy_function(points, reference, topology, implementation="warp")
    gradients = torch.autograd.grad(energy, (points, reference))

    torch.testing.assert_close(energy, torch.zeros_like(energy))
    for gradient in gradients:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))


@pytest.mark.cuda
@requires_module("warp")
@pytest.mark.parametrize(
    ("manifold_dim", "spatial_dim"),
    [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
)
def test_warp_simplex_specializations_match_torch(manifold_dim, spatial_dim):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Warp specialization parity")
    device = torch.device("cuda:0")
    reference, cells = _canonical_simplex(
        manifold_dim, spatial_dim, dtype=torch.float64, device=device
    )
    points = 1.08 * reference
    points = points.clone()
    points[-1, 0] += 0.07
    if spatial_dim > manifold_dim:
        points[-1, -1] += 0.05

    energy_cases = [
        (simplex_strain_energy, {}),
        (simplex_measure_energy, {"target_ratio": 0.96}),
    ]
    if manifold_dim == spatial_dim:
        energy_cases.append((simplex_inversion_energy, {"minimum_jacobian": 1.4}))

    for energy_function, kwargs in energy_cases:
        expected_output, expected_gradients = _run_with_gradients(
            energy_function,
            points,
            reference,
            cells,
            implementation="torch",
            kwargs=kwargs,
        )
        actual_output, actual_gradients = _run_with_gradients(
            energy_function,
            points,
            reference,
            cells,
            implementation="warp",
            kwargs=kwargs,
        )
        torch.testing.assert_close(
            actual_output, expected_output, atol=2e-10, rtol=2e-9
        )
        if energy_function is simplex_inversion_energy:
            assert torch.count_nonzero(expected_output) > 0
        for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
            torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-8)
            if energy_function is simplex_inversion_energy:
                assert torch.count_nonzero(expected) > 0


@pytest.mark.cuda
@requires_module("warp")
def test_warp_slender_float32_simplex_matches_analytic_torch_path():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Warp slender-simplex parity")
    device = torch.device("cuda:0")
    height = 3.0e-4
    reference = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, height]],
        dtype=torch.float32,
        device=device,
    )
    points = reference.clone()
    points[-1, 1] *= 1.1
    cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
    cases = (
        (simplex_strain_energy, {}),
        (simplex_measure_energy, {}),
        (simplex_inversion_energy, {"minimum_jacobian": 1.2}),
    )

    for function, kwargs in cases:
        expected_output, expected_gradients = _run_with_gradients(
            function,
            points,
            reference,
            cells,
            implementation="torch",
            kwargs=kwargs,
        )
        actual_output, actual_gradients = _run_with_gradients(
            function,
            points,
            reference,
            cells,
            implementation="warp",
            kwargs=kwargs,
        )
        torch.testing.assert_close(
            actual_output, expected_output, atol=2.0e-10, rtol=3.0e-4
        )
        for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=4.0e-4)


@pytest.mark.cuda
@requires_module("warp")
def test_warp_rotated_slender_float32_simplex_matches_torch_gradients():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Warp rotated-slender parity")
    device = torch.device("cuda:0")
    reference = torch.tensor(
        [[0.0, 0.0], [1.0e4, 1.0e4], [1.0e4, 10000.1]],
        dtype=torch.float32,
        device=device,
    )
    deformation = torch.tensor(
        [[1.05, 0.02], [-0.01, 0.97]], dtype=reference.dtype, device=device
    )
    points = reference @ deformation.T
    permutations = (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    )
    canonical_cells = torch.tensor([permutations[0]], dtype=torch.int64, device=device)
    cases = (
        (simplex_strain_energy, {}),
        (simplex_measure_energy, {}),
        (simplex_inversion_energy, {"minimum_jacobian": 1.2}),
    )

    for function, kwargs in cases:
        expected_output, expected_gradients = _run_with_gradients(
            function,
            points,
            reference,
            canonical_cells,
            implementation="torch",
            kwargs=kwargs,
        )
        assert torch.count_nonzero(expected_output) > 0
        for permutation in permutations:
            cells = torch.tensor([permutation], dtype=torch.int64, device=device)
            for implementation in ("torch", "warp"):
                actual_output, actual_gradients = _run_with_gradients(
                    function,
                    points,
                    reference,
                    cells,
                    implementation=implementation,
                    kwargs=kwargs,
                )
                torch.testing.assert_close(
                    actual_output, expected_output, atol=2.0e-5, rtol=1.0e-5
                )
                for actual, expected in zip(
                    actual_gradients, expected_gradients, strict=True
                ):
                    assert torch.isfinite(actual).all()
                    torch.testing.assert_close(
                        actual, expected, atol=3.0e-3, rtol=1.0e-5
                    )


@pytest.mark.cuda
@requires_module("warp")
@pytest.mark.parametrize(("manifold_dim", "spatial_dim"), [(1, 2), (2, 3)])
def test_warp_embedded_collapse_matches_zero_subgradient_contract(
    manifold_dim, spatial_dim
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Warp collapse parity")
    reference, cells = _canonical_simplex(
        manifold_dim, spatial_dim, dtype=torch.float64, device="cuda:0"
    )
    points = reference.clone()
    points[-1] = points[0]

    expected_output, expected_gradients = _run_with_gradients(
        simplex_measure_energy,
        points,
        reference,
        cells,
        implementation="torch",
    )
    actual_output, actual_gradients = _run_with_gradients(
        simplex_measure_energy,
        points,
        reference,
        cells,
        implementation="warp",
    )

    torch.testing.assert_close(actual_output, expected_output)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.cuda
@requires_module("warp")
def test_warp_nonfinite_current_geometry_matches_torch_nan_contract():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Warp nonfinite parity")
    device = torch.device("cuda:0")

    embedded_reference, embedded_cells = _canonical_simplex(
        2, 3, dtype=torch.float64, device=device
    )
    embedded_points = embedded_reference.clone()
    embedded_points[1, 1] = torch.inf
    planar_reference, planar_cells = _canonical_simplex(
        2, 2, dtype=torch.float64, device=device
    )
    planar_points = planar_reference.clone()
    planar_points[1, 1] = torch.inf

    cases = (
        (
            simplex_measure_energy,
            embedded_points,
            embedded_reference,
            embedded_cells,
            {},
        ),
        (
            simplex_inversion_energy,
            planar_points,
            planar_reference,
            planar_cells,
            {},
        ),
    )
    for function, points, reference, cells, kwargs in cases:
        expected = function(points, reference, cells, implementation="torch", **kwargs)
        actual = function(points, reference, cells, implementation="warp", **kwargs)
        assert torch.isnan(expected)
        assert torch.isnan(actual)


@pytest.mark.cuda
@requires_module("warp")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_warp_closed_volume_is_stable_under_large_translation(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Warp volume stability")
    device = torch.device("cuda:0")
    reference, triangles = _tetrahedron_surface(dtype=dtype, device=device)
    points = 1.25 * reference
    offset = 1.0e3 if dtype == torch.float32 else 1.0e9
    unused = reference.new_full((1, 3), torch.nan)
    translated_reference = torch.cat((unused, reference + offset), dim=0)
    translated_points = torch.cat((unused, points + offset), dim=0)

    baseline = closed_surface_volume_energy(
        points, reference, triangles, implementation="torch"
    )
    expected, expected_gradients = _run_with_gradients(
        closed_surface_volume_energy,
        translated_points,
        translated_reference,
        triangles + 1,
        implementation="torch",
    )
    actual, actual_gradients = _run_with_gradients(
        closed_surface_volume_energy,
        translated_points,
        translated_reference,
        triangles + 1,
        implementation="warp",
    )

    atol, rtol = (2.0e-6, 2.0e-6) if dtype == torch.float32 else (2.0e-12, 2.0e-12)
    torch.testing.assert_close(expected, baseline, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    gradient_atol, gradient_rtol = (
        (3.0e-5, 3.0e-5) if dtype == torch.float32 else (2.0e-10, 2.0e-9)
    )
    for warp_gradient, torch_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            warp_gradient,
            torch_gradient,
            atol=gradient_atol,
            rtol=gradient_rtol,
        )


@pytest.mark.cuda
@requires_module("warp")
def test_four_dimensional_embedding_falls_back_to_torch_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for device-dispatch coverage")
    reference, cells = _canonical_simplex(2, 4, device="cuda:0")
    points = 1.1 * reference

    expected = simplex_strain_energy(points, reference, cells, implementation="torch")
    actual = simplex_strain_energy(points, reference, cells)
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="Warp|warp|dimension|D=4"):
        simplex_strain_energy(points, reference, cells, implementation="warp")


@pytest.mark.cuda
@requires_module("warp")
def test_warp_high_valence_gradient_accumulation_above_one_block():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Warp stress test")
    device = torch.device("cuda:0")
    dtype = torch.float32
    num_cells = 600
    reference = torch.zeros((1 + 2 * num_cells, 2), dtype=dtype, device=device)
    reference[1::2, 0] = 1.0
    reference[2::2, 1] = 1.0
    cells = torch.stack(
        (
            torch.zeros(num_cells, dtype=torch.int64, device=device),
            1 + 2 * torch.arange(num_cells, dtype=torch.int64, device=device),
            2 + 2 * torch.arange(num_cells, dtype=torch.int64, device=device),
        ),
        dim=-1,
    )
    points = reference.clone()
    phase = torch.linspace(0, 2 * math.pi, num_cells, device=device)
    points[1::2, 1] = 0.02 * torch.sin(phase)
    points[2::2, 0] = 0.02 * torch.cos(phase)

    expected_output, expected_gradients = _run_with_gradients(
        simplex_strain_energy,
        points,
        reference,
        cells,
        implementation="torch",
    )
    actual_output, actual_gradients = _run_with_gradients(
        simplex_strain_energy,
        points,
        reference,
        cells,
        implementation="warp",
    )
    torch.testing.assert_close(actual_output, expected_output, atol=2e-6, rtol=2e-5)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.cuda
@requires_module("warp")
def test_warp_honors_nondefault_pytorch_stream():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the stream test")
    device = torch.device("cuda:0")
    reference, cells = _two_disjoint_triangles(dtype=torch.float32, device=device)
    source = reference.clone()
    source[1] += torch.tensor([0.15, 0.08], device=device)
    torch.cuda.synchronize(device)

    def run(points, rest):
        points = points.clone().requires_grad_()
        rest = rest.clone().requires_grad_()
        output = simplex_measure_energy(points, rest, cells, implementation="warp")
        gradients = torch.autograd.grad(output, (points, rest))
        return output.detach().clone(), tuple(
            gradient.detach().clone() for gradient in gradients
        )

    expected_output, expected_gradients = run(source, reference)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(int(5e7))
        actual_output, actual_gradients = run(source, reference)

    torch.cuda.current_stream(device).wait_stream(stream)
    torch.testing.assert_close(actual_output, expected_output)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.cuda
@requires_module("warp")
def test_warp_activates_tensor_device_for_multigpu_stream_guards(monkeypatch):
    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required for the multi-GPU stream test")

    target_device = torch.device("cuda:1")
    reference, cells = _canonical_simplex(
        2,
        2,
        dtype=torch.float32,
        device=target_device,
    )
    points = (1.1 * reference).requires_grad_()
    original_launch_context = FunctionSpec.warp_launch_context
    original_scope = FunctionSpec.warp_stream_scope
    observed_context_devices = []
    observed_scope_devices = []

    def assert_tensor_device_for_launch_context(tensor):
        observed_context_devices.append(torch.cuda.current_device())
        assert torch.cuda.current_device() == target_device.index
        return original_launch_context(tensor)

    @contextmanager
    def assert_tensor_device_is_current(*args, **kwargs):
        observed_scope_devices.append(torch.cuda.current_device())
        assert torch.cuda.current_device() == target_device.index
        with original_scope(*args, **kwargs):
            yield

    monkeypatch.setattr(
        FunctionSpec,
        "warp_launch_context",
        staticmethod(assert_tensor_device_for_launch_context),
    )
    monkeypatch.setattr(
        FunctionSpec,
        "warp_stream_scope",
        staticmethod(assert_tensor_device_is_current),
    )
    with torch.cuda.device("cuda:0"):
        output = simplex_strain_energy(
            points,
            reference,
            cells,
            implementation="warp",
        )
        output.backward()
        torch.cuda.synchronize(target_device)
        assert torch.cuda.current_device() == 0

    assert observed_context_devices == [target_device.index, target_device.index]
    assert observed_scope_devices == [target_device.index, target_device.index]


@requires_module("warp")
def test_warp_custom_ops_pass_pytorch_opcheck():
    from physicsnemo.nn.functional.geometry.deform._warp_impl.energy_op import (
        closed_surface_volume_contributions_warp_impl,
        hinge_bending_terms_warp_impl,
        simplex_inversion_terms_warp_impl,
        simplex_measure_components_warp_impl,
        simplex_stvk_terms_warp_impl,
    )

    reference_triangle, cells = _canonical_simplex(2, 2)
    points_triangle = reference_triangle.clone()
    points_triangle[1] += torch.tensor([0.12, 0.05], dtype=points_triangle.dtype)
    reference_hinge, hinges = _hinge(0.17)
    points_hinge, _ = _hinge(0.43)
    reference_tetra, triangles = _tetrahedron_surface()
    points_tetra = 1.07 * reference_tetra

    cases = (
        (
            simplex_stvk_terms_warp_impl,
            (
                points_triangle.unsqueeze(0).requires_grad_(),
                reference_triangle.unsqueeze(0).requires_grad_(),
                cells,
                1.0,
                1.0,
            ),
        ),
        (
            simplex_measure_components_warp_impl,
            (
                points_triangle.unsqueeze(0).requires_grad_(),
                reference_triangle.unsqueeze(0).requires_grad_(),
                cells,
            ),
        ),
        (
            simplex_inversion_terms_warp_impl,
            (
                points_triangle.unsqueeze(0).requires_grad_(),
                reference_triangle.unsqueeze(0).requires_grad_(),
                cells,
                1.2,
            ),
        ),
        (
            hinge_bending_terms_warp_impl,
            (
                points_hinge.unsqueeze(0).requires_grad_(),
                reference_hinge.unsqueeze(0).requires_grad_(),
                hinges,
            ),
        ),
        (
            closed_surface_volume_contributions_warp_impl,
            (
                points_tetra.unsqueeze(0).requires_grad_(),
                reference_tetra.unsqueeze(0).requires_grad_(),
                triangles,
            ),
        ),
    )
    for custom_op, args in cases:
        torch.library.opcheck(custom_op, args)


def test_backward_composes_through_rbf_control_displacements():
    reference = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
        dtype=torch.float64,
    )
    cells = torch.tensor(
        [[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]], dtype=torch.int64
    )
    controls = reference[:4].clone()
    control_displacements = torch.tensor(
        [[-0.04, 0.02], [0.08, 0.09], [-0.03, -0.05], [0.06, 0.12]],
        dtype=reference.dtype,
        requires_grad=True,
    )
    points = radial_basis_function_deform_points(
        reference,
        controls,
        control_displacements,
        implementation="torch",
    )
    objective = (
        simplex_strain_energy(points, reference, cells, implementation="torch")
        + 3 * total_measure_energy(points, reference, cells, implementation="torch")
        + 2
        * simplex_inversion_energy(
            points,
            reference,
            cells,
            minimum_jacobian=0.8,
            implementation="torch",
        )
    )
    objective.backward()

    assert control_displacements.grad is not None
    assert torch.isfinite(control_displacements.grad).all()
    assert torch.count_nonzero(control_displacements.grad) > 0


def test_torch_compile_fullgraph_forward_and_backward_smoke():
    reference, triangles = _tetrahedron_surface()
    points = reference.clone()
    points[1:] += torch.tensor(
        [[0.06, 0.02, -0.01], [-0.03, 0.05, 0.04], [0.01, -0.02, -0.07]],
        dtype=reference.dtype,
    )
    cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    hinges = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)

    def operation(current, rest):
        return (
            simplex_strain_energy(current, rest, cells, implementation="torch")
            + simplex_measure_energy(current, rest, cells, implementation="torch")
            + total_measure_energy(current, rest, cells, implementation="torch")
            + simplex_inversion_energy(current, rest, cells, implementation="torch")
            + surface_bending_energy(current, rest, hinges, implementation="torch")
            + closed_surface_volume_energy(
                current, rest, triangles, implementation="torch"
            )
        )

    eager_points = points.clone().requires_grad_()
    eager_reference = reference.clone().requires_grad_()
    eager_output = operation(eager_points, eager_reference)
    eager_gradients = torch.autograd.grad(eager_output, (eager_points, eager_reference))

    compiled = torch.compile(operation, fullgraph=True, backend="eager")
    compiled_points = points.clone().requires_grad_()
    compiled_reference = reference.clone().requires_grad_()
    compiled_output = compiled(compiled_points, compiled_reference)
    compiled_gradients = torch.autograd.grad(
        compiled_output, (compiled_points, compiled_reference)
    )
    torch.testing.assert_close(compiled_output, eager_output)
    for actual, expected in zip(compiled_gradients, eager_gradients, strict=True):
        torch.testing.assert_close(actual, expected)


def test_torch_compile_fullgraph_dynamic_simplex_forward_and_backward():
    reference, cells = _canonical_simplex(2, 2, dtype=torch.float32)
    points = reference @ reference.new_tensor([[1.04, 0.03], [-0.02, 0.97]])

    def operation(current, rest, topology):
        return (
            simplex_strain_energy(current, rest, topology, implementation="torch")
            + simplex_measure_energy(current, rest, topology, implementation="torch")
            + total_measure_energy(current, rest, topology, implementation="torch")
            + simplex_inversion_energy(
                current,
                rest,
                topology,
                minimum_jacobian=1.1,
                implementation="torch",
            )
        )

    eager_points = points.clone().requires_grad_()
    eager_reference = reference.clone().requires_grad_()
    eager_output = operation(eager_points, eager_reference, cells)
    eager_gradients = torch.autograd.grad(eager_output, (eager_points, eager_reference))

    compiled = torch.compile(
        operation,
        fullgraph=True,
        dynamic=True,
        backend="eager",
    )
    compiled_points = points.clone().requires_grad_()
    compiled_reference = reference.clone().requires_grad_()
    compiled_output = compiled(compiled_points, compiled_reference, cells)
    compiled_gradients = torch.autograd.grad(
        compiled_output, (compiled_points, compiled_reference)
    )

    torch.testing.assert_close(compiled_output, eager_output)
    for actual, expected in zip(compiled_gradients, eager_gradients, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.cuda
@requires_module("warp")
def test_warp_torch_compile_fullgraph_forward_and_backward_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the compiled Warp regression")
    device = torch.device("cuda:0")
    reference, triangles = _tetrahedron_surface(device=device)
    points = reference.clone()
    points[1:] += torch.tensor(
        [[0.06, 0.02, -0.01], [-0.03, 0.05, 0.04], [0.01, -0.02, -0.07]],
        dtype=reference.dtype,
        device=device,
    )
    cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64, device=device)
    hinges = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64, device=device)

    def operation(current, rest):
        return (
            simplex_strain_energy(current, rest, cells, implementation="warp")
            + surface_bending_energy(current, rest, hinges, implementation="warp")
            + closed_surface_volume_energy(
                current,
                rest,
                triangles,
                target_ratio=0.95,
                implementation="warp",
            )
        )

    eager_points = points.clone().requires_grad_()
    eager_reference = reference.clone().requires_grad_()
    eager_output = operation(eager_points, eager_reference)
    eager_gradients = torch.autograd.grad(eager_output, (eager_points, eager_reference))

    compiled = torch.compile(operation, fullgraph=True, backend="eager")
    compiled_points = points.clone().requires_grad_()
    compiled_reference = reference.clone().requires_grad_()
    compiled_output = compiled(compiled_points, compiled_reference)
    compiled_gradients = torch.autograd.grad(
        compiled_output, (compiled_points, compiled_reference)
    )

    torch.testing.assert_close(compiled_output, eager_output, atol=2e-10, rtol=2e-9)
    for actual, expected in zip(compiled_gradients, eager_gradients, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize(
    "target_ratio",
    [0.0, -1.0, 1e-100, float("nan"), float("inf"), True],
)
def test_torch_compile_rejects_invalid_python_scalars_like_eager(target_ratio):
    reference, cells = _canonical_simplex(2, 2, dtype=torch.float32)

    def invalid_target(points, rest):
        return simplex_measure_energy(
            points,
            rest,
            cells,
            target_ratio=target_ratio,
            implementation="torch",
        )

    with pytest.raises((TypeError, ValueError)):
        invalid_target(reference, reference)
    with pytest.raises((TypeError, ValueError)):
        torch.compile(invalid_target, backend="eager")(reference, reference)


def test_torch_compile_dynamic_scalar_runtime_guard_rejects_invalid_reuse():
    reference, cells = _canonical_simplex(2, 2, dtype=torch.float32)

    def operation(points, rest, target_ratio: float):
        return simplex_measure_energy(
            points,
            rest,
            cells,
            target_ratio=target_ratio,
            implementation="torch",
        )

    compiled = torch.compile(
        operation,
        backend="eager",
        fullgraph=True,
        dynamic=True,
    )
    torch.testing.assert_close(compiled(reference, reference, 1.0), torch.tensor(0.0))
    assert torch.isfinite(compiled(reference, reference, 0.9))
    with pytest.raises(RuntimeError, match="strictly positive"):
        compiled(reference, reference, 0.0)
    with pytest.raises(RuntimeError, match="coordinate dtype"):
        compiled(reference, reference, 1e-100)


@pytest.mark.parametrize("energy_function", LOCAL_ENERGY_FUNCTIONS)
def test_degenerate_reference_simplex_produces_nan(energy_function):
    reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    output = energy_function(reference, reference, cells, implementation="torch")
    assert torch.isnan(output)


@pytest.mark.parametrize("scale", [1.0e13, 1.0e-15])
def test_stvk_nonrepresentable_reference_measure_produces_nan(scale):
    reference, cells = _canonical_simplex(3, 3, dtype=torch.float32)
    reference = scale * reference
    points = 1.1 * reference

    output = simplex_strain_energy(
        points,
        reference,
        cells,
        reduction="none",
        implementation="torch",
    )

    assert torch.isnan(output).all()


@pytest.mark.cuda
@requires_module("warp")
@pytest.mark.parametrize("scale", [1.0e13, 1.0e-15])
def test_warp_stvk_nonrepresentable_reference_measure_matches_torch_nan(scale):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Warp reference-measure coverage")
    reference, cells = _canonical_simplex(
        3,
        3,
        dtype=torch.float32,
        device="cuda:0",
    )
    reference = scale * reference
    points = 1.1 * reference

    for implementation in ("torch", "warp"):
        output = simplex_strain_energy(
            points,
            reference,
            cells,
            reduction="none",
            implementation=implementation,
        )
        assert torch.isnan(output).all()


def test_degenerate_reference_total_measure_produces_nan():
    reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    cells = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    output = total_measure_energy(reference, reference, cells, implementation="torch")
    assert torch.isnan(output)


def test_degenerate_reference_hinge_produces_nan():
    reference = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    hinges = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    output = surface_bending_energy(
        reference, reference, hinges, implementation="torch"
    )
    assert torch.isnan(output)


@pytest.mark.parametrize("invalid_index", [-1, 10])
def test_torch_out_of_bounds_topology_produces_nan_per_primitive(invalid_index):
    reference, cells = _canonical_simplex(2, 2)
    points = reference.clone()
    points[2, 1] = 1.2
    invalid_cells = torch.cat(
        (cells, torch.tensor([[invalid_index, 1, 2]], dtype=torch.int64))
    )

    for energy_function in (
        simplex_strain_energy,
        simplex_measure_energy,
        simplex_inversion_energy,
    ):
        terms = energy_function(
            points,
            reference,
            invalid_cells,
            reduction="none",
            implementation="torch",
        )
        assert torch.isfinite(terms[0])
        assert torch.isnan(terms[1])

    assert torch.isnan(
        total_measure_energy(
            points,
            reference,
            invalid_cells,
            implementation="torch",
        )
    )

    reference_hinge, hinges = _hinge(0.0)
    points_hinge, _ = _hinge(0.3)
    invalid_hinges = torch.cat(
        (
            hinges,
            torch.tensor([[invalid_index, 1, 2, 0]], dtype=torch.int64),
        )
    )
    bending_terms = surface_bending_energy(
        points_hinge,
        reference_hinge,
        invalid_hinges,
        reduction="none",
        implementation="torch",
    )
    assert torch.isfinite(bending_terms[0])
    assert torch.isnan(bending_terms[1])

    reference_surface, triangles = _tetrahedron_surface()
    points_surface = 1.1 * reference_surface
    invalid_triangles = triangles.clone()
    invalid_triangles[1, 1] = invalid_index
    assert torch.isnan(
        closed_surface_volume_energy(
            points_surface,
            reference_surface,
            invalid_triangles,
            implementation="torch",
        )
    )


def test_invalid_shapes_topology_and_scalar_options_are_rejected():
    reference, cells = _canonical_simplex(2, 2)

    with pytest.raises(ValueError):
        simplex_strain_energy(
            reference.unsqueeze(0), reference, cells, implementation="torch"
        )
    int64_output = simplex_strain_energy(
        reference, reference, cells, implementation="torch"
    )
    int32_output = simplex_strain_energy(
        reference, reference, cells.to(torch.int32), implementation="torch"
    )
    torch.testing.assert_close(int32_output, int64_output)
    with pytest.raises(ValueError, match=r"cells must have shape \(M, K\)"):
        simplex_strain_energy(
            reference, reference, cells.flatten(), implementation="torch"
        )
    with pytest.raises(ValueError, match="cells must contain"):
        simplex_strain_energy(
            reference, reference, cells[:, :1], implementation="torch"
        )
    with pytest.raises(ValueError):
        simplex_strain_energy(
            reference, reference, cells, reduction="median", implementation="torch"
        )
    with pytest.raises(ValueError):
        simplex_strain_energy(
            reference,
            reference,
            cells,
            lame_lambda=-3.0,
            shear_modulus=1.0,
            implementation="torch",
        )
    max_float64 = torch.finfo(torch.float64).max
    reference_3d, cells_3d = _canonical_simplex(3, 3)
    with pytest.raises(ValueError, match="must be nonnegative"):
        simplex_strain_energy(
            reference_3d,
            reference_3d,
            cells_3d,
            lame_lambda=-max_float64,
            shear_modulus=0.75 * max_float64,
            implementation="torch",
        )
    for target_ratio in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            simplex_measure_energy(
                reference,
                reference,
                cells,
                target_ratio=target_ratio,
                implementation="torch",
            )
    for minimum_jacobian in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            simplex_inversion_energy(
                reference,
                reference,
                cells,
                minimum_jacobian=minimum_jacobian,
                implementation="torch",
            )


def test_domain_restrictions_and_invalid_global_references_are_rejected():
    embedded_triangle, cells = _canonical_simplex(2, 3)
    with pytest.raises(ValueError, match="full.?dimensional|spatial"):
        simplex_inversion_energy(
            embedded_triangle, embedded_triangle, cells, implementation="torch"
        )

    planar_hinge = torch.zeros((4, 2), dtype=torch.float64)
    hinges = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    with pytest.raises(ValueError, match="three-dimensional|D=3|3D"):
        surface_bending_energy(
            planar_hinge, planar_hinge, hinges, implementation="torch"
        )

    reference, triangles = _tetrahedron_surface()
    planar_reference = reference.clone()
    planar_reference[:, 2] = 0
    output = closed_surface_volume_energy(
        planar_reference,
        planar_reference,
        triangles,
        implementation="torch",
    )
    assert torch.isnan(output)

    empty_cells = torch.empty((0, 3), dtype=torch.int64)
    with pytest.raises(ValueError, match="empty|cell|measure"):
        total_measure_energy(
            torch.zeros((0, 2), dtype=torch.float64),
            torch.zeros((0, 2), dtype=torch.float64),
            empty_cells,
            implementation="torch",
        )
    empty_triangles = torch.empty((0, 3), dtype=torch.int64)
    with pytest.raises(ValueError, match="empty|triangle|volume"):
        closed_surface_volume_energy(
            torch.zeros((0, 3), dtype=torch.float64),
            torch.zeros((0, 3), dtype=torch.float64),
            empty_triangles,
            implementation="torch",
        )
