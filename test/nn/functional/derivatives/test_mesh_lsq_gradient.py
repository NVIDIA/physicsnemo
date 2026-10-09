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

from physicsnemo.nn.functional import mesh_lsq_gradient
from physicsnemo.nn.functional.derivatives import MeshLSQGradient
from test.conftest import requires_module
from test.nn.functional._parity_utils import clone_case


# Build deterministic KNN-CSR test data on random points.
def _make_case(device: str, n_entities: int, n_dims: int, k_neighbors: int):
    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(1234 + n_entities + n_dims)

    points = torch.rand((n_entities, n_dims), generator=generator, device=torch_device)
    dists = torch.cdist(points, points)
    knn = torch.topk(dists, k=k_neighbors + 1, largest=False, dim=1).indices[:, 1:]

    offsets = torch.arange(
        0,
        n_entities * k_neighbors + 1,
        k_neighbors,
        device=torch_device,
        dtype=torch.int64,
    )
    indices = knn.reshape(-1).to(torch.int64)
    return points, offsets, indices


# Validate torch LSQ reconstruction on an affine scalar field.
@pytest.mark.parametrize("n_dims", [1, 2, 3])
def test_mesh_lsq_gradient_torch(device: str, n_dims: int):
    points, offsets, indices = _make_case(
        device, n_entities=1024, n_dims=n_dims, k_neighbors=16
    )

    coeff = torch.arange(1, n_dims + 1, device=points.device, dtype=torch.float32)
    values = (points * coeff).sum(dim=-1)

    output = MeshLSQGradient.dispatch(
        points,
        values,
        offsets,
        indices,
        implementation="torch",
    )

    expected = coeff.view(1, -1).expand(points.shape[0], -1)
    torch.testing.assert_close(output, expected, atol=3e-3, rtol=3e-3)


# Validate warp backend parity against torch across benchmark representative inputs.
@requires_module("warp")
def test_mesh_lsq_gradient_backend_forward_parity(device: str):
    for _label, args, kwargs in MeshLSQGradient.make_inputs_forward(device=device):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        out_torch = MeshLSQGradient.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        out_warp = MeshLSQGradient.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        MeshLSQGradient.compare_forward(out_warp, out_torch)


# Validate warp backward parity against torch on differentiable value fields.
@requires_module("warp")
def test_mesh_lsq_gradient_backend_backward_parity(device: str):
    for _label, args, kwargs in MeshLSQGradient.make_inputs_backward(device=device):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        out_torch = MeshLSQGradient.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        out_torch.square().mean().backward()
        grad_torch = args_torch[1].grad
        assert grad_torch is not None

        out_warp = MeshLSQGradient.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        out_warp.square().mean().backward()
        grad_warp = args_warp[1].grad
        assert grad_warp is not None

        MeshLSQGradient.compare_backward(grad_warp, grad_torch)


@requires_module("warp")
def test_mesh_lsq_gradient_warp_supports_point_gradients(device: str):
    points, offsets, indices = _make_case(
        device, n_entities=768, n_dims=3, k_neighbors=12
    )
    base_values = (
        torch.sin(2.0 * torch.pi * points[:, 0])
        + 0.4 * torch.cos(2.0 * torch.pi * points[:, 1])
        + 0.2 * points[:, 2].square()
    ).to(torch.float32)

    points_warp = points.detach().clone().to(torch.float32).requires_grad_(True)
    values_warp = base_values.detach().clone().requires_grad_(True)
    out_warp = MeshLSQGradient.dispatch(
        points_warp,
        values_warp,
        offsets,
        indices,
        implementation="warp",
    )
    out_warp.square().mean().backward()
    grad_points_warp = points_warp.grad
    grad_values_warp = values_warp.grad
    assert grad_points_warp is not None
    assert grad_values_warp is not None
    assert torch.isfinite(grad_points_warp).all()
    assert torch.isfinite(grad_values_warp).all()


# Validate warp backend on 1D input parity against torch.
@requires_module("warp")
def test_mesh_lsq_gradient_warp(device: str):
    points, offsets, indices = _make_case(
        device, n_entities=512, n_dims=1, k_neighbors=16
    )
    values = torch.sin(2.0 * torch.pi * points[:, 0]).to(torch.float32)

    out_torch = MeshLSQGradient.dispatch(
        points,
        values,
        offsets,
        indices,
        implementation="torch",
    )
    out_warp = MeshLSQGradient.dispatch(
        points,
        values,
        offsets,
        indices,
        implementation="warp",
    )
    MeshLSQGradient.compare_forward(out_warp, out_torch)


# Validate benchmark input generation contract for forward inputs.
def test_mesh_lsq_gradient_make_inputs_forward(device: str):
    label, args, kwargs = next(iter(MeshLSQGradient.make_inputs_forward(device=device)))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    points, values, offsets, indices = args
    assert points.ndim == 2
    assert values.shape[0] == points.shape[0]
    assert offsets.ndim == 1
    assert indices.ndim == 1

    output = MeshLSQGradient.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    assert output.shape[0] == points.shape[0]
    assert output.shape[1] == points.shape[1]


# Validate benchmark input generation contract for backward inputs.
def test_mesh_lsq_gradient_make_inputs_backward(device: str):
    label, args, kwargs = next(
        iter(MeshLSQGradient.make_inputs_backward(device=device))
    )
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    values = args[1]
    assert values.requires_grad

    output = MeshLSQGradient.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    output.square().mean().backward()
    assert values.grad is not None


# Validate compare-forward hook contract.
def test_mesh_lsq_gradient_compare_forward_contract(device: str):
    _label, args, kwargs = next(
        iter(MeshLSQGradient.make_inputs_forward(device=device))
    )
    output = MeshLSQGradient.dispatch(*args, implementation="torch", **kwargs)
    reference = output.detach().clone()
    MeshLSQGradient.compare_forward(output, reference)


# Validate compare-backward hook contract.
def test_mesh_lsq_gradient_compare_backward_contract(device: str):
    _label, args, kwargs = next(
        iter(MeshLSQGradient.make_inputs_backward(device=device))
    )
    values = args[1]

    output = MeshLSQGradient.dispatch(*args, implementation="torch", **kwargs)
    output.square().mean().backward()

    assert values.grad is not None
    MeshLSQGradient.compare_backward(values.grad, values.grad.detach().clone())


# Validate exported API and input validation paths.
def test_mesh_lsq_gradient_error_handling(device: str):
    points, offsets, indices = _make_case(
        device, n_entities=128, n_dims=3, k_neighbors=8
    )
    values = torch.sin(points[:, 0])

    output = mesh_lsq_gradient(points, values, offsets, indices)
    assert output.shape == (points.shape[0], points.shape[1])
    assert output.dtype == torch.float32

    with pytest.raises(ValueError, match=r"must have shape \(n_entities \+ 1,\)"):
        MeshLSQGradient.dispatch(
            points,
            values,
            offsets[:-1],
            indices,
            implementation="torch",
        )

    with pytest.raises(ValueError, match=r"must equal len\(neighbor_indices\)"):
        bad_offsets = offsets.clone()
        bad_offsets[-1] = bad_offsets[-1] - 1
        MeshLSQGradient.dispatch(
            points,
            values,
            bad_offsets,
            indices,
            implementation="torch",
        )

    with pytest.raises(ValueError, match="neighbor_offsets must be non-decreasing"):
        bad_offsets = offsets.clone()
        mid = bad_offsets.shape[0] // 2
        bad_offsets[mid] = bad_offsets[mid - 1] - 1
        MeshLSQGradient.dispatch(
            points,
            values,
            bad_offsets,
            indices,
            implementation="torch",
        )

    with pytest.raises(ValueError, match="values leading dimension must match points"):
        MeshLSQGradient.dispatch(
            points,
            values[:-1],
            offsets,
            indices,
            implementation="torch",
        )

    with pytest.raises(TypeError, match="neighbor_offsets must be int32 or int64"):
        MeshLSQGradient.dispatch(
            points,
            values,
            offsets.to(torch.float32),
            indices,
            implementation="torch",
        )

    with pytest.raises(ValueError, match="must satisfy 0 <= index < n_entities"):
        bad_indices = indices.clone()
        bad_indices[0] = points.shape[0]
        MeshLSQGradient.dispatch(
            points,
            values,
            offsets,
            bad_indices,
            implementation="torch",
        )

    with pytest.raises(
        ValueError, match="safe_epsilon must be a finite positive value"
    ):
        MeshLSQGradient.dispatch(
            points,
            values,
            offsets,
            indices,
            safe_epsilon=0.0,
            implementation="torch",
        )

    if torch.cuda.is_available():
        other_device = torch.device("cuda" if points.device.type == "cpu" else "cpu")
        with pytest.raises(ValueError, match="must be on the same device"):
            MeshLSQGradient.dispatch(
                points,
                values,
                offsets.to(other_device),
                indices,
                implementation="torch",
            )


# Validate warp backend input validation paths mirror torch behavior.
@requires_module("warp")
def test_mesh_lsq_gradient_error_handling_warp(device: str):
    points, offsets, indices = _make_case(
        device, n_entities=128, n_dims=3, k_neighbors=8
    )
    values = torch.sin(points[:, 0])

    with pytest.raises(ValueError, match="neighbor_offsets must be non-decreasing"):
        bad_offsets = offsets.clone()
        mid = bad_offsets.shape[0] // 2
        bad_offsets[mid] = bad_offsets[mid - 1] - 1
        MeshLSQGradient.dispatch(
            points,
            values,
            bad_offsets,
            indices,
            implementation="warp",
        )

    with pytest.raises(
        ValueError, match="safe_epsilon must be a finite positive value"
    ):
        MeshLSQGradient.dispatch(
            points,
            values,
            offsets,
            indices,
            safe_epsilon=-1.0,
            implementation="warp",
        )

    if torch.cuda.is_available():
        other_device = torch.device("cuda" if points.device.type == "cpu" else "cpu")
        with pytest.raises(ValueError, match="must be on the same device"):
            MeshLSQGradient.dispatch(
                points,
                values,
                offsets.to(other_device),
                indices,
                implementation="warp",
            )


### small_lstsq: closed-form replacement for batched torch.linalg.lstsq


def _lstsq_reference(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Former formulation: CPU ``lstsq``, which gives minimum-norm solutions.

    The driver is ``gelsd`` rather than the default ``gelsy``: batched CPU
    ``gelsy`` intermittently returns zero for some exactly rank-deficient
    systems (e.g. ``planar_3d`` below in float32, at length scale 1e8).
    """
    solution = torch.linalg.lstsq(A.cpu(), B.cpu(), rcond=None, driver="gelsd").solution
    return solution.to(A.device)


def _make_stencils(kind: str, n_systems: int = 256) -> torch.Tensor:
    """Float64 stencil matrices (n_systems, k, d) of the given rank structure."""
    A = torch.randn(n_systems, 6, 3, dtype=torch.float64)
    t = torch.randn(n_systems, 6, 1, dtype=torch.float64)
    nonzero_column = torch.zeros(n_systems, 6, 1, dtype=torch.float64)
    tilted_normal = torch.ones(3, dtype=torch.float64) / 3**0.5
    return {
        "full_rank_3d": A,
        "full_rank_2d": A[..., :2],
        "full_rank_1d": A[..., :1],
        "anisotropic_1e-3": A * torch.tensor([1.0, 1.0, 1e-3], dtype=A.dtype),
        "planar_3d": torch.cat([A[..., :2], nonzero_column], dim=-1),
        "tilted_planar_3d": A
        - (A * tilted_normal).sum(-1, keepdim=True) * tilted_normal,
        "duplicate_column": torch.stack([A[..., 0], A[..., 1], A[..., 0]], dim=-1),
        "parallel_columns": torch.stack([A[..., 0], A[..., 1], 4 * A[..., 1]], dim=-1),
        "collinear_3d": t * torch.tensor([1.0, -2.0, 0.5], dtype=A.dtype),
        "collinear_2d": t * torch.tensor([1.0, -2.0], dtype=A.dtype),
        "underdetermined_k1_d3": A[:, :1],
        "underdetermined_k2_d3": A[:, :2],
        "underdetermined_k1_d2": A[:, :1, :2],
        "underdetermined_repeated_row": torch.cat([A[:, :1], A[:, :1]], dim=1),
        "all_zero": torch.zeros_like(A),
    }[kind]


@pytest.mark.parametrize(
    "kind",
    [
        "full_rank_3d",
        "full_rank_2d",
        "full_rank_1d",
        "anisotropic_1e-3",
        "planar_3d",
        "tilted_planar_3d",
        "duplicate_column",
        "parallel_columns",
        "collinear_3d",
        "collinear_2d",
        "underdetermined_k1_d3",
        "underdetermined_k2_d3",
        "underdetermined_k1_d2",
        "underdetermined_repeated_row",
        "all_zero",
    ],
)
@pytest.mark.parametrize("length_scale", [1e-8, 1.0, 1e8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_small_lstsq_matches_lstsq(
    device: str, kind: str, length_scale: float, dtype: torch.dtype
):
    """small_lstsq gives the minimum-norm solutions of CPU lstsq, also when rank-deficient."""
    from physicsnemo.nn.functional.derivatives.mesh_lsq_gradient._torch_impl import (
        small_lstsq,
    )

    A = (_make_stencils(kind) * length_scale).to(dtype=dtype, device=device)
    B = torch.randn(*A.shape[:-1], 2, dtype=dtype, device=device)

    solution = small_lstsq(A, B)
    reference = _lstsq_reference(A, B)

    assert solution.shape == reference.shape
    assert torch.isfinite(solution).all()
    # Error relative to each system's solution scale.
    error = (solution - reference).norm(dim=(-2, -1))
    scale = reference.norm(dim=(-2, -1)).clamp_min(torch.finfo(dtype).tiny)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    assert (error <= tolerance * scale).all(), (error / scale).max()


def test_small_lstsq_gradients(device: str):
    """small_lstsq is differentiable, with finite gradients on rank-deficient systems."""
    from physicsnemo.nn.functional.derivatives.mesh_lsq_gradient._torch_impl import (
        small_lstsq,
    )

    A = torch.randn(8, 6, 3, dtype=torch.float64, device=device, requires_grad=True)
    B = torch.randn(8, 6, 2, dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(small_lstsq, (A, B))

    for kind in ("planar_3d", "collinear_3d", "underdetermined_k2_d3", "all_zero"):
        A = _make_stencils(kind, n_systems=8).to(device).requires_grad_(True)
        B = torch.randn(*A.shape[:-1], 2, dtype=A.dtype, device=device)
        B.requires_grad_(True)
        small_lstsq(A, B).square().sum().backward()
        assert torch.isfinite(A.grad).all()
        assert torch.isfinite(B.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mesh_lsq_gradient_torch_rank_deficient_stencils(
    device: str, dtype: torch.dtype, monkeypatch
):
    """Planar and collinear 3D stencils give the minimum-norm gradients of CPU lstsq."""
    from physicsnemo.nn.functional.derivatives.mesh_lsq_gradient import _torch_impl

    points_planar, offsets, indices = _make_case(
        device, n_entities=256, n_dims=3, k_neighbors=8
    )
    points_planar = points_planar.to(dtype)
    points_planar[:, 2] = 0.0
    points_collinear = points_planar[:, :1] * torch.tensor(
        [1.0, 2.0, 0.5], dtype=dtype, device=device
    )
    # A neighborhood of one or two points is underdetermined in 3D.
    few_offsets = torch.arange(0, 2 * 256 + 1, 2, device=device).clamp_max(
        indices.shape[0] // 4
    )
    few_indices = indices[: indices.shape[0] // 4]

    for points, case_offsets, case_indices in (
        (points_planar, offsets, indices),
        (points_collinear, offsets, indices),
        (
            points_planar + 0.1 * torch.rand_like(points_planar),
            few_offsets,
            few_indices,
        ),
    ):
        values = torch.stack(
            [points[:, 0] - 3 * points[:, 1], torch.sin(4 * points).sum(-1)], dim=-1
        )
        output = _torch_impl.mesh_lsq_gradient_torch(
            points, values, case_offsets, case_indices
        )
        with monkeypatch.context() as patch:
            patch.setattr(_torch_impl, "small_lstsq", _lstsq_reference)
            reference = _torch_impl.mesh_lsq_gradient_torch(
                points, values, case_offsets, case_indices
            )

        assert torch.isfinite(output).all()
        tolerance = 1e-4 if dtype == torch.float32 else 1e-10
        torch.testing.assert_close(
            output, reference, atol=tolerance * reference.abs().max(), rtol=tolerance
        )
