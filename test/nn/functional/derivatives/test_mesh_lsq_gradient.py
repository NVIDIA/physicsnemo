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


# Validate the CSR checks that share one device-to-host transfer.
@pytest.mark.parametrize(
    "corrupt, message",
    [
        (lambda o, i: (o + 1, i), "neighbor_offsets must start at 0"),
        (
            lambda o, i: (o, i.index_fill(0, torch.tensor([3], device=i.device), -1)),
            "must satisfy 0 <= index < n_entities",
        ),
    ],
)
def test_mesh_lsq_gradient_csr_validation(device: str, corrupt, message):
    points, offsets, indices = _make_case(
        device, n_entities=32, n_dims=2, k_neighbors=4
    )
    bad_offsets, bad_indices = corrupt(offsets, indices)
    with pytest.raises(ValueError, match=message):
        MeshLSQGradient.dispatch(
            points,
            points[:, 0],
            bad_offsets,
            bad_indices,
            implementation="torch",
        )


# Validate grouping by neighbor count against independent per-entity solves.
def test_mesh_lsq_gradient_ragged_neighborhoods_match_per_entity_lstsq(device: str):
    generator = torch.Generator().manual_seed(5)
    n_entities = 30
    points = torch.rand(n_entities, 2, generator=generator, dtype=torch.float64)
    values = torch.sin(3 * points[:, 0]) + points[:, 1] ** 2
    # 3 to 5 distinct neighbors per entity, never the entity itself
    neighbors = []
    for i in range(n_entities):
        others = [j for j in torch.randperm(n_entities, generator=generator).tolist()]
        neighbors.append([j for j in others if j != i][: 3 + i % 3])
    offsets = torch.tensor([0] + [len(n) for n in neighbors]).cumsum(0)
    indices = torch.tensor(sum(neighbors, []))

    output = MeshLSQGradient.dispatch(
        points.to(device),
        values.to(device),
        offsets.to(device),
        indices.to(device),
        implementation="torch",
    )

    for i, n in enumerate(neighbors):
        dx = points[n] - points[i]
        sqrt_w = (1.0 / dx.norm(dim=-1))[:, None]  # inverse-distance-squared weights
        expected = torch.linalg.lstsq(
            sqrt_w * dx, sqrt_w * (values[n] - values[i])[:, None]
        ).solution[:, 0]
        torch.testing.assert_close(output[i].cpu(), expected)


# Validate that the torch backend returns empty gradients for zero entities.
@pytest.mark.parametrize("value_shape", [(), (2,), (2, 4)])
def test_mesh_lsq_gradient_torch_no_entities(device: str, value_shape):
    output = MeshLSQGradient.dispatch(
        torch.empty(0, 3, device=device),
        torch.empty(0, *value_shape, device=device),
        torch.zeros(1, dtype=torch.int64, device=device),
        torch.empty(0, dtype=torch.int64, device=device),
        implementation="torch",
    )
    assert output.shape == (0, 3, *value_shape)


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
@pytest.mark.parametrize("solver", ["small_lstsq", "batched_lstsq"])
def test_small_lstsq_matches_lstsq(
    device: str, kind: str, length_scale: float, dtype: torch.dtype, solver: str
):
    """small_lstsq and batched_lstsq (float64 normal equations for float32 inputs)
    give the minimum-norm solutions of CPU lstsq, also when rank-deficient."""
    from physicsnemo.utils import _small_linalg

    A = (_make_stencils(kind) * length_scale).to(dtype=dtype, device=device)
    B = torch.randn(*A.shape[:-1], 2, dtype=dtype, device=device)

    solution = getattr(_small_linalg, solver)(A, B)
    reference = _lstsq_reference(A, B)

    assert solution.shape == reference.shape
    assert torch.isfinite(solution).all()
    # Error relative to each system's solution scale.
    error = (solution - reference).norm(dim=(-2, -1))
    scale = reference.norm(dim=(-2, -1)).clamp_min(torch.finfo(dtype).tiny)
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    assert (error <= tolerance * scale).all(), (error / scale).max()


@pytest.mark.parametrize("solver", ["small_lstsq", "_normal_equations_lstsq"])
def test_small_lstsq_gradients(device: str, solver: str):
    """The solvers are differentiable, with finite gradients on rank-deficient systems."""
    from physicsnemo.utils import _small_linalg

    lstsq = getattr(_small_linalg, solver)
    A = torch.randn(8, 6, 3, dtype=torch.float64, device=device, requires_grad=True)
    B = torch.randn(8, 6, 2, dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(lstsq, (A, B))

    # The normal equations are for float32 inputs
    dtype = torch.float64 if solver == "small_lstsq" else torch.float32
    for kind in ("planar_3d", "collinear_3d", "underdetermined_k2_d3", "all_zero"):
        A = _make_stencils(kind, n_systems=8).to(device, dtype).requires_grad_(True)
        B = torch.randn(*A.shape[:-1], 2, dtype=A.dtype, device=device)
        B.requires_grad_(True)
        lstsq(A, B).square().sum().backward()
        assert torch.isfinite(A.grad).all()
        assert torch.isfinite(B.grad).all()


def _rank_deficient_case(case: str, device: str, dtype: torch.dtype):
    """Points, offsets and indices of 3D LSQ stencils that are rank-deficient."""
    if case == "single_planar":  # one point whose four neighbors lie in z = 0
        points = torch.tensor(
            [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]],
            dtype=dtype,
            device=device,
        )
        offsets = torch.tensor([0, 4, 4, 4, 4, 4], device=device)
        return points, offsets, torch.tensor([1, 2, 3, 4], device=device)
    points, offsets, indices = _make_case(
        device, n_entities=256, n_dims=3, k_neighbors=8
    )
    points = points.to(dtype)
    if case == "planar":  # every stencil in the plane z = 0
        points[:, 2] = 0.0
    elif case == "tilted":  # every stencil in the plane z = x
        points[:, 2] = points[:, 0]
    elif case == "collinear":
        points = points[:, :1] * torch.tensor(
            [1.0, 2.0, 0.5], dtype=dtype, device=device
        )
    elif case == "underdetermined":  # neighborhoods of two points
        offsets = torch.arange(0, 2 * 256 + 1, 2, device=device).clamp_max(
            indices.shape[0] // 4
        )
        indices = indices[: indices.shape[0] // 4]
    return points, offsets, indices


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "case", ["planar", "single_planar", "tilted", "collinear", "underdetermined"]
)
def test_mesh_lsq_gradient_torch_rank_deficient_stencils(
    device: str, dtype: torch.dtype, case: str, monkeypatch
):
    """Rank-deficient 3D stencils give the minimum-norm gradients of CPU lstsq,
    on every device. (CUDA lstsq, whose only driver gels assumes full rank, gave
    NaN for a single flat stencil and a large wrong answer for a tilted plane.)"""
    from physicsnemo.nn.functional.derivatives.mesh_lsq_gradient import _torch_impl

    points, offsets, indices = _rank_deficient_case(case, device, dtype)
    values = torch.stack(
        [points[:, 0] - 3 * points[:, 1], torch.sin(4 * points).sum(-1)], dim=-1
    )
    output = _torch_impl.mesh_lsq_gradient_torch(points, values, offsets, indices)
    with monkeypatch.context() as patch:
        patch.setattr(_torch_impl, "_uses_normal_equations", lambda *_: False)
        patch.setattr(_torch_impl, "batched_lstsq", _lstsq_reference)
        reference = _torch_impl.mesh_lsq_gradient_torch(
            points, values, offsets, indices
        )

    assert torch.isfinite(output).all()
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(
        output, reference, atol=tolerance * reference.abs().max(), rtol=tolerance
    )


@pytest.mark.parametrize("solver", ["small_lstsq", "batched_lstsq"])
def test_small_lstsq_propagates_nan_and_inf(device: str, solver: str):
    """A NaN or inf entry gives NaN rather than a plausible zero, in its own system only."""
    from physicsnemo.utils import _small_linalg

    small_lstsq = getattr(_small_linalg, solver)

    A = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device=device)
    A = A.expand(3, 3, 2).clone()
    B = torch.tensor([[1.0], [2.0], [3.0]], device=device).expand(3, 3, 1).clone()
    A[0, 0, 0] = float("nan")
    A[1, 0, 0] = float("inf")

    solution = small_lstsq(A, B)

    assert solution[0].isnan().all()
    assert solution[1].isnan().any()
    torch.testing.assert_close(solution[2], _lstsq_reference(A[2:], B[2:])[0])


@pytest.mark.parametrize("solver", ["small_lstsq", "batched_lstsq"])
def test_small_lstsq_bfloat16(device: str, solver: str):
    """16-bit systems are solved in float32, so the rank test is not swamped by eps."""
    from physicsnemo.utils import _small_linalg

    small_lstsq = getattr(_small_linalg, solver)

    generator = torch.Generator().manual_seed(0)
    x = torch.rand(2000, 20, generator=generator) * 2 - 1
    y = (torch.rand(2000, 20, generator=generator) * 2 - 1) / 10  # stretched 10:1
    A = torch.stack([x, y], dim=-1).to(device=device, dtype=torch.bfloat16)
    B = (x + y).unsqueeze(-1).to(device=device, dtype=torch.bfloat16)

    solution = small_lstsq(A, B)  # the gradient of f = x + y is (1, 1)

    assert solution.dtype == torch.bfloat16
    assert (solution.float() > 0.5).all()


@pytest.mark.parametrize(
    "dtype, n_unknowns",
    [
        (torch.float32, 2),  # normal equations
        (torch.float32, 3),
        (torch.float64, 3),  # small_lstsq
        (torch.float32, 4),
        (torch.bfloat16, 3),  # normal equations in float32
    ],
)
def test_lstsq_solver_matches_batched_lstsq(
    device: str, dtype: torch.dtype, n_unknowns: int
):
    """One solver of ``A`` serves right-hand sides of any width, bit for bit as
    ``batched_lstsq``, also with rank-deficient systems in the batch."""
    from physicsnemo.utils._small_linalg import _lstsq_solver, batched_lstsq

    generator = torch.Generator().manual_seed(0)
    A = torch.randn(300, 8, n_unknowns, generator=generator)
    A[:100, :, -1] = A[:100, :, 0]  # rank-deficient
    A[100:110] = 0.0
    A = A.to(device=device, dtype=dtype)
    solve = _lstsq_solver(A)
    for n_rhs in (1, 3, 1, 4):
        B = torch.randn(300, 8, n_rhs, generator=generator).to(device, dtype)
        assert torch.equal(solve(B), batched_lstsq(A, B))


@pytest.mark.parametrize(
    "dtype, n_unknowns",
    [(torch.float32, 1), (torch.float32, 2), (torch.float32, 3), (torch.bfloat16, 3)],
)
def test_segment_lstsq_solver_matches_batched_lstsq(
    device: str, dtype: torch.dtype, n_unknowns: int
):
    """Systems of 0 to 9 rows, assembled from consecutive rows and solved at
    once, get the solutions of ``batched_lstsq`` of the systems of each size,
    also when rank-deficient or zero, up to the rounding of their float64
    moments, which are summed in another order."""
    from physicsnemo.utils._small_linalg import _segment_lstsq_solver, batched_lstsq

    generator = torch.Generator().manual_seed(0)
    counts = torch.arange(10).repeat_interleave(30)
    counts = counts[torch.randperm(len(counts), generator=generator)]
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
    A = torch.randn(int(offsets[-1]), n_unknowns, generator=generator)
    if n_unknowns > 1:
        A[: len(A) // 3, -1] = A[: len(A) // 3, 0]  # rank-deficient
    A[len(A) // 3 : len(A) // 2] = 0.0
    A = A.to(device=device, dtype=dtype)
    counts, offsets = counts.to(device), offsets.to(device)
    solve = _segment_lstsq_solver(A, offsets)

    for n_rhs in (1, 3):
        B = torch.randn(len(A), n_rhs, generator=generator).to(device, dtype)
        X = solve(B)
        assert X.shape == (len(counts), n_unknowns, n_rhs)
        assert (X[counts == 0] == 0).all()
        for k in range(1, 10):
            systems = (counts == k).nonzero()[:, 0]
            rows = (offsets[systems, None] + torch.arange(k, device=device)).flatten()
            expected = batched_lstsq(
                A[rows].view(-1, k, n_unknowns), B[rows].view(-1, k, n_rhs)
            )
            # Identical on the GPUs and CPUs tested, but another order of the
            # float64 sums could flip a last bit
            scale = expected.float().abs().max().item()
            atol = (1e-7 if dtype == torch.float32 else 1e-2) * scale
            torch.testing.assert_close(X[systems], expected, rtol=0, atol=atol)
