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

from physicsnemo.nn.functional import mesh_lsq_hessian
from physicsnemo.nn.functional.derivatives import MeshLSQHessian
from test.conftest import requires_module
from test.nn.functional._parity_utils import clone_case

_IMPLEMENTATIONS = (
    "torch",
    pytest.param("warp", marks=requires_module("warp")),
)


def _complete_csr(
    n_entities: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a complete directed neighborhood without self edges."""
    all_indices = torch.arange(n_entities, device=device)
    neighbor_matrix = all_indices.expand(n_entities, -1)
    keep = neighbor_matrix != all_indices[:, None]
    indices = neighbor_matrix[keep].reshape(-1).to(torch.int64)
    n_neighbors = n_entities - 1
    offsets = torch.arange(
        0,
        n_entities * n_neighbors + 1,
        n_neighbors,
        device=device,
        dtype=torch.int64,
    )
    return offsets, indices


def _first_entity_csr(
    n_entities: int,
    neighbors: list[int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build CSR with one populated stencil and all other rows empty."""
    indices = torch.tensor(neighbors, device=device, dtype=torch.int64)
    offsets = torch.full(
        (n_entities + 1,),
        len(neighbors),
        device=device,
        dtype=torch.int64,
    )
    offsets[0] = 0
    return offsets, indices


def _random_points(
    n_entities: int,
    n_dims: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Build deterministic, generic point coordinates."""
    generator = torch.Generator(device=device)
    generator.manual_seed(1427 + 10 * n_entities + n_dims)
    return torch.rand(
        (n_entities, n_dims),
        generator=generator,
        device=device,
        dtype=dtype,
    )


def _expected_hessian(
    n_dims: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return a full symmetric Hessian with nonzero mixed derivatives."""
    if n_dims == 1:
        data = [[3.5]]
    elif n_dims == 2:
        data = [[2.0, 3.0], [3.0, -1.0]]
    else:
        data = [
            [2.0, 0.5, -1.25],
            [0.5, -3.0, 2.0],
            [-1.25, 2.0, 4.0],
        ]
    return torch.tensor(data, device=device, dtype=dtype)


def _quadratic_values(
    points: torch.Tensor,
    hessian: torch.Tensor,
) -> torch.Tensor:
    """Evaluate scalar or component-wise quadratic fields."""
    if hessian.ndim == 2:
        linear = torch.arange(
            1,
            points.shape[1] + 1,
            device=points.device,
            dtype=points.dtype,
        )
        return (
            0.5 * torch.einsum("ni,ij,nj->n", points, hessian, points)
            + points @ linear
            + 1.25
        )

    n_components = hessian.shape[0]
    linear = torch.arange(
        1,
        n_components * points.shape[1] + 1,
        device=points.device,
        dtype=points.dtype,
    ).reshape(n_components, points.shape[1])
    return (
        0.5 * torch.einsum("ni,cij,nj->nc", points, hessian, points)
        + torch.einsum("ni,ci->nc", points, linear)
        + 0.75
    )


@pytest.mark.parametrize("n_dims", [1, 2, 3])
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.float32, 5.0e-4), (torch.float64, 2.0e-9)],
)
def test_mesh_lsq_hessian_torch_exact_quadratic(
    device: str,
    n_dims: int,
    dtype: torch.dtype,
    tolerance: float,
):
    """Recover diagonal and mixed second derivatives of quadratic fields."""
    torch_device = torch.device(device)
    points = _random_points(20, n_dims, device=torch_device, dtype=dtype)
    offsets, indices = _complete_csr(points.shape[0], device=torch_device)
    expected = _expected_hessian(n_dims, device=torch_device, dtype=dtype)
    values = _quadratic_values(points, expected)

    output = MeshLSQHessian.dispatch(
        points,
        values,
        offsets,
        indices,
        implementation="torch",
    )

    assert output.shape == (points.shape[0], n_dims, n_dims)
    torch.testing.assert_close(
        output,
        expected.expand_as(output),
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(output, output.transpose(1, 2), atol=0.0, rtol=0.0)


@requires_module("warp")
@pytest.mark.parametrize("n_dims", [1, 2, 3])
def test_mesh_lsq_hessian_warp(device: str, n_dims: int):
    """Recover exact quadratic Hessians with the preferred Warp backend."""
    torch_device = torch.device(device)
    points = _random_points(
        20,
        n_dims,
        device=torch_device,
        dtype=torch.float32,
    )
    offsets, indices = _complete_csr(points.shape[0], device=torch_device)
    expected = _expected_hessian(
        n_dims,
        device=torch_device,
        dtype=torch.float32,
    )
    values = _quadratic_values(points, expected)

    output = MeshLSQHessian.dispatch(
        points,
        values,
        offsets,
        indices,
        implementation="warp",
    )

    assert output.shape == (points.shape[0], n_dims, n_dims)
    torch.testing.assert_close(
        output,
        expected.expand_as(output),
        atol=1.0e-3,
        rtol=1.0e-3,
    )
    torch.testing.assert_close(output, output.transpose(1, 2), atol=0.0, rtol=0.0)


@requires_module("warp")
def test_mesh_lsq_hessian_backend_forward_parity(device: str):
    """Compare Warp and Torch outputs on benchmark-representative inputs."""
    for _label, args, kwargs in MeshLSQHessian.make_inputs_forward(device=device):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        output_torch = MeshLSQHessian.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        output_warp = MeshLSQHessian.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        MeshLSQHessian.compare_forward(output_warp, output_torch)


@requires_module("warp")
def test_mesh_lsq_hessian_backend_backward_parity(device: str):
    """Compare Warp and Torch coordinate and value gradients."""
    for _label, args, kwargs in MeshLSQHessian.make_inputs_backward(device=device):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        output_torch = MeshLSQHessian.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        output_torch.square().mean().backward()
        point_grad_torch = args_torch[0].grad
        value_grad_torch = args_torch[1].grad
        assert point_grad_torch is not None
        assert value_grad_torch is not None

        output_warp = MeshLSQHessian.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        output_warp.square().mean().backward()
        point_grad_warp = args_warp[0].grad
        value_grad_warp = args_warp[1].grad
        assert point_grad_warp is not None
        assert value_grad_warp is not None

        MeshLSQHessian.compare_backward(point_grad_warp, point_grad_torch)
        MeshLSQHessian.compare_backward(value_grad_warp, value_grad_torch)


@requires_module("warp")
def test_mesh_lsq_hessian_default_dispatch_prefers_warp(device: str):
    """Select Warp by rank while retaining Torch as the sole baseline."""
    assert MeshLSQHessian.implementations() == ("warp", "torch")
    implementations = MeshLSQHessian._get_impls()
    assert implementations["warp"].rank == 0
    assert implementations["torch"].rank == 1
    assert implementations["torch"].baseline
    assert not implementations["warp"].baseline

    points = _random_points(12, 2, device=torch.device(device), dtype=torch.float32)
    offsets, indices = _complete_csr(points.shape[0], device=points.device)
    values = points[:, 0].square() + points[:, 1].square()
    output_default = mesh_lsq_hessian(points, values, offsets, indices)
    output_warp = mesh_lsq_hessian(
        points,
        values,
        offsets,
        indices,
        implementation="warp",
    )
    torch.testing.assert_close(output_default, output_warp)


@requires_module("warp")
def test_mesh_lsq_hessian_warp_fake_tensor():
    """Propagate custom-op shape and dtype without launching Warp."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    from physicsnemo.nn.functional.derivatives.mesh_lsq_hessian._warp_impl.op import (
        mesh_lsq_hessian_impl,
    )

    points = torch.empty((8, 2), dtype=torch.float32)
    values = torch.empty((8, 3), dtype=torch.float64)
    offsets = torch.arange(0, 41, 5, dtype=torch.int64)
    indices = torch.zeros(40, dtype=torch.int64)
    mode = FakeTensorMode()
    with mode:
        output = mesh_lsq_hessian_impl(
            mode.from_tensor(points),
            mode.from_tensor(values),
            mode.from_tensor(offsets),
            mode.from_tensor(indices),
            2.0,
            5,
            1.0e-10,
            -1.0,
        )

    assert output.shape == (8, 2, 2, 3)
    assert output.dtype == torch.float64


@requires_module("warp")
def test_mesh_lsq_hessian_warp_custom_op_contract():
    """Support schema checks and AOT-compiled point/value adjoints."""
    from physicsnemo.nn.functional.derivatives.mesh_lsq_hessian._warp_impl.op import (
        mesh_lsq_hessian_impl,
    )

    device = torch.device("cpu")
    points = _random_points(8, 2, device=device, dtype=torch.float32).requires_grad_()
    values = torch.rand(
        (8, 2),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    offsets, indices = _complete_csr(points.shape[0], device=device)
    op_args = (points, values, offsets, indices, 2.0, 5, 1.0e-10, -1.0)
    torch.library.opcheck(mesh_lsq_hessian_impl, args=op_args)

    def apply_hessian(
        input_points: torch.Tensor,
        input_values: torch.Tensor,
    ) -> torch.Tensor:
        return mesh_lsq_hessian_impl(
            input_points,
            input_values,
            offsets,
            indices,
            2.0,
            5,
            1.0e-10,
            -1.0,
        )

    compiled = torch.compile(apply_hessian, backend="aot_eager", fullgraph=True)
    output = compiled(points, values)
    grad_points, grad_values = torch.autograd.grad(
        output.square().sum(),
        (points, values),
    )
    assert torch.isfinite(grad_points).all()
    assert torch.isfinite(grad_values).all()


def test_mesh_lsq_hessian_tensor_values(device: str):
    """Preserve arbitrary value-component axes after the Hessian axes."""
    torch_device = torch.device(device)
    points = _random_points(24, 3, device=torch_device)
    offsets, indices = _complete_csr(points.shape[0], device=torch_device)
    base = _expected_hessian(3, device=torch_device)
    expected_components = torch.stack(
        (
            base,
            -0.5 * base,
            2.0 * base,
            torch.eye(3, device=torch_device, dtype=torch.float64),
        )
    )
    values = _quadratic_values(points, expected_components).reshape(24, 2, 2)

    output = mesh_lsq_hessian(points, values, offsets, indices, implementation="torch")

    assert output.shape == (24, 3, 3, 2, 2)
    expected = expected_components.permute(1, 2, 0).reshape(3, 3, 2, 2)
    torch.testing.assert_close(
        output,
        expected.expand_as(output),
        atol=3.0e-9,
        rtol=3.0e-9,
    )


@requires_module("warp")
def test_mesh_lsq_hessian_warp_tensor_values(device: str):
    """Preserve arbitrary value axes through the Warp component layout."""
    torch_device = torch.device(device)
    points = _random_points(24, 3, device=torch_device, dtype=torch.float32)
    offsets, indices = _complete_csr(points.shape[0], device=torch_device)
    base = _expected_hessian(3, device=torch_device, dtype=torch.float32)
    expected_components = torch.stack(
        (
            base,
            -0.5 * base,
            2.0 * base,
            torch.eye(3, device=torch_device, dtype=torch.float32),
        )
    )
    values = _quadratic_values(points, expected_components).reshape(24, 2, 2)

    output = mesh_lsq_hessian(
        points,
        values,
        offsets,
        indices,
        implementation="warp",
    )

    assert output.shape == (24, 3, 3, 2, 2)
    expected = expected_components.permute(1, 2, 0).reshape(3, 3, 2, 2)
    torch.testing.assert_close(
        output,
        expected.expand_as(output),
        atol=2.0e-3,
        rtol=2.0e-3,
    )


@pytest.mark.parametrize("scale", [1.0e-4, 1.0, 1.0e4])
def test_mesh_lsq_hessian_coordinate_scale_and_translation(device: str, scale: float):
    """Transform Hessians covariantly under coordinate scaling and translation."""
    torch_device = torch.device(device)
    base_points = _random_points(18, 2, device=torch_device) - 0.5
    translation = torch.tensor([13.0, -7.0], device=torch_device, dtype=torch.float64)
    points = scale * base_points + translation
    offsets, indices = _complete_csr(points.shape[0], device=torch_device)
    base_hessian = _expected_hessian(2, device=torch_device)
    values = _quadratic_values(base_points, base_hessian)

    output = mesh_lsq_hessian(points, values, offsets, indices, implementation="torch")
    expected = base_hessian / scale**2

    torch.testing.assert_close(
        output,
        expected.expand_as(output),
        atol=2.0e-7 * max(1.0, scale**-2),
        rtol=2.0e-8,
    )


@pytest.mark.parametrize("scale", [1.0e-6, 1.0e-5])
@pytest.mark.parametrize("implementation", _IMPLEMENTATIONS)
def test_mesh_lsq_hessian_float32_small_coordinate_scale(
    device: str,
    scale: float,
    implementation: str,
):
    """Keep valid float32 stencils well-scaled below the distance floor."""
    torch_device = torch.device(device)
    base_points = _random_points(20, 2, device=torch_device, dtype=torch.float32) - 0.5
    points = scale * base_points
    offsets, indices = _complete_csr(points.shape[0], device=torch_device)
    base_hessian = _expected_hessian(
        2,
        device=torch_device,
        dtype=torch.float32,
    )
    values = _quadratic_values(base_points, base_hessian)

    output = mesh_lsq_hessian(
        points,
        values,
        offsets,
        indices,
        implementation=implementation,
    )
    expected = base_hessian / scale**2

    torch.testing.assert_close(
        output,
        expected.expand_as(output),
        atol=5.0e-3 * scale**-2,
        rtol=5.0e-3,
    )


@pytest.mark.parametrize("implementation", _IMPLEMENTATIONS)
def test_mesh_lsq_hessian_rank_and_neighbor_guards(
    device: str,
    implementation: str,
):
    """Return finite zero curvature for deficient or explicitly skipped fits."""
    torch_device = torch.device(device)
    x = torch.linspace(-1.0, 1.0, 12, device=torch_device, dtype=torch.float64)
    collinear_points = torch.stack((x, torch.zeros_like(x)), dim=-1).requires_grad_(
        True
    )
    offsets, indices = _complete_csr(x.numel(), device=torch_device)
    values = x.square().detach().requires_grad_(True)

    rank_deficient = mesh_lsq_hessian(
        collinear_points,
        values,
        offsets,
        indices,
        min_neighbors=0,
        implementation=implementation,
    )
    skipped = mesh_lsq_hessian(
        collinear_points,
        values,
        offsets,
        indices,
        min_neighbors=x.numel(),
        implementation=implementation,
    )

    assert torch.isfinite(rank_deficient).all()
    torch.testing.assert_close(rank_deficient, torch.zeros_like(rank_deficient))
    torch.testing.assert_close(skipped, torch.zeros_like(skipped))
    rank_deficient.square().sum().backward()
    assert collinear_points.grad is not None
    assert values.grad is not None
    assert torch.isfinite(collinear_points.grad).all()
    assert torch.isfinite(values.grad).all()
    torch.testing.assert_close(
        collinear_points.grad,
        torch.zeros_like(collinear_points.grad),
    )
    torch.testing.assert_close(values.grad, torch.zeros_like(values.grad))


@pytest.mark.parametrize("implementation", _IMPLEMENTATIONS)
def test_mesh_lsq_hessian_ragged_csr(device: str, implementation: str):
    """Process valid ragged stencils while zeroing underdetermined entities."""
    torch_device = torch.device(device)
    points = _random_points(10, 2, device=torch_device)
    expected = _expected_hessian(2, device=torch_device)
    values = _quadratic_values(points, expected)

    offsets_list = [0]
    indices_list: list[int] = []
    for entity in range(points.shape[0]):
        candidates = [index for index in range(points.shape[0]) if index != entity]
        selected = candidates if entity < 5 else candidates[:4]
        indices_list.extend(selected)
        offsets_list.append(len(indices_list))
    offsets = torch.tensor(offsets_list, device=torch_device, dtype=torch.int64)
    indices = torch.tensor(indices_list, device=torch_device, dtype=torch.int64)

    output = mesh_lsq_hessian(
        points,
        values,
        offsets,
        indices,
        implementation=implementation,
    )
    tolerance = 2.0e-3 if implementation == "warp" else 2.0e-9

    torch.testing.assert_close(
        output[:5],
        expected.expand_as(output[:5]),
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(output[5:], torch.zeros_like(output[5:]))


def test_mesh_lsq_hessian_constant_and_affine_fields(device: str):
    """Return zero Hessians for fields with no quadratic component."""
    torch_device = torch.device(device)
    points = _random_points(20, 3, device=torch_device)
    offsets, indices = _complete_csr(points.shape[0], device=torch_device)
    affine = 2.0 + points @ torch.tensor(
        [1.5, -2.0, 0.25],
        device=torch_device,
        dtype=points.dtype,
    )
    values = torch.stack((torch.ones_like(affine), affine), dim=-1)

    output = mesh_lsq_hessian(points, values, offsets, indices, implementation="torch")
    torch.testing.assert_close(output, torch.zeros_like(output), atol=2.0e-9, rtol=0.0)


def test_mesh_lsq_hessian_backward(device: str):
    """Propagate finite gradients to both coordinates and field values."""
    torch_device = torch.device(device)
    points = _random_points(
        12,
        2,
        device=torch_device,
        dtype=torch.float32,
    ).requires_grad_(True)
    offsets, indices = _complete_csr(points.shape[0], device=torch_device)
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(9182)
    values = torch.randn(
        points.shape[0],
        device=torch_device,
        dtype=torch.float32,
        generator=generator,
        requires_grad=True,
    )

    output = mesh_lsq_hessian(points, values, offsets, indices, implementation="torch")
    output.square().mean().backward()

    assert points.grad is not None
    assert values.grad is not None
    assert torch.isfinite(points.grad).all()
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
@pytest.mark.parametrize("implementation", _IMPLEMENTATIONS)
def test_mesh_lsq_hessian_dtype_and_promoted_backward(
    device: str,
    dtype: torch.dtype,
    implementation: str,
):
    """Preserve the values dtype while promoting low-precision solves."""
    torch_device = torch.device(device)
    points = (
        _random_points(12, 2, device=torch_device, dtype=torch.float32)
        .to(dtype)
        .detach()
        .requires_grad_(True)
    )
    offsets, indices = _complete_csr(points.shape[0], device=torch_device)
    values = (
        torch.sin(points.detach().to(torch.float32)[:, 0])
        .to(dtype)
        .requires_grad_(True)
    )

    output = mesh_lsq_hessian(
        points,
        values,
        offsets,
        indices,
        implementation=implementation,
    )
    assert output.dtype == dtype
    assert torch.isfinite(output).all()

    output.to(torch.float32).square().mean().backward()
    assert points.grad is not None
    assert values.grad is not None
    assert torch.isfinite(points.grad).all()
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize("implementation", _IMPLEMENTATIONS)
def test_mesh_lsq_hessian_coincident_neighbors(
    device: str,
    implementation: str,
):
    """Ignore zero-distance samples, even when their values conflict."""
    torch_device = torch.device(device)
    points = _random_points(12, 2, device=torch_device)
    points[1] = points[0]
    points.requires_grad_(True)
    offsets, indices = _complete_csr(points.shape[0], device=torch_device)
    values = torch.linspace(
        -1.0,
        1.0,
        points.shape[0],
        device=torch_device,
        dtype=points.dtype,
    )
    values[0] = 3.0
    values[1] = -2.0
    values.requires_grad_(True)

    output = mesh_lsq_hessian(
        points,
        values,
        offsets,
        indices,
        implementation=implementation,
    )
    assert torch.isfinite(output).all()
    output.square().mean().backward()

    assert points.grad is not None
    assert values.grad is not None
    assert torch.isfinite(points.grad).all()
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize("implementation", _IMPLEMENTATIONS)
def test_mesh_lsq_hessian_duplicate_rows_are_invariant(
    device: str,
    implementation: str,
):
    """Keep scaling and rank policy unchanged when duplicate rows are appended."""
    torch_device = torch.device(device)
    angles = torch.arange(8, device=torch_device, dtype=torch.float64) * (
        torch.pi / 4.0
    )
    ring = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    points = torch.cat(
        (
            torch.zeros((1, 2), device=torch_device, dtype=torch.float64),
            ring,
            torch.zeros((1, 2), device=torch_device, dtype=torch.float64),
        ),
        dim=0,
    )
    expected = _expected_hessian(2, device=torch_device)
    values = _quadratic_values(points, expected)
    values[-1] = 17.0

    baseline_offsets, baseline_indices = _first_entity_csr(
        points.shape[0],
        list(range(1, 9)),
        device=torch_device,
    )
    augmented_offsets, augmented_indices = _first_entity_csr(
        points.shape[0],
        [*range(1, 9), *([9] * 24)],
        device=torch_device,
    )

    baseline = mesh_lsq_hessian(
        points,
        values,
        baseline_offsets,
        baseline_indices,
        implementation=implementation,
    )
    augmented = mesh_lsq_hessian(
        points,
        values,
        augmented_offsets,
        augmented_indices,
        implementation=implementation,
    )

    tolerance = 1.0e-3 if implementation == "warp" else 1.0e-10
    torch.testing.assert_close(
        augmented[0],
        baseline[0],
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        baseline[0],
        expected,
        atol=tolerance,
        rtol=tolerance,
    )


def test_mesh_lsq_hessian_symmetric_stencil_point_backward(device: str):
    """Keep point gradients finite when a symmetric stencil repeats spectra."""
    torch_device = torch.device(device)
    angles = torch.arange(8, device=torch_device, dtype=torch.float64) * (
        torch.pi / 4.0
    )
    ring = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    points = torch.cat(
        (
            torch.zeros((1, 2), device=torch_device, dtype=torch.float64),
            ring,
        ),
        dim=0,
    ).requires_grad_(True)
    values = torch.linspace(
        -1.0,
        2.0,
        points.shape[0],
        device=torch_device,
        dtype=torch.float64,
        requires_grad=True,
    )
    offsets, indices = _first_entity_csr(
        points.shape[0],
        list(range(1, 9)),
        device=torch_device,
    )

    output = mesh_lsq_hessian(points, values, offsets, indices, implementation="torch")
    output[0].square().sum().backward()

    assert points.grad is not None
    assert values.grad is not None
    assert torch.isfinite(points.grad).all()
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize("implementation", _IMPLEMENTATIONS)
def test_mesh_lsq_hessian_rcond_controls_rank(
    device: str,
    implementation: str,
):
    """Apply the user cutoff consistently to rank detection and solving."""
    torch_device = torch.device(device)
    points = _random_points(12, 2, device=torch_device)
    offsets, indices = _complete_csr(points.shape[0], device=torch_device)
    expected = _expected_hessian(2, device=torch_device)
    values = _quadratic_values(points, expected)

    output = mesh_lsq_hessian(
        points,
        values,
        offsets,
        indices,
        rcond=1.0,
        implementation=implementation,
    )
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_mesh_lsq_hessian_gradcheck(device: str):
    """Match numerical derivatives for a small full-rank stencil."""
    torch_device = torch.device(device)
    points = _random_points(8, 2, device=torch_device).requires_grad_(True)
    offsets, indices = _complete_csr(points.shape[0], device=torch_device)
    values = torch.linspace(
        -0.5,
        0.75,
        points.shape[0],
        device=torch_device,
        dtype=torch.float64,
        requires_grad=True,
    )

    def functional(points_input: torch.Tensor, values_input: torch.Tensor):
        """Evaluate the functional with fixed CSR topology for gradcheck."""
        return mesh_lsq_hessian(
            points_input,
            values_input,
            offsets,
            indices,
            implementation="torch",
        )

    assert torch.autograd.gradcheck(
        functional,
        (points, values),
        eps=1.0e-6,
        atol=2.0e-4,
        rtol=2.0e-3,
    )


def test_mesh_lsq_hessian_make_inputs_forward(device: str):
    """Validate the labeled forward benchmark-input contract."""
    forward_cases = list(MeshLSQHessian.make_inputs_forward(device=device))
    label, args, kwargs = forward_cases[0]
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    output = MeshLSQHessian.dispatch(*args, implementation="torch", **kwargs)
    assert output.shape[:3] == (
        args[0].shape[0],
        args[0].shape[1],
        args[0].shape[1],
    )


def test_mesh_lsq_hessian_make_inputs_backward(device: str):
    """Validate coordinate and value backward benchmark inputs."""
    backward_cases = list(MeshLSQHessian.make_inputs_backward(device=device))
    _label, args, kwargs = backward_cases[0]
    assert args[0].requires_grad
    assert args[1].requires_grad
    backward_output = MeshLSQHessian.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    backward_output.square().mean().backward()
    assert args[0].grad is not None
    assert args[1].grad is not None


def test_mesh_lsq_hessian_compare_forward_contract(device: str):
    """Validate the FunctionSpec forward comparison hook."""
    _label, args, kwargs = next(iter(MeshLSQHessian.make_inputs_forward(device=device)))
    output = MeshLSQHessian.dispatch(*args, implementation="torch", **kwargs)
    MeshLSQHessian.compare_forward(output, output.detach().clone())


def test_mesh_lsq_hessian_compare_backward_contract(device: str):
    """Validate the FunctionSpec backward comparison hook for both inputs."""
    _label, args, kwargs = next(
        iter(MeshLSQHessian.make_inputs_backward(device=device))
    )
    output = MeshLSQHessian.dispatch(*args, implementation="torch", **kwargs)
    output.square().mean().backward()

    assert args[0].grad is not None
    assert args[1].grad is not None
    MeshLSQHessian.compare_backward(args[0].grad, args[0].grad.detach().clone())
    MeshLSQHessian.compare_backward(args[1].grad, args[1].grad.detach().clone())


@pytest.mark.parametrize("implementation", _IMPLEMENTATIONS)
def test_mesh_lsq_hessian_error_handling(device: str, implementation: str):
    """Validate public dispatch and Hessian-specific error paths."""
    torch_device = torch.device(device)
    points = _random_points(12, 2, device=torch_device, dtype=torch.float32)
    offsets, indices = _complete_csr(points.shape[0], device=torch_device)
    values = points[:, 0].square()

    output = mesh_lsq_hessian(
        points,
        values,
        offsets,
        indices,
        implementation=implementation,
    )
    assert output.shape == (12, 2, 2)
    assert output.dtype == values.dtype

    with pytest.raises(ValueError, match="min_neighbors must be non-negative"):
        mesh_lsq_hessian(
            points,
            values,
            offsets,
            indices,
            min_neighbors=-1,
            implementation=implementation,
        )

    with pytest.raises(TypeError, match="must be an integer or None"):
        mesh_lsq_hessian(
            points,
            values,
            offsets,
            indices,
            min_neighbors=1.5,
            implementation=implementation,
        )

    with pytest.raises(ValueError, match="weight_power must be finite"):
        mesh_lsq_hessian(
            points,
            values,
            offsets,
            indices,
            weight_power=float("inf"),
            implementation=implementation,
        )

    for invalid_rcond in (-1.0, float("nan"), float("inf")):
        with pytest.raises(
            ValueError,
            match="rcond must be a finite non-negative value or None",
        ):
            mesh_lsq_hessian(
                points,
                values,
                offsets,
                indices,
                rcond=invalid_rcond,
                implementation=implementation,
            )

    with pytest.raises(
        ValueError,
        match="safe_epsilon must be a finite positive value",
    ):
        mesh_lsq_hessian(
            points,
            values,
            offsets,
            indices,
            safe_epsilon=0.0,
            implementation=implementation,
        )

    with pytest.raises(ValueError, match="must be non-decreasing"):
        bad_offsets = offsets.clone()
        bad_offsets[3] = bad_offsets[2] - 1
        mesh_lsq_hessian(
            points,
            values,
            bad_offsets,
            indices,
            implementation=implementation,
        )

    with pytest.raises(ValueError, match="must satisfy 0 <= index < n_entities"):
        bad_indices = indices.clone()
        bad_indices[0] = points.shape[0]
        mesh_lsq_hessian(
            points,
            values,
            offsets,
            bad_indices,
            implementation=implementation,
        )

    with pytest.raises(ValueError, match="must be 1D/2D/3D"):
        points_4d = torch.cat((points, points), dim=-1)
        mesh_lsq_hessian(
            points_4d,
            values,
            offsets,
            indices,
            implementation=implementation,
        )


@pytest.mark.parametrize("implementation", _IMPLEMENTATIONS)
def test_mesh_lsq_hessian_empty_input(device: str, implementation: str):
    """Preserve output shape for an empty but valid CSR batch."""
    torch_device = torch.device(device)
    points = torch.empty((0, 3), device=torch_device)
    values = torch.empty((0, 2), device=torch_device)
    offsets = torch.zeros(1, device=torch_device, dtype=torch.int64)
    indices = torch.empty(0, device=torch_device, dtype=torch.int64)

    output = mesh_lsq_hessian(
        points,
        values,
        offsets,
        indices,
        implementation=implementation,
    )
    assert output.shape == (0, 3, 3, 2)
