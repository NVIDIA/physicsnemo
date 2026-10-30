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

from physicsnemo.nn.functional import signed_distance_field
from physicsnemo.nn.functional.geometry import SignedDistanceField
from test.conftest import requires_module


# Build a simple tetrahedron surface mesh as four triangles.
def _tetrahedron_vertices() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


# Validate the warp-backed SDF implementation on a deterministic tetrahedron setup.
@requires_module("warp")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_signed_distance_field_warp(dtype: torch.dtype, device: str):
    device = torch.device(device)
    mesh_vertices = _tetrahedron_vertices().to(device=device, dtype=dtype)
    mesh_indices_flat = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        device=device,
        dtype=torch.int32,
    )
    query_points = torch.tensor(
        [[1.0, 1.0, 1.0], [0.05, 0.1, 0.1]],
        device=device,
        dtype=dtype,
    )

    sdf_out, hit_points, hit_faces = signed_distance_field(
        mesh_vertices=mesh_vertices,
        mesh_indices=mesh_indices_flat,
        input_points=query_points,
        use_sign_winding_number=False,
    )

    torch.testing.assert_close(
        sdf_out,
        torch.tensor([1.1547, -0.05], device=device, dtype=dtype),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        hit_points,
        torch.tensor(
            [[0.33333322, 0.33333334, 0.3333334], [0.0, 0.10, 0.10]],
            device=device,
            dtype=dtype,
        ),
        atol=1e-6,
        rtol=1e-6,
    )

    # Hit faces: int64 indices of the geometrically correct triangles. The
    # first query is nearest the far face [9, 10, 11] (face 3); the second is
    # nearest the x = 0 face [6, 7, 8] (face 2).
    assert hit_faces.dtype == torch.int64
    torch.testing.assert_close(
        hit_faces, torch.tensor([3, 2], device=device, dtype=torch.int64)
    )


# Queries beyond max_dist report the NaN/NaN/-1 miss sentinels.
@requires_module("warp")
def test_signed_distance_field_miss(device: str):
    device = torch.device(device)
    mesh_vertices = _tetrahedron_vertices().to(device=device, dtype=torch.float32)
    mesh_indices = torch.arange(12, device=device, dtype=torch.int32)
    query_points = torch.tensor(
        [[0.05, 0.1, 0.1], [100.0, 100.0, 100.0]],
        device=device,
        dtype=torch.float32,
    )

    sdf_out, hit_points, hit_faces = signed_distance_field(
        mesh_vertices, mesh_indices, query_points, max_dist=1.0
    )

    # In-band query resolves normally.
    assert torch.isfinite(sdf_out[0])
    assert torch.isfinite(hit_points[0]).all()
    assert hit_faces[0] >= 0

    # Out-of-band query: NaN distance/hit point, -1 face.
    assert torch.isnan(sdf_out[1])
    assert torch.isnan(hit_points[1]).all()
    assert hit_faces[1] == -1


# Validate SDF index-shape compatibility paths.
@requires_module("warp")
def test_signed_distance_field_index_layout_compatibility(device: str):
    device = torch.device(device)
    mesh_vertices = _tetrahedron_vertices().to(device=device, dtype=torch.float32)
    mesh_indices_flat = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        device=device,
        dtype=torch.int32,
    )
    mesh_indices_faces = mesh_indices_flat.reshape(-1, 3)
    query_points = torch.tensor([[0.1, 0.2, 0.3]], device=device, dtype=torch.float32)

    # Accept both flattened and (n_faces, 3) connectivity layouts.
    sdf_flat, hit_flat, faces_flat = signed_distance_field(
        mesh_vertices, mesh_indices_flat, query_points
    )
    sdf_faces, hit_faces, faces_faces = signed_distance_field(
        mesh_vertices, mesh_indices_faces, query_points
    )
    torch.testing.assert_close(sdf_flat, sdf_faces)
    torch.testing.assert_close(hit_flat, hit_faces)
    torch.testing.assert_close(faces_flat, faces_faces)


# Strided (transposed) vertices and queries give the same values and gradients
# as their contiguous copies.
@requires_module("warp")
def test_signed_distance_field_strided_inputs(device: str):
    device = torch.device(device)
    mesh_vertices = _tetrahedron_vertices().to(device=device, dtype=torch.float32)
    mesh_indices = torch.arange(12, device=device, dtype=torch.int32)
    query_points = torch.tensor(
        [[1.0, 1.0, 1.0], [0.05, 0.1, 0.1], [0.1, 0.2, -0.5]],
        device=device,
        dtype=torch.float32,
    )

    leaves = []
    outputs = []
    for strided in (False, True):
        # A leaf stored transposed, so .T is a view with column stride != 1.
        v = mesh_vertices.T.contiguous().requires_grad_()
        q = query_points.T.contiguous().requires_grad_()
        v_in, q_in = (v.T, q.T) if strided else (v.T.contiguous(), q.T.contiguous())
        assert v_in.is_contiguous() != strided and q_in.is_contiguous() != strided
        sdf_out, hit_points, hit_faces = signed_distance_field(v_in, mesh_indices, q_in)
        sdf_out.sum().backward()
        leaves.append((v.grad, q.grad))
        outputs.append((sdf_out.detach(), hit_points, hit_faces))

    for expected, actual in zip(outputs[0], outputs[1]):
        torch.testing.assert_close(actual, expected)
    for expected, actual in zip(leaves[0], leaves[1]):
        torch.testing.assert_close(actual, expected)


# Validate benchmark input generation contract for SDF.
@requires_module("warp")
def test_signed_distance_field_make_inputs_forward(device: str):
    label, args, kwargs = next(
        iter(SignedDistanceField.make_inputs_forward(device=device))
    )
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    sdf_out, hit_points, hit_faces = SignedDistanceField.dispatch(
        *args,
        implementation="warp",
        **kwargs,
    )
    assert sdf_out.ndim == 1
    assert hit_points.ndim == 2
    assert hit_points.shape[1] == 3
    assert hit_faces.ndim == 1
    assert hit_faces.dtype == torch.int64


# Validate backward benchmark input generation contract for SDF.
@requires_module("warp")
def test_signed_distance_field_make_inputs_backward(device: str):
    label, args, kwargs = next(
        iter(SignedDistanceField.make_inputs_backward(device=device))
    )
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    mesh_vertices, _, input_points = args
    assert mesh_vertices.requires_grad
    assert input_points.requires_grad

    sdf_out, _, _ = SignedDistanceField.dispatch(
        *args,
        implementation="warp",
        **kwargs,
    )
    sdf_out.square().mean().backward()
    assert torch.isfinite(mesh_vertices.grad).all()
    assert torch.isfinite(input_points.grad).all()


# On one triangle, the gradient has a closed form: a query gets the unit
# vector from its closest point, times the sign, and each vertex gets minus
# its barycentric weight in the closest point times that vector.
@requires_module("warp")
def test_signed_distance_field_backward_closed_form(device: str):
    device = torch.device(device)
    mesh_vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    mesh_indices = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)
    # Above the face, below it, beside an edge, beyond a vertex, and on the
    # face itself, where the face normal is the gradient.
    query_points = torch.tensor(
        [
            [0.2, 0.3, 0.5],
            [0.2, 0.3, -0.5],
            [0.5, -0.5, 0.5],
            [-1.0, -1.0, 1.0],
            [0.2, 0.3, 0.0],
        ],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )

    sdf_out, _, _ = signed_distance_field(mesh_vertices, mesh_indices, query_points)
    grad_vertices, grad_points = torch.autograd.grad(
        sdf_out.sum(), (mesh_vertices, query_points)
    )

    r2, r3 = 2.0**-0.5, 3.0**-0.5
    expected_points = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, -r2, r2],
            [-r3, -r3, r3],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float64,
    )
    # Barycentric weights of each query's closest point.
    weights = torch.tensor(
        [
            [0.5, 0.2, 0.3],
            [0.5, 0.2, 0.3],
            [0.5, 0.5, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.2, 0.3],
        ],
        device=device,
        dtype=torch.float64,
    )
    expected_vertices = -weights.T @ expected_points
    torch.testing.assert_close(grad_points, expected_points, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(grad_vertices, expected_vertices, atol=1e-6, rtol=0.0)


def _central_difference(fn, x: torch.Tensor, step: float) -> torch.Tensor:
    """Central finite-difference gradient of a scalar function of ``x``."""
    grad = torch.zeros_like(x)
    for i in range(x.numel()):
        offset = torch.zeros_like(x)
        offset.view(-1)[i] = step
        grad.view(-1)[i] = (fn(x + offset) - fn(x - offset)) / (2.0 * step)
    return grad


# The gradient matches finite differences on a closed surface, with either
# sign method.
@requires_module("warp")
@pytest.mark.parametrize("use_winding", [False, True])
def test_signed_distance_field_backward_finite_differences(
    use_winding: bool, device: str
):
    device = torch.device(device)
    mesh_vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float64,
    )
    mesh_indices = torch.tensor(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], device=device, dtype=torch.int32
    )
    # Inside and outside, nearest to faces, edges, and vertices, and away from
    # the medial axis, where the gradient jumps.
    query_points = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [0.1, 0.15, 0.2],
            [0.3, 0.3, -0.4],
            [-0.3, 0.2, 0.4],
            [1.2, -0.3, 0.1],
            [0.05, 0.6, 0.1],
        ],
        device=device,
        dtype=torch.float64,
    )
    weights = torch.linspace(0.5, 1.5, len(query_points), device=device)

    def weighted_sdf(vertices: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        sdf_out, _, _ = signed_distance_field(
            vertices, mesh_indices, points, use_sign_winding_number=use_winding
        )
        return (sdf_out * weights).sum()

    vertices = mesh_vertices.clone().requires_grad_(True)
    points = query_points.clone().requires_grad_(True)
    grad_vertices, grad_points = torch.autograd.grad(
        weighted_sdf(vertices, points), (vertices, points)
    )

    # The op computes in float32, so the step is large against its rounding.
    step = 1e-3
    fd_vertices = _central_difference(
        lambda v: weighted_sdf(v, query_points), mesh_vertices, step
    )
    fd_points = _central_difference(
        lambda q: weighted_sdf(mesh_vertices, q), query_points, step
    )
    torch.testing.assert_close(grad_vertices, fd_vertices, atol=1e-3, rtol=0.0)
    torch.testing.assert_close(grad_points, fd_points, atol=1e-3, rtol=0.0)


# Far from the origin, a query a few thousandths from a corner or an edge
# still gets the direction to its closest point. The coordinates and offsets
# are exact in float32.
@requires_module("warp")
@pytest.mark.parametrize(
    "query, direction, weights",
    [
        # Nearest the corner at (1000, 0, 0).
        (
            (999.9990234375, -0.0009765625, -0.0009765625),
            (-1.0, -1.0, -1.0),
            (1.0, 0.0),
        ),
        # Nearest the midpoint of the edge from (1000, 0, 0) to (1001, 0, 0).
        ((1000.5, -0.0009765625, -0.0009765625), (0.0, -1.0, -1.0), (0.5, 0.5)),
    ],
)
def test_signed_distance_field_backward_far_from_origin(
    query: tuple, direction: tuple, weights: tuple, device: str
):
    device = torch.device(device)
    mesh_vertices = torch.tensor(
        [
            [1000.0, 0.0, 0.0],
            [1001.0, 0.0, 0.0],
            [1000.0, 1.0, 0.0],
            [1000.0, 0.0, 1.0],
        ],
        device=device,
        requires_grad=True,
    )
    mesh_indices = torch.tensor(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], device=device, dtype=torch.int32
    )
    query_points = torch.tensor([query], device=device, requires_grad=True)

    sdf_out, _, _ = signed_distance_field(mesh_vertices, mesh_indices, query_points)
    grad_vertices, grad_points = torch.autograd.grad(
        sdf_out.sum(), (mesh_vertices, query_points)
    )

    expected = torch.tensor(direction, device=device)
    expected = expected / expected.norm()
    torch.testing.assert_close(grad_points[0], expected, atol=1e-5, rtol=0.0)
    # The closest point is a combination of vertices 0 and 1 only.
    expected_vertices = torch.zeros(4, 3, device=device)
    expected_vertices[0] = -weights[0] * expected
    expected_vertices[1] = -weights[1] * expected
    torch.testing.assert_close(grad_vertices, expected_vertices, atol=1e-5, rtol=0.0)


# Rounding the hit point to a float16 query dtype would turn the gradient,
# so the backward pass recomputes the closest point at full precision.
@requires_module("warp")
def test_signed_distance_field_backward_half_precision_queries(device: str):
    device = torch.device(device)
    mesh_vertices = torch.tensor(
        [[0.0, -1.0, 0.0], [2.0, -1.0, 2.0], [0.0, 3.0, 0.0]],
        device=device,
        requires_grad=True,
    )
    mesh_indices = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)
    query_points = torch.tensor(
        [[1.0, 0.0, 1.0009765625]],
        device=device,
        dtype=torch.float16,
        requires_grad=True,
    )

    sdf_out, _, _ = signed_distance_field(mesh_vertices, mesh_indices, query_points)
    grad_vertices, grad_points = torch.autograd.grad(
        sdf_out.sum(), (mesh_vertices, query_points)
    )

    # The closest point (1.00048828125, 0, 1.00048828125) lies between two
    # float16 values; the query sits on the side its face normal points to.
    direction = torch.tensor([-1.0, 0.0, 1.0], device=device) / 2.0**0.5
    assert grad_points.dtype == torch.float16
    torch.testing.assert_close(grad_points[0].float(), direction, atol=1e-3, rtol=0.0)
    weights = torch.tensor([0.249755859375, 0.500244140625, 0.25], device=device)
    torch.testing.assert_close(
        grad_vertices, -weights[:, None] * direction, atol=1e-6, rtol=0.0
    )


# Only the distance carries gradients; queries beyond max_dist contribute
# none, and their NaN distance does not leak into the vertex gradient.
@requires_module("warp")
def test_signed_distance_field_backward_outputs_and_misses(device: str):
    device = torch.device(device)
    mesh_vertices = _tetrahedron_vertices().to(device=device, dtype=torch.float32)
    mesh_vertices.requires_grad_(True)
    mesh_indices = torch.arange(12, device=device, dtype=torch.int32)
    query_points = torch.tensor(
        [[0.05, 0.1, 0.1], [100.0, 100.0, 100.0]],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    sdf_out, hit_points, hit_faces = signed_distance_field(
        mesh_vertices, mesh_indices, query_points, max_dist=1.0
    )
    assert sdf_out.requires_grad
    assert not hit_points.requires_grad
    assert not hit_faces.requires_grad

    grad_vertices, grad_points = torch.autograd.grad(
        sdf_out.sum(), (mesh_vertices, query_points)
    )
    # The in-band query sits inside, nearest the x = 0 face.
    torch.testing.assert_close(
        grad_points,
        torch.tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device),
        atol=1e-5,
        rtol=0.0,
    )
    assert torch.isfinite(grad_vertices).all()


# Validate SDF input and shape error handling paths.
@requires_module("warp")
def test_signed_distance_field_error_handling(device: str):
    device = torch.device(device)
    mesh_vertices = _tetrahedron_vertices().to(device=device, dtype=torch.float32)
    mesh_indices_flat = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        device=device,
        dtype=torch.int32,
    )
    query_points = torch.tensor([[0.1, 0.2, 0.3]], device=device, dtype=torch.float32)

    # Query points must have xyz in the last dimension.
    bad_queries = torch.randn(4, 2, device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match="last dimension of size 3"):
        signed_distance_field(mesh_vertices, mesh_indices_flat, bad_queries)

    # 2D mesh indices must be shaped as (n_faces, 3).
    bad_connectivity_shape = torch.zeros(4, 4, device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="shape \\(n_faces, 3\\)"):
        signed_distance_field(mesh_vertices, bad_connectivity_shape, query_points)

    # Connectivity may be 1D flattened or 2D triangular faces only.
    bad_connectivity_rank = torch.zeros(1, 2, 3, device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="1D flattened indices or 2D"):
        signed_distance_field(mesh_vertices, bad_connectivity_rank, query_points)


# Out-of-range index validation raises eagerly on CPU only: on CUDA it is a
# device-side assert (sync-free by design), which would poison the CUDA
# context if triggered in-process, so it is not exercised here.
@requires_module("warp")
def test_signed_distance_field_index_range_validation_cpu():
    mesh_vertices = _tetrahedron_vertices().to(dtype=torch.float32)
    query_points = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)

    with pytest.raises(ValueError, match="index existing vertices"):
        signed_distance_field(
            mesh_vertices,
            torch.tensor([0, 1, 12], dtype=torch.int32),
            query_points,
        )
    with pytest.raises(ValueError, match="index existing vertices"):
        signed_distance_field(
            mesh_vertices,
            torch.tensor([-1, 1, 2], dtype=torch.int32),
            query_points,
        )
