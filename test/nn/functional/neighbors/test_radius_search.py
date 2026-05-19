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

from physicsnemo.nn.functional import radius_search
from physicsnemo.nn.functional.neighbors import RadiusSearch
from physicsnemo.nn.functional.neighbors.radius_search._warp_impl import (
    radius_search_impl as radius_search_warp,
)
from test.conftest import requires_module


# Build a deterministic radius-search problem with known local neighbors.
def _build_problem(device: str):
    base = torch.linspace(0, 10, 11, device=device)
    x, y, z = torch.meshgrid(base, base, base, indexing="ij")
    queries = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)

    displacements = torch.tensor(
        [
            [-0.05, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.15, 0.0],
            [0.0, -0.2, 0.0],
            [0.0, 0.0, 0.25],
            [0.0, 0.0, -0.3],
        ],
        device=device,
    )
    points = queries[None, :, :] + displacements[:, None, :]
    points = points.reshape(-1, 3)
    return points, queries


# Validate result shapes and value bounds for radius-search outputs.
def _assert_radius_outputs(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int | None,
    return_dists: bool,
    return_points: bool,
    results,
) -> None:
    if return_points and return_dists:
        indices, selected_points, distances = results
    elif return_points:
        indices, selected_points = results
        distances = None
    elif return_dists:
        indices, distances = results
        selected_points = None
    else:
        indices = results
        selected_points = None
        distances = None

    if max_points is None:
        assert indices.shape[0] == 2
    else:
        assert indices.shape == (queries.shape[0], max_points)

    if distances is not None:
        assert (distances >= 0).all()
        assert (distances <= radius).all()

    if selected_points is not None:
        if max_points is None:
            assert selected_points.shape[0] == indices.shape[1]
            assert selected_points.shape[1] == 3
        else:
            assert selected_points.shape == (queries.shape[0], max_points, 3)

    # Valid indices are in bounds, with 0 as the sentinel for "unused".
    valid = (indices == 0) | ((indices >= 0) & (indices < points.shape[0]))
    assert valid.all()


# Validate the torch implementation across return modes.
@pytest.mark.parametrize("return_dists", [True, False])
@pytest.mark.parametrize("return_points", [True, False])
@pytest.mark.parametrize("max_points", [5, None])
def test_radius_search_torch(
    device: str,
    return_dists: bool,
    return_points: bool,
    max_points: int | None,
):
    points, queries = _build_problem(device)
    radius = 0.17
    results = radius_search(
        points=points,
        queries=queries,
        radius=radius,
        max_points=max_points,
        return_dists=return_dists,
        return_points=return_points,
        implementation="torch",
    )
    _assert_radius_outputs(
        points,
        queries,
        radius,
        max_points,
        return_dists,
        return_points,
        results,
    )


# Validate the warp implementation across return modes.
@requires_module("warp")
@pytest.mark.parametrize("return_dists", [True, False])
@pytest.mark.parametrize("return_points", [True, False])
@pytest.mark.parametrize("max_points", [5, None])
def test_radius_search_warp(
    device: str,
    return_dists: bool,
    return_points: bool,
    max_points: int | None,
):
    points, queries = _build_problem(device)
    radius = 0.17
    results = radius_search(
        points=points,
        queries=queries,
        radius=radius,
        max_points=max_points,
        return_dists=return_dists,
        return_points=return_points,
        implementation="warp",
    )
    _assert_radius_outputs(
        points,
        queries,
        radius,
        max_points,
        return_dists,
        return_points,
        results,
    )


# Validate benchmark input generation contract for radius search.
def test_radius_search_make_inputs_forward(device: str):
    label, args, kwargs = next(iter(RadiusSearch.make_inputs_forward(device=device)))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    output = RadiusSearch.dispatch(*args, implementation="torch", **kwargs)
    assert isinstance(output, tuple)


def test_radius_search_make_inputs_backward():
    label, args, kwargs = next(iter(RadiusSearch.make_inputs_backward(device="cpu")))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    points = args[0]
    queries = args[1]
    assert points.requires_grad
    assert queries.requires_grad

    _, output_points = RadiusSearch.dispatch(*args, implementation="torch", **kwargs)
    output_points.sum().backward()
    assert points.grad is not None


# Compare warp and torch forward outputs with order-invariant checks.
@requires_module("warp")
@pytest.mark.parametrize("max_points", [22, None])
def test_radius_search_backend_forward_parity(device: str, max_points: int | None):
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    points = torch.randn(53, 3, device=device)
    queries = torch.randn(21, 3, device=device)
    radius = 0.5

    idx_warp, pts_warp, dist_warp = radius_search(
        points,
        queries,
        radius=radius,
        max_points=max_points,
        return_dists=True,
        return_points=True,
        implementation="warp",
    )
    idx_torch, pts_torch, dist_torch = radius_search(
        points,
        queries,
        radius=radius,
        max_points=max_points,
        return_dists=True,
        return_points=True,
        implementation="torch",
    )

    RadiusSearch.compare_forward(
        (idx_warp, pts_warp, dist_warp),
        (idx_torch, pts_torch, dist_torch),
    )


# Compare warp and torch backward gradients on output points.
@requires_module("warp")
@pytest.mark.parametrize("max_points", [8, None])
def test_radius_search_backend_backward_parity(device: str, max_points: int | None):
    torch.manual_seed(42)
    points = torch.randn(88, 3, device=device, requires_grad=True)
    queries = torch.randn(57, 3, device=device, requires_grad=True)

    grads = {}
    for implementation in ("warp", "torch"):
        pts = points.clone().detach().requires_grad_(True)
        qrs = queries.clone().detach().requires_grad_(True)
        _, out_points = radius_search(
            pts,
            qrs,
            radius=0.5,
            max_points=max_points,
            return_dists=False,
            return_points=True,
            implementation=implementation,
        )
        out_points.sum().backward()
        grads[implementation] = (
            pts.grad.detach().clone() if pts.grad is not None else None,
            qrs.grad.detach().clone() if qrs.grad is not None else None,
        )

    pts_grad_warp, qrs_grad_warp = grads["warp"]
    pts_grad_torch, qrs_grad_torch = grads["torch"]
    assert pts_grad_warp is not None
    assert pts_grad_torch is not None
    RadiusSearch.compare_backward(pts_grad_warp, pts_grad_torch)

    # Query gradients are expected to be absent/unsupported for this op contract.
    assert qrs_grad_warp is None or torch.all(qrs_grad_warp == 0)
    assert qrs_grad_torch is None or torch.all(qrs_grad_torch == 0)


# Validate reduced-precision support for the warp backend.
@requires_module("warp")
@pytest.mark.parametrize("precision", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("max_points", [8, None])
def test_radius_search_reduced_precision(
    device: str,
    precision: torch.dtype,
    max_points: int | None,
):
    torch.manual_seed(42)
    points = torch.randn(88, 3, device=device, requires_grad=True).to(precision)
    queries = torch.randn(57, 3, device=device, requires_grad=True).to(precision)

    _, out_points = radius_search(
        points,
        queries,
        radius=0.5,
        max_points=max_points,
        return_dists=False,
        return_points=True,
        implementation="warp",
    )
    assert out_points.dtype == points.dtype


# Validate torch.compile support path for warp radius-search.
@requires_module("warp")
def test_radius_search_torch_compile_no_graph_break(device: str):
    if "cuda" in device:
        pytest.skip("Skipping radius search torch.compile on CUDA")
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available in this version of PyTorch")

    points = torch.randn(207, 3, device=device)
    queries = torch.randn(13, 3, device=device)

    def search_fn(points: torch.Tensor, queries: torch.Tensor):
        return radius_search(
            points,
            queries,
            radius=0.5,
            max_points=8,
            return_dists=True,
            return_points=True,
            implementation="warp",
        )

    eager = search_fn(points, queries)
    compiled = torch.compile(search_fn, fullgraph=True)(points, queries)
    for eager_tensor, compiled_tensor in zip(eager, compiled):
        torch.testing.assert_close(eager_tensor, compiled_tensor, atol=1e-6, rtol=1e-6)


# Validate custom-op schemas with torch opcheck.
@requires_module("warp")
def test_radius_search_opcheck(device: str):
    if device == "cpu":
        pytest.skip("CUDA only")
    points = torch.randn(100, 3, device=device)
    queries = torch.randn(10, 3, device=device)
    torch.library.opcheck(
        radius_search_warp,
        args=(points, queries, 0.5, 8, True, True),
    )


# Validate compare-forward hook contract for radius search.
def test_radius_search_compare_forward_contract(device: str):
    _, args, kwargs = next(iter(RadiusSearch.make_inputs_forward(device=device)))
    output = RadiusSearch.dispatch(*args, implementation="torch", **kwargs)
    reference = tuple(t.detach().clone() for t in output)
    RadiusSearch.compare_forward(output, reference)


# Validate compare-backward hook contract for radius search.
def test_radius_search_compare_backward_contract(device: str):
    _, args, kwargs = next(iter(RadiusSearch.make_inputs_backward(device=device)))
    points = args[0]
    queries = args[1]

    _, output_points = RadiusSearch.dispatch(*args, implementation="torch", **kwargs)
    output_points.sum().backward()
    assert points.grad is not None
    RadiusSearch.compare_backward(points.grad, points.grad.detach().clone())

    # Query gradients are optional for this op contract.
    if queries.grad is not None:
        RadiusSearch.compare_backward(queries.grad, queries.grad.detach().clone())


# Validate radius-search error handling paths.
@requires_module("warp")
def test_radius_search_error_handling(device: str):
    points, queries = _build_problem(device)
    if not torch.cuda.is_available():
        pytest.skip("device mismatch path requires CUDA")

    # Device mismatch is rejected by the warp custom-op implementation.
    cpu_points = points.to("cpu")
    cuda_queries = queries.to("cuda")
    with pytest.raises(ValueError, match="must be on the same device"):
        radius_search(
            points=cpu_points,
            queries=cuda_queries,
            radius=0.2,
            implementation="warp",
        )


# ---------------------------------------------------------------------------
# Batched radius-search tests  (B > 1)
# ---------------------------------------------------------------------------


def _build_batched_problem(device: str, batch_size: int, n_points=40, n_queries=15):
    """Build small (B, N, 3) / (B, Q, 3) point clouds for batched tests."""
    torch.manual_seed(0)
    points = torch.randn(batch_size, n_points, 3, device=device)
    queries = torch.randn(batch_size, n_queries, 3, device=device)
    return points, queries


def _assert_batched_radius_outputs(
    batch_size: int,
    n_points: int,
    n_queries: int,
    radius: float,
    max_points: int | None,
    return_dists: bool,
    return_points: bool,
    results,
) -> None:
    """Validate shapes and value bounds for batched radius-search outputs."""
    if return_points and return_dists:
        indices, selected_points, distances = results
    elif return_points:
        indices, selected_points = results
        distances = None
    elif return_dists:
        indices, distances = results
        selected_points = None
    else:
        indices = results
        selected_points = None
        distances = None

    if max_points is None:
        # Dynamic output: indices (3, total_count) with batch/query/point rows
        assert indices.ndim == 2
        assert indices.shape[0] == 3
        # Batch indices must be in [0, B)
        assert (indices[0] >= 0).all() and (indices[0] < batch_size).all()
        # Query indices must be in [0, Q)
        assert (indices[1] >= 0).all() and (indices[1] < n_queries).all()
        # Point indices must be in [0, N)
        assert (indices[2] >= 0).all() and (indices[2] < n_points).all()

        total = indices.shape[1]
        if selected_points is not None:
            assert selected_points.shape == (total, 3)
        if distances is not None:
            assert distances.shape == (total,)
            assert (distances >= 0).all()
            assert (distances <= radius).all()
    else:
        assert indices.shape == (batch_size, n_queries, max_points)
        # Valid indices: 0 (sentinel) or in [0, n_points)
        valid = (indices == 0) | ((indices >= 0) & (indices < n_points))
        assert valid.all()

        if selected_points is not None:
            assert selected_points.shape == (batch_size, n_queries, max_points, 3)
        if distances is not None:
            assert distances.shape == (batch_size, n_queries, max_points)
            assert (distances >= 0).all()
            assert (distances <= radius).all()


# Validate the torch implementation with batched inputs.
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("return_dists", [True, False])
@pytest.mark.parametrize("return_points", [True, False])
@pytest.mark.parametrize("max_points", [5, None])
def test_radius_search_batched_torch(
    device: str,
    batch_size: int,
    return_dists: bool,
    return_points: bool,
    max_points: int | None,
):
    n_points, n_queries = 40, 15
    points, queries = _build_batched_problem(device, batch_size, n_points, n_queries)
    radius = 1.5
    results = radius_search(
        points=points,
        queries=queries,
        radius=radius,
        max_points=max_points,
        return_dists=return_dists,
        return_points=return_points,
        implementation="torch",
    )
    _assert_batched_radius_outputs(
        batch_size,
        n_points,
        n_queries,
        radius,
        max_points,
        return_dists,
        return_points,
        results,
    )


# Validate the warp implementation with batched inputs.
@requires_module("warp")
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("return_dists", [True, False])
@pytest.mark.parametrize("return_points", [True, False])
@pytest.mark.parametrize("max_points", [5, None])
def test_radius_search_batched_warp(
    device: str,
    batch_size: int,
    return_dists: bool,
    return_points: bool,
    max_points: int | None,
):
    n_points, n_queries = 40, 15
    points, queries = _build_batched_problem(device, batch_size, n_points, n_queries)
    radius = 1.5
    results = radius_search(
        points=points,
        queries=queries,
        radius=radius,
        max_points=max_points,
        return_dists=return_dists,
        return_points=return_points,
        implementation="warp",
    )
    _assert_batched_radius_outputs(
        batch_size,
        n_points,
        n_queries,
        radius,
        max_points,
        return_dists,
        return_points,
        results,
    )


# Verify backward parity between warp and torch for batched inputs.
# Uses a small radius (0.5) so that max_points=8 does not truncate results,
# ensuring both backends select the same neighbors for deterministic gradient comparison.
@requires_module("warp")
@pytest.mark.parametrize("max_points", [8, None])
def test_radius_search_batched_backward_parity(device: str, max_points: int | None):
    torch.manual_seed(42)
    B, N, Q = 2, 88, 57
    points = torch.randn(B, N, 3, device=device, requires_grad=True)
    queries = torch.randn(B, Q, 3, device=device, requires_grad=True)

    grads = {}
    for implementation in ("warp", "torch"):
        pts = points.clone().detach().requires_grad_(True)
        qrs = queries.clone().detach().requires_grad_(True)
        _, out_points = radius_search(
            pts,
            qrs,
            radius=0.5,
            max_points=max_points,
            return_dists=False,
            return_points=True,
            implementation=implementation,
        )
        out_points.sum().backward()
        grads[implementation] = (
            pts.grad.detach().clone() if pts.grad is not None else None,
            qrs.grad.detach().clone() if qrs.grad is not None else None,
        )

    pts_grad_warp, _ = grads["warp"]
    pts_grad_torch, _ = grads["torch"]
    assert pts_grad_warp is not None
    assert pts_grad_torch is not None
    RadiusSearch.compare_backward(pts_grad_warp, pts_grad_torch)


# Verify 2D/3D input equivalence: unbatched result == batched[0] result.
@pytest.mark.parametrize("max_points", [8, None])
def test_radius_search_batched_2d_3d_equivalence(device: str, max_points: int | None):
    torch.manual_seed(7)
    pts_2d = torch.randn(40, 3, device=device)
    qs_2d = torch.randn(15, 3, device=device)
    radius = 1.5

    result_2d = radius_search(
        pts_2d,
        qs_2d,
        radius=radius,
        max_points=max_points,
        return_dists=True,
        return_points=True,
        implementation="torch",
    )
    result_3d = radius_search(
        pts_2d.unsqueeze(0),
        qs_2d.unsqueeze(0),
        radius=radius,
        max_points=max_points,
        return_dists=True,
        return_points=True,
        implementation="torch",
    )

    if max_points is not None:
        idx_2d, pts_out_2d, dists_2d = result_2d
        idx_3d, pts_out_3d, dists_3d = result_3d
        torch.testing.assert_close(idx_2d, idx_3d[0])
        torch.testing.assert_close(pts_out_2d, pts_out_3d[0])
        torch.testing.assert_close(dists_2d, dists_3d[0])
    else:
        idx_2d, pts_out_2d, dists_2d = result_2d
        idx_3d, pts_out_3d, dists_3d = result_3d
        # For dynamic output, 3D adds a batch-index row; strip it
        torch.testing.assert_close(idx_2d, idx_3d[1:])
        torch.testing.assert_close(pts_out_2d, pts_out_3d)
        torch.testing.assert_close(dists_2d, dists_3d)


# Verify that 4D+ inputs are rejected (no arbitrary batch dims).
def test_radius_search_batched_rejects_4d(device: str):
    pts_4d = torch.randn(2, 3, 10, 3, device=device)
    qs_4d = torch.randn(2, 3, 5, 3, device=device)
    with pytest.raises(ValueError, match="2D.*3D"):
        radius_search(
            pts_4d,
            qs_4d,
            radius=1.0,
            max_points=5,
            implementation="torch",
        )


# Validate batched torch.compile with deterministic (max_points) path.
@requires_module("warp")
def test_radius_search_batched_torch_compile(device: str):
    if "cuda" in device:
        pytest.skip("Skipping radius search torch.compile on CUDA")
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available in this version of PyTorch")

    B = 2
    points = torch.randn(B, 20, 3, device=device)
    queries = torch.randn(B, 8, 3, device=device)

    def search_fn(points: torch.Tensor, queries: torch.Tensor):
        return radius_search(
            points,
            queries,
            radius=1.5,
            max_points=8,
            return_dists=True,
            return_points=True,
            implementation="warp",
        )

    eager = search_fn(points, queries)
    compiled = torch.compile(search_fn, fullgraph=True)(points, queries)
    for eager_t, compiled_t in zip(eager, compiled):
        torch.testing.assert_close(eager_t, compiled_t, atol=1e-6, rtol=1e-6)


# Validate batched opcheck for custom-op schemas.
@requires_module("warp")
def test_radius_search_batched_opcheck(device: str):
    if device == "cpu":
        pytest.skip("CUDA only")
    B = 2
    points = torch.randn(B, 20, 3, device=device)
    queries = torch.randn(B, 8, 3, device=device)
    torch.library.opcheck(
        radius_search_warp,
        args=(points, queries, 1.5, 8, True, True),
    )
