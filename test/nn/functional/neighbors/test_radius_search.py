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

import warnings

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


@pytest.mark.parametrize("batched", [False, True])
def test_radius_search_torch_fewer_points_than_max_points(device: str, batched: bool):
    """Padded top-k slots must not be reported as neighbors.

    With fewer points than `max_points`, the padded slots had distance 0.0, so
    they passed the radius mask as phantom neighbors at point 0.
    """
    points = torch.tensor(
        [[1.0, 2.0, 3.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0]],
        device=device,
        requires_grad=True,
    )
    # First query has exactly one neighbor (point 1); second has none.
    queries = torch.tensor(
        [[10.0, 10.0, 10.1], [50.0, 50.0, 50.0]],
        device=device,
    )

    # Use batched or unbatched input
    if batched:
        points_in, queries_in = points[None], queries[None]
    else:
        points_in, queries_in = points, queries

    max_points = 5

    indices, selected_points, distances = radius_search(
        points=points_in,
        queries=queries_in,
        radius=0.5,
        max_points=max_points,
        return_dists=True,
        return_points=True,
        implementation="torch",
    )

    if batched:
        indices, selected_points, distances = (
            indices[0],
            selected_points[0],
            distances[0],
        )

    assert indices.shape == (2, max_points)
    assert selected_points.shape == (2, max_points, 3)
    assert distances.shape == (2, max_points)

    # Query 0: a single neighbor (point 1), remaining slots unused.
    assert indices[0, 0] == 1
    torch.testing.assert_close(selected_points[0, 0], points[1], atol=0.0, rtol=0.0)
    torch.testing.assert_close(distances[0, 0], torch.tensor(0.1, device=device))
    assert (indices[0, 1:] == 0).all()
    assert (selected_points[0, 1:] == 0).all()
    assert (distances[0, 1:] == 0).all()

    # Query 1: no neighbors at all, every slot unused and zero-filled.
    assert (indices[1] == 0).all()
    assert (selected_points[1] == 0).all()
    assert (distances[1] == 0).all()

    # Gradient only flows into the one real neighbor.
    selected_points.sum().backward()
    expected_grad = torch.zeros_like(points)
    expected_grad[1] = 1.0
    torch.testing.assert_close(points.grad, expected_grad)


def test_radius_search_torch_infinite_radius_fewer_points_than_max_points(
    device: str,
):
    """With an infinite radius, every point is a neighbor and padded slots stay unused."""
    points = torch.randn(3, 3, device=device, requires_grad=True)
    queries = torch.randn(2, 3, device=device)

    _, selected_points, distances = radius_search(
        points=points,
        queries=queries,
        radius=float("inf"),
        max_points=5,
        return_dists=True,
        return_points=True,
        implementation="torch",
    )

    # The 3 points fill slots 0-2 of each query; slots 3-4 are padding.
    assert (selected_points[:, 3:] == 0).all()
    assert (distances[:, 3:] == 0).all()

    # Each point is a neighbor of both queries, so each gets a gradient of 2.
    selected_points.sum().backward()
    torch.testing.assert_close(points.grad, torch.full_like(points, 2.0))


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


# Validate that compare-forward ignores the order of the neighbors of each query.
# With three queries, fixed-size indices have as many rows as batched dynamic ones,
# and without points, the distances for three slots pass as dynamic points.
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("max_points", [3, 4])
@pytest.mark.parametrize("return_points", [True, False])
def test_radius_search_compare_forward_reordered_neighbors(
    device: str, batched: bool, max_points: int, return_points: bool
):
    torch.manual_seed(0)
    batch = (2,) if batched else ()
    points = torch.rand(*batch, 50, 3, device=device)
    queries = torch.rand(*batch, 3, 3, device=device)

    # Every point is within the radius, so every slot holds a neighbor.
    output = radius_search(
        points,
        queries,
        radius=2.0,
        max_points=max_points,
        return_dists=True,
        return_points=return_points,
        implementation="torch",
    )
    # Reverse the neighbors of each query, which the indices hold along their
    # last dimension.
    neighbor_dim = output[0].ndim - 1
    reference = tuple(torch.flip(t, dims=[neighbor_dim]) for t in output)
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
    with pytest.raises(ValueError, match="rank 2"):
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


# ---------------------------------------------------------------------------
# Tests for points with different numbers of coordinates
# ---------------------------------------------------------------------------

_IMPLEMENTATIONS = ("torch", pytest.param("warp", marks=requires_module("warp")))

# The torch backend accepts any number of coordinates, the warp backend 1 to 3.
_DIM_CASES = [("torch", dim) for dim in (1, 2, 3, 4)] + [
    pytest.param("warp", dim, marks=requires_module("warp")) for dim in (1, 2, 3)
]


def _build_point_dim_problem(device: str, dim: int, batched: bool = False):
    """Build (50, dim) points and (10, dim) queries in the unit cube.

    The last query lies outside the cube, more than the 0.3 search radius from
    every point. With ``batched``, add a leading batch dimension of size 2.
    """
    # Generate on CPU so that every device gets the same points. None of them is
    # within float32 rounding of the 0.3 search radius from a query.
    torch.manual_seed(0)
    batch = (2,) if batched else ()
    points = torch.rand(*batch, 50, dim)
    queries = torch.rand(*batch, 10, dim)
    queries[..., -1, :] = 2.0
    return points.to(device), queries.to(device)


# Compare all neighbors within the radius, and gradients through them, with a
# brute-force cdist reference.
@pytest.mark.parametrize("implementation, dim", _DIM_CASES)
@pytest.mark.parametrize("batched", [False, True])
def test_radius_search_point_dims(
    device: str, implementation: str, dim: int, batched: bool
):
    points, queries = _build_point_dim_problem(device, dim, batched)
    points.requires_grad_(True)
    radius = 0.3
    indices, selected_points, distances = radius_search(
        points,
        queries,
        radius=radius,
        return_dists=True,
        return_points=True,
        implementation=implementation,
    )

    # Each (query, point) pair within the radius is found exactly once. Batched
    # indices start with a row of batch indices.
    reference = torch.cdist(queries.double(), points.detach().double())
    *batch_idx, query_idx, point_idx = indices
    found = torch.zeros_like(reference, dtype=torch.bool)
    found[(*batch_idx, query_idx, point_idx)] = True
    assert torch.equal(found, reference <= radius)
    assert indices.shape[1] == found.sum()

    assert selected_points.shape == (indices.shape[1], dim)
    assert selected_points.is_contiguous()
    torch.testing.assert_close(
        selected_points, points.detach()[(*batch_idx, point_idx)]
    )
    torch.testing.assert_close(
        distances, reference[(*batch_idx, query_idx, point_idx)].to(distances.dtype)
    )

    # Each neighbor passes a gradient of 1 to its point.
    selected_points.sum().backward()
    counts = found.sum(dim=-2).to(points.dtype)
    torch.testing.assert_close(points.grad, counts[..., None].expand_as(points))


# Validate the fixed-size outputs, and gradients through them, with max_points.
@pytest.mark.parametrize("implementation, dim", _DIM_CASES)
@pytest.mark.parametrize("batched", [False, True])
def test_radius_search_point_dims_max_points(
    device: str, implementation: str, dim: int, batched: bool
):
    points, queries = _build_point_dim_problem(device, dim, batched)
    points.requires_grad_(True)
    radius, max_points = 0.3, 4
    indices, selected_points, distances = radius_search(
        points,
        queries,
        radius=radius,
        max_points=max_points,
        return_dists=True,
        return_points=True,
        implementation=implementation,
    )
    assert indices.shape == (*queries.shape[:-1], max_points)
    assert selected_points.shape == (*queries.shape[:-1], max_points, dim)
    assert selected_points.is_contiguous()
    assert distances.shape == (*queries.shape[:-1], max_points)

    # Each query fills its first slots with distinct neighbors, as many as fit,
    # and leaves the other slots zero. The last query has no neighbors.
    reference = torch.cdist(queries.double(), points.detach().double())
    within = reference <= radius
    num_used = within.sum(dim=-1, keepdim=True).clamp(max=max_points)
    used = torch.arange(max_points, device=device) < num_used
    *batch_idx, query_idx, _ = torch.nonzero(used, as_tuple=True)
    point_idx = indices[used]
    assert within[(*batch_idx, query_idx, point_idx)].all()
    found = torch.zeros_like(within)
    found[(*batch_idx, query_idx, point_idx)] = True
    assert found.sum() == used.sum()
    torch.testing.assert_close(
        selected_points[used], points.detach()[(*batch_idx, point_idx)]
    )
    torch.testing.assert_close(
        distances[used],
        reference[(*batch_idx, query_idx, point_idx)].to(distances.dtype),
    )
    assert (indices[~used] == 0).all()
    assert (selected_points[~used] == 0).all()
    assert (distances[~used] == 0).all()

    # Each used slot passes a gradient of 1 to the point that it holds, and the
    # slots of a query hold distinct points.
    selected_points.sum().backward()
    counts = found.sum(dim=-2).to(points.dtype)
    torch.testing.assert_close(points.grad, counts[..., None].expand_as(points))


# Compare warp and torch outputs with the order-invariant compare hook.
@requires_module("warp")
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_radius_search_point_dims_backend_forward_parity(device: str, dim: int):
    points, queries = _build_point_dim_problem(device, dim)
    outputs = [
        radius_search(
            points,
            queries,
            radius=0.3,
            return_dists=True,
            return_points=True,
            implementation=implementation,
        )
        for implementation in ("warp", "torch")
    ]
    RadiusSearch.compare_forward(*outputs)


# Validate that both backends reject points without coordinates, and points and
# queries with different numbers of coordinates.
@pytest.mark.parametrize("implementation", _IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "points_dim, queries_dim, match",
    [
        (0, 0, "at least 1 coordinate"),
        (2, 3, "same number of coordinates"),
    ],
)
def test_radius_search_rejects_point_dims(
    device: str, implementation: str, points_dim: int, queries_dim: int, match: str
):
    points = torch.rand(20, points_dim, device=device)
    queries = torch.rand(5, queries_dim, device=device)
    with pytest.raises(ValueError, match=match):
        radius_search(points, queries, radius=0.3, implementation=implementation)


# Validate that the warp backend rejects more than 3 coordinates, and that the
# default selection uses the torch backend for them.
@requires_module("warp")
def test_radius_search_more_than_3_coordinates(device: str):
    points, queries = _build_point_dim_problem(device, 4)
    with pytest.raises(ValueError, match="1, 2, or 3 coordinates"):
        radius_search(points, queries, radius=0.3, implementation="warp")

    default = radius_search(points, queries, radius=0.3, return_dists=True)
    reference = radius_search(
        points, queries, radius=0.3, return_dists=True, implementation="torch"
    )
    for default_tensor, reference_tensor in zip(default, reference):
        torch.testing.assert_close(default_tensor, reference_tensor)


# Validate that the warp forward pass leaves .grad unset. Warp allocates a
# gradient for an array made from a tensor that requires grad, unless it is passed
# requires_grad=False.
@requires_module("warp")
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_radius_search_warp_forward_leaves_grad_unset(device: str, dim: int):
    points, queries = _build_point_dim_problem(device, dim, batched=True)
    points.requires_grad_(True)
    queries.requires_grad_(True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        radius_search(
            points,
            queries,
            radius=0.3,
            max_points=4,
            return_points=True,
            implementation="warp",
        )
    assert points.grad is None
    assert queries.grad is None
    # With 1 or 2 coordinates, Warp gets padded copies, which are not leaf
    # tensors, so reading their .grad would warn.
    assert not [w for w in caught if ".grad attribute" in str(w.message)]
