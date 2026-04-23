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
