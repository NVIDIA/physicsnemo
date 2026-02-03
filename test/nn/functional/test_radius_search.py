# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
from physicsnemo.nn.functional.radius_search._warp_impl import (
    radius_search_impl as radius_search_warp,
)


@pytest.mark.parametrize("return_dists", [True, False])
@pytest.mark.parametrize("return_points", [True, False])
@pytest.mark.parametrize("max_points", [5, None])
@pytest.mark.parametrize("implementation", ["warp", "torch"])
@pytest.mark.parametrize(
    "radius",
    [
        0.17,
    ],
)
def test_radius_search(
    device: str,
    return_dists: bool,
    return_points: bool,
    max_points: int | None,
    implementation: str,
    radius: float,
):
    """
    We use two utilities for this.

    There is a default pytorch implementation, which uses a brute force computation
    at significant expense of memory and computation.

    There is a warp backended implementation which uses hash maps and
    a better algorithm.

    the test here is to enforce agreement between the two utilities.
    There is no reason "Torch doesn't support pinned memory on mac"m
    that would be written here is identical to the torch backend tools.
    """

    assert radius > 0, "Radius must be positive"
    assert radius <= 0.5, "Radius must be less than 0.5 for the logic of this test."

    # We do is in a predictable, consistent way to ensure the results are valid.

    # First, generate the query points.  We take 11 points, from 0 to 10, and turn
    # that into a 3D grid.  (that's a point a 0.0, 1.0, ... 10.0)
    points = torch.linspace(0, 10, 11, device=device)
    x, y, z = torch.meshgrid(points, points, points, indexing="ij")
    query_space_points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)

    # Next, generate displacements.
    # We take 6 displacements, one in each cardinal direction, and increasing by 0.1 each time.
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

    # Add the displacements to each query point:
    search_space_points = query_space_points[None, :, :] + displacements[:, None, :]
    search_space_points = search_space_points.reshape(-1, 3)

    # Now, we have a grid of query points, and a grid of search space points.

    # Ok, now we can check results.

    # The points selected will depend on the radius.  Note the displacements are not overlapping because the
    # central points are spaced by 1.0 but the displacements stretch to < 0.5.

    results = radius_search(
        search_space_points,
        query_space_points,
        radius=radius,
        max_points=max_points,
        return_dists=return_dists,
        return_points=return_points,
        implementation=implementation,
    )

    # unpack:
    if return_points:
        if return_dists:
            indexes, points, dists = results
        else:
            indexes, points = results
    elif return_dists:
        indexes, dists = results
    else:
        indexes = results

    # Basic shape checks - there should be one index array per query point
    if max_points is not None:
        assert indexes.shape[0] == query_space_points.shape[0]
    else:
        assert indexes.shape[0] == 2
        # There is not really a constraint on the second dimension
        # (Except that we know exactly because of the input data,  for the test)
        # It's checked below.

    # # Check that no point has more matches than max_points (if specified)
    # if max_points is not None:
    #     assert (indexes != -1).sum(dim=1).max() <= max_points
    # else:
    #     # In this case, there should be no -1 items:
    #     assert (indexes == -1).sum() == 0

    # Check that found points are valid indices
    valid_indices = (indexes != 0) | (
        (indexes >= 0) & (indexes < search_space_points.shape[0])
    )
    assert valid_indices.all()

    # Check that all found points are within the radius
    if return_dists:
        assert (dists >= 0).all()
        assert (dists <= radius).all()

        # Points marked with -1 should have distance set to 0
        # (except the first one, since that *actually* matches point 0)
        if max_points is not None:
            mask = indexes == 0

            assert (dists[mask][1:] == 0).all()

    if return_points:
        if max_points is not None:
            assert points.shape[0] == query_space_points.shape[0]
            assert points.shape[1] == max_points
            assert points.shape[2] == 3

            points_selection = torch.where(indexes != 0)
            selected_indexes = indexes[points_selection]
            # Retrieve the points from the original tensor based on the index:
            index_selected_points = torch.index_select(
                search_space_points, 0, selected_indexes
            )

            flattened_points = points.reshape(-1, 3)
            flattened_indexes = indexes.reshape(-1)

            query_selected_points = flattened_points[flattened_indexes != 0]

            assert torch.allclose(index_selected_points, query_selected_points)

        else:
            assert points.shape[0] == indexes.shape[1]
            assert points.shape[1] == 3

            # Check that if we index into the original tensor,
            # We get back the same points.
            retrieved_points = search_space_points[indexes[1]]

            assert torch.allclose(retrieved_points, points)

    # Since we know the displacements are regular, we can check that each query point
    # finds exactly one point within radius 0.1 (the 0.05 displaced point)
    # This is how many are possible by the data:
    expected_matches = min(int(radius / 0.05), 6)
    # expected_matches = 1
    if max_points is not None:
        # Some limit has been imposed:
        expected_matches = min(max_points, expected_matches)
        # We sum the non 0 count
        # Note that the very first point should match itself ... so exlude
        # the first from the assertion.

        matches_per_query = (indexes != 0).sum(dim=1)
        assert (matches_per_query[1:] == expected_matches).all()

    else:
        # We should have exactly expected_matches match for each query point.
        assert indexes.shape[1] == expected_matches * query_space_points.shape[0]


def test_radius_search_torch_compile_no_graph_break(device):
    # Cuda curnently disabled in this test, but it does work.

    if "cuda" in device:
        pytest.skip("Skipping radius search torch compile for CUDA")

    import torch

    # Only test if torch.compile is available (PyTorch 2.0+)
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available in this version of PyTorch")

    # Prepare test data
    points = torch.randn(207, 3, device=device)
    queries = torch.randn(13, 3, device=device)
    radius = 0.5
    max_points = 8

    def search_fn(points, queries):
        return radius_search(
            points,
            queries,
            radius=radius,
            max_points=max_points,
            return_dists=True,
            return_points=True,
            implementation="warp",
        )

    # Run both and compare outputs
    out_eager = search_fn(points, queries)

    compiled_fn = torch.compile(search_fn, fullgraph=True)

    out_compiled = compiled_fn(points, queries)

    # Compare outputs (tuple of tensors)
    for eager, compiled in zip(out_eager, out_compiled):
        assert torch.allclose(eager, compiled, atol=1e-6)


def test_opcheck(device):
    if device == "cpu":
        pytest.skip("CUDA only")

    points = torch.randn(100, 3, device=device)
    queries = torch.randn(10, 3, device=device)
    radius = 0.5
    max_points = 8

    torch.library.opcheck(
        radius_search_warp, args=(points, queries, radius, max_points, True, True)
    )


@pytest.mark.parametrize("max_points", [22, None])
def test_radius_search_comparison(device, max_points):
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    points = torch.randn(53, 3, device=device)
    queries = torch.randn(21, 3, device=device)
    radius = 0.5

    return_points = True
    return_dists = True

    index_warp, out_points_warp, distance_warp = radius_search(
        points,
        queries,
        radius,
        max_points,
        return_dists,
        return_points,
        implementation="warp",
    )
    index_torch, out_points_torch, distance_torch = radius_search(
        points,
        queries,
        radius,
        max_points,
        return_dists,
        return_points,
        implementation="torch",
    )

    # The points may not come out in the same order in each.  So, we check only against the sums:
    if max_points is not None:
        assert torch.allclose(
            index_warp.sum(dim=1), index_torch.to(torch.int32).sum(dim=1)
        )
    else:
        assert torch.allclose(
            index_warp.sum(dim=1), index_torch.to(torch.int32).sum(dim=1)
        )

    if max_points is not None:
        assert torch.allclose(out_points_warp.sum(dim=1), out_points_torch.sum(dim=1))
    else:
        assert torch.allclose(
            out_points_warp.sum(dim=(0)), out_points_torch.sum(dim=(0))
        )

    if max_points is not None:
        assert torch.allclose(distance_warp.sum(dim=1), distance_torch.sum(dim=1))
    else:
        assert torch.allclose(distance_warp.sum(), distance_torch.sum())


@pytest.mark.parametrize("max_points", [8, None])
def test_radius_search_gradients(device, max_points):
    # Gradients are only supported to flow through the output points.
    # Therefore there are NO gradients if return_points=False

    # Additionally, we can only compare gradients of the points
    # Gradients of the queries doesn't make sense.

    torch.manual_seed(42)
    n_points = 88
    n_queries = 57
    radius = 0.5

    # Create points and queries with gradients enabled
    points = torch.randn(n_points, 3, device=device, requires_grad=True)
    queries = torch.randn(n_queries, 3, device=device, requires_grad=True)

    grads = {}
    for implementation in ["warp", "torch"]:
        # Clone inputs for each backend to avoid in-place ops
        pts = points.clone().detach().requires_grad_(True)
        qrs = queries.clone().detach().requires_grad_(True)
        index, out_points = radius_search(
            pts,
            qrs,
            radius=radius,
            max_points=max_points,
            return_dists=False,
            return_points=True,
            implementation=implementation,
        )
        # Only sum over valid distances (where index != -1)
        out_points.sum().backward()

        grads[implementation] = (
            pts.grad.detach().clone() if pts.grad is not None else None,
            qrs.grad.detach().clone() if qrs.grad is not None else None,
        )
    # Compare gradients between backends
    pts_grad_warp, qrs_grad_warp = grads["warp"]
    pts_grad_torch, qrs_grad_torch = grads["torch"]

    assert torch.allclose(pts_grad_warp, pts_grad_torch, atol=1e-5), (
        "Point gradients do not match"
    )

    # assert torch.allclose(qrs_grad_warp, qrs_grad_torch, atol=1e-5), "Query gradients do not match"


@pytest.mark.parametrize("precision", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("max_points", [8, None])
def test_radius_search_reduced_precision(device, precision, max_points):
    """
    This is a functionality based test.  We run in half precision and
    make sure results are reasonable, but exact agreement from the alg is tested
    elsewhere.
    """

    torch.manual_seed(42)
    n_points = 88
    n_queries = 57
    radius = 0.5

    # Create points and queries with gradients enabled
    points = torch.randn(n_points, 3, device=device, requires_grad=True).to(
        dtype=precision
    )
    queries = torch.randn(n_queries, 3, device=device, requires_grad=True).to(
        dtype=precision
    )

    index, out_points = radius_search(
        points,
        queries,
        radius=radius,
        max_points=max_points,
        return_dists=False,
        return_points=True,
        implementation="warp",
    )

    assert out_points.dtype == points.dtype
