# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

from physicsnemo.utils.neighbors import radius_search


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("return_dists", [True, False])
@pytest.mark.parametrize("return_points", [True, False])
@pytest.mark.parametrize("max_points", [5, None])
@pytest.mark.parametrize("backend", ["warp", "torch"])
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
    backend: str,
    radius: float,
):
    """
    We use two utilities for this.

    There is a default pytorch implementation, which uses a brute force computation
    at significant expense of memory and computation.

    There is a warp backended implementation which uses hash maps and
    a better algorithm.

    the test here is to enforce agreement between the two utilities.
    There is no reason to directly test the results: the algorithm
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
        backend=backend,
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

    print(f"Indexes shape: {indexes.shape}")
    # Basic shape checks - there should be one index array per query point
    if max_points is not None:
        assert indexes.shape[0] == query_space_points.shape[0]
    else:
        assert indexes.shape[0] == 2
        # There is not really a constraint on the second dimension
        # (Except that we know exactly because of the input data,  for the test)
        # It's checked below.

    # Check that no point has more matches than max_points (if specified)
    if max_points is not None:
        assert (indexes != -1).sum(dim=1).max() <= max_points
    else:
        # In this case, there should be no -1 items:
        assert (indexes == -1).sum() == 0

    # Check that found points are valid indices
    valid_indices = (indexes == -1) | (
        (indexes >= 0) & (indexes < search_space_points.shape[0])
    )
    assert valid_indices.all()

    # Check that all found points are within the radius
    if return_dists:
        assert (dists >= 0).all()
        assert (dists <= radius).all()

        # Points marked with -1 should have distance set to 0
        if max_points is not None:
            mask = indexes == -1
            assert (dists[mask] == 0).all()

    if return_points:

        print(points.shape)
        if max_points is not None:
            assert points.shape[0] == query_space_points.shape[0]
            assert points.shape[1] == max_points
            assert points.shape[2] == 3

            points_selection = torch.where(indexes != -1)
            selected_indexes = indexes[points_selection]
            # Retrieve the points from the original tensor based on the index:
            index_selected_points = torch.index_select(
                search_space_points, 0, selected_indexes
            )

            flattened_points = points.reshape(-1, 3)
            flattened_indexes = indexes.reshape(-1)

            query_selected_points = flattened_points[flattened_indexes != -1]

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
    print(radius, expected_matches)
    # expected_matches = 1
    if max_points is not None:
        # Some limit has been imposed:
        expected_matches = min(max_points, expected_matches)
        # We sum the non -1 count
        matches_per_query = (indexes != -1).sum(dim=1)
        assert (matches_per_query == expected_matches).all()

    else:

        # We should have exactly expected_matches match for each query point.
        assert indexes.shape[1] == expected_matches * query_space_points.shape[0]


if __name__ == "__main__":
    test_radius_search(
        device="cpu",
        return_dists=True,
        return_points=True,
        max_points=5,
        backend="warp",
        radius=0.17,
    )
