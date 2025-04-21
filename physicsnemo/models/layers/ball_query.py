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

from typing import Tuple

import torch
import warp as wp


@wp.kernel
def ball_query(
    points1: wp.array(dtype=wp.vec3),
    points2: wp.array(dtype=wp.vec3),
    grid: wp.uint64,
    k: wp.int32,
    radius: wp.float32,
    mapping: wp.array3d(dtype=wp.int32),
    num_neighbors: wp.array2d(dtype=wp.int32),
):

    # Get index of point1
    tid = wp.tid()

    # Get position from points1
    pos = points1[tid]

    # particle contact
    neighbors = wp.hash_grid_query(grid, pos, radius)

    # Keep track of the number of neighbors found
    nr_found = wp.int32(0)

    # loop through neighbors to compute density
    for index in neighbors:
        # Check if outside the radius
        pos2 = points2[index]
        if wp.length(pos - pos2) > radius:
            continue

        # Add neighbor to the list
        mapping[0, tid, nr_found] = index

        # Increment the number of neighbors found
        nr_found += 1

        # Break if we have found enough neighbors
        if nr_found == k:
            num_neighbors[0, tid] = k
            break

    # Set the number of neighbors
    num_neighbors[0, tid] = nr_found


@wp.kernel
def sparse_ball_query(
    points2: wp.array(dtype=wp.vec3),
    mapping: wp.array3d(dtype=wp.int32),
    num_neighbors: wp.array2d(dtype=wp.int32),
    outputs: wp.array4d(dtype=wp.float32),
):
    # Get index of point1
    p1 = wp.tid()

    # Get number of neighbors
    k = num_neighbors[0, p1]

    # Loop through neighbors
    for _k in range(k):
        # Get point2 index
        index = mapping[0, p1, _k]

        # Get position from points2
        pos = points2[index]

        # Set the output
        outputs[0, p1, _k, 0] = pos[0]
        outputs[0, p1, _k, 1] = pos[1]
        outputs[0, p1, _k, 2] = pos[2]


def _ball_query_forward_primative_(
    points1: torch.Tensor,
    points2: torch.Tensor,
    k: int,
    radius: float,
    hash_grid: wp.HashGrid,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # Create output tensors:
    mapping = torch.zeros(
        (1, points1.shape[0], k),
        dtype=torch.int32,
        device=points1.device,
        requires_grad=False,
    )
    num_neighbors = torch.zeros(
        (1, points1.shape[0]),
        dtype=torch.int32,
        device=points1.device,
        requires_grad=False,
    )
    outputs = torch.zeros(
        (1, points1.shape[0], k, 3),
        dtype=torch.float32,
        device=points1.device,
        requires_grad=(points1.requires_grad or points2.requires_grad),
    )

    # Convert from torch to warp
    points1 = wp.from_torch(points1, dtype=wp.vec3, requires_grad=points1.requires_grad)
    points2 = wp.from_torch(points2, dtype=wp.vec3, requires_grad=points2.requires_grad)

    wp_mapping = wp.from_torch(mapping, dtype=wp.int32, requires_grad=False)
    wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32, requires_grad=False)
    wp_outputs = wp.from_torch(
        outputs,
        dtype=wp.float32,
        requires_grad=(points1.requires_grad or points2.requires_grad),
    )

    # Build the grid
    hash_grid.build(points2, radius)

    # Run the kernel to get mapping
    wp.launch(
        ball_query,
        inputs=[
            points1,
            points2,
            hash_grid.id,
            k,
            radius,
        ],
        outputs=[
            wp_mapping,
            wp_num_neighbors,
        ],
        dim=[points1.shape[0]],
    )

    # Run the kernel to get outputs
    wp.launch(
        sparse_ball_query,
        inputs=[
            points2,
            wp_mapping,
            wp_num_neighbors,
        ],
        outputs=[
            wp_outputs,
        ],
        dim=[points1.shape[0]],
    )

    return mapping, num_neighbors, outputs


def _ball_query_backward_primative_(
    points1,
    points2,
    mapping,
    num_neighbors,
    outputs,
    grad_mapping,
    grad_num_neighbors,
    grad_outputs,
) -> Tuple[torch.Tensor, torch.Tensor]:

    p2_grad = torch.zeros_like(points2)

    # Run the kernel in adjoint mode
    wp.launch(
        sparse_ball_query,
        inputs=[
            wp.from_torch(points2, dtype=wp.vec3, requires_grad=points2.requires_grad),
            wp.from_torch(mapping, dtype=wp.int32, requires_grad=False),
            wp.from_torch(num_neighbors, dtype=wp.int32, requires_grad=False),
        ],
        outputs=[
            wp.from_torch(outputs, dtype=wp.float32, requires_grad=True),
        ],
        adj_inputs=[
            wp.from_torch(p2_grad, dtype=wp.vec3, requires_grad=points2.requires_grad),
            wp.from_torch(
                grad_mapping, dtype=wp.int32, requires_grad=mapping.requires_grad
            ),
            wp.from_torch(
                grad_num_neighbors,
                dtype=wp.int32,
                requires_grad=num_neighbors.requires_grad,
            ),
        ],
        adj_outputs=[
            wp.from_torch(grad_outputs, dtype=wp.float32),
        ],
        dim=[points1.shape[0]],
        adjoint=True,
    )

    return p2_grad


class BallQuery(torch.autograd.Function):
    """
    Warp based Ball Query.
    """

    @staticmethod
    def forward(
        ctx,
        points1,
        points2,
        k,
        radius,
        hash_grid,
    ):
        # Only works for batch size 1
        if points1.shape[0] != 1:
            raise AssertionError("Ball Query only works for batch size 1")

        ctx.k = k
        ctx.radius = radius

        # Make grid
        ctx.hash_grid = hash_grid

        # Apply the primitive.  Note the batch index is removed.
        mapping, num_neighbors, outputs = _ball_query_forward_primative_(
            points1[0],
            points2[0],
            k,
            radius,
            hash_grid,
        )
        ctx.save_for_backward(points1, points2, mapping, num_neighbors, outputs)

        return mapping, num_neighbors, outputs

    @staticmethod
    def backward(ctx, grad_mapping, grad_num_neighbors, grad_outputs):

        points1, points2, mapping, num_neighbors, outputs = ctx.saved_tensors

        # Apply the primitive
        p2_grad = _ball_query_backward_primative_(
            points1[0],
            points2[0],
            mapping,
            num_neighbors,
            outputs,
            grad_mapping,
            grad_num_neighbors,
            grad_outputs,
        )

        p2_grad = p2_grad.unsqueeze(0)

        # Return the gradients
        return (
            torch.zeros_like(points1),
            p2_grad,
            None,
            None,
            None,
        )


def ball_query_layer(points1, points2, k, radius, hash_grid):
    """
    Wrapper for BallQuery.apply to support a functional interface.
    """
    return BallQuery.apply(points1, points2, k, radius, hash_grid)


class BallQueryLayer(torch.nn.Module):
    """
    Torch layer for differentiable and accelerated Ball Query
    operation using Warp.
    Args:
        k (int): Number of neighbors.
        radius (float): Radius of influence.
        grid_size (int): Uniform grid resolution
    """

    def __init__(self, k, radius, grid_size=32):
        super().__init__()
        wp.init()
        self.k = k
        self.radius = radius
        self.hash_grid = wp.HashGrid(grid_size, grid_size, grid_size)

    def forward(self, points1, points2):
        # Why not just call the function directly?
        # Using a functional interface makes a distributed interception easier.
        return ball_query_layer(
            points1,
            points2,
            self.k,
            self.radius,
            self.hash_grid,
        )


if __name__ == "__main__":
    # Make function for saving point clouds
    import pyvista as pv

    def save_point_cloud(points, name):
        cloud = pv.PolyData(points.detach().cpu().numpy())
        cloud.save(name)

    # Check forward pass
    # Initialize tensors
    n = 1  # number of point clouds
    p1 = 128000  # 100000  # number of points in point cloud 1
    d = 3  # dimension of the points
    p2 = 39321  # 100000  # number of points in point cloud 2
    points1 = torch.rand(n, p1, d, device="cuda", requires_grad=True)

    points2 = torch.rand(n, p2, d, device="cuda", requires_grad=True)
    k = 256  # maximum number of neighbors
    radius = 0.1

    # Make ball query layer
    layer = BallQueryLayer(k, radius)

    # Make ball query
    with wp.ScopedTimer("ball query", active=True):
        mapping, num_neighbors, outputs = layer(
            points1,
            points2,
        )

    for i in range(20):
        p1 += 100
        p2 += 100
        points1 = torch.rand(n, p1, d, device="cuda", requires_grad=False)
        points2 = torch.rand(n, p2, d, device="cuda", requires_grad=False)

        mapping, num_neighbors, outputs = layer(
            points1,
            points2,
        )

    # Perform matrix multiplication as comparison for timing
    with wp.ScopedTimer("matrix multiplication 256", active=True):
        outputs2 = torch.matmul(points1[0], torch.ones(3, k, device="cuda"))

    # Save the point clouds
    save_point_cloud(points1[0], "point1.vtk")
    save_point_cloud(points2[0], "point2.vtk")
    save_point_cloud(outputs[0].reshape(-1, 3), "outputs.vtk")

    # Optimize the background points to move to the query points
    optimizer = torch.optim.SGD([points2], 0.01)

    # Test optimization
    for i in range(100):
        optimizer.zero_grad()
        # mapping, num_neighbors, outputs = layer(points1, points2, lengths1, lengths2)
        mapping, num_neighbors, outputs = layer(points1, points2)

        loss = (points1.unsqueeze(2) - outputs).pow(2).sum()
        loss.backward()
        optimizer.step()

        # Save the point clouds
        save_point_cloud(points1[0], "point1_{}.vtk".format(i))
        save_point_cloud(outputs[0].reshape(-1, 3), "outputs_{}.vtk".format(i))
