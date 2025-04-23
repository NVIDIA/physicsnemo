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


import torch
import warp as wp


class BallQuery(torch.autograd.Function):
    """
    Warp based Ball Query.

    Note: only differentiable with respect to points1 and points2.
    """

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
        """
        Performs ball query operation to find neighboring points within a specified radius.

        For each point in points1, finds up to k neighboring points from points2 that are
        within the specified radius. Uses a hash grid for efficient spatial queries.

        Note that the neighbors found are not strictly guaranteed to be the closest k neighbors,
        in the event that more than k neighbors are found within the radius.

        Args:
            points1: Array of query points
            points2: Array of points to search
            grid: Pre-computed hash grid for accelerated spatial queries
            k: Maximum number of neighbors to find for each query point
            radius: Maximum search radius for finding neighbors
            mapping: Output array to store indices of neighboring points. Should be instantiated as zeros(1, len(points1), k)
            num_neighbors: Output array to store the number of neighbors found for each query point. Should be instantiated as zeros(1, len(points1))
        """
        tid = wp.tid()

        # Get position from points1
        pos = points1[tid]

        # particle contact
        neighbors = wp.hash_grid_query(id=grid, point=pos, max_dist=radius)

        # Keep track of the number of neighbors found
        neighbors_found = wp.int32(0)

        # loop through neighbors to compute density
        for index in neighbors:
            # Check if outside the radius
            pos2 = points2[index]
            if wp.length(pos - pos2) > radius:
                continue

            # Add neighbor to the list
            mapping[0, tid, neighbors_found] = index

            # Increment the number of neighbors found
            neighbors_found += 1

            # Break if we have found enough neighbors
            if neighbors_found == k:
                num_neighbors[0, tid] = k
                break

        # Set the number of neighbors
        num_neighbors[0, tid] = neighbors_found

    @wp.kernel
    def sparse_ball_query(
        points2: wp.array(dtype=wp.vec3),
        mapping: wp.array3d(dtype=wp.int32),
        num_neighbors: wp.array2d(dtype=wp.int32),
        outputs: wp.array4d(dtype=wp.float32),
    ):
        tid = wp.tid()

        # Get number of neighbors
        k = num_neighbors[0, tid]

        # Loop through neighbors
        for _k in range(k):
            # Get point2 index
            index = mapping[0, tid, _k]

            # Get position from points2
            pos = points2[index]

            # Set the output
            outputs[0, tid, _k, 0] = pos[0]
            outputs[0, tid, _k, 1] = pos[1]
            outputs[0, tid, _k, 2] = pos[2]

    @staticmethod
    def forward(
        ctx,
        points1: torch.Tensor,
        points2: torch.Tensor,
        # lengths1: torch.Tensor,
        # lengths2: torch.Tensor,
        k: int,
        radius: float,
        hash_grid: wp.HashGrid,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Only works for batch size 1
        if points1.shape[0] != 1:
            raise AssertionError("nly works for batch size 1")

        # Convert from torch to warp
        ctx.points1 = wp.from_torch(
            points1[0], dtype=wp.vec3, requires_grad=points1.requires_grad
        )
        ctx.points2 = wp.from_torch(
            points2[0], dtype=wp.vec3, requires_grad=points2.requires_grad
        )
        # ctx.lengths1 = wp.from_torch(lengths1, dtype=wp.int32, requires_grad=False)
        # ctx.lengths2 = wp.from_torch(lengths2, dtype=wp.int32, requires_grad=False)
        ctx.k = k
        ctx.radius = radius

        # Allocate the mapping and outputs
        mapping = torch.zeros(
            1,
            points1.shape[1],
            k,
            dtype=torch.int32,
            device="cuda",
            requires_grad=False,
        )
        ctx.mapping = wp.from_torch(mapping, dtype=wp.int32, requires_grad=False)
        num_neighbors = torch.zeros(
            1,
            points1.shape[1],
            dtype=torch.int32,
            device="cuda",
            requires_grad=False,
        )
        ctx.num_neighbors = wp.from_torch(
            num_neighbors, dtype=wp.int32, requires_grad=False
        )
        outputs = torch.zeros(
            1,
            points1.shape[1],
            k,
            3,
            dtype=torch.float32,
            device="cuda",
            requires_grad=(points1.requires_grad or points2.requires_grad),
        )
        ctx.outputs = wp.from_torch(
            outputs,
            dtype=wp.float32,
            requires_grad=(points1.requires_grad or points2.requires_grad),
        )

        # Make grid
        ctx.hash_grid = hash_grid

        # Build the grid
        ctx.hash_grid.build(ctx.points2, radius)

        ctx.dim = [ctx.points1.shape[0]]

        # Run the kernel to get mapping
        wp.launch(
            BallQuery.ball_query,
            inputs=[
                ctx.points1,
                ctx.points2,
                ctx.hash_grid.id,
                k,
                radius,
            ],
            outputs=[
                ctx.mapping,
                ctx.num_neighbors,
            ],
            dim=ctx.dim,
        )

        # Run the kernel to get outputs
        wp.launch(
            BallQuery.sparse_ball_query,
            inputs=[
                ctx.points2,
                ctx.mapping,
                ctx.num_neighbors,
            ],
            outputs=[
                ctx.outputs,
            ],
            dim=ctx.dim,
        )

        return (
            wp.to_torch(ctx.mapping),
            wp.to_torch(ctx.num_neighbors),
            wp.to_torch(ctx.outputs),
        )

    @staticmethod
    def backward(
        ctx,
        grad_mapping: torch.Tensor,
        grad_num_neighbors: torch.Tensor,
        grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
        # Map incoming torch grads to our output variable
        ctx.outputs.grad = wp.from_torch(grad_outputs, dtype=wp.float32)

        # Run the kernel in adjoint mode
        wp.launch(
            BallQuery.sparse_ball_query,
            inputs=[
                ctx.points2,
                ctx.mapping,
                ctx.num_neighbors,
            ],
            outputs=[
                ctx.outputs,
            ],
            adj_inputs=[ctx.points2.grad, ctx.mapping.grad, ctx.num_neighbors.grad],
            adj_outputs=[
                ctx.outputs.grad,
            ],
            dim=ctx.dim,
            adjoint=True,
        )

        # Return the gradients
        return (
            wp.to_torch(ctx.points1.grad).unsqueeze(0),
            wp.to_torch(ctx.points2.grad).unsqueeze(0),
            # None,
            # None,
            None,
            None,
            None,
        )


class BallQueryLayer(torch.nn.Module):
    """
    Torch layer for differentiable and accelerated Ball Query
    operation using Warp.
    Args:
        k (int): Number of neighbors.
        radius (float): Radius of influence.
        grid_size (int): Resolution of the hash grid. (Assumed to be uniform in all dimensions.)
    """

    def __init__(self, k: int, radius: float, grid_size: int = 32):
        super().__init__()
        wp.init()
        self.k = k
        self.radius = radius
        self.hash_grid = wp.HashGrid(grid_size, grid_size, grid_size)

    def forward(
        self, points1: torch.Tensor, points2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs ball query operation to find neighboring points within a specified radius.

        For each point in points1, finds up to k neighboring points from points2 that are
        within the specified radius. Uses a hash grid for efficient spatial queries.

        Args:
            points1: Tensor of shape (batch_size, num_points1, 3) containing query points
            points2: Tensor of shape (batch_size, num_points2, 3) containing points to search

        Returns:
            tuple containing:
                - mapping: Tensor containing indices of neighboring points
                - num_neighbors: Tensor containing the number of neighbors found for each query point
                - outputs: Tensor containing features or coordinates of the neighboring points
        """
        return BallQuery.apply(
            points1,
            points2,
            # lengths1,
            # lengths2,
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
    # lengths1 = torch.full((n,), p1, dtype=torch.int32).cuda()
    # lengths2 = torch.full((n,), p2, dtype=torch.int32).cuda()
    k = 256  # maximum number of neighbors
    radius = 0.1

    # Make ball query layer
    layer = BallQueryLayer(k, radius)

    # Make ball query
    with wp.ScopedTimer("ball query", active=True):
        mapping, num_neighbors, outputs = layer(
            points1,
            points2,
            # lengths1,
            # lengths2,
        )

    for i in range(20):
        p1 += 100
        p2 += 100
        points1 = torch.rand(n, p1, d, device="cuda", requires_grad=False)
        points2 = torch.rand(n, p2, d, device="cuda", requires_grad=False)
        # lengths1 = torch.full((n,), p1, dtype=torch.int32).cuda()
        # lengths2 = torch.full((n,), p2, dtype=torch.int32).cuda()

        mapping, num_neighbors, outputs = layer(
            points1,
            points2,
            # lengths1,
            # lengths2,
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
