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

import importlib

import torch

from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.core.version_check import check_version_spec

WARP_AVAILABLE = check_version_spec("warp", "0.6.0", hard_fail=False)

if WARP_AVAILABLE:
    wp = importlib.import_module("warp")
    wp.config.quiet = True

    @wp.kernel
    def _bvh_query_distance(
        mesh_id: wp.uint64,
        points: wp.array(dtype=wp.vec3f),
        max_dist: wp.float32,
        sdf: wp.array(dtype=wp.float32),
        sdf_hit_point: wp.array(dtype=wp.vec3f),
        use_sign_winding_number: bool = False,
    ):
        """
        Computes the signed distance from each point in the given array `points`
        to the mesh represented by `mesh`,within the maximum distance `max_dist`,
        and stores the result in the array `sdf`.

        Parameters:
            mesh (wp.uint64): The identifier of the mesh.
            points (wp.array): An array of 3D points for which to compute the
                signed distance.
            max_dist (wp.float32): The maximum distance within which to search
                for the closest point on the mesh.
            sdf (wp.array): An array to store the computed signed distances.
            sdf_hit_point (wp.array): An array to store the computed hit points.
            sdf_hit_point_id (wp.array): An array to store the computed hit point ids.
            use_sign_winding_number (bool): Flag to use sign_winding_number method for SDF.

        Returns:
            None
        """
        tid = wp.tid()

        if use_sign_winding_number:
            res = wp.mesh_query_point_sign_winding_number(
                mesh_id, points[tid], max_dist
            )
        else:
            res = wp.mesh_query_point_sign_normal(mesh_id, points[tid], max_dist)

        mesh = wp.mesh_get(mesh_id)

        p0 = mesh.points[mesh.indices[3 * res.face + 0]]
        p1 = mesh.points[mesh.indices[3 * res.face + 1]]
        p2 = mesh.points[mesh.indices[3 * res.face + 2]]

        p_closest = res.u * p0 + res.v * p1 + (1.0 - res.u - res.v) * p2

        sdf[tid] = res.sign * wp.abs(wp.length(points[tid] - p_closest))
        sdf_hit_point[tid] = p_closest

    @torch.library.custom_op("physicsnemo::signed_distance_field", mutates_args=())
    def signed_distance_field_impl(
        mesh_vertices: torch.Tensor,
        mesh_indices: torch.Tensor,
        input_points: torch.Tensor,
        max_dist: float = 1e8,
        use_sign_winding_number: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the signed distance field (SDF) for a given mesh and input points.

        The mesh must be a surface mesh consisting of all triangles. Uses NVIDIA
        Warp for GPU acceleration.

        Parameters:
        ----------
            mesh_vertices (np.ndarray): Coordinates of the vertices of the mesh;
                shape: (n_vertices, 3)
            mesh_indices (np.ndarray): Indices corresponding to the faces of the
                mesh; shape: (n_faces, 3)
            input_points (np.ndarray): Coordinates of the points for which to
                compute the SDF; shape: (n_points, 3)
            max_dist (float, optional): Maximum distance within which
                to search for the closest point on the mesh. Default is 1e8.
            include_hit_points (bool, optional): Whether to include hit points in
                the output. Here,
            use_sign_winding_number (bool, optional): Whether to use sign winding
                number method for SDF. Default is False. If False, your mesh should
                be watertight to obtain correct results.
            return_cupy (bool, optional): Whether to return a CuPy array. Default is
                None, which means the function will automatically determine the
                appropriate return type based on the input types.

        Returns:
        -------
        Returns:
            tuple[torch.Tensor, torch.Tensor] of:
                - signed distance to the mesh, per input point
                - hit point, per input point. "hit points" are the points on the
                mesh that are closest to the input points, and hence, are
                defining the SDF.

        Example:
        -------
        >>> mesh_vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        >>> mesh_indices = torch.tensor((0, 1, 2))
        >>> input_points = torch.tensor((0.5, 0.5, 0.5))
        >>> signed_distance_field(mesh_vertices, mesh_indices, input_points)
        (tensor([0.5]), tensor([0.5, 0.5, 0.5]))
        """

        if input_points.shape[-1] != 3:
            raise ValueError(
                "Input points must be a tensor with last dimension of size 3"
            )

        input_shape = input_points.shape

        # Flatten the input points:
        input_points = input_points.reshape(-1, 3)

        N = len(input_points)

        # Allocate output tensors with torch:
        sdf = torch.zeros(N, dtype=torch.float32, device=input_points.device)
        sdf_hit_point = torch.zeros(
            N, 3, dtype=torch.float32, device=input_points.device
        )

        wp_launch_device, wp_launch_stream = FunctionSpec.warp_launch_context(
            input_points
        )

        with wp.ScopedStream(wp_launch_stream):
            wp.init()

            # zero copy the vertices, indices, and input points to warp:
            wp_vertices = wp.from_torch(mesh_vertices.to(torch.float32), dtype=wp.vec3)
            wp_indices = wp.from_torch(mesh_indices.to(torch.int32), dtype=wp.int32)
            wp_input_points = wp.from_torch(
                input_points.to(torch.float32), dtype=wp.vec3
            )

            # Convert output points:
            wp_sdf = wp.from_torch(sdf, dtype=wp.float32)
            wp_sdf_hit_point = wp.from_torch(sdf_hit_point, dtype=wp.vec3f)

            mesh = wp.Mesh(
                points=wp_vertices,
                indices=wp_indices,
                support_winding_number=use_sign_winding_number,
            )

            wp.launch(
                kernel=_bvh_query_distance,
                dim=N,
                inputs=[
                    mesh.id,
                    wp_input_points,
                    max_dist,
                    wp_sdf,
                    wp_sdf_hit_point,
                    use_sign_winding_number,
                ],
                device=wp_launch_device,
                stream=wp_launch_stream,
            )

        # Unflatten the output to be like the input:
        sdf = sdf.reshape(input_shape[:-1])
        sdf_hit_point = sdf_hit_point.reshape(input_shape)

        return sdf.to(input_points.dtype), sdf_hit_point.to(input_points.dtype)

    @signed_distance_field_impl.register_fake
    def signed_distance_field_impl_fake(
        mesh_vertices: torch.Tensor,
        mesh_indices: torch.Tensor,
        input_points: torch.Tensor,
        max_dist: float = 1e8,
        use_sign_winding_number: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mesh_vertices.device != input_points.device:
            raise RuntimeError(
                "mesh_vertices and input_points must be on the same device"
            )

        if mesh_vertices.device != mesh_indices.device:
            raise RuntimeError(
                "mesh_vertices and mesh_indices must be on the same device"
            )

        N = input_points.shape[0]

        sdf_output = torch.empty(
            N, 1, device=input_points.device, dtype=input_points.dtype
        )
        sdf_hit_point_output = torch.empty(
            N, 3, device=input_points.device, dtype=input_points.dtype
        )

        return sdf_output, sdf_hit_point_output

else:

    def signed_distance_field_impl(*args, **kwargs) -> None:
        # TODO: probably a better way to handle functionals with single
        # implementations like this. For now this works though.
        raise RuntimeError(
            "SDF ERROR - Warp is not available, please install warp with `pip install warp-lang` or see https://nvidia.github.io/warp/installation.html"
        )


class SignedDistanceField(FunctionSpec):
    """Compute signed distance to a mesh with a Warp-backed implementation."""

    @FunctionSpec.register(
        name="warp", required_imports=("warp>=0.6.0",), rank=0, baseline=True
    )
    def warp_forward(
        mesh_vertices: torch.Tensor,
        mesh_indices: torch.Tensor,
        input_points: torch.Tensor,
        max_dist: float = 1e8,
        use_sign_winding_number: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return signed_distance_field_impl(
            mesh_vertices,
            mesh_indices,
            input_points,
            max_dist=max_dist,
            use_sign_winding_number=use_sign_winding_number,
        )

    @classmethod
    def make_inputs(cls, device: torch.device | str = "cpu"):
        # TODO(ASV): Populate benchmark inputs in a follow-up PR.
        raise NotImplementedError

    @classmethod
    def compare(
        cls,
        output: tuple[torch.Tensor, torch.Tensor],
        reference: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        # TODO(ASV): Populate output comparison in a follow-up PR.
        raise NotImplementedError


signed_distance_field = SignedDistanceField.make_function("signed_distance_field")


__all__ = ["SignedDistanceField", "signed_distance_field"]
