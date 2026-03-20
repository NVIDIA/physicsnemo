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

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

# Warp is a required dependency in v2.0+.

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
        res = wp.mesh_query_point_sign_winding_number(mesh_id, points[tid], max_dist)
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
        raise ValueError("input_points must have last dimension of size 3")

    # Accept either flattened indices or face-triplet connectivity.
    if mesh_indices.ndim == 2:
        if mesh_indices.shape[-1] != 3:
            raise ValueError(
                "mesh_indices with 2 dimensions must have shape (n_faces, 3)"
            )
        mesh_indices = mesh_indices.reshape(-1)
    elif mesh_indices.ndim != 1:
        raise ValueError(
            "mesh_indices must be either 1D flattened indices or 2D (n_faces, 3)"
        )

    input_shape = input_points.shape

    # Flatten the input points:
    input_points = input_points.reshape(-1, 3)

    N = len(input_points)

    # Allocate output tensors with torch:
    sdf = torch.zeros(N, dtype=torch.float32, device=input_points.device)
    sdf_hit_point = torch.zeros(N, 3, dtype=torch.float32, device=input_points.device)

    wp_launch_device, wp_launch_stream = FunctionSpec.warp_launch_context(input_points)

    with wp.ScopedStream(wp_launch_stream):
        wp.init()

        # zero copy the vertices, indices, and input points to warp:
        wp_vertices = wp.from_torch(mesh_vertices.to(torch.float32), dtype=wp.vec3)
        wp_indices = wp.from_torch(
            mesh_indices.to(torch.int32).contiguous(), dtype=wp.int32
        )
        wp_input_points = wp.from_torch(input_points.to(torch.float32), dtype=wp.vec3)

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
        raise RuntimeError("mesh_vertices and input_points must be on the same device")

    if mesh_vertices.device != mesh_indices.device:
        raise RuntimeError("mesh_vertices and mesh_indices must be on the same device")

    N = input_points.shape[0]

    sdf_output = torch.empty(N, 1, device=input_points.device, dtype=input_points.dtype)
    sdf_hit_point_output = torch.empty(
        N, 3, device=input_points.device, dtype=input_points.dtype
    )

    return sdf_output, sdf_hit_point_output


class SignedDistanceField(FunctionSpec):
    """Compute the signed distance field (SDF) for a mesh and query points.

    The mesh must be a surface mesh consisting of triangles. This functional
    uses a Warp-backed implementation for accelerated execution.

    Parameters
    ----------
    mesh_vertices : torch.Tensor
        Coordinates of mesh vertices with shape ``(n_vertices, 3)``.
    mesh_indices : torch.Tensor
        Triangle connectivity indexing into ``mesh_vertices``. Expected shape is
        ``(n_faces, 3)`` or a flattened equivalent.
    input_points : torch.Tensor
        Query points at which to evaluate the signed distance, with shape
        ``(..., 3)``.
    max_dist : float, optional
        Maximum search distance for closest-point queries. Default is ``1e8``.
    use_sign_winding_number : bool, optional
        Whether to use winding-number-based sign computation. Default is
        ``False``. When ``False``, the mesh should be watertight for reliable
        signs.
    implementation : str, optional
        Explicit implementation name. Defaults to ``None``, which uses normal
        dispatch (currently the Warp implementation).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple ``(sdf, hit_points)`` where:
        - ``sdf`` contains signed distances at each query point.
        - ``hit_points`` contains the closest point on the mesh for each query.

    Examples
    --------
    >>> mesh_vertices = torch.tensor(
    ...     [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    ... )
    >>> mesh_indices = torch.tensor([(0, 1, 2)])
    >>> input_points = torch.tensor([(0.5, 0.5, 0.5)])
    >>> sdf, hit_points = signed_distance_field(
    ...     mesh_vertices, mesh_indices, input_points
    ... )
    """

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
        device = torch.device(device)

        # (label, n_rings, n_segments, n_query_points)
        cases = [
            ("small", 8, 20, 4096),
            ("medium", 16, 40, 16384),
            ("large", 32, 80, 65536),
        ]
        for label, n_rings, n_segments, num_points in cases:
            ### UV sphere vertex positions
            phi = torch.linspace(0, torch.pi, n_rings + 2, device=device)[1:-1]
            theta = torch.linspace(0, 2 * torch.pi, n_segments + 1, device=device)[:-1]
            phi_g, theta_g = torch.meshgrid(phi, theta, indexing="ij")

            sin_phi = phi_g.sin()
            ring_pts = torch.stack(
                [sin_phi * theta_g.cos(), sin_phi * theta_g.sin(), phi_g.cos()],
                dim=-1,
            ).reshape(-1, 3)

            mesh_vertices = torch.cat(
                [
                    torch.tensor([[0.0, 0.0, 1.0]], device=device),
                    ring_pts,
                    torch.tensor([[0.0, 0.0, -1.0]], device=device),
                ]
            )

            ### UV sphere triangle connectivity (vectorized)
            south_idx = n_rings * n_segments + 1
            j = torch.arange(n_segments, device=device)
            j_next = (j + 1) % n_segments

            north_fan = torch.stack([torch.zeros_like(j), 1 + j, 1 + j_next], dim=1)

            r = torch.arange(n_rings - 1, device=device).unsqueeze(1)  # (R, 1)
            base = 1 + r * n_segments
            p00, p01 = base + j, base + j_next
            p10, p11 = base + n_segments + j, base + n_segments + j_next
            body_tris = torch.stack(
                [
                    torch.stack([p00, p10, p11], dim=-1),
                    torch.stack([p00, p11, p01], dim=-1),
                ],
                dim=2,
            ).reshape(-1, 3)

            last = south_idx - n_segments
            south_fan = torch.stack(
                [last + j, torch.full_like(j, south_idx), last + j_next], dim=1
            )

            mesh_indices = (
                torch.cat([north_fan, body_tris, south_fan]).to(torch.int32).reshape(-1)
            )

            # Sample query points uniformly in [-1.5, 1.5]^3 (unit sphere + 50% padding).
            input_points = 3.0 * torch.rand(num_points, 3, device=device) - 1.5

            n_tris = 2 * n_rings * n_segments
            yield (
                f"{label}-uv-sphere-tris{n_tris}-query-points{num_points}",
                (mesh_vertices, mesh_indices, input_points),
                {"max_dist": 10.0, "use_sign_winding_number": False},
            )

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
