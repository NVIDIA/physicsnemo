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

import cupy as cp
import numpy as np
import warp as wp

wp.config.quiet = True


@wp.kernel
def _bvh_query_distance(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3f),
    max_dist: wp.float32,
    sdf: wp.array(dtype=wp.float32),
    sdf_hit_point: wp.array(dtype=wp.vec3f),
    sdf_hit_point_id: wp.array(dtype=wp.int32),
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
    sdf_hit_point_id[tid] = res.face


Array = np.ndarray | cp.ndarray


def signed_distance_field(
    mesh_vertices: Array,
    mesh_indices: Array,
    input_points: Array,
    max_dist: float = 1e8,
    include_hit_points: bool = False,
    include_hit_points_id: bool = False,
    use_sign_winding_number: bool = False,
    return_cupy: bool | None = None,
) -> Array | tuple[Array, ...]:
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
            the output. Here, "hit points" are the points on the mesh that are
            closest to the input points, and hence, are defining the SDF.
            Default is False.
        include_hit_points_id (bool, optional): Whether to include hit point
            IDs in the output. Default is False.
        use_sign_winding_number (bool, optional): Whether to use sign winding
            number method for SDF. Default is False. If False, your mesh should
            be watertight to obtain correct results.
        return_cupy (bool, optional): Whether to return a CuPy array. Default is
            None, which means the function will automatically determine the
            appropriate return type based on the input types.

    Returns:
    -------
    Returns:
        np.ndarray | cp.ndarray or tuple:
            - If both `include_hit_points` and `include_hit_points_id` are False
              (default), returns a 1D array of signed distances for each input
              point.
            - If `include_hit_points` is True, returns a tuple: (sdf,
              hit_points), where `hit_points` contains the closest mesh point
              for each input point.
            - If `include_hit_points_id` is True, returns a tuple: (sdf,
              hit_point_ids), where `hit_point_ids` contains the face index of
              the closest mesh face for each input point.
            - If both `include_hit_points` and `include_hit_points_id` are True,
              returns a tuple: (sdf, hit_points, hit_point_ids).
            - The returned array type (NumPy or CuPy) is determined by the
            `return_cupy` argument, or inferred from the input arrays.

    Example:
    -------
    >>> mesh_vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    >>> mesh_indices = np.array((0, 1, 2))
    >>> input_points = [(0.5, 0.5, 0.5)]
    >>> signed_distance_field(mesh_vertices, mesh_indices, input_points)
    array([0.5], dtype=float32)
    """
    if return_cupy is None:
        return_cupy = any(
            isinstance(arr, cp.ndarray)
            for arr in (mesh_vertices, mesh_indices, input_points)
        )

    wp.init()

    if isinstance(mesh_vertices, cp.ndarray):
        device = mesh_vertices.device
        wp_device = f"cuda:{device.id}"
    else:
        wp_device = wp.get_device()

    with wp.ScopedDevice(wp_device):
        mesh = wp.Mesh(
            points=wp.array(mesh_vertices, dtype=wp.vec3f, device=wp_device),
            indices=wp.array(mesh_indices, dtype=wp.int32, device=wp_device),
        )

        warp_input_points = wp.array(input_points, dtype=wp.vec3f, device=wp_device)

        N = len(warp_input_points)

        sdf = wp.empty(shape=(N,), dtype=wp.float32, device=wp_device)
        sdf_hit_point = wp.empty(shape=(N,), dtype=wp.vec3f, device=wp_device)
        sdf_hit_point_id = wp.empty(shape=(N,), dtype=wp.int32, device=wp_device)

        wp.launch(
            kernel=_bvh_query_distance,
            dim=N,
            inputs=[
                mesh.id,
                warp_input_points,
                max_dist,
                sdf,
                sdf_hit_point,
                sdf_hit_point_id,
                use_sign_winding_number,
            ],
            device=wp_device,
        )

        def convert(array: wp.array) -> np.ndarray | cp.ndarray:
            """Converts a Warp array to CuPy/NumPy based on the `return_cupy` flag."""
            if return_cupy:
                return cp.asarray(array)
            else:
                return array.numpy()

        arrays_to_return: list[np.ndarray | cp.ndarray] = [convert(sdf)]

        if include_hit_points:
            arrays_to_return.append(convert(sdf_hit_point))
        if include_hit_points_id:
            arrays_to_return.append(convert(sdf_hit_point_id))

        return (
            arrays_to_return[0]
            if len(arrays_to_return) == 1
            else tuple(arrays_to_return)
        )
