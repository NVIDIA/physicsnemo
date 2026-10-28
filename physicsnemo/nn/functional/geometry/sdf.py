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
from jaxtyping import Float, Int

from physicsnemo.core.function_spec import FunctionSpec

# Warp is a required dependency in v2.0+.

wp.config.log_level = wp.LOG_WARNING


@wp.kernel
def _bvh_query_distance(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3f),
    max_dist: wp.float32,
    sdf: wp.array(dtype=wp.float32),
    sdf_hit_point: wp.array(dtype=wp.vec3f),
    sdf_hit_face: wp.array(dtype=wp.int64),
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
        sdf_hit_face (wp.array): An array to store the hit face indices.
        use_sign_winding_number (bool): Flag to use sign_winding_number method for SDF.

    Queries that find no triangle within `max_dist` leave their output slots
    untouched; the host wrapper pre-fills them (NaN distance/hit point, -1 face).

    Returns:
        None
    """
    tid = wp.tid()

    if use_sign_winding_number:
        res = wp.mesh_query_point_sign_winding_number(mesh_id, points[tid], max_dist)
    else:
        res = wp.mesh_query_point_sign_normal(mesh_id, points[tid], max_dist)

    if not res.result:
        return

    mesh = wp.mesh_get(mesh_id)

    p0 = mesh.points[mesh.indices[3 * res.face + 0]]
    p1 = mesh.points[mesh.indices[3 * res.face + 1]]
    p2 = mesh.points[mesh.indices[3 * res.face + 2]]

    p_closest = res.u * p0 + res.v * p1 + (1.0 - res.u - res.v) * p2

    sdf[tid] = res.sign * wp.abs(wp.length(points[tid] - p_closest))
    sdf_hit_point[tid] = p_closest
    sdf_hit_face[tid] = wp.int64(res.face)


@torch.library.custom_op("physicsnemo::signed_distance_field", mutates_args=())
def signed_distance_field_impl(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    input_points: torch.Tensor,
    max_dist: float = 1e8,
    use_sign_winding_number: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the signed distance field (SDF) for a given mesh and input points.

    The mesh must be a surface mesh consisting of all triangles. Uses NVIDIA
    Warp for GPU acceleration.

    Parameters:
    ----------
        mesh_vertices (torch.Tensor): Coordinates of the vertices of the mesh;
            shape: (n_vertices, 3)
        mesh_indices (torch.Tensor): Indices corresponding to the faces of the
            mesh; shape: (n_faces, 3) or flattened (n_faces * 3,)
        input_points (torch.Tensor): Coordinates of the points for which to
            compute the SDF; shape: (..., 3)
        max_dist (float, optional): Maximum distance within which
            to search for the closest point on the mesh. Default is 1e8.
        use_sign_winding_number (bool, optional): Whether to use sign winding
            number method for SDF. Default is False. If False, your mesh should
            be watertight to obtain correct results.

    Returns:
    -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] of:
            - signed distance to the mesh, per input point; shape (...,)
            - hit point, per input point; shape (..., 3). "hit points" are the
            points on the mesh that are closest to the input points, and
            hence, are defining the SDF.
            - hit face, per input point; shape (...,), int64 index of the
            triangle containing the hit point.

        Queries with no triangle within ``max_dist`` return NaN for the
        distance and hit point and -1 for the hit face.

    Note:
        On CUDA inputs the mesh-index range validation is performed with a
        device-side assert so the op stays free of host synchronization; an
        out-of-range index surfaces as a CUDA assert at the next
        synchronization point rather than an immediate ``ValueError``.

        The signed distance is differentiable with respect to
        ``mesh_vertices`` and ``input_points`` (see
        :func:`_backward_signed_distance_field`). The hit points and hit faces
        are returned without gradients.
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
    if mesh_indices.numel() == 0 or mesh_indices.numel() % 3 != 0:
        raise ValueError("mesh_indices must contain complete triangle faces")

    n_vertices = mesh_vertices.shape[0]
    if mesh_indices.is_cuda:
        # Device-side assert: keeps the op free of host synchronization. An
        # out-of-range index fails at the next synchronization point.
        torch._assert_async(
            ((mesh_indices >= 0) & (mesh_indices < n_vertices)).all(),
            "mesh_indices must index existing vertices",
        )
    else:
        min_index = int(mesh_indices.min().item())
        max_index = int(mesh_indices.max().item())
        if min_index < 0 or max_index >= n_vertices:
            raise ValueError(
                "mesh_indices must index existing vertices; "
                f"got range [{min_index}, {max_index}] for "
                f"{n_vertices} vertices"
            )

    input_shape = input_points.shape
    out_dtype = input_points.dtype

    # Flatten the input points:
    input_points = input_points.reshape(-1, 3)

    N = len(input_points)

    # Allocate output tensors with torch, pre-filled with the miss sentinels
    # (NaN distance/hit point, -1 face) so the kernel only writes hits:
    sdf = torch.full(
        (N,), float("nan"), dtype=torch.float32, device=input_points.device
    )
    sdf_hit_point = torch.full(
        (N, 3), float("nan"), dtype=torch.float32, device=input_points.device
    )
    sdf_hit_face = torch.full((N,), -1, dtype=torch.int64, device=input_points.device)

    if N == 0:
        return (
            sdf.reshape(input_shape[:-1]).to(out_dtype),
            sdf_hit_point.reshape(input_shape).to(out_dtype),
            sdf_hit_face.reshape(input_shape[:-1]),
        )

    wp_launch_device, wp_launch_stream = FunctionSpec.warp_launch_context(input_points)

    with FunctionSpec.warp_stream_scope(wp_launch_stream):
        wp.init()

        # zero copy the vertices, indices, and input points to warp. Detach
        # the float inputs: from_torch would otherwise allocate a .grad buffer
        # on any input that requires grad. Gradients come from the registered
        # backward instead. Warp's vec3 needs contiguous rows, which a strided
        # view (e.g. a transpose) does not have.
        wp_vertices = wp.from_torch(
            mesh_vertices.detach().to(torch.float32).contiguous(), dtype=wp.vec3
        )
        wp_indices = wp.from_torch(
            mesh_indices.to(torch.int32).contiguous(), dtype=wp.int32
        )
        wp_input_points = wp.from_torch(
            input_points.detach().to(torch.float32).contiguous(), dtype=wp.vec3
        )

        # Convert output points:
        wp_sdf = wp.from_torch(sdf, dtype=wp.float32)
        wp_sdf_hit_point = wp.from_torch(sdf_hit_point, dtype=wp.vec3f)
        wp_sdf_hit_face = wp.from_torch(sdf_hit_face, dtype=wp.int64)

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
                wp_sdf_hit_face,
                use_sign_winding_number,
            ],
            device=wp_launch_device,
            stream=wp_launch_stream,
        )

    # Unflatten the output to be like the input:
    sdf = sdf.reshape(input_shape[:-1])
    sdf_hit_point = sdf_hit_point.reshape(input_shape)
    sdf_hit_face = sdf_hit_face.reshape(input_shape[:-1])

    return sdf.to(out_dtype), sdf_hit_point.to(out_dtype), sdf_hit_face


@signed_distance_field_impl.register_fake
def signed_distance_field_impl_fake(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    input_points: torch.Tensor,
    max_dist: float = 1e8,
    use_sign_winding_number: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mesh_vertices.device != input_points.device:
        raise RuntimeError("mesh_vertices and input_points must be on the same device")

    if mesh_vertices.device != mesh_indices.device:
        raise RuntimeError("mesh_vertices and mesh_indices must be on the same device")

    sdf_output = torch.empty(
        input_points.shape[:-1], device=input_points.device, dtype=input_points.dtype
    )
    sdf_hit_point_output = torch.empty(
        input_points.shape, device=input_points.device, dtype=input_points.dtype
    )
    sdf_hit_face_output = torch.empty(
        input_points.shape[:-1], device=input_points.device, dtype=torch.int64
    )

    return sdf_output, sdf_hit_point_output, sdf_hit_face_output


def _setup_signed_distance_field_context(
    ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: tuple
) -> None:
    """Save the tensors the backward pass needs.

    The hit points and hit faces are marked non-differentiable: only the
    signed distance carries gradients. The backward pass recomputes the
    closest point from the hit face, so the hit points are not saved.
    """
    mesh_vertices, mesh_indices, input_points, _max_dist, _use_winding = inputs
    sdf, hit_points, hit_faces = output
    ctx.mark_non_differentiable(hit_points, hit_faces)
    ctx.save_for_backward(mesh_vertices, mesh_indices, input_points, sdf, hit_faces)


def _closest_point_weights(
    query: Float[torch.Tensor, "n_queries 3"],
    tri: Float[torch.Tensor, "n_queries 3 3"],
) -> Float[torch.Tensor, "n_queries 3"]:
    """Barycentric weights of the point of each triangle closest to its query.

    Follows Ericson, *Real-Time Collision Detection* (2005), Sec. 5.1.5: the
    Voronoi region of the query (a corner, an edge, or the face) sets which
    weights are zero, so a query nearest a corner or an edge gets exact
    zeros. The face-region weights are ratios of Gram determinants, which
    cancel on thin faces, so call this in float64.
    """
    a, b, c = tri.unbind(dim=1)
    ab = b - a
    ac = c - a
    d1 = (ab * (query - a)).sum(-1)
    d2 = (ac * (query - a)).sum(-1)
    d3 = (ab * (query - b)).sum(-1)
    d4 = (ac * (query - b)).sum(-1)
    d5 = (ab * (query - c)).sum(-1)
    d6 = (ac * (query - c)).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    tiny = torch.finfo(query.dtype).tiny
    zero = torch.zeros_like(d1)
    one = torch.ones_like(d1)

    # Every denominator is a squared length or area, so it is >= 0.
    denom = (va + vb + vc).clamp(min=tiny)
    t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp(min=tiny)
    t_ac = d2 / (d2 - d6).clamp(min=tiny)
    t_ab = d1 / (d1 - d3).clamp(min=tiny)

    # The face interior is the default. Ericson tests corner A, corner B,
    # edge AB, corner C, edge AC, then edge BC, and returns at the first
    # match, so the regions are applied here in reverse and the first test
    # wins.
    weights = torch.stack([va / denom, vb / denom, vc / denom], dim=-1)
    regions = (
        ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), (zero, 1.0 - t_bc, t_bc)),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), (1.0 - t_ac, zero, t_ac)),
        ((d6 >= 0) & (d5 <= d6), (zero, zero, one)),
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), (1.0 - t_ab, t_ab, zero)),
        ((d3 >= 0) & (d4 <= d3), (zero, one, zero)),
        ((d1 <= 0) & (d2 <= 0), (one, zero, zero)),
    )
    for in_region, region_weights in regions:
        weights = torch.where(
            in_region.unsqueeze(-1), torch.stack(region_weights, dim=-1), weights
        )
    return weights


def _backward_signed_distance_field(
    ctx: torch.autograd.function.FunctionCtx,
    grad_sdf: torch.Tensor | None,
    grad_hit_points: torch.Tensor | None,
    grad_hit_faces: torch.Tensor | None,
) -> tuple[torch.Tensor | None, None, torch.Tensor | None, None, None]:
    r"""Backward pass of :func:`signed_distance_field_impl`.

    Let :math:`c = \sum_k \lambda_k p_k` be the closest point to a query
    :math:`q` on its nearest face, with vertices :math:`p_k` and barycentric
    weights :math:`\lambda_k`, and let :math:`s = \pm 1` be the sign. The
    closest point minimizes the distance, so by the envelope theorem it can
    be held fixed when differentiating:

    .. math::

        \frac{\partial\,\mathrm{sdf}}{\partial q} = s\,\frac{q - c}{\lVert q - c \rVert},
        \qquad
        \frac{\partial\,\mathrm{sdf}}{\partial p_k}
            = -\lambda_k\,\frac{\partial\,\mathrm{sdf}}{\partial q}.

    The closest point and its weights are recomputed in float64 on the hit
    face, from the query and the vertices. The ``hit_points`` output is
    rounded to the query dtype, and for float16 queries that rounding alone
    can turn :math:`q - c` by 45 degrees.

    The forward pass computes the sign in float32, so the sign is unreliable
    for a query within a few float32 ulps of the surface. Such a query uses
    the unit normal of its face, oriented by the face's winding, which is the
    gradient of the signed distance on a face of an outward-oriented
    surface. Queries beyond ``max_dist``, which have no hit face, contribute
    no gradient.

    The hit points and hit faces are non-differentiable outputs, so
    ``grad_hit_points`` and ``grad_hit_faces`` are ignored.
    """
    mesh_vertices, mesh_indices, input_points, sdf, hit_faces = ctx.saved_tensors
    need_vertices = ctx.needs_input_grad[0]
    need_points = ctx.needs_input_grad[2]
    if grad_sdf is None or not (need_vertices or need_points):
        return None, None, None, None, None

    dtype = torch.float64
    tiny = torch.finfo(dtype).tiny

    query = input_points.reshape(-1, 3).to(dtype)
    face = hit_faces.reshape(-1)
    hit = face >= 0
    # A miss carries a NaN distance, so its upstream gradient is zeroed.
    grad = torch.where(hit, grad_sdf.reshape(-1).to(dtype), 0.0)

    corners = mesh_indices.reshape(-1, 3).to(torch.long)[face.clamp(min=0)]
    tri = mesh_vertices.to(dtype)[corners]  # (n_queries, 3, 3)
    weights = _closest_point_weights(query, tri)
    closest = (weights.unsqueeze(-1) * tri).sum(dim=1)

    # The face's winding normal.
    normal = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=tiny)

    offset = query - closest
    distance = offset.norm(dim=-1)
    scale = torch.maximum(query.abs().amax(dim=-1), closest.abs().amax(dim=-1))
    near_surface = distance <= 8.0 * torch.finfo(torch.float32).eps * scale
    sign = torch.where(torch.signbit(sdf.reshape(-1)), -1.0, 1.0).to(dtype)
    direction = torch.where(
        near_surface.unsqueeze(-1),
        normal,
        (sign / distance.clamp(min=tiny)).unsqueeze(-1) * offset,
    )
    grad_query = torch.where(hit.unsqueeze(-1), grad.unsqueeze(-1) * direction, 0.0)

    grad_points = None
    if need_points:
        grad_points = grad_query.reshape(input_points.shape).to(input_points.dtype)

    grad_vertices = None
    if need_vertices:
        contributions = -weights.unsqueeze(-1) * grad_query.unsqueeze(1)
        # A miss's weights come from an arbitrary face; zero its contribution.
        contributions = torch.where(hit.view(-1, 1, 1), contributions, 0.0)
        grad_vertices = torch.zeros(
            mesh_vertices.shape, dtype=dtype, device=mesh_vertices.device
        )
        grad_vertices.index_add_(0, corners.reshape(-1), contributions.reshape(-1, 3))
        grad_vertices = grad_vertices.to(mesh_vertices.dtype)

    return grad_vertices, None, grad_points, None, None


signed_distance_field_impl.register_autograd(
    _backward_signed_distance_field,
    setup_context=_setup_signed_distance_field_context,
)


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
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple ``(sdf, hit_points, hit_faces)`` where:
        - ``sdf`` contains signed distances at each query point.
        - ``hit_points`` contains the closest point on the mesh for each query.
        - ``hit_faces`` contains the int64 index of the triangle holding each
          closest point.

        Queries with no triangle within ``max_dist`` return ``NaN`` for the
        distance and hit point and ``-1`` for the hit face.

    Notes
    -----
    ``sdf`` is differentiable with respect to ``mesh_vertices`` and
    ``input_points``. The closest point is held fixed (envelope theorem), so
    the gradient with respect to a query is the unit vector from its hit
    point, times the sign, and a vertex receives that gradient times minus
    its barycentric weight in the hit point. ``hit_points`` and ``hit_faces``
    carry no gradients. Queries beyond ``max_dist`` contribute no gradient.

    Examples
    --------
    >>> mesh_vertices = torch.tensor(
    ...     [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    ... )
    >>> mesh_indices = torch.tensor([(0, 1, 2)])
    >>> input_points = torch.tensor([(0.5, 0.5, 0.5)])
    >>> sdf, hit_points, hit_faces = signed_distance_field(
    ...     mesh_vertices, mesh_indices, input_points
    ... )
    """

    _BENCHMARK_CASES = (
        ("small", 8, 20, 4096),
        ("medium", 16, 40, 16384),
        ("large", 32, 80, 65536),
    )

    @FunctionSpec.register(
        name="warp", required_imports=("warp>=0.6.0",), rank=0, baseline=True
    )
    def warp_forward(
        mesh_vertices: Float[torch.Tensor, "num_vertices 3"],
        mesh_indices: torch.Tensor,
        input_points: Float[torch.Tensor, "... 3"],
        max_dist: float = 1e8,
        use_sign_winding_number: bool = False,
    ) -> tuple[
        Float[torch.Tensor, "..."],
        Float[torch.Tensor, "... 3"],
        Int[torch.Tensor, "..."],
    ]:
        return signed_distance_field_impl(
            mesh_vertices,
            mesh_indices,
            input_points,
            max_dist=max_dist,
            use_sign_winding_number=use_sign_winding_number,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        for label, n_rings, n_segments, num_points in cls._BENCHMARK_CASES:
            # Build UV-sphere vertex positions.
            phi = torch.linspace(0, torch.pi, n_rings + 2, device=device)[1:-1]
            theta = torch.linspace(0, 2 * torch.pi, n_segments + 1, device=device)[:-1]
            phi_g, theta_g = torch.meshgrid(phi, theta, indexing="ij")

            sin_phi = phi_g.sin()
            ring_points = torch.stack(
                [sin_phi * theta_g.cos(), sin_phi * theta_g.sin(), phi_g.cos()],
                dim=-1,
            ).reshape(-1, 3)

            mesh_vertices = torch.cat(
                [
                    torch.tensor([[0.0, 0.0, 1.0]], device=device),
                    ring_points,
                    torch.tensor([[0.0, 0.0, -1.0]], device=device),
                ]
            ).to(torch.float32)

            # Build UV-sphere triangle connectivity (vectorized).
            south_idx = n_rings * n_segments + 1
            j = torch.arange(n_segments, device=device)
            j_next = (j + 1) % n_segments

            north_fan = torch.stack([torch.zeros_like(j), 1 + j, 1 + j_next], dim=1)

            r = torch.arange(n_rings - 1, device=device).unsqueeze(1)
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

            # Sample query points in a padded box around the unit sphere.
            input_points = 3.0 * torch.rand(num_points, 3, device=device) - 1.5

            yield (
                f"{label}-uv-sphere-tris{2 * n_rings * n_segments}-query-points{num_points}",
                (mesh_vertices, mesh_indices, input_points),
                {"max_dist": 10.0, "use_sign_winding_number": False},
            )

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield the forward cases with vertices and queries requiring grad."""
        for label, args, kwargs in cls.make_inputs_forward(device=device):
            mesh_vertices, mesh_indices, input_points = args
            yield (
                label,
                (
                    mesh_vertices.requires_grad_(True),
                    mesh_indices,
                    input_points.requires_grad_(True),
                ),
                kwargs,
            )


signed_distance_field = SignedDistanceField.make_function("signed_distance_field")


__all__ = ["SignedDistanceField", "signed_distance_field"]
