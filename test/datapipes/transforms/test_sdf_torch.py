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

"""Tests for the Warp-free, mesh-BVH-backed signed distance field."""

import math

import pytest
import torch

from physicsnemo.datapipes.transforms._sdf_torch import signed_distance_field_mesh


# Build a simple tetrahedron surface mesh as four triangles (matches the
# reference fixture used by the Warp SDF test for cross-implementation parity).
def _tetrahedron_vertices() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


def _uv_sphere(n_rings: int = 40, n_segments: int = 80):
    """Build a UV-sphere triangle mesh (unit radius) for analytic SDF checks."""
    phi = torch.linspace(0, math.pi, n_rings + 2)[1:-1]
    theta = torch.linspace(0, 2 * math.pi, n_segments + 1)[:-1]
    phi_g, theta_g = torch.meshgrid(phi, theta, indexing="ij")
    sin_phi = phi_g.sin()
    ring = torch.stack(
        [sin_phi * theta_g.cos(), sin_phi * theta_g.sin(), phi_g.cos()], dim=-1
    ).reshape(-1, 3)
    vertices = torch.cat(
        [torch.tensor([[0.0, 0.0, 1.0]]), ring, torch.tensor([[0.0, 0.0, -1.0]])]
    ).float()

    south = n_rings * n_segments + 1
    j = torch.arange(n_segments)
    j_next = (j + 1) % n_segments
    north = torch.stack([torch.zeros_like(j), 1 + j, 1 + j_next], dim=1)
    r = torch.arange(n_rings - 1).unsqueeze(1)
    base = 1 + r * n_segments
    p00, p01 = base + j, base + j_next
    p10, p11 = base + n_segments + j, base + n_segments + j_next
    body = torch.stack(
        [torch.stack([p00, p10, p11], -1), torch.stack([p00, p11, p01], -1)], dim=2
    ).reshape(-1, 3)
    last = south - n_segments
    south_fan = torch.stack([last + j, torch.full_like(j, south), last + j_next], dim=1)
    faces = torch.cat([north, body, south_fan]).to(torch.int32).reshape(-1)
    return vertices, faces


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("use_winding", [False, True])
def test_sdf_tetrahedron_reference(dtype, use_winding, device):
    """Match the deterministic tetrahedron values from the Warp SDF test."""
    device = torch.device(device)
    mesh_vertices = _tetrahedron_vertices().to(device=device, dtype=dtype)
    mesh_indices = torch.arange(12, device=device, dtype=torch.int32)
    query_points = torch.tensor(
        [[1.0, 1.0, 1.0], [0.05, 0.1, 0.1]], device=device, dtype=dtype
    )

    sdf_out, hit_points = signed_distance_field_mesh(
        mesh_vertices,
        mesh_indices,
        query_points,
        use_sign_winding_number=use_winding,
    )

    torch.testing.assert_close(
        sdf_out,
        torch.tensor([1.1547, -0.05], device=device, dtype=dtype),
        atol=1e-4,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        hit_points,
        torch.tensor(
            [[0.33333322, 0.33333334, 0.3333334], [0.0, 0.10, 0.10]],
            device=device,
            dtype=dtype,
        ),
        atol=1e-4,
        rtol=1e-4,
    )


def test_sdf_index_layout_compatibility(device):
    """Flattened and (n_faces, 3) connectivity must give identical results."""
    device = torch.device(device)
    mesh_vertices = _tetrahedron_vertices().to(device=device, dtype=torch.float32)
    mesh_indices_flat = torch.arange(12, device=device, dtype=torch.int32)
    mesh_indices_faces = mesh_indices_flat.reshape(-1, 3)
    query_points = torch.tensor([[0.1, 0.2, 0.3]], device=device, dtype=torch.float32)

    sdf_flat, hit_flat = signed_distance_field_mesh(
        mesh_vertices, mesh_indices_flat, query_points
    )
    sdf_faces, hit_faces = signed_distance_field_mesh(
        mesh_vertices, mesh_indices_faces, query_points
    )
    torch.testing.assert_close(sdf_flat, sdf_faces)
    torch.testing.assert_close(hit_flat, hit_faces)


@pytest.mark.parametrize("use_winding", [False, True])
def test_sdf_sphere_analytic(use_winding, device):
    """SDF of a tessellated unit sphere matches the analytic ``|r| - 1``."""
    device = torch.device(device)
    vertices, faces = _uv_sphere()
    vertices = vertices.to(device)
    faces = faces.to(device)

    torch.manual_seed(0)
    query = (torch.rand(4096, 3, device=device) * 3.0 - 1.5).float()
    radius = query.norm(dim=-1)
    gt = radius - 1.0

    sdf_out, hit = signed_distance_field_mesh(
        vertices, faces, query, use_sign_winding_number=use_winding
    )

    # The error is dominated by the polygonal approximation of the sphere, not
    # the algorithm; a coarse tolerance captures that the magnitude is correct.
    torch.testing.assert_close(sdf_out, gt, atol=5e-3, rtol=0.0)

    # Sign must agree with the analytic field away from the surface.
    far = gt.abs() > 0.05
    assert torch.all(sdf_out[far].sign() == gt[far].sign())

    # Hit points lie (approximately) on the unit sphere.
    torch.testing.assert_close(
        hit.norm(dim=-1), torch.ones_like(radius), atol=5e-3, rtol=0.0
    )


def test_sdf_preserves_input_shape(device):
    """Output SDF/hit-point shapes follow the (possibly batched) query shape."""
    device = torch.device(device)
    vertices = _tetrahedron_vertices().to(device=device, dtype=torch.float32)
    faces = torch.arange(12, device=device, dtype=torch.int32)
    query = torch.rand(4, 5, 3, device=device)

    sdf_out, hit = signed_distance_field_mesh(vertices, faces, query)
    assert sdf_out.shape == (4, 5)
    assert hit.shape == (4, 5, 3)


def test_sdf_error_handling(device):
    """Input validation mirrors the Warp implementation's contract."""
    device = torch.device(device)
    vertices = _tetrahedron_vertices().to(device=device, dtype=torch.float32)
    faces = torch.arange(12, device=device, dtype=torch.int32)
    query = torch.tensor([[0.1, 0.2, 0.3]], device=device, dtype=torch.float32)

    bad_queries = torch.randn(4, 2, device=device)
    with pytest.raises(ValueError, match="last dimension of size 3"):
        signed_distance_field_mesh(vertices, faces, bad_queries)

    bad_connectivity_shape = torch.zeros(4, 4, device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match=r"shape \(n_faces, 3\)"):
        signed_distance_field_mesh(vertices, bad_connectivity_shape, query)

    bad_connectivity_rank = torch.zeros(1, 2, 3, device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="1D flattened indices or 2D"):
        signed_distance_field_mesh(vertices, bad_connectivity_rank, query)


# ---------------------------------------------------------------------------
# Triton GPU kernel parity (CUDA-only): the kernel is the fast path, the
# pure-PyTorch bounded-stack DFS is the reference oracle.
# ---------------------------------------------------------------------------

_CUDA = torch.cuda.is_available()


def _triton_available() -> bool:
    if not _CUDA:
        return False
    from physicsnemo.datapipes.transforms import _sdf_triton

    return _sdf_triton.available()


@pytest.mark.skipif(not _CUDA, reason="CUDA required for the Triton SDF kernel")
def test_sdf_triton_nearest_matches_torch_reference():
    """The Triton nearest-triangle kernel matches the torch DFS reference.

    Distances are unique, so they must agree tightly. The winning face / closest
    point can differ on exact ties, so those are compared via the query-to-point
    distance rather than the face index.
    """
    if not _triton_available():
        pytest.skip("triton not available")

    from physicsnemo.datapipes.transforms import _sdf_triton
    from physicsnemo.datapipes.transforms._sdf_torch import (
        _build_surface_mesh,
        _nearest_face_bvh,
    )
    from physicsnemo.mesh.spatial import BVH

    device = torch.device("cuda")
    vertices, faces = _uv_sphere()
    vertices = vertices.to(device)
    faces = faces.to(device)

    torch.manual_seed(0)
    query = (torch.rand(8192, 3, device=device) * 3.0 - 1.5).float()

    mesh, face_vertices, _ = _build_surface_mesh(vertices.float(), faces)
    bvh = BVH.from_mesh(mesh)

    ref_dist_sq, _, ref_pt = _nearest_face_bvh(bvh, face_vertices, query, 1e8)
    tri_dist_sq, _, tri_pt = _sdf_triton.nearest_triangle_triton(
        bvh, face_vertices, query, 1e8
    )

    torch.testing.assert_close(
        tri_dist_sq.sqrt(), ref_dist_sq.sqrt(), atol=1e-4, rtol=1e-4
    )
    d_ref = (query - ref_pt).norm(dim=-1)
    d_tri = (query - tri_pt).norm(dim=-1)
    torch.testing.assert_close(d_tri, d_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not _CUDA, reason="CUDA required for the Triton SDF kernel")
@pytest.mark.parametrize("use_winding", [False, True])
def test_sdf_triton_end_to_end_matches_reference(use_winding, monkeypatch):
    """Full signed_distance_field_mesh: Triton path matches the torch fallback."""
    if not _triton_available():
        pytest.skip("triton not available")

    from physicsnemo.datapipes.transforms import _sdf_triton

    device = torch.device("cuda")
    vertices, faces = _uv_sphere()
    vertices = vertices.to(device)
    faces = faces.to(device)

    torch.manual_seed(0)
    query = (torch.rand(4096, 3, device=device) * 3.0 - 1.5).float()

    # Triton fast path (default dispatch on CUDA).
    sdf_triton, _ = signed_distance_field_mesh(
        vertices, faces, query, use_sign_winding_number=use_winding
    )

    # Force the pure-PyTorch reference by disabling the Triton dispatch.
    monkeypatch.setattr(_sdf_triton, "available", lambda: False)
    sdf_ref, _ = signed_distance_field_mesh(
        vertices, faces, query, use_sign_winding_number=use_winding
    )

    torch.testing.assert_close(sdf_triton, sdf_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not _CUDA, reason="CUDA required for the Triton SDF kernel")
def test_winding_sign_triton_matches_exact():
    """Fast (Barnes-Hut) winding sign agrees with the exact winding sign.

    Signs are compared away from the surface, where the winding number is
    unambiguous; the Barnes-Hut approximation is only loose right at the surface.
    """
    if not _triton_available():
        pytest.skip("triton not available")

    from physicsnemo.datapipes.transforms import _sdf_triton
    from physicsnemo.datapipes.transforms._sdf_torch import (
        _build_surface_mesh,
        _winding_number_sign,
    )
    from physicsnemo.mesh.spatial import BVH

    device = torch.device("cuda")
    vertices, faces = _uv_sphere()
    vertices = vertices.to(device)
    faces = faces.to(device)

    torch.manual_seed(0)
    query = (torch.rand(8192, 3, device=device) * 3.0 - 1.5).float()
    radius = query.norm(dim=-1)
    away = (radius - 1.0).abs() > 0.05  # exclude the near-surface shell

    mesh, face_vertices, _ = _build_surface_mesh(vertices.float(), faces)
    bvh = BVH.from_mesh(mesh)

    sign_fast = _sdf_triton.winding_sign_triton(bvh, face_vertices, query)
    sign_exact = _winding_number_sign(face_vertices, query)

    assert torch.all(sign_fast[away] == sign_exact[away])
