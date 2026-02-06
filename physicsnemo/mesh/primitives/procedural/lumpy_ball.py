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

"""Lumpy ball volume mesh in 3D space.

A solid tetrahedral mesh built from concentric icosahedral shells with
optional radial noise. This is the volumetric analog to lumpy_sphere.

Dimensional: 3D manifold in 3D space (solid, no boundary on surface cells).
"""

import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import icosahedron_surface


def load(
    radius: float = 1.0,
    n_shells: int = 3,
    subdivisions: int = 2,
    noise_amplitude: float = 0.5,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> Mesh:
    """Create a lumpy ball volume mesh.

    Builds a solid ball from concentric icosahedral shells connected by
    tetrahedra. The mesh has naturally graded cell sizes (smaller near
    center, larger at surface) and mixed vertex valences inherited from
    the icosahedral structure.

    Parameters
    ----------
    radius : float
        Outer radius of the ball.
    n_shells : int
        Number of concentric shells (more = finer radial resolution).
        Must be at least 1.
    subdivisions : int
        Subdivision level per shell (more = finer angular resolution).
        Each level quadruples the number of faces.
    noise_amplitude : float
        Radial noise amplitude. 0 = perfect sphere, >0 = lumpy.
        Uses log-normal scaling like lumpy_sphere.
    seed : int
        Random seed for noise reproducibility.
    device : torch.device or str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=3, n_spatial_dims=3.

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.procedural import lumpy_ball
    >>> mesh = lumpy_ball.load(radius=1.0, n_shells=2, subdivisions=1)
    >>> mesh.n_manifold_dims, mesh.n_spatial_dims
    (3, 3)
    >>> mesh.n_cells  # 80 faces * (3*2 - 2) = 320
    320
    """
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius=}")
    if n_shells < 1:
        raise ValueError(f"n_shells must be at least 1, got {n_shells=}")
    if subdivisions < 0:
        raise ValueError(f"subdivisions must be non-negative, got {subdivisions=}")
    if noise_amplitude < 0:
        raise ValueError(
            f"noise_amplitude must be non-negative, got {noise_amplitude=}"
        )

    ### Step 1: Generate base icosahedron at unit radius
    template = icosahedron_surface.load(radius=1.0, device=device)

    ### Step 2: Apply noise to base icosahedron (if any)
    # Noise is applied to the base icosahedron (12 vertices) BEFORE subdivision.
    # This creates smooth, coherent lumps after subdivision, matching lumpy_sphere.
    # All shells are scaled versions of this noisy shape, ensuring shells remain
    # strictly nested and tetrahedra remain valid regardless of noise amplitude.
    if noise_amplitude > 0:
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = noise_amplitude * torch.randn(
            template.n_points, 1, generator=generator, device=device
        )
        # Log-normal scaling applied to base icosahedron (same as lumpy_sphere)
        template = Mesh(
            points=template.points * noise.exp(),
            cells=template.cells,
        )

    ### Step 3: Subdivide with loop scheme (if any)
    # Loop subdivision is an approximating scheme that smooths the noisy base
    # icosahedron into broad, coherent lumps.
    if subdivisions > 0:
        template = template.subdivide(subdivisions, "loop")

    n_verts_per_shell = template.n_points
    n_faces = template.n_cells

    ### Step 4: Generate shell radii (linear spacing from center to outer)
    # Vectorized: torch.arange instead of list comprehension
    shell_radii = (
        radius
        * torch.arange(1, n_shells + 1, device=device, dtype=torch.float32)
        / n_shells
    )

    ### Step 5: Build all vertices by scaling template
    # Vectorized: broadcasting instead of list comprehension
    # shell_radii: (n_shells,) -> (n_shells, 1, 1)
    # template.points: (n_verts, 3) -> (1, n_verts, 3)
    # Result: (n_shells, n_verts, 3) -> (n_shells * n_verts, 3)
    center = torch.zeros(1, 3, dtype=torch.float32, device=device)
    shell_points = (template.points.unsqueeze(0) * shell_radii.view(-1, 1, 1)).reshape(
        -1, 3
    )
    all_points = torch.cat([center, shell_points], dim=0)

    ### Step 6: Build core tetrahedra (center to innermost shell)
    # Vectorized: direct tensor operations instead of for loop
    # template.cells: (n_faces, 3), add offset 1 to shift to shell 1 indices
    shell1_faces = template.cells + 1  # (n_faces, 3)
    zeros_col = torch.zeros(n_faces, 1, dtype=torch.int64, device=device)
    core_cells = torch.cat([zeros_col, shell1_faces], dim=1)  # (n_faces, 4)

    ### Step 7: Build inter-shell tetrahedra (prism decomposition)
    # Each triangular prism between shells decomposes into 3 tetrahedra.
    # CRITICAL: We must use a CONSISTENT diagonal for each lateral rectangle,
    # regardless of which adjacent face we're processing. This is achieved by
    # sorting face vertices by their TEMPLATE index (not shell-offset index),
    # so that a < b < c and the decomposition is canonical.
    #
    # With sorted vertices (a < b < c), the Freudenthal decomposition is:
    #   tet1: (a_in, b_in, c_in, a_out)
    #   tet2: (b_in, c_in, a_out, b_out)
    #   tet3: (c_in, a_out, b_out, c_out)
    #
    # This guarantees that for each lateral rectangle, adjacent prisms use
    # the same diagonal, producing matching triangular faces.
    if n_shells > 1:
        # Sort face vertices by template index to ensure consistent decomposition
        sorted_faces = torch.sort(template.cells, dim=1)[0]  # (n_faces, 3)

        # Compute all shell pair offsets as tensors
        shell_indices = torch.arange(n_shells - 1, device=device)
        inner_offsets = 1 + shell_indices * n_verts_per_shell  # (n_shells-1,)
        outer_offsets = inner_offsets + n_verts_per_shell  # (n_shells-1,)

        # Broadcast sorted face indices across all shell pairs
        # sorted_faces: (n_faces, 3) -> (1, n_faces, 3)
        # offsets: (n_shells-1,) -> (n_shells-1, 1, 1)
        # Result: (n_shells-1, n_faces, 3)
        faces_expanded = sorted_faces.unsqueeze(0)
        inner_faces = faces_expanded + inner_offsets.view(-1, 1, 1)
        outer_faces = faces_expanded + outer_offsets.view(-1, 1, 1)

        # Extract individual vertex indices: each has shape (n_shells-1, n_faces)
        # Now a < b < c by template index, ensuring consistent diagonal choice
        a_in, b_in, c_in = inner_faces[..., 0], inner_faces[..., 1], inner_faces[..., 2]
        a_out, b_out, c_out = (
            outer_faces[..., 0],
            outer_faces[..., 1],
            outer_faces[..., 2],
        )

        # Build 3 tetrahedra per prism: each stack produces (n_shells-1, n_faces, 4)
        tet1 = torch.stack([a_in, b_in, c_in, a_out], dim=-1)
        tet2 = torch.stack([b_in, c_in, a_out, b_out], dim=-1)
        tet3 = torch.stack([c_in, a_out, b_out, c_out], dim=-1)

        # Interleave and flatten: (n_shells-1, n_faces, 3, 4) -> ((n_shells-1)*n_faces*3, 4)
        inter_shell_cells = torch.stack([tet1, tet2, tet3], dim=2).reshape(-1, 4)
    else:
        inter_shell_cells = torch.empty((0, 4), dtype=torch.int64, device=device)

    ### Step 8: Assemble all cells
    all_cells = torch.cat([core_cells, inter_shell_cells], dim=0)

    ### Step 9: Fix tetrahedron orientation
    # The vertex sorting for consistent diagonals may produce some tets with
    # negative orientation (inverted). Detect and fix by swapping two vertices.
    # Signed volume = (1/6) * (b-a) · ((c-a) × (d-a))
    # Positive = consistent orientation, Negative = inverted
    tet_verts = all_points[all_cells]  # (n_cells, 4, 3)
    v0, v1, v2, v3 = tet_verts[:, 0], tet_verts[:, 1], tet_verts[:, 2], tet_verts[:, 3]
    signed_volumes = torch.einsum(
        "ij,ij->i", v1 - v0, torch.cross(v2 - v0, v3 - v0, dim=1)
    )

    # Flip inverted tets by swapping vertices 2 and 3 (changes sign of volume)
    inverted = signed_volumes < 0
    if inverted.any():
        all_cells[inverted, 2], all_cells[inverted, 3] = (
            all_cells[inverted, 3].clone(),
            all_cells[inverted, 2].clone(),
        )

    return Mesh(points=all_points, cells=all_cells)
