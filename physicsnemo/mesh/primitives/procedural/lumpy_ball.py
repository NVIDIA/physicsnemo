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

"""Lumpy ball volume mesh in 3D space.

A solid tetrahedral mesh built from concentric icosahedral shells with
optional radial noise. This is the volumetric analog to lumpy_sphere.

Dimensional: 3D manifold in 3D space (solid, no boundary on surface cells).
"""

import torch
import torch.nn.functional as F

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import icosahedron_surface


def load(
    radius: float = 1.0,
    n_shells: int = 3,
    subdivisions: int = 2,
    noise_amplitude: float = 0.0,
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
        raise ValueError(f"noise_amplitude must be non-negative, got {noise_amplitude=}")

    ### Step 1: Generate shell template (subdivided icosahedron at unit radius)
    template = icosahedron_surface.load(radius=1.0, device=device)
    if subdivisions > 0:
        template = template.subdivide(subdivisions, "linear")
        # Project back to unit sphere
        template = Mesh(
            points=F.normalize(template.points, dim=-1),
            cells=template.cells,
        )

    n_verts_per_shell = template.n_points
    n_faces = template.n_cells

    ### Step 2: Generate shell radii (linear spacing from center to outer)
    shell_radii = [radius * (i + 1) / n_shells for i in range(n_shells)]

    ### Step 3: Apply noise to canonical template (if any)
    # Noise is applied ONCE to the unit template, then all shells are scaled
    # versions of this noisy shape. This ensures shells remain strictly nested
    # and tetrahedra remain valid regardless of noise amplitude.
    if noise_amplitude > 0:
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = noise_amplitude * torch.randn(
            n_verts_per_shell, 1, generator=generator, device=device
        )
        # Log-normal scaling applied to unit template (same as lumpy_sphere)
        noisy_template_points = template.points * noise.exp()
    else:
        noisy_template_points = template.points

    ### Step 4: Build all vertices by scaling noisy template
    # Center point at index 0
    center = torch.zeros(1, 3, dtype=torch.float32, device=device)

    # Shell vertices: scale noisy template to each radius
    shell_points = [noisy_template_points * r for r in shell_radii]

    # Concatenate: [center, shell_1_verts, shell_2_verts, ...]
    all_points = torch.cat([center] + shell_points, dim=0)

    ### Step 5: Build core tetrahedra (center to innermost shell)
    # For each face (a, b, c) on innermost shell, create tet (center, a, b, c)
    # Vertex indices in shell 1 start at offset 1
    core_cells = []
    offset = 1
    for face_idx in range(n_faces):
        face = template.cells[face_idx]
        a, b, c = face[0].item() + offset, face[1].item() + offset, face[2].item() + offset
        core_cells.append([0, a, b, c])  # 0 is center

    ### Step 6: Build inter-shell tetrahedra (prism decomposition)
    # Each triangular prism between shells decomposes into 3 tetrahedra:
    #   Prism vertices: inner (a, b, c), outer (a', b', c')
    #   Decomposition:
    #     tet1: (a, b, c, a')
    #     tet2: (b, c, a', b')
    #     tet3: (c, a', b', c')
    inter_shell_cells = []

    for shell_idx in range(n_shells - 1):
        inner_offset = 1 + shell_idx * n_verts_per_shell
        outer_offset = 1 + (shell_idx + 1) * n_verts_per_shell

        for face_idx in range(n_faces):
            face = template.cells[face_idx]
            a_in = face[0].item() + inner_offset
            b_in = face[1].item() + inner_offset
            c_in = face[2].item() + inner_offset
            a_out = face[0].item() + outer_offset
            b_out = face[1].item() + outer_offset
            c_out = face[2].item() + outer_offset

            # 3-tet decomposition of triangular prism
            inter_shell_cells.append([a_in, b_in, c_in, a_out])
            inter_shell_cells.append([b_in, c_in, a_out, b_out])
            inter_shell_cells.append([c_in, a_out, b_out, c_out])

    ### Step 7: Assemble and return Mesh
    all_cells_list = core_cells + inter_shell_cells
    all_cells = torch.tensor(all_cells_list, dtype=torch.int64, device=device)

    return Mesh(points=all_points, cells=all_cells)
