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

"""Lumpy sphere with radial noise in 3D space.

Dimensional: 2D manifold in 3D space (closed, no boundary, irregular).
"""

import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import icosahedron_surface


def load(
    radius: float = 1.0,
    subdivisions: int = 3,
    noise_amplitude: float = 0.5,
    seed: int = 0,
    device: str = "cpu",
) -> Mesh:
    """Create a lumpy sphere by adding radial noise to a sphere.

    Parameters
    ----------
    radius : float, optional
        Base radius of the sphere
    subdivisions : int, optional
        Number of subdivision levels
    noise_amplitude : float, optional
        Amplitude of radial noise
    seed : int, optional
        Random seed for reproducibility
    device : str, optional
        Compute device ('cpu' or 'cuda')

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=3
    """
    mesh = icosahedron_surface.load(radius=radius, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = noise_amplitude * torch.randn(
        mesh.n_points, 1, generator=generator, device=device
    )
    mesh.points = mesh.points * noise.exp()

    return mesh.subdivide(subdivisions, "loop")
