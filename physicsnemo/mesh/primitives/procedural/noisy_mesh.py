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

"""Add Gaussian noise to any mesh.

This is a generic utility for creating perturbed versions of meshes.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    base_mesh: Mesh,
    noise_scale: float = 0.1,
    seed: int = 0,
) -> Mesh:
    """Add Gaussian noise to mesh vertex positions.

    Parameters
    ----------
    base_mesh : Mesh
        Input mesh to perturb.
    noise_scale : float
        Standard deviation of Gaussian noise.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Mesh
        Mesh with same connectivity but perturbed vertex positions.
    """
    generator = torch.Generator(device=base_mesh.points.device).manual_seed(seed)

    # Generate noise with same shape as points
    noise = torch.randn(
        base_mesh.points.shape,
        dtype=base_mesh.points.dtype,
        device=base_mesh.points.device,
        generator=generator,
    )

    # Add scaled noise to points
    noisy_points = base_mesh.points + noise_scale * noise

    return Mesh(
        points=noisy_points,
        cells=base_mesh.cells,
        point_data=base_mesh.point_data,
        cell_data=base_mesh.cell_data,
        global_data=base_mesh.global_data,
    )
