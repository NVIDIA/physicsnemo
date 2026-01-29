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

import numpy as np
import pytest
import torch
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.core.version_check import check_version_spec
from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor
from physicsnemo.nn.functional import signed_distance_field

from .utils import numerical_shard_tensor_check

SCIPY_AVAILABLE = check_version_spec("scipy", hard_fail=False)

if SCIPY_AVAILABLE:
    from scipy.spatial import ConvexHull
else:
    pytest.skip(
        "scipy is not installed, cannot use ConvexHull", allow_module_level=True
    )


# This is from the domino datapipe, too:
def random_sample_on_unit_sphere(n_points):
    # Random points on the sphere:
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    cos_theta = np.random.uniform(-1, 1, n_points)
    theta = np.arccos(cos_theta)

    # Convert to x/y/z and stack:
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    points = np.stack([x, y, z], axis=1)
    return points


def mesh_vertices_and_indices(n_points):
    # We are generating a mesh on a random sphere.
    stl_points = random_sample_on_unit_sphere(n_points)

    # Generate the triangles with ConvexHull:
    hull = ConvexHull(stl_points)
    faces = hull.simplices  # (M, 3)

    return stl_points, faces


class SDFModule(torch.nn.Module):
    """
    This is a test module to run the SDF function ... don't use it elsewhere.
    """

    def __init__(self, max_dist=1e8, use_sign_winding_number=False):
        super().__init__()

        self.max_dist = max_dist
        self.use_sign_winding_number = use_sign_winding_number

    def forward(self, mesh_vertices, mesh_indices, input_points):
        return signed_distance_field(
            mesh_vertices,
            mesh_indices,
            input_points,
            self.max_dist,
            self.use_sign_winding_number,
        )


@pytest.mark.multigpu_static
@pytest.mark.parametrize("scatter_mesh", [True, False])
@pytest.mark.parametrize("scatter_inputs", [True, False])
def test_sdf_1dmesh(
    distributed_mesh,
    scatter_mesh: bool,
    scatter_inputs: bool,
):
    dm = DistributedManager()

    # Generate a mesh on a unit sphere:
    mesh_vertices, mesh_indices = mesh_vertices_and_indices(932)

    # Cast the vertices and indices to tensors:
    mesh_vertices = torch.tensor(mesh_vertices).to(dm.device)
    mesh_indices = torch.tensor(mesh_indices.flatten()).to(dm.device)

    # Distribute the inputs:
    mesh_placements = (Shard(0),) if scatter_mesh else (Replicate(),)
    input_placements = (Shard(0),) if scatter_inputs else (Replicate(),)

    sharded_mesh_vertices = scatter_tensor(
        mesh_vertices, 0, distributed_mesh, mesh_placements
    )
    sharded_mesh_indices = scatter_tensor(
        mesh_indices, 0, distributed_mesh, mesh_placements
    )

    # Generate random points in the volume:
    input_points = torch.randn(1043, 3).to(dm.device)

    sharded_input_points = scatter_tensor(
        input_points, 0, distributed_mesh, input_placements
    )

    module = SDFModule()

    numerical_shard_tensor_check(
        distributed_mesh,
        module,
        [sharded_mesh_vertices, sharded_mesh_indices, sharded_input_points],
        {},
        check_grads=False,
    )
