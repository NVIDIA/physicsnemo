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


def generate_test_data(bsize, nx, ny, nz, num_neigh, device):
    """Generate test data for DoMINO"""
    return {
        "pos_volume_closest": torch.randn(bsize, 50, 3).to(device),
        "pos_volume_center_of_mass": torch.randn(bsize, 50, 3).to(device),
        "pos_surface_center_of_mass": torch.randn(bsize, 50, 3).to(device),
        "geometry_coordinates": torch.randn(bsize, 50, 3).to(device),
        "grid": torch.randn(bsize, nx, ny, nz, 3).to(device),
        "surf_grid": torch.randn(bsize, nx, ny, nz, 3).to(device),
        "sdf_grid": torch.randn(bsize, nx, ny, nz).to(device),
        "sdf_surf_grid": torch.randn(bsize, nx, ny, nz).to(device),
        "sdf_nodes": torch.randn(bsize, 50, 1).to(device),
        "surface_mesh_centers": torch.randn(bsize, 50, 3).to(device),
        "surface_mesh_neighbors": torch.randn(bsize, 50, num_neigh, 3).to(device),
        "surface_normals": torch.randn(bsize, 50, 3).to(device),
        "surface_neighbors_normals": torch.randn(bsize, 50, num_neigh, 3).to(device),
        "surface_areas": torch.rand(bsize, 50).to(device) + 1e-6,
        "surface_neighbors_areas": torch.rand(bsize, 50, num_neigh).to(device) + 1e-6,
        "volume_mesh_centers": torch.randn(bsize, 50, 3).to(device),
        "volume_min_max": torch.randn(bsize, 2, 3).to(device),
        "surface_min_max": torch.randn(bsize, 2, 3).to(device),
        "global_params_values": torch.randn(bsize, 2, 1).to(device),
        "global_params_reference": torch.randn(bsize, 2, 1).to(device),
    }


def validate_output_shape_and_values(output, expected_shape, check_finite=True):
    """Validate output tensor shape and values"""
    if output is not None:
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )
        if check_finite:
            assert torch.isfinite(output).all(), "Output contains non-finite values"
        assert not torch.isnan(output).any(), "Output contains NaN values"
