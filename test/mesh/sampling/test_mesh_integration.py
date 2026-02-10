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

"""Tests for Mesh class integration with sampling."""

import torch

from physicsnemo.mesh.mesh import Mesh


class TestMeshSamplingIntegration:
    """Tests for Mesh.sample_data_at_points convenience method."""

    def test_mesh_sample_data_at_points_method(self):
        """Test that Mesh.sample_data_at_points delegates correctly."""
        ### Create a simple mesh with data
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ]
        )
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"temperature": torch.tensor([100.0, 200.0])},
            point_data={"value": torch.tensor([0.0, 1.0, 2.0, 3.0])},
        )

        ### Test cell data sampling
        queries = torch.tensor(
            [
                [0.25, 0.25],  # In first triangle
                [0.75, 0.75],  # In second triangle
            ]
        )

        result_cells = mesh.sample_data_at_points(queries, data_source="cells")
        assert torch.allclose(result_cells["temperature"][0], torch.tensor(100.0))
        assert torch.allclose(result_cells["temperature"][1], torch.tensor(200.0))

        ### Test point data sampling with interpolation
        result_points = mesh.sample_data_at_points(queries, data_source="points")
        # First query at (0.25, 0.25) in triangle [0,1,2] with values [0, 1, 2]
        # Should get some interpolated value
        assert not torch.isnan(result_points["value"][0])
        assert not torch.isnan(result_points["value"][1])

    def test_mesh_sample_outside_returns_nan(self):
        """Test that mesh sampling outside returns NaN."""
        ### Create a simple mesh
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"temperature": torch.tensor([100.0])},
        )

        ### Query outside
        queries = torch.tensor([[2.0, 2.0]])

        result = mesh.sample_data_at_points(queries, data_source="cells")
        assert torch.isnan(result["temperature"][0])

    def test_mesh_sample_with_projection(self):
        """Test mesh sampling with projection onto nearest cell."""
        ### Create a simple mesh
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(
            points=points,
            cells=cells,
            cell_data={"temperature": torch.tensor([100.0])},
        )

        ### Query outside but close
        queries = torch.tensor([[0.5, 0.6]])

        result = mesh.sample_data_at_points(
            queries,
            data_source="cells",
            project_onto_nearest_cell=True,
        )

        ### Should get a value (not NaN) because of projection
        assert not torch.isnan(result["temperature"][0])
