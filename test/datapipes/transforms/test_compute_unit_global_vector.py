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

"""Tests for the ComputeUnitGlobalVector transform."""

import pytest
import torch

from physicsnemo.datapipes.transforms.mesh import ComputeUnitGlobalVector
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.primitives.basic import single_triangle_3d


def _domain():
    return DomainMesh(
        interior=Mesh(points=torch.randn(12, 3)),
        boundaries={"wall": single_triangle_3d.load()},
        global_data={"U_inf": torch.tensor([3.0, 0.0, 4.0])},
    )


class TestComputeUnitGlobalVector:
    def test_direction_on_mesh(self):
        mesh = Mesh(
            points=torch.randn(5, 3),
            global_data={"v": torch.tensor([0.0, 5.0, 0.0])},
        )
        out = ComputeUnitGlobalVector(vector_field="v", output_field="v_dir")(mesh)
        torch.testing.assert_close(
            out.global_data["v_dir"], torch.tensor([0.0, 1.0, 0.0])
        )

    def test_direction_on_domain(self):
        transform = ComputeUnitGlobalVector(
            vector_field="U_inf", output_field="U_inf_dir"
        )
        out = transform.apply_to_domain(_domain())
        torch.testing.assert_close(
            out.global_data["U_inf_dir"], torch.tensor([0.6, 0.0, 0.8])
        )
        # The source vector is unchanged.
        torch.testing.assert_close(
            out.global_data["U_inf"], torch.tensor([3.0, 0.0, 4.0])
        )

    def test_source_dtype_preserved(self):
        # Rotation matrices are built at points.dtype, so the direction
        # must keep the source dtype for rotations to compose with it.
        mesh = Mesh(
            points=torch.randn(5, 3, dtype=torch.float64),
            global_data={"v": torch.tensor([3.0, 0.0, 4.0], dtype=torch.float64)},
        )
        out = ComputeUnitGlobalVector(vector_field="v", output_field="v_dir")(mesh)
        assert out.global_data["v_dir"].dtype == torch.float64
        rotated = out.rotate(angle=0.3, axis="z", transform_global_data=True)
        torch.testing.assert_close(
            torch.linalg.vector_norm(rotated.global_data["v_dir"]),
            torch.tensor(1.0, dtype=torch.float64),
        )

    def test_missing_field_raises(self):
        with pytest.raises(KeyError, match="not found in global_data"):
            ComputeUnitGlobalVector(
                vector_field="missing", output_field="d"
            ).apply_to_domain(_domain())

    def test_zero_vector_raises(self):
        mesh = Mesh(
            points=torch.randn(5, 3),
            global_data={"v": torch.tensor([0.0, 0.0, 0.0])},
        )
        with pytest.raises(ValueError, match="finite and positive"):
            ComputeUnitGlobalVector(vector_field="v", output_field="d")(mesh)
