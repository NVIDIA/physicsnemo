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

"""Tests for the SetGlobalField transform's DomainMesh path."""

import torch

from physicsnemo.datapipes.transforms.mesh import SetGlobalField
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.primitives.basic import single_triangle_3d


def _domain():
    """Build a small DomainMesh with one boundary and one global field."""
    return DomainMesh(
        interior=Mesh(points=torch.randn(12, 3)),
        boundaries={"wall": single_triangle_3d.load()},
        global_data={"U_inf": torch.tensor([3.0, 0.0, 4.0])},
    )


class TestSetGlobalFieldDomainLevel:
    """SetGlobalField writes both domain-level and sub-mesh global_data."""

    def test_domain_level_record_written(self):
        """The injected field lands in domain-level and sub-mesh global_data."""
        transform = SetGlobalField(fields={"reference_length": [1.0]})
        out = transform.apply_to_domain(_domain())
        expected = torch.tensor([1.0])
        for global_data in (
            out.global_data,
            out.interior.global_data,
            out.boundaries["wall"].global_data,
        ):
            torch.testing.assert_close(global_data["reference_length"], expected)

    def test_input_domain_is_not_modified(self):
        """Injection returns new metadata without mutating the input domain."""
        domain = _domain()
        original_u_inf = domain.global_data["U_inf"].clone()

        SetGlobalField(fields={"reference_length": [1.0]}).apply_to_domain(domain)

        assert "reference_length" not in domain.global_data.keys()
        assert "reference_length" not in domain.interior.global_data.keys()
        assert "reference_length" not in domain.boundaries["wall"].global_data.keys()
        torch.testing.assert_close(domain.global_data["U_inf"], original_u_inf)

    def test_existing_domain_fields_preserved(self):
        """Fields already in domain-level global_data survive injection."""
        out = SetGlobalField(fields={"x": [2.0]}).apply_to_domain(_domain())
        torch.testing.assert_close(
            out.global_data["U_inf"], torch.tensor([3.0, 0.0, 4.0])
        )

    def test_existing_domain_field_overwritten(self):
        """Injecting an existing key overwrites it, as documented."""
        out = SetGlobalField(fields={"U_inf": [1.0, 0.0, 0.0]}).apply_to_domain(
            _domain()
        )
        torch.testing.assert_close(
            out.global_data["U_inf"], torch.tensor([1.0, 0.0, 0.0])
        )

    def test_plain_mesh_path_unchanged(self):
        """The single-Mesh path still injects into the mesh's global_data."""
        mesh = Mesh(points=torch.randn(5, 3))
        out = SetGlobalField(fields={"x": [2.0]})(mesh)
        torch.testing.assert_close(out.global_data["x"], torch.tensor([2.0]))
