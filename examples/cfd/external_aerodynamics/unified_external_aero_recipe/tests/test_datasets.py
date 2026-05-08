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

"""Unit tests for the `_InjectMetadata` mesh transform in `src/datasets.py`.

`_InjectMetadata` is the recipe's first pipeline step: it writes the
dataset YAML's ``metadata:`` block into each sample's ``global_data`` so
downstream transforms (``NonDimensionalizeByMetadata``,
``SetGlobalField``, ...) can read freestream quantities from a single
canonical location. The transform must:

- Accept scalar / list / tuple / Tensor-valued metadata uniformly.
- Match the sample's device and dtype on injection.
- Preserve every other sample attribute (points, cells, point_data,
  cell_data, boundaries, interior).
- Work the same way for `Mesh` (single-mesh) and `DomainMesh` (domain
  with interior + boundaries).
"""

from __future__ import annotations

import pytest
import torch

from physicsnemo.mesh import DomainMesh, Mesh

from datasets import _InjectMetadata


### ---------------------------------------------------------------------------
### Mesh path
### ---------------------------------------------------------------------------


class TestInjectMetadataMesh:
    """Tests for `_InjectMetadata.__call__` (single-Mesh path)."""

    def test_scalar_list_and_tensor_values_all_become_float32_tensors(self):
        """Heterogeneous metadata values are coerced uniformly."""
        inj = _InjectMetadata(
            {
                "U_inf": [30.0, 0.0, 0.0],
                "rho_inf": 1.225,
                "L_ref": torch.tensor(5.0, dtype=torch.float64),
            }
        )
        mesh = Mesh(points=torch.zeros(5, 3))
        out = inj(mesh)
        assert out.global_data["U_inf"].tolist() == [30.0, 0.0, 0.0]
        ### approx() because 1.225 has no exact float32 representation.
        assert float(out.global_data["rho_inf"]) == pytest.approx(1.225)
        assert float(out.global_data["L_ref"]) == 5.0
        for k in ("U_inf", "rho_inf", "L_ref"):
            assert out.global_data[k].dtype == torch.float32

    def test_preserves_other_mesh_attributes(self):
        """Points, cells, point_data, cell_data are unchanged by injection."""
        mesh = Mesh(
            points=torch.randn(5, 3),
            cells=torch.randint(0, 5, (2, 3), dtype=torch.int64),
            point_data={"phi": torch.randn(5)},
            cell_data={"normals": torch.randn(2, 3)},
        )
        inj = _InjectMetadata({"U_inf": [1.0, 0.0, 0.0]})
        out = inj(mesh)
        assert torch.equal(out.points, mesh.points)
        assert torch.equal(out.cells, mesh.cells)
        assert torch.equal(out.point_data["phi"], mesh.point_data["phi"])
        assert torch.equal(out.cell_data["normals"], mesh.cell_data["normals"])

    def test_does_not_mutate_input_global_data(self):
        """Original mesh's `global_data` keeps its pre-call key set."""
        mesh = Mesh(
            points=torch.zeros(3, 3), global_data={"existing": torch.tensor(7.0)}
        )
        inj = _InjectMetadata({"U_inf": [1.0, 0.0, 0.0]})
        original_keys = set(mesh.global_data.keys())
        inj(mesh)
        assert set(mesh.global_data.keys()) == original_keys

    def test_existing_global_keys_are_preserved_alongside_injected_ones(self):
        """Pre-existing fields survive; injected fields are added."""
        mesh = Mesh(
            points=torch.zeros(3, 3), global_data={"existing": torch.tensor(7.0)}
        )
        inj = _InjectMetadata({"U_inf": [1.0, 0.0, 0.0]})
        out = inj(mesh)
        assert "existing" in out.global_data.keys()
        assert "U_inf" in out.global_data.keys()


### ---------------------------------------------------------------------------
### DomainMesh path
### ---------------------------------------------------------------------------


class TestInjectMetadataDomain:
    """Tests for `_InjectMetadata.apply_to_domain` (DomainMesh path)."""

    def _make_domain(self) -> DomainMesh:
        return DomainMesh(
            interior=Mesh(
                points=torch.randn(5, 3), point_data={"target": torch.randn(5)}
            ),
            boundaries={"vehicle": Mesh(points=torch.randn(3, 3))},
            global_data={"existing": torch.tensor(7.0)},
        )

    def test_lands_in_domain_global_data(self):
        """Domain-level ``global_data`` carries the injected fields."""
        dm = self._make_domain()
        inj = _InjectMetadata({"U_inf": [30.0, 0.0, 0.0], "L_ref": 5.0})
        out = inj.apply_to_domain(dm)
        assert "U_inf" in out.global_data.keys()
        assert float(out.global_data["L_ref"]) == 5.0
        assert "existing" in out.global_data.keys()

    def test_interior_and_boundaries_pass_through_unchanged(self):
        """Injection does not touch interior/boundaries."""
        dm = self._make_domain()
        inj = _InjectMetadata({"U_inf": [1.0, 0.0, 0.0]})
        out = inj.apply_to_domain(dm)
        assert out.interior is dm.interior
        assert out.boundaries is dm.boundaries

    def test_extra_repr_lists_field_names(self):
        """``extra_repr`` is sorted for stable display."""
        inj = _InjectMetadata({"L_ref": 5.0, "U_inf": [1.0, 0.0, 0.0]})
        ### sorted alphabetically by key.
        assert "fields=['L_ref', 'U_inf']" in repr(inj)
