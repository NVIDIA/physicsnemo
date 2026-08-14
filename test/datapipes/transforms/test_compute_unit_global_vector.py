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

from physicsnemo.datapipes.transforms.mesh import (
    ComputeUnitGlobalVector,
    MeshToDomainMesh,
)
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.primitives.basic import single_triangle_3d


def _domain():
    """Build a small DomainMesh with one boundary and one global field."""
    wall = single_triangle_3d.load()
    wall.global_data["wall_id"] = torch.tensor([7.0])
    return DomainMesh(
        interior=Mesh(points=torch.randn(12, 3)),
        boundaries={"wall": wall},
        global_data={"U_inf": torch.tensor([3.0, 0.0, 4.0])},
    )


class TestComputeUnitGlobalVector:
    """ComputeUnitGlobalVector derives v/|v| without touching the source."""

    def test_direction_on_mesh(self):
        """The unit direction is stored under the new key on a Mesh."""
        mesh = Mesh(
            points=torch.randn(5, 3),
            global_data={"v": torch.tensor([0.0, 5.0, 0.0])},
        )
        out = ComputeUnitGlobalVector(vector_field="v", output_field="v_dir")(mesh)
        torch.testing.assert_close(
            out.global_data["v_dir"], torch.tensor([0.0, 1.0, 0.0])
        )

    def test_direction_on_2d_mesh(self):
        """Rank-one vectors matching a 2-D mesh are supported."""
        mesh = Mesh(
            points=torch.randn(5, 2),
            global_data={"v": torch.tensor([3.0, 4.0])},
        )
        out = ComputeUnitGlobalVector(vector_field="v", output_field="v_dir")(mesh)
        torch.testing.assert_close(out.global_data["v_dir"], torch.tensor([0.6, 0.8]))

    def test_direction_on_domain(self):
        """The unit direction is stored in domain-level global_data."""
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

    def test_submesh_global_data_untouched(self):
        """Sub-mesh global_data is not modified on the DomainMesh path."""
        transform = ComputeUnitGlobalVector(
            vector_field="U_inf", output_field="U_inf_dir"
        )
        out = transform.apply_to_domain(_domain())
        assert "U_inf_dir" not in out.boundaries["wall"].global_data.keys()
        torch.testing.assert_close(
            out.boundaries["wall"].global_data["wall_id"], torch.tensor([7.0])
        )

    def test_before_mesh_to_domain_copies_direction_to_boundary(self):
        """Ordering before conversion copies the direction with global_data."""
        mesh = single_triangle_3d.load()
        mesh.global_data["v"] = torch.tensor([3.0, 0.0, 4.0])
        transform = ComputeUnitGlobalVector(vector_field="v", output_field="v_dir")
        domain = MeshToDomainMesh(interior_points="vertices")(transform(mesh))
        expected = torch.tensor([0.6, 0.0, 0.8])
        torch.testing.assert_close(domain.global_data["v_dir"], expected)
        torch.testing.assert_close(
            domain.boundaries["vehicle"].global_data["v_dir"], expected
        )

    def test_source_dtype_preserved(self):
        """The direction keeps the source dtype so rotations compose with it."""
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
        """A missing vector_field raises a KeyError naming available keys."""
        with pytest.raises(KeyError, match="not found in global_data"):
            ComputeUnitGlobalVector(
                vector_field="missing", output_field="d"
            ).apply_to_domain(_domain())

    def test_zero_vector_raises(self):
        """A zero-length vector raises instead of dividing by zero."""
        mesh = Mesh(
            points=torch.randn(5, 3),
            global_data={"v": torch.tensor([0.0, 0.0, 0.0])},
        )
        with pytest.raises(ValueError, match="must be positive"):
            ComputeUnitGlobalVector(vector_field="v", output_field="d")(mesh)

    def test_integer_vector_raises(self):
        """An integer vector raises instead of truncating to zeros."""
        mesh = Mesh(
            points=torch.randn(5, 3),
            global_data={"v": torch.tensor([3, 0, 4])},
        )
        with pytest.raises(ValueError, match="floating-point"):
            ComputeUnitGlobalVector(vector_field="v", output_field="d")(mesh)

    @pytest.mark.parametrize(
        "value",
        [
            torch.tensor(1.0),
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[[1.0, 0.0, 0.0]]]),
        ],
        ids=["scalar", "singleton_batch", "rank_three_singleton_batch"],
    )
    def test_non_rank_one_vector_raises(self, value):
        """Scalars and batched vectors raise, including singleton batches."""
        mesh = Mesh(
            points=torch.randn(5, 3),
            global_data={"v": value},
        )
        with pytest.raises(ValueError, match="single rank-one vector"):
            ComputeUnitGlobalVector(vector_field="v", output_field="d")(mesh)

    def test_spatial_dimension_mismatch_raises(self):
        """Vector length must match the mesh's spatial dimension."""
        mesh = Mesh(
            points=torch.randn(5, 3),
            global_data={"v": torch.tensor([3.0, 4.0])},
        )
        with pytest.raises(ValueError, match="length 3"):
            ComputeUnitGlobalVector(vector_field="v", output_field="d")(mesh)

    @pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
    def test_nonfinite_component_raises(self, nonfinite):
        """NaN and positive or negative infinity are rejected explicitly."""
        mesh = Mesh(
            points=torch.randn(5, 3),
            global_data={"v": torch.tensor([1.0, nonfinite, 0.0])},
        )
        with pytest.raises(ValueError, match="only finite values"):
            ComputeUnitGlobalVector(vector_field="v", output_field="d")(mesh)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("magnitude_name", ["large", "tiny"])
    def test_extreme_finite_vector_normalizes(self, dtype, magnitude_name):
        """Finite vectors normalize without norm overflow or underflow."""
        magnitude = (
            torch.finfo(dtype).max
            if magnitude_name == "large"
            else torch.finfo(dtype).tiny
        )
        mesh = Mesh(
            points=torch.randn(5, 3, dtype=dtype),
            global_data={"v": torch.tensor([magnitude, -magnitude, 0.0], dtype=dtype)},
        )
        out = ComputeUnitGlobalVector(vector_field="v", output_field="d")(mesh)
        expected = torch.tensor([2**-0.5, -(2**-0.5), 0.0], dtype=dtype)
        torch.testing.assert_close(out.global_data["d"], expected)

    def test_same_output_field_raises(self):
        """output_field == vector_field is rejected at construction."""
        with pytest.raises(ValueError, match="must differ"):
            ComputeUnitGlobalVector(vector_field="v", output_field="v")

    def test_registered_in_hydra_resolver(self):
        """The transform resolves through the ${dp:...} config resolver."""
        from omegaconf import OmegaConf

        import physicsnemo.datapipes  # noqa: F401  -- side-effect import

        cfg = OmegaConf.create({"_target_": "${dp:ComputeUnitGlobalVector}"})
        assert cfg._target_.endswith("ComputeUnitGlobalVector")
