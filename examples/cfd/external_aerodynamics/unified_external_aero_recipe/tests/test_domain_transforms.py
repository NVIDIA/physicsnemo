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

"""Tests for the recipe-local surface mesh transforms."""

from pathlib import Path

import pytest
import torch
from datasets import build_dataset
from domain_transforms import ComputeFreestreamDirection, DropDegenerateCells
from omegaconf import OmegaConf
from tensordict import TensorDict

from physicsnemo.datapipes.protocols import DatasetBase
from physicsnemo.mesh import DomainMesh, Mesh


def _two_triangles(second_degenerate: bool = False) -> Mesh:
    """Two triangles in the xy-plane; the second is optionally collapsed."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
        ]
    )
    if second_degenerate:
        points[5] = points[3]
    return Mesh(
        points=points,
        cells=torch.tensor([[0, 1, 2], [3, 4, 5]]),
        cell_data={"pressure": torch.tensor([10.0, 20.0])},
        global_data={"U_inf": torch.tensor([3.0, 0.0, 4.0])},
    )


class TestComputeFreestreamDirection:
    """ComputeFreestreamDirection on meshes and domains."""

    def test_adds_unit_direction_and_keeps_velocity(self):
        """The unit direction is added and the physical velocity is untouched."""
        mesh = _two_triangles()
        out = ComputeFreestreamDirection()(mesh)

        torch.testing.assert_close(
            out.global_data["U_inf_dir"], torch.tensor([0.6, 0.0, 0.8])
        )
        torch.testing.assert_close(out.global_data["U_inf"], mesh.global_data["U_inf"])
        assert "U_inf_dir" not in mesh.global_data.keys()
        assert torch.equal(out.points, mesh.points)
        assert torch.equal(out.cell_data["pressure"], mesh.cell_data["pressure"])

    def test_custom_field_names(self):
        """Input and output field names are configurable."""
        mesh = Mesh(
            points=torch.zeros(3, 3),
            cells=torch.tensor([[0, 1, 2]]),
            global_data={"velocity": torch.tensor([0.0, -2.0, 0.0])},
        )
        out = ComputeFreestreamDirection(
            velocity_field="velocity", output_field="direction"
        )(mesh)

        torch.testing.assert_close(
            out.global_data["direction"], torch.tensor([0.0, -1.0, 0.0])
        )

    def test_domain_writes_domain_level_global_data(self):
        """On a DomainMesh the direction lands on the domain-level global_data."""
        boundary = _two_triangles()
        domain = DomainMesh(
            interior=Mesh(points=torch.zeros(2, 3)),
            boundaries={"vehicle": boundary},
            global_data=TensorDict({"U_inf": torch.tensor([0.0, 5.0, 0.0])}),
        )
        out = ComputeFreestreamDirection().apply_to_domain(domain)

        torch.testing.assert_close(
            out.global_data["U_inf_dir"], torch.tensor([0.0, 1.0, 0.0])
        )
        assert "U_inf_dir" not in out.boundaries["vehicle"].global_data.keys()

    def test_missing_velocity_raises(self):
        """A missing velocity field is a KeyError naming the field."""
        mesh = Mesh(points=torch.zeros(3, 3), cells=torch.tensor([[0, 1, 2]]))
        with pytest.raises(KeyError, match="U_inf"):
            ComputeFreestreamDirection()(mesh)

    def test_zero_velocity_raises(self):
        """A zero freestream vector has no direction and raises."""
        mesh = Mesh(
            points=torch.zeros(3, 3),
            cells=torch.tensor([[0, 1, 2]]),
            global_data={"U_inf": torch.zeros(3)},
        )
        with pytest.raises(ValueError, match="finite and positive"):
            ComputeFreestreamDirection()(mesh)


class TestDropDegenerateCells:
    """DropDegenerateCells on healthy, degenerate, and cell-free meshes."""

    def test_healthy_mesh_passes_through(self):
        """A mesh without degenerate cells is returned as-is."""
        mesh = _two_triangles()
        assert DropDegenerateCells()(mesh) is mesh

    def test_drops_zero_area_cell_and_its_data(self):
        """A collapsed triangle is dropped together with its cell_data row."""
        mesh = _two_triangles(second_degenerate=True)
        with pytest.warns(UserWarning, match="dropping 1 cell"):
            out = DropDegenerateCells()(mesh)

        assert out.n_cells == 1
        assert torch.equal(out.cells, torch.tensor([[0, 1, 2]]))
        assert torch.equal(out.cell_data["pressure"], torch.tensor([10.0]))
        assert torch.all(out.cell_areas > 0)

    def test_point_cloud_passes_through(self):
        """A mesh without cells has nothing to drop."""
        cloud = Mesh(points=torch.rand(4, 3))
        assert DropDegenerateCells()(cloud) is cloud

    @pytest.mark.parametrize("height", [1e-4, 1e-30])
    def test_preserves_thin_triangle_with_cached_area(self, height):
        """A valid cross product must not be rejected by Gram cancellation."""
        mesh = Mesh(
            points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, height, 0.0]]),
            cells=torch.tensor([[0, 1, 2]]),
            cell_data={"pressure": torch.tensor([10.0])},
        )
        # Populate the area cache before filtering. The float32 Gram formula
        # currently cancels to zero for these valid triangles.
        _ = mesh.cell_areas

        out = DropDegenerateCells()(mesh)

        assert out is mesh
        assert out.n_cells == 1
        torch.testing.assert_close(out.cell_data["pressure"], torch.tensor([10.0]))

    def test_rechecks_coordinates_after_area_cache_was_populated(self):
        """Cached positive areas cannot hide a subsequently collapsed face."""
        mesh = _two_triangles()
        assert torch.all(mesh.cell_areas > 0)
        mesh.points[5].copy_(mesh.points[3])

        with pytest.warns(UserWarning, match="dropping 1 cell"):
            out = DropDegenerateCells()(mesh)

        assert out.n_cells == 1
        torch.testing.assert_close(out.cell_data["pressure"], torch.tensor([10.0]))

    @pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), -float("inf")])
    def test_drops_cell_with_nonfinite_coordinates(self, coordinate):
        """Non-finite coordinates exclude their cell and its target row."""
        mesh = _two_triangles()
        mesh.points[5, 1] = coordinate

        with pytest.warns(UserWarning, match="dropping 1 cell"):
            out = DropDegenerateCells()(mesh)

        assert out.n_cells == 1
        assert torch.isfinite(out.points[out.cells]).all()
        torch.testing.assert_close(out.cell_data["pressure"], torch.tensor([10.0]))

    def test_all_rejected_cells_return_empty_connectivity_and_data(self):
        """An entirely collapsed mesh keeps empty cell fields aligned."""
        mesh = _two_triangles()
        mesh.points.zero_()

        with pytest.warns(UserWarning, match="dropping 2 cell"):
            out = DropDegenerateCells()(mesh)

        assert out.cells.shape == (0, 3)
        assert out.cell_data["pressure"].shape == (0,)
        assert out.n_points == mesh.n_points

    def test_drops_faces_collapsed_by_coordinate_rounding(self):
        """The check sees geometry after a translation rounds vertices together."""
        mesh = _two_triangles()
        translated = mesh.translate(torch.full((3,), 1e8))
        assert torch.equal(translated.points[0], translated.points[1])

        with pytest.warns(UserWarning, match="dropping 2 cell"):
            out = DropDegenerateCells()(translated)

        assert out.n_cells == 0

    def test_other_simplex_dimensions_use_fresh_measure(self):
        """The general simplex path still drops a collapsed 2D triangle."""
        mesh = _two_triangles(second_degenerate=True)
        mesh = mesh.with_points(mesh.points[:, :2])

        with pytest.warns(UserWarning, match="dropping 1 cell"):
            out = DropDegenerateCells()(mesh)

        assert out.n_cells == 1
        torch.testing.assert_close(out.cell_data["pressure"], torch.tensor([10.0]))


@pytest.mark.parametrize("rotate", [False, True])
def test_drivaer_dataset_pipeline_preserves_thin_face_and_aligns_targets(
    tmp_path, rotate
):
    """Run the actual reader, configured transforms, and target conversion."""
    vehicle = Mesh(
        points=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1e-4, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [5.0, 1.0, 0.0],
            ]
        ),
        cells=torch.arange(9).reshape(3, 3),
        cell_data={
            "pMeanTrim": torch.tensor([100.0, 200.0, 300.0]),
            "wallShearStressMeanTrim": torch.zeros(3, 3),
        },
        global_data={"TimeValue": torch.tensor(0.0)},
    )
    source = DomainMesh(
        interior=Mesh(points=torch.zeros(0, 3)),
        boundaries={"vehicle": vehicle},
        global_data={
            "U_inf": torch.tensor([10.0, 0.0, 0.0]),
            "rho_inf": torch.tensor(2.0),
            "p_inf": torch.tensor(0.0),
            "L_ref": torch.tensor(1.0),
        },
    )
    sample_path = tmp_path / "run_001" / "sample.pdmsh"
    sample_path.parent.mkdir()
    source.save(sample_path)
    recipe = Path(__file__).resolve().parent.parent
    cfg = OmegaConf.merge(
        OmegaConf.load(recipe / "datasets" / "drivaer_ml_surface.yaml"),
        {
            "dataset_paths": {"drivaer_ml": str(tmp_path)},
            "sampling_resolution": 3,
        },
    )
    # Exercise the configured rotation and its insertion point. Translation's
    # pre-existing Uniform/list instantiation issue is outside this regression.
    cfg.pipeline.augmentations = [cfg.pipeline.augmentations[0]]
    dataset = build_dataset(cfg, augment=rotate, device=None, num_workers=1)
    try:
        with pytest.warns(UserWarning, match="dropping 1 cell"):
            domain, _ = dataset[0]
    finally:
        # MeshReader has no close() hook; release the dataset's prefetch pool.
        DatasetBase.close(dataset)

    boundary = domain.boundaries["vehicle"]
    assert boundary.n_cells == domain.interior.n_points == 2
    torch.testing.assert_close(
        domain.interior.point_data["pressure"], torch.tensor([1.0, 3.0])
    )
    torch.testing.assert_close(domain.interior.points, boundary.cell_centroids)
    torch.testing.assert_close(
        boundary.cell_data["normals"].norm(dim=-1), torch.ones(2)
    )
    torch.testing.assert_close(
        domain.global_data["U_inf_dir"],
        domain.global_data["U_inf"] / domain.global_data["U_inf"].norm(),
    )
    assert "TimeValue" not in domain.global_data
