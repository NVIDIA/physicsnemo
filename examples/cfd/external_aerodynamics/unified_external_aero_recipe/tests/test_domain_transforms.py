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
from domain_transforms import DropDegenerateCells
from omegaconf import OmegaConf

from physicsnemo.datapipes.protocols import DatasetBase
from physicsnemo.datapipes.transforms.mesh import MeshToDomainMesh
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus import cell_measures, point_measures, set_cell_measures


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
    )


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
        # Populate the area cache before filtering.
        _ = mesh.cell_areas

        out = DropDegenerateCells()(mesh)

        assert out is mesh
        assert out.n_cells == 1
        torch.testing.assert_close(out.cell_data["pressure"], torch.tensor([10.0]))
        domain = MeshToDomainMesh(
            interior_points="cell_centroids", cell_data_targets=["pressure"]
        )(out)
        torch.testing.assert_close(
            point_measures(domain.interior),
            torch.tensor([height / 2]),
            atol=0,
            rtol=1e-6,
        )

    def test_filter_and_centroids_preserve_sampling_correction(self):
        """Filtering slices complete measures rather than replacing them with areas."""
        mesh = _two_triangles(second_degenerate=True)
        set_cell_measures(mesh, torch.tensor([2.0, 8.0]))
        with pytest.warns(UserWarning, match="dropping 1 cell"):
            out = DropDegenerateCells()(mesh)
        domain = MeshToDomainMesh(
            interior_points="cell_centroids", cell_data_targets=["pressure"]
        )(out)
        torch.testing.assert_close(point_measures(domain.interior), torch.tensor([2.0]))
        torch.testing.assert_close(
            domain.interior.integrate_samples("pressure"), torch.tensor(20.0)
        )

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


@pytest.mark.parametrize("augment", [False, True])
def test_drivaer_dataset_pipeline_preserves_thin_face_and_aligns_targets(
    tmp_path, augment
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
    dataset = build_dataset(cfg, augment=augment, device=None, num_workers=1)
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
    measures = point_measures(domain.interior)
    assert torch.isfinite(measures).all() and (measures > 0).all()
    torch.testing.assert_close(measures, cell_measures(boundary))
    # A measure-weighted loss must include both retained targets and backpropagate.
    prediction = torch.zeros(2, requires_grad=True)
    error = prediction - domain.interior.point_data["pressure"]
    loss = domain.interior.integrate_samples(error.square()) / measures.sum()
    weights = measures / measures.sum()
    torch.testing.assert_close(loss, (weights * torch.tensor([1.0, 9.0])).sum())
    loss.backward()
    torch.testing.assert_close(prediction.grad, 2 * weights * error.detach())
    assert "TimeValue" not in domain.global_data
