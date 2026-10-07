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

"""Contracts for explicit measures across point and cell representations."""

import pytest
import torch

from physicsnemo.datapipes.readers.mesh import _subsample_mesh_points
from physicsnemo.datapipes.transforms.mesh import (
    MeshToDomainMesh,
    ScaleMesh,
    SubsampleMesh,
)
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import (
    EFFECTIVE_MEASURE_KEY,
    POINT_MEASURE_DIMENSION_KEY,
    cell_measures,
    lumped_point_measures,
    point_measures,
    scale_measures,
    set_cell_measures,
    set_point_measures,
)
from physicsnemo.mesh.primitives.basic import two_triangles_2d


def simplex(dimension, *, spatial_dimension=3):
    points = torch.cat(
        [torch.zeros(1, spatial_dimension), torch.eye(spatial_dimension)[:dimension]]
    )
    return Mesh(points=points, cells=torch.arange(dimension + 1).unsqueeze(0))


@pytest.mark.parametrize("connected", [False, True])
def test_point_quadrature_never_infers_a_measure(connected):
    source = two_triangles_2d.load()
    mesh = source if connected else Mesh(points=source.points)
    with pytest.raises(KeyError, match="explicit effective measures"):
        point_measures(mesh)
    with pytest.raises(KeyError, match="explicit effective measures"):
        mesh.integrate_samples(torch.ones(mesh.n_points))

    values = torch.arange(mesh.n_points, dtype=mesh.points.dtype) + 1
    weights = torch.arange(mesh.n_points, dtype=mesh.points.dtype) + 2
    set_point_measures(mesh, weights, dimension=2)
    torch.testing.assert_close(mesh.integrate_samples(values), (values * weights).sum())
    assert EFFECTIVE_MEASURE_KEY not in mesh.cell_data


def test_nodal_and_sample_integration_are_explicitly_distinct():
    mesh = two_triangles_2d.load()
    scale_measures(mesh, torch.tensor([2.0, 3.0]))
    values = torch.arange(mesh.n_points, dtype=mesh.points.dtype)
    p1 = mesh.integrate(values, data_source="points")
    set_point_measures(mesh, torch.ones(mesh.n_points), dimension=0)
    torch.testing.assert_close(mesh.integrate_samples(values), values.sum())
    torch.testing.assert_close(mesh.integrate(values, data_source="points"), p1)
    set_point_measures(mesh, lumped_point_measures(mesh), dimension=2)
    torch.testing.assert_close(mesh.integrate_samples(values), p1)
    torch.testing.assert_close(point_measures(mesh).sum(), cell_measures(mesh).sum())


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("factor", [-3.0, 0.0, 2.0])
@pytest.mark.parametrize("transform_fields", [False, True])
def test_centroid_conversion_commutes_with_uniform_scaling(
    dimension, factor, transform_fields
):
    source = simplex(dimension)
    source.cell_data["target"] = torch.tensor([7.0])
    scale_measures(source, 2.5)
    convert = MeshToDomainMesh(cell_data_targets=["target"])
    domain = convert(source)
    moved = ScaleMesh(
        factor,
        transform_point_data=transform_fields,
        transform_cell_data=transform_fields,
    ).apply_to_domain(domain)
    expected = cell_measures(source) * abs(factor) ** dimension
    torch.testing.assert_close(point_measures(moved.interior), expected)
    torch.testing.assert_close(cell_measures(moved.boundaries["vehicle"]), expected)
    torch.testing.assert_close(
        point_measures(convert(source.scale(factor)).interior), expected
    )
    torch.testing.assert_close(point_measures(domain.interior), cell_measures(source))
    assert int(moved.interior.global_data[POINT_MEASURE_DIMENSION_KEY]) == dimension


def test_cell_measures_follow_anisotropic_deformation():
    mesh = simplex(2)
    scale_measures(mesh, 3.0)
    moved = mesh.scale([2.0, 3.0, 4.0])
    torch.testing.assert_close(cell_measures(moved), moved.cell_areas * 3.0)
    displaced = mesh.with_points(mesh.points * torch.tensor([2.0, 3.0, 4.0]))
    torch.testing.assert_close(cell_measures(displaced), cell_measures(moved))


def test_point_support_is_required_for_anisotropic_surface_scaling():
    domain = MeshToDomainMesh()(simplex(2))
    before = point_measures(domain.interior).clone()
    with pytest.raises(ValueError, match="support geometry"):
        domain.scale([2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="preservation policy"):
        domain.interior.with_points(domain.interior.points * 2)
    fixed = domain.interior.with_points(
        domain.interior.points * 2, preserve_measures=True
    )
    torch.testing.assert_close(point_measures(fixed), before)
    torch.testing.assert_close(point_measures(domain.interior), before)


def test_counting_and_volume_point_measures_have_known_affine_scaling():
    cloud = Mesh(points=torch.randn(5, 3))
    set_point_measures(cloud, torch.ones(5), dimension=0)
    torch.testing.assert_close(
        point_measures(cloud.scale([2.0, 3.0, 4.0])), torch.ones(5)
    )
    torch.testing.assert_close(
        point_measures(cloud.with_points(cloud.points + 1)), torch.ones(5)
    )
    set_point_measures(cloud, torch.full((5,), 2.0), dimension=3)
    torch.testing.assert_close(
        point_measures(cloud.scale([2.0, 3.0, 4.0])), torch.full((5,), 48.0)
    )


def test_rigid_transforms_and_dtype_preserve_point_measures():
    cloud = MeshToDomainMesh()(simplex(2)).interior
    moved = (
        cloud.rotate(0.7, axis="x").translate([1.0, 2.0, 3.0]).to(dtype=torch.float64)
    )
    torch.testing.assert_close(point_measures(moved), point_measures(cloud).double())
    torch.testing.assert_close(
        point_measures(moved.scale(2.0)), point_measures(cloud).double() * 4
    )


@pytest.mark.parametrize("filter", ["linear", "loop", "butterfly"])
def test_subdivision_distributes_complete_cell_measures(filter):
    mesh = two_triangles_2d.load()
    scale_measures(mesh, 3.0)
    refined = mesh.subdivide(levels=2, filter=filter)
    torch.testing.assert_close(cell_measures(refined), refined.cell_areas * 3.0)
    if filter == "linear":
        torch.testing.assert_close(
            cell_measures(refined).sum(), cell_measures(mesh).sum()
        )
    set_point_measures(mesh, torch.ones(mesh.n_points), dimension=0)
    with pytest.raises(ValueError, match="cannot be interpolated"):
        mesh.subdivide(filter=filter)


@pytest.mark.parametrize("reader", [False, True])
def test_point_sampling_reweights_only_explicit_quadrature(reader):
    cloud = Mesh(points=torch.randn(10, 3))
    set_point_measures(cloud, torch.ones(10), dimension=0)
    if reader:
        sampled = _subsample_mesh_points(cloud, 4, torch.Generator().manual_seed(1))
    else:
        sampled = SubsampleMesh(n_points=4)(cloud)
    torch.testing.assert_close(point_measures(sampled), torch.full((4,), 2.5))
    torch.testing.assert_close(point_measures(cloud), torch.ones(10))
    ordinary = SubsampleMesh(n_points=4)(Mesh(points=cloud.points))
    with pytest.raises(KeyError):
        point_measures(ordinary)


def test_measure_storage_survives_serialization_slice_and_merge(tmp_path):
    cloud = Mesh(points=torch.randn(5, 3), point_data={"value": torch.arange(5.0)})
    set_point_measures(cloud, torch.arange(5.0) + 1, dimension=2)
    cloud.save(tmp_path / "cloud.pmsh")
    loaded = Mesh.load(tmp_path / "cloud.pmsh")
    selected = loaded.slice_points([1, 3])
    torch.testing.assert_close(point_measures(selected), torch.tensor([2.0, 4.0]))
    merged = Mesh.merge([selected, selected])
    torch.testing.assert_close(
        point_measures(merged.scale(2.0)), torch.tensor([8.0, 16.0, 8.0, 16.0])
    )
    assert merged.global_data[POINT_MEASURE_DIMENSION_KEY].shape == ()
    incompatible = selected.clone()
    set_point_measures(incompatible, point_measures(incompatible), dimension=1)
    with pytest.raises(ValueError, match="different measure dimensions"):
        Mesh.merge([selected, incompatible])


def test_centroid_conversion_and_vertex_conversion_preserve_existing_measures():
    source = simplex(2)
    set_cell_measures(source, torch.tensor([5.0]))
    set_point_measures(source, torch.tensor([1.0, 2.0, 3.0]), dimension=2)
    vertices = MeshToDomainMesh(interior_points="vertices")(source)
    torch.testing.assert_close(
        point_measures(vertices.interior), point_measures(source)
    )
    centroids = MeshToDomainMesh()(source)
    torch.testing.assert_close(point_measures(centroids.interior), torch.tensor([5.0]))
    torch.testing.assert_close(
        point_measures(centroids.boundaries["vehicle"]), point_measures(source)
    )


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("dual", [False, True])
def test_centroid_mesh_conversions_transfer_measures_without_mutating_source(
    explicit, dual
):
    """Centroid quadrature keeps the source dimension, including on dual graphs."""
    source = two_triangles_2d.load()
    source.cell_data["value"] = torch.tensor([2.0, 5.0])
    if explicit:
        scale_measures(source, torch.tensor([3.0, 7.0]))
    # Source vertex quadrature is independent of the new centroid quadrature.
    set_point_measures(source, torch.ones(source.n_points), dimension=0)
    expected = cell_measures(source).clone()
    converted = (
        source.to_dual_graph()
        if dual
        else source.to_point_cloud(point_source="cell_centroids")
    )
    torch.testing.assert_close(point_measures(converted), expected)
    torch.testing.assert_close(
        converted.integrate_samples("value"), source.integrate("value")
    )
    torch.testing.assert_close(point_measures(converted.scale(2.0)), expected * 4)
    assert int(converted.global_data[POINT_MEASURE_DIMENSION_KEY]) == 2
    assert int(source.global_data[POINT_MEASURE_DIMENSION_KEY]) == 0
    assert (EFFECTIVE_MEASURE_KEY in source.cell_data) == explicit
    if dual:
        torch.testing.assert_close(cell_measures(converted), converted.cell_areas)
    scale_measures(converted, 2.0, association="points")
    torch.testing.assert_close(cell_measures(source), expected)


def test_generic_field_interpolation_does_not_interpolate_measures():
    mesh = simplex(2)
    set_cell_measures(mesh, torch.tensor([7.0]))
    mesh.cell_data["f"] = torch.tensor([3.0])
    converted = mesh.cell_data_to_point_data()
    assert EFFECTIVE_MEASURE_KEY not in converted.point_data
    torch.testing.assert_close(converted.point_data["f"], torch.full((3,), 3.0))
    set_point_measures(mesh, torch.ones(3), dimension=2)
    mesh.point_data["g"] = torch.tensor([1.0, 2.0, 3.0])
    converted = mesh.point_data_to_cell_data()
    torch.testing.assert_close(cell_measures(converted), torch.tensor([7.0]))
    torch.testing.assert_close(converted.cell_data["g"], torch.tensor([2.0]))


def test_measure_validation_rejects_ambiguous_shapes_and_dimensions():
    mesh = simplex(2)
    with pytest.raises(ValueError, match="one scalar"):
        set_cell_measures(mesh, torch.ones(1, 1))
    with pytest.raises(ValueError, match="one scalar"):
        set_point_measures(mesh, torch.ones(3, 1), dimension=2)
    with pytest.raises(ValueError, match="dimension"):
        set_point_measures(mesh, torch.ones(3), dimension=4)
    set_point_measures(mesh, torch.ones(3), dimension=2)
    before = point_measures(mesh).clone()
    with pytest.raises(ValueError, match="scalar or have shape"):
        scale_measures(mesh, torch.ones(3, 1), association="points")
    torch.testing.assert_close(point_measures(mesh), before)


def test_differentiable_sample_quadrature():
    cloud = Mesh(points=torch.zeros(3, 2))
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    measures = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
    set_point_measures(cloud, measures, dimension=2)
    cloud.integrate_samples(values).sum().backward()
    torch.testing.assert_close(
        values.grad, measures.detach()[:, None].expand_as(values)
    )
    torch.testing.assert_close(measures.grad, values.detach().sum(-1))


def test_empty_sample_integral_and_nan_omission():
    cloud = Mesh(points=torch.empty(0, 3))
    set_point_measures(cloud, torch.empty(0), dimension=2)
    torch.testing.assert_close(
        cloud.integrate_samples(torch.empty(0, 2)), torch.zeros(2)
    )
    cloud = Mesh(points=torch.zeros(3, 2))
    set_point_measures(cloud, torch.tensor([1.0, 2.0, 3.0]), dimension=2)
    values = torch.tensor([1.0, float("nan"), 4.0])
    torch.testing.assert_close(cloud.integrate_samples(values), torch.tensor(13.0))
    assert torch.isnan(cloud.integrate_samples(values, nan_policy="propagate"))


def test_legacy_multiplier_cannot_silently_become_geometric_fallback():
    mesh = simplex(2)
    mesh.cell_data["_measure_weights"] = torch.tensor([3.0])
    with pytest.raises(ValueError, match="must be converted"):
        cell_measures(mesh)


def test_remeshing_requires_a_measure_transfer_rule():
    mesh = simplex(2)
    set_cell_measures(mesh, torch.tensor([1.0]))
    with pytest.raises(ValueError, match="conservative measure transfer"):
        mesh.remesh(n_clusters=3)


def test_partition_accumulates_effective_measures_without_overriding_geometry():
    from physicsnemo.mesh.remeshing import partition_cells

    mesh = two_triangles_2d.load()
    geometric = mesh.cell_areas.clone()
    set_cell_measures(mesh, torch.tensor([2.0, 5.0]))
    partition = partition_cells(mesh, seeds=mesh.cell_centroids)
    torch.testing.assert_close(partition.cluster_areas, torch.tensor([2.0, 5.0]))
    torch.testing.assert_close(mesh.cell_areas, geometric)
