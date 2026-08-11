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

"""Point-data transfer and adaptive-density tests for surface remeshing."""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import plane
from physicsnemo.mesh.remeshing import remesh


def _tilted_plane(subdivisions: int = 12) -> Mesh:
    """Create a planar triangle grid embedded obliquely in three dimensions."""
    base = plane.load(size=2.0, subdivisions=subdivisions)
    points = base.points.clone()
    points[:, 2] = 0.25 + 0.35 * points[:, 0] - 0.2 * points[:, 1]
    return Mesh(points=points, cells=base.cells.clone())


def _affine_field(points: torch.Tensor) -> torch.Tensor:
    """Evaluate a nonconstant affine scalar field in ambient coordinates."""
    coefficients = points.new_tensor([1.75, -0.8, 0.55])
    return 0.4 + points @ coefficients


def _leaf_keys(mesh: Mesh) -> set[str | tuple[str, ...]]:
    """Return the point-data leaf keys, including nested paths."""
    return set(mesh.point_data.keys(include_nested=True, leaves_only=True))


def test_remesh_transfers_affine_point_field_exactly():
    source = _tilted_plane(subdivisions=14)
    source.point_data["affine"] = _affine_field(source.points)

    output = remesh(
        source,
        64,
        max_iterations=2,
        transfer_point_data="affine",
    )

    assert _leaf_keys(output) == {"affine"}
    torch.testing.assert_close(
        output.point_data["affine"],
        _affine_field(output.points),
        rtol=2.0e-5,
        atol=2.0e-5,
    )


def test_remesh_transfers_selected_nested_point_data_only():
    source = _tilted_plane()
    source.point_data["flow", "pressure"] = _affine_field(source.points)
    source.point_data["flow", "unused"] = source.points[:, 0].square()
    source.point_data["temperature"] = 2.0 - source.points[:, 1]

    output = source.remesh(
        48,
        max_iterations=1,
        transfer_point_data=[("flow", "pressure"), "temperature"],
    )

    assert _leaf_keys(output) == {("flow", "pressure"), "temperature"}
    torch.testing.assert_close(
        output.point_data["flow", "pressure"],
        _affine_field(output.points),
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    torch.testing.assert_close(
        output.point_data["temperature"],
        2.0 - output.points[:, 1],
        rtol=2.0e-5,
        atol=2.0e-5,
    )


def test_remesh_accepts_single_component_tensor_dict_paths():
    source = _tilted_plane()
    source.point_data["field"] = _affine_field(source.points)
    source.point_data["resolution"] = torch.ones(source.n_points)

    output = source.remesh(
        48,
        max_iterations=1,
        transfer_point_data=("field",),
        resolution_field=("resolution",),
    )

    assert _leaf_keys(output) == {"field"}
    torch.testing.assert_close(
        output.point_data["field"],
        _affine_field(output.points),
        rtol=2.0e-5,
        atol=2.0e-5,
    )


def test_remesh_accepts_resolution_tensor_without_attaching_point_data():
    source = _tilted_plane(subdivisions=8)
    resolution = 1.0 + source.points[:, 0].square()

    direct = source.remesh(
        32,
        max_iterations=2,
        resolution_field=resolution,
    )

    assert _leaf_keys(source) == set()
    source.point_data["resolution"] = resolution
    attached = source.remesh(
        32,
        max_iterations=2,
        resolution_field="resolution",
    )
    torch.testing.assert_close(direct.points, attached.points)
    torch.testing.assert_close(direct.cells, attached.cells)


def test_remesh_point_data_transfer_preserves_field_autograd():
    source = _tilted_plane()
    source_values = (1.0 + source.points[:, 0].square()).detach().requires_grad_()
    source.point_data["design"] = source_values

    output = source.remesh(
        48,
        max_iterations=1,
        transfer_point_data="design",
        resolution_field="design",
    )
    output.point_data["design"].sum().backward()

    assert source_values.grad is not None
    assert torch.isfinite(source_values.grad).all()
    assert float(source_values.grad.min()) >= -5.0e-7
    torch.testing.assert_close(
        source_values.grad.sum(),
        source_values.new_tensor(output.n_points),
        rtol=1.0e-5,
        atol=1.0e-5,
    )


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [
        (torch.float16, 2.0e-3),
        (torch.bfloat16, 1.0e-2),
        (torch.float64, 2.0e-5),
    ],
)
def test_remesh_point_data_transfer_preserves_field_dtype(dtype, tolerance):
    source = _tilted_plane()
    source.point_data["field"] = _affine_field(source.points).to(dtype)

    output = source.remesh(
        48,
        max_iterations=1,
        transfer_point_data="field",
    )

    assert output.point_data["field"].dtype == dtype
    torch.testing.assert_close(
        output.point_data["field"].to(torch.float64),
        _affine_field(output.points).to(torch.float64),
        rtol=tolerance,
        atol=tolerance,
    )


def test_remesh_point_data_preservation_is_explicit_and_nonmutating():
    source = _tilted_plane()
    source.point_data["first"] = _affine_field(source.points)
    component_scales = source.points.new_tensor([0.5, 1.0, 2.0])
    source.point_data["second"] = (
        source.points[:, :2, None] * component_scales[None, None, :]
    )
    source.cell_data["region"] = torch.arange(source.n_cells)
    source.global_data["case_id"] = torch.tensor(17)
    original_first = source.point_data["first"].clone()
    original_second = source.point_data["second"].clone()

    discarded = source.remesh(48, max_iterations=1)
    transferred = source.remesh(
        48,
        max_iterations=1,
        transfer_point_data=True,
    )

    assert _leaf_keys(discarded) == set()
    assert len(discarded.cell_data.keys(include_nested=True, leaves_only=True)) == 0
    assert _leaf_keys(transferred) == {"first", "second"}
    assert len(transferred.cell_data.keys(include_nested=True, leaves_only=True)) == 0
    assert int(discarded.global_data["case_id"]) == 17
    assert int(transferred.global_data["case_id"]) == 17
    assert transferred.point_data["second"].shape == (transferred.n_points, 2, 3)
    torch.testing.assert_close(
        transferred.point_data["second"],
        transferred.points[:, :2, None] * component_scales[None, None, :],
    )
    torch.testing.assert_close(source.point_data["first"], original_first)
    torch.testing.assert_close(source.point_data["second"], original_second)
    assert "region" in source.cell_data


@pytest.mark.parametrize(
    "argument",
    [
        "missing",
        ["missing"],
        [("nested", "missing")],
    ],
    ids=["single-key", "key-list", "nested-key"],
)
def test_remesh_rejects_missing_transfer_point_data(argument):
    source = _tilted_plane(subdivisions=4)

    with pytest.raises(KeyError, match="missing"):
        source.remesh(
            16,
            transfer_point_data=argument,
        )


def test_remesh_rejects_missing_resolution_field():
    source = _tilted_plane(subdivisions=4)

    with pytest.raises(KeyError, match="missing"):
        source.remesh(
            16,
            resolution_field="missing",
        )


def test_remesh_rejects_invalid_resolution_field_type():
    source = _tilted_plane(subdivisions=4)

    with pytest.raises(TypeError, match="torch.Tensor, point_data key/path"):
        source.remesh(16, resolution_field=1.0)


def test_remesh_rejects_resolution_tensor_on_another_device():
    source = _tilted_plane(subdivisions=4)
    resolution = torch.ones(source.n_points, device="meta")

    with pytest.raises(ValueError, match="same device"):
        source.remesh(16, resolution_field=resolution)


def test_remesh_rejects_categorical_point_data_transfer():
    source = _tilted_plane(subdivisions=4)
    source.point_data["region"] = torch.zeros(source.n_points, dtype=torch.int64)

    with pytest.raises(TypeError, match="real floating-point"):
        source.remesh(
            16,
            transfer_point_data="region",
        )


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="float8 is unavailable in this PyTorch version",
)
def test_remesh_accepts_float8_fields():
    source = _tilted_plane(subdivisions=4)
    source.point_data["float8"] = (
        4.0 + 0.25 * source.points[:, 0] + 0.15 * source.points[:, 1]
    ).to(
        dtype=torch.float8_e4m3fn,
    )

    output = source.remesh(
        16,
        max_iterations=1,
        transfer_point_data="float8",
        resolution_field="float8",
    )

    assert output.point_data["float8"].dtype == torch.float8_e4m3fn
    expected = 4.0 + 0.25 * output.points[:, 0] + 0.15 * output.points[:, 1]
    torch.testing.assert_close(
        output.point_data["float8"].to(torch.float32),
        expected,
        rtol=0.15,
        atol=0.15,
    )


@pytest.mark.parametrize(
    ("case", "error", "match"),
    [
        ("nonscalar", ValueError, "shape"),
        ("integer", TypeError, "floating"),
        ("boolean", TypeError, "floating"),
        ("nan", ValueError, "resolution_field.*finite"),
        ("infinity", ValueError, "resolution_field.*finite"),
        ("zero", ValueError, "resolution_field.*strictly positive"),
        ("negative", ValueError, "resolution_field.*strictly positive"),
    ],
)
@pytest.mark.parametrize("direct", [False, True], ids=["point-data-key", "tensor"])
def test_remesh_rejects_invalid_resolution_field(case, error, match, direct):
    source = _tilted_plane(subdivisions=4)
    if case == "nonscalar":
        values = torch.ones(source.n_points, 2)
    elif case == "integer":
        values = torch.ones(source.n_points, dtype=torch.int64)
    elif case == "boolean":
        values = torch.ones(source.n_points, dtype=torch.bool)
    else:
        values = torch.ones(source.n_points)
        if case == "nan":
            values[0] = torch.nan
        elif case == "infinity":
            values[0] = torch.inf
        elif case == "zero":
            values[0] = 0.0
        elif case == "negative":
            values[0] = -1.0
    if direct:
        resolution_field = values
    else:
        source.point_data["density"] = values
        resolution_field = "density"

    with pytest.raises(error, match=match):
        source.remesh(
            16,
            resolution_field=resolution_field,
        )


def test_resolution_field_concentrates_vertices_in_high_resolution_region():
    source = plane.load(size=2.0, subdivisions=40)
    high_resolution_region = source.points[:, 0] > 0.35
    source.point_data["resolution"] = torch.where(
        high_resolution_region,
        source.points.new_tensor(2.0),
        source.points.new_tensor(1.0),
    )

    uniform = source.remesh(128, max_iterations=6)
    adaptive = source.remesh(
        128,
        max_iterations=6,
        resolution_field="resolution",
    )

    uniform_fraction = (uniform.points[:, 0] > 0.35).float().mean()
    adaptive_fraction = (adaptive.points[:, 0] > 0.35).float().mean()
    assert _leaf_keys(adaptive) == set()
    assert float(adaptive_fraction) > float(uniform_fraction) + 0.12


def test_resolution_field_reduces_equal_budget_field_preservation_error():
    source = plane.load(size=2.0, subdivisions=30)
    squared_radius = source.points[:, 0].square() + source.points[:, 1].square()
    source.point_data["field"] = torch.exp(-0.5 * squared_radius / 0.32**2)
    source.point_data["resolution"] = 1.0 + 3.0 * source.point_data["field"]

    uniform = source.remesh(
        120,
        max_iterations=4,
        transfer_point_data="field",
    )
    adaptive = source.remesh(
        120,
        max_iterations=4,
        transfer_point_data="field",
        resolution_field="resolution",
    )

    axis = torch.linspace(-0.75, 0.75, 31)
    y_grid, x_grid = torch.meshgrid(axis, axis, indexing="ij")
    query_points = torch.stack(
        (
            x_grid.flatten(),
            y_grid.flatten(),
            torch.zeros(x_grid.numel()),
        ),
        dim=1,
    )
    reference_field = source.sample_data_at_points(
        query_points,
        data_source="points",
    )["field"]
    uniform_field = uniform.sample_data_at_points(
        query_points,
        data_source="points",
    )["field"]
    adaptive_field = adaptive.sample_data_at_points(
        query_points,
        data_source="points",
    )["field"]

    valid = (
        torch.isfinite(reference_field)
        & torch.isfinite(uniform_field)
        & torch.isfinite(adaptive_field)
    )
    assert bool(valid.all())
    uniform_rmse = (
        (uniform_field[valid] - reference_field[valid]).square().mean().sqrt()
    )
    adaptive_rmse = (
        (adaptive_field[valid] - reference_field[valid]).square().mean().sqrt()
    )
    assert float(adaptive_rmse) < 0.8 * float(uniform_rmse)


@pytest.mark.parametrize("n_clusters", [256, 257, 512])
def test_resolution_field_remains_effective_across_initializer_threshold(n_clusters):
    source = plane.load(size=2.0, subdivisions=80)
    source.point_data["resolution"] = torch.where(
        source.points[:, 0] > 0.0,
        source.points.new_tensor(2.0),
        source.points.new_tensor(1.0),
    )

    adaptive = source.remesh(
        n_clusters,
        max_iterations=4,
        resolution_field="resolution",
    )

    high_resolution_fraction = (adaptive.points[:, 0] > 0.0).float().mean()
    cells = adaptive.cells.to(torch.int64)
    edges = torch.cat(
        (
            cells[:, (0, 1)],
            cells[:, (1, 2)],
            cells[:, (2, 0)],
        )
    )
    edges = torch.unique(torch.sort(edges, dim=1).values, dim=0)
    starts = adaptive.points[edges[:, 0]]
    ends = adaptive.points[edges[:, 1]]
    midpoints = 0.5 * (starts + ends)
    edge_lengths = torch.linalg.vector_norm(ends - starts, dim=1)
    low_resolution_edge = edge_lengths[midpoints[:, 0] < -0.15].median()
    high_resolution_edge = edge_lengths[midpoints[:, 0] > 0.15].median()

    assert float(high_resolution_fraction) > 0.72
    assert float(low_resolution_edge / high_resolution_edge) > 1.55


def test_constant_resolution_field_matches_uniform_remeshing():
    source = plane.load(size=2.0, subdivisions=20)
    source.point_data["resolution"] = torch.full(
        (source.n_points,),
        7.5,
        dtype=source.points.dtype,
    )

    uniform = source.remesh(64, max_iterations=4)
    controlled = source.remesh(
        64,
        max_iterations=4,
        resolution_field="resolution",
    )

    torch.testing.assert_close(controlled.points, uniform.points)
    torch.testing.assert_close(controlled.cells, uniform.cells)


def test_resolution_field_is_scale_invariant():
    source = plane.load(size=2.0, subdivisions=20)
    source.point_data["resolution"] = torch.where(
        source.points[:, 0] > 0.35,
        source.points.new_tensor(2.0),
        source.points.new_tensor(1.0),
    )

    reference = source.remesh(
        64,
        max_iterations=4,
        resolution_field="resolution",
    )
    source.point_data["resolution"].mul_(7.5)
    scaled = source.remesh(
        64,
        max_iterations=4,
        resolution_field="resolution",
    )

    torch.testing.assert_close(scaled.points, reference.points)
    torch.testing.assert_close(scaled.cells, reference.cells)
