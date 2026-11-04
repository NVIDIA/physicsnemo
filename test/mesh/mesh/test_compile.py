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

"""Regression tests for `Mesh` under `torch.compile`.

These tests guard against the `tensordict` 0.12.x regression in PR
`pytorch/tensordict#1552`, where the @tensorclass init wrapper's bypass branch
silently skipped both field-default normalization (`pytorch/tensordict#1709`)
and ``__post_init__`` (`pytorch/tensordict#1708`) under ``torch.compile``.

The bug manifested as:

* ``Mesh(points=p, cells=c).cell_normals`` raising ``AttributeError`` inside a
  compiled function (the cached property could not find the ``_cache`` field
  that ``__post_init__`` was supposed to materialize from its ``None``
  default).
* Silent miscomputation of any property whose result depends on a field
  normalized in ``__post_init__``.

These tests construct a ``Mesh`` *inside* a ``torch.compile``-traced function
and assert that:

1. The compiled call does not raise.
2. The compiled output matches the eager output exactly.

If either upstream regression returns (e.g. via a future tensordict pin bump,
or a refactor of ``Mesh`` that loses the workaround), these tests fail loudly
instead of waiting for the notebook-level CI to break.
"""

import threading

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import set_cell_measures
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

### Fixtures ###


@pytest.fixture
def triangle_3d() -> tuple[torch.Tensor, torch.Tensor]:
    """A single right triangle in the XY-plane of 3D space.

    The triangle has vertices ``(0,0,0)``, ``(1,0,0)``, ``(0,1,0)``, so:

    * ``cell_normals == [[0, 0, 1]]`` (unit +Z)
    * ``cell_areas == [0.5]``
    * ``cell_centroids == [[1/3, 1/3, 0]]``

    Small enough that compile overhead dominates wall time, keeping the test
    cheap to run on every CI invocation.
    """
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    cells = torch.tensor([[0, 1, 2]])
    return points, cells


### Tests ###


@pytest.mark.parametrize(
    "property_name",
    [
        # Cached properties: each reads from `self._cache`, which is a field
        # defaulted to None and materialized in __post_init__. Broken by both
        # upstream regressions under tensordict 0.12.x.
        "cell_normals",
        "cell_areas",
        "cell_centroids",
        "point_normals",
    ],
)
def test_cached_property_under_compile(
    property_name: str,
    triangle_3d: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Cached `Mesh` properties must produce the same output eager vs compiled.

    Regression test for `pytorch/tensordict#1708` and `pytorch/tensordict#1709`.
    """
    points, cells = triangle_3d

    def fn(p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return getattr(Mesh(points=p, cells=c), property_name)

    expected = fn(points, cells)
    compiled = torch.compile(fn, fullgraph=False)(points, cells)

    torch.testing.assert_close(compiled, expected)


@pytest.mark.parametrize("field_name", ["point_data", "cell_data", "global_data"])
def test_data_field_under_compile(
    field_name: str,
    triangle_3d: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Data-container fields (``point_data``/``cell_data``/``global_data``)
    default to ``None`` in the schema and are normalized to empty
    ``TensorDict`` instances in ``__post_init__``. Accessing them inside a
    compiled function must not raise.

    Regression test for `pytorch/tensordict#1708` and `pytorch/tensordict#1709`.
    """
    points, cells = triangle_3d

    ### Read .n_<thing> through the field as a proxy for "field exists" ###
    # We don't compare to a numerical reference here; we just want to confirm
    # the compiled function doesn't blow up on attribute access.
    def fn(p: torch.Tensor, c: torch.Tensor) -> int:
        m = Mesh(points=p, cells=c)
        return len(getattr(m, field_name))

    expected = fn(points, cells)
    compiled = torch.compile(fn, fullgraph=False)(points, cells)

    assert compiled == expected


def test_post_init_runs_under_compile(
    triangle_3d: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Construct a ``Mesh`` inside a compiled function, mutate ``_cache``
    through the side-effecting ``cell_normals`` getter, and read it back.

    This exercises the full ``__post_init__`` -> cached-property -> cache-write
    -> cache-read round-trip. If ``__post_init__`` is silently skipped (the
    `#1708` regression), ``self._cache`` is missing entirely and the first
    cache access in the property body raises.
    """
    points, cells = triangle_3d

    def fn(p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        m = Mesh(points=p, cells=c)
        # First access triggers `__post_init__`-materialized `_cache` and
        # writes the result into it.
        return m.cell_normals

    expected = torch.tensor([[0.0, 0.0, 1.0]])
    compiled = torch.compile(fn, fullgraph=False)(points, cells)

    torch.testing.assert_close(compiled, expected)


@pytest.mark.parametrize(
    "property_name",
    ["cell_normals", "cell_areas", "cell_centroids"],
)
def test_local_cell_geometry_under_fullgraph_compile(
    property_name: str,
    triangle_3d: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Local cell geometry should be capturable as one complete graph.

    Constructing the nested cache must not trigger a nested ``TensorDict.to``
    while Dynamo is tracing the ``Mesh`` constructor.
    """
    points, cells = triangle_3d

    def fn(p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return getattr(Mesh(points=p, cells=c), property_name)

    expected = fn(points, cells)
    compiled = torch.compile(fn, backend="eager", fullgraph=True)(points, cells)

    torch.testing.assert_close(compiled, expected)


def test_cache_rebuild_paths_under_fullgraph_compile(
    triangle_3d: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Operations that rebuild cache containers remain full-graph capturable."""
    points, cells = triangle_3d

    def fn(p: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
        mesh = Mesh(points=p, cells=c)
        _ = mesh.cell_areas
        mesh = mesh.translate([1.0, 2.0, 3.0])
        mesh = mesh.transform(
            torch.eye(3, dtype=p.dtype, device=p.device), assume_invertible=True
        )
        mesh = mesh.slice_cells(torch.tensor([0], device=c.device))
        mesh = mesh.pad(target_n_points=4, target_n_cells=2)
        mesh = mesh.with_data(point_data={"coordinate": mesh.points[:, 0]})
        return mesh.points, mesh.cells, mesh.cell_areas, mesh.point_data["coordinate"]

    expected = fn(points, cells)
    compiled = torch.compile(fn, backend="eager", fullgraph=True)(points, cells)

    for actual, reference in zip(compiled, expected):
        torch.testing.assert_close(actual, reference)


def test_with_points_checks_point_count_under_fullgraph_compile(
    triangle_3d: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Compiled coordinate replacement must preserve point indexing."""
    points, cells = triangle_3d

    def fn(p: torch.Tensor, c: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
        return Mesh(points=p, cells=c).with_points(replacement).points

    compiled = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
    torch.testing.assert_close(compiled(points, cells, points.clone()), points)

    with pytest.raises(RuntimeError, match="must preserve point indexing"):
        compiled(points, cells, points[:-1])


def test_with_cells_checks_simplex_type_under_fullgraph_compile(
    triangle_3d: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Compiled connectivity replacement must preserve simplex type."""
    points, cells = triangle_3d

    def fn(p: torch.Tensor, c: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
        return Mesh(points=p, cells=c).with_cells(replacement).cells

    compiled = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
    torch.testing.assert_close(compiled(points, cells, cells.clone()), cells)

    tetrahedra = torch.cat([cells, cells[:, :1]], dim=1)
    with pytest.raises(RuntimeError, match="must preserve simplex type"):
        compiled(points, cells, tetrahedra)


@pytest.mark.parametrize("backend", ["eager", "inductor"])
@pytest.mark.parametrize("dtype", [torch.int64, torch.uint64])
def test_integer_coordinates_under_fullgraph_compile(backend, dtype):
    """Integer promotion stays in one graph and retains its exactness check."""
    points = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=dtype)
    cells = torch.tensor([[0, 1, 2]])

    def area(p, c):
        return Mesh(points=p, cells=c).cell_areas

    compiled = torch.compile(area, backend=backend, fullgraph=True)
    torch.testing.assert_close(compiled(points, cells), torch.tensor([0.5]).double())

    inexact = points.clone()
    inexact[0, 0] = 2**53 + 1
    with pytest.raises(RuntimeError, match="cannot be represented exactly"):
        compiled(inexact, cells)


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_uint64_connectivity_under_fullgraph_compile(backend):
    """Compiled normalization rejects overflow instead of using negative IDs."""
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells = torch.tensor([[0, 1, 2]], dtype=torch.uint64)

    def area(p, c):
        return Mesh(points=p, cells=c).cell_areas

    compiled = torch.compile(area, backend=backend, fullgraph=True)
    torch.testing.assert_close(compiled(points, cells), torch.tensor([0.5]))

    overflow = torch.tensor([[0, 1, 2**64 - 1]], dtype=torch.uint64)
    with pytest.raises(RuntimeError, match="uint64 indices.*int64"):
        compiled(points, overflow)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
@pytest.mark.parametrize("cells_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize(
    "points, cells",
    [
        pytest.param(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[0, 1, 2], [1, 3, 2]],
            id="triangles",
        ),
        pytest.param(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            + [[1.0, 1.0, 1.0]],
            [[0, 1, 2, 3], [1, 2, 3, 4]],
            id="tetrahedra",
        ),
    ],
)
@pytest.mark.parametrize(
    "method", ["point_to_points_adjacency", "facet_mesh", "boundary_mesh"]
)
def test_facet_extraction_under_dynamic_fullgraph_compile(
    method, points, cells, cells_dtype, device
):
    """Facet extraction compiles with symbolic shapes and matches eager mode.

    Regression test for NVIDIA/physicsnemo#2138, where ``dynamic=True`` made
    the cell width symbolic and ``itertools.combinations`` could not trace it.
    """
    points = torch.tensor(points, device=device)
    cells = torch.tensor(cells, dtype=cells_dtype, device=device)

    def fn(p: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
        mesh = Mesh(points=p, cells=c)
        if method == "point_to_points_adjacency":
            adjacency = mesh.get_point_to_points_adjacency()
            return adjacency.offsets, adjacency.indices
        return (getattr(mesh, f"get_{method}")().cells,)

    expected = fn(points, cells)
    compiled = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)(
        points, cells
    )

    for actual, reference in zip(compiled, expected, strict=True):
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize(
    "property_name",
    [
        "cell_centroids",
        "cell_areas",
        "cell_normals",
        "point_normals",
        "gaussian_curvature_vertices",
    ],
)
def test_geometric_property_ignores_cache_under_compile(property_name: str) -> None:
    """Compiled geometric properties neither guard on nor write into ``_cache``.

    A warm and a cold input mesh share one graph, and the compiled call leaves
    the input mesh's cache keys unchanged.
    """
    cold = sphere_icosahedral.load(subdivisions=1)
    warm = Mesh(points=cold.points * 2.0, cells=cold.cells)
    getattr(warm, property_name)  # eager read fills the cache
    other_cold = Mesh(points=cold.points * 3.0, cells=cold.cells)

    torch._dynamo.reset()
    compiled = torch.compile(
        lambda mesh: getattr(mesh, property_name),
        backend="eager",
        fullgraph=True,
        dynamic=True,
    )
    with torch._dynamo.config.patch(error_on_recompile=True):
        for mesh in (cold, warm, other_cold):
            keys = set(mesh._cache.keys(True, True))
            actual = compiled(mesh)
            assert set(mesh._cache.keys(True, True)) == keys
            torch.testing.assert_close(actual, getattr(mesh, property_name))


def _in_another_thread_while_compiling(fn):
    """Return ``fn()``, run in a second thread while this thread compiles."""
    result = {}

    def backend(gm, example_inputs):
        thread = threading.Thread(target=lambda: result.update(value=fn()))
        thread.start()
        thread.join()
        return gm.forward

    torch._dynamo.reset()
    torch.compile(lambda x: torch.sin(x) + 1, backend=backend)(torch.randn(4))
    return result["value"]


def _triangle(dx: float = 1.0) -> Mesh:
    """A float32 triangle whose first edge has length ``dx``."""
    points = torch.tensor([[0.0, 0.0, 0.0], [dx, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return Mesh(points=points, cells=torch.tensor([[0, 1, 2]]))


def test_cache_is_kept_in_another_thread_while_compiling() -> None:
    """A mesh used in another thread keeps its cache while this thread compiles.

    Since torch 2.13, ``torch.compiler.is_compiling()`` is True in every thread
    during a compile, so it must not decide whether a data-loading thread's own
    code is being traced.
    """

    def areas_cached() -> bool:
        mesh = _triangle()
        return mesh.cell_areas is mesh.cell_areas

    assert _in_another_thread_while_compiling(areas_cached)


def test_measure_survives_a_collapse_in_another_thread_while_compiling() -> None:
    """A sliver's measure stays valid after its float32 vertices collapse.

    The cached area carries the measure through a translation that makes two
    vertices equal in float32, so a later scale meets no zero-area cell with a
    nonzero measure, even when the mesh was built while another thread compiled.
    """

    def collapse() -> Mesh:
        mesh = _triangle(dx=1e-5)
        set_cell_measures(mesh, mesh.cell_areas)
        return mesh.translate(torch.tensor([1000.0, 0.0, 0.0]))

    moved = _in_another_thread_while_compiling(collapse)
    assert torch.equal(moved.points[0], moved.points[1])
    moved.scale(2.0, assume_invertible=True)


def test_validation_runs_in_another_thread_while_compiling() -> None:
    """`Mesh` validation still runs in another thread while this thread compiles.

    Since torch 2.13, ``torch.compiler.is_compiling()`` is True in every thread
    during a compile, so it must not decide whether a data-loading thread's own
    code is being traced.
    """

    def build_with_float_cells() -> Exception | None:
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        try:
            Mesh(points=points, cells=torch.tensor([[0.0, 1.0, 2.0]]))
        except TypeError as error:
            return error
        return None

    assert isinstance(
        _in_another_thread_while_compiling(build_with_float_cells), TypeError
    )
