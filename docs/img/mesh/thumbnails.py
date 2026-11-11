# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render one 800 x 500 card thumbnail per mesh docs page, one object each, no text."""

import math
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import _style as S  # noqa: E402

from physicsnemo.mesh import Mesh  # noqa: E402
from physicsnemo.mesh.generate import (  # noqa: E402
    mesh_implicit_domain,
    sdf_box,
    sdf_difference,
    sdf_sphere,
)
from physicsnemo.mesh.geometry.dual_meshes import compute_circumcenters  # noqa: E402
from physicsnemo.mesh.io import from_pyvista, to_pyvista  # noqa: E402
from physicsnemo.mesh.neighbors import Adjacency  # noqa: E402
from physicsnemo.mesh.primitives.planar import (  # noqa: E402
    annulus_2d,
    circle_2d,
    l_shape,
    unit_square,
)
from physicsnemo.mesh.primitives.procedural import (  # noqa: E402
    lumpy_sphere,
    perturbed_grid,
)
from physicsnemo.mesh.primitives.surfaces import (  # noqa: E402
    cube_surface,
    mobius_strip,
    plane,
    sphere_icosahedral,
    torus,
)
from physicsnemo.mesh.projections import embed  # noqa: E402
from physicsnemo.mesh.repair import fill_holes  # noqa: E402
from physicsnemo.mesh.spatial import signed_distance_field  # noqa: E402

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "thumbnails"

SIZE = (800, 500)  # final card size
RENDER = (1280, 800)  # off-screen window, cropped to the object and fitted
MARGIN = 0.07  # white border on every side, as a fraction of the card height
SAMPLE_GRAY = "#cfd5da"  # neutral surface under colored sample points
WHITE_RIM = "#ffffff"

torch.manual_seed(0)


### Building blocks


def _pv(mesh):
    """Convert a Mesh to PyVista, padding 2D points with z = 0."""

    return to_pyvista(mesh)


def _xyz(points):
    """Return an (n, 3) numpy array, padding 2D points with z = 0."""

    points = torch.as_tensor(points).detach().float()
    if points.shape[-1] == 2:
        points = torch.nn.functional.pad(points, (0, 1))
    return points.cpu().numpy()


def _edges(mesh, radius, z=0.0):
    """Tubes along the unique edges of a mesh."""

    graph = mesh.to_edge_graph() if mesh.n_manifold_dims > 1 else mesh
    dataset = _pv(graph)
    if z:
        dataset.points[:, 2] += z
    return dataset.tube(radius=radius, n_sides=12)


def _dots(points, radius, color, rim=True, z=0.0):
    """Points as spheres. In top-down 2D views, a white rim sits under each dot."""

    xyz = _xyz(points)
    actors = []
    if rim:
        under = xyz.copy()
        under[:, 2] += z
        actors.append((S.spheres(under, 1.3 * radius), _flat(WHITE_RIM)))
    over = xyz.copy()
    over[:, 2] += z + 1.6 * radius
    actors.append((S.spheres(over, radius), _flat(color) if rim else _shaded(color)))
    return actors


def _flat(color, **kwargs):
    """Unlit flat color, for top-down 2D views."""

    return {"color": color, "lighting": False, **kwargs}


def _shaded(color, **kwargs):
    """Shaded color for 3D views. Diffuse 0.55 keeps saturated accents on their hue."""

    return {"color": color, "ambient": 0.5, "diffuse": 0.55, "specular": 0.0, **kwargs}


def _surface(mesh, color=S.SURFACE, edges=True, **kwargs):
    """A shaded 3D surface with its cell edges."""

    return (
        _pv(mesh),
        _shaded(
            color,
            show_edges=edges,
            edge_color=S.SURFACE_EDGE,
            line_width=1.0,
            **kwargs,
        ),
    )


def _arrows(points, vectors, factor, color, z=0.0, lit=False):
    """Arrow glyphs for a vector field."""

    cloud = pv.PolyData(_xyz(points))
    cloud.points[:, 2] += z
    cloud["v"] = _xyz(vectors)
    glyphs = cloud.glyph(orient="v", scale="v", factor=factor, geom=pv.Arrow())
    return (glyphs, _shaded(color) if lit else _flat(color))


def _sheet():
    """A gently curved Mesh[2, 3] sheet, as in the graph figures."""

    grid = perturbed_grid.load(n_x=6, n_y=6, perturbation_scale=0.04, seed=0)
    grid = embed(grid, target_n_spatial_dims=3)
    x, y, _ = (grid.points - 0.5).unbind(dim=-1)
    bump = 0.32 * torch.exp(-(x**2 + y**2) / 0.18)
    return grid.with_points(grid.points + bump[:, None] * torch.tensor([0.0, 0.0, 1.0]))


### One function per thumbnail. Each returns (actors, view), view "2d" or "3d".


def core():
    """Points and cells: a small planar mesh with one cell picked out."""

    points = torch.tensor(
        [[0.0, 0.0], [1.0, -0.15], [2.0, 0.05], [0.45, 0.95], [1.5, 0.9], [1.0, 0.45]]
    )
    cells = torch.tensor([[0, 1, 5], [1, 2, 4], [1, 4, 5], [0, 5, 3], [3, 5, 4]])
    mesh = Mesh(points=points, cells=cells)
    actors = [
        (_pv(mesh), _flat(S.SURFACE)),
        (_pv(mesh.slice_cells([2])), _flat(S.RESULT)),
        (_edges(mesh, 0.012, z=0.01), _flat(S.SURFACE_EDGE)),
        *_dots(mesh.points, 0.045, S.SECOND, z=0.02),
    ]
    return actors, "2d"


def fields():
    """A scalar field and a vector field, rotated together by 30 degrees."""

    mesh = unit_square.load(subdivisions=3).translate([-0.5, -0.5])
    x, y = mesh.points.unbind(dim=-1)
    mesh.point_data["pressure"] = x + y**2
    mesh.point_data["velocity"] = torch.stack([0.6 + 0.2 * y, 0.3 * x], dim=-1)
    rotated = mesh.rotate(math.radians(30.0), transform_point_data=True)
    square = Mesh(
        points=torch.tensor([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]),
        cells=torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]]),
    )
    surface = _pv(rotated)
    surface["pressure"] = rotated.point_data["pressure"].numpy()
    actors = [
        (_edges(square, 0.006, z=-0.01), _flat("#555555")),
        (
            surface,
            _flat(None, scalars="pressure", cmap=S.SEQUENTIAL, show_scalar_bar=False),
        ),
        (_edges(rotated, 0.004, z=0.005), _flat(S.EDGE)),
        _arrows(
            rotated.points, rotated.point_data["velocity"], 0.22, "#ffffff", z=0.02
        ),
    ]
    return actors, "2d"


def primitives():
    """A Mobius strip from the primitives catalog."""

    return [_surface(mobius_strip.load(n_circ=64, n_width=6))], "3d"


def io():
    """A voxel grid read through from_pyvista and split into tetrahedra."""

    grid = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    mesh = from_pyvista(grid)
    tets = _pv(mesh).shrink(0.78)
    return [
        (S.tubes(grid, 0.012), _shaded(S.EDGE)),
        (tets, _shaded(S.SURFACE, show_edges=True, edge_color=S.SURFACE_EDGE)),
    ], "3d"


def generate():
    """mesh_implicit_domain on a box minus a disk, with the boundary in vermillion."""

    phi = sdf_difference(
        sdf_box([-1.6, -1.0], [1.6, 1.0]), sdf_sphere([0.35, 0.0], 0.5)
    )
    mesh = mesh_implicit_domain(phi, ([-1.6, -1.0], [1.6, 1.0]), 0.12)
    return [
        (_pv(mesh), _flat(S.SURFACE)),
        (_edges(mesh, 0.006, z=0.01), _flat(S.SURFACE_EDGE)),
        (_edges(mesh.get_boundary_mesh(), 0.022, z=0.02), _flat(S.RESULT)),
    ], "2d"


def tessellation():
    """A polygon soup triangulated, one color per source polygon."""

    quad = [[0.0, 0.0], [1.0, 0.0], [1.1, 1.0], [0.0, 0.9]]
    hexagon = [
        [2.0 + 0.6 * math.cos(a), 0.45 + 0.6 * math.sin(a)]
        for a in np.linspace(0, 2 * math.pi, 7)[:-1]
    ]
    arrow = [
        [3.0, 0.25], [3.7, 0.25], [3.7, -0.1], [4.4, 0.45],
        [3.7, 1.0], [3.7, 0.65], [3.0, 0.65],
    ]  # fmt: skip
    loops = (quad, hexagon, arrow)
    points = torch.tensor([p for loop in loops for p in loop])
    sizes = torch.tensor([len(loop) for loop in loops])
    polygons = Adjacency(
        offsets=torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)]),
        indices=torch.arange(len(points)),
    )
    mesh = Mesh.from_polygons(
        points, polygons, cell_data={"polygon": torch.arange(len(loops))}
    )
    actors = []
    for index, color in enumerate((S.SECOND_FILL, S.FOURTH, S.FIXED)):
        part = mesh.slice_cells(mesh.cell_data["polygon"] == index)
        actors.append((_pv(part), _flat(color)))
    actors.append((_edges(mesh, 0.012, z=0.01), _flat("#ffffff")))
    outline = torch.cat(
        [torch.stack([o + torch.arange(n), o + (torch.arange(n) + 1) % n], 1)
         for o, n in zip(polygons.offsets[:-1].tolist(), sizes.tolist())]
    )  # fmt: skip
    actors.append(
        (
            _edges(Mesh(points=points, cells=outline), 0.022, z=0.02),
            _flat(S.SURFACE_EDGE),
        )
    )
    return actors, "2d"


def transformations():
    """A flat sheet morphed by one control handle."""

    sheet = plane.load(size=2.0, subdivisions=16)
    handle = torch.tensor([[0.0, 0.0, 0.0]])
    lift = torch.tensor([[0.0, 0.0, 0.7]])
    morphed = sheet.morph(handle, lift, radius=0.9)
    return [
        _surface(morphed),
        (S.spheres((handle + lift).numpy(), 0.085), _shaded(S.RESULT, ambient=0.6)),
    ], "3d"


def deformation():
    """A sphere pulled by a handle while anchors stay fixed, as in the energy figures."""

    sphere = sphere_icosahedral.load(subdivisions=3)
    anchors = torch.tensor(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    handle = torch.tensor([[0.0, 0.0, 1.0]])
    controls = torch.cat([anchors, handle])
    displacements = torch.zeros_like(controls)
    displacements[-1, 2] = 0.6
    pulled = sphere.radial_basis_function_deform(controls, displacements)
    return [
        _surface(pulled),
        (S.spheres((anchors * 1.02).numpy(), 0.085), _shaded(S.FIXED, ambient=0.6)),
        (
            S.spheres((handle + displacements[-1:]).numpy() * 1.02, 0.1),
            _shaded(S.RESULT, ambient=0.6),
        ),
    ], "3d"


def remeshing():
    """A fine lumpy sphere remeshed to fewer, evenly sized triangles."""

    from physicsnemo.mesh.remeshing import remesh

    fine = lumpy_sphere.load(subdivisions=5, noise_amplitude=0.3, seed=0)
    coarse = remesh(fine, 600)
    return [_surface(coarse, color=S.RESULT)], "3d"


def subdivision():
    """Loop subdivision of a cube, inside the coarse cage."""

    cube = cube_surface.load()
    smooth = cube.subdivide(levels=3, filter="loop")
    cage = _pv(cube.to_edge_graph()).tube(radius=0.012)
    return [
        _surface(smooth),
        (cage, _shaded(S.RESULT)),
        (S.spheres(cube.points.numpy(), 0.035), _shaded(S.RESULT)),
    ], "3d"


def smoothing():
    """One noisy sphere, smoothed by smooth_laplacian on the right half only."""

    from physicsnemo.mesh.smoothing import smooth_laplacian

    sphere = sphere_icosahedral.load(subdivisions=3)
    generator = torch.Generator().manual_seed(0)
    noise = 1.0 + 0.025 * torch.randn(sphere.n_points, 1, generator=generator)
    rough = sphere.with_points(sphere.points * noise)
    smooth = smooth_laplacian(rough, n_iter=40, relaxation_factor=0.3)
    # Blend from rough (left) to smooth (right), so the surface stays one piece.
    weight = torch.sigmoid(10.0 * (rough.points @ torch.tensor([1.0, 0.7, 0.0])))
    blended = rough.with_points(
        rough.points + weight[:, None] * (smooth.points - rough.points)
    )
    return [_surface(blended)], "3d"


def repair():
    """A sphere with a hole, and the cells that fill_holes adds."""

    sphere = sphere_icosahedral.load(subdivisions=2)
    view = torch.tensor(S.VIEW_VECTOR) / torch.tensor(S.VIEW_VECTOR).norm()
    centroids = sphere.cell_centroids / sphere.cell_centroids.norm(dim=-1, keepdim=True)
    holed = sphere.slice_cells(centroids @ view < 0.93).clean()
    filled, _ = fill_holes(holed, max_hole_edges=40)
    added = torch.arange(filled.n_cells) >= holed.n_cells
    return [
        _surface(filled.slice_cells(~added)),
        _surface(filled.slice_cells(added), color=S.RESULT),
    ], "3d"


def validation():
    """A jittered grid colored by quality_score."""

    mesh = perturbed_grid.load(n_x=13, n_y=9, perturbation_scale=0.015, seed=2)
    mesh = mesh.with_points(mesh.points * torch.tensor([1.6, 1.0]))
    surface = _pv(mesh)
    surface.cell_data["quality"] = mesh.quality_metrics["quality_score"].numpy()
    return [
        (surface, _flat(None, scalars="quality", cmap=S.ERROR, show_scalar_bar=False)),
        (_edges(mesh, 0.003, z=0.005), _flat("#ffffff")),
    ], "2d"


def boundaries():
    """An annulus and its boundary mesh in vermillion."""

    mesh = annulus_2d.load(n_radial=3, n_angular=24)
    return [
        (_pv(mesh), _flat(S.FAINT)),
        (_edges(mesh, 0.006, z=0.005), _flat(S.FAINT_EDGE)),
        (_edges(mesh.get_boundary_mesh(), 0.022, z=0.01), _flat(S.RESULT)),
    ], "2d"


def geometry():
    """Circumcentric dual edges over a triangle mesh, with one dual cell filled."""

    mesh = mesh_implicit_domain(
        sdf_box([-1.5, -1.0], [1.5, 1.0]), ([-1.5, -1.0], [1.5, 1.0]), 0.6
    )
    centers = compute_circumcenters(mesh.points[mesh.cells])
    source, target = mesh.get_cell_to_cells_adjacency().expand_to_pairs()
    pairs = torch.stack([source, target], 1)
    dual = Mesh(points=centers, cells=pairs[pairs[:, 0] < pairs[:, 1]])
    # Dual cell of the interior point nearest the middle: its cells' circumcenters in angular order.
    middle = mesh.points.mean(0)
    vertex = int((mesh.points - middle).norm(dim=-1).argmin())
    ring = mesh.get_point_to_cells_adjacency().to_list()[vertex]
    polygon = centers[ring]
    order = torch.atan2(*(polygon - mesh.points[vertex]).flip(-1).unbind(-1)).argsort()
    polygon = polygon[order]
    fan = Mesh(
        points=torch.cat([mesh.points[vertex : vertex + 1], polygon]),
        cells=torch.stack(
            [torch.zeros(len(ring), dtype=torch.long),
             torch.arange(1, len(ring) + 1),
             torch.arange(1, len(ring) + 1) % len(ring) + 1], 1
        ),
    )  # fmt: skip
    return [
        (_pv(mesh), _flat(S.FACE)),
        (_pv(fan), _flat(S.SECOND_FILL)),
        (_edges(mesh, 0.004, z=0.005), _flat(S.EDGE)),
        (_edges(dual, 0.006, z=0.01), _flat(S.SECOND)),
        *_dots(centers[ring], 0.018, S.SECOND, z=0.015),
        *_dots(mesh.points[vertex : vertex + 1].float(), 0.045, S.RESULT, z=0.03),
    ], "2d"


def calculus():
    """A surface field and its gradient, as tangent arrows."""

    sphere = sphere_icosahedral.load(subdivisions=2)
    x, y, z = sphere.points.unbind(-1)
    sphere.point_data["f"] = torch.sin(2 * x) * torch.cos(y) + z
    gradient = sphere.compute_point_derivatives("f").point_data["f_gradient"]
    surface = _pv(sphere)
    surface["f"] = sphere.point_data["f"].numpy()
    return [
        (surface, _shaded(None, scalars="f", cmap=S.ERROR, show_scalar_bar=False,
                          show_edges=True, edge_color=S.SURFACE_EDGE)),
        _arrows(sphere.points * 1.01, gradient, 0.16, S.RESULT, lit=True),
    ], "3d"  # fmt: skip


def curvature():
    """Gaussian curvature on a torus, diverging and centered at zero."""

    mesh = torus.load(n_major=72, n_minor=36)
    surface = _pv(mesh)
    surface["K"] = mesh.gaussian_curvature_vertices.numpy()
    bound = float(np.abs(surface["K"]).max())
    return [
        (surface, _shaded(None, scalars="K", cmap=S.DIVERGING, clim=(-bound, bound),
                          show_scalar_bar=False, smooth_shading=True)),
    ], "3d"  # fmt: skip


def neighbors():
    """A query point with its neighbor points and its incident cells."""

    mesh = perturbed_grid.load(n_x=6, n_y=6, perturbation_scale=0.04, seed=0)
    mesh = mesh.with_points(mesh.points * torch.tensor([1.5, 1.0]))
    middle = mesh.points.mean(0)
    query = int((mesh.points - middle).norm(dim=-1).argmin())
    near = mesh.get_point_to_points_adjacency().to_list()[query]
    ring = mesh.get_point_to_cells_adjacency().to_list()[query]
    spokes = Mesh(
        points=mesh.points,
        cells=torch.tensor([[query, other] for other in near]),
    )
    return [
        (_pv(mesh), _flat(S.FACE)),
        (_pv(mesh.slice_cells(ring)), _flat(S.SECOND_FILL)),
        (_edges(mesh, 0.004, z=0.005), _flat(S.EDGE)),
        (_edges(spokes, 0.011, z=0.01), _flat(S.SECOND)),
        *_dots(mesh.points[near], 0.03, S.SECOND, z=0.015),
        *_dots(mesh.points[query : query + 1], 0.042, S.RESULT, z=0.02),
    ], "2d"


def spatial():
    """A z = 0 slice of the signed distance to a closed surface."""

    surface = lumpy_sphere.load(subdivisions=4, noise_amplitude=0.4, seed=3)
    n = 240
    xs = torch.linspace(-2.0, 2.0, int(1.6 * n))
    ys = torch.linspace(-1.25, 1.25, n)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    query = torch.stack([grid_x, grid_y, torch.zeros_like(grid_x)], -1).reshape(-1, 3)
    sdf, _, _ = signed_distance_field(surface, query)
    image = pv.ImageData(
        dimensions=(len(xs), len(ys), 1),
        spacing=(float(xs[1] - xs[0]), float(ys[1] - ys[0]), 1.0),
        origin=(-2.0, -1.25, 0.0),
    )
    image["sdf"] = sdf.reshape(-1).numpy()
    bound = float(sdf.abs().max())
    contour = image.contour([0.0], scalars="sdf")
    contour.points[:, 2] += 0.01
    return [
        (image, _flat(None, scalars="sdf", cmap=S.DIVERGING, clim=(-bound, bound),
                      show_scalar_bar=False)),
        (contour.tube(radius=0.022), _flat(S.RESULT)),
    ], "2d"  # fmt: skip


def sampling():
    """Random points on a torus, colored by a field interpolated from the vertices."""

    mesh = torus.load(major_radius=1.0, minor_radius=0.4, n_major=24, n_minor=10)
    x, y, z = mesh.points.unbind(-1)
    mesh.point_data["f"] = 2 * x * y + z
    torch.manual_seed(0)
    cells = torch.randint(0, mesh.n_cells, (1500,))
    samples = mesh.sample_random_points_on_cells(cells)
    values = mesh.sample_data_at_points(samples, data_source="points")["f"]
    dot = pv.Sphere(radius=0.024, theta_resolution=10, phi_resolution=10)
    cloud = pv.PolyData(samples.numpy()).glyph(geom=dot, scale=False, orient=False)
    # Glyphs repeat each sphere's vertices, so repeat the values to match.
    cloud["f"] = np.repeat(values.numpy(), dot.n_points)
    return [
        _surface(mesh, color=SAMPLE_GRAY, edges=False),
        (cloud, _shaded(None, scalars="f", cmap=S.SEQUENTIAL, show_scalar_bar=False)),
    ], "3d"


def visualization():
    """Mesh.draw on a lumpy sphere colored by height, rendered by draw itself."""

    mesh = lumpy_sphere.load(subdivisions=4, noise_amplitude=0.4, seed=1)
    mesh.point_data["height"] = mesh.points[:, 2]
    return "draw", mesh


def graphs():
    """The edge graph of a curved sheet: blue nodes and edges over the faint surface."""

    sheet = _sheet()
    graph = sheet.get_facet_mesh()
    return [
        (_pv(sheet), _shaded(S.FAINT, opacity=0.9)),
        (_pv(graph).tube(radius=0.009), _shaded(S.SECOND)),
        (S.spheres(sheet.points.numpy(), 0.022), _shaded(S.SECOND)),
    ], "3d"


def domain_mesh():
    """Potential flow around a cylinder: C_p on the interior, two named boundaries."""

    mesh = annulus_2d.load(inner_radius=0.5, outer_radius=2.0, n_radial=8, n_angular=64)
    radius = mesh.points.norm(dim=-1)
    angle = torch.atan2(mesh.points[:, 1], mesh.points[:, 0])
    ratio = (0.5 / radius) ** 2
    pressure = 2 * ratio * torch.cos(2 * angle) - ratio**2
    surface = _pv(mesh)
    surface["cp"] = pressure.numpy()
    boundary = mesh.get_boundary_mesh()
    inner = boundary.points[boundary.cells].norm(dim=-1).mean(-1) < 1.0
    return [
        # C_p runs from -3 to 1. A symmetric range of 1 keeps the far field visible.
        (surface, _flat(None, scalars="cp", cmap=S.DIVERGING, clim=(-1.0, 1.0),
                        show_scalar_bar=False)),
        (_edges(boundary.slice_cells(inner), 0.035, z=0.01), _flat(S.RESULT)),
        (_edges(boundary.slice_cells(~inner), 0.035, z=0.01), _flat(S.SECOND)),
    ], "2d"  # fmt: skip


def getting_started():
    """A first mesh with a field attached: the curved sheet colored by a point field."""

    sheet = _sheet()
    x, y, _ = sheet.points.unbind(-1)
    surface = _pv(sheet)
    surface["f"] = (torch.sin(3 * x) + y).numpy()
    return [
        (surface, _shaded(None, scalars="f", cmap=S.SEQUENTIAL, show_scalar_bar=False,
                          show_edges=True, edge_color=S.SURFACE_EDGE)),
        (S.spheres(sheet.points.numpy(), 0.018), _shaded(S.SURFACE_EDGE)),
    ], "3d"  # fmt: skip


def ml_pipelines():
    """Three samples merged into one batch Mesh, one color per sample."""

    samples = [
        annulus_2d.load(n_radial=3, n_angular=24),
        l_shape.load(size=1.9, subdivisions=4).translate([1.35, -0.95]),
        circle_2d.load(radius=0.9, n_radial=3, n_angular=20).translate([4.6, 0.0]),
    ]
    batch = Mesh.merge(samples)
    sample = torch.repeat_interleave(
        torch.arange(len(samples)), torch.tensor([m.n_cells for m in samples])
    )
    actors = []
    for index, color in enumerate((S.SECOND_FILL, S.RESULT, S.FIXED)):
        actors.append((_pv(batch.slice_cells(sample == index)), _flat(color)))
    actors.append((_edges(batch, 0.008, z=0.01), _flat("#ffffff")))
    return actors, "2d"


THUMBNAILS = {
    function.__name__: function
    for function in (
        core, fields, primitives, io, generate, tessellation, transformations,
        deformation, remeshing, subdivision, smoothing, repair, validation,
        boundaries, geometry, calculus, curvature, neighbors, spatial, sampling,
        visualization, graphs, domain_mesh, getting_started, ml_pipelines,
    )
}  # fmt: skip


### Rendering


def _plotter(view):
    """An off-screen plotter on white with the house anti-aliasing."""

    plotter = pv.Plotter(off_screen=True, window_size=RENDER)
    plotter.enable_anti_aliasing("ssaa")
    plotter.set_background("white")
    return plotter


def _camera(plotter, view):
    """Top-down parallel view in 2D, the shared oblique view in 3D."""

    if view == "2d":
        plotter.enable_parallel_projection()
        plotter.view_xy()
    else:
        plotter.view_vector(S.VIEW_VECTOR, viewup=S.VIEW_UP)
    plotter.reset_camera()


def _fit(image):
    """Crop to the object and center it on the card, so every card has the same margins."""

    pixels = np.asarray(image)
    ink = (pixels < 245).any(axis=-1)
    rows, cols = np.nonzero(ink)
    crop = Image.fromarray(pixels).crop(
        (cols.min(), rows.min(), cols.max() + 1, rows.max() + 1)
    )
    width, height = SIZE
    margin = MARGIN * height
    scale = min((width - 2 * margin) / crop.width, (height - 2 * margin) / crop.height)
    crop = crop.resize(
        (round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS
    )
    card = Image.new("RGB", SIZE, "white")
    card.paste(crop, ((width - crop.width) // 2, (height - crop.height) // 2))
    return card


def render(name):
    """Render one thumbnail and return it as an 800 x 500 PIL image."""

    actors, view = THUMBNAILS[name]()
    plotter = _plotter(view)
    if actors == "draw":
        # The visualization card uses Mesh.draw itself on this plotter.
        view.draw(
            backend="pyvista",
            show=False,
            ax=plotter,
            point_scalars="height",
            alpha_points=0.0,
            backend_options={"ambient": 0.35, "show_scalar_bar": False},
        )
        view = "3d"
    else:
        for dataset, kwargs in actors:
            plotter.add_mesh(dataset, **kwargs)
    _camera(plotter, view)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return _fit(image)


def main():
    """Render the thumbnails named on the command line, or all of them."""

    names = sys.argv[1:] or list(THUMBNAILS)
    OUTPUT.mkdir(exist_ok=True)
    for name in names:
        path = OUTPUT / f"{name}.png"
        render(name).quantize(colors=256, method=Image.Quantize.FASTOCTREE).save(
            path, optimize=True
        )
        print(f"Saved {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
