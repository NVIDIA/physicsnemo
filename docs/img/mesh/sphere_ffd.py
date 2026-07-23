"""Render lattice free-form deformation of a sphere with its control lattice."""

from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from physicsnemo.mesh.io import to_pyvista
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "sphere_ffd.png"

ORIGIN = (-1.15, -1.15, -1.15)
EXTENT = (2.3, 2.3, 2.3)


def lattice_nodes(resolution, basis):
    """Basis-associated control positions, shape (*resolution, 3)."""
    if basis == "bernstein":
        coordinates = [torch.linspace(0.0, 1.0, n) for n in resolution]
    else:
        coordinates = [
            (torch.arange(n, dtype=torch.float32) - 1) / (n - 3) for n in resolution
        ]
    axes = [
        origin + extent * coordinate
        for coordinate, origin, extent in zip(coordinates, ORIGIN, EXTENT)
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


def lattice_wireframe(nodes):
    """Build a PyVista wireframe of the lattice edges along every axis."""
    n1, n2, n3, _ = nodes.shape
    index = np.arange(n1 * n2 * n3).reshape(n1, n2, n3)
    segments = []
    for a, b in (
        (index[:-1, :, :], index[1:, :, :]),
        (index[:, :-1, :], index[:, 1:, :]),
        (index[:, :, :-1], index[:, :, 1:]),
    ):
        segments.append(np.stack([a.ravel(), b.ravel()], axis=1))
    segments = np.concatenate(segments, axis=0)
    lines = np.column_stack([np.full(len(segments), 2), segments]).ravel()
    return pv.PolyData(nodes.reshape(-1, 3).numpy(), lines=lines)


def taper_displacements(nodes, strength=0.55):
    """Bernstein example: taper the lattice toward the top, growing with z."""
    z = (nodes[..., 2] - ORIGIN[2]) / EXTENT[2]
    shrink = -strength * z.square()
    return torch.stack(
        [shrink * nodes[..., 0], shrink * nodes[..., 1], torch.zeros_like(z)], dim=-1
    )


def sculpt_displacements(nodes):
    """B-spline example with three independently localized shape features."""
    displacements = torch.zeros_like(nodes)
    features = (
        # Center, displacement direction, and compact support radius.
        ((0.75, 0.0, 0.25), (0.9, 0.0, 0.2), 0.95),
        ((0.0, -0.75, -0.2), (0.0, -0.75, -0.2), 0.9),
        ((-0.2, 0.1, 0.75), (0.1, -0.1, -0.7), 0.85),
    )
    for center, direction, radius in features:
        center_t = nodes.new_tensor(center)
        direction_t = nodes.new_tensor(direction)
        distance = torch.linalg.vector_norm(nodes - center_t, dim=-1)
        normalized_distance = (distance / radius).clamp_max(1)
        weight = (1 - normalized_distance.square()).square()
        displacements += weight.unsqueeze(-1) * direction_t
    return displacements


sphere = sphere_icosahedral.load(radius=1.0, subdivisions=3)

bernstein_resolution = (4, 4, 4)
bernstein_nodes = lattice_nodes(bernstein_resolution, "bernstein")
bernstein_displacements = taper_displacements(bernstein_nodes)
tapered = sphere.free_form_deform(
    bernstein_displacements,
    origin=list(ORIGIN),
    extent=list(EXTENT),
    basis="bernstein",
)

bspline_resolution = (8, 8, 8)
bspline_nodes = lattice_nodes(bspline_resolution, "bspline")
bspline_displacements = sculpt_displacements(bspline_nodes)
sculpted = sphere.free_form_deform(
    bspline_displacements,
    origin=list(ORIGIN),
    extent=list(EXTENT),
    basis="bspline",
)

panels = (
    ("Original sphere\n4x4x4 lattice", sphere, bernstein_nodes, "#607080", 0.55),
    (
        "Bernstein taper\nevery node acts globally",
        tapered,
        bernstein_nodes + bernstein_displacements,
        "#c04040",
        0.55,
    ),
    (
        "Cubic B-spline sculpt\n8x8x8 lattice, three local features",
        sculpted,
        bspline_nodes + bspline_displacements,
        "#c04040",
        0.3,
    ),
)

plotter = pv.Plotter(shape=(1, 3), window_size=(2100, 800))
for column, (title, mesh, nodes, lattice_color, opacity) in enumerate(panels):
    plotter.subplot(0, column)
    plotter.add_mesh(
        to_pyvista(mesh), color="lightblue", show_edges=True, line_width=0.5
    )
    wireframe = lattice_wireframe(nodes)
    plotter.add_mesh(wireframe, color=lattice_color, line_width=1.5, opacity=opacity)
    plotter.add_points(
        wireframe.points,
        color=lattice_color,
        point_size=6.0,
        render_points_as_spheres=True,
        opacity=min(1.0, 2 * opacity),
    )
    plotter.add_text(title, font_size=13, color="black")
    plotter.camera_position = [
        (7.2, -7.2, 5.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
plotter.set_background("white")
plotter.render()
plotter.screenshot(OUTPUT, transparent_background=False)
plotter.close()

print(f"Saved {OUTPUT}")
