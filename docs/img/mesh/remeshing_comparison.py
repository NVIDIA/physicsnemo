"""Render dense and Warp-remeshed Stanford bunny surfaces."""

from pathlib import Path

import pyvista as pv
import torch

from physicsnemo.mesh.io import to_pyvista
from physicsnemo.mesh.remeshing import remesh

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "remeshing_comparison.png"
REPO_ROOT = Path(__file__).resolve().parents[3]
BUNNY = REPO_ROOT / "examples" / "minimal" / "mesh" / "assets" / "bunny.pt"

if not torch.cuda.is_available():
    raise RuntimeError("A CUDA device is required to render the Warp remeshing example")

source = torch.load(BUNNY, weights_only=False).subdivide(levels=2, filter="loop")
target_vertices = 600
result = remesh(source.to("cuda"), n_clusters=target_vertices).to("cpu")

panels = (
    ("Dense source", source, "#9ecae1"),
    ("Remeshed · GPU", result, "#76B900"),
)

center = source.points.mean(dim=0).tolist()
diagonal = float((source.points.amax(dim=0) - source.points.amin(dim=0)).norm())
eye = [
    center[0] + 1.25 * diagonal,
    center[1] - 1.25 * diagonal,
    center[2] + 0.75 * diagonal,
]

plotter = pv.Plotter(shape=(1, 2), window_size=(1050, 500))
plotter.enable_anti_aliasing("ssaa")
for column, (label, mesh, color) in enumerate(panels):
    plotter.subplot(0, column)
    plotter.add_mesh(
        to_pyvista(mesh),
        color=color,
        show_edges=True,
        edge_color="#304050",
        line_width=0.45,
        smooth_shading=True,
        ambient=0.25,
        diffuse=0.75,
    )
    plotter.add_text(
        f"{label}\n{mesh.n_points:,} vertices · {mesh.n_cells:,} triangles",
        font_size=11,
        color="black",
    )
    plotter.camera_position = [eye, center, (0.0, 0.0, 1.0)]
    plotter.camera.zoom(1.15)

plotter.set_background("white")
plotter.render()
plotter.screenshot(OUTPUT, transparent_background=False)
plotter.close()

print(f"Saved {OUTPUT}")
