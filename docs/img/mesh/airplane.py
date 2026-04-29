"""Render the PyVista stock airplane mesh as a plain wireframe surface."""

from pathlib import Path

import pyvista as pv

from physicsnemo.mesh.io import from_pyvista, to_pyvista

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "airplane.png"

### Load the airplane from PyVista examples and convert
pv_airplane = pv.examples.load_airplane()
mesh = from_pyvista(pv_airplane)

pv_mesh = to_pyvista(mesh)

plotter = pv.Plotter(window_size=(1400, 1000))
plotter.add_mesh(
    pv_mesh,
    color="lightblue",
    show_edges=True,
    line_width=0.5,
)
plotter.set_background("white")
### Scale-relative isometric-style camera. The airplane lives in a ~2000-unit
### bounding box, so the camera is offset by a fraction of the bounding diagonal.
center = mesh.points.mean(dim=0).numpy().tolist()
diag = float((mesh.points.amax(dim=0) - mesh.points.amin(dim=0)).norm())
eye = [center[0] - 0.7 * diag, center[1] - 0.7 * diag, center[2] + 0.6 * diag]
plotter.camera_position = [eye, center, (0, 0, 1)]
plotter.screenshot(OUTPUT, transparent_background=False)
plotter.close()

print(f"Saved {OUTPUT}")
