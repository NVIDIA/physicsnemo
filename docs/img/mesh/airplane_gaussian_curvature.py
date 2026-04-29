"""Render the PyVista stock airplane mesh colored by Gaussian curvature."""

from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from physicsnemo.mesh.io import from_pyvista, to_pyvista

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "airplane_gaussian_curvature.png"

### Load the airplane from PyVista examples and convert
pv_airplane = pv.examples.load_airplane()
mesh = from_pyvista(pv_airplane)

### Defensive cleanup; harmless on the airplane mesh and matches the bunny
### pipeline so users get a consistent recipe across the docs.
mesh = mesh.clean()

### Subdivide twice for smoother curvature estimation; Loop subdivision
### produces a limit surface that is C2 everywhere except at extraordinary
### vertices, so two levels dramatically reduce discrete-curvature noise.
mesh = mesh.subdivide(levels=2, filter="loop")

### Compute Gaussian curvature with log1p regularization for visualization
K = mesh.gaussian_curvature_vertices
K = torch.nan_to_num(K, nan=0.0)
K_reg = K.sign() * K.abs().log1p()

### Smooth the scalar field via iterated Laplacian diffusion to suppress
### per-vertex noise from the discrete curvature estimate.
adj = mesh.get_point_to_points_adjacency()
src, tgt = adj.expand_to_pairs()
for _ in range(50):
    neighbor_sum = torch.zeros_like(K_reg)
    counts = torch.zeros_like(K_reg)
    neighbor_sum.scatter_add_(0, tgt, K_reg[src])
    counts.scatter_add_(0, tgt, torch.ones_like(K_reg[src]))
    K_reg = 0.3 * K_reg + 0.7 * neighbor_sum / counts.clamp(min=1)

mesh.point_data["gaussian_curvature"] = K_reg

K_np = K_reg.numpy()
### The airplane is dominated by flat regions punctuated by very high
### curvature at sharp edges, so a tighter upper percentile (80 vs 95)
### lets the bulk variation occupy more of the colormap rather than being
### compressed to a single colour by extreme outliers at the wing tips.
low, high = np.percentile(K_np, 5), np.percentile(K_np, 80)

pv_mesh = to_pyvista(mesh)

plotter = pv.Plotter(window_size=(1400, 1000))
plotter.add_mesh(
    pv_mesh,
    scalars="gaussian_curvature",
    cmap="coolwarm",
    clim=(low, high),
    show_edges=False,
    scalar_bar_args={"title": "Gaussian Curvature", "color": "black"},
)
plotter.set_background("white")
### Scale-relative isometric-style camera, shared with the other airplane scripts.
center = mesh.points.mean(dim=0).numpy().tolist()
diag = float((mesh.points.amax(dim=0) - mesh.points.amin(dim=0)).norm())
eye = [center[0] - 0.7 * diag, center[1] - 0.7 * diag, center[2] + 0.6 * diag]
plotter.camera_position = [eye, center, (0, 0, 1)]
plotter.screenshot(OUTPUT, transparent_background=False)
plotter.close()

print(f"Saved {OUTPUT}")
