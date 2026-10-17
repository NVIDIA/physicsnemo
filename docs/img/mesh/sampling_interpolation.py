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

"""Render random surface samples colored by a field interpolated from vertices."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pyvista as pv
import torch
from PIL import Image

from physicsnemo.mesh.io import to_pyvista
from physicsnemo.mesh.primitives.surfaces import torus

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "sampling_interpolation.png"

N_SAMPLES = 5000
CMAP = "viridis"
EDGE_COLOR = "#304050"
VIEW_VECTOR = (0.0, -1.0, 0.9)

plt.rcParams.update({"font.size": 13})


def _dots(points, radius):
    """Draw points as small spheres: off-screen OpenGL ignores point sizes."""

    sphere = pv.Sphere(radius=radius, theta_resolution=8, phi_resolution=8)
    return points.glyph(geom=sphere, scale=False, orient=False)


def _render(layers):
    """Render PyVista layers off-screen and return the image."""

    plotter = pv.Plotter(window_size=(1000, 760))
    plotter.enable_anti_aliasing("ssaa")
    for dataset, options in layers:
        plotter.add_mesh(dataset, show_scalar_bar=False, **options)
    plotter.set_background("white")
    plotter.view_vector(VIEW_VECTOR, viewup=(0.0, 0.0, 1.0))
    plotter.reset_camera()
    plotter.camera.zoom(1.6)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def main():
    """Generate the documentation image."""

    torch.manual_seed(0)
    mesh = torus.load(major_radius=1.0, minor_radius=0.4, n_major=24, n_minor=10)
    x, y, z = mesh.points.unbind(dim=-1)
    mesh.point_data["f"] = 2.0 * x * y + z

    # Pick cells in proportion to their area, then a uniform point in each pick.
    cell_indices = torch.multinomial(mesh.cell_areas, N_SAMPLES, replacement=True)
    samples = mesh.sample_random_points_on_cells(cell_indices)
    values = mesh.sample_data_at_points(samples, data_source="points")["f"]
    # Linear interpolation stays within the range of the cell's vertex values.
    cell_values = mesh.point_data["f"][mesh.cells[cell_indices]]
    inside = (values >= cell_values.amin(dim=1) - 1e-5) & (
        values <= cell_values.amax(dim=1) + 1e-5
    )
    if not inside.all():
        raise RuntimeError("An interpolated value lies outside its cell's range")

    clim = (float(mesh.point_data["f"].min()), float(mesh.point_data["f"].max()))
    surface = to_pyvista(mesh)
    surface.point_data["f"] = mesh.point_data["f"].numpy()
    cloud = pv.PolyData(samples.numpy())
    cloud.point_data["f"] = values.numpy()

    vertex_image = _render(
        [
            (
                surface,
                {
                    "scalars": "f",
                    "cmap": CMAP,
                    "clim": clim,
                    "show_edges": True,
                    "edge_color": EDGE_COLOR,
                    "line_width": 0.6,
                },
            ),
            (
                _dots(pv.PolyData(mesh.points.numpy()), radius=0.022),
                {"color": EDGE_COLOR},
            ),
        ]
    )
    sample_image = _render(
        [
            (surface, {"color": "#d9dee2"}),
            (
                _dots(cloud, radius=0.016),
                {"scalars": "f", "cmap": CMAP, "clim": clim},
            ),
        ]
    )

    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
    axes[0].imshow(vertex_image)
    axes[0].set_title(f"Vertex field f on a torus\n{mesh.n_points} vertices")
    axes[1].imshow(sample_image)
    axes[1].set_title(
        f"{N_SAMPLES} random points on the surface\nf interpolated from the vertices"
    )
    for axis in axes:
        axis.set_axis_off()
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(plt.Normalize(*clim), CMAP),
        ax=axes,
        shrink=0.75,
        pad=0.02,
    )
    colorbar.set_label("f = 2xy + z")
    figure.savefig(OUTPUT, dpi=160, facecolor="white", bbox_inches="tight")
    # Drop the unused alpha channel to keep the file small.
    Image.open(OUTPUT).convert("RGB").save(OUTPUT, optimize=True)
    plt.close(figure)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
