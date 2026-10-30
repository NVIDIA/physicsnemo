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

"""Render dense and Warp-remeshed Stanford bunny surfaces."""

from pathlib import Path

import matplotlib.pyplot as plt
import pyvista as pv
import torch
from _style import RESULT, SURFACE, SURFACE_EDGE, mesh_label, save

from physicsnemo.mesh.io import to_pyvista
from physicsnemo.mesh.remeshing import remesh

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "remeshing_comparison.png"
REPO_ROOT = Path(__file__).resolve().parents[3]
BUNNY = REPO_ROOT / "examples" / "minimal" / "mesh" / "assets" / "bunny.pt"
TARGET_VERTICES = 600


def _render(mesh, color, camera):
    """Render one surface with its edges from a shared camera."""

    plotter = pv.Plotter(off_screen=True, window_size=(1000, 1000))
    plotter.enable_anti_aliasing("ssaa")
    plotter.set_background("white")
    plotter.add_mesh(
        to_pyvista(mesh),
        color=color,
        show_edges=True,
        edge_color=SURFACE_EDGE,
        line_width=0.6,
        smooth_shading=True,
        ambient=0.35,
        diffuse=0.7,
    )
    plotter.camera_position = camera
    plotter.camera.zoom(1.3)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def main():
    """Generate the documentation image."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA device is required to render the Warp remeshing example"
        )
    source = torch.load(BUNNY, weights_only=False).subdivide(levels=2, filter="loop")
    result = remesh(source.to("cuda"), n_clusters=TARGET_VERTICES).to("cpu")

    center = source.points.mean(dim=0).tolist()
    diagonal = float((source.points.amax(dim=0) - source.points.amin(dim=0)).norm())
    eye = [
        center[0] + 1.25 * diagonal,
        center[1] - 1.25 * diagonal,
        center[2] + 0.75 * diagonal,
    ]
    camera = [eye, center, (0.0, 0.0, 1.0)]

    panels = (
        ("dense source", source, SURFACE),
        (f"remesh(n_clusters={TARGET_VERTICES}) on GPU", result, RESULT),
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 6.0))
    for axis, (title, mesh, color) in zip(axes, panels, strict=True):
        axis.imshow(_render(mesh, color, camera))
        axis.set_title(f"{title}\n{mesh_label(mesh)}, {mesh.n_points:,} points")
        axis.set_axis_off()
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.9, wspace=0.02)
    save(figure, OUTPUT)


if __name__ == "__main__":
    main()
