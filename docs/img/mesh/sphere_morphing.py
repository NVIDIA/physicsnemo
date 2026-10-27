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

"""Render the original sphere and its single-control and multiple-control morphs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from _style import (
    ACCENT_LIGHTING,
    RESULT,
    SURFACE,
    SURFACE_EDGE,
    mesh_label,
    save,
    spheres,
)
from matplotlib.lines import Line2D

from physicsnemo.mesh.io import to_pyvista
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "sphere_morphing.png"

CAMERA = [(5.0, -5.0, 2.6), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
RADIUS = 1.0


def _render(actors, camera):
    """Render PyVista actors from a fixed camera, so that panels share one scale."""

    plotter = pv.Plotter(off_screen=True, window_size=(900, 900))
    plotter.enable_anti_aliasing("ssaa")
    plotter.set_background("white")
    for dataset, kwargs in actors:
        plotter.add_mesh(dataset, **{"ambient": 0.35, **kwargs})
    plotter.camera_position = camera
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def _surface(mesh):
    """Return the shaded surface actor of a mesh."""

    return (
        to_pyvista(mesh),
        {
            "color": SURFACE,
            "show_edges": True,
            "edge_color": SURFACE_EDGE,
            "line_width": 0.6,
            "diffuse": 0.7,
        },
    )


def _arrows(starts, vectors, color):
    """Draw displacement arrows as solid geometry."""

    arrows = [
        pv.Arrow(
            start=start,
            direction=vector,
            scale=float(np.linalg.norm(vector)),
            tip_radius=0.18,
            shaft_radius=0.06,
            tip_length=0.35,
        )
        for start, vector in zip(starts, vectors, strict=True)
    ]
    return (pv.merge(arrows), {"color": color, **ACCENT_LIGHTING})


def _legend(figure, entries):
    """Add a marker legend at the bottom center."""

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=12,
            label=label,
        )
        for label, color in entries
    ]
    figure.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)


def _panel(mesh, controls, displacements):
    """Render a morphed sphere with its handles and displacement arrows."""

    actors = [_surface(mesh)]
    if controls is not None:
        destinations = (controls + displacements).numpy()
        actors.append(
            (spheres(destinations, 0.06), {"color": RESULT, **ACCENT_LIGHTING})
        )
        actors.append(_arrows(destinations, 0.6 * displacements.numpy(), RESULT))
    return _render(actors, CAMERA)


def main():
    """Generate the documentation image."""

    mesh = sphere_icosahedral.load(subdivisions=3)
    top_index = mesh.points[:, 2].argmax()
    bottom_index = mesh.points[:, 2].argmin()

    single_points = mesh.points[top_index].unsqueeze(0)
    single_displacements = mesh.points.new_tensor([[0.0, 0.0, 0.5]])
    single_morph = mesh.morph(single_points, single_displacements, radius=RADIUS)

    multiple_points = mesh.points[torch.stack((top_index, bottom_index))]
    multiple_displacements = mesh.points.new_tensor([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]])
    multiple_morph = mesh.morph(
        multiple_points,
        multiple_displacements,
        radius=mesh.points.new_tensor([RADIUS, RADIUS]),
    )

    panels = (
        ("sphere_icosahedral", mesh, None, None),
        (
            "morph(), one control, +0.5 z",
            single_morph,
            single_points,
            single_displacements,
        ),
        (
            "morph(), two controls, +0.5 z and -0.5 z",
            multiple_morph,
            multiple_points,
            multiple_displacements,
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 5.6))
    for axis, (title, panel_mesh, controls, displacements) in zip(
        axes, panels, strict=True
    ):
        axis.imshow(_panel(panel_mesh, controls, displacements))
        axis.set_title(f"{title}\n{mesh_label(panel_mesh)}")
        axis.set_axis_off()
    _legend(figure, ((f"displaced controls, radius={RADIUS}", RESULT),))
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.07, top=0.88, wspace=0.02)
    save(figure, OUTPUT)


if __name__ == "__main__":
    main()
