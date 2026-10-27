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

"""Render one-handle and two-handle thin-plate-spline RBF deformations of a sphere."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from _style import (
    ACCENT_LIGHTING,
    FIXED,
    RESULT,
    SURFACE,
    SURFACE_EDGE,
    TEXT,
    mesh_label,
    save,
    spheres,
)
from matplotlib.lines import Line2D

from physicsnemo.mesh.io import to_pyvista
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "sphere_rbf_morph.png"

CAMERA = [(5.0, -5.0, 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]


def _controls(points: torch.Tensor) -> torch.Tensor:
    """Select six symmetric vertices spanning the affine polynomial terms."""
    indices = torch.stack(
        (
            points[:, 2].argmax(),
            points[:, 2].argmin(),
            points[:, 0].argmax(),
            points[:, 0].argmin(),
            points[:, 1].argmax(),
            points[:, 1].argmin(),
        )
    )
    return points[indices]


def _radial_basis_function_deform(mesh, controls, displacements):
    """Apply the exact RBF settings illustrated by this figure."""
    return mesh.radial_basis_function_deform(
        controls,
        displacements,
        kernel="thin_plate_spline",
        polynomial=True,
        smoothing=0.0,
        implementation="torch",
    )


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
    """Render a deformed sphere with fixed anchors, moved handles, and arrows."""

    controls = controls.numpy()
    displacements = displacements.numpy()
    active = np.linalg.norm(displacements, axis=1) > 0.0
    actors = [_surface(mesh)]
    if not active.any():
        # Before any move, a control is neither an anchor nor a handle.
        actors.append((spheres(controls, 0.06), {"color": TEXT, **ACCENT_LIGHTING}))
        return _render(actors, CAMERA)
    actors.append(
        (spheres(controls[~active], 0.06), {"color": FIXED, **ACCENT_LIGHTING})
    )
    if active.any():
        destinations = controls[active] + displacements[active]
        actors.append(
            (spheres(destinations, 0.07), {"color": RESULT, **ACCENT_LIGHTING})
        )
        directions = displacements[active] / np.linalg.norm(
            displacements[active], axis=1, keepdims=True
        )
        actors.append(_arrows(destinations, 0.35 * directions, RESULT))
    return _render(actors, CAMERA)


def main():
    """Generate the documentation image."""

    sphere = sphere_icosahedral.load(radius=1.0, subdivisions=3)
    controls = _controls(sphere.points)

    one_handle = torch.zeros_like(controls)
    one_handle[0, 2] = 0.55

    two_handles = torch.zeros_like(controls)
    two_handles[0, 2] = 0.55
    two_handles[1, 2] = -0.55

    pulled = _radial_basis_function_deform(sphere, controls, one_handle)
    stretched = _radial_basis_function_deform(sphere, controls, two_handles)

    panels = (
        ("sphere_icosahedral, six controls", sphere, torch.zeros_like(controls)),
        ("one handle +0.55 z, five anchors", pulled, one_handle),
        ("two handles +0.55 z and -0.55 z, four anchors", stretched, two_handles),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 5.6))
    for axis, (title, mesh, displacements) in zip(axes, panels, strict=True):
        axis.imshow(_panel(mesh, controls, displacements))
        axis.set_title(f"{title}\n{mesh_label(mesh)}")
        axis.set_axis_off()
    _legend(
        figure,
        (("controls", TEXT), ("fixed anchors", FIXED), ("moved handles", RESULT)),
    )
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.07, top=0.88, wspace=0.02)
    save(figure, OUTPUT)


if __name__ == "__main__":
    main()
