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

"""Render exact thin-plate-spline RBF deformation of a sphere."""

from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from physicsnemo.mesh.io import to_pyvista
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "sphere_rbf_morph.png"

MESH_COLOR = "lightblue"
ANCHOR_COLOR = "#c36a2d"
HANDLE_COLOR = "#5b9f00"


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


def _add_controls(plotter, controls, displacements, labels=()):
    """Draw fixed anchors and annotate moved handles on the deformed surface."""
    controls_np = controls.detach().cpu().numpy()
    displacements_np = displacements.detach().cpu().numpy()
    active = np.linalg.norm(displacements_np, axis=1) > 0.0

    if np.any(~active):
        plotter.add_points(
            controls_np[~active],
            color=ANCHOR_COLOR,
            point_size=12.0,
            render_points_as_spheres=True,
        )
    if not np.any(active):
        return

    vectors = displacements_np[active]
    destinations = controls_np[active] + vectors
    annotation_vectors = 0.3 * vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    plotter.add_points(
        destinations,
        color=HANDLE_COLOR,
        point_size=14.0,
        render_points_as_spheres=True,
    )
    plotter.add_arrows(
        destinations,
        annotation_vectors,
        mag=1.0,
        color=HANDLE_COLOR,
    )
    if labels:
        label_positions = destinations + annotation_vectors
        label_positions[:, 0] += 0.08
        plotter.add_point_labels(
            label_positions,
            list(labels),
            text_color="#397000",
            font_size=18,
            bold=True,
            shape=None,
            show_points=False,
        )


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
        (
            "Original\nRBF controls",
            sphere,
            torch.zeros_like(controls),
            (),
        ),
        (
            "One exact handle\nfive fixed anchors",
            pulled,
            one_handle,
            ("+0.55 z",),
        ),
        (
            "Two exact handles\nfour fixed anchors",
            stretched,
            two_handles,
            ("+0.55 z", "-0.55 z"),
        ),
    )

    plotter = pv.Plotter(shape=(1, 3), window_size=(1400, 650))
    for column, (title, mesh, displacements, labels) in enumerate(panels):
        plotter.subplot(0, column)
        plotter.add_mesh(
            to_pyvista(mesh),
            color=MESH_COLOR,
            show_edges=True,
            line_width=0.5,
        )
        _add_controls(plotter, controls, displacements, labels)
        plotter.add_text(
            title,
            position="upper_edge",
            font_size=13,
            color="black",
        )
        plotter.camera_position = [
            (5.0, -5.0, 3.8),
            (0.0, 0.0, 0.3),
            (0.0, 0.0, 1.0),
        ]

    plotter.set_background("white")
    plotter.render()
    plotter.screenshot(OUTPUT, transparent_background=False)
    plotter.close()
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
