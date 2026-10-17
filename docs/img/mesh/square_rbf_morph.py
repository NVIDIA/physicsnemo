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

"""Render a two-dimensional thin-plate-spline RBF deformation."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from _style import EDGE, FACE, FIXED, RESULT, clean_axis, mesh_label, save
from matplotlib.collections import PolyCollection

from physicsnemo.mesh.primitives.planar import unit_square

OUTPUT = Path(__file__).parent / "square_rbf_morph.png"

ANCHOR_COLOR = FIXED
HANDLE_COLOR = RESULT


def _draw_mesh(axis, mesh):
    """Draw one filled triangular mesh with consistent limits and styling."""

    points = mesh.points.detach().cpu().numpy()
    cells = mesh.cells.detach().cpu().numpy()
    triangles = PolyCollection(
        points[cells],
        facecolors=FACE,
        edgecolors=EDGE,
        linewidths=0.7,
    )
    axis.add_collection(triangles)
    clean_axis(axis, ((-0.05, 1.12), (-0.05, 1.42)))


def _draw_controls(axis, controls, displacements, *, deformed):
    """Draw fixed anchors and the moved handle in source or output position."""

    controls = controls.detach().cpu().numpy()
    displacements = displacements.detach().cpu().numpy()
    source = controls[-1]
    destination = source + displacements[-1]

    axis.scatter(
        controls[:-1, 0],
        controls[:-1, 1],
        s=130,
        color=ANCHOR_COLOR,
        edgecolor="white",
        linewidth=1.2,
        zorder=4,
        label="fixed anchors",
    )
    if deformed:
        axis.scatter(
            source[0],
            source[1],
            s=120,
            facecolor="none",
            edgecolor=HANDLE_COLOR,
            linewidth=1.8,
            zorder=4,
        )
        handle = destination
    else:
        handle = source
    axis.scatter(
        handle[0],
        handle[1],
        s=160,
        color=HANDLE_COLOR,
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
        label="moved handle",
    )
    axis.annotate(
        "",
        xy=destination,
        xytext=source,
        arrowprops={"arrowstyle": "-|>", "color": HANDLE_COLOR, "lw": 2.4},
        zorder=4,
    )
    if deformed:
        return
    dx, dy = displacements[-1]
    axis.text(
        destination[0] + 0.035,
        destination[1] - 0.01,
        f"displacement ({dx:.2f}, {dy:.2f})",
        color=HANDLE_COLOR,
        va="center",
    )


def main():
    """Generate the documentation image."""

    mesh = unit_square.load(subdivisions=4)
    controls = mesh.points.new_tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 1.0]]
    )
    displacements = torch.zeros_like(controls)
    displacements[-1] = controls.new_tensor([0.15, 0.35])
    morphed = mesh.radial_basis_function_deform(
        controls,
        displacements,
        kernel="thin_plate_spline",
        polynomial=True,
        smoothing=0.0,
        implementation="torch",
    )

    handle_index = torch.linalg.vector_norm(mesh.points - controls[-1], dim=1).argmin()
    torch.testing.assert_close(
        morphed.points[handle_index],
        controls[-1] + displacements[-1],
        atol=2.0e-5,
        rtol=2.0e-5,
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 7.6))
    panels = (
        ("unit_square, four anchors and one handle", mesh, False),
        ('kernel="thin_plate_spline", handle moves exactly', morphed, True),
    )
    for axis, (title, panel_mesh, deformed) in zip(axes, panels, strict=True):
        _draw_mesh(axis, panel_mesh)
        _draw_controls(axis, controls, displacements, deformed=deformed)
        axis.set_title(f"{title}\n{mesh_label(panel_mesh)}")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=False,
    )
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.08, top=0.9, wspace=0.06)
    save(figure, OUTPUT)


if __name__ == "__main__":
    main()
