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

"""Render the PyVista stock airplane mesh colored by Gaussian curvature."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from _style import DIVERGING, mesh_label, save
from matplotlib.colors import TwoSlopeNorm

from physicsnemo.mesh.io import from_pyvista, to_pyvista

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "airplane_gaussian_curvature.png"


def _smoothed_curvature(mesh):
    """Return the log-scaled Gaussian curvature, smoothed by neighbor averaging."""

    # The airplane has open boundary edges, where curvature is NaN.
    values = torch.nan_to_num(mesh.gaussian_curvature_vertices, nan=0.0)
    # Scale by the median magnitude first: the airplane's units make every |K| far
    # below one, where log1p is linear and the log scale would do nothing.
    scale = values.abs().median().clamp(min=torch.finfo(values.dtype).tiny)
    values = values.sign() * (values.abs() / scale).log1p()
    # Iterated neighbor averaging suppresses per-vertex noise of the estimate.
    source, target = mesh.get_point_to_points_adjacency().expand_to_pairs()
    for _ in range(50):
        neighbor_sum = torch.zeros_like(values)
        counts = torch.zeros_like(values)
        neighbor_sum.scatter_add_(0, target, values[source])
        counts.scatter_add_(0, target, torch.ones_like(values[source]))
        values = 0.3 * values + 0.7 * neighbor_sum / counts.clamp(min=1)
    return values


def main():
    """Generate the documentation image."""

    mesh = from_pyvista(pv.examples.load_airplane()).clean()
    # Loop subdivision gives a smoother surface for the discrete estimate.
    mesh = mesh.subdivide(levels=2, filter="loop")
    values = _smoothed_curvature(mesh).numpy()
    # Symmetric limits keep zero white. The 80th percentile keeps the sharp edges
    # from compressing the bulk variation, and both signs, into one color.
    limit = float(np.percentile(np.abs(values), 80))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    pv_mesh = to_pyvista(mesh)
    pv_mesh.point_data["curvature"] = values
    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1150))
    plotter.enable_anti_aliasing("ssaa")
    plotter.set_background("white")
    plotter.add_mesh(
        pv_mesh,
        scalars="curvature",
        cmap=DIVERGING,
        clim=(-limit, limit),
        show_scalar_bar=False,
        ambient=0.35,
        diffuse=0.7,
    )
    # Scale-relative camera, shared with airplane.py.
    center = mesh.points.mean(dim=0).numpy().tolist()
    diagonal = float((mesh.points.amax(dim=0) - mesh.points.amin(dim=0)).norm())
    eye = [
        center[0] - 0.7 * diagonal,
        center[1] - 0.7 * diagonal,
        center[2] + 0.6 * diagonal,
    ]
    plotter.camera_position = [eye, center, (0.0, 0.0, 1.0)]
    plotter.camera.zoom(1.25)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    # Crop the white border, so that the title sits close to the airplane.
    ink = (image[..., :3] < 250).any(axis=2)
    rows, cols = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    image = image[
        max(rows[0] - 20, 0) : rows[-1] + 20, max(cols[0] - 20, 0) : cols[-1] + 20
    ]

    figure, axis = plt.subplots(
        figsize=(11.0, 11.0 * image.shape[0] / image.shape[1] + 1.4)
    )
    axis.imshow(image)
    axis.set_title(f"gaussian_curvature_vertices\n{mesh_label(mesh)}")
    axis.set_axis_off()
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING),
        ax=axis,
        orientation="horizontal",
        fraction=0.04,
        pad=0.01,
        shrink=0.55,
    )
    colorbar.set_label(
        "Gaussian curvature, sign(K) log(1 + |K| / median |K|), smoothed"
    )
    ticks = np.linspace(-limit, limit, 5)
    colorbar.set_ticks(ticks, labels=[f"{tick:.2f}" for tick in ticks])
    save(figure, OUTPUT, tight=True, full_color=True)


if __name__ == "__main__":
    main()
