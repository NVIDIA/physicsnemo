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

"""Compare the error of LSQ and DEC vertex gradients on planar meshes."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LogNorm
from matplotlib.tri import Triangulation
from PIL import Image

from physicsnemo.mesh.primitives.procedural import perturbed_grid

OUTPUT = Path(__file__).parent / "calculus_lsq_vs_dec.png"

METHODS = {"lsq": ("LSQ", "#0072b2"), "dec": ("DEC", "#d55e00")}
PANELS = ("(a)", "(b)")
MAP_RESOLUTION = 17
RESOLUTIONS = (9, 17, 33, 65, 129)

plt.rcParams.update({"font.size": 13})


def _field(points):
    """Smooth test field f(x, y) = sin(2x) cos(y) + y^2 and its exact gradient."""

    x, y = points.unbind(dim=-1)
    value = torch.sin(2 * x) * torch.cos(y) + y**2
    gradient = torch.stack(
        [2 * torch.cos(2 * x) * torch.cos(y), -torch.sin(2 * x) * torch.sin(y) + 2 * y],
        dim=-1,
    )
    return value, gradient


def _gradient_errors(n):
    """Return the mesh, a boundary-vertex mask, and the error of each method."""

    # Jitter the interior vertices by 5% of the grid spacing.
    mesh = perturbed_grid.load(
        n_x=n, n_y=n, perturbation_scale=0.05 / (n - 1), seed=0
    ).to(torch.float64)
    value, exact = _field(mesh.points)
    mesh.point_data["f"] = value
    on_boundary = torch.zeros(mesh.n_points, dtype=torch.bool)
    on_boundary[mesh.get_boundary_mesh().cells.unique()] = True
    errors = {}
    for method in METHODS:
        gradient = mesh.compute_point_derivatives("f", method=method)
        errors[method] = torch.linalg.vector_norm(
            gradient.point_data["f_gradient"] - exact, dim=-1
        )
    return mesh, on_boundary, errors


def main():
    """Generate the documentation image."""

    figure, axes = plt.subplots(
        1, 3, figsize=(14.0, 5.4), gridspec_kw={"width_ratios": (1.0, 1.0, 1.15)}
    )
    map_axes, convergence_axis = axes[:2], axes[2]

    mesh, _, errors = _gradient_errors(MAP_RESOLUTION)
    points = mesh.points.numpy()
    triangulation = Triangulation(points[:, 0], points[:, 1], mesh.cells.numpy())
    norm = LogNorm(1.0e-3, 2.0e-1, clip=True)
    for axis, panel, (method, (label, _)) in zip(
        map_axes, PANELS, METHODS.items(), strict=True
    ):
        # Flat shading shows one error value per triangle (the mean of its vertices).
        image = axis.tripcolor(
            triangulation,
            errors[method].clamp_min(norm.vmin).numpy(),
            cmap="cividis",
            norm=norm,
            shading="flat",
        )
        axis.set_title(f"{panel} {label} error, h = 1/{MAP_RESOLUTION - 1}")
        axis.set_aspect("equal")
        axis.set_axis_off()
    colorbar = figure.colorbar(
        image, ax=map_axes, orientation="horizontal", fraction=0.05, pad=0.04
    )
    colorbar.set_label(r"$|\nabla_h f - \nabla f|$")

    sizes = [1.0 / (n - 1) for n in RESOLUTIONS]
    curves = {(method, part): [] for method in METHODS for part in (0, 1)}
    for n in RESOLUTIONS:
        _, on_boundary, errors = _gradient_errors(n)
        for method, error in errors.items():
            for part, mask in enumerate((~on_boundary, on_boundary)):
                rms = error[mask].square().mean().sqrt()
                curves[method, part].append(float(rms))
    for (method, part), values in curves.items():
        label, color = METHODS[method]
        convergence_axis.loglog(
            sizes,
            values,
            color=color,
            linestyle="--" if part else "-",
            marker="s" if part else "o",
            label=f"{label}, {'boundary' if part else 'interior'}",
        )
    reference = [0.5 * curves["lsq", 1][0] * h / sizes[0] for h in sizes]
    convergence_axis.loglog(
        sizes, reference, color="#777777", linestyle=":", label=r"slope 1, $O(h)$"
    )
    convergence_axis.set_xlabel("mesh spacing $h$")
    convergence_axis.set_ylabel("RMS gradient error")
    convergence_axis.set_title("(c) error against mesh size")
    convergence_axis.grid(True, which="both", alpha=0.3)
    convergence_axis.set_ylim(5.0e-4, 1.0)
    convergence_axis.legend(fontsize=11.5, loc="upper left")

    figure.suptitle(
        r"Gradient of $f(x, y) = \sin 2x \, \cos y + y^2$ on a jittered grid",
        fontsize=16,
        y=1.04,
    )
    figure.savefig(OUTPUT, dpi=160, facecolor="white", bbox_inches="tight")
    # Drop the unused alpha channel to keep the file small.
    Image.open(OUTPUT).convert("RGB").save(OUTPUT, optimize=True)
    plt.close(figure)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
