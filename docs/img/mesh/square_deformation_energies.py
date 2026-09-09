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

"""Render a 2D RBF design with differentiable deformation penalties."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.collections import PolyCollection

from physicsnemo.mesh.deformation import (
    simplex_inversion_energy,
    simplex_strain_energy,
    total_measure_energy,
)
from physicsnemo.mesh.primitives.planar import unit_square
from physicsnemo.nn.functional import radial_basis_function_deform_points

OUTPUT = Path(__file__).parent / "square_deformation_energies.png"

MESH_COLOR = "#a8c6c9"
EDGE_COLOR = "#263238"
ANCHOR_COLOR = "#c36a2d"
HANDLE_COLOR = "#5b9f00"
AUXILIARY_COLOR = "#6f5aa8"


def _backend() -> tuple[torch.device, str]:
    """Select the accelerated backend when a CUDA device is available."""

    if torch.cuda.is_available():
        return torch.device("cuda"), "warp"
    return torch.device("cpu"), "torch"


def _signed_area_ratios(
    points: torch.Tensor,
    reference_points: torch.Tensor,
    cells: torch.Tensor,
) -> torch.Tensor:
    """Compute signed area ratios independently for figure annotations."""

    current = points[cells]
    reference = reference_points[cells]
    current_edges = torch.stack(
        (current[:, 1] - current[:, 0], current[:, 2] - current[:, 0]), dim=-1
    )
    reference_edges = torch.stack(
        (
            reference[:, 1] - reference[:, 0],
            reference[:, 2] - reference[:, 0],
        ),
        dim=-1,
    )
    return torch.linalg.det(current_edges) / torch.linalg.det(reference_edges)


def _total_area(points: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Return the unsigned area used only for residual annotations."""

    triangles = points[cells]
    edges = torch.stack(
        (triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        dim=-1,
    )
    return 0.5 * torch.linalg.det(edges).abs().sum()


def _displacements(
    controls: torch.Tensor,
    auxiliary: torch.Tensor,
    target_displacement: torch.Tensor,
) -> torch.Tensor:
    """Assemble fixed, auxiliary, and target control displacements."""

    return torch.cat(
        (
            torch.zeros_like(controls[:4]),
            auxiliary,
            target_displacement.unsqueeze(0),
        )
    )


def _deform(
    reference_points: torch.Tensor,
    controls: torch.Tensor,
    auxiliary: torch.Tensor,
    target_displacement: torch.Tensor,
    implementation: str,
) -> torch.Tensor:
    """Evaluate the radial-basis deformation for one auxiliary-control state."""

    return radial_basis_function_deform_points(
        reference_points,
        controls,
        _displacements(controls, auxiliary, target_displacement),
        kernel="thin_plate_spline",
        polynomial=True,
        smoothing=0.0,
        implementation=implementation,
    )


def _draw_mesh(axis, points: torch.Tensor, cells: torch.Tensor) -> None:
    """Draw one triangulated panel with shared limits and styling."""

    points_np = points.detach().cpu().numpy()
    cells_np = cells.detach().cpu().numpy()
    axis.add_collection(
        PolyCollection(
            points_np[cells_np],
            facecolors=MESH_COLOR,
            edgecolors=EDGE_COLOR,
            linewidths=0.38,
        )
    )
    axis.set_xlim(-0.16, 1.35)
    axis.set_ylim(-0.16, 1.65)
    axis.set_aspect("equal")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.grid(False)


def _draw_controls(
    axis,
    controls: torch.Tensor,
    displacements: torch.Tensor,
    *,
    original: bool,
) -> None:
    """Draw anchors, auxiliary controls, and the target handle."""

    controls_np = controls.detach().cpu().numpy()
    displacements_np = displacements.detach().cpu().numpy()
    shown = controls_np if original else controls_np + displacements_np

    axis.scatter(
        shown[:4, 0],
        shown[:4, 1],
        s=55,
        color=ANCHOR_COLOR,
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
        label="fixed anchors",
    )
    axis.scatter(
        shown[4:8, 0],
        shown[4:8, 1],
        s=48,
        color=AUXILIARY_COLOR,
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
        label="optimized controls",
    )

    source = controls_np[-1]
    destination = source + displacements_np[-1]
    if not original:
        axis.scatter(
            source[0],
            source[1],
            s=58,
            facecolor="none",
            edgecolor=HANDLE_COLOR,
            linewidth=1.3,
            zorder=5,
        )
    handle = source if original else destination
    axis.scatter(
        handle[0],
        handle[1],
        s=72,
        color=HANDLE_COLOR,
        edgecolor="white",
        linewidth=0.7,
        zorder=6,
        label="target handle",
    )
    axis.annotate(
        "",
        xy=destination,
        xytext=source,
        arrowprops={"arrowstyle": "-|>", "color": HANDLE_COLOR, "lw": 1.8},
        zorder=4,
    )


def _annotate(axis, area_error: float, minimum_jacobian: float) -> None:
    """Place geometric residuals in a panel-local caption."""

    axis.text(
        0.5,
        1.015,
        f"relative area error {area_error:.2e}\n"
        f"minimum signed ratio {minimum_jacobian:.3f}",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        color="#455a64",
        fontsize=9.2,
        linespacing=1.15,
        clip_on=False,
    )


def main() -> None:
    """Generate the documentation image."""

    device, implementation = _backend()
    reference = unit_square.load(subdivisions=3).to(device=device, dtype=torch.float64)
    controls = reference.points.new_tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [1.0, 0.5],
            [0.0, 0.5],
            [0.5, 0.5],
            [0.5, 1.0],
        ]
    )
    target_displacement = controls.new_tensor([0.36, 0.48])
    unregularized_auxiliary = torch.zeros_like(controls[4:8])
    unregularized = _deform(
        reference.points,
        controls,
        unregularized_auxiliary,
        target_displacement,
        implementation,
    ).detach()

    auxiliary = torch.nn.Parameter(torch.zeros_like(unregularized_auxiliary))
    optimizer = torch.optim.Adam([auxiliary], lr=3.0e-2)
    for _ in range(350):
        optimizer.zero_grad()
        points = _deform(
            reference.points,
            controls,
            auxiliary,
            target_displacement,
            implementation,
        )
        loss = (
            0.05
            * simplex_strain_energy(reference, points, implementation=implementation)
            + 80.0
            * total_measure_energy(reference, points, implementation=implementation)
            + 500.0
            * simplex_inversion_energy(
                reference,
                points,
                minimum_jacobian=0.4,
                implementation=implementation,
            )
        )
        loss.backward()
        if not torch.isfinite(auxiliary.grad).all():
            raise RuntimeError("Non-finite deformation-energy gradient")
        optimizer.step()

    regularized = _deform(
        reference.points,
        controls,
        auxiliary,
        target_displacement,
        implementation,
    ).detach()
    reference_area = _total_area(reference.points, reference.cells)

    def metrics(points: torch.Tensor) -> tuple[float, float]:
        """Return total-area and minimum-Jacobian residuals."""

        ratios = _signed_area_ratios(points, reference.points, reference.cells)
        area_error = (_total_area(points, reference.cells) / reference_area - 1.0).abs()
        return area_error.item(), ratios.min().item()

    original_metrics = metrics(reference.points)
    unregularized_metrics = metrics(unregularized)
    regularized_metrics = metrics(regularized)
    if not regularized_metrics[0] < 0.2 * unregularized_metrics[0]:
        raise RuntimeError("Area penalty did not materially reduce the residual")
    if not (
        regularized_metrics[1] > 0.0
        and regularized_metrics[1] > unregularized_metrics[1]
    ):
        raise RuntimeError(
            "Inversion penalty did not improve the minimum signed element ratio"
        )

    panels = (
        (
            "Reference mesh\nfixed RBF controls",
            reference.points,
            torch.zeros_like(controls),
            original_metrics,
            True,
        ),
        (
            "Target deformation\nwithout geometric penalties",
            unregularized,
            _displacements(controls, unregularized_auxiliary, target_displacement),
            unregularized_metrics,
            False,
        ),
        (
            "Penalty-regularized\nstrain + area + inversion",
            regularized,
            _displacements(controls, auxiliary.detach(), target_displacement),
            regularized_metrics,
            False,
        ),
    )

    figure, axes = plt.subplots(1, 3, figsize=(15.6, 5.3))
    for axis, (title, points, displacements, residuals, original) in zip(
        axes, panels, strict=True
    ):
        _draw_mesh(axis, points, reference.cells)
        _draw_controls(axis, controls, displacements, original=original)
        _annotate(axis, *residuals)
        axis.set_title(title, fontsize=14, pad=40)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=10.5,
    )
    figure.subplots_adjust(
        left=0.045,
        right=0.985,
        bottom=0.19,
        top=0.79,
        wspace=0.22,
    )
    figure.savefig(OUTPUT, dpi=145, facecolor="white")
    plt.close(figure)
    print(f"Saved {OUTPUT} with {implementation} on {device}")


if __name__ == "__main__":
    main()
