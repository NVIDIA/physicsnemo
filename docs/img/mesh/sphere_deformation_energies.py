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

"""Render volume- and bending-regularized RBF deformation of a sphere."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from physicsnemo.mesh.deformation import (
    closed_surface_volume_energy,
    surface_bending_energy,
)
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.nn.functional import radial_basis_function_deform_points

OUTPUT = Path(__file__).parent / "sphere_deformation_energies.png"

MESH_COLOR = "#a8c6c9"
EDGE_COLOR = "#455a64"
ANCHOR_COLOR = "#c36a2d"
HANDLE_COLOR = "#5b9f00"
AUXILIARY_COLOR = "#6f5aa8"


def _backend() -> tuple[torch.device, str]:
    """Select the accelerated backend when a CUDA device is available."""

    if torch.cuda.is_available():
        return torch.device("cuda"), "warp"
    return torch.device("cpu"), "torch"


def _control_indices(points: torch.Tensor) -> torch.Tensor:
    """Select distinct sphere vertices nearest to prescribed directions."""

    directions = points.new_tensor(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.65],
            [-1.0, 0.0, 0.65],
            [0.0, 1.0, 0.65],
            [0.0, -1.0, 0.65],
            [1.0, 0.0, -0.55],
            [-1.0, 0.0, -0.55],
            [0.0, 1.0, -0.55],
            [0.0, -1.0, -0.55],
            [0.0, 0.0, 1.0],
        ]
    )
    directions = directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    indices = (directions @ points.T).argmax(dim=1)
    if torch.unique(indices).numel() != indices.numel():
        raise RuntimeError("Sphere subdivision did not provide distinct controls")
    return indices


def _displacements(
    controls: torch.Tensor,
    auxiliary: torch.Tensor,
    target_displacement: torch.Tensor,
) -> torch.Tensor:
    """Assemble fixed, auxiliary, and target control displacements."""

    return torch.cat(
        (
            torch.zeros_like(controls[:2]),
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


def _signed_volume(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Compute signed volume independently for figure annotations."""

    vertices = points[triangles]
    return (
        torch.sum(
            vertices[:, 0] * torch.linalg.cross(vertices[:, 1], vertices[:, 2], dim=1)
        )
        / 6.0
    )


def _draw_surface(axis, points: torch.Tensor, triangles: torch.Tensor) -> None:
    """Draw one triangle-surface panel with shared camera limits."""

    points_np = points.detach().cpu().numpy()
    triangles_np = triangles.detach().cpu().numpy()
    axis.add_collection3d(
        Poly3DCollection(
            points_np[triangles_np],
            facecolor=MESH_COLOR,
            edgecolor=EDGE_COLOR,
            linewidth=0.18,
            alpha=0.98,
            zorder=1,
        )
    )
    axis.set_xlim(-1.45, 1.55)
    axis.set_ylim(-1.45, 1.45)
    axis.set_zlim(-1.35, 1.75)
    axis.set_box_aspect((1.0, 1.0, 1.0), zoom=1.18)
    axis.view_init(elev=20.0, azim=-57.0)
    axis.set_axis_off()


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
        *shown[:2].T,
        s=24,
        color=ANCHOR_COLOR,
        depthshade=False,
        zorder=5,
    )
    axis.scatter(
        *shown[2:13].T,
        s=20,
        color=AUXILIARY_COLOR,
        depthshade=False,
        zorder=5,
    )

    source = controls_np[-1]
    destination = source + displacements_np[-1]
    handle = source if original else destination
    axis.scatter(
        *handle,
        s=36,
        color=HANDLE_COLOR,
        depthshade=False,
        zorder=6,
    )
    delta = destination - source
    axis.quiver(
        *source,
        *delta,
        color=HANDLE_COLOR,
        arrow_length_ratio=0.15,
        linewidth=1.8,
        zorder=5,
    )


def _annotate(axis, volume_error: float, bending: float) -> None:
    """Place geometric residuals in a panel-local annotation box."""

    axis.text2D(
        0.035,
        0.96,
        f"volume error: {volume_error:.2e}\nbending: {bending:.2e}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "#90a4ae", "alpha": 0.93},
    )


def main() -> None:
    """Generate the documentation image."""

    device, implementation = _backend()
    reference = sphere_icosahedral.load(radius=1.0, subdivisions=3).to(
        device=device, dtype=torch.float64
    )
    controls = reference.points[_control_indices(reference.points)]
    target_displacement = controls.new_tensor([0.12, -0.06, 0.58])
    unregularized_auxiliary = torch.zeros_like(controls[2:13])
    unregularized = _deform(
        reference.points,
        controls,
        unregularized_auxiliary,
        target_displacement,
        implementation,
    ).detach()

    auxiliary = torch.nn.Parameter(torch.zeros_like(unregularized_auxiliary))
    optimizer = torch.optim.Adam([auxiliary], lr=2.0e-2)
    for _ in range(400):
        optimizer.zero_grad()
        points = _deform(
            reference.points,
            controls,
            auxiliary,
            target_displacement,
            implementation,
        )
        loss = 250.0 * closed_surface_volume_energy(
            reference, points, implementation=implementation
        ) + 0.04 * surface_bending_energy(
            reference, points, implementation=implementation
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
    reference_volume = _signed_volume(reference.points, reference.cells)

    def metrics(points: torch.Tensor) -> tuple[float, float]:
        """Return enclosed-volume and hinge-bending residuals."""

        volume_error = (
            _signed_volume(points, reference.cells) / reference_volume - 1.0
        ).abs()
        bending = surface_bending_energy(
            reference, points, implementation=implementation
        )
        return volume_error.item(), bending.item()

    original_metrics = metrics(reference.points)
    unregularized_metrics = metrics(unregularized)
    regularized_metrics = metrics(regularized)
    if not regularized_metrics[0] < 0.2 * unregularized_metrics[0]:
        raise RuntimeError("Volume penalty did not materially reduce the residual")
    if not regularized_metrics[1] < unregularized_metrics[1]:
        raise RuntimeError("Bending penalty did not reduce the bending residual")

    panels = (
        (
            "Reference surface\nfixed RBF controls",
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
            "Penalty-regularized\nvolume + bending",
            regularized,
            _displacements(controls, auxiliary.detach(), target_displacement),
            regularized_metrics,
            False,
        ),
    )

    figure = plt.figure(figsize=(15.2, 4.35))
    axes = [
        figure.add_subplot(
            1,
            3,
            index + 1,
            projection="3d",
            computed_zorder=False,
        )
        for index in range(3)
    ]
    for axis, (title, points, displacements, residuals, original) in zip(
        axes, panels, strict=True
    ):
        _draw_surface(axis, points, reference.cells)
        _draw_controls(axis, controls, displacements, original=original)
        _annotate(axis, *residuals)
        axis.set_title(title, fontsize=14, pad=3)

    legend_handles = (
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=ANCHOR_COLOR,
            markeredgecolor="none",
            markersize=8,
            label="fixed anchors",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=AUXILIARY_COLOR,
            markeredgecolor="none",
            markersize=8,
            label="optimized controls",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=HANDLE_COLOR,
            markeredgecolor="none",
            markersize=8,
            label="target handle",
        ),
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.085),
        ncol=3,
        frameon=False,
        fontsize=10.5,
    )
    figure.subplots_adjust(
        left=0.015,
        right=0.985,
        bottom=0.14,
        top=0.84,
        wspace=0.02,
    )
    figure.savefig(
        OUTPUT,
        dpi=145,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(figure)
    print(f"Saved {OUTPUT} with {implementation} on {device}")


if __name__ == "__main__":
    main()
