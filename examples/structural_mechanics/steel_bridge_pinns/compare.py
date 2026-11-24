# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Comparison utilities: metrics and figures for CAE vs PINN predictions.

All metrics are evaluated on a dense grid that is independent of both the finite
element mesh (different resolution) and the PINN collocation points, so the numbers
measure interpolation/extrapolation quality rather than in-sample fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from steel_bridge_cae import PlaneStressSolution, SteelBridge  # noqa: E402
from steel_bridge_pinn import BridgeEnergyLoss, BridgePINN  # noqa: E402

__all__ = [
    "relative_l2",
    "predict_fields",
    "evaluate_model",
    "plot_field_comparison",
    "plot_error_maps",
    "plot_convergence",
    "plot_data_efficiency",
    "format_table",
]


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Relative L2 error ``||pred - ref|| / ||ref||``."""
    denominator = np.linalg.norm(reference.ravel())
    if denominator == 0.0:
        return float(np.linalg.norm(prediction.ravel()))
    return float(np.linalg.norm((prediction - reference).ravel()) / denominator)


@dataclass
class FieldPrediction:
    """PINN prediction sampled on a structured grid."""

    displacement: np.ndarray  # (nx, ny, 2)
    von_mises: np.ndarray  # (nx, ny)


def predict_fields(
    model: BridgePINN,
    bridge: SteelBridge,
    losses: BridgeEnergyLoss,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> FieldPrediction:
    """Evaluate displacement and von Mises stress of the PINN on a grid."""
    xx, yy = np.meshgrid(grid_x, grid_y, indexing="ij")
    points = torch.as_tensor(
        np.stack([xx.ravel(), yy.ravel()], axis=-1), dtype=torch.float32
    )
    model.eval()
    with torch.enable_grad():
        displacement = model(points)
        stress = losses.von_mises(model, points)
    displacement = displacement.detach().cpu().numpy()
    shape = (grid_x.size, grid_y.size)
    return FieldPrediction(
        displacement=displacement.reshape(*shape, 2),
        von_mises=stress.detach().cpu().numpy().reshape(*shape),
    )


def evaluate_model(
    model: BridgePINN,
    bridge: SteelBridge,
    losses: BridgeEnergyLoss,
    solution: PlaneStressSolution,
    nx: int = 241,
    ny: int = 61,
    bearing_margin: float = 0.25,
) -> dict[str, float]:
    """Compare a trained PINN against the CAE reference on a dense grid.

    ``bearing_margin`` excludes a strip next to both bearings from the stress
    metric: the reference solution has a stress singularity at the bearing pad
    corners (a point-like support in a continuum), which no smooth network can
    reproduce and which is not of engineering interest.
    """
    xx, yy = solution.grid(nx, ny)
    reference_magnitude, reference_vm, reference_disp = solution.grid_fields(nx, ny)
    prediction = predict_fields(model, bridge, losses, xx[:, 0], yy[0, :])

    predicted_magnitude = np.linalg.norm(prediction.displacement, axis=-1)

    bulk = (xx > bearing_margin * bridge.span) & (
        xx < (1.0 - bearing_margin) * bridge.span
    )
    stress_error = relative_l2(prediction.von_mises[bulk], reference_vm[bulk])

    gauge = np.array([[0.5 * bridge.span, 0.0]])
    reference_gauge = float(solution.stress_at(gauge)[0, 0])
    predicted_gauge = _gauge_stress(model, losses, gauge)

    return {
        "rel_l2_displacement": relative_l2(prediction.displacement, reference_disp),
        "rel_l2_displacement_magnitude": relative_l2(
            predicted_magnitude, reference_magnitude
        ),
        "rel_l2_von_mises": stress_error,
        "max_abs_displacement_error_mm": float(
            np.max(np.abs(prediction.displacement - reference_disp)) * 1e3
        ),
        "midspan_deflection_mm": float(
            -prediction.displacement[
                int(np.argmin(np.abs(xx[:, 0] - 0.5 * bridge.span))), 0, 1
            ]
            * 1e3
        ),
        "midspan_deflection_reference_mm": float(solution.midspan_deflection() * 1e3),
        "bottom_flange_stress_mpa": predicted_gauge / 1e6,
        "bottom_flange_stress_reference_mpa": reference_gauge / 1e6,
    }


def _gauge_stress(
    model: BridgePINN, losses: BridgeEnergyLoss, points: np.ndarray
) -> float:
    """Axial stress ``sigma_xx`` at a gauge location, in Pa."""
    tensor = torch.as_tensor(points, dtype=torch.float32)
    sigma = losses.stress(model, tensor)
    return float(sigma[0, 0].detach().cpu())


def plot_field_comparison(
    solution: PlaneStressSolution,
    prediction: FieldPrediction,
    path: Path,
    title: str = "",
) -> None:
    """Side-by-side displacement magnitude and von Mises stress fields."""
    bridge = solution.bridge
    magnitude_reference, stress_reference, _ = solution.grid_fields(
        prediction.displacement.shape[0], prediction.displacement.shape[1]
    )
    magnitude_prediction = np.linalg.norm(prediction.displacement, axis=-1)

    extent = [0.0, bridge.span, 0.0, bridge.depth]
    # Colour limits are taken from the reference solution so that every variant is
    # shown on the same scale; values above the limit are clipped.  The reference
    # stress is capped at its 99.5th percentile because of the bearing singularity.
    vmax_displacement = float(magnitude_reference.max()) * 1.05 * 1e3
    vmax_stress = float(np.percentile(stress_reference, 99.5)) / 1e6

    fig, axes = plt.subplots(2, 2, figsize=(11, 5.2), constrained_layout=True)
    panels = [
        (
            axes[0, 0],
            magnitude_reference * 1e3,
            "CAE: |u| [mm]",
            "viridis",
            vmax_displacement,
        ),
        (
            axes[0, 1],
            magnitude_prediction * 1e3,
            "PINN: |u| [mm]",
            "viridis",
            vmax_displacement,
        ),
        (
            axes[1, 0],
            stress_reference / 1e6,
            "CAE: von Mises [MPa]",
            "magma",
            vmax_stress,
        ),
        (
            axes[1, 1],
            prediction.von_mises / 1e6,
            "PINN: von Mises [MPa]",
            "magma",
            vmax_stress,
        ),
    ]
    for axis, values, label, cmap, vmax in panels:
        image = axis.imshow(
            values.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
        )
        axis.set_title(label, fontsize=10)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        fig.colorbar(image, ax=axis, shrink=0.85)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_maps(
    solution: PlaneStressSolution,
    prediction: FieldPrediction,
    path: Path,
) -> None:
    """Absolute displacement error and von Mises stress difference."""
    bridge = solution.bridge
    nx, ny = prediction.displacement.shape[0], prediction.displacement.shape[1]
    _, stress_reference, displacement_reference = solution.grid_fields(nx, ny)
    error = np.linalg.norm(prediction.displacement - displacement_reference, axis=-1)
    extent = [0.0, bridge.span, 0.0, bridge.depth]
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.8), constrained_layout=True)
    image = axes[0].imshow(
        (error * 1e3).T, origin="lower", extent=extent, aspect="auto", cmap="inferno"
    )
    axes[0].set_title("|u_PINN - u_CAE| [mm]", fontsize=10)
    fig.colorbar(image, ax=axes[0], shrink=0.9)
    difference = prediction.von_mises - stress_reference
    # Symmetric colour scale clipped at the 99th percentile: the bearing singularity
    # otherwise sets the scale and hides the differences in the rest of the domain.
    limit = float(np.percentile(np.abs(difference), 99.0)) / 1e6
    image = axes[1].imshow(
        (difference / 1e6).T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    axes[1].set_title(
        f"von Mises difference [MPa] (clipped at +/-{limit:.1f})", fontsize=10
    )
    fig.colorbar(image, ax=axes[1], shrink=0.9)
    for axis in axes:
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(
    histories: dict[str, list[dict[str, float]]],
    path: Path,
    reference_deflection_mm: float | None = None,
) -> None:
    """Validation error and midspan deflection of every trained variant."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    for name, history in histories.items():
        steps = [row["step"] for row in history]
        error = [row.get("rel_l2_displacement", np.nan) for row in history]
        deflection = [row.get("midspan_deflection_mm", np.nan) for row in history]
        axes[0].semilogy(steps, error, label=name)
        axes[1].plot(steps, deflection, label=name)
    if reference_deflection_mm is not None:
        axes[1].axhline(
            reference_deflection_mm, color="k", linestyle=":", label="CAE reference"
        )
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("relative L2 error (u)")
    axes[0].set_title("Error against the CAE reference")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("midspan deflection [mm]")
    axes[1].set_title("Midspan deflection")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_data_efficiency(rows: list[dict[str, float]], path: Path) -> None:
    """Displacement and stress error versus the number of CAE measurements."""
    by_variant: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    for variant, entries in by_variant.items():
        entries = sorted(entries, key=lambda item: item["sensors"])
        sensors = [entry["sensors"] for entry in entries]
        axes[0].semilogy(
            sensors,
            [entry["rel_l2_displacement"] for entry in entries],
            "o-",
            label=variant,
        )
        axes[1].semilogy(
            sensors,
            [entry["rel_l2_von_mises"] for entry in entries],
            "o-",
            label=variant,
        )
    ticks = sorted({row["sensors"] for row in rows})
    for axis, title in zip(axes, ("displacement", "von Mises stress")):
        axis.set_xscale("log")
        axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        axis.set_xticks(ticks)
        axis.set_xticklabels([f"{int(item)}" for item in ticks])
        axis.set_xlabel("number of CAE measurement points")
        axis.set_ylabel(f"relative L2 error ({title})")
        axis.set_title(f"{title.capitalize()} error", fontsize=10)
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def format_table(rows: list[dict[str, float]], columns: list[str]) -> str:
    """Markdown table from a list of metric dictionaries."""
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join([" --- "] * len(columns)) + "|"
    lines = [header, separator]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                cells.append(f"{value:.4g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
