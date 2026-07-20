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

r"""Integrate predicted Cp into lift / drag coefficients for SuperWing.

Reads the ``predictions.npz`` produced by ``inference.py`` and, for each
test case, integrates the surface pressure coefficient against the
cell-area-weighted outward normals derived from the vertex mesh
(``origingeom``), then rotates the resulting force vector into
freestream-aligned ``(C_D, C_L)`` using the case's angle of attack.

The integration is pressure-only — friction is intentionally omitted
for the tutorial. Both the predicted and ground-truth fields go through
the same integrator, so the parity plot is internally consistent even
though absolute values differ from the dataset's ``surface_coeffs``
(which include friction).

Outputs:

* ``forces.csv`` — per-case predicted vs ground-truth coefficients.
* ``forces_parity.png`` — CL and CD parity scatter plots.

Run as a script:

.. code-block:: bash

    python -m src.postprocessing.superwing_forces \
        --predictions outputs/<run>/inference/predictions.npz \
        --output outputs/<run>/inference/forces.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _cell_normals_and_areas(geom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute outward unit normals and cell areas from a vertex grid.

    Parameters
    ----------
    geom : np.ndarray
        Vertex coordinates of shape ``(3, I, J)`` (the ``origingeom``
        layout shipped by SuperWing).

    Returns
    -------
    normals : np.ndarray
        Unit normals of shape ``(I-1, J-1, 3)``.
    areas : np.ndarray
        Cell areas of shape ``(I-1, J-1)``.
    """
    # (3, I, J) -> (I, J, 3) for easier corner indexing.
    g = np.transpose(geom, (1, 2, 0))
    p0 = g[:-1, :-1, :]  # SW corner
    p1 = g[:-1, 1:, :]  # SE
    p2 = g[1:, 1:, :]  # NE
    p3 = g[1:, :-1, :]  # NW
    normals = np.cross(p2 - p0, p3 - p1, axis=-1)
    areas = 0.5 * (
        np.linalg.norm(np.cross(p1 - p0, p2 - p0, axis=-1), axis=-1)
        + np.linalg.norm(np.cross(p2 - p0, p3 - p0, axis=-1), axis=-1)
    )
    n_norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(n_norm, 1e-20)
    return normals.astype(np.float32), areas.astype(np.float32)


def integrate_pressure_forces(
    *,
    cp: np.ndarray,
    origingeom: np.ndarray,
    aoa_deg: float,
    ref_area: float,
) -> tuple[float, float]:
    r"""Integrate Cp on the wing surface into ``(C_L, C_D)``.

    Pressure-only: ``F_xyz = sum_cells (Cp * n * area)``, then rotate
    into freestream-aligned components by the angle of attack and scale
    by the reference area. Friction is intentionally omitted.

    Parameters
    ----------
    cp : np.ndarray
        Cell-centric pressure coefficient of shape ``(I-1, J-1)``.
    origingeom : np.ndarray
        Vertex mesh of shape ``(3, I, J)``.
    aoa_deg : float
        Angle of attack in degrees.
    ref_area : float
        Reference area used to non-dimensionalise the integrated force.

    Returns
    -------
    cl : float
        Lift coefficient.
    cd : float
        Drag coefficient.
    """
    normals, areas = _cell_normals_and_areas(origingeom)
    fxyz = np.sum(cp[..., None] * normals * areas[..., None], axis=(0, 1))
    fx, fy = float(fxyz[0]), float(fxyz[1])
    aoa = float(aoa_deg) * np.pi / 180.0
    cd = (fx * np.cos(aoa) - fy * np.sin(aoa)) / float(ref_area)
    cl = (fx * np.sin(aoa) + fy * np.cos(aoa)) / float(ref_area)
    return cl, cd


def integrate_pressure_forces_batch(
    *,
    cp: np.ndarray,
    origingeom: np.ndarray,
    aoa_deg: np.ndarray,
    ref_area: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Batched wrapper around :func:`integrate_pressure_forces`.

    Parameters
    ----------
    cp : np.ndarray
        Stack of Cp fields of shape ``(B, I-1, J-1)``.
    origingeom : np.ndarray
        Stack of vertex meshes of shape ``(B, 3, I, J)``.
    aoa_deg : np.ndarray
        Per-case angles of attack of shape ``(B,)``.
    ref_area : np.ndarray
        Per-case reference areas of shape ``(B,)``.

    Returns
    -------
    cl, cd : np.ndarray
        Coefficients of shape ``(B,)``.
    """
    b = int(cp.shape[0])
    cl = np.zeros((b,), dtype=np.float64)
    cd = np.zeros((b,), dtype=np.float64)
    for i in range(b):
        cl[i], cd[i] = integrate_pressure_forces(
            cp=cp[i],
            origingeom=origingeom[i],
            aoa_deg=float(aoa_deg[i]),
            ref_area=float(ref_area[i]),
        )
    return cl, cd


def _plot_parity(
    *,
    cl_pred: np.ndarray,
    cl_gt: np.ndarray,
    cd_pred: np.ndarray,
    cd_gt: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    for ax, gt, pred, label in (
        (axes[0], cl_gt, cl_pred, r"$C_L$"),
        (axes[1], cd_gt, cd_pred, r"$C_D$"),
    ):
        lo = float(min(gt.min(), pred.min()))
        hi = float(max(gt.max(), pred.max()))
        ax.scatter(gt, pred, s=14, alpha=0.7)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"{label} (ground truth, pressure-only)")
        ax.set_ylabel(f"{label} (predicted, pressure-only)")
        ax.set_title(f"{label} parity — {len(gt)} test cases")
        ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions.npz produced by inference.py.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output CSV path. The parity PNG is written next to it.",
    )
    return p.parse_args()


def main() -> None:
    """Command-line entry point — see module docstring."""
    args = _parse_args()
    preds = np.load(args.predictions, allow_pickle=False)

    cp_pred = preds["pred_field"][:, 0]
    cp_gt = preds["target_field"][:, 0]
    origingeom = preds["origingeom"]
    aoa_deg = preds["aoa_deg"]
    ref_area = preds["ref_area"]

    cl_pred, cd_pred = integrate_pressure_forces_batch(
        cp=cp_pred,
        origingeom=origingeom,
        aoa_deg=aoa_deg,
        ref_area=ref_area,
    )
    cl_gt, cd_gt = integrate_pressure_forces_batch(
        cp=cp_gt,
        origingeom=origingeom,
        aoa_deg=aoa_deg,
        ref_area=ref_area,
    )

    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    case_ids = preds["case_ids"]
    mach = preds["mach"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "case_id",
                "aoa_deg",
                "mach",
                "ref_area",
                "cl_pred",
                "cl_gt",
                "cl_err",
                "cd_pred",
                "cd_gt",
                "cd_err",
            ]
        )
        for i in range(int(case_ids.shape[0])):
            w.writerow(
                [
                    str(case_ids[i]),
                    f"{float(aoa_deg[i]):.4f}",
                    f"{float(mach[i]):.4f}",
                    f"{float(ref_area[i]):.6f}",
                    f"{float(cl_pred[i]):.6f}",
                    f"{float(cl_gt[i]):.6f}",
                    f"{float(cl_pred[i] - cl_gt[i]):.6f}",
                    f"{float(cd_pred[i]):.6f}",
                    f"{float(cd_gt[i]):.6f}",
                    f"{float(cd_pred[i] - cd_gt[i]):.6f}",
                ]
            )
    print(f"Wrote {out_csv}")

    parity_path = out_csv.with_name("forces_parity.png")
    _plot_parity(
        cl_pred=cl_pred,
        cl_gt=cl_gt,
        cd_pred=cd_pred,
        cd_gt=cd_gt,
        output_path=parity_path,
    )
    print(f"Wrote {parity_path}")


if __name__ == "__main__":
    main()
