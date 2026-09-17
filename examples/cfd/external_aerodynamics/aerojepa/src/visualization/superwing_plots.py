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

r"""Field-level surface-flow plots for the SuperWing tutorial.

Renders one PNG per surface channel — ``Cp``, ``Cf_tau``, ``Cf_z`` —
showing ground truth, prediction, and absolute error side-by-side on
the dataset's native ``(128, 256)`` cell grid. Used by ``inference.py``
to materialise ``docs/img/inference_cp_field.png`` (and friends) after
training.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SUPERWING_CHANNEL_LABELS: tuple[str, ...] = (
    r"$C_p$",
    r"$C_{f,\tau}$",
    r"$C_{f,z}$",
)
SUPERWING_CHANNEL_SLUGS: tuple[str, ...] = ("cp", "cf_tau", "cf_z")


def denormalize_field(
    field: np.ndarray,
    *,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> np.ndarray:
    r"""Reverse the dataset's per-channel standardisation.

    Parameters
    ----------
    field : np.ndarray
        Normalised field of shape ``(C, H, W)`` or ``(N, C)``.
    target_mean : np.ndarray
        Per-channel mean of shape ``(C,)`` from the normalisation stats.
    target_std : np.ndarray
        Per-channel std of shape ``(C,)`` from the normalisation stats.

    Returns
    -------
    np.ndarray
        Field in physical units with the same shape as ``field``.
    """
    mean = np.asarray(target_mean, dtype=np.float32)
    std = np.asarray(target_std, dtype=np.float32)
    if field.ndim == 3:
        return field * std[:, None, None] + mean[:, None, None]
    if field.ndim == 2 and field.shape[-1] == mean.shape[0]:
        return field * std[None, :] + mean[None, :]
    raise ValueError(
        f"Unsupported field shape {tuple(field.shape)}; expected (C, H, W) "
        f"or (N, C={mean.shape[0]})."
    )


def _panel(ax, image: np.ndarray, *, title: str, cmap: str, vmin: float, vmax: float):
    im = ax.imshow(
        image,
        cmap=cmap,
        origin="lower",
        aspect="auto",
        vmin=float(vmin),
        vmax=float(vmax),
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_surface_field(
    *,
    predicted: np.ndarray,
    target: np.ndarray,
    output_dir: str | Path,
    case_id: str,
    channels: tuple[int, ...] = (0, 1, 2),
    channel_labels: tuple[str, ...] = SUPERWING_CHANNEL_LABELS,
    channel_slugs: tuple[str, ...] = SUPERWING_CHANNEL_SLUGS,
    cmap_field: str = "viridis",
    cmap_error: str = "magma",
) -> list[Path]:
    r"""Save one ``GT | Pred | |Error|`` PNG per channel.

    The output filename is ``{case_id}_{channel_slug}.png`` under
    ``output_dir``.

    Parameters
    ----------
    predicted : np.ndarray
        Predicted field of shape ``(C, H, W)`` (in physical units).
    target : np.ndarray
        Ground-truth field of shape ``(C, H, W)`` (in physical units).
    output_dir : str or Path
        Directory to write PNGs into. Created if missing.
    case_id : str
        Stem of the output filename (e.g. ``"geo0042_cond3_s000571"``).
    channels : tuple of int, optional
        Which channels to plot. Default ``(0, 1, 2)``.
    channel_labels : tuple of str, optional
        LaTeX labels matched to ``channels``. Default
        :data:`SUPERWING_CHANNEL_LABELS`.
    channel_slugs : tuple of str, optional
        Filename-safe slugs matched to ``channels``. Default
        :data:`SUPERWING_CHANNEL_SLUGS`.
    cmap_field : str, optional
        Colormap for the GT and prediction panels. Default ``"viridis"``.
    cmap_error : str, optional
        Colormap for the absolute-error panel. Default ``"magma"``.

    Returns
    -------
    list of Path
        Absolute paths of the written PNGs, one per channel.
    """
    if predicted.shape != target.shape:
        raise ValueError(
            "predicted and target must have the same shape; "
            f"got {tuple(predicted.shape)} vs {tuple(target.shape)}."
        )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ch in channels:
        gt = np.asarray(target[ch])
        pred = np.asarray(predicted[ch])
        err = np.abs(pred - gt)
        vmin = float(min(gt.min(), pred.min()))
        vmax = float(max(gt.max(), pred.max()))
        err_vmax = float(err.max()) if err.size else 1.0

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        label = (
            channel_labels[ch]
            if 0 <= int(ch) < len(channel_labels)
            else f"channel {ch}"
        )
        im_gt = _panel(
            axes[0], gt, title=f"{label} (GT)", cmap=cmap_field, vmin=vmin, vmax=vmax
        )
        _panel(
            axes[1],
            pred,
            title=f"{label} (Pred)",
            cmap=cmap_field,
            vmin=vmin,
            vmax=vmax,
        )
        im_err = _panel(
            axes[2],
            err,
            title=f"|{label}| error",
            cmap=cmap_error,
            vmin=0.0,
            vmax=err_vmax,
        )
        fig.colorbar(im_gt, ax=axes[:2].ravel().tolist(), shrink=0.85)
        fig.colorbar(im_err, ax=axes[2], shrink=0.85)
        fig.suptitle(case_id, fontsize=10)

        slug = (
            channel_slugs[ch] if 0 <= int(ch) < len(channel_slugs) else f"ch{int(ch)}"
        )
        out_path = out_dir / f"{case_id}_{slug}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        written.append(out_path)
    return written
