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

from collections.abc import Mapping

from matplotlib import pyplot as plt
import numpy as np


def _normalize_backgrounds(background):
    """Ensure background inputs are handled uniformly."""
    if background is None:
        return {}
    if isinstance(background, Mapping):
        return background
    if isinstance(background, (list, tuple)):
        return {f"background_{idx}": arr for idx, arr in enumerate(background)}
    return {"background": background}


def validation_plot(generated, truth, input_state, variable, background=None):
    """Produce validation plot created during training.

    Args:
        generated: Generated output array
        truth: Ground truth array
        input_state: Input state array (t=0)
        variable: Variable name for title
        background: Optional background channel(s) - dict, list, or single array

    Returns:
        matplotlib figure
    """
    backgrounds = _normalize_backgrounds(background)
    num_panels = 3 + max(len(backgrounds), 1)
    fig, axes = plt.subplots(
        1, num_panels, sharex=True, sharey=True, figsize=(4 * num_panels, 5)
    )
    if num_panels == 1:
        axes = [axes]
    a, b, c = axes[:3]
    vmin, vmax = truth.min(), truth.max()
    im = a.imshow(generated, vmin=vmin, vmax=vmax)
    a.set_title("generated, {}".format(variable))
    plt.colorbar(im, fraction=0.046, pad=0.04)
    im = b.imshow(truth, vmin=vmin, vmax=vmax)
    b.set_title("truth")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if input_state is None:
        c.set_title("input (none)")
        c.axis("off")
    else:
        im = c.imshow(input_state, vmin=vmin, vmax=vmax)
        c.set_title("input")
        plt.colorbar(im, fraction=0.046, pad=0.04)
    if backgrounds:
        for idx, (name, bg) in enumerate(backgrounds.items()):
            ax = axes[3 + idx]
            gmin, gmax = float(np.nanmin(bg)), float(np.nanmax(bg))
            im = ax.imshow(bg, vmin=gmin, vmax=gmax)
            ax.set_title(f"background: {name}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        # fallback: show empty panel to keep layout predictable
        ax = axes[3]
        ax.axis("off")
    return fig


color_limits = {
    "u10m": (-5, 5),
    "v10": (-5, 5),
    "t2m": (260, 310),
    "tcwv": (0, 60),
    "msl": (0.1, 0.3),
    "refc": (-10, 30),
}


def inference_plot(
    background,
    state_pred,
    state_true,
    plot_var_background,
    plot_var_state,
    initial_time,
    lead_time,
):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    state_error = state_pred - state_true

    if plot_var_state in color_limits:
        im = ax[0].imshow(
            state_pred,
            origin="lower",
            cmap="magma",
            clim=color_limits[plot_var_state],
        )
    else:
        im = ax[0].imshow(state_pred, origin="lower", cmap="magma")

    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    ax[0].set_title(
        "Predicted, {}, \n initial time {} \n lead_time {} hours".format(
            plot_var_state, initial_time, lead_time
        )
    )
    if plot_var_state in color_limits:
        im = ax[1].imshow(
            state_true,
            origin="lower",
            cmap="magma",
            clim=color_limits[plot_var_state],
        )
    else:
        im = ax[1].imshow(state_true, origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    ax[1].set_title("Actual, {}".format(plot_var_state))
    if plot_var_background in color_limits:
        im = ax[2].imshow(
            background,
            origin="lower",
            cmap="magma",
            clim=color_limits[plot_var_background],
        )
    else:
        im = ax[2].imshow(
            background,
            origin="lower",
            cmap="magma",
        )
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    ax[2].set_title("Background, {}".format(plot_var_background))
    maxerror = np.max(np.abs(state_error))
    im = ax[3].imshow(
        state_error,
        origin="lower",
        cmap="RdBu_r",
        vmax=maxerror,
        vmin=-maxerror,
    )
    fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)
    ax[3].set_title("Error, {}".format(plot_var_state))

    return fig
