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
import os

from matplotlib import pyplot as plt
import numpy as np
import wandb

from utils.spectrum import ps1d_plots


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

    vmin, vmax = truth.min(), truth.max()

    def _make_fig(data, title, vmin=vmin, vmax=vmax):
        fig, ax = plt.subplots(1, figsize=(6, 7.5), dpi=100)
        im = ax.imshow(data, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        return fig

    yield ("generated", _make_fig(generated, f"Generated {variable}"))
    yield ("truth", _make_fig(truth, "Truth"))
    yield ("input", _make_fig(input_state, "Input"))

    backgrounds = _normalize_backgrounds(background)
    num_panels = 3 + max(len(backgrounds), 1)
    for name, bg in backgrounds.items():
        gmin, gmax = float(np.nanmin(bg)), float(np.nanmax(bg))
        yield (
            f"background_{name}",
            _make_fig(bg, f"Background: {name}", vmin=gmin, vmax=gmax),
        )


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


def save_validation_plots(trainer, plot_outputs, plot_state, plot_background):
    r"""
    Save validation plots to disk and wandb.

    Parameters
    ----------
    trainer: trainer.Trainer
        The trainer object used for plotting.
    plot_outputs : torch.Tensor or None
        Model outputs to visualize.
    plot_state : tuple or None
        Tuple of (input_state, target_state) for comparison plots.
    plot_background : torch.Tensor or None
        Background conditioning for context panels.
    """
    if trainer.dist.rank != 0 or plot_outputs is None or plot_state is None:
        return

    fields = trainer.cfg.training.validation_plot_variables

    for i in range(plot_outputs.shape[0]):
        image = plot_outputs[i].cpu().numpy()
        figs, spec_ratios = ps1d_plots(
            plot_outputs[i], plot_state[1][i], fields, trainer.state_channels
        )

        for f_ in fields:
            f_idx = trainer.state_channels.index(f_)
            image_dir = os.path.join(trainer.cfg.training.rundir, "images", f_)
            os.makedirs(image_dir, exist_ok=True)

            bg_panels = _prepare_background_panels(trainer, plot_background, i)
            validation_figs = validation_plot(
                image[f_idx],
                plot_state[1][i, f_idx].cpu().numpy(),
                plot_state[0][i, f_idx].cpu().numpy(),
                f_,
                bg_panels,
            )
            for name, fig in validation_figs:
                fig.savefig(
                    os.path.join(
                        image_dir, f"{trainer.total_steps}_{i}_{f_}_{name}.png"
                    ),
                    bbox_inches="tight",
                )
                trainer.logger.log_figure(f"{name}/{f_}", fig)
                plt.close(fig)

            figs[f"PS1D_{f_}"].savefig(
                os.path.join(image_dir, f"{trainer.total_steps}_{i}_{f_}_spec.png")
            )

            for figname, plot in figs.items():
                trainer.logger.log_figure(figname, plot)

            plt.close("all")

    for key, value in spec_ratios.items():
        trainer.logger.log_value(f"spectrum/valid/{key}", value)


def _prepare_background_panels(trainer, plot_background, batch_idx) -> dict | None:
    r"""
    Prepare background panels for validation plot.

    Parameters
    ----------
    trainer: trainer.Trainer
        The trainer object used for plotting.
    plot_background : torch.Tensor or None
        Background conditioning tensor of shape :math:`(B, C, H, W)`.
    batch_idx : int
        Index of the batch sample to extract.

    Returns
    -------
    dict or None
        Dictionary mapping channel names to numpy arrays, or None if no background.
    """
    if plot_background is None:
        return None

    selected = trainer.validation_bg_channels or (
        [trainer.background_channels[0]] if trainer.background_channels else []
    )
    panels = {}

    for bg in selected:
        if isinstance(bg, int):
            if bg < 0 or bg >= plot_background.shape[1]:
                continue
            label = (
                trainer.background_channels[bg]
                if bg < len(trainer.background_channels)
                else f"ch_{bg}"
            )
            idx = bg
        else:
            if bg not in trainer.background_channels:
                continue
            idx = trainer.background_channels.index(bg)
            label = bg
        panels[label] = plot_background[batch_idx, idx].detach().cpu().numpy()

    return panels if panels else None
