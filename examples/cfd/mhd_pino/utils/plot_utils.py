# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import os
import math
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm
import traceback

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import imageio
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from IPython.display import HTML, display


def plot_spectra_mhd(
    k,
    pred_spectra_kin,
    true_spectra_kin,
    pred_spectra_mag,
    true_spectra_mag,
    index_t=-1,
    name="Re100",
    save_path=None,
    save_suffix=None,
    font_size=None,
    sci_limits=None,
    style_kin_pred="b-",
    style_kin_true="k-",
    style_mag_pred="b--",
    style_mag_true="k--",
    xmin=0,
    xmax=200,
    ymin=1e-10,
    ymax=None,
):
    "Plots spectra of predicted and true outputs"
    if font_size is not None:
        plt.rcParams.update({"font.size": font_size})

    if sci_limits is not None:
        plt.rcParams.update({"axes.formatter.limits": sci_limits})

    E_kin_pred = pred_spectra_kin[index_t]
    E_mag_pred = pred_spectra_mag[index_t]

    E_kin_true = true_spectra_kin[index_t]
    E_mag_true = true_spectra_mag[index_t]

    fig = plt.figure(figsize=(6, 5))

    plt.loglog(k, E_kin_pred, style_kin_pred, label="$E_{kin}$ Pred")
    plt.loglog(k, E_kin_true, style_kin_true, label="$E_{kin}$ True")
    plt.loglog(k, E_mag_pred, style_mag_pred, label="$E_{mag}$ Pred")
    plt.loglog(k, E_mag_true, style_mag_true, label="$E_{mag}$ True")

    plt.xlabel("k")
    plt.ylabel("E(k)")
    plt.axis([xmin, xmax, ymin, ymax])

    plt.title(f"Spectra ${name}$")
    plt.legend(loc="upper right")

    if save_path is not None:
        if save_suffix is not None:
            figure_path = f"{save_path}_spectra_{save_suffix}.png"
        else:
            figure_path = f"{save_path}_spectra.png"
        plt.savefig(figure_path, bbox_inches="tight")

    return fig


def plot_predictions_mhd(
    pred,
    true,
    inputs,
    index_t=-1,
    names=[],
    save_path=None,
    save_suffix=None,
    font_size=None,
    sci_limits=None,
    shading="auto",
    cmap="jet",
):
    "Plots images of predictions and absolute error"
    if font_size is not None:
        plt.rcParams.update({"font.size": font_size})

    if sci_limits is not None:
        plt.rcParams.update({"axes.formatter.limits": sci_limits})
    # Plot
    fig = plt.figure(figsize=(24, 5 * len(names)))

    # Make plots for each field
    for index, name in enumerate(names):
        Nt, Nx, Ny, Nfields = pred.shape
        u_pred = pred[index_t, ..., index]
        u_true = true[index_t, ..., index]
        u_err = u_pred - u_true

        initial_data = inputs[0, ..., 3:]
        u0 = initial_data[..., index]

        x = inputs[0, :, 0, 1]
        y = inputs[0, 0, :, 2]
        X, Y = torch.meshgrid(x, y, indexing="ij")
        t = inputs[index_t, 0, 0, 0]

        plt.subplot(len(names), 4, index * 4 + 1)
        plt.pcolormesh(X, Y, u0, cmap=cmap, shading=shading)
        plt.colorbar()
        plt.title(f"Intial Condition ${name}_0(x,y)$")
        plt.tight_layout()
        plt.axis("square")
        plt.axis("off")

        plt.subplot(len(names), 4, index * 4 + 2)
        plt.pcolormesh(X, Y, u_true, cmap=cmap, shading=shading)
        plt.colorbar()
        plt.title(f"Exact ${name}(x,y,t={t:.2f})$")
        plt.tight_layout()
        plt.axis("square")
        plt.axis("off")

        plt.subplot(len(names), 4, index * 4 + 3)
        plt.pcolormesh(X, Y, u_pred, cmap=cmap, shading=shading)
        plt.colorbar()
        plt.title(f"Predict ${name}(x,y,t={t:.2f})$")
        plt.axis("square")
        plt.tight_layout()
        plt.axis("off")

        plt.subplot(len(names), 4, index * 4 + 4)
        plt.pcolormesh(X, Y, u_pred - u_true, cmap=cmap, shading=shading)
        plt.colorbar()
        plt.title(f"Absolute Error ${name}(x,y,t={t:.2f})$")
        plt.tight_layout()
        plt.axis("square")
        plt.axis("off")

    if save_path is not None:
        if save_suffix is not None:
            figure_path = f"{save_path}_{save_suffix}.png"
        else:
            figure_path = f"{save_path}.png"
        plt.savefig(figure_path, bbox_inches="tight")
    # plt.show()
    # return fig
    plt.close()


def generate_movie_2D(
    preds_y,
    test_y,
    test_x,
    key=0,
    plot_title="",
    field=0,
    val_cbar_index=-1,
    err_cbar_index=-1,
    val_clim=None,
    err_clim=None,
    font_size=None,
    movie_dir="",
    movie_name="movie.gif",
    frame_basename="movie",
    frame_ext="jpg",
    cmap="jet",
    shading="gouraud",
    remove_frames=True,
):
    "Generates a movie of the exact, predicted, and absolute error fields"
    frame_files = []

    if movie_dir:
        os.makedirs(movie_dir, exist_ok=True)

    if font_size is not None:
        plt.rcParams.update({"font.size": font_size})

    pred = preds_y[key][..., field]
    true = test_y[key][..., field]
    inputs = test_x[key]
    error = pred - true

    Nt, Nx, Ny = pred.shape

    t = inputs[:, 0, 0, 0]
    x = inputs[0, :, 0, 1]
    y = inputs[0, 0, :, 2]
    X, Y = torch.meshgrid(x, y, indexing="ij")

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]

    pcm1 = ax1.pcolormesh(
        X, Y, true[val_cbar_index], cmap=cmap, label="true", shading=shading
    )
    pcm2 = ax2.pcolormesh(
        X, Y, pred[val_cbar_index], cmap=cmap, label="pred", shading=shading
    )
    pcm3 = ax3.pcolormesh(
        X, Y, error[err_cbar_index], cmap=cmap, label="error", shading=shading
    )

    if val_clim is None:
        val_clim = pcm1.get_clim()
    if err_clim is None:
        err_clim = pcm3.get_clim()

    pcm1.set_clim(val_clim)
    plt.colorbar(pcm1, ax=ax1)
    ax1.axis("square")
    ax1.set_axis_off()

    pcm2.set_clim(val_clim)
    plt.colorbar(pcm2, ax=ax2)
    ax2.axis("square")
    ax2.set_axis_off()

    pcm3.set_clim(err_clim)
    plt.colorbar(pcm3, ax=ax3)
    ax3.axis("square")
    ax3.set_axis_off()

    plt.tight_layout()

    for i in range(Nt):
        # Exact
        ax1.clear()
        pcm1 = ax1.pcolormesh(X, Y, true[i], cmap=cmap, label="true", shading=shading)
        pcm1.set_clim(val_clim)
        ax1.set_title(f"Exact {plot_title}: $t={t[i]:.2f}$")
        ax1.axis("square")
        ax1.set_axis_off()

        # Predictions
        ax2.clear()
        pcm2 = ax2.pcolormesh(X, Y, pred[i], cmap=cmap, label="pred", shading=shading)
        pcm2.set_clim(val_clim)
        ax2.set_title(f"Predict {plot_title}: $t={t[i]:.2f}$")
        ax2.axis("square")
        ax2.set_axis_off()

        # Error
        ax3.clear()
        pcm3 = ax3.pcolormesh(X, Y, error[i], cmap=cmap, label="error", shading=shading)
        pcm3.set_clim(err_clim)
        ax3.set_title(f"Error {plot_title}: $t={t[i]:.2f}$")
        ax3.axis("square")
        ax3.set_axis_off()

        #         plt.tight_layout()
        fig.canvas.draw()

        if movie_dir:
            frame_path = os.path.join(movie_dir, f"{frame_basename}-{i:03}.{frame_ext}")
            frame_files.append(frame_path)
            plt.savefig(frame_path, bbox_inches="tight")

    if movie_dir:
        movie_path = os.path.join(movie_dir, movie_name)
        with imageio.get_writer(movie_path, mode="I") as writer:
            for frame in frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)

    if movie_dir and remove_frames:
        for frame in frame_files:
            try:
                os.remove(frame)
            except:
                pass


def plot_predictions_mhd_plotly(
    pred,
    true,
    inputs,
    index=0,
    index_t=-1,
    name="u",
    save_path=None,
    font_size=None,
    shading="auto",
    cmap="jet",
):
    "Plots images of predictions and absolute error to be saved to wandb"
    Nt, Nx, Ny, Nfields = pred.shape
    u_pred = pred[index_t, ..., index]
    u_true = true[index_t, ..., index]

    ic = inputs[0, ..., 3:]
    u_ic = ic[..., index]
    u_err = u_pred - u_true

    x = inputs[0, :, 0, 1]
    y = inputs[0, 0, :, 2]
    X, Y = torch.meshgrid(x, y, indexing="ij")
    t = inputs[index_t, 0, 0, 0]

    zmin = u_true.min().item()
    zmax = u_true.max().item()
    labels = {"color": name}

    # Initial Conditions
    title_ic = f"{name}0"
    fig_ic = px.imshow(
        u_ic,
        binary_string=False,
        color_continuous_scale=cmap,
        labels=labels,
        title=title_ic,
    )
    fig_ic.update_xaxes(showticklabels=False)
    fig_ic.update_yaxes(showticklabels=False)

    # Predictions
    title_pred = f"Predict {name}: t={t:.2f}"
    fig_pred = px.imshow(
        u_pred,
        binary_string=False,
        color_continuous_scale=cmap,
        labels=labels,
        title=title_pred,
    )
    fig_pred.update_xaxes(showticklabels=False)
    fig_pred.update_yaxes(showticklabels=False)

    # Ground Truth
    title_true = f"Exact {name}: t={t:.2f}"
    fig_true = px.imshow(
        u_true,
        binary_string=False,
        color_continuous_scale=cmap,
        labels=labels,
        title=title_true,
    )
    fig_true.update_xaxes(showticklabels=False)
    fig_true.update_yaxes(showticklabels=False)

    # Ground Truth
    title_err = f"Error {name}: t={t:.2f}"
    fig_err = px.imshow(
        u_err,
        binary_string=False,
        color_continuous_scale=cmap,
        labels=labels,
        title=title_err,
    )
    fig_err.update_xaxes(showticklabels=False)
    fig_err.update_yaxes(showticklabels=False)

    return fig_ic, fig_pred, fig_true, fig_err
