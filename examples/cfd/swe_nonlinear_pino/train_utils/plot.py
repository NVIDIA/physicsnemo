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

import torch
import imageio
import os
import matplotlib.pyplot as plt
from .utils import get_grid3d


def plot_predictions(
    key,
    key_t,
    test_x,
    test_y,
    preds_y,
    print_index=False,
    save_path=None,
    font_size=None,
):
    """Plot PINO predictions on dataset"""
    if font_size is not None:
        plt.rcParams.update({"font.size": font_size})

    Nsamples, Nx, Ny, Nt, Nfields = preds_y.shape

    a = test_x[key]
    # Nt, Nx, _ = a.shape
    h0 = a[..., 0, -1]
    # v0 = a[..., 0, -1]
    pred_h = preds_y[key, ..., key_t, 0]
    pred_u = preds_y[key, ..., key_t, 1]
    pred_v = preds_y[key, ..., key_t, 2]
    true_h = test_y[key, ..., key_t, 0]
    true_u = test_y[key, ..., key_t, 1]
    true_v = test_y[key, ..., key_t, 2]

    # T = a[:,:,2]
    # X = a[:,:,1]
    # x = X[0]
    x = torch.linspace(0, 1, Nx + 1)[:-1]
    y = torch.linspace(0, 1, Nx + 1)[:-1]
    X, Y = torch.meshgrid(x, y, indexing="ij")
    u0 = torch.zeros_like(X)
    v0 = torch.zeros_like(X)
    t = a[0, 0, key_t, 2]
    grid_x, grid_y, grid_t = get_grid3d(Nx, Nt)

    fig = plt.figure(figsize=(24, 15))
    plt.subplot(3, 4, 1)

    plt.pcolormesh(X, Y, h0, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    plt.title("Initial Condition $\eta(x,y)$")
    plt.tight_layout()
    plt.axis("square")

    plt.subplot(3, 4, 2)
    plt.pcolormesh(X, Y, true_h, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    #     plt.title(f'Exact $\eta(x,y,t={t:.2f})$')
    plt.title(f"Exact $\eta(x,y,t={int(t)})$")

    plt.tight_layout()
    plt.axis("square")

    plt.subplot(3, 4, 3)
    plt.pcolormesh(X, Y, pred_h, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    #     plt.title(f'Predict $\eta(x,y,t={t:.2f})$')
    plt.title(f"Predict $\eta(x,y,t={int(t)})$")
    plt.axis("square")

    plt.tight_layout()

    plt.subplot(3, 4, 4)
    plt.pcolormesh(X, Y, pred_h - true_h, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    plt.title("Absolute Error $\eta$")
    plt.tight_layout()
    plt.axis("square")

    plt.subplot(3, 4, 5)
    plt.pcolormesh(X, Y, u0, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    plt.title("Initial Condition $u(x,y)$")
    plt.tight_layout()
    plt.axis("square")

    plt.subplot(3, 4, 6)
    plt.pcolormesh(X, Y, true_u, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    #     plt.title(f'Exact $u(x,y,t={t:.2f})$')
    plt.title(f"Exact $u(x,y,t={int(t)})$")
    plt.tight_layout()
    plt.axis("square")

    plt.subplot(3, 4, 7)
    plt.pcolormesh(X, Y, pred_u, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    #     plt.title(f'Predict $u(x,y,t={t:.2f})$')
    plt.title(f"Predict $u(x,y,t={int(t)})$")
    plt.axis("square")

    plt.tight_layout()

    plt.subplot(3, 4, 8)
    plt.pcolormesh(X, Y, pred_u - true_u, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    plt.title("Absolute Error u")
    plt.tight_layout()
    plt.axis("square")

    plt.subplot(3, 4, 9)
    plt.pcolormesh(X, Y, v0, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    plt.title("Initial Condition $v(x,y)$")
    plt.tight_layout()
    plt.axis("square")

    plt.subplot(3, 4, 10)
    plt.pcolormesh(X, Y, true_v, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    #     plt.title(f'Exact $v(x,y,t={t:.2f})$')
    plt.title(f"Exact $v(x,y,t={int(t)})$")
    plt.tight_layout()
    plt.axis("square")

    plt.subplot(3, 4, 11)
    plt.pcolormesh(X, Y, pred_v, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    #     plt.title(f'Predict $v(x,y,t={t:.2f})$')
    plt.title(f"Predict $v(x,y,t={int(t)})$")
    plt.axis("square")

    plt.tight_layout()

    plt.subplot(3, 4, 12)
    plt.pcolormesh(X, Y, pred_v - true_v, cmap="jet", shading="gouraud")
    plt.colorbar()
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    plt.title("Absolute Error v")
    plt.tight_layout()
    plt.axis("square")

    if save_path is not None:
        plt.savefig(f"{save_path}.png", bbox_inches="tight")
    plt.close()


def generate_movie(
    key,
    test_x,
    test_y,
    preds_y,
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
    remove_frames=True,
):
    """Generates a movie on test predictions"""
    frame_files = []

    if movie_dir:
        os.makedirs(movie_dir, exist_ok=True)

    if font_size is not None:
        plt.rcParams.update({"font.size": font_size})

    if len(preds_y.shape) == 4:
        Nsamples, Nx, Ny, Nt = preds_y.shape
        preds_y = preds_y.reshape(Nsamples, Nx, Ny, Nt, 1)
        test_y = test_y.reshape(Nsamples, Nx, Ny, Nt, 1)
    Nsamples, Nx, Ny, Nt, Nfields = preds_y.shape

    pred = preds_y[key, ..., field]
    true = test_y[key, ..., field]
    error = pred - true

    a = test_x[key]
    x = torch.linspace(0, 1, Nx + 1)[:-1]
    y = torch.linspace(0, 1, Ny + 1)[:-1]
    X, Y = torch.meshgrid(x, y, indexing="ij")
    t = a[0, 0, :, 2]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]

    pcm1 = ax1.pcolormesh(
        X, Y, true[..., val_cbar_index], cmap="jet", label="true", shading="gouraud"
    )
    pcm2 = ax2.pcolormesh(
        X, Y, pred[..., val_cbar_index], cmap="jet", label="pred", shading="gouraud"
    )
    pcm3 = ax3.pcolormesh(
        X, Y, error[..., err_cbar_index], cmap="jet", label="error", shading="gouraud"
    )

    if val_clim is None:
        val_clim = pcm1.get_clim()
    if err_clim is None:
        err_clim = pcm3.get_clim()

    pcm1.set_clim(val_clim)
    plt.colorbar(pcm1, ax=ax1)
    ax1.axis("square")

    pcm2.set_clim(val_clim)
    plt.colorbar(pcm2, ax=ax2)
    ax2.axis("square")

    pcm3.set_clim(err_clim)
    plt.colorbar(pcm3, ax=ax3)
    ax3.axis("square")

    plt.tight_layout()

    for i in range(Nt):
        # Exact
        ax1.clear()
        pcm1 = ax1.pcolormesh(
            X, Y, true[..., i], cmap="jet", label="true", shading="gouraud"
        )
        pcm1.set_clim(val_clim)
        ax1.set_title(f"Exact {plot_title}: $t={t[i]:.2f}$")
        ax1.axis("square")

        # Predictions
        ax2.clear()
        pcm2 = ax2.pcolormesh(
            X, Y, pred[..., i], cmap="jet", label="pred", shading="gouraud"
        )
        pcm2.set_clim(val_clim)
        ax2.set_title(f"Predict {plot_title}: $t={t[i]:.2f}$")
        ax2.axis("square")

        # Error
        ax3.clear()
        pcm3 = ax3.pcolormesh(
            X, Y, error[..., i], cmap="jet", label="error", shading="gouraud"
        )
        pcm3.set_clim(err_clim)
        ax3.set_title(f"Error {plot_title}: $t={t[i]:.2f}$")
        ax3.axis("square")

        #         plt.tight_layout()
        fig.canvas.draw()

        if movie_dir:
            frame_path = os.path.join(movie_dir, f"{frame_basename}-{i:03}.{frame_ext}")
            frame_files.append(frame_path)
            plt.savefig(frame_path)

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
