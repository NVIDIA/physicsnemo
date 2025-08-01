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

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import os

import matplotlib.pyplot as plt
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader
import torch.nn.functional as F

from physicsnemo.models.fno import FNO
from physicsnemo.launch.logging import LaunchLogger
from physicsnemo.launch.utils.checkpoint import save_checkpoint

from train_utils.datasets import DataLoader2D_swe
from swe_nl_pde import SWE_NL
from train_utils.losses import (
    swe_loss,
    ic_loss,
    pino_loss_swe_nonlin,
    physicsnemo_fourier,
)
from train_utils.plot import plot_predictions, generate_movie


def test_step(model, dataloader, log, cfg, swe_nl_node, device, option):
    """Test Step"""
    model.eval()

    batch_size = cfg.batchsize
    padding = cfg.model.fno.padding

    Nx = cfg.data.nx
    Ny = cfg.data.nx
    Nt = cfg.data.nt + 1
    Ntest = cfg.data.n_test
    in_dim = cfg.model.fno.in_channels
    out_dim = cfg.model.fno.out_channels
    key_t = (Nt - 1) // 1

    test_x = np.zeros((Ntest, Nx, Ny, Nt, in_dim))
    preds_y = np.zeros((Ntest, Nx, Ny, Nt, out_dim))
    test_y0 = np.zeros((Ntest, Nx, Ny, out_dim))
    test_y = np.zeros((Ntest, Nx, Ny, Nt, out_dim))

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            x, y = data
            x, y = x.to(device), y.to(device)
            x_in = F.pad(x, (0, 0, 0, padding), "constant", 0)
            x_in = x_in.permute(0, 4, 3, 1, 2)
            # compute forward pass
            out = (
                model(x_in)
                .permute(0, 3, 4, 2, 1)
                .reshape(batch_size, Nx, Ny, Nt + padding, out_dim)
            )
            out = out[..., :-padding, :]
            pred_y = out.reshape(y.shape)

            if cfg.loss.derivative == "original":
                loss_pde = PINO_loss_swe_nonlin(
                    out,
                    g=cfg.data.g,
                    nu=cfg.data.nu,
                    h_weight=cfg.loss.h_loss,
                    u_weight=cfg.loss.u_loss,
                    v_weight=cfg.loss.v_loss,
                )
            elif cfg.loss.derivative == "physicsnemo":
                loss_pde = physicsnemo_fourier(
                    out,
                    swe_nl_node,
                    h_weight=cfg.loss.h_loss,
                    u_weight=cfg.loss.u_loss,
                    v_weight=cfg.loss.v_loss,
                )

            # Compute data loss
            loss_l2 = swe_loss(out, y, H=cfg.data.H)

            # convert data to numpy
            test_x[i] = x.cpu().numpy()
            test_y[i] = y.cpu().numpy()
            test_y0[i] = (
                x[..., 0, -out_dim:].cpu().numpy()
            )  # same way as in training code
            preds_y[i] = pred_y.cpu().numpy()

            log.log_minibatch(
                {"test loss_l2": loss_l2.detach(), "test loss_pde": loss_pde.detach()}
            )

        if option == "plot":
            figures_dir = to_absolute_path("outputs_pino/figures/")
            os.makedirs(figures_dir, exist_ok=True)

            for key in range(len(preds_y)):
                save_path = os.path.join(figures_dir, f"SWE_{key}")
                plot_predictions(
                    key,
                    key_t,
                    test_x,
                    test_y,
                    preds_y,
                    print_index=True,
                    save_path=save_path,
                )
        elif option == "movie":
            movie_dir = to_absolute_path("outputs_pino/movies/")

            for key in range(len(preds_y)):
                if key == cfg.movie.nmovies:
                    break
                for field, name in enumerate(["eta", "u", "v"]):
                    movie_name = f"SWE_NonLinear_{key}_{name}.gif"
                    frame_basename = f"SWE_NonLinear_{key}_{name}_frame"
                    if field == 0:
                        name = "\\" + name
                    plot_title = f"${name}$"

                    generate_movie(
                        key,
                        test_x,
                        test_y,
                        preds_y,
                        plot_title=plot_title,
                        field=field,
                        val_cbar_index=cfg.movie.val_cbar_index,
                        err_cbar_index=cfg.movie.err_cbar_index,
                        movie_dir=movie_dir,
                        movie_name=movie_name,
                        frame_basename=frame_basename,
                        frame_ext=cfg.movie.frame_ext,
                        remove_frames=cfg.movie.remove_frames,
                        font_size=cfg.movie.font_size,
                    )


@hydra.main(version_base="1.3", config_path=".", config_name="config_pino.yaml")
def main(cfg: DictConfig):

    # CUDA support
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    LaunchLogger.initialize()

    # Define 2D Nonlinear Shallow Waters PDEs
    swe_nl = SWE_NL(g=cfg.data.g, nu=cfg.data.nu)
    swe_nl_node = swe_nl.make_nodes()

    # Load in dataset and make dataloader
    data = torch.load(
        to_absolute_path("datasets/swe_nl_dataset.pt"),
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    dataset = DataLoader2D_swe(
        data, cfg.data.nx, cfg.data.nt, cfg.data.sub, cfg.data.sub_t
    )

    train_loader = dataset.make_loader(
        cfg.data.n_train, cfg.batchsize, start=0, train=True
    )
    test_loader = dataset.make_loader(
        cfg.data.n_test, cfg.batchsize, start=cfg.data.n_train, train=False
    )

    # Define FNO model architecture
    model = FNO(
        in_channels=cfg.model.fno.in_channels,
        out_channels=cfg.model.fno.out_channels,
        decoder_layers=cfg.model.fno.decoder_layers,
        decoder_layer_size=cfg.model.fno.decoder_layer_size,
        dimension=cfg.model.fno.dimension,
        latent_channels=cfg.model.fno.latent_channels,
        num_fno_layers=cfg.model.fno.num_fno_layers,
        num_fno_modes=cfg.model.fno.num_fno_modes,
        padding=cfg.model.fno.padding,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), betas=(0.9, 0.999), lr=cfg.start_lr
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.milestones, gamma=cfg.gamma
    )

    S, T = dataset.S, dataset.T
    batch_size = cfg.batchsize
    padding = cfg.model.fno.padding
    nfields = cfg.model.fno.out_channels

    for epoch in range(cfg.max_epochs):
        # wrap epoch in launch logger for console logs
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(train_loader),
            epoch_alert_freq=10,
        ) as log:
            for x, y in train_loader:
                # Pad input to be put into model
                x, y = x.to(device), y.to(device)
                x_in = F.pad(x, (0, 0, 0, padding), "constant", 0)
                x_in = x_in.permute(0, 4, 3, 1, 2)
                # compute forward pass and unpad output
                out = (
                    model(x_in)
                    .permute(0, 3, 4, 2, 1)
                    .reshape(batch_size, S, S, T + padding, nfields)
                )
                out = out[..., :-padding, :]
                s0 = x[..., 0, -1]

                # Compute PDE loss using 'physicsnemo' functions or method from 'original' paper
                if cfg.loss.derivative == "original":
                    loss_pde = pino_loss_swe_nonlin(
                        out,
                        g=cfg.data.g,
                        nu=cfg.data.nu,
                        h_weight=cfg.loss.h_loss,
                        u_weight=cfg.loss.u_loss,
                        v_weight=cfg.loss.v_loss,
                    )
                elif cfg.loss.derivative == "physicsnemo":
                    loss_pde = physicsnemo_fourier(
                        out,
                        swe_nl_node,
                        h_weight=cfg.loss.h_loss,
                        u_weight=cfg.loss.u_loss,
                        v_weight=cfg.loss.v_loss,
                    )

                # Compute data loss
                loss_l2 = swe_loss(out, y, H=cfg.data.H)

                # Compute initial condition loss
                loss_ic = ic_loss(out, s0)

                # Compute total loss
                loss = (
                    cfg.loss.ic_loss * loss_ic
                    + cfg.loss.xy_loss * loss_l2
                    + cfg.loss.f_loss * loss_pde
                )

                # Backward pass and optimizer
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                log.log_minibatch(
                    {
                        "train loss": loss,
                        "train loss_ic": loss_ic.detach(),
                        "train loss_l2": loss_l2.detach(),
                        "train loss_pde": loss_pde.detach(),
                    }
                )
            # Learning rate update
            scheduler.step()
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
        save_checkpoint(
            "./checkpoints",
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )

        # Test model every 15 epochs
        if (epoch + 1) % 15 == 0:
            with LaunchLogger("test", epoch=epoch) as log:
                test_step(model, test_loader, log, cfg, swe_nl_node, device, "plot")

    # Generates movies of predictions
    with LaunchLogger("test", epoch=epoch) as log:
        test_step(model, test_loader, log, cfg, swe_nl_node, device, "movie")


if __name__ == "__main__":
    main()
