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

from physicsnemo.datapipes.cae.mesh_datapipe import MeshDatapipe
from physicsnemo.distributed import DistributedManager
import vtk
from physicsnemo.models.unet import UNet
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import torch
import hydra
import matplotlib.pyplot as plt
import torch.nn.functional as F
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import PythonLogger, LaunchLogger
from hydra.utils import to_absolute_path
from torch.nn.parallel import DistributedDataParallel
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from apex import optimizers
import os
import numpy as np
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes


def dilate_mask_3d(mask, padding_size):
    """Dilate a 3D mask by a specified padding size."""

    inverted_mask = (~mask.bool()).float()

    kernel_size = 2 * padding_size + 1
    kernel = torch.ones(
        (kernel_size, kernel_size, kernel_size), dtype=torch.float32
    ).to(mask.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    dilated_result = torch.clamp(
        torch.nn.functional.conv3d(inverted_mask, kernel, padding=padding_size), 0, 1
    )
    dilated_result = (~dilated_result.bool()).float()

    return dilated_result


def reshape_fortran(x, shape):
    """Based on https://stackoverflow.com/questions/63960352/reshaping-order-in-pytorch-fortran-like-index-ordering"""
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


@torch.no_grad()
def validation_step(
    model, dataset, pos_embed_tensor, epoch, plotting=False, device=None, name="default"
):
    loss_epoch = 0.0
    num_samples = 0.0

    nx, ny, nz = 960, 96, 80
    for i, data in enumerate(dataset):
        bs, _, chans = data[0]["x"].shape

        var = reshape_fortran(data[0]["x"], (bs, nx, ny, nz, chans))

        mask = torch.permute(var[..., 6:7], (0, 4, 1, 2, 3))
        invar = torch.permute(var[..., 5:6], (0, 4, 1, 2, 3))  # Grab Wall Distance
        invar = torch.cat((invar, pos_embed_tensor), axis=1)
        outvar = torch.permute(
            var[..., 0:5], (0, 4, 1, 2, 3)
        )  # Grab U components, T and P
        pred_outvar = model(invar)
        outvar = outvar * mask
        pred_outvar = pred_outvar * mask
        loss_epoch += F.mse_loss(outvar, pred_outvar)

        num_samples += invar.shape[0]

        if plotting:
            if i == 0:
                for chan in range(outvar.size(1)):
                    fig, ax = plt.subplots(1, 3)
                    vmin, vmax = np.min(
                        outvar[i, chan, :, :, nz // 2].detach().cpu().numpy()
                    ), np.max(outvar[i, chan, :, :, nz // 2].detach().cpu().numpy())
                    # plot z slices
                    im = ax[0].imshow(
                        outvar[i, chan, :, :, nz // 2].detach().cpu().numpy(),
                        vmin=vmin,
                        vmax=vmax,
                    )
                    fig.colorbar(im, ax=ax[0])
                    im = ax[1].imshow(
                        pred_outvar[i, chan, :, :, nz // 2].detach().cpu().numpy(),
                        vmin=vmin,
                        vmax=vmax,
                    )
                    fig.colorbar(im, ax=ax[1])
                    im = ax[2].imshow(
                        (
                            pred_outvar[i, chan, :, :, nz // 2]
                            - outvar[i, chan, :, :, nz // 2]
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    fig.colorbar(im, ax=ax[2])

                    ax[0].set_aspect("equal")
                    ax[1].set_aspect("equal")
                    ax[2].set_aspect("equal")

                    ax[0].set_title("True")
                    ax[1].set_title("Pred")
                    ax[2].set_title("Diff")

                    plt.savefig(f"chan_{chan}_epoch_{epoch}_mid_z_slice_{name}.png")
                    plt.close()

    return loss_epoch.detach() / num_samples


@hydra.main(
    version_base="1.2", config_path="conf", config_name="config_physics_informed"
)
def main(cfg: DictConfig) -> None:

    logger = PythonLogger("main")  # General python logger
    LaunchLogger.initialize()

    nx, ny, nz = 960, 96, 80

    # Compute positional embeddings
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)

    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    x_freq_sin = np.sin(xv * 72 * np.pi / 2)
    x_freq_cos = np.cos(xv * 72 * np.pi / 2)
    y_freq_sin = np.sin(yv * 8 * np.pi / 2)
    y_freq_cos = np.cos(yv * 8 * np.pi / 2)
    z_freq_sin = np.sin(zv * 8 * np.pi / 2)
    z_freq_cos = np.cos(zv * 8 * np.pi / 2)
    pos_embed = np.stack(
        (
            xv,
            x_freq_sin,
            x_freq_cos,
            yv,
            y_freq_sin,
            y_freq_cos,
            zv,
            z_freq_sin,
            z_freq_cos,
        ),
        axis=0,
    )

    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    pos_embed_tensor = torch.from_numpy(pos_embed).to(torch.float).to(dist.device)
    pos_embed_tensor = pos_embed_tensor.repeat(
        cfg.train_batch_size, 1, 1, 1, 1
    )  # repeat along the batch size dim

    model = UNet(
        in_channels=10,
        out_channels=5,
        model_depth=5,
        feature_map_channels=[32, 32, 64, 64, 128, 128, 256, 256, 512, 512],
        num_conv_blocks=2,
    ).to(dist.device)

    bounds = (0, 40, -3.95, 0.05, 0, 3.2)  # bounding box coordinates
    nx, ny, nz = 960, 96, 80

    # Define mean and std dictionaries
    mean_dict = {
        "T": 39,
        "U": 1.5983600616455078,
        "p": 6.1226935386657715,
        "wallDistance": 0.6676982045173645,
    }
    std_dict = {
        "T": 4,
        "U": 1.3656059503555298,
        "p": 4.166020393371582,
        "wallDistance": 0.45233625173568726,
    }

    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)

    phy_informer = PhysicsInformer(
        required_outputs=["continuity", "momentum_x", "momentum_y", "momentum_z"],
        equations=ns,
        grad_method="finite_difference",
        device=dist.device,
        fd_dx=[
            (bounds[1] - bounds[0]) / nx,
            (bounds[3] - bounds[2]) / ny,
            (bounds[5] - bounds[4]) / nz,
        ],
    )

    # Distributed learning (Data parallel)
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
        )

    # Initialize the dataset
    data_dir = to_absolute_path("./datasets/train/")
    dataset = MeshDatapipe(
        data_dir=data_dir,
        file_format="vtu",
        variables=["U", "T", "p", "wallDistance", "vtkValidPointMask"],
        num_variables=7,
        num_samples=cfg.train_num_samples,
        batch_size=cfg.train_batch_size,
        num_workers=1,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
        shuffle=True,
    )

    # Initialize the validation dataset
    if dist.rank == 0:
        pos_embed_tensor_val = (
            torch.from_numpy(pos_embed).to(torch.float).to(dist.device)
        )
        pos_embed_tensor_val = pos_embed_tensor_val.repeat(
            cfg.val_batch_size, 1, 1, 1, 1
        )  # repeat along the batch size dim
        val_data_dir = to_absolute_path("./datasets/test/")
        val_dataset = MeshDatapipe(
            data_dir=val_data_dir,
            file_format="vtu",
            variables=["U", "T", "p", "wallDistance", "vtkValidPointMask"],
            num_variables=7,
            num_samples=cfg.val_num_samples,
            batch_size=cfg.val_batch_size,
            num_workers=1,
            device=dist.device,
            process_rank=dist.rank,
            world_size=dist.world_size,
            shuffle=False,
        )

        train_dataset_plotting = MeshDatapipe(
            data_dir=data_dir,
            file_format="vtu",
            variables=["U", "T", "p", "wallDistance", "vtkValidPointMask"],
            num_variables=7,
            num_samples=16,
            batch_size=cfg.val_batch_size,
            num_workers=1,
            device=dist.device,
            process_rank=dist.rank,
            world_size=dist.world_size,
            shuffle=False,
        )

    optimizer = optimizers.FusedAdam(
        model.parameters(), betas=(0.9, 0.999), lr=cfg.start_lr, weight_decay=0.0
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=cfg.lr_scheduler_gamma
    )

    # Attempt to load latest checkpoint if one exists
    loaded_epoch = load_checkpoint(
        "./checkpoints",
        models=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=dist.device,
    )

    for epoch in range(max(1, loaded_epoch + 1), cfg.max_epochs + 1):  # epochs
        with LaunchLogger(
            "train", epoch=epoch, num_mini_batch=len(dataset), epoch_alert_freq=1
        ) as log:
            for i, data in enumerate(dataset):
                optimizer.zero_grad()
                bs, _, chans = data[0]["x"].shape

                var = reshape_fortran(data[0]["x"], (bs, nx, ny, nz, chans))

                mask = torch.permute(var[..., 6:7], (0, 4, 1, 2, 3))
                mask_dilated = dilate_mask_3d(mask, 3)
                invar = torch.permute(
                    var[..., 5:6], (0, 4, 1, 2, 3)
                )  # Grab Wall Distance
                invar = torch.cat(
                    (invar, pos_embed_tensor), axis=1
                )  # Concat along channel dim
                outvar = torch.permute(
                    var[..., 0:5], (0, 4, 1, 2, 3)
                )  # Grab U components, T and P
                pred_outvar = model(invar)
                phy_losses = phy_informer.forward(
                    {
                        "u": pred_outvar[:, 0:1] * std_dict["U"] + mean_dict["U"],
                        "v": pred_outvar[:, 1:2] * std_dict["U"] + mean_dict["U"],
                        "w": pred_outvar[:, 2:3] * std_dict["U"] + mean_dict["U"],
                        "p": pred_outvar[:, 4:5] * std_dict["p"] + mean_dict["p"],
                    }
                )

                phy_loss = 0.0
                for key in phy_losses.keys():
                    phy_loss += torch.mean(mask_dilated * phy_losses[key] ** 2)

                outvar = outvar * mask
                pred_outvar = pred_outvar * mask
                data_loss = F.mse_loss(outvar, pred_outvar)
                loss = data_loss + cfg.phy_wt * phy_loss
                loss.backward()
                optimizer.step()
                scheduler.step()

                log.log_minibatch({"Mini-batch data loss": data_loss.detach()})
                log.log_minibatch({"Mini-batch phy loss": phy_loss.detach()})
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        if dist.world_size > 1:
            torch.distributed.barrier()

        if dist.rank == 0:
            with LaunchLogger("valid", epoch=epoch) as log:
                train_loss = validation_step(
                    model,
                    train_dataset_plotting,
                    pos_embed_tensor_val,
                    epoch,
                    plotting=True,
                    name="train",
                )
                val_loss = validation_step(
                    model,
                    val_dataset,
                    pos_embed_tensor_val,
                    epoch,
                    plotting=True,
                    name="val",
                )
                log.log_epoch({"Val loss": val_loss, "Train loss": train_loss})

        if epoch % 20 == 0 and dist.rank == 0:
            save_checkpoint(
                "./checkpoints",
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )


if __name__ == "__main__":
    main()
