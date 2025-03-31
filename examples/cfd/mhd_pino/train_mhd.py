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
from math import ceil
import torch
import plotly
import os

from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel
from omegaconf import OmegaConf

from physicsnemo.models.fno import FNO
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import (
    PythonLogger,
    LaunchLogger,
)
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.sym.hydra import to_absolute_path

from losses import LossMHD, LossMHD_PhysicsNeMo
from torch.optim import AdamW
from dataloaders import Dedalus2DDataset, MHDDataloader
from utils.plot_utils import plot_predictions_mhd, plot_predictions_mhd_plotly
import wandb

dtype = torch.float
# dtype = torch.double
torch.set_default_dtype(dtype)


@hydra.main(version_base="1.3", config_path="config", config_name="mhd_Re250.yaml")
def main(cfg: DictConfig) -> None:
    """Training for the 2D Darcy flow benchmark problem.

    This training script demonstrates how to set up a data-driven model for a 2D Darcy flow
    using Fourier Neural Operators (FNO) and acts as a benchmark for this type of operator.
    Training data is generated in-situ via the Darcy2D data loader from PhysicsNeMo. Darcy2D
    continuously generates data previously unseen by the model, i.e. the model is trained
    over a single epoch of a training set consisting of
    (cfg.training.max_pseudo_epochs*cfg.training.pseudo_epoch_sample_size) unique samples.
    Pseudo_epochs were introduced to leverage the LaunchLogger and its MLFlow integration.
    """

    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # initialize monitoring
    log = PythonLogger(name="mhd_pino")
    log.file_logging()

    wandb_dir = cfg.wandb_params.wandb_dir
    wandb_project = cfg.wandb_params.wandb_project
    wandb_group = cfg.wandb_params.wandb_group

    initialize_wandb(
        project=wandb_project,
        entity="fresleven",
        mode="offline",
        group=wandb_group,
        config=dict(cfg),
        results_dir=wandb_dir,
    )

    LaunchLogger.initialize(use_wandb=cfg.use_wandb)  # PhysicsNeMo launch logger

    # Load config file parameters
    model_params = cfg.model_params
    dataset_params = cfg.dataset_params
    train_loader_params = cfg.train_loader_params
    val_loader_params = cfg.val_loader_params
    test_loader_params = cfg.test_loader_params
    loss_params = cfg.loss_params
    optimizer_params = cfg.optimizer_params
    train_params = cfg.train_params
    wandb_params = cfg.wandb_params

    load_ckpt = cfg.load_ckpt
    output_dir = cfg.output_dir
    use_wandb = cfg.use_wandb

    output_dir = to_absolute_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    data_dir = dataset_params.data_dir
    ckpt_path = train_params.ckpt_path

    # Construct dataloaders
    dataset_train = Dedalus2DDataset(
        dataset_params.data_dir,
        output_names=dataset_params.output_names,
        field_names=dataset_params.field_names,
        num_train=dataset_params.num_train,
        num_test=dataset_params.num_test,
        use_train=True,
    )
    dataset_val = Dedalus2DDataset(
        data_dir,
        output_names=dataset_params.output_names,
        field_names=dataset_params.field_names,
        num_train=dataset_params.num_train,
        num_test=dataset_params.num_test,
        use_train=False,
    )

    mhd_dataloader_train = MHDDataloader(
        dataset_train,
        sub_x=dataset_params.sub_x,
        sub_t=dataset_params.sub_t,
        ind_x=dataset_params.ind_x,
        ind_t=dataset_params.ind_t,
    )
    mhd_dataloader_val = MHDDataloader(
        dataset_val,
        sub_x=dataset_params.sub_x,
        sub_t=dataset_params.sub_t,
        ind_x=dataset_params.ind_x,
        ind_t=dataset_params.ind_t,
    )

    dataloader_train, sampler_train = mhd_dataloader_train.create_dataloader(
        batch_size=train_loader_params.batch_size,
        shuffle=train_loader_params.shuffle,
        num_workers=train_loader_params.num_workers,
        pin_memory=train_loader_params.pin_memory,
        distributed=dist.distributed,
    )
    dataloader_val, sampler_val = mhd_dataloader_val.create_dataloader(
        batch_size=val_loader_params.batch_size,
        shuffle=val_loader_params.shuffle,
        num_workers=val_loader_params.num_workers,
        pin_memory=val_loader_params.pin_memory,
        distributed=dist.distributed,
    )

    # define FNO model
    model = FNO(
        in_channels=model_params.in_dim,
        out_channels=model_params.out_dim,
        decoder_layers=model_params.decoder_layers,
        decoder_layer_size=model_params.fc_dim,
        dimension=model_params.dimension,
        latent_channels=model_params.layers,
        num_fno_layers=model_params.num_fno_layers,
        num_fno_modes=model_params.modes,
        padding=[model_params.pad_z, model_params.pad_y, model_params.pad_x],
    ).to(dist.device)

    # Set up DistributedDataParallel if using more than a single process.
    # The `distributed` property of DistributedManager can be used to
    # check this.
    if dist.distributed:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            model = DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],  # Set the device_id to be
                # the local rank of this process on
                # this node
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    # Construct optimizer and scheduler
    # optimizer = Adam(model.parameters(), betas=optimizer_params['betas'], lr=optimizer_params['lr'])
    optimizer = AdamW(
        model.parameters(),
        betas=optimizer_params.betas,
        lr=optimizer_params.lr,
        weight_decay=0.1,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=optimizer_params.milestones, gamma=optimizer_params.gamma
    )

    # Construct Loss class
    if cfg.derivative == "physicsnemo":
        mhd_loss = LossMHD_PhysicsNeMo(**loss_params)
    elif cfg.derivative == "original":
        mhd_loss = LossMHD(**loss_params)

    # Load model from checkpoint (if exists)
    loaded_epoch = 0
    if load_ckpt:
        loaded_epoch = load_checkpoint(
            ckpt_path, model, optimizer, scheduler, device=dist.device
        )

    # Training Loop
    epochs = train_params.epochs
    ckpt_freq = train_params.ckpt_freq
    names = dataset_params.fields
    input_norm = torch.tensor(model_params.input_norm).to(dist.device)
    output_norm = torch.tensor(model_params.output_norm).to(dist.device)

    for epoch in range(max(1, loaded_epoch + 1), epochs + 1):
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(dataloader_train),
            epoch_alert_freq=1,
        ) as log:

            if dist.distributed:
                sampler_train.set_epoch(epoch)

            # Train Loop
            model.train()

            for i, (inputs, outputs) in enumerate(dataloader_train):
                inputs = inputs.type(torch.FloatTensor).to(dist.device)
                outputs = outputs.type(torch.FloatTensor).to(dist.device)
                # Zero Gradients
                optimizer.zero_grad()
                # Compute Predictions
                pred = (
                    model((inputs / input_norm).permute(0, 4, 1, 2, 3)).permute(
                        0, 2, 3, 4, 1
                    )
                    * output_norm
                )
                # Compute Loss
                loss, loss_dict = mhd_loss(pred, outputs, inputs, return_loss_dict=True)
                # Compute Gradients for Back Propagation
                loss.backward()
                # Update Weights
                optimizer.step()

                log.log_minibatch(loss_dict)

            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
            scheduler.step()

        with LaunchLogger("valid", epoch=epoch) as log:
            # Val loop
            model.eval()
            val_loss_dict = {}
            plot_count = 0
            plot_dict = {name: {} for name in names}
            with torch.no_grad():
                for i, (inputs, outputs) in enumerate(dataloader_val):
                    inputs = inputs.type(dtype).to(dist.device)
                    outputs = outputs.type(dtype).to(dist.device)

                    # Compute Predictions
                    pred = (
                        model((inputs / input_norm).permute(0, 4, 1, 2, 3)).permute(
                            0, 2, 3, 4, 1
                        )
                        * output_norm
                    )
                    # Compute Loss
                    loss, loss_dict = mhd_loss(
                        pred, outputs, inputs, return_loss_dict=True
                    )

                    log.log_minibatch(loss_dict)

                    # Get prediction plots to log for wandb
                    # Do for number of batches specified in the config file
                    if (i < wandb_params.wandb_num_plots) and (
                        epoch % wandb_params.wandb_plot_freq == 0
                    ):
                        # Add all predictions in batch
                        for j, _ in enumerate(pred):
                            # Make plots for each field
                            for index, name in enumerate(names):
                                # Generate figure
                                if use_wandb:
                                    figs = plot_predictions_mhd_plotly(
                                        pred[j].cpu(),
                                        outputs[j].cpu(),
                                        inputs[j].cpu(),
                                        index=index,
                                        name=name,
                                    )
                                    # Add figure to plot dict
                                    plot_dict[name] = {
                                        f"{plot_type}-{plot_count}": wandb.Html(
                                            plotly.io.to_html(fig)
                                        )
                                        for plot_type, fig in zip(
                                            wandb_params.wandb_plot_types, figs
                                        )
                                    }

                            plot_count += 1

                    # Get prediction plots and save images locally
                    if (i < 2) and (epoch % wandb_params.wandb_plot_freq == 0):
                        # Add all predictions in batch
                        for j, _ in enumerate(pred):
                            # Generate figure
                            plot_predictions_mhd(
                                pred[j].cpu(),
                                outputs[j].cpu(),
                                inputs[j].cpu(),
                                names=names,
                                save_path=os.path.join(
                                    output_dir,
                                    "MHD_" + cfg.derivative + "_" + str(dist.rank),
                                ),
                                save_suffix=i,
                            )

            if use_wandb and epoch % wandb_params.wandb_plot_freq == 0:
                wandb.log({"plots": plot_dict})

            if epoch % ckpt_freq == 0 and dist.rank == 0:
                save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch=epoch)


if __name__ == "__main__":
    main()
