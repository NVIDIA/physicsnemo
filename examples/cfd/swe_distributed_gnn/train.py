# ignore_header_test
# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# This example was adopted from torch-harmonics and modified to
# highlight the usage of GraphCast on the example of training
# a neural solver for the Shallow-Water-Equations (SWE) and
# to showcase the utilities in PhysicsNeMo when it comes to
# distributing a model in a tensor-parallel.

import argparse
import hydra
from omegaconf import DictConfig, OmegaConf
import json
import os
import time
import uuid
import logging
from typing import Optional

import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from torch.utils.data import DataLoader
from torch.cuda import amp

import numpy as np
import matplotlib.pyplot as plt

from shallow_water_pde_dataset import ShallowWaterPDEDataset

from physicsnemo.distributed import (
    DistributedManager,
    mark_module_as_shared,
    ProcessGroupConfig,
    ProcessGroupNode,
)
from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet


logger = logging.getLogger(__name__)
# as we write logs to an output directory, starting many jobs at the same
# time can lead to conflicts, to avoid this, let hydra define a random
# uuid appended to the output path to avoid these conflicts
OmegaConf.register_new_resolver("rand_uuid", lambda: int(uuid.uuid4()))


def l2loss_sphere(prd, tar, solver, relative=False, squared=True):
    loss = solver.integrate_grid((prd - tar) ** 2, dimensionless=True).sum(dim=-1)
    if relative:
        loss = loss / solver.integrate_grid(tar**2, dimensionless=True).sum(dim=-1)

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss


def l2loss(prd, tar, *args, **kwargs):
    loss = ((prd - tar) ** 2).mean()
    return loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# rolls out the model and compares to the classical solver
def autoregressive_inference(
    model,
    dataset,
    path_root,
    nsteps,
    dist_manager,
    cfg,
):
    nskip = 1
    plot_channel = 1
    nics = 20

    if cfg.model.loss_fn == "l2":
        loss_fn = l2loss
    elif cfg.model.loss_fn == "l2sphere":
        loss_fn = l2loss_sphere
    else:
        raise ValueError(
            f"loss_fn={cfg.model.loss_fn} not supported, expected 'l2' or 'l2sphere'."
        )

    losses = np.zeros(nics)
    graphcast_times = np.zeros(nics)
    nwp_times = np.zeros(nics)

    for iic in range(nics):
        if dist_manager.rank == 0:
            ic = dataset.solver.random_initial_condition(mach=0.1)
            inp_mean = dataset.inp_mean
            inp_var = dataset.inp_var

            prd = (dataset.solver.spec2grid(ic) - inp_mean) / torch.sqrt(inp_var)
            prd = prd.unsqueeze(0)
            uspec = ic.clone()

        else:
            prd = torch.empty((1, 3, dataset.nlat, dataset.nlon), device=model.device)

        # ML model
        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(1, cfg.data.autoreg_steps + 1):
            # evaluate the ML model
            prd = model(prd)

            if (
                (dist_manager.rank == 0)
                and (iic == nics - 1)
                and (nskip > 0)
                and (i % nskip == 0)
            ):

                # do plotting, as we aggregated output only on rank0, only use
                # data from rank 0
                fig = plt.figure(figsize=(7.5, 6))
                dataset.solver.plot_griddata(prd[0, plot_channel], fig, vmax=4, vmin=-4)
                plt.savefig(os.path.join(path_root, f"pred_{i//nskip}.png"))
                plt.clf()

        torch.cuda.synchronize()
        graphcast_times[iic] = time.time() - start_time

        # classical model, not parallel so only on rank 0
        if dist_manager.rank == 0:
            torch.cuda.synchronize()
            start_time = time.time()
            for i in range(1, cfg.data.autoreg_steps + 1):

                # advance classical model
                uspec = dataset.solver.timestep(uspec, nsteps)

                if iic == nics - 1 and i % nskip == 0 and nskip > 0:
                    ref = (dataset.solver.spec2grid(uspec) - inp_mean) / torch.sqrt(
                        inp_var
                    )

                    fig = plt.figure(figsize=(7.5, 6))
                    dataset.solver.plot_griddata(
                        ref[plot_channel], fig, vmax=4, vmin=-4
                    )
                    plt.savefig(os.path.join(path_root, f"truth_{i//nskip}.png"))
                    plt.clf()

            torch.cuda.synchronize()
            nwp_times[iic] = time.time() - start_time

            ref = dataset.solver.spec2grid(uspec)
            prd = prd * torch.sqrt(inp_var) + inp_mean

            losses[iic] = loss_fn(prd, ref, solver=dataset.solver, relative=True).item()

    return losses, graphcast_times, nwp_times


# training function
def train_model(
    model,
    dataloader,
    optimizer,
    scheduler,
    gscaler,
    dist_manager,
    metrics,
    cfg,
):
    if cfg.model.loss_fn == "l2":
        loss_fn = l2loss
    elif cfg.model.loss_fn == "l2sphere":
        loss_fn = l2loss_sphere
    else:
        raise ValueError(
            f"loss_fn={cfg.model.loss_fn} not supported, expected 'l2' or 'l2sphere'."
        )
    torch.cuda.synchronize()
    train_start = time.time()

    dist_manager = DistributedManager()
    try:
        dp_group_size = dist_manager.group_size("data_parallel")
    except:
        dp_group_size = 1

    # count iterations
    iters = 0

    metrics["current_lr"] = []
    metrics["epoch_time"] = []
    metrics["acc_loss"] = []
    metrics["val_loss"] = []
    metrics["epochs"] = []

    for epoch in tqdm(range(cfg.data.num_epochs)):
        # time each epoch
        torch.cuda.synchronize()
        epoch_start = time.time()

        dataloader.dataset.set_initial_condition("random")
        dataloader.dataset.set_num_examples(cfg.data.num_examples // dp_group_size)

        # get the solver for its convenience functions
        solver = dataloader.dataset.solver

        # do the training
        acc_loss = 0
        model.train()

        for inp, tar in tqdm(dataloader, leave=False):
            with amp.autocast(enabled=cfg.model.enable_amp):
                prd = model(inp)
                for _ in range(cfg.data.num_future):
                    prd = model(prd)
                loss = loss_fn(prd, tar, solver=solver, relative=False)

            acc_loss += loss.item() * inp.size(0)
            optimizer.zero_grad(set_to_none=True)
            gscaler.scale(loss).backward()
            gscaler.step(optimizer)
            gscaler.update()

            iters += 1

        acc_loss = acc_loss / len(dataloader.dataset)

        dataloader.dataset.set_initial_condition("random")
        dataloader.dataset.set_num_examples(cfg.data.num_valid)

        # perform validation
        valid_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inp, tar in dataloader:
                prd = model(inp)
                for _ in range(cfg.data.num_future):
                    prd = model(prd)
                loss = loss_fn(prd, tar, solver=solver, relative=True)
                valid_loss += loss.item() * inp.size(0)

        valid_loss = valid_loss / len(dataloader.dataset)

        torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start

        if metrics is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            metrics["epochs"].append(epoch)
            metrics["current_lr"].append(current_lr)
            metrics["epoch_time"].append(epoch_time)
            metrics["acc_loss"].append(acc_loss)
            metrics["val_loss"].append(valid_loss)

        scheduler.step()

    torch.cuda.synchronize()
    train_time = time.time() - train_start

    if dist_manager.rank == 0:
        logger.info(f"Training took {train_time}.")
        logger.info("Training Metrics")
        for k, v in metrics.items():
            logger.info(f"{k}: {v}")


def get_random_data(dataset, device):
    inp = torch.randn((1, 3, dataset.nlat, dataset.nlon), device=device)
    tar = torch.randn((1, 3, dataset.nlat, dataset.nlon), device=device)
    return inp, tar


def train_model_on_dummy_data(
    model,
    dataloader,
    optimizer,
    scheduler,
    gscaler,
    dist_manager,
    metrics,
    cfg,
):
    torch.cuda.synchronize()
    train_start = time.time()

    dist_manager = DistributedManager()
    try:
        dp_group_size = dist_manager.group_size("data_parallel")
    except:
        dp_group_size = 1
    print(dp_group_size)
    raise ValueError
    # count iterations
    iters = 0

    metrics["current_lr"] = []
    metrics["epoch_time"] = []
    metrics["acc_loss"] = []
    metrics["val_loss"] = []
    metrics["epochs"] = []

    for epoch in tqdm(range(cfg.data.num_epochs)):
        # time each epoch
        torch.cuda.synchronize()
        epoch_start = time.time()

        dataloader.dataset.set_num_examples(cfg.data.num_examples // dp_group_size)

        # do the training
        acc_loss = 0
        model.train()

        for train_batch in tqdm(range(dataloader.dataset.num_examples), leave=False):
            inp, tar = get_random_data(dataloader.dataset, model.device)
            with amp.autocast(enabled=cfg.model.enable_amp):
                prd = model(inp)
                for _ in range(cfg.data.num_future):
                    prd = model(prd)
                loss = l2loss(prd, tar)

            acc_loss += loss.item() * inp.size(0)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            gscaler.step(optimizer)
            gscaler.update()

            iters += 1

        acc_loss = acc_loss / len(dataloader.dataset)

        dataloader.dataset.set_num_examples(cfg.data.num_valid)

        # perform validation
        valid_loss = 0.0
        model.eval()
        with torch.no_grad():
            for val_batch in range(cfg.data.num_valid):
                inp, tar = get_random_data(dataloader.dataset, model.device)
                prd = model(inp)
                for _ in range(cfg.data.num_future):
                    prd = model(prd)
                loss = l2loss(prd, tar)
                valid_loss += loss.item() * inp.size(0)

        valid_loss = valid_loss / len(dataloader.dataset)

        torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start

        if metrics is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            metrics["epochs"].append(epoch)
            metrics["current_lr"].append(current_lr)
            metrics["epoch_time"].append(epoch_time)
            metrics["acc_loss"].append(acc_loss)
            metrics["val_loss"].append(valid_loss)

        scheduler.step()

    torch.cuda.synchronize()
    train_time = time.time() - train_start

    if dist_manager.rank == 0:
        logger.info(f"Training took {train_time}.")
        logger.info("Training Metrics")
        for k, v in metrics.items():
            logger.info(f"{k}: {v}")


@hydra.main(
    version_base="1.3", config_path=".", config_name="config_graphcast_swe.yaml"
)
def main(cfg: DictConfig):
    # realistic FP32 / TF32 settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # default of 0.703125 gives (256, 512)
    input_res = (
        int(180.0 / cfg.data.angular_resolution),
        int(360.0 / cfg.data.angular_resolution),
    )

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    # initialize DistributedManager and define process group
    # across which graph is partitioned and model is parallel across
    # for now, based
    DistributedManager.initialize()
    if DistributedManager().distributed:
        graph_partition_pg_name = "model_parallel"
        world_size = torch.distributed.get_world_size()
        graph_partition_size = cfg.model.graph_partition_size
        if graph_partition_size < 0:
            graph_partition_size = world_size
        if not world_size % graph_partition_size == 0:
            raise ValueError(
                f"Partition Size ({graph_partition_size}) must divide World Size ({world_size}) evenly."
            )
        world = ProcessGroupNode("world")
        pg_config = ProcessGroupConfig(world)
        pg_config.add_node(ProcessGroupNode("data_parallel"), parent=world)
        pg_config.add_node(ProcessGroupNode("model_parallel"), parent=world)
        pg_sizes = {
            "model_parallel": graph_partition_size,
            "data_parallel": world_size // graph_partition_size,
        }
        pg_config.set_leaf_group_sizes(pg_sizes)
        DistributedManager.create_groups_from_config(
            pg_config,
            verbose=True,
        )
    else:
        world_size = 1
        graph_partition_size = 1
        graph_partition_pg_name = None

    dist_manager = DistributedManager()

    hyperparameters = {
        "world_size": world_size,
        **OmegaConf.to_container(cfg, resolve=True),
    }

    if dist_manager.rank == 0:
        logger.info(f"starting to run ...")

    model = GraphCastNet(
        multimesh_level=cfg.data.multimesh_level,
        input_res=input_res,
        input_dim_grid_nodes=3,
        input_dim_mesh_nodes=3,
        input_dim_edges=4,
        output_dim_grid_nodes=3,
        processor_layers=cfg.model.processor_layers,
        hidden_dim=cfg.model.hidden_dim,
        partition_size=graph_partition_size,
        partition_group_name=graph_partition_pg_name,
        # simplified data-loading scheme: only rank 0 has valid inputs
        # model then takes care of scattering these onto participating ranks
        expect_partitioned_input=False,
        global_features_on_rank_0=True,
        # simplilfied loss computation, to allow e.g. the l2_loss_sphere
        # without having to distribute this loss computation, valid
        # output is only on rank 0, model aggregates the output accordingly
        produce_aggregated_output=True,
        produce_aggregated_output_on_all_ranks=False,
        use_lat_lon_partitioning=cfg.model.use_lat_lon_partitioning,
    ).to(device=dist_manager.device)

    if dist_manager.distributed and dist_manager.group_size("data_parallel") > 1:
        model = DistributedDataParallel(
            model,
            process_group=dist_manager.group("data_parallel"),
            device_ids=[dist_manager.local_rank],
            output_device=dist_manager.device,
        )

    # since model is "tensor-parallel" in graph-partition
    # mark model as "shared" which sets gradient hooks and
    # aggregates gradients in the backward pass accordingly
    if (
        dist_manager.distributed
        and dist_manager.group_size(graph_partition_pg_name) > 1
    ):
        mark_module_as_shared(model, graph_partition_pg_name)

    num_params = count_parameters(model)
    # prepare dicts containing models and corresponding metrics
    metrics = {**hyperparameters}

    if dist_manager.rank == 0:
        logger.info(f"number of trainable params: {num_params}")
    metrics["num_params"] = num_params

    if cfg.load_checkpoint:
        if cfg.checkpoint_dir is None:
            raise ValueError(
                "load_checkpoint=True but cfg.checkpoint_dir is not set, abort."
            )
        file_name = os.path.join(cfg.checkpoint_dir, "checkpoints", "model.mdlus")
        model.load(file_name, map_location=manager.device)

    nsteps = cfg.data.dt // cfg.data.dt_solver
    mp_rank = (
        0 if graph_partition_size <= 1 else dist_manager.group_rank("model_parallel")
    )
    dataset = ShallowWaterPDEDataset(
        dt=cfg.data.dt,
        nsteps=nsteps,
        dims=input_res,
        device=dist_manager.device,
        normalize=True,
        rank=mp_rank,
    )

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, persistent_workers=False
    )

    if cfg.run_training:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.lr_milestones, gamma=cfg.gamma
        )
        gscaler = amp.GradScaler(enabled=cfg.model.enable_amp)

        torch.cuda.synchronize()
        start_time = time.time()

        if dist_manager.rank == 0:
            logger.info(f"Training, started ...")

        if cfg.data.dummy_data:
            train_fn = train_model_on_dummy_data
        else:
            train_fn = train_model

        train_fn(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            gscaler=gscaler,
            dist_manager=dist_manager,
            metrics=metrics,
            cfg=cfg,
        )

        torch.cuda.synchronize()
        training_time = time.time() - start_time

        run_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        if dist_manager.rank == 0:
            os.makedirs(os.path.join(run_output_dir, "figures"), exist_ok=True)
            os.makedirs(os.path.join(run_output_dir, "output_data"), exist_ok=True)
            if cfg.save_checkpoint:
                os.makedirs(os.path.join(run_output_dir, "checkpoints"), exist_ok=True)
                file_name = os.path.join(run_output_dir, "checkpoints", "model.mdlus")
                model.save(file_name)

    # skip inference for dummy_data
    if not cfg.data.dummy_data:
        # set seed
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)

        with torch.inference_mode():
            losses, graphcast_times, nwp_times = autoregressive_inference(
                model,
                dataset,
                os.path.join(run_output_dir, "figures"),
                nsteps,
                dist_manager=dist_manager,
                cfg=cfg,
            )
            metrics["loss_mean"] = np.mean(losses)
            metrics["loss_std"] = np.std(losses)
            metrics["graphcast_time_mean"] = np.mean(graphcast_times)
            metrics["graphcast_time_std"] = np.std(graphcast_times)
            metrics["nwp_time_mean"] = np.mean(nwp_times)
            metrics["nwp_time_std"] = np.std(nwp_times)

    max_mem = torch.tensor(
        [
            torch.cuda.max_memory_allocated() * 1.0 / (1024**3),
            torch.cuda.max_memory_reserved() * 1.0 / (1024**3),
        ],
        dtype=torch.float32,
        device=dist_manager.device,
    ).view(1, 2)
    if dist_manager.rank == 0:
        gather_list = [
            torch.empty_like(max_mem) for r in range(dist_manager.world_size)
        ]
    else:
        gather_list = []

    if DistributedManager().distributed:
        torch.distributed.gather(max_mem, gather_list, dst=0)
    else:
        max_mem = gather_list[0]

    if dist_manager.rank == 0:
        if cfg.run_training:
            max_mem = torch.vstack(gather_list)
            metrics["training_time"] = training_time
            metrics[f"max_memory_allocated_gib"] = max_mem[:, 0].tolist()
            metrics[f"max_memory_reserved_gib"] = max_mem[:, 1].tolist()

        with open(
            os.path.join(run_output_dir, "output_data", "metrics.json"), "w"
        ) as f:
            json.dump(metrics, f)

    if DistributedManager().distributed:
        DistributedManager.cleanup()


if __name__ == "__main__":
    main()
