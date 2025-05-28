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

import time
import hydra
from hydra.utils import to_absolute_path
import torch
import wandb
import dgl
from dgl.dataloading import GraphDataLoader
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn

from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from utils import custom_loss, compute_physics_loss

# Custom collate function that checks if each item is a tuple (graph, physics_data) or a plain graph.
def collate_fn(batch):
    if isinstance(batch[0], tuple):
        graphs, physics_list = zip(*batch)
        batched_graph = dgl.batch(graphs)
        physics_data = {}
        # For each key, build a tensor by stacking the scalar values from each sample.
        for key in physics_list[0].keys():
            physics_data[key] = torch.tensor(
                [d[key] for d in physics_list], dtype=torch.float
            )
        return batched_graph, physics_data
    else:
        return dgl.batch(batch)


class MGNTrainer:
    def __init__(self, cfg: DictConfig, rank_zero_logger: RankZeroLoggingWrapper):
        # Ensure distributed manager is initialized.
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.amp = cfg.amp
        self.noise_type = cfg.noise_type

        # Physics loss settings.
        self.use_physics_loss = cfg.get("use_physics_loss", False)
        self.delta_t = cfg.get("delta_t", 1200.0)
        self.physics_loss_weight = cfg.get("physics_loss_weight", 1.0)

        # Set activation function.
        mlp_act = "relu"
        if cfg.recompute_activation:
            rank_zero_logger.info(
                "Setting MLP activation to SiLU for recompute_activation."
            )
            mlp_act = "silu"

        rank_zero_logger.info("Initializing HydroGraphDataset...")
        # Pass the flag to the dataset so it returns physics data only if needed.
        dataset = HydroGraphDataset(
            name="hydrograph_dataset",
            data_dir=cfg.data_dir,
            prefix="M80",
            num_samples=500,
            n_time_steps=cfg.n_time_steps,
            k=4,
            noise_type=cfg.noise_type,
            noise_std=0.01,
            hydrograph_ids_file="train.txt",
            split="train",
            force_reload=False,
            verbose=False,
            return_physics=self.use_physics_loss,
        )
        self.dataloader = GraphDataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            use_ddp=self.dist.world_size > 1,
            num_workers=cfg.num_dataloader_workers,
            collate_fn=collate_fn,
        )
        rank_zero_logger.info("Dataset and dataloader initialization complete.")

        rank_zero_logger.info("Instantiating MeshGraphKAN model...")
        self.model = MeshGraphKAN(
            cfg.num_input_features,
            cfg.num_edge_features,
            cfg.num_output_features,
            mlp_activation_fn=mlp_act,
            do_concat_trick=cfg.do_concat_trick,
            num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
            recompute_activation=cfg.recompute_activation,
        )
        if cfg.jit:
            if not self.model.meta.jit:
                raise ValueError("MeshGraphKAN is not yet JIT-compatible.")
            self.model = torch.jit.script(self.model).to(self.dist.device)
        else:
            self.model = self.model.to(self.dist.device)
        rank_zero_logger.info("Model instantiated successfully.")

        if cfg.watch_model and not cfg.jit and self.dist.rank == 0:
            wandb.watch(self.model)

        if self.dist.world_size > 1:
            rank_zero_logger.info("Wrapping model in DistributedDataParallel...")
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        self.model.train()
        self.criterion = nn.MSELoss()
        try:
            if cfg.use_apex:
                from apex.optimizers import FusedAdam

                self.optimizer = FusedAdam(self.model.parameters(), lr=cfg.lr)
            else:
                self.optimizer = None
        except ImportError:
            rank_zero_logger.warning(
                "NVIDIA Apex is not installed; FusedAdam optimizer will not be used."
            )
            self.optimizer = None
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        rank_zero_logger.info(f"Using optimizer: {self.optimizer.__class__.__name__}")

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: cfg.lr_decay_rate**epoch
        )
        self.scaler = GradScaler()

        rank_zero_logger.info("Loading checkpoint if available...")
        if self.dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )
        rank_zero_logger.info(
            f"Checkpoint loaded. Starting training from epoch {self.epoch_init}."
        )

    def train(self, batch):
        if self.use_physics_loss:
            graph, physics_data = batch
        else:
            graph = batch
            physics_data = None
        graph = graph.to(self.dist.device)
        if physics_data is not None:
            physics_data = {k: v.to(self.dist.device) for k, v in physics_data.items()}
        self.optimizer.zero_grad()
        loss, loss_dict = self.forward(graph, physics_data)
        self.backward(loss)
        self.scheduler.step()
        return loss, loss_dict

    def forward(self, graph, physics_data):
        if self.noise_type == "pushforward":
            with autocast(enabled=self.amp):
                X = graph.ndata["x"]
                n_static = 12  # assumed static features dimension
                n_time = (X.shape[1] - n_static) // 2
                static_part = X[:, :n_static]
                water_depth_full = X[:, n_static : n_static + n_time]
                volume_full = X[:, n_static + n_time : n_static + 2 * n_time]
                # For one-step prediction, use dynamic features from indices 1: (last n_time_steps)
                water_depth_window_one = water_depth_full[:, 1:]
                volume_window_one = volume_full[:, 1:]
                X_one = torch.cat(
                    [static_part, water_depth_window_one, volume_window_one], dim=1
                )
                pred_one = self.model(X_one, graph.edata["x"], graph)
                one_step_loss = self.criterion(pred_one, graph.ndata["y"])

                # Stability branch (example implementation)
                water_depth_window_stab = water_depth_full[:, : n_time - 1]
                volume_window_stab = volume_full[:, : n_time - 1]
                X_stab = torch.cat(
                    [static_part, water_depth_window_stab, volume_window_stab], dim=1
                )
                pred_stab = self.model(X_stab, graph.edata["x"], graph)
                pred_stab_detached = pred_stab.detach()
                water_depth_updated = torch.cat(
                    [
                        water_depth_full[:, 1:2],
                        water_depth_full[:, 1:2] + pred_stab_detached[:, 0:1],
                    ],
                    dim=1,
                )
                volume_updated = torch.cat(
                    [
                        volume_full[:, 1:2],
                        volume_full[:, 1:2] + pred_stab_detached[:, 1:2],
                    ],
                    dim=1,
                )
                X_stab_updated = torch.cat(
                    [static_part, water_depth_updated, volume_updated], dim=1
                )
                pred_stab2 = self.model(X_stab_updated, graph.edata["x"], graph)
                stability_loss = self.criterion(pred_stab2, graph.ndata["y"])

                loss = one_step_loss + stability_loss
                loss_dict = {
                    "total_loss": loss,
                    "loss_one": one_step_loss,
                    "loss_stability": stability_loss,
                }
                if self.use_physics_loss and physics_data is not None:
                    phy_loss = compute_physics_loss(
                        pred_one, physics_data, graph, delta_t=self.delta_t
                    )
                    loss = loss + self.physics_loss_weight * phy_loss
                    loss_dict["physics_loss"] = phy_loss
            return loss, loss_dict
        else:
            with autocast(enabled=self.amp):
                pred = self.model(graph.ndata["x"], graph.edata["x"], graph)
                mse_loss = self.criterion(pred, graph.ndata["y"])
                loss = mse_loss
                loss_dict = {"total_loss": loss, "mse_loss": mse_loss}
                if self.use_physics_loss and physics_data is not None:
                    phy_loss = compute_physics_loss(
                        pred, physics_data, graph, delta_t=self.delta_t
                    )
                    loss = loss + self.physics_loss_weight * phy_loss
                    loss_dict["physics_loss"] = phy_loss
            return loss, loss_dict

    def backward(self, loss):
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    initialize_wandb(
        project="Modulus-Launch",
        entity="Modulus",
        name="Vortex_Shedding-Training",
        group="Vortex_Shedding-DDP-Group",
        mode=cfg.wandb_mode,
    )
    logger = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging()
    rank_zero_logger.info(f"Starting training process with configuration: {cfg}")
    trainer = MGNTrainer(cfg, rank_zero_logger)
    rank_zero_logger.info("Beginning training loop...")
    start_time = time.time()

    for epoch in range(trainer.epoch_init, cfg.epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch in trainer.dataloader:
            loss, loss_dict = trainer.train(batch)
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else float("inf")
        rank_zero_logger.info(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4e}")

        wandb.log(
            {
                "total_loss": loss_dict["total_loss"].detach().cpu(),
                "loss_one": loss_dict.get("loss_one", torch.tensor(0.0)).detach().cpu(),
                "loss_stability": loss_dict.get("loss_stability", torch.tensor(0.0))
                .detach()
                .cpu(),
                "physics_loss": loss_dict.get("physics_loss", torch.tensor(0.0))
                .detach()
                .cpu(),
                "epoch": epoch,
            }
        )

        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0:
            save_checkpoint(
                to_absolute_path(cfg.ckpt_path),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            rank_zero_logger.info(f"Checkpoint saved at epoch {epoch}.")

        elapsed = time.time() - start_time
        rank_zero_logger.info(f"Epoch {epoch} duration: {elapsed:.2f} seconds.")
        start_time = time.time()

    rank_zero_logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()
