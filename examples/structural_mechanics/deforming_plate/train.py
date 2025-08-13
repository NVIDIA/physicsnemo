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
from tqdm import tqdm

from omegaconf import DictConfig

from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from deforming_plate_dataset import DeformingPlateDataset
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

from helpers import add_world_edges

import os

os.makedirs(os.path.expanduser("~/.dgl"), exist_ok=True)

from torch.utils.tensorboard import SummaryWriter


class InMemoryTimeStepDataset(torch.utils.data.Dataset):
    """In-memory dataset."""

    def __init__(self, sample_dir):
        self.data = []
        sample_files = sorted(
            [
                os.path.join(sample_dir, f)
                for f in os.listdir(sample_dir)
                if f.startswith("sample_") and f.endswith(".pt")
            ]
        )
        print(f"Found {len(sample_files)} sample files")
        for sample_file in sample_files:
            sample_data = torch.load(
                sample_file, map_location="cpu", weights_only=False
            )
            self.data.extend(sample_data)  # Flatten all time steps into one list
        print(f"Loaded the dataset with {len(self.data)} samples")

    def __getitem__(self, idx):
        return self.data[
            idx
        ]  # dict with graph, mesh_edge_features, world_edge_features

    def __len__(self):
        return len(self.data)


class LazyTimeStepDataset(torch.utils.data.Dataset):
    """Lazy dataset."""

    def __init__(self, sample_dir, num_time_steps):
        self.sample_files = sorted(
            [
                os.path.join(sample_dir, f)
                for f in os.listdir(sample_dir)
                if f.startswith("sample_") and f.endswith(".pt")
            ]
        )
        self.num_steps = num_time_steps - 1
        self.total_samples = len(self.sample_files) * self.num_steps
        print(
            f"Found {len(self.sample_files)} sample files, {self.total_samples} samples in total."
        )

    def __getitem__(self, idx):
        file_idx = idx // self.num_steps
        idx_in_file = idx % self.num_steps
        sample_file = self.sample_files[file_idx]
        sample_data = torch.load(sample_file, map_location="cpu", weights_only=False)
        return sample_data[idx_in_file]

    def __len__(self):
        return self.total_samples


class MGNTrainer:
    def __init__(self, cfg: DictConfig, rank_zero_logger: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()

        self.amp = cfg.amp
        # MGN with recompute_activation currently supports only SiLU activation function.
        mlp_act = "relu"
        if cfg.recompute_activation:
            rank_zero_logger.info(
                "Setting MLP activation to SiLU required by recompute_activation."
            )
            mlp_act = "silu"

        # dataset = InMemoryTimeStepDataset(to_absolute_path(cfg.preprocess_output_dir))
        dataset = LazyTimeStepDataset(
            to_absolute_path(cfg.preprocess_output_dir), cfg.num_training_time_steps
        )
        if self.dist.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.dist.world_size,
                rank=self.dist.rank,
                shuffle=True,
            )
        else:
            sampler = None

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=(sampler is None),  # Only shuffle if not using sampler
            drop_last=True,
            pin_memory=True,
            num_workers=cfg.num_dataloader_workers,
            sampler=sampler,
            collate_fn=lambda batch: batch[0],
        )
        self.sampler = sampler

        # instantiate the model
        self.model = HybridMeshGraphNet(
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
                raise ValueError("MeshGraphNet is not yet JIT-compatible.")
            self.model = torch.jit.script(self.model).to(self.dist.device)
        else:
            self.model = self.model.to(self.dist.device)

        # distributed data parallel for multi-node training
        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        # enable train mode
        self.model.train()

        # instantiate loss, optimizer, and scheduler
        self.criterion = torch.nn.MSELoss()

        self.optimizer = None
        try:
            if cfg.use_apex:
                from apex.optimizers import FusedAdam

                self.optimizer = FusedAdam(self.model.parameters(), lr=cfg.lr)
        except ImportError:
            rank_zero_logger.warning(
                "NVIDIA Apex (https://github.com/nvidia/apex) is not installed, "
                "FusedAdam optimizer will not be used."
            )
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        rank_zero_logger.info(f"Using {self.optimizer.__class__.__name__} optimizer")

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: cfg.lr_decay_rate**epoch
        )
        self.scaler = GradScaler()

        # load checkpoint
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

        if self.dist.rank == 0:
            self.writer = SummaryWriter(
                log_dir=to_absolute_path(cfg.tensorboard_log_dir)
            )

    def train(self, graph, mesh_edge_features, world_edge_features, epoch):
        mesh_edge_features = mesh_edge_features.to(self.dist.device)
        world_edge_features = world_edge_features.to(self.dist.device)
        self.optimizer.zero_grad()
        loss = self.forward(graph, mesh_edge_features, world_edge_features)
        self.backward(loss)
        self.scheduler.step()
        return loss

    def forward(self, graph, mesh_edge_features, world_edge_features):
        # forward pass
        with autocast(enabled=self.amp):
            pred = self.model(
                graph.ndata["x"], mesh_edge_features, world_edge_features, graph
            )
            loss = self.criterion(pred, graph.ndata["y"])
            return loss

    def backward(self, loss):
        # backward pass
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    rank_zero_logger.file_logging()

    trainer = MGNTrainer(cfg, rank_zero_logger)
    start = time.time()
    rank_zero_logger.info("Training started...")
    for epoch in range(trainer.epoch_init, cfg.epochs):
        if trainer.sampler is not None:
            trainer.sampler.set_epoch(epoch)
        start = time.time()
        # Wrap the dataloader with tqdm and add description with epoch info
        progress_bar = tqdm(
            trainer.dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", leave=False
        )

        for item in progress_bar:
            graph = item["graph"].to(dist.device)
            mesh_edge_features = item["mesh_edge_features"].to(dist.device)
            world_edge_features = item["world_edge_features"].to(dist.device)
            loss = trainer.train(graph, mesh_edge_features, world_edge_features, epoch)

            # Update tqdm postfix with current loss (converted to scalar)
            progress_bar.set_postfix(loss=f"{loss.item():.3e}")
            del graph, mesh_edge_features, world_edge_features
            torch.cuda.empty_cache()

        rank_zero_logger.info(
            f"epoch: {epoch + 1}, loss: {loss:10.3e}, time per epoch: {(time.time() - start):10.3e}"
        )
        if dist.rank == 0:
            trainer.writer.add_scalar("loss", loss.detach().cpu().item(), epoch)
            current_lr = trainer.optimizer.param_groups[0]["lr"]
            trainer.writer.add_scalar("learning_rate", current_lr, epoch)

        # save checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0:
            save_checkpoint(
                to_absolute_path(cfg.ckpt_path),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch + 1,
            )
            logger.info(f"Saved model on rank {dist.rank}")
        torch.cuda.empty_cache()
        start = time.time()
    rank_zero_logger.info("Training completed!")
    if dist.rank == 0:
        trainer.writer.close()


if __name__ == "__main__":
    main()
