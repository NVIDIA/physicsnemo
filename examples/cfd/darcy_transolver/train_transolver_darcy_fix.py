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

# Configuration imports:
import hydra
from omegaconf import DictConfig, OmegaConf
import json
import time
from math import ceil

# Base PyTorch imports:
import torchinfo
import torch
import torch.distributed as dist


from torch.optim import lr_scheduler, AdamW
from torch.nn.parallel import DistributedDataParallel as DDP

# PyTorch Data tools
from torch.utils.data import DataLoader, DistributedSampler

from torch.utils.tensorboard import SummaryWriter

from utils.testloss import TestLoss

# Model imports from PhysicsNeMo
from physicsnemo.models.transolver import Transolver
from physicsnemo.distributed import DistributedManager

from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper

from darcy_datapipe_fix import Darcy2D_fix
from validator_fix import GridValidator

from physicsnemo.utils.profiling import Profiler
from contextlib import nullcontext


prof = Profiler()


def forward_train_full_loop(
    model: torch.nn.Module,
    loss_fun: callable,
    optimizer: torch.optim.Optimizer,
    pos: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    y_normalizer,
    precision_context,
    scaler: torch.cuda.amp.GradScaler = None,
) -> torch.Tensor:
    """
    Forward and backward pass for one iteration, with optional mixed precision training.

    Args:
        model (torch.nn.Module): The model to train.
        loss_fun (callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        pos (torch.Tensor): Position tensor (embedding).
        x (torch.Tensor): Input tensor.
        y (torch.Tensor): Target tensor.
        y_normalizer: Normalizer for the target tensor.
        precision_context: Context manager for precision (e.g., autocast).
        scaler (torch.cuda.amp.GradScaler, optional): GradScaler for mixed precision.

    Returns:
        torch.Tensor: The computed loss for this minibatch.
    """
    dm = DistributedManager()
    with precision_context:
        pred = model(embedding=pos, fx=x.unsqueeze(-1)).squeeze(-1)
        pred = y_normalizer.decode(pred)
        loss = loss_fun(pred, y)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_dataloader: DataLoader,
    loss_fun: callable,
    y_normalizer,
    precision_context,
    scaler: torch.cuda.amp.GradScaler,
) -> torch.Tensor:
    """
    One epoch of training. Returns the loss from the last minibatch used, averaged across replicas.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        train_dataloader (DataLoader): Training data loader.
        loss_fun (callable): Loss function.
        y_normalizer: Normalizer for the target tensor.
        precision_context: Context manager for precision (e.g., autocast).
        scaler (torch.cuda.amp.GradScaler): GradScaler for mixed precision.

    Returns:
        torch.Tensor: The averaged loss from the last minibatch.
    """
    for i, batch in enumerate(train_dataloader):
        pos, x, y = batch
        loss = forward_train_full_loop(
            model,
            loss_fun,
            optimizer,
            pos,
            x,
            y,
            y_normalizer,
            precision_context,
            scaler,
        )
        scheduler.step()

    # At the end of the epoch, reduce the last local loss if needed:
    dm = DistributedManager()
    if dm.world_size > 1:
        dist.all_reduce(loss.detach(), op=dist.ReduceOp.SUM)
        loss = loss / dm.world_size

    return loss


def val_epoch(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    loss_fun: callable,
    y_normalizer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One epoch of validation. Returns the loss averaged across the entire validation set.

    Args:
        model (torch.nn.Module): The model to validate.
        test_dataloader (DataLoader): Validation data loader.
        loss_fun (callable): Loss function.
        y_normalizer: Normalizer for the target tensor.

    Returns:
        tuple: (val_loss, pred, y, RL2)
            val_loss (torch.Tensor): Averaged validation loss.
            pred (torch.Tensor): Last batch predictions.
            y (torch.Tensor): Last batch targets.
            RL2 (torch.Tensor): Averaged relative L2 error.
    """
    val_loss = None
    RL2 = None
    for i, batch in enumerate(test_dataloader):
        pos, x, y = batch
        with torch.no_grad():
            pred = model(embedding=pos, fx=x.unsqueeze(-1)).squeeze(-1)
            pred = y_normalizer.decode(pred)
            loss = loss_fun(pred, y)

            # Compute per-sample relative L2 error
            diff = pred.reshape(y.shape) - y
            rel_l2 = torch.norm(diff.view(diff.shape[0], -1), dim=1) / torch.norm(
                y.view(y.shape[0], -1), dim=1
            )
            rel_l2_mean = rel_l2.mean()

            if RL2 is None:
                RL2 = rel_l2_mean
            else:
                RL2 += rel_l2_mean
            if val_loss is None:
                val_loss = loss
            else:
                val_loss += loss

    val_loss = val_loss / len(test_dataloader)
    RL2 = RL2 / len(test_dataloader)

    dm = DistributedManager()
    if dm.world_size > 1:
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(RL2, op=dist.ReduceOp.SUM)
        val_loss = val_loss / dm.world_size
        RL2 = RL2 / dm.world_size
    return val_loss, pred, y, RL2


@hydra.main(version_base="1.3", config_path=".", config_name="config_fix.yaml")
def darcy_trainer(cfg: DictConfig) -> None:
    """
    Training entry point for the 2D Darcy flow benchmark problem.

    Args:
        cfg (DictConfig): Configuration object loaded by Hydra.
    """
    ########################################################################
    # Initialize distributed tools
    ########################################################################
    DistributedManager.initialize()  # Only call this once in the entire script!
    dm = DistributedManager()  # call if required elsewhere

    ########################################################################
    # Initialize monitoring and logging
    ########################################################################
    logger = RankZeroLoggingWrapper(PythonLogger(name="darcy_transolver"), dm)
    logger.file_logging()

    # === TensorBoard SummaryWriter ===
    # Only rank 0 writes logs to avoid duplication in DDP
    writer = None
    if dm.rank == 0:
        log_dir = f"{cfg.output_dir}/runs/{cfg.run_id}"
        writer = SummaryWriter(log_dir=log_dir)

    ########################################################################
    # Print the configuration to log
    ########################################################################
    logger.info(json.dumps(OmegaConf.to_container(cfg), indent=4))

    ########################################################################
    # define model
    ########################################################################
    model = Transolver(
        functional_dim=cfg.model.functional_dim,
        out_dim=cfg.model.out_dim,
        embedding_dim=cfg.model.embedding_dim,
        n_layers=cfg.model.n_layers,
        n_hidden=cfg.model.n_hidden,
        dropout=cfg.model.dropout,
        n_head=cfg.model.n_head,
        act=cfg.model.act,
        mlp_ratio=cfg.model.mlp_ratio,
        slice_num=cfg.model.slice_num,
        unified_pos=cfg.model.unified_pos,
        ref=cfg.model.ref,
        structured_shape=[cfg.data.resolution, cfg.data.resolution],
        use_te=cfg.model.use_te,
        time_input=cfg.model.time_input,
    ).to(dm.device)

    logger.info(f"\n{torchinfo.summary(model, verbose=0)}")

    if dm.world_size > 1:
        model = DDP(model, device_ids=[dm.rank])

    ########################################################################
    # define loss and optimizer
    ########################################################################
    loss_fun = TestLoss(size_average=True)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.scheduler.initial_lr,
        weight_decay=cfg.scheduler.weight_decay,
    )

    ########################################################################
    # Create the data pipes and samplers
    ########################################################################

    train_datapipe = Darcy2D_fix(
        resolution=cfg.data.resolution,
        batch_size=cfg.data.batch_size,
        train_path=cfg.data.train_path,
        is_test=False,
    )
    # Sampler ensures disjoint instances on each rank
    train_sampler = DistributedSampler(
        train_datapipe, num_replicas=dm.world_size, rank=dm.rank, shuffle=True
    )
    # DataLoader handles the batching
    train_dataloader = DataLoader(
        train_datapipe,
        batch_size=cfg.data.batch_size // dm.world_size,
        sampler=train_sampler,
        drop_last=True,
    )
    # Reuse the train normalizer for the test data:
    # (The normalizer puts the inputs and targets to mean 0, std=1.0)
    x_normalizer, y_normalizer = train_datapipe.__get_normalizer__()

    test_datapipe = Darcy2D_fix(
        resolution=cfg.data.resolution,
        batch_size=cfg.data.batch_size,
        train_path=cfg.data.test_path,
        is_test=True,
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
    )
    test_sampler = DistributedSampler(
        test_datapipe, num_replicas=dm.world_size, rank=dm.rank, shuffle=False
    )
    test_dataloader = DataLoader(
        test_datapipe,
        batch_size=cfg.data.batch_size // dm.world_size,
        sampler=test_sampler,
        drop_last=True,
    )

    # calculate steps per pseudo epoch
    steps_per_pseudo_epoch = ceil(
        cfg.training.pseudo_epoch_sample_size / cfg.data.batch_size
    )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.scheduler.initial_lr,
        steps_per_epoch=steps_per_pseudo_epoch,
        epochs=cfg.training.max_pseudo_epochs,
    )

    validator = GridValidator(output_dir=f"{cfg.output_dir}/runs/{cfg.run_id}/plots")

    ckpt_args = {
        "path": f"{cfg.output_dir}/runs/{cfg.run_id}/checkpoints",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    loaded_pseudo_epoch = load_checkpoint(device=dm.device, **ckpt_args)

    validation_iters = ceil(cfg.validation.sample_size / cfg.data.batch_size)

    if cfg.training.pseudo_epoch_sample_size % cfg.data.batch_size != 0:
        logger.warning(
            f"increased pseudo_epoch_sample_size to multiple of \
                      batch size: {steps_per_pseudo_epoch * cfg.data.batch_size}"
        )
    if cfg.validation.sample_size % cfg.data.batch_size != 0:
        logger.warning(
            f"increased validation sample size to multiple of \
                      batch size: {validation_iters * cfg.data.batch_size}"
        )

    # Initialize GradScaler for mixed precision training
    if cfg.precision == "fp16":
        precision_context = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        scaler = torch.amp.GradScaler("cuda")
    elif cfg.precision == "bf16":
        precision_context = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        scaler = None
    else:
        precision_context = nullcontext()
        scaler = None

    if loaded_pseudo_epoch == 0:
        logger.success("Training started...")
    else:
        logger.warning(
            f"Resuming training from pseudo epoch {loaded_pseudo_epoch + 1}."
        )

    # Get the first batch of the test dataset for plotting

    with prof:
        for pseudo_epoch in range(
            max(1, loaded_pseudo_epoch + 1), cfg.training.max_pseudo_epochs + 1
        ):
            # --- TRAINING ---
            train_start = time.time()
            loss = train_epoch(
                model,
                optimizer,
                scheduler,
                train_dataloader,
                loss_fun,
                y_normalizer,
                precision_context,
                scaler,
            )
            train_time = time.time() - train_start

            # After training epoch, e.g. after loss, train_time, optimizer, etc. are available:
            if torch.cuda.is_available():
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
            else:
                gpu_mem_reserved = 0

            lr = optimizer.param_groups[0]["lr"]

            header = "mode\tEpoch\tloss\ttime\tLR\t\tGPU_mem"
            values = f"train\t{pseudo_epoch}\t{loss.item():.4f}\t{train_time:.2f}\t{lr:.4e}\t{gpu_mem_reserved:.2f}"

            log_string = f"\n{header}\n{values}"
            logger.info(log_string)

            # --- TensorBoard logging (only on rank 0) ---
            if dm.rank == 0 and writer is not None:
                # Images/sec/GPU: (num images processed in train_epoch) / train_time / num_gpus
                # Each batch processes batch_size // world_size images, for steps_per_pseudo_epoch steps
                images_per_epoch = len(train_dataloader) * (
                    cfg.data.batch_size // dm.world_size
                )
                images_per_sec_per_gpu = images_per_epoch / train_time

                writer.add_scalar("loss/train", loss.item(), pseudo_epoch)
                writer.add_scalar("time_per_epoch/train", train_time, pseudo_epoch)
                writer.add_scalar(
                    "images_per_sec_per_gpu/train", images_per_sec_per_gpu, pseudo_epoch
                )
                writer.add_scalar("learning_rate/train", lr, pseudo_epoch)

            # save checkpoint
            if pseudo_epoch % cfg.training.rec_results_freq == 0 and dm.rank == 0:
                save_checkpoint(**ckpt_args, epoch=pseudo_epoch)

            # --- VALIDATION ---
            if pseudo_epoch % cfg.validation.validation_pseudo_epochs == 0:
                val_start = time.time()
                val_loss, pred, y, RL2 = val_epoch(
                    model, test_dataloader, loss_fun, y_normalizer
                )
                val_time = time.time() - val_start

                header = "mode\tEpoch\tloss\tRL2\ttime"
                values = f"val\t{pseudo_epoch}\t{val_loss.item():.4f}\t{RL2.item():.4f}\t{val_time:.2f}"

                log_string = f"\n{header}\n{values}"
                logger.info(log_string)

                # --- TensorBoard logging (only on rank 0) ---
                if dm.rank == 0 and writer is not None:
                    # Validation images/sec/GPU
                    val_images = validation_iters * (
                        cfg.data.batch_size // dm.world_size
                    )
                    val_images_per_sec_per_gpu = val_images / val_time
                    writer.add_scalar("loss/val", val_loss.item(), pseudo_epoch)
                    writer.add_scalar("RL2/val", RL2.item(), pseudo_epoch)
                    writer.add_scalar("time_per_epoch/val", val_time, pseudo_epoch)
                    writer.add_scalar(
                        "images_per_sec_per_gpu/val",
                        val_images_per_sec_per_gpu,
                        pseudo_epoch,
                    )

                if dm.rank == 0:
                    validator.make_plot(pred, y, pseudo_epoch, test_datapipe.s)

        # update learning rate
        # if pseudo_epoch % cfg.scheduler.decay_pseudo_epochs == 0:

    if dm.rank == 0 and writer is not None:
        writer.close()
    logger.success("Training completed *yay*")


if __name__ == "__main__":
    # prof.enable("line_profile")
    # prof.enable("torch")
    # prof.initialize()
    darcy_trainer()

    # prof.finalize()
