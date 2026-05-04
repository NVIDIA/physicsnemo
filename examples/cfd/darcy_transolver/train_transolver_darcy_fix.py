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

# Configuration imports:
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import json
import os
import time
from datetime import datetime, timezone
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
from physicsnemo.distributed import DistributedManager
from physicsnemo.optim import CombinedOptimizer

from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

from darcy_datapipe_fix import Darcy2D_fix
from validator_fix import GridValidator

from physicsnemo.utils.profiling import Profiler
from contextlib import nullcontext


prof = Profiler()


_GEOTRANSOLVER_TARGETS = {
    "physicsnemo.experimental.models.geotransolver.GeoTransolver",
}


def make_model_forward(cfg: DictConfig) -> callable:
    """
    Return a forward callable that uses the right keyword arguments for the
    configured model.

    GeoTransolver uses (local_embedding, geometry) while Transolver/FLARE
    use (fx, embedding).  The decision is made once at startup from the Hydra
    config, avoiding fragile isinstance checks through DDP/compile wrappers.

    Args:
        cfg (DictConfig): Full Hydra config (reads model._target_).

    Returns:
        callable: ``fn(model, pos, x) -> Tensor``
    """
    if cfg.model._target_ in _GEOTRANSOLVER_TARGETS:

        def _forward(model, pos, x):
            combined_inputs = torch.cat([pos, x.unsqueeze(-1)], dim=-1)
            return model(
                local_embedding=combined_inputs, geometry=combined_inputs
            ).squeeze(-1)

    else:

        def _forward(model, pos, x):
            return model(embedding=pos, fx=x.unsqueeze(-1)).squeeze(-1)

    return _forward


def build_optimizer(
    model: torch.nn.Module,
    cfg: DictConfig,
) -> torch.optim.Optimizer:
    """
    Build optimizer based on config.  Supports AdamW and Muon.

    Muon is applied to 2D weight matrices; remaining parameters (biases, norms,
    embeddings) are handled by AdamW.  When both groups exist they are wrapped in
    ``CombinedOptimizer``.

    Args:
        model (torch.nn.Module): The model (possibly DDP-wrapped).
        cfg (DictConfig): Full Hydra config (reads optimizer.type, scheduler.initial_lr,
            scheduler.weight_decay).

    Returns:
        torch.optim.Optimizer: The configured optimizer.
    """
    opt_type = cfg.optimizer.type
    lr = cfg.scheduler.initial_lr
    weight_decay = cfg.scheduler.weight_decay

    if opt_type == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if opt_type == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise ImportError(
                "Muon optimizer requires PyTorch >= 2.9. "
                "Install a newer PyTorch or use optimizer.type=adamw."
            )
        base_model = model.module if hasattr(model, "module") else model
        muon_params = [p for p in base_model.parameters() if p.ndim == 2]
        other_params = [p for p in base_model.parameters() if p.ndim != 2]

        if muon_params and other_params:
            return CombinedOptimizer(
                [
                    torch.optim.Muon(
                        muon_params,
                        lr=lr,
                        weight_decay=weight_decay,
                        adjust_lr_fn="match_rms_adamw",
                    ),
                    AdamW(
                        other_params,
                        lr=lr,
                        weight_decay=weight_decay,
                        betas=(0.9, 0.999),
                        eps=1.0e-8,
                    ),
                ]
            )
        elif muon_params:
            return torch.optim.Muon(
                muon_params,
                lr=lr,
                weight_decay=weight_decay,
                adjust_lr_fn="match_rms_adamw",
            )
        else:
            return AdamW(other_params, lr=lr, weight_decay=weight_decay)

    raise ValueError(
        f"Unsupported optimizer type: {opt_type!r}. Use 'adamw' or 'muon'."
    )


def forward_train_full_loop(
    model: torch.nn.Module,
    model_forward: callable,
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
        model_forward (callable): Forward callable from ``make_model_forward``.
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
        pred = model_forward(model, pos, x)
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
    model_forward: callable,
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
        model_forward (callable): Forward callable from ``make_model_forward``.
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
            model_forward,
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
    model_forward: callable,
    test_dataloader: DataLoader,
    loss_fun: callable,
    y_normalizer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One epoch of validation. Returns the loss averaged across the entire validation set.

    Args:
        model (torch.nn.Module): The model to validate.
        model_forward (callable): Forward callable from ``make_model_forward``.
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
            pred = model_forward(model, pos, x)
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

    # === TensorBoard SummaryWriters ===
    # Separate train/val writers so TensorBoard can overlay matching scalars
    train_writer = None
    val_writer = None
    metrics_file = None
    if dm.rank == 0:
        log_dir = f"{cfg.output_dir}/runs/{cfg.run_id}"
        train_writer = SummaryWriter(log_dir=f"{log_dir}/train")
        val_writer = SummaryWriter(log_dir=f"{log_dir}/val")

        # === JSONL metrics log (append-safe for resumed runs) ===
        metrics_path = os.path.join(log_dir, "metrics.jsonl")
        os.makedirs(log_dir, exist_ok=True)
        metrics_file = open(metrics_path, "a")

    ########################################################################
    # Print the configuration to log
    ########################################################################
    logger.info(json.dumps(OmegaConf.to_container(cfg), indent=4))

    ########################################################################
    # define model
    ########################################################################
    model = instantiate(cfg.model).to(dm.device)
    model_forward = make_model_forward(cfg)

    logger.info(f"\n{torchinfo.summary(model, verbose=0)}")

    if dm.world_size > 1:
        model = DDP(model, device_ids=[dm.rank])

    ########################################################################
    # define loss and optimizer
    ########################################################################
    loss_fun = TestLoss(size_average=True)
    optimizer = build_optimizer(model, cfg)

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

    total_steps = steps_per_pseudo_epoch * cfg.training.max_pseudo_epochs
    if cfg.optimizer.type == "muon":
        warmup_steps = steps_per_pseudo_epoch * 2
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-2, total_iters=warmup_steps
                ),
                lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_steps - warmup_steps,
                    eta_min=cfg.scheduler.initial_lr * 0.1,
                ),
            ],
            milestones=[warmup_steps],
        )
    else:
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

    # Compile after checkpoint loading to avoid triggering recompilation
    model = torch.compile(model)

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
                model_forward,
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
            if dm.rank == 0 and train_writer is not None:
                # Images/sec/GPU: (num images processed in train_epoch) / train_time / num_gpus
                # Each batch processes batch_size // world_size images, for steps_per_pseudo_epoch steps
                images_per_epoch = len(train_dataloader) * (
                    cfg.data.batch_size // dm.world_size
                )
                images_per_sec_per_gpu = images_per_epoch / train_time

                train_writer.add_scalar("loss", loss.item(), pseudo_epoch)
                train_writer.add_scalar("time_per_epoch", train_time, pseudo_epoch)
                train_writer.add_scalar(
                    "images_per_sec_per_gpu", images_per_sec_per_gpu, pseudo_epoch
                )
                train_writer.add_scalar("learning_rate", lr, pseudo_epoch)

            # --- JSONL metrics record (training fields) ---
            metrics_record = None
            if dm.rank == 0 and metrics_file is not None:
                images_per_epoch = len(train_dataloader) * (
                    cfg.data.batch_size // dm.world_size
                )
                metrics_record = {
                    "pseudo_epoch": pseudo_epoch,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "train_loss": loss.item(),
                    "train_time_s": train_time,
                    "learning_rate": lr,
                    "images_per_sec_per_gpu": images_per_epoch / train_time,
                    "gpu_mem_reserved_gb": gpu_mem_reserved,
                }

            # save checkpoint
            if pseudo_epoch % cfg.training.rec_results_freq == 0 and dm.rank == 0:
                save_checkpoint(**ckpt_args, epoch=pseudo_epoch)

            # --- VALIDATION ---
            if pseudo_epoch % cfg.validation.validation_pseudo_epochs == 0:
                val_start = time.time()
                val_loss, pred, y, RL2 = val_epoch(
                    model, model_forward, test_dataloader, loss_fun, y_normalizer
                )
                val_time = time.time() - val_start

                header = "mode\tEpoch\tloss\tRL2\ttime"
                values = f"val\t{pseudo_epoch}\t{val_loss.item():.4f}\t{RL2.item():.4f}\t{val_time:.2f}"

                log_string = f"\n{header}\n{values}"
                logger.info(log_string)

                # --- TensorBoard logging (only on rank 0) ---
                if dm.rank == 0 and val_writer is not None:
                    # Validation images/sec/GPU
                    val_images = validation_iters * (
                        cfg.data.batch_size // dm.world_size
                    )
                    val_images_per_sec_per_gpu = val_images / val_time
                    val_writer.add_scalar("loss", val_loss.item(), pseudo_epoch)
                    val_writer.add_scalar("RL2", RL2.item(), pseudo_epoch)
                    val_writer.add_scalar("time_per_epoch", val_time, pseudo_epoch)
                    val_writer.add_scalar(
                        "images_per_sec_per_gpu",
                        val_images_per_sec_per_gpu,
                        pseudo_epoch,
                    )

                # --- JSONL metrics record (validation fields) ---
                if metrics_record is not None:
                    val_images = validation_iters * (
                        cfg.data.batch_size // dm.world_size
                    )
                    metrics_record["val_loss"] = val_loss.item()
                    metrics_record["val_rl2"] = RL2.item()
                    metrics_record["val_time_s"] = val_time
                    metrics_record["val_images_per_sec_per_gpu"] = val_images / val_time

                if dm.rank == 0 and cfg.validation.save_plots:
                    validator.make_plot(pred, y, pseudo_epoch, test_datapipe.s)

            # --- Flush JSONL record for this pseudo-epoch ---
            if metrics_record is not None:
                metrics_file.write(json.dumps(metrics_record) + "\n")
                metrics_file.flush()

        # update learning rate
        # if pseudo_epoch % cfg.scheduler.decay_pseudo_epochs == 0:

    if dm.rank == 0:
        if train_writer is not None:
            train_writer.close()
        if val_writer is not None:
            val_writer.close()
        if metrics_file is not None:
            metrics_file.close()
    logger.success("Training completed *yay*")


if __name__ == "__main__":
    # prof.enable("line_profile")
    # prof.enable("torch")
    # prof.initialize()
    darcy_trainer()

    # prof.finalize()
