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

# Core python imports:
import os
import time
from pathlib import Path
from typing import Literal, Any, Callable, Sequence
import collections
from contextlib import nullcontext

from collections.abc import Sequence

# Configuration:
import hydra
import omegaconf
from omegaconf import DictConfig

# Pytorch imports:
import torch
from torch.optim import Optimizer
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist

# For metrics and model printouts:
from tabulate import tabulate
import torchinfo

# For loading dataset stats:
import numpy as np

# Physicsnemo imports ...
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.profiling import profile, Profiler
from physicsnemo.datapipes.cae.transolver_datapipe import (
    create_transolver_dataset,
    TransolverDataPipe,
)

# Local folder imports for this example
from metrics import metrics_fn

# tensorwise is to handle single-point-cloud or multi-point-cloud running.
# it's a decorator that will automatically unzip one or more of a list of tensors,
# run the funtcion, and rezip the results.
from utils import tensorwise

# Special import, if transformer engine is available:
from physicsnemo.core.version_check import check_version_spec

TE_AVAILABLE = check_version_spec("transformer_engine", hard_fail=False)

if TE_AVAILABLE:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
else:
    te, Format, DelayedScaling = None, None, None

# This will go away when checkpointing is refined further below:
torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
torch.serialization.add_safe_globals([Any])
torch.serialization.add_safe_globals([list])
torch.serialization.add_safe_globals([collections.defaultdict])
torch.serialization.add_safe_globals([dict])
torch.serialization.add_safe_globals([int])
torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
torch.serialization.add_safe_globals([omegaconf.base.Metadata])


class CombinedOptimizer(Optimizer):
    """Combine multiple PyTorch optimizers into a single Optimizer-like interface.

    The wrapper concatenates the *param_groups* from all contained optimizers so
    that learning-rate schedulers (e.g., ReduceLROnPlateau, CosineAnnealingLR)
    operate transparently across every parameter. Only a minimal subset of the
    *torch.optim.Optimizer* API is implemented—extend as needed.

    Note:
        This will get upstreamed to physicsnemo shortly.  Don't count on this
        class existing here in the future!

        In other words, this is already marked for deprecation!
    """

    def __init__(
        self,
        optimizers: Sequence[Optimizer],
        torch_compile_kwargs: dict[str, Any] | None = None,
    ):
        if not optimizers:
            raise ValueError("`optimizers` must contain at least one optimizer.")

        self.optimizers = optimizers

        # Collect parameter groups from all optimizers. We pass an empty
        # *defaults* dict because hyper-parameters are managed by the inner
        # optimizers, not this wrapper.
        param_groups = [g for opt in optimizers for g in opt.param_groups]
        super().__init__(param_groups, defaults={})

        if torch_compile_kwargs is None:
            self.step_fns: list[Callable] = [opt.step for opt in optimizers]
        else:
            self.step_fns: list[Callable] = [
                torch.compile(opt.step, **torch_compile_kwargs) for opt in optimizers
            ]

    def zero_grad(self, *args, **kwargs) -> None:
        """Nullify gradients"""
        for opt in self.optimizers:
            opt.zero_grad(*args, **kwargs)

    def step(self, closure=None) -> None:
        for step_fn in self.step_fns:
            if closure is None:
                step_fn()
            else:
                step_fn(closure)

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)

        self.param_groups = [g for opt in self.optimizers for g in opt.param_groups]


def get_autocast_context(precision: str) -> nullcontext:
    """
    Returns the appropriate autocast context for mixed precision training.

    Args:
        precision (str): The desired precision. Supported values are "float16", "bfloat16", or any other string for no autocast.

    Returns:
        Context manager: An autocast context for the specified precision, or a nullcontext if precision is not recognized.
    """
    if precision == "float16":
        return autocast("cuda", dtype=torch.float16)
    elif precision == "bfloat16":
        return autocast("cuda", dtype=torch.bfloat16)
    elif precision == "float8" and TE_AVAILABLE:
        fp8_format = Format.HYBRID
        fp8_recipe = DelayedScaling(
            fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max"
        )
        return te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
    else:
        return nullcontext()


@tensorwise
def cast_precisions(tensor: torch.Tensor, precision: str) -> torch.Tensor:
    """
    Casts the tensors to the specified precision.

    We are careful to take either a tensor or list of tensors, and return the same format.
    """

    match precision:
        case "float16":
            dtype = torch.float16
        case "bfloat16":
            dtype = torch.bfloat16
        case _:
            dtype = None

    if dtype is not None:
        return tensor.to(dtype)
    else:
        return tensor


@tensorwise
def pad_input_for_fp8(
    features: torch.Tensor,
    embeddings: torch.Tensor,
    geometry: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Pads the input features tensor so that the concatenated feature and embedding dimension is a multiple of 16,
    which is required for FP8 operations.  Only the features is updated.

    Args:
        features (torch.Tensor): The input features tensor of shape (..., feature_dim).
        embeddings (torch.Tensor): The embeddings tensor of shape (..., embedding_dim).

    Returns:
        torch.Tensor: The padded features tensor, so that (features.shape[-1] + embeddings.shape[-1]) is a multiple of 16.
    """
    fx_dim = features.shape[-1] + embeddings.shape[-1]
    if fx_dim % 16 != 0:
        pad_size = 16 - (fx_dim % 16)
        features = torch.nn.functional.pad(features, (0, pad_size))
        fx_dim = features.shape[-1] + embeddings.shape[-1]

    if geometry is not None:
        geometry_dim = geometry.shape[-1] if geometry is not None else 0
        if geometry_dim % 16 != 0:
            pad_size = 16 - (geometry_dim % 16)
            geometry = torch.nn.functional.pad(geometry, (0, pad_size))
            geometry_dim = geometry.shape[-1]

    return features, geometry


@tensorwise
def unpad_output_for_fp8(
    outputs: torch.Tensor, output_pad_size: int | None
) -> torch.Tensor:
    """
    Removes the padding from the output tensor that was added for FP8 compatibility.

    Args:
        outputs (torch.Tensor): The output tensor of shape (..., output_dim + pad_size) if padded.
        output_pad_size (int | None): The number of padded elements to remove from the last dimension. If None, no unpadding is performed.

    Returns:
        torch.Tensor: The unpadded output tensor.
    """
    # Remove the padded outputs:
    if output_pad_size is not None:
        return outputs[:, :, :-output_pad_size]
    return outputs


@tensorwise
def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the loss for the model.
    """
    return torch.nn.functional.mse_loss(outputs, targets)


def forward_pass(
    batch: dict,
    model: torch.nn.Module,
    precision: str,
    output_pad_size: int | None,
    dist_manager: DistributedManager,
    data_mode: Literal["surface", "volume"],
    datapipe: TransolverDataPipe,
):
    """
    Run the forward pass of the model for one batch, including metrics and loss calculation.

    Transolver takes just one tensor for features, embeddings.
    Typhon takes a  list of tensors, for each.

    Typhon needs a `geometry` tensor, so that's the switch we use to distinguish.

    """

    features = batch["fx"]
    embeddings = batch["embeddings"]
    targets = batch["fields"]

    # Cast precisions:
    features = cast_precisions(features, precision=precision)
    embeddings = cast_precisions(embeddings, precision=precision)
    if "geometry" in batch.keys():
        geometry = cast_precisions(batch["geometry"], precision=precision)
    else:
        geometry = None

    all_metrics = {}
    if datapipe.config.model_type == "combined":
        # This is hard coded for Typhon.  If you have more point clouds,
        # your mileage may vary.
        modes = ["surface", "volume"]
    elif datapipe.config.model_type == "surface":
        modes = [
            "surface",
        ]
    elif datapipe.config.model_type == "volume":
        modes = [
            "volume",
        ]

    with get_autocast_context(precision):
        # For fp8, we may have to pad the inputs:
        if precision == "float8" and TE_AVAILABLE:
            features, geometry = pad_input_for_fp8(features, embeddings, geometry)

        if "geometry" in batch.keys():
            local_positions = embeddings[:, :, :3]
            # This is the Typhon path
            outputs = model(
                global_embedding=features,
                local_embedding=embeddings,
                geometry=geometry,
                local_positions=local_positions,
            )

            outputs = unpad_output_for_fp8(outputs, output_pad_size)
            # Loss per point cloud:
            loss = loss_fn(outputs, targets)
            # Log them too:
            for i, mode in enumerate(modes):
                all_metrics[f"loss/{mode}"] = loss.item()
            # Averaging over point cloud inputs, instead of summing.
            full_loss = torch.mean(loss)

        else:
            # This is the Transolver path
            outputs = model(fx=features, embedding=embeddings)
            outputs = unpad_output_for_fp8(outputs, output_pad_size)
            full_loss = torch.nn.functional.mse_loss(outputs, targets)

            all_metrics[f"loss/{modes[0]}"] = full_loss

    air_density = batch["air_density"] if "air_density" in batch.keys() else None
    stream_velocity = (
        batch["stream_velocity"] if "stream_velocity" in batch.keys() else None
    )

    unscaled_outputs = tensorwise(datapipe.unscale_model_targets)(
        outputs,
        air_density=air_density,
        stream_velocity=stream_velocity,
        factor_type=modes,
    )
    unscaled_targets = tensorwise(datapipe.unscale_model_targets)(
        targets,
        air_density=air_density,
        stream_velocity=stream_velocity,
        factor_type=modes,
    )
    metrics = metrics_fn(unscaled_outputs, unscaled_targets, dist_manager, modes)

    # In the combined mode, this is a list of dicts.  Merge them.
    metrics = (
        {k: v for d in metrics for k, v in d.items()}
        if isinstance(metrics, list)
        else metrics
    )
    all_metrics.update(metrics)

    return full_loss, all_metrics, (unscaled_outputs, unscaled_targets)


@profile
def train_epoch(
    dataloader,
    epoch_len: int,
    model: torch.nn.Module,
    output_pad_size: int | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    logger: PythonLogger,
    writer: SummaryWriter,
    epoch: int,
    cfg: DictConfig,
    dist_manager: DistributedManager,
    scaler: GradScaler | None = None,
) -> float:
    """
    Train the model for one epoch.

    Args:
        dataloader: Training data loader
        model (torch.nn.Module): The neural network model to train.
        epoch_len (int): Length of the epoch.
        output_pad_size (int | None): Optional output padding size for lowest precisions (FP8).
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        logger (PythonLogger): Logger for training progress.
        writer (SummaryWriter): TensorBoard writer for logging metrics.
        epoch (int): Current epoch number.
        cfg (DictConfig): Hydra configuration object.
        dist_manager (DistributedManager): Distributed manager from physicsnemo.
        scaler (GradScaler | None, optional): Gradient scaler for mixed precision training.
    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    total_metrics = {}

    precision = getattr(cfg, "precision", "float32")
    start_time = time.time()

    for i, batch in enumerate(dataloader):
        # TransolverX has a different forward pass:

        loss, metrics, _ = forward_pass(
            batch,
            model,
            precision,
            output_pad_size,
            dist_manager,
            cfg.data.mode,
            dataloader,
        )

        optimizer.zero_grad()
        if precision == "float16" and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if not isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
            scheduler.step()

        end_time = time.time()

        # Logging
        this_loss = loss.detach().item()
        total_loss += this_loss

        if i == 0:
            total_metrics = metrics
        else:
            total_metrics = {k: total_metrics[k] + metrics[k] for k in metrics.keys()}

        duration = end_time - start_time
        start_time = end_time
        images_per_second = 1 / duration

        mem_usage = torch.cuda.memory_reserved() / 1024**3

        logger.info(
            f"Epoch {epoch} [{i}/{epoch_len}] Loss: {this_loss:.6f} Duration: {duration:.2f}s Mem: {mem_usage:.2f}GB"
        )
        if dist_manager.rank == 0:
            writer.add_scalar(
                "batch/learning_rate",
                optimizer.param_groups[0]["lr"],
                i + epoch_len * epoch,
            )
            writer.add_scalar("batch/loss", this_loss, i + epoch_len * epoch)
            writer.add_scalar(
                "batch/throughpu_per_gpu", images_per_second, i + epoch_len * epoch
            )
            for metric_name, metric_value in metrics.items():
                writer.add_scalar(
                    f"batch/{metric_name}", metric_value, i + epoch_len * epoch
                )

        if cfg.profile and i >= 10:
            break  # Stop profiling after 10 batches

    avg_loss = total_loss / epoch_len
    avg_metrics = {k: v / epoch_len for k, v in total_metrics.items()}
    if dist_manager.rank == 0:
        writer.add_scalar("epoch/loss", avg_loss, epoch)
        for metric_name, metric_value in avg_metrics.items():
            writer.add_scalar(f"epoch/{metric_name}", metric_value, epoch)
        # Print average metrics using tabulate
        metrics_table = tabulate(
            [[k, v] for k, v in avg_metrics.items()],
            headers=["Metric", "Average Value"],
            tablefmt="pretty",
        )
        print(f"\nEpoch {epoch} Average Metrics:\n{metrics_table}\n")
    return avg_loss


@profile
def val_epoch(
    dataloader,
    epoch_len: int,
    model: torch.nn.Module,
    output_pad_size: int | None,
    logger: PythonLogger,
    val_writer: SummaryWriter,
    epoch: int,
    cfg: DictConfig,
    dist_manager: DistributedManager,
) -> float:
    """
    Run validation for one epoch.

    Args:
        dataloader: Validation data loader.
        epoch_len (int): Length of the epoch.
        model (torch.nn.Module): The model to evaluate.
        output_pad_size (int | None): Optional output padding size for lowest precisions (FP8).
        logger (PythonLogger): Logger for validation progress.
        val_writer (SummaryWriter): TensorBoard writer for logging validation metrics.
        epoch (int): Current epoch number.
        cfg (DictConfig): Hydra configuration object.
        dist_manager (DistributedManager): Distributed manager instance.
    Returns:
        float: The average validation loss for the epoch.
    """

    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_metrics = {}

    precision = getattr(cfg.training, "precision", "float32")

    start_time = time.time()
    with torch.no_grad():  # Disable gradient computation
        for i, batch in enumerate(dataloader):
            loss, metrics, _ = forward_pass(
                batch,
                model,
                precision,
                output_pad_size,
                dist_manager,
                cfg.data.mode,
                dataloader,
            )

            if i == 0:
                total_metrics = metrics
            else:
                total_metrics = {
                    k: total_metrics[k] + metrics[k] for k in metrics.keys()
                }

            # Logging
            this_loss = loss.detach().item()
            total_loss += this_loss

            end_time = time.time()
            duration = end_time - start_time
            start_time = end_time

            logger.info(
                f"Val [{i}/{epoch_len}] Loss: {this_loss:.6f} Duration: {duration:.2f}s"
            )
            # We don't add individual loss measurements to tensorboard in the validation loop.

            if cfg.profile and i >= 10:
                break  # Stop profiling after 10 batches

    avg_loss = total_loss / epoch_len
    avg_metrics = {k: v / epoch_len for k, v in total_metrics.items()}
    if dist_manager.rank == 0:
        val_writer.add_scalar("epoch/loss", avg_loss, epoch)
        for metric_name, metric_value in avg_metrics.items():
            val_writer.add_scalar(f"epoch/{metric_name}", metric_value, epoch)
        # Print average metrics using tabulate
        metrics_table = tabulate(
            [[k, v] for k, v in avg_metrics.items()],
            headers=["Metric", "Average Value"],
            tablefmt="pretty",
        )
        print(f"\nEpoch {epoch} Validation Average Metrics:\n{metrics_table}\n")
    return avg_loss


def update_model_params_for_fp8(cfg, logger) -> tuple | None:
    """
    Adjusts model configuration parameters to ensure compatibility with FP8 computations.

    The output shape will be padded to a multiple of 16.  The input shape
    is padded dynamically in the forward pass, but that is printed here
    for information.

    Args:
        cfg: Configuration object with model and training attributes.
        logger: Logger object for info messages.

    Returns:
        tuple: (cfg, output_pad_size) if precision is "float8", where output_pad_size is the amount
               of padding added to the output dimension (or None if no padding was needed).
    """
    # we have to manipulate the output shape
    # to enable fp8 computations with transformer_engine.
    # need the input and output to be divisible by 16.
    # if (cfg.model.embedding_dim + cfg.model.functional_dim) % 16 != 0:

    output_pad_size = None
    if cfg.precision == "float8":
        if cfg.model.out_dim % 16 != 0:
            # pad the output:
            output_pad_size = 16 - (cfg.model.out_dim % 16)
            cfg.model.out_dim += output_pad_size
            logger.info(
                f"Padding output dimension to {cfg.model.out_dim} for fp8 autocast"
            )

        # This part is informational only:
        if (cfg.model.functional_dim + cfg.model.embedding_dim) % 16 != 0:
            input_pad_size = 16 - (
                (cfg.model.functional_dim + cfg.model.embedding_dim) % 16
            )
            cfg.model.functional_dim += input_pad_size
            logger.info(
                f"Padding input dimension to {cfg.model.functional_dim} and {cfg.model.embedding_dim} for fp8 autocast"
            )

    return cfg, output_pad_size


@profile
def main(cfg: DictConfig):
    """Main training function

    Args:
        cfg: Hydra configuration object
    """

    DistributedManager.initialize()

    # Set up distributed training
    dist_manager = DistributedManager()

    # Set up logging
    logger = RankZeroLoggingWrapper(PythonLogger(name="training"), dist_manager)

    # Set checkpoint directory - defaults to output_dir if not specified
    checkpoint_dir = getattr(cfg, "checkpoint_dir", None)
    if checkpoint_dir is None:
        checkpoint_dir = cfg.output_dir

    if dist_manager.rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        writer = SummaryWriter(
            log_dir=os.path.join(
                cfg.output_dir + "/" + cfg.run_id + "/train",
            )
        )
        val_writer = SummaryWriter(
            log_dir=os.path.join(
                cfg.output_dir + "/" + cfg.run_id + "/val",
            )
        )
    else:
        writer = None
        val_writer = None

    logger.info(f"Config:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")
    logger.info(f"Output directory: {cfg.output_dir}/{cfg.run_id}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}/{cfg.run_id}/checkpoints")

    cfg, output_pad_size = update_model_params_for_fp8(cfg, logger)

    # Set up model
    # (Using partial convert to get lists, etc., instead of ListConfigs.)
    model = hydra.utils.instantiate(cfg.model, _convert_="partial")
    logger.info(f"\n{torchinfo.summary(model, verbose=0)}")

    model.to(dist_manager.device)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist_manager.local_rank],
        output_device=dist_manager.device,
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {num_params}")

    # Load the normalization file from configured directory (defaults to current dir)
    norm_dir = getattr(cfg.data, "normalization_dir", ".")
    if cfg.data.mode == "surface" or cfg.data.mode == "combined":
        norm_file = str(Path(norm_dir) / "surface_fields_normalization.npz")
        norm_data = np.load(norm_file)
        surface_factors = {
            "mean": torch.from_numpy(norm_data["mean"]).to(dist_manager.device),
            "std": torch.from_numpy(norm_data["std"]).to(dist_manager.device),
        }
    else:
        surface_factors = None

    if cfg.data.mode == "volume" or cfg.data.mode == "combined":
        norm_file = str(Path(norm_dir) / "volume_fields_normalization.npz")
        norm_data = np.load(norm_file)
        volume_factors = {
            "mean": torch.from_numpy(norm_data["mean"]).to(dist_manager.device),
            "std": torch.from_numpy(norm_data["std"]).to(dist_manager.device),
        }
    else:
        volume_factors = None

    # Training dataset
    train_dataloader = create_transolver_dataset(
        cfg.data,
        phase="train",
        surface_factors=surface_factors,
        volume_factors=volume_factors,
    )

    # Validation dataset

    val_dataloader = create_transolver_dataset(
        cfg.data,
        phase="val",
        surface_factors=surface_factors,
        volume_factors=volume_factors,
    )

    num_replicas = dist_manager.world_size
    data_rank = dist_manager.rank

    # Set up distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataloader,
        num_replicas=num_replicas,
        rank=data_rank,
        shuffle=True,
        drop_last=True,
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataloader,
        num_replicas=num_replicas,
        rank=data_rank,
        shuffle=False,  # No shuffling for validation
        drop_last=True,
    )

    muon_params = [p for p in model.parameters() if p.ndim == 2]
    other_params = [p for p in model.parameters() if p.ndim != 2]

    # Set up optimizer and scheduler
    optimizer = hydra.utils.instantiate(cfg.training.optimizer, params=other_params)

    optimizer = CombinedOptimizer(
        optimizers=[
            torch.optim.Muon(
                muon_params,
                lr=cfg.training.optimizer.lr,
                weight_decay=cfg.training.optimizer.weight_decay,
                adjust_lr_fn="match_rms_adamw",
            ),
            optimizer,
        ],
    )

    # Set up learning rate scheduler based on config
    scheduler_cfg = cfg.training.scheduler
    scheduler_name = scheduler_cfg.name
    scheduler_params = dict(scheduler_cfg.params)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)

    precision = cfg.precision
    scaler = GradScaler() if precision == "float16" else None

    if precision == "float8" and not TE_AVAILABLE:
        raise ImportError(
            "TransformerEngine is not installed.  Please install it to use float8 precision."
        )

    ckpt_args = {
        "path": f"{checkpoint_dir}/{cfg.run_id}/checkpoints",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }

    loaded_epoch = load_checkpoint(device=dist_manager.device, **ckpt_args)

    if cfg.compile:
        model = torch.compile(model)

    # Training loop
    logger.info("Starting training...")
    for epoch in range(loaded_epoch, cfg.training.num_epochs):
        # Set the epoch in the samplers
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        train_dataloader.dataset.set_indices(list(train_sampler))
        val_dataloader.dataset.set_indices(list(val_sampler))

        start_time = time.time()
        # Training phase
        with Profiler():
            train_loss = train_epoch(
                train_dataloader,
                len(list(train_sampler)),
                model,
                output_pad_size,
                optimizer,
                scheduler,
                logger,
                writer,
                epoch,
                cfg,
                dist_manager,
                scaler,
            )
            end_time = time.time()
            train_duration = end_time - start_time

            start_time = time.time()
            # Validation phase
            val_loss = val_epoch(
                val_dataloader,
                len(list(val_sampler)),
                model,
                output_pad_size,
                logger,
                val_writer,
                epoch,
                cfg,
                dist_manager,
            )
            end_time = time.time()
            val_duration = end_time - start_time

        # Log epoch results
        logger.info(
            f"Epoch [{epoch}/{cfg.training.num_epochs}] Train Loss: {train_loss:.6f} [duration: {train_duration:.2f}s] Val Loss: {val_loss:.6f} [duration: {val_duration:.2f}s]"
        )

        # save checkpoint
        if epoch % cfg.training.save_interval == 0 and dist_manager.rank == 0:
            save_checkpoint(**ckpt_args, epoch=epoch + 1)

        if scheduler_name == "StepLR":
            scheduler.step()

    logger.info("Training completed!")


@hydra.main(version_base=None, config_path="conf", config_name="train_surface")
def launch(cfg: DictConfig):
    """Launch training with hydra configuration

    Args:
        cfg: Hydra configuration object
    """

    # If you want to use `line_profiler` or PyTorch's profiler, enable them here.

    profiler = Profiler()
    if cfg.profile:
        profiler.enable("torch")
        profiler.enable("line_profiler")
    profiler.initialize()
    main(cfg)
    profiler.finalize()


if __name__ == "__main__":
    launch()
