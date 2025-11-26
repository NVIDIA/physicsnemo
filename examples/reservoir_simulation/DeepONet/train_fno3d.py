#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Training script for U-FNO CO2 sequestration model."""

import hydra
from omegaconf import DictConfig
from pathlib import Path
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import mlflow
import mlflow.pytorch

from ufno import UFNONet
from physicsnemo_unet import StandaloneUNet
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import PythonLogger, LaunchLogger

from dataset import create_dataloaders
from losses import get_loss_function, UnifiedLoss
from metrics import mean_relative_error, mean_plume_error
from utils import dnorm_dP
from data_validation import validate_batch_dimensions, print_validation_summary


@hydra.main(version_base="1.3", config_path="conf", config_name="training_config")
def main(cfg: DictConfig) -> None:
    """Main training function for FNO3D CO2 sequestration model."""

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Helper variable for MLFlow logging (only on rank 0)
    use_mlflow = cfg.logging.use_mlflow and dist.rank == 0

    # Set random seeds for reproducibility
    if hasattr(cfg, "seed"):
        import random
        import numpy as np

        seed = cfg.seed + dist.rank  # Different seed per rank for data augmentation
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Set deterministic behavior if requested
        if cfg.compute.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        elif cfg.compute.benchmark:
            torch.backends.cudnn.benchmark = True

    # Initialize logger
    logger = PythonLogger(name="fno3d_co2")
    logger.file_logging()
    LaunchLogger.initialize()

    # Print header (only on rank 0)
    if dist.rank == 0:
        logger.info("=" * 80)
        logger.info(
            f"U-FNO Training | Variable: {cfg.data.variable} | GPUs: {dist.world_size}"
        )
        logger.info("=" * 80)

    # Auto-set num_workers based on num_gpus if not specified
    num_workers = cfg.data.num_workers
    if num_workers is None and hasattr(cfg.compute, "num_gpus"):
        num_workers = cfg.compute.num_gpus * 2  # 2 workers per GPU
    elif num_workers is None:
        num_workers = 4  # Default fallback

    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=cfg.data.data_path,
        variable=cfg.data.variable,
        batch_size=cfg.training.batch_size,
        normalize=cfg.data.normalize,
        num_workers=num_workers,
        device=dist.device,
    )

    # Print data info (only on rank 0)
    if dist.rank == 0:
        effective_batch_size = cfg.training.batch_size * dist.world_size
        logger.info(
            f"Data: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)} | Batch size={cfg.training.batch_size} per GPU (Effective: {effective_batch_size})"
        )

    # Validate data dimensions before training (dynamic validation)
    if dist.rank == 0:
        logger.info("Validating data dimensions...")

        # Get a sample batch to check dimensions
        sample_inputs, sample_targets = next(iter(train_loader))

        # Validate using centralized validation function
        validate_batch_dimensions(sample_inputs, sample_targets, cfg.data.variable)

        # Print validation summary
        print_validation_summary(
            input_shape=tuple(sample_inputs.shape),
            target_shape=tuple(sample_targets.shape),
            variable=cfg.data.variable,
            is_batch=True,
            logger=logger,
        )

    # Create model based on model_type
    model_type = cfg.arch.model_type.lower()

    if model_type == "ufno":
        # U-FNO or Conv-FNO: Fourier layers + enhancement
        num_unet = cfg.arch.ufno.num_unet_layers
        num_conv = cfg.arch.ufno.num_conv_layers

        if num_unet > 0 and num_conv > 0:
            # Conv-U-FNO: Both U-Net and Conv layers (not recommended but possible)
            logger.warning(
                "⚠️  WARNING: Using both U-Net and Conv layers (Conv-U-FNO). This is not recommended but supported."
            )
            logger.warning(
                "⚠️  Consider using either U-FNO (num_conv_layers=0) or Conv-FNO (num_unet_layers=0) for better performance."
            )
            logger.info(
                f"Creating Conv-U-FNO model (FNO: {cfg.arch.ufno.num_fno_layers}, U-Net: {num_unet}, Conv: {num_conv})"
            )
            model_arch_name = f"convufno_{cfg.arch.ufno.unet_type}"
        elif num_unet > 0:
            # U-FNO
            logger.info(
                f"Creating U-FNO model (FNO layers: {cfg.arch.ufno.num_fno_layers}, U-Net layers: {num_unet}, U-Net type: {cfg.arch.ufno.unet_type})"
            )
            model_arch_name = f"ufno_{cfg.arch.ufno.unet_type}"
        elif num_conv > 0:
            # Conv-FNO
            logger.info(
                f"Creating Conv-FNO model (FNO layers: {cfg.arch.ufno.num_fno_layers}, Conv layers: {num_conv})"
            )
            model_arch_name = "convfno"
        else:
            # Standard FNO
            logger.info(
                f"Creating standard FNO model (FNO layers: {cfg.arch.ufno.num_fno_layers})"
            )
            model_arch_name = "fno"

        model = UFNONet(
            in_channels=cfg.arch.ufno.in_channels,
            out_channels=cfg.arch.ufno.out_channels,
            width=cfg.arch.ufno.width,
            modes1=cfg.arch.ufno.modes1,
            modes2=cfg.arch.ufno.modes2,
            modes3=cfg.arch.ufno.modes3,
            num_fno_layers=cfg.arch.ufno.num_fno_layers,
            num_unet_layers=num_unet,
            num_conv_layers=num_conv,
            padding=cfg.arch.ufno.padding,
            conv_kernel_size=cfg.arch.ufno.conv_kernel_size,
            unet_kernel_size=cfg.arch.ufno.unet_kernel_size,
            unet_dropout=cfg.arch.ufno.unet_dropout,
            unet_type=cfg.arch.ufno.unet_type,
            activation_fn=cfg.arch.ufno.activation_fn,
            lifting_type=cfg.arch.ufno.lifting_type,
            lifting_layers=cfg.arch.ufno.lifting_layers,
            lifting_width=cfg.arch.ufno.lifting_width,
            decoder_type=cfg.arch.ufno.decoder_type,
            decoder_layers=cfg.arch.ufno.decoder_layers,
            decoder_width=cfg.arch.ufno.decoder_width,
        ).to(dist.device)

    elif model_type == "unet":
        # Standalone U-Net (no Fourier layers)
        unet_type = cfg.arch.unet.unet_type
        logger.info(f"Creating standalone U-Net model (type: {unet_type})")

        if unet_type == "physicsnemo":
            unet_kwargs = dict(cfg.arch.unet.physicsnemo)
        elif unet_type == "custom":
            unet_kwargs = dict(cfg.arch.unet.custom)
        else:
            raise ValueError(f"Unknown unet_type: {unet_type}")

        model = StandaloneUNet(
            in_channels=cfg.arch.unet.in_channels,
            out_channels=cfg.arch.unet.out_channels,
            unet_type=unet_type,
            **unet_kwargs,
        ).to(dist.device)
        model_arch_name = f"unet_{unet_type}"

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'ufno' or 'unet'.")

    # Wrap model with DistributedDataParallel for multi-GPU training
    if dist.world_size > 1:
        model = DDP(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.local_rank,
            find_unused_parameters=False,
        )

    # Count trainable parameters
    model_for_counting = model.module if isinstance(model, DDP) else model
    if hasattr(model_for_counting, "count_params"):
        trainable_params = model_for_counting.count_params()
    else:
        trainable_params = sum(
            p.numel() for p in model_for_counting.parameters() if p.requires_grad
        )

    # Print model info (only on rank 0)
    if dist.rank == 0:
        logger.info(
            f"Model: {model.__class__.__name__} | Parameters: {trainable_params:,}"
        )

    # Create training loss function
    loss_fn = get_loss_function(cfg.loss)

    # Create validation loss function (same as training loss for fair comparison)
    from omegaconf import DictConfig

    if cfg.loss.base_loss_type == "simple_relative_l2":
        # Use same simple loss for validation
        val_loss_fn = loss_fn
    else:
        # Use UnifiedLoss with same masking but no derivatives for validation
        val_loss_cfg = DictConfig(
            {
                "base_loss_type": cfg.loss.base_loss_type,
                "use_mask": cfg.loss.use_mask,  # Use same mask as training
                "use_derivative": False,  # No derivatives in validation
                "reduction": cfg.loss.get("reduction", "sum"),
            }
        )
        val_loss_fn = get_loss_function(val_loss_cfg)

    # Print loss info (only on rank 0)
    if dist.rank == 0:
        if cfg.loss.base_loss_type == "simple_relative_l2":
            loss_info = f"Train Loss: SIMPLE_RELATIVE_L2 (no mask, no derivatives) | Val Loss: SIMPLE_RELATIVE_L2"
        else:
            loss_info = f"Train Loss: {cfg.loss.base_loss_type.upper()} | Val Loss: {cfg.loss.base_loss_type.upper()}"
            if cfg.loss.use_derivative:
                loss_info += f" (+Derivative w={cfg.loss.derivative_weight})"
            if cfg.loss.use_mask:
                loss_info += " (+Masking)"
        logger.info(loss_info)

    # Create optimizer and scheduler
    optimizer = Adam(
        model.parameters(),
        lr=cfg.training.initial_lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    # Create scheduler based on config type
    scheduler_type = cfg.scheduler.type.lower()
    if scheduler_type == "step":
        scheduler = StepLR(
            optimizer, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma
        )
    elif scheduler_type == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=cfg.scheduler.gamma)
    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. Must be 'step' or 'exponential'"
        )

    # Initialize AMP GradScaler if enabled
    scaler = GradScaler() if cfg.training.use_amp else None

    # Print optimizer info (only on rank 0)
    if dist.rank == 0:
        logger.info(
            f"Optimizer: Adam (lr={cfg.training.initial_lr}) | AMP: {cfg.training.use_amp}"
        )

    # Setup MLFlow tracking if enabled
    if use_mlflow:
        mlflow.set_experiment(cfg.logging.experiment_name)

        # Enable automatic system metrics logging (CPU, GPU, memory, disk, network)
        mlflow.enable_system_metrics_logging()

        mlflow.start_run()

        # Log hyperparameters
        mlflow.log_params(
            {
                "batch_size": cfg.training.batch_size,
                "epochs": cfg.training.epochs,
                "learning_rate": cfg.training.initial_lr,
                "optimizer": "Adam",
                "train_loss": cfg.loss.base_loss_type,
                "val_loss": cfg.loss.base_loss_type
                if cfg.loss.base_loss_type == "simple_relative_l2"
                else cfg.loss.base_loss_type,
                "loss_masking": cfg.loss.use_mask,
                "loss_derivative": cfg.loss.use_derivative,
                "loss_derivative_weight": cfg.loss.derivative_weight
                if cfg.loss.use_derivative
                else 0.0,
                "loss_derivative_dim": cfg.loss.derivative_dim
                if cfg.loss.use_derivative
                else None,
                "loss_reduction": cfg.loss.get("reduction", "sum"),
                "architecture": "U-FNO" if cfg.arch.ufno.num_unet_layers > 0 else "FNO",
                "has_unet": cfg.arch.ufno.num_unet_layers > 0,
                "data_format": "H × W × T",
                "in_channels": cfg.arch.ufno.in_channels,
                "out_channels": cfg.arch.ufno.out_channels,
                "width": cfg.arch.ufno.width,
                "modes1": cfg.arch.ufno.modes1,
                "modes2": cfg.arch.ufno.modes2,
                "modes3": cfg.arch.ufno.modes3,
                "num_fno_layers": cfg.arch.ufno.num_fno_layers,
                "num_unet_layers": cfg.arch.ufno.num_unet_layers,
                "padding": cfg.arch.ufno.padding,
                "activation_fn": cfg.arch.activation_fn,
                "use_amp": cfg.training.use_amp,
                "use_graphs": cfg.training.use_graphs,
                "variable": cfg.data.variable,
                "trainable_parameters": trainable_params,
            }
        )

    # Setup checkpointing (make absolute path since chdir=False)
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        # If relative, make it relative to the working directory (U-FNO folder)
        checkpoint_dir = Path.cwd() / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if resuming
    start_epoch = 1
    best_val_loss = float("inf")
    best_val_mre = float("inf")

    if (
        hasattr(cfg.training, "resume_from_checkpoint")
        and cfg.training.resume_from_checkpoint
    ):
        checkpoint_path = Path(cfg.training.resume_from_checkpoint)
        if checkpoint_path.exists():
            if dist.rank == 0:
                logger.info(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=dist.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint.get("val_loss", float("inf"))
            best_val_mre = checkpoint.get("val_mre", float("inf"))
            if dist.rank == 0:
                logger.success(
                    f"Resumed from epoch {checkpoint['epoch']}, best val loss: {best_val_loss:.6f}, best val MRE: {best_val_mre:.6f}"
                )
                logger.info(
                    f"Continuing training from epoch {start_epoch} to {cfg.training.epochs}"
                )
        else:
            if dist.rank == 0:
                logger.warning(
                    f"Checkpoint not found at {checkpoint_path}, starting from scratch"
                )
    else:
        if dist.rank == 0:
            logger.success("Starting training from scratch...")

    # Print training start header (only on rank 0)
    if dist.rank == 0:
        logger.info("=" * 80)
        logger.info("Starting training...")
        logger.info("=" * 80)

    # Training loop
    for epoch in range(start_epoch, cfg.training.epochs + 1):
        # Set epoch for distributed sampler (ensures proper shuffling across epochs)
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # Training phase
        with LaunchLogger(
            "train", epoch=epoch, num_mini_batch=len(train_loader)
        ) as log:
            model.train()
            total_loss = 0.0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Move data to GPU
                inputs = inputs.to(dist.device)
                targets = targets.to(dist.device)

                optimizer.zero_grad()

                # Forward pass with optional AMP
                if cfg.training.use_amp:
                    with autocast():
                        pred = model(inputs)
                        # UnifiedLoss always accepts inputs (for masking/derivatives)
                        loss = loss_fn(pred, targets, inputs)
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(inputs)
                    # UnifiedLoss always accepts inputs (for masking/derivatives)
                    loss = loss_fn(pred, targets, inputs)
                    loss.backward()
                    optimizer.step()

                # Aggregate loss across GPUs for accurate logging
                if dist.world_size > 1:
                    loss_tensor = loss.detach().clone()
                    torch.distributed.all_reduce(
                        loss_tensor, op=torch.distributed.ReduceOp.SUM
                    )
                    loss_tensor = loss_tensor / dist.world_size
                    total_loss += loss_tensor
                else:
                    total_loss += loss.detach()

            avg_train_loss = total_loss / len(train_loader)
            log.log_epoch({"loss": avg_train_loss})

            # Log to MLFlow (only on rank 0)
            if cfg.logging.use_mlflow and dist.rank == 0:
                mlflow.log_metric("train_loss", float(avg_train_loss), step=epoch)

        # Validation phase
        if epoch % cfg.training.validate_freq == 0:
            with LaunchLogger("valid", epoch=epoch) as log:
                model.eval()
                total_val_loss = 0.0
                mre_list = []

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(dist.device)
                        targets = targets.to(dist.device)

                        # Forward pass with optional AMP
                        if cfg.training.use_amp:
                            with autocast():
                                pred = model(inputs)
                                # Use Relative L2 loss for validation
                                val_loss = val_loss_fn(pred, targets, inputs)
                        else:
                            pred = model(inputs)
                            # Use Relative L2 loss for validation
                            val_loss = val_loss_fn(pred, targets, inputs)

                        # Aggregate validation loss across GPUs
                        if dist.world_size > 1:
                            val_loss_tensor = val_loss.detach().clone()
                            torch.distributed.all_reduce(
                                val_loss_tensor, op=torch.distributed.ReduceOp.SUM
                            )
                            val_loss_tensor = val_loss_tensor / dist.world_size
                            total_val_loss += val_loss_tensor
                        else:
                            total_val_loss += val_loss.detach()

                        # Calculate MRE on rank 0 only (for logging)
                        if dist.rank == 0:
                            # Denormalize predictions and targets based on variable type
                            pred_cpu = pred.cpu().numpy()
                            targets_cpu = targets.cpu().numpy()
                            inputs_cpu = inputs.cpu().numpy()

                            # Denormalize predictions and targets (only for pressure)
                            if cfg.data.variable == "pressure":
                                pred_denorm = dnorm_dP(pred_cpu)
                                targets_denorm = dnorm_dP(targets_cpu)
                            else:  # saturation - already in physical units [0, 1]
                                pred_denorm = pred_cpu
                                targets_denorm = targets_cpu

                            # Compute appropriate metric for each sample in the batch
                            for i in range(pred_denorm.shape[0]):
                                # Extract mask from input (permeability channel)
                                mask = inputs_cpu[i, :, :, 0, 0] != 0
                                thickness = np.sum(mask[:, 0])

                                # Extract masked region
                                y_true = targets_denorm[i][mask].reshape(
                                    (thickness, 200, 24)
                                )
                                y_pred = pred_denorm[i][mask].reshape(
                                    (thickness, 200, 24)
                                )

                                # Compute metric based on variable type
                                if cfg.data.variable == "pressure":
                                    # MRE for pressure
                                    metric = mean_relative_error(y_pred, y_true)
                                else:  # saturation
                                    # MPE for saturation
                                    metric = mean_plume_error(y_pred, y_true)
                                mre_list.append(metric)

                avg_val_loss = total_val_loss / len(val_loader)
                avg_metric = np.mean(mre_list) if len(mre_list) > 0 else 0.0

                # Determine metric name based on variable type
                metric_name = "MRE" if cfg.data.variable == "pressure" else "MPE"
                metric_key = "val_mre" if cfg.data.variable == "pressure" else "val_mpe"

                # Print validation metrics (only on rank 0)
                if dist.rank == 0:
                    logger.info(
                        f"Epoch {epoch}: Val Loss = {avg_val_loss:.6f} | Val {metric_name} = {avg_metric:.6f} ({avg_metric * 100:.2f}%)"
                    )

                # Log to MLFlow (only on rank 0)
                if cfg.logging.use_mlflow and dist.rank == 0:
                    mlflow.log_metric("val_loss", float(avg_val_loss), step=epoch)
                    mlflow.log_metric(metric_key, float(avg_metric), step=epoch)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_mre = avg_metric

                    # Print and log best validation loss (only on rank 0)
                    if dist.rank == 0:
                        logger.success(
                            f"New best validation: Loss = {best_val_loss:.6f} | {metric_name} = {best_val_mre:.6f} ({best_val_mre * 100:.2f}%)"
                        )

                    # Log best loss to MLFlow (only on rank 0)
                    if cfg.logging.use_mlflow and dist.rank == 0:
                        mlflow.log_metric(
                            "best_val_loss", float(best_val_loss), step=epoch
                        )
                        mlflow.log_metric(
                            f"best_{metric_key}", float(best_val_mre), step=epoch
                        )

                    # Save best model (only on rank 0 to avoid race condition)
                    if dist.rank == 0:
                        best_model_path = (
                            checkpoint_dir
                            / f"best_model_{cfg.data.variable}_{model_arch_name}.pth"
                        )
                        # For DDP models, save the underlying module's state_dict
                        model_to_save = (
                            model.module if isinstance(model, DDP) else model
                        )

                        # Prepare model config to save with checkpoint
                        model_config = {
                            "model_type": model_type,
                            "model_arch_name": model_arch_name,
                            "variable": cfg.data.variable,
                        }

                        if model_type == "ufno":
                            model_config.update(
                                {
                                    "in_channels": cfg.arch.ufno.in_channels,
                                    "out_channels": cfg.arch.ufno.out_channels,
                                    "width": cfg.arch.ufno.width,
                                    "modes1": cfg.arch.ufno.modes1,
                                    "modes2": cfg.arch.ufno.modes2,
                                    "modes3": cfg.arch.ufno.modes3,
                                    "num_fno_layers": cfg.arch.ufno.num_fno_layers,
                                    "num_unet_layers": num_unet,
                                    "num_conv_layers": num_conv,
                                    "padding": cfg.arch.ufno.padding,
                                    "conv_kernel_size": cfg.arch.ufno.conv_kernel_size,
                                    "unet_kernel_size": cfg.arch.ufno.unet_kernel_size,
                                    "unet_dropout": cfg.arch.ufno.unet_dropout,
                                    "unet_type": cfg.arch.ufno.unet_type,
                                    "activation_fn": cfg.arch.ufno.activation_fn,
                                    "lifting_type": cfg.arch.ufno.lifting_type,
                                    "lifting_layers": cfg.arch.ufno.lifting_layers,
                                    "lifting_width": cfg.arch.ufno.lifting_width,
                                    "decoder_type": cfg.arch.ufno.decoder_type,
                                    "decoder_layers": cfg.arch.ufno.decoder_layers,
                                    "decoder_width": cfg.arch.ufno.decoder_width,
                                }
                            )
                        elif model_type == "unet":
                            model_config.update(
                                {
                                    "in_channels": cfg.arch.unet.in_channels,
                                    "out_channels": cfg.arch.unet.out_channels,
                                    "unet_type": unet_type,
                                }
                            )
                            if unet_type == "physicsnemo":
                                model_config["unet_kwargs"] = dict(
                                    cfg.arch.unet.physicsnemo
                                )
                            elif unet_type == "custom":
                                model_config["unet_kwargs"] = dict(cfg.arch.unet.custom)

                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": model_to_save.state_dict(),
                                "val_loss": best_val_loss,
                                "val_mre": best_val_mre,
                                "model_config": model_config,
                            },
                            best_model_path,
                        )

                        # Log model to MLFlow
                        if cfg.logging.use_mlflow:
                            mlflow.log_artifact(str(best_model_path))

        # Learning rate scheduling (StepLR steps automatically every step_size epochs)
        scheduler.step()

    # Print training completion (only on rank 0)
    if dist.rank == 0:
        metric_name = "MRE" if cfg.data.variable == "pressure" else "MPE"
        logger.success("Training completed! 🎉")
        logger.info(
            f"Best validation: Loss = {best_val_loss:.6f} | {metric_name} = {best_val_mre:.6f} ({best_val_mre * 100:.2f}%)"
        )

    # End MLFlow run (only on rank 0)
    if cfg.logging.use_mlflow and dist.rank == 0:
        metric_key = (
            "final_best_val_mre"
            if cfg.data.variable == "pressure"
            else "final_best_val_mpe"
        )
        mlflow.log_metric("final_best_val_loss", float(best_val_loss))
        mlflow.log_metric(metric_key, float(best_val_mre))
        mlflow.end_run()
        logger.info("MLFlow run completed")


if __name__ == "__main__":
    main()
