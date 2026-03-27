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

r"""
Pretraining module for source domain training.

Compatible with neuralop 2.0.0 API.
"""

import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split

from neuralop.training import AdamW
from neuralop.losses import LpLoss
from neuralop import get_model

from physicsnemo.launch.utils.checkpoint import save_checkpoint
from training.trainer import save_model_checkpoint

from datasets import FloodDatasetWithQueryPoints, NormalizedDataset
from data_processing import FloodGINODataProcessor, GINOWrapper, LpLossWrapper
from utils.normalization import collect_all_fields, stack_and_fit_transform
from training.trainer import NeuralOperatorTrainer


def create_scheduler(optimizer, config, logger=None):
    r"""
    Create learning rate scheduler based on config.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    config : Any
        Configuration object.
    logger : Any, optional
        Optional logger instance.

    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler
        Scheduler instance.

    Raises
    ------
    ValueError
        If scheduler name is unknown.
    """
    scheduler_name = config.training.get("scheduler", "StepLR")
    if scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.training.get("gamma", 0.5),
            patience=config.training.get("scheduler_patience", 5),
            mode="min",
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.get("scheduler_T_max", 200)
        )
    elif scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.training.get("step_size", 50),
            gamma=config.training.get("gamma", 0.5),
        )
    else:
        raise ValueError(f"Unknown scheduler {scheduler_name}")

    if logger:
        # Safely log debug message - PythonLogger doesn't have debug method
        # Access underlying logging.Logger if available (PythonLogger wraps it)
        try:
            # Handle RankZeroLoggingWrapper which wraps PythonLogger
            if hasattr(logger, 'obj') and hasattr(logger.obj, 'logger'):
                logger.obj.logger.debug(f"Created {scheduler_name} scheduler")
            # Handle direct PythonLogger
            elif hasattr(logger, 'logger') and hasattr(logger.logger, 'debug'):
                logger.logger.debug(f"Created {scheduler_name} scheduler")
            # Fallback: try direct debug method (for loggers that support it)
            elif hasattr(logger, 'debug'):
                logger.debug(f"Created {scheduler_name} scheduler")
        except (AttributeError, TypeError):
            # Skip debug logging if not available (not critical for scheduler creation)
            pass
    return scheduler


def pretrain_model(config, device, is_logger, source_data_config, logger=None):
    r"""
    Pretrain model on source domain data.

    Compatible with neuralop 2.0.0 Trainer API.

    Parameters
    ----------
    config : Any
        Configuration object.
    device : str or torch.device
        Device to train on.
    is_logger : bool
        Whether this process is the logger.
    source_data_config : Any
        Source data configuration.
    logger : Any, optional
        Optional logger instance.

    Returns
    -------
    Tuple[nn.Module, Dict[str, Any], Any]
        Tuple of (model, normalizers, trainer).
    """
    if logger is None:
        # Fallback to print if no logger provided
        def log_info(msg):
            print(msg)

        def log_debug(msg):
            pass

        logger = type("Logger", (), {"info": lambda self, msg: log_info(msg), "debug": lambda self, msg: log_debug(msg)})()

    logger.info("Starting pretraining on source domain...")
    
    # Create source dataset
    logger.info(f"Loading source dataset from: {source_data_config.root}")
    source_full_dataset = FloodDatasetWithQueryPoints(
        data_root=source_data_config.root,
        n_history=source_data_config.n_history,
        xy_file=getattr(source_data_config, "xy_file", None),
        query_res=getattr(source_data_config, "query_res", [64, 64]),
        static_files=getattr(source_data_config, "static_files", []),
        dynamic_patterns=getattr(source_data_config, "dynamic_patterns", {}),
        boundary_patterns=getattr(source_data_config, "boundary_patterns", {}),
        raise_on_smaller=True,
        skip_before_timestep=getattr(source_data_config, "skip_before_timestep", 0),
        noise_type=getattr(source_data_config, "noise_type", "none"),
        noise_std=getattr(source_data_config, "noise_std", None)
    )
    
    # Split into train/val
    train_sz_source = int(0.9 * len(source_full_dataset))
    source_train_raw, source_val_raw = random_split(
        source_full_dataset,
        [train_sz_source, len(source_full_dataset) - train_sz_source]
    )
    logger.info(f"Source domain: total={len(source_full_dataset)}, train={train_sz_source}, val={len(source_val_raw)}")
    
    # Collect and normalize training data
    logger.info("Collecting and normalizing training data...")
    geom_s_tr, static_s_tr, boundary_s_tr, dyn_s_tr, tgt_s_tr = collect_all_fields(source_train_raw, True)
    normalizers, big_source_train = stack_and_fit_transform(
        geom_s_tr, static_s_tr, boundary_s_tr, dyn_s_tr, tgt_s_tr
    )
    source_train_ds = NormalizedDataset(
        geometry=big_source_train["geometry"],
        static=big_source_train["static"],
        boundary=big_source_train["boundary"],
        dynamic=big_source_train["dynamic"],
        target=big_source_train["target"],
        query_res=source_data_config.query_res
    )
    source_train_loader = DataLoader(
        source_train_ds, batch_size=source_data_config.batch_size, shuffle=True
    )
    
    # Collect and normalize validation data
    logger.info("Collecting and normalizing validation data...")
    geom_s_val, static_s_val, boundary_s_val, dyn_s_val, tgt_s_val = collect_all_fields(source_val_raw, True)
    _, big_source_val = stack_and_fit_transform(
        geom_s_val, static_s_val, boundary_s_val, dyn_s_val, tgt_s_val,
        normalizers=normalizers, fit_normalizers=False
    )
    source_val_ds = NormalizedDataset(
        geometry=big_source_val["geometry"],
        static=big_source_val["static"],
        boundary=big_source_val["boundary"],
        dynamic=big_source_val["dynamic"],
        target=big_source_val["target"],
        query_res=source_data_config.query_res
    )
    source_val_loader = DataLoader(
        source_val_ds, batch_size=source_data_config.batch_size, shuffle=False
    )
    
    # Create model
    logger.info("Creating GINO model...")
    # Convert config.model to dict to avoid struct mode issues with neuralop's get_model
    # neuralop's get_model tries to pop from config, which doesn't work with struct mode
    # It expects config.model to exist, so we wrap it in a new OmegaConf DictConfig
    # (not in struct mode) that supports both attribute and dict access
    from omegaconf import OmegaConf
    model_config_dict = OmegaConf.to_container(config.model, resolve=True)
    
    # Extract autoregressive parameter before passing to get_model (GINO doesn't accept it)
    autoregressive = model_config_dict.pop("autoregressive", False)
    
    # Create a wrapper config that neuralop expects: {"model": {...}}
    # Convert to OmegaConf DictConfig (not struct mode) so it supports attribute access
    wrapper_config = OmegaConf.create({"model": model_config_dict})
    model = get_model(wrapper_config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {n_params:,} parameters")
    
    # Wrap model to filter out unexpected kwargs (like 'y') from Trainer
    # Enable autoregressive residual connection if specified in config
    model = GINOWrapper(model, autoregressive=autoregressive)
    
    # Create optimizer and scheduler
    lr = config.training.get("learning_rate", 1e-4)
    weight_decay = config.training.get("weight_decay", 1e-4)
    optimizer_src = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    logger.info(f"Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")
    scheduler_src = create_scheduler(optimizer_src, config, logger)
    
    # Create loss and data processor
    # Get loss type from config, default to 'l2'
    def create_loss(loss_type_str, default="l2"):
        """Helper function to create loss function from string."""
        loss_type_str = loss_type_str.lower()
        if loss_type_str == "l1":
            return LpLossWrapper(LpLoss(d=2, p=1)), "l1"
        elif loss_type_str == "l2":
            return LpLossWrapper(LpLoss(d=2, p=2)), "l2"
        else:
            logger.warning(f"Unknown loss type '{loss_type_str}', defaulting to '{default}'")
            return LpLossWrapper(LpLoss(d=2, p=2)), default
    
    training_loss_type = config.training.get("training_loss", "l2")
    training_loss_fn, training_loss_name = create_loss(training_loss_type)
    logger.info(f"Using {training_loss_name.upper()} loss for training")
    
    # Use testing_loss for evaluation if specified, otherwise use training_loss
    testing_loss_type = config.training.get("testing_loss", training_loss_type)
    eval_loss_fn, eval_loss_name = create_loss(testing_loss_type, default=training_loss_name)
    if testing_loss_type.lower() != training_loss_type.lower():
        logger.info(f"Using {eval_loss_name.upper()} loss for evaluation (different from training)")
    data_processor = FloodGINODataProcessor(
        device=device,
        target_norm=normalizers.get("target", None),
        inverse_test=True
    )
    data_processor.wrap(model)
    
    # Create trainer using PhysicsNeMo-style trainer
    n_epochs = config.training.get("n_epochs_source", config.training.get("n_epochs", 100))
    logger.info(f"Creating NeuralOperatorTrainer for {n_epochs} epochs...")
    trainer_src = NeuralOperatorTrainer(
        model=model,
        n_epochs=n_epochs,
        data_processor=data_processor,
        device=device,
        wandb_log=config.wandb.get("log", False),
        verbose=is_logger,
        logger=logger if hasattr(logger, 'info') else None,
    )

    # Train using neuralop 2.0.0 API
    save_dir = os.path.join(config.checkpoint.get("save_dir", "./checkpoints"), "pretrain")
    logger.info(f"Starting training... Checkpoints will be saved to: {save_dir}")
    logger.info(f"Training samples: {len(source_train_ds)}, Validation samples: {len(source_val_ds)}")

    # Get checkpoint saving options from config
    save_best = config.checkpoint.get("save_best", None)
    save_every = config.checkpoint.get("save_every", None)
    
    trainer_src.train(
        train_loader=source_train_loader,
        test_loaders={"source_val": source_val_loader},
        optimizer=optimizer_src,
        scheduler=scheduler_src,
        training_loss=training_loss_fn,
        eval_losses={eval_loss_name: eval_loss_fn},
        save_dir=save_dir,
        save_best=save_best,  # Save best model based on validation metric (from config)
        save_every=save_every,  # Save checkpoint every N epochs (from config)
        resume_from_dir=config.checkpoint.get("resume_from_source", None),
    )
    
    # Explicitly save final pretrained model checkpoint using PhysicsNeMo checkpoint system
    # Ensure directory exists before saving
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving final pretrained model checkpoint to {save_dir}")
    
    # Save model using helper function that handles PyTorch submodules
    # If it returns True, model was saved separately (as PyTorch model)
    model_saved_separately = save_model_checkpoint(
        model=model,
        save_dir=save_dir,
        epoch=n_epochs - 1,  # Final epoch (0-indexed)
        metadata={"stage": "pretrain", "final_epoch": True},
    )
    
    # Save optimizer, scheduler, and metadata using PhysicsNeMo
    # Include model only if it wasn't saved separately
    save_checkpoint(
        path=save_dir,
        models=None if model_saved_separately else model,
        optimizer=optimizer_src,
        scheduler=scheduler_src,
        scaler=None,
        epoch=n_epochs - 1,
        metadata={"stage": "pretrain", "final_epoch": True},
    )
    logger.info("Saved pretrained model checkpoint using PhysicsNeMo format")
    
    # Save normalizers to checkpoint directory
    normalizers_path = os.path.join(save_dir, "normalizers.pt")
    torch.save(normalizers, normalizers_path)
    logger.info(f"Saved normalizers to {normalizers_path}")
    
    logger.info("Pretraining completed!")
    return model, normalizers, trainer_src
