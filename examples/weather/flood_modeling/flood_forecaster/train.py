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
Training script for FloodForecaster using GINO with domain adaptation.

This script implements a two-stage training pipeline:
1. Pretraining on source domain
2. Domain adaptation on source + target domains

For rollout evaluation and visualization, use inference.py instead.
"""

import os
import sys
from typing import Any, Dict, Optional

import hydra
import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split

from neuralop import get_model
from neuralop.utils import get_wandb_api_key

from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.launch.logging.wandb import initialize_wandb

from datasets import (
    FloodDatasetWithQueryPoints,
    NormalizedDataset,
)
from data_processing import FloodGINODataProcessor
from training.pretraining import pretrain_model
from training.domain_adaptation import adapt_model
from utils.normalization import (
    collect_all_fields,
    stack_and_fit_transform,
)


def _register_hydra_resolvers() -> None:
    r"""
    Register custom Hydra resolvers if needed.
    
    This function can be extended to register custom resolvers for the config.
    Currently, OmegaConf provides built-in resolvers like oc.env for environment
    variables, so no custom registration is needed.
    
    Note: The config uses ${VAR:default} syntax which is Hydra's legacy
    environment variable interpolation. If this causes issues, consider migrating
    to ${oc.env:VAR,default} syntax in the config file.
    """
    # Placeholder for future custom resolvers if needed
    # oc.env is already built into OmegaConf, so no registration needed
    pass


def safe_config_to_dict(
    cfg: DictConfig,
    exclude_keys: Optional[list] = None,
    logger: Optional[RankZeroLoggingWrapper] = None,
) -> Dict[str, Any]:
    r"""
    Safely convert OmegaConf DictConfig to a Python dictionary for wandb logging.
    
    This function handles unresolved interpolations gracefully by:
    1. Attempting full resolution first
    2. Filtering out problematic keys if resolution fails
    3. Falling back to partial resolution if needed
    
    Parameters
    ----------
    cfg : DictConfig
        The OmegaConf configuration object to convert.
    exclude_keys : list, optional
        List of top-level keys to exclude from the output (e.g., ['rollout_data']).
        Defaults to ['rollout_data'] since it's only needed for inference.
    logger : RankZeroLoggingWrapper, optional
        Logger instance for warning messages. If None, warnings are suppressed.
    
    Returns
    -------
    Dict[str, Any]
        A Python dictionary representation of the config, suitable for wandb logging.
    
    Examples
    --------
    >>> config_dict = safe_config_to_dict(cfg, exclude_keys=['rollout_data'])
    >>> wandb.init(config=config_dict)
    """
    if exclude_keys is None:
        exclude_keys = ["rollout_data"]  # Default: exclude rollout_data (only for inference)
    
    # Strategy 1: Try full resolution first
    try:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        # Remove excluded keys
        for key in exclude_keys:
            config_dict.pop(key, None)
        return config_dict
    except Exception as e:
        if logger:
            logger.info(
                f"Config resolution encountered unresolved interpolations: {type(e).__name__}. "
                f"Attempting filtered resolution..."
            )
    
    # Strategy 2: Filter problematic keys, then resolve
    try:
        # Get unresolved config as dict
        config_dict_unresolved = OmegaConf.to_container(cfg, resolve=False)
        
        # Remove excluded keys
        filtered_dict = {
            key: value
            for key, value in config_dict_unresolved.items()
            if key not in exclude_keys
        }
        
        # Create new config from filtered dict and try to resolve
        # Use struct=False to allow modifications during resolution
        filtered_cfg = OmegaConf.create(filtered_dict)
        OmegaConf.set_struct(filtered_cfg, False)
        
        try:
            config_dict = OmegaConf.to_container(filtered_cfg, resolve=True)
            return config_dict
        except Exception:
            # If resolution still fails, try resolving each top-level key individually
            partially_resolved = {}
            for key, value in filtered_dict.items():
                try:
                    # Try to resolve this key's section
                    key_cfg = OmegaConf.create({key: value})
                    resolved_key = OmegaConf.to_container(key_cfg, resolve=True)
                    partially_resolved[key] = resolved_key[key]
                except Exception:
                    # If this key fails, include it unresolved
                    if isinstance(value, DictConfig):
                        partially_resolved[key] = OmegaConf.to_container(value, resolve=False)
                    else:
                        partially_resolved[key] = value
            return partially_resolved
    except Exception as e:
        if logger:
            logger.warning(
                f"Could not fully resolve config: {type(e).__name__}. "
                f"Using partially resolved config for wandb logging."
            )
    
    # Strategy 3: Fallback to unresolved config (last resort)
    config_dict = OmegaConf.to_container(cfg, resolve=False)
    for key in exclude_keys:
        config_dict.pop(key, None)
    return config_dict


def log_section(logger: RankZeroLoggingWrapper, title: str, char: str = "=", width: int = 60):
    r"""Log a section header for visual separation."""
    separator = char * width
    logger.info("")
    logger.info(separator)
    logger.info(title)
    logger.info(separator)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def train_flood_forecaster(cfg: DictConfig) -> None:
    r"""
    Main training pipeline for FloodForecaster.

    This function orchestrates the complete training workflow:
    1. Configuration loading and device setup
    2. Pretraining on source domain
    3. Domain adaptation on source + target domains

    After training completes, use inference.py to perform rollout evaluation
    and generate visualizations.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.

    Raises
    ------
    SystemExit
        If critical errors occur during execution.
    """
    # Register custom Hydra resolvers for environment variable interpolation
    _register_hydra_resolvers()
    
    # Initialize distributed manager (must be called first)
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize logging
    log = PythonLogger(name="flood_forecaster")
    log_rank_zero = RankZeroLoggingWrapper(log, dist)

    log_section(log_rank_zero, "FLOOD FORECASTER - Training and Evaluation Pipeline")

    try:
        # Get device from distributed manager or config
        device = dist.device if dist.device is not None else cfg.distributed.device
        is_logger = dist.rank == 0

        # Log device information prominently
        log_rank_zero.info("=" * 50)
        log_rank_zero.info(f"PyTorch version: {torch.__version__}")
        log_rank_zero.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log_rank_zero.info(f"CUDA version: {torch.version.cuda}")
            log_rank_zero.info(f"GPU device: {torch.cuda.get_device_name(0)}")
            log_rank_zero.info(
                f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        log_rank_zero.info(f"Using device: {device}")
        log_rank_zero.info(f"Distributed: rank={dist.rank}, world_size={dist.world_size}")
        log_rank_zero.info("=" * 50)

        if not torch.cuda.is_available():
            log_rank_zero.warning("CUDA is not available! Training will be very slow on CPU.")
            log_rank_zero.warning("Please check your PyTorch installation with CUDA support.")

        # Adjust FNO modes if needed (access via OmegaConf)
        if (
            hasattr(cfg, "source_data")
            and hasattr(cfg.source_data, "resolution")
            and hasattr(cfg.model, "fno_n_modes")
            and cfg.source_data.resolution < cfg.model.fno_n_modes[0]
        ):
            cfg.model.fno_n_modes = [cfg.source_data.resolution] * len(cfg.model.fno_n_modes)
            # Safely log debug message - PythonLogger doesn't have debug method
            try:
                # Check if logger has a 'logger' attribute (underlying logging.Logger)
                # RankZeroLoggingWrapper wraps PythonLogger which has a 'logger' attribute
                if hasattr(log_rank_zero, 'obj') and hasattr(log_rank_zero.obj, 'logger'):
                    log_rank_zero.obj.logger.debug(f"Adjusted FNO modes to: {cfg.model.fno_n_modes}")
                elif hasattr(log_rank_zero, 'logger') and hasattr(log_rank_zero.logger, 'debug'):
                    log_rank_zero.logger.debug(f"Adjusted FNO modes to: {cfg.model.fno_n_modes}")
                # Fallback: try direct debug method (for loggers that support it)
                elif hasattr(log_rank_zero, 'debug'):
                    log_rank_zero.debug(f"Adjusted FNO modes to: {cfg.model.fno_n_modes}")
            except (AttributeError, TypeError):
                # Skip debug logging if not available (not critical)
                pass

        # Initialize wandb if logging is enabled
        if cfg.wandb.log and is_logger:
            log_rank_zero.info("Initializing Weights & Biases logging...")
            # Try to login if API key is available, but don't fail if it's not
            # wandb.init() will handle authentication automatically if user has logged in via CLI
            try:
                api_key = get_wandb_api_key()
                if api_key:
                    wandb.login(key=api_key)
                    log_rank_zero.info("W&B API key found and logged in")
            except (KeyError, FileNotFoundError, Exception) as e:
                # API key not found - this is OK, wandb.init() will use existing login or prompt
                log_rank_zero.info(
                    "W&B API key not found in environment or file. "
                    "Will use existing wandb login or prompt for authentication."
                )
            
            wandb_name = (
                cfg.wandb.name
                if cfg.wandb.name
                else f"flood-run_{getattr(cfg.source_data, 'resolution', 64)}"
            )
            
            # Safely convert config to dict for wandb, handling unresolved interpolations gracefully
            # This excludes 'rollout_data' by default since it's only needed for inference
            wandb_config_dict = safe_config_to_dict(
                cfg,
                exclude_keys=["rollout_data"],  # Not needed for training logging
                logger=log_rank_zero,
            )
            
            wandb_init_args = dict(
                config=wandb_config_dict,
                name=wandb_name,
                group=cfg.wandb.group,
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
            )
            if cfg.wandb.sweep:
                for key in wandb.config.keys():
                    if hasattr(cfg, "params"):
                        cfg.params[key] = wandb.config[key]
            wandb.init(**wandb_init_args)
            log_rank_zero.info(f"W&B initialized: project={cfg.wandb.project}, name={wandb_name}")

        # Stage 1: Pretraining on source domain
        log_section(log_rank_zero, "Stage 1: Pretraining on Source Domain")
        model, normalizers, trainer_src = pretrain_model(
            config=cfg,
            device=device,
            is_logger=is_logger,
            source_data_config=cfg.source_data,
            logger=log_rank_zero,
        )

        # Recreate source loaders for domain adaptation
        log_rank_zero.info("Recreating source loaders for domain adaptation...")
        source_full_dataset = FloodDatasetWithQueryPoints(
            data_root=cfg.source_data.root,
            n_history=cfg.source_data.n_history,
            xy_file=cfg.source_data.get("xy_file", None),
            query_res=cfg.source_data.get("query_res", [64, 64]),
            static_files=cfg.source_data.get("static_files", []),
            dynamic_patterns=cfg.source_data.get("dynamic_patterns", {}),
            boundary_patterns=cfg.source_data.get("boundary_patterns", {}),
            raise_on_smaller=True,
            skip_before_timestep=cfg.source_data.get("skip_before_timestep", 0),
            noise_type=cfg.source_data.get("noise_type", "none"),
            noise_std=cfg.source_data.get("noise_std", None),
        )
        train_sz_source = int(0.9 * len(source_full_dataset))
        source_train_raw, source_val_raw = random_split(
            source_full_dataset,
            [train_sz_source, len(source_full_dataset) - train_sz_source],
        )

        # Move normalizers to CPU for data transformation
        for norm in normalizers.values():
            norm.to("cpu")

        geom_s_tr, static_s_tr, boundary_s_tr, dyn_s_tr, tgt_s_tr = collect_all_fields(
            source_train_raw, True
        )
        _, big_source_train = stack_and_fit_transform(
            geom_s_tr,
            static_s_tr,
            boundary_s_tr,
            dyn_s_tr,
            tgt_s_tr,
            normalizers=normalizers,
            fit_normalizers=False,
        )
        source_train_ds = NormalizedDataset(
            geometry=big_source_train["geometry"],
            static=big_source_train["static"],
            boundary=big_source_train["boundary"],
            dynamic=big_source_train["dynamic"],
            target=big_source_train["target"],
            query_res=cfg.source_data.query_res,
        )
        source_train_loader = DataLoader(
            source_train_ds, batch_size=cfg.source_data.batch_size, shuffle=True
        )

        geom_s_val, static_s_val, boundary_s_val, dyn_s_val, tgt_s_val = collect_all_fields(
            source_val_raw, True
        )
        _, big_source_val = stack_and_fit_transform(
            geom_s_val,
            static_s_val,
            boundary_s_val,
            dyn_s_val,
            tgt_s_val,
            normalizers=normalizers,
            fit_normalizers=False,
        )
        source_val_ds = NormalizedDataset(
            geometry=big_source_val["geometry"],
            static=big_source_val["static"],
            boundary=big_source_val["boundary"],
            dynamic=big_source_val["dynamic"],
            target=big_source_val["target"],
            query_res=cfg.source_data.query_res,
        )
        source_val_loader = DataLoader(
            source_val_ds, batch_size=cfg.source_data.batch_size, shuffle=False
        )

        # Stage 2: Domain adaptation
        log_section(log_rank_zero, "Stage 2: Domain Adaptation")
        data_processor = trainer_src.data_processor
        
        # Calculate wandb step offset to continue from pretraining
        # neuralop Trainer uses step=epoch+1, so if pretraining ran for n_epochs_source,
        # domain adaptation should start from step n_epochs_source + 1
        n_epochs_source = cfg.training.get("n_epochs_source", cfg.training.get("n_epochs", 100))
        wandb_step_offset = n_epochs_source if (cfg.wandb.log and is_logger) else 0
        
        model, domain_classifier, trainer_adapt = adapt_model(
            model=model,
            normalizers=normalizers,
            data_processor=data_processor,
            config=cfg,
            device=device,
            is_logger=is_logger,
            source_train_loader=source_train_loader,
            source_val_loader=source_val_loader,
            target_data_config=cfg.target_data,
            logger=log_rank_zero,
            wandb_step_offset=wandb_step_offset,
        )

        if cfg.wandb.log and is_logger:
            wandb.finish()
            log_rank_zero.info("W&B logging finished")

        log_section(log_rank_zero, "Training Complete!")
        log_rank_zero.info("")
        log_rank_zero.info("To perform rollout evaluation and generate visualizations,")
        log_rank_zero.info("run: python inference.py --config-path conf --config-name config")
        log_rank_zero.info("")

    except KeyboardInterrupt:
        log_rank_zero.warning("Training interrupted by user")
        if cfg.wandb.log and is_logger:
            wandb.finish()
        sys.exit(1)
    except Exception as e:
        import traceback
        log_rank_zero.error(f"Fatal error in main pipeline: {e}")
        log_rank_zero.error(traceback.format_exc())
        if cfg.wandb.log and is_logger:
            wandb.finish()
        raise


if __name__ == "__main__":
    train_flood_forecaster()

