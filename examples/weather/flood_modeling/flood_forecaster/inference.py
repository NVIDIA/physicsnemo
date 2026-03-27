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
Inference script for FloodForecaster using trained GINO model.

This script loads a trained model checkpoint and performs rollout evaluation
on test data, generating visualizations and metrics.
"""

import sys
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from neuralop import get_model

from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.launch.utils.checkpoint import load_checkpoint

from datasets import FloodRolloutTestDatasetNew, NormalizedRolloutTestDataset
from data_processing import FloodGINODataProcessor, GINOWrapper
from inference.rollout import rollout_prediction
from utils.normalization import collect_all_fields, transform_with_existing_normalizers


def log_section(logger: RankZeroLoggingWrapper, title: str, char: str = "=", width: int = 60):
    r"""Log a section header for visual separation."""
    separator = char * width
    logger.info("")
    logger.info(separator)
    logger.info(title)
    logger.info(separator)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run_inference(cfg: DictConfig) -> None:
    r"""
    Run inference using a trained model checkpoint.

    This function loads a trained model and performs rollout evaluation:
    1. Load model from checkpoint
    2. Load and normalize test data
    3. Perform autoregressive rollout
    4. Generate visualizations and metrics

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.

    Raises
    ------
    SystemExit
        If critical errors occur during execution.
    """
    # Initialize distributed manager (must be called first)
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize logging
    log = PythonLogger(name="flood_forecaster_inference")
    log_rank_zero = RankZeroLoggingWrapper(log, dist)

    log_section(log_rank_zero, "FLOOD FORECASTER - Inference and Evaluation")

    try:
        # Get device from distributed manager or config
        device = dist.device if dist.device is not None else cfg.distributed.device

        # Log device information
        log_rank_zero.info("=" * 50)
        log_rank_zero.info(f"PyTorch version: {torch.__version__}")
        log_rank_zero.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log_rank_zero.info(f"CUDA version: {torch.version.cuda}")
            log_rank_zero.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        log_rank_zero.info(f"Using device: {device}")
        log_rank_zero.info(f"Distributed: rank={dist.rank}, world_size={dist.world_size}")
        log_rank_zero.info("=" * 50)

        # Check checkpoint path
        checkpoint_path = cfg.checkpoint.get("resume_from_adapt") or cfg.checkpoint.get("resume_from_source")
        if checkpoint_path is None:
            log_rank_zero.error("No checkpoint path specified in config.checkpoint.resume_from_adapt or resume_from_source")
            sys.exit(1)

        checkpoint_path = Path(to_absolute_path(checkpoint_path))
        if not checkpoint_path.exists():
            log_rank_zero.error(f"Checkpoint path does not exist: {checkpoint_path}")
            sys.exit(1)

        log_rank_zero.info(f"Loading model from checkpoint: {checkpoint_path}")

        # Create model (same as training)
        log_rank_zero.info("Creating GINO model...")
        # Convert config.model to dict to avoid struct mode issues with neuralop's get_model
        # neuralop's get_model tries to pop from config, which doesn't work with struct mode
        # It expects config.model to exist, so we wrap it in a new OmegaConf DictConfig
        # (not in struct mode) that supports both attribute and dict access
        model_config_dict = OmegaConf.to_container(cfg.model, resolve=True)
        # Create a wrapper config that neuralop expects: {"model": {...}}
        # Extract autoregressive parameter before passing to get_model (GINO doesn't accept it)
        autoregressive = model_config_dict.pop("autoregressive", False)
        
        # Convert to OmegaConf DictConfig (not struct mode) so it supports attribute access
        wrapper_config = OmegaConf.create({"model": model_config_dict})
        gino_model = get_model(wrapper_config)
        gino_model = gino_model.to(device)

        # Create GINOWrapper first (checkpoints are saved as GINOWrapper in PhysicsNeMo format)
        # Enable autoregressive residual connection if specified in config
        model = GINOWrapper(gino_model, autoregressive=autoregressive)
        model = model.to(device)

        # Load checkpoint into the model
        # Support both PhysicsNeMo format (new) and neuralop format (old) for backward compatibility
        checkpoint_loaded = False
        
        # Try PhysicsNeMo format first (new format)
        # Checkpoints are saved as GINOWrapper (PhysicsNeMo Module), so we load into the wrapper
        try:
            metadata_dict = {}
            load_checkpoint(
                path=str(checkpoint_path),
                models=model,  # Load into GINOWrapper (PhysicsNeMo Module)
                optimizer=None,
                scheduler=None,
                scaler=None,
                epoch=None,  # Load latest checkpoint
                metadata_dict=metadata_dict,
                device=device,
            )
            log_rank_zero.info("Loaded checkpoint using PhysicsNeMo format")
            checkpoint_loaded = True
        except (FileNotFoundError, KeyError, ValueError) as e:
            # Fall back to neuralop format (old format)
            log_rank_zero.info(f"PhysicsNeMo checkpoint not found, trying neuralop format: {e}")
            
        # Try neuralop format (old format) if PhysicsNeMo format failed
        # For old format, we need to load into the inner gino_model, not the wrapper
        if not checkpoint_loaded:
            try:
                from neuralop.training.training_state import load_training_state
                
                # Check for neuralop checkpoint files
                if (checkpoint_path / "best_model_state_dict.pt").exists():
                    save_name = "best_model"
                elif (checkpoint_path / "model_state_dict.pt").exists():
                    save_name = "model"
                else:
                    log_rank_zero.error(f"No checkpoint found in {checkpoint_path}")
                    log_rank_zero.error("Tried both PhysicsNeMo and neuralop formats")
                    sys.exit(1)

                # Load checkpoint using neuralop format into the inner model
                # Extract the inner model from GINOWrapper for loading
                inner_model = model.model if hasattr(model, 'model') else gino_model
                inner_model, _, _, _, _ = load_training_state(
                    save_dir=checkpoint_path,
                    save_name=save_name,
                    model=inner_model,
                    optimizer=None,
                    scheduler=None,
                )
                log_rank_zero.info(f"Loaded checkpoint using neuralop format: {save_name}")
                checkpoint_loaded = True
            except Exception as e:
                log_rank_zero.error(f"Failed to load checkpoint in both formats: {e}")
                sys.exit(1)
        
        if not checkpoint_loaded:
            log_rank_zero.error("Failed to load checkpoint in any format")
            sys.exit(1)

        # Load normalizers from checkpoint if available
        # Normalizers are saved in both pretrain and adapt folders, so check both locations
        normalizers_path = checkpoint_path / "normalizers.pt"
        if not normalizers_path.exists():
            # If loading from adapt folder, also check pretrain folder as fallback
            # (normalizers are saved in both places and are the same)
            if checkpoint_path.name == "adapt":
                pretrain_path = checkpoint_path.parent / "pretrain" / "normalizers.pt"
                if pretrain_path.exists():
                    normalizers_path = pretrain_path
                    log_rank_zero.info(f"Normalizers not found in adapt folder, checking pretrain folder...")
        
        if normalizers_path.exists():
            log_rank_zero.info(f"Loading normalizers from: {normalizers_path}")
            normalizers = torch.load(normalizers_path, map_location=device)
            log_rank_zero.info("Normalizers loaded successfully")
        else:
            # Fallback: recreate normalizers from source data if not saved
            log_rank_zero.info("Normalizers not found in checkpoint or pretrain folder. Recreating from source data...")
            from datasets import FloodDatasetWithQueryPoints, NormalizedDataset
            from utils.normalization import stack_and_fit_transform

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
                noise_type="none",
                noise_std=None,
            )

            # Use a subset to fit normalizers (faster)
            from torch.utils.data import random_split

            train_sz = min(100, int(0.9 * len(source_full_dataset)))  # Use up to 100 samples
            source_train_subset, _ = random_split(source_full_dataset, [train_sz, len(source_full_dataset) - train_sz])

            geom, static, boundary, dyn, tgt = collect_all_fields(source_train_subset, True)
            normalizers, _ = stack_and_fit_transform(geom, static, boundary, dyn, tgt)
            log_rank_zero.info("Normalizers recreated from source data")

        # Create data processor
        data_processor = FloodGINODataProcessor(
            device=device,
            target_norm=normalizers.get("target", None),
            inverse_test=True,
        )
        data_processor.wrap(model)

        # Load rollout test dataset
        log_section(log_rank_zero, "Loading Rollout Test Dataset")
        rollout_test_dataset = FloodRolloutTestDatasetNew(
            rollout_data_root=cfg.rollout_data.root,
            n_history=cfg.source_data.n_history,
            rollout_length=cfg.source_data.rollout_length,
            xy_file=cfg.rollout_data.get("xy_file", None),
            query_res=cfg.source_data.get("query_res", [32, 32]),
            static_files=cfg.rollout_data.get("static_files", []),
            dynamic_patterns=cfg.rollout_data.get("dynamic_patterns", {}),
            boundary_patterns=cfg.rollout_data.get("boundary_patterns", {}),
            raise_on_smaller=True,
            skip_before_timestep=cfg.source_data.get("skip_before_timestep", 0),
        )
        log_rank_zero.info(f"Loaded {len(rollout_test_dataset)} rollout test samples")

        # Collect and normalize rollout data
        (
            rollout_geom,
            rollout_static,
            rollout_boundary,
            rollout_dyn,
            _,
            rollout_cell_area,
        ) = collect_all_fields(rollout_test_dataset, expect_target=False)

        # Move normalizers to CPU for data transformation
        for norm in normalizers.values():
            norm.to("cpu")

        transformed_rollout = transform_with_existing_normalizers(
            rollout_geom, rollout_static, rollout_boundary, rollout_dyn, normalizers
        )

        normalized_rollout_samples = [
            {
                "run_id": rollout_test_dataset.valid_run_ids[i],
                "geometry": transformed_rollout["geometry"][i],
                "static": transformed_rollout["static"][i],
                "boundary": transformed_rollout["boundary"][i],
                "dynamic": transformed_rollout["dynamic"][i],
                "cell_area": rollout_cell_area[i],
            }
            for i in range(len(rollout_test_dataset))
        ]

        # Run rollout prediction
        log_section(log_rank_zero, "Running Rollout Prediction")
        rollout_prediction(
            model=model,
            rollout_dataset=NormalizedRolloutTestDataset(normalized_rollout_samples, cfg.source_data.query_res),
            rollout_length=cfg.source_data.rollout_length,
            history_steps=cfg.source_data.n_history,
            dynamic_norm=normalizers["dynamic"],
            target_norm=normalizers["target"],
            boundary_norm=normalizers["boundary"],
            device=device,
            skip_before_timestep=cfg.source_data.get("skip_before_timestep", 0),
            dt=cfg.source_data.dt,
            out_dir=cfg.rollout.out_dir,
            logger=log_rank_zero,
        )

        log_section(log_rank_zero, "Inference Complete!")

    except KeyboardInterrupt:
        log_rank_zero.warning("Inference interrupted by user")
        sys.exit(1)
    except Exception as e:
        import traceback
        log_rank_zero.error(f"Fatal error in inference pipeline: {e}")
        log_rank_zero.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    run_inference()

