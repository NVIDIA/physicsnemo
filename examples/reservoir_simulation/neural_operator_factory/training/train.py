#!/usr/bin/env python3
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

"""Training script for neural operator reservoir simulation models."""

import sys
from pathlib import Path

# Add parent directory (neural_operator_factory/) to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from models.xdeeponet import DeepONet3DWrapper, DeepONetWrapper
from models.xfno import FNO4DNet, UFNONet
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from utils.checkpoint import load_checkpoint, save_checkpoint


def print_model_architecture(model, model_type: str, dimensions: str, cfg, logger):
    """Print detailed model architecture for any model type."""
    logger.info("=" * 80)
    logger.info("MODEL ARCHITECTURE")
    logger.info("=" * 80)

    # Get the actual model (unwrap DDP if needed)
    if hasattr(model, "module"):
        actual_model = model.module
    else:
        actual_model = model

    # Print model type and dimensions
    logger.info(f"Dimensions: {dimensions.upper()}")
    logger.info(f"Model Type: {model_type.upper()}")

    if model_type == "xdeeponet":
        variant = cfg.arch.xdeeponet.get("variant", "u_deeponet")
        logger.info(f"Variant: {variant}")
        logger.info("")

        # Branch configuration
        branch1_cfg = cfg.arch.xdeeponet.get("branch1", {})
        b1_enc = branch1_cfg.get("encoder", {})
        b1_layers = branch1_cfg.get("layers", {})
        b1_enc_type = (
            b1_enc.get("type", "linear") if isinstance(b1_enc, dict) else b1_enc
        )
        logger.info("Branch 1:")
        logger.info(f"  Encoder: {b1_enc_type}")
        logger.info("  In Channels: auto (inferred from input tensor)")
        logger.info(f"  Fourier Layers: {b1_layers.get('num_fourier_layers', 0)}")
        logger.info(f"  UNet Layers: {b1_layers.get('num_unet_layers', 0)}")
        logger.info(f"  Conv Layers: {b1_layers.get('num_conv_layers', 0)}")
        logger.info(f"  Layer Activation: {b1_layers.get('activation_fn', 'sin')}")

        if variant in ["mionet", "fourier_mionet", "tno"]:
            branch2_cfg = cfg.arch.xdeeponet.get("branch2", {})
            b2_enc = branch2_cfg.get("encoder", {})
            b2_layers = branch2_cfg.get("layers", {})
            b2_enc_type = (
                b2_enc.get("type", "linear") if isinstance(b2_enc, dict) else b2_enc
            )
            logger.info("Branch 2:")
            logger.info(f"  Encoder: {b2_enc_type}")
            logger.info(f"  Fourier Layers: {b2_layers.get('num_fourier_layers', 0)}")
            logger.info(f"  UNet Layers: {b2_layers.get('num_unet_layers', 0)}")
            logger.info(f"  Layer Activation: {b2_layers.get('activation_fn', 'sin')}")

        # Trunk configuration
        trunk_cfg = cfg.arch.xdeeponet.get("trunk", {})
        trunk_input = trunk_cfg.get("input_type", "time")
        in_features = (4 if dimensions == "4d" else 3) if trunk_input == "grid" else 1
        coord_desc = "x,y,z,t" if dimensions == "4d" else "x,y,t"
        logger.info("Trunk:")
        logger.info(
            f"  Input Type: {trunk_input} ({coord_desc if trunk_input == 'grid' else 'just t'})"
        )
        logger.info(f"  In Features: {in_features}")
        logger.info(f"  Hidden Width: {trunk_cfg.get('hidden_width', 128)}")
        logger.info(f"  Num Layers: {trunk_cfg.get('num_layers', 6)}")
        logger.info(f"  Activation: {trunk_cfg.get('activation_fn', 'sin')}")

        # Decoder configuration
        logger.info("Decoder:")
        logger.info(f"  Type: {cfg.arch.xdeeponet.get('decoder_type', 'mlp')}")
        logger.info(f"  Width: {cfg.arch.xdeeponet.get('decoder_width', 128)}")
        logger.info(f"  Layers: {cfg.arch.xdeeponet.get('decoder_layers', 2)}")
        logger.info(
            f"  Activation: {cfg.arch.xdeeponet.get('decoder_activation_fn', 'relu')}"
        )

        logger.info(f"Latent Width: {cfg.arch.xdeeponet.get('width', 64)}")
        logger.info(f"Padding: {cfg.arch.xdeeponet.get('padding', 8)}")

    elif model_type == "xfno":
        xfno_cfg = cfg.arch.xfno
        logger.info(f"Out Channels: {xfno_cfg.out_channels}")
        logger.info(f"Width: {xfno_cfg.width}")
        if dimensions == "4d":
            logger.info(
                f"Modes: ({xfno_cfg.modes1}, {xfno_cfg.modes2}, {xfno_cfg.modes3}, {xfno_cfg.modes4})"
            )
        else:
            logger.info(
                f"Modes: ({xfno_cfg.modes1}, {xfno_cfg.modes2}, {xfno_cfg.modes3})"
            )
        logger.info(f"FNO Layers: {xfno_cfg.num_fno_layers}")
        if dimensions == "3d":
            logger.info(f"U-Net Layers: {xfno_cfg.num_unet_layers}")
            logger.info(f"Conv Layers: {xfno_cfg.num_conv_layers}")
            logger.info(
                f"Lifting: type={xfno_cfg.lifting_type}, layers={xfno_cfg.lifting_layers}"
            )
        else:
            logger.info(f"Coord Features: {xfno_cfg.coord_features}")
        logger.info(f"Activation: {xfno_cfg.activation_fn}")
        logger.info(
            f"Decoder: layers={xfno_cfg.decoder_layers}, width={xfno_cfg.decoder_width}"
        )

    # Print full model structure
    logger.info("")
    logger.info("Full Model Structure:")
    logger.info("-" * 80)
    for line in str(actual_model).split("\n"):
        logger.info(line)
    logger.info("-" * 80)

    # Count parameters per component
    logger.info("")
    logger.info("Parameter Counts:")
    total_params = 0
    for name, module in actual_model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += params
        logger.info(f"  {name}: {params:,} parameters")
    logger.info(f"  TOTAL: {total_params:,} parameters")
    logger.info("=" * 80)


from data.dataloader import create_dataloaders  # noqa: E402
from data.validation import (  # noqa: E402
    print_validation_summary,
    validate_batch_dimensions,
)
from utils.co2_normalization import dnorm_dP  # noqa: E402

from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.launch.logging import LaunchLogger, PythonLogger  # noqa: E402
from training.ar_utils import (  # noqa: E402
    ar_validate_full_rollout,
    compute_unroll_steps,
    get_training_stage,
    live_rollout_step,
    rollout_step,
    teacher_forcing_step,
)
from training.losses import get_loss_function  # noqa: E402
from training.metrics import (  # noqa: E402
    compute_relative_l2_error,
    mean_absolute_error,
    mean_plume_error,
    mean_relative_error,
)

# Registry of denormalization functions that can be selected via config.
_DENORM_REGISTRY = {
    "dnorm_dP": dnorm_dP,
}


def _get_batch_mask(inputs, mask_channel, mask_per_sample, static_mask):
    """Construct the spatial mask for the current batch.

    Works for any structured-grid dataset (3D or 4D).  When the mask
    is static (identical across all samples), returns ``(*spatial)``.
    When it varies per sample, returns ``(B, *spatial)`` so each
    sample's loss is computed only on its own active cells.
    Returns *None* when no mask channel is available.
    """
    if mask_channel is None:
        return None
    if not mask_per_sample:
        return static_mask
    # Per-sample: inputs shape is (B, *spatial, T, C).
    # Returns (B, *spatial) boolean mask.
    return inputs[..., 0, mask_channel] != 0


# Registry of validation metric functions (numpy-based, operate on flat arrays).
def _rmse_np(y_pred, y_true):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


_METRIC_REGISTRY = {
    "rmse": ("RMSE", "val_rmse", _rmse_np),
    "mae": ("MAE", "val_mae", mean_absolute_error),
    "mre": ("MRE", "val_mre", mean_relative_error),
    "mpe": ("MPE", "val_mpe", mean_plume_error),
    "relative_l2": ("RelL2", "val_relative_l2", compute_relative_l2_error),
}


def _live_rollout_ddp_safe(
    model, dist, inputs, targets, loss_fn, ar_common, max_steps=None
):
    """Run live_rollout_step with correct DDP gradient synchronization.

    live_rollout_step performs multiple forward passes with a single backward,
    which conflicts with DDP's per-forward gradient hooks.  This wrapper
    disables DDP sync during the step, then manually AllReduces gradients
    so all GPUs apply identical weight updates.
    """
    kwargs = (
        dict(ar_common, max_steps=max_steps)
        if max_steps is not None
        else dict(ar_common)
    )

    if isinstance(model, DDP):
        with model.no_sync():
            loss = live_rollout_step(model, inputs, targets, loss_fn, **kwargs)
        for param in model.parameters():
            if param.grad is not None:
                torch.distributed.all_reduce(
                    param.grad, op=torch.distributed.ReduceOp.SUM
                )
                param.grad /= dist.world_size
    else:
        loss = live_rollout_step(model, inputs, targets, loss_fn, **kwargs)

    return loss


@hydra.main(version_base="1.3", config_path="../conf", config_name="training_config")
def main(cfg: DictConfig) -> None:
    """Main training function for neural operator reservoir simulation models."""

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
    logger = PythonLogger(name="nof_train")
    logger.file_logging()
    LaunchLogger.initialize()

    # Print header (only on rank 0)
    if dist.rank == 0:
        dimensions = cfg.arch.dimensions.lower()
        model_type = cfg.arch.model.lower()
        if model_type == "xdeeponet":
            model_name = cfg.arch.xdeeponet.variant.replace("_", "-").upper()
        else:
            model_name = model_type.upper()
        logger.info("=" * 80)
        logger.info(
            f"{model_name} ({dimensions.upper()}) Training | Variable: {cfg.data.variable} | GPUs: {dist.world_size}"
        )
        logger.info("=" * 80)

    # Auto-set num_workers based on num_gpus if not specified
    num_workers = cfg.data.num_workers
    if num_workers is None and hasattr(cfg.compute, "num_gpus"):
        num_workers = cfg.compute.num_gpus * 2  # 2 workers per GPU
    elif num_workers is None:
        num_workers = 4  # Default fallback

    # Get dimensions from config (used for model selection and data validation)
    expected_dimensions = cfg.arch.dimensions.lower()

    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=cfg.data.data_path,
        batch_size=cfg.training.batch_size,
        normalize=cfg.data.normalize,
        num_workers=num_workers,
        device=dist.device,
        input_file=cfg.data.get("input_file", None),
        output_file=cfg.data.get("output_file", None),
        variable=cfg.data.get("variable", None),
        expected_dimensions=expected_dimensions,
        use_mask=cfg.data.get("mask_enabled", False),
        mask_channel=cfg.data.get("mask_channel", None),
        num_timesteps=cfg.data.get("num_timesteps", None),
    )

    # Masking metadata from dataset
    ds = train_loader.dataset
    mask_channel = getattr(ds, "mask_channel", None)
    mask_per_sample = getattr(ds, "mask_per_sample", False)
    static_mask = ds.get_static_mask()
    if static_mask is not None:
        static_mask = static_mask.to(dist.device)

    # Detect variants with branch2
    regime = cfg.training.get("regime", "full_mapping").lower()
    _variant = (
        cfg.arch.xdeeponet.get("variant", "")
        if cfg.arch.model.lower() == "xdeeponet"
        else ""
    )
    is_tno = _variant == "tno"
    has_branch2 = _variant in ("mionet", "fourier_mionet", "tno")
    if is_tno:
        if regime != "autoregressive":
            raise ValueError("TNO variant requires regime: autoregressive")
        if dist.rank == 0:
            logger.info("TNO mode: branch2 receives previous solution state")
    elif has_branch2 and dist.rank == 0:
        logger.info(f"MIONet mode ({_variant}): branch2 processes scalar inputs")

    # Print data info (only on rank 0)
    if dist.rank == 0:
        effective_batch_size = cfg.training.batch_size * dist.world_size
        logger.info(
            f"Data: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)} | Batch size={cfg.training.batch_size} per GPU (Effective: {effective_batch_size})"
        )

    # Validate data dimensions against config
    if dist.rank == 0:
        logger.info("Validating data dimensions...")

        # Get a sample batch to check dimensions
        sample_inputs, sample_targets = next(iter(train_loader))

        # Validate using centralized validation function
        validation_info = validate_batch_dimensions(
            sample_inputs, sample_targets, cfg.data.get("variable", "unknown")
        )
        detected_dimensions = validation_info["dimensions"]

        # Check that detected dimensions match config
        if detected_dimensions != expected_dimensions:
            raise ValueError(
                f"❌ Dimension mismatch! Config specifies '{expected_dimensions}' but data is '{detected_dimensions}'.\n"
                f"   Config: arch.dimensions = {expected_dimensions}\n"
                f"   Data: Input shape {tuple(sample_inputs.shape)} → {detected_dimensions}\n"
                f"   Please update arch.dimensions in config to match your dataset."
            )

        # Print validation summary
        print_validation_summary(
            input_shape=tuple(sample_inputs.shape),
            target_shape=tuple(sample_targets.shape),
            variable=cfg.data.get("variable", "unknown"),
            is_batch=True,
            logger=logger,
        )

    # Create model based on dimensions and model type
    dimensions = cfg.arch.dimensions.lower()
    model_type = cfg.arch.model.lower()

    # Get in_channels from first batch (for auto-discovery)
    sample_inputs, _ = next(iter(train_loader))
    in_channels = sample_inputs.shape[-1]  # Last dimension is channels

    # Account for feedback channel (appended during AR training)
    _ar_feedback_init = cfg.training.get(
        "regime", "full_mapping"
    ).lower() == "autoregressive" and cfg.training.autoregressive.get(
        "use_feedback_channel", False
    )
    if _ar_feedback_init:
        in_channels += 1

    if model_type == "xfno":
        xfno_cfg = cfg.arch.xfno

        if dimensions == "4d":
            # 4D FNO (3D spatial + time) - Pure FNO only
            logger.info(
                f"Creating FNO4D model (FNO layers: {xfno_cfg.num_fno_layers}, "
                f"modes: [{xfno_cfg.modes1}, {xfno_cfg.modes2}, {xfno_cfg.modes3}, {xfno_cfg.modes4}])"
            )
            model = FNO4DNet(
                in_channels=in_channels,
                out_channels=xfno_cfg.out_channels,
                width=xfno_cfg.width,
                modes1=xfno_cfg.modes1,
                modes2=xfno_cfg.modes2,
                modes3=xfno_cfg.modes3,
                modes4=xfno_cfg.modes4,
                num_fno_layers=xfno_cfg.num_fno_layers,
                padding=xfno_cfg.padding,
                activation_fn=xfno_cfg.activation_fn,
                lifting_layers=xfno_cfg.lifting_layers,
                decoder_layers=xfno_cfg.decoder_layers,
                decoder_width=xfno_cfg.decoder_width,
                coord_features=xfno_cfg.coord_features,
            ).to(dist.device)
            model_arch_name = "fno4d"
        else:
            # 3D FNO (2D spatial + time) - With optional U-Net/Conv
            num_unet = xfno_cfg.num_unet_layers
            num_conv = xfno_cfg.num_conv_layers

            if num_unet > 0 and num_conv > 0:
                logger.warning("⚠️  Using both U-Net and Conv layers (Conv-U-FNO).")
                model_arch_name = f"convufno_{xfno_cfg.unet_type}"
            elif num_unet > 0:
                model_arch_name = f"ufno_{xfno_cfg.unet_type}"
            elif num_conv > 0:
                model_arch_name = "convfno"
            else:
                model_arch_name = "fno"

            logger.info(
                f"Creating {model_arch_name.upper()} model (FNO: {xfno_cfg.num_fno_layers}, "
                f"U-Net: {num_unet}, Conv: {num_conv})"
            )

            model = UFNONet(
                in_channels=in_channels,
                out_channels=xfno_cfg.out_channels,
                width=xfno_cfg.width,
                modes1=xfno_cfg.modes1,
                modes2=xfno_cfg.modes2,
                modes3=xfno_cfg.modes3,
                num_fno_layers=xfno_cfg.num_fno_layers,
                num_unet_layers=num_unet,
                num_conv_layers=num_conv,
                padding=xfno_cfg.padding,
                conv_kernel_size=xfno_cfg.conv_kernel_size,
                unet_kernel_size=xfno_cfg.unet_kernel_size,
                unet_dropout=xfno_cfg.unet_dropout,
                unet_type=xfno_cfg.unet_type,
                activation_fn=xfno_cfg.activation_fn,
                lifting_type=xfno_cfg.lifting_type,
                lifting_layers=xfno_cfg.lifting_layers,
                lifting_width=xfno_cfg.lifting_width,
                decoder_type=xfno_cfg.decoder_type,
                decoder_layers=xfno_cfg.decoder_layers,
                decoder_width=xfno_cfg.decoder_width,
                decoder_activation_fn=xfno_cfg.get("decoder_activation_fn", None),
            ).to(dist.device)

    elif model_type == "xdeeponet":
        xdeeponet_cfg = cfg.arch.xdeeponet
        variant = xdeeponet_cfg.variant

        # Build branch configs from yaml
        branch1_config = dict(xdeeponet_cfg.branch1)
        branch2_config = (
            dict(xdeeponet_cfg.branch2)
            if variant in ["mionet", "fourier_mionet", "tno"]
            else None
        )
        trunk_config = dict(xdeeponet_cfg.trunk)

        if dimensions == "4d":
            # 4D DeepONet (3D spatial + time)
            logger.info(
                f"Creating DeepONet3D model (variant: {variant}, "
                f"branch1: {branch1_config.get('encoder', 'spatial')}, width: {xdeeponet_cfg.width})"
            )
            model = DeepONet3DWrapper(
                padding=xdeeponet_cfg.padding,
                variant=variant,
                width=xdeeponet_cfg.width,
                branch1_config=branch1_config,
                branch2_config=branch2_config,
                trunk_config=trunk_config,
                decoder_type=xdeeponet_cfg.get("decoder_type", "mlp"),
                decoder_width=xdeeponet_cfg.decoder_width,
                decoder_layers=xdeeponet_cfg.decoder_layers,
                decoder_activation_fn=xdeeponet_cfg.get(
                    "decoder_activation_fn", "relu"
                ),
            ).to(dist.device)
            b1_enc = branch1_config.get("encoder", "spatial")
            b1_enc_name = (
                b1_enc.get("type", "linear") if not isinstance(b1_enc, str) else b1_enc
            )
            model_arch_name = f"deeponet3d_{variant}_{b1_enc_name}"
        else:
            # 3D DeepONet (2D spatial + time)
            logger.info(
                f"Creating DeepONet model (variant: {variant}, "
                f"branch1: {branch1_config.get('encoder', 'spatial')}, width: {xdeeponet_cfg.width})"
            )
            model = DeepONetWrapper(
                padding=xdeeponet_cfg.padding,
                variant=variant,
                width=xdeeponet_cfg.width,
                branch1_config=branch1_config,
                branch2_config=branch2_config,
                trunk_config=trunk_config,
                decoder_type=xdeeponet_cfg.get("decoder_type", "mlp"),
                decoder_width=xdeeponet_cfg.decoder_width,
                decoder_layers=xdeeponet_cfg.decoder_layers,
                decoder_activation_fn=xdeeponet_cfg.get(
                    "decoder_activation_fn", "relu"
                ),
            ).to(dist.device)
            b1_enc = branch1_config.get("encoder", "spatial")
            b1_enc_name = (
                b1_enc.get("type", "linear") if not isinstance(b1_enc, str) else b1_enc
            )
            model_arch_name = f"deeponet_{variant}_{b1_enc_name}"

    else:
        raise ValueError(f"Unknown model: {model_type}. Use 'xfno' or 'xdeeponet'.")

    # Set temporal projection output window before any forward pass
    if (
        regime == "autoregressive"
        and hasattr(model, "_temporal_projection")
        and model._temporal_projection
    ):
        ar_K_init = cfg.training.autoregressive.output_window
        model.set_output_window(ar_K_init)
        if dist.rank == 0:
            logger.info(f"Temporal projection decoder: output window K={ar_K_init}")

    # Initialize lazy modules with a dummy forward pass (required for DDP)
    # This is needed because nn.LazyLinear doesn't know its input size until first forward
    _ar_feedback_init = regime == "autoregressive" and cfg.training.autoregressive.get(
        "use_feedback_channel", False
    )
    if dist.rank == 0:
        logger.info("Initializing model with dummy forward pass...")
    with torch.no_grad():
        dummy_batch = next(iter(train_loader))
        dummy_input = dummy_batch[0].to(dist.device)
        if _ar_feedback_init:
            dummy_fb = torch.zeros_like(dummy_input[..., :1])
            dummy_input = torch.cat([dummy_input, dummy_fb], dim=-1)
        if is_tno:
            dummy_target = dummy_batch[1].to(dist.device)
            _L = cfg.training.autoregressive.input_window
            dummy_b2 = dummy_target[..., :_L]
            _ = model(dummy_input, x_branch2=dummy_b2)
        elif has_branch2:
            dummy_b2 = dummy_input[:, 0, 0, 0, :]
            _ = model(dummy_input, x_branch2=dummy_b2)
        else:
            _ = model(dummy_input)
    if dist.rank == 0:
        logger.info("Model initialization complete.")

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
        # Print detailed model architecture
        print_model_architecture(model, model_type, dimensions, cfg, logger)

    # Create training loss function
    loss_fn = get_loss_function(cfg.loss, variable=cfg.data.get("variable", None))

    # Create validation loss function (same as training loss for fair comparison)
    from omegaconf import DictConfig

    # Validation loss: same base losses, no derivatives, no physics losses
    val_loss_cfg = DictConfig(
        {
            "types": list(cfg.loss.types),
            "weights": list(cfg.loss.weights),
            "reduction": cfg.loss.get("reduction", "mean"),
        }
    )
    val_loss_fn = get_loss_function(val_loss_cfg)

    # Print loss info (only on rank 0)
    if dist.rank == 0:
        types_str = "+".join(
            f"{w}*{t}" for t, w in zip(cfg.loss.types, cfg.loss.weights)
        )
        loss_info = f"Train Loss: {types_str}"
        if cfg.loss.get("derivative", {}).get("enabled", False):
            loss_info += f" (+Derivative w={cfg.loss.derivative.weight}, dims={list(cfg.loss.derivative.dims)})"
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
        mlflow_params = {
            "dimensions": dimensions,
            "model_type": model_type,
            "model_arch_name": model_arch_name,
            "batch_size": cfg.training.batch_size,
            "epochs": cfg.training.epochs,
            "learning_rate": cfg.training.initial_lr,
            "optimizer": "Adam",
            "train_loss": "+".join(list(cfg.loss.types)),
            "loss_masking": cfg.data.get("mask_enabled", False),
            "loss_derivative": cfg.loss.get("derivative", {}).get("enabled", False),
            "use_amp": cfg.training.use_amp,
            "use_graphs": cfg.training.use_graphs,
            "variable": cfg.data.variable,
            "trainable_parameters": trainable_params,
            "in_channels": in_channels,
        }

        if model_type == "xfno":
            xfno_cfg = cfg.arch.xfno
            mlflow_params.update(
                {
                    "width": xfno_cfg.width,
                    "modes1": xfno_cfg.modes1,
                    "modes2": xfno_cfg.modes2,
                    "modes3": xfno_cfg.modes3,
                    "num_fno_layers": xfno_cfg.num_fno_layers,
                    "padding": xfno_cfg.padding,
                    "activation_fn": xfno_cfg.activation_fn,
                }
            )
            if dimensions == "4d":
                mlflow_params["modes4"] = xfno_cfg.modes4
            else:
                mlflow_params["num_unet_layers"] = xfno_cfg.num_unet_layers
                mlflow_params["num_conv_layers"] = xfno_cfg.num_conv_layers
        elif model_type == "xdeeponet":
            xdeeponet_cfg = cfg.arch.xdeeponet
            mlflow_params.update(
                {
                    "variant": xdeeponet_cfg.variant,
                    "width": xdeeponet_cfg.width,
                    "padding": xdeeponet_cfg.padding,
                    "branch1_encoder": xdeeponet_cfg.branch1.get("encoder", {}).get(
                        "type", "linear"
                    )
                    if isinstance(xdeeponet_cfg.branch1.get("encoder"), dict)
                    else xdeeponet_cfg.branch1.get("encoder", "spatial"),
                }
            )

        mlflow.log_params(mlflow_params)

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
    _resume_ckpt = None

    if (
        hasattr(cfg.training, "resume_from_checkpoint")
        and cfg.training.resume_from_checkpoint
    ):
        checkpoint_path = Path(cfg.training.resume_from_checkpoint)
        if checkpoint_path.exists():
            if dist.rank == 0:
                logger.info(f"Loading checkpoint from: {checkpoint_path}")
            _resume_ckpt = load_checkpoint(checkpoint_path, device=dist.device)

            # Validate that the checkpoint architecture matches the current model
            _resume_ckpt.get("model_config", {})

            model_to_load = model.module if isinstance(model, DDP) else model
            model_to_load.load_state_dict(_resume_ckpt["model_state_dict"])

            start_epoch = _resume_ckpt["epoch"] + 1
            best_val_loss = _resume_ckpt.get("val_loss", float("inf"))
            best_val_mre = _resume_ckpt.get("val_mre", float("inf"))

            if dist.rank == 0:
                logger.success(
                    f"Resumed from epoch {_resume_ckpt['epoch']}, "
                    f"best val loss: {best_val_loss:.6f}"
                )
        else:
            if dist.rank == 0:
                logger.warning(
                    f"Checkpoint not found at {checkpoint_path}, starting from scratch"
                )
    else:
        if dist.rank == 0:
            logger.success("Starting training from scratch...")

    # Restore optimizer and scheduler state if resuming
    if _resume_ckpt is not None:
        if "optimizer_state_dict" in _resume_ckpt:
            optimizer.load_state_dict(_resume_ckpt["optimizer_state_dict"])
            if dist.rank == 0:
                logger.info(
                    f"Restored optimizer state (LR={optimizer.param_groups[0]['lr']:.2e})"
                )
        if "scheduler_state_dict" in _resume_ckpt:
            scheduler.load_state_dict(_resume_ckpt["scheduler_state_dict"])
            if dist.rank == 0:
                logger.info("Restored scheduler state")
        del _resume_ckpt  # free memory

    # ---------------------------------------------------------------------------
    # Determine training regime
    # ---------------------------------------------------------------------------
    regime = cfg.training.get("regime", "full_mapping").lower()
    if regime == "autoregressive":
        ar_cfg = cfg.training.autoregressive
        ar_L = ar_cfg.input_window
        ar_K = ar_cfg.output_window
        ar_stride = ar_cfg.get("stride", None)
        tf_epochs = ar_cfg.teacher_forcing_epochs
        pf_epochs = ar_cfg.get("pushforward_epochs", 0)
        ro_epochs = ar_cfg.get("rollout_epochs", 0)
        total_epochs = tf_epochs + pf_epochs + ro_epochs
        ar_noise_std = ar_cfg.get("noise_std", 0.0)
        ar_feedback = ar_cfg.get("use_feedback_channel", False)
        ar_max_unroll = ar_cfg.get(
            "max_unroll", ar_cfg.get("pushforward_max_unroll", 5)
        )
        ar_lr_reset = ar_cfg.get("lr_reset_factor", 1.0)
        ar_rollout_mode = ar_cfg.get("rollout_mode", "detached").lower()

        if dist.rank == 0:
            logger.info("=" * 80)
            logger.info(f"AUTOREGRESSIVE TRAINING | L={ar_L}, K={ar_K}")
            logger.info(f"  Stage 1 — Teacher Forcing:  {tf_epochs} epochs")
            if pf_epochs > 0:
                logger.info(
                    f"  Stage 2 — Pushforward:      {pf_epochs} epochs (unroll 1 -> {ar_max_unroll})"
                )
            logger.info(
                f"  Stage 3 — Rollout:          {ro_epochs} epochs ({ar_rollout_mode})"
            )
            logger.info(f"  Total: {total_epochs} epochs")
            if ar_noise_std > 0:
                logger.info(f"  Noise: std={ar_noise_std}")
            if ar_feedback:
                logger.info("  Feedback channel: enabled")
            if not is_tno and not ar_feedback:
                logger.warning(
                    "Autoregressive training without TNO or feedback channel: "
                    "the model will not receive its own predictions as input. "
                    "Set autoregressive.use_feedback_channel: true for real AR feedback."
                )
            logger.info("=" * 80)
    else:
        total_epochs = cfg.training.epochs
        if dist.rank == 0:
            logger.info("=" * 80)
            logger.info("FULL-MAPPING TRAINING")
            logger.info(f"  Epochs: {total_epochs}")
            logger.info("=" * 80)

    # Training loop
    for epoch in range(start_epoch, total_epochs + 1):
        # Set epoch for distributed sampler (ensures proper shuffling across epochs)
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # Training step
        with LaunchLogger(
            "train", epoch=epoch, num_mini_batch=len(train_loader)
        ) as log:
            model.train()
            total_loss = 0.0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(dist.device)
                targets = targets.to(dist.device)
                optimizer.zero_grad()

                batch_mask = _get_batch_mask(
                    inputs, mask_channel, mask_per_sample, static_mask
                )

                if regime == "autoregressive":
                    stage = get_training_stage(epoch, tf_epochs, pf_epochs, ro_epochs)
                    ar_common = dict(
                        L=ar_L,
                        K=ar_K,
                        spatial_mask=batch_mask,
                        is_tno=is_tno,
                        noise_std=ar_noise_std,
                        feedback_channel=1 if ar_feedback else None,
                        stride=ar_stride,
                    )
                    if stage == "teacher_forcing":
                        loss = teacher_forcing_step(
                            model,
                            inputs,
                            targets,
                            loss_fn,
                            **ar_common,
                        )
                    elif stage == "pushforward":
                        unroll = compute_unroll_steps(
                            epoch,
                            tf_epochs + 1,
                            pf_epochs,
                            ar_max_unroll,
                        )
                        loss = _live_rollout_ddp_safe(
                            model,
                            dist,
                            inputs,
                            targets,
                            loss_fn,
                            ar_common,
                            max_steps=unroll,
                        )
                    else:
                        if ar_rollout_mode == "live_gradients":
                            loss = _live_rollout_ddp_safe(
                                model,
                                dist,
                                inputs,
                                targets,
                                loss_fn,
                                ar_common,
                            )
                        else:
                            loss = rollout_step(
                                model,
                                inputs,
                                targets,
                                loss_fn,
                                **ar_common,
                            )
                else:
                    # Full-mapping: single forward pass over entire trajectory
                    fwd_kwargs = {}
                    if has_branch2 and not is_tno:
                        fwd_kwargs["x_branch2"] = inputs[:, 0, 0, 0, :]

                    if cfg.training.use_amp:
                        with autocast():
                            pred = model(inputs, **fwd_kwargs)
                            loss = loss_fn(
                                pred, targets, inputs, spatial_mask=batch_mask
                            )
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        if dist.world_size > 1:
                            loss_tensor = loss.detach().clone()
                            torch.distributed.all_reduce(
                                loss_tensor, op=torch.distributed.ReduceOp.SUM
                            )
                            total_loss += loss_tensor / dist.world_size
                        else:
                            total_loss += loss.detach()
                        continue
                    else:
                        pred = model(inputs, **fwd_kwargs)
                        loss = loss_fn(pred, targets, inputs, spatial_mask=batch_mask)

                # Backward + step
                if regime == "autoregressive":
                    # All AR step functions call backward() internally
                    # and return detached scalars for logging.
                    optimizer.step()
                else:
                    loss.backward()
                    optimizer.step()

                if dist.world_size > 1:
                    loss_tensor = loss.detach().clone()
                    torch.distributed.all_reduce(
                        loss_tensor, op=torch.distributed.ReduceOp.SUM
                    )
                    total_loss += loss_tensor / dist.world_size
                else:
                    total_loss += loss.detach()

            avg_train_loss = total_loss / len(train_loader)

            # Log stage transitions and LR reset for AR
            if regime == "autoregressive":
                stage = get_training_stage(epoch, tf_epochs, pf_epochs, ro_epochs)
                prev_stage = (
                    get_training_stage(epoch - 1, tf_epochs, pf_epochs, ro_epochs)
                    if epoch > 1
                    else None
                )
                if prev_stage is not None and stage != prev_stage:
                    if ar_lr_reset != 1.0:
                        for pg in optimizer.param_groups:
                            pg["lr"] *= ar_lr_reset
                    if dist.rank == 0:
                        new_lr = optimizer.param_groups[0]["lr"]
                        logger.info("=" * 60)
                        logger.info(
                            f"STAGE TRANSITION: {prev_stage.upper().replace('_', ' ')} -> {stage.upper().replace('_', ' ')} (LR={new_lr:.2e})"
                        )
                        if stage == "pushforward":
                            logger.info(
                                f"  Pushforward curriculum: unroll 1 -> {ar_max_unroll} over {pf_epochs} epochs"
                            )
                        logger.info("=" * 60)

            log.log_epoch({"loss": avg_train_loss})

            if cfg.logging.use_mlflow and dist.rank == 0:
                mlflow.log_metric("train_loss", float(avg_train_loss), step=epoch)

        # Validation step
        if epoch % cfg.training.validate_freq == 0:
            with LaunchLogger("valid", epoch=epoch) as log:
                model.eval()
                total_val_loss = 0.0
                mre_list = []

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(dist.device)
                        targets = targets.to(dist.device)

                        val_batch_mask = _get_batch_mask(
                            inputs, mask_channel, mask_per_sample, static_mask
                        )

                        # Forward pass — same regime as training
                        if regime == "autoregressive":
                            pred = ar_validate_full_rollout(
                                model,
                                inputs,
                                targets,
                                L=ar_L,
                                K=ar_K,
                                is_tno=is_tno,
                                feedback_channel=1 if ar_feedback else None,
                            )
                        else:
                            val_fwd = {}
                            if has_branch2 and not is_tno:
                                val_fwd["x_branch2"] = inputs[:, 0, 0, 0, :]
                            pred = model(inputs, **val_fwd)

                        if cfg.training.use_amp:
                            with autocast():
                                val_loss = val_loss_fn(
                                    pred, targets, inputs, spatial_mask=val_batch_mask
                                )
                        else:
                            val_loss = val_loss_fn(
                                pred, targets, inputs, spatial_mask=val_batch_mask
                            )

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

                        # Calculate validation metric on rank 0 only (for logging)
                        if dist.rank == 0:
                            pred_cpu = pred.cpu().numpy()
                            targets_cpu = targets.cpu().numpy()
                            inputs_cpu = inputs.cpu().numpy()

                            # Optional denormalization (config-driven)
                            denorm_name = cfg.data.get("denormalize_fn", None)
                            if denorm_name and denorm_name in _DENORM_REGISTRY:
                                denorm_fn = _DENORM_REGISTRY[denorm_name]
                                pred_denorm = denorm_fn(pred_cpu)
                                targets_denorm = denorm_fn(targets_cpu)
                            else:
                                pred_denorm = pred_cpu
                                targets_denorm = targets_cpu

                            # Resolve metric function from config
                            val_metric_choice = cfg.data.get("val_metric", "rmse")
                            if val_metric_choice not in _METRIC_REGISTRY:
                                raise ValueError(
                                    f"Unknown val_metric '{val_metric_choice}'. "
                                    f"Choices: {list(_METRIC_REGISTRY.keys())}"
                                )
                            _, _, metric_fn = _METRIC_REGISTRY[val_metric_choice]

                            for i in range(pred_denorm.shape[0]):
                                if mask_channel is not None:
                                    mask_i = inputs_cpu[i, ..., 0, mask_channel] != 0
                                    y_pred = pred_denorm[i][mask_i]
                                    y_true = targets_denorm[i][mask_i]
                                else:
                                    y_pred = pred_denorm[i].ravel()
                                    y_true = targets_denorm[i].ravel()

                                mre_list.append(metric_fn(y_pred, y_true))

                avg_val_loss = total_val_loss / len(val_loader)
                avg_metric = np.mean(mre_list) if len(mre_list) > 0 else 0.0

                # Metric display name and logging key from config
                val_metric_choice = cfg.data.get("val_metric", "rmse")
                metric_name, metric_key, _ = _METRIC_REGISTRY[val_metric_choice]

                is_ratio_metric = val_metric_choice in ("mre", "mpe", "relative_l2")

                # Print validation metrics (only on rank 0)
                if dist.rank == 0:
                    if is_ratio_metric:
                        logger.info(
                            f"Epoch {epoch}: Val Loss = {avg_val_loss:.6f} | Val {metric_name} = {avg_metric:.6f} ({avg_metric * 100:.2f}%)"
                        )
                    else:
                        logger.info(
                            f"Epoch {epoch}: Val Loss = {avg_val_loss:.6f} | Val {metric_name} = {avg_metric:.6f}"
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
                        if is_ratio_metric:
                            logger.success(
                                f"New best validation: Loss = {best_val_loss:.6f} | {metric_name} = {best_val_mre:.6f} ({best_val_mre * 100:.2f}%)"
                            )
                        else:
                            logger.success(
                                f"New best validation: Loss = {best_val_loss:.6f} | {metric_name} = {best_val_mre:.6f}"
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
                        # Prepare model config to save with checkpoint
                        model_config = {
                            "dimensions": dimensions,
                            "model_type": model_type,
                            "model_arch_name": model_arch_name,
                            "variable": cfg.data.variable,
                            "in_channels": in_channels,
                            "feedback_channel": 1 if _ar_feedback_init else None,
                        }

                        if model_type == "xfno":
                            xfno_cfg = cfg.arch.xfno
                            model_config.update(
                                {
                                    "out_channels": xfno_cfg.out_channels,
                                    "width": xfno_cfg.width,
                                    "modes1": xfno_cfg.modes1,
                                    "modes2": xfno_cfg.modes2,
                                    "modes3": xfno_cfg.modes3,
                                    "num_fno_layers": xfno_cfg.num_fno_layers,
                                    "padding": xfno_cfg.padding,
                                    "activation_fn": xfno_cfg.activation_fn,
                                    "decoder_layers": xfno_cfg.decoder_layers,
                                    "decoder_width": xfno_cfg.decoder_width,
                                }
                            )
                            if dimensions == "4d":
                                model_config.update(
                                    {
                                        "modes4": xfno_cfg.modes4,
                                        "coord_features": xfno_cfg.coord_features,
                                        "lifting_layers": xfno_cfg.lifting_layers,
                                    }
                                )
                            else:
                                model_config.update(
                                    {
                                        "num_unet_layers": xfno_cfg.num_unet_layers,
                                        "num_conv_layers": xfno_cfg.num_conv_layers,
                                        "unet_type": xfno_cfg.unet_type,
                                        "lifting_type": xfno_cfg.lifting_type,
                                        "lifting_layers": xfno_cfg.lifting_layers,
                                        "lifting_width": xfno_cfg.lifting_width,
                                        "decoder_type": xfno_cfg.decoder_type,
                                    }
                                )

                        elif model_type == "xdeeponet":
                            xdeeponet_cfg = cfg.arch.xdeeponet
                            model_config.update(
                                {
                                    "variant": xdeeponet_cfg.variant,
                                    "width": xdeeponet_cfg.width,
                                    "padding": xdeeponet_cfg.padding,
                                    "branch1_config": dict(xdeeponet_cfg.branch1),
                                    "trunk_config": dict(xdeeponet_cfg.trunk),
                                    "decoder_type": xdeeponet_cfg.get(
                                        "decoder_type", "mlp"
                                    ),
                                    "decoder_width": xdeeponet_cfg.decoder_width,
                                    "decoder_layers": xdeeponet_cfg.decoder_layers,
                                    "decoder_activation_fn": xdeeponet_cfg.get(
                                        "decoder_activation_fn", "relu"
                                    ),
                                }
                            )
                            if xdeeponet_cfg.variant in [
                                "mionet",
                                "fourier_mionet",
                                "tno",
                            ]:
                                model_config["branch2_config"] = dict(
                                    xdeeponet_cfg.branch2
                                )
                            if (
                                xdeeponet_cfg.get("decoder_type", "mlp")
                                == "temporal_projection"
                            ):
                                model_config["output_window"] = (
                                    cfg.training.autoregressive.output_window
                                )

                        save_checkpoint(
                            path=best_model_path,
                            model=model,
                            epoch=epoch,
                            val_loss=best_val_loss,
                            metric_key=metric_key,
                            metric_value=best_val_mre,
                            model_config=model_config,
                        )

                        # Log model to MLFlow
                        if cfg.logging.use_mlflow:
                            mlflow.log_artifact(str(best_model_path))

        # Learning rate scheduling (StepLR steps automatically every step_size epochs)
        scheduler.step()

    # Resolve metric metadata once for final summary / MLflow
    val_metric_choice = cfg.data.get("val_metric", "rmse")
    metric_name, metric_key, _ = _METRIC_REGISTRY[val_metric_choice]
    is_ratio_metric = val_metric_choice in ("mre", "mpe", "relative_l2")

    # Print training completion (only on rank 0)
    if dist.rank == 0:
        logger.success("Training completed!")
        if is_ratio_metric:
            logger.info(
                f"Best validation: Loss = {best_val_loss:.6f} | {metric_name} = {best_val_mre:.6f} ({best_val_mre * 100:.2f}%)"
            )
        else:
            logger.info(
                f"Best validation: Loss = {best_val_loss:.6f} | {metric_name} = {best_val_mre:.6f}"
            )

    # End MLFlow run (only on rank 0)
    if cfg.logging.use_mlflow and dist.rank == 0:
        mlflow.log_metric("final_best_val_loss", float(best_val_loss))
        mlflow.log_metric(f"final_best_{metric_key}", float(best_val_mre))
        mlflow.end_run()
        logger.info("MLFlow run completed")


if __name__ == "__main__":
    main()
