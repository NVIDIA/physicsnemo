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

from collections.abc import Sequence
from typing import Optional, Dict, Any, Tuple, List
import os
import time
import numpy as np
import torch
import psutil
from physicsnemo.core import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.diffusion.metrics import EDMLoss, EDMLossLogUniform
from physicsnemo.diffusion.utils import InfiniteSampler

from physicsnemo.utils import save_checkpoint, load_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from utils.nn import (
    diffusion_model_forward,
    regression_loss_fn,
    get_preconditioned_architecture,
    build_network_condition_and_target,
    unpack_batch,
)
from utils.plots import validation_plot
from utils.optimizers import build_optimizer
from utils.schedulers import init_scheduler, step_scheduler
from datasets import dataset_classes
from datasets.dataset import worker_init
import matplotlib.pyplot as plt
import wandb
from utils.spectrum import ps1d_plots
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf


logger = PythonLogger("train")


class Trainer:
    r"""
    StormCast Trainer class.

    Encapsulates all training logic including model and optimizer setup,
    training and validation loops, checkpointing, logging, and validation plotting.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object containing training, model, dataset, and sampler settings.

    Attributes
    ----------
    cfg : DictConfig
        Configuration object.
    dist : DistributedManager
        Distributed training manager.
    device : torch.device
        Device for training (CUDA or CPU).
    net : Module
        The neural network model.
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    scheduler : torch.optim.lr_scheduler._LRScheduler or None
        Learning rate scheduler.
    total_steps : int
        Current training step count.
    val_loss : float
        Latest validation loss.

    Examples
    --------
    >>> from omegaconf import OmegaConf
    >>> cfg = OmegaConf.load("config.yaml")
    >>> trainer = Trainer(cfg)
    >>> trainer.train()
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.start_time = time.time()

        # Distributed setup
        self.dist = DistributedManager()
        self.device = self.dist.device
        self.logger0 = RankZeroLoggingWrapper(logger, self.dist)

        # Parse config
        self._parse_config()

        # Setup seeds and backends
        self._setup_seeds_and_backends()

        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_ddp()

        # Resume or init
        self._resume_or_init()

        # Training state
        self.train_steps = 0
        self.avg_train_loss = 0.0
        self.valid_time = -1.0
        self.wandb_logs = {}

    # =========================================================================
    # Configuration
    # =========================================================================

    def _validate_config(self, cfg):
        r"""
        Validate configuration entries.

        Parameters
        ----------
        cfg : DictConfig
            Configuration object to validate.

        Raises
        ------
        ValueError
            If required config sections or keys are missing or invalid.
        """
        errors = []

        # Required top-level sections
        required_sections = ["training", "model", "dataset"]
        for section in required_sections:
            if not hasattr(cfg, section):
                errors.append(f"Missing required config section: '{section}'")

        if errors:
            raise ValueError(
                f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        # Training config validation
        training_required = [
            "batch_size",
            "total_train_steps",
            "lr",
            "lr_rampup_steps",
            "seed",
            "log_to_wandb",
            "rundir",
            "print_progress_freq",
            "checkpoint_freq",
            "validation_freq",
            "num_data_workers",
        ]
        for key in training_required:
            if not hasattr(cfg.training, key):
                errors.append(f"Missing required training config: '{key}'")

        # Validate training values
        if hasattr(cfg.training, "batch_size") and cfg.training.batch_size <= 0:
            errors.append("training.batch_size must be positive")
        if (
            hasattr(cfg.training, "total_train_steps")
            and cfg.training.total_train_steps <= 0
        ):
            errors.append("training.total_train_steps must be positive")
        if hasattr(cfg.training, "lr") and cfg.training.lr <= 0:
            errors.append("training.lr must be positive")
        if (
            hasattr(cfg.training, "lr_rampup_steps")
            and cfg.training.lr_rampup_steps < 0
        ):
            errors.append("training.lr_rampup_steps must be non-negative")

        # Model config validation
        model_required = ["model_name", "regression_conditions", "diffusion_conditions"]
        for key in model_required:
            if not hasattr(cfg.model, key):
                errors.append(f"Missing required model config: '{key}'")

        if hasattr(cfg.model, "model_name"):
            valid_models = ["regression", "diffusion"]
            if cfg.model.model_name not in valid_models:
                errors.append(
                    f"model.model_name must be one of {valid_models}, got '{cfg.model.model_name}'"
                )

        # Loss config validation
        if hasattr(cfg.training, "loss"):
            loss_cfg = cfg.training.loss
            if hasattr(loss_cfg, "type"):
                valid_loss_types = ["regression", "edm"]
                if loss_cfg.type not in valid_loss_types:
                    errors.append(
                        f"training.loss.type must be one of {valid_loss_types}, got '{loss_cfg.type}'"
                    )

            if hasattr(loss_cfg, "sigma_distribution"):
                valid_dists = ["lognormal", "loguniform"]
                if loss_cfg.sigma_distribution not in valid_dists:
                    errors.append(
                        f"training.loss.sigma_distribution must be one of {valid_dists}"
                    )

            if hasattr(loss_cfg, "sigma_data") and loss_cfg.sigma_data <= 0:
                errors.append("training.loss.sigma_data must be positive")

        # Dataset config validation
        if not hasattr(cfg.dataset, "name"):
            errors.append("Missing required dataset config: 'name'")

        # Performance config validation
        if hasattr(cfg.training, "perf"):
            perf_cfg = cfg.training.perf
            if hasattr(perf_cfg, "fp_optimizations"):
                valid_fp = ["fp32", "amp-fp16", "amp-bf16"]
                if perf_cfg.fp_optimizations not in valid_fp:
                    errors.append(
                        f"training.perf.fp_optimizations must be one of {valid_fp}"
                    )
            # Boolean checks for CUDA backend settings
            for bool_key in [
                "allow_tf32",
                "allow_fp16_reduced_precision",
                "use_apex_gn",
                "torch_compile",
            ]:
                if hasattr(perf_cfg, bool_key) and not isinstance(
                    getattr(perf_cfg, bool_key), bool
                ):
                    errors.append(f"training.perf.{bool_key} must be a boolean")

        # Sampler config validation (for diffusion)
        if hasattr(cfg, "sampler") and hasattr(cfg.sampler, "args"):
            if (
                hasattr(cfg.sampler.args, "num_steps")
                and cfg.sampler.args.num_steps <= 0
            ):
                errors.append("sampler.args.num_steps must be positive")

        if errors:
            raise ValueError(
                f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        self.logger0.info("Configuration validated successfully")

    def _parse_config(self):
        r"""
        Parse and store configuration values.

        Extracts and stores batch sizes, training parameters, validation config,
        model type, performance options, and checkpoint paths from the configuration.
        """
        cfg = self.cfg

        # Validate required config sections
        self._validate_config(cfg)

        # Batch sizes
        self.batch_size = cfg.training.batch_size
        if cfg.training.batch_size_per_gpu == "auto":
            self.local_batch_size = self.batch_size // self.dist.world_size
        else:
            self.local_batch_size = cfg.training.batch_size_per_gpu
        assert self.batch_size % (self.local_batch_size * self.dist.world_size) == 0
        self.num_accumulation_rounds = self.batch_size // (
            self.local_batch_size * self.dist.world_size
        )

        # Training params
        self.total_train_steps = cfg.training.total_train_steps
        self.warmup_steps = cfg.training.lr_rampup_steps
        self.log_to_wandb = cfg.training.log_to_wandb
        self.log_to_tensorboard = cfg.training.get("log_to_tensorboard", False)

        # TensorBoard writer (rank 0 only)
        self.writer = None
        if self.log_to_tensorboard and self.dist.rank == 0:
            tb_dir = os.path.join(cfg.training.rundir, "tensorboard")
            os.makedirs(tb_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_dir)
            self.logger0.info(f"TensorBoard logging enabled: {tb_dir}")

        # Validation config
        self.validation_steps = cfg.training.get("validation_steps", 1)
        self.validation_bg_channels = (
            cfg.training.get("validation_plot_background_variables", []) or []
        )

        # Model type
        self.loss_type = cfg.training.loss.type
        self.net_name = "regression" if self.loss_type == "regression" else "diffusion"
        self.condition_list = (
            cfg.model.regression_conditions
            if self.net_name == "regression"
            else cfg.model.diffusion_conditions
        )

        # Performance options
        self._parse_perf_config()

        # Paths
        self.ckpt_path = os.path.join(
            cfg.training.rundir, f"checkpoints_{self.net_name}"
        )

    def _parse_perf_config(self):
        r"""
        Parse performance configuration.

        Extracts AMP settings, torch.compile options, Apex GroupNorm settings,
        and CUDA backend configurations (TF32, fp16 reduced precision).
        """
        perf_cfg = self.cfg.training.get("perf", {})
        fp_opt = perf_cfg.get(
            "fp_optimizations", self.cfg.training.get("fp_optimizations", "fp32")
        )

        self.enable_amp = fp_opt.startswith("amp")
        self.amp_dtype = torch.float16 if fp_opt == "amp-fp16" else torch.bfloat16
        self.use_torch_compile = perf_cfg.get("torch_compile", False)
        self.use_apex_gn = perf_cfg.get("use_apex_gn", False)
        self.memory_format = (
            torch.channels_last if self.use_apex_gn else torch.preserve_format
        )

        # CUDA backend settings (configurable via perf section)
        self.cudnn_benchmark = self.cfg.training.get("cudnn_benchmark", True)
        self.allow_tf32 = perf_cfg.get("allow_tf32", False)
        self.allow_fp16_reduced_precision = perf_cfg.get(
            "allow_fp16_reduced_precision", False
        )

        if self.use_apex_gn:
            self.logger0.info("Using Apex GroupNorm with channels_last memory format")

    def _setup_seeds_and_backends(self, step: int = 0):
        r"""
        Configure random seeds and CUDA backends.

        Parameters
        ----------
        step : int, optional
            Current training step for seed offset calculation, by default 0.
        """
        seed_offset = (
            self.cfg.training.seed * self.dist.world_size * max(step, 1)
            + self.dist.rank
        )
        np.random.seed(seed_offset % (1 << 31))
        torch.manual_seed(seed_offset % (1 << 31))

        # Apply CUDA backend settings from perf config
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        torch.backends.cudnn.allow_tf32 = self.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            self.allow_fp16_reduced_precision
        )

    # =========================================================================
    # Data Setup
    # =========================================================================

    def _setup_data(self):
        r"""
        Create datasets and dataloaders.

        Initializes training and validation datasets, creates infinite samplers
        for distributed training, and sets up PyTorch DataLoaders with pinned memory.
        """
        self.logger0.info("Loading dataset...")

        dataset_cls = dataset_classes[self.cfg.dataset.name]
        del self.cfg.dataset.name

        self.dataset_train = dataset_cls(self.cfg.dataset, train=True)
        self.dataset_valid = dataset_cls(self.cfg.dataset, train=False)

        self.state_channels = self.dataset_train.state_channels()
        self.background_channels = self.dataset_train.background_channels()
        self.lead_time_steps = self.dataset_train.lead_time_steps

        # Samplers
        sampler = InfiniteSampler(
            dataset=self.dataset_train,
            rank=self.dist.rank,
            num_replicas=self.dist.world_size,
            seed=self.cfg.training.seed,
        )
        valid_sampler = InfiniteSampler(
            dataset=self.dataset_valid,
            rank=self.dist.rank,
            num_replicas=self.dist.world_size,
            seed=self.cfg.training.seed,
        )

        # Dataloaders
        num_workers = self.cfg.training.num_data_workers
        self.data_loader = torch.utils.data.DataLoader(
            dataset=self.dataset_train,
            batch_size=self.local_batch_size,
            num_workers=num_workers,
            sampler=sampler,
            worker_init_fn=worker_init,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2 if num_workers > 0 else None,
        )
        self.valid_data_loader = torch.utils.data.DataLoader(
            dataset=self.dataset_valid,
            batch_size=self.local_batch_size,
            num_workers=num_workers,
            sampler=valid_sampler,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2 if num_workers > 0 else None,
        )

        self.dataset_iterator = iter(self.data_loader)

        # Invariants
        invariant_array = self.dataset_train.get_invariants()
        if invariant_array is not None:
            self.invariant_tensor = (
                torch.from_numpy(invariant_array)
                .to(
                    dtype=torch.float32,
                    device=self.device,
                    non_blocking=True,
                    memory_format=self.memory_format,
                )
                .unsqueeze(0)
                .repeat(self.local_batch_size, 1, 1, 1)
            )
        else:
            self.invariant_tensor = None

    # =========================================================================
    # Model Setup
    # =========================================================================

    def _setup_model(self):
        r"""
        Construct and configure the neural network.

        Builds the preconditioned architecture (regression or diffusion) based on
        configuration, loads regression network if needed for conditioning, and
        applies memory format optimizations if Apex GroupNorm is enabled.
        """
        self.logger0.info("Constructing network...")

        # Load regression net if needed
        self._load_regression_net()

        # Compute condition channels
        num_cond = {
            "state": len(self.state_channels),
            "background": len(self.background_channels),
            "regression": len(self.state_channels),
            "invariant": 0
            if self.invariant_tensor is None
            else self.invariant_tensor.shape[1],
        }
        num_condition_channels = sum(num_cond[c] for c in self.condition_list)

        self.logger0.info(f"Model conditions: {self.condition_list}")
        self.logger0.info(f"Background channels: {self.background_channels}")
        self.logger0.info(f"State channels: {self.state_channels}")
        self.logger0.info(f"Condition channels: {num_condition_channels}")

        # Build network
        # Convert model config to native Python types for JSON serialization in checkpoints
        model_cfg = OmegaConf.to_container(self.cfg.model, resolve=True)
        hyperparams = model_cfg.get("hyperparameters", {})
        self.net = get_preconditioned_architecture(
            name=self.net_name,
            img_resolution=self.dataset_train.image_shape(),
            target_channels=len(self.state_channels),
            conditional_channels=num_condition_channels,
            spatial_embedding=model_cfg["spatial_pos_embed"],
            lead_time_steps=self.lead_time_steps,
            amp_mode=self.enable_amp,
            **hyperparams,
        )
        self.logger0.info(self.net)
        self.net.train().requires_grad_(True).to(
            device=self.device, memory_format=self.memory_format
        )

        # Keep uncompiled version for validation sampling
        self.net_uncompiled = self.net

    def _load_regression_net(self):
        r"""
        Load pretrained regression network if needed.

        Loads the regression network from checkpoint when 'regression' is in the
        condition list. Sets the network to eval mode with gradients disabled.
        """
        if "regression" not in self.condition_list:
            self.regression_net = None
            return

        self.regression_net = Module.from_checkpoint(
            self.cfg.model.regression_weights,
            override_args={"use_apex_gn": self.use_apex_gn}
            if self.use_apex_gn
            else None,
        )
        if self.enable_amp:
            self.regression_net.amp_mode = self.enable_amp
        self.regression_net = (
            self.regression_net.eval().requires_grad_(False).to(self.device)
        )
        self.regression_net.to(memory_format=self.memory_format)

    # =========================================================================
    # Loss and Optimizer Setup
    # =========================================================================

    def _setup_loss(self):
        r"""
        Create the loss function.

        For regression models, uses MSE loss. For diffusion models, creates EDM loss
        with configurable sigma distribution (lognormal or loguniform).
        Optionally compiles the loss function with torch.compile.
        """
        self.logger0.info("Setting up loss function...")

        if self.loss_type == "regression":
            self.loss_fn = regression_loss_fn
            if self.use_torch_compile:
                self.logger0.info("Compiling loss function with torch.compile...")
                self.loss_fn = torch.compile(self.loss_fn)
            return

        # EDM loss
        loss_params = self.cfg.training.loss
        sigma_data = loss_params.get("sigma_data", 0.5)
        if isinstance(sigma_data, Sequence):
            sigma_data = torch.as_tensor(
                list(sigma_data), dtype=torch.float32, device=self.device
            )[None, :, None, None]

        sigma_dist = loss_params.get("sigma_distribution", "lognormal")
        if sigma_dist == "lognormal":
            loss_cls, param_names = EDMLoss, ("P_mean", "P_std")
        elif sigma_dist == "loguniform":
            loss_cls, param_names = EDMLossLogUniform, ("sigma_min", "sigma_max")
        else:
            raise ValueError(f"Unknown sigma distribution: {sigma_dist}")

        params = {k: v for k, v in loss_params.items() if k in param_names}
        self.logger0.info(f"Using loss: {sigma_dist}, params: {params or 'default'}")
        self.loss_fn = loss_cls(sigma_data=sigma_data, **params)

        if self.use_torch_compile:
            self.logger0.info("Compiling loss function with torch.compile...")
            self.loss_fn = torch.compile(self.loss_fn)

    def _setup_optimizer(self):
        r"""
        Create optimizer and scheduler.

        Builds optimizer using configuration (Adam, AdamW, or StableAdamW).
        Optionally initializes a learning rate scheduler for decay after warmup.
        """
        self.logger0.info("Setting up optimizer...")

        self.optimizer = build_optimizer(
            self.net.parameters(),
            self.cfg.training.get("optimizer", {}),
            lr=self.cfg.training.lr,
        )

        self.scheduler, self.scheduler_name = init_scheduler(
            self.optimizer,
            self.cfg.training.get("scheduler", None),
            warmup_steps=self.warmup_steps,
            total_steps=self.total_train_steps,
        )
        if self.scheduler:
            self.logger0.info(f"Using scheduler: {self.scheduler_name}")

        self.augment_pipe = None

    def _setup_ddp(self):
        r"""
        Setup DistributedDataParallel.

        Wraps the network in DDP for distributed training with gradient
        synchronization across processes.
        """
        self.ddp = torch.nn.parallel.DistributedDataParallel(
            self.net,
            device_ids=[self.device],
            broadcast_buffers=False,
            bucket_cap_mb=35,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    def _resume_or_init(self):
        r"""
        Resume from checkpoint or initialize training.

        Attempts to load model, optimizer, and scheduler state from checkpoint.
        If no checkpoint exists, optionally loads initial weights from a separate file.
        Re-seeds RNG for reproducibility after checkpoint load.
        """
        self.logger0.info(f'Trying to resume from "{self.ckpt_path}"...')

        # Load checkpoint with metadata
        metadata_dict = {}
        self.total_steps = load_checkpoint(
            path=self.ckpt_path,
            models=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=None
            if self.cfg.training.resume_checkpoint == "latest"
            else self.cfg.training.resume_checkpoint,
            metadata_dict=metadata_dict,
        )

        # Load validation loss from metadata
        self.val_loss = metadata_dict.get("val_loss", -1.0)

        if self.total_steps == 0:
            self.logger0.info("No resumable state found.")
            init_weights = self.cfg.training.get("initial_weights", None)
            if init_weights is None:
                self.logger0.info("Starting training from scratch...")
            else:
                self.logger0.info(f"Loading initial weights from {init_weights}...")
                self.net.load(init_weights)

        # Re-seed for reproducibility
        self._setup_seeds_and_backends(self.total_steps)

    # =========================================================================
    # Training Step
    # =========================================================================

    def train_step(self) -> torch.Tensor:
        r"""
        Execute a single training step with gradient accumulation.

        Performs forward pass, loss computation, backward pass, and optimizer step.
        Supports gradient accumulation over multiple batches, gradient clipping,
        and manual learning rate warmup.

        Returns
        -------
        torch.Tensor
            The computed loss tensor (synchronized across ranks if distributed).
        """
        self.optimizer.zero_grad(set_to_none=True)
        loss = None

        for _ in range(self.num_accumulation_rounds):
            batch = next(self.dataset_iterator)
            mem_format = self.memory_format
            background, state, mask, lead_time_label = unpack_batch(
                batch, self.device, memory_format=mem_format
            )

            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
                condition, target, _ = build_network_condition_and_target(
                    background,
                    state,
                    self.invariant_tensor,
                    lead_time_label=lead_time_label,
                    regression_net=self.regression_net,
                    condition_list=self.condition_list,
                    regression_condition_list=self.cfg.model.regression_conditions,
                )
                # Only pass lead_time_label if the model supports it
                loss_kwargs = {}
                if lead_time_label is not None:
                    loss_kwargs["lead_time_label"] = lead_time_label
                loss = self.loss_fn(
                    net=self.ddp,
                    images=target,
                    condition=condition,
                    augment_pipe=self.augment_pipe,
                    **loss_kwargs,
                )

                if mask is not None:
                    loss = loss * mask

            if self.log_to_wandb:
                channelwise_loss = loss.mean(dim=(0, 2, 3))
                self.wandb_logs["channelwise_loss"] = {
                    f"ChLoss/{ch}": channelwise_loss[i].item()
                    for i, ch in enumerate(self.state_channels)
                }

            loss_value = loss.sum() / len(self.state_channels)
            loss_value.backward()

        # Gradient clipping
        clip_grad_norm = self.cfg.training.get("clip_grad_norm", -1)
        if clip_grad_norm > 0:
            clip_grad_norm_(self.net.parameters(), clip_grad_norm)

        # Manual LR warmup (linear ramp) - only during warmup phase
        # After warmup, let the scheduler control the LR
        if self.total_steps < self.warmup_steps:
            # Use (total_steps + 1) so that at step warmup_steps-1, lr_scale = 1.0
            lr_scale = (self.total_steps + 1) / self.warmup_steps
            for g in self.optimizer.param_groups:
                g["lr"] = self.cfg.training.lr * lr_scale

        # Clean NaN gradients
        for param in self.net.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                )

        self.optimizer.step()
        step_scheduler(
            self.scheduler,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
        )

        # Sync loss across ranks
        if self.dist.world_size > 1:
            torch.distributed.barrier()
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)

        return loss

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(
        self,
    ) -> Tuple[float, Optional[torch.Tensor], Optional[List], Optional[torch.Tensor]]:
        r"""
        Run validation loop.

        Evaluates model on validation set with deterministic seeding for reproducibility.
        Collects outputs from the first batch for visualization.

        Returns
        -------
        val_loss : float
            Average validation loss across all validation steps.
        plot_outputs : torch.Tensor or None
            Model outputs from first batch for plotting.
        plot_state : List or None
            Input/target state tensors from first batch.
        plot_background : torch.Tensor or None
            Background conditioning from first batch.
        """
        # Set seed for reproducible validation
        np.random.seed(0)
        torch.manual_seed(0)

        valid_iter = iter(self.valid_data_loader)
        valid_loss_sum = torch.zeros((), device=self.device)
        plot_outputs, plot_state, plot_background = None, None, None

        with torch.no_grad():
            for v_i in range(self.validation_steps):
                batch = next(valid_iter)
                mem_format = self.memory_format
                background, state, mask, lead_time_label = unpack_batch(
                    batch, self.device, memory_format=mem_format
                )

                with torch.autocast(
                    "cuda", dtype=self.amp_dtype, enabled=self.enable_amp
                ):
                    condition, target, reg_out = build_network_condition_and_target(
                        background,
                        state,
                        self.invariant_tensor,
                        lead_time_label=lead_time_label,
                        regression_net=self.regression_net,
                        condition_list=self.condition_list,
                        regression_condition_list=self.cfg.model.regression_conditions,
                    )

                    loss_kwargs = (
                        {"return_model_outputs": True}
                        if self.net_name == "regression"
                        else {}
                    )
                    # Only pass lead_time_label if the model supports it
                    if lead_time_label is not None:
                        loss_kwargs["lead_time_label"] = lead_time_label

                    valid_loss = self.loss_fn(
                        net=self.net,
                        images=target,
                        condition=condition,
                        augment_pipe=self.augment_pipe,
                        **loss_kwargs,
                    )

                    # Apply mask
                    if mask is not None:
                        if isinstance(valid_loss, tuple):
                            valid_loss = (valid_loss[0] * mask, valid_loss[1])
                        else:
                            valid_loss = valid_loss * mask

                    # Save first batch for plotting
                    if v_i == 0:
                        plot_state, plot_background = state, background
                        plot_outputs = self._get_plot_outputs(
                            valid_loss, condition, state, lead_time_label, reg_out
                        )
                    elif self.loss_type == "regression":
                        valid_loss, _ = valid_loss

                    valid_loss_sum += (
                        valid_loss.mean()
                        if not isinstance(valid_loss, tuple)
                        else valid_loss[0].mean()
                    )

        # Sync across ranks
        if self.dist.world_size > 1:
            torch.distributed.barrier()
            torch.distributed.all_reduce(
                valid_loss_sum, op=torch.distributed.ReduceOp.AVG
            )

        val_loss = (valid_loss_sum / max(self.validation_steps, 1)).item()

        step_scheduler(
            self.scheduler,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            metric=val_loss,
        )

        return val_loss, plot_outputs, plot_state, plot_background

    def _get_plot_outputs(self, valid_loss, condition, state, lead_time_label, reg_out):
        r"""
        Get outputs for validation plotting.

        Parameters
        ----------
        valid_loss : torch.Tensor or tuple
            Validation loss, or tuple of (loss, outputs) for regression.
        condition : torch.Tensor
            Conditioning tensor for the model.
        state : tuple
            Tuple of (input_state, target_state) tensors.
        lead_time_label : torch.Tensor or None
            Lead time embedding indices if using lead time conditioning.
        reg_out : torch.Tensor or None
            Regression network output for residual addition.

        Returns
        -------
        torch.Tensor
            Model outputs for visualization.
        """
        if self.net_name == "diffusion":
            outputs = diffusion_model_forward(
                self.net_uncompiled,
                condition,
                state[1].shape,
                sampler_args=dict(self.cfg.sampler.args),
                lead_time_label=lead_time_label,
            )
            if "regression" in self.condition_list:
                outputs += reg_out
            return outputs
        else:
            # Regression model - valid_loss is (loss_tensor, output_images)
            valid_loss_tensor, output_images = valid_loss
            return output_images

    # =========================================================================
    # Plotting and Logging
    # =========================================================================

    def save_validation_plots(self, plot_outputs, plot_state, plot_background):
        r"""
        Save validation plots to disk and wandb.

        Parameters
        ----------
        plot_outputs : torch.Tensor or None
            Model outputs to visualize.
        plot_state : tuple or None
            Tuple of (input_state, target_state) for comparison plots.
        plot_background : torch.Tensor or None
            Background conditioning for context panels.
        """
        if self.dist.rank != 0 or plot_outputs is None or plot_state is None:
            return

        fields = self.cfg.training.validation_plot_variables

        for i in range(plot_outputs.shape[0]):
            image = plot_outputs[i].cpu().numpy()
            figs, spec_ratios = ps1d_plots(
                plot_outputs[i], plot_state[1][i], fields, self.state_channels
            )

            for f_ in fields:
                f_idx = self.state_channels.index(f_)
                image_dir = os.path.join(self.cfg.training.rundir, "images", f_)
                os.makedirs(image_dir, exist_ok=True)

                bg_panels = self._prepare_background_panels(plot_background, i)
                input_state = (
                    plot_state[0][i, f_idx].cpu().numpy()
                    if plot_state[0] is not None
                    else None
                )
                fig = validation_plot(
                    image[f_idx],
                    plot_state[1][i, f_idx].cpu().numpy(),
                    input_state,
                    f_,
                    bg_panels,
                )
                fig.savefig(os.path.join(image_dir, f"{self.total_steps}_{i}_{f_}.png"))
                figs[f"PS1D_{f_}"].savefig(
                    os.path.join(image_dir, f"{self.total_steps}_{i}_{f_}_spec.png")
                )

                if self.log_to_wandb:
                    for figname, plot in figs.items():
                        self.wandb_logs[figname] = wandb.Image(plot)
                    self.wandb_logs[f"generated_{f_}"] = wandb.Image(fig)

                if self.writer is not None:
                    for figname, plot in figs.items():
                        self.writer.add_figure(figname, plot, self.total_steps)
                    self.writer.add_figure(f"generated_{f_}", fig, self.total_steps)

                    plt.close("all")

        if self.log_to_wandb:
            self.wandb_logs.update(spec_ratios)
            wandb.log(self.wandb_logs, step=self.total_steps)

    def _prepare_background_panels(self, plot_background, batch_idx) -> Optional[Dict]:
        r"""
        Prepare background panels for validation plot.

        Parameters
        ----------
        plot_background : torch.Tensor or None
            Background conditioning tensor of shape :math:`(B, C, H, W)`.
        batch_idx : int
            Index of the batch sample to extract.

        Returns
        -------
        dict or None
            Dictionary mapping channel names to numpy arrays, or None if no background.
        """
        if plot_background is None:
            return None

        selected = self.validation_bg_channels or (
            [self.background_channels[0]] if self.background_channels else []
        )
        panels = {}

        for bg in selected:
            if isinstance(bg, int):
                if bg < 0 or bg >= plot_background.shape[1]:
                    continue
                label = (
                    self.background_channels[bg]
                    if bg < len(self.background_channels)
                    else f"ch_{bg}"
                )
                idx = bg
            else:
                if bg not in self.background_channels:
                    continue
                idx = self.background_channels.index(bg)
                label = bg
            panels[label] = plot_background[batch_idx, idx].detach().cpu().numpy()

        return panels if panels else None

    def log_progress(self):
        r"""
        Log training progress.

        Prints a summary line with step count, timing, memory usage, learning rate,
        and loss values. Resets step counters and memory statistics after logging.
        """
        current_time = time.time()
        lr = self.optimizer.param_groups[0]["lr"]

        fields = [
            f"steps {self.total_steps:<5d}",
            f"samples {self.total_steps * self.batch_size}",
            f"tot_time {current_time - self.start_time:.2f}",
            f"step_time {(current_time - self.train_start) / max(self.train_steps, 1):.2f}",
            f"valid_time {self.valid_time:.2f}",
            f"cpumem {psutil.Process(os.getpid()).memory_info().rss / 2**30:<6.2f}",
            f"gpumem {torch.cuda.max_memory_allocated(self.device) / 2**30:<6.2f}",
            f"lr {lr:.6g}",
            f"train_loss {self.avg_train_loss / max(self.train_steps, 1):<6.5f}",
            f"val_loss {self.val_loss:<6.5f}",
        ]
        self.logger0.info(" ".join(fields))

        # Reset counters
        self.train_steps = 0
        self.train_start = time.time()
        self.avg_train_loss = 0
        torch.cuda.reset_peak_memory_stats()

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def save_checkpoint(self):
        r"""
        Save training checkpoint with metadata.

        Saves model weights, optimizer state, scheduler state, and validation loss
        to the checkpoint directory. Only rank 0 saves to avoid file conflicts.
        """
        if self.dist.rank == 0:
            save_checkpoint(
                path=self.ckpt_path,
                models=self.net,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=self.total_steps,
                metadata={"val_loss": self.val_loss},
            )

    # =========================================================================
    # Main Training Loop
    # =========================================================================

    def train(self):
        r"""
        Main training loop.

        Runs training until total_train_steps is reached. Handles training steps,
        validation, logging, and checkpointing according to configured frequencies.
        Cleans up TensorBoard writer on exit.
        """
        self.logger0.info(
            f"Training up to {self.total_train_steps} steps from step {self.total_steps}..."
        )
        # resetting in log_progress
        self.train_start = time.time()

        while self.total_steps < self.total_train_steps:
            # Training step
            loss = self.train_step()
            train_loss = loss.mean().cpu().item()
            self.avg_train_loss += train_loss
            self.train_steps += 1
            self.total_steps += 1

            # Logging
            lr = self.optimizer.param_groups[0]["lr"]
            if self.log_to_wandb:
                self.wandb_logs["loss"] = train_loss / self.train_steps
                self.wandb_logs["lr"] = lr
            if self.writer is not None:
                self.writer.add_scalar("loss/train", train_loss, self.total_steps)
                self.writer.add_scalar("lr", lr, self.total_steps)

            # Validation
            if self.total_steps % self.cfg.training.validation_freq == 0:
                valid_start = time.time()
                self.val_loss, plot_outputs, plot_state, plot_background = (
                    self.validate()
                )

                if self.log_to_wandb:
                    self.wandb_logs["valid_loss"] = self.val_loss
                if self.writer is not None:
                    self.writer.add_scalar(
                        "loss/valid", self.val_loss, self.total_steps
                    )

                self.save_validation_plots(plot_outputs, plot_state, plot_background)
                self.valid_time = time.time() - valid_start

            # Log progress
            if self.total_steps % self.cfg.training.print_progress_freq == 0:
                self.log_progress()

            # Checkpointing
            done = self.total_steps >= self.total_train_steps
            if (
                done or self.total_steps % self.cfg.training.checkpoint_freq == 0
            ) and self.total_steps != 0:
                self.save_checkpoint()
                self._setup_seeds_and_backends(self.total_steps)

        # Cleanup
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        self.logger0.info("\nExiting...")
