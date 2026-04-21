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

"""Trainer class for StormCast/StormScope training."""

import os
import time

import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch.nn.utils import clip_grad_norm_
import psutil
from physicsnemo.core import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint, load_model_weights, save_checkpoint

from physicsnemo.diffusion.metrics.losses import WeightedMSEDSMLoss
from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler, NoiseScheduler

from utils.loss import (
    RegressionLoss,
    SigmaBinTracker,
    build_noise_scheduler,
)

from utils.config import MainConfig
from utils.logging import ExperimentLogger
from utils.nn import (
    diffusion_model_forward,
    get_preconditioned_natten_dit,
    get_preconditioned_unet,
    build_network_condition_and_target,
    unpack_batch,
)
from utils.optimizers import build_optimizer
from utils.parallel import ParallelHelper
from utils.plots import save_validation_plots
from utils.schedulers import init_scheduler, step_scheduler
from datasets import dataset_classes


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
    train_noise_scheduler : NoiseScheduler or None
        For diffusion training, the noise scheduler used for training-time
        sigma sampling (same object as held by the loss). Set before applying
        ``torch.compile`` to the loss so callers always access the real
        scheduler. ``None`` for regression runs.

    Examples
    --------
    >>> from omegaconf import OmegaConf
    >>> cfg = OmegaConf.load("config.yaml")
    >>> trainer = Trainer(cfg)
    >>> trainer.train()
    """

    def __init__(self, cfg: DictConfig):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        self.cfg = MainConfig(**cfg_dict)  # validates config, including types
        self.logger = ExperimentLogger("train", self.cfg)
        self.logger.info("Configuration validated successfully")

        self.start_time = time.time()

        # Distributed setup
        self.dist = DistributedManager()
        self.device = self.dist.device
        domain_parallel_size = self.cfg.training.domain_parallel_size
        self.use_shard_tensor = (
            domain_parallel_size > 1
        ) or self.cfg.training.force_sharding
        self.parallel_helper = ParallelHelper(
            domain_parallel_size=domain_parallel_size,
            use_shard_tensor=self.use_shard_tensor,
        )
        if self.use_shard_tensor and (
            self.parallel_helper.local_batch_size(cfg.training.batch_size) > 1
        ):
            raise ValueError(
                "Domain parallelism is only available with a local batch size of 1."
            )

        # Parse config
        self._parse_config()

        # Initialize components
        self._setup_data()

        # All ranks use the same seed so parameter initialization is identical.
        # FSDP sync_module_states and distribute_tensor also broadcast from
        # rank 0, but explicit seeding avoids silent dependence on those.
        torch.manual_seed(self.cfg.training.seed)

        # Create model and move to device
        self.net = self._setup_model()
        self.logger.info(str(self.net))
        self.net.train().requires_grad_(True).to(
            device=self.device, memory_format=self.memory_format
        )

        # Load regression net if needed
        self.regression_net = self._load_regression_net()

        # Sharding and FSDP wrapping
        if self.use_shard_tensor:
            self.logger.info(
                "Distributing model with FSDP and sharding for domain parallelism"
            )
        else:
            self.logger.info("Distributing model with FSDP")
        self.net = self.parallel_helper.distribute_model(self.net)
        if self.regression_net is not None:
            self.regression_net = self.parallel_helper.distribute_model(
                self.regression_net
            )
        if self.invariant_tensor is not None:
            self.invariant_tensor = self.parallel_helper.distribute_tensor(
                self.invariant_tensor
            )

        # Create optimizer on the distributed model
        (self.optimizer, self.scheduler) = self._setup_optimizer(self.net)

        # Resume from checkpoint (all ranks participate)
        (self.total_steps, self.val_loss) = self._resume_or_init()

        # Loss function (``train_noise_scheduler`` is set inside for diffusion)
        self.train_noise_scheduler: NoiseScheduler | None = None
        self.loss_fn = self._setup_loss()
        self.sigma_bin_tracker = SigmaBinTracker(
            self.cfg.training.loss, self.device, self.loss_type
        )
        if self.sigma_bin_tracker.enabled:
            self.logger.info(
                f"Sigma-bin tracking enabled with edges: {self.sigma_bin_tracker.edges}"
            )

        # Training state
        self.train_steps = 0
        self.avg_train_loss = 0.0
        self.valid_time = -1.0

        # Put RNG in a deterministic per-rank state for any operations
        # between __init__ and the first train_step.  train_step will call
        # _setup_seeds again with the current step before doing real work.
        self._setup_seeds(self.total_steps)

    # =========================================================================
    # Configuration
    # =========================================================================

    def _parse_config(self):
        r"""
        Parse and store configuration values.

        Extracts and stores batch sizes, training parameters, validation config,
        model type, performance options, and checkpoint paths from the configuration.
        """
        cfg = self.cfg

        # Batch sizes
        self.batch_size = cfg.training.batch_size
        max_local_batch_size = self.parallel_helper.local_batch_size(self.batch_size)
        if cfg.training.batch_size_per_gpu == "auto":
            self.local_batch_size = max_local_batch_size
        else:
            self.local_batch_size = cfg.training.batch_size_per_gpu
            assert max_local_batch_size % self.local_batch_size == 0
        self.num_accumulation_rounds = max_local_batch_size // self.local_batch_size
        assert (
            self.batch_size * self.parallel_helper.domain_parallel_size
            == self.dist.world_size * max_local_batch_size
        )

        # Training params
        self.total_train_steps = cfg.training.total_train_steps
        self.warmup_steps = cfg.training.scheduler.lr_rampup_steps

        # Validation config
        self.validation_steps = cfg.training.validation_steps
        self.validation_bg_channels = cfg.training.validation_plot_background_channels

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
        perf_cfg = self.cfg.training.perf
        fp_opt = perf_cfg.fp_optimizations

        self.enable_amp = fp_opt.startswith("amp")
        self.amp_dtype = torch.float16 if fp_opt == "amp-fp16" else torch.bfloat16
        self.use_torch_compile = perf_cfg.torch_compile
        self.use_apex_gn = perf_cfg.use_apex_gn
        use_channels_last = self.use_apex_gn or (self.cfg.model.architecture == "dit")
        self.memory_format = (
            torch.channels_last if use_channels_last else torch.preserve_format
        )

        # CUDA backend settings (configurable via perf section)
        self.cudnn_benchmark = self.cfg.training.cudnn_benchmark
        self.allow_tf32 = perf_cfg.allow_tf32
        self.allow_fp16_reduced_precision = perf_cfg.allow_fp16_reduced_precision

        if self.use_apex_gn:
            self.logger.info("Using Apex GroupNorm with channels_last memory format")

        # Apply CUDA backend settings from perf config
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        if self.allow_tf32:
            torch.backends.cudnn.conv.fp32_precision = "tf32"
            torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            self.allow_fp16_reduced_precision
        )

    def _setup_seeds(self, step: int = 0):
        r"""
        Set deterministic per-rank, per-step RNG seeds.

        Called at the start of each training step so that randomness (diffusion
        sigma sampling, noise generation) is reproducible.  Each
        ``(step, rank)`` pair maps to a unique seed:

        * Different ranks produce different random sequences, as required by
          data-parallel training where each rank processes different data.
        * The same ``(seed, step, rank)`` triple always reproduces the same
          sequence across identical runs.
        * Within a model-parallel (domain) group, diffusion sigma values
          are kept consistent via broadcast (handled by the
          :class:`~physicsnemo.diffusion.DomainParallelNoiseScheduler`
          that wraps the noise scheduler), not by sharing a seed, so that
          spatial noise generated by ``torch.randn_like`` remains
          independent per shard.

        Parameters
        ----------
        step : int, optional
            Current training step, by default 0.
        """
        seed = self.cfg.training.seed + step * self.dist.world_size + self.dist.rank
        np.random.seed(seed % (1 << 31))
        torch.manual_seed(seed)

    # =========================================================================
    # Data Setup
    # =========================================================================

    def _setup_data(self):
        r"""
        Create datasets and dataloaders.

        Initializes training and validation datasets, creates infinite samplers
        for distributed training, and sets up PyTorch DataLoaders with pinned memory.
        """
        self.logger.info("Loading dataset...")

        dataset_cls = dataset_classes[self.cfg.dataset.name]
        dataset_kwargs = self.cfg.dataset.__dict__.copy()
        del dataset_kwargs["name"]
        self.dataset_train = dataset_cls(dataset_kwargs, train=True)
        self.dataset_valid = dataset_cls(dataset_kwargs, train=False)

        self.state_channels = self.dataset_train.state_channels()
        self.background_channels = self.dataset_train.background_channels()
        self.scalar_cond_channels = self.dataset_train.scalar_condition_channels()
        self.lead_time_steps = self.dataset_train.lead_time_steps

        # Dataloaders
        num_workers = self.cfg.training.num_data_workers
        self.train_dataloader = self.parallel_helper.sharded_dataloader(
            self.dataset_train,
            batch_size=self.local_batch_size,
            num_workers=num_workers,
            seed=self.cfg.training.seed,
        )
        self.dataset_iterator = self.parallel_helper.sharded_data_iter(
            self.train_dataloader
        )

        # Invariants
        invariant_array = self.dataset_train.get_invariants()
        if invariant_array is not None:
            self.invariant_tensor = (
                torch.from_numpy(invariant_array)
                .unsqueeze(0)
                .to(device=self.device, memory_format=self.memory_format)
                .repeat(self.local_batch_size, 1, 1, 1)
            )
        else:
            self.invariant_tensor = None
            if (
                "invariant" in self.cfg.model.diffusion_conditions
                or "invariant" in self.cfg.model.regression_conditions
            ):
                self.logger.info(
                    "Invariant conditions specified in model configuration, but dataset provides no invariants. Ignoring invariant conditions."
                )

        if (
            self.cfg.model.architecture != "dit"
        ) and self.dataset_train.scalar_condition_channels():
            raise ValueError(
                "Scalar conditions are only supported for the 'dit' architecture."
            )

    # =========================================================================
    # Model Setup
    # =========================================================================

    def _setup_model(self) -> Module:
        r"""
        Construct and configure the neural network.

        Builds the preconditioned architecture (regression or diffusion) based on
        configuration, loads regression network if needed for conditioning, and
        applies memory format optimizations if Apex GroupNorm is enabled.

        Returns
        -------
        physicsnemo.core.Module
            The network to be trained.
        """
        self.logger.info("Constructing network...")

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

        self.logger.info(f"Model conditions: {self.condition_list}")
        self.logger.info(f"Background channels: {self.background_channels}")
        self.logger.info(f"State channels: {self.state_channels}")
        self.logger.info(f"Condition channels: {num_condition_channels}")

        # Build network
        model_cfg = self.cfg.model
        if model_cfg.architecture == "unet":
            net = get_preconditioned_unet(
                name=self.net_name,
                img_resolution=self.dataset_train.image_shape(),
                target_channels=len(self.state_channels),
                conditional_channels=num_condition_channels,
                lead_time_steps=self.lead_time_steps,
                amp_mode=self.enable_amp,
                use_apex_gn=self.use_apex_gn,
                **model_cfg.hyperparameters,
            )
        elif model_cfg.architecture == "dit":
            net = get_preconditioned_natten_dit(
                img_resolution=self.dataset_train.image_shape(),
                target_channels=len(self.state_channels),
                conditional_channels=num_condition_channels,
                scalar_condition_channels=len(self.scalar_cond_channels),
                lead_time_steps=self.lead_time_steps,
                **model_cfg.hyperparameters,
            )
        else:
            raise ValueError("model.architecture must be 'unet' or 'dit'")

        return net

    def _load_regression_net(self) -> Module | None:
        r"""
        Load pretrained regression network if needed.

        Loads the regression network from checkpoint when 'regression' is in the
        condition list. Sets the network to eval mode with gradients disabled.

        Returns
        -------
        physicsnemo.core.Module | None
            The regression net, or None if no regression net is used.
        """
        if "regression" not in self.condition_list:
            return None

        regression_net = Module.from_checkpoint(
            self.cfg.model.regression_weights,
            override_args={"use_apex_gn": self.use_apex_gn}
            if self.use_apex_gn
            else None,
        )
        if self.enable_amp:
            regression_net.amp_mode = self.enable_amp
        return (
            regression_net.eval()
            .requires_grad_(False)
            .to(device=self.device, memory_format=self.memory_format)
        )

    # =========================================================================
    # Loss and Optimizer Setup
    # =========================================================================

    def _setup_loss(self) -> WeightedMSEDSMLoss | RegressionLoss:
        r"""
        Create the loss function.

        For regression models, uses :class:`~utils.loss.RegressionLoss`.
        For diffusion models, creates a
        :class:`~physicsnemo.diffusion.metrics.losses.WeightedMSEDSMLoss`
        with an
        :class:`~physicsnemo.diffusion.noise_schedulers.EDMNoiseScheduler`.
        When domain parallelism is active the scheduler is wrapped via
        :meth:`~utils.parallel.ParallelHelper.make_domain_parallel_scheduler`
        so that sampled sigmas are broadcast across spatial shards.

        A separate sampling scheduler is also created for validation-time
        diffusion sampling.

        When ``training.perf.torch_compile`` is enabled (and domain parallelism
        is off), the loss callable is wrapped with :func:`torch.compile` so the
        forward path through the preconditioned model and loss is compiled.

        Returns
        -------
        WeightedMSEDSMLoss | RegressionLoss
            The loss function (possibly ``torch.compile``-wrapped).
        """
        self.logger.info("Setting up loss function...")

        self.sampling_scheduler = None

        compile_loss = self.use_torch_compile and not self.use_shard_tensor
        if self.use_torch_compile and self.use_shard_tensor:
            self.logger.info(
                "Skipping torch.compile on loss: not supported with "
                "domain parallelism / ShardTensor."
            )

        if self.loss_type == "regression":
            loss_fn = RegressionLoss(self.net)
            if compile_loss:
                self.logger.info("Compiling loss function with torch.compile...")
                loss_fn = torch.compile(loss_fn)
            return loss_fn

        loss_params = self.cfg.training.loss
        self.logger.info(
            f"Using modern diffusion loss: {loss_params.sigma_distribution}"
        )

        noise_scheduler = build_noise_scheduler(
            loss_params,
            self.logger,
        )
        noise_scheduler = self.parallel_helper.make_domain_parallel_scheduler(
            noise_scheduler,
        )
        loss_fn = WeightedMSEDSMLoss(self.net, noise_scheduler, reduction="none")
        self.train_noise_scheduler = loss_fn.noise_scheduler

        sa = self.cfg.sampler.args.__dict__
        sampling_scheduler = EDMNoiseScheduler(
            sigma_min=sa.get("sigma_min", 0.002),
            sigma_max=sa.get("sigma_max", 80.0),
            rho=sa.get("rho", 7.0),
        )
        self.sampling_scheduler = self.parallel_helper.make_domain_parallel_scheduler(
            sampling_scheduler
        )

        if compile_loss:
            self.logger.info("Compiling loss function with torch.compile...")
            loss_fn = torch.compile(loss_fn)

        return loss_fn

    def _setup_optimizer(
        self, net: torch.nn.Module
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
        r"""
        Create optimizer and scheduler.

        Builds optimizer using configuration (Adam or AdamW).
        Optionally initializes a learning rate scheduler for decay after warmup.

        Parameters
        ----------
        net : physicsnemo.core.Module
            The module for which the optimizer is created.

        Returns
        -------
        optimizer: torch.optim.Optimizer
            The optimizer for the given network.
        scheduler: float
            The learning rate scheduler, or None if no scheduler is used.
        """
        self.logger.info("Setting up optimizer...")

        optimizer = build_optimizer(net.parameters(), self.cfg.training.optimizer)

        scheduler, scheduler_name = init_scheduler(
            optimizer,
            self.cfg.training.scheduler,
            total_steps=self.total_train_steps,
            logger=self.logger,
        )
        if scheduler:
            self.logger.info(f"Using scheduler: {scheduler_name}")

        return (optimizer, scheduler)

    def _resume_or_init(self) -> tuple[int, float]:
        r"""
        Resume from checkpoint or initialize training.

        All ranks participate.  The distributed checkpoint utilities handle
        gathering (save) and scattering (load) of FSDP / ShardTensor state
        automatically.

        Returns
        -------
        total_steps: int
            The number of training steps that the loaded checkpoint was trained for,
            or 0 if checkpoint was not loaded.
        val_loss: float
            The validation loss saved in the checkpoint metadata, or -1.0 if checkpoint
            was not loaded.
        """
        self.logger.info(f'Trying to resume from "{self.ckpt_path}"...')

        metadata_dict: dict = {}
        total_steps = load_checkpoint(
            path=self.ckpt_path,
            models=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=None
            if self.cfg.training.resume_checkpoint == "latest"
            else self.cfg.training.resume_checkpoint,
            metadata_dict=metadata_dict,
        )

        val_loss = metadata_dict.get("val_loss", -1.0)

        if total_steps == 0:
            self.logger.info("No resumable state found.")
            init_weights = self.cfg.training.initial_weights
            if init_weights is not None:
                self.logger.info(f"Loading initial weights from {init_weights}...")
                load_model_weights(self.net, init_weights)
            else:
                self.logger.info("Starting training from scratch...")

        return (total_steps, val_loss)

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
        self._setup_seeds(self.total_steps)
        self.optimizer.zero_grad(set_to_none=True)
        loss = None
        channelwise_loss = torch.zeros((), device=self.device, requires_grad=False)
        self.sigma_bin_tracker.reset()

        for _ in range(self.num_accumulation_rounds):
            batch = next(self.dataset_iterator)
            background, state, mask, lead_time_label, scalar_conditions = unpack_batch(
                batch, self.device, memory_format=self.memory_format
            )

            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
                condition, target, _ = build_network_condition_and_target(
                    background,
                    state,
                    self.invariant_tensor,
                    lead_time_label=lead_time_label,
                    scalar_conditions=scalar_conditions,
                    regression_net=self.regression_net,
                    condition_list=self.condition_list,
                    regression_condition_list=self.cfg.model.regression_conditions,
                )
                del background, state, scalar_conditions

                weight = (
                    mask
                    if mask is not None
                    else self.parallel_helper.replicate_tensor(
                        torch.ones((), device=target.device, dtype=target.dtype)
                    )
                )
                loss_kwargs = {}
                if lead_time_label is not None:
                    loss_kwargs["lead_time_label"] = lead_time_label

                sigma = None
                if self.loss_type != "regression":
                    assert self.train_noise_scheduler is not None
                    sigma = self.train_noise_scheduler.sample_time(
                        target.shape[0],
                        device=target.device,
                        dtype=target.dtype,
                    )
                    loss_kwargs["t"] = sigma

                loss = self.loss_fn(target, weight, condition=condition, **loss_kwargs)

                self.sigma_bin_tracker.update(loss, sigma)

            channelwise_loss_step = loss.detach().mean(dim=(0, 2, 3))
            if self.use_shard_tensor:
                channelwise_loss_step = channelwise_loss_step.to_local()
            channelwise_loss = channelwise_loss + channelwise_loss_step

            loss_value = loss.sum() / len(self.state_channels)
            loss_value.backward()

        for ch, value in zip(self.state_channels, channelwise_loss):
            self.logger.log_value(
                f"loss/train/{ch}", value / self.num_accumulation_rounds
            )

        self.sigma_bin_tracker.log(self.logger, world_size=self.dist.world_size)

        # Gradient clipping
        if self.cfg.training.clip_grad_norm > 0:
            clip_grad_norm_(self.net.parameters(), self.cfg.training.clip_grad_norm)

        # Manual LR warmup (linear ramp) - only during warmup phase
        # After warmup, let the scheduler control the LR
        if self.total_steps < self.warmup_steps:
            # Use (total_steps + 1) so that at step warmup_steps-1, lr_scale = 1.0
            lr_scale = (self.total_steps + 1) / self.warmup_steps
            for g in self.optimizer.param_groups:
                g["lr"] = self.cfg.training.optimizer.lr * lr_scale

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
            logger=self.logger,
        )

        # Sync loss across ranks
        if self.dist.world_size > 1:
            torch.distributed.barrier()
            if self.use_shard_tensor:
                loss = loss.detach().mean().to_local()
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)

        return loss

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(
        self,
    ) -> tuple[
        float, torch.Tensor | None, list[torch.Tensor] | None, torch.Tensor | None
    ]:
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
        # Fixed validation seed tied to config so results are reproducible across
        # runs.  Uses a large offset to avoid overlap with training-step seeds.
        val_seed = self.cfg.training.seed + (1 << 30) + self.dist.rank
        np.random.seed(val_seed % (1 << 31))
        torch.manual_seed(val_seed)

        valid_dataloader = self.parallel_helper.sharded_dataloader(
            self.dataset_valid,
            batch_size=self.local_batch_size,
            seed=0,
            num_workers=0,  # self.cfg.training.num_data_workers,
            shuffle=False,
        )
        valid_iter = self.parallel_helper.sharded_data_iter(
            valid_dataloader, self.validation_steps
        )
        valid_loss_sum = torch.zeros((), device=self.device)
        plot_outputs, plot_state, plot_background = None, None, None

        with torch.no_grad():
            for v_i, batch in enumerate(valid_iter):
                background, state, mask, lead_time_label, scalar_conditions = (
                    unpack_batch(batch, self.device, memory_format=self.memory_format)
                )

                with torch.autocast(
                    "cuda", dtype=self.amp_dtype, enabled=self.enable_amp
                ):
                    condition, target, reg_out = build_network_condition_and_target(
                        background,
                        state,
                        self.invariant_tensor,
                        lead_time_label=lead_time_label,
                        scalar_conditions=scalar_conditions,
                        regression_net=self.regression_net,
                        condition_list=self.condition_list,
                        regression_condition_list=self.cfg.model.regression_conditions,
                    )

                    weight = (
                        mask
                        if mask is not None
                        else self.parallel_helper.replicate_tensor(
                            torch.ones((), device=target.device, dtype=target.dtype)
                        )
                    )
                    loss_kwargs = {}
                    if lead_time_label is not None:
                        loss_kwargs["lead_time_label"] = lead_time_label

                    valid_loss = self.loss_fn(
                        target, weight, condition=condition, **loss_kwargs
                    )

                    if v_i == 0:
                        plot_state, plot_background = state, background
                        plot_outputs = self._get_plot_outputs(
                            condition, state, lead_time_label, reg_out
                        )

                    valid_loss_mean_step = valid_loss.mean(dim=(0, 2, 3))
                    if self.use_shard_tensor:
                        valid_loss_mean_step = valid_loss_mean_step.to_local()
                    valid_loss_sum = valid_loss_sum + valid_loss_mean_step

        # Sync across ranks
        if self.dist.world_size > 1:
            torch.distributed.barrier()
            torch.distributed.all_reduce(
                valid_loss_sum, op=torch.distributed.ReduceOp.AVG
            )

        val_loss = (valid_loss_sum / max(self.validation_steps, 1)).cpu().numpy()

        step_scheduler(
            self.scheduler,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            metric=val_loss.mean(),
            logger=self.logger,
        )

        return val_loss, plot_outputs, plot_state, plot_background

    def _get_plot_outputs(self, condition, state, lead_time_label, reg_out):
        r"""
        Get outputs for validation plotting.

        For diffusion models, runs full reverse-ODE sampling.  For regression
        models, runs a forward pass to obtain the prediction.

        Parameters
        ----------
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
                self.net,
                condition,
                shape=state[1].shape,
                scheduler=self.sampling_scheduler,
                dtype=state[1].dtype,
                device=state[1].device,
                sampler_args=self.cfg.sampler.args.__dict__,
                lead_time_label=lead_time_label,
            )
            if "regression" in self.condition_list:
                outputs += reg_out
            return outputs
        else:
            labels = (
                {} if lead_time_label is None else {"lead_time_label": lead_time_label}
            )
            return self.net(x=condition, **labels)

    # =========================================================================
    # Logging
    # =========================================================================

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
        self.logger.info(" ".join(fields))

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

        All ranks participate; the checkpoint utilities handle gathering
        FSDP / ShardTensor state automatically and only rank 0 writes files.
        """
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
        Cleans up logger on exit.
        """
        self.logger.info(
            f"Training up to {self.total_train_steps} steps from step {self.total_steps}..."
        )
        # resetting in log_progress
        self.train_start = time.time()
        run_steps = 0
        max_run_steps = self.cfg.training.max_run_steps

        while self.total_steps < self.total_train_steps:
            # Training step
            self.logger.step = self.total_steps + 1
            loss = self.train_step()
            train_loss = loss.mean().cpu().item()
            self.avg_train_loss += train_loss
            self.train_steps += 1
            self.total_steps += 1

            # Logging
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.log_value("loss/train", train_loss)
            self.logger.log_value("lr", lr)

            # Validation
            if self.total_steps % self.cfg.training.validation_freq == 0:
                valid_start = time.time()
                val_loss_channel, plot_outputs, plot_state, plot_background = (
                    self.validate()
                )
                self.val_loss = float(val_loss_channel.mean())

                self.logger.log_value("loss/valid", self.val_loss)
                for ch, value in zip(self.state_channels, val_loss_channel):
                    self.logger.log_value(f"loss/valid/{ch}", value)

                if self.use_shard_tensor:
                    plot_outputs = (
                        None if plot_outputs is None else plot_outputs.full_tensor()
                    )
                    plot_state = (
                        None
                        if plot_state is None
                        else [
                            s.full_tensor() if s is not None else None
                            for s in plot_state
                        ]
                    )
                    plot_background = (
                        None
                        if plot_background is None
                        else plot_background.full_tensor()
                    )
                save_validation_plots(self, plot_outputs, plot_state, plot_background)
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

            self.logger.dump()

            run_steps += 1
            if (max_run_steps is not None) and run_steps >= max_run_steps:
                self.logger.info(f"Trained for max_run_steps={max_run_steps}, quitting")
                break

        # Cleanup
        self.logger.finalize()

        self.logger.info("\nExiting...")
