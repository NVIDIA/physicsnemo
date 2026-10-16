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

"""Trainer class for corrdiff-era5-hrrr training."""

from contextlib import nullcontext
import itertools
import math
import os
import time

from matplotlib import pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from physicsnemo.diffusion.metrics.losses import WeightedFlowMatchingLoss
from physicsnemo.diffusion.noise_schedulers import RectifiedFlowNoiseScheduler
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint, load_model_weights, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

from datasets import dataset_classes
from datasets.dataset import worker_init
from utils.nn import build_condition, get_dit, sample_flow
from utils.plots import comparison_plot


class ResumableSampler(DistributedSampler):
    r"""Distributed sampler that skips the first ``start`` indices of the next
    epoch, so that a resumed run continues where it stopped."""

    start = 0

    def __iter__(self):
        indices = list(super().__iter__())[self.start :]
        self.start = 0
        return iter(indices)


class Trainer:
    r"""
    Trainer for x-prediction rectified flow matching.

    The trainer draws times :math:`t` uniformly from
    :math:`[t_{\min}, t_{\max}]` and builds noisy states on the
    rectified-flow path
    :math:`\mathbf{x}_t = (1 - t) \mathbf{x}_0 + t \boldsymbol{\epsilon}`.
    The network predicts the clean state :math:`\hat{\mathbf{x}}_0` and the
    loss is the mean squared error
    :math:`\|\hat{\mathbf{x}}_0 - \mathbf{x}_0\|^2`.

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
        The GPU that this rank trains on.
    net : TimeScaledModel
        The network, as saved in the checkpoints.
    train_net : torch.nn.Module
        ``net`` wrapped in ``DistributedDataParallel`` on more than one GPU,
        otherwise ``net`` itself.
    optimizer : torch.optim.Optimizer
        AdamW optimizer.
    scheduler : torch.optim.lr_scheduler.LambdaLR
        Learning rate schedule.
    noise_scheduler : RectifiedFlowNoiseScheduler
        Noise scheduler of the loss and of the validation sampling.
    loss_fn : WeightedFlowMatchingLoss
        Training and validation loss.
    total_steps : int
        Number of training steps done so far.
    val_loss : float
        Most recent validation loss, or -1 before the first validation.

    Examples
    --------
    >>> from hydra import compose, initialize
    >>> from physicsnemo.distributed import DistributedManager
    >>> DistributedManager.initialize()
    >>> with initialize(version_base=None, config_path="config"):
    ...     cfg = compose(config_name="test_diffusion")  # mock data
    >>> trainer = Trainer(cfg)
    >>> trainer.train()
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dist = DistributedManager()
        self.device = self.dist.device
        self.logger = RankZeroLoggingWrapper(PythonLogger("train"), self.dist)
        self.start_time = time.time()

        self._parse_config()
        self._setup_data()

        # Same initialization on all ranks
        torch.manual_seed(cfg.training.seed)
        self.net = self._setup_model()
        (self.optimizer, self.scheduler) = self._setup_optimizer()
        (self.total_steps, self.val_loss) = self._resume_or_init()
        self.dataset_iterator = self._train_batches()

        # Data parallelism: DDP averages the gradients over the ranks
        self.train_net = self.net
        if self.dist.world_size > 1:
            self.train_net = DistributedDataParallel(
                self.net, device_ids=[self.dist.local_rank], broadcast_buffers=False
            )
        self.loss_fn = self._setup_loss()

        self.tensorboard = None
        if cfg.training.log_to_tensorboard and (self.dist.rank == 0):
            self.tensorboard = SummaryWriter(
                log_dir=os.path.join(cfg.training.rundir, "tensorboard")
            )

    # =========================================================================
    # Setup
    # =========================================================================

    def _parse_config(self):
        r"""Parse the batch sizes, the floating point precision, and the
        checkpoint path."""
        cfg = self.cfg.training

        # Batch sizes
        world_size = self.dist.world_size
        if cfg.batch_size % world_size != 0:
            raise ValueError(
                "training.batch_size must be divisible by the number of GPUs."
            )
        max_local_batch_size = cfg.batch_size // world_size
        if cfg.batch_size_per_gpu == "auto":
            self.local_batch_size = max_local_batch_size
        else:
            self.local_batch_size = min(cfg.batch_size_per_gpu, max_local_batch_size)
        if max_local_batch_size % self.local_batch_size != 0:
            raise ValueError(
                "training.batch_size / num_gpus must be divisible by training.batch_size_per_gpu."
            )
        self.num_accumulation_rounds = max_local_batch_size // self.local_batch_size

        # Floating point precision
        fp_optimizations = cfg.perf.fp_optimizations
        if fp_optimizations not in ("fp32", "amp-bf16"):
            raise ValueError(
                "training.perf.fp_optimizations must be 'fp32' or 'amp-bf16'."
            )
        self.enable_amp = fp_optimizations == "amp-bf16"

        self.ckpt_path = os.path.join(cfg.rundir, "checkpoints")

    def _setup_data(self):
        r"""Create the datasets and dataloaders."""
        self.logger.info("Loading dataset...")
        params = OmegaConf.to_container(self.cfg.dataset, resolve=True)
        dataset_cls = dataset_classes[params.pop("name")]
        self.dataset_train = dataset_cls(train=True, **params)
        self.dataset_valid = dataset_cls(train=False, **params)

        self.state_channels = self.dataset_train.state_channels()
        self.background_channels = self.dataset_train.background_channels()
        self.scalar_condition_channels = self.dataset_train.scalar_condition_channels()

        (rank, world_size) = (self.dist.rank, self.dist.world_size)
        num_workers = self.cfg.training.num_data_workers
        self.train_sampler = ResumableSampler(
            self.dataset_train,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=self.cfg.training.seed,
            drop_last=True,
        )
        self.train_dataloader = DataLoader(
            self.dataset_train,
            batch_size=self.local_batch_size,
            sampler=self.train_sampler,
            num_workers=num_workers,
            worker_init_fn=worker_init,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
        )
        if len(self.train_dataloader) == 0:
            raise ValueError(
                f"The training dataset has {len(self.dataset_train)} samples, "
                f"too few for one batch of {self.local_batch_size} on each GPU."
            )
        self.valid_dataloader = DataLoader(
            self.dataset_valid,
            batch_size=self.local_batch_size,
            sampler=DistributedSampler(
                self.dataset_valid, num_replicas=world_size, rank=rank, shuffle=False
            ),
            num_workers=num_workers,
            worker_init_fn=worker_init,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    def _train_batches(self):
        r"""Iterate over the training data indefinitely, reshuffling every epoch.
        A resumed run continues with the epoch and the batch where it stopped."""
        batches_done = self.total_steps * self.num_accumulation_rounds
        (epoch, batches_in_epoch) = divmod(batches_done, len(self.train_dataloader))
        self.train_sampler.start = batches_in_epoch * self.local_batch_size
        while True:
            self.train_sampler.set_epoch(epoch)
            yield from self.train_dataloader
            epoch += 1

    def _setup_model(self) -> torch.nn.Module:
        r"""Construct the network."""
        self.logger.info("Constructing network...")
        self.logger.info(f"Background channels: {self.background_channels}")
        self.logger.info(f"Scalar condition channels: {self.scalar_condition_channels}")
        self.logger.info(f"State channels: {self.state_channels}")

        net = get_dit(
            target_channels=len(self.state_channels),
            conditional_channels=len(self.background_channels),
            scalar_condition_channels=len(self.scalar_condition_channels),
            img_resolution=self.dataset_train.image_shape(),
            **OmegaConf.to_container(self.cfg.model.hyperparameters, resolve=True),
        )
        num_params = sum(p.numel() for p in net.parameters())
        self.logger.info(f"Number of parameters: {num_params / 1e6:.1f}M")
        return net.train().requires_grad_(True).to(self.device)

    def _setup_optimizer(
        self,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        r"""Create the AdamW optimizer and the learning rate schedule: a linear
        warmup followed by a cosine decay to zero or a constant learning rate."""
        cfg = self.cfg.training
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.betas),
            weight_decay=cfg.optimizer.weight_decay,
        )

        warmup_steps = cfg.scheduler.lr_rampup_steps
        if cfg.scheduler.lr_decay not in ("cosine", "constant"):
            raise ValueError(
                "training.scheduler.lr_decay must be 'cosine' or 'constant'."
            )

        def lr_factor(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            if cfg.scheduler.lr_decay == "constant":
                return 1.0
            progress = (step - warmup_steps) / max(
                cfg.total_train_steps - warmup_steps, 1
            )
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        return (optimizer, scheduler)

    def _resume_or_init(self) -> tuple[int, float]:
        r"""
        Resume from the latest checkpoint in the run directory, if one exists.
        Otherwise, load ``training.initial_weights`` if given.

        Returns
        -------
        total_steps : int
            The number of training steps done so far.
        val_loss : float
            The validation loss stored in the checkpoint, or -1 if there is none.
        """
        self.logger.info(f'Trying to resume from "{self.ckpt_path}"...')
        metadata = {}
        total_steps = load_checkpoint(
            path=self.ckpt_path,
            models=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metadata_dict=metadata,
            device=self.device,
        )
        if total_steps == 0:
            initial_weights = self.cfg.training.initial_weights
            if initial_weights is not None:
                if not os.path.isfile(initial_weights) or not initial_weights.endswith(
                    ".mdlus"
                ):
                    raise ValueError(
                        "training.initial_weights must point to a physicsnemo .mdlus checkpoint"
                    )
                self.logger.info(f"Loading initial weights from {initial_weights}...")
                load_model_weights(self.net, initial_weights)
            else:
                self.logger.info("Starting training from scratch...")
        return (total_steps, metadata.get("val_loss", -1.0))

    def _setup_loss(self) -> WeightedFlowMatchingLoss:
        r"""
        Create the x-prediction rectified-flow loss.

        The :class:`~physicsnemo.diffusion.noise_schedulers.RectifiedFlowNoiseScheduler`
        draws :math:`t` uniformly from :math:`[t_{\min}, t_{\max}]` and builds
        :math:`\mathbf{x}_t = (1 - t) \mathbf{x}_0 + t \boldsymbol{\epsilon}`
        with unit loss weight. With ``prediction_type="x0"``, the
        :class:`~physicsnemo.diffusion.metrics.losses.WeightedFlowMatchingLoss`
        passes the network output and the clean state through
        ``x0_to_flow_fn`` and compares the results. The identity function
        compares them directly, so the per-element loss is
        ``weight * (x0_pred - x0) ** 2``. With the all-ones ``weight`` that the
        trainer passes, it equals the loss of
        :class:`~physicsnemo.diffusion.metrics.losses.MSEDSMLoss`.
        It stays finite for all :math:`t` up to the pure-noise endpoint
        :math:`t = 1`.
        """
        loss_cfg = self.cfg.training.loss
        self.logger.info(
            f"Using the x-prediction rectified flow loss with t in [{loss_cfg.t_min}, {loss_cfg.t_max}]"
        )
        self.noise_scheduler = RectifiedFlowNoiseScheduler(
            t_min=loss_cfg.t_min, t_max=loss_cfg.t_max
        )
        return WeightedFlowMatchingLoss(
            self.train_net,
            self.noise_scheduler,
            prediction_type="x0",
            # Compare x0_pred with x0 directly; the rectified flow
            # noise_scheduler.x0_to_flow would weight the error by 1 / t^2
            x0_to_flow_fn=lambda x0, x_t, t: x0,
            reduction="none",
        )

    def _setup_seeds(self, step: int):
        r"""Seed the random number generators differently for each step and rank."""
        seed = self.cfg.training.seed + step * self.dist.world_size + self.dist.rank
        np.random.seed(seed % (1 << 31))
        torch.manual_seed(seed)

    def _unpack_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | TensorDict]:
        r"""Move a batch to the GPU and split it into the target and the condition."""
        (target, background) = (
            batch[k].to(self.device, non_blocking=True) for k in ("state", "background")
        )
        scalar_conditions = batch.get("scalar_conditions")
        if scalar_conditions is not None:
            scalar_conditions = scalar_conditions.to(self.device, non_blocking=True)
        return (target, build_condition(background, scalar_conditions))

    # =========================================================================
    # Training and validation
    # =========================================================================

    def train_step(self) -> float:
        r"""
        Execute a single training step with gradient accumulation.

        Returns
        -------
        float
            The training loss averaged over the batch and the ranks.
        """
        self._setup_seeds(self.total_steps)
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=self.device)

        for i in range(self.num_accumulation_rounds):
            (target, condition) = self._unpack_batch(next(self.dataset_iterator))

            # Average the gradients over the ranks only in the last round
            last_round = i == self.num_accumulation_rounds - 1
            sync_context = (
                nullcontext()
                if last_round or self.train_net is self.net
                else self.train_net.no_sync()
            )
            # Same loss weight for every element; the weight could, for
            # example, mask out regions or weight channels differently
            weight = torch.ones_like(target)
            with sync_context:
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=self.enable_amp
                ):
                    loss = self.loss_fn(target, weight, condition=condition).mean()
                (loss / self.num_accumulation_rounds).backward()
            loss_sum += loss.detach()

        # Replace non-finite gradients, clip the gradient norm, and update the weights
        for param in self.net.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                )
        if self.cfg.training.clip_grad_norm > 0:
            clip_grad_norm_(self.net.parameters(), self.cfg.training.clip_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        loss = loss_sum / self.num_accumulation_rounds
        if self.dist.world_size > 1:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        return loss.item()

    @torch.no_grad()
    def validate(self) -> torch.Tensor:
        r"""
        Compute the validation loss over ``training.validation_steps`` batches
        and plot a sample generated for the first validation batch.

        Returns
        -------
        torch.Tensor
            The validation loss for each state channel.
        """
        # Fixed seed, so that the validation loss always uses the same noise
        torch.manual_seed(self.cfg.training.seed + self.dist.rank)
        self.net.eval()

        channel_loss = torch.zeros(len(self.state_channels), device=self.device)
        num_batches = 0
        for batch in itertools.islice(
            self.valid_dataloader, self.cfg.training.validation_steps
        ):
            (target, condition) = self._unpack_batch(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.enable_amp):
                loss = self.loss_fn(
                    target, torch.ones_like(target), condition=condition
                )
            channel_loss += loss.mean(dim=(0, 2, 3))
            if (num_batches == 0) and (self.dist.rank == 0):
                self._save_validation_plots(
                    target[:1], condition[:1], batch["background"][:1]
                )
            num_batches += 1

        channel_loss /= max(num_batches, 1)
        if self.dist.world_size > 1:
            torch.distributed.all_reduce(
                channel_loss, op=torch.distributed.ReduceOp.AVG
            )
        self.net.train()
        return channel_loss

    def _save_validation_plots(
        self,
        target: torch.Tensor,
        condition: torch.Tensor | TensorDict,
        background: torch.Tensor,
    ):
        r"""Generate a sample for one validation example and plot it next to
        the background and the truth."""
        sampler_args = OmegaConf.to_container(self.cfg.sampler.args, resolve=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.enable_amp):
            generated = sample_flow(
                self.net, condition, target.shape, self.noise_scheduler, **sampler_args
            )

        dataset = self.dataset_valid
        generated = dataset.denormalize_state(generated[0].float().cpu().numpy())
        truth = dataset.denormalize_state(target[0].cpu().numpy())
        background = dataset.denormalize_background(background[0].numpy())

        images_path = os.path.join(self.cfg.training.rundir, "images")
        os.makedirs(images_path, exist_ok=True)
        for variable in self.cfg.training.validation_plot_variables:
            c = self.state_channels.index(variable)
            panels = [("Generated", generated[c]), ("Truth", truth[c])]
            if variable in self.background_channels:
                c_bg = self.background_channels.index(variable)
                panels.insert(0, ("Background", background[c_bg]))
            fig = comparison_plot(
                panels, variable, title=f"{variable}, step {self.total_steps}"
            )
            fig.savefig(
                os.path.join(images_path, f"{variable}_{self.total_steps:09d}.png")
            )
            if self.tensorboard is not None:
                self.tensorboard.add_figure(
                    f"validation/{variable}", fig, self.total_steps
                )
            plt.close(fig)

    # =========================================================================
    # Logging and checkpointing
    # =========================================================================

    def log_value(self, key: str, value: float):
        r"""Log a value to TensorBoard, if enabled."""
        if self.tensorboard is not None:
            self.tensorboard.add_scalar(key, value, self.total_steps)

    def log_progress(self, train_loss: float, step_time: float):
        r"""Print a summary line of the training progress."""
        fields = [
            f"steps {self.total_steps:<7d}",
            f"samples {self.total_steps * self.cfg.training.batch_size}",
            f"tot_time {time.time() - self.start_time:.1f}",
            f"step_time {step_time:.3f}",
            f"gpumem {torch.cuda.max_memory_allocated(self.device) / 2**30:.2f}",
            f"lr {self.optimizer.param_groups[0]['lr']:.4g}",
            f"train_loss {train_loss:.5f}",
            f"val_loss {self.val_loss:.5f}",
        ]
        self.logger.info(" ".join(fields))
        torch.cuda.reset_peak_memory_stats(self.device)

    def save_checkpoint(self):
        r"""Save the model and training state. Every rank holds the same
        model, so only rank 0 writes the files."""
        if self.dist.rank == 0:
            save_checkpoint(
                path=self.ckpt_path,
                models=self.net,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=self.total_steps,
                metadata={"val_loss": self.val_loss},
            )
            self._remove_old_checkpoints()
        if self.dist.world_size > 1:
            torch.distributed.barrier()

    def _remove_old_checkpoints(self):
        r"""Keep only the ``training.checkpoints_to_keep`` most recent pairs of
        model and training state files; ``None`` keeps every checkpoint."""
        keep = self.cfg.training.checkpoints_to_keep
        if keep is None:
            return
        steps = sorted(
            int(name.split(".")[-2])
            for name in os.listdir(self.ckpt_path)
            if name.startswith("checkpoint.0.") and name.endswith(".pt")
        )
        model_name = type(self.net).__name__
        for step in steps[: max(len(steps) - keep, 0)]:
            for name in (f"checkpoint.0.{step}.pt", f"{model_name}.0.{step}.mdlus"):
                path = os.path.join(self.ckpt_path, name)
                if os.path.exists(path):
                    os.remove(path)

    # =========================================================================
    # Main training loop
    # =========================================================================

    def train(self):
        r"""
        Main training loop.

        Runs training up to ``training.total_train_steps``, with validation,
        logging, and checkpointing at the configured frequencies.
        """
        cfg = self.cfg.training
        self.logger.info(
            f"Training up to {cfg.total_train_steps} steps from step {self.total_steps}..."
        )
        (loss_sum, num_steps, interval_start) = (0.0, 0, time.time())

        while self.total_steps < cfg.total_train_steps:
            train_loss = self.train_step()
            self.total_steps += 1
            (loss_sum, num_steps) = (loss_sum + train_loss, num_steps + 1)
            self.log_value("loss/train", train_loss)
            self.log_value("lr", self.optimizer.param_groups[0]["lr"])

            if self.total_steps % cfg.validation_freq == 0:
                channel_loss = self.validate().cpu().numpy()
                self.val_loss = float(channel_loss.mean())
                self.log_value("loss/valid", self.val_loss)
                for channel, value in zip(
                    self.state_channels, channel_loss, strict=True
                ):
                    self.log_value(f"loss/valid/{channel}", value)

            if self.total_steps % cfg.print_progress_freq == 0:
                step_time = (time.time() - interval_start) / num_steps
                self.log_progress(loss_sum / num_steps, step_time)
                (loss_sum, num_steps, interval_start) = (0.0, 0, time.time())

            if (self.total_steps % cfg.checkpoint_freq == 0) or (
                self.total_steps == cfg.total_train_steps
            ):
                self.save_checkpoint()

        if self.tensorboard is not None:
            self.tensorboard.close()
        self.logger.info("Training finished.")
