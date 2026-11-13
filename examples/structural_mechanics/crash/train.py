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

"""Distributed training loop of the crash recipes.

``Trainer`` builds the datasets, the model and the optimizer from a Hydra
recipe, runs one optimizer step per ``SimSample`` and validates with the pooled
normalized MSE; a ``Trainer`` subclass named by ``training.trainer_class`` may
extend the validation (the PX Yaris recipes do). The loop knows no model or
dataset by name: every model family is served through the same
``model(sample=..., data_stats=...)`` call (one-shot models return ``[N, T, F]``,
time-conditional models one frame ``[N, 3]`` in training, optionally as a tuple
with an ``auxiliary_prediction``, and the trajectory ``[N, T, 3]`` in
evaluation), and model wrappers declare the sample kind they consume
(``input_kind``) while datapipes declare the kinds they serve (``sample_kinds``).
"""

import logging
import math
import os
import random
import sys
import time
from copy import deepcopy

sys.path.insert(0, os.path.dirname(__file__))

import hydra
import numpy as np
import omegaconf
import torch
from hydra.utils import get_class, get_object, instantiate
from omegaconf import DictConfig
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

# Optional: tabulate for metrics tables, torchinfo for model summary
_tabulate = OptionalImport("tabulate")
_torchinfo = OptionalImport("torchinfo")

from datapipes.datapipe import GLOBAL_STATS_FILE, SimSample, simsample_collate
from metrics.losses import PositionLoss
from omegaconf import open_dict
from utils.distributed import initialize_datasets
from utils.optimizers import build_muon_optimizer
from utils.random_state import (
    configure_deterministic_training,
    gather_rng_states,
    restore_rank_rng,
)
from utils.warm_start import load_warm_start


class ValidationEarlyStopping:
    """Track validation improvements and stop after a bounded stale period."""

    def __init__(self, patience: int = 0, min_delta: float = 0.0) -> None:
        if patience < 0:
            raise ValueError("early_stopping_patience cannot be negative")
        if min_delta < 0.0:
            raise ValueError("early_stopping_min_delta cannot be negative")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = math.inf
        self.stale_evaluations = 0

    @property
    def enabled(self) -> bool:
        """Return whether a positive early-stopping patience was configured."""
        return self.patience > 0

    def update(self, metric: float) -> tuple[bool, bool]:
        """Return ``(should_stop, improved)`` for a validation metric."""

        improved = math.isfinite(metric) and metric < self.best - self.min_delta
        if improved:
            self.best = metric
            self.stale_evaluations = 0
        else:
            self.stale_evaluations += 1
        should_stop = self.enabled and self.stale_evaluations >= self.patience
        return should_stop, improved


def average_model_gradients(model: torch.nn.Module, world_size: int) -> None:
    """Synchronously average dense gradients after backward releases activations.

    Every trainable parameter takes part, a missing gradient as zeros, so all
    ranks pack the same flat buffer even when a data-dependent branch left a
    module unused on one of them (a mismatch would abort under Gloo and hang
    under NCCL). Such a parameter ends up with the average of the other ranks'
    gradients.
    """

    if world_size <= 1:
        return
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable parameters for distributed gradient averaging")

    groups: dict[tuple[torch.device, torch.dtype], list[torch.nn.Parameter]] = {}
    for parameter in parameters:
        groups.setdefault((parameter.device, parameter.dtype), []).append(parameter)
    for grouped in groups.values():
        flat = torch.cat(
            [
                (
                    p.grad.reshape(-1)
                    if p.grad is not None
                    else torch.zeros(p.numel(), device=p.device, dtype=p.dtype)
                )
                for p in grouped
            ]
        )
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat.div_(world_size)
        offset = 0
        for p in grouped:
            count = p.numel()
            chunk = flat[offset : offset + count].view_as(p)
            if p.grad is None:
                p.grad = chunk.clone()
            else:
                p.grad.copy_(chunk)
            offset += count


def check_global_stats_for_resume(cfg) -> None:
    """Refuse to resume a run whose stats directory predates global-feature standardization.

    The datapipe standardizes the global features with train-split statistics
    stored in ``stats/global_stats.json``. A run directory written before that
    holds checkpoints but no such file; recomputing it silently would resume
    the run on rescaled inputs.
    """
    if not cfg.datapipe.get("global_features_filepath"):
        return
    ckpt_path = str(cfg.training.ckpt_path)
    has_checkpoint = os.path.isdir(ckpt_path) and any(
        name.endswith((".pt", ".mdlus")) for name in os.listdir(ckpt_path)
    )
    stats_path = os.path.join(
        str(cfg.datapipe.get("stats_dir", "stats")), GLOBAL_STATS_FILE
    )
    if has_checkpoint and not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"{ckpt_path} holds a checkpoint but {stats_path} does not exist: this run "
            "directory predates the standardization of the global features with "
            "train-split statistics, and resuming would recompute them and continue "
            "on rescaled inputs. Start the run afresh, or point training.ckpt_path "
            "and datapipe.stats_dir at a new run directory."
        )


def resolve_trainer_class(path: str | None):
    """Return the Trainer subclass named by ``training.trainer_class`` (default Trainer)."""
    if path is None:
        return Trainer
    trainer_class = get_class(str(path))

    # When train.py runs as a script its Trainer lives in ``__main__`` while
    # subclasses import ``train.Trainer``; compare by identity of the class
    # definition (name and defining file), not by object identity.
    def is_trainer(cls):
        return cls.__name__ == "Trainer" and cls.__module__ in ("train", "__main__")

    if not (
        isinstance(trainer_class, type)
        and any(is_trainer(base) for base in trainer_class.__mro__)
    ):
        raise TypeError(f"training.trainer_class={path!r} is not a Trainer subclass")
    return trainer_class


def _warmup_cosine_factor(warmup_epochs, epochs, start_lr, end_lr):
    """Per-epoch multiplier for LambdaLR: 1 during the warmup epochs (the
    per-step linear rise is applied by the trainer), then the cosine from
    ``start_lr`` to ``end_lr`` over the remaining epochs, reaching ``end_lr``
    after the last one exactly like CosineAnnealingLR(T_max=epochs)."""
    span = max(1, epochs - warmup_epochs)

    def factor(epoch):
        if epoch < warmup_epochs:
            return 1.0
        progress = min(1.0, (epoch - warmup_epochs) / span)
        lr = end_lr + (start_lr - end_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr / start_lr

    return factor


def trainable_parameters(model):
    """The parameters an optimizer should update: a frozen teacher's are skipped."""
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


class Trainer:
    """Trainer for crash simulation models with unified SimSample input.

    Parameters
    ----------
    cfg : DictConfig
        The composed Hydra recipe (``reader``, ``datapipe``, ``model``,
        ``training``, ``inference``).
    logger0 : RankZeroLoggingWrapper
        Logger that prints on rank zero.

    Training options read from ``cfg.training`` beyond the defaults of
    ``conf/training/default.yaml``: ``loss`` / ``loss_space`` / ``loss_scale_mm``
    (:class:`metrics.losses.PositionLoss`), ``auxiliary_loss_weight`` (position loss on a
    model's ``auxiliary_prediction``), ``lr_warmup_epochs``, ``warm_start_model`` (initialise from another
    run's weights), ``manual_gradient_allreduce``, ``reproducible_sampling`` /
    ``sampler_seed``, ``deterministic`` and ``early_stopping_*``.
    """

    def __init__(self, cfg: DictConfig, logger0: RankZeroLoggingWrapper):
        if not DistributedManager.is_initialized():
            raise RuntimeError(
                "DistributedManager must be initialized before the trainer"
            )
        self.dist = DistributedManager()
        self.cfg = cfg
        self.rollout_steps = cfg.training.num_time_steps - 1
        if self.rollout_steps < 1:
            raise ValueError("training.num_time_steps must be at least 2")
        self.amp = cfg.training.amp
        amp_dtype_name = cfg.training.get("amp_dtype", "float16")
        amp_dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        if amp_dtype_name not in amp_dtypes:
            raise ValueError(
                "training.amp_dtype must be 'float16' or 'bfloat16'; "
                f"got {amp_dtype_name!r}"
            )
        self.amp_dtype = amp_dtypes[amp_dtype_name]
        requested_manual_allreduce = bool(
            cfg.training.get("manual_gradient_allreduce", False)
        )
        self.manual_gradient_allreduce = (
            requested_manual_allreduce and self.dist.world_size > 1
        )
        if (
            self.manual_gradient_allreduce
            and self.amp
            and self.amp_dtype == torch.float16
        ):
            raise ValueError(
                "manual_gradient_allreduce currently requires bfloat16 or fp32"
            )
        self.last_loss_components: dict[str, torch.Tensor] = {}
        # Models may return an auxiliary prediction next to the final one (for
        # example a provisional position head); its position loss enters with
        # this weight, 0 by default.
        self.auxiliary_loss_weight = float(
            cfg.training.get("auxiliary_loss_weight", 0.0)
        )
        if self.auxiliary_loss_weight < 0.0:
            raise ValueError("training.auxiliary_loss_weight cannot be negative")

        # --- Consistency check between model and datapipe ---
        # Model wrappers declare the sample kind they consume (``input_kind``:
        # "graph" or "points") and datapipes the kinds they serve
        # (``sample_kinds``); targets without the attributes, factory functions
        # included, are not checked.
        model_class = get_object(cfg.model._target_)
        datapipe_class = get_object(cfg.datapipe._target_)
        input_kind = getattr(model_class, "input_kind", None)
        sample_kinds = getattr(datapipe_class, "sample_kinds", None)
        if (
            input_kind is not None
            and sample_kinds is not None
            and input_kind not in sample_kinds
        ):
            raise ValueError(
                f"Model {cfg.model._target_} consumes {input_kind!r} samples, but "
                f"the datapipe {cfg.datapipe._target_} serves {list(sample_kinds)}."
            )

        # Dataset
        reader = instantiate(cfg.reader)
        logging.getLogger().setLevel(logging.INFO)
        dataset_kwargs = {
            "name": "crash_train",
            "reader": reader,
            "split": "train",
            "logger": logger0,
        }
        val_cfg = None
        if cfg.training.num_validation_samples > 0:
            self.num_validation_replicas = min(
                self.dist.world_size, cfg.training.num_validation_samples
            )
            self.num_validation_samples = (
                cfg.training.num_validation_samples
                // self.num_validation_replicas
                * self.num_validation_replicas
            )
            logger0.info(f"Number of validation samples: {self.num_validation_samples}")
            val_cfg = deepcopy(cfg.datapipe)
            with open_dict(val_cfg):
                val_cfg.data_dir = cfg.training.raw_data_dir_validation
                val_cfg.num_samples = self.num_validation_samples

        def build_datasets(stats_mode):
            training = instantiate(
                cfg.datapipe, **dataset_kwargs, stats_mode=stats_mode
            )
            validation = None
            if val_cfg is not None:
                validation = instantiate(
                    val_cfg,
                    name="crash_validation",
                    reader=reader,
                    split="validation",
                    logger=logger0,
                    stats_mode="load",
                    sample_type="all_time_steps",
                )
            return training, validation

        # A corrector recipe names the frozen network's checkpoint; a wrong path
        # fails here, before the datasets start, not minutes later.
        if "teacher_weights" in cfg.model and not omegaconf.OmegaConf.is_missing(
            cfg.model, "teacher_weights"
        ):
            teacher_weights = str(cfg.model.teacher_weights)
            if not os.path.isfile(teacher_weights):
                raise FileNotFoundError(
                    f"model.teacher_weights={teacher_weights} does not exist; it must "
                    "be the trained network's .mdlus file"
                )
        # A run directory from before global-feature standardization fails here
        # instead of resuming on silently rescaled inputs.
        check_global_stats_for_resume(cfg)
        # Includes validation warmup: no rank may reach model NCCL broadcasts
        # while another is still building static masks on the CPU.
        dataset, val_dataset = initialize_datasets(
            build_datasets,
            timeout_seconds=float(
                cfg.training.get("data_startup_timeout_seconds", 3600)
            ),
        )
        logger0.info(
            "CPU dataset startup complete on all ranks; GPU training may start"
        )
        sample_target = dataset[0].node_target
        self.target_channels = int(sample_target.shape[-1])
        logging.getLogger().setLevel(logging.INFO)
        # Move stats to device
        self.data_stats = dict(
            node={k: v.to(self.dist.device) for k, v in dataset.node_stats.items()},
            edge={
                k: v.to(self.dist.device)
                for k, v in getattr(dataset, "edge_stats", {}).items()
            },
            feature={
                k: v.to(self.dist.device)
                for k, v in getattr(dataset, "feature_stats", {}).items()
            },
            global_features={
                "mean": dataset.global_stats["global_mean"].to(self.dist.device),
                "std": dataset.global_stats["global_std"].to(self.dist.device),
                "keys": list(dataset.global_features_keys or []),
            },
        )

        # Sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.dist.world_size,
            rank=self.dist.rank,
            shuffle=True,
            seed=int(cfg.training.get("sampler_seed", 0)),
        )

        self.loader_generators = {}
        if cfg.training.get("reproducible_sampling", False):
            for index, name in enumerate(("train", "validation")):
                self.loader_generators[name] = torch.Generator().manual_seed(
                    int(cfg.training.get("seed", 42))
                    + self.dist.rank
                    + (index + 1) * 1000003
                )

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,  # variable N per sample
            shuffle=(sampler is None),
            drop_last=True,
            pin_memory=True,
            num_workers=cfg.training.num_dataloader_workers,
            sampler=sampler,
            collate_fn=simsample_collate,
            generator=self.loader_generators.get("train"),
        )
        self.sampler = sampler

        if cfg.training.num_validation_samples > 0:
            if self.dist.rank < self.num_validation_replicas:
                # Sampler
                if self.dist.world_size > 1:
                    sampler = DistributedSampler(
                        val_dataset,
                        num_replicas=self.num_validation_replicas,
                        rank=self.dist.rank,
                        shuffle=False,
                        drop_last=True,
                    )
                else:
                    sampler = None

                self.val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=1,  # variable N per sample
                    shuffle=(sampler is None),
                    drop_last=True,
                    pin_memory=True,
                    num_workers=cfg.training.num_dataloader_workers,
                    sampler=sampler,
                    collate_fn=simsample_collate,
                    generator=self.loader_generators.get("validation"),
                )
            else:
                self.val_dataloader = torch.utils.data.DataLoader(
                    torch.utils.data.Subset(val_dataset, []), batch_size=1
                )

        # Model
        self.model = instantiate(cfg.model)
        logging.getLogger().setLevel(logging.INFO)
        warm_start = cfg.training.get("warm_start_model")
        if warm_start:
            # Fresh run initialised from another run's weights; a resumed run's
            # own checkpoint (loaded below) takes precedence.
            report = load_warm_start(self.model, warm_start)
            logger0.info(
                f"Warm start from {warm_start}: {len(report['loaded'])} tensors loaded"
            )
        self.model.to(self.dist.device)
        self.model.train()

        # Log model summary and parameter count (optional: torchinfo)
        if self.dist.rank == 0:
            num_params = sum(p.numel() for p in self.model.parameters())
            num_trainable = sum(p.numel() for p in trainable_parameters(self.model))
            logger0.info(
                f"Model parameters: {num_params:,} ({num_trainable:,} trainable)"
            )
            if _torchinfo.available:
                try:
                    logger0.info(f"\n{_torchinfo.summary(self.model, verbose=0)}")
                except Exception:
                    logger0.info(
                        "(torchinfo summary skipped: model requires sample input)"
                    )

        # Memory-bound meshes synchronize after backward so NCCL workspace does
        # not overlap peak rollout activations. Rank-zero state is broadcast once,
        # then gradients are averaged before every optimizer step.
        if self.dist.world_size > 1:
            if self.manual_gradient_allreduce:
                with torch.no_grad():
                    for tensor in list(self.model.parameters()) + list(
                        self.model.buffers()
                    ):
                        torch.distributed.broadcast(tensor, src=0)
                logger0.info("Using post-backward synchronous gradient averaging")
            else:
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[self.dist.local_rank],
                    output_device=self.dist.device,
                    broadcast_buffers=self.dist.broadcast_buffers,
                    find_unused_parameters=self.dist.find_unused_parameters,
                    gradient_as_bucket_view=True,
                )

        # Loss (defaults reproduce the plain MSE on standardized positions)
        self.criterion = torch.nn.MSELoss()
        self._configure_position_loss(cfg)

        # Optimizer (Muon requires PyTorch >= 2.9); frozen parameters are skipped.
        opt_name = cfg.training.get("optimizer", "adam")
        if opt_name not in ["adam", "adamw", "muon"]:
            raise ValueError(f"Unsupported optimizer: {opt_name}")
        if opt_name == "muon":
            self.optimizer = build_muon_optimizer(self.model, cfg)
        elif opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                trainable_parameters(self.model),
                lr=cfg.training.start_lr,
                weight_decay=cfg.training.optimizer_weight_decay,
                fused=torch.cuda.is_available(),
            )
        else:
            self.optimizer = torch.optim.Adam(
                trainable_parameters(self.model),
                lr=cfg.training.start_lr,
                fused=torch.cuda.is_available(),
            )
        logger0.info(f"Using {self.optimizer.__class__.__name__} optimizer")

        # Scheduler: cosine per epoch; with training.lr_warmup_epochs > 0 the
        # learning rate first rises linearly per optimizer step over those epochs
        # (applied in train()), then the cosine runs over the remaining epochs.
        self.lr_warmup_epochs = int(cfg.training.get("lr_warmup_epochs", 0))
        if not 0 <= self.lr_warmup_epochs < int(cfg.training.epochs):
            raise ValueError(
                "training.lr_warmup_epochs must be non-negative and below training.epochs"
            )
        self._lr_warmup_progress = (-1, 0)  # (epoch, optimizer steps seen in it)
        if self.lr_warmup_epochs == 0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.end_lr
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                _warmup_cosine_factor(
                    self.lr_warmup_epochs,
                    int(cfg.training.epochs),
                    float(cfg.training.start_lr),
                    float(cfg.training.end_lr),
                ),
            )
        self.scaler = GradScaler(
            "cuda", enabled=self.amp and self.amp_dtype == torch.float16
        )

        # Checkpoint
        self.checkpoint_metadata = {}
        if self.dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            cfg.training.ckpt_path,
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=cfg.training.get("load_epoch"),
            device=self.dist.device,
            metadata_dict=self.checkpoint_metadata,
        )

        if self.dist.rank == 0:
            self.writer = SummaryWriter(log_dir=cfg.training.tensorboard_log_dir)
        # Restore last, after all constructor/loading side effects. Each rank
        # restores its own stream, never rank zero's stream broadcast to all.
        if self.epoch_init:
            if cfg.training.get(
                "deterministic", False
            ) and not self.checkpoint_metadata.get("deterministic_algorithms", False):
                raise ValueError(
                    "Deterministic restart requires a deterministic-execution checkpoint"
                )
            states = self.checkpoint_metadata.get("rng_states")
            if states is None:
                if cfg.training.get("reproducible_sampling", False):
                    raise ValueError(
                        "Reproducible restart requires a v2 RNG checkpoint"
                    )
                logger0.warning(
                    "Checkpoint has no RNG state; restart is not replay-equivalent"
                )
            elif not cfg.training.get("restore_rng", True):
                # Evaluation (or a restart at another world size) loads weights
                # only; replay equivalence is not needed there.
                logger0.info("RNG state not restored (training.restore_rng=false)")
            else:
                # Replay equivalence is only promised with reproducible sampling
                # or deterministic execution; a plain restart at another world
                # size goes on with fresh streams.
                replay_required = bool(
                    cfg.training.get("reproducible_sampling", False)
                    or cfg.training.get("deterministic", False)
                )
                restored = restore_rank_rng(
                    states,
                    self.dist.rank,
                    self.dist.world_size,
                    self.loader_generators,
                    required=replay_required,
                )
                if not restored:
                    logger0.warning(
                        f"Checkpoint holds RNG states for {len(states)} ranks, this "
                        f"restart has {self.dist.world_size}; the per-rank RNG states "
                        "were not restored and the restart is not replay-equivalent "
                        "(training.restore_rng=false silences this)"
                    )

    def _configure_position_loss(self, cfg):
        """``training.loss`` / ``loss_space`` / ``loss_scale_mm``."""
        self.position_loss = PositionLoss(
            kind=cfg.training.get("loss", "mse"),
            space=cfg.training.get("loss_space", "normalized"),
            scale_mm=float(cfg.training.get("loss_scale_mm", 10.0)),
            pos_std=self.data_stats["node"].get("pos_std"),
        )

    def _loss_reference(self, sample):
        """The frame the position targets are displacements from, for the
        relative L2 loss: ``node_features["coords"]`` (the undeformed mesh for a
        time-conditional sample, the current frame for a rollout). ``None`` for
        the other loss kinds, which do not use it."""
        if self.position_loss.kind != "relative_l2":
            return None
        features = getattr(sample, "node_features", None)
        coords = features.get("coords") if features is not None else None
        if coords is None:
            raise ValueError("training.loss=relative_l2 needs node_features['coords']")
        return coords

    def _supervised_position_loss(self, prediction, target, sample):
        """Position loss with the configured kind and space."""
        return self.position_loss(
            prediction, target, reference=self._loss_reference(sample)
        )

    def _apply_lr_warmup(self, epoch: int) -> None:
        """Linear per-step warmup to each group's base rate during the first
        ``lr_warmup_epochs`` epochs; the epoch scheduler takes over afterwards."""
        if self.lr_warmup_epochs <= 0 or epoch >= self.lr_warmup_epochs:
            return
        steps_per_epoch = max(1, len(self.dataloader))
        last_epoch, seen = self._lr_warmup_progress
        seen = seen + 1 if last_epoch == epoch else 1
        self._lr_warmup_progress = (epoch, seen)
        scale = min(
            1.0,
            (epoch * steps_per_epoch + seen)
            / (self.lr_warmup_epochs * steps_per_epoch),
        )
        for group in self.optimizer.param_groups:
            group["lr"] = group["initial_lr"] * scale

    def train(self, sample: SimSample, epoch: int = 0):
        """One optimizer step on ``sample``; returns the detached loss."""
        self.optimizer.zero_grad()
        self._apply_lr_warmup(epoch)
        loss = self.forward(sample)
        self.backward(loss)
        return loss.detach()

    def forward(self, sample: SimSample):
        """The training loss of one sample (every model family)."""
        with autocast(device_type="cuda", enabled=self.amp, dtype=self.amp_dtype):
            model_kwargs = {}
            if self.auxiliary_loss_weight > 0.0:
                model_kwargs["return_auxiliary"] = True
            model_output = self.model(
                sample=sample, data_stats=self.data_stats, **model_kwargs
            )
            target = sample.node_target
            if hasattr(model_output, "auxiliary_prediction"):
                # The model also returned an auxiliary prediction (for example a
                # provisional position): supervise it with the configured weight.
                prediction_loss = self._supervised_position_loss(
                    model_output.prediction, target, sample
                )
                auxiliary_loss = self.position_loss(
                    model_output.auxiliary_prediction,
                    target,
                    reference=self._loss_reference(sample),
                )
                self.last_loss_components = {
                    "trajectory": prediction_loss.detach(),
                    "auxiliary": auxiliary_loss.detach(),
                }
                return prediction_loss + self.auxiliary_loss_weight * auxiliary_loss
            trajectory_loss = self._supervised_position_loss(
                model_output, target, sample
            )
            self.last_loss_components = {"trajectory": trajectory_loss.detach()}
            return trajectory_loss

    def backward(self, loss):
        """Backward pass, gradient averaging and the optimizer step."""
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            if self.manual_gradient_allreduce:
                self.scaler.unscale_(self.optimizer)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                average_model_gradients(self.model, self.dist.world_size)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.manual_gradient_allreduce:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                average_model_gradients(self.model, self.dist.world_size)
            self.optimizer.step()

    @torch.no_grad()
    def validate(self, epoch):
        """Pooled normalized MSE over the validation loader (per time step and channel)."""
        self.model.eval()

        MSE = torch.zeros(1, device=self.dist.device)
        MSE_w_time = torch.zeros(self.rollout_steps, device=self.dist.device)
        MSE_w_channel = torch.zeros(self.target_channels, device=self.dist.device)
        MSE_w_time_channel = torch.zeros(
            self.rollout_steps,
            self.target_channels,
            device=self.dist.device,
        )
        for sample in self.val_dataloader:
            sample = sample[0].to(self.dist.device)  # SimSample .to()

            # Model forward - returns [N, T, Fo]
            with autocast(device_type="cuda", enabled=self.amp, dtype=self.amp_dtype):
                pred = self.model(sample=sample, data_stats=self.data_stats)

            # Target is [N, T, Fo]
            target = sample.node_target

            # Compute and add error
            SqError = torch.square(pred - target)
            MSE_w_time += torch.mean(
                SqError, dim=(0, 2)
            )  # mean over N, Fo per timestep
            MSE_w_channel += torch.mean(SqError, dim=(0, 1))
            MSE_w_time_channel += torch.mean(SqError, dim=0)
            MSE += torch.mean(SqError)

        # Sum errors across all ranks
        if self.dist.world_size > 1:
            torch.distributed.all_reduce(MSE, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(MSE_w_time, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(
                MSE_w_channel, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                MSE_w_time_channel, op=torch.distributed.ReduceOp.SUM
            )

        val_stats = {
            "MSE_w_time": MSE_w_time / self.num_validation_samples,
            "MSE_w_channel": MSE_w_channel / self.num_validation_samples,
            "MSE_w_time_channel": (MSE_w_time_channel / self.num_validation_samples),
            "MSE": MSE / self.num_validation_samples,
        }

        self.model.train()  # Switch back to training mode
        return val_stats


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run distributed crash training from a Hydra experiment configuration."""
    configure_deterministic_training(bool(cfg.training.get("deterministic", False)))
    DistributedManager.initialize()
    dist = DistributedManager()

    seed = int(cfg.training.get("seed", 42))
    rank_seed = seed + dist.rank
    random.seed(rank_seed)
    np.random.seed(rank_seed % (2**32))
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)

    logger = PythonLogger("main")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger0.file_logging()

    # Log full config and paths
    logger0.info(f"Config:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")
    logger0.info(f"Output directory: {cfg.training.tensorboard_log_dir}")
    logger0.info(f"Checkpoint directory: {cfg.training.ckpt_path}")
    logger0.info(f"Random seed: {seed} (rank seed: {rank_seed})")
    logger0.info(
        f"Deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}; "
        f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}"
    )
    stats_dir = getattr(cfg.datapipe, "stats_dir")
    logger0.info(f"Stats directory: {stats_dir}")

    trainer = resolve_trainer_class(cfg.training.get("trainer_class"))(cfg, logger0)
    early_stopping = ValidationEarlyStopping(
        patience=int(cfg.training.get("early_stopping_patience", 0)),
        min_delta=float(cfg.training.get("early_stopping_min_delta", 0.0)),
    )
    saved_early_stopping = trainer.checkpoint_metadata.get("early_stopping", {})
    early_stopping.best = saved_early_stopping.get("best", math.inf)
    early_stopping.stale_evaluations = saved_early_stopping.get("stale_evaluations", 0)
    logger0.info("Training started...")

    for epoch in range(trainer.epoch_init, cfg.training.epochs):
        if trainer.sampler is not None:
            trainer.sampler.set_epoch(epoch)

        total_loss = 0.0
        component_totals: dict[str, float] = {}
        num_batches = 0
        start = time.time()
        batch_start = start
        epoch_len = len(trainer.dataloader)
        log_every = max(1, epoch_len // 10)  # Log ~10 times per epoch

        for batch_idx, sample in enumerate(trainer.dataloader):
            sample = sample[0].to(dist.device)  # SimSample .to()
            loss = trainer.train(sample, epoch)
            total_loss += loss.detach().item()
            for component, value in trainer.last_loss_components.items():
                component_totals[component] = component_totals.get(component, 0.0) + (
                    value.item()
                )
            num_batches += 1

            # Per-batch progress
            if (batch_idx + 1) % log_every == 0 or batch_idx == 0:
                batch_duration = time.time() - batch_start
                mem_gb = (
                    torch.cuda.memory_reserved() / 1024**3
                    if torch.cuda.is_available()
                    else 0.0
                )
                logger0.info(
                    f"Epoch {epoch + 1} [{batch_idx + 1}/{epoch_len}] "
                    f"Loss: {loss.detach().item():.6f} "
                    f"Duration: {batch_duration:.2f}s Mem: {mem_gb:.2f}GB"
                )
            batch_start = time.time()

        trainer.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        epoch_duration = time.time() - start
        logger0.info(
            f"Epoch {epoch + 1}/{cfg.training.epochs} "
            f"avg_loss: {avg_loss:.6f} "
            f"lr: {trainer.optimizer.param_groups[0]['lr']:.3e} "
            f"duration: {epoch_duration:.2f}s"
        )

        if dist.rank == 0:
            trainer.writer.add_scalar("loss", avg_loss, epoch)
            for component, total in component_totals.items():
                trainer.writer.add_scalar(
                    f"loss/{component}", total / max(num_batches, 1), epoch
                )
            trainer.writer.add_scalar(
                "learning_rate", trainer.optimizer.param_groups[0]["lr"], epoch
            )

        if dist.world_size > 1:
            torch.distributed.barrier()

        # Validation
        should_stop = False
        if (
            cfg.training.num_validation_samples > 0
            and (epoch + 1) % cfg.training.validation_freq == 0
        ):
            val_stats = trainer.validate(epoch)

            # Log validation metrics
            mse_val = val_stats["MSE"].item()
            mse_w_time = val_stats["MSE_w_time"]
            mse_w_channel = val_stats["MSE_w_channel"]
            mse_w_time_channel = val_stats["MSE_w_time_channel"]
            logger0.info(f"Validation epoch {epoch + 1}: MSE: {mse_val:.6f}")
            logger0.info(
                f"Validation epoch {epoch + 1}: channel_MSE: "
                + ", ".join(
                    f"channel_{index}={value.item():.6f}"
                    for index, value in enumerate(mse_w_channel)
                )
            )
            logger0.info(
                f"Validation epoch {epoch + 1}: final_timestep_channel_MSE: "
                + ", ".join(
                    f"channel_{index}={value.item():.6f}"
                    for index, value in enumerate(mse_w_time_channel[-1])
                )
            )
            if _tabulate.available and dist.rank == 0:
                rows = [["MSE (overall)", f"{mse_val:.6f}"]]
                for i, m in enumerate(mse_w_time):
                    rows.append([f"timestep_{i}_MSE", f"{m.item():.6f}"])
                logger0.info(
                    f"\nValidation metrics:\n{_tabulate.tabulate(rows, headers=['Metric', 'Value'], tablefmt='pretty')}\n"
                )

            if dist.rank == 0:
                # Log to tensorboard
                trainer.writer.add_scalar("val/MSE", val_stats["MSE"].item(), epoch)

                # Log individual timestep relative errors
                for i in range(len(val_stats["MSE_w_time"])):
                    trainer.writer.add_scalar(
                        f"val/timestep_{i}_MSE",
                        val_stats["MSE_w_time"][i].item(),
                        epoch,
                    )
                for channel in range(len(mse_w_channel)):
                    trainer.writer.add_scalar(
                        f"val/channel_{channel}_MSE",
                        mse_w_channel[channel].item(),
                        epoch,
                    )
                    trainer.writer.add_scalar(
                        f"val/final_timestep_channel_{channel}_MSE",
                        mse_w_time_channel[-1, channel].item(),
                        epoch,
                    )

            should_stop, improved = early_stopping.update(mse_val)
            if early_stopping.enabled:
                logger0.info(
                    "Early stopping: "
                    f"best={early_stopping.best:.6f}, "
                    f"stale={early_stopping.stale_evaluations}/"
                    f"{early_stopping.patience}, improved={improved}"
                )
            if should_stop:
                logger0.info(
                    f"Early stopping at epoch {epoch + 1} after "
                    f"{early_stopping.stale_evaluations} validation evaluations "
                    "without sufficient improvement."
                )

        # Save after validation: its DataLoader and any stochastic evaluation
        # must be reflected in the state used for the next training epoch. An
        # early stop saves too, so the final state is on disk.
        if should_stop or (epoch + 1) % cfg.training.save_checkpoint_freq == 0:
            metadata = {
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "rng_states": gather_rng_states(trainer.loader_generators),
                "early_stopping": {
                    "best": early_stopping.best,
                    "stale_evaluations": early_stopping.stale_evaluations,
                },
            }
            if dist.rank == 0:
                save_checkpoint(
                    cfg.training.ckpt_path,
                    models=trainer.model,
                    optimizer=trainer.optimizer,
                    scheduler=trainer.scheduler,
                    scaler=trainer.scaler,
                    epoch=epoch + 1,
                    metadata=metadata,
                )
                logger.info(f"Saved model on rank {dist.rank}")
        if should_stop:
            break

    logger0.info("Training completed!")
    if dist.rank == 0:
        trainer.writer.close()


if __name__ == "__main__":
    main()
