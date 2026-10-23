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

import logging
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))

import hydra
import numpy as np
import omegaconf
import torch
from hydra.utils import get_class, instantiate
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

# Import unified datapipe and utils
from data_startup import initialize_datasets
from datapipe import SimSample, simsample_collate
from omegaconf import open_dict
from training_state import (
    configure_deterministic_training,
    gather_rng_states,
    restore_rank_rng,
)
from utils import build_muon_optimizer


def execution_end_epoch(
    epoch_init: int, epochs: int, max_epochs_this_run: int | None
) -> int:
    """Bound a smoke/allocation without shortening the optimizer's LR schedule."""
    if max_epochs_this_run is None:
        return epochs
    if (
        isinstance(max_epochs_this_run, bool)
        or not isinstance(max_epochs_this_run, int)
        or max_epochs_this_run <= 0
    ):
        raise ValueError("max_epochs_this_run must be a positive integer or null")
    return min(epochs, epoch_init + max_epochs_this_run)


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


@dataclass(frozen=True)
class LinearTeacherForcingSchedule:
    """Linear transition-level teacher-forcing schedule for rollout training."""

    warmup_epochs: int = 0
    decay_epochs: int = 0
    start_probability: float = 1.0
    end_probability: float = 1.0

    def __post_init__(self) -> None:
        if self.warmup_epochs < 0:
            raise ValueError("teacher-forcing warmup_epochs cannot be negative")
        if self.decay_epochs < 0:
            raise ValueError("teacher-forcing decay_epochs cannot be negative")
        for name, value in (
            ("start_probability", self.start_probability),
            ("end_probability", self.end_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"teacher-forcing {name} must be in [0, 1]")

    def probability(self, epoch: int) -> float:
        """Return the teacher-forcing probability for a zero-based epoch."""
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        if epoch < self.warmup_epochs:
            return self.start_probability
        if self.decay_epochs == 0:
            return self.end_probability
        progress = min((epoch - self.warmup_epochs) / self.decay_epochs, 1.0)
        return self.start_probability + progress * (
            self.end_probability - self.start_probability
        )


def masked_acceleration_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    supervision_mask: torch.Tensor,
) -> torch.Tensor:
    """MSE over transitions whose input state came from the reference trajectory."""

    if (
        prediction.shape != target.shape
        or prediction.ndim != 3
        or prediction.shape[-1] != 3
    ):
        raise ValueError(
            "Acceleration prediction and target must have matching [N, T, 3] shapes"
        )
    if supervision_mask.shape != (prediction.shape[1],):
        raise ValueError("Acceleration supervision mask must have shape [T]")
    # Accumulate in fp32: a bumper graph has enough nodes for an fp16 element
    # count to overflow before division.
    squared_error = torch.square(prediction.float() - target.float())
    selected = supervision_mask.to(dtype=squared_error.dtype).view(1, -1, 1)
    denominator = selected.sum() * prediction.shape[0] * prediction.shape[2]
    return torch.sum(squared_error * selected) / denominator.clamp_min(1.0)


def average_model_gradients(model: torch.nn.Module, world_size: int) -> None:
    """Synchronously average dense gradients after backward releases activations."""

    if world_size <= 1:
        return
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("No gradients are available for distributed averaging")

    groups: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for gradient in gradients:
        groups.setdefault((gradient.device, gradient.dtype), []).append(gradient)
    for grouped_gradients in groups.values():
        flat = torch.cat([gradient.reshape(-1) for gradient in grouped_gradients])
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat.div_(world_size)
        offset = 0
        for gradient in grouped_gradients:
            count = gradient.numel()
            gradient.copy_(flat[offset : offset + count].view_as(gradient))
            offset += count


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


class Trainer:
    """Trainer for crash simulation models with unified SimSample input."""

    def __init__(self, cfg: DictConfig, logger0: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.cfg = cfg
        initial_history_steps = int(cfg.datapipe.get("initial_history_steps", 1))
        self.rollout_steps = cfg.training.num_time_steps - initial_history_steps
        if self.rollout_steps < 1:
            raise ValueError(
                "training.num_time_steps must exceed datapipe.initial_history_steps"
            )
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
        if self.manual_gradient_allreduce and self.amp_dtype == torch.float16:
            raise ValueError(
                "manual_gradient_allreduce currently requires bfloat16 or fp32"
            )
        self.acceleration_loss_weight = float(
            cfg.training.get("acceleration_loss_weight", 0.0)
        )
        if self.acceleration_loss_weight < 0.0:
            raise ValueError("training.acceleration_loss_weight cannot be negative")
        schedule_cfg = cfg.training.get("teacher_forcing_schedule")
        self.teacher_forcing_schedule = (
            None
            if schedule_cfg is None
            else LinearTeacherForcingSchedule(
                warmup_epochs=int(schedule_cfg.get("warmup_epochs", 0)),
                decay_epochs=int(schedule_cfg.get("decay_epochs", 0)),
                start_probability=float(schedule_cfg.get("start_probability", 1.0)),
                end_probability=float(schedule_cfg.get("end_probability", 1.0)),
            )
        )
        self.last_loss_components: dict[str, torch.Tensor] = {}
        # Time-conditional DeFormer: auxiliary supervision of the provisional
        # (coarse) position head, weight 0 for every other recipe.
        self.coarse_loss_weight = float(cfg.training.get("coarse_loss_weight", 0.0))
        if self.coarse_loss_weight < 0.0:
            raise ValueError("training.coarse_loss_weight cannot be negative")
        self.penetration_cfg = cfg.training.get("penetration_loss")
        if self.penetration_cfg is not None:
            from penetration_loss import penetration_weight

            penetration_weight(
                0,
                **{
                    key: self.penetration_cfg[key]
                    for key in ("max_weight", "warmup_epochs", "ramp_epochs")
                },
            )
            if not (
                cfg.datapipe.get("penetration_loss_metadata", False)
                or cfg.datapipe.get("contact_surface", False)
            ):
                raise ValueError(
                    "penetration loss requires datapipe.penetration_loss_metadata "
                    "(or datapipe.contact_surface on the Yaris time-conditional path)"
                )

        # --- Consistency check between model and datapipe ---
        model_name = cfg.model._target_
        datapipe_name = cfg.datapipe._target_
        graph_model_markers = (
            "MeshGraphNet",
            "MeshTransolver",
            "MeshGeoTransolver",
            "MeshGeoFLARE",
        )
        is_graph_model = any(marker in model_name for marker in graph_model_markers)

        if is_graph_model and "GraphDataset" not in datapipe_name:
            raise ValueError(
                f"Model {model_name} requires a graph datapipe, "
                f"but you selected {datapipe_name}."
            )
        if (
            not is_graph_model
            and "Transolver" in model_name
            and "PointCloudDataset" not in datapipe_name
        ):
            raise ValueError(
                f"Model {model_name} requires a point-cloud datapipe, "
                f"but you selected {datapipe_name}."
            )
        if "FIGConvUNet" in model_name and "PointCloudDataset" not in datapipe_name:
            raise ValueError(
                f"Model {model_name} requires a point-cloud datapipe, "
                f"but you selected {datapipe_name}."
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
            from utils import load_warm_start

            report = load_warm_start(
                self.model,
                warm_start,
                allowed_missing_prefixes=tuple(
                    cfg.training.get(
                        "warm_start_allow_missing",
                        [
                            "refinement_contact_block.",
                            "wall_contact.",
                            "coarse_feedback.",
                            "global_motion_head.",
                        ],
                    )
                ),
            )
            logger0.info(
                f"Warm start from {warm_start}: {len(report['loaded'])} tensors "
                f"loaded, {len(report['missing'])} fresh: {report['missing'][:6]}"
            )
        self.model.to(self.dist.device)
        self.model.train()

        # Log model summary and parameter count (optional: torchinfo)
        if self.dist.rank == 0:
            num_params = sum(p.numel() for p in self.model.parameters())
            logger0.info(f"Model parameters: {num_params:,}")
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

        # Optimizer (Muon requires PyTorch >= 2.9)
        opt_name = cfg.training.get("optimizer", "adam")
        if opt_name not in ["adam", "adamw", "muon"]:
            raise ValueError(f"Unsupported optimizer: {opt_name}")
        if opt_name == "muon":
            self.optimizer = build_muon_optimizer(self.model, cfg)
        elif opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.training.start_lr,
                weight_decay=cfg.training.optimizer_weight_decay,
                fused=torch.cuda.is_available(),
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=cfg.training.start_lr,
                fused=torch.cuda.is_available(),
            )
        logger0.info(f"Using {self.optimizer.__class__.__name__} optimizer")

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.end_lr
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
        self._setup_ema(cfg)
        # Concrete dropout (learned per-layer rates in the FLARE blocks when
        # model.concrete_dropout=true): entropy regulariser weight.
        self.concrete_dropout_lambda = float(
            cfg.training.get("concrete_dropout_lambda", 0.0)
        )
        if self.concrete_dropout_lambda < 0:
            raise ValueError("training.concrete_dropout_lambda cannot be negative")
        # Stochastic node subsets for the supervised position loss (training only).
        self.loss_node_fraction = float(cfg.training.get("loss_node_fraction", 1.0))
        if not 0.0 < self.loss_node_fraction <= 1.0:
            raise ValueError("training.loss_node_fraction must be in (0, 1]")
        self.penetration_signature = (
            None
            if self.penetration_cfg is None
            else {
                "version": "initial-pair-allowance-node-max-v1",
                "config": omegaconf.OmegaConf.to_container(
                    self.penetration_cfg, resolve=True
                ),
            }
        )
        if (
            self.epoch_init
            and self.checkpoint_metadata.get("penetration_loss")
            != self.penetration_signature
        ):
            raise ValueError(
                "Penetration-loss settings differ from the resumed checkpoint"
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
                    "Legacy checkpoint has no RNG state; restart is not replay-equivalent"
                )
            elif not cfg.training.get("restore_rng", True):
                # Evaluation (or a restart at another world size) loads weights
                # only; replay equivalence is not needed there.
                logger0.info("RNG state not restored (training.restore_rng=false)")
            else:
                restore_rank_rng(
                    states, self.dist.rank, self.dist.world_size, self.loader_generators
                )

    # ----- exponential moving average of the weights -----
    def _unwrapped_model(self):
        return (
            self.model.module
            if isinstance(self.model, DistributedDataParallel)
            else self.model
        )

    def _setup_ema(self, cfg):
        """``training.ema_decay`` > 0 keeps an EMA copy used for validation.

        The copy lives next to the model, is updated after every optimizer
        step with a warm-up on the decay, is saved as ``ema/<Class>.0.<E>.mdlus``
        next to the checkpoints and restored from there on resume (otherwise
        it restarts from the loaded weights).
        """
        self.ema_decay = float(cfg.training.get("ema_decay", 0.0))
        self.ema_model = None
        self.ema_updates = 0
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("training.ema_decay must be in [0, 1)")
        if self.ema_decay == 0.0:
            return
        source = self._unwrapped_model()
        self.ema_model = deepcopy(source).to(self.dist.device)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()
        path = self._ema_path(cfg.training.ckpt_path, self.epoch_init)
        if self.epoch_init > 0 and path.exists():
            from utils import load_warm_start

            load_warm_start(self.ema_model, path, allowed_missing_prefixes=())
            self.ema_updates = self.epoch_init * max(1, len(self.dataloader))
        self.ema_source_epochs = self.epoch_init

    @staticmethod
    def _ema_path(ckpt_path, epoch):
        from pathlib import Path

        return Path(ckpt_path) / "ema" / f"MeshGeoFLARETimeConditional.0.{epoch}.mdlus"

    @torch.no_grad()
    def _update_ema(self):
        if self.ema_model is None:
            return
        self.ema_updates += 1
        decay = min(
            self.ema_decay, (1.0 + self.ema_updates) / (10.0 + self.ema_updates)
        )
        source = self._unwrapped_model()
        for target, value in zip(self.ema_model.parameters(), source.parameters()):
            if target.is_floating_point():
                target.lerp_(value.detach().to(target.dtype), 1.0 - decay)
            else:
                target.copy_(value)
        for target, value in zip(self.ema_model.buffers(), source.buffers()):
            target.copy_(value)

    @contextmanager
    def ema_weights(self):
        """Temporarily put the EMA weights into the trained model (no-op without EMA)."""
        if self.ema_model is None:
            yield
            return
        source = self._unwrapped_model()
        with torch.no_grad():
            backup = [p.detach().clone() for p in source.parameters()]
            backup_buffers = [b.detach().clone() for b in source.buffers()]
            for target, value in zip(source.parameters(), self.ema_model.parameters()):
                target.copy_(value)
            for target, value in zip(source.buffers(), self.ema_model.buffers()):
                target.copy_(value)
        try:
            yield
        finally:
            with torch.no_grad():
                for target, value in zip(source.parameters(), backup):
                    target.copy_(value)
                for target, value in zip(source.buffers(), backup_buffers):
                    target.copy_(value)

    def save_ema(self, epoch):
        """Rank zero writes the EMA weights for ``epoch`` (1-based) next to the checkpoints."""
        if self.ema_model is None or self.dist.rank != 0:
            return
        path = self._ema_path(self.cfg.training.ckpt_path, epoch)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.ema_model.save(str(path))

    def _configure_position_loss(self, cfg):
        """training.loss / loss_space / frame_weighting / edge_loss_weight options."""
        from losses import FrameWeighter, PositionLoss

        pos_std = self.data_stats["node"].get("pos_std")
        self.position_loss = PositionLoss(
            kind=cfg.training.get("loss", "mse"),
            space=cfg.training.get("loss_space", "normalized"),
            scale_mm=float(cfg.training.get("loss_scale_mm", 10.0)),
            huber_delta_mm=float(cfg.training.get("huber_delta_mm", 3.0)),
            pos_std=pos_std,
        )
        weighting = cfg.training.get("frame_weighting", "none")
        if weighting not in ("none", "ema"):
            raise ValueError("training.frame_weighting must be 'none' or 'ema'")
        self.frame_weighter = (
            FrameWeighter(
                self.rollout_steps,
                float(cfg.training.get("frame_weighting_ema", 0.99)),
            )
            if weighting == "ema"
            else None
        )
        self.edge_loss_weight = float(cfg.training.get("edge_loss_weight", 0.0))
        if self.edge_loss_weight < 0.0:
            raise ValueError("training.edge_loss_weight cannot be negative")

    def _supervised_position_loss(self, prediction, target, sample):
        """Position loss with the configured kind, space and frame weighting."""
        if self.model.training and self.loss_node_fraction < 1.0:
            keep = (
                torch.rand(prediction.shape[0], device=prediction.device)
                < self.loss_node_fraction
            )
            if bool(keep.any()):
                prediction, target = prediction[keep], target[keep]
        if self.frame_weighter is None:
            return self.position_loss(prediction, target)
        from losses import frame_indices

        frames = frame_indices(sample, target, self.rollout_steps)
        return self.frame_weighter.apply(
            frames, self.position_loss.per_frame(prediction, target)
        )

    def _edge_term(self, prediction, target, sample):
        if self.edge_loss_weight <= 0.0:
            return None
        from losses import edge_consistency_loss

        graph = getattr(sample, "graph", None)
        index = getattr(graph, "knn_index", None)
        mask = getattr(graph, "knn_mask", None)
        if index is None or mask is None:
            raise ValueError(
                "training.edge_loss_weight needs graph.knn_index/knn_mask (kNN datapipe)"
            )
        return self.edge_loss_weight * edge_consistency_loss(
            prediction,
            target,
            index,
            mask,
            self.position_loss.pos_std
            if self.position_loss.space == "physical"
            else None,
            self.position_loss.scale_mm,
        )

    def train(self, sample: SimSample, epoch: int = 0):
        """Run the configured training budget with validation and restart checkpoints."""
        self.optimizer.zero_grad()
        loss = self.forward(sample, epoch)
        loss = self._add_concrete_dropout_term(loss)
        self.backward(loss)
        return loss.detach()

    def _add_concrete_dropout_term(self, loss):
        if self.concrete_dropout_lambda <= 0:
            return loss
        from physicsnemo.nn import collect_concrete_dropout_losses

        term = collect_concrete_dropout_losses(self._unwrapped_model())
        if not term.requires_grad:
            return loss
        term = term.to(loss.device, loss.dtype)
        self.last_loss_components["concrete_dropout"] = term.detach()
        return loss + self.concrete_dropout_lambda * term

    def concrete_dropout_rates(self):
        """Learned dropout probabilities per ConcreteDropout module (empty if none)."""
        from physicsnemo.nn import get_concrete_dropout_rates

        return get_concrete_dropout_rates(self._unwrapped_model())

    def forward(self, sample: SimSample, epoch: int = 0):
        with autocast(device_type="cuda", enabled=self.amp, dtype=self.amp_dtype):
            # Model forward - returns [N, T, Fo]
            model_kwargs = {}
            if self.teacher_forcing_schedule is not None:
                model_kwargs["teacher_forcing_probability"] = (
                    self.teacher_forcing_schedule.probability(epoch)
                )
            if self.acceleration_loss_weight > 0.0 or self.coarse_loss_weight > 0.0:
                model_kwargs["return_auxiliary"] = True
            model_output = self.model(
                sample=sample, data_stats=self.data_stats, **model_kwargs
            )

            # Target is [N, T, Fo]
            target = sample.node_target
            if hasattr(model_output, "coarse_prediction"):
                prediction_loss = self._supervised_position_loss(
                    model_output.prediction, target, sample
                )
                coarse_loss = self.position_loss(model_output.coarse_prediction, target)
                self.last_loss_components = {
                    "trajectory": prediction_loss.detach(),
                    "coarse": coarse_loss.detach(),
                }
                total = prediction_loss + self.coarse_loss_weight * coarse_loss
                edge = self._edge_term(model_output.prediction, target, sample)
                if edge is not None:
                    self.last_loss_components["edge"] = edge.detach()
                    total = total + edge
                return self.add_penetration_loss(
                    total, model_output.prediction, sample, epoch
                )
            is_autoregressive_output = all(
                hasattr(model_output, field)
                for field in (
                    "trajectory",
                    "normalized_acceleration",
                    "target_normalized_acceleration",
                    "acceleration_supervision_mask",
                )
            )
            if is_autoregressive_output:
                trajectory_loss = self._supervised_position_loss(
                    model_output.trajectory, target, sample
                )
                acceleration_loss = masked_acceleration_mse(
                    model_output.normalized_acceleration,
                    model_output.target_normalized_acceleration,
                    model_output.acceleration_supervision_mask,
                )
                total_loss = trajectory_loss + (
                    self.acceleration_loss_weight * acceleration_loss
                )
                self.last_loss_components = {
                    "trajectory": trajectory_loss.detach(),
                    "acceleration": acceleration_loss.detach(),
                }
                edge = self._edge_term(model_output.trajectory, target, sample)
                if edge is not None:
                    self.last_loss_components["edge"] = edge.detach()
                    total_loss = total_loss + edge
                return self.add_penetration_loss(
                    total_loss, model_output.trajectory, sample, epoch
                )
            if self.acceleration_loss_weight > 0.0:
                raise TypeError(
                    "Acceleration loss requires an autoregressive model that returns "
                    "AutoregressiveRolloutOutput"
                )
            trajectory_loss = self._supervised_position_loss(
                model_output, target, sample
            )
            self.last_loss_components = {"trajectory": trajectory_loss.detach()}
            edge = self._edge_term(model_output, target, sample)
            if edge is not None:
                self.last_loss_components["edge"] = edge.detach()
                trajectory_loss = trajectory_loss + edge
            return self.add_penetration_loss(
                trajectory_loss, model_output, sample, epoch
            )

    def add_penetration_loss(self, base_loss, trajectory, sample, epoch):
        if self.penetration_cfg is None:
            return base_loss
        from penetration_loss import penetration_weight, rollout_penetration_loss

        weight = penetration_weight(
            epoch,
            **{
                key: self.penetration_cfg[key]
                for key in ("max_weight", "warmup_epochs", "ramp_epochs")
            },
        )
        self.last_loss_components["penetration_weight"] = base_loss.new_tensor(weight)
        if weight == 0:
            return base_loss
        if trajectory.ndim == 2:
            # One predicted frame (time-conditional model): the model's own
            # crossing-aware penalty on its surface-contact search.
            model = (
                self.model.module
                if isinstance(self.model, DistributedDataParallel)
                else self.model
            )
            penalty_fn = getattr(model, "penetration_penalty", None)
            if penalty_fn is None:
                raise TypeError(
                    "penetration loss on a single predicted frame needs a model "
                    "with penetration_penalty (the time-conditional DeFormer)"
                )
            penalty, diagnostics = penalty_fn(
                trajectory,
                sample,
                self.data_stats,
                tolerance_mm=float(self.penetration_cfg["tolerance_mm"]),
                length_scale_mm=float(self.penetration_cfg["length_scale_mm"]),
            )
        else:
            penalty, diagnostics = rollout_penetration_loss(
                trajectory, sample.graph, self.data_stats, self.penetration_cfg
            )
        self.last_loss_components.update(diagnostics)
        self.last_loss_components["penetration"] = penalty.detach()
        self.last_loss_components["penetration_weighted"] = weight * penalty.detach()
        return base_loss + weight * penalty

    def backward(self, loss):
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
        self._update_ema()

    @torch.no_grad()
    def validate(self, epoch):
        """Run validation error computation"""
        self.model.eval()

        MSE = torch.zeros(1, device=self.dist.device)
        MSE_w_time = torch.zeros(self.rollout_steps, device=self.dist.device)
        MSE_w_channel = torch.zeros(self.target_channels, device=self.dist.device)
        MSE_w_time_channel = torch.zeros(
            self.rollout_steps,
            self.target_channels,
            device=self.dist.device,
        )
        for idx, sample in enumerate(self.val_dataloader):
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

    end_epoch = execution_end_epoch(
        trainer.epoch_init, cfg.training.epochs, cfg.training.get("max_epochs_this_run")
    )
    memory_diagnostics = cfg.training.get("memory_diagnostics", False)
    for epoch in range(trainer.epoch_init, end_epoch):
        if trainer.sampler is not None:
            trainer.sampler.set_epoch(epoch)
        trainer.dataloader.dataset.set_epoch(epoch)

        total_loss = 0.0
        component_totals: dict[str, float] = {}
        num_batches = 0
        start = time.time()
        batch_start = start
        epoch_len = len(trainer.dataloader)
        log_every = max(1, epoch_len // 10)  # Log ~10 times per epoch

        for batch_idx, sample in enumerate(trainer.dataloader):
            sample = sample[0].to(dist.device)  # SimSample .to()
            if memory_diagnostics and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            loss = trainer.train(sample, epoch)
            if memory_diagnostics and torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                logger.info(
                    f"CUDA_MEMORY rank={dist.rank} epoch={epoch + 1} batch={batch_idx + 1} "
                    f"allocated={torch.cuda.memory_allocated()} "
                    f"reserved={torch.cuda.memory_reserved()} "
                    f"peak_allocated={torch.cuda.max_memory_allocated()} "
                    f"peak_reserved={torch.cuda.max_memory_reserved()} "
                    f"driver_used={total_bytes - free_bytes} driver_total={total_bytes}"
                )
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
            if trainer.teacher_forcing_schedule is not None:
                trainer.writer.add_scalar(
                    "teacher_forcing_probability",
                    trainer.teacher_forcing_schedule.probability(epoch),
                    epoch,
                )
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
            with trainer.ema_weights():
                val_stats = trainer.validate(epoch)
            rates = trainer.concrete_dropout_rates()
            if rates:
                logger0.info(
                    f"Validation epoch {epoch + 1}: concrete dropout rates "
                    + ", ".join(f"{name}={rate:.3f}" for name, rate in rates.items())
                )
                if getattr(trainer, "writer", None) is not None:
                    for name, rate in rates.items():
                        trainer.writer.add_scalar(f"dropout_rates/{name}", rate, epoch)

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
        # must be reflected in the state used for the next training epoch.
        if (epoch + 1) % cfg.training.save_checkpoint_freq == 0:
            metadata = {
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "rng_states": gather_rng_states(trainer.loader_generators),
                "early_stopping": {
                    "best": early_stopping.best,
                    "stale_evaluations": early_stopping.stale_evaluations,
                },
            }
            if trainer.penetration_signature is not None:
                metadata["penetration_loss"] = trainer.penetration_signature
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
                trainer.save_ema(epoch + 1)
                logger.info(f"Saved model on rank {dist.rank}")
        if should_stop:
            break

    if end_epoch < cfg.training.epochs:
        logger0.info(
            f"Execution limit reached at epoch {end_epoch}; target remains {cfg.training.epochs}."
        )
    else:
        logger0.info("Training completed!")
    if dist.rank == 0:
        trainer.writer.close()


if __name__ == "__main__":
    main()
