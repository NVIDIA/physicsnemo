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

"""
Training Script for Transolver VOF Prediction.

Uses per-timestep interface masking: at each rollout step, loss is computed
ONLY on nodes in the interface band (determined from ground truth).
Bulk nodes produce exactly zero gradient — they are excluded from the loss
entirely, not down-weighted.

The forward pass still processes all nodes so the transformer has full
spatial context for attention.
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(__file__))

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

import torch
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from typing import Any, Callable, Sequence

from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils import load_checkpoint, save_checkpoint

from datapipe import SimSample, simsample_collate
from rollout import compute_interface_band


def _to_tensor(value, dtype=torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _stats_to_device(stats: dict, device: torch.device, dtype=torch.float32) -> dict:
    return {k: _to_tensor(v, dtype=dtype).to(device) for k, v in stats.items()}


class CombinedOptimizer(Optimizer):
    """Combine multiple PyTorch optimizers under a single Optimizer-like interface."""

    def __init__(
        self,
        optimizers: Sequence[Optimizer],
        torch_compile_kwargs: dict[str, Any] | None = None,
    ):
        if not optimizers:
            raise ValueError("`optimizers` must contain at least one optimizer.")
        self.optimizers = optimizers
        param_groups = [g for opt in optimizers for g in opt.param_groups]
        super().__init__(param_groups, defaults={})
        if torch_compile_kwargs is None:
            self.step_fns: list[Callable] = [opt.step for opt in optimizers]
        else:
            self.step_fns: list[Callable] = [
                torch.compile(opt.step, **torch_compile_kwargs) for opt in optimizers
            ]

    def zero_grad(self, *args, **kwargs) -> None:
        """Zero gradients on every wrapped optimizer."""
        for opt in self.optimizers:
            opt.zero_grad(*args, **kwargs)

    def step(self, closure=None) -> None:
        """Run a single optimization step on every wrapped optimizer."""
        for step_fn in self.step_fns:
            if closure is None:
                step_fn()
            else:
                step_fn(closure)

    def state_dict(self):
        """Return a state dict aggregating the state of all wrapped optimizers."""
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        """Load aggregated state into each wrapped optimizer and refresh param groups."""
        for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)
        self.param_groups = [g for opt in self.optimizers for g in opt.param_groups]


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Timestep Interface Loss
# ═══════════════════════════════════════════════════════════════════════════════


class PerTimestepInterfaceLoss(torch.nn.Module):
    """

    Interface-band MSE loss computed per rollout timestep.

    Rationale
    ---------
    In VOF (volume-of-fluid) problems, the vast majority of nodes are
    either fully filled (VOF ≈ 1) or fully empty (VOF ≈ 0). The small
    fraction of nodes near the fluid interface (``vof_lo < VOF < vof_hi``)
    is where all of the interesting dynamics happens — flow front
    advancement, curvature changes, wetting behavior.

    A naïve MSE loss over all nodes is dominated by the trivial bulk
    regions, so the model learns to match the 0/1 plateaus and barely
    trains on the interface. This class removes that imbalance by
    restricting the loss to nodes inside the interface band at each
    timestep.

    How the band is determined
    --------------------------
    At each timestep:

    1. Nodes with partially-filled ground-truth VOF (``vof_lo < gt < vof_hi``)
       define the interface "core".
    2. The core is expanded to a band by including all nodes whose
       position along the thickness axis is within ``±expansion`` of the
       core. The axis is auto-detected as the direction with the smallest
       spread of core nodes (typical for thin flow fronts).
    3. ``expansion`` is either ``absolute_expansion`` (preferred for
       normalized coordinates) or ``band_fraction * domain_extent``.

    Interaction with the rollout model
    ----------------------------------
    The model (``TransolverAutoregressiveRollout``) always runs on the
    full mesh so that attention has global spatial context — interface
    dynamics depend on far-away boundary conditions. This loss does not
    change the forward pass; it simply zeros out the gradient
    contribution of bulk nodes by excluding them from the loss average.

    During training, the band mask is rebuilt from *ground-truth* VOF at
    every timestep so that the loss follows the true advancing interface,
    not the model's (initially bad) predictions. At inference time no
    masking is applied.

    Args:
        vof_lo: Lower VOF threshold for interface-core detection.
        vof_hi: Upper VOF threshold for interface-core detection.
        band_fraction: Band expansion as a fraction of the domain extent
                       along the detected axis. Used only when
                       ``absolute_expansion`` is ``None``.
        interface_axis: Axis along which to expand the core into a band
                        (0, 1, 2, or -1 for auto-detect).
        absolute_expansion: Explicit band expansion in coordinate units.
                            Recommended for z-score-normalized coords
                            where domain-fraction is not meaningful.

    Returns:
        Tuple of ``(loss, interface_pct)``:
          - ``loss``: Scalar MSE averaged over valid (non-empty) timesteps.
          - ``interface_pct``: Mean percentage of nodes in the band over
            the valid timesteps. Used for logging.

    """

    def __init__(
        self,
        vof_lo: float = 0.01,
        vof_hi: float = 0.99,
        band_fraction: float = 0.05,
        interface_axis: int = -1,
        absolute_expansion: float = None,
    ):
        super().__init__()
        self.vof_lo = vof_lo
        self.vof_hi = vof_hi
        self.band_fraction = band_fraction
        self.interface_axis = interface_axis
        self.absolute_expansion = absolute_expansion

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        T, N, _ = pred.shape
        device = pred.device

        total_loss = torch.zeros(1, device=device)
        total_interface_nodes = 0
        num_valid_steps = 0

        for t in range(T):
            gt_t = target[t, :, 0]

            band_mask = compute_interface_band(
                vof=gt_t,
                coords=coords,
                vof_lo=self.vof_lo,
                vof_hi=self.vof_hi,
                band_fraction=self.band_fraction,
                interface_axis=self.interface_axis,
                absolute_expansion=self.absolute_expansion,
            )

            M = band_mask.sum().item()
            if M == 0:
                continue

            pred_masked = pred[t, band_mask, :]
            target_masked = target[t, band_mask, :]

            step_loss = (pred_masked - target_masked).pow(2).mean()
            total_loss = total_loss + step_loss

            total_interface_nodes += M
            num_valid_steps += 1

        if num_valid_steps > 0:
            total_loss = total_loss / num_valid_steps
            avg_pct = (total_interface_nodes / num_valid_steps) / N * 100.0
        else:
            total_loss = (pred - target).pow(2).mean()
            avg_pct = 100.0

        return total_loss, avg_pct


class Trainer:
    """
    Trainer with per-timestep interface-only loss.

    Key difference from union-mask approach:
        - Forward pass: ALL nodes (full spatial context for attention)
        - Loss: ONLY interface nodes at each timestep (5-20% of nodes)
        - Bulk nodes: zero gradient contribution
    """

    def __init__(self, cfg: DictConfig, logger0: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.cfg = cfg
        self.logger = logger0

        self.rollout_steps = cfg.training.num_time_steps - 1
        self.amp = cfg.training.amp

        # ====== Interface loss config ======
        iface_cfg = getattr(cfg.training, "interface_mask", {})
        self.criterion = PerTimestepInterfaceLoss(
            vof_lo=getattr(iface_cfg, "vof_lo", 0.01),
            vof_hi=getattr(iface_cfg, "vof_hi", 0.99),
            band_fraction=getattr(iface_cfg, "band_fraction", 0.05),
            interface_axis=getattr(iface_cfg, "interface_axis", -1),
            absolute_expansion=getattr(iface_cfg, "absolute_expansion", None),
        )

        logger0.info(
            f"Per-timestep interface loss: VOF in ({self.criterion.vof_lo}, "
            f"{self.criterion.vof_hi}), band={self.criterion.band_fraction * 100:.0f}%"
        )

        # ====== Dataset Setup ======
        reader = instantiate(cfg.reader)
        logging.getLogger().setLevel(logging.INFO)

        dataset = instantiate(
            cfg.datapipe,
            name="vof_train",
            reader=reader,
            split="train",
            logger=logger0,
        )
        logging.getLogger().setLevel(logging.INFO)

        self.data_stats = dict(
            node=_stats_to_device(dataset.node_stats, self.dist.device),
            feature=_stats_to_device(dataset.feature_stats, self.dist.device),
        )

        sampler = DistributedSampler(
            dataset,
            num_replicas=self.dist.world_size,
            rank=self.dist.rank,
            shuffle=True,
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=cfg.training.num_dataloader_workers,
            sampler=sampler,
            collate_fn=simsample_collate,
        )
        self.sampler = sampler

        # ====== Validation Dataset ======
        self.val_dataloader = None
        self.num_validation_samples = 0
        self.num_validation_replicas = 0

        if cfg.training.num_validation_samples > 0:
            self._setup_validation(cfg, reader, logger0)

        # ====== Model Setup ======
        self.model = instantiate(cfg.model)
        logging.getLogger().setLevel(logging.INFO)
        self.model.to(self.dist.device)
        self.model.train()

        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        # ====== Optimization Setup ======
        base_lr = cfg.training.start_lr
        weight_decay = getattr(cfg.training, "weight_decay", 1.0e-4)
        muon_params = [p for p in self.model.parameters() if p.ndim == 2]
        other_params = [p for p in self.model.parameters() if p.ndim != 2]
        base_opt = torch.optim.AdamW(
            other_params,
            lr=base_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1.0e-8,
        )
        muon_opt = torch.optim.Muon(
            muon_params,
            lr=base_lr,
            weight_decay=weight_decay,
            adjust_lr_fn="match_rms_adamw",
        )
        self.optimizer = CombinedOptimizer(optimizers=[muon_opt, base_opt])

        # ====== Scheduler: CosineAnnealing with warm restarts ======
        scheduler_T0 = getattr(cfg.training, "scheduler_T0", 50)
        scheduler_T_mult = getattr(cfg.training, "scheduler_T_mult", 2)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=scheduler_T0,
            T_mult=scheduler_T_mult,
            eta_min=cfg.training.end_lr,
        )

        self.scaler = GradScaler("cuda", enabled=self.amp)

        # ====== Checkpoint Loading ======
        if self.dist.world_size > 1:
            torch.distributed.barrier()

        self.epoch_init = load_checkpoint(
            cfg.training.ckpt_path,
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )

        # ====== TensorBoard ======
        if self.dist.rank == 0:
            self.writer = SummaryWriter(log_dir=cfg.training.tensorboard_log_dir)

    def _setup_validation(self, cfg: DictConfig, reader, logger0):
        self.num_validation_replicas = min(
            self.dist.world_size, cfg.training.num_validation_samples
        )
        self.num_validation_samples = (
            cfg.training.num_validation_samples
            // self.num_validation_replicas
            * self.num_validation_replicas
        )
        logger0.info(f"Number of validation samples: {self.num_validation_samples}")

        val_cfg = self.cfg.datapipe.copy()
        with open_dict(val_cfg):
            val_cfg.data_dir = self.cfg.training.raw_data_dir_validation
            val_cfg.num_samples = self.num_validation_samples

        val_dataset = instantiate(
            val_cfg,
            name="vof_validation",
            reader=reader,
            split="validation",
            logger=logger0,
        )

        if self.dist.rank < self.num_validation_replicas:
            val_sampler = None
            if self.dist.world_size > 1:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=self.num_validation_replicas,
                    rank=self.dist.rank,
                    shuffle=False,
                    drop_last=True,
                )
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                drop_last=True,
                pin_memory=True,
                num_workers=cfg.training.num_dataloader_workers,
                sampler=val_sampler,
                collate_fn=simsample_collate,
            )
        else:
            self.val_dataloader = DataLoader(
                torch.utils.data.Subset(val_dataset, []),
                batch_size=1,
            )

    def train_step(self, sample: SimSample) -> tuple[torch.Tensor, float]:
        """
        Returns:
            (loss, interface_pct) for logging
        """
        self.optimizer.zero_grad()
        loss, interface_pct = self._forward(sample)
        self._backward(loss)
        return loss.detach(), interface_pct

    def _forward(self, sample: SimSample) -> tuple[torch.Tensor, float]:
        """
        Forward pass on ALL nodes, loss on ONLY interface nodes per timestep.
        """
        with autocast(device_type="cuda", enabled=self.amp):
            # Model forward on full domain: [T, N, 1]
            pred = self.model(sample=sample, data_stats=self.data_stats)

            # Target: [N, T] -> [T, N, 1]
            target_flat = sample.node_target  # [N, T]
            T = target_flat.size(1)

            assert T == self.rollout_steps, (
                f"Target time steps {T} != expected {self.rollout_steps}"
            )

            target = target_flat.transpose(0, 1).unsqueeze(-1)  # [T, N, 1]
            coords = sample.node_features["coords"]  # [N, 3]

            # Per-timestep interface-only loss
            loss, interface_pct = self.criterion(pred, target, coords)
            return loss, interface_pct

    def _backward(self, loss: torch.Tensor):
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=25.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=25.0)
            self.optimizer.step()

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Validate with both full-domain and per-timestep interface metrics."""
        self.model.eval()

        MSE_full = torch.zeros(1, device=self.dist.device)
        MSE_iface = torch.zeros(1, device=self.dist.device)
        MSE_w_time = torch.zeros(self.rollout_steps, device=self.dist.device)
        MSE_iface_w_time = torch.zeros(self.rollout_steps, device=self.dist.device)
        num_samples = 0

        for sample in self.val_dataloader:
            sample = sample[0].to(self.dist.device)

            pred_seq = self.model(sample=sample, data_stats=self.data_stats)

            N = sample.node_target.size(0)
            T = sample.node_target.size(1)

            if T != self.rollout_steps:
                continue

            exact_seq = sample.node_target.transpose(0, 1).unsqueeze(-1)
            coords = sample.node_features["coords"]
            sq_error = (pred_seq - exact_seq).pow(2)

            MSE_w_time += sq_error.mean(dim=(1, 2))
            MSE_full += sq_error.mean()

            for t in range(T):
                gt_t = exact_seq[t, :, 0]
                band = compute_interface_band(
                    gt_t,
                    coords,
                    vof_lo=self.criterion.vof_lo,
                    vof_hi=self.criterion.vof_hi,
                    band_fraction=self.criterion.band_fraction,
                    interface_axis=self.criterion.interface_axis,
                    absolute_expansion=self.criterion.absolute_expansion,
                )

                if band.any():
                    iface_mse = sq_error[t, band, :].mean()
                    MSE_iface_w_time[t] += iface_mse
                    MSE_iface += iface_mse / T

            num_samples += 1

        if self.dist.world_size > 1:
            for tensor in [MSE_full, MSE_iface, MSE_w_time, MSE_iface_w_time]:
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)

        total = max(self.num_validation_samples, 1)
        val_stats = {
            "MSE": MSE_full / total,
            "MSE_interface": MSE_iface / total,
            "MSE_w_time": MSE_w_time / total,
            "MSE_interface_w_time": MSE_iface_w_time / total,
        }

        self.model.train()
        return val_stats


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: initialize distributed state and launch the trainer."""
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("train")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger0.file_logging()

    trainer = Trainer(cfg, logger0)

    # ══════════════════════════════════════════════════════════════════════
    # Pre-training summary (rank 0 only)
    # ══════════════════════════════════════════════════════════════════════
    if dist.rank == 0:
        # ── Model parameters ──────────────────────────────────────────────
        model_raw = (
            trainer.model.module
            if isinstance(trainer.model, DistributedDataParallel)
            else trainer.model
        )
        total_params = sum(p.numel() for p in model_raw.parameters())
        trainable_params = sum(
            p.numel() for p in model_raw.parameters() if p.requires_grad
        )
        muon_params = sum(p.numel() for p in model_raw.parameters() if p.ndim == 2)
        other_params = trainable_params - muon_params

        logger0.info("")
        logger0.info("=" * 72)
        logger0.info("  TRAINING CONFIGURATION")
        logger0.info("=" * 72)

        # ── Data ──────────────────────────────────────────────────────────
        logger0.info("")
        logger0.info("  ┌─ Data ────────────────────────────────────────────────┐")
        logger0.info(f"  │  Train dir:          {cfg.training.raw_data_dir}")
        logger0.info(f"  │  Validation dir:     {cfg.training.raw_data_dir_validation}")
        logger0.info(f"  │  Train samples:      {cfg.training.num_samples}")
        logger0.info(f"  │  Validation samples: {cfg.training.num_validation_samples}")
        logger0.info(f"  │  Time steps (T):     {cfg.training.num_time_steps}")
        logger0.info(f"  │  Rollout steps:      {trainer.rollout_steps}")
        logger0.info(f"  │  Dataloader workers: {cfg.training.num_dataloader_workers}")
        logger0.info("  └────────────────────────────────────────────────────────┘")

        # ── Model ─────────────────────────────────────────────────────────
        logger0.info("")
        logger0.info("  ┌─ Model ───────────────────────────────────────────────┐")
        logger0.info(f"  │  Architecture:       {model_raw.__class__.__name__}")
        logger0.info(f"  │  Total parameters:   {total_params:,}")
        logger0.info(f"  │  Trainable:          {trainable_params:,}")
        logger0.info(f"  │    Muon (2D):        {muon_params:,}")
        logger0.info(f"  │    AdamW (other):    {other_params:,}")
        if hasattr(model_raw, "rollout_steps"):
            logger0.info(f"  │  Rollout steps:      {model_raw.rollout_steps}")
        if hasattr(model_raw, "num_fourier_frequencies"):
            logger0.info(
                f"  │  Fourier freqs:      {model_raw.num_fourier_frequencies}"
            )
        if hasattr(cfg, "model"):
            model_cfg = cfg.model
            for key in [
                "functional_dim",
                "out_dim",
                "geometry_dim",
                "slice_num",
                "n_layers",
            ]:
                val = getattr(model_cfg, key, None)
                if val is not None:
                    logger0.info(f"  │  {key + ':':<20} {val}")
        logger0.info("  └────────────────────────────────────────────────────────┘")

        # ── Optimization ──────────────────────────────────────────────────
        scheduler_T0 = getattr(cfg.training, "scheduler_T0", 50)
        scheduler_T_mult = getattr(cfg.training, "scheduler_T_mult", 2)

        logger0.info("")
        logger0.info("  ┌─ Optimization ────────────────────────────────────────┐")
        logger0.info(f"  │  Epochs:             {cfg.training.epochs}")
        logger0.info(f"  │  Start LR:           {cfg.training.start_lr}")
        logger0.info(f"  │  End LR (eta_min):   {cfg.training.end_lr}")
        logger0.info(f"  │  Scheduler:          CosineAnnealingWarmRestarts")
        logger0.info(f"  │    T_0:              {scheduler_T0}")
        logger0.info(f"  │    T_mult:           {scheduler_T_mult}")
        logger0.info(
            f"  │  Weight decay:       {getattr(cfg.training, 'weight_decay', 1e-4)}"
        )
        logger0.info(f"  │  Grad clip max_norm: 25.0")
        logger0.info(f"  │  AMP enabled:        {cfg.training.amp}")
        logger0.info("  └────────────────────────────────────────────────────────┘")

        # ── Interface loss ────────────────────────────────────────────────
        c = trainer.criterion
        logger0.info("")
        logger0.info("  ┌─ Interface Loss ──────────────────────────────────────┐")
        logger0.info(f"  │  VOF thresholds:     ({c.vof_lo}, {c.vof_hi})")
        logger0.info(f"  │  Band fraction:      {c.band_fraction}")
        logger0.info(f"  │  Absolute expansion: {c.absolute_expansion}")
        logger0.info(f"  │  Interface axis:     {c.interface_axis}  (-1 = auto)")
        logger0.info("  └────────────────────────────────────────────────────────┘")

        # ── Infrastructure ────────────────────────────────────────────────
        logger0.info("")
        logger0.info("  ┌─ Infrastructure ──────────────────────────────────────┐")
        logger0.info(f"  │  World size:         {dist.world_size}")
        logger0.info(f"  │  Device:             {dist.device}")
        logger0.info(f"  │  Checkpoint dir:     {cfg.training.ckpt_path}")
        logger0.info(f"  │  TensorBoard dir:    {cfg.training.tensorboard_log_dir}")
        logger0.info(
            f"  │  Save every:         {cfg.training.save_chckpoint_freq} epochs"
        )
        logger0.info(f"  │  Validate every:     {cfg.training.validation_freq} epochs")
        if trainer.epoch_init > 0:
            logger0.info(f"  │  Resumed from epoch: {trainer.epoch_init}")
        logger0.info("  └────────────────────────────────────────────────────────┘")

        # ── Per-layer parameter breakdown (compact) ───────────────────────
        logger0.info("")
        logger0.info("  ┌─ Layer Parameter Breakdown ───────────────────────────┐")
        logger0.info(f"  │  {'Layer':<40} {'Params':>10}  │")
        logger0.info(f"  │  {'─' * 40} {'─' * 10}  │")
        for name, param in model_raw.named_parameters():
            if param.requires_grad:
                logger0.info(f"  │  {name:<40} {param.numel():>10,}  │")
        logger0.info("  └────────────────────────────────────────────────────────┘")

        logger0.info("")
        logger0.info(f"  Total parameters:     {total_params:>12,}")
        logger0.info(f"  Trainable parameters: {trainable_params:>12,}")
        logger0.info(
            f"  Model size:           {total_params * 4 / 1024**2:>11.2f} MB  (fp32)"
        )

        logger0.info("")
        logger0.info("=" * 72)
        logger0.info("  STARTING TRAINING")
        logger0.info("=" * 72)
        logger0.info("")

    # ══════════════════════════════════════════════════════════════════════
    # Training loop
    # ══════════════════════════════════════════════════════════════════════

    for epoch in range(trainer.epoch_init, cfg.training.epochs):
        if trainer.sampler is not None:
            trainer.sampler.set_epoch(epoch)

        total_loss = 0.0
        total_iface_pct = 0.0
        num_batches = 0
        start_time = time.time()

        for sample in trainer.dataloader:
            sample = sample[0].to(dist.device)
            loss, iface_pct = trainer.train_step(sample)
            total_loss += loss.item()
            total_iface_pct += iface_pct
            num_batches += 1

        trainer.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        avg_pct = total_iface_pct / max(num_batches, 1)
        epoch_time = time.time() - start_time

        logger0.info(
            f"Epoch {epoch + 1:4d}/{cfg.training.epochs} | "
            f"Loss: {avg_loss:.4e} | "
            f"Iface: {avg_pct:.1f}% | "
            f"LR: {trainer.optimizer.param_groups[0]['lr']:.3e} | "
            f"Time: {epoch_time:.2f}s"
        )

        if dist.rank == 0:
            trainer.writer.add_scalar("train/loss", avg_loss, epoch)
            trainer.writer.add_scalar("train/interface_pct", avg_pct, epoch)
            trainer.writer.add_scalar(
                "train/learning_rate",
                trainer.optimizer.param_groups[0]["lr"],
                epoch,
            )

        if dist.world_size > 1:
            torch.distributed.barrier()

        if dist.rank == 0 and (epoch + 1) % cfg.training.save_chckpoint_freq == 0:
            save_checkpoint(
                cfg.training.ckpt_path,
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch + 1,
            )
            logger0.info(f"Checkpoint saved at epoch {epoch + 1}")

        if (
            cfg.training.num_validation_samples > 0
            and (epoch + 1) % cfg.training.validation_freq == 0
        ):
            val_stats = trainer.validate(epoch)

            mse = val_stats["MSE"].item()
            mse_i = val_stats["MSE_interface"].item()

            logger0.info(
                f"Validation | Full RMSE: {mse**0.5:.4e} | "
                f"Interface RMSE: {mse_i**0.5:.4e}"
            )

            if dist.rank == 0:
                trainer.writer.add_scalar("val/MSE", mse, epoch)
                trainer.writer.add_scalar("val/RMSE", mse**0.5, epoch)
                trainer.writer.add_scalar("val/MSE_interface", mse_i, epoch)
                trainer.writer.add_scalar("val/RMSE_interface", mse_i**0.5, epoch)

                for t in range(len(val_stats["MSE_w_time"])):
                    trainer.writer.add_scalar(
                        f"val/t{t + 1:02d}_MSE",
                        val_stats["MSE_w_time"][t].item(),
                        epoch,
                    )
                    trainer.writer.add_scalar(
                        f"val/t{t + 1:02d}_MSE_iface",
                        val_stats["MSE_interface_w_time"][t].item(),
                        epoch,
                    )

    logger0.info("=" * 60)
    logger0.info("Training completed!")
    logger0.info("=" * 60)

    if dist.rank == 0:
        trainer.writer.close()


if __name__ == "__main__":
    main()
