# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from datasets import dataset_classes
from omegaconf import OmegaConf
from utils.config import TrainMainConfig
from utils.loss import (
    build_area_weights,
    build_channel_weights,
    ensemble_mean_mse,
    fair_crps,
)
from utils.metrics import (
    crps_per_variable_per_lead,
    derived_variable_crps,
    energy_score_per_lead,
    ensemble_rmse_per_variable_per_lead,
    plot_metric_vs_lead,
    plot_power_spectra,
    plot_rank_histograms,
    power_spectra_per_variable,
    rank_histogram_per_variable,
    save_summary,
    spread_skill_per_variable_per_lead,
)
from utils.nn import build_model
from utils.parallel import ParallelHelper

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper


def find_latest_model_checkpoint(checkpoint_dir: Path) -> str:
    candidates = sorted(checkpoint_dir.glob("*.mdlus"))
    if not candidates:
        raise FileNotFoundError(f"No .mdlus checkpoints found in {checkpoint_dir}")
    return str(candidates[-1])


class Trainer:
    def __init__(self, cfg):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        self.cfg = TrainMainConfig(**cfg_dict)

        self.dist = DistributedManager()
        self.device = self.dist.device
        # Rank-0-only logger mirrors the StormCast convention
        # (examples/weather/stormcast/utils/logging.ExperimentLogger) — uses
        # physicsnemo.utils.logging.PythonLogger so output flushes on each
        # record instead of sitting in a print() stdio buffer under srun.
        self.logger = RankZeroLoggingWrapper(PythonLogger("fgn"), self.dist)
        self.logger.info("Trainer.__init__ starting")

        # Data + domain parallel setup. For single-process runs we skip the
        # ParallelHelper entirely: DistributedManager may be in its fallback
        # "single process" state (no process group), which is incompatible
        # with ShardTensor mesh creation. StormCast's trainer always builds a
        # ParallelHelper because it assumes a real distributed init; the FGN
        # recipe keeps a no-helper path so the CPU-only smoke test stays
        # runnable without an init_process_group call.
        self.parallel_helper: ParallelHelper | None = None
        domain_parallel_size = int(self.cfg.training.domain_parallel_size)
        force_sharding = bool(self.cfg.training.force_sharding)
        self.use_shard_tensor = domain_parallel_size > 1 or force_sharding
        if self.dist.world_size > 1 or self.use_shard_tensor:
            self.parallel_helper = ParallelHelper(
                domain_parallel_size=domain_parallel_size,
                use_shard_tensor=self.use_shard_tensor,
            )
            if (
                self.use_shard_tensor
                and self.parallel_helper.local_batch_size(
                    int(self.cfg.training.batch_size)
                )
                > 1
            ):
                raise ValueError("Domain parallelism requires a local batch size of 1")

        self.checkpoint_dir = (
            Path(self.cfg.training.rundir) / self.cfg.training.checkpoint_dir
        )
        if self.dist.rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # All ranks use the same seed so parameter initialization is identical.
        torch.manual_seed(int(self.cfg.training.seed))

        dataset_cls = dataset_classes[self.cfg.dataset.name]
        self.logger.info(f"Building datasets: {self.cfg.dataset.name}")
        self.train_dataset = dataset_cls(self.cfg.dataset, train=True)
        self.valid_dataset = dataset_cls(self.cfg.dataset, train=False)
        self.logger.info(
            f"Dataset ready: train={len(self.train_dataset)} val={len(self.valid_dataset)}"
        )

        self.logger.info("Fetching dataset invariants")
        invariants = self.train_dataset.get_invariants()
        self.invariants = None
        invariant_channels = 0
        if invariants is not None:
            self.invariants = torch.from_numpy(invariants).to(
                self.device, dtype=torch.float32
            )
            invariant_channels = int(self.invariants.shape[0])

        self.logger.info("Building model")
        self.model = build_model(
            self.cfg,
            state_channels=len(self.train_dataset.state_channels()),
            background_channels=len(self.train_dataset.background_channels()),
            invariant_channels=invariant_channels,
        ).to(self.device)
        self.logger.info(
            f"Model ready on {self.device} "
            f"(params={sum(p.numel() for p in self.model.parameters()):,})"
        )

        # Wrap with FSDP / ShardTensor when running distributed. Domain-
        # sharded invariant tensor so forward passes on sharded inputs find
        # the invariant in the same layout.
        if self.parallel_helper is not None:
            self.model = self.parallel_helper.distribute_model(self.model)
            if self.invariants is not None and self.use_shard_tensor:
                self.invariants = self.parallel_helper.distribute_tensor(
                    self.invariants
                )

        # Optimizer must be built after FSDP wrapping.
        opt_cfg = self.cfg.training.optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(opt_cfg.lr),
            betas=tuple(opt_cfg.betas),
            weight_decay=float(opt_cfg.weight_decay),
        )
        # LR schedule: linear warmup then cosine decay (paper Table A.2).
        warmup = int(opt_cfg.lr_warmup_steps)
        total = int(self.cfg.training.total_train_steps)
        lr_min = float(opt_cfg.lr_min)
        lr_max = float(opt_cfg.lr)

        def _lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return step / warmup
            if total <= warmup:
                return 1.0
            import math
            progress = (step - warmup) / max(1, total - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min / lr_max + (1.0 - lr_min / lr_max) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=_lr_lambda
        )

        # Train/val loaders: ranks get disjoint contiguous index slices via
        # ParallelHelper.sharded_dataloader. Single-process falls back to a
        # plain DataLoader so we don't depend on a process group.
        batch_size = int(self.cfg.training.batch_size)
        num_workers = int(self.cfg.training.num_data_workers)
        seed = int(self.cfg.training.seed)
        if self.parallel_helper is not None:
            local_batch = self.parallel_helper.local_batch_size(batch_size)
            self.train_loader = self.parallel_helper.sharded_dataloader(
                self.train_dataset,
                batch_size=local_batch,
                seed=seed,
                num_workers=num_workers,
                shuffle=True,
            )
            self.valid_loader = self.parallel_helper.sharded_dataloader(
                self.valid_dataset,
                batch_size=local_batch,
                seed=seed + 1,
                num_workers=0,
                shuffle=False,
            )
            # Cap validation length: the parallel_helper sampler is infinite
            # by design (StormCast convention), so we bound iteration the
            # same way StormCast does — `sharded_data_iter(loader, N)`. By
            # default sweep one local epoch over each rank's shard.
            local_valid = max(
                1,
                len(self.valid_dataset)
                // (max(self.dist.world_size, 1) * max(local_batch, 1)),
            )
            self.validation_steps = int(
                getattr(self.cfg.training, "validation_steps", local_valid)
                or local_valid
            )
        else:
            from datasets.dataset import worker_init
            from torch.utils.data import DataLoader

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=worker_init if num_workers else None,
            )
            self.valid_loader = DataLoader(
                self.valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
            self.validation_steps = None  # plain DataLoader is finite

        # Optional per-channel + cos(lat) loss weights. Channel weights
        # follow GraphCast/GenCast scheme with geopotential halved per FGN
        # §2.2.3; area weights normalise cos(lat) so the mean row-sum over
        # latitudes equals 1 (preserves loss scale when toggled on/off).
        self.loss_weights: torch.Tensor | None = None
        channel_w = None
        if bool(self.cfg.training.loss.use_channel_weights):
            channel_w = torch.from_numpy(
                build_channel_weights(self.train_dataset.state_channels())
            ).to(self.device, dtype=torch.float32)
        area_w = None
        if bool(self.cfg.training.loss.use_area_weights):
            H, _ = self.train_dataset.image_shape()
            area_w = torch.from_numpy(build_area_weights(H)).to(
                self.device, dtype=torch.float32
            )  # shape (H, 1)
        if channel_w is not None or area_w is not None:
            # Build a (1, C, H, W)-broadcastable tensor.
            H, W = self.train_dataset.image_shape()
            combined = torch.ones(1, 1, H, W, device=self.device, dtype=torch.float32)
            if channel_w is not None:
                combined = combined * channel_w.view(1, -1, 1, 1)
            if area_w is not None:
                combined = combined * area_w.view(1, 1, H, 1)
            self.loss_weights = combined

        self.step = 0
        self.best_val_loss = float("inf")
        self._resume_if_needed()

    def _resume_if_needed(self) -> None:
        resume = self.cfg.training.resume_checkpoint
        if resume is None:
            return
        if not self.checkpoint_dir.exists():
            return
        epoch = None if resume == "latest" else int(resume)
        metadata = {}
        loaded = load_checkpoint(
            self.checkpoint_dir,
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            metadata_dict=metadata,
            device=self.device,
        )
        self.step = int(loaded)
        if metadata.get("best_val_loss") is not None:
            self.best_val_loss = float(metadata["best_val_loss"])

    def _step_ensemble(
        self,
        history: torch.Tensor,
        background: torch.Tensor,
        invariants: torch.Tensor | None,
        num_samples: int,
    ) -> torch.Tensor:
        """Run `num_samples` forward passes of the model, one per latent draw.

        Returns a tensor of shape ``(B, num_samples, C, H, W)``.
        """

        members = []
        for _ in range(num_samples):
            latent = torch.randn(
                history.shape[0],
                int(self.cfg.model.latent_dim),
                device=self.device,
                dtype=torch.float32,
            )
            members.append(
                self.model(
                    history=history,
                    latent=latent,
                    background=background,
                    invariants=invariants,
                )
            )
        return torch.stack(members, dim=1)

    def _loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        history = batch["history"].to(self.device, dtype=torch.float32)
        target = batch["target"].to(self.device, dtype=torch.float32)
        background = batch["background"].to(self.device, dtype=torch.float32)

        # Normalize target layout to (B, K, C, H, W); datasets may emit (B, C, H, W).
        if target.ndim == 4:
            target = target.unsqueeze(1)
        if target.ndim != 5:
            raise ValueError(
                f"target must have shape [B, K, C, H, W] or [B, C, H, W], got {tuple(target.shape)}"
            )

        ar_steps = int(target.shape[1])
        cfg_ar = int(getattr(self.cfg.training, "ar_steps", 1))
        if cfg_ar != ar_steps:
            raise ValueError(
                f"training.ar_steps={cfg_ar} but dataset produced {ar_steps} future frames; "
                "set future_frames to match ar_steps"
            )

        invariants = None
        if self.invariants is not None:
            invariants = self.invariants.unsqueeze(0).expand(
                history.shape[0], -1, -1, -1
            )

        num_samples = int(self.cfg.training.loss.num_samples)
        mse_weight = float(self.cfg.training.loss.mse_weight)

        # For each rollout step, run N-member ensemble, score against that
        # step's ground truth, then advance history by appending each member's
        # prediction (so the N trajectories diverge in parallel).
        # History shape per member: (B, T, C, H, W).
        B, T, C, H, W = history.shape
        per_member_hist = (
            history.unsqueeze(1).expand(B, num_samples, T, C, H, W).contiguous()
        )

        step_losses: list[torch.Tensor] = []
        for k in range(ar_steps):
            members = []
            for n in range(num_samples):
                hist_n = per_member_hist[:, n]
                latent = torch.randn(
                    hist_n.shape[0],
                    int(self.cfg.model.latent_dim),
                    device=self.device,
                    dtype=torch.float32,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    members.append(
                        self.model(
                            history=hist_n,
                            latent=latent,
                            background=background,
                            invariants=invariants,
                        )
                    )
            preds = torch.stack(members, dim=1).float()  # (B, N, C, H, W)

            step_loss = fair_crps(preds, target[:, k], weights=self.loss_weights)
            if mse_weight > 0.0:
                step_loss = step_loss + mse_weight * ensemble_mean_mse(
                    preds, target[:, k], weights=self.loss_weights
                )
            step_losses.append(step_loss)

            if k < ar_steps - 1:
                # Paper §3: predicted-only channels (e.g. tp06) must not be
                # fed back as input on the next AR step — mirrors
                # earth2studio gencast_mini's zeroing of tp12 in inputs.
                # Clone before mutating because ``preds`` is still used in
                # ``step_loss`` and autograd is tracking it.
                next_frame = preds
                output_only = self.train_dataset.output_only_channels()
                if output_only:
                    next_frame = next_frame.clone()
                    for ci in output_only:
                        next_frame[:, :, ci].zero_()
                per_member_hist = torch.cat(
                    [per_member_hist[:, :, 1:], next_frame.unsqueeze(2)], dim=2
                )

        return torch.stack(step_losses).mean()

    def _validation_loss(self) -> float:
        self.model.eval()
        losses = []
        # Mirror StormCast: with parallel_helper the sampler is infinite, so
        # bound iteration via sharded_data_iter(loader, N). Plain DataLoader
        # path is finite and falls through to the default for-loop.
        if self.parallel_helper is not None:
            iterator = self.parallel_helper.sharded_data_iter(
                self.valid_loader, self.validation_steps
            )
        else:
            iterator = self.valid_loader
        with torch.no_grad():
            losses.extend(float(self._loss(batch).detach().cpu()) for batch in iterator)
        self.model.train()
        return sum(losses) / max(len(losses), 1)

    def _run_validation_metrics(self) -> None:
        """Figure 2 + 3 diagnostics on a single validation batch.

        Runs an ensemble rollout across all ``ar_steps`` lead times and
        writes per-variable CRPS / RMSE / spread-skill / rank hist / 1D
        power spectra to ``rundir/validation/step=<step>/``. No-op on
        non-rank-0 ranks.
        """
        if self.dist.rank != 0:
            return
        try:
            batch = next(iter(self.valid_loader))
        except StopIteration:
            return

        self.model.eval()
        history = batch["history"].to(self.device, dtype=torch.float32)
        target = batch["target"].to(self.device, dtype=torch.float32)
        background = batch["background"].to(self.device, dtype=torch.float32)
        if target.ndim == 4:
            target = target.unsqueeze(1)
        K = target.shape[1]

        invariants = None
        if self.invariants is not None:
            invariants = self.invariants.unsqueeze(0).expand(
                history.shape[0], -1, -1, -1
            )

        M = int(self.cfg.training.validation_ensemble_size)
        latent_dim = int(self.cfg.model.latent_dim)

        # N parallel trajectories diverge step-by-step exactly as in the
        # training loop, but we don't need gradients.
        B, T, C, H, W = history.shape
        per_member_hist = history.unsqueeze(1).expand(B, M, T, C, H, W).contiguous()
        preds_all: list[torch.Tensor] = []
        with torch.no_grad():
            for k in range(K):
                members = []
                for n in range(M):
                    latent = torch.randn(
                        B, latent_dim, device=self.device, dtype=torch.float32
                    )
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                        pred = self.model(
                            history=per_member_hist[:, n],
                            latent=latent,
                            background=background,
                            invariants=invariants,
                        )
                    members.append(pred.float())
                preds = torch.stack(members, dim=1)  # (B, M, C, H, W)
                preds_all.append(preds)
                if k < K - 1:
                    # Paper §3: zero predicted-only channels (e.g. tp06)
                    # before feeding them back as next-step history.
                    next_frame = preds
                    output_only = self.train_dataset.output_only_channels()
                    if output_only:
                        next_frame = next_frame.clone()
                        for ci in output_only:
                            next_frame[:, :, ci].zero_()
                    per_member_hist = torch.cat(
                        [per_member_hist[:, :, 1:], next_frame.unsqueeze(2)], dim=2
                    )

        self.model.train()
        ensemble = torch.stack(preds_all, dim=1)  # (B, K, M, C, H, W)

        variables = list(self.train_dataset.state_channels())
        crps_kc = crps_per_variable_per_lead(ensemble, target)
        rmse_kc = ensemble_rmse_per_variable_per_lead(ensemble, target)
        spread_kc, skill_kc, ratio_kc = spread_skill_per_variable_per_lead(
            ensemble, target
        )
        ranks_cb = rank_histogram_per_variable(ensemble, target)
        es_k = energy_score_per_lead(ensemble, target)
        derived = derived_variable_crps(ensemble, target, variables)
        ensemble_mean = ensemble.mean(dim=2)
        k_vec, ens_spec, tgt_spec = power_spectra_per_variable(ensemble_mean, target)

        out_dir = Path(self.cfg.training.rundir) / "validation" / f"step={self.step}"
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "crps_per_lead_per_channel": crps_kc,
            "rmse_per_lead_per_channel": rmse_kc,
            "spread_per_lead_per_channel": spread_kc,
            "skill_per_lead_per_channel": skill_kc,
            "spread_skill_ratio": ratio_kc,
            "rank_histograms": ranks_cb,
            "energy_score_per_lead": es_k,
            "variables": np.array(variables, dtype=object),
            "lead_steps": np.arange(1, K + 1, dtype=np.int64),
            "power_spectrum_k": k_vec,
            "power_spectrum_forecast": ens_spec,
            "power_spectrum_truth": tgt_spec,
        }
        for dname, vals in derived.items():
            summary[f"derived_crps_{dname}"] = vals
        save_summary(summary, str(out_dir / "metrics.npz"))

        leads = np.arange(1, K + 1)
        plot_metric_vs_lead(
            crps_kc,
            variables,
            leads,
            "CRPS",
            "fCRPS per lead (lower is better)",
            str(out_dir / "crps_vs_lead.png"),
        )
        plot_metric_vs_lead(
            rmse_kc,
            variables,
            leads,
            "ensemble-mean RMSE",
            "Ensemble-mean RMSE per lead",
            str(out_dir / "rmse_vs_lead.png"),
        )
        plot_metric_vs_lead(
            ratio_kc,
            variables,
            leads,
            "spread / skill",
            "Spread-skill ratio (1.0 = calibrated)",
            str(out_dir / "spread_skill_vs_lead.png"),
            hline_y=1.0,
        )
        plot_rank_histograms(ranks_cb, variables, str(out_dir / "rank_histograms.png"))
        # Energy score is a (K,) scalar — plot as a single-series lead curve.
        plot_metric_vs_lead(
            es_k[:, None],
            ["multivariate"],
            leads,
            "energy score",
            "Energy score per lead (lower is better)",
            str(out_dir / "energy_score_vs_lead.png"),
        )
        plot_power_spectra(
            k_vec,
            ens_spec,
            tgt_spec,
            variables,
            lead_idx=K - 1,
            out_path=str(out_dir / f"power_spectra_lead{K}.png"),
        )

    def save_checkpoint(self) -> None:
        save_checkpoint(
            self.checkpoint_dir,
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.step,
            metadata={"best_val_loss": self.best_val_loss},
        )

    def _make_train_iter(self) -> Iterator:
        # When domain parallelism is active, sharded_data_iter handles both
        # data-parallel sample routing and spatial scatter (ShardTensor).
        # Mirrors StormCast's pattern (stormcast/utils/trainer.py).
        if self.parallel_helper is not None:
            remaining = int(self.cfg.training.total_train_steps) - self.step
            return self.parallel_helper.sharded_data_iter(
                self.train_loader, num_samples=remaining
            )
        # Plain single-process / DDP path: restart the DataLoader on exhaustion.
        def _plain() -> Iterator:
            loader_iter = iter(self.train_loader)
            while True:
                try:
                    yield next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self.train_loader)
                    yield next(loader_iter)

        return _plain()

    def train(self) -> None:
        self.model.train()
        total_steps = int(self.cfg.training.total_train_steps)

        for batch in self._make_train_iter():
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._loss(batch)
            loss.backward()

            clip = float(self.cfg.training.clip_grad_norm)
            if clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            self.optimizer.step()
            self.scheduler.step()
            self.step += 1

            if self.step % int(self.cfg.training.print_progress_freq) == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(
                    f"step={self.step} train_loss={float(loss.detach().cpu()):.6f} lr={lr:.3e}"
                )

            if self.step % int(self.cfg.training.validation_freq) == 0:
                val_loss = self._validation_loss()
                self.best_val_loss = min(self.best_val_loss, val_loss)
                self.logger.info(f"step={self.step} val_loss={val_loss:.6f}")
                if bool(self.cfg.training.validation_metrics):
                    self._run_validation_metrics()

            if self.step % int(self.cfg.training.checkpoint_freq) == 0:
                self.save_checkpoint()

            if self.step >= total_steps:
                break

        if self.step % int(self.cfg.training.checkpoint_freq) != 0:
            self.save_checkpoint()
