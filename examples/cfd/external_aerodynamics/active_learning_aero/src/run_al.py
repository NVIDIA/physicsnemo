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

"""Active learning loop for GeoTransolver + GP on DrivAerStar.

Loads a pretrained Fastback checkpoint, constructs a multi-class pool
(Fastback + Notchback + Estateback), and iteratively selects the most
informative samples via joint UQ or random acquisition.

Usage::

    python run_al.py --config-name=al_config \\
        ++initial_checkpoint=/path/to/checkpoints_combined \\
        ++al_class_F_path=/data/datasets/drivaerstar/surface_files_zarr/class_F/val \\
        ++al_class_N_path=/data/datasets/drivaerstar/surface_files_zarr/class_N/val \\
        ++al_class_E_path=/data/datasets/drivaerstar/surface_files_zarr/class_E/val
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from queue import Queue

import hydra
import numpy as np
import omegaconf
import torch
from omegaconf import DictConfig, OmegaConf

torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
torch.serialization.add_safe_globals([list])
torch.serialization.add_safe_globals([collections.defaultdict])
torch.serialization.add_safe_globals([dict])
torch.serialization.add_safe_globals([int])
torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
torch.serialization.add_safe_globals([omegaconf.base.Metadata])

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.experimental.uq import VariationalGPHead

from utils import CombinedOptimizer
from gp_utils import (
    apply_spectral_norm_to_model,
    create_embedding_reduction,
    gp_ramp_weight,
)

from data_pool import AeroDataPool, load_manifests
from strategies import (
    ClassBalancedRandomQueryStrategy,
    DummyLabelStrategy,
    JointUQQueryStrategy,
    LatentNoveltyQueryStrategy,
    RandomQueryStrategy,
)
from aero_metrology import FieldMetrologyStrategy
from al_train_step import train_one_batch


@hydra.main(version_base=None, config_path="conf", config_name="al_config")
def main(cfg: DictConfig) -> None:
    """Run the active learning experiment."""
    DistributedManager.initialize()
    dist_manager = DistributedManager()
    device = dist_manager.device
    logger = RankZeroLoggingWrapper(PythonLogger(name="active_learning"), dist_manager)

    # ---- Config ----
    al_rounds = getattr(cfg, "al_rounds", 5)
    samples_per_round = getattr(cfg, "samples_per_round", 50)
    test_per_class = getattr(cfg, "test_samples_per_class", 100)
    acquisition = getattr(cfg, "acquisition", "joint_uq")
    random_seed = getattr(cfg, "random_seed", 42)
    fine_tune_epochs = getattr(cfg, "fine_tune_epochs", 20)
    fine_tune_lr = getattr(cfg, "fine_tune_lr", 5e-4)
    precision = getattr(cfg, "precision", "float32")
    embed_dim = getattr(cfg, "embed_dim", 32)
    feat_dim = getattr(cfg, "embedding_feat_dim", 256)
    n_inducing = getattr(cfg, "n_inducing", 128)
    use_spectral_norm = getattr(cfg, "spectral_norm_embedding", True)
    normalize_embeddings = getattr(cfg, "normalize_embeddings", True)
    embedding_target_scale = getattr(cfg, "embedding_target_scale", 1.0)
    lambda_gp = getattr(cfg, "lambda_gp", 0.01)
    lambda_consistency = getattr(cfg, "lambda_consistency", 1.0)
    consistency_detach = getattr(cfg, "consistency_detach_transolver", False)
    consistency_every_n = getattr(cfg, "consistency_every_n_steps", 1)
    accumulation_steps = getattr(cfg.training, "gradient_accumulation_steps", 1)
    save_interval = getattr(cfg.training, "save_interval", 10)
    prefetch_depth = int(OmegaConf.select(cfg, "dataloader.prefetch_depth", default=0))

    initial_checkpoint = cfg.initial_checkpoint
    if initial_checkpoint is None:
        raise ValueError(
            "Must provide ++initial_checkpoint=/path/to/checkpoints_combined"
        )

    manifest_dir = getattr(cfg, "manifest_dir", None)
    if manifest_dir is None:
        raise ValueError("Must provide ++manifest_dir=/path/to/manifests/")

    pool_by_class, test_by_class, paths_by_class = load_manifests(manifest_dir)
    class_paths = paths_by_class

    # ---- Normalization ----
    norm_dir = getattr(cfg.data, "normalization_dir", ".")
    norm_file = str(Path(norm_dir) / "surface_fields_normalization.npz")
    norm_data = np.load(norm_file)
    surface_factors = {
        "mean": torch.from_numpy(norm_data["mean"]).to(device),
        "std": torch.from_numpy(norm_data["std"]).to(device),
    }

    # ---- Build data pools from manifests ----
    logger.info("Building data pools from manifests...")

    train_pool = AeroDataPool(
        data_cfg=cfg.data,
        class_paths=class_paths,
        surface_factors=surface_factors,
        local_indices_by_class=pool_by_class,
        train_indices=torch.LongTensor([]),
    )

    test_pool = AeroDataPool(
        data_cfg=cfg.data,
        class_paths=class_paths,
        surface_factors=surface_factors,
        local_indices_by_class=test_by_class,
        train_indices=torch.arange(sum(len(v) for v in test_by_class.values())).long(),
    )

    # ---- Original Fastback training data (replay to prevent forgetting) ----
    base_train_path = getattr(cfg.data.train, "data_path", None)
    base_train_datapipe = None
    if base_train_path is not None:
        from physicsnemo.datapipes.cae.transolver_datapipe import (
            create_transolver_dataset,
        )

        base_train_datapipe = create_transolver_dataset(
            cfg.data,
            phase="train",
            surface_factors=surface_factors,
            volume_factors=None,
        )
        logger.info(
            f"Base training data (Fastback replay): "
            f"{len(base_train_datapipe.dataset)} samples from {base_train_path}"
        )
    else:
        logger.warning(
            "No base training data path provided (data.train.data_path). "
            "Fine-tuning will use only AL-selected samples — risk of forgetting."
        )

    logger.info(
        f"AL Pool: {train_pool.total_samples} samples, "
        f"Test: {test_pool.total_samples} samples"
    )

    # ---- Build models ----
    ls_range = tuple(getattr(cfg, "gp_lengthscale_range", [0.01, 1.0]))
    ls_prior_cfg = getattr(cfg, "gp_lengthscale_prior", None)
    ls_prior = tuple(ls_prior_cfg) if ls_prior_cfg is not None else None
    os_prior_cfg = getattr(cfg, "gp_outputscale_prior", None)
    os_prior = tuple(os_prior_cfg) if os_prior_cfg is not None else None
    mlp_hidden_cfg = getattr(cfg, "gp_mlp_hidden", None)
    mlp_hidden = list(mlp_hidden_cfg) if mlp_hidden_cfg is not None else None

    model = hydra.utils.instantiate(cfg.model, _convert_="partial")
    sn_backbone = getattr(cfg, "spectral_norm_backbone", False)
    sn_coeff = getattr(cfg, "spectral_norm_coeff", 1.0)
    if sn_backbone:
        apply_spectral_norm_to_model(model, coeff=sn_coeff)
    model.to(device)

    embedding_reduction = create_embedding_reduction(
        pooling=getattr(cfg, "embedding_pooling", "attention"),
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        spectral_norm=use_spectral_norm,
        normalize=normalize_embeddings,
        target_scale=embedding_target_scale,
    )
    embedding_reduction.to(device)

    n_train_for_gp = train_pool.total_samples
    gp = VariationalGPHead(
        input_dim=embed_dim,
        n_inducing=n_inducing,
        n_train=n_train_for_gp,
        lengthscale_range=ls_range,
        lengthscale_prior=ls_prior,
        outputscale_prior=os_prior,
        mlp_hidden=mlp_hidden,
    )
    gp.to(device)

    # ---- Load checkpoint ----
    logger.info(f"Loading checkpoint from {initial_checkpoint}")
    load_checkpoint(
        path=initial_checkpoint,
        models=[model, embedding_reduction, gp],
        device=device,
    )

    # ---- Output directory ----
    out_dir = Path(cfg.output_dir) / cfg.run_id / acquisition
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resume from previous run if checkpoints exist ----
    resume_round = 0
    resume_epoch = -1
    resuming_mid_round = False
    existing_ckpts = sorted(
        out_dir.glob("checkpoint_round_*"),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if existing_ckpts:
        last_ckpt = existing_ckpts[-1]
        round_num = int(last_ckpt.name.split("_")[-1])
        indices_path = last_ckpt / "train_indices.pt"

        if indices_path.exists():
            is_incomplete = (last_ckpt / "round_started").exists() and not (
                last_ckpt / "round_complete"
            ).exists()

            if is_incomplete:
                has_training_state = any(last_ckpt.glob("checkpoint.*.pt"))
                if has_training_state:
                    loaded_epoch = load_checkpoint(
                        path=str(last_ckpt),
                        models=[model, embedding_reduction, gp],
                        device=device,
                    )
                    resume_epoch = loaded_epoch
                    logger.info(
                        f"Resuming incomplete round {round_num} from epoch "
                        f"{loaded_epoch + 1}/{fine_tune_epochs}: {last_ckpt}"
                    )
                else:
                    # Selection saved but training not started yet; load
                    # model from the previous completed round if available.
                    prev_ckpt = out_dir / f"checkpoint_round_{round_num - 1}"
                    if prev_ckpt.exists() and (prev_ckpt / "train_indices.pt").exists():
                        load_checkpoint(
                            path=str(prev_ckpt),
                            models=[model, embedding_reduction, gp],
                            device=device,
                        )
                    logger.info(
                        f"Resuming incomplete round {round_num} from start "
                        f"(selection preserved): {last_ckpt}"
                    )

                train_pool.train_indices = torch.load(indices_path, weights_only=True)
                resume_round = round_num - 1
                resuming_mid_round = True
                logger.info(
                    f"Restored training pool: {len(train_pool)} samples "
                    f"from {indices_path}"
                )
            else:
                # Fully completed round (has sentinel or old-style checkpoint)
                resume_round = round_num
                logger.info(
                    f"Resuming from completed round {resume_round}: {last_ckpt}"
                )
                load_checkpoint(
                    path=str(last_ckpt),
                    models=[model, embedding_reduction, gp],
                    device=device,
                )
                train_pool.train_indices = torch.load(indices_path, weights_only=True)
                logger.info(
                    f"Restored training pool: {len(train_pool)} samples "
                    f"from {indices_path}"
                )

    # ---- DDP ----
    if dist_manager.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_manager.local_rank],
            output_device=device,
        )

    # ---- Strategies ----
    match acquisition:
        case "joint_uq":
            query_strategy = JointUQQueryStrategy(
                max_samples=samples_per_round, precision=precision
            )
        case "class_balanced_random":
            query_strategy = ClassBalancedRandomQueryStrategy(
                max_samples=samples_per_round, seed=random_seed
            )
        case "random":
            query_strategy = RandomQueryStrategy(
                max_samples=samples_per_round, seed=random_seed
            )
        case "latent_novelty":
            knn_k = int(getattr(cfg, "latent_novelty_knn_k", 10))
            query_strategy = LatentNoveltyQueryStrategy(
                max_samples=samples_per_round,
                precision=precision,
                knn_k=knn_k,
                cold_start_seed=random_seed,
            )
        case _:
            raise ValueError(
                f"Unknown acquisition strategy: {acquisition!r}. "
                f"Expected one of: 'joint_uq', 'random', 'class_balanced_random', "
                f"'latent_novelty'."
            )

    metrology = FieldMetrologyStrategy(precision=precision)
    label_strategy = DummyLabelStrategy()

    strategy_kwargs = {
        "gp_head": gp,
        "embedding_reduction": embedding_reduction,
        "surface_factors": surface_factors,
        "device": device,
    }

    # ---- Simple driver-like object for strategies to attach to ----
    class _DriverStub:
        def __init__(self, train_pool, test_pool, model):
            self.training_pool = train_pool
            self.validation_pool = test_pool
            self.learner = model
            self.active_learning_step_idx = 0
            self.log_dir = out_dir

    driver_stub = _DriverStub(train_pool, test_pool, model)
    query_strategy.attach(driver_stub)
    metrology.attach(driver_stub)
    label_strategy.attach(driver_stub)

    import logging as _logging

    query_strategy.logger.setLevel(_logging.INFO)
    metrology.logger.setLevel(_logging.INFO)

    # ---- Load existing metrology records if resuming ----
    metrics_path = out_dir / "validation_metrics.json"
    if (resume_round > 0 or resuming_mid_round) and metrics_path.exists():
        metrology.load_records(metrics_path)
        logger.info(f"Loaded {len(metrology.records)} existing metric records")

    is_rank0 = dist_manager.rank == 0

    # ---- Initial evaluation (round 0, no training) — skip if resuming ----
    if resume_round == 0 and not resuming_mid_round:
        logger.info("=== Round 0: baseline evaluation ===")
        metrology.compute(**strategy_kwargs)
        if is_rank0:
            metrology.serialize_records(metrics_path)

    # ---- Active learning loop ----
    start_round = resume_round + 1
    for al_round in range(start_round, al_rounds + 1):
        driver_stub.active_learning_step_idx = al_round
        ckpt_dir = out_dir / f"checkpoint_round_{al_round}"
        logger.info(f"\n{'=' * 60}")
        logger.info(f"=== Active Learning Round {al_round}/{al_rounds} ===")

        if resuming_mid_round:
            logger.info(
                f"Resuming mid-round (skipping sample selection). "
                f"Training pool size: {len(train_pool)}"
            )
            resuming_mid_round = False
        else:
            logger.info(f"Training pool size: {len(train_pool)}")
            logger.info(f"Unlabeled pool size: {len(train_pool.unlabeled_indices())}")

            # ---- Query (all ranks score in parallel) ----
            query_queue: Queue = Queue()
            query_strategy.sample(query_queue, **strategy_kwargs)

            selected_indices: list[int] = []
            label_queue: Queue = Queue()
            label_strategy.label(query_queue, label_queue)
            while not label_queue.empty():
                selected_indices.append(label_queue.get())

            # ---- Add to training pool (all ranks, same indices) ----
            for flat_idx in selected_indices:
                train_pool.append(flat_idx)

            logger.info(f"Training pool after selection: {len(train_pool)}")

            # ---- Persist selection + train indices immediately ----
            if is_rank0:
                history_path = out_dir / "selection_history.json"
                existing_history = []
                if history_path.exists():
                    with open(history_path) as f:
                        existing_history = json.load(f)
                existing_history.extend(query_strategy.selection_history)
                with open(history_path, "w") as f:
                    json.dump(existing_history, f, indent=2)
                query_strategy.selection_history.clear()

                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(train_pool.train_indices, ckpt_dir / "train_indices.pt")
                (ckpt_dir / "round_started").touch()

        # ---- Fine-tune ----
        # Fresh optimizer + cosine schedule each round (Ash & Adams, NeurIPS 2020).
        # Same base LR every round — new OOD samples need full learning capacity.
        # LR structure matches baseline training: backbone at base_lr,
        # embedding reduction at base_lr, GP variational/kernel params at 10x base_lr.
        base_lr = fine_tune_lr
        model.train()
        embedding_reduction.train()
        gp.train()

        backbone_model = model.module if hasattr(model, "module") else model
        param_groups = [
            {"params": backbone_model.parameters(), "lr": base_lr * 0.1},
            {"params": embedding_reduction.parameters(), "lr": base_lr},
        ]
        if hasattr(gp, "gp_layer"):
            param_groups.extend(
                [
                    {
                        "params": gp.gp_layer.variational_parameters(),
                        "lr": base_lr * 10,
                    },
                    {"params": gp.gp_layer.hyperparameters(), "lr": base_lr * 10},
                    {"params": gp.likelihood.parameters(), "lr": base_lr * 10},
                ]
            )
            if gp.feature_extractor is not None:
                param_groups.append(
                    {"params": gp.feature_extractor.parameters(), "lr": base_lr}
                )
        else:
            param_groups.append({"params": gp.parameters(), "lr": base_lr})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=fine_tune_epochs, eta_min=base_lr * 0.01
        )

        # Restore optimizer/scheduler state if resuming mid-round
        if resume_epoch >= 0:
            load_checkpoint(
                path=str(ckpt_dir),
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=resume_epoch,
            )
            start_epoch = resume_epoch + 1
            logger.info(
                f"Restored optimizer/scheduler from epoch {resume_epoch}, "
                f"continuing from epoch {start_epoch}"
            )
            resume_epoch = -1
        else:
            start_epoch = 0

        # Build DDP-sharded DataLoaders for both data sources
        num_replicas = dist_manager.world_size
        data_rank = dist_manager.rank

        # ---- Replay cap (CAL-style partial replay) ----
        # If replay_cap is set, each round draws a fresh random subset of
        # `replay_cap` Fastback samples instead of replaying the full base
        # training corpus.  Subset is deterministic across DDP ranks via a
        # per-round seed so all ranks shard the same indices.
        replay_cap = getattr(cfg, "replay_cap", None)
        replay_seed_base = int(getattr(cfg, "replay_seed_base", 12345))
        if base_train_datapipe is not None and replay_cap is not None:
            full_n = len(base_train_datapipe.dataset)
            cap = min(int(replay_cap), full_n)
            replay_gen = torch.Generator()
            replay_gen.manual_seed(replay_seed_base + al_round)
            replay_indices = torch.randperm(full_n, generator=replay_gen)[:cap].tolist()
            base_dataset_for_round = torch.utils.data.Subset(
                base_train_datapipe.dataset, replay_indices
            )
            if is_rank0:
                logger.info(
                    f"  Replay cap: {cap}/{full_n} Fastback samples this round "
                    f"(replay seed={replay_seed_base + al_round})"
                )
        elif base_train_datapipe is not None:
            base_dataset_for_round = base_train_datapipe.dataset
        else:
            base_dataset_for_round = None

        # Base Fastback training data with DistributedSampler
        base_sampler = None
        base_dl = None
        n_base = 0
        if base_dataset_for_round is not None:
            base_sampler = torch.utils.data.distributed.DistributedSampler(
                base_dataset_for_round,
                num_replicas=num_replicas,
                rank=data_rank,
                shuffle=True,
                drop_last=True,
            )
            base_dl = torch.utils.data.DataLoader(
                base_dataset_for_round,
                batch_size=1,
                sampler=base_sampler,
                num_workers=0,
            )
            n_base = len(base_dataset_for_round)

        # AL-selected data with DistributedSampler
        al_sampler = None
        al_dl = None
        if len(train_pool) > 0:
            al_sampler = torch.utils.data.distributed.DistributedSampler(
                train_pool,
                num_replicas=num_replicas,
                rank=data_rank,
                shuffle=True,
                drop_last=False,
            )
            al_dl = torch.utils.data.DataLoader(
                train_pool,
                batch_size=1,
                sampler=al_sampler,
                num_workers=0,
            )

        n_al = len(train_pool)
        per_gpu_base = n_base // num_replicas if base_dl else 0
        per_gpu_al = (n_al + num_replicas - 1) // num_replicas if al_dl else 0
        logger.info(
            f"Fine-tuning for {fine_tune_epochs} epochs: "
            f"{n_base} base + {n_al} AL = {n_base + n_al} total "
            f"(~{per_gpu_base + per_gpu_al} per GPU) "
            f"(LR: backbone={base_lr * 0.1:.2e}, embed={base_lr:.2e}, "
            f"GP variational/kernel={base_lr * 10:.2e}, "
            f"prefetch_depth={prefetch_depth})"
        )

        for epoch in range(start_epoch, fine_tune_epochs):
            epoch_loss = 0.0
            n_batches = 0

            # Base training data (Fastback replay, DDP-sharded)
            if base_dl is not None:
                base_sampler.set_epoch(epoch)
                for batch in base_dl:
                    batch = base_train_datapipe(
                        {
                            k: v[0] if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()
                        }
                    )
                    batch = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    epoch_loss += train_one_batch(
                        batch,
                        backbone_model,
                        embedding_reduction,
                        gp,
                        surface_factors,
                        device,
                        precision,
                        optimizer,
                        lambda_gp,
                        lambda_consistency,
                        consistency_detach,
                        consistency_every_n,
                        n_batches,
                        dist_manager,
                        accumulation_steps=accumulation_steps,
                    )
                    n_batches += 1

            # AL-selected samples (DDP-sharded)
            if al_dl is not None:
                al_sampler.set_epoch(epoch)
                ### In-process prefetch via ``AeroDataPool.prefetch`` (mirrors
                ### the same hook in train_ceiling.py).  ``prefetch_depth=0``
                ### preserves the legacy synchronous behavior; >0 materializes
                ### the per-rank sampler permutation -- deterministic for this
                ### (seed, epoch) -- and submits reads ahead on each per-class
                ### CAEDataset's ThreadPoolExecutor so file I/O overlaps GPU
                ### compute.  In-process on purpose: AeroDataPool holds
                ### GPU-resident surface_factors that are not safe to pickle
                ### across DataLoader worker subprocess boundaries.
                al_sampler_order: list[int] = []
                if prefetch_depth > 0:
                    al_sampler_order = list(iter(al_sampler))
                    for j in range(min(prefetch_depth, len(al_sampler_order))):
                        local_idx = al_sampler_order[j]
                        flat_idx = int(train_pool.train_indices[local_idx].item())
                        train_pool.prefetch(flat_idx)
                for i, batch in enumerate(al_dl):
                    if prefetch_depth > 0:
                        next_j = i + prefetch_depth
                        if next_j < len(al_sampler_order):
                            next_local = al_sampler_order[next_j]
                            next_flat = int(train_pool.train_indices[next_local].item())
                            train_pool.prefetch(next_flat)
                    batch = {
                        k: v.squeeze(0).to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    epoch_loss += train_one_batch(
                        batch,
                        backbone_model,
                        embedding_reduction,
                        gp,
                        surface_factors,
                        device,
                        precision,
                        optimizer,
                        lambda_gp,
                        lambda_consistency,
                        consistency_detach,
                        consistency_every_n,
                        n_batches,
                        dist_manager,
                        accumulation_steps=accumulation_steps,
                    )
                    n_batches += 1

            scheduler.step()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"  Epoch {epoch + 1}/{fine_tune_epochs} "
                    f"avg_loss={epoch_loss / max(n_batches, 1):.6f} "
                    f"({n_batches} batches)"
                )

            # Periodic intra-round checkpoint (survives Slurm timeouts)
            if is_rank0 and (
                (epoch + 1) % save_interval == 0 or epoch == fine_tune_epochs - 1
            ):
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    path=str(ckpt_dir),
                    models=[backbone_model, embedding_reduction, gp],
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                )

        # ---- Evaluate (all ranks compute, rank 0 saves) ----
        metrology.compute(**strategy_kwargs)
        if is_rank0:
            metrology.serialize_records(metrics_path)

        # ---- Mark round as complete ----
        if is_rank0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "round_complete").touch()

    if is_rank0:
        logger.info(f"Saved selection history to {out_dir / 'selection_history.json'}")
        logger.info(f"Saved validation metrics to {metrics_path}")

    logger.info("Active learning experiment complete.")


if __name__ == "__main__":
    main()
