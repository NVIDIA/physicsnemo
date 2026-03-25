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
Unified External Aerodynamics - Surface Training Script

Trains a GeoTransolver (or other point-cloud model) on surface pressure
and wall shear stress using the mesh datapipe infrastructure.

Usage::

    # Single-GPU
    python src/train.py

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=N src/train.py

    # I/O benchmark: iterate dataloaders without model logic
    python src/train.py benchmark_io=true profile=true
    python src/train.py benchmark_io=true training.benchmark_max_steps=20
"""

import os
import sys
import time
import collections
from contextlib import nullcontext
from pathlib import Path

import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf

import torch
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from tabulate import tabulate

from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.profiling import profile, Profiler

from physicsnemo import datapipes  # noqa: F401 - registers ${dp:...} resolver
from physicsnemo.datapipes import DataLoader

from datasets import build_surface_dataset, load_dataset_config
from collate import surface_collate
from metrics import MetricCalculator
from loss import LossCalculator
from utils import build_muon_optimizer

from physicsnemo.core.version_check import check_version_spec

TE_AVAILABLE = check_version_spec("transformer_engine", hard_fail=False)

if TE_AVAILABLE:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
else:
    te, Format, DelayedScaling = None, None, None

torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
torch.serialization.add_safe_globals([collections.defaultdict])
torch.serialization.add_safe_globals([dict])
torch.serialization.add_safe_globals([int])
torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
torch.serialization.add_safe_globals([omegaconf.base.Metadata])


def get_autocast_context(precision: str):
    """Return an autocast context manager for the given precision.

    Parameters
    ----------
    precision : str
        One of ``"float16"``, ``"bfloat16"``, ``"float8"``, or ``"float32"``.
        For ``"float8"``, Transformer Engine must be available.

    Returns
    -------
    contextlib.AbstractContextManager
        An autocast context manager for the requested precision, or a
        no-op ``nullcontext`` when no casting is needed.
    """
    if precision == "float16":
        return autocast("cuda", dtype=torch.float16)
    elif precision == "bfloat16":
        return autocast("cuda", dtype=torch.bfloat16)
    elif precision == "float8" and TE_AVAILABLE:
        fp8_format = Format.HYBRID
        fp8_recipe = DelayedScaling(
            fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max"
        )
        return te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
    else:
        return nullcontext()


def forward_pass(
    batch: dict[str, torch.Tensor],
    model: torch.nn.Module,
    precision: str,
    loss_calculator: LossCalculator,
    metric_calculator: MetricCalculator,
) -> tuple[torch.Tensor, dict[str, float], tuple]:
    """Run forward pass, compute loss and metrics.

    Parameters
    ----------
    batch : dict
        Keys: ``geometry`` (B,N,3), ``local_embedding`` (B,N,6),
        ``global_embedding`` (B,1,3), ``fields`` (B,N,4).
    model : torch.nn.Module
        Point-cloud model with forward(local_embedding, geometry, global_embedding, ...).
    precision : str
        One of "float32", "float16", "bfloat16", "float8".
    loss_calculator : LossCalculator
    metric_calculator : MetricCalculator

    Returns
    -------
    loss, metrics_dict, (outputs, targets)
    """
    targets = batch.pop("fields")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(precision)
    if dtype is not None:
        batch = {k: v.to(dtype) for k, v in batch.items()}

    with get_autocast_context(precision):
        outputs = model(**batch)
        loss, loss_dict = loss_calculator(outputs, targets)

    metrics = {k: v.item() for k, v in loss_dict.items()}
    with torch.no_grad():
        metrics.update(metric_calculator(outputs, targets))

    return loss, metrics, (outputs, targets)


@profile
def train_epoch(
    dataloader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_calculator: LossCalculator,
    metric_calculator: MetricCalculator,
    logger,
    writer,
    epoch: int,
    cfg: DictConfig,
    dist_manager: DistributedManager,
    scaler: GradScaler | None = None,
) -> tuple[float, dict[str, float]]:
    """Run one training epoch over the dataloader.

    Iterates through all batches, computes forward pass, back-propagates
    gradients, and logs per-step and per-epoch statistics to TensorBoard.

    Parameters
    ----------
    dataloader : DataLoader
        Training dataloader yielding ``dict[str, Tensor]`` batches.
    model : torch.nn.Module
        The model to train (already on ``dist_manager.device``).
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning-rate scheduler.  Updated per step or per epoch depending
        on ``cfg.training.scheduler_update_mode``.
    loss_calculator : LossCalculator
        Computes the training loss from model outputs and targets.
    metric_calculator : MetricCalculator
        Computes evaluation metrics (L1, L2, MAE, etc.).
    logger : RankZeroLoggingWrapper
        Logger for console output.
    writer : SummaryWriter or None
        TensorBoard writer for training scalars (rank-0 only).
    epoch : int
        Current epoch index (0-based).
    cfg : DictConfig
        Full Hydra config; uses ``cfg.profile`` and ``cfg.training``.
    dist_manager : DistributedManager
        Distributed training manager.
    scaler : torch.amp.GradScaler or None, optional
        Gradient scaler for mixed-precision (float16) training.

    Returns
    -------
    avg_loss : float
        Mean training loss over all batches.
    avg_metrics : dict[str, float]
        Mean per-metric values over all batches.
    """
    model.train()
    total_loss = 0.0
    total_metrics: dict[str, float] = {}
    precision = getattr(cfg, "precision", "float32")
    n_batches = 0
    num_steps = len(dataloader)
    epoch_t0 = time.perf_counter()

    step_t0 = time.perf_counter()
    for i, batch in enumerate(dataloader):
        batch = {k: v.to(dist_manager.device) for k, v in batch.items()}

        loss, metrics, _ = forward_pass(
            batch, model, precision, loss_calculator, metric_calculator
        )

        optimizer.zero_grad()
        if precision == "float16" and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if cfg.training.get("scheduler_update_mode", "epoch") == "step":
            scheduler.step()

        this_loss = loss.detach().item()
        total_loss += this_loss
        n_batches += 1

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + (
                v if isinstance(v, float) else v.item()
            )

        step_dt = time.perf_counter() - step_t0

        mem_gb = (
            torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
        )
        logger.info(
            f"Epoch {epoch} [{i + 1}/{num_steps}] "
            f"Loss: {this_loss:.6f} "
            f"Step: {step_dt:.3f}s "
            f"Mem: {mem_gb:.2f}GB"
        )

        if cfg.profile and i >= 10:
            break
        step_t0 = time.perf_counter()

    epoch_dt = time.perf_counter() - epoch_t0
    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {k: v / max(n_batches, 1) for k, v in total_metrics.items()}

    logger.info(
        f"Epoch {epoch} train done in {epoch_dt:.1f}s "
        f"({n_batches} steps, {epoch_dt / max(n_batches, 1):.3f}s/step avg)"
    )

    if dist_manager.rank == 0 and writer is not None:
        writer.add_scalar("epoch/loss", avg_loss, epoch)
        for k, v in avg_metrics.items():
            writer.add_scalar(f"epoch/{k}", v, epoch)

    return avg_loss, avg_metrics


@profile
def val_epoch(
    dataloader,
    model: torch.nn.Module,
    loss_calculator: LossCalculator,
    metric_calculator: MetricCalculator,
    logger,
    val_writer,
    epoch: int,
    cfg: DictConfig,
    dist_manager: DistributedManager,
    phys_metric_calculator: MetricCalculator | None = None,
    target_config: dict[str, str] | None = None,
    normalizer=None,
    nondim_transform=None,
    metadata: dict | None = None,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Run one validation epoch and optionally compute physical-space metrics.

    When ``phys_metric_calculator`` and ``metadata`` are both provided,
    model outputs and targets are converted back to physical units via
    :func:`_to_physical` and a second set of metrics is computed.

    Parameters
    ----------
    dataloader : DataLoader
        Validation dataloader yielding ``dict[str, Tensor]`` batches.
    model : torch.nn.Module
        The model to evaluate (already on ``dist_manager.device``).
    loss_calculator : LossCalculator
        Computes the validation loss.
    metric_calculator : MetricCalculator
        Computes normalised-space metrics.
    logger : RankZeroLoggingWrapper
        Logger for console output.
    val_writer : SummaryWriter or None
        TensorBoard writer for validation scalars (rank-0 only).
    epoch : int
        Current epoch index (0-based).
    cfg : DictConfig
        Full Hydra config; uses ``cfg.profile`` and ``cfg.precision``.
    dist_manager : DistributedManager
        Distributed training manager.
    phys_metric_calculator : MetricCalculator or None, optional
        If provided, computes metrics in physical (dimensional) space.
    target_config : dict[str, str] or None, optional
        Maps target field names to their types (``"scalar"`` / ``"vector"``).
        Required when ``phys_metric_calculator`` is given.
    normalizer : NormalizeMeshFields or None, optional
        The z-score normalizer extracted from the training pipeline.
    nondim_transform : NonDimensionalizeByMetadata or None, optional
        The non-dimensionalization transform from the training pipeline.
    metadata : dict or None, optional
        Freestream conditions (``U_inf``, ``rho_inf``, ``p_inf``, etc.)
        needed to invert non-dimensionalization.

    Returns
    -------
    avg_loss : float
        Mean validation loss over all batches.
    avg_metrics : dict[str, float]
        Mean normalised-space metrics.
    avg_phys_metrics : dict[str, float]
        Mean physical-space metrics (empty dict if not computed).
    """
    model.eval()
    total_loss = 0.0
    total_metrics: dict[str, float] = {}
    total_phys_metrics: dict[str, float] = {}
    precision = getattr(cfg, "precision", "float32")
    n_batches = 0
    num_steps = len(dataloader)
    epoch_t0 = time.perf_counter()
    compute_phys = phys_metric_calculator is not None and metadata

    with torch.no_grad():
        step_t0 = time.perf_counter()
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(dist_manager.device) for k, v in batch.items()}

            loss, metrics, (outputs, targets) = forward_pass(
                batch, model, precision, loss_calculator, metric_calculator
            )

            if compute_phys:
                phys_out = _to_physical(
                    outputs, target_config, normalizer, nondim_transform, metadata
                )
                phys_tgt = _to_physical(
                    targets, target_config, normalizer, nondim_transform, metadata
                )
                phys_metrics = phys_metric_calculator(phys_out, phys_tgt)
                for k, v in phys_metrics.items():
                    val = v if isinstance(v, float) else v.item()
                    total_phys_metrics[k] = total_phys_metrics.get(k, 0.0) + val

            step_dt = time.perf_counter() - step_t0
            total_loss += loss.item()
            n_batches += 1
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + (
                    v if isinstance(v, float) else v.item()
                )

            logger.info(
                f"Val Epoch {epoch} [{i + 1}/{num_steps}] "
                f"Loss: {loss.item():.6f} "
                f"Step: {step_dt:.3f}s"
            )

            if cfg.profile and i >= 10:
                break
            step_t0 = time.perf_counter()

    epoch_dt = time.perf_counter() - epoch_t0
    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {k: v / max(n_batches, 1) for k, v in total_metrics.items()}
    avg_phys_metrics = {k: v / max(n_batches, 1) for k, v in total_phys_metrics.items()}

    logger.info(
        f"Epoch {epoch} val done in {epoch_dt:.1f}s "
        f"({n_batches} steps, {epoch_dt / max(n_batches, 1):.3f}s/step avg)"
    )

    if dist_manager.rank == 0 and val_writer is not None:
        val_writer.add_scalar("epoch/loss", avg_loss, epoch)
        for k, v in avg_metrics.items():
            val_writer.add_scalar(f"epoch/{k}", v, epoch)
        for k, v in avg_phys_metrics.items():
            base = k.removeprefix("phys/")
            val_writer.add_scalar(f"epoch/phys/{base}", v, epoch)

    return avg_loss, avg_metrics, avg_phys_metrics


@profile
def benchmark_io_epoch(
    dataloader,
    label: str,
    logger,
    max_steps: int | None = None,
) -> None:
    """Iterate a dataloader without any model logic and report I/O timing.

    Parameters
    ----------
    dataloader : DataLoader
        Dataloader to benchmark.
    label : str
        Human-readable label for logging (e.g. ``"train"`` or ``"val"``).
    logger : RankZeroLoggingWrapper
        Logger for console output.
    max_steps : int or None, optional
        Stop after this many batches.  ``None`` means exhaust the loader.
    """
    import statistics

    num_steps = len(dataloader)
    times: list[float] = []

    step_t0 = time.perf_counter()
    for i, batch in enumerate(dataloader):
        dt = time.perf_counter() - step_t0
        times.append(dt)

        mem_gb = (
            torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
        )
        shapes = "  ".join(f"{k}:{tuple(v.shape)}" for k, v in batch.items())
        logger.info(
            f"  [{label}] [{i + 1}/{num_steps}] "
            f"dt={dt:.4f}s  Mem={mem_gb:.2f}GB  {shapes}"
        )

        if max_steps is not None and i + 1 >= max_steps:
            break
        step_t0 = time.perf_counter()

    if not times:
        logger.info(f"  [{label}] empty dataloader")
        return

    total = sum(times)
    mean = statistics.mean(times)
    med = statistics.median(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    p95 = sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else times[0]

    logger.info(
        f"  [{label}] {len(times)} batches in {total:.2f}s  "
        f"mean={mean:.4f}s  median={med:.4f}s  std={std:.4f}s  p95={p95:.4f}s  "
        f"throughput={len(times) / total:.2f} batches/sec"
    )


def _extract_pipeline_transforms(datasets: list) -> tuple:
    """Find NormalizeMeshFields and NonDimensionalizeByMetadata in transform chains.

    Returns (normalizer, nondim) instances from the first dataset that has them,
    or (None, None) if not found.
    """
    from physicsnemo.datapipes.transforms.mesh import NormalizeMeshFields
    from nondim import NonDimensionalizeByMetadata

    normalizer = None
    nondim = None
    for ds in datasets:
        for t in getattr(ds, "transforms", []):
            if isinstance(t, NormalizeMeshFields) and normalizer is None:
                normalizer = t
            if isinstance(t, NonDimensionalizeByMetadata) and nondim is None:
                nondim = t
    return normalizer, nondim


def build_dataloaders(cfg: DictConfig):
    """Build train and val dataloaders from dataset configs."""
    recipe_root = Path(__file__).resolve().parent.parent
    batch_size = cfg.training.get("batch_size", 1)
    sampling_resolution = cfg.dataset.get("sampling_resolution", None)
    augment = cfg.get("augment", False)
    dist_manager = DistributedManager()
    use_distributed = dist_manager.world_size > 1

    # DataLoader / MeshDataset performance tuning from cfg.dataloader
    dl_cfg = cfg.get("dataloader", {})
    prefetch_factor = dl_cfg.get("prefetch_factor", 2)
    num_streams = dl_cfg.get("num_streams", 4)
    use_streams = dl_cfg.get("use_streams", False)
    num_workers = dl_cfg.get("num_workers", 1)
    pin_memory = dl_cfg.get("pin_memory", False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_datasets = []
    val_datasets = []
    first_metadata = None
    for ds_key in cfg.data:
        ds_cfg_block = cfg.data[ds_key]
        config_path = recipe_root / ds_cfg_block.config
        if not config_path.exists():
            continue
        train_dir = ds_cfg_block.get("train_dir", "")
        if train_dir and not Path(train_dir).exists():
            continue
        ds_yaml = load_dataset_config(config_path)
        if sampling_resolution is not None:
            ds_yaml = OmegaConf.merge(
                ds_yaml, {"sampling_resolution": sampling_resolution}
            )
        if first_metadata is None:
            first_metadata = OmegaConf.to_container(
                OmegaConf.select(ds_yaml, "metadata", default=OmegaConf.create({})),
                resolve=True,
            )
        train_datasets.append(
            build_surface_dataset(
                ds_yaml,
                augment=augment,
                device=device,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        )

        val_datadir = OmegaConf.select(ds_yaml, "val_datadir", default=None)
        if val_datadir and Path(val_datadir).exists():
            val_yaml = OmegaConf.merge(ds_yaml, {"train_datadir": val_datadir})
            val_datasets.append(
                build_surface_dataset(
                    val_yaml,
                    augment=False,
                    device=device,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
            )

    if not train_datasets:
        raise RuntimeError("No valid datasets found. Check data paths in config.")

    normalizer, nondim_transform = _extract_pipeline_transforms(train_datasets)

    if len(train_datasets) == 1:
        train_dataset = train_datasets[0]
    else:
        from physicsnemo.datapipes import MultiDataset

        train_dataset = MultiDataset(*train_datasets, output_strict=False)

    if val_datasets:
        if len(val_datasets) == 1:
            val_dataset = val_datasets[0]
        else:
            from physicsnemo.datapipes import MultiDataset

            val_dataset = MultiDataset(*val_datasets, output_strict=False)
    else:
        val_dataset = train_dataset

    train_sampler = None
    val_sampler = None
    if use_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, drop_last=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=surface_collate,
        drop_last=True,
        prefetch_factor=prefetch_factor,
        num_streams=num_streams,
        use_streams=use_streams,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=surface_collate,
        drop_last=False,
        prefetch_factor=prefetch_factor,
        num_streams=num_streams,
        use_streams=use_streams,
    )

    return train_loader, val_loader, normalizer, nondim_transform, first_metadata or {}


_NONDIM_TYPE_MAP = {"scalar": "pressure", "vector": "stress"}


def _to_physical(
    tensor: torch.Tensor,
    target_config: dict[str, str],
    normalizer,
    nondim_transform,
    metadata: dict,
) -> torch.Tensor:
    """Convert a model-space tensor (normalized + non-dim) back to physical units.

    Chains two inverse operations using the existing transform instances:
    1. ``NormalizeMeshFields.inverse_tensor`` -- undo z-score normalization
    2. ``NonDimensionalizeByMetadata.inverse_tensor`` -- undo non-dimensionalization
    """
    if not metadata:
        return tensor

    out = tensor
    device, dtype = tensor.device, tensor.dtype

    # Step 1: undo z-score normalization
    if normalizer is not None:
        out = normalizer.inverse_tensor(out, target_config)

    # Step 2: undo non-dimensionalization
    if nondim_transform is not None:
        nondim_fields = {
            name: _NONDIM_TYPE_MAP.get(ftype, ftype)
            for name, ftype in target_config.items()
        }
        U_inf = torch.tensor(metadata["U_inf"], dtype=dtype, device=device)
        rho_inf = torch.tensor(metadata["rho_inf"], dtype=dtype, device=device)
        p_inf = torch.tensor(metadata["p_inf"], dtype=dtype, device=device)
        q_inf = 0.5 * rho_inf * (U_inf * U_inf).sum()
        U_inf_mag = (U_inf * U_inf).sum().sqrt()
        out = nondim_transform.inverse_tensor(
            out, nondim_fields, q_inf, p_inf, U_inf_mag
        )

    return out


@profile
def main(cfg: DictConfig):
    """Run the full training loop, or I/O-only benchmark when ``benchmark_io=true``.

    Orchestrates the complete training workflow:

    1. Initialise distributed training and TensorBoard writers.
    2. Build train/val dataloaders and extract pipeline transforms.
    3. If ``cfg.benchmark_io`` is true, iterate dataloaders to measure
       I/O throughput and return early (no model, no optimizer).
    4. Otherwise, instantiate the model, optimizer, and run the normal
       train/val epoch loop with checkpointing.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config containing ``model``, ``training``, ``dataset``,
        ``data``, ``output_dir``, ``run_id``, ``precision``, ``compile``,
        ``profile``, ``benchmark_io``, and related keys.
    """
    DistributedManager.initialize()
    dist_manager = DistributedManager()
    logger = RankZeroLoggingWrapper(PythonLogger(name="training"), dist_manager)

    checkpoint_dir = getattr(cfg, "checkpoint_dir", None) or cfg.output_dir

    writer = None
    val_writer = None
    if dist_manager.rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        writer = SummaryWriter(
            log_dir=os.path.join(cfg.output_dir, cfg.run_id, "train")
        )
        val_writer = SummaryWriter(
            log_dir=os.path.join(cfg.output_dir, cfg.run_id, "val")
        )

    logger.info(f"Config:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")

    train_loader, val_loader, normalizer, nondim_transform, ds_metadata = (
        build_dataloaders(cfg)
    )
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    # -- I/O benchmark mode: iterate dataloaders, skip model entirely -----------
    if cfg.get("benchmark_io", False):
        num_epochs = cfg.training.num_epochs
        max_steps = cfg.training.get("benchmark_max_steps", None)
        logger.info(
            f"benchmark_io=True  — benchmarking dataloader I/O only "
            f"({num_epochs} epoch(s), max_steps={max_steps})"
        )
        with torch.no_grad(), Profiler():
            for epoch in range(num_epochs):
                logger.info(f"--- Epoch {epoch + 1}/{num_epochs} ---")
                train_loader.set_epoch(epoch)
                benchmark_io_epoch(train_loader, "train", logger, max_steps=max_steps)
                benchmark_io_epoch(val_loader, "val", logger, max_steps=max_steps)
        logger.info("benchmark_io complete!")
        return

    # -- Normal training path ---------------------------------------------------
    model = hydra.utils.instantiate(cfg.model, _convert_="partial")
    logger.info(f"Model: {model.__class__.__name__}")
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {num_params:,}")

    model.to(dist_manager.device)

    if dist_manager.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_manager.local_rank],
            output_device=dist_manager.device,
        )

    if normalizer is not None:
        logger.info(
            f"Normalization: {', '.join(f'{k}({v["type"]})' for k, v in normalizer.stats.items())}"
        )

    optimizer = build_muon_optimizer(model, cfg, compile_optimizer=cfg.compile)
    logger.info(f"Optimizer: {optimizer}")
    scheduler = hydra.utils.instantiate(cfg.training.scheduler, optimizer=optimizer)

    precision = cfg.precision
    scaler = GradScaler() if precision == "float16" else None

    ds_cfg = cfg.dataset
    targets = omegaconf.OmegaConf.to_container(ds_cfg.targets, resolve=True)
    metrics_list = omegaconf.OmegaConf.to_container(
        ds_cfg.get("metrics", ["l1", "l2", "mae"]), resolve=True
    )

    active_data_keys = list(cfg.data.keys())
    if len(active_data_keys) > 1:
        prefix = cfg.get("model_type", "merged")
        logger.info(
            f"Multiple datasets active ({', '.join(active_data_keys)}); "
            f"using prefix '{prefix}' for metrics"
        )
    else:
        prefix = ""

    phys_prefix = f"phys/{prefix}" if prefix else "phys"
    metric_calculator = MetricCalculator(
        target_config=targets, metrics=metrics_list, prefix=prefix
    )
    use_phys_metrics = len(active_data_keys) == 1
    phys_metric_calculator = (
        MetricCalculator(
            target_config=targets, metrics=metrics_list, prefix=phys_prefix
        )
        if use_phys_metrics
        else None
    )
    loss_calculator = LossCalculator(
        target_config=targets,
        loss_type=cfg.training.get("loss_type", "huber"),
        prefix=prefix,
    )
    logger.info(f"Loss: {loss_calculator}")
    logger.info(f"Metrics: {metric_calculator}")

    ckpt_args = {
        "path": os.path.join(checkpoint_dir, cfg.run_id, "checkpoints"),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    loaded_epoch = load_checkpoint(device=dist_manager.device, **ckpt_args)

    if cfg.compile:
        model = torch.compile(model)

    num_epochs = cfg.training.num_epochs
    logger.info(f"Starting training for {num_epochs} epochs...")

    # Unless profiling is enabled, this is a null context:
    with Profiler():
        for epoch in range(loaded_epoch, num_epochs):
            logger.info(f"--- Epoch {epoch + 1}/{num_epochs} ---")
            train_loader.set_epoch(epoch)

            train_loss, train_metrics = train_epoch(
                train_loader,
                model,
                optimizer,
                scheduler,
                loss_calculator,
                metric_calculator,
                logger,
                writer,
                epoch,
                cfg,
                dist_manager,
                scaler,
            )

            val_loss, val_metrics, val_phys_metrics = val_epoch(
                val_loader,
                model,
                loss_calculator,
                metric_calculator,
                logger,
                val_writer,
                epoch,
                cfg,
                dist_manager,
                phys_metric_calculator=phys_metric_calculator,
                target_config=targets,
                normalizer=normalizer,
                nondim_transform=nondim_transform,
                metadata=ds_metadata,
            )

            if dist_manager.rank == 0:
                all_keys = list(dict.fromkeys(list(train_metrics) + list(val_metrics)))
                phys_by_base = {}
                for k, v in val_phys_metrics.items():
                    base = k.removeprefix("phys/")
                    phys_by_base[base] = v

                headers = ["Metric", "Train", "Val"]
                if use_phys_metrics:
                    headers.append("Val (phys)")

                rows = []
                for k in all_keys:
                    row = [
                        k,
                        f"{train_metrics.get(k, float('nan')):.6f}",
                        f"{val_metrics.get(k, float('nan')):.6f}",
                    ]
                    if use_phys_metrics and not k.startswith("loss/"):
                        row.append(f"{phys_by_base.get(k, float('nan')):.6f}")
                    elif use_phys_metrics:
                        row.append("")
                    rows.append(row)

                table = tabulate(rows, headers=headers, tablefmt="pretty")
                logger.info(
                    f"\nEpoch [{epoch}/{cfg.training.num_epochs}] "
                    f"Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}\n"
                    f"{table}\n"
                )

            if epoch % cfg.training.save_interval == 0 and dist_manager.rank == 0:
                save_checkpoint(**ckpt_args, epoch=epoch + 1)
                if normalizer is not None:
                    norm_path = os.path.join(ckpt_args["path"], "norm_stats.pt")
                    torch.save(normalizer.stats, norm_path)

            if cfg.training.get("scheduler_update_mode", "epoch") == "epoch":
                scheduler.step()

    if dist_manager.rank == 0:
        if writer is not None:
            writer.close()
        if val_writer is not None:
            val_writer.close()

    logger.info("Training completed!")


@hydra.main(version_base=None, config_path="../conf", config_name="train_surface")
def launch(cfg: DictConfig):
    """Hydra entry point: configure profiling and delegate to :func:`main`.

    Parameters
    ----------
    cfg : DictConfig
        Hydra-composed config loaded from ``conf/train_surface.yaml``.
        When ``cfg.profile`` is truthy, torch profiling is enabled.
    """
    profiler = Profiler()
    if cfg.profile:
        profiler.enable("torch")
    profiler.initialize()
    main(cfg)
    profiler.finalize()


if __name__ == "__main__":
    launch()
