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
Unified External Aerodynamics Training Script

Trains a point-cloud model (GeoTransolver, Transolver, etc.) on surface
or volume fields using the mesh datapipe infrastructure.

Usage::

    # Single-GPU
    python src/train.py

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=N src/train.py

    # I/O benchmark: iterate dataloaders without model logic
    python src/train.py benchmark_io=true profile=true
    python src/train.py benchmark_io=true +training.benchmark_max_steps=20
"""

import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf

import json
from datetime import datetime, timezone

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

from datasets import (
    build_dataset,
    load_dataset_config,
    load_manifest,
    resolve_manifest_indices,
    ManifestSampler,
)
from collate import build_collate_fn
from metrics import MetricCalculator
from loss import LossCalculator
from utils import build_muon_optimizer, set_seed

from physicsnemo.core.version_check import OptionalImport

te = OptionalImport("transformer_engine.pytorch")
te_recipe = OptionalImport("transformer_engine.common.recipe")
TE_AVAILABLE = te.available


def _flatten_config(d: dict, parent: str = "", sep: str = ".") -> dict[str, str]:
    """Recursively flatten a nested dict into dot-separated key/value pairs."""
    items: dict[str, str] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            items.update(_flatten_config(v, key, sep))
        else:
            items[key] = str(v)
    return items


def _log_to_tensorboard(writer, metrics_dict, prefix, global_step):
    """Write metrics to TensorBoard with structured tag prefixes.

    Loss entries (keys starting with ``loss/``) are logged as
    ``{prefix}/{key}`` (e.g. ``iteration/loss/pressure``).  All other
    entries are treated as evaluation metrics and logged as
    ``{prefix}/metrics/{key}`` (e.g. ``epoch/metrics/pressure_l2``).
    """
    if writer is None:
        return
    for k, v in metrics_dict.items():
        if k.startswith("loss/"):
            tag = f"{prefix}/{k}"
        else:
            tag = f"{prefix}/metrics/{k}"
        val = v if isinstance(v, (int, float)) else v.item()
        writer.add_scalar(tag, val, global_step=global_step)


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
        fp8_format = te_recipe.Format.HYBRID
        fp8_recipe = te_recipe.DelayedScaling(
            fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max"
        )
        return te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
    else:
        return nullcontext()


def _recursive_to(obj, *args, **kwargs):
    """Apply ``.to()`` recursively through nested dicts/lists of tensors."""
    if isinstance(obj, torch.Tensor):
        return obj.to(*args, **kwargs)
    if isinstance(obj, dict):
        return {k: _recursive_to(v, *args, **kwargs) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_recursive_to(v, *args, **kwargs) for v in obj)
    return obj


def forward_pass(
    batch: dict[str, torch.Tensor],
    model: torch.nn.Module,
    precision: str,
    loss_calculator: LossCalculator,
    metric_calculator: MetricCalculator,
    *,
    broadcast_global: bool = False,
) -> tuple[torch.Tensor, dict[str, float], tuple]:
    """Run forward pass, compute loss and metrics.

    Parameters
    ----------
    batch : dict
        Model-ready batch produced by the collate function.  Must contain
        a ``"fields"`` key holding the prediction targets (popped before
        the forward call).  Values may be tensors or nested dicts of
        tensors (e.g. DoMINO's ``data_dict``).
    model : torch.nn.Module
        Point-cloud model whose ``forward`` accepts the remaining batch
        keys as keyword arguments.
    precision : str
        One of "float32", "float16", "bfloat16", "float8".
    loss_calculator : LossCalculator
    metric_calculator : MetricCalculator
    broadcast_global : bool, default False
        When ``True``, any tensor with spatial dimension 1 (e.g. global
        features shaped ``(B, 1, C)``) is expanded to match the largest
        spatial dimension in the batch.  Required for Transolver, whose
        ``forward`` concatenates ``[embedding, fx]`` along the last dim
        and therefore needs matching spatial sizes.

    Returns
    -------
    loss, metrics_dict, (outputs, targets)
    """
    targets = batch.pop("fields")

    if broadcast_global:
        max_n = max(
            v.shape[1]
            for v in batch.values()
            if isinstance(v, torch.Tensor) and v.ndim >= 3
        )
        batch = {
            k: v.expand(-1, max_n, -1)
            if isinstance(v, torch.Tensor) and v.ndim >= 3 and v.shape[1] == 1
            else v
            for k, v in batch.items()
        }

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(precision)
    if dtype is not None:
        batch = {k: _recursive_to(v, dtype) for k, v in batch.items()}

    with get_autocast_context(precision):
        outputs = model(**batch)

        # Models like DoMINO return (vol_output, surf_output); extract the
        # non-None element for single-mode training.
        if isinstance(outputs, tuple):
            outputs = next(o for o in outputs if o is not None)

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
    epoch: int,
    cfg: DictConfig,
    dist_manager: DistributedManager,
    scaler: GradScaler | None = None,
    broadcast_global: bool = False,
    train_writer: SummaryWriter | None = None,
    log_jsonl=None,
) -> tuple[float, dict[str, float]]:
    """Run one training epoch over the dataloader.

    Iterates through all batches, computes forward pass, back-propagates
    gradients, and logs per-step and per-epoch statistics to TensorBoard and JSONL.

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
    epoch : int
        Current epoch index (0-based).
    cfg : DictConfig
        Full Hydra config; uses ``cfg.profile`` and ``cfg.training``.
    dist_manager : DistributedManager
        Distributed training manager.
    scaler : torch.amp.GradScaler or None, optional
        Gradient scaler for mixed-precision (float16) training.
    broadcast_global : bool, default False
        Expand global (B,1,C) tensors to match the spatial dimension
        of other batch tensors before forwarding.

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
        batch = {k: _recursive_to(v, dist_manager.device) for k, v in batch.items()}

        loss, metrics, (outputs, targets) = forward_pass(
            batch,
            model,
            precision,
            loss_calculator,
            metric_calculator,
            broadcast_global=broadcast_global,
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

        # Per-step TensorBoard + JSONL logging
        global_step = epoch * num_steps + i
        if dist_manager.rank == 0:
            if train_writer is not None:
                _log_to_tensorboard(train_writer, metrics, "iteration", global_step)
                current_lr = scheduler.get_last_lr()[0]
                train_writer.add_scalar(
                    "iteration/lr", current_lr, global_step=global_step
                )
                train_writer.add_scalar(
                    "iteration/performance/mem_gb", mem_gb, global_step=global_step
                )
                train_writer.add_scalar(
                    "iteration/performance/step_time_s",
                    step_dt,
                    global_step=global_step,
                )
            if log_jsonl is not None:
                step_metrics = {
                    "loss": this_loss,
                    "mem_gb": mem_gb,
                    "step_time_s": step_dt,
                }
                step_metrics.update(metrics)
                log_jsonl({"phase": "step", "global_step": global_step, **step_metrics})

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

    if dist_manager.rank == 0:
        _log_to_tensorboard(train_writer, avg_metrics, "epoch", epoch)
        if log_jsonl is not None:
            epoch_log = {"loss": avg_loss}
            epoch_log.update(avg_metrics)
            log_jsonl({"phase": "train", "epoch": epoch, **epoch_log})

    return avg_loss, avg_metrics


@profile
def val_epoch(
    dataloader,
    model: torch.nn.Module,
    loss_calculator: LossCalculator,
    metric_calculator: MetricCalculator,
    logger,
    epoch: int,
    cfg: DictConfig,
    dist_manager: DistributedManager,
    broadcast_global: bool = False,
    val_writer: SummaryWriter | None = None,
    log_jsonl=None,
) -> tuple[float, dict[str, float]]:
    """Run one validation epoch.

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
    epoch : int
        Current epoch index (0-based).
    cfg : DictConfig
        Full Hydra config; uses ``cfg.profile`` and ``cfg.precision``.
    dist_manager : DistributedManager
        Distributed training manager.
    broadcast_global : bool, default False
        Expand global (B,1,C) tensors to match the spatial dimension
        of other batch tensors before forwarding.

    Returns
    -------
    avg_loss : float
        Mean validation loss over all batches.
    avg_metrics : dict[str, float]
        Mean normalised-space metrics.
    """
    model.eval()
    total_loss = 0.0
    total_metrics: dict[str, float] = {}
    precision = getattr(cfg, "precision", "float32")
    n_batches = 0
    num_steps = len(dataloader)
    epoch_t0 = time.perf_counter()

    with torch.no_grad():
        step_t0 = time.perf_counter()
        for i, batch in enumerate(dataloader):
            batch = {k: _recursive_to(v, dist_manager.device) for k, v in batch.items()}

            loss, metrics, _ = forward_pass(
                batch,
                model,
                precision,
                loss_calculator,
                metric_calculator,
                broadcast_global=broadcast_global,
            )

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

    logger.info(
        f"Epoch {epoch} val done in {epoch_dt:.1f}s "
        f"({n_batches} steps, {epoch_dt / max(n_batches, 1):.3f}s/step avg)"
    )

    if dist_manager.rank == 0:
        _log_to_tensorboard(val_writer, avg_metrics, "epoch", epoch)
        if log_jsonl is not None:
            val_log = {"loss": avg_loss}
            val_log.update(avg_metrics)
            log_jsonl({"phase": "val", "epoch": epoch, **val_log})

    return avg_loss, avg_metrics


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
        for k, v in batch.items():
            v_flat = v.float()
            logger.info(
                f"    {k:20s}  "
                f"min={v_flat.min().item(): .6e}  "
                f"mean={v_flat.mean().item(): .6e}  "
                f"std={v_flat.std().item(): .6e}  "
                f"max={v_flat.max().item(): .6e}"
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
    """Build train and val dataloaders from dataset configs.

    Supports two split strategies:

    **Directory-based** (existing): separate ``train_datadir`` and
    ``val_datadir`` in the dataset YAML. Each split gets its own reader
    and dataset.

    **Manifest-based** (new): a single ``datadir`` in the dataset YAML
    with ``train_manifest`` and ``val_manifest`` in the training config's
    ``data.<key>`` block. One reader/dataset covers the full directory;
    ``ManifestSampler`` restricts each loader to the correct subset of
    indices.
    """
    recipe_root = Path(__file__).resolve().parent.parent
    batch_size = cfg.training.get("batch_size", 1)
    sampling_resolution = cfg.dataset.get("sampling_resolution", None)
    augment = cfg.get("augment", False)
    dist_manager = DistributedManager()
    use_distributed = dist_manager.world_size > 1
    collate_fn = build_collate_fn(
        cfg.get("data_mapping", "geotransolver_automotive_surface")
    )

    # DataLoader / MeshDataset performance tuning from cfg.dataloader
    dl_cfg = cfg.get("dataloader", {})
    prefetch_factor = dl_cfg.get("prefetch_factor", 2)
    num_streams = dl_cfg.get("num_streams", 4)
    use_streams = dl_cfg.get("use_streams", False)
    num_workers = dl_cfg.get("num_workers", 1)
    pin_memory = dl_cfg.get("pin_memory", False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sampler_seed = cfg.training.get("seed", 0) or 0

    train_datasets = []
    val_datasets = []
    # When using manifest-based splits, we collect indices per dataset
    # and build samplers instead of separate datasets.
    manifest_train_indices: list[int] | None = None
    manifest_val_indices: list[int] | None = None
    using_manifests = False
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

        # --- Manifest-based splits ---
        # Two config styles are supported:
        #
        # Style A (separate files):
        #   train_manifest: /path/to/train_runs.txt
        #   val_manifest:   /path/to/val_runs.txt
        #
        # Style B (single dict manifest with split keys):
        #   manifest:    /path/to/manifest.json
        #   train_split: single_aoa_4_train
        #   val_split:   single_aoa_4_val
        #
        # Both styles accept an optional ``datadir`` to override
        # ``train_datadir`` in the dataset YAML with the root directory
        # containing all runs.
        #
        # NOTE (limitation): only ONE ``data.<key>`` block may carry a
        # manifest today.  If multiple blocks have manifest/train_split,
        # the later block silently overwrites the earlier block's indices
        # and the resulting ``ManifestSampler`` is indexed against the
        # last reader's local positions rather than the MultiDataset's
        # concatenated positions.  To merge splits via MultiDataset (e.g.
        # train on single_aoa_4 + single_aoa_12 together), this loop must
        # first be extended to collect per-block (offset, indices) pairs
        # and build a single sampler over offset-shifted indices against
        # the MultiDataset.  Tracked as a follow-up.
        train_manifest = ds_cfg_block.get("train_manifest", None)
        val_manifest = ds_cfg_block.get("val_manifest", None)
        manifest = ds_cfg_block.get("manifest", None)
        train_split = ds_cfg_block.get("train_split", None)
        val_split = ds_cfg_block.get("val_split", None)

        # Derive manifest from the dataset's train_datadir when not
        # explicitly provided in the training config.
        if manifest is None and train_split is not None:
            train_datadir = OmegaConf.select(ds_yaml, "train_datadir", default=None)
            if train_datadir:
                derived = Path(str(train_datadir)) / "manifest.json"
                if derived.exists():
                    manifest = str(derived)

        has_manifest = train_manifest is not None or (
            manifest is not None and train_split is not None
        )

        if has_manifest:
            using_manifests = True
            # When using manifests, the reader must see ALL runs under one
            # root. The config block can provide ``datadir`` to override the
            # dataset YAML's ``train_datadir`` with the parent directory that
            # contains every run (train + val).
            datadir = ds_cfg_block.get("datadir", None)
            if datadir:
                ds_yaml = OmegaConf.merge(ds_yaml, {"train_datadir": datadir})
            dataset = build_dataset(
                ds_yaml,
                augment=augment,
                device=device,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            train_datasets.append(dataset)

            # Resolve train indices
            if train_manifest is not None:
                train_entries = load_manifest(train_manifest)
            else:
                train_entries = load_manifest(manifest, split=train_split)
            manifest_train_indices = resolve_manifest_indices(
                dataset.reader, train_entries
            )

            # Resolve val indices
            if val_manifest is not None:
                val_entries = load_manifest(val_manifest)
                manifest_val_indices = resolve_manifest_indices(
                    dataset.reader, val_entries
                )
            elif val_split is not None:
                val_entries = load_manifest(manifest, split=val_split)
                manifest_val_indices = resolve_manifest_indices(
                    dataset.reader, val_entries
                )
            continue

        # --- Directory-based splits (existing path) ---
        train_datasets.append(
            build_dataset(
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
                build_dataset(
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

    if using_manifests:
        # Manifest path: single dataset, split via samplers
        rank = dist_manager.rank if use_distributed else 0
        world_size = dist_manager.world_size if use_distributed else 1

        train_sampler = ManifestSampler(
            manifest_train_indices,
            shuffle=True,
            seed=sampler_seed,
            rank=rank,
            world_size=world_size,
            drop_last=True,
        )
        if manifest_val_indices is not None:
            val_sampler = ManifestSampler(
                manifest_val_indices,
                shuffle=False,
                seed=sampler_seed,
                rank=rank,
                world_size=world_size,
                drop_last=False,
            )
        else:
            val_sampler = train_sampler
        val_dataset = train_dataset
    else:
        # Directory-based path: separate datasets per split
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
                train_dataset,
                shuffle=True,
                drop_last=True,
                seed=sampler_seed,
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                shuffle=False,
                drop_last=False,
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
        drop_last=True,
        prefetch_factor=prefetch_factor,
        num_streams=num_streams,
        use_streams=use_streams,
        seed=sampler_seed,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
        drop_last=False,
        prefetch_factor=prefetch_factor,
        num_streams=num_streams,
        use_streams=use_streams,
        seed=sampler_seed,
    )

    return train_loader, val_loader, normalizer, nondim_transform, first_metadata or {}


@profile
def main(cfg: DictConfig):
    """Run the full training loop, or I/O-only benchmark when ``benchmark_io=true``.

    Orchestrates the complete training workflow:

    1. Initialise distributed training and TensorBoard/JSONL logging.
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
        ``profile``, ``benchmark_io``, ``logging``, and related keys.
    """
    DistributedManager.initialize()
    dist_manager = DistributedManager()
    logger = RankZeroLoggingWrapper(PythonLogger(name="training"), dist_manager)

    seed = cfg.training.get("seed", None)
    set_seed(seed, rank=dist_manager.rank)
    logger.info(f"Random seed: {seed} (rank offset: {dist_manager.rank})")

    checkpoint_dir = getattr(cfg, "checkpoint_dir", None) or cfg.output_dir

    # -- Logging setup (rank 0 only) ----------------------------------------------
    train_writer = None
    val_writer = None
    log_jsonl = None
    run_dir = os.path.join(cfg.output_dir, cfg.run_id)
    if dist_manager.rank == 0:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        train_writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb", "train"))
        val_writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb", "val"))
        metrics_path = os.path.join(run_dir, "metrics.jsonl")

        def log_jsonl(record: dict):
            record["ts"] = datetime.now(timezone.utc).isoformat()
            with open(metrics_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")

    logger.info(f"Config:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")

    train_loader, val_loader, normalizer, _, ds_metadata = build_dataloaders(cfg)
    logger.info(f"Train samples: {len(train_loader.sampler)}")
    logger.info(f"Val samples: {len(val_loader.sampler)}")

    # -- Log dataset metadata (rank 0) --------------------------------------------
    recipe_root = Path(__file__).resolve().parent.parent
    if dist_manager.rank == 0 and log_jsonl is not None:
        log_jsonl(
            {
                "phase": "dataset",
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "metadata": ds_metadata or {},
            }
        )

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
        if dist_manager.rank == 0:
            if train_writer is not None:
                train_writer.close()
            if val_writer is not None:
                val_writer.close()
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

    # -- Log full config + model params (rank 0) ---------------------------------
    if dist_manager.rank == 0:
        flat_cfg = _flatten_config(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
        )
        if log_jsonl is not None:
            log_jsonl(
                {
                    "phase": "config",
                    "model": model.__class__.__name__,
                    "num_parameters": num_params,
                    "params": flat_cfg,
                }
            )

        # Save the full resolved config
        resolved_yaml = omegaconf.OmegaConf.to_yaml(cfg, resolve=True)
        config_artifact_path = os.path.join(run_dir, "resolved_config.yaml")
        with open(config_artifact_path, "w") as f:
            f.write(resolved_yaml)

    ds_cfg = cfg.dataset
    targets = omegaconf.OmegaConf.to_container(ds_cfg.targets, resolve=True)
    metrics_list = omegaconf.OmegaConf.to_container(
        ds_cfg.get("metrics", ["l1", "l2", "mae"]), resolve=True
    )
    metric_calculator = MetricCalculator(
        target_config=targets,
        metrics=metrics_list,
    )
    loss_calculator = LossCalculator(
        target_config=targets,
        loss_type=cfg.training.get("loss_type", "huber"),
    )
    broadcast_global = cfg.get("broadcast_global", False)
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
                epoch,
                cfg,
                dist_manager,
                scaler,
                broadcast_global=broadcast_global,
                train_writer=train_writer,
                log_jsonl=log_jsonl,
            )

            val_loss, val_metrics = val_epoch(
                val_loader,
                model,
                loss_calculator,
                metric_calculator,
                logger,
                epoch,
                cfg,
                dist_manager,
                broadcast_global=broadcast_global,
                val_writer=val_writer,
                log_jsonl=log_jsonl,
            )

            if dist_manager.rank == 0:
                all_keys = list(dict.fromkeys(list(train_metrics) + list(val_metrics)))

                rows = [
                    [
                        k,
                        f"{train_metrics.get(k, float('nan')):.6f}",
                        f"{val_metrics.get(k, float('nan')):.6f}",
                    ]
                    for k in all_keys
                ]

                table = tabulate(
                    rows, headers=["Metric", "Train", "Val"], tablefmt="pretty"
                )
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
        if train_writer is not None:
            train_writer.close()
        if val_writer is not None:
            val_writer.close()

    logger.info("Training completed!")


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="train_geotransolver_automotive_surface",
)
def launch(cfg: DictConfig):
    """Hydra entry point: configure profiling and delegate to :func:`main`.

    Parameters
    ----------
    cfg : DictConfig
        Hydra-composed config (override with ``--config-name``).
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
