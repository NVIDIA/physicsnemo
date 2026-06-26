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
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import nullcontext
from typing import Any, Literal, cast

import hydra
import torch
import torch.distributed as dist
from datasets import build_dataloaders
from loss import LossCalculator
from metrics import MetricCalculator, resolve_metrics
from omegaconf import DictConfig, OmegaConf
from output_normalize import IOType, normalize_output_to_tensordict, require_output_type
from tabulate import tabulate
from tensordict import TensorDict
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from utils import (
    FieldType,
    Phase,
    Precision,
    build_muon_optimizer,
    get_autocast_context,
    make_jsonl_logger,
    recursive_to_device,
    resolve_dict,
    set_seed,
)

from physicsnemo import datapipes  # noqa: F401 - registers ${dp:...} resolver
from physicsnemo.datapipes import DataLoader
from physicsnemo.distributed import DistributedManager
from physicsnemo.mesh import MESH_FIELD_ASSOCIATIONS, DomainMesh, Mesh
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.profiling import Profiler, profile

### When `cfg.profile` is set, every train / val epoch breaks out of its
### batch loop after this many steps. Keeps profiling traces short enough
### to be useful without changing the rest of the training contract.
_PROFILE_MAX_STEPS = 10


### ---------------------------------------------------------------------------
### Config
### ---------------------------------------------------------------------------


def _flatten_config(
    d: dict[str, Any], parent: str = "", sep: str = "."
) -> dict[str, str]:
    """Recursively flatten a nested dict into dot-separated key/value pairs."""
    items: dict[str, str] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            items.update(_flatten_config(v, key, sep))
        else:
            items[key] = str(v)
    return items


### ---------------------------------------------------------------------------
### Aggregation
### ---------------------------------------------------------------------------


def _reduce_and_average(
    loss_sum: float,
    losses_td: TensorDict | None,
    metrics_td: TensorDict | None,
    n_samples: int,
    *,
    device: torch.device | str,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Collapse rank-local loss/metric *sums* + a sample count into global means.

    Under DDP each rank only sees its own shard, so the numbers we log are
    meaningless until reduced across ranks. This takes a rank-local *sum*
    (``loss_sum`` plus the 0-D ``losses_td`` / ``metrics_td`` leaves) and the
    matching sample count, reduces them across ranks once, and divides by the
    *global* count to produce sample-weighted means. It is granularity-neutral
    and called at two boundaries:

    - Per step, with ``n_samples == 1`` (one sample per batch), so the logged
      iteration curves are global all-rank means rather than rank-0's shard.
    - Per epoch, with ``n_samples == n_local`` and the running epoch sums, so
      the summary is a global mean over the whole dataset.

    Reducing sums and a count (rather than per-rank means) is what makes the
    result correct for uneven shards: ``global_sum / global_count`` weights
    every sample equally no matter how the dataset split across ranks. The
    values are packed into a single ``float32`` buffer, so each call costs one
    ``all_reduce`` and one device-to-host sync (the ``.tolist()``). It mirrors
    the inference-side reducer ``infer._allreduce_sums``.

    Args:
        loss_sum: Rank-local sum of scalar losses over the ``n_samples`` being
            collapsed -- one step's ``loss.item()`` per step, or the epoch
            running sum -- not a mean.
        losses_td: Rank-local per-field loss sum: a 0-D (``batch_size=[]``)
            ``TensorDict`` whose leaves are summed scalar losses, one per loss
            term. ``None`` is the "zero samples" sentinel (see Notes); it does
            not arise on the per-step path, where a batch is always present.
        metrics_td: The matching per-field metric-sum accumulator, with the
            same ``None`` sentinel. Seeded in lock-step with ``losses_td``, so
            the two are ``None`` together or populated together.
        n_samples: Number of samples this rank contributed to ``loss_sum`` and
            the accumulators (``1`` per step, ``n_local`` per epoch; equal to
            the step count because the recipe runs ``batch_size == 1``).
        device: Device on which to build the reduction buffer. Must be the
            rank's collective/compute device (``dist_manager.device``) so the
            NCCL ``all_reduce`` runs on the correct device.

    Returns:
        A ``(avg_loss, avg_losses, avg_metrics)`` tuple where ``avg_loss`` is
        the global mean loss, ``avg_losses`` is ``{loss_name: global_mean}``,
        and ``avg_metrics`` is ``{metric_name: global_mean}``. The dict keys
        and their order are taken from ``losses_td`` / ``metrics_td``. On the
        ``None`` sentinel it returns ``(loss_sum / max(n_samples, 1), {}, {})``
        without entering the collective.

    Notes:
        Single-process (or ``world_size == 1``) skips the reduction, so the
        result is identical to plain ``sum / n_samples`` averaging and
        single-GPU logs are unchanged.

        Calling this per step adds one collective per iteration, which is only
        deadlock-free because every rank issues the same number of collectives
        -- i.e. every rank runs the same step count. The recipe's samplers
        guarantee that: train uses ``drop_last=True`` and val pads to even
        shards, so no rank finishes early and skips a step's ``all_reduce``.

        The one fused ``all_reduce`` is valid only because every rank packs
        the same leaves in the same order, which holds since all ranks share
        one ``target_config`` (identical loss/metric keys). The ``None`` early
        return similarly assumes ranks are seeded together: under DDP every
        rank gets at least one sample, so the accumulators are non-``None`` on
        all ranks at once.
    """
    if losses_td is None or metrics_td is None:
        return loss_sum / max(n_samples, 1), {}, {}
    loss_keys = cast(list[str], list(losses_td.keys()))
    metric_keys = cast(list[str], list(metrics_td.keys()))
    leaves = cast(
        list[torch.Tensor], list(losses_td.values()) + list(metrics_td.values())
    )
    ### [loss_sum, n_samples, *loss_sums, *metric_sums] -> one collective.
    packed = torch.cat(
        [
            torch.tensor([loss_sum, float(n_samples)], device=device),
            torch.stack(leaves).float().to(device),
        ]
    )
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(packed)
    reduced_loss, reduced_n, *leaf_sums = packed.tolist()
    n = max(reduced_n, 1.0)
    n_loss = len(loss_keys)
    averaged = [v / n for v in leaf_sums]
    return (
        reduced_loss / n,
        dict(zip(loss_keys, averaged[:n_loss])),
        dict(zip(metric_keys, averaged[n_loss:])),
    )


### ---------------------------------------------------------------------------
### Logging
### ---------------------------------------------------------------------------


def _log_to_tensorboard(
    writer: SummaryWriter | None,
    values: Mapping[str, float | torch.Tensor],
    tag_prefix: str,
    global_step: int,
) -> None:
    """Write a flat ``{name: scalar}`` mapping to TensorBoard under ``tag_prefix/<name>``.

    No-op when *writer* is ``None``. The caller chooses *tag_prefix* to
    namespace the entries (e.g. ``"epoch"`` vs ``"iteration/metrics"``).
    """
    if writer is None:
        return
    for k, v in values.items():
        writer.add_scalar(f"{tag_prefix}/{k}", v, global_step=global_step)


### ---------------------------------------------------------------------------
### Forward pass
### ---------------------------------------------------------------------------


def forward_pass(
    batch: dict[str, Any],
    model: torch.nn.Module,
    precision: Precision,
    loss_calculator: LossCalculator,
    metric_calculator: MetricCalculator,
    *,
    output_type: IOType,
    target_config: dict[str, FieldType],
) -> tuple[torch.Tensor, TensorDict, TensorDict]:
    """Run a forward pass + loss + metrics on one collated batch.

    Args:
        batch: ``{"forward_kwargs": ..., "targets": TensorDict}`` produced
            by the collate function. ``"targets"`` is a TensorDict with
            batch_size ``[N]`` (mesh-input mode) or ``[1, N]``
            (tensor-input mode).
        model: Model whose ``forward`` accepts the resolved
            ``forward_kwargs`` as keyword arguments.
        precision: One of ``"float32"``, ``"float16"``, or ``"bfloat16"``.
            Wraps the forward call in the matching ``torch.autocast``
            context; inputs keep their native dtype.
        loss_calculator: Returns ``(loss, loss_td)`` from
            ``(pred, target)`` TensorDicts.
        metric_calculator: Returns a per-field metrics ``TensorDict``.
        output_type: ``"mesh"`` or ``"tensors"``; controls how the model
            output is unpacked into a TensorDict.
        target_config: ``{name: "scalar"|"vector"}``; used to split tensor
            outputs and validate Mesh outputs.

    Returns:
        ``(loss, loss_td, metric_td)``. The two TensorDicts are kept
        separate so callers can route them to different log namespaces
        without textual key inspection. Per-field values are returned
        as **detached, on-device 0-D tensors** (no ``.item()`` sync
        here): the caller decides when to sync, so the loss kernels can
        overlap with backward instead of being serialised by an
        in-line D2H transfer.
    """
    forward_kwargs = batch["forward_kwargs"]
    targets: TensorDict = batch["targets"]

    ### Inputs keep their native dtype; autocast handles model-internal precision.
    with get_autocast_context(precision):
        output = model(**forward_kwargs)

    pred_td = normalize_output_to_tensordict(output, target_config, output_type)

    ### Loss runs in float32 to avoid bf16 precision loss in the reduction.
    pred_f32 = pred_td.float()
    target_f32 = targets.float()

    loss, loss_td = loss_calculator(pred_f32, target_f32)
    with torch.no_grad():
        metric_td = metric_calculator(pred_f32, target_f32)
    ### Detach (don't sync) the per-field TDs so the caller controls when
    ### a D2H copy happens; running ``.item()`` here would serialise the
    ### forward kernels against the host. ``TensorDict.detach()`` walks
    ### every leaf in one fast-apply pass.
    return loss, loss_td.detach(), metric_td.detach()


### ---------------------------------------------------------------------------
### Epoch loops
### ---------------------------------------------------------------------------


def _run_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_calculator: LossCalculator,
    metric_calculator: MetricCalculator,
    logger: Any,
    epoch: int,
    cfg: DictConfig,
    dist_manager: DistributedManager,
    *,
    mode: Literal["train", "val"],
    output_type: IOType,
    target_config: dict[str, FieldType],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: GradScaler | None = None,
    writer: SummaryWriter | None = None,
    log_jsonl: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[float, dict[str, float]]:
    """Run one training-or-validation epoch.

    Train and val share the same per-batch loop (``forward_pass`` +
    metric accumulation + per-step console log + per-epoch summary).
    Train mode additionally runs the backward / optimizer / scheduler
    step and emits per-step TensorBoard + JSONL entries (``phase: "step"``);
    val mode wraps the loop in ``torch.no_grad()``, skips TensorBoard
    per-step logging, and emits a lighter-weight JSONL record per step
    (``phase: "val_step"``) carrying ``epoch``, ``val_step``, ``loss``
    and ``step_time_s``.

    Args:
        mode: ``"train"`` or ``"val"``. ``"train"`` requires *optimizer*
            and *scheduler*; ``"val"`` ignores them.
        scaler: GradScaler for fp16 (train mode only).
        writer: TensorBoard writer for the matching split. Per-epoch
            metrics are written to it on rank 0; per-step metrics are
            written only in train mode.
        log_jsonl: Optional ``record -> None`` callback for JSONL logs.
            See ``forward_pass`` and ``main`` docstrings for the rest of
            the parameters.
    """
    is_train = mode == "train"
    if is_train and (optimizer is None or scheduler is None):
        raise ValueError("train mode requires both optimizer and scheduler")
    if is_train:
        model.train()
    else:
        model.eval()

    grad_ctx = nullcontext() if is_train else torch.no_grad()
    log_prefix = "Epoch" if is_train else "Val Epoch"
    is_rank0 = dist_manager.rank == 0

    ### `total_loss` is a Python float fed by the per-step print line's
    ### sync; `total_losses_td` / `total_metrics_td` are on-device
    ### TensorDict accumulators (one 0-D leaf per field) that defer
    ### their D2H transfer to the single batched ``.tolist()`` at
    ### end-of-epoch. ``None`` here means "not yet seeded"; the first
    ### iteration clones the per-step TensorDict to break aliasing.
    ### ``n_local`` below is this rank's step/sample count. The averaging
    ### denominator is the GLOBAL count that ``_reduce_and_average``
    ### all-reduces from each rank's ``n_local`` at end-of-epoch; the local
    ### value is reused directly only for the per-rank step-rate line.
    total_loss = 0.0
    total_losses_td: TensorDict | None = None
    total_metrics_td: TensorDict | None = None
    precision = getattr(cfg, "precision", "float32")
    n_local = 0
    num_steps = len(dataloader)
    epoch_t0 = time.perf_counter()

    with grad_ctx:
        step_t0 = time.perf_counter()
        for i, batch in enumerate(dataloader):
            batch = recursive_to_device(batch, dist_manager.device)

            loss, losses, metrics = forward_pass(
                batch,
                model,
                precision,
                loss_calculator,
                metric_calculator,
                output_type=output_type,
                target_config=target_config,
            )

            if is_train:
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

            ### Accumulate on-device with no sync. First iteration clones
            ### so subsequent in-place ``add_`` calls don't alias the
            ### per-step TDs; both accumulators are seeded in lock-step
            ### (the joint ``is None`` check exists to satisfy the type
            ### checker, which can't see the invariant from per-variable
            ### narrowing).
            if total_losses_td is None or total_metrics_td is None:
                total_losses_td = losses.clone()
                total_metrics_td = metrics.clone()
            else:
                total_losses_td.add_(losses)
                total_metrics_td.add_(metrics)
            n_local += 1

            ### Per-step sync for the print line; lands after backward +
            ### optimizer.step so it overlaps with queued GPU work.
            this_loss = loss.detach().item()
            total_loss += this_loss

            step_dt = time.perf_counter() - step_t0
            mem_gb = (
                torch.cuda.memory_reserved() / 1024**3
                if torch.cuda.is_available()
                else 0
            )

            ### Reduce this step's loss + metrics across ranks so the iteration
            ### logs are global all-rank means, not rank-0's shard. This is a
            ### collective: EVERY rank must call it, so it sits outside the
            ### rank-0 logging gate below. ``n_samples=1`` is this step's local
            ### sample count (one batch == one sample), matching the
            ### ``n_local += 1`` accumulation above. Equal per-rank step counts
            ### keep the per-step collective deadlock-free (see
            ### ``_reduce_and_average``).
            step_loss, step_losses, step_metrics = _reduce_and_average(
                this_loss, losses, metrics, 1, device=dist_manager.device
            )

            ### Train mode includes Mem in the per-step line; val drops it
            ### because the no_grad path is the lowest-noise place to look.
            mem_str = f" Mem: {mem_gb:.2f}GB" if is_train else ""
            logger.info(
                f"{log_prefix} {epoch} [{i + 1}/{num_steps}] "
                f"Loss: {step_loss:.6f} "
                f"Step: {step_dt:.3f}s"
                f"{mem_str}"
            )

            ### Per-step TensorBoard: train only (val_writer is intentionally
            ### epoch-only to keep dashboards uncluttered). Per-step JSONL is
            ### emitted in both modes so downstream tooling can compute val
            ### step-time statistics directly instead of inferring them from
            ### ``val_ts - train_ts``. The logged loss / metrics are the global
            ### all-rank means from ``_reduce_and_average`` above (not rank-0's
            ### shard); rank 0 is only the writer.
            if is_rank0:
                if is_train:
                    global_step = epoch * num_steps + i
                    if writer is not None:
                        ### Loss keys already start with `loss/`, so the iteration
                        ### prefix yields tags like `iteration/loss/pressure`;
                        ### metric tags get an explicit `iteration/metrics/...`
                        ### namespace so we never have to split by string prefix.
                        _log_to_tensorboard(
                            writer, step_losses, "iteration", global_step
                        )
                        _log_to_tensorboard(
                            writer, step_metrics, "iteration/metrics", global_step
                        )
                        writer.add_scalar(
                            "iteration/lr",
                            scheduler.get_last_lr()[0],
                            global_step=global_step,
                        )
                        writer.add_scalar(
                            "iteration/performance/mem_gb",
                            mem_gb,
                            global_step=global_step,
                        )
                        writer.add_scalar(
                            "iteration/performance/step_time_s",
                            step_dt,
                            global_step=global_step,
                        )
                    if log_jsonl is not None:
                        log_jsonl(
                            {
                                "phase": "train_step",
                                "global_step": global_step,
                                "loss": step_loss,
                                "mem_gb": mem_gb,
                                "step_time_s": step_dt,
                                **step_losses,
                                **step_metrics,
                            }
                        )
                elif log_jsonl is not None:
                    ### Val per-step record. ``epoch`` is explicit (unlike
                    ### the train ``step`` records, which the parser infers
                    ### from surrounding ``train`` markers) so val_step
                    ### records can be associated with an epoch without
                    ### relying on surrounding context. ``mem_gb`` is
                    ### intentionally omitted -- the no_grad path is the
                    ### lowest-noise place to measure step time and we
                    ### don't want allocator state hopping in TB.
                    log_jsonl(
                        {
                            "phase": "val_step",
                            "epoch": epoch,
                            "val_step": i,
                            "loss": step_loss,
                            "step_time_s": step_dt,
                            **step_losses,
                            **step_metrics,
                        }
                    )

            if cfg.profile and i >= _PROFILE_MAX_STEPS:
                break
            step_t0 = time.perf_counter()

    epoch_dt = time.perf_counter() - epoch_t0
    n = max(n_local, 1)
    ### Reduce the epoch sums + sample count across ranks once, so logged
    ### loss/metrics are the GLOBAL averages (not rank-0's shard) under
    ### DDP. `n` above is kept local for the per-rank step-rate line below.
    avg_loss, avg_losses, avg_metrics = _reduce_and_average(
        total_loss,
        total_losses_td,
        total_metrics_td,
        n_local,
        device=dist_manager.device,
    )

    logger.info(
        f"Epoch {epoch} {mode} done in {epoch_dt:.1f}s "
        f"({n_local} steps, {epoch_dt / n:.3f}s/step avg)"
    )

    if is_rank0:
        _log_to_tensorboard(writer, avg_losses, "epoch", epoch)
        _log_to_tensorboard(writer, avg_metrics, "epoch/metrics", epoch)
        if log_jsonl is not None:
            summary_phase: Phase = "train_summary" if is_train else "val_summary"
            log_jsonl(
                {
                    "phase": summary_phase,
                    "epoch": epoch,
                    "loss": avg_loss,
                    **avg_losses,
                    **avg_metrics,
                }
            )

    return avg_loss, {**avg_losses, **avg_metrics}


@profile
def train_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_calculator: LossCalculator,
    metric_calculator: MetricCalculator,
    logger: Any,
    epoch: int,
    cfg: DictConfig,
    dist_manager: DistributedManager,
    scaler: GradScaler | None = None,
    *,
    output_type: IOType,
    target_config: dict[str, FieldType],
    train_writer: SummaryWriter | None = None,
    log_jsonl: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[float, dict[str, float]]:
    """Run one training epoch (delegates to :func:`_run_epoch` in train mode)."""
    return _run_epoch(
        dataloader,
        model,
        loss_calculator,
        metric_calculator,
        logger,
        epoch,
        cfg,
        dist_manager,
        mode="train",
        output_type=output_type,
        target_config=target_config,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        writer=train_writer,
        log_jsonl=log_jsonl,
    )


@profile
def val_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_calculator: LossCalculator,
    metric_calculator: MetricCalculator,
    logger: Any,
    epoch: int,
    cfg: DictConfig,
    dist_manager: DistributedManager,
    *,
    output_type: IOType,
    target_config: dict[str, FieldType],
    val_writer: SummaryWriter | None = None,
    log_jsonl: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[float, dict[str, float]]:
    """Run one validation epoch (delegates to :func:`_run_epoch` in val mode)."""
    return _run_epoch(
        dataloader,
        model,
        loss_calculator,
        metric_calculator,
        logger,
        epoch,
        cfg,
        dist_manager,
        mode="val",
        output_type=output_type,
        target_config=target_config,
        writer=val_writer,
        log_jsonl=log_jsonl,
    )


### ---------------------------------------------------------------------------
### I/O benchmarking
### ---------------------------------------------------------------------------


def _walk_batch_for_logging(
    value: Any, prefix: str = ""
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(dotted_name, Tensor)`` pairs from a batch (nested dicts / TensorDicts of tensors / Mesh).

    The TensorDict branch delegates the recursion to ``TD.flatten_keys('.')``
    rather than driving it from Python via ``.items()`` -- a TD's own
    flattening produces dotted leaf paths in one call. The plain ``dict``
    branch keeps the manual visitor because dicts may contain mixed
    Tensor / Mesh / nested-dict values that need the full recursion.
    """
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, TensorDict):
        for key, leaf in value.flatten_keys(".").items():
            sub = f"{prefix}.{key}" if prefix else key
            yield sub, leaf
    elif isinstance(value, dict):
        for k, v in value.items():
            sub = f"{prefix}.{k}" if prefix else str(k)
            yield from _walk_batch_for_logging(v, sub)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            sub = f"{prefix}[{i}]" if prefix else f"[{i}]"
            yield from _walk_batch_for_logging(v, sub)
    elif isinstance(value, DomainMesh):
        ### Recurse into interior, boundaries, and domain-level global_data
        ### so I/O benchmarks see every leaf the model would actually
        ### consume (point_data targets, boundary cell_data inputs, etc).
        yield from _walk_batch_for_logging(value.interior, f"{prefix}.interior")
        for bname in value.boundary_names:
            yield from _walk_batch_for_logging(
                value.boundaries[bname], f"{prefix}.boundaries.{bname}"
            )
        if value.global_data.keys():
            yield from _walk_batch_for_logging(
                value.global_data, f"{prefix}.global_data"
            )
    elif isinstance(value, Mesh):
        ### Mesh-level inputs: emit geometry tensors + every per-element /
        ### per-vertex / per-sample field. Each *_data attribute is itself
        ### a TensorDict, so the TD branch above handles dotted leaf paths.
        yield (f"{prefix}.points", value.points)
        if value.n_cells > 0:
            yield (f"{prefix}.cells", value.cells)
        for section in MESH_FIELD_ASSOCIATIONS:
            td = getattr(value, section)
            if td.keys():
                yield from _walk_batch_for_logging(td, f"{prefix}.{section}")


@profile
def benchmark_io_epoch(
    dataloader: DataLoader,
    label: str,
    logger: Any,
    max_steps: int | None = None,
) -> None:
    """Iterate a dataloader without any model logic and report I/O timing.

    Args:
        dataloader: Dataloader to benchmark.
        label: Human-readable label for logging (e.g. ``"train"`` or
            ``"val"``).
        logger: Logger for console output.
        max_steps: Stop after this many batches. ``None`` means exhaust
            the loader.
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

        named_tensors = list(_walk_batch_for_logging(batch))
        shapes = "  ".join(f"{name}:{tuple(t.shape)}" for name, t in named_tensors)
        logger.info(
            f"  [{label}] [{i + 1}/{num_steps}] "
            f"dt={dt:.4f}s  Mem={mem_gb:.2f}GB  {shapes}"
        )
        for name, t in named_tensors:
            v_flat = t.float() if t.is_floating_point() else t.to(torch.float32)
            logger.info(
                f"    {name:30s}  "
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


### ---------------------------------------------------------------------------
### Driver
### ---------------------------------------------------------------------------


@profile
def main(cfg: DictConfig) -> None:
    """Run the full training loop, or I/O-only benchmark when ``benchmark_io=true``.

    Orchestrates the complete training workflow:

    1. Initialise distributed training and TensorBoard/JSONL logging.
    2. Build train/val dataloaders and extract pipeline transforms.
    3. If ``cfg.benchmark_io`` is true, iterate dataloaders to measure
       I/O throughput and return early (no model, no optimizer).
    4. Otherwise, instantiate the model, optimizer, and run the normal
       train/val epoch loop with checkpointing.

    Args:
        cfg: Hydra config containing ``model``, ``training``, ``dataset``,
            ``data``, ``output_dir``, ``run_id``, ``precision``,
            ``compile``, ``profile``, ``benchmark_io``, ``logging``, and
            related keys.
    """

    DistributedManager.initialize()
    dist_manager = DistributedManager()
    device = dist_manager.device
    is_rank0 = dist_manager.rank == 0
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
    if is_rank0:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        train_writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb", "train"))
        val_writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb", "val"))
        log_jsonl = make_jsonl_logger(os.path.join(run_dir, "metrics.jsonl"))

    train_loader, val_loader, normalizer, dataset_info = build_dataloaders(cfg)
    target_config: dict[str, FieldType] = dataset_info["targets"]
    ### `metrics_list` is derived later from cfg.metrics (recipe-side);
    ### build_dataloaders no longer ships a "metrics" key in dataset_info.

    ### Log the resolved config AFTER build_dataloaders() because that's
    ### where `cfg.out_dim` is auto-derived from the chosen dataset's
    ### `targets:` block; resolving earlier would fail on the model
    ### template's `out_dim: ${out_dim}` interpolation.
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    logger.info(f"Train samples: {len(train_loader.sampler)}")
    logger.info(f"Val samples: {len(val_loader.sampler)}")
    logger.info(f"Targets (from dataset YAML): {target_config}")

    # -- Log dataset metadata (rank 0) --------------------------------------------
    if is_rank0 and log_jsonl is not None:
        ### Use len(sampler) so manifest mode (where train and val share
        ### one underlying dataset) reports the actual per-split count,
        ### not the always-identical len(dataset). PyTorch always assigns
        ### a sampler (a default SequentialSampler when none is passed),
        ### so len(loader.sampler) is always defined.
        log_jsonl(
            {
                "phase": "dataset",
                "train_samples": len(train_loader.sampler),
                "val_samples": len(val_loader.sampler),
                "dataset_size": len(train_loader.dataset),
                "targets": target_config,
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
        if is_rank0:
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

    model.to(device)

    if dist_manager.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_manager.local_rank],
            output_device=device,
        )

    if normalizer is not None:
        norm_summary = ", ".join(
            f"{k}({v['type']})" for k, v in normalizer.stats.items()
        )
        logger.info(f"Normalization: {norm_summary}")

    optimizer = build_muon_optimizer(model, cfg, compile_optimizer=cfg.compile)
    logger.info(f"Optimizer: {optimizer}")
    scheduler = hydra.utils.instantiate(cfg.training.scheduler, optimizer=optimizer)

    precision = cfg.precision
    scaler = GradScaler() if precision == "float16" else None

    # -- Log full config + model params (rank 0) ---------------------------------
    if is_rank0:
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
        resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
        config_artifact_path = os.path.join(run_dir, "resolved_config.yaml")
        with open(config_artifact_path, "w") as f:
            f.write(resolved_yaml)

    ### `target_config` was loaded from the chosen dataset YAML's
    ### `targets:` block by `build_dataloaders`.  The metrics list is
    ### now recipe-side: train.yaml's `cfg.metrics` is the canonical
    ### source. Falls back to the recipe's default set when unset.
    metrics_list = resolve_metrics(cfg)

    field_weights = resolve_dict(cfg, "training.field_weights")

    metric_calculator = MetricCalculator(
        target_config=target_config,
        metrics=metrics_list,
    )
    loss_calculator = LossCalculator(
        target_config=target_config,
        loss_type=cfg.training.get("loss_type", "huber"),
        field_weights=field_weights,
    )
    output_type = require_output_type(cfg)
    logger.info(f"Loss: {loss_calculator}")
    logger.info(f"Metrics: {metric_calculator}")
    logger.info(
        f"Model contract: input_type={cfg.input_type}, output_type={output_type}"
    )

    ckpt_args = {
        "path": os.path.join(checkpoint_dir, cfg.run_id, "checkpoints"),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }
    loaded_epoch = load_checkpoint(device=device, **ckpt_args)

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
                output_type=output_type,
                target_config=target_config,
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
                output_type=output_type,
                target_config=target_config,
                val_writer=val_writer,
                log_jsonl=log_jsonl,
            )

            if is_rank0:
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

            if epoch % cfg.training.save_interval == 0 and is_rank0:
                save_checkpoint(**ckpt_args, epoch=epoch + 1)
                if normalizer is not None:
                    norm_path = os.path.join(ckpt_args["path"], "norm_stats.pt")
                    torch.save(normalizer.stats, norm_path)

            if cfg.training.get("scheduler_update_mode", "epoch") == "epoch":
                scheduler.step()

    if is_rank0:
        if train_writer is not None:
            train_writer.close()
        if val_writer is not None:
            val_writer.close()

    logger.info("Training completed!")


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="train",
)
def launch(cfg: DictConfig) -> None:
    """Hydra entry point: configure profiling and delegate to :func:`main`.

    Args:
        cfg: Hydra-composed config (override with ``--config-name``).
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
