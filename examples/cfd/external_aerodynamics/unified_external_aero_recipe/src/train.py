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

import json
import logging
import os
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import hydra
import omegaconf
import torch
from collate import build_collate_fn
from datasets import (
    ManifestSampler,
    build_dataset,
    find_normalizer,
    load_dataset_config,
    load_manifest,
    resolve_manifest_indices,
    resolve_manifest_spec,
    validate_dataset_consistency,
)
from loss import LossCalculator
from metrics import DEFAULT_METRICS, MetricCalculator, MetricName
from omegaconf import DictConfig, OmegaConf
from output_normalize import IOType, normalize_output_to_tensordict
from tabulate import tabulate
from tensordict import TensorDict
from torch.amp import GradScaler, autocast
from torch.utils.data import Sampler
from torch.utils.tensorboard import SummaryWriter
from utils import FieldType, build_muon_optimizer, set_seed

from physicsnemo import datapipes  # noqa: F401 - registers ${dp:...} resolver
from physicsnemo.core.version_check import OptionalImport
from physicsnemo.datapipes import DataLoader, MeshDataset, MultiDataset
from physicsnemo.distributed import DistributedManager
from physicsnemo.mesh import MESH_FIELD_ASSOCIATIONS, DomainMesh, Mesh
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.profiling import Profiler, profile

te = OptionalImport("transformer_engine.pytorch")
te_recipe = OptionalImport("transformer_engine.common.recipe")
TE_AVAILABLE = te.available

_LOGGER = logging.getLogger("training.build_dataloaders")

### When `cfg.profile` is set, every train / val epoch breaks out of its
### batch loop after this many steps. Keeps profiling traces short enough
### to be useful without changing the rest of the training contract.
_PROFILE_MAX_STEPS = 10

### Allowed mixed-precision modes for the autocast context + input pre-cast.
### Validated only structurally (via the type), not at runtime: an unknown
### value falls through to a no-op autocast in `get_autocast_context`.
Precision: TypeAlias = Literal["float32", "float16", "bfloat16", "float8"]


def _resolve_dict(cfg: DictConfig, path: str) -> dict[str, Any] | None:
    """Resolve `cfg.<path>` to a plain dict, or ``None`` if missing/empty.

    Wraps the OmegaConf incantation
    ``OmegaConf.to_container(OmegaConf.select(cfg, path, default=...), resolve=True) or None``
    that would otherwise repeat at every read site.
    """
    selected = OmegaConf.select(cfg, path, default=OmegaConf.create({}))
    container = OmegaConf.to_container(selected, resolve=True)
    return container or None


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


def _to_float_dicts(
    losses_td: TensorDict | None,
    metrics_td: TensorDict | None,
    *,
    n: int = 1,
) -> tuple[dict[str, float], dict[str, float]]:
    """Stack both TDs' 0-D leaves, divide by *n*, and ``.tolist()`` in one D2H sync.

    Used at both per-step (``n=1``) and per-epoch (``n=batch_count``)
    boundaries: collapses ``2 * n_fields`` ``.item()`` calls into a single
    ``.tolist()`` over a stacked 1-D tensor. Either TD being ``None``
    (the "not yet seeded" sentinel for zero-step epochs) returns
    ``({}, {})``.
    """
    if losses_td is None or metrics_td is None:
        return {}, {}
    ### Bridge TensorDict's wider key/value types to the runtime contract
    ### this recipe enforces: every loss / metric leaf is a 0-D scalar
    ### Tensor keyed by str.
    loss_keys = cast(list[str], list(losses_td.keys()))
    metric_keys = cast(list[str], list(metrics_td.keys()))
    loss_tensors = cast(list[torch.Tensor], list(losses_td.values()))
    metric_tensors = cast(list[torch.Tensor], list(metrics_td.values()))
    flat = (torch.stack(loss_tensors + metric_tensors) / n).tolist()
    n_loss = len(loss_keys)
    return (
        dict(zip(loss_keys, flat[:n_loss])),
        dict(zip(metric_keys, flat[n_loss:])),
    )


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


def get_autocast_context(precision: Precision):
    """Return an autocast context manager for the given precision.

    Args:
        precision: One of ``"float16"``, ``"bfloat16"``, ``"float8"``, or
            ``"float32"``. For ``"float8"``, Transformer Engine must be
            available.

    Returns:
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


### Callable types for the recursive walker. ``LeafFn`` is the per-Tensor
### transform (mandatory); ``ContainerFn`` is the optional override
### applied to tensor-aware containers (Mesh / DomainMesh / TensorDict)
### when the default ``container.apply(leaf_fn)`` semantics aren't enough.
LeafFn = Callable[[torch.Tensor], torch.Tensor]
ContainerFn = Callable[[Any], Any]


def _recursive_apply(
    obj: Any,
    leaf_fn: LeafFn,
    *,
    container_fn: ContainerFn | None = None,
) -> Any:
    """Walk a nested structure, applying ``leaf_fn`` to every Tensor leaf.

    Tensor-aware containers (Mesh, DomainMesh, TensorDict) are routed
    through ``container_fn``. By default, ``container_fn`` delegates to
    ``container.apply(leaf_fn)``, which walks every tensor leaf in
    lock-step but does NOT touch container-level metadata
    (``TensorDict.device`` in particular stays at whatever it was).
    Override ``container_fn`` (e.g. ``lambda c: c.to(device)``) when the
    metadata change matters -- ``TensorDict`` treats ``device is None``
    as "leaves may be on any device", so device moves must go through
    ``.to(device)`` to be observable on the container.

    Plain dicts / lists / tuples are walked recursively. Note that
    ``TensorDict`` is matched in the container branch above, so it does
    NOT fall into the ``isinstance(obj, dict)`` branch (it isn't a
    ``dict`` subclass, but the explicit container check is what makes
    this work). Anything else passes through unchanged.
    """
    if container_fn is None:
        container_fn = lambda c: c.apply(leaf_fn)  # noqa: E731
    if isinstance(obj, torch.Tensor):
        return leaf_fn(obj)
    if isinstance(obj, (Mesh, DomainMesh, TensorDict)):
        return container_fn(obj)
    if isinstance(obj, dict):
        return {
            k: _recursive_apply(v, leaf_fn, container_fn=container_fn)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return type(obj)(
            _recursive_apply(v, leaf_fn, container_fn=container_fn) for v in obj
        )
    return obj


def _recursive_to_device(obj: Any, device: torch.device | str) -> Any:
    """Move every tensor / Mesh / DomainMesh / TensorDict in a nested value to *device*.

    Containers go through ``.to(device)`` (not ``.apply(...)``) so that
    ``TensorDict.device`` is updated alongside the leaves; otherwise a
    later consumer reading ``td.device`` would still see ``None`` even
    though the underlying tensors have already moved.
    """
    return _recursive_apply(
        obj,
        lambda t: t.to(device),
        container_fn=lambda c: c.to(device),
    )


def _recursive_cast_floats(obj: Any, dtype: torch.dtype) -> Any:
    """Cast floating-point tensors in a nested value to *dtype*; skip everything else.

    Non-float tensors (e.g. ``Mesh.cells`` in int64) pass through
    unchanged via the same ``is_floating_point()`` guard at every level.
    Tensor-aware containers use the default ``.apply(leaf_fn)`` path so
    integer leaves stay integer (a plain ``container.to(dtype)`` would
    cast them too).
    """
    return _recursive_apply(obj, lambda t: t.to(dtype) if t.is_floating_point() else t)


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
        precision: One of ``"float32"``, ``"float16"``, ``"bfloat16"``,
            ``"float8"``. Float kwargs are pre-cast to this dtype before
            the autocast context wraps the forward call.
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

    ### Pre-cast float kwargs to the autocast dtype before entering the
    ### autocast context. This is load-bearing for mesh-input models like
    ### GLOBE: torch.autocast intercepts the first op of each tensor it
    ### sees, but it does NOT reach inside Mesh / DomainMesh / TensorDict
    ### leaves -- the cells / point_data / boundary tensors stay at their
    ### original dtype unless we materialize the cast here. For tensor-
    ### input models, the pre-cast is partially redundant with autocast
    ### but harmless and keeps a single code path.
    ###
    ### `_recursive_cast_floats` skips integer tensors (e.g. Mesh.cells)
    ### so connectivity stays valid through bf16 / fp16 training. fp8 and
    ### fp32 fall through (no pre-cast) -- fp8 is handled inside
    ### `te.fp8_autocast`, fp32 needs no cast at all.
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(precision)
    if dtype is not None:
        forward_kwargs = _recursive_cast_floats(forward_kwargs, dtype)

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
    step and emits per-step TensorBoard + JSONL entries; val mode wraps
    the loop in ``torch.no_grad()`` and skips the per-step writer logging.

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

    ### `total_loss` is a Python float fed by the per-step print line's
    ### sync; `total_losses_td` / `total_metrics_td` are on-device
    ### TensorDict accumulators (one 0-D leaf per field) that defer
    ### their D2H transfer to the single batched ``.tolist()`` at
    ### end-of-epoch. ``None`` here means "not yet seeded"; the first
    ### iteration clones the per-step TensorDict to break aliasing.
    total_loss = 0.0
    total_losses_td: TensorDict | None = None
    total_metrics_td: TensorDict | None = None
    precision = getattr(cfg, "precision", "float32")
    n_batches = 0
    num_steps = len(dataloader)
    epoch_t0 = time.perf_counter()

    with grad_ctx:
        step_t0 = time.perf_counter()
        for i, batch in enumerate(dataloader):
            batch = _recursive_to_device(batch, dist_manager.device)

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
            n_batches += 1

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
            ### Train mode includes Mem in the per-step line; val drops it
            ### because the no_grad path is the lowest-noise place to look.
            mem_str = f" Mem: {mem_gb:.2f}GB" if is_train else ""
            logger.info(
                f"{log_prefix} {epoch} [{i + 1}/{num_steps}] "
                f"Loss: {this_loss:.6f} "
                f"Step: {step_dt:.3f}s"
                f"{mem_str}"
            )

            ### Per-step TensorBoard + JSONL: train only. Val emits one
            ### epoch-level entry below to keep dashboards uncluttered.
            if is_train and dist_manager.rank == 0:
                global_step = epoch * num_steps + i
                losses_floats, metrics_floats = _to_float_dicts(losses, metrics)
                if writer is not None:
                    ### Loss keys already start with `loss/`, so the iteration
                    ### prefix yields tags like `iteration/loss/pressure`;
                    ### metric tags get an explicit `iteration/metrics/...`
                    ### namespace so we never have to split by string prefix.
                    _log_to_tensorboard(writer, losses_floats, "iteration", global_step)
                    _log_to_tensorboard(
                        writer, metrics_floats, "iteration/metrics", global_step
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
                            "phase": "step",
                            "global_step": global_step,
                            "loss": this_loss,
                            "mem_gb": mem_gb,
                            "step_time_s": step_dt,
                            **losses_floats,
                            **metrics_floats,
                        }
                    )

            if cfg.profile and i >= _PROFILE_MAX_STEPS:
                break
            step_t0 = time.perf_counter()

    epoch_dt = time.perf_counter() - epoch_t0
    n = max(n_batches, 1)
    avg_loss = total_loss / n
    avg_losses, avg_metrics = _to_float_dicts(total_losses_td, total_metrics_td, n=n)

    logger.info(
        f"Epoch {epoch} {mode} done in {epoch_dt:.1f}s "
        f"({n_batches} steps, {epoch_dt / n:.3f}s/step avg)"
    )

    if dist_manager.rank == 0:
        _log_to_tensorboard(writer, avg_losses, "epoch", epoch)
        _log_to_tensorboard(writer, avg_metrics, "epoch/metrics", epoch)
        if log_jsonl is not None:
            log_jsonl(
                {
                    "phase": mode,
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


def _resolve_manifest_indices_from_spec(
    reader: Any, manifest_spec: dict[str, Any]
) -> tuple[list[int], list[int] | None]:
    """Resolve a manifest spec to ``(train_indices, val_indices_or_None)``."""
    if manifest_spec["train_manifest"] is not None:
        train_entries = load_manifest(manifest_spec["train_manifest"])
    else:
        train_entries = load_manifest(
            manifest_spec["manifest"], split=manifest_spec["train_split"]
        )
    train_indices = resolve_manifest_indices(reader, train_entries)

    if manifest_spec["val_manifest"] is not None:
        val_entries = load_manifest(manifest_spec["val_manifest"])
        val_indices = resolve_manifest_indices(reader, val_entries)
    elif manifest_spec["val_split"] is not None:
        val_entries = load_manifest(
            manifest_spec["manifest"], split=manifest_spec["val_split"]
        )
        val_indices = resolve_manifest_indices(reader, val_entries)
    else:
        val_indices = None
    return train_indices, val_indices


def _build_collate(
    cfg: DictConfig, target_config: dict[str, FieldType]
) -> Callable[[list[tuple[Any, Any]]], dict[str, Any]]:
    """Build the per-sample collate from the training YAML's I/O contract."""
    if not target_config:
        raise ValueError(
            "Dataset YAML must declare a non-empty `targets:` block. "
            "Targets are the single source of truth for prediction field "
            "names + types."
        )
    input_type = cfg.get("input_type", None)
    if input_type is None:
        raise ValueError(
            "Training YAML must declare `input_type` (one of 'mesh', 'tensors')."
        )
    forward_kwargs_spec = _resolve_dict(cfg, "forward_kwargs")
    if not forward_kwargs_spec:
        raise ValueError(
            "Training YAML must declare a non-empty `forward_kwargs:` block."
        )
    return build_collate_fn(
        input_type=input_type,
        forward_kwargs_spec=forward_kwargs_spec,
        target_config=target_config,
    )


def _combine_datasets(
    datasets: list[MeshDataset],
) -> MeshDataset | MultiDataset:
    """Wrap a list of `MeshDataset`s in a `MultiDataset` if there's more than one."""
    if len(datasets) == 1:
        return datasets[0]
    return MultiDataset(*datasets, output_strict=False)


def _build_directory_samplers(
    train_dataset: Any,
    val_dataset: Any,
    *,
    use_distributed: bool,
    sampler_seed: int,
) -> tuple[Sampler | None, Sampler | None]:
    """Per-split :class:`DistributedSampler` pair for **directory-mode** datasets.

    Used when each split has its own dataset (separate ``train_datadir``
    and ``val_datadir`` in the dataset YAML); manifest-mode shares a
    single dataset across splits and uses :func:`_build_manifest_samplers`
    instead. Returns ``(None, None)`` on a single rank, where torch's
    default sequential sampler is sufficient.
    """
    if not use_distributed:
        return None, None
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True, drop_last=True, seed=sampler_seed
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, shuffle=False, drop_last=False
    )
    return train_sampler, val_sampler


def _build_manifest_samplers(
    train_indices: list[int],
    val_indices: list[int] | None,
    *,
    dist_manager: DistributedManager,
    sampler_seed: int,
) -> tuple[ManifestSampler, ManifestSampler]:
    """ManifestSamplers (with distributed sharding when world_size > 1)."""
    use_distributed = dist_manager.world_size > 1
    rank = dist_manager.rank if use_distributed else 0
    world_size = dist_manager.world_size if use_distributed else 1

    train_sampler = ManifestSampler(
        train_indices,
        shuffle=True,
        seed=sampler_seed,
        rank=rank,
        world_size=world_size,
        drop_last=True,
    )
    ### When no explicit val split is configured, fall back to the train
    ### indices but build a separate non-shuffled, no-drop sampler so val
    ### iteration is deterministic and covers every sample.
    if val_indices is None:
        val_indices = train_indices
    val_sampler = ManifestSampler(
        val_indices,
        shuffle=False,
        seed=sampler_seed,
        rank=rank,
        world_size=world_size,
        drop_last=False,
    )
    return train_sampler, val_sampler


def build_dataloaders(
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader, "NormalizeMeshFields | None", dict[str, Any]]:
    """Build train and val dataloaders from dataset configs.

    Supports two split strategies:

    **Directory-based** (existing): separate ``train_datadir`` and
    ``val_datadir`` in the dataset YAML. Each split gets its own reader
    and dataset.

    **Manifest-based** (new): a single ``datadir`` in the dataset YAML
    with ``train_manifest`` and ``val_manifest`` (or ``manifest`` +
    ``train_split`` / ``val_split``) in the training config's
    ``data.<key>`` block. One reader/dataset covers the full directory;
    :class:`ManifestSampler` restricts each loader to the correct subset
    of indices.

    NOTE (limitation): only ONE ``data.<key>`` block may carry a
    manifest today. If multiple blocks have ``manifest`` /
    ``train_split``, the later block silently overwrites the earlier
    block's indices and the resulting :class:`ManifestSampler` is
    indexed against the last reader's local positions rather than the
    :class:`MultiDataset`'s concatenated positions. To merge splits via
    :class:`MultiDataset` (e.g. train on single_aoa_4 + single_aoa_12
    together), this loop must first be extended to collect per-block
    ``(offset, indices)`` pairs and build a single sampler over
    offset-shifted indices against the :class:`MultiDataset`. Tracked
    as a follow-up.
    """
    recipe_root = Path(__file__).resolve().parent.parent
    batch_size = cfg.training.get("batch_size", 1)
    if batch_size != 1:
        raise NotImplementedError(
            f"This recipe requires batch_size=1, got batch_size={batch_size}. "
            f"All models in this recipe assume B=1; the YAML field is "
            f"reserved for future use."
        )
    sampling_resolution = cfg.dataset.get("sampling_resolution", None)
    augment = cfg.get("augment", False)
    dist_manager = DistributedManager()
    use_distributed = dist_manager.world_size > 1

    ### DataLoader / MeshDataset performance tuning from cfg.dataloader
    dl_cfg = cfg.get("dataloader", {})
    prefetch_factor = dl_cfg.get("prefetch_factor", 2)
    num_streams = dl_cfg.get("num_streams", 4)
    use_streams = dl_cfg.get("use_streams", False)
    num_workers = dl_cfg.get("num_workers", 1)
    pin_memory = dl_cfg.get("pin_memory", False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sampler_seed = cfg.training.get("seed", 0) or 0

    ### Per-block accumulators. Manifest mode collects indices into the
    ### single train_dataset; directory mode collects val_datasets per
    ### block. Only one of (manifest_*_indices, val_datasets) is populated
    ### per dataset block, but they're tracked across blocks here for the
    ### final assembly step below.
    train_datasets: list = []
    val_datasets: list = []
    manifest_train_indices: list[int] | None = None
    manifest_val_indices: list[int] | None = None
    using_manifests = False
    first_metadata: dict | None = None
    first_targets: dict[str, str] | None = None
    first_metrics: list[str] | None = None

    for ds_key in cfg.data:
        ds_cfg_block = cfg.data[ds_key]
        config_path = recipe_root / ds_cfg_block.config
        if not config_path.exists():
            ### Warn-and-skip on a missing dataset config so a typo in
            ### `data.<key>.config` surfaces in the run log rather than
            ### vanishing as an empty dataloader at training time.
            _LOGGER.warning(
                f"Skipping dataset {ds_key!r}: config file not found at "
                f"{str(config_path)!r}. Check `data.{ds_key}.config` in the "
                f"training YAML."
            )
            continue
        train_dir = ds_cfg_block.get("train_dir", "")
        if train_dir and not Path(train_dir).exists():
            _LOGGER.warning(
                f"Skipping dataset {ds_key!r}: train_dir {str(train_dir)!r} "
                f"does not exist. Check `data.{ds_key}.train_dir` in the "
                f"training YAML or the `dataset_paths` interpolation it "
                f"resolves to."
            )
            continue

        ds_yaml = load_dataset_config(config_path)
        if sampling_resolution is not None:
            ds_yaml = OmegaConf.merge(
                ds_yaml, {"sampling_resolution": sampling_resolution}
            )

        ### Read the dataset YAML's contract block so we can validate
        ### consistency across multi-dataset training.
        ds_targets = OmegaConf.to_container(
            OmegaConf.select(ds_yaml, "targets", default=OmegaConf.create({})),
            resolve=True,
        )
        ds_metrics = OmegaConf.to_container(
            OmegaConf.select(ds_yaml, "metrics", default=OmegaConf.create([])),
            resolve=True,
        )
        ds_metadata = OmegaConf.to_container(
            OmegaConf.select(ds_yaml, "metadata", default=OmegaConf.create({})),
            resolve=True,
        )
        if first_targets is None:
            first_targets, first_metrics, first_metadata = (
                ds_targets,
                ds_metrics,
                ds_metadata,
            )
        else:
            validate_dataset_consistency(
                ds_key,
                ds_targets,
                ds_metrics,
                ds_metadata,
                first_targets,
                first_metrics,
                first_metadata,
            )

        manifest_spec = resolve_manifest_spec(ds_yaml, ds_cfg_block)
        if manifest_spec is not None:
            using_manifests = True
            ### Manifest mode: the reader must see ALL runs under one
            ### root. The config block can provide ``datadir`` to override
            ### the dataset YAML's ``train_datadir`` with the parent
            ### directory that contains every run (train + val).
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
            ### NOTE: this overwrites any prior block's indices; see the
            ### docstring's multi-block limitation note.
            manifest_train_indices, manifest_val_indices = (
                _resolve_manifest_indices_from_spec(dataset.reader, manifest_spec)
            )
            continue

        ### Directory mode: separate readers / datasets per split.
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

    normalizer = find_normalizer(train_datasets)
    collate_fn = _build_collate(cfg, first_targets or {})
    train_dataset = _combine_datasets(train_datasets)

    if using_manifests:
        ### Manifest mode: train and val share one underlying dataset;
        ### the samplers carve out the per-split index sets.
        val_dataset = train_dataset
        train_sampler, val_sampler = _build_manifest_samplers(
            manifest_train_indices,
            manifest_val_indices,
            dist_manager=dist_manager,
            sampler_seed=sampler_seed,
        )
    else:
        ### Directory mode: separate datasets per split, with per-rank
        ### DistributedSamplers when world_size > 1.
        val_dataset = _combine_datasets(val_datasets) if val_datasets else train_dataset
        train_sampler, val_sampler = _build_directory_samplers(
            train_dataset,
            val_dataset,
            use_distributed=use_distributed,
            sampler_seed=sampler_seed,
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

    dataset_info = {
        "metadata": first_metadata or {},
        "targets": first_targets or {},
        "metrics": first_metrics or list(DEFAULT_METRICS),
    }
    return train_loader, val_loader, normalizer, dataset_info


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

    train_loader, val_loader, normalizer, dataset_info = build_dataloaders(cfg)
    target_config: dict[str, FieldType] = dataset_info["targets"]
    metrics_list: list[MetricName] = dataset_info["metrics"]
    ds_metadata: dict = dataset_info["metadata"]
    logger.info(f"Train samples: {len(train_loader.sampler)}")
    logger.info(f"Val samples: {len(val_loader.sampler)}")
    logger.info(f"Targets (from dataset YAML): {target_config}")

    # -- Log dataset metadata (rank 0) --------------------------------------------
    if dist_manager.rank == 0 and log_jsonl is not None:
        log_jsonl(
            {
                "phase": "dataset",
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "metadata": ds_metadata or {},
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

    ### `target_config` and `metrics_list` were loaded from the dataset YAML
    ### by `build_dataloaders` -- see the dataset_info dict above. The
    ### training YAML may override the metrics list with a (typically
    ### shorter) `dataset.metrics` selection.
    metrics_override = OmegaConf.select(cfg, "dataset.metrics", default=None)
    if metrics_override is not None:
        metrics_list = OmegaConf.to_container(metrics_override, resolve=True)

    field_weights = _resolve_dict(cfg, "training.field_weights")

    metric_calculator = MetricCalculator(
        target_config=target_config,
        metrics=metrics_list,
    )
    loss_calculator = LossCalculator(
        target_config=target_config,
        loss_type=cfg.training.get("loss_type", "huber"),
        field_weights=field_weights,
    )
    output_type = cfg.get("output_type", None)
    if output_type is None:
        raise ValueError(
            "Training YAML must declare `output_type` (one of 'mesh', 'tensors')."
        )
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
