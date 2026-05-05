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

# Suppress all warnings during import. In multi-GPU training (up to 400
# ranks), every process emits identical import-time warnings (e.g.
# ExperimentalFeatureWarning, Warp DeprecationWarning) and srun merges them
# into one log file. Re-enabled for rank 0 after distributed init below.
import warnings

warnings.filterwarnings("ignore")

import contextlib
import logging
import os
from collections import defaultdict
from datetime import datetime
from itertools import count
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import matplotlib as mpl
import torch
import torch.nn.functional as F
import torchinfo
from dataset import (
    DrivAerMLDataSet,
    DrivAerMLSample,
    postprocess,
    visualize_comparison,
)
from jaxtyping import Float
from mlflow.tracking.fluent import (
    active_run,
    log_artifact,
    log_metrics,
    set_experiment,
    start_run,
)
from tensordict import TensorDict
from torch.distributed import ReduceOp, all_reduce, barrier
from torch.profiler import record_function
from torch.utils.data import DataLoader
from tqdm import tqdm
from utilities import (
    log_hyperparameters,
    resilient,
    sanitize_metric_name,
)

from physicsnemo.core import get_physicsnemo_pkg_info
from physicsnemo.distributed import DistributedManager
from physicsnemo.experimental.models.globe.model import GLOBE
from physicsnemo.experimental.utils import (
    disable_autotune_printing,
    prefetch_map,
    silence_compile_logs_on_non_zero_ranks,
)
from physicsnemo.optim import CombinedOptimizer
from physicsnemo.utils.checkpoint import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.profiling import Profiler

mpl.use("agg")  # Allows headless plotting
disable_autotune_printing()

Split = Literal["train", "validation"]
splits: list[Split] = ["train", "validation"]

# MLflow's system-metrics monitor thread can collide with the main thread
# when both write to SQLite on Lustre, causing "database is locked" errors.
# Wrapping with retry ensures transient failures never kill training.
log_artifact = resilient(log_artifact)
log_metrics = resilient(log_metrics)


def main(
    data_dir: Path | None = None,
    output_name: str | None = None,
    amp: bool = True,
    use_compile: bool = True,
    compile_mode: Literal[
        "default",
        "max-autotune-no-cudagraphs",
    ] = "default",  # max-autotune-no-cudagraphs has a CUDA illegal memory error. Due to buggy triton kernel in no-grad.
    n_prediction_points: int | None = None,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-4,
    use_muon: bool = True,
    muon_method: Literal["original", "match_rms_adamw"] = "match_rms_adamw",
    train_randomize_face_centers: bool = True,
    seed: int = 0,
    error_scales: dict[str, float] | None = None,
    n_communication_hyperlayers: int = 2,
    hidden_layer_sizes: tuple[int, ...] = (256, 256, 256),
    n_latent_scalars: int = 8,
    n_latent_vectors: int = 4,
    n_spherical_harmonics: int = 4,
    theta: float = 1.0,
    leaf_size: int = 1,
    n_faces_per_boundary: int = 80_000,
    patience_steps: int = 1600,
    use_profiler: bool = True,
    make_images: bool = False,
    save_every: int = 1,
    use_mlflow: bool = True,
    mlflow_experiment: str = "GLOBE_DrivAerML",
    gradient_clip_norm: float | None = 1.0,
    network_type: Literal["pade", "mlp"] = "pade",
    self_regularization_beta: float | None = 0.01,
    latent_compression_scale: float | None = 100.0,
    expand_far_targets: bool = True,
):
    """Train GLOBE on DrivAerML.

    Args:
        data_dir: Path to the DrivAerML dataset root (containing ``run_N/``
            subdirectories).  Falls back to ``DRIVAER_DATA_DIR`` env var.
        output_name: Name for the output directory.  Defaults to a timestamp.
        amp: Enable automatic mixed precision (bfloat16).
        use_compile: Enable ``torch.compile`` for the forward/loss function.
        compile_mode: Compilation mode for ``torch.compile``.
        n_prediction_points: Surface points sampled per training iteration.
            ``None`` (default) uses ``n_faces_per_boundary``.
        learning_rate: Base learning rate (sqrt-scaled by world size).
        weight_decay: Weight decay factor.
        use_muon: Use Muon optimizer for 2D parameters (matrix weights).
        muon_method: Muon learning-rate adjustment method.
        train_randomize_face_centers: Sample random points inside faces
            instead of centroids during training.
        seed: Random seed.
        error_scales: Per-field loss scaling.  Keys must match output field
            names.  Defaults to ``{"C_p": 1.0, "C_f": 0.01}``.
        n_communication_hyperlayers: GLOBE boundary-to-boundary comm layers.
        hidden_layer_sizes: Kernel MLP architecture.
        n_latent_scalars: Scalar latent channels between hyperlayers.
        n_latent_vectors: Vector latent channels between hyperlayers.
        n_spherical_harmonics: Legendre polynomial terms (default 4 for 3D).
        theta: Barnes-Hut opening angle. Larger values are more
            aggressive (more approximation, faster). 0 = exact.
        leaf_size: Maximum sources per leaf node in the Barnes-Hut tree.
        n_faces_per_boundary: Target boundary mesh face count after decimation.
        patience_steps: ReduceLROnPlateau patience expressed in gradient
            steps (world-size independent).  Converted to epochs internally.
        use_profiler: Enable PyTorch profiler (rank 0 only).
        make_images: Generate visualization images during training.
        save_every: Save a checkpoint every this many epochs.
        use_mlflow: Enable MLflow experiment tracking.
        mlflow_experiment: MLflow experiment name.
        expand_far_targets: If True, expand far-field target nodes to
            individual points so target-side approximation is removed
            (more kernel evaluations, often more stable training).
    """
    ### [Config Processing]
    if data_dir is None:
        if _data_dir_str := os.environ.get("DRIVAER_DATA_DIR"):
            data_dir = Path(_data_dir_str)
        else:
            raise ValueError(
                "DrivAerML data directory not specified.  Pass `data_dir` or "
                "set the DRIVAER_DATA_DIR environment variable."
            )
    data_dir = Path(data_dir)

    if output_name is None:
        output_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(__file__).parent / "output" / output_name
    cache_dir = Path(__file__).parent / "cache"

    if n_prediction_points is None:
        n_prediction_points = n_faces_per_boundary

    error_scale_config = {
        "C_p": 1.0,
        "C_f": 0.01,
    } | ({} if error_scales is None else error_scales)

    config_settings = locals()

    ### [Distributed Training Setup]
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    torch.cuda.set_device(device)
    silence_compile_logs_on_non_zero_ranks(dist.rank)

    if dist.rank == 0:
        logging.basicConfig(level=logging.INFO)
        warnings.resetwarnings()  # undo module-level suppression for rank 0
        # MLflow's SQLAlchemy store and system-metrics monitor emit verbose
        # tracebacks on transient SQLite "database is locked" errors.  Our
        # resilient() wrapper reports final failures as one-liners.
        logging.getLogger("mlflow.store.db.utils").setLevel(logging.CRITICAL)
        logging.getLogger("mlflow.system_metrics.system_metrics_monitor").setLevel(
            logging.CRITICAL
        )
    else:
        logging.disable(logging.ERROR)
    logger = PythonLogger("globe.drivaer.train")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger0.info(f"{dist.world_size = }")

    error_scales: TensorDict[str, Float[torch.Tensor, ""]] = TensorDict(
        error_scale_config,
        device=device,
    )

    ### [Output Directory Setup]
    torch_compile_cache_dir = output_dir / "torch_compile_cache"
    torch_compile_cache = torch_compile_cache_dir / f"rank_{dist.rank}.compile_cache"
    checkpoint_dir = output_dir / "checkpoints"
    best_model_path = output_dir / "best_model.mdlus"
    profiling_dir = output_dir / "profiling"
    shutdown_file = output_dir / "SHUTDOWN"

    for directory in (checkpoint_dir, torch_compile_cache_dir, profiling_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if dist.rank == 0:
        shutdown_file.unlink(missing_ok=True)

    ### [PyTorch Configuration]
    autocast_ctx = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=amp
    )
    torch.set_float32_matmul_precision("high")  # Allows use of Tensor Cores in matmuls
    torch.manual_seed(seed)

    ### [Dataset Preparation]
    sample_paths: dict[Split, list[Path]] = {
        split: DrivAerMLDataSet.get_split_paths(data_dir, split) for split in splits
    }
    dataloaders: dict[Split, DataLoader] = {
        split: DrivAerMLDataSet.make_dataloader(
            sample_paths[split],
            cache_dir,
            world_size=dist.world_size,
            rank=dist.rank,
            n_faces_per_boundary=n_faces_per_boundary,
        )
        for split in splits
    }

    ### [Model]
    # Reference area: constant aRefRef = 2.170 m² from the DrivAerML spec
    model = GLOBE(
        n_spatial_dims=3,
        output_field_ranks={
            "C_p": 0,
            "C_f": 1,
        },
        boundary_source_data_ranks={
            "vehicle": {},
            "no_slip_floor": {},
            "slip_floor": {},
        },
        reference_length_names=["L_ref", "delta_turb"],
        reference_area=2.170,
        global_data_ranks=None,
        n_communication_hyperlayers=n_communication_hyperlayers,
        hidden_layer_sizes=hidden_layer_sizes,
        n_latent_scalars=n_latent_scalars,
        n_latent_vectors=n_latent_vectors,
        n_spherical_harmonics=n_spherical_harmonics,
        theta=theta,
        leaf_size=leaf_size,
        network_type=network_type,
        self_regularization_beta=self_regularization_beta,
        latent_compression_scale=latent_compression_scale,
        expand_far_targets=expand_far_targets,
    ).to(device)

    logger0.info(f"{output_dir.name=!r}")

    base_model = model

    # TODO: candidate for upstreaming to physicsnemo once torch.compiler
    # cache APIs stabilize (currently experimental in PyTorch).
    if use_compile and torch_compile_cache.exists():
        torch.compiler.load_cache_artifacts(torch_compile_cache.read_bytes())

    # Different MultiscaleKernel instances have different MLP output sizes.
    # Without this, Dynamo guards on parameter shapes and recompiles for each
    # kernel branch, quickly exhausting the recompile limit.
    torch._dynamo.config.force_parameter_static_shapes = False
    torch._dynamo.config.capture_scalar_outputs = True

    # The GLOBE model stores latent channels as individually-named TensorDict
    # entries (one key per scalar/vector channel).  Dynamo specializes on each
    # key, so the default limit of 8 is exhausted mid-forward and remaining
    # code falls back to eager.
    torch._dynamo.config.cache_size_limit = 64

    ### [Distribute the model across GPUs]
    if dist.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=device,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    ### [Optimizer and Scheduler Setup]
    # No automatic LR scaling for batch size / world_size.  Muon
    # orthogonalizes the gradient, fixing the update norm independent of
    # batch size, so the maximum safe LR is determined by loss-landscape
    # curvature, not gradient noise.  See arXiv:2502.16982 Sec 2.2.
    if use_muon:
        # Muon is designed for matrix-shaped parameters (2D weight tensors
        # of linear layers); biases, norms, and other non-matrix parameters
        # fall back to RAdam.  This ndim==2 split is the standard Muon
        # recommendation.
        optimizer = CombinedOptimizer(
            optimizers=[
                torch.optim.Muon(
                    [p for p in model.parameters() if p.ndim == 2],
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    adjust_lr_fn=muon_method,
                ),
                torch.optim.RAdam(
                    [p for p in model.parameters() if p.ndim != 2],
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    decoupled_weight_decay=True,
                    foreach=True,
                ),
            ],
        )
    else:
        optimizer = torch.optim.RAdam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            decoupled_weight_decay=True,
            foreach=True,
        )
    patience_epochs = max(1, patience_steps // len(dataloaders["train"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=patience_epochs,
        min_lr=learning_rate / 64,
        threshold=1e-3,
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp)

    ### [Checkpoint Save/Load]
    metadata_dict: dict[str, Any] = {}
    epoch = load_checkpoint(
        checkpoint_dir,
        models=base_model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        metadata_dict=metadata_dict,
        device=dist.device,
    )
    logger0.info(
        f"Resuming training from epoch {epoch}"
        if epoch > 0
        else "Starting training from scratch."
    )
    best_loss = metadata_dict.get("best_loss", float("inf"))
    last_image_epoch = metadata_dict.get("last_image_epoch", -float("inf"))
    last_image_loss = metadata_dict.get("last_image_loss", float("inf"))
    mlflow_run_id: str | None = metadata_dict.get("mlflow_run_id")

    ### [Scheduler patience normalization]
    # ReduceLROnPlateau measures patience in epochs.  When world_size
    # changes, epoch length changes (fewer steps per epoch with more
    # GPUs), so we recompute patience in epochs from the step-based
    # target and rescale the bad-epoch counter accordingly.
    scheduler.patience = patience_epochs
    loaded_world_size = metadata_dict.get("world_size")
    if loaded_world_size is not None and loaded_world_size != dist.world_size:
        ws_ratio = dist.world_size / loaded_world_size
        scheduler.num_bad_epochs = round(scheduler.num_bad_epochs * ws_ratio)

    ### [First-Launch Diagnostics]
    # Verbose diagnostics (torchinfo, GLOBE debug, graph break summary) are
    # only emitted on the very first SLURM launch (epoch==0). Subsequent
    # --dependency=singleton restarts skip them entirely. Within the first
    # launch, debug logging and graph break capture are disabled after the
    # first training batch completes.
    is_first_launch = (epoch == 0) and dist.rank == 0
    _globe_logger: logging.Logger | None = None

    if is_first_launch:
        torchinfo.summary(base_model, depth=4)
        _globe_logger = logging.getLogger("globe")
        _globe_logger.setLevel(logging.DEBUG)
        torch._logging.set_logs(graph_breaks=True, recompiles=True)

    ### [MLflow Setup]
    mlflow_run_ctx: contextlib.AbstractContextManager = contextlib.nullcontext()
    if dist.rank == 0 and use_mlflow:
        set_experiment(experiment_name=mlflow_experiment)
        if mlflow_run_id:
            try:
                mlflow_run_ctx = start_run(
                    run_id=mlflow_run_id, log_system_metrics=True
                )
                logger0.info(f"Resumed MLflow run {mlflow_run_id}")
            except Exception:
                warnings.warn(
                    f"Could not resume MLflow run {mlflow_run_id!r}, creating new run"
                )
                mlflow_run_id = None
        if not mlflow_run_id:
            mlflow_run_ctx = start_run(
                run_name=f"{output_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags={"output_name": output_name},
                log_system_metrics=True,
            )

    ### [Hyperparameter Logging]
    if dist.rank == 0 and epoch == 0:
        log_hyperparameters(
            log_dir=output_dir,
            model=base_model,
            other_hyperparameters={
                **config_settings,
                "optimizer": optimizer.__class__.__name__,
                "scheduler": scheduler.__class__.__name__,
                "scaler": scaler.__class__.__name__,
                "physicsnemo_pkg_info": get_physicsnemo_pkg_info(),
                "world_size": dist.world_size,
                **{f"n_{split}_samples": len(sample_paths[split]) for split in splits},
            },
        )
        if use_mlflow:
            log_artifact(str(output_dir / "hyperparameters.yaml"))

    ### [Training and Testing]
    @torch.compile(
        dynamic=True,
        mode=compile_mode,
        disable=not use_compile,
    )
    def run_batch(
        sample: DrivAerMLSample,
    ) -> tuple[torch.Tensor, TensorDict[str, Float[torch.Tensor, ""]]]:
        """Forward pass + loss for one sample."""
        pred_mesh = model(**sample.model_input_kwargs)
        batch_loss_components = pred_mesh.point_data.apply(
            field_loss_fn,
            sample.prediction_mesh.point_data,
            error_scales.expand_as(pred_mesh.point_data),
        ).mean(dim=0)
        batch_loss = batch_loss_components.stack_from_tensordict().sum()
        return batch_loss, batch_loss_components

    def run_epoch(split: Split) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run one epoch of training or testing."""
        training = split == "train"
        dataloaders[split].sampler.set_epoch(epoch=epoch)
        model.train(training)

        all_batch_losses: list[torch.Tensor] = []
        all_batch_loss_components: dict[str, list[torch.Tensor]] = defaultdict(list)

        for sample in tqdm(
            prefetch_map(
                dataloaders[split],
                lambda s: s.prepare(
                    n_prediction_points,
                    device,
                    randomize_vehicle=training and train_randomize_face_centers,
                ),
            ),
            desc=f"{epoch:d} {split.title()}",
            unit=" samples",
            disable=dist.rank != 0 or epoch > 10,
        ):
            torch.compiler.cudagraph_mark_step_begin()

            with (
                autocast_ctx,
                contextlib.nullcontext() if training else torch.no_grad(),
                record_function("main_processing_loop"),
            ):
                if training:
                    optimizer.zero_grad()

                with record_function("forward"):
                    batch_loss, batch_loss_components = run_batch(sample)

                if training:
                    if torch.isnan(batch_loss):
                        warnings.warn(f"{batch_loss=} at: {dist.rank=}, {epoch=}")
                    with record_function("backward"):
                        scaler.scale(batch_loss).backward()
                    if gradient_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=gradient_clip_norm
                        )
                    with record_function("optimizer_step"):
                        scaler.step(optimizer)
                        scaler.update()

                all_batch_losses.append(batch_loss.detach().clone())
                for k, v in batch_loss_components.items():
                    all_batch_loss_components[k].append(v.detach().clone())

            if training:
                profiler.step()

            ### Disable all first-launch diagnostics after the first batch.
            ### Re-entry guard: globe_logger.level is DEBUG only between
            ### first-launch setup and the first cleanup pass.
            if _globe_logger is not None and _globe_logger.level == logging.DEBUG:
                _globe_logger.setLevel(logging.INFO)
                torch._logging.set_logs(graph_breaks=False, recompiles=False)

        ### [Distributed comms]
        keys = ["loss", *all_batch_loss_components.keys()]
        all_values = torch.stack(
            [
                torch.nanmean(torch.stack(all_batch_losses)),
                *(
                    torch.nanmean(torch.stack(all_batch_loss_components[k]))
                    for k in keys[1:]
                ),
            ]
        )
        if dist.world_size > 1:
            all_reduce(all_values, op=ReduceOp.AVG)
        epoch_loss = all_values[0]
        epoch_loss_components = dict(zip(keys[1:], all_values[1:]))

        logger0.info(
            " | ".join(
                [
                    f"{epoch:d=} {split.title():<{max(len(s.title()) for s in splits)}}",
                    f"Loss: {epoch_loss:7.3g}",
                    *[f"{k}: {v:7.3g}" for k, v in epoch_loss_components.items()],
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}",
                ]
            )
        )
        return epoch_loss, epoch_loss_components

    ### [Profiler Setup]
    profiler = Profiler()
    if use_profiler and dist.rank == 0 and (not any(profiling_dir.iterdir())):
        profiler.enable("torch").reconfigure(
            schedule=torch.profiler.schedule(wait=5, warmup=1, active=1, repeat=1),
            on_trace_ready_path=profiling_dir,
            with_stack=False,
        )
    # Co-locate the optional summary tables (cpu_time.txt, gpu_time.txt)
    # next to the trace JSON in `profiling_dir/torch/`, instead of the
    # default `./physicsnemo_profiling_outputs/torch/` (cwd-relative).
    profiler.output_path = profiling_dir
    profiler.initialize()

    with mlflow_run_ctx, profiler:
        ### [Training Loop]

        if dist.rank == 0:
            time_last_epoch = perf_counter()

        def checkpoint_metadata() -> dict[str, Any]:
            return {
                "best_loss": best_loss,
                "last_image_epoch": last_image_epoch,
                "last_image_loss": last_image_loss,
                "mlflow_run_id": (
                    _run.info.run_id if use_mlflow and (_run := active_run()) else None
                ),
                "world_size": dist.world_size,
            }

        def save_ckpt() -> None:
            save_checkpoint(
                checkpoint_dir,
                models=base_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                metadata=checkpoint_metadata(),
            )

        for epoch in count(start=epoch + 1):
            loss = {}
            loss_components = {}
            for split in splits:
                with record_function(f"epoch_{epoch}_{split}"):
                    loss[split], loss_components[split] = run_epoch(split)

            scheduler.step(loss["train"])

            if dist.world_size > 1:
                barrier()

            ### [Logging and Checkpointing]
            if dist.rank == 0:
                if epoch % save_every == 0:
                    save_ckpt()
                if loss["validation"] < best_loss:
                    best_loss = loss["validation"]
                    base_model.save(best_model_path)
                    if use_mlflow:
                        log_artifact(str(best_model_path), artifact_path="best_model")

                if use_mlflow:
                    log_metrics(
                        {
                            **{f"{split}_loss": loss[split].item() for split in splits},
                            **{
                                f"{split}_loss_components/{sanitize_metric_name(k)}": v.item()
                                for split in splits
                                for k, v in loss_components[split].items()
                            },
                            "lr": optimizer.param_groups[0]["lr"],
                            "system/vram_gb": torch.cuda.memory_stats()[
                                "reserved_bytes.all.peak"
                            ]
                            / 1024**3,
                            "system/seconds_per_epoch": (time_now := perf_counter())
                            - time_last_epoch,
                        },
                        step=epoch,
                    )
                    time_last_epoch = time_now

            if shutdown_file.exists():
                logger0.info("Quitting due to shutdown request.")
                if dist.rank == 0:
                    save_ckpt()
                break

            ### [Visualization]
            if (
                make_images
                and (loss["train"] / last_image_loss < 0.9)
                and (epoch > last_image_epoch + 200)
            ):
                if dist.rank == 0:
                    logger0.info("Generating visualization images...")
                    for split in splits:
                        generate_visualization(
                            model=base_model,
                            dataset=dataloaders[split].dataset,
                            n_faces_per_boundary=n_faces_per_boundary,
                            device=device,
                            autocast_ctx=autocast_ctx,
                            output_dir=output_dir,
                            split=split,
                            epoch=epoch,
                            use_mlflow=use_mlflow,
                            logger=logger0,
                        )

                last_image_epoch, last_image_loss = epoch, loss["train"]
                if dist.world_size > 1:
                    barrier()

            ### [torch.compile Caching]
            if use_compile and not torch_compile_cache.exists():
                artifacts_bytes, cache_info = torch.compiler.save_cache_artifacts()
                torch_compile_cache.write_bytes(artifacts_bytes)
                logger.info(f"Saved torch.compile cache to {torch_compile_cache}.")


def field_loss_fn(
    pred: Float[torch.Tensor, "n_points ..."],
    true: Float[torch.Tensor, "n_points ..."],
    error_scale: Float[torch.Tensor, ""],
) -> Float[torch.Tensor, " n_points"]:
    """Per-point Huber loss with NaN masking and per-field scaling.

    Computes ``||(pred - true) / error_scale||`` with Huber smoothing
    (delta=1) and masks out NaN entries in *true*.

    Args:
        pred: Predicted values, ``(n_points,)`` or ``(n_points, 3)``.
        true: Ground truth (same shape). NaN entries are masked.
        error_scale: Scalar scaling factor for this field.

    Returns:
        Per-point loss of shape ``(n_points,)``.
    """
    error = torch.where(
        torch.isnan(true),
        torch.zeros_like(true),
        (pred - true) / error_scale,
    )
    if error.ndim > 1:
        error = error.norm(dim=-1)
    return 2 * F.huber_loss(error, torch.zeros_like(error), reduction="none", delta=1.0)


def generate_visualization(
    model: torch.nn.Module,
    dataset: DrivAerMLDataSet,
    *,
    n_faces_per_boundary: int,
    device: torch.device,
    autocast_ctx: contextlib.AbstractContextManager,
    output_dir: Path,
    split: str,
    epoch: int,
    use_mlflow: bool,
    logger: Any,
) -> None:
    """Run inference on sample 0 of a split and produce comparison visualizations.

    Generates a pred/true/error image, computes integrated force coefficients,
    and logs both to MLflow (if enabled) and the console.

    Args:
        model: Trained GLOBE model (will be set to eval mode).
        dataset: Dataset to draw sample 0 from.
        n_faces_per_boundary: Face count for subsampling the prediction mesh.
        device: Device to run inference on.
        autocast_ctx: Autocast context for mixed precision.
        output_dir: Directory for saving visualization images.
        split: Split name (for labeling output files and log messages).
        epoch: Current epoch (for labeling output files and MLflow steps).
        use_mlflow: Whether to log artifacts and metrics to MLflow.
        logger: Logger instance for console output.
    """
    viz_sample = dataset[0]

    ### Subsample prediction surface for speed
    if viz_sample.prediction_mesh.n_cells > n_faces_per_boundary:
        viz_sample.prediction_mesh = DrivAerMLDataSet.subsample_mesh(
            viz_sample.prediction_mesh,
            n_faces_per_boundary,
            geometry_only=False,
        )

    viz_sample = viz_sample.to(device)
    with torch.no_grad(), autocast_ctx:
        model.eval()
        pred_mesh = model(**viz_sample.model_input_kwargs)

    combined = postprocess(
        pred_mesh=pred_mesh.to(device="cpu"),
        sample=viz_sample.to(device="cpu"),
    )

    save_path = output_dir / f"viz_{split}_epoch_{epoch}.png"
    visualize_comparison(combined, save_path=save_path, backend="matplotlib")
    if use_mlflow:
        log_artifact(str(save_path), artifact_path="visualization")

    ### Log force coefficients
    pred_coeffs = combined.global_data["pred"].to_dict()  # ty: ignore[unresolved-attribute]
    true_coeffs = combined.global_data["true"].to_dict()  # ty: ignore[unresolved-attribute]

    logger.info(
        f"Force coefficients ({split}):"
        + "".join(
            f"\n  {k}: pred={pred_coeffs[k]:.5f}"
            f"  true={true_coeffs[k]:.5f}"
            f"  err={pred_coeffs[k] - true_coeffs[k]:+.5f}"
            for k in ("Cd", "Cl", "Cs")
        )
    )
    if use_mlflow:
        log_metrics(
            {
                f"force_coeffs/{split}_{k}_{src}": coeffs[k]
                for src, coeffs in [
                    ("pred", pred_coeffs),
                    ("true", true_coeffs),
                ]
                for k in pred_coeffs
            },
            step=epoch,
        )


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
