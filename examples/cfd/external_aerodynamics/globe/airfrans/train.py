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

import contextlib
import logging
import os
import warnings
from collections import defaultdict
from datetime import datetime
from itertools import count
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn.functional as F
import torchinfo
from dataset import AirFRANSDataSet, AirFRANSSample, compute_max_mesh_sizes
from tensordict import TensorDict
from torch.profiler import record_function
from tqdm import tqdm
from utilities import (
    disable_autotune_printing,
    get_physicsnemo_pkg_info,
    install_graceful_shutdown,
    log_hyperparameters,
    sanitize_metric_name,
)

from physicsnemo.distributed import DistributedManager
from physicsnemo.experimental.models.globe.model import GLOBE
from physicsnemo.optim import CombinedOptimizer
from physicsnemo.utils.checkpoint import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

mpl.use("agg")  # Allows headless plotting
disable_autotune_printing()  # Silences the verbose output of `torch.compile(..., mode="max-autotune")`.


def main(
    data_dir: Path | None = None,
    output_name: str | None = None,
    amp: bool = False,
    use_compile: bool = True,
    compile_mode: Literal[
        "default", "max-autotune-no-cudagraphs", "reduce-overhead", "max-autotune"
    ] = "max-autotune",
    points_per_iter: int = 2048,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    use_muon: bool = True,
    muon_method: Literal["original", "match_rms_adamw"] = "original",
    train_face_downsampling_ratio: float = 1.0,
    train_randomize_face_centers: bool = True,
    seed: int = 0,
    error_scales: dict[str, float] | None = None,
    n_communication_hyperlayers: int = 2,
    hidden_layer_sizes: tuple[int, ...] = (64, 64, 64),
    n_latent_scalars: int = 12,
    n_latent_vectors: int = 6,
    n_spherical_harmonics: int = 1,
    airfrans_task: Literal["full", "scarce", "reynolds", "aoa"] = "full",
    profile: bool = True,
    make_images: bool = True,
    use_mlflow: bool = True,
    mlflow_experiment: str = "GLOBE_AirFRANS",
):
    """Train the GLOBE model on AirFRANS dataset.

    Args:
        data_dir: Path to the AirFRANS dataset directory. Resolution order:
            1. This argument (if provided)
            2. AIRFRANS_DATA_DIR environment variable (set automatically by run.sh)
        output_name: Name for output directory. If None, uses current timestamp.
        amp: Enable automatic mixed precision (AMP) training for faster computation.
        use_compile: Enable torch.compile for model optimization and performance.
        compile_mode: Mode for torch.compile.
        points_per_iter: Number of points to sample per training iteration.
        learning_rate: Initial learning rate for the Adam optimizer.
        weight_decay: Weight decay (L2 regularization) factor for the optimizer.
        train_face_downsampling_ratio: Ratio of faces to keep when downsampling boundary meshes.
        train_randomize_face_centers: Whether to use random points inside faces instead of centroids.
        seed: Random seed for reproducibility across runs.
        error_scales: Dictionary specifying error scales for loss components. If None, uses default scales.
        n_communication_hyperlayers: Number of boundary-to-boundary communication layers.
        hidden_layer_sizes: Hidden layer sizes for the kernel MLP architecture.
        n_latent_scalars: Number of scalar latent channels propagated between hyperlayers.
        n_latent_vectors: Number of vector latent channels propagated between hyperlayers.
        n_spherical_harmonics: Number of Legendre polynomial terms for angle features.
        airfrans_task: Which AirFRANS dataset task to train on.
        profile: Enable PyTorch profiler for performance analysis.
        make_images: Whether to make images for visualization.
        use_mlflow: Enable MLflow experiment tracking. Requires MLFLOW_TRACKING_URI to be set
            in the environment (see run.sh). When False, training still logs to console and
            saves hyperparameters to YAML, but skips all MLflow calls.
        mlflow_experiment: MLflow experiment name. Ignored when use_mlflow is False.

    Note:
        Output directory is created under the script's parent directory in an 'output' folder.
        Error scales control the relative weighting of different physical fields in the loss.
        When profiling is enabled, results are saved to output_dir/profiling/ as Chrome trace files.
    """
    ### [Config Processing]
    if data_dir is None:
        if data_dir := os.environ.get("AIRFRANS_DATA_DIR"):
            data_dir = Path(data_dir)
        else:
            raise ValueError("AirFRANS data directory not specified. Pass `data_dir` or set the AIRFRANS_DATA_DIR environment variable.")
    data_dir = Path(data_dir)

    if output_name is None:
        output_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(__file__).parent / "output" / output_name
    cache_dir = Path(__file__).parent / "cache"

    # Parse error scales
    error_scales = {
        "ΔU/|U_inf|": 1.0,
        "C_p": 1.0,
        "C_pt": 1.0,
        "ln(1+nut/nu)": 5.0,
        "C_F,shear": 0.01,
    } | ({} if error_scales is None else error_scales)

    config_settings = locals()

    ### [Distributed Training Setup]
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    torch.cuda.set_device(device)

    logging.basicConfig(level=logging.INFO)
    logger = PythonLogger("globe.airfrans.train")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger0.info(f"{dist.world_size = }")

    error_scales: TensorDict = TensorDict(error_scales, device=device)
    if dist.rank == 0:
        torch._logging.set_logs(graph_breaks=True, recompiles=True)

    ### [Output Directory Setup]
    torch_compile_cache_dir = output_dir / "torch_compile_cache"
    torch_compile_cache = torch_compile_cache_dir / f"rank_{dist.rank}.compile_cache"
    checkpoint_dir = output_dir / "checkpoints"
    best_model_path = output_dir / "best_model.mdlus"
    profiling_dir = output_dir / "profiling"

    if dist.rank == 0:
        for directory in (
            checkpoint_dir,
            torch_compile_cache_dir,
            profiling_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    ### [Signal Handling]
    shutdown_requested = install_graceful_shutdown(rank=dist.rank)

    ### [PyTorch Configuration]
    autocast_ctx = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=amp
    )
    torch.set_float32_matmul_precision("high")  # Allows use of Tensor Cores in matmuls
    torch.manual_seed(seed)

    ### [Dataset Preparation]
    train_sample_paths = AirFRANSDataSet.get_split_paths(
        data_dir, airfrans_task, "train"
    )
    valid_sample_paths = AirFRANSDataSet.get_split_paths(
        data_dir, airfrans_task, "test"
    )

    train_dataloader = AirFRANSDataSet.make_dataloader(
        train_sample_paths,
        cache_dir,
        world_size=dist.world_size,
        rank=dist.rank,
    )
    valid_dataloader = AirFRANSDataSet.make_dataloader(
        valid_sample_paths,
        cache_dir,
        world_size=dist.world_size,
        rank=dist.rank,
    )

    ### [Model]
    model = GLOBE(
        n_spatial_dims=2,
        output_fields={
            "ΔU/|U_inf|": "vector",
            "C_p": "scalar",
            "C_pt": "scalar",
            "ln(1+nut/nu)": "scalar",
            "C_F,shear": "vector",
        },
        boundary_condition_names=["no_slip"],
        boundary_condition_n_source_scalars={"no_slip": 0},
        boundary_condition_n_source_vectors={"no_slip": 0},
        reference_length_names=["chord", "delta_FS"],
        reference_area=torch.tensor(1.0, device=device),
        n_global_scalars=0,
        n_global_vectors=1,
        n_communication_hyperlayers=n_communication_hyperlayers,
        hidden_layer_sizes=hidden_layer_sizes,
        n_latent_scalars=n_latent_scalars,
        n_latent_vectors=n_latent_vectors,
        n_spherical_harmonics=n_spherical_harmonics,
    ).to(device)

    if dist.rank == 0:
        torchinfo.summary(model, depth=20)
    logger0.info(f"{output_dir.name=!r}")

    base_model = model

    # TODO: candidate for upstreaming to physicsnemo once torch.compiler
    # cache APIs stabilize (currently experimental in PyTorch).
    if use_compile and torch_compile_cache.exists():
        torch.compiler.load_cache_artifacts(torch_compile_cache.read_bytes())

    ### [Distribute the model across GPUs]
    if dist.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=device,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    ### [Compute Maximum Mesh Sizes Per BC Type and Split]
    train_max_sizes = compute_max_mesh_sizes(
        train_dataloader,
        device,
        face_downsampling_ratio=train_face_downsampling_ratio,
        rank=dist.rank,
    )
    valid_max_sizes = compute_max_mesh_sizes(
        valid_dataloader,
        device,
        rank=dist.rank,
    )

    ### [Optimizer and Scheduler Setup]
    learning_rate *= (dist.world_size * points_per_iter / 2048) ** 0.5
    if use_muon:
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
            model.parameters(),  # ty: ignore[unresolved-attribute]
            lr=learning_rate,
            weight_decay=weight_decay,
            decoupled_weight_decay=True,
            foreach=True,
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=400,
        min_lr=learning_rate / 64,
        threshold=0,
    )
    scaler = torch.amp.GradScaler(device=device.type)

    ### [Checkpoint Save/Load]
    mlflow_run_id: str | None = None
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
    if epoch > 0:
        logger0.info(f"Resuming training from epoch {epoch}")
        best_loss = metadata_dict.get("best_loss", float("inf"))
        last_image_epoch = metadata_dict.get("last_image_epoch", -float("inf"))
        last_image_loss = metadata_dict.get("last_image_loss", float("inf"))
        mlflow_run_id = metadata_dict.get("mlflow_run_id")
    else:
        logger0.info("Starting training from scratch.")
        best_loss = float("inf")
        last_image_epoch = -float("inf")
        last_image_loss = float("inf")

    ### [MLflow Setup]
    mlflow_run_ctx: contextlib.AbstractContextManager = contextlib.nullcontext()
    if dist.rank == 0 and use_mlflow:
        mlflow.set_experiment(experiment_name=mlflow_experiment)
        if mlflow_run_id:
            try:
                mlflow_run_ctx = mlflow.start_run(run_id=mlflow_run_id)
                logger0.info(f"Resumed MLflow run {mlflow_run_id}")
            except Exception:
                warnings.warn(
                    f"Could not resume MLflow run {mlflow_run_id!r}, creating new run"
                )
                mlflow_run_id = None
        if not mlflow_run_id:
            mlflow_run_ctx = mlflow.start_run(
                run_name=f"{output_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags={
                    "airfrans_task": airfrans_task,
                    "output_name": output_name,
                },
            )

    ### [Hyperparameter Logging]
    if dist.rank == 0:
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
                "n_train_samples": len(train_sample_paths),
                "n_valid_samples": len(valid_sample_paths),
                "train_sample_paths": train_sample_paths,
                "valid_sample_paths": valid_sample_paths,
            },
        )
        if use_mlflow:
            mlflow.log_artifact(str(output_dir / "hyperparameters.yaml"))

    ### [Training and Validation]
    @torch.compile(
        dynamic=False,
        mode=compile_mode,
        disable=not use_compile,
    )
    def run_batch(sample: AirFRANSSample) -> tuple[torch.Tensor, TensorDict]:
        """Runs a single batch (always just one sample) through the model and computes the loss."""
        pred_results = model(**sample.model_input_kwargs)
        batch_loss_components = pred_results.apply(
            field_loss_fn,
            sample.interior_mesh.point_data,  # ground truth
            error_scales.expand_as(pred_results),
        ).mean(dim=0)  # Mean over points
        batch_loss = batch_loss_components.stack_from_tensordict().sum()
        return batch_loss, batch_loss_components

    def run_epoch(training: bool) -> torch.Tensor:
        """Run one epoch of training or validation. Returns the average total loss across all batches."""
        dataloader = train_dataloader if training else valid_dataloader
        dataloader.sampler.set_epoch(epoch=epoch)  # ty: ignore[unresolved-attribute]
        model.train(training)

        all_batch_losses: list[torch.Tensor] = []
        all_batch_loss_components: dict[str, list[torch.Tensor]] = defaultdict(list)

        for sample in tqdm(
            dataloader,
            desc=f"{epoch:d} {'Train' if training else 'Valid'}",
            unit=" samples",
            disable=dist.rank != 0 or epoch > 10,
        ):
            torch.compiler.cudagraph_mark_step_begin()
            with record_function("data_subsampling"):
                ### Subsample interior points (on CPU to reduce GPU transfer)
                n_points = min(points_per_iter, sample.interior_mesh.n_points)
                mask = torch.randperm(sample.interior_mesh.n_points)[:n_points]
                sample.interior_mesh = sample.interior_mesh.slice_points(mask)

                ### Subsample boundary mesh cells during training
                if training:
                    for bc_type, mesh in sample.boundary_meshes.items():
                        if train_face_downsampling_ratio != 1.0:
                            mesh._cache["cell", "areas"] = (
                                mesh.cell_areas / train_face_downsampling_ratio
                            )
                            new_n_cells = int(
                                mesh.n_cells * train_face_downsampling_ratio
                            )
                            mesh = mesh.slice_cells(
                                torch.randperm(mesh.n_cells)[:new_n_cells]
                            )
                        if train_randomize_face_centers:
                            mesh._cache["cell", "centroids"] = (
                                mesh.sample_random_points_on_cells()
                            )
                        sample.boundary_meshes[bc_type] = mesh

                ### Pad boundary meshes to fixed size for static compilation
                max_sizes = train_max_sizes if training else valid_max_sizes
                for bc_type, mesh in sample.boundary_meshes.items():
                    ### Pre-cache normals before entering torch.compile.
                    # Areas and centroids are already cached above; normals must also
                    # be cached here so the computation+cache-write path is never
                    # traced by Dynamo (avoiding graph breaks and recompilations).
                    _ = mesh.cell_normals
                    sample.boundary_meshes[bc_type] = mesh.pad(
                        target_n_points=max_sizes[bc_type]["n_points"],
                        target_n_cells=max_sizes[bc_type]["n_cells"],
                        data_padding_value=0.0,
                    )

            with record_function("data_transfer"):
                sample = sample.to(device)

            with (
                autocast_ctx,
                contextlib.nullcontext() if training else torch.no_grad(),
                record_function("main_processing_loop"),
            ):
                if training:
                    optimizer.zero_grad()
                batch_loss, batch_loss_components = run_batch(sample)
                if training:
                    if torch.isnan(batch_loss):
                        warnings.warn(
                            f"{batch_loss=} at: {dist.rank=}, {epoch=}, {training=}"
                        )
                    scaler.scale(batch_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                all_batch_losses.append(batch_loss.detach().clone())
                for k, v in batch_loss_components.items():
                    all_batch_loss_components[k].append(v.detach().clone())

        # [Distributed comms]
        keys = ["loss", *all_batch_loss_components.keys()]
        all_values = torch.stack([
            torch.nanmean(torch.stack(all_batch_losses)),
            *(
                torch.nanmean(torch.stack(all_batch_loss_components[k]))
                for k in keys[1:]
            ),
        ])
        if dist.world_size > 1:
            torch.distributed.all_reduce(all_values, op=torch.distributed.ReduceOp.AVG)
        epoch_loss = all_values[0]
        epoch_loss_components = dict(zip(keys[1:], all_values[1:]))

        logger0.info(
            " | ".join(
                [
                    f"{epoch:d=} {'Train' if training else 'Valid'}",
                    f"Loss: {epoch_loss:7.3g}",
                    *[f"{k}: {v:7.3g}" for k, v in epoch_loss_components.items()],
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}",
                ]
            )
        )
        if dist.rank == 0:
            if use_mlflow:
                split = "train" if training else "valid"
                mlflow.log_metrics(
                    {
                        f"{split}_loss": epoch_loss.item(),
                        **{
                            f"{split}_loss_components/{sanitize_metric_name(k)}": v.item()
                            for k, v in epoch_loss_components.items()
                        },
                    },
                    step=epoch,
                )

        return epoch_loss

    ### [Profiler Setup]
    profile = profile and dist.rank == 0 and (not any(profiling_dir.iterdir()))
    profiler_ctx = (
        torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=4, warmup=1, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(profiling_dir), worker_name=f"worker_{dist.rank}"
            ),
            with_stack=True,
        )
        if profile
        else contextlib.nullcontext()
    )
    with mlflow_run_ctx, profiler_ctx as profiler:
        ### [Training Loop]

        if dist.rank == 0:
            time_last_epoch = perf_counter()

        for epoch in count(start=epoch + 1):
            with record_function(f"epoch_{epoch}_train"):
                train_loss = run_epoch(training=True)

            with record_function(f"epoch_{epoch}_valid"):
                valid_loss = run_epoch(training=False)

            scheduler.step(train_loss)

            if profile:
                profiler.step()  # ty: ignore[possibly-missing-attribute]

            ### [Logging and Checkpointing]
            if dist.rank == 0:
                ### [Checkpointing]
                if epoch % (25 * dist.world_size) == 0:
                    save_checkpoint(
                        checkpoint_dir,
                        models=base_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        metadata={
                            "best_loss": best_loss,
                            "last_image_epoch": last_image_epoch,
                            "last_image_loss": last_image_loss,
                            "mlflow_run_id": (
                                mlflow.active_run().info.run_id
                                if use_mlflow and mlflow.active_run()
                                else None
                            ),
                        },
                    )
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    base_model.save(best_model_path)
                    if use_mlflow:
                        mlflow.log_artifact(
                            str(best_model_path), artifact_path="best_model"
                        )

                ### [MLflow Scalars Logging]
                if use_mlflow:
                    time_now = perf_counter()
                    mlflow.log_metrics(
                        {
                            "lr": optimizer.param_groups[0]["lr"],
                            "system/vram_gb": torch.cuda.memory_stats()[
                                "reserved_bytes.all.peak"
                            ]
                            / 1024**3,
                            "system/seconds_per_epoch": time_now - time_last_epoch,
                        },
                        step=epoch,
                    )
                    time_last_epoch = time_now

            ### [MLflow Image Logging]
            if (
                make_images
                and (train_loss / last_image_loss < 0.9)
                and (epoch > last_image_epoch + 200)
            ):
                if dist.rank == 0:
                    logger0.info("Generating visualization images...")
                    for split, paths in [
                        ("train", train_sample_paths),
                        ("valid", valid_sample_paths),
                    ]:
                        sample_path = paths[0]
                        viz_sample = AirFRANSDataSet.preprocess(sample_path).to(device)
                        with torch.no_grad(), autocast_ctx:
                            base_model.eval()
                            pred_results = base_model(
                                **viz_sample.model_input_kwargs,
                                chunk_size=points_per_iter,
                            )
                        AirFRANSDataSet.postprocess(
                            pred_results.to(device="cpu", dtype=torch.float64),
                            sample_path,
                            fields_to_plot="pred",
                            show=False,
                        )
                        plt.tight_layout(h_pad=0.1, w_pad=0)
                        plt.gcf().set_dpi(300)
                        if use_mlflow:
                            mlflow.log_figure(
                                plt.gcf(),
                                f"visualization/{split}_sample_epoch_{epoch}.png",
                            )
                        plt.close()
                last_image_epoch, last_image_loss = epoch, train_loss

            ### [torch.compile Caching]
            if use_compile and not torch_compile_cache.exists():
                artifacts_bytes, cache_info = torch.compiler.save_cache_artifacts()  # ty: ignore[not-iterable]
                torch_compile_cache.write_bytes(artifacts_bytes)
                logger.info(f"Saved torch.compile cache to {torch_compile_cache}.")

            if shutdown_requested():
                logger0.info("Quitting due to shutdown request.")
                if dist.rank == 0:
                    save_checkpoint(
                        checkpoint_dir,
                        models=base_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        metadata={
                            "best_loss": best_loss,
                            "last_image_epoch": last_image_epoch,
                            "last_image_loss": last_image_loss,
                            "mlflow_run_id": (
                                mlflow.active_run().info.run_id
                                if use_mlflow and mlflow.active_run()
                                else None
                            ),
                        },
                    )
                break


def field_loss_fn(
    pred: torch.Tensor, true: torch.Tensor, error_scale: torch.Tensor
) -> torch.Tensor:
    """Per-point Huber loss for GLOBE field predictions, with NaN masking.

    Computes the scaled error ``(pred - true) / error_scale``, masks out
    points where ``true`` is NaN, takes the vector norm for multi-component
    fields, and applies a Huber loss (delta=1) with a factor of 2.

    Args:
        pred: Predicted field values, shape ``(n_points,)`` or ``(n_points, n_dims)``.
        true: Ground-truth field values (same shape). NaN entries are masked.
        error_scale: Per-field scaling factor broadcastable to *pred*.

    Returns:
        Per-point loss tensor of shape ``(n_points,)``.
    """
    error = torch.where(
        torch.isnan(true),
        torch.zeros_like(true),
        (pred - true) / error_scale,
    )
    if error.ndim > 1:
        error = error.norm(dim=-1)
    return 2 * F.huber_loss(error, torch.zeros_like(error), reduction="none", delta=1.0)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
