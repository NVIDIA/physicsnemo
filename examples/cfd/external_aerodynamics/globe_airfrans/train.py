# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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
import json
import signal
from collections import defaultdict
from datetime import datetime
from itertools import count
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn.functional as F
import torchinfo
from tensordict import TensorDict
from torch.profiler import record_function
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from physicsnemo.distributed import DistributedManager
from physicsnemo.mesh.utilities._cache import set_cached
from combined_optimizer import CombinedOptimizer
from config import get_data_dir
from dataset import AirFRANSDataSet
from utilities import (
    get_latest_checkpoint_path,
    get_physicsnemo_pkg_info,
    log_hyperparameters,
    reduce_over_ranks,
    sanitize_metric_name,
    to,
    disable_autotune_printing,
)
from physicsnemo.models.globe.model import GLOBE

mpl.use("agg")  # Allows headless plotting
disable_autotune_printing()  # Silences the verbose output of `torch.compile(..., mode="max-autotune")`.


def main(
    output_name: str | None = None,
    amp: bool = False,
    compile: bool = True,
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
):
    """Train the GLOBE model on AirFRANS dataset.

    Args:
        output_name: Name for output directory. If None, uses current timestamp.
        amp: Enable automatic mixed precision (AMP) training for faster computation.
        compile: Enable torch.compile for model optimization and performance.
        compile_mode: Mode for torch.compile.
        points_per_iter: Number of points to sample per training iteration.
        learning_rate: Initial learning rate for the Adam optimizer.
        weight_decay: Weight decay (L2 regularization) factor for the optimizer.
        train_face_downsampling_ratio: Ratio of faces to keep when downsampling boundary meshes.
        train_random_face_centers: Whether to use random points inside faces instead of centroids.
        train_random_face_centers_alpha: Concentration parameter for Dirichlet distribution used to
            generate random face centers. Alpha=1 gives uniform distribution over the simplex.
            Larger values (e.g., alpha=3) concentrate samples toward the center of each face.
        seed: Random seed for reproducibility across runs.
        error_scales: Dictionary specifying error scales for loss components. If None, uses default scales.
        hidden_layer_sizes: List of hidden layer sizes for the MLP architecture.
        bc_encoding_n_scalars: Number of scalar features in boundary condition encoding.
        bc_encoding_n_vectors: Number of vector features in boundary condition encoding.
        airfrans_task: Which AirFRANS dataset task to train on.
        profile: Enable PyTorch profiler for performance analysis.
        make_images: Whether to make images for visualization.

    Note:
        Data directory is automatically determined based on hostname (local vs. EOS cluster).
        Output directory is created under the script's parent directory in an 'output' folder.
        Error scales control the relative weighting of different physical fields in the loss.
        When profiling is enabled, results are saved to output_dir/profiling/ as Chrome trace files.
    """
    ### [Config Processing]
    data_dir = get_data_dir()

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
    if dist.rank == 0:
        print(f"{dist.world_size=} 🌎")
    print(f"Howdy from {dist.rank=}! 🤠")

    error_scales: TensorDict = TensorDict(error_scales, device=device)
    if dist.rank == 0:
        torch._logging.set_logs(graph_breaks=True, recompiles=True)

    ### [Output Directory Setup]
    torch_compile_cache_dir = output_dir / "torch_compile_cache"
    torch_compile_cache = torch_compile_cache_dir / f"rank_{dist.rank}.compile_cache"
    models_dir = output_dir / "models"
    best_model_dir = models_dir / "best_model"
    profiling_dir = output_dir / "profiling"
    shared_mlflow_dir = Path(__file__).parent / "output" / "mlruns"

    if dist.rank == 0:
        for directory in (
            models_dir,
            best_model_dir,
            torch_compile_cache_dir,
            profiling_dir,
            shared_mlflow_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        ### [MLflow Setup]
        mlflow.set_tracking_uri(f"file://{shared_mlflow_dir.absolute()}")
        mlflow.set_experiment(experiment_name="GLOBE_AirFRANS")
        mlflow.start_run(
            run_name=f"{output_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={
                "airfrans_task": airfrans_task,
                "output_name": output_name,
            },
        )

    ### [Signal Handling]
    shutdown_received = False

    def _handle_signal(signum: int, frame) -> None:
        nonlocal shutdown_received
        if dist.rank == 0:
            print(f"{signal.Signals(signum).name} received; quitting after this epoch.")
        shutdown_received = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGQUIT, _handle_signal)

    ### [PyTorch Configuration]
    autocast_ctx = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=amp
    )
    torch.cuda.set_per_process_memory_fraction(0.99)
    torch.set_float32_matmul_precision("high")  # Allows use of Tensor Cores in matmuls
    torch.manual_seed(seed)

    ### [Dataset Preparation]
    manifest = json.loads((data_dir / "manifest.json").read_text())
    train_sample_paths = [data_dir / f for f in manifest[f"{airfrans_task}_train"]]
    valid_taskname = "full" if airfrans_task == "scarce" else airfrans_task
    valid_sample_paths = [data_dir / f for f in manifest[f"{valid_taskname}_test"]]

    ### [DataLoader Creation]
    def make_dataloader(sample_paths: list[Path], num_workers: int) -> DataLoader:
        dataset = AirFRANSDataSet(
            sample_paths=sample_paths,
            cache_dir=cache_dir,
        )
        return DataLoader(
            dataset,
            sampler=DistributedSampler(
                dataset=dataset, num_replicas=dist.world_size, rank=dist.rank
            ),
            batch_size=None,
            collate_fn=lambda x: x,
            num_workers=num_workers,
            prefetch_factor=32 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        )

    train_dataloader = make_dataloader(train_sample_paths, num_workers=8)
    valid_dataloader = make_dataloader(valid_sample_paths, num_workers=8)

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
        print(f"{output_dir.name=!r}")

    base_model = model

    if compile and torch_compile_cache.exists():
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
    def compute_max_mesh_sizes_distributed(
        training: bool,
    ) -> dict[str, dict[str, int]]:
        """Compute max n_points and n_cells per BC type using distributed dataloader.

        Each rank processes its subset of data on the (train/validation) split,
        then all_reduce to get global max.
        """
        # Dictionary mapping BC type to max sizes: {bc_type: {n_points: int, n_cells: int}}
        max_sizes = defaultdict(lambda: {"n_points": 0, "n_cells": 0})

        for input_dict, _ in tqdm(
            train_dataloader if training else valid_dataloader,
            desc=f"Computing max mesh sizes on {'Train' if training else 'Valid'} data (rank {dist.rank})",
            disable=dist.rank != 0,
        ):
            for bc_type, mesh in input_dict["boundary_meshes"].items():
                max_sizes[bc_type]["n_points"] = max(
                    max_sizes[bc_type]["n_points"], mesh.n_points
                )
                if training and train_face_downsampling_ratio != 1.0:
                    n_cells = int(mesh.n_cells * train_face_downsampling_ratio)
                else:
                    n_cells = mesh.n_cells
                max_sizes[bc_type]["n_cells"] = max(
                    max_sizes[bc_type]["n_cells"],
                    n_cells,
                )

        # Reduce across all ranks to get global max
        for bc_type in max_sizes.keys():
            size_tensor = torch.tensor(
                [max_sizes[bc_type]["n_points"], max_sizes[bc_type]["n_cells"]],
                device=device,
            )
            size_tensor = reduce_over_ranks(size_tensor, op="max")
            max_sizes[bc_type]["n_points"] = int(size_tensor[0])
            max_sizes[bc_type]["n_cells"] = int(size_tensor[1])

        if dist.rank == 0:
            print(
                f"Max mesh sizes on {'Train' if training else 'Valid'} data (rank {dist.rank}): {dict(max_sizes)}"
            )

        return dict(max_sizes)

    train_max_sizes = compute_max_mesh_sizes_distributed(training=True)
    valid_max_sizes = compute_max_mesh_sizes_distributed(training=False)

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
    def save_checkpoint(save_dir: Path, keep_only_latest: bool = False) -> None:
        checkpoint_path = save_dir / f"{base_model.__class__.__name__}.{epoch:d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_loss": best_loss,
                "last_image_epoch": last_image_epoch,
                "last_image_loss": last_image_loss,
            },
            checkpoint_path,
        )
        if keep_only_latest:
            for old_checkpoint in save_dir.glob("*.pt"):
                if old_checkpoint != checkpoint_path:
                    old_checkpoint.unlink()

    # Try to load a pre-existing checkpoint
    previous_checkpoint: Path | None = get_latest_checkpoint_path(output_dir=output_dir)
    if previous_checkpoint:
        if dist.rank == 0:
            print(f"Resuming training from checkpoint: {previous_checkpoint.name!r}")
        checkpoint = torch.load(previous_checkpoint, map_location=device)
        epoch: int = checkpoint["epoch"]
        base_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        best_loss = checkpoint.get("best_loss", float("inf"))
        last_image_epoch = checkpoint.get("last_image_epoch", -float("inf"))
        last_image_loss = checkpoint.get("last_image_loss", float("inf"))
    else:
        if dist.rank == 0:
            print("Starting training from scratch.")
        epoch = 0
        best_loss = float("inf")
        last_image_epoch = -float("inf")
        last_image_loss = float("inf")

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

    ### [Training and Validation]
    def field_loss_fn(
        pred: torch.Tensor, true: torch.Tensor, error_scale: torch.Tensor
    ) -> torch.Tensor:
        error = torch.where(
            torch.isnan(true),
            torch.zeros_like(true),
            (pred - true) / error_scale,
        )
        if error.ndim > 1:
            error = error.norm(dim=-1)
        return 2 * F.huber_loss(
            error, torch.zeros_like(error), reduction="none", delta=1.0
        )

    @torch.compile(
        dynamic=False,
        mode=compile_mode,
        disable=not compile,
    )
    def run_batch(
        input_dict: TensorDict, true_results: TensorDict
    ) -> tuple[torch.Tensor, TensorDict]:
        """Runs a single batch (always just one sample) through the model and computes the loss."""
        pred_results = model(
            prediction_points=input_dict["prediction_points"],
            boundary_meshes=input_dict["boundary_meshes"],  # type: dict[str, Mesh]
            reference_lengths=input_dict["reference_lengths"],  # type: dict[str, torch.Tensor]
            global_scalars=input_dict["global_scalars"],  # type: TensorDict
            global_vectors=input_dict["global_vectors"],  # type: TensorDict
            chunk_size=None,
            verbose=False,
        )
        batch_loss_components = pred_results.apply(
            field_loss_fn,
            true_results,
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

        for input_dict, true_results in tqdm(
            dataloader,
            desc=f"{epoch:d} {'Train' if training else 'Valid'}",
            unit=" samples",
            disable=dist.rank != 0 or epoch > 10,
        ):
            torch.compiler.cudagraph_mark_step_begin()
            with record_function("data_transfer"):
                input_dict: dict[str, Any] = to(input_dict, device=device)
                true_results: TensorDict = to(true_results, device=device)

            with record_function("data_subsampling"):
                # Subsample points for this iteration
                prediction_points = input_dict["prediction_points"]
                n_points = min(points_per_iter, len(prediction_points))
                mask = torch.randperm(len(prediction_points), device=device)[:n_points]
                input_dict["prediction_points"] = prediction_points[mask]
                true_results = true_results[mask]

                ### Subsample boundary mesh cells during training
                if training:
                    for bc_type, mesh in input_dict["boundary_meshes"].items():
                        if train_face_downsampling_ratio != 1.0:
                            set_cached(
                                mesh.cell_data,
                                "areas",
                                mesh.cell_areas / train_face_downsampling_ratio,
                            )
                            new_n_cells = int(
                                mesh.n_cells * train_face_downsampling_ratio
                            )
                            mesh = mesh.slice_cells(
                                torch.randperm(mesh.n_cells, device=device)[
                                    :new_n_cells
                                ]
                            )
                        if train_randomize_face_centers:
                            set_cached(
                                mesh.cell_data,
                                "centroids",
                                mesh.sample_random_points_on_cells(),
                            )
                        input_dict["boundary_meshes"][bc_type] = mesh

                ### Pad boundary meshes to fixed size for static compilation
                max_sizes = train_max_sizes if training else valid_max_sizes
                for bc_type, mesh in input_dict["boundary_meshes"].items():
                    ### Pre-cache normals before entering torch.compile.
                    # Areas and centroids are already cached above; normals must also
                    # be cached here so the computation+cache-write path is never
                    # traced by Dynamo (avoiding graph breaks and recompilations).
                    _ = mesh.cell_normals
                    input_dict["boundary_meshes"][bc_type] = mesh.pad(
                        target_n_points=max_sizes[bc_type]["n_points"],
                        target_n_cells=max_sizes[bc_type]["n_cells"],
                        data_padding_value=0.0,
                    )

            with (
                autocast_ctx,
                contextlib.nullcontext() if training else torch.no_grad(),
                record_function("main_processing_loop"),
            ):
                if training:
                    optimizer.zero_grad()
                batch_loss, batch_loss_components = run_batch(
                    input_dict=input_dict, true_results=true_results
                )
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

        epoch_loss = reduce_over_ranks(
            torch.nanmean(torch.stack(all_batch_losses)), op="mean"
        )
        epoch_loss_components = {
            k: reduce_over_ranks(torch.nanmean(torch.stack(v)), op="mean")
            for k, v in all_batch_loss_components.items()
        }

        if dist.rank == 0:  # Logging on rank 0 only
            print(
                " | ".join(
                    [
                        f"{epoch:d=} {'Train' if training else 'Valid'}",
                        f"Loss: {epoch_loss:7.3g}",
                        *[f"{k}: {v:7.3g}" for k, v in epoch_loss_components.items()],
                        f"LR: {optimizer.param_groups[0]['lr']:.2e}",
                    ]
                )
            )
            split = "train" if training else "valid"
            mlflow.log_metric(f"{split}_loss", epoch_loss.item(), step=epoch)
            for field_name, loss_value in epoch_loss_components.items():
                mlflow.log_metric(
                    f"{split}_loss_components/{sanitize_metric_name(field_name)}",
                    loss_value.item(),
                    step=epoch,
                )

        return epoch_loss

    ### [Profiler Setup]
    profile = profile and dist.rank == 0 and (not any(profiling_dir.iterdir()))
    with (
        torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=4, warmup=1, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(profiling_dir), worker_name=f"worker_{dist.rank}"
            ),
            with_stack=True,
        )
        if profile
        else contextlib.nullcontext()
    ) as profiler:
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
                    save_checkpoint(save_dir=models_dir, keep_only_latest=True)
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    save_checkpoint(save_dir=best_model_dir, keep_only_latest=True)

                ### [MLflow Scalars Logging]
                mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)
                mlflow.log_metric(
                    "system/vram_gb",
                    torch.cuda.memory_stats()["reserved_bytes.all.peak"] / 1024**3,
                    step=epoch,
                )
                time_now = perf_counter()
                mlflow.log_metric(
                    "system/seconds_per_epoch",
                    (time_now - time_last_epoch),
                    step=epoch,
                )
                time_last_epoch = time_now

            ### [MLflow Image Logging]
            if (
                make_images
                and (train_loss / last_image_loss < 0.9)
                and (epoch > last_image_epoch + 200)
            ):

                def log_images(training: bool) -> None:
                    sample_path = (
                        train_sample_paths if training else valid_sample_paths
                    )[0]
                    input_dict, _ = AirFRANSDataSet.preprocess(sample_path)
                    input_dict = to(input_dict, device=device)

                    with torch.no_grad(), autocast_ctx:
                        base_model.eval()
                        pred_results = base_model(
                            prediction_points=input_dict["prediction_points"],
                            boundary_meshes=input_dict["boundary_meshes"],
                            reference_lengths=input_dict["reference_lengths"],
                            global_scalars=input_dict["global_scalars"],
                            global_vectors=input_dict["global_vectors"],
                            chunk_size=points_per_iter,
                            verbose=False,
                        )

                    AirFRANSDataSet.postprocess(
                        to(
                            pred_results,
                            device=torch.device("cpu"),
                            dtype=torch.float64,
                        ),
                        sample_path,
                        fields_to_plot="pred",
                        show=False,
                    )
                    plt.tight_layout(h_pad=0.1, w_pad=0)
                    plt.gcf().set_dpi(300)

                    # Log figure to MLflow (only on rank 0 where MLflow is initialized)
                    if dist.rank == 0:
                        split = "train" if training else "valid"
                        mlflow.log_figure(
                            plt.gcf(),
                            f"visualization/{split}_sample_epoch_{epoch}.png",
                        )
                    plt.close()

                if dist.rank == 0:
                    print("Generating visualization images...")
                    log_images(training=True)
                    log_images(training=False)
                last_image_epoch, last_image_loss = epoch, train_loss

            ### [torch.compile Caching]
            if compile and not torch_compile_cache.exists():
                artifacts_bytes, cache_info = torch.compiler.save_cache_artifacts()  # ty: ignore[not-iterable]
                torch_compile_cache.write_bytes(artifacts_bytes)
                print(f"Saved torch.compile cache to {torch_compile_cache}.")

            if shutdown_received:
                if dist.rank == 0:
                    print("Quitting due to shutdown request.")
                    save_checkpoint(save_dir=models_dir, keep_only_latest=False)
                    mlflow.end_run()
                break


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
