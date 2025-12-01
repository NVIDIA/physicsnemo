# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

from pathlib import Path

import numpy as np
import torch
import torchinfo
import typing
import collections
from typing import Literal

import hydra
import omegaconf
from omegaconf import DictConfig
from physicsnemo.models.transolver.transolver import Transolver
from physicsnemo.launch.utils import load_checkpoint
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper

from sklearn.metrics import r2_score

from physicsnemo.distributed import DistributedManager

import time

from physicsnemo.datapipes.cae.transolver_datapipe import (
    create_transolver_dataset,
    TransolverDataPipe,
)
from train import forward_pass
from tabulate import tabulate

# import transformer_engine.pytorch as te
# from transformer_engine.common.recipe import Format, DelayedScaling
from torch.amp import autocast
from contextlib import nullcontext

from train import (
    get_autocast_context,
    pad_input_for_fp8,
    unpad_output_for_fp8,
    update_model_params_for_fp8,
)

# torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
# torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
# torch.serialization.add_safe_globals([typing.Any])
# torch.serialization.add_safe_globals([list])
# torch.serialization.add_safe_globals([collections.defaultdict])
# torch.serialization.add_safe_globals([dict])
# torch.serialization.add_safe_globals([int])
# torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
# torch.serialization.add_safe_globals([omegaconf.base.Metadata])


@torch.no_grad()
def compute_force_coefficients(
    normals: torch.Tensor,
    area: torch.Tensor,
    coeff: float,
    p: torch.Tensor,
    wss: torch.Tensor,
    force_direction: torch.Tensor = np.array([1, 0, 0]),
):
    """
    Computes force coefficients for a given mesh. Output includes the pressure and skin
    friction components. Can be used to compute lift and drag.
    For drag, use the `force_direction` as the direction of the motion,
    e.g. [1, 0, 0] for flow in x direction.
    For lift, use the `force_direction` as the direction perpendicular to the motion,
    e.g. [0, 1, 0] for flow in x direction and weight in y direction.

    Parameters:
    -----------
    normals: torch.Tensor
        The surface normals on cells of the mesh
    area: torch.Tensor
        The surface areas of each cell
    coeff: float
        Reciprocal of dynamic pressure times the frontal area, i.e. 2/(A * rho * U^2)
    p: torch.Tensor
        Pressure distribution on the mesh (on each cell)
    wss: torch.Tensor
        Wall shear stress distribution on the mesh (on each cell)
    force_direction: torch.Tensor
        Direction to compute the force, default is np.array([1, 0, 0])

    Returns:
    --------
    c_total: float
        Computed total force coefficient
    c_p: float
        Computed pressure force coefficient
    c_f: float
        Computed skin friction coefficient
    """

    # Compute coefficients
    c_p = coeff * torch.sum(torch.sum(normals * force_direction, dim=-1) * area * p)
    c_f = -coeff * torch.sum(torch.sum(wss * force_direction, dim=-1) * area)

    # Compute total force coefficients
    c_total = c_p + c_f

    return c_total, c_p, c_f


def batched_inference_loop(
    batch: dict,
    model: torch.nn.Module,
    precision: str,
    data_mode: Literal["surface", "volume"],
    batch_resolution: int,
    output_pad_size: int | None,
    dist_manager: DistributedManager,
    datapipe: TransolverDataPipe,
) -> tuple[float, dict, tuple[torch.Tensor, torch.Tensor]]:
    N = batch["embeddings"].shape[1]
    # This generates a random ordering of the input points,
    # Which we'll then slice up into inputs to the model.
    indices = torch.randperm(N, device=batch["fx"].device)

    index_blocks = torch.split(indices, batch_resolution)

    global_preds_targets = []
    global_weight = 0.0
    start = time.time()
    for i, index_block in enumerate(index_blocks):
        # We compute the local_batch by slicing from embeddings and fields:
        local_embeddings = batch["embeddings"][:, index_block]
        local_fields = batch["fields"][:, index_block]

        # fx does not need to be sliced for TransolverX:
        if "geometry" not in batch.keys():
            local_fx = batch["fx"][:, index_block]
        else:
            local_fx = batch["fx"]

        local_batch = {
            "fx": local_fx,
            "embeddings": local_embeddings,
            "fields": local_fields,
        }

        if "air_density" in batch.keys() and "stream_velocity" in batch.keys():
            local_batch["air_density"] = batch["air_density"]
            local_batch["stream_velocity"] = batch["stream_velocity"]

        if "geometry" in batch.keys():
            local_batch["geometry"] = batch["geometry"]

        # Run the forward inference pass:
        local_loss, local_metrics, local_preds_targets = forward_pass(
            local_batch,
            model,
            precision,
            output_pad_size,
            dist_manager,
            data_mode,
            datapipe,
        )

        # Accumulate the loss and metrics:
        # (Still on the GPU)
        weight = index_block.shape[0] / N
        global_weight += weight
        if i == 0:
            metrics = {k: local_metrics[k] * weight for k in local_metrics.keys()}
            loss = local_loss * weight
        else:
            metrics = {
                k: metrics[k] + local_metrics[k] * weight for k in metrics.keys()
            }
            loss += local_loss * weight

        global_preds_targets.append(local_preds_targets)

        end = time.time()
        elapsed = end - start
        print(
            f"Completed sub-batch {i} of {len(index_blocks)} in {elapsed:.4f} seconds"
        )
        start = end

    # Now, compute the overall loss, metrics, and coefficients:
    metrics = {k: v / global_weight for k, v in metrics.items()}
    loss = loss / global_weight

    global_predictions = torch.cat([l[0] for l in global_preds_targets], dim=1)
    global_targets = torch.cat([l[1] for l in global_preds_targets], dim=1)

    # Now, we have to *unshuffle* the prediction to the original index
    inverse_indices = torch.empty_like(indices)
    inverse_indices[indices] = torch.arange(indices.size(0), device=indices.device)
    # Suppose prediction is of shape [batch, N, ...]
    global_predictions = global_predictions[:, inverse_indices]
    global_targets = global_targets[:, inverse_indices]
    return loss, metrics, (global_predictions, global_targets)


def inference(cfg: DictConfig) -> None:
    """
    Run inference on a validation Zarr dataset using a trained Transolver model.

    Args:
        cfg (DictConfig): Hydra configuration object containing model, data, and training settings.

    Returns:
        None
    """
    DistributedManager.initialize()

    dist_manager = DistributedManager()

    logger = RankZeroLoggingWrapper(PythonLogger(name="training"), dist_manager)

    cfg, output_pad_size = update_model_params_for_fp8(cfg, logger)

    logger.info(f"Config:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")

    # Set up model
    model = hydra.utils.instantiate(cfg.model)
    logger.info(f"\n{torchinfo.summary(model, verbose=0)}")

    if cfg.checkpoint_dir is not None:
        checkpoint_dir = cfg.checkpoint_dir
    else:
        checkpoint_dir = f"{cfg.output_dir}/{cfg.run_id}/checkpoints"

    ckpt_args = {
        "path": checkpoint_dir,
        "models": model,
    }

    loaded_epoch = load_checkpoint(device=dist_manager.device, **ckpt_args)
    logger.info(f"loaded epoch: {loaded_epoch}")
    model.to(dist_manager.device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {num_params}")

    # Load the normalization file from configured directory (defaults to current dir)
    norm_dir = getattr(cfg.data, "normalization_dir", ".")
    if cfg.data.mode == "surface":
        norm_file = str(Path(norm_dir) / "surface_fields_normalization.npz")
    elif cfg.data.mode == "volume":
        norm_file = str(Path(norm_dir) / "volume_fields_normalization.npz")

    norm_data = np.load(norm_file)
    norm_factors = {
        "mean": torch.from_numpy(norm_data["mean"]).to(dist_manager.device),
        "std": torch.from_numpy(norm_data["std"]).to(dist_manager.device),
    }

    if cfg.compile:
        model = torch.compile(model, dynamic=True)
    model.eval()

    # For INFERENCE, we deliberately set the resolution in the data pipe to NONE
    # so there is not downsampling.  We still batch it in the inference script
    # for memory usage constraints.

    batch_resolution = cfg.data.resolution
    cfg.data.resolution = None
    ## Make sure to read the whole data sample for volume:
    if cfg.data.mode == "volume":
        cfg.data.volume_sample_from_disk = False

    # And we need the mesh features for drag, lift in surface data:
    if cfg.data.mode == "surface":
        cfg.data.return_mesh_features = True

    # Validation dataset
    val_dataset = create_transolver_dataset(
        cfg.data,
        phase="val",
        scaling_factors=norm_factors,
    )

    results = []
    start = time.time()
    for batch_idx, batch in enumerate(val_dataset):
        with torch.no_grad():
            loss, metrics, (global_predictions, global_targets) = (
                batched_inference_loop(
                    batch,
                    model,
                    cfg.precision,
                    cfg.data.mode,
                    batch_resolution,
                    output_pad_size,
                    dist_manager,
                    val_dataset,
                )
            )
        end = time.time()
        elapsed = end - start
        logger.info(f"Finished batch {batch_idx} in {elapsed:.4f} seconds")
        start = time.time()

        if cfg.data.mode == "surface":
            coeff = 1.0

            # Compute the drag and loss coefficients:
            # (Index on [0] is to remove the 1 batch index)
            pred_pressure, pred_shear = torch.split(
                global_predictions[0], (1, 3), dim=-1
            )

            pred_pressure = pred_pressure.reshape(-1)
            pred_drag_coeff, _, _ = compute_force_coefficients(
                batch["surface_normals"][0],
                batch["surface_areas"],
                coeff,
                pred_pressure,
                pred_shear,
                torch.tensor([[1, 0, 0]], device=dist_manager.device),
            )

            pred_lift_coeff, _, _ = compute_force_coefficients(
                batch["surface_normals"][0],
                batch["surface_areas"],
                coeff,
                pred_pressure,
                pred_shear,
                torch.tensor([[0, 0, 1]], device=dist_manager.device),
            )

            # air_density = batch["air_density"] if "air_density" in batch.keys() else None
            # stream_velocity = batch["stream_velocity"] if "stream_velocity" in batch.keys() else None
            # true_fields = val_dataset.unscale_model_targets(batch["fields"], air_density=air_density, stream_velocity=stream_velocity)
            true_pressure, true_shear = torch.split(global_targets[0], (1, 3), dim=-1)

            true_pressure = true_pressure.reshape(-1)
            true_drag_coeff, _, _ = compute_force_coefficients(
                batch["surface_normals"][0],
                batch["surface_areas"],
                coeff,
                true_pressure,
                true_shear,
                torch.tensor([[1, 0, 0]], device=dist_manager.device),
            )

            true_lift_coeff, _, _ = compute_force_coefficients(
                batch["surface_normals"][0],
                batch["surface_areas"],
                coeff,
                true_pressure,
                true_shear,
                torch.tensor([[0, 0, 1]], device=dist_manager.device),
            )

            pred_lift_coeff = pred_lift_coeff.item()
            pred_drag_coeff = pred_drag_coeff.item()

            # Extract metric values and convert tensors to floats
            l2_pressure = (
                metrics["l2_pressure_surf"].item()
                if hasattr(metrics["l2_pressure_surf"], "item")
                else metrics["l2_pressure_surf"]
            )
            l2_shear_x = (
                metrics["l2_shear_x"].item()
                if hasattr(metrics["l2_shear_x"], "item")
                else metrics["l2_shear_x"]
            )
            l2_shear_y = (
                metrics["l2_shear_y"].item()
                if hasattr(metrics["l2_shear_y"], "item")
                else metrics["l2_shear_y"]
            )
            l2_shear_z = (
                metrics["l2_shear_z"].item()
                if hasattr(metrics["l2_shear_z"], "item")
                else metrics["l2_shear_z"]
            )

            results.append(
                [
                    batch_idx,
                    f"{loss:.4f}",
                    f"{l2_pressure:.4f}",
                    f"{l2_shear_x:.4f}",
                    f"{l2_shear_y:.4f}",
                    f"{l2_shear_z:.4f}",
                    f"{pred_drag_coeff:.4f}",
                    f"{pred_lift_coeff:.4f}",
                    f"{true_drag_coeff:.4f}",
                    f"{true_lift_coeff:.4f}",
                    f"{elapsed:.4f}",
                ]
            )

        elif cfg.data.mode == "volume":
            # Extract metric values and convert tensors to floats
            l2_pressure = (
                metrics["l2_pressure_vol"].item()
                if hasattr(metrics["l2_pressure_vol"], "item")
                else metrics["l2_pressure_vol"]
            )
            l2_velocity_x = (
                metrics["l2_velocity_x"].item()
                if hasattr(metrics["l2_velocity_x"], "item")
                else metrics["l2_velocity_x"]
            )
            l2_velocity_y = (
                metrics["l2_velocity_y"].item()
                if hasattr(metrics["l2_velocity_y"], "item")
                else metrics["l2_velocity_y"]
            )
            l2_velocity_z = (
                metrics["l2_velocity_z"].item()
                if hasattr(metrics["l2_velocity_z"], "item")
                else metrics["l2_velocity_z"]
            )
            l2_nut = (
                metrics["l2_nut"].item()
                if hasattr(metrics["l2_nut"], "item")
                else metrics["l2_nut"]
            )

            results.append(
                [
                    batch_idx,
                    f"{loss:.4f}",
                    f"{l2_pressure:.4f}",
                    f"{l2_velocity_x:.4f}",
                    f"{l2_velocity_y:.4f}",
                    f"{l2_velocity_z:.4f}",
                    f"{l2_nut:.4f}",
                    f"{elapsed:.4f}",
                ]
            )

    if cfg.data.mode == "surface":
        pred_drag_coeffs = [r[6] for r in results]
        pred_lift_coeffs = [r[7] for r in results]
        true_drag_coeffs = [r[8] for r in results]
        true_lift_coeffs = [r[9] for r in results]

        # Compute the R2 scores for lift and drag:
        r2_lift = r2_score(true_lift_coeffs, pred_lift_coeffs)
        r2_drag = r2_score(true_drag_coeffs, pred_drag_coeffs)

        headers = [
            "Batch",
            "Loss",
            "L2 Pressure",
            "L2 Shear X",
            "L2 Shear Y",
            "L2 Shear Z",
            "Predicted Drag Coefficient",
            "Pred Lift Coefficient",
            "True Drag Coefficient",
            "True Lift Coefficient",
            "Elapsed (s)",
        ]
        logger.info(
            f"Results:\n{tabulate(results, headers=headers, tablefmt='github')}"
        )
        logger.info(f"R2 score for lift: {r2_lift:.4f}")
        logger.info(f"R2 score for drag: {r2_drag:.4f}")

    elif cfg.data.mode == "volume":
        headers = [
            "Batch",
            "Loss",
            "L2 Pressure",
            "L2 Velocity X",
            "L2 Velocity Y",
            "L2 Velocity Z",
            "L2 Nut",
            "Elapsed (s)",
        ]
        logger.info(
            f"Results:\n{tabulate(results, headers=headers, tablefmt='github')}"
        )

    # Calculate means for each metric (skip batch index)
    if results:
        # Convert string values back to float for mean calculation
        arr = np.array(results)[:, 1:].astype(float)
        means = arr.mean(axis=0)
        mean_row = ["Mean"] + [f"{m:.4f}" for m in means]
        logger.info(
            f"Summary:\n{tabulate([mean_row], headers=headers, tablefmt='github')}"
        )


@hydra.main(version_base=None, config_path="conf", config_name="train_surface")
def launch(cfg: DictConfig) -> None:
    """
    Launch inference with Hydra configuration.

    Args:
        cfg (DictConfig): Hydra configuration object.

    Returns:
        None
    """
    inference(cfg)


if __name__ == "__main__":
    launch()
