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

import numpy as np
import torch
import torchinfo

import hydra
from omegaconf import DictConfig
from physicsnemo.models.transolver.transolver import Transolver
from physicsnemo.launch.utils import load_checkpoint
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper

from physicsnemo.distributed import DistributedManager

import time

from datapipe import DomainParallelZarrDataset

from train import forward_pass
from tabulate import tabulate

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
from torch.amp import autocast
from contextlib import nullcontext

from train import (
    get_autocast_context,
    pad_input_for_fp8,
    unpad_output_for_fp8,
    update_model_params_for_fp8
)




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

    # Set up model
    model = hydra.utils.instantiate(cfg.model)
    logger.info(f"\n{torchinfo.summary(model, verbose=0)}")
    model.eval()
    model.to(dist_manager.device)
    
    if cfg.training.compile:
        model = torch.compile(model)

    # Validation dataset

    val_dataset = DomainParallelZarrDataset(
        data_path=cfg.data.val.data_path,  # Assuming validation data path is configured
        device_mesh=None,
        placements=None,
        max_workers=cfg.data.max_workers,
        pin_memory=cfg.data.pin_memory,
        keys_to_read=cfg.data.data_keys,
        large_keys=cfg.data.large_keys,
    )

    ckpt_args = {
        "path": f"{cfg.output_dir}/{cfg.run_id}/checkpoints",
        "models": model,
    }

    # Load the normalization factors:
    norm_file = "surface_fields_normalization.npz"
    norm_data = np.load(norm_file)
    norm_factors = {
        "mean": torch.from_numpy(norm_data["mean"]).to(dist_manager.device),
        "std": torch.from_numpy(norm_data["std"]).to(dist_manager.device),
    }

    loaded_epoch = load_checkpoint(device=dist_manager.device, **ckpt_args)
    logger.info(f"loaded epoch: {loaded_epoch}")

    results = []
    start = time.time()
    for batch_idx in range(len(val_dataset)):

        batch = val_dataset[batch_idx]
        with torch.no_grad():
            loss, metrics = forward_pass(
                batch,
                model,
                cfg.training.precision,
                output_pad_size,
                dist_manager,
                cfg,
                norm_factors,
            )

        # Extract metric values and convert tensors to floats
        l2_pressure = (
            metrics["l2_pressure"].item()
            if hasattr(metrics["l2_pressure"], "item")
            else metrics["l2_pressure"]
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

        end = time.time()
        elapsed = end - start
        logger.info(f"Finished batch {batch_idx} in {elapsed:.4f} seconds")
        results.append(
            [
                batch_idx,
                f"{loss:.4f}",
                f"{l2_pressure:.4f}",
                f"{l2_shear_x:.4f}",
                f"{l2_shear_y:.4f}",
                f"{l2_shear_z:.4f}",
                f"{elapsed:.4f}",
            ]
        )

        start = time.time()

    headers = [
        "Batch",
        "Loss",
        "L2 Pressure",
        "L2 Shear X",
        "L2 Shear Y",
        "L2 Shear Z",
        "Elapsed (s)",
    ]
    logger.info(f"Results:\n{tabulate(results, headers=headers, tablefmt='github')}")

    # Calculate means for each metric (skip batch index)
    if results:
        # Convert string values back to float for mean calculation
        arr = np.array(results)[:, 1:].astype(float)
        means = arr.mean(axis=0)
        mean_row = ["Mean"] + [f"{m:.4f}" for m in means]
        logger.info(f"Summary:\n{tabulate([mean_row], headers=headers, tablefmt='github')}")


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