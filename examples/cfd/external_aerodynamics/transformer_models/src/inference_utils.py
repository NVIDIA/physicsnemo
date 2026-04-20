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

"""Shared inference utilities for Transolver/GeoTransolver models.

Provides batched inference, MC-Dropout inference, and MC-Dropout model setup
used by both zarr and VTK inference scripts.
"""

import time
from typing import Literal

import torch

from physicsnemo.distributed import DistributedManager
from physicsnemo.datapipes.cae.transolver_datapipe import TransolverDataPipe
from physicsnemo.nn import ConcreteDropout, get_concrete_dropout_rates

from train import forward_pass


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
    """Run inference in sub-batches to manage memory.

    Splits the input points into blocks of ``batch_resolution``, runs
    ``forward_pass`` on each block, and reassembles the predictions in
    the original point order.

    Parameters
    ----------
    batch : dict
        Input batch dictionary with keys ``embeddings``, ``fields``, ``fx``,
        and optionally ``geometry``, ``air_density``, ``stream_velocity``.
    model : torch.nn.Module
        Trained model.
    precision : str
        Precision setting (e.g. ``"fp32"``, ``"fp16"``).
    data_mode : Literal["surface", "volume"]
        Data mode.
    batch_resolution : int
        Number of points per sub-batch.
    output_pad_size : int | None
        Output padding for FP8.
    dist_manager : DistributedManager
        Distributed manager.
    datapipe : TransolverDataPipe
        Data pipeline (used by ``forward_pass`` for unscaling).

    Returns
    -------
    tuple[float, dict, tuple[torch.Tensor, torch.Tensor]]
        ``(loss, metrics, (predictions, targets))``
    """
    N = batch["embeddings"].shape[1]
    indices = torch.randperm(N, device=batch["fx"].device)
    index_blocks = torch.split(indices, batch_resolution)

    global_preds_targets = []
    global_weight = 0.0
    start = time.time()
    for i, index_block in enumerate(index_blocks):
        local_embeddings = batch["embeddings"][:, index_block]
        local_fields = batch["fields"][:, index_block]

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

        local_loss, local_metrics, local_preds_targets = forward_pass(
            local_batch,
            model,
            precision,
            output_pad_size,
            dist_manager,
            data_mode,
            datapipe,
        )

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

    metrics = {k: v / global_weight for k, v in metrics.items()}
    loss = loss / global_weight

    global_predictions = torch.cat([l[0][0] for l in global_preds_targets], dim=1)
    global_targets = torch.cat([l[1][0] for l in global_preds_targets], dim=1)

    inverse_indices = torch.empty_like(indices)
    inverse_indices[indices] = torch.arange(indices.size(0), device=indices.device)
    global_predictions = global_predictions[:, inverse_indices]
    global_targets = global_targets[:, inverse_indices]
    return loss, metrics, (global_predictions, global_targets)


def mc_dropout_inference_loop(
    batch: dict,
    model: torch.nn.Module,
    precision: str,
    data_mode: Literal["surface", "volume"],
    batch_resolution: int,
    output_pad_size: int | None,
    dist_manager: DistributedManager,
    datapipe: TransolverDataPipe,
    n_samples: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, dict, torch.Tensor]:
    """Run MC-Dropout inference: N stochastic forward passes to estimate uncertainty.

    Parameters
    ----------
    batch : dict
        Input batch dictionary.
    model : torch.nn.Module
        Model with ConcreteDropout layers in train mode.
    precision : str
        Precision setting.
    data_mode : Literal["surface", "volume"]
        Data mode.
    batch_resolution : int
        Batch resolution for sub-batching.
    output_pad_size : int | None
        Output padding for FP8.
    dist_manager : DistributedManager
        Distributed manager.
    datapipe : TransolverDataPipe
        Data pipeline.
    n_samples : int
        Number of stochastic forward passes.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, dict, torch.Tensor]
        ``(mean_predictions, std_predictions, all_predictions, mean_loss,
        mean_metrics, targets)`` where ``all_predictions`` has shape
        ``(n_samples, batch, N, C)``.
    """
    all_predictions = []
    all_losses = []
    all_metrics_list = []
    targets = None

    for sample_idx in range(n_samples):
        start = time.time()
        loss, metrics, (preds, targets) = batched_inference_loop(
            batch,
            model,
            precision,
            data_mode,
            batch_resolution,
            output_pad_size,
            dist_manager,
            datapipe,
        )
        elapsed = time.time() - start
        print(f"  MC sample {sample_idx + 1}/{n_samples} in {elapsed:.2f}s")

        all_predictions.append(preds)
        all_losses.append(loss.item() if hasattr(loss, "item") else float(loss))
        all_metrics_list.append(metrics)

    stacked = torch.stack(all_predictions, dim=0)
    mean_predictions = stacked.mean(dim=0)
    std_predictions = stacked.std(dim=0)

    mean_loss = sum(all_losses) / n_samples
    mean_metrics = {}
    for key in all_metrics_list[0]:
        vals = [m[key] for m in all_metrics_list]
        mean_metrics[key] = (
            sum(v.item() if hasattr(v, "item") else float(v) for v in vals) / n_samples
        )

    return mean_predictions, std_predictions, stacked, mean_loss, mean_metrics, targets


def setup_mc_dropout(model, cfg, logger):
    """Set up MC-Dropout mode if enabled via config.

    When ``mc_dropout_samples > 0``, puts the model in eval mode but
    re-enables ConcreteDropout layers for stochastic forward passes.
    Falls back to standard eval if no ConcreteDropout layers are found.

    Parameters
    ----------
    model : torch.nn.Module
        The model to configure.
    cfg : DictConfig
        Hydra config; reads ``mc_dropout_samples`` (default 0).
    logger
        Logger for info/warning messages.

    Returns
    -------
    int
        The effective number of MC-Dropout samples (may be set to 0
        if no ConcreteDropout layers are found).
    """
    mc_dropout_samples = getattr(cfg, "mc_dropout_samples", 0)
    if mc_dropout_samples > 0:
        model.eval()
        for m in model.modules():
            if isinstance(m, ConcreteDropout):
                m.train()
        dropout_rates = get_concrete_dropout_rates(model)
        if dropout_rates:
            rates = list(dropout_rates.values())
            logger.info(
                f"MC-Dropout enabled with {mc_dropout_samples} samples. "
                f"Learned rates: min={min(rates):.4f} max={max(rates):.4f} "
                f"mean={sum(rates) / len(rates):.4f}"
            )
        else:
            logger.warning(
                "mc_dropout_samples > 0 but no ConcreteDropout layers found. "
                "Was the model trained with concrete_dropout=true?"
            )
            mc_dropout_samples = 0
            model.eval()
    else:
        model.eval()

    return mc_dropout_samples
