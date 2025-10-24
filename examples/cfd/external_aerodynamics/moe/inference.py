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

import os
import torch
import numpy as np
import pyvista as pv
import logging
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from dataset import ProcessedVTPDataset
from model import MoEGatingNet
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils import load_checkpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Inference")


def denormalize(data, mean, std):
    return data * std + mean


def run_inference(dataset, cfg):
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    rank = dist.rank
    world_size = dist.world_size

    if rank == 0:
        logger.info(f"Distributed inference: {world_size} GPUs")
        os.makedirs(cfg.output_dir, exist_ok=True)

    # Load normalization stats
    norm_dir = os.path.join(to_absolute_path(cfg.preprocessed_data_dir), "stats")
    p_mean = np.load(os.path.join(norm_dir, "pMeanTrim_mean.npy"))
    p_std = np.load(os.path.join(norm_dir, "pMeanTrim_std.npy"))
    shear_mean = np.load(os.path.join(norm_dir, "wallShearStressMeanTrim_mean.npy"))
    shear_std = np.load(os.path.join(norm_dir, "wallShearStressMeanTrim_std.npy"))

    # Distributed sampler for partitioning data
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        collate_fn=lambda x: x,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=True,
        drop_last=False,
    )

    # Initialize models with same configuration as training
    pressure_gating = MoEGatingNet(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_experts=cfg.num_experts,
        num_feature_per_expert=cfg.num_feature_per_expert_pressure,
        use_moe_bias=cfg.use_moe_bias,
        include_normals=cfg.include_normals,
    ).to(device)

    shear_gating = MoEGatingNet(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_experts=cfg.num_experts,
        num_feature_per_expert=cfg.num_feature_per_expert_shear,
        use_moe_bias=cfg.use_moe_bias,
        include_normals=cfg.include_normals,
    ).to(device)

    # Wrap with DistributedDataParallel
    if world_size > 1:
        pressure_gating = torch.nn.parallel.DistributedDataParallel(
            pressure_gating, device_ids=[dist.local_rank], output_device=dist.local_rank
        )
        shear_gating = torch.nn.parallel.DistributedDataParallel(
            shear_gating, device_ids=[dist.local_rank], output_device=dist.local_rank
        )

    # Load checkpoint
    iter_init = load_checkpoint(
        cfg.checkpoint_dir,
        models=[pressure_gating, shear_gating],
        device=device,
    )

    pressure_gating.eval()
    shear_gating.eval()

    if rank == 0:
        logger.info("Running inference...")
        pbar = tqdm(dataloader, desc="Inference", unit="batch")
    else:
        pbar = dataloader

    for batch in pbar:
        sample = batch[0]
        mesh = sample["mesh"]  # PyVista mesh

        # Prepare input features
        p_preds = (
            torch.from_numpy(
                np.concatenate(
                    [
                        sample["p_pred_xmgn"],
                        sample["p_pred_fignet"],
                        sample["p_pred_domino"],
                    ],
                    axis=1,
                )
            )
            .float()
            .to(device)
        )  # (N, 3)
        normals = torch.from_numpy(sample["normals"]).float().to(device)  # (N, 3)
        p_gating_input = torch.cat([p_preds, normals], dim=1)  # (N, 6)

        shear_preds = (
            torch.from_numpy(
                np.stack(
                    [
                        sample["shear_pred_xmgn"],
                        sample["shear_pred_fignet"],
                        sample["shear_pred_domino"],
                    ],
                    axis=1,
                )
            )
            .float()
            .to(device)
        )  # (N, 3, 3)
        shear_preds_flat = shear_preds.reshape(shear_preds.shape[0], -1)  # (N, 9)
        shear_gating_input = torch.cat([shear_preds_flat, normals], dim=1)  # (N, 12)

        # Gating and mixture
        with torch.no_grad():
            gate_p = pressure_gating(p_gating_input)  # (N, 3)
            pred_p = torch.sum(gate_p * p_preds, dim=1, keepdim=True)  # (N, 1)
            gate_shear = shear_gating(shear_gating_input)  # (N, 3)
            pred_shear = torch.sum(
                gate_shear.unsqueeze(-1) * shear_preds, dim=1
            )  # (N, 3)

        # Denormalize predictions
        pred_p_denorm = denormalize(pred_p.cpu().numpy(), p_mean, p_std)  # (N, 1)
        pred_shear_denorm = denormalize(
            pred_shear.cpu().numpy(), shear_mean, shear_std
        )  # (N, 3)

        # Save gating scores as fields (order: xmgn, fignet, domino)
        # gate_p: (N, 3) -- columns: [xmgn, fignet, domino]
        mesh.point_data["xmgn_pressure_score"] = (
            gate_p[:, 0].cpu().numpy().reshape(-1, 1)
        )
        mesh.point_data["fignet_pressure_score"] = (
            gate_p[:, 1].cpu().numpy().reshape(-1, 1)
        )
        mesh.point_data["domino_pressure_score"] = (
            gate_p[:, 2].cpu().numpy().reshape(-1, 1)
        )

        # Save shear gating scores as fields (order: xmgn, fignet, domino)
        # gate_shear: (N, 3) -- columns: [xmgn, fignet, domino]
        mesh.point_data["xmgn_shear_score"] = (
            gate_shear[:, 0].cpu().numpy().reshape(-1, 1)
        )
        mesh.point_data["fignet_shear_score"] = (
            gate_shear[:, 1].cpu().numpy().reshape(-1, 1)
        )
        mesh.point_data["domino_shear_score"] = (
            gate_shear[:, 2].cpu().numpy().reshape(-1, 1)
        )

        # Append predictions to mesh
        mesh.point_data["pMeanTrimPred_MoE"] = pred_p_denorm.reshape(-1)
        mesh.point_data["wallShearStressMeanTrimPred_MoE"] = pred_shear_denorm

        # Save mesh
        out_path = os.path.join(cfg.output_dir, f"{sample['id']}_moe_pred.vtp")
        mesh.save(out_path)
        # Synchronize after each save to avoid conflicts
        torch.distributed.barrier()

        # Avoid memory leaks
        del sample, mesh, p_preds, normals, p_gating_input
        del shear_preds, shear_preds_flat, shear_gating_input
        del gate_p, pred_p, gate_shear, pred_shear
        del pred_p_denorm, pred_shear_denorm
        torch.cuda.empty_cache()

    if rank == 0:
        logger.info(f"Inference complete. Results saved to {cfg.output_dir}")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(
        f"Loading validation dataset from {to_absolute_path(cfg.preprocessed_data_dir)}..."
    )
    dataset = ProcessedVTPDataset(
        os.path.join(to_absolute_path(cfg.preprocessed_data_dir), "val"),
        os.path.join(to_absolute_path(cfg.preprocessed_data_dir), "stats"),
        normalize=True,
        return_mesh=True,
    )
    logger.info("Dataset loaded.")
    run_inference(dataset, cfg)


if __name__ == "__main__":
    main()
