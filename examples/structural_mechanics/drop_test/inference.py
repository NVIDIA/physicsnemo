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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Inference script that saves a single VTU file per run.

Output VTU uses the exact mesh geometry (t0 reference) with point data:
- displacement_pred_t*, displacement_exact_t*, displacement_diff_t*
- Von_Mises_pred_t*, Von_Mises_exact_t*, Von_Mises_diff_t* (or other dynamic targets)

All fields are on the exact mesh; diff = predicted - exact.
"""

import os
import sys
import logging
import tempfile
import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(__file__))

import hydra
from hydra.utils import to_absolute_path, instantiate
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader

from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils import load_checkpoint

from datapipe import simsample_collate

EPS = 1e-8


def denormalize_positions(
    y: torch.Tensor, pos_mean: torch.Tensor, pos_std: torch.Tensor
) -> torch.Tensor:
    """Denormalize node positions [N,3] or [T,N,3]."""
    if y.ndim == 2:  # [N,3]
        return y * (pos_std.view(1, -1) + EPS) + pos_mean.view(1, -1)
    elif y.ndim == 3:  # [T,N,3]
        return y * (pos_std.view(1, 1, -1) + EPS) + pos_mean.view(1, 1, -1)
    else:
        raise ValueError(f"Expected [N,3] or [T,N,3], got {y.shape}")


def _extract_extra_fields(data: torch.Tensor, target_series: dict, T: int) -> dict:
    """Extract extra fields from [N,T,Fo] tensor. Returns {name: list of [N,C] per timestep}."""
    if data.shape[2] <= 3 or not target_series:
        return {}
    out, idx = {}, 3
    for name, tgt in target_series.items():
        C = 1 if tgt.ndim == 2 else tgt.shape[-1]
        field = data[:, :, idx : idx + C]  # [N,T,C]
        out[name] = [field[:, t, :] for t in range(T)]
        idx += C
    return out


def _denormalize_extra_fields(
    extra: dict, dyn_stats: dict, log_transform: bool = False
) -> dict:
    """Denormalize extra fields using dynamic target stats.

    If log_transform is True, applies expm1 (inverse of log1p) after
    denormalization to recover physical-space values.
    """
    if not extra or not dyn_stats:
        return extra
    out = {}
    for name, field_seq in extra.items():
        mu_key, std_key = f"{name}_mean", f"{name}_std"
        if mu_key not in dyn_stats or std_key not in dyn_stats:
            out[name] = field_seq
            continue
        mu = dyn_stats[mu_key]
        std = dyn_stats[std_key]
        denorm_seq = []
        for f in field_seq:
            val = f * (std + EPS) + mu
            if log_transform:
                val = torch.expm1(val).clamp(min=0.0)
            denorm_seq.append(val)
        out[name] = denorm_seq
    return out


def save_single_vtu(
    mesh_ref: pv.UnstructuredGrid,
    pos0: np.ndarray,
    pred_pos: list,
    exact_pos: list,
    pred_extra: dict,
    exact_extra: dict,
    output_path: str,
    prefix: str = "frame",
):
    """
    Save a single VTU file with all timesteps on the exact mesh.

    Uses mesh_ref geometry (cells, connectivity from exact mesh at t0).
    Point data arrays are added for each timestep:
      - displacement_pred_t*, displacement_exact_t*, displacement_diff_t*
      - {field}_pred_t*, {field}_exact_t*, {field}_diff_t* for each extra field
    """
    T = len(pred_pos)
    N = pos0.shape[0]

    # Clone mesh - keep exact geometry (cells) but use t0 points as reference
    mesh = mesh_ref.copy()
    mesh.points = pos0

    for t in range(T):
        # pred/exact index t corresponds to timestep t+1 (t0 is the initial condition)
        tag = f"{prefix}_{t + 1:03d}"
        pred_np = pred_pos[t] if torch.is_tensor(pred_pos[t]) else pred_pos[t]
        exact_np = exact_pos[t] if torch.is_tensor(exact_pos[t]) else exact_pos[t]
        if torch.is_tensor(pred_np):
            pred_np = pred_np.detach().cpu().numpy()
        if torch.is_tensor(exact_np):
            exact_np = exact_np.detach().cpu().numpy()

        # Displacement from t0
        disp_pred = pred_np - pos0
        disp_exact = exact_np - pos0
        disp_diff = disp_pred - disp_exact

        mesh.point_data[f"displacement_pred_{tag}"] = disp_pred
        mesh.point_data[f"displacement_exact_{tag}"] = disp_exact
        mesh.point_data[f"displacement_diff_{tag}"] = disp_diff

        # Extra fields (Von_Mises, etc.)
        if pred_extra:
            for name, field_seq in pred_extra.items():
                pred_f = field_seq[t]
                exact_f = (
                    exact_extra[name][t]
                    if exact_extra
                    and name in exact_extra
                    and t < len(exact_extra[name])
                    else None
                )
                pred_fnp = (
                    pred_f.detach().cpu().numpy()
                    if torch.is_tensor(pred_f)
                    else np.asarray(pred_f)
                )
                if pred_fnp.ndim == 1:
                    pred_fnp = pred_fnp[:, np.newaxis]
                pred_fnp = pred_fnp.squeeze()

                mesh.point_data[f"{name}_pred_{tag}"] = pred_fnp

                if exact_f is not None:
                    exact_fnp = (
                        exact_f.detach().cpu().numpy()
                        if torch.is_tensor(exact_f)
                        else np.asarray(exact_f)
                    )
                    if exact_fnp.ndim == 1:
                        exact_fnp = exact_fnp[:, np.newaxis]
                    exact_fnp = exact_fnp.squeeze()
                    diff_fnp = pred_fnp - exact_fnp

                    mesh.point_data[f"{name}_exact_{tag}"] = exact_fnp
                    mesh.point_data[f"{name}_diff_{tag}"] = diff_fnp

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mesh.save(output_path)
    logging.info(f"Saved single VTU: {output_path}")


class InferenceWorkerSingleVtu:
    """Inference worker that saves one VTU per run with all fields on exact mesh."""

    def __init__(self, cfg: DictConfig, logger: PythonLogger, dist: DistributedManager):
        self.cfg = cfg
        self.logger = logger
        self.dist = dist
        self.device = dist.device

        self.model = instantiate(cfg.model)
        logging.getLogger().setLevel(logging.INFO)
        self.model.to(self.device)
        self.model.eval()

        ckpt_path = cfg.training.ckpt_path
        load_checkpoint(ckpt_path, models=self.model, device=self.dist.device)
        self.logger.info(f"[Rank {self.dist.rank}] Loaded checkpoint {ckpt_path}")

        self.frame_prefix = cfg.inference.get("frame_prefix", "frame")
        self.out_dir = cfg.inference.get("output_dir_single_vtu", "./single_vtu_output")
        self.T = cfg.training.num_time_steps - 1
        self.num_workers = cfg.training.num_dataloader_workers
        self.log_transform_targets = cfg.datapipe.get("log_transform_targets", False)

    @torch.no_grad()
    def run_on_single_run(self, run_path: str, run_name: str):
        """Run inference rollout on one simulation and write predicted VTU files."""
        self.logger.info(f"[Rank {self.dist.rank}] Processing run: {run_name}")

        reader = instantiate(self.cfg.reader)
        dataset = instantiate(
            self.cfg.datapipe,
            name="drop_test_single_vtu",
            reader=reader,
            split="test",
            num_steps=self.cfg.training.num_time_steps,
            num_samples=1,
            logger=self.logger,
            data_dir=run_path,
            sample_type="all_time_steps",
        )

        data_stats = dict(
            node={k: v.to(self.device) for k, v in dataset.node_stats.items()},
            edge={
                k: v.to(self.device)
                for k, v in getattr(dataset, "edge_stats", {}).items()
            },
            feature={
                k: v.to(self.device)
                for k, v in getattr(dataset, "feature_stats", {}).items()
            },
            dynamic_target={
                k: v.to(self.device)
                for k, v in getattr(dataset, "dynamic_target_stats", {}).items()
            },
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=simsample_collate,
        )

        vtu_frames_dir = os.path.join(os.getcwd(), f"output_{run_name}")
        pos_mean = data_stats["node"]["pos_mean"]
        pos_std = data_stats["node"]["pos_std"]

        for sample in dataloader:
            if isinstance(sample, list):
                sample = sample[0]
            sample = sample.to(self.device)

            N, T, Fo = sample.node_target.shape
            assert T == self.T, f"Target T={T} != rollout steps {self.T}"

            pred = self.model(sample=sample, data_stats=data_stats)
            target_series = getattr(sample, "target_series", None) or {}

            pred_pos = pred[:, :, :3].transpose(0, 1)
            pred_extra = _extract_extra_fields(pred, target_series, self.T)
            exact_pos = sample.node_target[:, :, :3].transpose(0, 1)
            exact_extra = _extract_extra_fields(
                sample.node_target, target_series, self.T
            )

            pred_pos_denorm = denormalize_positions(pred_pos, pos_mean, pos_std)
            exact_pos_denorm = denormalize_positions(exact_pos, pos_mean, pos_std)
            dyn_stats = data_stats.get("dynamic_target", {})
            log_t = self.log_transform_targets
            pred_extra_denorm = _denormalize_extra_fields(
                pred_extra, dyn_stats, log_transform=log_t
            )
            exact_extra_denorm = _denormalize_extra_fields(
                exact_extra, dyn_stats, log_transform=log_t
            )

            if not os.path.isdir(vtu_frames_dir):
                self.logger.warning(
                    f"[Rank {self.dist.rank}] Missing frames dir {vtu_frames_dir}; skipping."
                )
                break

            # Load reference mesh from exact frame (t0) - use exact geometry
            frame0_path = os.path.join(vtu_frames_dir, f"{self.frame_prefix}_000.vtu")
            if not os.path.exists(frame0_path):
                self.logger.warning(f"Missing {frame0_path}; skipping.")
                break

            mesh_ref = pv.read(frame0_path)
            pos0 = np.array(mesh_ref.points, dtype=np.float64)

            out_path = os.path.join(
                self.out_dir, f"rank{self.dist.rank}", f"{run_name}_comparison.vtu"
            )
            save_single_vtu(
                mesh_ref=mesh_ref,
                pos0=pos0,
                pred_pos=[p for p in pred_pos_denorm],
                exact_pos=[p for p in exact_pos_denorm],
                pred_extra=pred_extra_denorm or {},
                exact_extra=exact_extra_denorm or {},
                output_path=out_path,
                prefix=self.frame_prefix,
            )
            break

        self.logger.info(f"[Rank {self.dist.rank}] Finished run: {run_name}")


@hydra.main(
    version_base="1.3",
    config_path="conf",
    config_name="drop_test_geotransolver_oneshot",
)
def main(cfg: DictConfig):
    """Hydra entry point: shard runs across ranks and write predicted VTUs."""
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("inference_single_vtu")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger0.file_logging()
    logging.getLogger().setLevel(logging.INFO)

    parent_dir = to_absolute_path(cfg.inference.raw_data_dir_test)
    if not os.path.isdir(parent_dir):
        logger0.error(f"Parent directory not found: {parent_dir}")
        return

    run_items = [
        f.path
        for f in os.scandir(parent_dir)
        if f.is_file() and f.name.lower().endswith(".vtu")
    ]
    run_items.sort()
    run_names = [os.path.splitext(os.path.basename(p))[0] for p in run_items]

    if len(run_items) == 0:
        logger0.error(f"No .vtu files found under: {parent_dir}")
        return

    logger0.info(f"Found {len(run_items)} runs under {parent_dir}")

    my_items = run_items[dist.rank :: dist.world_size]
    my_names = run_names[dist.rank :: dist.world_size]
    logger.info(f"[Rank {dist.rank}] Assigned {len(my_items)} runs.")

    worker = InferenceWorkerSingleVtu(cfg, logger, dist)

    for run_path, run_name in zip(my_items, my_names):
        with tempfile.TemporaryDirectory(prefix="drop_test_single_vtu_") as tmp:
            link_path = os.path.join(tmp, os.path.basename(run_path))
            os.symlink(run_path, link_path)
            worker.run_on_single_run(tmp, run_name=run_name)

    if dist.rank == 0:
        logger0.info("Inference (single VTU) completed successfully.")


if __name__ == "__main__":
    main()
