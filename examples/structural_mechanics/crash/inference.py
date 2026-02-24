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

import os, sys, logging, tempfile
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
        return y * pos_std.view(1, -1) + pos_mean.view(1, -1)
    elif y.ndim == 3:  # [T,N,3]
        return y * pos_std.view(1, 1, -1) + pos_mean.view(1, 1, -1)
    else:
        raise AssertionError(f"Expected [N,3] or [T,N,3], got {y.shape}")


def save_vtp_sequence(
    preds,
    exacts,
    vtp_frames_dir,  # directory containing frame_XXX.vtp for this run
    out_pred_dir,
    out_exact_dir=None,
    prefix="frame",
):
    """
    Save a sequence of predicted (and optional exact) positions to VTP files.
    preds/exacts: list of [N,3] torch.Tensors
    """
    os.makedirs(out_pred_dir, exist_ok=True)
    if exacts is not None and out_exact_dir is not None:
        os.makedirs(out_exact_dir, exist_ok=True)

    T = len(preds)
    for t in range(T):
        vtp_file = os.path.join(vtp_frames_dir, f"{prefix}_{t:03d}.vtp")
        if not os.path.exists(vtp_file):
            logging.warning(f"Missing VTP frame: {vtp_file}, skipping timestep {t}.")
            continue

        pred_np = preds[t].detach().cpu().numpy()
        mesh_pred = pv.read(vtp_file)
        if pred_np.shape[0] != mesh_pred.n_points:
            logging.warning(
                f"Point mismatch at t={t}: pred {pred_np.shape[0]} vs mesh {mesh_pred.n_points}"
            )
            continue

        # Predicted
        mesh_pred.points = pred_np
        mesh_pred.point_data["prediction"] = pred_np
        mesh_pred.save(os.path.join(out_pred_dir, f"{prefix}_{t:03d}_pred.vtp"))

        # Exact + difference
        if exacts is not None and out_exact_dir is not None:
            exact_np = exacts[t].detach().cpu().numpy()
            if exact_np.shape[0] != mesh_pred.n_points:
                logging.warning(
                    f"Exact mismatch at t={t}: {exact_np.shape[0]} vs mesh {mesh_pred.n_points}"
                )
            else:
                mesh_exact = pv.read(vtp_file)
                mesh_exact.points = exact_np
                mesh_exact.point_data["exact"] = exact_np
                mesh_exact.point_data["difference"] = pred_np - exact_np
                mesh_exact.save(
                    os.path.join(out_exact_dir, f"{prefix}_{t:03d}_exact.vtp")
                )


class InferenceWorker:
    """
    Creates the model once and runs inference on a single run-directory at a time.
    Each rank calls `run_on_single_run(run_path)` for its assigned runs.
    """

    def __init__(self, cfg: DictConfig, logger: PythonLogger, dist: DistributedManager):
        self.cfg = cfg
        self.logger = logger
        self.dist = dist
        self.device = dist.device

        # Build model once per rank
        self.model = instantiate(cfg.model)
        logging.getLogger().setLevel(logging.INFO)
        self.model.to(self.device)
        self.model.eval()

        ckpt_path = cfg.training.ckpt_path
        load_checkpoint(ckpt_path, models=self.model, device=self.device)
        self.logger.info(f"[Rank {dist.rank}] Loaded checkpoint {ckpt_path}")

        # For VTP exporting
        self.vtp_prefix = cfg.inference.get("vtp_prefix", "frame")
        self.write_vtp = True

        # Output roots
        self.out_pred_root = cfg.inference.get("output_dir_pred", "./predicted_vtps")
        self.out_exact_root = cfg.inference.get("output_dir_exact", "./exact_vtps")

        # How many timesteps to roll out
        self.T = cfg.training.num_time_steps - 1
        self.Fo = 3

        # Dataloader workers (for single-sample run datasets this can be 0 or small)
        self.num_workers = cfg.training.num_dataloader_workers

    @torch.no_grad()
    def run_on_single_run(self, run_path: str):
        """
        Process a single run directory: build a one-run dataset with a temporary symlink, run inference, and save outputs.
        """
        run_name = os.path.basename(run_path)
        self.logger.info(f"[Rank {self.dist.rank}] Processing run: {run_name}")

        # Instantiate a dataset that sees exactly one run
        reader = instantiate(self.cfg.reader)
        dataset = instantiate(
            self.cfg.datapipe,
            name="crash_test",
            reader=reader,
            split="test",
            num_steps=self.cfg.training.num_time_steps,
            num_samples=1,
            logger=self.logger,
            data_dir=run_path,
        )

        # Data stats for de/normalization
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
        )

        # Simple 1-sample loader
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=simsample_collate,
        )

        # VTP frames directory generated by the dataset for THIS run
        vtp_frames_dir = os.path.join(os.getcwd(), f"output_{run_name}")

        pos_mean = data_stats["node"]["pos_mean"]
        pos_std = data_stats["node"]["pos_std"]

        for local_idx, sample in enumerate(dataloader):
            if isinstance(sample, list):
                sample = sample[0]
            sample = sample.to(self.device)

            # Forward rollout: expected to return [T,N,3]
            pred_seq = self.model(sample=sample, data_stats=data_stats)

            # Exact sequence (if provided)
            exact_seq = None
            if sample.node_target is not None:
                N = sample.node_target.size(0)
                assert sample.node_target.size(1) == self.T * self.Fo
                exact_seq = (
                    sample.node_target.view(N, self.T, self.Fo)
                    .transpose(0, 1)
                    .contiguous()
                )

            # Denormalize
            pred_seq_denorm = [
                denormalize_positions(pred_seq[t], pos_mean, pos_std)
                for t in range(pred_seq.size(0))
            ]
            exact_seq_denorm = (
                [
                    denormalize_positions(exact_seq[t], pos_mean, pos_std)
                    for t in range(self.T)
                ]
                if exact_seq is not None
                else None
            )

            # Save VTPs (rank-separated, run-separated)
            if self.write_vtp:
                sample_tag = f"{run_name}"
                pred_dir = os.path.join(
                    self.out_pred_root, f"rank{self.dist.rank}", sample_tag
                )
                exact_dir = (
                    os.path.join(
                        self.out_exact_root, f"rank{self.dist.rank}", sample_tag
                    )
                    if exact_seq_denorm
                    else None
                )

                if not os.path.isdir(vtp_frames_dir):
                    self.logger.warning(
                        f"[Rank {self.dist.rank}] Missing VTP frames dir {vtp_frames_dir}; skipping export."
                    )
                else:
                    save_vtp_sequence(
                        preds=pred_seq_denorm,
                        exacts=exact_seq_denorm,
                        vtp_frames_dir=vtp_frames_dir,
                        out_pred_dir=pred_dir,
                        out_exact_dir=exact_dir,
                        prefix=self.vtp_prefix,
                    )

        self.logger.info(f"[Rank {self.dist.rank}] Finished run: {run_name}")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Initialize distributed (one process per GPU via torchrun)
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("inference")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger0.file_logging()
    logging.getLogger().setLevel(logging.INFO)

    # Discover all Run* directories under the parent test folder
    parent_dir = to_absolute_path(cfg.inference.raw_data_dir_test)
    if not os.path.isdir(parent_dir):
        logger0.error(f"Parent directory not found: {parent_dir}")
        return

    run_dirs = [d.path for d in os.scandir(parent_dir) if d.is_dir()]
    run_dirs.sort()

    if len(run_dirs) == 0:
        logger0.error(f"No run directories found under: {parent_dir}")
        return

    logger0.info(f"Found {len(run_dirs)} runs under {parent_dir}")

    # Shard run list across ranks: rank r processes run_dirs[r::world_size]
    my_runs = run_dirs[dist.rank :: dist.world_size]
    logger.info(f"[Rank {dist.rank}] Assigned {len(my_runs)} runs.")

    worker = InferenceWorker(cfg, logger, dist)

    for run_path in my_runs:
        worker.run_on_single_run(run_path)

    if dist.rank == 0:
        logger0.info("Inference completed successfully.")


if __name__ == "__main__":
    main()
