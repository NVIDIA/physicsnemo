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

import os
import sys
import logging
import tempfile
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


def save_vtp_sequence(
    preds,
    exacts,
    vtp_frames_dir,  # directory containing frame_XXX.vtp for this run
    out_pred_dir,
    out_exact_dir=None,
    prefix="frame",
    extra_fields=None,  # dict[name -> list of [N,C] tensors per timestep]
    exact_extra_fields=None,  # optional exact values for extra fields
):
    """
    Save a sequence of predicted (and optional exact) positions to VTP files.
    preds/exacts: tensor [T,N,3] or list of [N,3] torch.Tensors
    extra_fields: dict mapping field name to list of [N,C] tensors (one per timestep)
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

        # Predicted positions
        mesh_pred.points = pred_np
        mesh_pred.point_data["prediction"] = pred_np

        # Add predicted extra fields (stress, strain, etc.)
        if extra_fields:
            for name, field_seq in extra_fields.items():
                field_np = field_seq[t].detach().cpu().numpy()  # [N] or [N,C]
                if field_np.shape[0] != mesh_pred.n_points:
                    logging.warning(
                        f"Field '{name}' size mismatch at t={t}: "
                        f"{field_np.shape[0]} vs {mesh_pred.n_points}"
                    )
                    continue
                if field_np.ndim == 1 or field_np.shape[1] == 1:
                    mesh_pred.point_data[f"pred_{name}"] = field_np.squeeze()
                else:
                    mesh_pred.point_data[f"pred_{name}"] = field_np

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

                # Add exact extra fields and compute differences
                if exact_extra_fields:
                    for name, field_seq in exact_extra_fields.items():
                        exact_field_np = (
                            field_seq[t].detach().cpu().numpy()
                        )  # [N] or [N,C]
                        if exact_field_np.shape[0] != mesh_exact.n_points:
                            logging.warning(
                                f"Exact field '{name}' size mismatch at t={t}"
                            )
                            continue
                        if exact_field_np.ndim == 1 or exact_field_np.shape[1] == 1:
                            mesh_exact.point_data[f"exact_{name}"] = (
                                exact_field_np.squeeze()
                            )
                        else:
                            mesh_exact.point_data[f"exact_{name}"] = exact_field_np

                        if extra_fields and name in extra_fields:
                            pred_field_np = extra_fields[name][t].detach().cpu().numpy()
                            if pred_field_np.shape == exact_field_np.shape:
                                if (
                                    pred_field_np.ndim == 1
                                    or pred_field_np.shape[1] == 1
                                ):
                                    mesh_exact.point_data[f"pred_{name}"] = (
                                        pred_field_np.squeeze()
                                    )
                                    mesh_exact.point_data[f"diff_{name}"] = (
                                        pred_field_np - exact_field_np
                                    ).squeeze()
                                else:
                                    mesh_exact.point_data[f"pred_{name}"] = (
                                        pred_field_np
                                    )
                                    mesh_exact.point_data[f"diff_{name}"] = (
                                        pred_field_np - exact_field_np
                                    )

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
        # Fo (features per timestep) is computed from the first sample in run_on_single_run
        # to support both position-only (Fo=3) and dynamic targets (Fo=3+sum(C_k))

        # Dataloader workers (for single-sample run datasets this can be 0 or small)
        self.num_workers = cfg.training.num_dataloader_workers

    @torch.no_grad()
    def run_on_single_run(self, run_path: str, run_name: str):
        """
        Process a single run: build a one-run dataset, run inference, and save outputs.

        ``run_path`` is a temp dir containing a symlinked .vtp file; ``run_name``
        is the stem of that file so output paths match the dataset.
        """
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

            # Model returns [N, T, Fo]; Fo = 3 for position-only, 3+sum(C_k) with dynamic targets
            N, T, Fo = sample.node_target.shape
            assert T == self.T, (
                f"Target T={T} does not match model rollout steps {self.T}"
            )

            # Forward rollout: model returns [N, T, Fo]
            pred = self.model(sample=sample, data_stats=data_stats)
            target_series = getattr(sample, "target_series", None) or {}

            # Extract positions [T, N, 3] and extra fields
            pred_pos = pred[:, :, :3].transpose(0, 1)  # [T, N, 3]
            pred_extra = _extract_extra_fields(pred, target_series, self.T)

            exact_pos = exact_extra = None
            if sample.node_target is not None:
                exact_pos = sample.node_target[:, :, :3].transpose(0, 1)  # [T, N, 3]
                exact_extra = _extract_extra_fields(
                    sample.node_target, target_series, self.T
                )

            # Denormalize positions
            pred_pos_denorm = denormalize_positions(pred_pos, pos_mean, pos_std)
            exact_pos_denorm = (
                denormalize_positions(exact_pos, pos_mean, pos_std)
                if exact_pos is not None
                else None
            )

            if self.write_vtp and os.path.isdir(vtp_frames_dir):
                pred_dir = os.path.join(
                    self.out_pred_root, f"rank{self.dist.rank}", run_name
                )
                exact_dir = (
                    os.path.join(self.out_exact_root, f"rank{self.dist.rank}", run_name)
                    if exact_pos_denorm is not None
                    else None
                )
                save_vtp_sequence(
                    preds=pred_pos_denorm,
                    exacts=exact_pos_denorm,
                    vtp_frames_dir=vtp_frames_dir,
                    out_pred_dir=pred_dir,
                    out_exact_dir=exact_dir,
                    prefix=self.vtp_prefix,
                    extra_fields=pred_extra or None,
                    exact_extra_fields=exact_extra or None,
                )
                if pred_extra:
                    self.logger.info(
                        f"[Rank {self.dist.rank}] Saved predicted fields: {list(pred_extra.keys())}"
                    )
            elif self.write_vtp:
                self.logger.warning(
                    f"[Rank {self.dist.rank}] Missing VTP frames dir {vtp_frames_dir}; skipping export."
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

    # Discover all .vtp files in the parent directory (flat layout)
    parent_dir = to_absolute_path(cfg.inference.raw_data_dir_test)
    if not os.path.isdir(parent_dir):
        logger0.error(f"Parent directory not found: {parent_dir}")
        return

    run_items = [
        f.path
        for f in os.scandir(parent_dir)
        if f.is_file() and f.name.lower().endswith(".vtp")
    ]
    run_items.sort()
    run_names = [os.path.splitext(os.path.basename(p))[0] for p in run_items]

    if len(run_items) == 0:
        logger0.error(f"No .vtp files found under: {parent_dir}")
        return

    logger0.info(f"Found {len(run_items)} runs under {parent_dir}")

    # Shard run list across ranks: rank r processes run_items[r::world_size]
    my_items = run_items[dist.rank :: dist.world_size]
    my_names = run_names[dist.rank :: dist.world_size]
    logger.info(f"[Rank {dist.rank}] Assigned {len(my_items)} runs.")

    worker = InferenceWorker(cfg, logger, dist)

    for run_path, run_name in zip(my_items, my_names):
        with tempfile.TemporaryDirectory(prefix="crash_inference_") as tmp:
            link_path = os.path.join(tmp, os.path.basename(run_path))
            os.symlink(run_path, link_path)
            worker.run_on_single_run(tmp, run_name=run_name)

    if dist.rank == 0:
        logger0.info("Inference completed successfully.")


if __name__ == "__main__":
    main()
