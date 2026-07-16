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

r"""Run inference on the SuperWing test split with a trained AeroJEPA model.

Loads a checkpoint, encodes each test case through the context and
target encoders, predicts target tokens with the JEPA predictor head,
decodes the surface field at the full ``128 x 256`` grid, denormalises
back to physical units, and writes:

* ``<output_dir>/predictions.npz`` — stacked per-case predictions and
  metadata (consumed by the CL/CD post-processing script).
* ``<output_dir>/plots/<case_id>_{cp,cf_tau,cf_z}.png`` — three field
  plots per case for the first ``num_plots`` cases.

Inference is embarrassingly parallel over test cases: each rank runs a
disjoint stride of the split and the results are gathered onto rank 0,
which writes the single ``predictions.npz``. The GPU count is taken from
the launcher -- nothing to configure or merge.

Usage::

    # Single GPU:
    python inference.py \
        checkpoint=outputs/<run-name>/checkpoints/best.pt \
        data.path=/path/to/SuperWing_Dataset \
        output_dir=inference

    # Multi-GPU (one process per GPU):
    torchrun --nproc_per_node=<#GPUs> inference.py \
        checkpoint=outputs/<run-name>/checkpoints/best.pt \
        data.path=/path/to/SuperWing_Dataset \
        output_dir=inference
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from physicsnemo.distributed import DistributedManager
from physicsnemo.experimental.models.aerojepa import TokenSet
from src.datapipes import (
    SuperWingDataset,
    build_superwing_split_manifest,
    compute_superwing_normalization_stats,
    superwing_collate,
)
from src.datapipes.superwing import SUPERWING_GRID_SHAPE
from src.training import (
    ExponentialMovingAverage,
    get_autocast_context,
    move_batch_to_device,
    set_seed,
)
from src.visualization import denormalize_field, plot_surface_field


log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Setup helpers (mirror the train.py builders)
# --------------------------------------------------------------------------- #


def _ensure_superwing_artifacts(data_cfg: DictConfig) -> tuple[str, str]:
    """Return (split_manifest_path, normalization_stats_path), building if missing."""
    root = Path(str(data_cfg.path))
    split_path = (
        Path(str(data_cfg.split_manifest))
        if data_cfg.split_manifest
        else root / "split_by_geometry.json"
    )
    stats_path = (
        Path(str(data_cfg.normalization_stats_path))
        if data_cfg.normalization_stats_path
        else root / "normalization_stats_train.json"
    )

    if not split_path.exists():
        log.info("Building split manifest at %s", split_path)
        manifest = build_superwing_split_manifest(
            root_dir=str(root),
            train_ratio=float(data_cfg.train_ratio),
            val_ratio=float(data_cfg.val_ratio),
            seed=int(data_cfg.split_seed),
        )
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with split_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    if not stats_path.exists():
        log.info("Building normalization stats at %s", stats_path)
        compute_superwing_normalization_stats(
            root_dir=str(root),
            split_manifest_path=str(split_path),
            gen_param_columns=list(data_cfg.gen_params_columns),
            gen_param_names=list(data_cfg.gen_params_names),
            max_target_samples=int(data_cfg.normalization_max_target_samples),
            save_path=str(stats_path),
        )

    return str(split_path), str(stats_path)


def _build_test_loader(
    data_cfg: DictConfig,
    *,
    split_manifest_path: str,
    normalization_stats_path: str,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    dataset = SuperWingDataset(
        root_dir=str(data_cfg.path),
        split="test",
        split_manifest_path=split_manifest_path,
        normalization_stats_path=normalization_stats_path,
        surface_points=int(data_cfg.surface_points),
        target_encoder_points=int(data_cfg.target_encoder_points),
        query_points=int(data_cfg.query_points),
        eval_full_grid_query=True,
        return_origingeom=True,
        return_full_fields=True,
        deterministic_sampling=True,
        normalize_xyz=bool(data_cfg.normalize_xyz),
    )
    # Strided shard: rank `rank` owns cases rank, rank+world_size, ...
    # Passing an explicit index list as the sampler means each rank only reads
    # its own cases (lighter on the shared filesystem than reading the full
    # split on every rank and skipping in the loop).
    sampler = (
        list(range(int(rank), len(dataset), int(world_size)))
        if int(world_size) > 1
        else None
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=int(data_cfg.num_workers),
        pin_memory=bool(data_cfg.pin_memory),
        collate_fn=superwing_collate,
        drop_last=False,
    )


def _load_checkpoint(
    *,
    model: torch.nn.Module,
    ckpt_path: Path,
    use_ema: bool,
    device: torch.device,
) -> None:
    """Load model weights, applying EMA shadow when present and requested."""
    payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    if use_ema and "ema_shadow" in payload:
        ema = ExponentialMovingAverage(
            model, decay=float(payload.get("ema_decay", 0.999))
        )
        ema.shadow = {k: v.to(device) for k, v in payload["ema_shadow"].items()}
        ema.apply_to(model)
        log.info("Applied EMA shadow from checkpoint.")
    elif use_ema:
        log.info("Checkpoint has no EMA shadow; using live weights.")


# --------------------------------------------------------------------------- #
# Per-case prediction
# --------------------------------------------------------------------------- #


def _slice_batch_sample(batch: dict, idx: int) -> dict:
    sample: dict = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and not k.endswith("_n"):
            n_key = f"{k}_n"
            if n_key in batch:
                length = int(batch[n_key][idx].item())
                sample[k] = v[idx, :length]
            else:
                sample[k] = v[idx]
        elif isinstance(v, list):
            sample[k] = v[idx]
        else:
            sample[k] = v
    return sample


@torch.no_grad()
def _predict_one_case(
    *,
    model: torch.nn.Module,
    sample: dict,
    device: torch.device,
    precision: str,
    chunk_size: int,
) -> torch.Tensor:
    """Run encode -> predict -> chunked decode for one sample.

    Returns a CPU tensor of shape ``(N_q, C)`` with the decoded field in
    normalised units.
    """
    with get_autocast_context(device, precision):
        context_tokens, cond_global = model.encode_geometry(
            context_pos=sample["context_pos"],
            context_feat=sample["context_feat"],
            gen_params=sample["gen_params"],
        )
        target_coords = model.build_target_token_coords(
            point_positions=sample["target_surface_pos"]
        )
        pred_features = model.predict_field_tokens(
            context_tokens=context_tokens,
            target_positions=target_coords,
            conditions=sample["gen_params"].unsqueeze(0),
        )
    if pred_features.ndim == 3 and int(pred_features.shape[0]) == 1:
        pred_features = pred_features[0]

    target_tokens = TokenSet(
        features=pred_features,
        coords=target_coords,
        mask=torch.ones(
            (int(target_coords.shape[0]),), dtype=torch.bool, device=device
        ),
        global_token=None,
    )
    return model.decode_field_chunked(
        target_tokens=target_tokens,
        cond_global=cond_global,
        query_pos=sample["query_pos"].cpu(),
        query_sdf=sample["query_sdf"].cpu(),
        chunk_size=int(chunk_size),
        precision=str(precision),
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


@hydra.main(
    config_path="conf",
    config_name="config_inference",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point — see module docstring."""
    DistributedManager.initialize()
    dm = DistributedManager()
    world_size = dm.world_size
    rank = dm.rank
    is_main = rank == 0
    device = dm.device

    set_seed(int(cfg.seed))
    hydra_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir = hydra_dir / cfg.output_dir
    plots_dir = out_dir / "plots"
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

    # Build the split/normalization artifacts once on rank 0; other ranks wait
    # at the barrier, then read the finished files.
    if is_main:
        split_path, stats_path = _ensure_superwing_artifacts(cfg.data)
    if world_size > 1:
        dist.barrier()
    if not is_main:
        split_path, stats_path = _ensure_superwing_artifacts(cfg.data)
    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)
    target_mean = np.asarray(stats["target_mean"], dtype=np.float32)
    target_std = np.asarray(stats["target_std"], dtype=np.float32)

    loader = _build_test_loader(
        cfg.data,
        split_manifest_path=split_path,
        normalization_stats_path=stats_path,
        rank=rank,
        world_size=world_size,
    )
    n_local = len(loader.sampler) if loader.sampler is not None else len(loader.dataset)
    if world_size > 1:
        log.info(
            "Test cases: %d total; rank %d/%d owns %d cases.",
            len(loader.dataset),
            rank,
            world_size,
            n_local,
        )
    else:
        log.info("Test cases: %d", len(loader.dataset))

    model = hydra.utils.instantiate(cfg.model).to(device).eval()
    ckpt_path = Path(str(cfg.checkpoint))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    _load_checkpoint(
        model=model, ckpt_path=ckpt_path, use_ema=bool(cfg.use_ema), device=device
    )

    pred_fields: list[np.ndarray] = []
    target_fields: list[np.ndarray] = []
    case_ids: list[str] = []
    gen_params_list: list[np.ndarray] = []
    solver_coeffs_list: list[np.ndarray] = []
    surface_coeffs_list: list[np.ndarray] = []
    aoa_list: list[float] = []
    mach_list: list[float] = []
    geom_idx_list: list[int] = []
    sample_idx_list: list[int] = []
    ref_area_list: list[float] = []
    half_span_list: list[float] = []
    origingeom_list: list[np.ndarray] = []
    global_idx_list: list[int] = []

    n_plots = int(cfg.num_plots)
    H, W = SUPERWING_GRID_SHAPE

    for local_idx, batch in enumerate(loader):
        # Global position in the full (unsharded) test order — used to restore
        # the original case ordering after the ranks are gathered.
        global_idx = rank + local_idx * world_size
        batch = move_batch_to_device(batch, device)
        sample = _slice_batch_sample(batch, 0)
        pred_flat = _predict_one_case(
            model=model,
            sample=sample,
            device=device,
            precision=str(cfg.precision),
            chunk_size=int(cfg.decoder_chunk_size),
        )

        # (N_q, C) flat -> (C, H, W) grid. This assumes the ``query_pos`` used
        # for ``eval_full_grid_query=True`` is emitted in row-major (H, W) order
        # so the flat predictions reshape back to the grid correctly; the
        # SuperWing dataset guarantees that ordering (SUPERWING_GRID_SHAPE). A
        # different query ordering would silently scramble the output grid.
        pred_norm_chw = (
            pred_flat.detach().cpu().numpy().reshape(H, W, -1).transpose(2, 0, 1)
        )
        pred_chw = denormalize_field(
            pred_norm_chw, target_mean=target_mean, target_std=target_std
        )
        target_chw = sample["target_full"].detach().cpu().numpy()

        pred_fields.append(pred_chw)
        target_fields.append(target_chw)
        case_ids.append(str(sample["case_id"]))
        gen_params_list.append(sample["gen_params"].detach().cpu().numpy())
        solver_coeffs_list.append(sample["solver_coeffs"].detach().cpu().numpy())
        surface_coeffs_list.append(sample["surface_coeffs"].detach().cpu().numpy())
        aoa_list.append(float(sample["aoa_deg"]))
        mach_list.append(float(sample["mach"]))
        geom_idx_list.append(int(sample["geom_idx"]))
        sample_idx_list.append(int(sample["sample_idx"]))
        ref_area_list.append(float(sample["ref_area"]))
        half_span_list.append(float(sample["half_span"]))
        origingeom_list.append(sample["origingeom_full"].detach().cpu().numpy())
        global_idx_list.append(global_idx)

        if (local_idx + 1) % 20 == 0:
            log.info("Processed %d / %d local cases", local_idx + 1, n_local)

    # This rank's contribution, stacked into arrays keyed like the final dump.
    shard_payload = {
        "pred_field": np.stack(pred_fields, axis=0),
        "target_field": np.stack(target_fields, axis=0),
        "case_ids": np.asarray(case_ids),
        "gen_params": np.stack(gen_params_list, axis=0),
        "solver_coeffs": np.stack(solver_coeffs_list, axis=0),
        "surface_coeffs": np.stack(surface_coeffs_list, axis=0),
        "aoa_deg": np.asarray(aoa_list, dtype=np.float32),
        "mach": np.asarray(mach_list, dtype=np.float32),
        "geom_idx": np.asarray(geom_idx_list, dtype=np.int64),
        "sample_idx": np.asarray(sample_idx_list, dtype=np.int64),
        "ref_area": np.asarray(ref_area_list, dtype=np.float32),
        "half_span": np.asarray(half_span_list, dtype=np.float32),
        "origingeom": np.stack(origingeom_list, axis=0),
    }
    global_indices = np.asarray(global_idx_list, dtype=np.int64)

    # Gather every rank's payload onto all ranks (NCCL supports all_gather, not
    # gather); only rank 0 uses the result to write the single dump.
    if world_size > 1:
        gathered: list = [None] * world_size
        dist.all_gather_object(gathered, (global_indices, shard_payload))
    else:
        gathered = [(global_indices, shard_payload)]

    if is_main:
        order = np.argsort(
            np.concatenate([g[0] for g in gathered], axis=0), kind="stable"
        )
        merged = {
            key: np.concatenate([g[1][key] for g in gathered], axis=0)[order]
            for key in shard_payload
        }

        npz_path = out_dir / "predictions.npz"
        np.savez_compressed(
            npz_path, **merged, target_mean=target_mean, target_std=target_std
        )
        log.info("Saved predictions to %s (%d cases)", npz_path, order.shape[0])

        for i in range(min(n_plots, int(merged["case_ids"].shape[0]))):
            case_id = str(merged["case_ids"][i])
            written = plot_surface_field(
                predicted=merged["pred_field"][i],
                target=merged["target_field"][i],
                output_dir=plots_dir,
                case_id=case_id,
            )
            log.info("Plotted case %s -> %d PNGs", case_id, len(written))

    DistributedManager.cleanup()


if __name__ == "__main__":
    main()
