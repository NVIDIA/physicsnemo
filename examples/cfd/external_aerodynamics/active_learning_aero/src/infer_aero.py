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

"""Visualize predictions from a trained surface-CFD checkpoint as VTP point clouds.

Reuses the exact ``AeroDataPool`` + ``build_surface_factors`` plumbing from
the AL training scripts so the model sees identical inputs at inference
time as it did during training (modulo ``resolution=None``, which we force so
the VTPs contain every surface point, not a 60k-subsample).  Predictions are
then un-standardized back to physical units (Pa for pressure, Pa for wall
shear stress) and written alongside ground-truth fields.

Usage (single GPU, container shell)::

    python src/infer_aero.py \\
        --config-name=geotransolver_surface \\
        ++manifest_dir=src/manifests \\
        ++data.geometry_sampling=200000 \\
        ++data.return_mesh_features=true \\
        ++data.include_geometry=true \\
        ++data.broadcast_global_features=false \\
        ++data.reference_scale=[5.0,5.0,5.0] \\
        ++data.physics_nondim.enabled=true \\
        ++data.physics_nondim.U_inf=30.0 \\
        ++data.physics_nondim.rho_inf=1.225 \\
        ++data.physics_nondim.p_inf=0.0 \\
        ++data.physics_nondim.wss_factor=0.00183 \\
        ++data.classes=[SF] \\
        ++run_id=geotransolver/surface/shiftsuv_ceiling_golden_sf \\
        ++infer.n_samples=10

Outputs land under ``runs/<run_id>/inference_vtps/`` -- one ``.vtp`` per
sample plus a ``summary.json`` with per-sample Cd / residual stats.
"""

from __future__ import annotations

import collections
import json
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import torch
from omegaconf import DictConfig, OmegaConf

### Same ``add_safe_globals`` block as ``train_ceiling.py`` -- the
### checkpoint loader pickles a dict that needs these classes whitelisted.
torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
torch.serialization.add_safe_globals([list])
torch.serialization.add_safe_globals([collections.defaultdict])
torch.serialization.add_safe_globals([dict])
torch.serialization.add_safe_globals([int])
torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
torch.serialization.add_safe_globals([omegaconf.base.Metadata])

from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.utils import load_checkpoint  # noqa: E402
from physicsnemo.utils.logging import PythonLogger  # noqa: E402

from utils import cast_precisions, get_autocast_context  # noqa: E402
from aero_physics import (  # noqa: E402
    DRAG_COEFF_SCALE,
    FRONTAL_AREA,
    REFERENCE_DENSITY,
    REFERENCE_VELOCITY,
    compute_drag_from_subsampled_outputs,
    compute_drag_target_from_batch,
)

from data_pool import (  # noqa: E402
    build_pool,
    build_surface_factors,
    load_manifests,
)


def _filter_classes(
    pool_by_class: dict[str, list[int]],
    test_by_class: dict[str, list[int]],
    paths_by_class: dict[str, str],
    keep: list[str] | None,
    logger,
) -> tuple[dict, dict, dict]:
    """Restrict manifests to the listed classes (mirrors train_ceiling)."""
    if keep is None:
        return pool_by_class, test_by_class, paths_by_class
    keep_set = {str(c) for c in keep}
    missing = keep_set - set(paths_by_class.keys())
    if missing:
        raise ValueError(
            f"data.classes requested {sorted(missing)} but manifest_dir "
            f"only provides {sorted(paths_by_class.keys())}"
        )
    pool_by_class = {c: pool_by_class[c] for c in keep_set}
    test_by_class = {c: test_by_class[c] for c in keep_set}
    paths_by_class = {c: paths_by_class[c] for c in keep_set}
    logger.info(
        f"Class filter: keeping {sorted(keep_set)} "
        f"(train={sum(len(v) for v in pool_by_class.values())}, "
        f"val={sum(len(v) for v in test_by_class.values())})"
    )
    return pool_by_class, test_by_class, paths_by_class


def _force_resolution_none(cfg: DictConfig) -> None:
    """Force the surface datapipe to keep every mesh point (no subsampling).

    Done in-place via ``OmegaConf`` so we don't need to thread a separate
    inference config; the rest of the pipeline (``include_normals``,
    ``reference_scale``, ``physics_nondim``, etc.) stays exactly as the
    training run had it.
    """
    OmegaConf.set_struct(cfg, False)
    cfg.data.resolution = None
    OmegaConf.set_struct(cfg, True)


@hydra.main(version_base=None, config_path="conf", config_name="geotransolver_surface")
def main(cfg: DictConfig) -> None:
    """Inference driver: load checkpoint, run on val pool, write .vtp clouds."""
    ### Try to import pyvista lazily -- the host python has no VTK install but
    ### the training container does.  Fail with a clear message if not present.
    try:
        import pyvista as pv  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "pyvista is required for VTP writing.  Run this script inside the "
            "training container (the same one your SLURM script uses)."
        ) from e

    DistributedManager.initialize()
    dm = DistributedManager()
    device = dm.device
    if dm.world_size > 1:
        raise RuntimeError(
            "This script is single-process by design (no DDP).  "
            "Launch with `python ...`, not `torchrun`."
        )
    logger = PythonLogger(name="infer_aero")

    manifest_dir = getattr(cfg, "manifest_dir", None)
    if manifest_dir is None:
        raise ValueError("Must provide ++manifest_dir=src/manifests")

    ### Inference knobs (all opt-in via Hydra overrides):
    ###   ++infer.n_samples=N    -- write VTPs for the first N val samples
    ###   ++infer.save_all=true  -- override n_samples and dump every val sample
    ###   ++infer.indices=[3,7]  -- write VTPs only for these specific val
    ###                             positions (0-indexed within ``val_pool``)
    n_samples = int(OmegaConf.select(cfg, "infer.n_samples", default=5))
    save_all = bool(OmegaConf.select(cfg, "infer.save_all", default=False))
    explicit_indices = OmegaConf.select(cfg, "infer.indices", default=None)
    precision = getattr(cfg, "precision", "float32")

    ### Velocity-ratio correction for the DrivAerML constants baked into
    ### ``gp_utils.compute_drag_target_from_batch``.  This is independent
    ### of normalization: every training recipe so far (Fastback-norm or
    ### physics-nondim, MSE or Huber) routes Cd through the same hardcoded
    ### ``REFERENCE_VELOCITY=40`` formula, so the *logged* Cd is always
    ### off from raw ShiftSUV Cd by ``(40 / U_inf)^2``.  Configurable via
    ### ``++infer.u_inf`` so the same script works for any chain.  Default
    ### picks up ``data.physics_nondim.U_inf`` when set (golden runs), else
    ### falls back to ``30.0`` (the ShiftSUV freestream).  Set
    ### ``++infer.u_inf=40.0`` to disable the correction (matches DrivAerML
    ### datasets where the constants are already right).
    pn = OmegaConf.select(cfg, "data.physics_nondim", default=None)
    if pn is not None and bool(OmegaConf.select(pn, "enabled", default=False)):
        default_u_inf = float(OmegaConf.select(pn, "U_inf", default=30.0))
    else:
        default_u_inf = 30.0
    u_inf = float(OmegaConf.select(cfg, "infer.u_inf", default=default_u_inf))
    cd_raw_correction = (REFERENCE_VELOCITY / u_inf) ** 2
    logger.info(
        f"Cd raw-units correction: U_inf={u_inf} (REF={REFERENCE_VELOCITY}) "
        f"-> multiply logged Cd by {cd_raw_correction:.4f}"
    )

    _force_resolution_none(cfg)

    pool_by_class, test_by_class, paths_by_class = load_manifests(manifest_dir)
    keep = OmegaConf.select(cfg, "data.classes", default=None)
    pool_by_class, test_by_class, paths_by_class = _filter_classes(
        pool_by_class, test_by_class, paths_by_class, keep, logger
    )

    surface_factors = build_surface_factors(cfg, device, logger)
    val_pool = build_pool(cfg.data, paths_by_class, test_by_class, surface_factors)
    logger.info(f"Val pool: {val_pool.total_samples} samples (full mesh, no subsample)")

    ### Model -- non-DDP wrapper because this is a single-process script.
    ### ``load_checkpoint`` handles both DDP and bare modules; the saved
    ### ``GeoTransolver.X.mdlus`` was unwrapped on save so loading into a
    ### plain ``nn.Module`` is correct.
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(device)
    ### Checkpoint resolution:
    ###   * if ``++checkpoint_dir=...`` is set, use it verbatim (lets us
    ###     point at AL per-round dirs like
    ###     ``runs/<run_id>/<strategy>/checkpoint_round_<N>``);
    ###   * otherwise derive ``runs/<run_id>/checkpoints`` to match the
    ###     ceiling trainer's layout.
    explicit_ckpt = OmegaConf.select(cfg, "checkpoint_dir", default=None)
    if explicit_ckpt is not None:
        checkpoint_dir = str(explicit_ckpt)
    else:
        checkpoint_dir = f"{cfg.output_dir}/{cfg.run_id}/checkpoints"
    loaded_epoch = load_checkpoint(
        path=str(checkpoint_dir),
        models=model,
        device=device,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Loaded checkpoint from {checkpoint_dir} @ epoch={loaded_epoch} "
        f"({n_params:,} params)"
    )

    ### Output layout: one dir per run, timestamped subdir per inference run
    ### so re-runs don't clobber each other.  For AL chains where multiple
    ### per-round checkpoints share a single ``run_id``, the explicit ckpt
    ### dir's basename (e.g. ``checkpoint_round_64``) is folded into the
    ### stamp so different rounds land in distinct dirs.
    out_dir = Path(cfg.output_dir) / cfg.run_id / "inference_vtps"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if explicit_ckpt is not None:
        ckpt_tag = Path(str(explicit_ckpt)).name
        out_dir = out_dir / f"{ckpt_tag}_epoch{loaded_epoch}_{stamp}"
    else:
        out_dir = out_dir / f"epoch{loaded_epoch}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing VTPs to {out_dir}")

    ### Pick which val positions to visualize.
    n_total = val_pool.total_samples
    if explicit_indices is not None:
        sample_idxs = [int(i) for i in explicit_indices]
    elif save_all:
        sample_idxs = list(range(n_total))
    else:
        sample_idxs = list(range(min(n_samples, n_total)))
    logger.info(f"Inferring on {len(sample_idxs)} sample(s): {sample_idxs}")

    summary: list[dict] = []

    for n, i in enumerate(sample_idxs):
        if i >= n_total:
            logger.warn(f"Skipping i={i}: out of range (val_pool has {n_total})")
            continue

        flat_idx = int(val_pool.train_indices[i].item())
        ds_idx, local_idx = val_pool._flat_to_local[flat_idx]
        cls_label = val_pool.class_of(flat_idx)

        t0 = time.time()

        ### CRITICAL: clone raw positions BEFORE running the datapipe.
        ### ``preprocess_surface_data`` does ``positions -= center_of_mass``
        ### in-place when ``resolution=None``, which would otherwise mutate
        ### the cached zarr tensor inside ``CAEDataset``.  Cloning the
        ### tensor we hand back to pyvista guarantees we report the raw
        ### simulation coordinates (in meters), not the centered/scaled
        ### version the model consumed.
        raw_sample = val_pool._raw_datasets[ds_idx][local_idx]
        raw_positions = raw_sample["surface_mesh_centers"].detach().clone()

        batch = val_pool._datapipes[ds_idx](raw_sample)
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        features = cast_precisions(batch["fx"], precision)
        embeddings = cast_precisions(batch["embeddings"], precision)
        geometry = (
            cast_precisions(batch["geometry"], precision)
            if "geometry" in batch
            else None
        )
        local_positions = embeddings[:, :, :3]

        with torch.no_grad(), get_autocast_context(precision):
            outputs = model(
                global_embedding=features,
                local_embedding=embeddings,
                geometry=geometry,
                local_positions=local_positions,
            )

        ### Un-standardize.  In the physics-nondim mode this gives Pa for
        ### pressure (mean=0, std=q_inf  =>  Cp * q_inf = p) and Pa for each
        ### WSS component (mean=0, std=q_inf * wss_factor).  Match the dtype
        ### of the factors so autocast outputs don't trip an op-type error.
        ###
        ### Shape note: ``TransolverDataPipe.__call__`` already adds a
        ### batch dim, so model I/O is ``(1, N, 4)``.  We squeeze that out
        ### before broadcasting against per-channel ``mean/std`` ``(4,)``
        ### tensors so the resulting per-point arrays are ``(N, 4)`` and
        ### line up 1:1 with ``raw_positions (N, 3)`` for the VTP.
        pred_norm = outputs.squeeze(0).float()
        true_norm = batch["fields"].squeeze(0).float()
        mean = surface_factors["mean"].float()
        std = surface_factors["std"].float()
        pred_phys = pred_norm * std + mean
        true_phys = true_norm * std + mean

        ### Cd (same path the training validator uses) -- multiply through
        ### ``DRAG_COEFF_SCALE`` to recover the *logged* Cd, then apply the
        ### velocity-ratio correction to get the *raw* simulation Cd.
        pred_cd_logged = (
            compute_drag_from_subsampled_outputs(
                outputs, batch, surface_factors, device
            ).item()
            * DRAG_COEFF_SCALE
        )
        true_cd_logged = (
            compute_drag_target_from_batch(batch, surface_factors, device).item()
            * DRAG_COEFF_SCALE
        )
        pred_cd_raw = pred_cd_logged * cd_raw_correction
        true_cd_raw = true_cd_logged * cd_raw_correction

        ### Per-sample field residual stats (in Pa) for quick sanity printout.
        p_true = true_phys[..., 0].cpu().numpy()
        p_pred = pred_phys[..., 0].cpu().numpy()
        wss_true = true_phys[..., 1:4].cpu().numpy()
        wss_pred = pred_phys[..., 1:4].cpu().numpy()
        p_res = p_true - p_pred
        wss_res = wss_true - wss_pred

        ### Build a point-cloud VTP.  Raw positions live in meters; field
        ### units are Pa throughout.
        pts = raw_positions.cpu().numpy().astype(np.float32)
        if pts.shape[0] != p_true.shape[0]:
            ### Should never happen with resolution=None, but guard anyway:
            ### a mismatch means the datapipe subsampled despite our override
            ### and the per-point arrays are inconsistent with the geometry.
            raise RuntimeError(
                f"Point count mismatch: positions={pts.shape[0]} fields="
                f"{p_true.shape[0]}.  Check that data.resolution override "
                "took effect."
            )

        cloud = pv.PolyData(pts)
        cloud.point_data["pressure_true_Pa"] = p_true
        cloud.point_data["pressure_pred_Pa"] = p_pred
        cloud.point_data["pressure_residual_Pa"] = p_res
        cloud.point_data["wss_true_Pa"] = wss_true
        cloud.point_data["wss_pred_Pa"] = wss_pred
        cloud.point_data["wss_residual_Pa"] = wss_res

        ### Optional: a couple of scalar magnitudes are handy in Paraview.
        cloud.point_data["wss_true_mag_Pa"] = np.linalg.norm(wss_true, axis=1)
        cloud.point_data["wss_pred_mag_Pa"] = np.linalg.norm(wss_pred, axis=1)
        cloud.point_data["wss_residual_mag_Pa"] = np.linalg.norm(wss_res, axis=1)

        out_path = out_dir / f"flat{flat_idx:04d}_{cls_label}.vtp"
        cloud.save(str(out_path))

        elapsed = time.time() - t0
        rec = {
            "flat_idx": flat_idx,
            "class": cls_label,
            "n_points": int(pts.shape[0]),
            "true_cd_logged": true_cd_logged,
            "pred_cd_logged": pred_cd_logged,
            "true_cd_raw": true_cd_raw,
            "pred_cd_raw": pred_cd_raw,
            "cd_residual_raw": true_cd_raw - pred_cd_raw,
            "pressure_residual_Pa_rms": float(np.sqrt(np.mean(p_res**2))),
            "pressure_residual_Pa_max_abs": float(np.max(np.abs(p_res))),
            "wss_residual_Pa_rms": float(
                np.sqrt(np.mean(np.linalg.norm(wss_res, axis=1) ** 2))
            ),
            "vtp_path": str(out_path),
            "infer_seconds": elapsed,
        }
        summary.append(rec)

        logger.info(
            f"[{n + 1}/{len(sample_idxs)}] flat_idx={flat_idx:>4d} class={cls_label} "
            f"N={pts.shape[0]} | Cd_raw true={true_cd_raw:+.4f} pred={pred_cd_raw:+.4f} "
            f"resid={true_cd_raw - pred_cd_raw:+.4f} | "
            f"p_res_rms={rec['pressure_residual_Pa_rms']:.2f}Pa "
            f"wss_res_rms={rec['wss_residual_Pa_rms']:.3f}Pa | "
            f"{elapsed:.2f}s -> {out_path.name}"
        )

    ### Aggregate summary for offline analysis.  We write JSON next to the
    ### VTPs so the whole inference run is self-describing.
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "checkpoint_epoch": loaded_epoch,
                "checkpoint_dir": str(checkpoint_dir),
                "run_id": cfg.run_id,
                "cd_raw_correction": cd_raw_correction,
                "u_inf_for_cd_correction": u_inf,
                "drag_coeff_scale": DRAG_COEFF_SCALE,
                "drivaerml_constants": {
                    "FRONTAL_AREA": FRONTAL_AREA,
                    "REFERENCE_DENSITY": REFERENCE_DENSITY,
                    "REFERENCE_VELOCITY": REFERENCE_VELOCITY,
                },
                "samples": summary,
            },
            f,
            indent=2,
        )
    logger.info(f"Summary written to {summary_path}")
    logger.info(f"Done. {len(summary)} VTPs in {out_dir}")


if __name__ == "__main__":
    main()
