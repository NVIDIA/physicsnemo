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

"""Field-GP inference on raw VTK surfaces: per-point fields + per-point UQ.

Reads boundary VTP files, runs the GeoTransolver backbone in chunks to extract
per-point features, feeds them to the trained :class:`FieldVariationalGPHead`, and writes
the GP posterior **mean** (the field prediction) *and* **standard deviation**
(the per-point uncertainty) for all four surface channels back to a VTP.  Color
the resulting mesh by any ``*Std`` array to see which car regions are most
uncertain — produced in a single forward pass.

Usage::

    python inference_field_gp.py \
        +vtk_inference.input_dir=/path/to/runs \
        +vtk_inference.output_dir=/path/to/output \
        +vtk_inference.air_density=1.2050 \
        +vtk_inference.stream_velocity=30.0 \
        ++checkpoint_dir=/path/to/runs/<run_id>/checkpoints_field_gp

The '+' prefix adds new config keys; '++' overrides existing ones.  Expects the
same input directory layout as ``inference_on_vtk.py``.
"""

from __future__ import annotations

import collections
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np
import omegaconf
import pyvista as pv
import torch
import torchinfo
from jaxtyping import Float
from omegaconf import DictConfig

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

from train import (
    Precision,
    cast_precisions,
    get_autocast_context,
    update_model_params_for_fp8,
    validate_precision,
)
from field_gp_utils import probe_feature_dim

# Reuse the VTK I/O + datapipe helpers from the deterministic VTK inference.
from inference_on_vtk import build_data_dict, create_datapipe

if TYPE_CHECKING:
    # Annotation only: the head is built by hydra from cfg.gp_head, and importing
    # it eagerly would trade its own "install gpytorch" message for an
    # ImportError on the name.
    from physicsnemo.experimental.uq import FieldVariationalGPHead

# physicsnemo's ``.mdlus`` checkpoints store the model's instantiation arguments
# alongside its weights, and those arrive from hydra as OmegaConf nodes rather
# than plain containers. ``torch.load`` runs with ``weights_only=True``, which
# refuses to unpickle any type not on its allowlist, so loading a trained
# backbone fails on the config until these are registered. Every training and
# inference script in this directory carries the same block.
torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
torch.serialization.add_safe_globals([Any])
torch.serialization.add_safe_globals([list])
torch.serialization.add_safe_globals([collections.defaultdict])
torch.serialization.add_safe_globals([dict])
torch.serialization.add_safe_globals([int])
torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
torch.serialization.add_safe_globals([omegaconf.base.Metadata])


@torch.no_grad()
def field_gp_predict_full_mesh(
    batch: dict,
    model: torch.nn.Module,
    head: FieldVariationalGPHead,
    chunk_size: int,
    precision: Precision,
) -> tuple[Float[torch.Tensor, "points tasks"], Float[torch.Tensor, "points tasks"]]:
    """Run the backbone in chunks, GP-predict per chunk, stitch to full mesh.

    A full boundary mesh does not fit in one forward pass, so the points are
    shuffled, split into chunks, and the predictions scattered back into mesh
    order.  The shuffle matches training, where the backbone's global attention
    also sees a random subset of the mesh rather than a contiguous region; it
    comes off torch's global RNG, so two runs over the same mesh agree only if
    the process is seeded.

    Parameters
    ----------
    batch : dict
        One geometry from the datapipe, full mesh (no point subsampling).
    model : torch.nn.Module
        GeoTransolver backbone accepting ``return_point_features=True``.
    head : FieldVariationalGPHead
        Trained GP head, in eval mode.
    chunk_size : int
        Points per chunk.
    precision : Precision
        Precision for the backbone forward pass.

    Returns
    -------
    tuple of Float[torch.Tensor, "points tasks"]
        ``(mean, std)`` in the GP's normalized target space, on the CPU and
        ordered to match the input mesh cells.
    """
    N = batch["embeddings"].shape[1]
    indices = torch.randperm(N, device=batch["fx"].device)
    index_blocks = torch.split(indices, chunk_size)

    features = cast_precisions(batch["fx"], precision)
    geometry = (
        cast_precisions(batch["geometry"], precision) if "geometry" in batch else None
    )

    mean_blocks: list[torch.Tensor] = []
    std_blocks: list[torch.Tensor] = []
    for index_block in index_blocks:
        local_emb = cast_precisions(batch["embeddings"][:, index_block], precision)
        local_pos = local_emb[:, :, :3]
        with get_autocast_context(precision):
            _, point_features = model(
                global_embedding=features,
                local_embedding=local_emb,
                geometry=geometry,
                local_positions=local_pos,
                return_point_features=True,
            )
        pred = head.predict(point_features)
        mean_blocks.append(pred.mean.squeeze(0).cpu())
        std_blocks.append(pred.variance.clamp_min(0).sqrt().squeeze(0).cpu())

    mean_stitched = torch.cat(mean_blocks, dim=0)
    std_stitched = torch.cat(std_blocks, dim=0)

    inverse = torch.empty_like(indices, device="cpu")
    inverse[indices.cpu()] = torch.arange(N)
    return mean_stitched[inverse], std_stitched[inverse]


class CheckpointError(RuntimeError):
    """No usable checkpoint was found, which no individual run can recover from."""


def _require_checkpoint_files(checkpoint_dir: str, epoch: int | None) -> None:
    """Raise unless both the backbone and head checkpoint files are present.

    ``load_checkpoint`` returns 0 instead of raising when it finds nothing, so
    without this inference would run to completion on randomly initialized
    weights and write plausible-looking VTK output.
    """
    tag = "*" if epoch is None else str(epoch)
    patterns = (f"GeoTransolver.0.{tag}.mdlus", f"FieldVariationalGPHead.0.{tag}.pt")
    missing = [p for p in patterns if not list(Path(checkpoint_dir).glob(p))]
    if missing:
        raise CheckpointError(
            f"No usable field-GP checkpoint in {checkpoint_dir}: nothing matches "
            f"{missing}.  Pass ++checkpoint_dir=... (and +checkpoint_epoch=N) "
            "pointing at a trained run."
        )


def write_field_gp_predictions_to_vtk(
    vtp_path: str | Path,
    output_path: str | Path,
    mean_norm: Float[torch.Tensor, "points tasks"],
    std_norm: Float[torch.Tensor, "points tasks"],
    surface_factors: dict[str, torch.Tensor],
) -> None:
    """Write GP mean field + per-point std (UQ) to a VTP.

    Adds these cell-data arrays:

    * ``PredictedPressure`` / ``PredictedWallShearStress`` — GP mean field
      (physical units).
    * ``StdPressure`` / ``StdWallShearStress`` — GP std in physical units.

    Everything is written in physical units.  The targets are standardized
    *physical* fields — the DrivAerStar statistics are Pa-scale — so
    unstandardizing is the whole conversion and no dynamic-pressure factor
    applies.  Note the four channels carry very different normalization scales,
    so the physical stds are not comparable across channels; divide each by its
    ``surface_factors["std"]`` entry to recover the dimensionless form when a
    cross-channel UQ map is what you want.

    The unstandardizing is done on the tensors, but stays on the host: PyVista
    holds cell data as NumPy arrays, so the results have to arrive on the CPU
    whatever happens, and both operands are already there — the predictions were
    moved off the device chunk by chunk to keep a full mesh's worth of points out
    of GPU memory.  It is one affine pass over four columns, which costs nothing
    next to reading and writing the mesh itself.
    """
    vtp_path = Path(vtp_path)
    if not vtp_path.is_file():
        raise FileNotFoundError(f"No surface mesh to annotate at {vtp_path}")
    mesh = pv.read(vtp_path)
    output_mesh = mesh.copy()

    field_mean = surface_factors["mean"].detach().cpu().reshape(-1)
    field_std = surface_factors["std"].detach().cpu().reshape(-1)

    # Mean field: unstandardize (x * std + mean) back to physical units.
    mean_unscaled = (mean_norm * field_std + field_mean).numpy()
    output_mesh.cell_data["PredictedPressure"] = mean_unscaled[:, 0]
    output_mesh.cell_data["PredictedWallShearStress"] = mean_unscaled[:, 1:4]

    # Std is a scale: maps through the affine unstandardize as std * field_std.
    std_unscaled = (std_norm * field_std).numpy()
    output_mesh.cell_data["StdPressure"] = std_unscaled[:, 0]
    output_mesh.cell_data["StdWallShearStress"] = std_unscaled[:, 1:4]

    output_mesh.save(output_path)


def inference_field_gp(cfg: DictConfig) -> None:
    """Run field-GP per-point UQ inference over a directory of VTK runs."""
    DistributedManager.initialize()
    dist_manager = DistributedManager()
    logger = RankZeroLoggingWrapper(
        PythonLogger(name="field_gp_vtk_inference"), dist_manager
    )

    cfg, output_pad_size = update_model_params_for_fp8(cfg, logger)
    logger.info(f"Config:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")

    if not cfg.get("vtk_inference", None):
        raise ValueError(
            "vtk_inference config section is required. "
            "Add it via command line with '+vtk_inference.input_dir=...' etc."
        )
    vtk_cfg = cfg.vtk_inference
    if not vtk_cfg.get("input_dir", None):
        raise ValueError("vtk_inference.input_dir is required")
    if not vtk_cfg.get("output_dir", None):
        raise ValueError("vtk_inference.output_dir is required")

    input_dir = Path(vtk_cfg.input_dir)
    output_dir = Path(vtk_cfg.output_dir)
    # Freestream conditions of the dataset being scored, used by build_data_dict
    # to fill the global-parameter channels the backbone was trained with (they
    # play no part in the VTK arrays, which stay in the trained units). The
    # defaults are the DrivAerStar operating point: 1.2050 kg/m^3 at 15 C and
    # 101.325 kPa, and a 30 m/s inlet. Override both for any other dataset.
    air_density = vtk_cfg.get("air_density", 1.2050)
    stream_velocity = vtk_cfg.get("stream_velocity", 30.0)
    run_indices = vtk_cfg.get("run_indices", None)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_mode = cfg.data.mode
    if data_mode != "surface":
        raise ValueError("inference_field_gp currently supports data.mode=surface only")

    device = dist_manager.device
    precision = validate_precision(cfg.precision)

    # ---- Backbone ----
    model = hydra.utils.instantiate(cfg.model, _convert_="partial")
    logger.info(f"\n{torchinfo.summary(model, verbose=0)}")
    model.to(device)
    model.eval()

    # ---- Normalization factors ----
    norm_dir = cfg.data.normalization_dir
    norm_file = str(Path(norm_dir) / "surface_fields_normalization.npz")
    norm_data = np.load(norm_file)
    surface_factors = {
        "mean": torch.from_numpy(norm_data["mean"]).to(device),
        "std": torch.from_numpy(norm_data["std"]).to(device),
    }

    # ---- Datapipe (full mesh, manual chunking) ----
    datapipe = create_datapipe(cfg, data_mode, device, surface_factors, None)
    # Chunk the full mesh at the same point count the head was trained on.
    chunk_size = cfg.data.resolution

    # ---- Locate runs ----
    if run_indices is not None:
        run_dirs = [input_dir / f"run_{idx}" for idx in run_indices]
    else:
        run_dirs = sorted(
            d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("run_")
        )
    this_device_runs = run_dirs[dist_manager.rank :: dist_manager.world_size]
    logger.info(f"Rank {dist_manager.rank} processing {len(this_device_runs)} runs")

    # ---- Build the GP head once (probe feature dim on the first run) ----
    head = None
    # Optional overrides rather than recipe knobs: absent means "derive it".
    checkpoint_dir = cfg.get("checkpoint_dir", None) or (
        f"{cfg.output_dir}/{cfg.run_id}/checkpoints_field_gp"
    )
    checkpoint_epoch = cfg.get("checkpoint_epoch", None)
    _require_checkpoint_files(checkpoint_dir, checkpoint_epoch)

    for run_dir in this_device_runs:
        run_idx = int(run_dir.name.split("_")[1])
        logger.info(f"Processing run {run_idx}: {run_dir}")
        start_time = time.time()

        try:
            data_dict = build_data_dict(
                run_dir=run_dir,
                data_mode=data_mode,
                device=device,
                air_density=air_density,
                stream_velocity=stream_velocity,
                run_idx=run_idx,
            )
            batch = datapipe(data_dict)

            if head is None:
                # A boundary mesh is far larger than one chunk, so probe on a
                # slice of it rather than the whole thing.
                feature_dim = probe_feature_dim(
                    model, batch, precision, max_points=1024
                )
                # n_train only normalizes the training ELBO, so any value works
                # here; cfg.gp_head is what must match the checkpoint.
                head = hydra.utils.instantiate(
                    cfg.gp_head,
                    input_dim=feature_dim,
                    n_train=1,
                    _convert_="all",
                )
                head.to(device)
                loaded_epoch = load_checkpoint(
                    path=checkpoint_dir,
                    models=[model, head],
                    device=device,
                    epoch=checkpoint_epoch,
                )
                if loaded_epoch == 0:
                    raise CheckpointError(
                        f"load_checkpoint restored nothing from {checkpoint_dir} "
                        f"(epoch={checkpoint_epoch}); the models are still randomly "
                        "initialized."
                    )
                logger.info(
                    f"Loaded field-GP checkpoint (epoch {loaded_epoch}) from "
                    f"{checkpoint_dir}; feature_dim={feature_dim}, "
                    f"gp_dim={head.gp_input_dim}"
                )
                model.eval()
                head.eval()
                head.likelihood.eval()

            mean_norm, std_norm = field_gp_predict_full_mesh(
                batch, model, head, chunk_size, precision
            )

            run_output_dir = output_dir / run_dir.name
            run_output_dir.mkdir(parents=True, exist_ok=True)
            vtp_path = run_dir / f"boundary_{run_idx}.vtp"
            if not vtp_path.exists():
                vtp_path = list(run_dir.glob("boundary_*.vtp"))[0]
            output_vtp = run_output_dir / f"field_gp_boundary_{run_idx}.vtp"
            write_field_gp_predictions_to_vtk(
                str(vtp_path),
                str(output_vtp),
                mean_norm,
                std_norm,
                surface_factors,
            )
            elapsed = time.time() - start_time
            logger.info(f"Saved field-GP UQ to {output_vtp} ({elapsed:.2f}s)")

        except CheckpointError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error processing run {run_idx}: {e}")
            import traceback

            traceback.print_exc()
            continue

    logger.info("Field-GP inference complete!")


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="geotransolver_surface_field_gp",
)
def launch(cfg: DictConfig) -> None:
    """Hydra entry point for field-GP VTK inference."""
    inference_field_gp(cfg)


if __name__ == "__main__":
    launch()
