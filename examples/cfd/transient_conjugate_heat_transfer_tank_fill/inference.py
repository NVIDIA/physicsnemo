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

"""Inference script for the transient conjugate heat-transfer DoMINO model.

This script loads raw CFD simulation dumps directly (``sim_X`` folders with
``*.vtu``/``*.boundaries.vtu`` files), runs the trained model, and writes one VTK
file per timestep containing both predictions and (optionally) ground truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import hydra
import numpy as np
import pyvista as pv
import torch
from omegaconf import DictConfig, OmegaConf

from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataPipe
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.models.domino.model import DoMINO
from physicsnemo.models.domino.utils.utils import dict_to_device
from physicsnemo.utils import load_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

# Local preprocessing helpers
from process_data import process_sim, list_steps
from utils import (
    component_layout,
    count_global_features,
    count_variable_components,
    load_scaling,
    resolve_path,
    split_fields_by_step,
    to_torch_sample,
)


def write_surface_step(
    mesh_path: Path,
    out_path: Path,
    pred_fields: Dict[str, np.ndarray],
    gt_fields: Dict[str, np.ndarray] | None,
) -> None:
    """Write a surface step to a file."""

    # Read the surface mesh and triangulate it
    mesh = pv.read(mesh_path).extract_geometry().triangulate()

    # Write the predicted and ground truth fields to the mesh
    n_cells = mesh.n_cells
    for name, arr in pred_fields.items():
        arr_np = np.asarray(arr)
        if arr_np.shape[0] != n_cells:
            raise ValueError(
                f"Surface field '{name}' has {arr_np.shape[0]} cells, expected {n_cells}."
            )
        mesh.cell_data[f"pred_{name}"] = arr_np
    if gt_fields is not None:
        for name, arr in gt_fields.items():
            arr_np = np.asarray(arr)
            if arr_np.shape[0] != n_cells:
                raise ValueError(
                    f"Surface field '{name}' has {arr_np.shape[0]} cells, expected {n_cells}."
                )
            mesh.cell_data[f"gt_{name}"] = arr_np

    # Save the mesh to a file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(out_path)


def write_volume_step(
    step_dir: Path,
    region_names: Sequence[str],
    out_dir: Path,
    pred_fields: Dict[str, np.ndarray],
    gt_fields: Dict[str, np.ndarray] | None,
) -> None:
    """Write a volume step to a file."""

    # Loop through all volume regions
    start = 0
    for region in region_names:
        # Read the volume mesh
        mesh_path = step_dir / region
        mesh = pv.read(mesh_path)

        # Write the predicted and ground truth fields to the mesh
        n_cells = mesh.n_cells
        end = start + n_cells
        for name, arr in pred_fields.items():
            region_arr = np.asarray(arr)[start:end]
            if region_arr.shape[0] != n_cells:
                raise ValueError(
                    f"Volume field '{name}' region '{region}' has {region_arr.shape[0]} cells, expected {n_cells}."
                )
            mesh.cell_data[f"pred_{name}"] = region_arr
        if gt_fields is not None:
            for name, arr in gt_fields.items():
                region_arr = np.asarray(arr)[start:end]
                if region_arr.shape[0] != n_cells:
                    raise ValueError(
                        f"Volume field '{name}' region '{region}' has {region_arr.shape[0]} cells, expected {n_cells}."
                    )
                mesh.cell_data[f"gt_{name}"] = region_arr

        # Save the mesh to a file
        out_path = out_dir / region / f"{step_dir.name}.vtu"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.save(out_path)
        start = end


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)

    # Initialize distributed manager and logger
    DistributedManager.initialize()
    dist = DistributedManager()
    logger = RankZeroLoggingWrapper(PythonLogger("inference"), dist)

    # Get model type for processing
    model_type = str(cfg.model.model_type).lower()
    include_surface = model_type in {"surface", "combined"}
    include_volume = model_type in {"volume", "combined"}
    if not include_surface and not include_volume:
        raise ValueError(
            "At least one of surface or volume predictions must be enabled."
        )

    # Get target steps
    target_steps = int(cfg.data.get("future_steps", 1))

    # Resolve paths
    raw_root = resolve_path(cfg.data.raw_dir)
    output_root = resolve_path(cfg.inference.output_dir)
    stats_dir = resolve_path(
        cfg.inference.get("stats_dir", Path(cfg.data.processed_dir) / "stats")
    )
    checkpoint_dir_cfg = cfg.inference.checkpoint or cfg.train.checkpoint_dir
    checkpoint_dir = resolve_path(checkpoint_dir_cfg)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory '{checkpoint_dir}' not found. "
            "Set inference.checkpoint or train.checkpoint_dir to a valid path."
        )

    # Get surface and volume specifications
    surface_specs = (
        list(cfg.variables.surface.solution.items()) if include_surface else []
    )
    volume_specs = list(cfg.variables.volume.solution.items()) if include_volume else []
    global_params = cfg.variables.get("global_parameters", {})
    global_order = list(global_params.keys())
    global_reference = [
        global_params[name].get("reference", 1.0) for name in global_order
    ]

    # Load scaling factors
    volume_scaling, surface_scaling = load_scaling(
        stats_dir, include_surface, include_volume
    )
    if cfg.model.normalization and (
        (include_surface and surface_scaling is None)
        or (include_volume and volume_scaling is None)
    ):
        raise FileNotFoundError(
            f"Missing scaling factors under {stats_dir}. Run preprocessing or disable normalization."
        )

    # Make data pipeline
    use_gpu = str(cfg.inference.device).lower() != "cpu" and torch.cuda.is_available()
    datapipe = DoMINODataPipe(
        input_path=raw_root,
        phase="test",
        model_type=model_type,
        grid_resolution=cfg.model.interp_res,
        normalize_coordinates=True,
        sampling=False,
        sample_in_bbox=False,
        volume_points_sample=cfg.model.volume_points_sample,
        surface_points_sample=cfg.model.surface_points_sample,
        geom_points_sample=cfg.model.geom_points_sample,
        volume_factors=volume_scaling,
        surface_factors=surface_scaling,
        scaling_type=cfg.model.normalization,
        bounding_box_dims=cfg.data.bounding_box,
        bounding_box_dims_surf=cfg.data.bounding_box_surface,
        volume_sample_from_disk=False,
        num_surface_neighbors=cfg.model.num_neighbors_surface,
        surface_sampling_algorithm=cfg.model.surface_sampling_algorithm,
        gpu_preprocessing=use_gpu and cfg.data.gpu_preprocessing,
        gpu_output=use_gpu and cfg.data.gpu_output,
    )

    # Initialize model
    device = datapipe.output_device if use_gpu else torch.device("cpu")
    num_global_features = count_global_features(cfg, target_steps)
    output_features_surf = (
        count_variable_components(cfg.variables.surface.solution) * target_steps
        if include_surface
        else None
    )
    output_features_vol = (
        count_variable_components(cfg.variables.volume.solution) * target_steps
        if include_volume
        else None
    )
    model = DoMINO(
        input_features=3,
        output_features_vol=output_features_vol,
        output_features_surf=output_features_surf,
        global_features=num_global_features,
        model_parameters=cfg.model,
    ).to(device)

    # Load checkpoint
    load_checkpoint(str(checkpoint_dir), models=model, device=device)
    model.eval()

    # Get simulation directories
    sim_dirs: List[Path] = []
    for entry in sorted([p for p in raw_root.iterdir() if p.is_dir()]):
        subdirs = sorted([child for child in entry.iterdir() if child.is_dir()])
        sim_dirs.extend(subdirs or [entry])
    logger.info(f"Running inference on {len(sim_dirs)} simulations found in {raw_root}")

    # Get surface and volume layouts
    surface_layout = (
        component_layout(cfg.variables.surface.solution) if include_surface else []
    )
    volume_layout = (
        component_layout(cfg.variables.volume.solution) if include_volume else []
    )

    # Process each simulation
    for sim_dir in sim_dirs:
        # Process simulation
        logger.info(f"Processing simulation '{sim_dir.name}'")
        sample, meta = process_sim(
            sim_dir,
            surface_specs,
            volume_specs,
            include_surface,
            include_volume,
            target_steps,
            global_order,
            global_reference,
        )

        # Convert sample to torch tensor and batch
        torch_sample = to_torch_sample(sample, datapipe.preproc_device)
        batch = datapipe(torch_sample)
        batch = dict_to_device(batch, device)

        # Run model
        with torch.no_grad():
            pred_vol, pred_surf = model(batch)
            pred_vol, pred_surf = datapipe.unscale_model_outputs(pred_vol, pred_surf)
            gt_vol, gt_surf = datapipe.unscale_model_outputs(
                batch.get("volume_fields"), batch.get("surface_fields")
            )

        # Split fields by step
        steps = list_steps(sim_dir)
        usable_steps = min(target_steps, max(0, len(steps) - 1))
        if usable_steps == 0:
            logger.warning(f"No timesteps found for {sim_dir}, skipping.")
            continue
        if pred_surf is not None:
            surf_pred_steps = split_fields_by_step(
                pred_surf.squeeze(0).cpu().numpy(), surface_layout, usable_steps
            )
            surf_gt_steps = (
                split_fields_by_step(
                    gt_surf.squeeze(0).cpu().numpy(), surface_layout, usable_steps
                )
                if gt_surf is not None and cfg.inference.get("write_ground_truth", True)
                else [None] * usable_steps
            )
        else:
            surf_pred_steps = []
            surf_gt_steps = []
        if pred_vol is not None:
            vol_pred_steps = split_fields_by_step(
                pred_vol.squeeze(0).cpu().numpy(), volume_layout, usable_steps
            )
            vol_gt_steps = (
                split_fields_by_step(
                    gt_vol.squeeze(0).cpu().numpy(), volume_layout, usable_steps
                )
                if gt_vol is not None and cfg.inference.get("write_ground_truth", True)
                else [None] * usable_steps
            )
        else:
            vol_pred_steps = []
            vol_gt_steps = []

        # Get step count
        step_count = usable_steps
        if surf_pred_steps:
            step_count = min(step_count, len(surf_pred_steps))
        if vol_pred_steps:
            step_count = min(step_count, len(vol_pred_steps))
        if step_count == 0:
            logger.warning(f"No usable steps after splitting for {sim_dir}, skipping.")
            continue
        if step_count < usable_steps:
            logger.warning(
                f"Only {step_count} steps available for {sim_dir} (requested {usable_steps}); writing available steps."
            )

        # Get region names
        region_names: Sequence[str] = []
        if include_volume:
            init_step = steps[0]
            region_names = [
                p.name
                for p in sorted(init_step.glob("*.vtu"))
                if not p.name.endswith(".boundaries.vtu")
            ]

        sim_out = output_root / sim_dir.name

        # Write inference results
        for idx in range(step_count):
            step_dir = steps[idx + 1]
            step_tag = step_dir.name
            if include_surface:
                surface_file = step_dir / f"{step_tag}.boundaries.vtu"
                out_path = sim_out / "surface" / f"{step_tag}.vtp"
                write_surface_step(
                    surface_file, out_path, surf_pred_steps[idx], surf_gt_steps[idx]
                )
            if include_volume and region_names:
                vol_out = sim_out / "volume"
                write_volume_step(
                    step_dir,
                    region_names,
                    vol_out,
                    vol_pred_steps[idx],
                    vol_gt_steps[idx],
                )

        logger.info(f"Wrote inference results for '{sim_dir.name}' to {sim_out}")


if __name__ == "__main__":
    main()
