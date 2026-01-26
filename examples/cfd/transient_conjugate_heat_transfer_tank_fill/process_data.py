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
"""Lightweight preprocessing for the Domino transient conjugate heat-transfer data.

This script reads the raw CFD simulation dumps, packs each simulation into a
single ``.npz`` file, and writes simple scaling stats. Outputs land in
``<processed_dir>/train``, ``<processed_dir>/val``, and ``<processed_dir>/stats``.

Raw data layout (per simulation):
  <raw_dir>/<sim_name>/<sim_name>/
  ├─ sim_0/                   # initial timestep (geometry only)
  │  ├─ sim_0.boundaries.vtu  # surface mesh (boundary triangles)
  │  └─ FLUID0_REG0.vtu, SOLID*.vtu  # volume regions (cell-centered)
  ├─ sim_1/, sim_2/, ..., sim_N/   # subsequent timesteps with surface/volume fields
  └─ probe_points_*.csv, inlet_*.csv, residuals.csv ... (aux files, ignored)
  The outer folder often duplicates <sim_name>; we traverse both levels to find sim_* folders.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import hydra
import numpy as np
import pyvista as pv
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from multiprocessing import Pool


_FLOAT = r"[-+]?\d+(?:\.\d+)?"
_NAME_RE = re.compile(
    rf"(?P<prefix>.*?)(?P<pressure>{_FLOAT})bar_(?P<temperature>{_FLOAT})C_(?P<runtime>{_FLOAT})s"
)

DEFAULT_COIL_MAPPING: Dict[str, float] = {
    "nocoil": 0.0,
    "frontcoil": 1.0,
    "midcoil": 2.0,
    "backcoil": 3.0,
}


def parse_params(sim_name: str) -> Dict[str, float]:
    """Parse simulation parameters from folder name (including coil name)."""

    match = _NAME_RE.search(sim_name)
    if match is None:
        raise ValueError(
            f"Simulation folder '{sim_name}' does not encode pressure/temp/runtime."
        )
    params: Dict[str, float] = {
        "pressure_bar": float(match.group("pressure")),
        "inlet_temperature_C": float(match.group("temperature")),
        "run_time_s": float(match.group("runtime")),
    }
    prefix = match.group("prefix").rstrip("_")
    coil_tag = prefix.split("-")[-1].strip()
    if coil_tag:
        key = coil_tag.lower()
        if key not in DEFAULT_COIL_MAPPING:
            raise ValueError(f"Coil tag '{coil_tag}' not defined for '{sim_name}'.")
        params["coil_position"] = DEFAULT_COIL_MAPPING[key]
    return params


def list_steps(sim_dir: Path) -> List[Path]:
    """Return timestep folders in numeric order (sim_0, sim_1, ...)."""

    def key(path: Path) -> Tuple[int, str]:
        return int(path.name.split("_")[-1]), path.name

    return sorted(
        [p for p in sim_dir.iterdir() if p.is_dir() and p.name.startswith("sim_")],
        key=key,
    )


def gather_fields(mesh: pv.DataSet, specs: Sequence[Tuple[str, str]]) -> np.ndarray:
    """Extract requested scalar/vector cell-data into a 2D array."""
    cols: List[np.ndarray] = []
    for name, kind in specs:
        if name not in mesh.cell_data:
            raise KeyError(f"Missing field '{name}' in {mesh}")
        arr = np.asarray(mesh.cell_data[name])
        if kind == "scalar":
            arr = arr.reshape(mesh.n_cells, 1)
        elif kind == "vector":
            arr = arr.reshape(mesh.n_cells, -1)[:, :3]
        else:
            raise ValueError(f"Unsupported variable kind '{kind}' for '{name}'")
        cols.append(arr.astype(np.float32))
    if not cols:
        return np.zeros((mesh.n_cells, 0), dtype=np.float32)
    return np.concatenate(cols, axis=1)


def pad_to_channels(data: np.ndarray, channels: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pad or trim channel dimension and return a matching validity mask."""
    if data.shape[1] == channels:
        mask = np.ones_like(data, dtype=np.float32)
        return data, mask
    mask = np.ones_like(data, dtype=np.float32)
    if data.shape[1] > channels:
        return data[:, :channels], mask[:, :channels]
    pad = channels - data.shape[1]
    zeros = np.zeros((data.shape[0], pad), dtype=np.float32)
    return np.concatenate([data, zeros], axis=1), np.concatenate(
        [mask, np.zeros_like(zeros)], axis=1
    )


def process_sim(
    sim_dir: Path,
    surface_specs: Sequence[Tuple[str, str]],
    volume_specs: Sequence[Tuple[str, str]],
    include_surface: bool,
    include_volume: bool,
    future_steps: int,
    global_order: Sequence[str],
    global_reference: Sequence[float] | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Process a single simulation and return a sample and metadata."""

    # Get list of steps
    steps = list_steps(sim_dir)
    if len(steps) < 2:
        raise RuntimeError(
            f"Simulation '{sim_dir}' must contain at least two timesteps."
        )

    # Get initial step
    init = steps[0]
    init_tag = init.name

    # Get boundary file
    boundary = init / f"{init_tag}.boundaries.vtu"
    if not boundary.exists():
        raise FileNotFoundError(f"Missing surface file '{boundary}'")

    # Base surface geometry from first timestep.
    surface_mesh = pv.read(boundary).extract_geometry().triangulate()
    stl_centers = surface_mesh.cell_centers().points.astype(np.float32)
    stl_faces = surface_mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
    normals = surface_mesh.cell_normals.astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    areas = (
        surface_mesh.compute_cell_sizes(length=False, area=True, volume=False)
        .cell_data["Area"]
        .astype(np.float32)
    )
    face_set = surface_mesh.cell_data.get("face_set")

    # Get volumes if needed
    volume_regions = [
        p for p in init.glob("*.vtu") if not p.name.endswith(".boundaries.vtu")
    ]
    volume_centers = None
    if include_volume and volume_regions:
        volume_centers = np.concatenate(
            [
                pv.read(region).cell_centers().points.astype(np.float32)
                for region in sorted(volume_regions)
            ],
            axis=0,
        )

    # Collect future field frames up to target_steps.
    target_steps = int(future_steps)
    usable_steps = min(target_steps, len(steps) - 1)
    surface_frames: List[np.ndarray] = []
    volume_frames: List[np.ndarray] = []
    region_names = [p.name for p in sorted(volume_regions)]

    # Process each step
    for step in steps[1 : usable_steps + 1]:
        # Get step tag
        tag = step.name

        # Get surface fields if needed
        if include_surface:
            surf_mesh = (
                pv.read(step / f"{tag}.boundaries.vtu").extract_geometry().triangulate()
            )
            vals = gather_fields(surf_mesh, surface_specs)
            if vals.shape[0] != stl_centers.shape[0]:
                raise ValueError(f"Surface cell count mismatch in {step}")
            surface_frames.append(vals)

        # Get volume fields if needed
        if include_volume and region_names:
            chunks = []
            for name in region_names:
                mesh = pv.read(step / name)
                chunks.append(gather_fields(mesh, volume_specs))
            merged = np.concatenate(chunks, axis=0)
            if (
                volume_centers is not None
                and merged.shape[0] != volume_centers.shape[0]
            ):
                raise ValueError(f"Volume cell count mismatch in {step}")
            volume_frames.append(merged)

    # Concatenate surface and volume fields
    surface_fields = (
        np.concatenate(surface_frames, axis=1)
        if surface_frames
        else np.zeros((stl_centers.shape[0], 0), dtype=np.float32)
    )
    volume_fields = np.concatenate(volume_frames, axis=1) if volume_frames else None

    # Get number of components for surface and volume
    surface_components = sum(3 if kind == "vector" else 1 for _, kind in surface_specs)
    volume_components = sum(3 if kind == "vector" else 1 for _, kind in volume_specs)

    # Pad surface fields if needed
    if include_surface:
        expected_surface = surface_components * target_steps
        surface_fields, surface_mask = pad_to_channels(surface_fields, expected_surface)
    else:
        surface_mask = None

    if include_volume and volume_fields is not None:
        expected_volume = volume_components * target_steps
        volume_fields, volume_mask = pad_to_channels(volume_fields, expected_volume)
    else:
        volume_mask = None

    params = parse_params(sim_dir.name)
    global_values = np.array([params[k] for k in global_order], dtype=np.float32)
    global_values = np.expand_dims(global_values, -1)
    global_reference_arr = np.expand_dims(
        np.array(global_reference, dtype=np.float32), -1
    )

    # Create sample for Domino datapipe format
    sample: Dict[str, np.ndarray] = {
        "stl_coordinates": surface_mesh.points.astype(np.float32),
        "stl_centers": stl_centers,
        "stl_faces": stl_faces.flatten().astype(np.float32),
        "stl_areas": areas,
        "global_params_values": global_values,
        "global_params_reference": global_reference_arr,
    }

    if include_surface:
        sample.update(
            {
                "surface_mesh_centers": stl_centers.copy(),
                "surface_normals": normals,
                "surface_areas": areas.copy(),
                "surface_fields": surface_fields.astype(np.float32),
                "surface_valid_mask": surface_mask.astype(np.float32),
            }
        )
        if face_set is not None:
            sample["surface_face_set"] = np.asarray(face_set, dtype=np.int32)

    if include_volume and volume_fields is not None and volume_centers is not None:
        sample.update(
            {
                "volume_mesh_centers": volume_centers,
                "volume_fields": volume_fields.astype(np.float32),
                "volume_valid_mask": volume_mask.astype(np.float32),
            }
        )

    meta = {
        "name": sim_dir.name,
        "future_steps": target_steps,
        "surface_points": stl_centers.shape[0],
        "volume_points": 0 if volume_centers is None else volume_centers.shape[0],
        "surface_channels": surface_fields.shape[1] if include_surface else 0,
        "volume_channels": 0 if volume_fields is None else volume_fields.shape[1],
        "parameters": params,
    }
    return sample, meta


def compute_stats(npz_dir: Path, stats_dir: Path) -> None:
    """Compute per-channel min/max for surface and volume fields."""

    # Run through all samples and compute statistics for surface and volume fields
    surf_max = surf_min = vol_max = vol_min = None
    for npz_path in sorted(npz_dir.glob("*.npz")):
        with np.load(npz_path) as data:
            # Compute statistics for surface fields
            if "surface_fields" in data:
                arr = data["surface_fields"]
                if arr.size == 0:
                    continue
                smax = arr.max(axis=0)
                smin = arr.min(axis=0)
                surf_max = smax if surf_max is None else np.maximum(surf_max, smax)
                surf_min = smin if surf_min is None else np.minimum(surf_min, smin)

            # Compute statistics for volume fields
            if "volume_fields" in data:
                arr = data["volume_fields"]
                if arr.size == 0:
                    continue
                vmax = arr.max(axis=0)
                vmin = arr.min(axis=0)
                vol_max = vmax if vol_max is None else np.maximum(vol_max, vmax)
                vol_min = vmin if vol_min is None else np.minimum(vol_min, vmin)

    # Save statistics
    stats_dir.mkdir(parents=True, exist_ok=True)
    if surf_max is not None:
        np.save(
            stats_dir / "surface_scaling_factors.npy",
            np.stack([surf_max, surf_min]).astype(np.float32),
        )
    if vol_max is not None:
        np.save(
            stats_dir / "volume_scaling_factors.npy",
            np.stack([vol_max, vol_min]).astype(np.float32),
        )


def meta_from_existing_npz(
    npz_path: Path,
    sim_name: str,
    future_steps: int,
    include_surface: bool,
    include_volume: bool,
) -> Dict[str, object]:
    """Construct metadata from an existing processed npz without reprocessing raw data."""
    params = parse_params(sim_name)
    with np.load(npz_path) as data:
        surface_pts = (
            data["surface_mesh_centers"].shape[0]
            if "surface_mesh_centers" in data
            else 0
        )
        volume_pts = (
            data["volume_mesh_centers"].shape[0] if "volume_mesh_centers" in data else 0
        )
        surface_ch = (
            data["surface_fields"].shape[1]
            if include_surface and "surface_fields" in data
            else 0
        )
        volume_ch = (
            data["volume_fields"].shape[1]
            if include_volume and "volume_fields" in data
            else 0
        )
    return {
        "name": sim_name,
        "future_steps": future_steps,
        "surface_points": surface_pts,
        "volume_points": volume_pts,
        "surface_channels": surface_ch,
        "volume_channels": volume_ch,
        "parameters": params,
    }


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Get config values
    OmegaConf.set_struct(cfg, False)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    data_cfg = cfg_dict["data"]
    var_cfg = cfg_dict["variables"]
    model_cfg = cfg_dict["model"]

    # Get paths for data and stats
    raw_dir = Path(data_cfg["raw_dir"]).expanduser().resolve()
    processed_root = Path(data_cfg["processed_dir"]).expanduser().resolve()
    train_dir = processed_root / "train"
    val_dir = processed_root / "val"
    stats_dir = processed_root / "stats"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Get data processing parameters
    future = int(data_cfg.get("future_steps", 1))
    include_surface = str(model_cfg.get("model_type", "surface")).lower() in {
        "surface",
        "combined",
    }
    include_volume = str(model_cfg.get("model_type", "surface")).lower() in {
        "volume",
        "combined",
    }

    # Get variable specifications
    surface_specs = [
        (name, kind) for name, kind in var_cfg["surface"]["solution"].items()
    ]
    volume_specs = [
        (name, kind) for name, kind in var_cfg["volume"]["solution"].items()
    ]
    config_globals = var_cfg.get("global_parameters", {})
    global_order = list(config_globals.keys())
    global_reference = [
        config_globals[name].get("reference", 1.0) for name in global_order
    ]

    # Process simulations
    # NOTE: simulation name is repeated, ie, comb_20-noCoil_300bar_-40C_90s/comb_20-noCoil_300bar_-40C_90s/
    # TODO: remove duplicate simulation names
    metas = []
    sim_dirs = []
    for entry in sorted([p for p in raw_dir.iterdir() if p.is_dir()]):
        subdirs = sorted([child for child in entry.iterdir() if child.is_dir()])
        sim_dirs.extend(subdirs or [entry])

    # Determine split
    explicit_train = data_cfg.get("splits", {}).get("train", []) or []
    explicit_val = data_cfg.get("splits", {}).get("val", []) or []
    if explicit_train or explicit_val:
        train_set = set(explicit_train)
        val_set = set(explicit_val)
    else:
        val_fraction = float(data_cfg.get("val_fraction", 0.2))
        total = len(sim_dirs)
        num_val = max(1, min(total - 1, int(round(val_fraction * total))))
        # Evenly spaced selection across sorted list
        val_indices = set(
            int(idx) for idx in np.linspace(0, total - 1, num_val, dtype=int).tolist()
        )
        train_set = set(i for i in range(total) if i not in val_indices)
        val_set = val_indices
    train_dirs = [sim_dirs[i] for i in sorted(train_set)]
    val_dirs = [sim_dirs[i] for i in sorted(val_set)]

    # Create worker arguments for train set and skip already-processed samples
    worker_args = []
    for sim_dir in train_dirs:
        out_path = train_dir / f"{sim_dir.name}.npz"
        if out_path.exists():
            meta = meta_from_existing_npz(
                out_path, sim_dir.name, future, include_surface, include_volume
            )
            meta["split"] = "train"
            metas.append(meta)
        else:
            worker_args.append(
                (
                    sim_dir,
                    surface_specs,
                    volume_specs,
                    include_surface,
                    include_volume,
                    future,
                    global_order,
                    global_reference,
                )
            )
    num_workers = max(1, int(data_cfg.get("preprocess_workers", 1)))

    # Process train set
    if num_workers == 1:
        iterable = (process_sim(*args) for args in worker_args)
        pool = None
    else:
        pool = Pool(processes=num_workers)
        iterable = pool.imap(_process_sim_wrapper, worker_args)

    try:
        for sample, meta in tqdm(
            iterable, total=len(worker_args), desc="Processing train simulations"
        ):
            np.savez_compressed(train_dir / f"{meta['name']}.npz", **sample)
            meta = dict(meta)
            meta["split"] = "train"
            metas.append(meta)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # Process validation set (parallelized like train for consistency), skipping existing npz
    val_worker_args = []
    for sim_dir in val_dirs:
        out_path = val_dir / f"{sim_dir.name}.npz"
        if out_path.exists():
            meta = meta_from_existing_npz(
                out_path, sim_dir.name, future, include_surface, include_volume
            )
            meta["split"] = "val"
            metas.append(meta)
        else:
            val_worker_args.append(
                (
                    sim_dir,
                    surface_specs,
                    volume_specs,
                    include_surface,
                    include_volume,
                    future,
                    global_order,
                    global_reference,
                )
            )

    if num_workers == 1:
        val_iterable = (process_sim(*args) for args in val_worker_args)
        val_pool = None
    else:
        val_pool = Pool(processes=num_workers)
        val_iterable = val_pool.imap(_process_sim_wrapper, val_worker_args)

    try:
        for sample, meta in tqdm(
            val_iterable, total=len(val_worker_args), desc="Processing val simulations"
        ):
            np.savez_compressed(val_dir / f"{meta['name']}.npz", **sample)
            meta = dict(meta)
            meta["split"] = "val"
            metas.append(meta)
    finally:
        if val_pool is not None:
            val_pool.close()
            val_pool.join()

    # Save metadata
    with (processed_root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"samples": metas, "global_param_order": global_order or []}, f, indent=2
        )

    # Compute statistics
    compute_stats(train_dir, stats_dir)
    print(
        f"Wrote {len(metas)} samples to train={train_dir}, val={val_dir} and stats to {stats_dir}"
    )


def _process_sim_wrapper(args):
    return process_sim(*args)


if __name__ == "__main__":
    main()
