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

"""
Inference script for running trained Transolver/GeoTransolver models on raw VTK files.

This script reads VTP (surface) and VTU (volume) files directly, processes them through
the TransolverDataPipe, runs batched inference, and saves predictions back to VTK files.

Usage (surface inference with GeoTransolver):
    python inference_on_vtk.py --config-name=geotransolver_surface \
        +vtk_inference.input_dir=/path/to/runs \
        +vtk_inference.output_dir=/path/to/output \
        +vtk_inference.air_density=1.2050 \
        +vtk_inference.stream_velocity=30.0

Usage (volume inference with GeoTransolver):
    python inference_on_vtk.py --config-name=geotransolver_volume \
        +vtk_inference.input_dir=/path/to/runs \
        +vtk_inference.output_dir=/path/to/output

Usage (surface inference with Transolver):
    python inference_on_vtk.py --config-name=transolver_surface \
        +vtk_inference.input_dir=/path/to/runs \
        +vtk_inference.output_dir=/path/to/output

Note: The '+' prefix adds new config keys that don't exist in the base config.

Expected input directory structure:
    input_dir/
    ├── run_1/
    │   ├── boundary_1.vtp              # Surface mesh
    │   ├── volume_1.vtu                # Volume mesh
    │   └── drivaer_1_single_solid.stl  # STL geometry
    ├── run_2/
    │   └── ...
    └── ...
"""

from pathlib import Path
from typing import Literal
import time

import numpy as np
import torch
import torchinfo
import pyvista as pv

import hydra
import omegaconf
from omegaconf import DictConfig

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

from physicsnemo.datapipes.cae.transolver_datapipe import TransolverDataPipe

from train import update_model_params_for_fp8

from inference_on_zarr import batched_inference_loop


# =============================================================================
# VTK File Reading Functions
# =============================================================================


def read_stl_geometry(stl_path: str, device: torch.device) -> dict[str, torch.Tensor]:
    """
    Read STL file and extract geometry data for SDF calculation.

    Parameters
    ----------
    stl_path : str
        Path to the STL file (e.g., drivaer_N_single_solid.stl).
    device : torch.device
        Device to place tensors on.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing:
        - stl_coordinates: Vertex coordinates, shape (num_vertices, 3)
        - stl_faces: Face indices (flattened), shape (num_faces * 3,)
        - stl_centers: Cell centers, shape (num_cells, 3)
    """
    mesh = pv.read(stl_path)

    # Get vertex coordinates
    stl_coordinates = torch.from_numpy(np.asarray(mesh.points)).to(
        device=device, dtype=torch.float32
    )

    # Get face indices - pyvista stores as [n_verts, v0, v1, v2, n_verts, v0, v1, v2, ...]
    # We reshape to extract just the vertex indices for triangles
    faces = mesh.faces.reshape(-1, 4)[:, 1:]  # Remove the count column
    stl_faces = torch.from_numpy(faces.flatten()).to(device=device, dtype=torch.int32)

    # Get cell centers
    stl_centers = torch.from_numpy(np.asarray(mesh.cell_centers().points)).to(
        device=device, dtype=torch.float32
    )

    return {
        "stl_coordinates": stl_coordinates,
        "stl_faces": stl_faces,
        "stl_centers": stl_centers,
    }


def read_surface_from_vtp(
    vtp_path: str, device: torch.device, n_output_fields: int = 4
) -> dict[str, torch.Tensor]:
    """
    Read VTP (PolyData) file and extract surface mesh data.

    Parameters
    ----------
    vtp_path : str
        Path to the VTP file (e.g., boundary_N.vtp).
    device : torch.device
        Device to place tensors on.
    n_output_fields : int
        Number of output fields (default 4: pressure + 3 wall shear stress components).

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing:
        - surface_mesh_centers: Cell center coordinates, shape (num_cells, 3)
        - surface_normals: Cell normals, shape (num_cells, 3)
        - surface_areas: Cell areas, shape (num_cells,)
        - surface_fields: Dummy zeros for inference, shape (num_cells, n_output_fields)
    """
    mesh = pv.read(vtp_path)

    # Get cell centers
    surface_mesh_centers = torch.from_numpy(np.asarray(mesh.cell_centers().points)).to(
        device=device, dtype=torch.float32
    )

    # Get cell normals (normalized)
    normals = np.asarray(mesh.cell_normals)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    surface_normals = torch.from_numpy(normals).to(device=device, dtype=torch.float32)

    # Compute cell areas
    cell_sizes = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    surface_areas = torch.from_numpy(np.asarray(cell_sizes.cell_data["Area"])).to(
        device=device, dtype=torch.float32
    )

    # Create dummy fields for inference (zeros)
    num_cells = surface_mesh_centers.shape[0]
    surface_fields = torch.zeros(
        (num_cells, n_output_fields), device=device, dtype=torch.float32
    )

    return {
        "surface_mesh_centers": surface_mesh_centers,
        "surface_normals": surface_normals,
        "surface_areas": surface_areas,
        "surface_fields": surface_fields,
    }


def read_volume_from_vtu(
    vtu_path: str, device: torch.device, n_output_fields: int = 5
) -> dict[str, torch.Tensor]:
    """
    Read VTU (UnstructuredGrid) file and extract volume mesh data.

    Parameters
    ----------
    vtu_path : str
        Path to the VTU file (e.g., volume_N.vtu).
    device : torch.device
        Device to place tensors on.
    n_output_fields : int
        Number of output fields (default 5: 3 velocity + pressure + turbulent viscosity).

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing:
        - volume_mesh_centers: Cell center coordinates, shape (num_cells, 3)
        - volume_fields: Dummy zeros for inference, shape (num_cells, n_output_fields)
    """
    mesh = pv.read(vtu_path)

    # Get cell centers
    volume_mesh_centers = torch.from_numpy(np.asarray(mesh.cell_centers().points)).to(
        device=device, dtype=torch.float32
    )

    # Create dummy fields for inference (zeros)
    num_cells = volume_mesh_centers.shape[0]
    volume_fields = torch.zeros(
        (num_cells, n_output_fields), device=device, dtype=torch.float32
    )

    return {
        "volume_mesh_centers": volume_mesh_centers,
        "volume_fields": volume_fields,
    }


# =============================================================================
# Data Dict Builder
# =============================================================================


def build_data_dict(
    run_dir: Path,
    data_mode: Literal["surface", "volume", "combined"],
    device: torch.device,
    air_density: float,
    stream_velocity: float,
    run_idx: int,
) -> dict[str, torch.Tensor]:
    """
    Build a complete data dictionary from VTK files for a single run.

    This function reads VTP, VTU, and STL files from a run directory and
    combines them into a dictionary compatible with TransolverDataPipe.process_data().

    Parameters
    ----------
    run_dir : Path
        Path to the run directory containing VTK files.
    data_mode : Literal["surface", "volume", "combined"]
        Which data to load - surface, volume, or both.
    device : torch.device
        Device to place tensors on.
    air_density : float
        Air density value for the simulation.
    stream_velocity : float
        Stream velocity value for the simulation.
    run_idx : int
        The run index (used for file naming conventions).

    Returns
    -------
    dict[str, torch.Tensor]
        Complete data dictionary for the datapipe.
    """
    data_dict = {}

    # Always read STL geometry (needed for SDF in volume mode, center of mass calculation)
    stl_path = run_dir / f"drivaer_{run_idx}_single_solid.stl"
    if stl_path.exists():
        stl_data = read_stl_geometry(str(stl_path), device)
        data_dict.update(stl_data)
    else:
        # Try alternative naming
        stl_files = list(run_dir.glob("*_single_solid.stl"))
        if stl_files:
            stl_data = read_stl_geometry(str(stl_files[0]), device)
            data_dict.update(stl_data)
        else:
            raise FileNotFoundError(f"No STL file found in {run_dir}")

    # Read surface data if needed
    if data_mode in ["surface", "combined"]:
        vtp_path = run_dir / f"boundary_{run_idx}.vtp"
        if not vtp_path.exists():
            # Try alternative naming
            vtp_files = list(run_dir.glob("boundary_*.vtp"))
            if vtp_files:
                vtp_path = vtp_files[0]
            else:
                raise FileNotFoundError(f"No VTP file found in {run_dir}")

        surface_data = read_surface_from_vtp(str(vtp_path), device)
        data_dict.update(surface_data)

    # Read volume data if needed
    if data_mode in ["volume", "combined"]:
        vtu_path = run_dir / f"volume_{run_idx}.vtu"
        if not vtu_path.exists():
            # Try alternative naming
            vtu_files = list(run_dir.glob("volume_*.vtu"))
            if vtu_files:
                vtu_path = vtu_files[0]
            else:
                raise FileNotFoundError(f"No VTU file found in {run_dir}")

        volume_data = read_volume_from_vtu(str(vtu_path), device)
        data_dict.update(volume_data)

    # Add flow parameters
    data_dict["air_density"] = torch.tensor(
        [air_density], device=device, dtype=torch.float32
    )
    data_dict["stream_velocity"] = torch.tensor(
        [stream_velocity], device=device, dtype=torch.float32
    )

    return data_dict


# =============================================================================
# Prediction Writer
# =============================================================================


def write_surface_predictions_to_vtk(
    vtp_path: str,
    output_path: str,
    predictions: torch.Tensor,
    air_density: float,
    stream_velocity: float,
) -> None:
    """
    Write surface predictions to a VTP file.

    Parameters
    ----------
    vtp_path : str
        Path to the original VTP file (to copy mesh structure).
    output_path : str
        Path to write the output VTP file.
    predictions : torch.Tensor
        Model predictions, shape (num_cells, 4) - [pressure, wss_x, wss_y, wss_z].
    air_density : float
        Air density for dimensional scaling.
    stream_velocity : float
        Stream velocity for dimensional scaling.
    """
    mesh = pv.read(vtp_path)
    output_mesh = mesh.copy()

    # Convert to numpy
    pred_np = predictions.cpu().numpy()

    # Split into pressure and wall shear stress
    pred_pressure = pred_np[:, 0]  # Shape: (num_cells,)
    pred_wss = pred_np[:, 1:4]  # Shape: (num_cells, 3)

    # Scale to physical units
    dynamic_pressure = air_density * stream_velocity**2
    pred_pressure = pred_pressure * dynamic_pressure
    pred_wss = pred_wss * dynamic_pressure

    # Add to mesh
    output_mesh.cell_data["PredictedPressure"] = pred_pressure
    output_mesh.cell_data["PredictedWallShearStress"] = pred_wss

    # Save
    output_mesh.save(output_path)


def write_volume_predictions_to_vtk(
    vtu_path: str,
    output_path: str,
    predictions: torch.Tensor,
    air_density: float,
    stream_velocity: float,
) -> None:
    """
    Write volume predictions to a VTU file.

    Parameters
    ----------
    vtu_path : str
        Path to the original VTU file (to copy mesh structure).
    output_path : str
        Path to write the output VTU file.
    predictions : torch.Tensor
        Model predictions, shape (num_cells, 5) - [vel_x, vel_y, vel_z, pressure, nut].
    air_density : float
        Air density for dimensional scaling.
    stream_velocity : float
        Stream velocity for dimensional scaling.
    """
    mesh = pv.read(vtu_path)
    output_mesh = mesh.copy()

    # Convert to numpy
    pred_np = predictions.cpu().numpy()

    # Split into velocity, pressure, and turbulent viscosity
    pred_velocity = pred_np[:, 0:3]  # Shape: (num_cells, 3)
    pred_pressure = pred_np[:, 3]  # Shape: (num_cells,)
    pred_nut = pred_np[:, 4]  # Shape: (num_cells,)

    # Scale to physical units
    dynamic_pressure = air_density * stream_velocity**2
    pred_velocity = pred_velocity * stream_velocity
    pred_pressure = pred_pressure * dynamic_pressure
    pred_nut = pred_nut * dynamic_pressure

    # Add to mesh
    output_mesh.cell_data["PredictedVelocity"] = pred_velocity
    output_mesh.cell_data["PredictedPressure"] = pred_pressure
    output_mesh.cell_data["PredictedNut"] = pred_nut

    # Save
    output_mesh.save(output_path)


# =============================================================================
# Main Inference Function
# =============================================================================


def create_datapipe(
    cfg: DictConfig,
    data_mode: Literal["surface", "volume", "combined"],
    device: torch.device,
    surface_factors: dict | None,
    volume_factors: dict | None,
) -> TransolverDataPipe:
    """
    Create a TransolverDataPipe configured for inference.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration.
    data_mode : Literal["surface", "volume", "combined"]
        Data mode for the datapipe.
    device : torch.device
        Device for tensors.
    surface_factors : dict | None
        Normalization factors for surface fields.
    volume_factors : dict | None
        Normalization factors for volume fields.

    Returns
    -------
    TransolverDataPipe
        Configured datapipe for inference.
    """
    # Build overrides from config
    overrides = {}

    optional_keys = [
        "include_normals",
        "include_sdf",
        "broadcast_global_features",
        "include_geometry",
        "geometry_sampling",
        "translational_invariance",
        "reference_origin",
        "scale_invariance",
        "reference_scale",
    ]

    for key in optional_keys:
        if cfg.data.get(key, None) is not None:
            overrides[key] = cfg.data[key]

    # Create the datapipe with no resolution limit (we handle batching ourselves)
    datapipe = TransolverDataPipe(
        input_path=None,  # We're not using the dataset iterator
        model_type=data_mode,
        resolution=None,  # No downsampling - we batch manually
        surface_factors=surface_factors,
        volume_factors=volume_factors,
        scaling_type="mean_std_scaling",
        return_mesh_features=True,  # For surface areas/normals if needed
        **overrides,
    )

    # Move reference scale to device if needed
    if datapipe.config.scale_invariance and datapipe.config.reference_scale is not None:
        datapipe.config.reference_scale = datapipe.config.reference_scale.to(device)

    return datapipe


def inference_on_vtk(cfg: DictConfig) -> None:
    """
    Main inference function for VTK files.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """
    # Initialize distributed
    DistributedManager.initialize()
    dist_manager = DistributedManager()

    logger = RankZeroLoggingWrapper(PythonLogger(name="vtk_inference"), dist_manager)

    # Update config for FP8 if needed
    cfg, output_pad_size = update_model_params_for_fp8(cfg, logger)

    logger.info(f"Config:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")

    # Get VTK inference config - these are added via command line with '+' prefix
    if not cfg.get("vtk_inference", None):
        raise ValueError(
            "vtk_inference config section is required. "
            "Add it via command line with '+vtk_inference.input_dir=...' etc."
        )

    vtk_cfg = cfg.vtk_inference

    # Required parameters
    if not vtk_cfg.get("input_dir", None):
        raise ValueError("vtk_inference.input_dir is required")
    if not vtk_cfg.get("output_dir", None):
        raise ValueError("vtk_inference.output_dir is required")

    input_dir = Path(vtk_cfg.input_dir)
    output_dir = Path(vtk_cfg.output_dir)

    # Optional parameters with defaults
    air_density = vtk_cfg.get("air_density", 1.2050)
    stream_velocity = vtk_cfg.get("stream_velocity", 30.0)
    run_indices = vtk_cfg.get("run_indices", None)

    logger.info(f"VTK Inference Settings:")
    logger.info(f"  input_dir: {input_dir}")
    logger.info(f"  output_dir: {output_dir}")
    logger.info(f"  air_density: {air_density}")
    logger.info(f"  stream_velocity: {stream_velocity}")
    logger.info(f"  run_indices: {run_indices}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine data mode
    data_mode = cfg.data.mode

    # Set up model
    model = hydra.utils.instantiate(cfg.model)
    logger.info(f"\n{torchinfo.summary(model, verbose=0)}")

    # Load checkpoint
    if cfg.checkpoint_dir is not None:
        checkpoint_dir = cfg.checkpoint_dir
    else:
        checkpoint_dir = f"{cfg.output_dir}/{cfg.run_id}/checkpoints"

    ckpt_args = {
        "path": checkpoint_dir,
        "models": model,
    }

    loaded_epoch = load_checkpoint(device=dist_manager.device, **ckpt_args)
    logger.info(f"Loaded checkpoint from epoch: {loaded_epoch}")

    model.to(dist_manager.device)
    model.eval()

    if cfg.compile:
        model = torch.compile(model, dynamic=True)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of model parameters: {num_params}")

    # Load normalization factors
    norm_dir = getattr(cfg.data, "normalization_dir", ".")

    surface_factors = None
    volume_factors = None

    if data_mode in ["surface", "combined"]:
        norm_file = str(Path(norm_dir) / "surface_fields_normalization.npz")
        if Path(norm_file).exists():
            norm_data = np.load(norm_file)
            surface_factors = {
                "mean": torch.from_numpy(norm_data["mean"]).to(dist_manager.device),
                "std": torch.from_numpy(norm_data["std"]).to(dist_manager.device),
            }
            logger.info(f"Loaded surface normalization from {norm_file}")

    if data_mode in ["volume", "combined"]:
        norm_file = str(Path(norm_dir) / "volume_fields_normalization.npz")
        if Path(norm_file).exists():
            norm_data = np.load(norm_file)
            volume_factors = {
                "mean": torch.from_numpy(norm_data["mean"]).to(dist_manager.device),
                "std": torch.from_numpy(norm_data["std"]).to(dist_manager.device),
            }
            logger.info(f"Loaded volume normalization from {norm_file}")

    # Create datapipe
    datapipe = create_datapipe(
        cfg, data_mode, dist_manager.device, surface_factors, volume_factors
    )

    # Get batch resolution from config
    batch_resolution = cfg.data.resolution

    # Find all run directories
    if run_indices is not None:
        run_dirs = [input_dir / f"run_{idx}" for idx in run_indices]
    else:
        run_dirs = sorted(
            [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        )

    logger.info(f"Found {len(run_dirs)} run directories to process")

    # Distribute runs across ranks
    this_device_runs = run_dirs[dist_manager.rank :: dist_manager.world_size]
    logger.info(f"Rank {dist_manager.rank} processing {len(this_device_runs)} runs")

    # Process each run
    for run_dir in this_device_runs:
        run_idx = int(run_dir.name.split("_")[1])
        logger.info(f"Processing run {run_idx}: {run_dir}")

        start_time = time.time()

        try:
            # Build data dictionary from VTK files
            data_dict = build_data_dict(
                run_dir=run_dir,
                data_mode=data_mode,
                device=dist_manager.device,
                air_density=air_density,
                stream_velocity=stream_velocity,
                run_idx=run_idx,
            )

            # Process through datapipe (adds batch dimension)
            batch = datapipe(data_dict)

            # Run batched inference using imported function from inference_on_zarr
            with torch.no_grad():
                _, _, (predictions, _) = batched_inference_loop(
                    batch=batch,
                    model=model,
                    precision=cfg.precision,
                    data_mode=data_mode,
                    batch_resolution=batch_resolution,
                    output_pad_size=output_pad_size,
                    dist_manager=dist_manager,
                    datapipe=datapipe,
                )

            # Remove batch dimension and get predictions
            predictions = predictions.squeeze(0)

            # Write predictions to output files
            run_output_dir = output_dir / run_dir.name
            run_output_dir.mkdir(parents=True, exist_ok=True)

            if data_mode in ["surface", "combined"]:
                vtp_path = run_dir / f"boundary_{run_idx}.vtp"
                if not vtp_path.exists():
                    vtp_path = list(run_dir.glob("boundary_*.vtp"))[0]

                output_vtp = run_output_dir / f"pred_boundary_{run_idx}.vtp"
                write_surface_predictions_to_vtk(
                    str(vtp_path),
                    str(output_vtp),
                    predictions,
                    air_density,
                    stream_velocity,
                )
                logger.info(f"Saved surface predictions to {output_vtp}")

            if data_mode in ["volume", "combined"]:
                vtu_path = run_dir / f"volume_{run_idx}.vtu"
                if not vtu_path.exists():
                    vtu_path = list(run_dir.glob("volume_*.vtu"))[0]

                output_vtu = run_output_dir / f"pred_volume_{run_idx}.vtu"
                write_volume_predictions_to_vtk(
                    str(vtu_path),
                    str(output_vtu),
                    predictions,
                    air_density,
                    stream_velocity,
                )
                logger.info(f"Saved volume predictions to {output_vtu}")

            elapsed = time.time() - start_time
            logger.info(f"Completed run {run_idx} in {elapsed:.2f} seconds")

        except Exception as e:
            logger.error(f"Error processing run {run_idx}: {e}")
            import traceback

            traceback.print_exc()
            continue

    logger.info("Inference complete!")


# =============================================================================
# Entry Point
# =============================================================================


@hydra.main(version_base=None, config_path="conf", config_name="geotransolver_surface")
def launch(cfg: DictConfig) -> None:
    """
    Launch VTK inference with Hydra configuration.

    Uses existing geotransolver/transolver configs. VTK-specific parameters
    must be added via command line with '+' prefix:
        +vtk_inference.input_dir=/path/to/runs
        +vtk_inference.output_dir=/path/to/output
        +vtk_inference.air_density=1.2050  (optional, default: 1.2050)
        +vtk_inference.stream_velocity=30.0  (optional, default: 30.0)
        +vtk_inference.run_indices=[1,2,3]  (optional, default: all runs)

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """
    inference_on_vtk(cfg)


if __name__ == "__main__":
    launch()
