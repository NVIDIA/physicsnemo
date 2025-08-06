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

import numpy as np
import torch
import torchinfo
import hydra
from omegaconf import DictConfig

import pyvista as pv
from physicsnemo.models.transolver.transolver import Transolver
from physicsnemo.launch.utils import load_checkpoint
from physicsnemo.launch.logging import RankZeroLoggingWrapper, PythonLogger

from physicsnemo.distributed import DistributedManager

import vtk
from vtk.util import numpy_support
import os
import math
import time

from train import (
    update_model_params_for_fp8,
    cast_precisions,
    get_autocast_context,
    pad_input_for_fp8,
    unpad_output_for_fp8,
)

def read_data_from_stl(
    stl_path: str,
    air_density: float = 1.2050,
    stream_velocity: float = 30.0,
) -> dict:
    """
    Reads mesh and surface data from an STL file and prepares a batch dictionary for inference.

    Args:
        stl_path (str): Path to the STL file.
        air_density (float, optional): Air density value. Defaults to 1.2050.
        stream_velocity (float, optional): Stream velocity value. Defaults to 30.0.

    Returns:
        dict: Batch dictionary with mesh centers, normals, sizes, air density, and stream velocity as torch tensors.
    """

    dm = DistributedManager()

    mesh = pv.read(stl_path)

    batch = {}

    batch["surface_mesh_centers"] = np.asarray(mesh.cell_centers().points)

    normals = np.asarray(mesh.cell_normals)
    # Normalize cell normals
    surface_normals = (
        surface_normals / np.linalg.norm(surface_normals, axis=1)[:, np.newaxis]
    )
    batch["surface_normals"] = surface_normals
    surface_areas = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    batch["surface_mesh_sizes"] = np.array(surface_areas.cell_data["Area"])

    batch["air_density"] = np.array([air_density], dtype="float32")
    batch["stream_velocity"] = np.array([stream_velocity], dtype="float32")

    batch = {
        k: torch.from_numpy(v).to(device=dm.device, dtype=torch.float32)
        for k, v in batch.items()
    }

    batch = {k: torch.unsqueeze(v, dim=0) for k, v in batch.items()}

    return batch


def read_data_from_vtp(
    vtp_path: str,
    air_density: float = 1.2050,
    stream_velocity: float = 30.0,
) -> tuple:
    """
    Reads mesh and surface data from a VTP file and prepares a batch dictionary for inference.

    Args:
        vtp_path (str): Path to the VTP file.
        air_density (float, optional): Air density value. Defaults to 1.2050.
        stream_velocity (float, optional): Stream velocity value. Defaults to 30.0.

    Returns:
        tuple: (mesh, batch) where mesh is a pyvista mesh object and batch is a dictionary of torch tensors.
    """

    dm = DistributedManager()

    mesh = pv.read(vtp_path)

    batch = {}

    batch["surface_mesh_centers"] = np.asarray(mesh.cell_centers().points)
    batch["surface_normals"] = np.asarray(mesh.cell_normals)
    surface_areas = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    batch["surface_mesh_sizes"] = np.array(surface_areas.cell_data["Area"])

    batch["CpMeanTrim"] = np.asarray(mesh.cell_data["CpMeanTrim"])
    batch["pMeanTrim"] = np.asarray(mesh.cell_data["pMeanTrim"])
    batch["wallShearStressMeanTrim"] = np.asarray(
        mesh.cell_data["wallShearStressMeanTrim"]
    )

    batch["air_density"] = np.array([air_density], dtype="float32")
    batch["stream_velocity"] = np.array([stream_velocity], dtype="float32")

    # From VTP we can also exctract the ground-truth results:
    batch = {
        k: torch.from_numpy(v).to(device=dm.device, dtype=torch.float32)
        for k, v in batch.items()
    }

    batch = {k: torch.unsqueeze(v, dim=0) for k, v in batch.items()}

    return mesh, batch


def preprocess_data(
    batch: dict,
) -> tuple:
    """
    Preprocesses the batch data to generate node features and embeddings for the model.

    Args:
        batch (dict): Batch dictionary containing mesh and physical properties.

    Returns:
        tuple: (node_features, embeddings) as torch tensors.
    """

    mesh_centers = batch["surface_mesh_centers"]
    normals = batch["surface_normals"]
    node_features = torch.stack(
        [batch["air_density"], batch["stream_velocity"]], axis=-1
    )

    # Calculate center of mass
    sizes = batch["surface_mesh_sizes"]

    total_weighted_position = torch.einsum("ki,kij->kj", sizes, mesh_centers)
    total_size = torch.sum(sizes)
    center_of_mass = total_weighted_position[None, ...] / total_size

    # Subtract the COM from the centers:
    mesh_centers = mesh_centers - center_of_mass

    embeddings = torch.cat(
        [
            mesh_centers,
            normals,
        ],
        dim=-1,
    )
    node_features = node_features.expand(1, embeddings.shape[1], -1)

    return node_features, embeddings


def model_inference(model, features, embeddings, precision, output_pad_size):
    
    # Cast precisions:
    features, embeddings = cast_precisions(features, embeddings, precision)
    with get_autocast_context(precision):

        # For fp8, we may have to pad the inputs:
        if precision == "float8":
            features = pad_input_for_fp8(features, embeddings)

        outputs = model(features, embeddings)

        outputs = unpad_output_for_fp8(outputs, output_pad_size)

    return outputs


def process_vtp_file(
    vtp_file: str,
    model: torch.nn.Module,
    norm_factors: dict,
    output_folder: str,
    precision: str,
    output_pad_size: int,
    batch_size: int = 500_000,
) -> None:
    """
    Processes a single VTP file: runs inference, computes errors, and writes predictions to a new VTP file.

    Args:
        vtp_file (str): Path to the VTP file.
        model (torch.nn.Module): The trained model for inference.
        norm_factors (dict): Normalization factors for output.
        output_folder (str): Directory to save output files.
        batch_size (int, optional): Batch size for inference. Defaults to 500_000.

    Returns:
        None
    """

    # First, load the data and mesh from the file:
    try:
        mesh, batch = read_data_from_vtp(vtp_file)
    except FileNotFoundError as e:
        print(f"File not found: {vtp_file}")
        return

    # Run preprocessing to prepare the data for the model:
    fx, embedding = preprocess_data(batch)

    with torch.no_grad():

        if batch_size > fx.shape[1]:

            outputs = model_inference(model, fx, embedding, precision, output_pad_size)

            prediction = (
                outputs * norm_factors["std"] + norm_factors["mean"]
            )

        else:
            # Split the indices by a batch size.  We shuffle the cells into
            # the batches (don't forget to unshuffle later!)
            indices = torch.randperm(fx.shape[1], device=fx.device)

            index_blocks = torch.split(indices, batch_size)

            predictions = []
            for i, index_block in enumerate(index_blocks):
                local_fx = fx[:, index_block]
                local_embedding = embedding[:, index_block]
                
                # Just in the fp8 case, we have to pad the batch shape, too:
                sample_shape = local_fx.shape[1]
                if precision == "float8" and sample_shape % 8 != 0:
                    # NOTE: this padding is along axis 1, not -1!
                    padding = 8 - (sample_shape % 8)

                    # Create zero tensors to pad
                    fx_pad = torch.zeros(*local_fx.shape[:1], padding, *local_fx.shape[2:], dtype=local_fx.dtype, device=local_fx.device)
                    emb_pad = torch.zeros(*local_embedding.shape[:1], padding, *local_embedding.shape[2:], dtype=local_embedding.dtype, device=local_embedding.device)
                   
                    # Concatenate along dim=1
                    local_fx = torch.cat([local_fx, fx_pad], dim=1)
                    local_embedding = torch.cat([local_embedding, emb_pad], dim=1)
                    
                outputs = model_inference(model, local_fx, local_embedding, precision, output_pad_size)
                
                # And, if we padded, we have to unpad the output:
                if precision == "float8" and sample_shape % 8 != 0:
                    outputs = outputs[:, :-padding, :]

                predictions.append(
                    outputs * norm_factors["std"]+ norm_factors["mean"]
                )

            prediction = torch.cat(predictions, dim=1)

            # Now, we have to *unshuffle* the prediction to the original index
            inverse_indices = torch.empty_like(indices)
            inverse_indices[indices] = torch.arange(
                indices.size(0), device=indices.device
            )
            # Suppose prediction is of shape [batch, N, ...]
            prediction = prediction[:, inverse_indices]

        pred_pressure, pred_shear = torch.split(prediction, (1, 3), dim=-1)
        pred_pressure = pred_pressure.squeeze(-1)

        pred_pressure = pred_pressure * (
            batch["air_density"] * batch["stream_velocity"] ** 2
        )
        pred_shear = pred_shear * (batch["air_density"] * batch["stream_velocity"] ** 2)
        target_pressure = batch["pMeanTrim"]
        target_shear = batch["wallShearStressMeanTrim"]

        pressure_l2_num = (pred_pressure - target_pressure) ** 2
        pressure_l2_num = torch.sum(pressure_l2_num, dim=1)
        pressure_l2_num = torch.sqrt(pressure_l2_num)

        pressure_l2_denom = target_pressure**2
        pressure_l2_denom = torch.sum(pressure_l2_denom, dim=1)
        pressure_l2_denom = torch.sqrt(pressure_l2_denom)

        pressure_l2 = pressure_l2_num / pressure_l2_denom

        shear_l2_num = (pred_shear - target_shear) ** 2
        shear_l2_num = torch.sum(shear_l2_num, dim=1)
        shear_l2_num = torch.sqrt(shear_l2_num)

        shear_l2_denom = target_shear**2
        shear_l2_denom = torch.sum(shear_l2_denom, dim=1)
        shear_l2_denom = torch.sqrt(shear_l2_denom)

        shear_l2 = shear_l2_num / shear_l2_denom

        print(f"pressure l2: {pressure_l2}")
        print(f"shear l2: {shear_l2}")

    # Write the output to a new .vtp file.  Clone the old information:
    output_mesh = mesh.copy()
    # Convert tensors to numpy arrays and squeeze batch dimension
    pred_pressure_np = pred_pressure[0].cpu().numpy()
    pred_shear_np = pred_shear[0].cpu().numpy()
    # Add arrays to the mesh as cell data
    output_mesh.cell_data["PredictedPressure"] = pred_pressure_np
    output_mesh.cell_data["PredictedShear"] = pred_shear_np
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    # Construct output file path
    base_name = os.path.basename(vtp_file)
    output_path = os.path.join(output_folder, f"pred_{base_name}")
    # Write to file
    output_mesh.save(output_path)
    print(f"Saved prediction VTP to: {output_path}")


def inference_on_vtp_or_stl(cfg: DictConfig) -> None:
    """
    Main inference loop for processing multiple VTP or STL files using the provided configuration.

    Args:
        cfg (DictConfig): Hydra configuration object.

    Returns:
        None
    """

    DistributedManager.initialize()

    dist_manager = DistributedManager()

    run_id = cfg.run_id


    logger = RankZeroLoggingWrapper(PythonLogger(name="training"), dist_manager)

    cfg, output_pad_size = update_model_params_for_fp8(cfg, logger)

    # Set up model
    model = hydra.utils.instantiate(cfg.model)
    logger.info(f"\n{torchinfo.summary(model, verbose=0)}")
    model.eval()
    model.to(dist_manager.device)

    ckpt_args = {
        "path": f"{cfg.output_dir}/{cfg.run_id}/checkpoints",
        "models": model,
    }

    # Load the normalization factors:
    norm_file = "surface_fields_normalization.npz"
    norm_data = np.load(norm_file)
    norm_factors = {
        "mean": torch.from_numpy(norm_data["mean"]).to(dist_manager.device),
        "std": torch.from_numpy(norm_data["std"]).to(dist_manager.device),
    }

    # Restore the model:
    loaded_epoch = load_checkpoint(device=dist_manager.device, **ckpt_args)
    print(f"loaded epoch: {loaded_epoch}")

    stl_input_path = "/group_data/datasets/drivaer_aws/drivaer_data_full/"
    vtp_output_path = f"/user_data/datasets/drivaer_aws/drivaer_data_full-test/{run_id}/"

    all_files = list(range(1, 501))

    all_files = [stl_input_path + f"/run_{i}/boundary_{i}.vtp" for i in all_files]

    # Remove files that already exist in the output directory
    filtered_files = []
    for file_path in all_files:
        base_name = os.path.basename(file_path)
        output_path = os.path.join(vtp_output_path, f"pred_{base_name}")
        if not os.path.exists(output_path):
            filtered_files.append(file_path)
    all_files = filtered_files

    # print(f"all files: {all_files} of length {len(all_files)}")

    this_device_files = all_files[dist_manager.rank :: dist_manager.world_size]

    print(
        f"Rank {dist_manager.rank} of {dist_manager.world_size} is processing {len(this_device_files)} files"
    )
    for vtp_file in this_device_files:
        start = time.time()

        # Process files:
        process_vtp_file(
            vtp_file, 
            model, 
            norm_factors, 
            vtp_output_path, 
            precision=cfg.training.precision,
            output_pad_size=output_pad_size,
            batch_size=cfg.data.resolution
        )
        end = time.time()
        print(f"time taken: {end - start}")


@hydra.main(version_base=None, config_path="conf", config_name="train_surface")
def launch(cfg: DictConfig) -> None:
    """Launch training with hydra configuration

    Args:
        cfg: Hydra configuration object
    """
    inference_on_vtp_or_stl(cfg)


if __name__ == "__main__":
    launch()