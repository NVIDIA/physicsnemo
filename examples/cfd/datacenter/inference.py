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

from physicsnemo.datapipes.cae.mesh_datapipe import MeshDatapipe
from physicsnemo.distributed import DistributedManager
import vtk
from physicsnemo.models.unet import UNet
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import torch
import hydra
import matplotlib.pyplot as plt
import torch.nn.functional as F
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import PythonLogger, LaunchLogger
from hydra.utils import to_absolute_path
from torch.nn.parallel import DistributedDataParallel
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from apex import optimizers
import os
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from physicsnemo.sym.geometry.primitives_3d import Box, Channel
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk
import itertools


def reshape_fortran(x, shape):
    """Based on https://stackoverflow.com/questions/63960352/reshaping-order-in-pytorch-fortran-like-index-ordering"""
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def generate_mask(points, sample):
    num_racks, width, gap, translate, length, height = (
        sample[1],
        sample[2],
        sample[3],
        sample[4],
        sample[5],
        sample[6],
    )

    rack_x = 600 / 1000
    rack_y = 50 / 1000
    rack_z = 2200 / 1000

    width = width * 2 / 1000
    length = length / 1000
    height = height / 1000

    origin = (0, 0.05, 0)

    w1_x = gap / 2 / 1000  # the x distance of the left wall
    geo = Box(
        (origin[0] + w1_x, origin[1], origin[2]),
        (origin[0] + w1_x + rack_x, origin[1] + rack_y, origin[2] + rack_z),
    )
    geo = geo.repeat(
        gap / 1000 + rack_x,
        repeat_lower=(0, 0, 0),
        repeat_higher=(int(num_racks - 1), 0, 0),
        center=(
            origin[0] + w1_x + rack_x / 2,
            origin[1] + rack_y / 2,
            origin[2] + rack_z / 2,
        ),
    )

    geo_block_pos_y = Box(
        (origin[0] - w1_x, origin[1] - rack_y, origin[2]),
        (origin[0] + w1_x, origin[1] + 2, origin[2] + rack_z),
    )
    geo_block_neg_y = Box(
        (origin[0] - w1_x, origin[1] - width - 2 * rack_y - 2, origin[2]),
        (origin[0] + w1_x, origin[1] - width - rack_y, origin[2] + rack_z),
    )

    geo_block_pos_y = geo_block_pos_y.repeat(
        gap / 1000 + rack_x,
        repeat_lower=(0, 0, 0),
        repeat_higher=(int(num_racks), 0, 0),
        center=(origin[0], origin[1] - rack_y / 2 + 1, origin[2] + rack_z / 2),
    )

    geo_block_neg_y = geo_block_neg_y.repeat(
        gap / 1000 + rack_x,
        repeat_lower=(0, 0, 0),
        repeat_higher=(int(num_racks), 0, 0),
        center=(
            origin[0],
            origin[1] - width - 3 * rack_y / 2 - 1,
            origin[2] + rack_z / 2,
        ),
    )

    geo_block = geo_block_pos_y + geo_block_neg_y

    rack_top_pos_x = Box(
        (origin[0] - 5, origin[1] - rack_y, origin[2] + rack_z),
        (origin[0] + length + 5, origin[1] + 2, origin[2] + height + 10),
    )
    rack_top_neg_x = Box(
        (origin[0] - 5, origin[1] - width - 2 * rack_y - 2, origin[2] + rack_z),
        (origin[0] + length + 5, origin[1] - width - rack_y, origin[2] + height + 10),
    )

    geo_block = geo_block + rack_top_pos_x + rack_top_neg_x

    hot_aisle_bounds = (
        (origin[0], origin[1] - width - 2 * rack_y, origin[2]),
        (origin[0] + length, origin[1], origin[2] + height),
    )

    hot_aisle = Channel(
        (origin[0] - 5, origin[1] - width - 2, origin[2]),
        (origin[0] + length + 5, origin[1] + 2, origin[2] + height + 10),
    )

    hot_aisle = hot_aisle - geo_block

    # Compute SDF on the points
    sdf = hot_aisle.sdf(points, params={})

    return sdf["sdf"], hot_aisle_bounds


def save_to_vtu(data_dict, bounds, output_file):
    num_cells_x, num_cells_y, num_cells_z = next(iter(data_dict.values())).shape
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    dx = (x_max - x_min) / (num_cells_x - 1)
    dy = (y_max - y_min) / (num_cells_y - 1)
    dz = (z_max - z_min) / (num_cells_z - 1)

    # Create an unstructured grid
    points = vtk.vtkPoints()
    grid = vtk.vtkUnstructuredGrid()

    # Insert points
    for k in range(num_cells_z):
        for j in range(num_cells_y):
            for i in range(num_cells_x):
                points.InsertNextPoint(x_min + i * dx, y_min + j * dy, z_min + k * dz)

    grid.SetPoints(points)

    # Create cells
    for k in range(num_cells_z - 1):
        for j in range(num_cells_y - 1):
            for i in range(num_cells_x - 1):
                pt_ids = [
                    i + j * num_cells_x + k * num_cells_x * num_cells_y,
                    (i + 1) + j * num_cells_x + k * num_cells_x * num_cells_y,
                    (i + 1) + (j + 1) * num_cells_x + k * num_cells_x * num_cells_y,
                    i + (j + 1) * num_cells_x + k * num_cells_x * num_cells_y,
                    i + j * num_cells_x + (k + 1) * num_cells_x * num_cells_y,
                    (i + 1) + j * num_cells_x + (k + 1) * num_cells_x * num_cells_y,
                    (i + 1)
                    + (j + 1) * num_cells_x
                    + (k + 1) * num_cells_x * num_cells_y,
                    i + (j + 1) * num_cells_x + (k + 1) * num_cells_x * num_cells_y,
                ]
                grid.InsertNextCell(vtk.VTK_HEXAHEDRON, 8, pt_ids)

    # Add data arrays to the grid
    for var_name, array in data_dict.items():
        array = np.asfortranarray(array)
        flat_array = array.flatten(order="F")
        vtk_array = numpy_to_vtk(flat_array, deep=True)
        vtk_array.SetName(var_name)
        grid.GetPointData().AddArray(vtk_array)

    # Write the unstructured grid to a VTU file
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(grid)
    writer.Write()


@hydra.main(version_base="1.2", config_path="conf", config_name="config_inference")
def main(cfg: DictConfig) -> None:

    print("Inference Started!")

    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    nx, ny, nz = 960, 96, 80

    # Compute positional embeddings
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)

    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    x_freq_sin = np.sin(xv * 72 * np.pi / 2)
    x_freq_cos = np.cos(xv * 72 * np.pi / 2)
    y_freq_sin = np.sin(yv * 8 * np.pi / 2)
    y_freq_cos = np.cos(yv * 8 * np.pi / 2)
    z_freq_sin = np.sin(zv * 8 * np.pi / 2)
    z_freq_cos = np.cos(zv * 8 * np.pi / 2)
    pos_embed = np.stack(
        (
            xv,
            x_freq_sin,
            x_freq_cos,
            yv,
            y_freq_sin,
            y_freq_cos,
            zv,
            z_freq_sin,
            z_freq_cos,
        ),
        axis=0,
    )

    model = UNet(
        in_channels=10,
        out_channels=5,
        model_depth=5,
        feature_map_channels=[32, 32, 64, 64, 128, 128, 256, 256, 512, 512],
        num_conv_blocks=2,
    ).to(dist.device)

    loaded_epoch = load_checkpoint(
        to_absolute_path("./outputs/checkpoints/"),
        models=model,
        device=dist.device,
    )

    grid_dims = (nx, ny, nz)  # dimensions of the grid
    bounds = (0, 40, -3.95, 0.05, 0, 3.2)  # bounding box coordinates

    # Define the bounds and resolution of the Cartesian grid
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    num_cells_x, num_cells_y, num_cells_z = grid_dims
    dx = (x_max - x_min) / (num_cells_x - 1)
    dy = (y_max - y_min) / (num_cells_y - 1)
    dz = (z_max - z_min) / (num_cells_z - 1)

    x = np.linspace(x_min, x_max, num_cells_x)
    y = np.linspace(y_min, y_max, num_cells_y)
    z = np.linspace(z_min, z_max, num_cells_z)

    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")

    points = {
        "x": xv,
        "y": yv,
        "z": zv,
    }

    # Generate custom samples
    racks = np.linspace(35, 55, 6)
    length = 40000
    widths = 3500 / 2
    heights = 2900
    combinations = list(itertools.product(racks))

    # Define mean and std dictionaries
    mean_dict = {
        "T": 39,
        "U": 1.5983600616455078,
        "p": 6.1226935386657715,
        "wallDistance": 0.6676982045173645,
    }
    std_dict = {
        "T": 4,
        "U": 1.3656059503555298,
        "p": 4.166020393371582,
        "wallDistance": 0.45233625173568726,
    }

    model.eval()

    for design in combinations:
        print("Computing: ", design)
        rack, width, height = design[0], widths, heights
        gap = (length / rack) - 600
        sample = (
            0,
            rack,
            width,
            gap,
            0,
            length,
            height,
        )  # case num and translate var dont matter

        sdf, hot_aisle_bounds = generate_mask(points, sample)
        mask = np.where(
            (sdf > 0)
            & (zv < hot_aisle_bounds[1][2])
            & (yv > hot_aisle_bounds[0][1])
            & (xv < hot_aisle_bounds[1][0]),
            1,
            0,
        )

        sdf = ((sdf - mean_dict["wallDistance"]) / std_dict["wallDistance"]) * mask

        invar_np = np.concatenate(
            (np.expand_dims(sdf, 0), pos_embed), axis=0
        )  # concat along channel dim
        invar_np = np.expand_dims(invar_np, 0)  # add batch dim
        invar_tensor = torch.from_numpy(invar_np).to(dist.device).to(torch.float)

        with torch.no_grad():
            pred_outvar = model(invar_tensor)

        pred_outvar_np = pred_outvar.detach().cpu().numpy()

        output_filename = f"results_{rack}_{length}_{width}_{height}.vtu"
        var = {
            "u_x_pred": pred_outvar_np[0, 0],
            "u_y_pred": pred_outvar_np[0, 1],
            "u_z_pred": pred_outvar_np[0, 2],
            "T_pred": pred_outvar_np[0, 3],
            "p_pred": pred_outvar_np[0, 4],
            "wallDistance": invar_np[0, 0],
            "mask": mask,
        }
        save_to_vtu(var, bounds, output_filename)

    print("Inference complete")


if __name__ == "__main__":
    main()
