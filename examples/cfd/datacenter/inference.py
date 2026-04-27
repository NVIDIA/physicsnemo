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
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, LaunchLogger
from hydra.utils import to_absolute_path
from torch.nn.parallel import DistributedDataParallel
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from apex import optimizers
import itertools
import os
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


def _box_sdf(points, lower, upper):
    """Euclidean signed distance for an axis-aligned box (positive inside)."""
    cx = 0.5 * (lower[0] + upper[0])
    cy = 0.5 * (lower[1] + upper[1])
    cz = 0.5 * (lower[2] + upper[2])
    hx = 0.5 * (upper[0] - lower[0])
    hy = 0.5 * (upper[1] - lower[1])
    hz = 0.5 * (upper[2] - lower[2])
    dx = np.abs(points[:, 0] - cx) - hx
    dy = np.abs(points[:, 1] - cy) - hy
    dz = np.abs(points[:, 2] - cz) - hz
    outside = np.sqrt(
        np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2 + np.maximum(dz, 0) ** 2
    )
    inside = np.minimum(np.maximum(np.maximum(dx, dy), dz), 0)
    return -(outside + inside)


def _sdf_union(*sdfs):
    """CSG union: positive where any operand is positive."""
    return np.maximum.reduce(sdfs)


def _sdf_subtract(a, b):
    """CSG subtraction (A - B): inside A and outside B."""
    return np.minimum(a, -b)


def _repeated_boxes_sdf(
    points, lower, upper, spacing, repeat_lower, repeat_higher, center
):
    """SDF for repeated boxes: evaluate each copy and take the union (max).

    Uses the Euclidean box SDF per copy, combined via ``max`` (CSG union).
    The center parameter defines the center of the original (un-repeated) box;
    copies are offset by ``i * spacing`` along x from that center.
    """
    combined = np.full(len(points), -np.inf)
    cx = center[0] if center is not None else 0.5 * (lower[0] + upper[0])
    half_x = 0.5 * (upper[0] - lower[0])
    for i in range(repeat_lower, repeat_higher + 1):
        offset = i * spacing
        lo = (cx - half_x + offset, lower[1], lower[2])
        hi = (cx + half_x + offset, upper[1], upper[2])
        combined = np.maximum(combined, _box_sdf(points, lo, hi))
    return combined


def reshape_fortran(x, shape):
    """Based on https://stackoverflow.com/questions/63960352/reshaping-order-in-pytorch-fortran-like-index-ordering"""
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def generate_mask(points, sample):
    """
    Generate a mask
    """
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

    w1_x = gap / 2 / 1000
    spacing = gap / 1000 + rack_x

    # Wall blocks (pos_y and neg_y) repeated along x
    sdf_block_pos_y = _repeated_boxes_sdf(
        points,
        (origin[0] - w1_x, origin[1] - rack_y, origin[2]),
        (origin[0] + w1_x, origin[1] + 2, origin[2] + rack_z),
        spacing=spacing,
        repeat_lower=0,
        repeat_higher=int(num_racks),
        center=(origin[0], origin[1] - rack_y / 2 + 1, origin[2] + rack_z / 2),
    )
    sdf_block_neg_y = _repeated_boxes_sdf(
        points,
        (origin[0] - w1_x, origin[1] - width - 2 * rack_y - 2, origin[2]),
        (origin[0] + w1_x, origin[1] - width - rack_y, origin[2] + rack_z),
        spacing=spacing,
        repeat_lower=0,
        repeat_higher=int(num_racks),
        center=(
            origin[0],
            origin[1] - width - 3 * rack_y / 2 - 1,
            origin[2] + rack_z / 2,
        ),
    )

    # Rack-top boxes
    sdf_rack_top_pos = _box_sdf(
        points,
        (origin[0] - 5, origin[1] - rack_y, origin[2] + rack_z),
        (origin[0] + length + 5, origin[1] + 2, origin[2] + height + 10),
    )
    sdf_rack_top_neg = _box_sdf(
        points,
        (origin[0] - 5, origin[1] - width - 2 * rack_y - 2, origin[2] + rack_z),
        (origin[0] + length + 5, origin[1] - width - rack_y, origin[2] + height + 10),
    )

    # Union of wall blocks + rack tops (racks are NOT subtracted from the channel,
    # matching the original code where the rack variable is unused in the CSG)
    sdf_block = _sdf_union(
        sdf_block_pos_y, sdf_block_neg_y, sdf_rack_top_pos, sdf_rack_top_neg
    )

    # Channel (no x-boundaries — Euclidean SDF on y and z only)
    cy = 0.5 * ((origin[1] - width - 2) + (origin[1] + 2))
    cz = 0.5 * (origin[2] + (origin[2] + height + 10))
    hy = 0.5 * ((origin[1] + 2) - (origin[1] - width - 2))
    hz = 0.5 * ((origin[2] + height + 10) - origin[2])
    dy = np.abs(points[:, 1] - cy) - hy
    dz = np.abs(points[:, 2] - cz) - hz
    outside_ch = np.sqrt(np.maximum(dy, 0) ** 2 + np.maximum(dz, 0) ** 2)
    inside_ch = np.minimum(np.maximum(dy, dz), 0)
    sdf_channel = -(outside_ch + inside_ch)

    # hot_aisle = channel - blocks (inside channel AND outside blocks)
    sdf_hot_aisle = _sdf_subtract(sdf_channel, sdf_block)

    hot_aisle_bounds = (
        (origin[0], origin[1] - width - 2 * rack_y, origin[2]),
        (origin[0] + length, origin[1], origin[2] + height),
    )

    return sdf_hot_aisle, hot_aisle_bounds


def save_to_vtu(data_dict, bounds, output_file):
    """Save a dict of 3-D arrays to a VTU file on a rectilinear grid."""
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
    """Run datacenter inference."""
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
