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

"""
This code defines a standalone distributed inference pipeline the DoMINO model. 
This inference pipeline can be used to evaluate the model given an STL and
an inflow speed. The pre-trained model checkpoint can be specified in this script
or inferred from the config file. The results are calculated on a point cloud
sampled in the volume around the STL and on the surface of the STL. They are stored
in a dictionary, which can be written out for visualization.
"""

import os
import time

import hydra, re
from hydra import compose, initialize
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch

from modulus.models.domino.model import DoMINO
from modulus.utils.domino.utils import *
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel
from modulus.distributed import DistributedManager

from numpy.typing import NDArray
from typing import Any, Iterable, List, Literal, Mapping, Optional, Union, Callable
import warp as wp
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import pyvista as pv

try:
    from modulus.sym.geometry.tessellation import Tessellation

    SYM_AVAILABLE = True
except ImportError:
    SYM_AVAILABLE = False


def combine_stls(stl_path, stl_files):
    meshes = []
    combined_mesh = pv.PolyData()
    for file in stl_files:
        if ".stl" in file and "single_solid" not in file:
            stl_file_path = os.path.join(stl_path, file)
            reader = pv.get_reader(stl_file_path)
            mesh_stl = reader.read()
            combined_mesh = combined_mesh.merge(mesh_stl)
            # meshes.append(mesh_stl)
            break
    # combined_mesh = pv.merge(meshes)
    return combined_mesh


def plot(truth, prediction, var, save_path, axes_titles=None, plot_error=True):
    if plot_error:
        c = 3
    else:
        c = 2
    fig, axes = plt.subplots(1, c, figsize=(15, 5))
    error = truth - prediction
    # Plot Truth
    im = axes[0].imshow(
        truth,
        cmap="jet",
        vmax=np.ma.masked_invalid(truth).max(),
        vmin=np.ma.masked_invalid(truth).min(),
    )
    axes[0].axis("off")
    cbar = fig.colorbar(im, ax=axes[0], orientation="vertical")
    cbar.ax.tick_params(labelsize=12)
    if axes_titles is None:
        axes[0].set_title(f"{var} Truth")
    else:
        axes[0].set_title(axes_titles[0])

    # Plot Predicted
    im = axes[1].imshow(
        prediction,
        cmap="jet",
        vmax=np.ma.masked_invalid(prediction).max(),
        vmin=np.ma.masked_invalid(prediction).min(),
    )
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], orientation="vertical")
    cbar.ax.tick_params(labelsize=12)
    if axes_titles is None:
        axes[1].set_title(f"{var} Predicted")
    else:
        axes[1].set_title(axes_titles[1])

    if plot_error:
        # Plot Error
        im = axes[2].imshow(
            error,
            cmap="jet",
            vmax=np.ma.masked_invalid(error).max(),
            vmin=np.ma.masked_invalid(error).min(),
        )
        axes[2].axis("off")
        cbar = fig.colorbar(im, ax=axes[2], orientation="vertical")
        cbar.ax.tick_params(labelsize=12)
        if axes_titles is None:
            axes[2].set_title(f"{var} Error")
        else:
            axes[2].set_title(axes_titles[2])

        MAE = np.mean(np.ma.masked_invalid((error)))

        if MAE:
            fig.suptitle(f"MAE {MAE}", fontsize=18, x=0.5)

    plt.tight_layout()

    path_to_save_path = os.path.join(save_path)
    plt.savefig(path_to_save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()


@wp.kernel
def _bvh_query_distance(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3f),
    max_dist: wp.float32,
    sdf: wp.array(dtype=wp.float32),
    sdf_hit_point: wp.array(dtype=wp.vec3f),
    sdf_hit_point_id: wp.array(dtype=wp.int32),
):

    """
    Computes the signed distance from each point in the given array `points`
    to the mesh represented by `mesh`,within the maximum distance `max_dist`,
    and stores the result in the array `sdf`.

    Parameters:
        mesh (wp.uint64): The identifier of the mesh.
        points (wp.array): An array of 3D points for which to compute the
            signed distance.
        max_dist (wp.float32): The maximum distance within which to search
            for the closest point on the mesh.
        sdf (wp.array): An array to store the computed signed distances.
        sdf_hit_point (wp.array): An array to store the computed hit points.
        sdf_hit_point_id (wp.array): An array to store the computed hit point ids.

    Returns:
        None
    """
    tid = wp.tid()

    res = wp.mesh_query_point_sign_winding_number(mesh, points[tid], max_dist)

    mesh_ = wp.mesh_get(mesh)

    p0 = mesh_.points[mesh_.indices[3 * res.face + 0]]
    p1 = mesh_.points[mesh_.indices[3 * res.face + 1]]
    p2 = mesh_.points[mesh_.indices[3 * res.face + 2]]

    p_closest = res.u * p0 + res.v * p1 + (1.0 - res.u - res.v) * p2

    sdf[tid] = res.sign * wp.abs(wp.length(points[tid] - p_closest))
    sdf_hit_point[tid] = p_closest
    sdf_hit_point_id[tid] = res.face


def signed_distance_field(
    mesh_vertices: list[tuple[float, float, float]],
    mesh_indices: NDArray[float],
    input_points: list[tuple[float, float, float]],
    max_dist: float = 1e8,
    include_hit_points: bool = False,
    include_hit_points_id: bool = False,
    device: int = 0,
) -> wp.array:
    """
    Computes the signed distance field (SDF) for a given mesh and input points.

    Parameters:
    ----------
        mesh_vertices (list[tuple[float, float, float]]): List of vertices defining the mesh.
        mesh_indices (list[tuple[int, int, int]]): List of indices defining the triangles of the mesh.
        input_points (list[tuple[float, float, float]]): List of input points for which to compute the SDF.
        max_dist (float, optional): Maximum distance within which to search for
            the closest point on the mesh. Default is 1e8.
        include_hit_points (bool, optional): Whether to include hit points in
            the output. Default is False.
        include_hit_points_id (bool, optional): Whether to include hit point
            IDs in the output. Default is False.

    Returns:
    -------
        wp.array: An array containing the computed signed distance field.

    Example:
    -------
    >>> mesh_vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    >>> mesh_indices = np.array((0, 1, 2))
    >>> input_points = [(0.5, 0.5, 0.5)]
    >>> signed_distance_field(mesh_vertices, mesh_indices, input_points).numpy()
    Module ...
    array([0.5], dtype=float32)
    """

    wp.init()
    # mesh = wp.Mesh(
    #     wp.array(mesh_vertices.cpu(), dtype=wp.vec3), wp.array(mesh_indices.cpu(), dtype=wp.int32)
    # )
    mesh = wp.Mesh(
        wp.from_torch(mesh_vertices, dtype=wp.vec3),
        wp.from_torch(mesh_indices, dtype=wp.int32),
    )

    sdf_points = wp.from_torch(input_points, dtype=wp.vec3)
    sdf = wp.zeros(shape=sdf_points.shape, dtype=wp.float32)
    sdf_hit_point = wp.zeros(shape=sdf_points.shape, dtype=wp.vec3f)
    sdf_hit_point_id = wp.zeros(shape=sdf_points.shape, dtype=wp.int32)
    wp.launch(
        kernel=_bvh_query_distance,
        dim=len(sdf_points),
        inputs=[mesh.id, sdf_points, max_dist, sdf, sdf_hit_point, sdf_hit_point_id],
    )
    if include_hit_points and include_hit_points_id:
        return (
            wp.to_torch(sdf),
            wp.to_torch(sdf_hit_point),
            wp.to_torch(sdf_hit_point_id),
        )
    elif include_hit_points:
        return (wp.to_torch(sdf), wp.to_torch(sdf_hit_point))
    elif include_hit_points_id:
        return (wp.to_torch(sdf), wp.to_torch(sdf_hit_point_id))
    else:
        return wp.to_torch(sdf)


def shuffle_array_torch(surface_vertices, geometry_points, device):
    idx = torch.unsqueeze(
        torch.randperm(surface_vertices.shape[0])[:geometry_points], -1
    ).to(device)
    idx = idx.repeat(1, 3)
    surface_sampled = torch.gather(surface_vertices, 0, idx)
    return surface_sampled


class inferenceDataPipe:
    def __init__(
        self,
        device: int = 0,
        grid_resolution: Optional[list] = [256, 96, 64],
        normalize_coordinates: bool = False,
        geom_points_sample: int = 300000,
        positional_encoding: bool = False,
        surface_vertices=None,
        surface_indices=None,
        surface_areas=None,
        surface_centers=None,
        use_sdf_basis=False,
    ):
        self.surface_vertices = surface_vertices
        self.surface_indices = surface_indices
        self.surface_areas = surface_areas
        self.surface_centers = surface_centers
        self.device = device
        self.grid_resolution = grid_resolution
        self.normalize_coordinates = normalize_coordinates
        self.geom_points_sample = geom_points_sample
        self.positional_encoding = positional_encoding
        self.use_sdf_basis = use_sdf_basis
        torch.manual_seed(int(42 + torch.cuda.current_device()))
        self.data_dict = {}

    def clear_dict(self):
        del self.data_dict

    def clear_volume_dict(self):
        del self.data_dict["volume_mesh_centers"]
        del self.data_dict["pos_enc_closest"]
        del self.data_dict["pos_normals_com"]
        del self.data_dict["sdf_nodes"]

    def create_grid_torch(self, mx, mn, nres):
        start_time = time.time()
        dx = torch.linspace(mn[0], mx[0], nres[0], device=self.device)
        dy = torch.linspace(mn[1], mx[1], nres[1], device=self.device)
        dz = torch.linspace(mn[2], mx[2], nres[2], device=self.device)

        xv, yv, zv = torch.meshgrid(dx, dy, dz, indexing="ij")
        xv = torch.unsqueeze(xv, -1)
        yv = torch.unsqueeze(yv, -1)
        zv = torch.unsqueeze(zv, -1)
        grid = torch.cat((xv, yv, zv), axis=-1)
        return grid

    def process_surface_mesh(self, bounding_box=None, bounding_box_surface=None):
        # Use coarse mesh to calculate SDF
        surface_vertices = self.surface_vertices
        surface_indices = self.surface_indices
        surface_areas = self.surface_areas
        surface_centers = self.surface_centers

        start_time = time.time()

        if bounding_box is None:
            # Create a bounding box
            s_max = torch.amax(surface_vertices, 0)
            s_min = torch.amin(surface_vertices, 0)

            c_max = s_max + (s_max - s_min) / 2
            c_min = s_min - (s_max - s_min) / 2
            c_min[2] = s_min[2]
        else:
            c_min = bounding_box[0]
            c_max = bounding_box[1]

        if bounding_box_surface is None:
            # Create a bounding box
            s_max = torch.amax(surface_vertices, 0)
            s_min = torch.amin(surface_vertices, 0)

            surf_max = s_max + (s_max - s_min) / 2
            surf_min = s_min - (s_max - s_min) / 2
            surf_min[2] = s_min[2]
        else:
            surf_min = bounding_box_surface[0]
            surf_max = bounding_box_surface[1]

        nx, ny, nz = self.grid_resolution

        grid = self.create_grid_torch(c_max, c_min, self.grid_resolution)
        grid_reshaped = torch.reshape(grid, (nx * ny * nz, 3))

        # SDF on grid
        sdf_grid = signed_distance_field(
            surface_vertices, surface_indices, grid_reshaped, device=self.device
        )
        sdf_grid = torch.reshape(sdf_grid, (nx, ny, nz))

        surface_areas = torch.unsqueeze(surface_areas, -1)
        center_of_mass = torch.sum(surface_centers * surface_areas, 0) / torch.sum(
            surface_areas
        )

        s_grid = self.create_grid_torch(surf_max, surf_min, self.grid_resolution)
        surf_grid_reshaped = torch.reshape(s_grid, (nx * ny * nz, 3))

        surf_sdf_grid = signed_distance_field(
            surface_vertices, surface_indices, surf_grid_reshaped, device=self.device
        )
        surf_sdf_grid = torch.reshape(surf_sdf_grid, (nx, ny, nz))

        if self.normalize_coordinates:
            grid = 2.0 * (grid - c_min) / (c_max - c_min) - 1.0
            s_grid = 2.0 * (s_grid - surf_min) / (surf_max - surf_min) - 1.0

        surface_vertices = torch.unsqueeze(surface_vertices, 0)
        grid = torch.unsqueeze(grid, 0)
        s_grid = torch.unsqueeze(s_grid, 0)
        sdf_grid = torch.unsqueeze(sdf_grid, 0)
        surf_sdf_grid = torch.unsqueeze(surf_sdf_grid, 0)
        max_min = [c_min, c_max]
        surf_max_min = [surf_min, surf_max]
        center_of_mass = center_of_mass

        return (
            surface_vertices,
            grid,
            sdf_grid,
            max_min,
            s_grid,
            surf_sdf_grid,
            surf_max_min,
            center_of_mass,
        )

    def sample_stl_points(
        self,
        num_points,
        stl_centers,
        stl_area,
        stl_normals,
        max_min,
        center_of_mass,
        bounding_box=None,
        stencil_size=7,
    ):
        if bounding_box is not None:
            c_max = bounding_box[1]
            c_min = bounding_box[0]
        else:
            c_min = max_min[0]
            c_max = max_min[1]

        start_time = time.time()

        nx, ny, nz = self.grid_resolution

        idx = np.arange(stl_centers.shape[0])
        # np.random.shuffle(idx)
        if num_points is not None:
            idx = idx[:num_points]

        surface_coordinates = stl_centers
        surface_normals = stl_normals
        surface_area = stl_area

        if stencil_size > 1:
            interp_func = KDTree(surface_coordinates)
            dd, ii = interp_func.query(surface_coordinates, k=stencil_size)
            surface_neighbors = surface_coordinates[ii]
            surface_neighbors = surface_neighbors[:, 1:] + 1e-6
            surface_neighbors_normals = surface_normals[ii]
            surface_neighbors_normals = surface_neighbors_normals[:, 1:]
            surface_neighbors_area = surface_area[ii]
            surface_neighbors_area = surface_neighbors_area[:, 1:]
        else:
            surface_neighbors = np.expand_dims(surface_coordinates, 1) + 1e-6
            surface_neighbors_normals = np.expand_dims(surface_normals, 1)
            surface_neighbors_area = np.expand_dims(surface_area, 1)

        surface_coordinates = torch.from_numpy(surface_coordinates).to(self.device)
        surface_normals = torch.from_numpy(surface_normals).to(self.device)
        surface_area = torch.from_numpy(surface_area).to(self.device)
        surface_neighbors = torch.from_numpy(surface_neighbors).to(self.device)
        surface_neighbors_normals = torch.from_numpy(surface_neighbors_normals).to(
            self.device
        )
        surface_neighbors_area = torch.from_numpy(surface_neighbors_area).to(
            self.device
        )

        pos_normals_com = surface_coordinates - center_of_mass

        if self.normalize_coordinates:
            surface_coordinates = (
                2.0 * (surface_coordinates - c_min) / (c_max - c_min) - 1.0
            )
            surface_neighbors = (
                2.0 * (surface_neighbors - c_min) / (c_max - c_min) - 1.0
            )

        surface_coordinates = surface_coordinates[idx]
        surface_area = surface_area[idx]
        surface_normals = surface_normals[idx]
        pos_normals_com = pos_normals_com[idx]
        surface_coordinates = torch.unsqueeze(surface_coordinates, 0)
        surface_normals = torch.unsqueeze(surface_normals, 0)
        surface_area = torch.unsqueeze(surface_area, 0)
        pos_normals_com = torch.unsqueeze(pos_normals_com, 0)

        surface_neighbors = surface_neighbors[idx]
        surface_neighbors_normals = surface_neighbors_normals[idx]
        surface_neighbors_area = surface_neighbors_area[idx]
        surface_neighbors = torch.unsqueeze(surface_neighbors, 0)
        surface_neighbors_normals = torch.unsqueeze(surface_neighbors_normals, 0)
        surface_neighbors_area = torch.unsqueeze(surface_neighbors_area, 0)

        scaling_factors = [c_max, c_min]

        return (
            surface_coordinates,
            surface_neighbors,
            surface_normals,
            surface_neighbors_normals,
            surface_area,
            surface_neighbors_area,
            pos_normals_com,
            scaling_factors,
            idx,
        )

    def sample_points_on_surface(
        self,
        num_points_surf,
        max_min,
        center_of_mass,
        stl_path,
        bounding_box=None,
        stencil_size=7,
    ):
        if bounding_box is not None:
            c_max = bounding_box[1]
            c_min = bounding_box[0]
        else:
            c_min = max_min[0]
            c_max = max_min[1]

        start_time = time.time()

        nx, ny, nz = self.grid_resolution

        obj = Tessellation.from_stl(stl_path, airtight=False)

        boundary = obj.sample_boundary(num_points_surf)
        surface_coordinates = np.concatenate(
            [
                np.float32(boundary["x"]),
                np.float32(boundary["y"]),
                np.float32(boundary["z"]),
            ],
            axis=1,
        )
        surface_normals = np.concatenate(
            [
                np.float32(boundary["normal_x"]),
                np.float32(boundary["normal_y"]),
                np.float32(boundary["normal_z"]),
            ],
            axis=1,
        )

        surface_area = np.float32(boundary["area"])

        interp_func = KDTree(surface_coordinates)
        dd, ii = interp_func.query(surface_coordinates, k=stencil_size)
        surface_neighbors = surface_coordinates[ii]
        surface_neighbors = surface_neighbors[:, 1:]
        surface_neighbors_normals = surface_normals[ii]
        surface_neighbors_normals = surface_neighbors_normals[:, 1:]
        surface_neighbors_area = surface_area[ii]
        surface_neighbors_area = surface_neighbors_area[:, 1:]

        surface_coordinates = torch.from_numpy(surface_coordinates).to(self.device)
        surface_normals = torch.from_numpy(surface_normals).to(self.device)
        surface_area = torch.from_numpy(surface_area).to(self.device)
        surface_neighbors = torch.from_numpy(surface_neighbors).to(self.device)
        surface_neighbors_normals = torch.from_numpy(surface_neighbors_normals).to(
            self.device
        )
        surface_neighbors_area = torch.from_numpy(surface_neighbors_area).to(
            self.device
        )

        pos_normals_com = surface_coordinates - center_of_mass

        if self.normalize_coordinates:
            surface_coordinates = (
                2.0 * (surface_coordinates - c_min) / (c_max - c_min) - 1.0
            )

        surface_coordinates = torch.unsqueeze(surface_coordinates, 0)
        surface_normals = torch.unsqueeze(surface_normals, 0)
        surface_area = torch.unsqueeze(surface_area, 0)
        pos_normals_com = torch.unsqueeze(pos_normals_com, 0)

        surface_neighbors = torch.unsqueeze(surface_neighbors, 0)
        surface_neighbors_normals = torch.unsqueeze(surface_neighbors_normals, 0)
        surface_neighbors_area = torch.unsqueeze(surface_neighbors_area, 0)

        scaling_factors = [c_max, c_min]

        return (
            surface_coordinates,
            surface_neighbors,
            surface_normals,
            surface_neighbors_normals,
            surface_area,
            surface_neighbors_area,
            pos_normals_com,
            scaling_factors,
        )

    def sample_points_in_volume(
        self, num_points_vol, max_min, center_of_mass, bounding_box=None
    ):

        if bounding_box is not None:
            c_max = bounding_box[1]
            c_min = bounding_box[0]
        else:
            c_min = max_min[0]
            c_max = max_min[1]

        start_time = time.time()

        nx, ny, nz = self.grid_resolution
        for k in range(10):
            if k > 0:
                num_pts_vol = num_points_vol - int(volume_coordinates.shape[0] / 2)
            else:
                num_pts_vol = int(1.25 * num_points_vol)

            volume_coordinates_sub = (c_max - c_min) * torch.rand(
                num_pts_vol, 3, device=self.device, dtype=torch.float32
            ) + c_min

            sdf_nodes, sdf_node_closest_point = signed_distance_field(
                self.surface_vertices,
                self.surface_indices,
                volume_coordinates_sub,
                include_hit_points=True,
                device=self.device,
            )
            sdf_nodes = torch.unsqueeze(sdf_nodes, -1)

            idx = torch.unsqueeze(torch.where((sdf_nodes > 0))[0], -1)
            idx = idx.repeat(1, volume_coordinates_sub.shape[1])
            if k == 0:
                volume_coordinates = torch.gather(volume_coordinates_sub, 0, idx)
            else:
                volume_coordinates_1 = torch.gather(volume_coordinates_sub, 0, idx)
                volume_coordinates = torch.cat(
                    (volume_coordinates, volume_coordinates_1), axis=0
                )

            if volume_coordinates.shape[0] > num_points_vol:
                volume_coordinates = volume_coordinates[:num_points_vol]
                break

        sdf_nodes, sdf_node_closest_point = signed_distance_field(
            self.surface_vertices,
            self.surface_indices,
            volume_coordinates,
            include_hit_points=True,
            device=self.device,
        )
        sdf_nodes = torch.unsqueeze(sdf_nodes, -1)

        pos_normals_closest = volume_coordinates - sdf_node_closest_point
        pos_normals_com = volume_coordinates - center_of_mass

        if self.normalize_coordinates:
            volume_coordinates = (
                2.0 * (volume_coordinates - c_min) / (c_max - c_min) - 1.0
            )

        volume_coordinates = torch.unsqueeze(volume_coordinates, 0)
        pos_normals_com = torch.unsqueeze(pos_normals_com, 0)

        if self.use_sdf_basis:
            pos_normals_closest = torch.unsqueeze(pos_normals_closest, 0)
            sdf_nodes = torch.unsqueeze(sdf_nodes, 0)

        scaling_factors = [c_max, c_min]
        return (
            volume_coordinates,
            pos_normals_com,
            pos_normals_closest,
            sdf_nodes,
            scaling_factors,
        )


class dominoInference:
    def __init__(
        self,
        cfg: DictConfig,
        dist: None,
        cached_geo_encoding: False,
    ):

        self.cfg = cfg
        self.dist = dist
        self.stream_velocity = None
        self.stencil_size = None
        self.stl_path = None
        self.stl_vertices = None
        self.stl_centers = None
        self.surface_areas = None
        self.mesh_indices_flattened = None
        self.length_scale = 1.0
        if self.dist is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = self.dist.device

        self.air_density = torch.full((1, 1), 1.205, dtype=torch.float32).to(
            self.device
        )
        self.num_vol_vars, self.num_surf_vars = self.get_num_variables()
        self.model = None
        self.grid_resolution = torch.tensor(self.cfg.model.interp_res).to(self.device)
        self.vol_factors = None
        self.bounding_box_min_max = None
        self.bounding_box_surface_min_max = None
        self.center_of_mass = None
        self.grid = None
        self.geometry_encoding = None
        self.geometry_encoding_surface = None
        self.cached_geo_encoding = cached_geo_encoding
        self.out_dict = {}

    def get_geometry_encoding(self):
        return self.geometry_encoding

    def get_geometry_encoding_surface(self):
        return self.geometry_encoding_surface

    def get_out_dict(self):
        return self.out_dict

    def clear_out_dict(self):
        self.out_dict.clear()

    def initialize_data_processor(self):
        self.ifp = inferenceDataPipe(
            device=self.device,
            surface_vertices=self.stl_vertices,
            surface_indices=self.mesh_indices_flattened,
            surface_areas=self.surface_areas,
            surface_centers=self.stl_centers,
            grid_resolution=self.grid_resolution,
            normalize_coordinates=True,
            geom_points_sample=300000,
            positional_encoding=False,
            use_sdf_basis=self.cfg.model.use_sdf_in_basis_func,
        )

    def load_bounding_box(self):
        if (
            self.cfg.data.bounding_box.min is not None
            and self.cfg.data.bounding_box.max is not None
        ):
            c_min = torch.from_numpy(
                np.array(self.cfg.data.bounding_box.min, dtype=np.float32)
            ).to(self.device)
            c_max = torch.from_numpy(
                np.array(self.cfg.data.bounding_box.max, dtype=np.float32)
            ).to(self.device)
            self.bounding_box_min_max = [c_min, c_max]

        if (
            self.cfg.data.bounding_box_surface.min is not None
            and self.cfg.data.bounding_box_surface.max is not None
        ):
            c_min = torch.from_numpy(
                np.array(self.cfg.data.bounding_box_surface.min, dtype=np.float32)
            ).to(self.device)
            c_max = torch.from_numpy(
                np.array(self.cfg.data.bounding_box_surface.max, dtype=np.float32)
            ).to(self.device)
            self.bounding_box_surface_min_max = [c_min, c_max]

    def load_volume_scaling_factors(self):
        vol_factors = np.array(
            [
                [2.2642279,  2.2397292,  1.8689916,  0.7547227],
                [
                    -1.2899836, -2.2787743, -1.866153 , -2.7116761
                ],
            ],
            dtype=np.float32,
        )

        vol_factors = torch.from_numpy(vol_factors).to(self.device)

        return vol_factors

    def load_surface_scaling_factors(self):
        surf_factors = np.array(
            [
                [0.8215038 ,  0.01063187,  0.01514608,  0.01327803],
                [-2.1505525 , -0.01865184, -0.01514422, -0.0121509],
            ],
            dtype=np.float32,
        )

        surf_factors = torch.from_numpy(surf_factors).to(self.device)
        return surf_factors

    def read_stl(self):
        stl_files = get_filenames(self.stl_path)
        mesh_stl = combine_stls(self.stl_path, stl_files)
        if self.cfg.eval.refine_stl:
            mesh_stl = mesh_stl.subdivide(nsub=2, subfilter='linear')#.smooth(n_iter=20)
        stl_vertices = mesh_stl.points
        length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))
        stl_centers = mesh_stl.cell_centers().points
        # Assuming triangular elements
        stl_faces = np.array(mesh_stl.faces).reshape((-1, 4))[:, 1:]
        mesh_indices_flattened = stl_faces.flatten()

        surface_areas = mesh_stl.compute_cell_sizes(
            length=False, area=True, volume=False
        )
        surface_areas = np.array(surface_areas.cell_data["Area"])

        surface_normals = np.array(mesh_stl.cell_normals, dtype=np.float32)

        self.stl_vertices = torch.from_numpy(np.float32(stl_vertices)).to(self.device)
        self.stl_centers = torch.from_numpy(np.float32(stl_centers)).to(self.device)
        self.surface_areas = torch.from_numpy(np.float32(surface_areas)).to(self.device)
        self.stl_normals = -1.0 * torch.from_numpy(np.float32(surface_normals)).to(
            self.device
        )
        self.mesh_indices_flattened = torch.from_numpy(
            np.int32(mesh_indices_flattened)
        ).to(self.device)
        self.length_scale = length_scale
        self.mesh_stl = mesh_stl

    def read_stl_trimesh(
        self, stl_vertices, stl_faces, stl_centers, surface_normals, surface_areas
    ):
        mesh_indices_flattened = stl_faces.flatten()
        length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))
        self.stl_vertices = torch.from_numpy(stl_vertices).to(self.device)
        self.stl_centers = torch.from_numpy(stl_centers).to(self.device)
        self.stl_normals = -1.0 * torch.from_numpy(surface_normals).to(self.device)
        self.surface_areas = torch.from_numpy(surface_areas).to(self.device)
        self.mesh_indices_flattened = torch.from_numpy(
            np.int32(mesh_indices_flattened)
        ).to(self.device)
        self.length_scale = length_scale

    def get_num_variables(self):
        volume_variable_names = list(self.cfg.variables.volume.solution.keys())
        num_vol_vars = 0
        for j in volume_variable_names:
            if self.cfg.variables.volume.solution[j] == "vector":
                num_vol_vars += 3
            else:
                num_vol_vars += 1

        surface_variable_names = list(self.cfg.variables.surface.solution.keys())
        num_surf_vars = 0
        for j in surface_variable_names:
            if self.cfg.variables.surface.solution[j] == "vector":
                num_surf_vars += 3
            else:
                num_surf_vars += 1
        return num_vol_vars, num_surf_vars

    def initialize_model(self, model_path):
        model = (
            DoMINO(
                input_features=3,
                output_features_vol=self.num_vol_vars,
                output_features_surf=self.num_surf_vars,
                model_parameters=self.cfg.model,
            )
            .to(self.device)
            .eval()
        )
        model = torch.compile(model, disable=True)

        checkpoint_iter = torch.load(
            to_absolute_path(model_path), map_location=self.dist.device
        )

        model.load_state_dict(checkpoint_iter)

        if self.dist is not None:
            if self.dist.world_size > 1:
                model = DistributedDataParallel(
                    model,
                    device_ids=[self.dist.local_rank],
                    output_device=self.dist.device,
                    broadcast_buffers=self.dist.broadcast_buffers,
                    find_unused_parameters=self.dist.find_unused_parameters,
                    gradient_as_bucket_view=True,
                    static_graph=True,
                )

        self.model = model
        self.vol_factors = self.load_volume_scaling_factors()
        self.surf_factors = self.load_surface_scaling_factors()
        self.load_bounding_box()

    def set_stream_velocity(self, stream_velocity):
        self.stream_velocity = torch.full(
            (1, 1), stream_velocity, dtype=torch.float32
        ).to(self.device)

    def set_stencil_size(self, stencil_size):
        self.stencil_size = stencil_size

    def set_air_density(self, air_density):
        self.air_density = torch.full((1, 1), air_density, dtype=torch.float32).to(
            self.device
        )

    def set_stl_path(self, filename):
        self.stl_path = filename

    @torch.no_grad()
    def compute_geo_encoding(self, cached_geom_path=None):
        start_time = time.time()

        if not self.cached_geo_encoding:
            (
                surface_vertices,
                grid,
                sdf_grid,
                max_min,
                s_grid,
                surf_sdf_grid,
                surf_max_min,
                center_of_mass,
            ) = self.ifp.process_surface_mesh(
                self.bounding_box_min_max, self.bounding_box_surface_min_max
            )
            if self.bounding_box_min_max is None:
                self.bounding_box_min_max = max_min
            if self.bounding_box_surface_min_max is None:
                self.bounding_box_surface_min_max = surf_max_min
            self.center_of_mass = center_of_mass
            self.grid = grid
            self.s_grid = s_grid
            self.sdf_grid = sdf_grid
            self.surf_sdf_grid = surf_sdf_grid
            self.out_dict["sdf"] = sdf_grid

            geo_encoding, geo_encoding_surface = self.calculate_geometry_encoding(
                surface_vertices, grid, sdf_grid, s_grid, surf_sdf_grid, self.model
            )
        else:
            out_dict_cached = torch.load(cached_geom_path, map_location=self.device)
            self.bounding_box_min_max = out_dict_cached["bounding_box_min_max"]
            self.grid = out_dict_cached["grid"]
            self.sdf_grid = out_dict_cached["sdf_grid"]
            self.center_of_mass = out_dict_cached["com"]
            geo_encoding = out_dict_cached["geo_encoding"]
            geo_encoding_surface = out_dict_cached["geo_encoding_surface"]
            self.out_dict["sdf"] = self.sdf_grid
        torch.cuda.synchronize()
        print("Time taken for geo encoding = %f" % (time.time() - start_time))

        self.geometry_encoding = geo_encoding
        self.geometry_encoding_surface = geo_encoding_surface

    def compute_forces(self):
        pressure = self.out_dict["pressure_surface"]
        wall_shear = self.out_dict["wall-shear-stress"]
        # sampling_indices = self.out_dict["sampling_indices"]

        surface_normals = self.stl_normals[self.sampling_indices]
        surface_areas = self.surface_areas[self.sampling_indices]

        drag_force = torch.sum(
            pressure[0, :, 0] * surface_normals[:, 0] * surface_areas
            - wall_shear[0, :, 0] * surface_areas
        )
        lift_force = torch.sum(
            pressure[0, :, 0] * surface_normals[:, 2] * surface_areas
            - wall_shear[0, :, 2] * surface_areas
        )

        self.out_dict["drag_force"] = drag_force
        self.out_dict["lift_force"] = lift_force

    @torch.inference_mode()
    def compute_surface_solutions(self, num_sample_points=None, plot_solutions=False):
        total_time = 0.0
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        geo_encoding = self.geometry_encoding_surface
        j = 0

        with autocast(enabled=True):
            start_event.record()
            (
                surface_mesh_centers,
                surface_neighbors,
                surface_normals,
                surface_neighbors_normals,
                surface_areas,
                surface_neighbors_areas,
                pos_normals_com,
                surf_scaling_factors,
                sampling_indices,
            ) = self.ifp.sample_stl_points(
                num_sample_points,
                self.stl_centers.cpu().numpy(),
                self.surface_areas.cpu().numpy(),
                self.stl_normals.cpu().numpy(),
                max_min=self.bounding_box_surface_min_max,
                center_of_mass=self.center_of_mass,
                stencil_size=self.stencil_size,
            )
            end_event.record()
            end_event.synchronize()
            cur_time = start_event.elapsed_time(end_event) / 1000.0
            print(f"sample_points_in_surface time (s): {cur_time:.4f}")
            # vol_coordinates_all.append(volume_mesh_centers)
            surface_coordinates_all = surface_mesh_centers

            inner_time = time.time()
            start_event.record()
            if num_sample_points == None:
                point_batch_size = 512_000
                num_points = surface_coordinates_all.shape[1]
                subdomain_points = int(np.floor(num_points / point_batch_size))
                surface_solutions = torch.zeros(1, num_points, self.num_surf_vars).to(
                    self.device
                )
                for p in range(subdomain_points + 1):
                    start_idx = p * point_batch_size
                    end_idx = (p + 1) * point_batch_size
                    surface_solutions_batch = self.compute_solution_on_surface(
                        geo_encoding,
                        surface_mesh_centers[:, start_idx:end_idx],
                        surface_neighbors[:, start_idx:end_idx],
                        surface_normals[:, start_idx:end_idx],
                        surface_neighbors_normals[:, start_idx:end_idx],
                        surface_areas[:, start_idx:end_idx],
                        surface_neighbors_areas[:, start_idx:end_idx],
                        pos_normals_com[:, start_idx:end_idx],
                        self.s_grid,
                        self.model,
                        inlet_velocity=self.stream_velocity,
                        air_density=self.air_density,
                    )
                    surface_solutions[:, start_idx:end_idx] = surface_solutions_batch
            else:
                point_batch_size = 512_000
                num_points = num_sample_points
                subdomain_points = int(np.floor(num_points / point_batch_size))
                surface_solutions = torch.zeros(1, num_points, self.num_surf_vars).to(
                    self.device
                )
                for p in range(subdomain_points + 1):
                    start_idx = p * point_batch_size
                    end_idx = (p + 1) * point_batch_size
                    surface_solutions_batch = self.compute_solution_on_surface(
                        geo_encoding,
                        surface_mesh_centers[:, start_idx:end_idx],
                        surface_neighbors[:, start_idx:end_idx],
                        surface_normals[:, start_idx:end_idx],
                        surface_neighbors_normals[:, start_idx:end_idx],
                        surface_areas[:, start_idx:end_idx],
                        surface_neighbors_areas[:, start_idx:end_idx],
                        pos_normals_com[:, start_idx:end_idx],
                        self.s_grid,
                        self.model,
                        inlet_velocity=self.stream_velocity,
                        air_density=self.air_density,
                    )
                    # print(torch.amax(surface_solutions_batch, (0, 1)), torch.amin(surface_solutions_batch, (0, 1)))
                    surface_solutions[:, start_idx:end_idx] = surface_solutions_batch

            # print(surface_solutions.shape)
            end_event.record()
            end_event.synchronize()
            cur_time = start_event.elapsed_time(end_event) / 1000.0
            print(f"compute_solution time (s): {cur_time:.4f}")
            total_time += float(time.time() - inner_time)
            surface_solutions_all = surface_solutions
            print(
                "Time taken for compute solution on surface for=%f, %f"
                % (time.time() - inner_time, torch.cuda.utilization(self.device))
            )
        cmax = surf_scaling_factors[0]
        cmin = surf_scaling_factors[1]

        surface_coordinates_all = torch.reshape(
            surface_coordinates_all, (1, num_points, 3)
        )
        surface_solutions_all = torch.reshape(surface_solutions_all, (1, num_points, 4))

        if self.surf_factors is not None:
            surface_solutions_all = unnormalize(
                surface_solutions_all, self.surf_factors[0], self.surf_factors[1]
            )

        self.out_dict["surface_coordinates"] = (
            0.5 * (surface_coordinates_all + 1.0) * (cmax - cmin) + cmin
        )
        self.out_dict["pressure_surface"] = (
            surface_solutions_all[:, :, :1]
            * self.stream_velocity**2.0
            * self.air_density
        )
        self.out_dict["wall-shear-stress"] = (
            surface_solutions_all[:, :, 1:4]
            * self.stream_velocity**2.0
            * self.air_density
        )
        self.sampling_indices = sampling_indices

    @torch.inference_mode()
    def compute_volume_solutions(self, num_sample_points, plot_solutions=False):
        total_time = 0.0
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        geo_encoding = self.geometry_encoding
        j = 0

        # Compute volume
        point_batch_size = 512_000
        num_points = num_sample_points
        subdomain_points = int(np.floor(num_points / point_batch_size))
        volume_solutions = torch.zeros(1, num_points, self.num_vol_vars).to(self.device)
        volume_coordinates = torch.zeros(1, num_points, 3).to(self.device)

        for p in range(subdomain_points + 1):
            start_idx = p * point_batch_size
            end_idx = (p + 1) * point_batch_size
            if end_idx > num_points:
                point_batch_size = num_points - start_idx
                end_idx = num_points

            with autocast(enabled=True):
                inner_time = time.time()
                start_event.record()
                volume_mesh_centers, pos_normals_com, pos_normals_closest, sdf_nodes, scaling_factors = (
                    self.ifp.sample_points_in_volume(
                        num_points_vol=point_batch_size,
                        max_min=self.bounding_box_min_max,
                        center_of_mass=self.center_of_mass,
                    )
                )
                end_event.record()
                end_event.synchronize()
                cur_time = start_event.elapsed_time(end_event) / 1000.0
                print(f"sample_points_in_volume time (s): {cur_time:.4f}")

                volume_coordinates[:, start_idx:end_idx] = volume_mesh_centers

                start_event.record()
                
                volume_solutions_batch = self.compute_solution_in_volume(
                    geo_encoding,
                    volume_mesh_centers,
                    sdf_nodes,
                    pos_normals_closest,
                    pos_normals_com,
                    self.grid,
                    self.model,
                    use_sdf_basis=self.cfg.model.use_sdf_in_basis_func,
                    inlet_velocity=self.stream_velocity,
                    air_density=self.air_density,
                )
                volume_solutions[:, start_idx:end_idx] = volume_solutions_batch
                end_event.record()
                end_event.synchronize()
                cur_time = start_event.elapsed_time(end_event) / 1000.0
                print(f"compute_solution time (s): {cur_time:.4f}")
                total_time += float(time.time() - inner_time)
                # volume_solutions_all = volume_solutions
                print(
                    "Time taken for compute solution in volume for =%f"
                    % (time.time() - inner_time)
                )
                # print("Points processed:", end_idx)
            print("Total time measured = %f" % total_time)
            print("Points processed:", end_idx)

        cmax = scaling_factors[0]
        cmin = scaling_factors[1]
        volume_coordinates_all = volume_coordinates
        volume_solutions_all = volume_solutions

        cmax = scaling_factors[0]
        cmin = scaling_factors[1]

        volume_coordinates_all = torch.reshape(
            volume_coordinates_all, (1, num_sample_points, 3)
        )
        volume_solutions_all = torch.reshape(
            volume_solutions_all, (1, num_sample_points, self.num_vol_vars)
        )

        if self.vol_factors is not None:
            volume_solutions_all = unnormalize(
                volume_solutions_all, self.vol_factors[0], self.vol_factors[1]
            )

        self.out_dict["coordinates"] = (
            0.5 * (volume_coordinates_all + 1.0) * (cmax - cmin) + cmin
        )
        self.out_dict["velocity"] = (
            volume_solutions_all[:, :, :3] * self.stream_velocity
        )
        self.out_dict["pressure"] = (
            volume_solutions_all[:, :, 3:4]
            * self.stream_velocity**2.0
            * self.air_density
        )
        # self.out_dict["turbulent-kinetic-energy"] = (
        #     volume_solutions_all[:, :, 4:5]
        #     * self.stream_velocity**2.0
        #     * self.air_density
        # )
        # self.out_dict["turbulent-viscosity"] = (
        #     volume_solutions_all[:, :, 5:] * self.stream_velocity * self.length_scale
        # )
        self.out_dict["bounding_box_dims"] = torch.vstack(self.bounding_box_min_max)

        if plot_solutions:
            print("Plotting solutions")
            plot_save_path = os.path.join(self.cfg.output, "plots/contours/")
            create_directory(plot_save_path)

            p_grid = 0.5 * (self.grid + 1.0) * (cmax - cmin) + cmin
            p_grid = p_grid.cpu().numpy()
            sdf_grid = self.sdf_grid.cpu().numpy()
            volume_coordinates_all = (
                0.5 * (volume_coordinates_all + 1.0) * (cmax - cmin) + cmin
            )
            volume_solutions_all[:, :, :3] = (
                volume_solutions_all[:, :, :3] * self.stream_velocity
            )
            volume_solutions_all[:, :, 3:4] = (
                volume_solutions_all[:, :, 3:4]
                * self.stream_velocity**2.0
                * self.air_density
            )
            # volume_solutions_all[:, :, 4:5] = (
            #     volume_solutions_all[:, :, 4:5]
            #     * self.stream_velocity**2.0
            #     * self.air_density
            # )
            # volume_solutions_all[:, :, 5] = (
            #     volume_solutions_all[:, :, 5] * self.stream_velocity * self.length_scale
            # )
            volume_coordinates_all = volume_coordinates_all.cpu().numpy()
            volume_solutions_all = volume_solutions_all.cpu().numpy()

            # ND interpolation on a grid
            prediction_grid = nd_interpolator(
                volume_coordinates_all, volume_solutions_all[0], p_grid[0]
            )
            nx, ny, nz, vars = prediction_grid.shape
            idx = np.where(sdf_grid[0] < 0.0)
            prediction_grid[idx] = float("inf")
            axes_titles = ["y/4 plane", "y/2 plane"]

            plot(
                prediction_grid[:, int(ny / 4), :, 0],
                prediction_grid[:, int(ny / 2), :, 0],
                var="x-vel",
                save_path=plot_save_path + f"x-vel-midplane_{self.stream_velocity}.png",
                axes_titles=axes_titles,
                plot_error=False,
            )
            plot(
                prediction_grid[:, int(ny / 4), :, 1],
                prediction_grid[:, int(ny / 2), :, 1],
                var="y-vel",
                save_path=plot_save_path + f"y-vel-midplane_{self.stream_velocity}.png",
                axes_titles=axes_titles,
                plot_error=False,
            )
            plot(
                prediction_grid[:, int(ny / 4), :, 2],
                prediction_grid[:, int(ny / 2), :, 2],
                var="z-vel",
                save_path=plot_save_path + f"z-vel-midplane_{self.stream_velocity}.png",
                axes_titles=axes_titles,
                plot_error=False,
            )
            plot(
                prediction_grid[:, int(ny / 4), :, 3],
                prediction_grid[:, int(ny / 2), :, 3],
                var="pres",
                save_path=plot_save_path + f"pres-midplane_{self.stream_velocity}.png",
                axes_titles=axes_titles,
                plot_error=False,
            )
            # plot(
            #     prediction_grid[:, int(ny / 4), :, 4],
            #     prediction_grid[:, int(ny / 2), :, 4],
            #     var="tke",
            #     save_path=plot_save_path + f"tke-midplane_{self.stream_velocity}.png",
            #     axes_titles=axes_titles,
            #     plot_error=False,
            # )
            # plot(
            #     prediction_grid[:, int(ny / 4), :, 5],
            #     prediction_grid[:, int(ny / 2), :, 5],
            #     var="nut",
            #     save_path=plot_save_path + f"nut-midplane_{self.stream_velocity}.png",
            #     axes_titles=axes_titles,
            #     plot_error=False,
            # )

    def cold_start(self, cached_geom_path=None):
        print("Cold start")
        self.compute_geo_encoding(cached_geom_path)
        self.compute_volume_solutions(num_sample_points=10)
        self.clear_out_dict()

    @torch.no_grad()
    def calculate_geometry_encoding(
        self, geo_centers, p_grid, sdf_grid, s_grid, sdf_surf_grid, model
    ):

        vol_min = self.bounding_box_min_max[0]
        vol_max = self.bounding_box_min_max[1]
        surf_min = self.bounding_box_surface_min_max[0]
        surf_max = self.bounding_box_surface_min_max[1]

        geo_centers_vol = 2.0 * (geo_centers - vol_min) / (vol_max - vol_min) - 1
        if self.dist.world_size == 1:
            encoding_g_vol = model.geo_rep_volume(geo_centers_vol, p_grid, sdf_grid)
        else:
            encoding_g_vol = model.module.geo_rep_volume(geo_centers_vol, p_grid, sdf_grid)

        geo_centers_surf = 2.0 * (geo_centers - surf_min) / (surf_max - surf_min) - 1

        if self.dist.world_size == 1:
            encoding_g_surf = model.geo_rep_surface(geo_centers_surf, s_grid, sdf_surf_grid)
        else:
            encoding_g_surf = model.module.geo_rep_surface(
                geo_centers_surf, s_grid, sdf_surf_grid
            )

        if self.dist.world_size == 1:
            encoding_g_surf1 = model.geo_rep_surface1(geo_centers_surf, s_grid, sdf_surf_grid)
        else:
            encoding_g_surf1 = model.module.geo_rep_surface1(
                geo_centers_surf, s_grid, sdf_surf_grid
            )

        geo_encoding = 0.5 * encoding_g_surf1 + 0.5 * encoding_g_vol
        geo_encoding_surface = 0.5 * encoding_g_surf
        return geo_encoding, geo_encoding_surface

    @torch.no_grad()
    def compute_solution_on_surface(
        self,
        geo_encoding,
        surface_mesh_centers,
        surface_mesh_neighbors,
        surface_normals,
        surface_neighbors_normals,
        surface_areas,
        surface_neighbors_areas,
        pos_normals_com,
        s_grid,
        model,
        inlet_velocity,
        air_density,
    ):

        if self.dist.world_size == 1:
            geo_encoding_local = model.geo_encoding_local(
                geo_encoding, surface_mesh_centers, s_grid, mode="surface"
            )
        else:
            geo_encoding_local = model.module.geo_encoding_local(
                geo_encoding, surface_mesh_centers, s_grid, mode="surface"
            )

        pos_encoding = pos_normals_com
        surface_areas = torch.unsqueeze(surface_areas, -1)
        surface_neighbors_areas = torch.unsqueeze(surface_neighbors_areas, -1)

        if self.dist.world_size == 1:
            pos_encoding = model.position_encoder(pos_encoding, eval_mode="surface")
            tpredictions_batch = model.calculate_solution_with_neighbors(
                surface_mesh_centers,
                geo_encoding_local,
                pos_encoding,
                surface_mesh_neighbors,
                surface_normals,
                surface_neighbors_normals,
                surface_areas,
                surface_neighbors_areas,
                inlet_velocity,
                air_density,
            )
        else:
            pos_encoding = model.module.position_encoder(
                pos_encoding, eval_mode="surface"
            )
            tpredictions_batch = model.module.calculate_solution_with_neighbors(
                surface_mesh_centers,
                geo_encoding_local,
                pos_encoding,
                surface_mesh_neighbors,
                surface_normals,
                surface_neighbors_normals,
                surface_areas,
                surface_neighbors_areas,
                inlet_velocity,
                air_density,
            )

        return tpredictions_batch

    @torch.no_grad()
    def compute_solution_in_volume(
        self,
        geo_encoding,
        volume_mesh_centers,
        sdf_nodes,
        pos_enc_closest,
        pos_normals_com,
        p_grid,
        model,
        use_sdf_basis,
        inlet_velocity,
        air_density,
    ):

        if self.dist.world_size == 1:
            geo_encoding_local = model.geo_encoding_local(
                geo_encoding, volume_mesh_centers, p_grid, mode="volume"
            )
        else:
            geo_encoding_local = model.module.geo_encoding_local(
                geo_encoding, volume_mesh_centers, p_grid, mode="volume"
            )
        if use_sdf_basis:
            pos_encoding = torch.cat(
                (sdf_nodes, pos_enc_closest, pos_normals_com), axis=-1
            )
        else:
            pos_encoding = pos_normals_com

        if self.dist.world_size == 1:
            pos_encoding = model.position_encoder(pos_encoding, eval_mode="volume")
            tpredictions_batch = model.calculate_solution(
                volume_mesh_centers,
                geo_encoding_local,
                pos_encoding,
                inlet_velocity,
                air_density,
                num_sample_points=self.stencil_size,
                eval_mode="volume",
            )
        else:
            pos_encoding = model.module.position_encoder(
                pos_encoding, eval_mode="volume"
            )
            tpredictions_batch = model.module.calculate_solution(
                volume_mesh_centers,
                geo_encoding_local,
                pos_encoding,
                inlet_velocity,
                air_density,
                num_sample_points=self.stencil_size,
                eval_mode="volume",
            )
        return tpredictions_batch


if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval)
    with initialize(version_base="1.3", config_path="conf"):
        cfg = compose(config_name="config")

    DistributedManager.initialize()
    dist = DistributedManager()

    if dist.world_size > 1:
        torch.distributed.barrier()

    input_path = cfg.eval.test_path
    dirnames = get_filenames(input_path)
    dev_id = torch.cuda.current_device()
    num_files = int(len(dirnames) / 8)
    dirnames_per_gpu = dirnames[int(num_files * dev_id) : int(num_files * (dev_id + 1))]

    domino = dominoInference(cfg, dist, False)
    domino.initialize_model(
        model_path="/lustre/rranade/modulus_dev/modulus_forked/modulus/examples/cfd/external_aerodynamics/domino/outputs/AWS_Dataset/19/models/DoMINO.0.201.pt"
    )

    for count, dirname in enumerate(dirnames_per_gpu):
        # print(f"Processing file {dirname}")
        filepath = os.path.join(input_path, dirname)

        STREAM_VELOCITY = 30.0
        AIR_DENSITY = 1.205

        # Neighborhood points sampled for evaluation, tradeoff between accuracy and speed
        STENCIL_SIZE = (
            7  # Higher stencil size -> more accuracy but more evaluation time
        )

        domino.set_stl_path(filepath)
        domino.set_stream_velocity(STREAM_VELOCITY)
        domino.set_stencil_size(STENCIL_SIZE)

        domino.read_stl()

        domino.initialize_data_processor()

        # Calculate geometry encoding
        domino.compute_geo_encoding()

        # Calculate volume solutions
        domino.compute_volume_solutions(
            num_sample_points=10_256_000, plot_solutions=False
        )

        # Calculate surface solutions
        domino.compute_surface_solutions()
        domino.compute_forces()
        out_dict = domino.get_out_dict()

        print(
            "Dirname:",
            dirname,
            "Drag:",
            out_dict["drag_force"],
            "Lift:",
            out_dict["lift_force"],
        )
        vtp_path = f"/lustre/rranade/modulus_dev/modulus_forked/modulus/examples/cfd/external_aerodynamics/domino/pred_{dirname}_4.vtp"
        domino.mesh_stl.save(vtp_path)
        # vtp_path = f"/lustre/rranade/modulus_dev/modulus_demo/modulus_rishi/modulus/examples/cfd/external_aerodynamics/domino_gtc_demo/sensitivity_pred_{dirname}.vtp"
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(f"{vtp_path}")
        reader.Update()
        polydata_surf = reader.GetOutput()

        surfParam_vtk = numpy_support.numpy_to_vtk(out_dict["pressure_surface"][0].cpu().numpy())
        surfParam_vtk.SetName(f"Pressure")
        polydata_surf.GetCellData().AddArray(surfParam_vtk)

        surfParam_vtk = numpy_support.numpy_to_vtk(out_dict["wall-shear-stress"][0].cpu().numpy())
        surfParam_vtk.SetName(f"Wall-shear-stress")
        polydata_surf.GetCellData().AddArray(surfParam_vtk)

        write_to_vtp(polydata_surf, vtp_path)
        exit()
