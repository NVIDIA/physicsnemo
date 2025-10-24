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
This code processes mesh data from .stl and .vtp files to create partitioned
graphs for large scale training. It first converts meshes to triangular format
and extracts surface triangles, vertices, and relevant attributes such as pressure
and shear stress. Using nearest neighbors, the code interpolates these attributes
for a sampled boundary of points, and constructs a graph based on these points, with
node features like coordinates, normals, pressure, and shear stress, as well as edge
features representing relative displacement. The graph is partitioned into subgraphs,
and the partitions are saved. The code supports parallel processing to handle multiple
samples simultaneously, improving efficiency. Additionally, it provides an option to
save the point cloud of each graph for visualization purposes.
"""

import os
import vtk
import pyvista as pv
import numpy as np
import torch
import dgl
import hydra

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import NearestNeighbors
from dgl.data.utils import save_graphs
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from physicsnemo.datapipes.cae.readers import read_vtp
from physicsnemo.sym.geometry.tessellation import Tessellation


def convert_to_triangular_mesh(
    polydata, write=False, output_filename="surface_mesh_triangular.vtu"
):
    """Converts a vtkPolyData object to a triangular mesh."""
    tet_filter = vtk.vtkDataSetTriangleFilter()
    tet_filter.SetInputData(polydata)
    tet_filter.Update()

    tet_mesh = pv.wrap(tet_filter.GetOutput())

    if write:
        tet_mesh.save(output_filename)

    return tet_mesh


def extract_surface_triangles(tet_mesh):
    """Extracts the surface triangles from a triangular mesh."""
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(tet_mesh)
    surface_filter.Update()

    surface_mesh = pv.wrap(surface_filter.GetOutput())
    triangle_indices = []
    faces = surface_mesh.faces.reshape((-1, 4))
    for face in faces:
        if face[0] == 3:
            triangle_indices.extend([face[1], face[2], face[3]])
        else:
            raise ValueError("Face is not a triangle")

    return triangle_indices


def fetch_mesh_vertices(mesh):
    """Fetches the vertices of a mesh."""
    points = mesh.GetPoints()
    num_points = points.GetNumberOfPoints()
    vertices = [points.GetPoint(i) for i in range(num_points)]
    return vertices


def add_edge_features(graph):
    """
    Add relative displacement and displacement norm as edge features to the graph.
    The calculations are done using the 'pos' attribute in the
    node data of each graph. The resulting edge features are stored in the 'x' attribute
    in the edge data of each graph.

    This method will modify the graph in-place.

    Returns
    -------
    dgl.DGLGraph
        Graph with updated edge features.
    """

    pos = graph.ndata.get("coordinates")
    if pos is None:
        raise ValueError(
            "'coordinates' does not exist in the node data of one or more graphs."
        )

    row, col = graph.edges()
    row = row.long()
    col = col.long()

    disp = pos[row] - pos[col]
    disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
    graph.edata["x"] = torch.cat((disp, disp_norm), dim=-1)

    return graph


# Define this function outside of any local scope so it can be pickled
def run_task(params):
    """Wrapper function to unpack arguments for process_run."""
    return process_run(*params)


def process_partition(graph, num_partitions, halo_hops):
    """
    Helper function to partition a single graph and include node and edge features.
    """
    # Perform the partitioning
    partitioned = dgl.metis_partition(
        graph, k=num_partitions, extra_cached_hops=halo_hops, reshuffle=True
    )

    # For each partition, restore node and edge features
    partition_list = []
    for _, subgraph in partitioned.items():
        subgraph.ndata["coordinates"] = graph.ndata["coordinates"][
            subgraph.ndata[dgl.NID]
        ]
        subgraph.ndata["normals"] = graph.ndata["normals"][subgraph.ndata[dgl.NID]]
        subgraph.ndata["area"] = graph.ndata["area"][subgraph.ndata[dgl.NID]]
        subgraph.ndata["pressure"] = graph.ndata["pressure"][subgraph.ndata[dgl.NID]]
        subgraph.ndata["shear_stress"] = graph.ndata["shear_stress"][
            subgraph.ndata[dgl.NID]
        ]
        if "x" in graph.edata:
            subgraph.edata["x"] = graph.edata["x"][subgraph.edata[dgl.EID]]

        partition_list.append(subgraph)

    return partition_list


def process_run(
    run_path, point_list, node_degree, num_partitions, halo_hops, save_point_cloud=False
):
    """Process a single run directory to generate a multi-level graph and apply partitioning."""
    run_id = os.path.basename(run_path).split("_")[-1]

    stl_file = os.path.join(run_path, f"drivaer_{run_id}_single_solid.stl")
    vtp_file = os.path.join(run_path, f"boundary_{run_id}.vtp")

    # Path to save the list of partitions
    partition_file_path = to_absolute_path(f"partitions/graph_partitions_{run_id}.bin")

    if os.path.exists(partition_file_path):
        print(f"Partitions for run {run_id} already exist. Skipping...")
        return

    if not os.path.exists(stl_file) or not os.path.exists(vtp_file):
        print(f"Warning: Missing files for run {run_id}. Skipping...")
        return

    try:
        # Load the STL and VTP files
        obj = Tessellation.from_stl(stl_file, airtight=False)
        surface_mesh = read_vtp(vtp_file)
        surface_mesh = convert_to_triangular_mesh(surface_mesh)
        surface_vertices = fetch_mesh_vertices(surface_mesh)
        surface_mesh = surface_mesh.cell_data_to_point_data()
        node_attributes = surface_mesh.point_data
        pressure_ref = node_attributes["pMeanTrim"]
        shear_stress_ref = node_attributes["wallShearStressMeanTrim"]

        # Sort the list of points in ascending order
        sorted_points = sorted(point_list)

        # Initialize arrays to store all points, normals, and areas
        all_points = np.empty((0, 3))
        all_normals = np.empty((0, 3))
        all_areas = np.empty((0, 1))
        edge_sources = []
        edge_destinations = []

        # Precompute the nearest neighbors for surface vertices
        nbrs_surface = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(
            surface_vertices
        )

        for num_points in sorted_points:
            # Sample the boundary points for the current level
            boundary = obj.sample_boundary(num_points)
            points = np.concatenate(
                [boundary["x"], boundary["y"], boundary["z"]], axis=1
            )
            normals = np.concatenate(
                [boundary["normal_x"], boundary["normal_y"], boundary["normal_z"]],
                axis=1,
            )
            area = boundary["area"]

            # Concatenate new points with the previous ones
            all_points = np.vstack([all_points, points])
            all_normals = np.vstack([all_normals, normals])
            all_areas = np.vstack([all_areas, area])

            # Construct edges for the combined point cloud at this level
            nbrs_points = NearestNeighbors(
                n_neighbors=node_degree + 1, algorithm="ball_tree"
            ).fit(all_points)
            _, indices_within = nbrs_points.kneighbors(all_points)
            src_within = [i for i in range(len(all_points)) for _ in range(node_degree)]
            dst_within = indices_within[:, 1:].flatten()

            # Add the within-level edges
            edge_sources.extend(src_within)
            edge_destinations.extend(dst_within)

        # Now, compute pressure and shear stress for the final combined point cloud
        _, indices = nbrs_surface.kneighbors(all_points)
        indices = indices.flatten()

        pressure = pressure_ref[indices]
        shear_stress = shear_stress_ref[indices]

    except Exception as e:
        print(f"Error processing run {run_id}: {e}. Skipping this run...")
        return

    try:
        # Create the final graph with multi-level edges
        graph = dgl.graph((edge_sources, edge_destinations))
        graph = dgl.remove_self_loop(graph)
        graph = dgl.to_simple(graph)
        graph = dgl.to_bidirected(graph, copy_ndata=True)
        graph = dgl.add_self_loop(graph)

        graph.ndata["coordinates"] = torch.tensor(all_points, dtype=torch.float32)
        graph.ndata["normals"] = torch.tensor(all_normals, dtype=torch.float32)
        graph.ndata["area"] = torch.tensor(all_areas, dtype=torch.float32)
        graph.ndata["pressure"] = torch.tensor(pressure, dtype=torch.float32).unsqueeze(
            -1
        )
        graph.ndata["shear_stress"] = torch.tensor(shear_stress, dtype=torch.float32)
        graph = add_edge_features(graph)

        # Partition the graph
        partitioned_graphs = process_partition(graph, num_partitions, halo_hops)

        # Save the partitions
        save_graphs(partition_file_path, partitioned_graphs)

        if save_point_cloud:
            point_cloud = pv.PolyData(graph.ndata["coordinates"].numpy())
            point_cloud["coordinates"] = graph.ndata["coordinates"].numpy()
            point_cloud["normals"] = graph.ndata["normals"].numpy()
            point_cloud["area"] = graph.ndata["area"].numpy()
            point_cloud["pressure"] = graph.ndata["pressure"].numpy()
            point_cloud["shear_stress"] = graph.ndata["shear_stress"].numpy()
            point_cloud.save(f"point_clouds/point_cloud_{run_id}.vtp")

    except Exception as e:
        print(
            f"Error while constructing graph or saving data for run {run_id}: {e}. Skipping this run..."
        )
        return


def process_all_runs(
    base_path,
    num_points,
    node_degree,
    num_partitions,
    halo_hops,
    num_workers=16,
    save_point_cloud=False,
):
    """Process all runs in the base directory in parallel."""

    run_dirs = [
        os.path.join(base_path, d)
        for d in os.listdir(base_path)
        if d.startswith("run_") and os.path.isdir(os.path.join(base_path, d))
    ]

    tasks = [
        (run_dir, num_points, node_degree, num_partitions, halo_hops, save_point_cloud)
        for run_dir in run_dirs
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for _ in tqdm(
            pool.map(run_task, tasks),
            total=len(tasks),
            desc="Processing Runs",
            unit="run",
        ):
            pass


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    process_all_runs(
        base_path=to_absolute_path(cfg.data_path),
        num_points=cfg.num_nodes,
        node_degree=cfg.node_degree,
        num_partitions=cfg.num_partitions,
        halo_hops=cfg.num_message_passing_layers,
        num_workers=cfg.num_preprocess_workers,
        save_point_cloud=cfg.save_point_clouds,
    )


if __name__ == "__main__":
    main()
