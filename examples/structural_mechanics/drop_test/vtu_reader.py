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

import os
import re
import numpy as np
import pyvista as pv

from utils import load_global_features, get_global_features_for_run


def find_run_folders(base_data_dir):
    """Return a list of absolute VTU file paths; each file is a separate sample."""
    if not os.path.isdir(base_data_dir):
        return []
    vtus = [
        os.path.join(base_data_dir, f)
        for f in os.listdir(base_data_dir)
        if f.lower().endswith(".vtu")
    ]

    def natural_key(name):
        return [
            int(s) if s.isdigit() else s.lower()
            for s in re.findall(r"\d+|\D+", os.path.basename(name))
        ]

    return sorted(vtus, key=natural_key)


def extract_mesh_connectivity_from_unstructured_grid(grid: pv.UnstructuredGrid):
    """Extract mesh connectivity (list of cells with node indices) from UnstructuredGrid (tet/hex)."""
    cells = grid.cells
    connectivity = []
    i = 0
    n = cells.size
    while i < n:
        fsz = int(cells[i])
        ids = cells[i + 1 : i + 1 + fsz].tolist()
        if len(ids) >= 3:
            connectivity.append(ids)
        i += 1 + fsz
    return connectivity


def load_vtu_file(vtu_path):
    """Load positions over time, connectivity, and other point data from a single VTU file.

    Expects UnstructuredGrid (tet/hex) with displacement fields in point_data named like:
      - displacement_t0.000, displacement_t0.005, ..., displacement_t0.100
    Also loads velocity_, acceleration_, temperature_, residual_forces_ if present.
    Returns:
        pos_raw: (timesteps, num_nodes, 3) absolute positions (coords + displacement_t)
        mesh_connectivity: list[list[int]]
        point_data_dict: dict of other point data arrays (e.g., thickness, velocity, etc.)
    """
    mesh = pv.read(vtu_path)
    if not isinstance(mesh, pv.UnstructuredGrid):
        raise ValueError(
            f"Drop test VTU expects UnstructuredGrid (tet/hex), got {type(mesh).__name__}"
        )

    coords = np.array(mesh.points, dtype=np.float64)

    # Collect displacement vector arrays (3 components) and sort naturally
    # Supports: displacement_t0.000, displacement_t0000, displacement_t0001, etc.
    disp_names = [
        name
        for name in mesh.point_data.keys()
        if re.match(r"displacement_t0\.[0-9]{3,}$", name)
    ]
    if not disp_names:
        disp_names = [
            name
            for name in mesh.point_data.keys()
            if re.match(r"displacement_t[0-9]+$", name)
        ]
    if not disp_names:
        disp_names = [
            name for name in mesh.point_data.keys() if name.startswith("displacement_t")
        ]
    if not disp_names:
        raise ValueError(f"No displacement fields found in {vtu_path}")

    def natural_key(name):
        return [
            int(s) if s.isdigit() else s.lower() for s in re.findall(r"\d+|\D+", name)
        ]

    disp_names = sorted(disp_names, key=natural_key)

    # Curator convention: displacement_t0 is always the zero vector (it's written as
    # `filtered_pos_raw[0] - reference_coords`, which equals zero by construction).
    # We add disp unconditionally — at t=0 this resolves to coords + 0 = coords, and
    # at t>0 it gives the absolute position. This handles malformed inputs where the
    # first frame is non-zero the same way as well-formed inputs.
    pos_list = []
    for name in disp_names:
        disp = np.asarray(mesh.point_data[name])
        if disp.ndim != 2 or disp.shape[1] != 3:
            raise ValueError(
                f"Point-data array '{name}' must be a 3-component vector (got shape {disp.shape})."
            )
        pos_list.append(coords + disp)

    pos_raw = np.stack(pos_list, axis=0)
    mesh_connectivity = extract_mesh_connectivity_from_unstructured_grid(mesh)

    # Extract all other point data fields (not displacement, velocity, acceleration, etc. per-timestep)
    point_data_dict = {}
    for name in mesh.point_data.keys():
        if not name.startswith("displacement_"):
            point_data_dict[name] = np.asarray(mesh.point_data[name])

    # Extract cell data and convert to point data
    if mesh.cell_data:
        converted = mesh.cell_data_to_point_data(pass_cell_data=True)
        cell_point_names = [
            name
            for name in converted.point_data.keys()
            if name.startswith("cell_effective_plastic_strain_")
            or name.startswith("cell_stress_vm_")
        ]
        if cell_point_names:
            cell_point_names = sorted(cell_point_names, key=natural_key)
            for name in cell_point_names:
                arr = np.asarray(converted.point_data[name])
                point_name = name.replace("cell_", "", 1)
                point_data_dict[point_name] = arr

    return pos_raw, mesh_connectivity, point_data_dict


def build_edges_from_mesh_connectivity(mesh_connectivity):
    """Build unique edges from mesh connectivity (cells of any size)."""
    edges = set()
    for cell in mesh_connectivity:
        n = len(cell)
        for idx in range(n):
            edge = tuple(sorted((cell[idx], cell[(idx + 1) % n])))
            edges.add(edge)
    return edges


def collect_mesh_pos(
    output_dir, pos_raw, filtered_mesh_connectivity, write_vtu=False, logger=None
):
    """Write VTU files for each timestep and collect mesh/point data (UnstructuredGrid tet/hex)."""
    n_timesteps = pos_raw.shape[0]
    mesh_pos_all = []
    pos0 = pos_raw[0]
    for t in range(n_timesteps):
        pos = pos_raw[t, :, :]

        cells = []
        cell_types = []
        for cell in filtered_mesh_connectivity:
            n = len(cell)
            if n == 4:
                cells.extend([4, *cell])
                cell_types.append(pv.CellType.TETRA)
            elif n == 8:
                cells.extend([8, *cell])
                cell_types.append(pv.CellType.HEXAHEDRON)
            else:
                continue

        cells = np.array(cells)
        cell_types = np.array(cell_types)
        mesh = pv.UnstructuredGrid(cells, cell_types, pos)

        disp = pos - pos0
        mesh.point_data["displacement"] = disp

        if write_vtu:
            filename = os.path.join(output_dir, f"frame_{t:03d}.vtu")
            mesh.save(filename)
            if logger:
                logger.info(f"Saved: {filename}")

        mesh_pos_all.append(pos)
    return np.stack(mesh_pos_all)


def process_vtu_data(
    data_dir,
    num_samples=2,
    write_vtu=False,
    global_features_filepath: str | None = None,
    logger=None,
):
    """
    Preprocesses VTU drop test (solid) simulation data in a given directory.
    Each .vtu file is treated as one sample. For each sample, computes edges from connectivity,
    keeps all nodes, and optionally writes VTU files for each timestep.
    Returns lists of source/destination node indices and point data for all samples.
    """
    processed_runs = 0
    base_data_dir = data_dir
    vtu_files = find_run_folders(base_data_dir)
    srcs, dsts = [], []
    point_data_all = []
    global_features_all = []

    if not vtu_files:
        raise FileNotFoundError(f"No .vtu files found in: {base_data_dir}")

    if global_features_filepath is not None:
        all_global_features = load_global_features(global_features_filepath)

    for vtu_path in vtu_files:
        if logger:
            logger.info(f"Processing {vtu_path}...")
        output_dir = f"./output_{os.path.splitext(os.path.basename(vtu_path))[0]}"
        os.makedirs(output_dir, exist_ok=True)

        run_id = os.path.splitext(os.path.basename(vtu_path))[0]
        if global_features_filepath is not None:
            global_features = get_global_features_for_run(
                all_global_features,
                run_id,
            )
        else:
            global_features = {}

        pos_raw, mesh_connectivity, point_data_dict = load_vtu_file(vtu_path)

        filtered_pos_raw = pos_raw
        filtered_mesh_connectivity = mesh_connectivity

        edges = build_edges_from_mesh_connectivity(filtered_mesh_connectivity)
        edge_arr = np.array(list(edges), dtype=np.int64)
        assert edge_arr.min() >= 0 and edge_arr.max() < filtered_pos_raw.shape[1]

        src, dst = np.array(list(edges)).T
        srcs.append(src)
        dsts.append(dst)

        mesh_pos_all = collect_mesh_pos(
            output_dir,
            filtered_pos_raw,
            filtered_mesh_connectivity,
            write_vtu=write_vtu,
            logger=logger,
        )

        record = {
            "coords": mesh_pos_all,
            "point_data": point_data_dict,
        }

        point_data_all.append(record)
        global_features_all.append(global_features)

        processed_runs += 1
        if processed_runs >= num_samples:
            break

    return srcs, dsts, point_data_all, global_features_all


class Reader:
    """
    Reader for VTU files (drop test, solid elements).
    """

    def __init__(self):
        pass

    def __call__(
        self,
        data_dir: str,
        num_samples: int,
        split: str | None = None,
        global_features_filepath: str | None = None,
        logger=None,
        **kwargs,
    ):
        write_vtu = False if split in ("train", "validation") else True
        return process_vtu_data(
            data_dir=data_dir,
            num_samples=num_samples,
            write_vtu=write_vtu,
            global_features_filepath=global_features_filepath,
            logger=logger,
        )
