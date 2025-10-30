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

import os
import re
import numpy as np
import pyvista as pv


def find_run_folders(base_data_dir):
    """Return a list of absolute VTP file paths; each file is a separate sample."""
    if not os.path.isdir(base_data_dir):
        return []
    vtps = [
        os.path.join(base_data_dir, f)
        for f in os.listdir(base_data_dir)
        if f.lower().endswith(".vtp")
    ]

    def natural_key(name):
        return [
            int(s) if s.isdigit() else s.lower()
            for s in re.findall(r"\d+|\D+", os.path.basename(name))
        ]

    return sorted(vtps, key=natural_key)


def extract_mesh_connectivity_from_polydata(poly: pv.PolyData):
    """Extract mesh connectivity (list of cells with node indices) from a PolyData."""
    faces = poly.faces
    connectivity = []
    i = 0
    n = faces.size
    while i < n:
        fsz = int(faces[i])
        ids = faces[i + 1 : i + 1 + fsz].tolist()
        if len(ids) >= 3:
            connectivity.append(ids)
        i += 1 + fsz
    return connectivity


def load_vtp_file(vtp_path):
    """Load positions over time and connectivity from a single VTP file.

    Expects displacement fields in point_data named like:
      - displacement_t0.000, displacement_t0.005, ..., displacement_t0.100
    Returns:
        pos_raw: (timesteps, num_nodes, 3) absolute positions (coords + displacement_t)
        mesh_connectivity: list[list[int]]
    """
    poly = pv.read(vtp_path)
    if not isinstance(poly, pv.PolyData):
        poly = poly.extract_surface().cast_to_polydata()

    coords = np.array(poly.points, dtype=np.float64)

    # Collect displacement vector arrays (3 components) and sort naturally
    disp_names = [
        name
        for name in poly.point_data.keys()
        if re.match(r"displacement_t0\.[0-9]{3}$", name)
    ]
    if not disp_names:
        disp_names = [
            name for name in poly.point_data.keys() if name.startswith("displacement_t")
        ]
    if not disp_names:
        raise ValueError(f"No displacement fields found in {vtp_path}")

    def natural_key(name):
        return [
            int(s) if s.isdigit() else s.lower() for s in re.findall(r"\d+|\D+", name)
        ]

    disp_names = sorted(disp_names, key=natural_key)

    pos_list = []
    for idx, name in enumerate(disp_names):
        disp = np.asarray(poly.point_data[name])
        if disp.ndim != 2 or disp.shape[1] != 3:
            raise ValueError(
                f"Point-data array '{name}' must be a 3-component vector (got shape {disp.shape})."
            )
        # Force zero displacement at t0: pos_raw[0] = coords
        if idx == 0:
            pos_list.append(coords)
        else:
            pos_list.append(coords + disp)

    pos_raw = np.stack(pos_list, axis=0)
    mesh_connectivity = extract_mesh_connectivity_from_polydata(poly)
    return pos_raw, mesh_connectivity


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
    output_dir, pos_raw, filtered_mesh_connectivity, write_vtp=False, logger=None
):
    """Write VTP files for each timestep and collect mesh/point data."""
    n_timesteps = pos_raw.shape[0]
    mesh_pos_all = []
    pos0 = pos_raw[0]  # reference for displacement
    for t in range(n_timesteps):
        pos = pos_raw[t, :, :]

        faces = []
        for cell in filtered_mesh_connectivity:
            if len(cell) == 3:
                faces.extend([3, *cell])
            elif len(cell) == 4:
                faces.extend([4, *cell])
            elif len(cell) > 4:
                continue

        faces = np.array(faces)
        mesh = pv.PolyData(pos, faces)

        # Add displacement vector relative to t0
        disp = pos - pos0
        mesh.point_data["displacement"] = disp

        if write_vtp:
            filename = os.path.join(output_dir, f"frame_{t:03d}.vtp")
            mesh.save(filename)
            if write_vtp and logger:
                logger.info(f"Saved: {filename}")

        mesh_pos_all.append(pos)
    return np.stack(mesh_pos_all)


def process_vtp_data(data_dir, num_samples=2, write_vtp=False, logger=None):
    """
    Preprocesses VTP crash simulation data in a given directory.
    Each .vtp file is treated as one sample. For each sample, computes edges from connectivity,
    keeps all nodes, and optionally writes VTP files for each timestep.
    Returns lists of source/destination node indices and point data for all samples.
    """
    processed_runs = 0
    base_data_dir = data_dir
    vtp_files = find_run_folders(base_data_dir)
    srcs, dsts = [], []
    point_data_all = []

    if not vtp_files:
        if logger:
            logger.error("No .vtp files found in:", base_data_dir)
        exit(1)

    for vtp_path in vtp_files:
        if logger:
            logger.info(f"Processing {vtp_path}...")
        output_dir = f"./output_{os.path.splitext(os.path.basename(vtp_path))[0]}"
        os.makedirs(output_dir, exist_ok=True)

        pos_raw, mesh_connectivity = load_vtp_file(vtp_path)

        # Use unfiltered data
        filtered_pos_raw = pos_raw
        filtered_mesh_connectivity = mesh_connectivity

        # Build edges and sanity-check ranges
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
            write_vtp=write_vtp,
            logger=logger,
        )
        point_data_all.append({"coords": mesh_pos_all})

        processed_runs += 1
        if processed_runs >= num_samples:
            break

    return srcs, dsts, point_data_all


class Reader:
    """
    Reader for VTP files.
    """

    def __init__(self):
        pass

    def __call__(
        self,
        data_dir: str,
        num_samples: int,
        split: str | None = None,
        logger=None,
        **kwargs,
    ):
        write_vtp = False if split == "train" else True
        return process_vtp_data(
            data_dir=data_dir,
            num_samples=num_samples,
            write_vtp=write_vtp,
            logger=logger,
        )
