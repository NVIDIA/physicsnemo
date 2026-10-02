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

import hashlib
import os
import re

import numpy as np
import pyvista as pv
import torch
from utils import get_global_features_for_run, load_global_features

VTP_CACHE_VERSION = 1


def _vtp_cache_path(cache_dir: str, vtp_path: str) -> str:
    """Return a collision-resistant cache path for a source VTP."""

    source = os.path.realpath(vtp_path)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    stem = os.path.splitext(os.path.basename(vtp_path))[0]
    return os.path.join(cache_dir, f"{stem}-{digest}.pt")


def _load_vtp_cache(cache_path: str, vtp_path: str, include_contact_topology=False):
    """Load a cache entry when it still describes the exact source file."""

    if not os.path.isfile(cache_path):
        return None
    source_stat = os.stat(vtp_path)
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    if (
        payload.get("version") != VTP_CACHE_VERSION
        or payload.get("source") != os.path.realpath(vtp_path)
        or payload.get("source_size") != source_stat.st_size
        or payload.get("source_mtime_ns") != source_stat.st_mtime_ns
        or (include_contact_topology and "mesh_cells" not in payload)
    ):
        return None
    result = (
        payload["src"].numpy(),
        payload["dst"].numpy(),
        payload["coords"].numpy(),
        {key: value.numpy() for key, value in payload["point_data"].items()},
    )
    return (
        result + (payload["mesh_cells"].numpy(),)
        if include_contact_topology
        else result
    )


def _save_vtp_cache(
    cache_path: str,
    vtp_path: str,
    src: np.ndarray,
    dst: np.ndarray,
    coords: np.ndarray,
    point_data: dict[str, np.ndarray],
    mesh_cells: np.ndarray | None = None,
) -> None:
    """Atomically publish a tensor-only cache entry safe for weights-only load."""

    source_stat = os.stat(vtp_path)
    payload = {
        "version": VTP_CACHE_VERSION,
        "source": os.path.realpath(vtp_path),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        # The datapipe immediately casts coordinates to float32. Caching that
        # representation preserves its exact inputs while halving cache size.
        "src": torch.from_numpy(np.ascontiguousarray(src, dtype=np.int64)),
        "dst": torch.from_numpy(np.ascontiguousarray(dst, dtype=np.int64)),
        "coords": torch.from_numpy(np.ascontiguousarray(coords, dtype=np.float32)),
        "point_data": {
            key: torch.from_numpy(np.ascontiguousarray(value))
            for key, value in point_data.items()
        },
    }
    if mesh_cells is not None:
        payload["mesh_cells"] = torch.from_numpy(
            np.ascontiguousarray(mesh_cells, dtype=np.int64)
        )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    temporary_path = f"{cache_path}.tmp-{os.getpid()}"
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, cache_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


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
    """Load positions over time, connectivity, and other point data from a single VTP file.

    Expects displacement fields in point_data named like:
      - displacement_t0.000, displacement_t0.005, ..., displacement_t0.100
    Returns:
        pos_raw: (timesteps, num_nodes, 3) absolute positions (coords + displacement_t)
        mesh_connectivity: list[list[int]]
        point_data_dict: dict of other point data arrays (e.g., thickness)
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

    # Extract all other point data fields (not displacement fields)
    point_data_dict = {}
    for name in poly.point_data.keys():
        if not name.startswith("displacement_"):
            point_data_dict[name] = np.asarray(poly.point_data[name])

    # Extract cell data and convert to point data
    if poly.cell_data:
        converted = poly.cell_data_to_point_data(pass_cell_data=True)
        cell_point_names = [
            name
            for name in converted.point_data.keys()
            if name.startswith("cell_effective_plastic_strain_")
            or name.startswith("cell_stress_vm_")
        ]
        if cell_point_names:

            def natural_key(name):
                return [
                    int(s) if s.isdigit() else s.lower()
                    for s in re.findall(r"\d+|\D+", name)
                ]

            cell_point_names = sorted(cell_point_names, key=natural_key)
            for name in cell_point_names:
                arr = np.asarray(converted.point_data[name])
                # Drop the 'cell_' prefix to reflect point semantics
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
    output_dir, pos_raw, filtered_mesh_connectivity, write_vtp=False, logger=None
):
    """Write VTP files for each timestep and collect mesh/point data."""
    # Training and validation only need the already assembled position tensor.
    # Constructing a 385k-node PyVista mesh at every timestep is pure overhead
    # when no VTP is requested (26 large mesh constructions per simulation for
    # the GM crash data).
    if not write_vtp:
        return np.asarray(pos_raw)

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


def process_vtp_data(
    data_dir,
    num_samples=2,
    write_vtp=False,
    global_features_filepath: str | None = None,
    cache_dir: str | None = None,
    include_contact_topology: bool = False,
    logger=None,
):
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
    global_features_all = []

    if not vtp_files:
        if logger:
            logger.error(f"No .vtp files found in: {base_data_dir}")
        exit(1)

    # Load global features
    if global_features_filepath is not None:
        all_global_features = load_global_features(global_features_filepath)

    for vtp_path in vtp_files:
        cached = None
        cache_path = None
        if cache_dir is not None and not write_vtp:
            cache_path = _vtp_cache_path(cache_dir, vtp_path)
            cached = _load_vtp_cache(cache_path, vtp_path, include_contact_topology)
        if logger:
            action = "Loading cached" if cached is not None else "Processing"
            logger.info(f"{action} {vtp_path}...")
        output_dir = f"./output_{os.path.splitext(os.path.basename(vtp_path))[0]}"
        if write_vtp:
            os.makedirs(output_dir, exist_ok=True)

        # Get global features for this run
        run_id = os.path.splitext(os.path.basename(vtp_path))[0]
        if global_features_filepath is not None:
            global_features = get_global_features_for_run(
                all_global_features,
                run_id,
            )
        else:
            global_features = {}

        if cached is None:
            pos_raw, mesh_connectivity, point_data_dict = load_vtp_file(vtp_path)
            mesh_cells = (
                np.asarray(
                    [
                        value
                        for cell in mesh_connectivity
                        for value in (len(cell), *cell)
                    ],
                    dtype=np.int64,
                )
                if include_contact_topology
                else None
            )

            # Use unfiltered data
            filtered_pos_raw = pos_raw
            filtered_mesh_connectivity = mesh_connectivity

            # Build edges and sanity-check ranges
            edges = build_edges_from_mesh_connectivity(filtered_mesh_connectivity)
            edge_arr = np.array(list(edges), dtype=np.int64)
            assert edge_arr.min() >= 0 and edge_arr.max() < filtered_pos_raw.shape[1]

            src, dst = edge_arr.T
            mesh_pos_all = collect_mesh_pos(
                output_dir,
                filtered_pos_raw,
                filtered_mesh_connectivity,
                write_vtp=write_vtp,
                logger=logger,
            )
            if cache_path is not None:
                _save_vtp_cache(
                    cache_path,
                    vtp_path,
                    src,
                    dst,
                    mesh_pos_all,
                    point_data_dict,
                    mesh_cells=mesh_cells,
                )
        else:
            src, dst, mesh_pos_all, point_data_dict = cached[:4]
            mesh_cells = cached[4] if include_contact_topology else None
        srcs.append(src)
        dsts.append(dst)

        # Create record with coords and all other point data fields
        record = {
            "coords": mesh_pos_all,
            "point_data": point_data_dict,
        }
        if mesh_cells is not None:
            record["mesh_cells"] = mesh_cells

        point_data_all.append(record)
        global_features_all.append(global_features)

        processed_runs += 1
        if processed_runs >= num_samples:
            break

    return srcs, dsts, point_data_all, global_features_all


class Reader:
    """
    Reader for VTP files.
    """

    def __init__(self, cache_dir: str | None = None, include_contact_topology=False):
        self.cache_dir = cache_dir
        self.include_contact_topology = bool(include_contact_topology)

    def __call__(
        self,
        data_dir: str,
        num_samples: int,
        split: str | None = None,
        global_features_filepath: str | None = None,
        logger=None,
        **kwargs,
    ):
        write_vtp = False if split in ("train", "validation") else True
        return process_vtp_data(
            data_dir=data_dir,
            num_samples=num_samples,
            write_vtp=write_vtp,
            global_features_filepath=global_features_filepath,
            cache_dir=self.cache_dir,
            include_contact_topology=self.include_contact_topology,
            logger=logger,
        )
