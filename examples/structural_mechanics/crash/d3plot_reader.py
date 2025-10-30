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

import numpy as np
import os
import pyvista as pv

from lasso.dyna import D3plot, ArrayType
from typing import Optional


def find_run_folders(base_data_dir: str) -> list[str]:
    """
    Find run directories containing LS-DYNA d3plot files.

    Args:
        base_data_dir: Path to base directory containing multiple simulation runs.

    Returns:
        List of run folder names containing `d3plot` files.
    """
    run_dirs = []
    if os.path.isdir(base_data_dir):
        for item in os.listdir(base_data_dir):
            item_path = os.path.join(base_data_dir, item)
            if os.path.isdir(item_path) and os.path.exists(
                os.path.join(item_path, "d3plot")
            ):
                run_dirs.append(item)
    return run_dirs


def parse_k_file(k_file_path: str) -> dict[int, float]:
    """
    Parse LS-DYNA keyword (.k) file to extract part thickness values.

    Args:
        k_file_path: Path to `.k` file.

    Returns:
        Dictionary mapping part ID -> thickness.
    """
    part_to_section: dict[int, int] = {}
    section_thickness: dict[int, float] = {}

    with open(k_file_path, "r") as f:
        lines = [
            line.strip() for line in f if line.strip() and not line.startswith("$")
        ]

    i = 0
    while i < len(lines):
        line = lines[i]
        if "*PART" in line.upper():
            # After *PART:
            # i+1 = part name (skip)
            # i+2 = part id, section id, material id
            if i + 2 < len(lines):
                tokens = lines[i + 2].split()
                if len(tokens) >= 2:
                    try:
                        part_id = int(tokens[0])
                        section_id = int(tokens[1])
                        part_to_section[part_id] = section_id
                    except ValueError:
                        pass
            i += 3
        elif "*SECTION_SHELL" in line.upper():
            # Multiple sections can be defined under one *SECTION_SHELL keyword
            # Each section has two lines: header line and thickness line
            i += 1  # Skip the *SECTION_SHELL line
            while i < len(lines) and not lines[i].startswith("*"):
                # Check if this line looks like a section header (starts with a number)
                if lines[i] and lines[i][0].isdigit():
                    header_tokens = lines[i].split()
                    section_id = None
                    if header_tokens:
                        try:
                            section_id = int(header_tokens[0])
                        except ValueError:
                            pass

                    thickness_line = lines[i + 1] if i + 1 < len(lines) else ""
                    thickness_values = []
                    for t in thickness_line.split():
                        try:
                            thickness_values.append(float(t))
                        except ValueError:
                            continue

                    # Average ignoring zeros
                    if thickness_values:
                        non_zero = [t for t in thickness_values if t > 0.0]
                        thickness = (
                            sum(non_zero) / len(non_zero)
                            if non_zero
                            else sum(thickness_values) / len(thickness_values)
                        )
                    else:
                        thickness = 0.0

                    if section_id is not None:
                        section_thickness[section_id] = thickness

                    i += 2
                else:
                    i += 1
        else:
            i += 1

    return {
        pid: section_thickness.get(sid, 0.0) for pid, sid in part_to_section.items()
    }


def load_d3plot_data(data_path: str):
    """
    Load node coordinates, displacements, connectivity, and part IDs from d3plot.

    Args:
        data_path: Path to `d3plot` file.

    Returns:
        coords: (num_nodes, 3)
        pos_raw: (timesteps, num_nodes, 3)
        mesh_connectivity: element connectivity
        part_ids: array of part IDs for elements
        actual_part_ids: optional mapping of part indices to IDs
    """
    dp = D3plot(data_path)
    coords = dp.arrays[ArrayType.node_coordinates]
    pos_raw = dp.arrays[ArrayType.node_displacement]
    mesh_connectivity = dp.arrays[ArrayType.element_shell_node_indexes]
    part_ids = dp.arrays[ArrayType.element_shell_part_indexes]
    actual_part_ids = dp.arrays.get(ArrayType.part_ids, None)
    return coords, pos_raw, mesh_connectivity, part_ids, actual_part_ids


def compute_node_type(pos_raw: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """
    Identify structural vs wall nodes based on displacement variation.

    Args:
        pos_raw: (timesteps, num_nodes, 3) raw displacement trajectories
        threshold: max displacement below which a node is considered "wall"

    Returns:
        node_type: (num_nodes,) uint8 array where 1=wall, 0=structure
    """
    variation = np.max(np.abs(pos_raw - pos_raw[0:1, :, :]), axis=0)
    variation = np.max(variation, axis=1)
    is_wall = variation < threshold
    return np.where(is_wall, 1, 0).astype(np.uint8)


def build_edges_from_mesh_connectivity(mesh_connectivity) -> set:
    """
    Build unique edges from mesh connectivity.

    Args:
        mesh_connectivity: list of elements (list[int])

    Returns:
        Set of unique edges (i,j)
    """
    edges = set()
    for cell in mesh_connectivity:
        n = len(cell)
        for idx in range(n):
            edges.add(tuple(sorted((cell[idx], cell[(idx + 1) % n]))))
    return edges


def compute_node_thickness(
    mesh_connectivity,
    part_ids,
    part_thickness_map: dict[int, float],
    actual_part_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute average node thickness from connected elements.

    Args:
        mesh_connectivity: Element connectivity array
        part_ids: Part IDs for each element
        part_thickness_map: Mapping part ID -> thickness
        actual_part_ids: Optional explicit part IDs array

    Returns:
        node_thickness: (num_nodes,) average thickness per node
    """
    # Map part index -> actual part ID
    if actual_part_ids is not None:
        part_index_to_id = {i: pid for i, pid in enumerate(actual_part_ids) if i > 0}
    else:
        sorted_part_ids = sorted(part_thickness_map.keys())
        part_index_to_id = {i + 1: pid for i, pid in enumerate(sorted_part_ids)}

    element_thickness = np.zeros(len(part_ids))
    for i, part_index in enumerate(part_ids):
        actual_part_id = part_index_to_id.get(part_index)
        if actual_part_id is not None:
            element_thickness[i] = part_thickness_map.get(actual_part_id, 0.0)

    max_node_idx = max(max(cell) for cell in mesh_connectivity)
    node_thickness = np.zeros(max_node_idx + 1)
    node_count = np.zeros(max_node_idx + 1)

    for i, element in enumerate(mesh_connectivity):
        thickness = element_thickness[i]
        for node_idx in element:
            node_thickness[node_idx] += thickness
            node_count[node_idx] += 1

    nonzero = node_count > 0
    node_thickness[nonzero] /= node_count[nonzero]
    return node_thickness


def collect_mesh_pos(
    output_dir: str,
    pos_raw: np.ndarray,
    filtered_mesh_connectivity,
    node_thickness: np.ndarray,
    write_vtp: bool = False,
    logger=None,
) -> np.ndarray:
    """
    Collect mesh positions per timestep, optionally writing VTP files.

    Args:
        output_dir: directory to write VTPs
        pos_raw: (timesteps, num_nodes, 3)
        filtered_mesh_connectivity: element connectivity list
        node_thickness: per-node thickness array
        write_vtp: whether to save VTPs
        logger: optional logger

    Returns:
        mesh_pos_all: (timesteps, num_nodes, 3)
    """
    n_timesteps = pos_raw.shape[0]
    mesh_pos_all = []

    for t in range(n_timesteps):
        pos = pos_raw[t, :, :]

        faces = []
        for cell in filtered_mesh_connectivity:
            if len(cell) == 3:  # Triangle
                faces.extend([3, *cell])
            elif len(cell) == 4:  # Quad shell
                faces.extend([4, *cell])
            else:  # Higher order elements not supported
                continue

        faces = np.array(faces)
        mesh = pv.PolyData(pos, faces)

        if len(node_thickness) >= len(pos):
            mesh.point_data["thickness"] = node_thickness[: len(pos)]

        if write_vtp:
            filename = os.path.join(output_dir, f"frame_{t:03d}.vtp")
            mesh.save(filename)
            if logger:
                logger.info(f"Saved VTP: {filename}")

        mesh_pos_all.append(pos)

    return np.stack(mesh_pos_all)


def process_d3plot_data(
    data_dir: str,
    num_samples: int,
    wall_node_disp_threshold: float = 1.0,
    write_vtp: bool = False,
    logger=None,
):
    """
    Main entry: Preprocess LS-DYNA crash simulation data for a directory.

    Args:
        data_dir: path to folder containing run subdirectories
        num_samples: max number of runs to process
        wall_node_disp_threshold: displacement threshold for filtering wall nodes
        write_vtp: whether to write per-timestep VTP files
        logger: optional logger

    Returns:
        srcs, dsts: edge connectivity arrays
        point_data_all: list of dicts {"mesh_pos": ..., "thickness": ...}
    """
    run_folders = find_run_folders(data_dir)
    if not run_folders:
        raise ValueError(f"No run folders found in {data_dir}")

    if logger is None:
        import logging

        logger = logging.getLogger(__name__)

    srcs, dsts, point_data_all = [], [], []

    for run_i, run_folder in enumerate(sorted(run_folders)):
        if run_i >= num_samples:
            break

        logger.info(f"Processing run: {run_folder}")
        data_path = os.path.join(data_dir, run_folder, "d3plot")
        output_dir = f"./output_{run_folder}"
        os.makedirs(output_dir, exist_ok=True)

        coords, pos_raw, mesh_connectivity, part_ids, actual_part_ids = (
            load_d3plot_data(data_path)
        )

        flat = np.fromiter(
            (n for cell in mesh_connectivity for n in cell), dtype=np.int64
        )
        assert flat.min() >= 0 and flat.max() < coords.shape[0], (
            f"Connectivity out of bounds: min={flat.min()}, max={flat.max()}, num_nodes={coords.shape[0]}"
        )

        # Parse thickness from .k file if present
        node_thickness = np.zeros(len(coords))
        k_file_path = None
        for f in os.listdir(os.path.join(data_dir, run_folder)):
            if f.endswith(".k"):
                k_file_path = os.path.join(data_dir, run_folder, f)
                break

        if k_file_path and os.path.exists(k_file_path):
            part_thickness_map = parse_k_file(k_file_path)
            node_thickness = compute_node_thickness(
                mesh_connectivity, part_ids, part_thickness_map, actual_part_ids
            )
        else:
            logger.warning(f"No .k file in {run_folder}, defaulting thickness=0")

        # Identify structural nodes
        node_type = compute_node_type(pos_raw, threshold=wall_node_disp_threshold)
        keep_nodes = sorted(np.where(node_type == 0)[0])
        node_map = {old: new for new, old in enumerate(keep_nodes)}

        # Filter arrays
        filtered_pos_raw = pos_raw[:, keep_nodes, :]
        filtered_thickness = node_thickness[keep_nodes]

        # Remap connectivity
        filtered_mesh_connectivity = []
        for cell in mesh_connectivity:
            mapped = [node_map[n] for n in cell if n in node_map]
            if len(mapped) >= 3:
                filtered_mesh_connectivity.append(mapped)

        used = np.unique([n for cell in filtered_mesh_connectivity for n in cell])
        if used.size == 0:
            raise ValueError("No cells left after filtering; lower threshold?")

        # Compact indexing if needed
        num_kept = filtered_pos_raw.shape[1]
        if used.min() != 0 or used.max() != num_kept - 1 or used.size != num_kept:
            remap2 = {old: new for new, old in enumerate(used.tolist())}
            filtered_pos_raw = filtered_pos_raw[:, used, :]
            filtered_thickness = filtered_thickness[used]
            filtered_mesh_connectivity = [
                [remap2[n] for n in cell] for cell in filtered_mesh_connectivity
            ]
            num_kept = filtered_pos_raw.shape[1]

        # Sanity checks
        used = np.unique([n for cell in filtered_mesh_connectivity for n in cell])
        assert used.min() == 0 and used.max() == num_kept - 1

        edges = build_edges_from_mesh_connectivity(filtered_mesh_connectivity)
        edge_arr = np.array(list(edges), dtype=np.int64)
        assert edge_arr.min() >= 0 and edge_arr.max() < num_kept

        src, dst = np.array(list(edges)).T
        srcs.append(src)
        dsts.append(dst)

        # Collect mesh pos and store
        mesh_pos_all = collect_mesh_pos(
            output_dir,
            filtered_pos_raw,
            filtered_mesh_connectivity,
            filtered_thickness,
            write_vtp,
            logger,
        )
        point_data_all.append({"coords": mesh_pos_all, "thickness": filtered_thickness})

    return srcs, dsts, point_data_all


class Reader:
    """
    Reader for LS-DYNA d3plot files.

    Args:
        wall_node_disp_threshold: threshold for filtering wall nodes
    """

    def __init__(self, wall_node_disp_threshold: float = 1.0):
        self.wall_node_disp_threshold = wall_node_disp_threshold

    def __call__(
        self,
        data_dir: str,
        num_samples: int,
        split: str,
        logger=None,
    ):
        write_vtp = False if split == "train" else True
        return process_d3plot_data(
            data_dir=data_dir,
            num_samples=num_samples,
            wall_node_disp_threshold=self.wall_node_disp_threshold,
            write_vtp=write_vtp,
            logger=logger,
        )
