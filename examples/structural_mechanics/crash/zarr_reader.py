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
import zarr


def find_zarr_stores(base_data_dir: str) -> list[str]:
    """
    Find all Zarr stores (directories ending with .zarr) in the base directory.

    Args:
        base_data_dir: Path to directory containing Zarr stores.

    Returns:
        List of Zarr store paths sorted naturally.
    """
    if not os.path.isdir(base_data_dir):
        return []

    zarr_stores = [
        os.path.join(base_data_dir, f)
        for f in os.listdir(base_data_dir)
        if f.endswith(".zarr") and os.path.isdir(os.path.join(base_data_dir, f))
    ]

    def natural_key(name):
        """Natural sort key to handle numeric sorting."""
        return [
            int(s) if s.isdigit() else s.lower()
            for s in re.findall(r"\d+|\D+", os.path.basename(name))
        ]

    return sorted(zarr_stores, key=natural_key)


def load_zarr_store(zarr_path: str):
    """
    Load mesh positions, edges, and all point data fields from a Zarr store.

    Args:
        zarr_path: Path to the Zarr store directory.

    Returns:
        mesh_pos: (timesteps, num_nodes, 3) temporal positions
        edges: (num_edges, 2) edge connectivity
        point_data_dict: Dictionary of all point data fields (e.g., thickness, etc.)
    """
    store = zarr.open(zarr_path, mode="r")

    # Read mesh positions (temporal coordinates)
    if "mesh_pos" not in store:
        raise KeyError(f"'mesh_pos' not found in Zarr store {zarr_path}")
    mesh_pos = np.array(store["mesh_pos"][:], dtype=np.float64)

    # Read edges
    if "edges" not in store:
        raise KeyError(f"'edges' not found in Zarr store {zarr_path}")
    edges = np.array(store["edges"][:], dtype=np.int64)

    # Extract all other datasets as point data (excluding mesh-level data)
    # Skip: mesh_pos, edges, mesh_connectivity_* (these are not per-node features)
    point_data_dict = {}
    for name in store.keys():
        if name in ("mesh_pos", "edges"):
            continue
        if name.startswith("mesh_connectivity_"):
            continue
        # Read as point data feature
        point_data_dict[name] = np.array(store[name][:], dtype=np.float32)

    return mesh_pos, edges, point_data_dict


def process_zarr_data(
    data_dir: str,
    num_samples: int,
    logger=None,
):
    """
    Process Zarr crash simulation data from a given directory.

    Each .zarr store is treated as one sample. Reads mesh positions, edges,
    and all available point data fields (e.g., thickness, etc.) from the Zarr stores.

    Args:
        data_dir: Directory containing .zarr stores
        num_samples: Maximum number of samples to process
        logger: Optional logger for logging progress

    Returns:
        srcs: List of source node indices for edges (one array per sample)
        dsts: List of destination node indices for edges (one array per sample)
        point_data_all: List of dicts with 'coords' and all point data fields
    """
    zarr_stores = find_zarr_stores(data_dir)

    if not zarr_stores:
        if logger:
            logger.error(f"No .zarr stores found in: {data_dir}")
        raise ValueError(f"No .zarr stores found in: {data_dir}")

    srcs, dsts = [], []
    point_data_all = []

    processed_runs = 0
    for zarr_path in zarr_stores:
        if processed_runs >= num_samples:
            break

        if logger:
            logger.info(f"Processing Zarr store: {os.path.basename(zarr_path)}")

        try:
            mesh_pos, edges, point_data_dict = load_zarr_store(zarr_path)

            # Validate shapes
            if mesh_pos.ndim != 3 or mesh_pos.shape[-1] != 3:
                raise ValueError(
                    f"mesh_pos must be [T,N,3], got {mesh_pos.shape} in {zarr_path}"
                )

            if edges.ndim != 2 or edges.shape[-1] != 2:
                raise ValueError(
                    f"edges must be [E,2], got {edges.shape} in {zarr_path}"
                )

            num_nodes = mesh_pos.shape[1]

            # Validate point data features
            for name, data in point_data_dict.items():
                if data.ndim == 1:
                    if len(data) != num_nodes:
                        raise ValueError(
                            f"Point data '{name}' length {len(data)} doesn't match "
                            f"number of nodes {num_nodes} in {zarr_path}"
                        )
                elif data.ndim == 2:
                    if data.shape[0] != num_nodes:
                        raise ValueError(
                            f"Point data '{name}' shape {data.shape} doesn't match "
                            f"number of nodes {num_nodes} in {zarr_path}"
                        )
                else:
                    raise ValueError(
                        f"Point data '{name}' must be [N] or [N,K], got shape {data.shape} in {zarr_path}"
                    )

            # Validate edge indices are within bounds
            if edges.size > 0:
                if edges.min() < 0 or edges.max() >= num_nodes:
                    raise ValueError(
                        f"Edge indices out of bounds [0, {num_nodes - 1}] in {zarr_path}"
                    )

            # Extract source and destination node indices from edges
            src, dst = edges.T
            srcs.append(src)
            dsts.append(dst)

            # Create record with coordinates and all point data fields
            record = {"coords": mesh_pos}
            record.update(point_data_dict)  # Add all point data features dynamically
            point_data_all.append(record)

            processed_runs += 1

        except Exception as e:
            if logger:
                logger.error(f"Error processing {zarr_path}: {e}")
            raise

    if logger:
        logger.info(f"Successfully processed {processed_runs} Zarr stores")

    return srcs, dsts, point_data_all


class Reader:
    """
    Reader for Zarr crash simulation stores.

    This reader loads preprocessed crash simulation data from Zarr stores
    created by the PhysicsNeMo Curator ETL pipeline.
    """

    def __init__(self):
        """Initialize the Zarr reader."""
        pass

    def __call__(
        self,
        data_dir: str,
        num_samples: int,
        split: str | None = None,
        logger=None,
        **kwargs,
    ):
        """
        Load Zarr crash simulation data.

        Args:
            data_dir: Directory containing .zarr stores
            num_samples: Number of samples to load
            split: Data split ('train', 'validation', 'test') - not used for Zarr
            logger: Optional logger
            **kwargs: Additional arguments (ignored)

        Returns:
            srcs: List of source node arrays for graph edges
            dsts: List of destination node arrays for graph edges
            point_data: List of dicts with 'coords' and all available point data fields
        """
        return process_zarr_data(
            data_dir=data_dir,
            num_samples=num_samples,
            logger=logger,
        )
