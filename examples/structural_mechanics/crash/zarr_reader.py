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


def _validate_point_data_shapes(
    store,
    zarr_path: str,
    num_nodes: int,
):
    """Validate point-data array shapes using Zarr metadata only."""
    for name in store.keys():
        if name in ("mesh_pos", "edges"):
            continue
        if name.startswith("mesh_connectivity_"):
            continue

        data = store[name]
        if data.ndim == 1:
            if data.shape[0] != num_nodes:
                raise ValueError(
                    f"Point data '{name}' length {data.shape[0]} doesn't match "
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


def validate_zarr_store(zarr_path: str) -> int:
    """
    Validate a Zarr store and return the number of mesh nodes.

    Reads only metadata and edge indices (small arrays), not full mesh trajectories.
    """
    store = zarr.open(zarr_path, mode="r")

    if "mesh_pos" not in store:
        raise KeyError(f"'mesh_pos' not found in Zarr store {zarr_path}")
    mesh_pos = store["mesh_pos"]
    if mesh_pos.ndim != 3 or mesh_pos.shape[-1] != 3:
        raise ValueError(
            f"mesh_pos must be [T,N,3], got {mesh_pos.shape} in {zarr_path}"
        )

    if "edges" not in store:
        raise KeyError(f"'edges' not found in Zarr store {zarr_path}")
    edges = store["edges"]
    if edges.ndim != 2 or edges.shape[-1] != 2:
        raise ValueError(f"edges must be [E,2], got {edges.shape} in {zarr_path}")

    num_nodes = mesh_pos.shape[1]
    _validate_point_data_shapes(store, zarr_path, num_nodes)

    edges_arr = np.array(edges[:], dtype=np.int64)
    if edges_arr.size > 0:
        if edges_arr.min() < 0 or edges_arr.max() >= num_nodes:
            raise ValueError(
                f"Edge indices out of bounds [0, {num_nodes - 1}] in {zarr_path}"
            )

    return num_nodes


def load_zarr_edges(zarr_path: str):
    """Load edge connectivity from a Zarr store."""
    store = zarr.open(zarr_path, mode="r")
    if "edges" not in store:
        raise KeyError(f"'edges' not found in Zarr store {zarr_path}")
    edges = np.array(store["edges"][:], dtype=np.int64)
    if edges.ndim != 2 or edges.shape[-1] != 2:
        raise ValueError(f"edges must be [E,2], got {edges.shape} in {zarr_path}")
    return edges


def load_zarr_store(zarr_path: str, num_timesteps: int | None = None):
    """
    Load mesh positions, edges, and all point data fields from a Zarr store.

    Args:
        zarr_path: Path to the Zarr store directory.
        num_timesteps: Optional cap on timesteps read from ``mesh_pos``.

    Returns:
        mesh_pos: (timesteps, num_nodes, 3) temporal positions
        edges: (num_edges, 2) edge connectivity
        point_data_dict: Dictionary of all point data fields (e.g., thickness, etc.)
    """
    store = zarr.open(zarr_path, mode="r")

    if "mesh_pos" not in store:
        raise KeyError(f"'mesh_pos' not found in Zarr store {zarr_path}")
    mesh_pos_arr = store["mesh_pos"]
    if num_timesteps is not None:
        mesh_pos = np.array(mesh_pos_arr[:num_timesteps], dtype=np.float64)
    else:
        mesh_pos = np.array(mesh_pos_arr[:], dtype=np.float64)

    if "edges" not in store:
        raise KeyError(f"'edges' not found in Zarr store {zarr_path}")
    edges = np.array(store["edges"][:], dtype=np.int64)

    point_data_dict = {}
    for name in store.keys():
        if name in ("mesh_pos", "edges"):
            continue
        if name.startswith("mesh_connectivity_"):
            continue
        point_data_dict[name] = np.array(store[name][:], dtype=np.float32)

    return mesh_pos, edges, point_data_dict


def materialize_zarr_record(
    zarr_path: str,
    num_timesteps: int | None = None,
) -> dict:
    """
    Materialize a lazy Zarr record into an in-memory reader dict.

    Returns a dict with ``coords`` and all point-data fields, matching the eager
    reader output.
    """
    mesh_pos, edges, point_data_dict = load_zarr_store(
        zarr_path, num_timesteps=num_timesteps
    )
    num_nodes = mesh_pos.shape[1]

    if mesh_pos.ndim != 3 or mesh_pos.shape[-1] != 3:
        raise ValueError(
            f"mesh_pos must be [T,N,3], got {mesh_pos.shape} in {zarr_path}"
        )
    if edges.ndim != 2 or edges.shape[-1] != 2:
        raise ValueError(f"edges must be [E,2], got {edges.shape} in {zarr_path}")

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

    if edges.size > 0:
        if edges.min() < 0 or edges.max() >= num_nodes:
            raise ValueError(
                f"Edge indices out of bounds [0, {num_nodes - 1}] in {zarr_path}"
            )

    record = {"coords": mesh_pos}
    record.update(point_data_dict)
    return record


def process_zarr_data(
    data_dir: str,
    num_samples: int,
    logger=None,
    lazy_load: bool = False,
):
    """
    Process Zarr crash simulation data from a given directory.

    Each .zarr store is treated as one sample. Reads mesh positions, edges,
    and all available point data fields (e.g., thickness, etc.) from the Zarr stores.

    When ``lazy_load=True``, only edge connectivity is loaded eagerly. Mesh
    trajectories and point features are loaded on demand via
    :func:`materialize_zarr_record`.

    Args:
        data_dir: Directory containing .zarr stores
        num_samples: Maximum number of samples to process
        logger: Optional logger for logging progress
        lazy_load: Defer loading heavy mesh/point arrays until materialization

    Returns:
        srcs: List of source node indices for edges (one array per sample)
        dsts: List of destination node indices for edges (one array per sample)
        point_data_all: List of dicts with 'coords' and all point data fields,
            or lazy handles when ``lazy_load=True``.
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
            if lazy_load:
                num_nodes = validate_zarr_store(zarr_path)
                edges = load_zarr_edges(zarr_path)
                src, dst = edges.T
                srcs.append(src)
                dsts.append(dst)
                point_data_all.append(
                    {
                        "_lazy": True,
                        "_zarr_path": zarr_path,
                        "_num_nodes": num_nodes,
                    }
                )
            else:
                mesh_pos, edges, point_data_dict = load_zarr_store(zarr_path)

                if mesh_pos.ndim != 3 or mesh_pos.shape[-1] != 3:
                    raise ValueError(
                        f"mesh_pos must be [T,N,3], got {mesh_pos.shape} in {zarr_path}"
                    )

                if edges.ndim != 2 or edges.shape[-1] != 2:
                    raise ValueError(
                        f"edges must be [E,2], got {edges.shape} in {zarr_path}"
                    )

                num_nodes = mesh_pos.shape[1]

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

                if edges.size > 0:
                    if edges.min() < 0 or edges.max() >= num_nodes:
                        raise ValueError(
                            f"Edge indices out of bounds [0, {num_nodes - 1}] in {zarr_path}"
                        )

                src, dst = edges.T
                srcs.append(src)
                dsts.append(dst)

                record = {"coords": mesh_pos}
                record.update(point_data_dict)
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

    By default ``lazy_load=True`` so multi-GPU training does not materialize
    every run on every rank at dataset construction time.
    """

    def __init__(self, lazy_load: bool = True):
        """
        Args:
            lazy_load: When True, defer loading mesh trajectories and point
                features until the datapipe accesses a sample.
        """
        self.lazy_load = lazy_load

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
            point_data: List of dicts with 'coords' and all available point data fields,
                or lazy handles when ``lazy_load=True``.
            global_features: Empty list (Zarr stores do not embed global features)
        """
        srcs, dsts, point_data = process_zarr_data(
            data_dir=data_dir,
            num_samples=num_samples,
            logger=logger,
            lazy_load=self.lazy_load,
        )
        return srcs, dsts, point_data, []
