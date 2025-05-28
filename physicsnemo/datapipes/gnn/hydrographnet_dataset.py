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

# ruff: noqa: S324,F821,S113

"""
HydroGraphDataset module

This module defines a DGLDataset for hydrograph-based graphs. It includes utility functions
for downloading data, computing normalization statistics, and processing both static and dynamic
data required to build a graph for each hydrograph sample.

The dataset supports two modes:
    - Training: Each sample is a sliding window sample.
    - Testing: Each sample corresponds to an entire hydrograph.

For testing, each sample returns a tuple (graph, rollout_data) containing the initial graph and
a dictionary of future hydrograph data for evaluation.
"""

import hashlib
import json
import logging
import math
import os
import random
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any, List, Optional, Union

import dgl
import numpy as np
import requests
import torch
from dgl.data import DGLDataset
from scipy.spatial import KDTree
from tqdm import tqdm

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# ---------------------------
# Download Utility Functions
# ---------------------------
def calculate_md5(fpath: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """
    Calculate the MD5 checksum of a file.

    Args:
        fpath (str or Path): Path to the file.
        chunk_size (int): Size of each chunk to read from the file.

    Returns:
        str: MD5 checksum of the file.
    """
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath: Union[str, Path], md5: str, **kwargs: Any) -> bool:
    """
    Check if the file at fpath has the expected MD5 checksum.

    Args:
        fpath (str or Path): Path to the file.
        md5 (str): Expected MD5 checksum.
        **kwargs: Additional keyword arguments for calculate_md5.

    Returns:
        bool: True if the file's checksum matches; False otherwise.
    """
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath: Union[str, Path], md5: Optional[str] = None) -> bool:
    """
    Verify the integrity of a file by checking its existence and, optionally, its MD5 checksum.

    Args:
        fpath (str or Path): File path to check.
        md5 (Optional[str]): Expected MD5 checksum (if any).

    Returns:
        bool: True if the file exists (and matches the checksum if provided); False otherwise.
    """
    fpath = Path(fpath)
    if not fpath.is_file():
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def download_from_url(
    url: str,
    root: Union[str, Path],
    filename: Optional[Union[str, Path]] = None,
    md5: Optional[str] = None,
    size: Optional[int] = None,
    chunk_size: int = 256 * 64,
    extract: bool = True,
) -> None:
    """
    Download a file from a URL, verify its integrity, and optionally extract it.

    Args:
        url (str): URL of the file to download.
        root (str or Path): Directory where the file will be saved.
        filename (Optional[str or Path]): Optional file name; if not provided, it is derived from the URL.
        md5 (Optional[str]): Expected MD5 checksum.
        size (Optional[int]): Expected file size.
        chunk_size (int): Chunk size for downloading.
        extract (bool): If True, extract the file if it is a tar or zip archive.
    """
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    if not filename:
        filename = url.split("/")[-1]
    fpath = root / filename
    if check_integrity(fpath, md5):
        logger.info(f"Using downloaded and verified file: {fpath}")
    else:
        logger.info(f"Downloading {url} to {fpath} ...")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            with open(fpath, "wb") as f, tqdm(
                desc=str(fpath),
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())
                        bar.update(len(chunk))
        if size is not None and fpath.stat().st_size != size:
            raise RuntimeError("Downloaded file has unexpected size.")
        if not check_integrity(fpath, md5):
            raise RuntimeError("File not found or corrupted.")
        logger.info(f"Saved to {fpath} successfully.")
    if extract:
        # Extract tar or zip archives
        if fpath.suffix in [".tar", ".gz", ".tgz"]:
            logger.info(f"Extracting tar archive {fpath}...")
            with tarfile.open(fpath, "r:*") as archive:
                archive.extractall(path=root)
                names = ", ".join(archive.getnames())
            logger.info(f"Extracted files: {names}")
        elif fpath.suffix == ".zip":
            logger.info(f"Extracting zip archive {fpath}...")
            with zipfile.ZipFile(fpath, "r") as z:
                z.extractall(path=root)
                names = ", ".join(z.namelist())
            logger.info(f"Extracted files: {names}")


def download_from_zenodo_record(
    record_id: str,
    root: Union[str, Path],
    files_to_download: Optional[List[str]] = None,
) -> None:
    """
    Download dataset files from a Zenodo record.

    Args:
        record_id (str): The Zenodo record ID.
        root (str or Path): Directory where files will be saved.
        files_to_download (Optional[List[str]]): Specific files to download; if None, download all.
    """
    zenodo_api_url = "https://zenodo.org/api/records/"
    url = f"{zenodo_api_url}{record_id}"
    logger.info(f"Fetching Zenodo record info for record ID {record_id} ...")
    resp = requests.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"Error: request failed with status code {resp.status_code}")
    response_json = resp.json()
    for file_record in response_json["files"]:
        fname = file_record["key"]
        if files_to_download is None or fname in files_to_download:
            file_url = file_record["links"]["self"]
            file_md5 = file_record["checksum"][4:]
            file_size = file_record["size"]
            download_from_url(
                url=file_url,
                root=root,
                filename=fname,
                md5=file_md5,
                size=file_size,
                extract=True,
            )


def ensure_data_available(data_dir: Union[str, Path]) -> None:
    """
    Ensure that the dataset is available in the specified directory.
    If not found, download the dataset from Zenodo.

    Args:
        data_dir (str or Path): Path to the data directory.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        logger.info(
            f"Data directory {data_dir} not found. Downloading dataset from Zenodo..."
        )
        download_from_zenodo_record(ZENODO_RECORD_ID, data_dir, FILES_TO_DOWNLOAD)
    else:
        logger.info(f"Data directory {data_dir} already exists. Skipping download.")


# Global constants for Zenodo record and filenames.
ZENODO_RECORD_ID = "14969507"
FILES_TO_DOWNLOAD = None

STATIC_NORM_STATS_FILE = "static_norm_stats.json"
DYNAMIC_NORM_STATS_FILE = "dynamic_norm_stats.json"


# ---------------------------
# HydroGraphDataset Class
# ---------------------------
class HydroGraphDataset(DGLDataset):
    """
    DGL Dataset for hydrograph-based graphs.

    This dataset processes both static and dynamic data to construct graphs for each hydrograph.
    It supports two modes:
        - Training ("train"): Each sample is a sliding window sample.
        - Testing ("test"): Each sample is a full hydrograph with rollout data.

    Attributes:
        data_dir (str): Directory where the dataset is located.
        prefix (str): Prefix for file names.
        num_samples (int): Maximum number of hydrograph samples.
        n_time_steps (int): Number of time steps used in the sliding window.
        k (int): Number of nearest neighbors for graph connectivity.
        noise_std (float): Standard deviation for added noise.
        noise_type (str): Type of noise to apply.
        hydrograph_ids_file (Optional[str]): File containing hydrograph IDs.
        split (str): Split type ("train" or "test").
        rollout_length (int): Number of rollout time steps (used in test mode).
        return_physics (bool): Flag to include physics data in __getitem__ output.
    """

    def __init__(
        self,
        name: str = "hydrograph_dataset",
        data_dir: Union[str, Path] = "data_directory",
        prefix: str = "M80",
        num_samples: int = 500,
        n_time_steps: int = 10,
        k: int = 4,
        noise_std: float = 0.01,
        noise_type: str = "none",
        hydrograph_ids_file: Optional[str] = None,
        split: str = "train",
        rollout_length: Optional[int] = None,
        force_reload: bool = False,
        verbose: bool = False,
        return_physics: bool = False,
    ):

        if split not in {"train", "test"}:
            raise ValueError(f"Invalid split '{split}'. Expected 'train' or 'test'.")

        # Initialize dataset attributes.
        self.data_dir = str(data_dir)
        ensure_data_available(self.data_dir)
        self.prefix = prefix
        self.num_samples = num_samples
        self.n_time_steps = n_time_steps
        self.k = k
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.hydrograph_ids_file = hydrograph_ids_file
        self.split = split
        # rollout_length is only used when split=="test"
        self.rollout_length = rollout_length if rollout_length is not None else 0
        self.return_physics = return_physics

        # Placeholders for static and dynamic data, indices, and normalization stats.
        self.static_data = {}
        self.dynamic_data = []
        self.sample_index = []
        self.hydrograph_ids = []
        self.static_stats = {}
        self.dynamic_stats = {}

        # Call the parent class constructor.
        super().__init__(name=name, force_reload=force_reload, verbose=verbose)

    def process(self) -> None:
        """
        Process the dataset to load static and dynamic data and compute necessary normalization stats.
        """
        if self.split == "train":
            # For training, load constant data and compute static normalization stats.
            (
                xy_coords,
                area,
                area_denorm,
                elevation,
                slope,
                aspect,
                curvature,
                manning,
                flow_accum,
                infiltration,
                self.static_stats,
            ) = self.load_constant_data(
                self.data_dir, self.prefix, norm_stats_static=None
            )
            self.save_norm_stats(self.static_stats, STATIC_NORM_STATS_FILE)
        else:
            # For test or validation, load precomputed normalization stats.
            self.static_stats = self.load_norm_stats(STATIC_NORM_STATS_FILE)
            (
                xy_coords,
                area,
                area_denorm,
                elevation,
                slope,
                aspect,
                curvature,
                manning,
                flow_accum,
                infiltration,
                _,
            ) = self.load_constant_data(
                self.data_dir, self.prefix, norm_stats_static=self.static_stats
            )

        # Build the graph connectivity using a k-d tree.
        num_nodes = xy_coords.shape[0]
        kdtree = KDTree(xy_coords)
        _, neighbors = kdtree.query(xy_coords, k=self.k + 1)
        edge_index = np.vstack(
            [(i, nbr) for i, nbrs in enumerate(neighbors) for nbr in nbrs if nbr != i]
        ).T
        edge_features = self.create_edge_features(xy_coords, edge_index)

        # Store static data.
        self.static_data = {
            "xy_coords": xy_coords,
            "area": area,
            "area_denorm": area_denorm,
            "elevation": elevation,
            "slope": slope,
            "aspect": aspect,
            "curvature": curvature,
            "manning": manning,
            "flow_accum": flow_accum,
            "infiltration": infiltration,
            "edge_index": edge_index,
            "edge_features": edge_features,
        }

        # Read hydrograph IDs either from a file or from the directory.
        if self.hydrograph_ids_file is not None:
            file_path = os.path.join(self.data_dir, self.hydrograph_ids_file)
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    lines = f.readlines()
                self.hydrograph_ids = [line.strip() for line in lines if line.strip()]
            else:
                raise FileNotFoundError(f"Hydrograph IDs file not found: {file_path}")
        else:
            all_files = os.listdir(self.data_dir)
            self.hydrograph_ids = []
            for f in all_files:
                if f.startswith(f"{self.prefix}_WD_") and f.endswith(".txt"):
                    parts = f.split("_")
                    if len(parts) >= 3:
                        hid = os.path.splitext(parts[2])[0]
                        self.hydrograph_ids.append(hid)
        if len(self.hydrograph_ids) > self.num_samples:
            self.hydrograph_ids = random.sample(self.hydrograph_ids, self.num_samples)

        # Process dynamic data (water depth, inflow, volume, precipitation) for each hydrograph.
        temp_dynamic_data = []
        water_depth_list = []
        volume_list = []
        precipitation_list = []
        inflow_list = []
        for hid in tqdm(self.hydrograph_ids, desc="Processing Hydrographs"):
            (
                water_depth,
                inflow_hydrograph,
                volume,
                precipitation,
            ) = self.load_dynamic_data(
                self.data_dir, hid, self.prefix, num_points=num_nodes
            )
            temp_dynamic_data.append(
                {
                    "water_depth": water_depth,
                    "inflow_hydrograph": inflow_hydrograph,
                    "volume": volume,
                    "precipitation": precipitation,
                    "hydro_id": hid,
                }
            )
            water_depth_list.append(water_depth.flatten())
            volume_list.append(volume.flatten())
            precipitation_list.append(precipitation.flatten())
            inflow_list.append(inflow_hydrograph.flatten())

        # Compute dynamic normalization statistics for training or load precomputed stats.
        if self.split == "train":
            self.dynamic_stats = {}
            water_depth_all = np.concatenate(water_depth_list)
            self.dynamic_stats["water_depth"] = {
                "mean": float(np.mean(water_depth_all)),
                "std": float(np.std(water_depth_all)),
            }
            volume_all = np.concatenate(volume_list)
            self.dynamic_stats["volume"] = {
                "mean": float(np.mean(volume_all)),
                "std": float(np.std(volume_all)),
            }
            precipitation_all = np.concatenate(precipitation_list)
            self.dynamic_stats["precipitation"] = {
                "mean": float(np.mean(precipitation_all)),
                "std": float(np.std(precipitation_all)),
            }
            inflow_all = np.concatenate(inflow_list)
            self.dynamic_stats["inflow_hydrograph"] = {
                "mean": float(np.mean(inflow_all)),
                "std": float(np.std(inflow_all)),
            }
            self.save_norm_stats(self.dynamic_stats, DYNAMIC_NORM_STATS_FILE)
        else:
            self.dynamic_stats = self.load_norm_stats(DYNAMIC_NORM_STATS_FILE)

        # Normalize the dynamic data.
        self.dynamic_data = []
        for dyn in temp_dynamic_data:
            dyn_std = {
                "water_depth": self.normalize(
                    dyn["water_depth"],
                    self.dynamic_stats["water_depth"]["mean"],
                    self.dynamic_stats["water_depth"]["std"],
                ),
                "volume": self.normalize(
                    dyn["volume"],
                    self.dynamic_stats["volume"]["mean"],
                    self.dynamic_stats["volume"]["std"],
                ),
                "precipitation": self.normalize(
                    dyn["precipitation"],
                    self.dynamic_stats["precipitation"]["mean"],
                    self.dynamic_stats["precipitation"]["std"],
                ),
                "inflow_hydrograph": self.normalize(
                    dyn["inflow_hydrograph"],
                    self.dynamic_stats["inflow_hydrograph"]["mean"],
                    self.dynamic_stats["inflow_hydrograph"]["std"],
                ),
                "hydro_id": dyn["hydro_id"],
            }
            self.dynamic_data.append(dyn_std)

        # Build sample indices for training (sliding window) or validate test data.
        if self.split == "train":
            for h_idx, dyn in enumerate(self.dynamic_data):
                T = dyn["water_depth"].shape[0]
                if self.noise_type == "pushforward":
                    max_t = T - self.n_time_steps - 1
                else:
                    max_t = T - self.n_time_steps
                for t in range(max_t):
                    self.sample_index.append((h_idx, t))
            self.length = len(self.sample_index)
        elif self.split == "test":
            for dyn in self.dynamic_data:
                T = dyn["water_depth"].shape[0]
                if T < self.n_time_steps + self.rollout_length:
                    raise ValueError(
                        f"Hydrograph {dyn['hydro_id']} does not have enough time steps for the specified rollout_length."
                    )
            self.length = len(self.dynamic_data)

    def __getitem__(self, idx: int):
        """
        Retrieve a graph sample (and associated physics data if required).

        Args:
            idx (int): Index of the sample.

        Returns:
            Depending on the split:
                - Training: A DGL graph with node features, edge features, and target values, optionally
                  along with a dictionary of physics data.
                - Testing: A tuple (graph, rollout_data) where rollout_data contains future hydrograph data.
        """
        sd = self.static_data
        if self.split != "test":
            # Training mode: use sliding window sample.
            hydro_idx, t_idx = self.sample_index[idx]
            dyn = self.dynamic_data[hydro_idx]

            # Determine the end index for the dynamic window.
            end_index = (
                t_idx + self.n_time_steps + 1
                if self.noise_type == "pushforward"
                else t_idx + self.n_time_steps
            )

            # Compute node features and future flow/precipitation values.
            node_features, future_flow, future_precip = self.create_node_features(
                sd["xy_coords"],
                sd["area"],
                sd["elevation"],
                sd["slope"],
                sd["aspect"],
                sd["curvature"],
                sd["manning"],
                sd["flow_accum"],
                sd["infiltration"],
                dyn["water_depth"][t_idx:end_index, :],
                dyn["volume"][t_idx:end_index, :],
                dyn["precipitation"],
                t_idx,
                self.n_time_steps,
                dyn["inflow_hydrograph"],
            )
            target_time = t_idx + self.n_time_steps
            prev_time = target_time - 1
            # Compute target differences for water depth and volume.
            target_depth = (
                dyn["water_depth"][target_time, :] - dyn["water_depth"][prev_time, :]
            )
            target_volume = dyn["volume"][target_time, :] - dyn["volume"][prev_time, :]
            target = np.stack([target_depth, target_volume], axis=1)

            # Create the graph with DGL.
            src, dst = sd["edge_index"]
            g = dgl.graph((src, dst))
            g.edata["x"] = torch.tensor(sd["edge_features"], dtype=torch.float)
            g.ndata["x"] = torch.tensor(node_features, dtype=torch.float)
            g.ndata["y"] = torch.tensor(target, dtype=torch.float)

            # Determine if physics data should be returned.
            need_physics = self.return_physics or (self.noise_type == "pushforward")
            if need_physics:
                # Compute physics data in the denormalized domain.
                past_volume = float(np.sum(dyn["volume"][prev_time, :]))
                future_volume = (
                    float(np.sum(dyn["volume"][target_time + 1, :]))
                    if (target_time + 1 < dyn["volume"].shape[0])
                    else float(np.sum(dyn["volume"][target_time, :]))
                )
                avg_inflow_norm = float(
                    (
                        dyn["inflow_hydrograph"][prev_time]
                        + dyn["inflow_hydrograph"][target_time]
                    )
                    / 2
                )
                avg_precip_norm = float(
                    (
                        dyn["precipitation"][prev_time]
                        + dyn["precipitation"][target_time]
                    )
                    / 2
                )
                denorm_avg_inflow = (
                    avg_inflow_norm * self.dynamic_stats["inflow_hydrograph"]["std"]
                    + self.dynamic_stats["inflow_hydrograph"]["mean"]
                )
                denorm_avg_precip = (
                    avg_precip_norm * self.dynamic_stats["precipitation"]["std"]
                    + self.dynamic_stats["precipitation"]["mean"]
                )

                # --- New: Compute next-step inflow and precipitation for physics loss term2 ---
                if (target_time + 1) < dyn["inflow_hydrograph"].shape[0]:
                    next_inflow_norm = dyn["inflow_hydrograph"][target_time + 1]
                    next_precip_norm = dyn["precipitation"][target_time + 1]
                else:
                    next_inflow_norm = dyn["inflow_hydrograph"][target_time]
                    next_precip_norm = dyn["precipitation"][target_time]
                denorm_next_inflow = (
                    next_inflow_norm * self.dynamic_stats["inflow_hydrograph"]["std"]
                    + self.dynamic_stats["inflow_hydrograph"]["mean"]
                )
                denorm_next_precip = (
                    next_precip_norm * self.dynamic_stats["precipitation"]["std"]
                    + self.dynamic_stats["precipitation"]["mean"]
                )

                # Build the complete physics data dictionary.
                full_physics_data = {
                    "flow_future": float(
                        future_flow * self.dynamic_stats["inflow_hydrograph"]["std"]
                        + self.dynamic_stats["inflow_hydrograph"]["mean"]
                    ),
                    "precip_future": float(
                        future_precip * self.dynamic_stats["precipitation"]["std"]
                        + self.dynamic_stats["precipitation"]["mean"]
                    ),
                    "past_volume": past_volume,
                    "future_volume": future_volume,
                    "avg_inflow": denorm_avg_inflow,
                    "avg_precipitation": denorm_avg_precip,
                    "next_inflow": denorm_next_inflow,
                    "next_precip": denorm_next_precip,
                    "volume_mean": float(self.dynamic_stats["volume"]["mean"]),
                    "volume_std": float(self.dynamic_stats["volume"]["std"]),
                    "inflow_mean": float(
                        self.dynamic_stats["inflow_hydrograph"]["mean"]
                    ),
                    "inflow_std": float(self.dynamic_stats["inflow_hydrograph"]["std"]),
                    "precip_mean": float(self.dynamic_stats["precipitation"]["mean"]),
                    "precip_std": float(self.dynamic_stats["precipitation"]["std"]),
                    "num_nodes": float(sd["xy_coords"].shape[0]),
                    "area_sum": float(np.sum(sd["area_denorm"])),
                    "infiltration_area_sum": float(
                        np.sum(
                            self.denormalize(
                                sd["infiltration"],
                                self.static_stats["infiltration"]["mean"],
                                self.static_stats["infiltration"]["std"],
                            )
                            * sd["area_denorm"]
                        )
                    )
                    / 100.0,
                }
                # For pushforward noise without full physics data requested.
                if not self.return_physics and self.noise_type == "pushforward":
                    physics_data = {
                        "flow_future": full_physics_data["flow_future"],
                        "precip_future": full_physics_data["precip_future"],
                        "next_inflow": full_physics_data["next_inflow"],
                        "next_precip": full_physics_data["next_precip"],
                    }
                else:
                    physics_data = full_physics_data
                return g, physics_data
            else:
                return g
        else:
            # Test mode: Each sample returns a graph and a rollout data dictionary.
            dyn = self.dynamic_data[idx]
            node_features, _, _ = self.create_node_features(
                sd["xy_coords"],
                sd["area"],
                sd["elevation"],
                sd["slope"],
                sd["aspect"],
                sd["curvature"],
                sd["manning"],
                sd["flow_accum"],
                sd["infiltration"],
                dyn["water_depth"][0 : self.n_time_steps, :],
                dyn["volume"][0 : self.n_time_steps, :],
                dyn["precipitation"],
                0,
                self.n_time_steps,
                dyn["inflow_hydrograph"],
            )
            src, dst = sd["edge_index"]
            g = dgl.graph((src, dst))
            g.edata["x"] = torch.tensor(sd["edge_features"], dtype=torch.float)
            g.ndata["x"] = torch.tensor(node_features, dtype=torch.float)
            rollout_data = {
                "inflow": torch.tensor(
                    dyn["inflow_hydrograph"][
                        self.n_time_steps : self.n_time_steps + self.rollout_length
                    ],
                    dtype=torch.float,
                ),
                "precipitation": torch.tensor(
                    dyn["precipitation"][
                        self.n_time_steps : self.n_time_steps + self.rollout_length
                    ],
                    dtype=torch.float,
                ),
                "water_depth_gt": torch.tensor(
                    dyn["water_depth"][
                        self.n_time_steps : self.n_time_steps + self.rollout_length
                    ],
                    dtype=torch.float,
                ),
                "volume_gt": torch.tensor(
                    dyn["volume"][
                        self.n_time_steps : self.n_time_steps + self.rollout_length
                    ],
                    dtype=torch.float,
                ),
            }
            return g, rollout_data

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.length

    @staticmethod
    def normalize(
        data: np.ndarray,
        mean: Union[float, list, np.ndarray],
        std: Union[float, list, np.ndarray],
        epsilon: float = 1e-8,
    ) -> np.ndarray:
        """
        Normalize the data using the provided mean and standard deviation.

        Args:
            data (np.ndarray): Data to normalize.
            mean (float, list, or np.ndarray): Mean value(s).
            std (float, list, or np.ndarray): Standard deviation value(s).
            epsilon (float): Small constant to avoid division by zero.

        Returns:
            np.ndarray: Normalized data.
        """
        mean = np.array(mean) if isinstance(mean, list) else mean
        std = np.array(std) if isinstance(std, list) else std
        return (data - mean) / (std + epsilon)

    @staticmethod
    def denormalize(
        data: np.ndarray,
        mean: Union[float, list, np.ndarray],
        std: Union[float, list, np.ndarray],
        epsilon: float = 1e-8,
    ) -> np.ndarray:
        """
        Denormalize the data using the provided mean and standard deviation.

        Args:
            data (np.ndarray): Normalized data.
            mean (float, list, or np.ndarray): Mean value(s) used for normalization.
            std (float, list, or np.ndarray): Standard deviation used for normalization.
            epsilon (float): Small constant to avoid division by zero.

        Returns:
            np.ndarray: Denormalized data.
        """
        mean = np.array(mean) if isinstance(mean, list) else mean
        std = np.array(std) if isinstance(std, list) else std
        return data * (std + epsilon) + mean

    def apply_noise_to_feature(
        self, data: np.ndarray, noise_type: str, noise_std: float
    ) -> np.ndarray:
        """
        Apply specified noise to a feature matrix.

        Args:
            data (np.ndarray): Input data of shape (T, num_nodes).
            noise_type (str): Type of noise ("none", "only_last", "correlated", "uncorrelated", "random_walk").
            noise_std (float): Standard deviation of the noise.

        Returns:
            np.ndarray: Data with noise applied.
        """
        if noise_type in ["none", "pushforward"]:
            return data
        T, num_nodes = data.shape
        if noise_type == "only_last":
            noise = np.random.normal(0, noise_std, size=(1, num_nodes))
            data_modified = data.copy()
            data_modified[-1] += noise[0]
            return data_modified
        elif noise_type == "correlated":
            noise = np.random.normal(0, noise_std, size=(1, num_nodes))
            return data + noise
        elif noise_type == "uncorrelated":
            noise = np.random.normal(0, noise_std, size=(T, num_nodes))
            return data + noise
        elif noise_type == "random_walk":
            noise_increments = np.random.normal(
                0, noise_std / math.sqrt(T), size=(T, num_nodes)
            )
            noise_cumulative = np.cumsum(noise_increments, axis=0)
            return data + noise_cumulative
        else:
            logger.warning(f"Unknown noise_type={noise_type}, skipping noise.")
            return data

    def save_norm_stats(self, stats: dict, filename: str) -> None:
        """
        Save normalization statistics to a JSON file.

        Args:
            stats (dict): Dictionary of normalization statistics.
            filename (str): Filename to save the stats.
        """
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, "w") as f:
            json.dump(stats, f)

    def load_norm_stats(self, filename: str) -> dict:
        """
        Load normalization statistics from a JSON file.

        Args:
            filename (str): Filename from which to load the stats.

        Returns:
            dict: Normalization statistics.
        """
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, "r") as f:
            stats = json.load(f)
        return stats

    def load_constant_data(
        self, folder: str, prefix: str, norm_stats_static: Optional[dict] = None
    ):
        """
        Load and standardize static (constant) data such as coordinates, elevation, and flow accumulation.

        Args:
            folder (str): Directory where the static data files are located.
            prefix (str): Prefix for file names.
            norm_stats_static (Optional[dict]): Precomputed static normalization statistics.

        Returns:
            Tuple containing standardized static data and the updated normalization stats.
        """
        epsilon = 1e-8
        stats = norm_stats_static if norm_stats_static is not None else {}

        def standardize(data: np.ndarray, key: str) -> np.ndarray:
            """
            Standardize data by subtracting the mean and dividing by the standard deviation.
            """
            if key in stats:
                mean_val = np.array(stats[key]["mean"])
                std_val = np.array(stats[key]["std"])
            else:
                mean_val = np.mean(data, axis=0)
                std_val = np.std(data, axis=0)
                stats[key] = {"mean": mean_val.tolist(), "std": std_val.tolist()}
            return (data - mean_val) / (std_val + epsilon)

        # Load each file using the given prefix.
        xy_path = os.path.join(folder, f"{prefix}_XY.txt")
        ca_path = os.path.join(folder, f"{prefix}_CA.txt")
        ce_path = os.path.join(folder, f"{prefix}_CE.txt")
        cs_path = os.path.join(folder, f"{prefix}_CS.txt")
        aspect_path = os.path.join(folder, f"{prefix}_A.txt")
        curvature_path = os.path.join(folder, f"{prefix}_CU.txt")
        manning_path = os.path.join(folder, f"{prefix}_N.txt")
        flow_accum_path = os.path.join(folder, f"{prefix}_FA.txt")
        infiltration_path = os.path.join(folder, f"{prefix}_IP.txt")

        xy_coords = np.loadtxt(xy_path, delimiter="\t")
        xy_coords = standardize(xy_coords, "xy_coords")
        area_denorm = np.loadtxt(ca_path, delimiter="\t")[: xy_coords.shape[0]].reshape(
            -1, 1
        )
        area = standardize(area_denorm, "area")
        elevation = np.loadtxt(ce_path, delimiter="\t")[: xy_coords.shape[0]].reshape(
            -1, 1
        )
        elevation = standardize(elevation, "elevation")
        slope = np.loadtxt(cs_path, delimiter="\t")[: xy_coords.shape[0]].reshape(-1, 1)
        slope = standardize(slope, "slope")
        aspect = np.loadtxt(aspect_path, delimiter="\t")[: xy_coords.shape[0]].reshape(
            -1, 1
        )
        aspect = standardize(aspect, "aspect")
        curvature = np.loadtxt(curvature_path, delimiter="\t")[
            : xy_coords.shape[0]
        ].reshape(-1, 1)
        curvature = standardize(curvature, "curvature")
        manning = np.loadtxt(manning_path, delimiter="\t")[
            : xy_coords.shape[0]
        ].reshape(-1, 1)
        manning = standardize(manning, "manning")
        flow_accum = np.loadtxt(flow_accum_path, delimiter="\t")[
            : xy_coords.shape[0]
        ].reshape(-1, 1)
        flow_accum = standardize(flow_accum, "flow_accum")
        infiltration = np.loadtxt(infiltration_path, delimiter="\t")[
            : xy_coords.shape[0]
        ].reshape(-1, 1)
        infiltration = standardize(infiltration, "infiltration")
        return (
            xy_coords,
            area,
            area_denorm,
            elevation,
            slope,
            aspect,
            curvature,
            manning,
            flow_accum,
            infiltration,
            stats,
        )

    def load_dynamic_data(
        self,
        folder: str,
        hydrograph_id: str,
        prefix: str,
        num_points: int,
        interval: int = 1,
        skip: int = 72,
    ):
        """
        Load dynamic data (water depth, inflow, volume, and precipitation) for a given hydrograph.

        Args:
            folder (str): Directory where the dynamic data files are located.
            hydrograph_id (str): Identifier for the hydrograph.
            prefix (str): Prefix for file names.
            num_points (int): Number of spatial points (nodes).
            interval (int): Sampling interval.
            skip (int): Number of initial time steps to skip.

        Returns:
            Tuple of np.ndarray: (water_depth, inflow_hydrograph, volume, precipitation)
        """
        wd_path = os.path.join(folder, f"{prefix}_WD_{hydrograph_id}.txt")
        inflow_path = os.path.join(folder, f"{prefix}_US_InF_{hydrograph_id}.txt")
        volume_path = os.path.join(folder, f"{prefix}_V_{hydrograph_id}.txt")
        precipitation_path = os.path.join(folder, f"{prefix}_Pr_{hydrograph_id}.txt")
        water_depth = np.loadtxt(wd_path, delimiter="\t")[skip::interval, :num_points]
        inflow_hydrograph = np.loadtxt(inflow_path, delimiter="\t")[skip::interval, 1]
        volume = np.loadtxt(volume_path, delimiter="\t")[skip::interval, :num_points]
        precipitation = np.loadtxt(precipitation_path, delimiter="\t")[skip::interval]
        # Limit data until 25 time steps after the peak inflow.
        peak_time_idx = np.argmax(inflow_hydrograph)
        water_depth = water_depth[: peak_time_idx + 25]
        volume = volume[: peak_time_idx + 25]
        precipitation = (
            precipitation[: peak_time_idx + 25] * 2.7778e-7
        )  # Unit conversion
        inflow_hydrograph = inflow_hydrograph[: peak_time_idx + 25]
        return water_depth, inflow_hydrograph, volume, precipitation

    def create_node_features(
        self,
        xy_coords: np.ndarray,
        area: np.ndarray,
        elevation: np.ndarray,
        slope: np.ndarray,
        aspect: np.ndarray,
        curvature: np.ndarray,
        manning: np.ndarray,
        flow_accum: np.ndarray,
        infiltration: np.ndarray,
        water_depth: np.ndarray,
        volume: np.ndarray,
        precipitation_data: np.ndarray,
        time_step: int,
        n_time_steps: int,
        inflow_hydrograph: np.ndarray,
    ) -> (np.ndarray, float, float):
        """
        Create node features by combining static and dynamic data.

        Args:
            xy_coords (np.ndarray): Spatial coordinates.
            area (np.ndarray): Normalized area.
            elevation (np.ndarray): Normalized elevation.
            slope (np.ndarray): Normalized slope.
            aspect (np.ndarray): Normalized aspect.
            curvature (np.ndarray): Normalized curvature.
            manning (np.ndarray): Normalized Manning coefficient.
            flow_accum (np.ndarray): Normalized flow accumulation.
            infiltration (np.ndarray): Normalized infiltration.
            water_depth (np.ndarray): Dynamic water depth data (time x nodes).
            volume (np.ndarray): Dynamic volume data (time x nodes).
            precipitation_data (np.ndarray): Dynamic precipitation data.
            time_step (int): Starting time step.
            n_time_steps (int): Number of time steps in the window.
            inflow_hydrograph (np.ndarray): Dynamic inflow data.

        Returns:
            Tuple:
                - features (np.ndarray): Node feature matrix.
                - future_inflow (float): Future inflow at time_step+n_time_steps.
                - future_precip (float): Future precipitation at time_step+n_time_steps.
        """
        # Apply noise if required (excluding "none" and "pushforward").
        if self.noise_type not in ["none", "pushforward"]:
            window_slice = slice(time_step, time_step + n_time_steps)
            water_depth[window_slice, :] = self.apply_noise_to_feature(
                water_depth[window_slice, :], self.noise_type, self.noise_std
            )
            volume[window_slice, :] = self.apply_noise_to_feature(
                volume[window_slice, :], self.noise_type, self.noise_std
            )
        num_nodes = xy_coords.shape[0]
        # Create static copies of inflow and precipitation for each node.
        flow_hydrograph_current_step = np.full(
            (num_nodes, 1), inflow_hydrograph[time_step]
        )
        precip_current_step = np.full((num_nodes, 1), precipitation_data[time_step])
        # Concatenate all features horizontally.
        features = np.hstack(
            [
                xy_coords,
                area,
                elevation,
                slope,
                aspect,
                curvature,
                manning,
                flow_accum,
                infiltration,
                flow_hydrograph_current_step,
                precip_current_step,
                water_depth.T,
                volume.T,
            ]
        )
        future_inflow = inflow_hydrograph[time_step + n_time_steps]
        future_precip = precipitation_data[time_step + n_time_steps]
        return features, future_inflow, future_precip

    def create_edge_features(
        self, xy_coords: np.ndarray, edge_index: np.ndarray
    ) -> np.ndarray:
        """
        Create edge features based on the relative positions of connected nodes.

        Args:
            xy_coords (np.ndarray): Node spatial coordinates.
            edge_index (np.ndarray): Array containing source and destination indices for each edge.

        Returns:
            np.ndarray: Concatenated edge features (relative coordinates and normalized distance).
        """
        row, col = edge_index
        relative_coords = xy_coords[row] - xy_coords[col]
        distance = np.linalg.norm(relative_coords, axis=1)
        epsilon = 1e-8
        # Normalize relative coordinates and distance.
        relative_coords = (relative_coords - np.mean(relative_coords, axis=0)) / (
            np.std(relative_coords, axis=0) + epsilon
        )
        distance = (distance - np.mean(distance)) / (np.std(distance) + epsilon)
        return np.hstack([relative_coords, distance[:, None]])
