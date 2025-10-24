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

import os
import glob
import pyvista as pv
import numpy as np
import csv
import shutil
import hydra
import logging
from omegaconf import DictConfig
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from physicsnemo.launch.logging import LaunchLogger

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)


def process_and_save_vtp(args):
    id_, xmgn_path, fignet_path, domino_path, output_dir = args

    # Read meshes
    xmgn_mesh = pv.read(xmgn_path)
    fignet_mesh = pv.read(fignet_path)
    domino_mesh = pv.read(domino_path)
    domino_mesh = domino_mesh.cell_data_to_point_data()

    # Use geometry, normals, and area from xmgn
    mesh = xmgn_mesh.copy()
    mesh = mesh.compute_normals(point_normals=True, cell_normals=False)
    mesh = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    mesh.point_data["Normals"] = mesh.point_normals
    mesh.cell_data["Area"] = mesh["Area"]

    # True fields (from any mesh, here xmgn)
    mesh.point_data["pMeanTrim"] = xmgn_mesh.point_data["pMeanTrim"]
    mesh.point_data["wallShearStressMeanTrim"] = xmgn_mesh.point_data[
        "wallShearStressMeanTrim"
    ]

    # Predicted fields for each model
    mesh.point_data["pMeanTrimPred_xmgn"] = xmgn_mesh.point_data["pMeanTrimPred"]
    mesh.point_data["pMeanTrimPred_fignet"] = fignet_mesh.point_data["pMeanTrimPred"]
    mesh.point_data["pMeanTrimPred_domino"] = domino_mesh.point_data["pMeanTrimPred"]

    mesh.point_data["wallShearStressMeanTrimPred_xmgn"] = xmgn_mesh.point_data[
        "wallShearStressMeanTrimPred"
    ]
    mesh.point_data["wallShearStressMeanTrimPred_fignet"] = fignet_mesh.point_data[
        "wallShearStressMeanTrimPred"
    ]
    mesh.point_data["wallShearStressMeanTrimPred_domino"] = domino_mesh.point_data[
        "wallShearStressMeanTrimPred"
    ]

    # Remove original predicted fields to avoid redundancy
    for field in ["pMeanTrimPred", "wallShearStressMeanTrimPred"]:
        if field in mesh.point_data:
            del mesh.point_data[field]

    # Save to output directory
    out_path = os.path.join(output_dir, f"processed_{id_}.vtp")
    mesh.save(out_path)

    # Return true fields and area for normalization stats
    return (
        mesh.point_data["pMeanTrim"],
        mesh.point_data["wallShearStressMeanTrim"],
        mesh.cell_data["Area"],
    )


def split_data(output_dir, validation_ids_csv):
    """
    Split processed VTP files into train/val directories based on validation IDs.

    Args:
        output_dir: Directory containing processed VTP files
        validation_ids_csv: Path to CSV file containing validation IDs
    """
    # Step 1: Read validation IDs from CSV (skip header)
    val_ids = set()
    if os.path.exists(validation_ids_csv):
        with open(validation_ids_csv, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if row:  # Check if row is not empty
                    val_ids.add(row[0])
    else:
        logging.warning(
            f"Validation IDs CSV file {validation_ids_csv} not found. Skipping data splitting."
        )
        return

    # Step 2: Create destination directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    stats_dir = os.path.join(output_dir, "stats")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    # Step 3: Move .vtp files based on ID
    vtp_files = glob.glob(os.path.join(output_dir, "processed_*.vtp"))
    train_count = 0
    val_count = 0

    for file_path in vtp_files:
        filename = os.path.basename(file_path)
        # Extract ID from filename (remove 'processed_' prefix and '.vtp' suffix)
        id_ = filename[10:-4]  # len('processed_') = 10, len('.vtp') = 4

        if id_ in val_ids:
            shutil.move(file_path, os.path.join(val_dir, filename))
            val_count += 1
        else:
            shutil.move(file_path, os.path.join(train_dir, filename))
            train_count += 1

    # Step 4: Move stats files
    stats_files = [
        "wallShearStressMeanTrim_mean.npy",
        "wallShearStressMeanTrim_std.npy",
        "pMeanTrim_mean.npy",
        "pMeanTrim_std.npy",
        "Area_mean.npy",
        "Area_std.npy",
    ]

    for stats_file in stats_files:
        src_path = os.path.join(output_dir, stats_file)
        if os.path.exists(src_path):
            shutil.move(src_path, os.path.join(stats_dir, stats_file))

    logging.info(f"Data splitting complete:")
    logging.info(f"   - Training files: {train_count}")
    logging.info(f"   - Validation files: {val_count}")
    logging.info(f"   - Stats files moved to {stats_dir}")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Initialize PhysicsNeMo logging
    LaunchLogger.initialize()

    os.chdir(hydra.utils.get_original_cwd())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = cfg.preprocessed_data_dir
    os.makedirs(output_dir, exist_ok=True)

    # Create logger instance
    logger = LaunchLogger("Preprocessor")

    logging.info("Starting file discovery...")

    xmgn_files = glob.glob(
        os.path.join(
            cfg.xmgn_data_dir, f"{cfg.xmgn_filename_prefix}*{cfg.xmgn_filename_suffix}"
        )
    )
    fignet_files = glob.glob(
        os.path.join(
            cfg.fignet_data_dir,
            f"{cfg.fignet_filename_prefix}*{cfg.fignet_filename_suffix}",
        )
    )
    domino_files = glob.glob(
        os.path.join(
            cfg.domino_data_dir,
            f"{cfg.domino_filename_prefix}*{cfg.domino_filename_suffix}",
        )
    )

    get_id = lambda path, prefix, suffix: os.path.basename(path)[
        len(prefix) : -len(suffix)
    ]
    xmgn_ids = {
        get_id(f, cfg.xmgn_filename_prefix, cfg.xmgn_filename_suffix)
        for f in xmgn_files
    }
    fignet_ids = {
        get_id(f, cfg.fignet_filename_prefix, cfg.fignet_filename_suffix)
        for f in fignet_files
    }
    domino_ids = {
        get_id(f, cfg.domino_filename_prefix, cfg.domino_filename_suffix)
        for f in domino_files
    }

    common_ids = sorted(xmgn_ids & fignet_ids & domino_ids)

    logging.info(f"File discovery complete. Found {len(common_ids)} common files.")

    # Step 1: Read validation IDs first
    val_ids = set()
    if os.path.exists(cfg.validation_ids_csv):
        with open(cfg.validation_ids_csv, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if row:  # Check if row is not empty
                    val_ids.add(row[0])
        logging.info(f"Found {len(val_ids)} validation IDs")
    else:
        logging.warning(
            f"Validation IDs CSV file {cfg.validation_ids_csv} not found. All data will be used for training."
        )

    # Step 2: Separate train and validation IDs
    train_ids = [id_ for id_ in common_ids if id_ not in val_ids]
    validation_ids = [id_ for id_ in common_ids if id_ in val_ids]

    logging.info(f"Training samples: {len(train_ids)}")
    logging.info(f"Validation samples: {len(validation_ids)}")

    args_list = []
    for id_ in common_ids:
        xmgn_path = os.path.join(
            cfg.xmgn_data_dir,
            f"{cfg.xmgn_filename_prefix}{id_}{cfg.xmgn_filename_suffix}",
        )
        fignet_path = os.path.join(
            cfg.fignet_data_dir,
            f"{cfg.fignet_filename_prefix}{id_}{cfg.fignet_filename_suffix}",
        )
        domino_path = os.path.join(
            cfg.domino_data_dir,
            f"{cfg.domino_filename_prefix}{id_}{cfg.domino_filename_suffix}",
        )
        args_list.append((id_, xmgn_path, fignet_path, domino_path, output_dir))

    logging.info(f"Starting parallel processing with {os.cpu_count()} workers...")

    # Step 3: Process training data first (for stats computation)
    pMeanTrim_all = []
    wallShearStressMeanTrim_all = []
    area_all = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for pMeanTrim, wallShear, area in tqdm(
            executor.map(process_and_save_vtp, args_list),
            total=len(args_list),
        ):
            pMeanTrim_all.append(pMeanTrim)
            wallShearStressMeanTrim_all.append(wallShear)
            area_all.append(area)

    logging.info(f"Parallel processing complete. Processed {len(args_list)} files.")
    logging.info("Starting statistics computation...")

    # Concatenate all values and compute stats (same as original)
    pMeanTrim_all = np.concatenate(pMeanTrim_all)  # (total_points,)
    wallShearStressMeanTrim_all = np.vstack(
        wallShearStressMeanTrim_all
    )  # (total_points, 3)
    area_all = np.concatenate(area_all)  # (total_cells,)

    pMeanTrim_mean = np.mean(pMeanTrim_all)
    pMeanTrim_std = np.std(pMeanTrim_all)
    wallShear_mean = np.mean(wallShearStressMeanTrim_all, axis=0)  # (3,)
    wallShear_std = np.std(wallShearStressMeanTrim_all, axis=0)  # (3,)
    area_mean = np.mean(area_all)
    area_std = np.std(area_all)

    logging.info("Statistics computed.")

    # Log the computed statistics
    logger.log_epoch(
        {
            "pMeanTrim_mean": pMeanTrim_mean,
            "pMeanTrim_std": pMeanTrim_std,
            "wallShear_mean_x": wallShear_mean[0],
            "wallShear_mean_y": wallShear_mean[1],
            "wallShear_mean_z": wallShear_mean[2],
            "wallShear_std_x": wallShear_std[0],
            "wallShear_std_y": wallShear_std[1],
            "wallShear_std_z": wallShear_std[2],
            "area_mean": area_mean,
            "area_std": area_std,
            "total_points": len(pMeanTrim_all),
            "total_cells": len(area_all),
        }
    )

    np.save(os.path.join(output_dir, "pMeanTrim_mean.npy"), pMeanTrim_mean)
    np.save(os.path.join(output_dir, "pMeanTrim_std.npy"), pMeanTrim_std)
    np.save(
        os.path.join(output_dir, "wallShearStressMeanTrim_mean.npy"), wallShear_mean
    )
    np.save(os.path.join(output_dir, "wallShearStressMeanTrim_std.npy"), wallShear_std)
    np.save(os.path.join(output_dir, "Area_mean.npy"), area_mean)
    np.save(os.path.join(output_dir, "Area_std.npy"), area_std)

    logging.info("Statistics saved to files.")
    logging.info("Starting data splitting...")

    # Step 4: Process validation data (no stats collection)
    if validation_ids:
        val_args_list = []
        for id_ in validation_ids:
            xmgn_path = os.path.join(
                cfg.xmgn_data_dir,
                f"{cfg.xmgn_filename_prefix}{id_}{cfg.xmgn_filename_suffix}",
            )
            fignet_path = os.path.join(
                cfg.fignet_data_dir,
                f"{cfg.fignet_filename_prefix}{id_}{cfg.fignet_filename_suffix}",
            )
            domino_path = os.path.join(
                cfg.domino_data_dir,
                f"{cfg.domino_filename_prefix}{id_}{cfg.domino_filename_suffix}",
            )
            val_args_list.append((id_, xmgn_path, fignet_path, domino_path, output_dir))

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            for _ in tqdm(
                executor.map(process_and_save_vtp, val_args_list),
                total=len(val_args_list),
            ):
                pass  # No stats collection for validation data

    # Split data into train/val directories
    split_data(output_dir, cfg.validation_ids_csv)

    logging.info("Data splitting complete.")


if __name__ == "__main__":
    main()
