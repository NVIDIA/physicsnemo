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
This module provides functionality to convert STL files with multiple solids
to another STL file with a single combined solid. It includes support for
processing multiple files in parallel with progress tracking.
"""

import os
import trimesh
import hydra

from multiprocessing import Pool
from tqdm import tqdm
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


def process_stl_file(task):
    stl_path = task

    # Load the STL file using trimesh
    mesh = trimesh.load_mesh(stl_path)

    # If the STL file contains multiple solids (as a Scene object)
    if isinstance(mesh, trimesh.Scene):
        # Extract all geometries (solids) from the scene
        meshes = list(mesh.geometry.values())

        # Combine all the solids into a single mesh
        combined_mesh = trimesh.util.concatenate(meshes)
    else:
        # If it's a single solid, no need to combine
        combined_mesh = mesh

    # Prepare the output file path (next to the original file)
    base_name, ext = os.path.splitext(stl_path)
    output_file_path = to_absolute_path(f"{base_name}_single_solid{ext}")

    # Save the new combined mesh as an STL file
    combined_mesh.export(output_file_path)

    return f"Processed: {stl_path} -> {output_file_path}"


def process_directory(data_path, num_workers=16):
    """Process all STL files in the given directory using multiprocessing with progress tracking."""
    tasks = []
    for root, _, files in os.walk(data_path):
        stl_files = [f for f in files if f.endswith(".stl")]
        for stl_file in stl_files:
            stl_path = os.path.join(root, stl_file)

            # Add the STL file to the tasks list (no need for output dir, saving next to the original)
            tasks.append(stl_path)

    # Use multiprocessing to process the tasks with progress tracking
    with Pool(num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_stl_file, tasks),
            total=len(tasks),
            desc="Processing STL Files",
            unit="file",
        ):
            pass


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Process the directory with multiple STL files
    process_directory(
        to_absolute_path(cfg.data_path), num_workers=cfg.num_preprocess_workers
    )


if __name__ == "__main__":
    main()
