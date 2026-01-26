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

"""
Tutorial 4: Hydra Configuration for DataPipes
==============================================

This tutorial demonstrates how to configure PhysicsNeMo DataPipes entirely
through Hydra YAML files with minimal Python code.

The key insight: Hydra's `instantiate()` can build the entire datapipe
(reader, transforms, dataset, dataloader) from configuration alone using
recursive instantiation.

Prerequisites
-------------
Generate synthetic data before running:

    # For point cloud data:
    python generate_variable_points_data.py -n 100 -s "coords:3 features:8" --min-points 50000 --max-points 100000 -b zarr -o output/pointcloud_data/

Run this tutorial:

    # Use point cloud configuration (default)
    python tutorial_04_hydra_config.py --config-name tutorial_04_pointcloud

    # Override values from command line
    python tutorial_04_hydra_config.py --config-name tutorial_04_pointcloud \\
        dataloader.batch_size=8

    # Override transform parameters
    python tutorial_04_hydra_config.py --config-name tutorial_04_pointcloud \\
        subsample.n_points=5000

Configuration Files
-------------------
- conf/tutorial_04_pointcloud.yaml  - Point cloud pipeline with subsampling
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from physicsnemo.datapipes import DataLoader


@hydra.main(
    version_base=None,
    config_path="./conf",
    config_name="tutorial_04_pointcloud",
)
def main(cfg: DictConfig):
    """
    Main entry point - demonstrates Hydra-based datapipe configuration.

    The entire pipeline is built from the YAML configuration with a single
    instantiate call that recursively builds DataLoader -> Dataset -> Reader + Transforms.
    """
    print()
    print("=" * 70)
    print("Tutorial 4: Hydra Configuration for DataPipes")
    print("=" * 70)
    print()

    # Show the resolved configuration
    print("Resolved Configuration:")
    print("-" * 70)
    print(OmegaConf.to_yaml(cfg))
    print("-" * 70)
    print()

    # Build entire datapipe from config with a single instantiate call
    # Hydra recursively instantiates: DataLoader -> Dataset -> Reader + Transforms
    print("Building datapipe from configuration (single instantiate call)...")
    dataloader: DataLoader = hydra.utils.instantiate(cfg.dataloader)
    dataset = dataloader.dataset

    print(f"  Reader: {dataset.reader}")
    # Handle different transform configurations
    if dataset.transforms is None:
        transform_names = []
    elif hasattr(dataset.transforms, "transforms"):
        # Compose wraps multiple transforms
        transform_names = [type(t).__name__ for t in dataset.transforms.transforms]
    else:
        # Single transform
        transform_names = [type(dataset.transforms).__name__]
    print(f"  Transforms: {transform_names}")
    print(f"  Dataset: {len(dataset)} samples")
    print(
        f"  DataLoader: {len(dataloader)} batches (batch_size={cfg.dataloader.batch_size})"
    )
    print()

    # Run training loop
    print("Training Loop:")
    print("-" * 70)

    num_epochs = cfg.training.get("num_epochs", 2)
    log_interval = cfg.training.get("log_interval", 1)

    for epoch in range(num_epochs):
        for batch_idx, batch_data in enumerate(dataloader):
            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}:")
                for key in batch_data.keys():
                    shape = tuple(batch_data[key].shape)
                    print(f"    {key}: {shape} on device {batch_data[key].device}")

        print(f"Epoch {epoch} complete: {len(dataloader)} batches")

    print("-" * 70)
    print()

    # Cleanup
    dataset.close()

    # Print summary
    print("=" * 70)
    print("Tutorial 4 Complete!")
    print()
    print("Key takeaways:")
    print("  1. Define datapipes entirely in YAML configuration")
    print("  2. Use a single hydra.utils.instantiate() call to build everything")
    print(
        "  3. Hydra recursively instantiates: DataLoader -> Dataset -> Reader + Transforms"
    )
    print("  4. Override any parameter from command line:")
    print("       python tutorial_04_hydra_config.py dataloader.batch_size=8")
    print("  5. Override transform parameters (using top-level keys from defaults):")
    print("       python tutorial_04_hydra_config.py subsample.n_points=5000")
    print("=" * 70)


if __name__ == "__main__":
    main()
