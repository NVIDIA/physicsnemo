# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This file provides utilities to compute normalization statistics (mean, std, min, max)
for a given field in a dataset, typically used for preprocessing in CFD workflows.
"""

from pathlib import Path
import time

import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from physicsnemo.datapipes.cae.cae_dataset import CAEDataset


def compute_mean_std_min_max(
    dataset: CAEDataset,
    field_key: str,
    max_samples: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the mean, standard deviation, minimum, and maximum for a specified field
    across all samples in a dataset.

    Uses a numerically stable online algorithm for mean and variance.

    Args:
        dataset (CAEDataset): The dataset to process.
        field_key (str): The key for the field to normalize.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            mean, std, min, max tensors for the field.
    """
    N = torch.tensor(
        0, dtype=torch.int64, device="cpu"
    )  # Total number of elements processed
    mean = None
    M2 = None  # Sum of squares of differences from the current mean
    min_val = None
    max_val = None

    time_start = time.time()
    for i in range(len(dataset)):
        print(f"reading file: {i}")
        data = dataset[i][field_key]
        if mean is None:
            # Initialize accumulators based on the shape of the data
            mean = torch.zeros(data.shape[-1], device=data.device)
            M2 = torch.zeros(data.shape[-1], device=data.device)
            min_val = torch.full((data.shape[-1],), float("inf"), device=data.device)
            max_val = torch.full((data.shape[-1],), float("-inf"), device=data.device)
        n = data.shape[0]
        N += n

        # Compute batch statistics
        batch_mean = data.mean(axis=(0,))
        batch_M2 = ((data - batch_mean) ** 2).sum(axis=(0,))
        batch_n = data.shape[0]

        # Update min/max
        batch_min = data.amin(dim=(0,))
        batch_max = data.amax(dim=(0,))
        min_val = torch.minimum(min_val, batch_min)
        max_val = torch.maximum(max_val, batch_max)

        # Update running mean and M2 (Welford's algorithm)
        delta = batch_mean - mean
        mean = mean + delta * (batch_n / N)
        M2 = M2 + batch_M2 + delta**2 * (batch_n * N) / N
        time_end = time.time()
        print(f"Time taken for file {i}: {time_end - time_start:.2f} seconds")
        time_start = time.time()
        if i >= max_samples:
            break

    var = M2 / (N - 1)
    std = torch.sqrt(var)
    return mean, std, min_val, max_val


@hydra.main(version_base="1.3", config_path="conf", config_name="train_surface")
def main(cfg: DictConfig) -> None:
    """
    Script entry point for computing normalization statistics for a specified field
    in a dataset, using configuration from a YAML file.

    The computed statistics are printed and saved to a .npz file.
    """

    # Choose which field to normalize (can be overridden via command line)
    field_key: str = cfg.data.mode + "_fields"

    # Normalization directory can be configured (backward compatible: defaults to current directory)
    normalization_dir: str = getattr(cfg.data, "normalization_dir", ".")

    # Construct full path using pathlib (cross-platform, concise)
    workspace_path: str = str(
        Path(normalization_dir) / f"{field_key}_normalization.npz"
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create the dataset using configuration parameters
    dataset = CAEDataset(
        data_dir=cfg.data.train.data_path,
        keys_to_read=[
            field_key,
        ],
        keys_to_read_if_available={},
        output_device=device,
        preload_depth=cfg.data.preload_depth,
        pin_memory=cfg.data.pin_memory,
    )
    # Compute normalization statistics
    mean, std, min_val, max_val = compute_mean_std_min_max(dataset, field_key, 100)
    print(f"Mean for {field_key}: {mean}")
    print(f"Std for {field_key}: {std}")
    print(f"Min for {field_key}: {min_val}")
    print(f"Max for {field_key}: {max_val}")

    # Save statistics to configured workspace path
    print(f"Saving normalization statistics to: {workspace_path}")
    np.savez(
        workspace_path,
        mean=mean.cpu().numpy(),
        std=std.cpu().numpy(),
        min=min_val.cpu().numpy(),
        max=max_val.cpu().numpy(),
    )
    print(f"Successfully saved normalization file: {workspace_path}")


if __name__ == "__main__":
    main()
