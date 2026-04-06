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

"""
Scalar channel detection utilities for MIONet.

This module provides utilities to automatically detect which input channels
contain constant (scalar) values that should be processed separately by
the MIONet's second branch.
"""

from functools import partial
from typing import Dict, List, Optional, Tuple

import torch


def detect_scalar_channels(
    sample_input: torch.Tensor,
    threshold: float = 1e-6,
    num_samples_to_check: int = 1,
) -> Dict:
    """
    Detect which channels contain constant (scalar) values.

    A channel is considered "scalar" if it contains the same value at every
    spatial and temporal location. This is detected using multiple checks:
    1. Standard deviation < threshold
    2. Min value equals max value
    3. All values are close to the first value

    Args:
        sample_input: Single sample tensor of shape (H, W, T, C) or batch (B, H, W, T, C)
        threshold: Tolerance for detecting constants (default: 1e-6)
        num_samples_to_check: Number of samples to verify consistency (if batch provided)

    Returns:
        Dictionary containing:
            - 'scalar_indices': List of channel indices that are constant
            - 'spatial_indices': List of channel indices that vary spatially
            - 'scalar_values': Tensor of shape (C_scalar,) with the constant value for each scalar channel
            - 'num_scalar_channels': Number of scalar channels detected
            - 'num_spatial_channels': Number of spatial channels detected

    Example:
        >>> sample = dataset[0][0]  # Get first input sample
        >>> result = detect_scalar_channels(sample)
        >>> print(f"Scalar channels: {result['scalar_indices']}")
        >>> print(f"Spatial channels: {result['spatial_indices']}")
    """
    # Handle batch dimension if present
    if sample_input.dim() == 5:
        # (B, H, W, T, C) -> use first sample
        sample_input = sample_input[0]

    if sample_input.dim() != 4:
        raise ValueError(
            f"Expected input shape (H, W, T, C) or (B, H, W, T, C), "
            f"got shape {sample_input.shape}"
        )

    num_channels = sample_input.shape[-1]
    scalar_indices = []
    spatial_indices = []
    scalar_values = []

    for c in range(num_channels):
        channel_data = sample_input[..., c]  # (H, W, T)

        # Flatten for easier analysis
        flat_data = channel_data.flatten()
        first_value = flat_data[0]

        # Multiple checks for robustness:
        # 1. Standard deviation check
        std_check = channel_data.std().item() < threshold

        # 2. Min equals max check
        min_val = channel_data.min().item()
        max_val = channel_data.max().item()
        minmax_check = abs(max_val - min_val) < threshold

        # 3. All values close to first value
        allclose_check = torch.allclose(
            channel_data,
            torch.full_like(channel_data, first_value.item()),
            atol=threshold,
            rtol=0,
        )

        # 4. Unique values check (should be 1 for scalar)
        unique_check = len(torch.unique(flat_data)) == 1

        # Channel is scalar if ALL checks pass
        is_constant = std_check and minmax_check and allclose_check and unique_check

        if is_constant:
            scalar_indices.append(c)
            scalar_values.append(first_value.item())
        else:
            spatial_indices.append(c)

    # Convert scalar values to tensor
    scalar_values_tensor = torch.tensor(scalar_values, dtype=sample_input.dtype)

    return {
        "scalar_indices": scalar_indices,
        "spatial_indices": spatial_indices,
        "scalar_values": scalar_values_tensor,
        "num_scalar_channels": len(scalar_indices),
        "num_spatial_channels": len(spatial_indices),
    }


def verify_scalar_consistency(
    dataset,
    scalar_indices: List[int],
    num_samples: int = 10,
    threshold: float = 1e-6,
) -> Tuple[bool, Optional[str]]:
    """
    Verify that detected scalar channels are consistent across multiple samples.

    This checks that the same channels are scalar across different samples
    in the dataset (though the scalar values may differ per sample).

    Args:
        dataset: PyTorch dataset to check
        scalar_indices: List of channel indices detected as scalar
        num_samples: Number of samples to check
        threshold: Tolerance for scalar detection

    Returns:
        Tuple of (is_consistent, error_message)
        - is_consistent: True if all samples have same scalar channels
        - error_message: None if consistent, otherwise describes the inconsistency
    """
    num_to_check = min(num_samples, len(dataset))

    for i in range(num_to_check):
        sample_input, _ = dataset[i]
        result = detect_scalar_channels(sample_input, threshold=threshold)

        if set(result["scalar_indices"]) != set(scalar_indices):
            return False, (
                f"Inconsistent scalar channels at sample {i}. "
                f"Expected {scalar_indices}, got {result['scalar_indices']}"
            )

    return True, None


def create_mionet_collate_fn(
    scalar_indices: List[int],
    spatial_indices: List[int],
):
    """
    Create a custom collate function for MIONet that separates scalar and spatial inputs.

    Args:
        scalar_indices: List of channel indices that are scalar
        spatial_indices: List of channel indices that are spatial

    Returns:
        Collate function for use with DataLoader

    Example:
        >>> collate_fn = create_mionet_collate_fn([3, 5, 7], [0, 1, 2, 4, 6, 8, 9, 10, 11])
        >>> dataloader = DataLoader(dataset, collate_fn=collate_fn, ...)
    """

    def mionet_collate_fn(batch, scalar_idx, spatial_idx):
        """
        Custom collate that separates scalar and spatial inputs.

        Returns:
            Tuple of (spatial_inputs, scalar_inputs, targets)
            - spatial_inputs: (B, H, W, T, C_spatial) - channels that vary spatially
            - scalar_inputs: (B, C_scalar) - scalar values for each sample
            - targets: (B, H, W, T) - target outputs
        """
        inputs, targets = zip(*batch)
        inputs = torch.stack(inputs)  # (B, H, W, T, C)
        targets = torch.stack(targets)  # (B, H, W, T)

        # Separate spatial and scalar channels
        spatial_inputs = inputs[..., spatial_idx]  # (B, H, W, T, C_spatial)

        # For scalar channels, extract the constant value (take from position [0,0,0])
        # since it's the same everywhere in the spatial/temporal domain
        scalar_inputs = inputs[:, 0, 0, 0, scalar_idx]  # (B, C_scalar)

        return spatial_inputs, scalar_inputs, targets

    # Return partial function with indices bound
    return partial(
        mionet_collate_fn, scalar_idx=scalar_indices, spatial_idx=spatial_indices
    )


def log_scalar_detection_results(
    result: Dict,
    logger=None,
    channel_names: Optional[List[str]] = None,
):
    """
    Log the results of scalar channel detection.

    Args:
        result: Dictionary from detect_scalar_channels()
        logger: Optional logger object (uses print if None)
        channel_names: Optional list of channel names for better logging
    """
    log_fn = logger.info if logger else print
    success_fn = logger.success if logger and hasattr(logger, "success") else log_fn

    log_fn("=" * 60)
    log_fn("SCALAR CHANNEL DETECTION RESULTS")
    log_fn("=" * 60)

    # Scalar channels
    log_fn(f"Scalar channels ({result['num_scalar_channels']}):")
    for i, idx in enumerate(result["scalar_indices"]):
        name = channel_names[idx] if channel_names else f"Channel {idx}"
        value = result["scalar_values"][i].item()
        log_fn(f"  [{idx}] {name}: value = {value:.6f}")

    # Spatial channels
    log_fn(f"Spatial channels ({result['num_spatial_channels']}):")
    for idx in result["spatial_indices"]:
        name = channel_names[idx] if channel_names else f"Channel {idx}"
        log_fn(f"  [{idx}] {name}")

    log_fn("=" * 60)
    success_fn(
        f"✅ Detection complete: {result['num_scalar_channels']} scalar, "
        f"{result['num_spatial_channels']} spatial channels"
    )
