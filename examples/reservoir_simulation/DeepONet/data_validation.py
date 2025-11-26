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
Data validation utilities for CO2 sequestration models.

This module provides dynamic validation functions to ensure data dimensions
are correctly formatted before training or evaluation.
"""

import torch
from typing import Tuple, Optional


def validate_batch_dimensions(
    inputs: torch.Tensor, targets: torch.Tensor, variable: str = "unknown"
) -> Tuple[int, Tuple[int, int, int], int]:
    """
    Validate dimensions of a batch of training/evaluation data.

    Args:
        inputs: Input tensor, expected shape (batch, H, W, T, channels)
        targets: Target tensor, expected shape (batch, H, W, T)
        variable: Name of the variable being predicted (e.g., 'pressure', 'saturation')

    Returns:
        Tuple of (batch_size, spatial_dims, num_channels)

    Raises:
        ValueError: If data dimensions are invalid or inconsistent
    """
    # Check input dimensions: must be 5D (batch, H, W, T, channels)
    if len(inputs.shape) != 5:
        raise ValueError(
            f"❌ Invalid input shape! Expected 5D tensor (batch, H, W, T, channels), "
            f"got {len(inputs.shape)}D tensor with shape {inputs.shape}. "
            f"Input data must be arranged as (batch, height, width, time, channels)."
        )

    # Extract dimensions
    batch_size = inputs.shape[0]
    spatial_dims = tuple(inputs.shape[1:4])  # H, W, T
    num_channels = inputs.shape[4]

    # Check output dimensions: must be 4D (batch, H, W, T)
    if len(targets.shape) != 4:
        raise ValueError(
            f"❌ Invalid target shape! Expected 4D tensor (batch, H, W, T), "
            f"got {len(targets.shape)}D tensor with shape {targets.shape}. "
            f"Target data must be arranged as (batch, height, width, time)."
        )

    # Check that spatial dimensions match between input and output
    target_spatial_dims = tuple(targets.shape[1:4])
    if target_spatial_dims != spatial_dims:
        raise ValueError(
            f"❌ Spatial dimension mismatch! "
            f"Input spatial dims: {spatial_dims} (H, W, T), "
            f"Target spatial dims: {target_spatial_dims} (H, W, T). "
            f"Input and target must have the same spatial dimensions."
        )

    # Check that batch dimensions match
    if targets.shape[0] != batch_size:
        raise ValueError(
            f"❌ Batch size mismatch! "
            f"Input batch size: {batch_size}, "
            f"Target batch size: {targets.shape[0]}. "
            f"Batch dimensions must match."
        )

    # Sanity checks on dimension values
    if spatial_dims[0] < 1 or spatial_dims[1] < 1 or spatial_dims[2] < 1:
        raise ValueError(
            f"❌ Invalid spatial dimensions: {spatial_dims}. "
            f"All spatial dimensions (H, W, T) must be positive integers."
        )

    if num_channels < 1:
        raise ValueError(
            f"❌ Invalid number of channels: {num_channels}. "
            f"Must have at least 1 input channel."
        )

    return batch_size, spatial_dims, num_channels


def validate_sample_dimensions(
    input_sample: torch.Tensor, target_sample: torch.Tensor, variable: str = "unknown"
) -> Tuple[Tuple[int, int, int], int]:
    """
    Validate dimensions of a single sample (for evaluation).

    Args:
        input_sample: Input tensor, expected shape (H, W, T, channels)
        target_sample: Target tensor, expected shape (H, W, T)
        variable: Name of the variable being predicted

    Returns:
        Tuple of (spatial_dims, num_channels)

    Raises:
        ValueError: If data dimensions are invalid or inconsistent
    """
    # Check input dimensions: must be 4D (H, W, T, channels)
    if len(input_sample.shape) != 4:
        raise ValueError(
            f"❌ Invalid input shape! Expected 4D tensor (H, W, T, channels), "
            f"got {len(input_sample.shape)}D tensor with shape {input_sample.shape}. "
            f"Input data must be arranged as (height, width, time, channels)."
        )

    # Extract dimensions
    spatial_dims = tuple(input_sample.shape[0:3])  # H, W, T
    num_channels = input_sample.shape[3]

    # Check output dimensions: must be 3D (H, W, T)
    if len(target_sample.shape) != 3:
        raise ValueError(
            f"❌ Invalid target shape! Expected 3D tensor (H, W, T), "
            f"got {len(target_sample.shape)}D tensor with shape {target_sample.shape}. "
            f"Target data must be arranged as (height, width, time)."
        )

    # Check that spatial dimensions match between input and output
    target_spatial_dims = tuple(target_sample.shape)
    if target_spatial_dims != spatial_dims:
        raise ValueError(
            f"❌ Spatial dimension mismatch! "
            f"Input spatial dims: {spatial_dims} (H, W, T), "
            f"Target spatial dims: {target_spatial_dims} (H, W, T). "
            f"Input and target must have the same spatial dimensions."
        )

    # Sanity checks on dimension values
    if spatial_dims[0] < 1 or spatial_dims[1] < 1 or spatial_dims[2] < 1:
        raise ValueError(
            f"❌ Invalid spatial dimensions: {spatial_dims}. "
            f"All spatial dimensions (H, W, T) must be positive integers."
        )

    if num_channels < 1:
        raise ValueError(
            f"❌ Invalid number of channels: {num_channels}. "
            f"Must have at least 1 input channel."
        )

    return spatial_dims, num_channels


def print_validation_summary(
    input_shape: Tuple,
    target_shape: Tuple,
    variable: str,
    is_batch: bool = True,
    logger: Optional[any] = None,
):
    """
    Print a formatted summary of validation results.

    Args:
        input_shape: Shape of input tensor
        target_shape: Shape of target tensor
        variable: Name of the variable being predicted
        is_batch: Whether shapes include batch dimension
        logger: Optional logger object (if None, uses print)
    """
    log_func = logger.success if logger else print
    info_func = logger.info if logger else print

    log_func("✅ Data validation passed!")

    if is_batch:
        batch_size = input_shape[0]
        spatial_dims = input_shape[1:4]
        num_channels = input_shape[4]

        info_func(f"   Input shape: {input_shape} → (batch, H, W, T, channels)")
        info_func(f"   Target shape: {target_shape} → (batch, H, W, T)")
        info_func(f"   Batch size: {batch_size}")
    else:
        spatial_dims = input_shape[0:3]
        num_channels = input_shape[3]

        info_func(f"   Input shape: {input_shape} → (H, W, T, channels)")
        info_func(f"   Target shape: {target_shape} → (H, W, T)")

    info_func(
        f"   Spatial dimensions: {spatial_dims[0]}×{spatial_dims[1]}×{spatial_dims[2]} (H×W×T)"
    )
    info_func(f"   Input channels: {num_channels}")
    info_func(f"   Variable: {variable}")
