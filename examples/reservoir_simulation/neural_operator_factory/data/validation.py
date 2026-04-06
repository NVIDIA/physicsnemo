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
Data validation utilities for reservoir simulation models.

Supports both 3D (2D spatial + time) and 4D (3D spatial + time) datasets:
- 3D: Input (B, H, W, T, C), Output (B, H, W, T)
- 4D: Input (B, X, Y, Z, T, C), Output (B, X, Y, Z, T)
"""

from typing import Dict, Optional, Tuple

import torch


def detect_dimensions(inputs: torch.Tensor) -> str:
    """
    Detect data dimensions from input tensor.

    Parameters
    ----------
    inputs : torch.Tensor
        Input tensor (batched)

    Returns
    -------
    str
        '3d' for (B, H, W, T, C) or '4d' for (B, X, Y, Z, T, C)
    """
    ndim = inputs.dim()
    if ndim == 5:
        return "3d"
    elif ndim == 6:
        return "4d"
    else:
        raise ValueError(
            f"Cannot detect dimensions from {ndim}D tensor. "
            f"Expected 5D (3d) or 6D (4d), got shape {tuple(inputs.shape)}"
        )


def validate_batch_dimensions(
    inputs: torch.Tensor, targets: torch.Tensor, variable: str = "unknown"
) -> Dict:
    """
    Validate dimensions of a batch of training/evaluation data.

    Automatically detects 3D or 4D data format.

    Parameters
    ----------
    inputs : torch.Tensor
        Input tensor
        - 3D: (batch, H, W, T, channels)
        - 4D: (batch, X, Y, Z, T, channels)
    targets : torch.Tensor
        Target tensor
        - 3D: (batch, H, W, T)
        - 4D: (batch, X, Y, Z, T)
    variable : str
        Name of the variable being predicted

    Returns
    -------
    Dict
        Dictionary with keys:
        - dimensions: '3d' or '4d'
        - batch_size: int
        - spatial_shape: tuple
        - time_steps: int
        - num_channels: int

    Raises
    ------
    ValueError
        If data dimensions are invalid or inconsistent
    """
    input_ndim = inputs.dim()
    output_ndim = targets.dim()

    # Determine expected format based on input dimensions
    if input_ndim == 5:
        # 3D format: (B, H, W, T, C)
        dimensions = "3d"
        expected_output_ndim = 4
        dim_names = "H, W, T"
        spatial_slice = slice(1, 3)
        time_idx = 3
        channel_idx = 4

    elif input_ndim == 6:
        # 4D format: (B, X, Y, Z, T, C)
        dimensions = "4d"
        expected_output_ndim = 5
        dim_names = "X, Y, Z, T"
        spatial_slice = slice(1, 4)
        time_idx = 4
        channel_idx = 5

    else:
        raise ValueError(
            f"❌ Invalid input shape! Expected 5D (B, H, W, T, C) for 3D data "
            f"or 6D (B, X, Y, Z, T, C) for 4D data, "
            f"got {input_ndim}D tensor with shape {tuple(inputs.shape)}."
        )

    # Validate output dimensions
    if output_ndim != expected_output_ndim:
        raise ValueError(
            f"❌ Invalid target shape for {dimensions.upper()} data! "
            f"Expected {expected_output_ndim}D tensor (B, {dim_names}), "
            f"got {output_ndim}D tensor with shape {tuple(targets.shape)}."
        )

    # Extract dimensions
    batch_size = inputs.shape[0]
    spatial_shape = tuple(inputs.shape[spatial_slice])
    time_steps = inputs.shape[time_idx]
    num_channels = inputs.shape[channel_idx]

    # Validate batch size match
    if targets.shape[0] != batch_size:
        raise ValueError(
            f"❌ Batch size mismatch! Input: {batch_size}, Target: {targets.shape[0]}"
        )

    # Validate spatial+temporal dimensions match
    input_spatiotemporal = tuple(
        inputs.shape[1:-1]
    )  # All dims except batch and channels
    target_spatiotemporal = tuple(targets.shape[1:])  # All dims except batch

    if input_spatiotemporal != target_spatiotemporal:
        raise ValueError(
            f"❌ Spatial/temporal dimension mismatch! "
            f"Input: {input_spatiotemporal}, Target: {target_spatiotemporal}"
        )

    # Sanity checks
    if any(d < 1 for d in spatial_shape):
        raise ValueError(f"❌ Invalid spatial dimensions: {spatial_shape}")
    if time_steps < 1:
        raise ValueError(f"❌ Invalid time steps: {time_steps}")
    if num_channels < 1:
        raise ValueError(f"❌ Invalid number of channels: {num_channels}")

    return {
        "dimensions": dimensions,
        "batch_size": batch_size,
        "spatial_shape": spatial_shape,
        "time_steps": time_steps,
        "num_channels": num_channels,
    }


def validate_sample_dimensions(
    input_sample: torch.Tensor, target_sample: torch.Tensor, variable: str = "unknown"
) -> Dict:
    """
    Validate dimensions of a single sample (unbatched).

    Parameters
    ----------
    input_sample : torch.Tensor
        Input tensor (single sample)
        - 3D: (H, W, T, channels)
        - 4D: (X, Y, Z, T, channels)
    target_sample : torch.Tensor
        Target tensor (single sample)
        - 3D: (H, W, T)
        - 4D: (X, Y, Z, T)
    variable : str
        Name of the variable being predicted

    Returns
    -------
    Dict
        Dictionary with keys: dimensions, spatial_shape, time_steps, num_channels
    """
    input_ndim = input_sample.dim()
    output_ndim = target_sample.dim()

    if input_ndim == 4:
        # 3D: (H, W, T, C)
        dimensions = "3d"
        expected_output_ndim = 3
        dim_names = "H, W, T"
        spatial_slice = slice(0, 2)
        time_idx = 2
        channel_idx = 3

    elif input_ndim == 5:
        # 4D: (X, Y, Z, T, C)
        dimensions = "4d"
        expected_output_ndim = 4
        dim_names = "X, Y, Z, T"
        spatial_slice = slice(0, 3)
        time_idx = 3
        channel_idx = 4

    else:
        raise ValueError(
            f"❌ Invalid input shape! Expected 4D (H, W, T, C) for 3D data "
            f"or 5D (X, Y, Z, T, C) for 4D data, "
            f"got {input_ndim}D tensor with shape {tuple(input_sample.shape)}."
        )

    if output_ndim != expected_output_ndim:
        raise ValueError(
            f"❌ Invalid target shape for {dimensions.upper()} data! "
            f"Expected {expected_output_ndim}D tensor ({dim_names}), "
            f"got {output_ndim}D tensor with shape {tuple(target_sample.shape)}."
        )

    spatial_shape = tuple(input_sample.shape[spatial_slice])
    time_steps = input_sample.shape[time_idx]
    num_channels = input_sample.shape[channel_idx]

    # Validate spatial+temporal match
    input_spatiotemporal = tuple(input_sample.shape[:-1])
    target_spatiotemporal = tuple(target_sample.shape)

    if input_spatiotemporal != target_spatiotemporal:
        raise ValueError(
            f"❌ Spatial/temporal dimension mismatch! "
            f"Input: {input_spatiotemporal}, Target: {target_spatiotemporal}"
        )

    return {
        "dimensions": dimensions,
        "spatial_shape": spatial_shape,
        "time_steps": time_steps,
        "num_channels": num_channels,
    }


def print_validation_summary(
    input_shape: Tuple,
    target_shape: Tuple,
    variable: str,
    is_batch: bool = True,
    logger: Optional[object] = None,
):
    """
    Print a formatted summary of validation results.

    Automatically detects 3D or 4D format from shapes.

    Parameters
    ----------
    input_shape : Tuple
        Shape of input tensor
    target_shape : Tuple
        Shape of target tensor
    variable : str
        Name of the variable being predicted
    is_batch : bool
        Whether shapes include batch dimension
    logger : object, optional
        Logger object with .success() and .info() methods
    """
    log_func = logger.success if logger else print
    info_func = logger.info if logger else print

    # Detect dimensions
    if is_batch:
        is_4d = len(input_shape) == 6
        batch_size = input_shape[0]
        spatial_start = 1
    else:
        is_4d = len(input_shape) == 5
        batch_size = None
        spatial_start = 0

    if is_4d:
        dim_label = "4D"
        dim_names = ("X", "Y", "Z", "T")
        spatial_end = spatial_start + 3
    else:
        dim_label = "3D"
        dim_names = ("H", "W", "T")
        spatial_end = spatial_start + 2

    spatial_shape = input_shape[spatial_start:spatial_end]
    time_steps = input_shape[spatial_end]
    num_channels = input_shape[-1]

    log_func(f"✅ Data validation passed! ({dim_label})")

    if is_batch:
        spatial_str = " × ".join(
            f"{dim_names[i]}={spatial_shape[i]}" for i in range(len(spatial_shape))
        )
        info_func(
            f"   Input shape: {input_shape} → (batch, {', '.join(dim_names)}, channels)"
        )
        info_func(f"   Target shape: {target_shape} → (batch, {', '.join(dim_names)})")
        info_func(f"   Batch size: {batch_size}")
    else:
        info_func(f"   Input shape: {input_shape} → ({', '.join(dim_names)}, channels)")
        info_func(f"   Target shape: {target_shape} → ({', '.join(dim_names)})")

    spatial_str = " × ".join(str(s) for s in spatial_shape)
    info_func(f"   Spatial dimensions: {spatial_str} ({' × '.join(dim_names[:-1])})")
    info_func(f"   Time steps: {time_steps}")
    info_func(f"   Input channels: {num_channels}")
    info_func(f"   Variable: {variable}")


def get_dimension_info(tensor: torch.Tensor, is_batch: bool = True) -> Dict:
    """
    Extract dimension information from a tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor
    is_batch : bool
        Whether tensor includes batch dimension

    Returns
    -------
    Dict with dimension information
    """
    ndim = tensor.dim()

    if is_batch:
        if ndim == 5:
            return {
                "dimensions": "3d",
                "batch_size": tensor.shape[0],
                "spatial_shape": tuple(tensor.shape[1:3]),
                "time_steps": tensor.shape[3],
                "num_channels": tensor.shape[4],
            }
        elif ndim == 6:
            return {
                "dimensions": "4d",
                "batch_size": tensor.shape[0],
                "spatial_shape": tuple(tensor.shape[1:4]),
                "time_steps": tensor.shape[4],
                "num_channels": tensor.shape[5],
            }
        elif ndim == 4:
            # Output tensor (3D)
            return {
                "dimensions": "3d",
                "batch_size": tensor.shape[0],
                "spatial_shape": tuple(tensor.shape[1:3]),
                "time_steps": tensor.shape[3],
                "num_channels": 1,
            }
        elif ndim == 5:
            # Output tensor (4D)
            return {
                "dimensions": "4d",
                "batch_size": tensor.shape[0],
                "spatial_shape": tuple(tensor.shape[1:4]),
                "time_steps": tensor.shape[4],
                "num_channels": 1,
            }
    else:
        if ndim == 4:
            return {
                "dimensions": "3d",
                "spatial_shape": tuple(tensor.shape[0:2]),
                "time_steps": tensor.shape[2],
                "num_channels": tensor.shape[3],
            }
        elif ndim == 5:
            return {
                "dimensions": "4d",
                "spatial_shape": tuple(tensor.shape[0:3]),
                "time_steps": tensor.shape[3],
                "num_channels": tensor.shape[4],
            }

    raise ValueError(f"Cannot extract dimension info from {ndim}D tensor")
