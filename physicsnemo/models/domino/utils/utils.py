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
Utilities for data processing and training with the DoMINO model architecture.

This module provides essential utilities for computational fluid dynamics data processing,
mesh manipulation, field normalization, and geometric computations. It supports both
torch.Tensor operations on either CPU or GPU.
"""

from pathlib import Path
from typing import Any, Sequence

import torch

from physicsnemo.nn.functional import knn


def calculate_center_of_mass(
    centers: torch.Tensor, sizes: torch.Tensor
) -> torch.Tensor:
    """Calculate the center of mass for a collection of elements.

    Computes the volume-weighted centroid of mesh elements, commonly used
    in computational fluid dynamics for mesh analysis and load balancing.

    Args:
        centers: torch.Tensor of shape (n_elements, 3) containing the centroid
            coordinates of each element.
        sizes: torch.Tensor of shape (n_elements,) containing the volume
            or area of each element used as weights.

    Returns:
        torch.Tensor of shape (1, 3) containing the x, y, z coordinates of the center of mass.

    Raises:
        ValueError: If centers and sizes have incompatible shapes.

    Examples:
        >>> import torch
        >>> centers = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        >>> sizes = torch.tensor([1.0, 2.0, 3.0])
        >>> com = calculate_center_of_mass(centers, sizes)
        >>> torch.allclose(com, torch.tensor([[4.0/3.0, 4.0/3.0, 4.0/3.0]]))
        True
    """

    total_weighted_position = torch.einsum("i,ij->j", sizes, centers)
    total_size = torch.sum(sizes)

    return total_weighted_position[None, ...] / total_size


def normalize(
    field: torch.Tensor,
    max_val: float | torch.Tensor | None = None,
    min_val: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize field values to the range [-1, 1].

    Applies min-max normalization to scale field values to a symmetric range
    around zero. This is commonly used in machine learning preprocessing to
    ensure numerical stability and faster convergence.

    Args:
        field: Input field tensor to be normalized.
        max_val: Maximum values for normalization, can be scalar or array.
            If None, computed from the field data.
        min_val: Minimum values for normalization, can be scalar or array.
            If None, computed from the field data.

    Returns:
        Normalized field with values in the range [-1, 1].

    Raises:
        ZeroDivisionError: If max_val equals min_val (zero range).

    Examples:
        >>> import torch
        >>> field = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> normalized = normalize(field, 5.0, 1.0)
        >>> torch.allclose(normalized, torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]))
        True
        >>> # Auto-compute min/max
        >>> normalized_auto = normalize(field)
        >>> torch.allclose(normalized_auto, torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]))
        True
    """

    if max_val is None:
        max_val, _ = field.max(axis=0, keepdim=True)
    if min_val is None:
        min_val, _ = field.min(axis=0, keepdim=True)

    field_range = max_val - min_val
    return 2.0 * (field - min_val) / field_range - 1.0


def unnormalize(
    normalized_field: torch.Tensor,
    max_val: float | torch.Tensor,
    min_val: float | torch.Tensor,
) -> torch.Tensor:
    """Reverse the normalization process to recover original field values.

    Transforms normalized values from the range [-1, 1] back to their original
    physical range using the stored min/max values.

    Args:
        normalized_field: Field values in the normalized range [-1, 1].
        max_val: Maximum values used in the original normalization.
        min_val: Minimum values used in the original normalization.

    Returns:
        Field values restored to their original physical range.

    Examples:
        >>> import torch
        >>> normalized = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        >>> max_val = torch.tensor(5.0)
        >>> min_val = torch.tensor(1.0)
        >>> original = unnormalize(normalized, max_val, min_val)
        >>> torch.allclose(original, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
        True
    """
    field_range = max_val - min_val
    return (normalized_field + 1.0) * field_range * 0.5 + min_val


def standardize(
    field: torch.Tensor,
    mean: float | torch.Tensor | None = None,
    std: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Standardize field values to have zero mean and unit variance.

    Applies z-score normalization to center the data around zero with
    standard deviation of one. This is preferred over min-max normalization
    when the data follows a normal distribution.

    Args:
        field: Input field tensor to be standardized.
        mean: Mean values for standardization. If None, computed from field data.
        std: Standard deviation values for standardization. If None, computed from field data.

    Returns:
        Standardized field with approximately zero mean and unit variance.

    Raises:
        ZeroDivisionError: If std contains zeros.

    Examples:
        >>> import torch
        >>> field = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> mean = torch.tensor(3.0)
        >>> std = torch.sqrt(torch.tensor(2.5))
        >>> standardized = standardize(field, mean, std)
        >>> torch.allclose(standardized, torch.tensor([-1.265, -0.632, 0.0, 0.632, 1.265]), atol=1e-3)
        True
        >>> # Auto-compute mean/std
        >>> standardized_auto = standardize(field)
        >>> torch.allclose(torch.mean(standardized_auto), torch.tensor(0.0))
        True
        >>> torch.allclose(torch.std(standardized_auto), torch.tensor(1.0))
        True
    """

    if mean is None:
        mean = field.mean(axis=0, keepdim=True)
    if std is None:
        std = field.std(axis=0, keepdim=True)

    return (field - mean) / std


def unstandardize(
    standardized_field: torch.Tensor,
    mean: float | torch.Tensor,
    std: float | torch.Tensor,
) -> torch.Tensor:
    """Reverse the standardization process to recover original field values.

    Transforms standardized values (zero mean, unit variance) back to their
    original distribution using the stored mean and standard deviation.

    Args:
        standardized_field: Field values with zero mean and unit variance.
        mean: Mean values used in the original standardization.
        std: Standard deviation values used in the original standardization.

    Returns:
        Field values restored to their original distribution.

    Examples:
        >>> import torch
        >>> standardized = torch.tensor([-1.265, -0.632, 0.0, 0.632, 1.265])
        >>> mean = torch.tensor(3.0)
        >>> std = torch.sqrt(torch.tensor(2.5))
        >>> original = unstandardize(standardized, mean, std)
        >>> torch.allclose(original, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), atol=1e-3)
        True
    """
    return standardized_field * std + mean


def calculate_normal_positional_encoding(
    coordinates_a: torch.Tensor,
    coordinates_b: torch.Tensor | None = None,
    cell_dimensions: Sequence[float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Calculate sinusoidal positional encoding for 3D coordinates.

    This function computes transformer-style positional encodings for 3D spatial
    coordinates, enabling neural networks to understand spatial relationships.
    The encoding uses sinusoidal functions at different frequencies to create
    unique representations for each spatial position.

    Args:
        coordinates_a: Primary coordinates tensor of shape (n_points, 3).
        coordinates_b: Optional secondary coordinates for computing relative positions.
            If provided, the encoding is computed for (coordinates_a - coordinates_b).
        cell_dimensions: Characteristic length scales for x, y, z dimensions used
            for normalization. Defaults to unit dimensions.

    Returns:
        torch.Tensor of shape (n_points, 12) containing positional encodings with
        4 encoding dimensions per spatial axis (x, y, z).

    Examples:
        >>> import torch
        >>> coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        >>> cell_size = [0.1, 0.1, 0.1]
        >>> encoding = calculate_normal_positional_encoding(coords, cell_dimensions=cell_size)
        >>> encoding.shape
        torch.Size([2, 12])
        >>> # Relative positioning example
        >>> coords_b = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        >>> encoding_rel = calculate_normal_positional_encoding(coords, coords_b, cell_size)
        >>> encoding_rel.shape
        torch.Size([2, 12])
    """
    dx, dy, dz = cell_dimensions[0], cell_dimensions[1], cell_dimensions[2]

    if coordinates_b is not None:
        normals = coordinates_a - coordinates_b
        pos_x = torch.cat(calculate_pos_encoding(normals[:, 0] / dx, d=4), dim=-1)
        pos_y = torch.cat(calculate_pos_encoding(normals[:, 1] / dy, d=4), dim=-1)
        pos_z = torch.cat(calculate_pos_encoding(normals[:, 2] / dz, d=4), dim=-1)
        pos_normals = torch.cat((pos_x, pos_y, pos_z), dim=0).reshape(-1, 12)
    else:
        normals = coordinates_a
        pos_x = torch.cat(calculate_pos_encoding(normals[:, 0] / dx, d=4), dim=-1)
        pos_y = torch.cat(calculate_pos_encoding(normals[:, 1] / dy, d=4), dim=-1)
        pos_z = torch.cat(calculate_pos_encoding(normals[:, 2] / dz, d=4), dim=-1)
        pos_normals = torch.cat((pos_x, pos_y, pos_z), dim=0).reshape(-1, 12)

    return pos_normals


def nd_interpolator(
    coordinates: torch.Tensor, field: torch.Tensor, grid: torch.Tensor, k: int = 2
) -> torch.Tensor:
    """Perform n-dimensional interpolation using k-nearest neighbors.

    This function interpolates field values from scattered points to a regular
    grid using k-nearest neighbor averaging. It's useful for reconstructing
    fields on regular grids from irregular measurement points.

    Args:
        coordinates: torch.Tensor of shape (n_points, n_dims) containing source point coordinates.
        field: torch.Tensor of shape (n_points, n_fields) containing field values at source points.
        grid: torch.Tensor of shape (n_field_points, n_dims) containing target grid points for interpolation.
        k: Number of nearest neighbors to use for interpolation.

    Returns:
        Interpolated field values at grid points using k-nearest neighbor averaging.


    Examples:
        >>> import torch
        >>> # Simple 2D interpolation example
        >>> coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        >>> field_vals = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        >>> grid_points = torch.tensor([[0.5, 0.5]])
        >>> result = nd_interpolator(coords, field_vals, grid_points)
        >>> result.shape[0] == 1  # One grid point
        True
    """
    neighbor_indices, distances = knn(coordinates, grid, k=k)

    field_grid = field[neighbor_indices]
    field_grid = torch.mean(field_grid, dim=1)
    return field_grid


def pad(arr: torch.Tensor, n_points: int, pad_value: float = 0.0) -> torch.Tensor:
    """Pad 2D tensor with constant values to reach target size.

    This function extends a 2D tensor by adding rows filled with a constant
    value. It's commonly used to standardize tensor sizes in batch processing
    for machine learning applications.

    Args:
        arr: Input tensor of shape (n_points, n_features) to be padded.
        n_points: Target number of points (rows) after padding.
        pad_value: Constant value used for padding. Defaults to 0.0.

    Returns:
        Padded tensor of shape (n_points, n_features). If n_points <= arr.shape[0],
        returns the original tensor unchanged.

    Examples:
        >>> import torch
        >>> arr = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> padded = pad(arr, 4, -1.0)
        >>> padded.shape
        torch.Size([4, 2])
        >>> torch.allclose(padded[:2], arr)
        True
        >>> bool(torch.all(padded[2:] == -1.0))
        True
        >>> # No padding needed
        >>> same = pad(arr, 2)
        >>> torch.allclose(same, arr)
        True
    """

    if n_points <= arr.shape[0]:
        return arr

    n_pad = n_points - arr.shape[0]
    arr_padded = torch.nn.functional.pad(
        arr,
        (
            0,
            0,
            0,
            n_pad,
        ),
        mode="constant",
        value=pad_value,
    )
    return arr_padded


def pad_inp(arr: torch.Tensor, n_points: int, pad_value: float = 0.0) -> torch.Tensor:
    """Pad 3D tensor with constant values to reach target size.

    This function extends a 3D tensor by adding entries along the first dimension
    filled with a constant value. Used for standardizing 3D tensor sizes in
    batch processing workflows.

    Args:
        arr: Input tensor of shape (n_points, height, width) to be padded.
        n_points: Target number of points along first dimension after padding.
        pad_value: Constant value used for padding. Defaults to 0.0.

    Returns:
        Padded tensor of shape (n_points, height, width). If n_points <= arr.shape[0],
        returns the original tensor unchanged.

    Examples:
        >>> import torch
        >>> arr = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
        >>> padded = pad_inp(arr, 4, 0.0)
        >>> padded.shape
        torch.Size([4, 1, 2])
        >>> torch.allclose(padded[:2], arr)
        True
        >>> bool(torch.all(padded[2:] == 0.0))
        True
    """
    if n_points <= arr.shape[0]:
        return arr

    n_pad = n_points - arr.shape[0]
    arr_padded = torch.nn.functional.pad(
        arr,
        (
            0,
            0,
            0,
            0,
            0,
            n_pad,
        ),
        mode="constant",
        value=pad_value,
    )
    return arr_padded


def shuffle_array(
    points: torch.Tensor,
    n_points: int,
    weights: torch.Tensor = None,
):
    """
    Randomly sample points from tensor without replacement.

    This function performs random sampling from the input tensor, selecting
    n_points points without replacement. It's commonly used for creating training
    subsets and data augmentation in machine learning workflows.

    Optionally, you can provide weights to use in the sampling.

    Note: the implementation with torch.multinomial is constrained to 2^24 points.
    If the input is larger than that, it will be split and sampled from each chunk.

    Args:
        points: Input tensor to sample from, shape (n_points, ...).
        n_points: Number of points to sample. If greater than arr.shape[0],
            all points are returned.
        weights: Optional weights for sampling. If None, uniform weights are used.

    Returns:
        Tuple containing:
        - Sampled tensor subset
        - Indices of the selected points

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(42)  # For reproducible results
        >>> data = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> subset, indices = shuffle_array(data, 2)
        >>> subset.shape
        torch.Size([2, 2])
        >>> indices.shape
        torch.Size([2])
        >>> len(torch.unique(indices)) == 2  # No duplicates
        True
    """

    N_input_points = points.shape[0]

    if N_input_points < n_points:
        return points, torch.arange(N_input_points)

    # If there are no weights, use uniform weights:
    if weights is None:
        weights = torch.ones(points.shape[0], device=points.device)

    # Using torch multinomial for this.
    # Multinomial can't work with more than 2^24 input points.

    # So apply chunking and stich back together in that case.
    # Assume each chunk gets a number proportional to it's size,
    # (but make sure they add up to n_points!)

    max_chunk_size = 2**24

    N_chunks = (N_input_points // max_chunk_size) + 1

    # Divide the weights into these chunks
    chunk_weights = torch.chunk(weights, N_chunks)

    # Determine how mant points to compute per chunk:
    points_per_chunk = [
        round(n_points * c.shape[0] / N_input_points) for c in chunk_weights
    ]

    gap = n_points - sum(points_per_chunk)

    if gap > 0:
        for g in range(gap):
            points_per_chunk[g] += 1
    elif gap < 0:
        for g in range(-gap):
            points_per_chunk[g] -= 1

    # Create a list of indexes per chunk:
    idx_chunks = [
        torch.multinomial(
            w,
            p,
            replacement=False,
        )
        for w, p in zip(chunk_weights, points_per_chunk)
    ]

    # Stitch the chunks back together:
    idx = torch.cat(idx_chunks)

    # Apply the selection:
    points_selected = points[idx]

    return points_selected, idx


def shuffle_array_without_sampling(
    arr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffle tensor order without changing the number of elements.

    This function reorders all elements in the tensor randomly while preserving
    all data points. It's useful for randomizing data order before training
    while maintaining the complete dataset.

    Args:
        arr: Input tensor to shuffle, shape (n_points, ...).

    Returns:
        Tuple containing:
        - Shuffled tensor with same shape as input
        - Permutation indices used for shuffling

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(42)  # For reproducible results
        >>> data = torch.tensor([[1], [2], [3], [4]])
        >>> shuffled, indices = shuffle_array_without_sampling(data)
        >>> shuffled.shape
        torch.Size([4, 1])
        >>> indices.shape
        torch.Size([4])
        >>> set(indices.tolist()) == set(range(4))  # All original indices present
        True
    """
    idx = torch.randperm(arr.shape[0])
    return arr[idx], idx


def create_directory(filepath: str | Path) -> None:
    """Create directory and all necessary parent directories.

    This function creates a directory at the specified path, including any
    necessary parent directories. It's equivalent to `mkdir -p` in Unix systems.

    Args:
        filepath: Path to the directory to create. Can be string or Path object.

    """
    Path(filepath).mkdir(parents=True, exist_ok=True)


def get_filenames(filepath: str | Path, exclude_dirs: bool = False) -> list[str]:
    """Get list of filenames in a directory with optional directory filtering.

    This function returns all items in a directory, with options to exclude
    subdirectories. It handles special cases like .zarr directories which
    are treated as files even when exclude_dirs is True.

    Args:
        filepath: Path to the directory to list. Can be string or Path object.
        exclude_dirs: If True, exclude subdirectories from results.
            Exception: .zarr directories are always included as they represent
            data files in array storage format.

    Returns:
        List of filenames/directory names found in the specified directory.

    Raises:
        FileNotFoundError: If the specified directory does not exist.

    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Directory {filepath} does not exist")

    filenames = []
    for item in path.iterdir():
        if exclude_dirs and item.is_dir():
            # Include directories ending with .zarr even when exclude_dirs is True
            if item.name.endswith(".zarr"):
                filenames.append(item.name)
            continue
        filenames.append(item.name)
    return filenames


def calculate_pos_encoding(nx: torch.Tensor, d: int = 8) -> list[torch.Tensor]:
    """Calculate sinusoidal positional encoding for transformer architectures.

    This function computes positional encodings using alternating sine and cosine
    functions at different frequencies. These encodings help neural networks
    understand positional relationships in sequences or spatial data.

    Args:
        nx: Input positions/coordinates to encode.
        d: Encoding dimensionality. Must be even number. Defaults to 8.

    Returns:
        List of d tensors containing alternating sine and cosine encodings.
        Each pair (sin, cos) uses progressively lower frequencies.

    Examples:
        >>> import torch
        >>> positions = torch.tensor([0.0, 1.0, 2.0])
        >>> encodings = calculate_pos_encoding(positions, d=4)
        >>> len(encodings)
        4
        >>> all(enc.shape == (3,) for enc in encodings)
        True
    """
    vec = []
    for k in range(int(d / 2)):
        vec.append(torch.sin(nx / 10000 ** (2 * k / d)))
        vec.append(torch.cos(nx / 10000 ** (2 * k / d)))
    return vec


def combine_dict(old_dict: dict[Any, Any], new_dict: dict[Any, Any]) -> dict[Any, Any]:
    """Combine two dictionaries by adding values for matching keys.

    This function performs element-wise addition of dictionary values for
    keys that exist in both dictionaries. It's commonly used for accumulating
    statistics or metrics across multiple iterations.

    Args:
        old_dict: Base dictionary to update.
        new_dict: Dictionary with values to add to old_dict.

    Returns:
        Updated old_dict with combined values.

    Note:
        This function modifies old_dict in place and returns it.
        Values must support the + operator.

    Examples:
        >>> stats1 = {"loss": 0.5, "accuracy": 0.8}
        >>> stats2 = {"loss": 0.3, "accuracy": 0.1}
        >>> combined = combine_dict(stats1, stats2)
        >>> combined["loss"]
        0.8
        >>> combined["accuracy"]
        0.9
    """
    for key in old_dict.keys():
        old_dict[key] += new_dict[key]
    return old_dict


def create_grid(
    max_coords: torch.Tensor, min_coords: torch.Tensor, resolution: torch.Tensor
) -> torch.Tensor:
    """Create a 3D regular grid from coordinate bounds and resolution.

    This function generates a regular 3D grid spanning from min_coords to
    max_coords with the specified resolution in each dimension. The resulting
    grid is commonly used for interpolation, visualization, and regular sampling.

    Args:
        max_coords: Maximum coordinates [x_max, y_max, z_max] for the grid bounds.
        min_coords: Minimum coordinates [x_min, y_min, z_min] for the grid bounds.
        resolution: Number of grid points [nx, ny, nz] in each dimension.

    Returns:
        Grid tensor of shape (nx, ny, nz, 3) containing 3D coordinates for each
        grid point. The last dimension contains [x, y, z] coordinates.

    Examples:
        >>> import torch
        >>> min_bounds = torch.tensor([0.0, 0.0, 0.0])
        >>> max_bounds = torch.tensor([1.0, 1.0, 1.0])
        >>> grid_res = torch.tensor([2, 2, 2])
        >>> grid = create_grid(max_bounds, min_bounds, grid_res)
        >>> grid.shape
        torch.Size([2, 2, 2, 3])
        >>> torch.allclose(grid[0, 0, 0], torch.tensor([0.0, 0.0, 0.0]))
        True
        >>> torch.allclose(grid[1, 1, 1], torch.tensor([1.0, 1.0, 1.0]))
        True
    """
    # Linspace to make evenly spaced steps along each axis:
    dd = [
        torch.linspace(
            min_coords[i],
            max_coords[i],
            resolution[i],
            dtype=max_coords.dtype,
            device=max_coords.device,
        )
        for i in range(3)
    ]

    # Combine them with meshgrid:
    xv, yv, zv = torch.meshgrid(*dd, indexing="ij")

    xv = xv.unsqueeze(-1)
    yv = yv.unsqueeze(-1)
    zv = zv.unsqueeze(-1)
    grid = torch.concatenate((xv, yv, zv), axis=-1)
    return grid


def mean_std_sampling(
    field: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, tolerance: float = 3.0
) -> list[int]:
    """Identify outlier points based on statistical distance from mean.

    This function identifies data points that are statistical outliers by
    checking if they fall outside mean ± tolerance*std for any field component.
    It's useful for data cleaning and identifying regions of interest in CFD data.

    Args:
        field: Input field tensor of shape (n_points, n_components).
        mean: Mean values for each field component, shape (n_components,).
        std: Standard deviation for each component, shape (n_components,).
        tolerance: Number of standard deviations to use as outlier threshold.
            Defaults to 3.0 (99.7% of normal distribution).

    Returns:
        List of indices identifying outlier points that exceed the statistical threshold.

    Examples:
        >>> import torch
        >>> # Create test data with outliers
        >>> field = torch.tensor([[1.0], [2.0], [3.0], [10.0]])  # 10.0 is outlier
        >>> field_mean = torch.tensor([2.0])
        >>> field_std = torch.tensor([1.0])
        >>> outliers = mean_std_sampling(field, field_mean, field_std, 2.0)
        >>> 3 in outliers  # Index 3 (value 10.0) should be detected as outlier
        True
    """

    idx_all = []
    for v in range(field.shape[-1]):
        fv = field[:, v]
        idx = torch.where(
            (fv > mean[v] + tolerance * std[v]) | (fv < mean[v] - tolerance * std[v])
        )
        if len(idx[0]) != 0:
            idx_all += list(idx[0])

    return idx_all


def dict_to_device(
    state_dict: dict[str, Any], device: Any, exclude_keys: list[str] | None = None
) -> dict[str, Any]:
    """Move dictionary values to specified device (GPU/CPU).

    This function transfers PyTorch tensors in a dictionary to the specified
    compute device while preserving the dictionary structure. It's commonly
    used for moving model parameters and data between CPU and GPU.

    Args:
        state_dict: Dictionary containing tensors and other values.
        device: Target device (e.g., torch.device('cuda:0')).
        exclude_keys: List of keys to skip during device transfer.
            Defaults to ["filename"] if None.

    Returns:
        New dictionary with tensors moved to the specified device.
        Non-tensor values and excluded keys are preserved as-is.

    Examples:
        >>> import torch
        >>> data = {"weights": torch.randn(10, 10), "filename": "model.pt"}
        >>> device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        >>> gpu_data = dict_to_device(data, device)
    """
    if exclude_keys is None:
        exclude_keys = ["filename"]

    new_state_dict = {}
    for k, v in state_dict.items():
        if k not in exclude_keys:
            new_state_dict[k] = v.to(device)
    return new_state_dict


def area_weighted_shuffle_array(
    arr: torch.Tensor, n_points: int, area: torch.Tensor, area_factor: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform area-weighted random sampling from tensor.

    This function samples points from a tensor with probability proportional to
    their associated area weights. This is particularly useful in CFD applications
    where larger cells or surface elements should have higher sampling probability.

    Args:
        arr: Input tensor to sample from, shape (n_points, ...).
        n_points: Number of points to sample. If greater than arr.shape[0],
            samples all available points.
        area: Area weights for each point, shape (n_points,). Larger values
            indicate higher sampling probability.
        area_factor: Exponent applied to area weights to control sampling bias.
            Values > 1.0 increase bias toward larger areas, values < 1.0 reduce bias.
            Defaults to 1.0 (linear weighting).

    Returns:
        Tuple containing:
        - Sampled tensor subset weighted by area
        - Indices of the selected points

    Note:
        For GPU tensors, the sampling is performed on the current device.
        The sampling uses torch.multinomial for efficient weighted sampling.

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(42)  # For reproducible results
        >>> mesh_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        >>> cell_areas = torch.tensor([0.1, 0.1, 0.1, 10.0])  # Last point has much larger area
        >>> subset, indices = area_weighted_shuffle_array(mesh_data, 2, cell_areas)
        >>> subset.shape
        torch.Size([2, 1])
        >>> indices.shape
        torch.Size([2])
        >>> # The point with large area (index 3) should likely be selected
        >>> len(set(indices)) <= 2  # At most 2 unique indices
        True
        >>> # Use higher area_factor for stronger bias toward large areas
        >>> subset_biased, _ = area_weighted_shuffle_array(mesh_data, 2, cell_areas, area_factor=2.0)
    """

    # Calculate area-weighted probabilities
    sampling_probabilities = area**area_factor
    sampling_probabilities /= sampling_probabilities.sum()  # Normalize to sum to 1

    return shuffle_array(arr, n_points, sampling_probabilities)


def solution_weighted_shuffle_array(
    arr: torch.Tensor,
    n_points: int,
    solution: torch.Tensor,
    scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform solution-weighted random sampling from tensor.

    This function samples points from a tensor with probability proportional to
    their associated solution weights. This is particularly useful in CFD applications
    where larger cells or surface elements should have higher sampling probability.

    Args:
        arr: Input tensor to sample from, shape (n_points, ...).
        n_points: Number of points to sample. If greater than arr.shape[0],
            samples all available points.
        solution: Solution weights for each point, shape (n_points,). Larger values
            indicate higher sampling probability.
        scaling_factor: Exponent applied to solution weights to control sampling bias.
            Values > 1.0 increase bias toward larger solution fields, values < 1.0 reduce bias.
            Defaults to 1.0 (linear weighting).

    Returns:
        Tuple containing:
        - Sampled tensor subset weighted by solution fields
        - Indices of the selected points

    Note:
        For GPU tensors, the sampling is performed on the current device.
        The sampling uses torch.multinomial for efficient weighted sampling.

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(42)  # For reproducible results
        >>> mesh_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        >>> solution = torch.tensor([0.1, 0.1, 0.1, 10.0])  # Last point has much larger solution field
        >>> subset, indices = solution_weighted_shuffle_array(mesh_data, 2, solution)
        >>> subset.shape
        torch.Size([2, 1])
        >>> indices.shape
        torch.Size([2])
        >>> # The point with large area (index 3) should likely be selected
        >>> len(set(indices)) <= 2  # At most 2 unique indices
        True
        >>> # Use higher scaling_factor for stronger bias toward large solution fields
        >>> subset_biased, _ = solution_weighted_shuffle_array(mesh_data, 2, solution, scaling_factor=2.0)
    """

    # Calculate solution-weighted probabilities
    sampling_probabilities = solution**scaling_factor
    sampling_probabilities /= sampling_probabilities.sum()  # Normalize to sum to 1

    return shuffle_array(arr, n_points, sampling_probabilities)


def sample_points_on_mesh(
    mesh_coordinates: torch.Tensor,
    mesh_faces: torch.Tensor,
    n_points: int,
    mesh_areas: torch.Tensor | None = None,
    mesh_normals: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Uniformly sample points on a mesh.

    Will use area-weighted sampling to select mesh regions, then uniform
    sampling within those triangles.
    """

    # First, if we don't have the areas, compute them:
    faces_reshaped = mesh_faces.reshape(-1, 3)

    if mesh_areas is None or mesh_normals is None:
        # We have to do 90% of the work for both of these,
        # to get either.  So check at the last minute:
        faces_reshaped_p0 = faces_reshaped[:, 0]
        faces_reshaped_p1 = faces_reshaped[:, 1]
        faces_reshaped_p2 = faces_reshaped[:, 2]
        d1 = mesh_coordinates[faces_reshaped_p1] - mesh_coordinates[faces_reshaped_p0]
        d2 = mesh_coordinates[faces_reshaped_p2] - mesh_coordinates[faces_reshaped_p0]
        inferred_mesh_normals = torch.linalg.cross(d1, d2, dim=1)
        normals_norm = torch.linalg.norm(inferred_mesh_normals, dim=1)
        inferred_mesh_normals = inferred_mesh_normals / normals_norm.unsqueeze(1)
        if mesh_normals is None:
            mesh_normals = inferred_mesh_normals
        if mesh_areas is None:
            mesh_areas = 0.5 * normals_norm

    # Next, use the areas to compute a weighted sampling of the triangles:
    target_triangles = torch.multinomial(
        mesh_areas,
        n_points,
        replacement=True,
    )

    target_faces = faces_reshaped[target_triangles]

    # Next, generate random points within each selected triangle.
    # We'll map two uniform distributions to the points in the triangles.
    # See https://stackoverflow.com/questions/47410054/generate-random-locations-within-a-triangular-domain
    # and the original reference https://www.cs.princeton.edu/%7Efunk/tog02.pdf
    # for more information
    r1 = torch.rand((n_points, 1), device=mesh_coordinates.device)
    r2 = torch.rand((n_points, 1), device=mesh_coordinates.device)

    s1 = torch.sqrt(r1)

    local_coords = torch.stack(
        (1.0 - s1, (1.0 - r2) * s1, r2 * s1),
        dim=1,
    )

    barycentric_coordinates = torch.sum(
        mesh_coordinates[target_faces] * local_coords, dim=1
    )

    # Apply the selection to the other tensors, too:
    target_areas = mesh_areas[target_triangles]
    target_normals = mesh_normals[target_triangles]

    return barycentric_coordinates, target_triangles, target_areas, target_normals
