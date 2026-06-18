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
@Author : Clement Etienam
Pure array/tensor manipulation utilities shared across all sub-modules.
"""

# Standard Library
import re

# Third-party Libraries
import numpy as np
import torch

# Local Modules
from utils.logging_utils import setup_logging


def replace_nans_and_infs(tensor, value=0.0):
    """Replace NaN and Inf entries of a tensor in-place with a constant value.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor modified in-place.
    value : float, optional
        Replacement scalar value. Default is 0.0.

    Returns
    -------
    torch.Tensor
        The same tensor with NaN/Inf entries replaced.
    """
    tensor[torch.isnan(tensor) | torch.isinf(tensor)] = value
    return tensor


def replace_large_and_invalid_values(arr, placeholder=0.0):
    """Replace NaN, Inf, and float32-overflow values in an array with a placeholder.

    Parameters
    ----------
    arr : np.ndarray
        Input array modified in-place.
    placeholder : float, optional
        Scalar replacement for invalid entries. Default is 0.0.

    Returns
    -------
    np.ndarray
        Array with all invalid entries replaced by ``placeholder``.
    """
    threshold = np.finfo(np.float32).max
    invalid_indices = (np.isnan(arr)) | (np.isinf(arr)) | (np.abs(arr) > threshold)
    arr[invalid_indices] = placeholder
    return arr


def clean_dict_arrays(data_dict):
    """Apply ``replace_large_and_invalid_values`` to every array in a dictionary.

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping string keys to ``np.ndarray`` values.

    Returns
    -------
    dict
        The same dictionary with all arrays cleaned in-place.
    """
    for key in data_dict:
        data_dict[key] = replace_large_and_invalid_values(data_dict[key])
    return data_dict


def clip_and_convert_to_float32(array):
    """Clip an array to float32 representable range and cast it to float32.

    Parameters
    ----------
    array : np.ndarray
        Input array of any dtype.

    Returns
    -------
    np.ndarray
        Float32 array clipped to ``[np.finfo(np.float32).min, np.finfo(np.float32).max]``.
    """
    max_float32 = np.finfo(np.float32).max
    min_float32 = np.finfo(np.float32).min
    array_clipped = np.clip(array, min_float32, max_float32)
    # array_clipped = round_array_to_4dp(array_clipped)
    return array_clipped.astype(np.float32)


def clip_and_convert_to_float3(array):
    """Clip an array to float32 range and return it as float32 (alias variant).

    Parameters
    ----------
    array : np.ndarray
        Input array of any dtype.

    Returns
    -------
    np.ndarray
        Float32 array clipped to the float32 representable range.
    """
    max_float32 = np.finfo(np.float32).max
    min_float32 = np.finfo(np.float32).min

    array_clipped = np.clip(array, min_float32, max_float32)
    # array_clipped = round_array_to_4dp(array_clipped)
    return array_clipped.astype(np.float32)


def replace_nan_with_zero(tensor):
    """Replace NaN and Inf entries in a tensor with the mean of the finite elements.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of any shape that may contain NaN or Inf.

    Returns
    -------
    torch.Tensor
        Tensor with invalid entries replaced by the finite-element mean (or 1e-6).
    """
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)
    invalid_mask = nan_mask | inf_mask
    valid_elements = tensor[~invalid_mask]  # Elements that are not NaN or Inf
    if valid_elements.numel() > 0:  # Ensure there are valid elements to calculate mean
        mean_value = valid_elements.mean()
    else:
        mean_value = torch.tensor(1e-6, device=tensor.device)
    return torch.where(invalid_mask, mean_value, tensor)


def sort_key(s):
    """Extract the number from the filename for sorting."""
    return int(re.search(r"\d+", s).group())


def Split_Matrix(matrix, sizee):
    """Split a matrix into equal parts along axis 0.

    Parameters
    ----------
    matrix : np.ndarray
        Input array to split along the first axis.
    sizee : int
        Number of equal sections to split into.

    Returns
    -------
    list of np.ndarray
        List of sub-arrays resulting from the split.
    """
    return np.split(matrix, sizee, axis=0)


def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


def rescale_linear_numpy_pytorch(array, new_min, new_max, minimum, maximum):
    """Rescale an arrary linearly."""
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


def rescale_linear_pytorch_numpy(array, new_min, new_max, minimum, maximum):
    """Rescale an arrary linearly."""
    m = (maximum - minimum) / (new_max - new_min)
    b = minimum - m * new_min
    return m * array + b


def round_array_to_4dp(arr):
    """Round all elements of an array to four decimal places.

    Parameters
    ----------
    arr : array-like
        Input data convertible to a NumPy array.

    Returns
    -------
    np.ndarray or None
        Array rounded to 4 decimal places, or ``None`` if an error occurs.
    """
    try:
        # Convert input to a numpy array if it's not already
        arr = np.asarray(arr)
        return np.around(arr, 4)
    except Exception as e:
        logger = setup_logging()
        logger.error(f"An error occurred: {e!s}")
        return None  # You can choose to return None or handle the error differently


def scale_operation(tensor, target_min, target_max):
    """
    Clean invalid values and normalise a numpy array by its maximum.

    Parameters
    ----------
    tensor : np.ndarray
        Array to clean and normalise; modified in-place for NaN/Inf.
    target_min : float
        Unused target minimum (reserved for interface consistency).
    target_max : float
        Unused target maximum (reserved for interface consistency).

    Returns
    -------
    min_val : float
        Minimum value of the cleaned array.
    max_val : float
        Maximum value of the cleaned array.
    rescaled_tensor : np.ndarray
        Array divided by max_val.
    """
    tensor[np.isnan(tensor)] = 0  # Replace NaN with 0
    tensor[np.isinf(tensor)] = 0  # Replace infinity with 0
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    rescaled_tensor = tensor / max_val
    return min_val, max_val, rescaled_tensor


def Make_correct(array):
    """Reorder a 5-D array/tensor from (N, C, nz, H, W) to (N, C, H, W, nz).

    Works with both numpy arrays and torch tensors. Stays on whatever
    device the input is on — no host transfers, no copies.
    """
    if isinstance(array, torch.Tensor):
        # torch: permute(0, 1, 3, 4, 2). Add .contiguous() so downstream
        # ops that expect contiguous memory (e.g. Conv3d, view) work.
        return array.permute(0, 1, 3, 4, 2).contiguous()
    else:
        # numpy: transpose(0, 1, 3, 4, 2). The .copy() makes the result
        # contiguous in memory (transpose returns a view, which can break
        # later code that expects a real array).
        return np.ascontiguousarray(np.transpose(array, (0, 1, 3, 4, 2)))


def linear_interp(x, xp, fp):
    """Perform 1-D piecewise-linear interpolation using PyTorch tensors.

    Parameters
    ----------
    x : torch.Tensor
        Query points at which to interpolate.
    xp : torch.Tensor
        1-D tensor of sorted breakpoint coordinates.
    fp : torch.Tensor
        1-D tensor of function values at each breakpoint in ``xp``.

    Returns
    -------
    torch.Tensor
        Interpolated values at the query points ``x``.
    """
    contiguous_xp = xp.contiguous()
    left_indices = torch.clamp(
        torch.searchsorted(contiguous_xp, x) - 1, 0, len(contiguous_xp) - 2
    )
    denominators = contiguous_xp[left_indices + 1] - contiguous_xp[left_indices]
    close_to_zero = denominators.abs() < 1e-10
    denominators[close_to_zero] = 1.0  # or any non-zero value to avoid NaN
    return (
        ((fp[left_indices + 1] - fp[left_indices]) / denominators)
        * (x - contiguous_xp[left_indices])
    ) + fp[left_indices]


def interp_torch(cuda, reference_matrix1, reference_matrix2, tensor1):
    """Apply linear interpolation to a tensor using a reference lookup table.

    Parameters
    ----------
    cuda : str
        Device identifier (e.g., ``"cuda:0"`` or ``"cpu"``); currently unused.
    reference_matrix1 : torch.Tensor
        1-D sorted breakpoint coordinates.
    reference_matrix2 : torch.Tensor
        1-D function values corresponding to ``reference_matrix1``.
    tensor1 : torch.Tensor
        Query tensor to interpolate; processed in a single chunk.

    Returns
    -------
    list of torch.Tensor
        List containing the single interpolated chunk tensor.
    """
    chunk_size = 1
    chunks = torch.chunk(tensor1, chunks=chunk_size, dim=0)
    processed_chunks = []
    for start_idx in range(chunk_size):
        interpolated_chunk = linear_interp(
            chunks[start_idx], reference_matrix1, reference_matrix2
        )
        processed_chunks.append(interpolated_chunk)
    torch.cuda.empty_cache()
    return processed_chunks


def convert_back(rescaled_tensor, target_min, target_max, min_val, max_val):
    """Reverse a max-normalisation by multiplying back by the original maximum.

    Parameters
    ----------
    rescaled_tensor : np.ndarray or torch.Tensor
        Normalised tensor produced by fit_operation.
    target_min : float
        Unused (reserved for interface consistency).
    target_max : float
        Unused (reserved for interface consistency).
    min_val : float
        Unused (reserved for interface consistency).
    max_val : float
        Original maximum value used during normalisation.

    Returns
    -------
    np.ndarray or torch.Tensor
        Tensor scaled back to original range.
    """
    return rescaled_tensor * max_val


def fit_operation(tensor, target_min, target_max, tensor_min, tensor_max):
    """Normalise a tensor by dividing by its maximum value.

    Parameters
    ----------
    tensor : np.ndarray
        Array to normalise.
    target_min : float
        Unused target minimum (reserved for interface consistency).
    target_max : float
        Unused target maximum (reserved for interface consistency).
    tensor_min : float
        Unused observed minimum (reserved for interface consistency).
    tensor_max : float
        Divisor used to normalise the tensor.

    Returns
    -------
    rescaled_tensor : np.ndarray
        Tensor divided by tensor_max.
    """
    return tensor / tensor_max


def find_first_numeric_row(df):
    """Find the first row in the DataFrame where all data is numeric.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame whose rows will be scanned.

    Returns
    -------
    int or None
        Zero-based index of the first fully-numeric row, or None if absent.
    """
    for i in range(len(df)):
        if df.iloc[i].apply(np.isreal).all():
            return i
    return None
