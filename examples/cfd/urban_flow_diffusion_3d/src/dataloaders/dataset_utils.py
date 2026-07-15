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

import numpy as np
import torch
import h5py
from typing import Optional, Union

from src.utils import select_random


def select_random_field(data, num_elements=1, seed=None, combined_channels=True):
    """
    Select random ground-truth snapshots from a UflowDataset3D's raw field
    storage (``dataset.data``), handling both HDF5 layouts it can produce:
    a single combined ``data`` array/dataset (``combined_channels=True``),
    or a list of 3 separate ``U``/``V``/``W`` h5py Datasets
    (``combined_channels=False``) -- the latter has no ``.shape`` of its own
    and can't be fancy-indexed directly the way ``select_random`` expects.

    Mirrors ``select_random``'s return convention: a single leading-dim-less
    snapshot when ``num_elements == 1``, otherwise a batch.
    """
    if combined_channels:
        return select_random(data, num_elements=num_elements, seed=seed)

    u, v, w = data
    # h5py fancy-indexing requires increasing-order indices; select_random
    # already returns them sorted ascending.
    indices = select_random(u, num_elements=num_elements, seed=seed, only_indices=True)
    stacked = np.stack([u[indices], v[indices], w[indices]], axis=1)
    return stacked[0] if num_elements == 1 else stacked


def normalize(x: Union[np.ndarray, torch.Tensor], mins, maxs, eps=1e-9):
    """
    Normalize a tensor (torch or numpy) channel-wise to [-1, 1] using given min and max.

    Args:
        x: Tensor with shape (C, H, W) or (C, D, H, W)
        mins: array-like or tensor with shape [C]
        maxs: array-like or tensor with shape [C]

    Returns:
        Normalized tensor with same type and shape.
    """
    if isinstance(x, np.ndarray):
        mins = np.reshape(mins, (-1,) + (1,) * (x.ndim - 1))
        maxs = np.reshape(maxs, (-1,) + (1,) * (x.ndim - 1))
        x_scaled = (x - mins) / (maxs - mins + eps)
        return 2 * x_scaled - 1
    elif isinstance(x, torch.Tensor):
        mins = torch.as_tensor(mins, dtype=x.dtype, device=x.device).view(
            -1, *[1] * (x.dim() - 1)
        )
        maxs = torch.as_tensor(maxs, dtype=x.dtype, device=x.device).view(
            -1, *[1] * (x.dim() - 1)
        )
        x_scaled = (x - mins) / (maxs - mins + eps)
        return 2 * x_scaled - 1
    else:
        raise TypeError("Input must be a numpy array or torch tensor")


def renormalize(x_norm: Union[np.ndarray, torch.Tensor], mins, maxs, eps=1e-9):
    """
    Reverses normalization back to original range.

    Args:
        x_norm: Normalized data in [-1, 1]
        mins, maxs: Original min/max values per channel

    Returns:
        Renormalized tensor or array.
    """
    if isinstance(x_norm, np.ndarray):
        mins = np.reshape(mins, (-1,) + (1,) * (x_norm.ndim - 1))
        maxs = np.reshape(maxs, (-1,) + (1,) * (x_norm.ndim - 1))
        x_rescaled = (x_norm + 1) / 2
        return x_rescaled * (maxs - mins + eps) + mins
    elif isinstance(x_norm, torch.Tensor):
        mins = torch.as_tensor(mins, dtype=x_norm.dtype, device=x_norm.device).view(
            -1, *[1] * (x_norm.dim() - 1)
        )
        maxs = torch.as_tensor(maxs, dtype=x_norm.dtype, device=x_norm.device).view(
            -1, *[1] * (x_norm.dim() - 1)
        )
        x_rescaled = (x_norm + 1) / 2
        return x_rescaled * (maxs - mins + eps) + mins
    else:
        raise TypeError("Input must be a numpy array or torch tensor")


def combine_fields(
    u: Optional[np.ndarray] = None,
    v: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Combine velocity fields u, v, and optionally w into a single (N, C, ...) array.

    Returns:
        np.ndarray of shape [N, C, ...]
    """
    if u is None or v is None:
        raise ValueError("At least u and v fields must be provided.")

    u = np.expand_dims(u, axis=1)
    v = np.expand_dims(v, axis=1)

    if w is not None:
        w = np.expand_dims(w, axis=1)
        data = np.concatenate((u, v, w), axis=1)
    else:
        data = np.concatenate((u, v), axis=1)

    print(f"✅ Combined field shape: {data.shape}")
    return data


def rescale_and_crop_ds4(preds, min_scaler, max_scaler, crop_hw):
    """This should be only used for ds4, only applicable when padding is not symmetric"""
    preds_np = ((preds + 1.0) / 2).clip(0, 1).cpu().numpy()
    preds_np = preds_np[:, :, :, : crop_hw[0], : crop_hw[1]]
    return min_scaler + (max_scaler - min_scaler) * preds_np


def resize_spatial_tensor_yz(x, target_hw, pad_value=0):
    """
    Resize the spatial dimensions (H, W) of a tensor to match `target_hw`
    via symmetric cropping or padding.

    Works for 4D or 5D tensors (N, C, H, W) or (N, C, D, H, W) for both numpy and torch.

    Parameters:
    - x: Input tensor (numpy.ndarray or torch.Tensor)
    - target_hw: Tuple/List (target_height, target_width)
    - pad_value: Constant value to use for padding (default=0)

    Returns:
    - Resized tensor with shape (..., target_height, target_width)
    """
    is_torch = torch.is_tensor(x)
    current_h, current_w = x.shape[-2:]
    target_h, target_w = target_hw

    delta_h = target_h - current_h
    delta_w = target_w - current_w

    # Compute cropping indices or padding widths
    if delta_h < 0:
        crop_top = -delta_h // 2
        crop_bottom = crop_top + target_h
    else:
        pad_top = delta_h // 2
        pad_bottom = delta_h - pad_top

    if delta_w < 0:
        crop_left = -delta_w // 2
        crop_right = crop_left + target_w
    else:
        pad_left = delta_w // 2
        pad_right = delta_w - pad_left

    if delta_h < 0 or delta_w < 0:
        # Cropping
        if is_torch:
            return x[..., crop_top:crop_bottom, crop_left:crop_right]
        else:
            return x[..., crop_top:crop_bottom, crop_left:crop_right]

    elif delta_h > 0 or delta_w > 0:
        # Padding
        if is_torch:
            pad = [pad_left, pad_right, pad_top, pad_bottom]  # (W1, W2, H1, H2)
            return F.pad(x, pad=pad, mode="constant", value=pad_value)
        else:
            pad_width = [(0, 0)] * x.ndim
            pad_width[-2] = (pad_top, pad_bottom)
            pad_width[-1] = (pad_left, pad_right)
            return np.pad(x, pad_width, mode="constant", constant_values=pad_value)

    return x  # No resizing needed


def load_h5(file_path, load_components="U", verbose=True):
    """
    Loads velocity + coordinate fields from an HDF5 file.
    """
    imgU = imgV = imgW = None

    with h5py.File(file_path, "r") as hf:
        if verbose:
            print(f"📂 Opened HDF5: {file_path}")
            print(f"📁 Keys: {list(hf.keys())}")

        x = hf["x"][:]
        y = hf["y"][:]
        z = hf["z"][:] if "z" in hf else None
        t = hf["t"][:]

        imgU = hf["U"][:]
        if load_components.upper() in ("UV", "UVW"):
            imgV = hf["V"][:]
        if load_components.upper() == "UVW":
            imgW = hf["W"][:]

    if load_components.upper() == "U":
        return imgU, x, y, z, t
    elif load_components.upper() == "UV":
        return imgU, imgV, x, y, z, t
    elif load_components.upper() == "UVW":
        return imgU, imgV, imgW, x, y, z, t
    else:
        raise ValueError(f"Invalid component mode: {load_components}")


def lazy_load_h5(file_path, load_components="U", verbose=True):
    """
    Returns h5py dataset handles without loading them into memory.
    """
    hf = h5py.File(file_path, "r")
    x = hf["x"][:]
    y = hf["y"][:]
    z = hf["z"][:]
    t = hf["t"][:]

    imgU = hf["U"]
    if verbose:
        print(f"Lazy loaded U with shape: {imgU.shape}")

    if load_components.upper() in ("UV", "UVW"):
        imgV = hf["V"]
        print(f"Lazy loaded V with shape: {imgV.shape}")

    if load_components.upper() == "UVW":
        imgW = hf["W"]
        print(f"Lazy loaded W with shape: {imgW.shape}")

    if load_components.upper() == "U":
        return hf, imgU, x, y, z, t
    elif load_components.upper() == "UV":
        return hf, imgU, imgV, x, y, z, t
    elif load_components.upper() == "UVW":
        return hf, imgU, imgV, imgW, x, y, z, t


def get_precision(dtype_str):
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "amp-fp16": torch.float16,
        "amp-bf16": torch.bfloat16,
    }.get(dtype_str, torch.float32)


def move_batch_to_device(batch, device, dtype=torch.float32):
    """
    Moves each tensor in the batch dict to the given device and dtype,
    and ensures tensors are contiguous in memory.

    Args:
        batch (dict): Dictionary of batched tensors.
        device (torch.device): CUDA/CPU device.
        dtype (torch.dtype): Floating point precision (default: float32).

    Returns:
        dict: Same structure, with tensors on the correct device and contiguous.
    """
    return {
        k: v.to(device=device, dtype=dtype, non_blocking=True).contiguous()
        if torch.is_tensor(v)
        else v
        for k, v in batch.items()
    }
