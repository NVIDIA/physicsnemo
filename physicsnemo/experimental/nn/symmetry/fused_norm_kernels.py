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

"""Fused Warp GPU kernels for equivariant normalization layers.

This file contains pure warp kernels. No PyTorch code here.
Two-pass design:
  - Kernel 1 (reduce): 3D grid (batch, l, channels), tile reduction over channels
  - Kernel 2 (normalize): 3D grid (batch, l, channels), pure SIMT
"""

import warp as wp


@wp.func
def inv_rms_transform(norm_sum: float, inv_num_channels: float, eps: float) -> float:
    """Convert accumulated sum-of-squares to inverse RMS.

    Computes: 1 / sqrt(norm_sum / num_channels + eps)
    Using pre-computed inv_num_channels = 1.0 / num_channels.
    """
    return 1.0 / wp.sqrt(norm_sum * inv_num_channels + eps)


@wp.kernel
def layernorm_grid_reduce(
    x: wp.array4d(dtype=float),
    inv_rms: wp.array(dtype=float),  # 1D flattened [batch * lmax_p1]
    per_degree_norm_weight: wp.array2d(dtype=float),
    lmax_p1: int,
    mmax: int,
    num_channels: int,
    inv_num_channels: float,
    eps: float,
):
    """Compute per-degree inverse RMS with tile reduction over channels.

    Launched with wp.launch(dim=(batch, lmax+1, num_channels), block_dim=num_channels).
    Threads within a block cooperatively reduce the channel dimension.

    Parameters
    ----------
    x : [batch, lmax+1, mmax+1, 2*channels]
        Input features.
    inv_rms : [batch * lmax_p1]
        Output: inverse RMS per (batch, l). Pre-zeroed by caller.
    per_degree_norm_weight : [lmax+1, mmax+1]
        Weights per (l, m).
    lmax_p1 : int
        lmax + 1.
    mmax : int
        Maximum order.
    num_channels : int
        Number of channels (= block_dim).
    inv_num_channels : float
        Pre-computed 1.0 / num_channels.
    eps : float
        Epsilon for numerical stability.
    """
    batch_idx, l_idx, c = wp.tid()
    num_valid_m = wp.min(l_idx, mmax) + 1

    # Each thread accumulates its per-channel contribution
    local_norm = float(0.0)
    for m in range(num_valid_m):
        w = per_degree_norm_weight[l_idx, m]
        for ri in range(2):
            if m == 0 and ri == 1:
                continue
            val = x[batch_idx, l_idx, m, ri * num_channels + c]
            local_norm = local_norm + w * val * val

    # Cooperative tile reduction across channels within the block
    t = wp.tile(local_norm)
    s = wp.tile_sum(t)

    # Transform sum-of-squares to inverse RMS
    inv_num_channels_tile = wp.tile(inv_num_channels)
    eps_tile = wp.tile(eps)
    s_transformed = wp.tile_map(inv_rms_transform, s, inv_num_channels_tile, eps_tile)

    # Store result — one write per (batch, l) block
    flat_idx = batch_idx * lmax_p1 + l_idx
    wp.tile_atomic_add(inv_rms, s_transformed, offset=flat_idx)


@wp.kernel
def layernorm_grid_reduce_submean(
    x: wp.array4d(dtype=float),
    l0_mean: wp.array(dtype=float),
    inv_rms: wp.array(dtype=float),  # 1D flattened [batch * lmax_p1]
    per_degree_norm_weight: wp.array2d(dtype=float),
    lmax_p1: int,
    mmax: int,
    num_channels: int,
    inv_num_channels: float,
    eps: float,
):
    """Compute per-degree inverse RMS with tile reduction over channels, subtracting l=0 mean.

    Launched with wp.launch(dim=(batch, lmax+1, num_channels), block_dim=num_channels).
    Threads within a block cooperatively reduce the channel dimension.

    Parameters
    ----------
    x : [batch, lmax+1, mmax+1, 2*channels]
        Input features.
    l0_mean : [batch]
        Pre-computed mean of l=0, m=0, real channels.
    inv_rms : [batch * lmax_p1]
        Output: inverse RMS per (batch, l). Pre-zeroed by caller.
    per_degree_norm_weight : [lmax+1, mmax+1]
        Weights per (l, m).
    lmax_p1 : int
        lmax + 1.
    mmax : int
        Maximum order.
    num_channels : int
        Number of channels (= block_dim).
    inv_num_channels : float
        Pre-computed 1.0 / num_channels.
    eps : float
        Epsilon for numerical stability.
    """
    batch_idx, l_idx, c = wp.tid()
    num_valid_m = wp.min(l_idx, mmax) + 1

    # Each thread accumulates its per-channel contribution
    local_norm = float(0.0)
    for m in range(num_valid_m):
        w = per_degree_norm_weight[l_idx, m]
        for ri in range(2):
            if m == 0 and ri == 1:
                continue
            val = x[batch_idx, l_idx, m, ri * num_channels + c]
            if l_idx == 0 and m == 0 and ri == 0:
                val = val - l0_mean[batch_idx]
            local_norm = local_norm + w * val * val

    # Cooperative tile reduction across channels within the block
    t = wp.tile(local_norm)
    s = wp.tile_sum(t)

    # Transform sum-of-squares to inverse RMS
    inv_num_channels_tile = wp.tile(inv_num_channels)
    eps_tile = wp.tile(eps)
    s_transformed = wp.tile_map(inv_rms_transform, s, inv_num_channels_tile, eps_tile)

    # Store result — one write per (batch, l) block
    flat_idx = batch_idx * lmax_p1 + l_idx
    wp.tile_atomic_add(inv_rms, s_transformed, offset=flat_idx)


@wp.kernel
def layernorm_grid_normalize(
    x: wp.array4d(dtype=float),
    output: wp.array4d(dtype=float),
    inv_rms: wp.array2d(dtype=float),
    affine_weight: wp.array2d(dtype=float),
    grid_mask: wp.array3d(dtype=float),
    mmax: int,
    num_channels: int,
):
    """Normalize features using pre-computed statistics. Pure SIMT, one thread per channel.

    Launched with wp.launch(dim=(batch, lmax+1, num_channels)).

    Parameters
    ----------
    x : [batch, lmax+1, mmax+1, 2*channels]
        Input features.
    output : [batch, lmax+1, mmax+1, 2*channels]
        Output features.
    inv_rms : [batch, lmax+1]
        Pre-computed inverse RMS.
    affine_weight : [lmax+1, channels]
        Scale parameters.
    grid_mask : [lmax+1, mmax+1, 2]
        Validity mask combining (l,m) validity and m=0 imaginary zeroing.
        1.0 for valid positions, 0.0 for invalid.
    mmax : int
        Maximum order.
    num_channels : int
        Number of channels.
    """
    batch_idx, l_idx, c = wp.tid()

    inv_rms_val = inv_rms[batch_idx, l_idx]
    aw = affine_weight[l_idx, c]

    for m in range(mmax + 1):
        for ri in range(2):
            val = x[batch_idx, l_idx, m, ri * num_channels + c]
            val = val * inv_rms_val
            val = val * aw
            val = val * grid_mask[l_idx, m, ri]
            output[batch_idx, l_idx, m, ri * num_channels + c] = val


@wp.kernel
def layernorm_grid_normalize_submean(
    x: wp.array4d(dtype=float),
    output: wp.array4d(dtype=float),
    l0_mean: wp.array(dtype=float),
    inv_rms: wp.array2d(dtype=float),
    affine_weight: wp.array2d(dtype=float),
    grid_mask: wp.array3d(dtype=float),
    mmax: int,
    num_channels: int,
):
    """Normalize features using pre-computed statistics, subtracting l=0 mean. Pure SIMT, one thread per channel.

    Launched with wp.launch(dim=(batch, lmax+1, num_channels)).

    Parameters
    ----------
    x : [batch, lmax+1, mmax+1, 2*channels]
        Input features.
    output : [batch, lmax+1, mmax+1, 2*channels]
        Output features.
    l0_mean : [batch]
        Pre-computed l=0 channel mean.
    inv_rms : [batch, lmax+1]
        Pre-computed inverse RMS.
    affine_weight : [lmax+1, channels]
        Scale parameters.
    grid_mask : [lmax+1, mmax+1, 2]
        Validity mask combining (l,m) validity and m=0 imaginary zeroing.
        1.0 for valid positions, 0.0 for invalid.
    mmax : int
        Maximum order.
    num_channels : int
        Number of channels.
    """
    batch_idx, l_idx, c = wp.tid()

    inv_rms_val = inv_rms[batch_idx, l_idx]
    aw = affine_weight[l_idx, c]

    for m in range(mmax + 1):
        for ri in range(2):
            val = x[batch_idx, l_idx, m, ri * num_channels + c]
            if l_idx == 0 and m == 0 and ri == 0:
                val = val - l0_mean[batch_idx]
            val = val * inv_rms_val
            val = val * aw
            val = val * grid_mask[l_idx, m, ri]
            output[batch_idx, l_idx, m, ri * num_channels + c] = val


@wp.kernel
def layernorm_grid_normalize_submean_bias(
    x: wp.array4d(dtype=float),
    output: wp.array4d(dtype=float),
    l0_mean: wp.array(dtype=float),
    inv_rms: wp.array2d(dtype=float),
    affine_weight: wp.array2d(dtype=float),
    affine_bias: wp.array(dtype=float),
    grid_mask: wp.array3d(dtype=float),
    mmax: int,
    num_channels: int,
):
    """Normalize features using pre-computed statistics, subtracting l=0 mean and adding bias. Pure SIMT, one thread per channel.

    Launched with wp.launch(dim=(batch, lmax+1, num_channels)).

    Parameters
    ----------
    x : [batch, lmax+1, mmax+1, 2*channels]
        Input features.
    output : [batch, lmax+1, mmax+1, 2*channels]
        Output features.
    l0_mean : [batch]
        Pre-computed l=0 channel mean.
    inv_rms : [batch, lmax+1]
        Pre-computed inverse RMS.
    affine_weight : [lmax+1, channels]
        Scale parameters.
    affine_bias : [channels]
        Bias for l=0.
    grid_mask : [lmax+1, mmax+1, 2]
        Validity mask combining (l,m) validity and m=0 imaginary zeroing.
        1.0 for valid positions, 0.0 for invalid.
    mmax : int
        Maximum order.
    num_channels : int
        Number of channels.
    """
    batch_idx, l_idx, c = wp.tid()

    inv_rms_val = inv_rms[batch_idx, l_idx]
    aw = affine_weight[l_idx, c]

    for m in range(mmax + 1):
        for ri in range(2):
            val = x[batch_idx, l_idx, m, ri * num_channels + c]
            if l_idx == 0 and m == 0 and ri == 0:
                val = val - l0_mean[batch_idx]
            val = val * inv_rms_val
            val = val * aw
            if l_idx == 0 and m == 0 and ri == 0:
                val = val + affine_bias[c]
            val = val * grid_mask[l_idx, m, ri]
            output[batch_idx, l_idx, m, ri * num_channels + c] = val


# =============================================================================
# RMSNorm Kernels
# =============================================================================


@wp.kernel
def rmsnorm_grid_reduce(
    x: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    norm_stats: wp.array(dtype=float),  # [batch], pre-zeroed
    balance_weight: wp.array2d(dtype=float),  # [lmax+1, mmax+1]
    mmax: int,
    num_channels: int,
):
    """Compute global norm statistics with tile reduction over channels.

    Launched with wp.launch(dim=(batch, lmax+1, num_channels), block_dim=num_channels).
    Multiple blocks (different l) accumulate into the same norm_stats[batch] via atomics.

    Parameters
    ----------
    x : [batch, lmax+1, mmax+1, 2*channels]
        Input features.
    norm_stats : [batch]
        Output: accumulated sum-of-squares per batch. Pre-zeroed by caller.
        Multiple l-blocks write to same batch slot via atomic add.
    balance_weight : [lmax+1, mmax+1]
        Degree balancing weights per (l, m).
    mmax : int
        Maximum order.
    num_channels : int
        Number of channels (= block_dim).
    """
    batch_idx, l_idx, c = wp.tid()
    num_valid_m = wp.min(l_idx, mmax) + 1

    # Each thread accumulates its per-channel contribution
    local_norm = float(0.0)
    for m in range(num_valid_m):
        w = balance_weight[l_idx, m]
        for ri in range(2):
            if m == 0 and ri == 1:
                continue
            val = x[batch_idx, l_idx, m, ri * num_channels + c]
            local_norm = local_norm + w * val * val

    # Cooperative tile reduction across channels within the block
    t = wp.tile(local_norm)
    s = wp.tile_sum(t)

    # Store raw sum-of-squares
    # Multiple l-blocks atomically accumulate into the same batch slot
    wp.tile_atomic_add(norm_stats, s, offset=batch_idx)


@wp.kernel
def rmsnorm_grid_reduce_submean(
    x: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    l0_mean: wp.array(dtype=float),  # [batch]
    norm_stats: wp.array(dtype=float),  # [batch], pre-zeroed
    balance_weight: wp.array2d(dtype=float),  # [lmax+1, mmax+1]
    mmax: int,
    num_channels: int,
):
    """Compute global norm statistics with tile reduction over channels, subtracting l=0 mean.

    Launched with wp.launch(dim=(batch, lmax+1, num_channels), block_dim=num_channels).
    Multiple blocks (different l) accumulate into the same norm_stats[batch] via atomics.

    Parameters
    ----------
    x : [batch, lmax+1, mmax+1, 2*channels]
        Input features.
    l0_mean : [batch]
        Pre-computed mean of l=0, m=0, real channels.
    norm_stats : [batch]
        Output: accumulated sum-of-squares per batch. Pre-zeroed by caller.
        Multiple l-blocks write to same batch slot via atomic add.
    balance_weight : [lmax+1, mmax+1]
        Degree balancing weights per (l, m).
    mmax : int
        Maximum order.
    num_channels : int
        Number of channels (= block_dim).
    """
    batch_idx, l_idx, c = wp.tid()
    num_valid_m = wp.min(l_idx, mmax) + 1

    # Each thread accumulates its per-channel contribution
    local_norm = float(0.0)
    for m in range(num_valid_m):
        w = balance_weight[l_idx, m]
        for ri in range(2):
            if m == 0 and ri == 1:
                continue
            val = x[batch_idx, l_idx, m, ri * num_channels + c]
            if l_idx == 0 and m == 0 and ri == 0:
                val = val - l0_mean[batch_idx]
            local_norm = local_norm + w * val * val

    # Cooperative tile reduction across channels within the block
    t = wp.tile(local_norm)
    s = wp.tile_sum(t)

    # Store raw sum-of-squares
    # Multiple l-blocks atomically accumulate into the same batch slot
    wp.tile_atomic_add(norm_stats, s, offset=batch_idx)


@wp.kernel
def rmsnorm_grid_normalize(
    x: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    output: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    norm_stats: wp.array(dtype=float),  # [batch] - raw accumulated sum
    affine_weight: wp.array2d(dtype=float),  # [lmax+1, channels]
    grid_mask: wp.array3d(dtype=float),  # [lmax+1, mmax+1, 2]
    mmax: int,
    num_channels: int,
    inv_num_channels: float,
    eps: float,
):
    """Normalize features using global statistics. Pure SIMT, one thread per channel.

    Launched with wp.launch(dim=(batch, lmax+1, num_channels)).
    Computes inv_rms inline from raw norm_stats per-thread.

    Parameters
    ----------
    x : [batch, lmax+1, mmax+1, 2*channels]
        Input features.
    output : [batch, lmax+1, mmax+1, 2*channels]
        Output features.
    norm_stats : [batch]
        Raw accumulated sum-of-squares from reduce kernel.
    affine_weight : [lmax+1, channels]
        Scale parameters.
    grid_mask : [lmax+1, mmax+1, 2]
        Validity mask combining (l,m) validity and m=0 imaginary zeroing.
        1.0 for valid positions, 0.0 for invalid.
    mmax : int
        Maximum order.
    num_channels : int
        Number of channels.
    inv_num_channels : float
        Pre-computed 1.0 / (2 * num_channels) for averaging.
    eps : float
        Epsilon for numerical stability.
    """
    batch_idx, l_idx, c = wp.tid()

    # Compute inv_rms inline from raw norm_stats (redundant but cheap, value is L1-cached)
    inv_rms_val = 1.0 / wp.sqrt(norm_stats[batch_idx] * inv_num_channels + eps)
    aw = affine_weight[l_idx, c]

    for m in range(mmax + 1):
        for ri in range(2):
            val = x[batch_idx, l_idx, m, ri * num_channels + c]
            val = val * inv_rms_val
            val = val * aw
            val = val * grid_mask[l_idx, m, ri]
            output[batch_idx, l_idx, m, ri * num_channels + c] = val


@wp.kernel
def rmsnorm_grid_normalize_submean(
    x: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    output: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    l0_mean: wp.array(dtype=float),  # [batch]
    norm_stats: wp.array(dtype=float),  # [batch] - raw accumulated sum
    affine_weight: wp.array2d(dtype=float),  # [lmax+1, channels]
    grid_mask: wp.array3d(dtype=float),  # [lmax+1, mmax+1, 2]
    mmax: int,
    num_channels: int,
    inv_num_channels: float,
    eps: float,
):
    """Normalize features using global statistics, subtracting l=0 mean. Pure SIMT, one thread per channel.

    Launched with wp.launch(dim=(batch, lmax+1, num_channels)).
    Computes inv_rms inline from raw norm_stats per-thread.

    Parameters
    ----------
    x : [batch, lmax+1, mmax+1, 2*channels]
        Input features.
    output : [batch, lmax+1, mmax+1, 2*channels]
        Output features.
    l0_mean : [batch]
        Pre-computed l=0 channel mean.
    norm_stats : [batch]
        Raw accumulated sum-of-squares from reduce kernel.
    affine_weight : [lmax+1, channels]
        Scale parameters.
    grid_mask : [lmax+1, mmax+1, 2]
        Validity mask combining (l,m) validity and m=0 imaginary zeroing.
        1.0 for valid positions, 0.0 for invalid.
    mmax : int
        Maximum order.
    num_channels : int
        Number of channels.
    inv_num_channels : float
        Pre-computed 1.0 / (2 * num_channels) for averaging.
    eps : float
        Epsilon for numerical stability.
    """
    batch_idx, l_idx, c = wp.tid()

    # Compute inv_rms inline from raw norm_stats (redundant but cheap, value is L1-cached)
    inv_rms_val = 1.0 / wp.sqrt(norm_stats[batch_idx] * inv_num_channels + eps)
    aw = affine_weight[l_idx, c]

    for m in range(mmax + 1):
        for ri in range(2):
            val = x[batch_idx, l_idx, m, ri * num_channels + c]
            if l_idx == 0 and m == 0 and ri == 0:
                val = val - l0_mean[batch_idx]
            val = val * inv_rms_val
            val = val * aw
            val = val * grid_mask[l_idx, m, ri]
            output[batch_idx, l_idx, m, ri * num_channels + c] = val


@wp.kernel
def rmsnorm_grid_normalize_submean_bias(
    x: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    output: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    l0_mean: wp.array(dtype=float),  # [batch]
    norm_stats: wp.array(dtype=float),  # [batch] - raw accumulated sum
    affine_weight: wp.array2d(dtype=float),  # [lmax+1, channels]
    affine_bias: wp.array(dtype=float),  # [channels]
    grid_mask: wp.array3d(dtype=float),  # [lmax+1, mmax+1, 2]
    mmax: int,
    num_channels: int,
    inv_num_channels: float,
    eps: float,
):
    """Normalize features using global statistics, subtracting l=0 mean and adding bias. Pure SIMT, one thread per channel.

    Launched with wp.launch(dim=(batch, lmax+1, num_channels)).
    Computes inv_rms inline from raw norm_stats per-thread.

    Parameters
    ----------
    x : [batch, lmax+1, mmax+1, 2*channels]
        Input features.
    output : [batch, lmax+1, mmax+1, 2*channels]
        Output features.
    l0_mean : [batch]
        Pre-computed l=0 channel mean.
    norm_stats : [batch]
        Raw accumulated sum-of-squares from reduce kernel.
    affine_weight : [lmax+1, channels]
        Scale parameters.
    affine_bias : [channels]
        Bias for l=0.
    grid_mask : [lmax+1, mmax+1, 2]
        Validity mask combining (l,m) validity and m=0 imaginary zeroing.
        1.0 for valid positions, 0.0 for invalid.
    mmax : int
        Maximum order.
    num_channels : int
        Number of channels.
    inv_num_channels : float
        Pre-computed 1.0 / (2 * num_channels) for averaging.
    eps : float
        Epsilon for numerical stability.
    """
    batch_idx, l_idx, c = wp.tid()

    # Compute inv_rms inline from raw norm_stats (redundant but cheap, value is L1-cached)
    inv_rms_val = 1.0 / wp.sqrt(norm_stats[batch_idx] * inv_num_channels + eps)
    aw = affine_weight[l_idx, c]

    for m in range(mmax + 1):
        for ri in range(2):
            val = x[batch_idx, l_idx, m, ri * num_channels + c]
            if l_idx == 0 and m == 0 and ri == 0:
                val = val - l0_mean[batch_idx]
            val = val * inv_rms_val
            val = val * aw
            if l_idx == 0 and m == 0 and ri == 0:
                val = val + affine_bias[c]
            val = val * grid_mask[l_idx, m, ri]
            output[batch_idx, l_idx, m, ri * num_channels + c] = val
