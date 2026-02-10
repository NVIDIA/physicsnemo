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

    # Store raw sum-of-squares (transformation to inv_rms happens in normalize kernel)
    flat_idx = batch_idx * lmax_p1 + l_idx
    wp.tile_atomic_add(inv_rms, s, offset=flat_idx)


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

    # Store raw sum-of-squares (transformation to inv_rms happens in normalize kernel)
    flat_idx = batch_idx * lmax_p1 + l_idx
    wp.tile_atomic_add(inv_rms, s, offset=flat_idx)


@wp.kernel
def layernorm_grid_normalize(
    x: wp.array4d(dtype=float),
    output: wp.array4d(dtype=float),
    norm_stats: wp.array2d(dtype=float),  # [batch, lmax+1] - raw sum-of-squares
    affine_weight: wp.array2d(dtype=float),
    grid_mask: wp.array3d(dtype=float),
    mmax: int,
    num_channels: int,
    inv_num_channels: float,
    eps: float,
):
    """Normalize features using pre-computed statistics. Pure SIMT, one thread per channel.

    Launched with wp.launch(dim=(batch, lmax+1, num_channels)).

    Parameters
    ----------
    x : [batch, lmax+1, mmax+1, 2*channels]
        Input features.
    output : [batch, lmax+1, mmax+1, 2*channels]
        Output features.
    norm_stats : [batch, lmax+1]
        Raw sum-of-squares per (batch, l) from reduce kernel.
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
        Pre-computed 1.0 / num_channels.
    eps : float
        Epsilon for numerical stability.
    """
    batch_idx, l_idx, c = wp.tid()

    # Compute inv_rms inline from raw norm_stats
    inv_rms_val = 1.0 / wp.sqrt(norm_stats[batch_idx, l_idx] * inv_num_channels + eps)
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
    norm_stats: wp.array2d(dtype=float),  # [batch, lmax+1] - raw sum-of-squares
    affine_weight: wp.array2d(dtype=float),
    grid_mask: wp.array3d(dtype=float),
    mmax: int,
    num_channels: int,
    inv_num_channels: float,
    eps: float,
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
    norm_stats : [batch, lmax+1]
        Raw sum-of-squares per (batch, l) from reduce kernel.
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
        Pre-computed 1.0 / num_channels.
    eps : float
        Epsilon for numerical stability.
    """
    batch_idx, l_idx, c = wp.tid()

    # Compute inv_rms inline from raw norm_stats
    inv_rms_val = 1.0 / wp.sqrt(norm_stats[batch_idx, l_idx] * inv_num_channels + eps)
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
    norm_stats: wp.array2d(dtype=float),  # [batch, lmax+1] - raw sum-of-squares
    affine_weight: wp.array2d(dtype=float),
    affine_bias: wp.array(dtype=float),
    grid_mask: wp.array3d(dtype=float),
    mmax: int,
    num_channels: int,
    inv_num_channels: float,
    eps: float,
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
    norm_stats : [batch, lmax+1]
        Raw sum-of-squares per (batch, l) from reduce kernel.
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
        Pre-computed 1.0 / num_channels.
    eps : float
        Epsilon for numerical stability.
    """
    batch_idx, l_idx, c = wp.tid()

    # Compute inv_rms inline from raw norm_stats
    inv_rms_val = 1.0 / wp.sqrt(norm_stats[batch_idx, l_idx] * inv_num_channels + eps)
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


@wp.kernel
def layernormsh_lgt0_reduce(
    x_lgt0: wp.array4d(dtype=float),  # [batch, lmax, mmax+1, 2*channels]
    norm_stats: wp.array(dtype=float),  # [batch], pre-zeroed
    balance_weight_lgt0: wp.array2d(dtype=float),  # [lmax, mmax+1]
    mmax: int,
    num_channels: int,
):
    """Compute global norm statistics for l>0 with tile reduction over channels.

    This kernel operates on the l>0 slice only (excluding l=0 scalar component).
    Launched with wp.launch(dim=(batch, lmax, num_channels), block_dim=num_channels).
    Multiple blocks (different l) accumulate into the same norm_stats[batch] via atomics.

    Parameters
    ----------
    x_lgt0 : [batch, lmax, mmax+1, 2*channels]
        Input features for l>0 degrees (l=1 to lmax).
    norm_stats : [batch]
        Output: accumulated sum-of-squares per batch. Pre-zeroed by caller.
        Multiple l-blocks write to same batch slot via atomic add.
    balance_weight_lgt0 : [lmax, mmax+1]
        Degree balancing weights for l>0 per (l, m).
    mmax : int
        Maximum order.
    num_channels : int
        Number of channels (= block_dim).
    """
    batch_idx, l_idx, c = wp.tid()
    # l_idx is 0-based within l>0 slice, so actual degree is l_idx + 1
    l_actual = l_idx + 1
    num_valid_m = wp.min(l_actual, mmax) + 1

    # Each thread accumulates its per-channel contribution
    local_norm = float(0.0)
    for m in range(num_valid_m):
        w = balance_weight_lgt0[l_idx, m]
        for ri in range(2):
            if m == 0 and ri == 1:
                continue
            val = x_lgt0[batch_idx, l_idx, m, ri * num_channels + c]
            local_norm = local_norm + w * val * val

    # Cooperative tile reduction across channels within the block
    t = wp.tile(local_norm)
    s = wp.tile_sum(t)

    # Store raw sum-of-squares
    # Multiple l-blocks atomically accumulate into the same batch slot
    wp.tile_atomic_add(norm_stats, s, offset=batch_idx)


@wp.kernel
def layernormsh_lgt0_normalize(
    x_lgt0: wp.array4d(dtype=float),  # [batch, lmax, mmax+1, 2*channels]
    output_lgt0: wp.array4d(dtype=float),  # [batch, lmax, mmax+1, 2*channels]
    norm_stats: wp.array(dtype=float),  # [batch]
    affine_weight: wp.array2d(dtype=float),  # [lmax, channels]
    grid_mask_lgt0: wp.array3d(dtype=float),  # [lmax, mmax+1, 2]
    mmax: int,
    num_channels: int,
    inv_num_channels: float,
    eps: float,
):
    """Normalize l>0 features using global statistics. Pure SIMT, one thread per channel.

    This kernel operates on the l>0 slice only (excluding l=0 scalar component).
    Launched with wp.launch(dim=(batch, lmax, num_channels)).
    Computes inv_rms inline from raw norm_stats per-thread.

    Parameters
    ----------
    x_lgt0 : [batch, lmax, mmax+1, 2*channels]
        Input features for l>0 degrees (l=1 to lmax).
    output_lgt0 : [batch, lmax, mmax+1, 2*channels]
        Output features for l>0 degrees.
    norm_stats : [batch]
        Raw accumulated sum-of-squares from reduce kernel.
    affine_weight : [lmax, channels]
        Scale parameters for l>0 degrees.
    grid_mask_lgt0 : [lmax, mmax+1, 2]
        Validity mask for l>0 combining (l,m) validity and m=0 imaginary zeroing.
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
            val = x_lgt0[batch_idx, l_idx, m, ri * num_channels + c]
            val = val * inv_rms_val
            val = val * aw
            val = val * grid_mask_lgt0[l_idx, m, ri]
            output_lgt0[batch_idx, l_idx, m, ri * num_channels + c] = val


# =============================================================================
# Backward Kernels for RMSNorm
# =============================================================================


@wp.kernel
def rmsnorm_backward_reduce(
    grad_output: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    output: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    go_dot_o: wp.array(dtype=float),  # [batch], pre-zeroed
    mmax: int,
    num_channels: int,
):
    """Compute go_dot_o[b] = sum(grad_output * output) for backward pass.

    Launched with wp.launch(dim=(batch, lmax+1, num_channels), block_dim=num_channels).
    Uses tile reduction for efficient accumulation across channels and atomic add across l-blocks.

    Parameters
    ----------
    grad_output : [batch, lmax+1, mmax+1, 2*channels]
        Upstream gradient from loss.
    output : [batch, lmax+1, mmax+1, 2*channels]
        Forward pass output (saved from forward).
    go_dot_o : [batch]
        Output: inner product per batch. Pre-zeroed by caller.
        Multiple l-blocks write to same batch slot via atomic add.
    mmax : int
        Maximum order.
    num_channels : int
        Number of channels (= block_dim).
    """
    batch_idx, l_idx, c = wp.tid()
    num_valid_m = wp.min(l_idx, mmax) + 1

    # Each thread accumulates its per-channel contribution to the inner product
    local_sum = float(0.0)
    for m in range(num_valid_m):
        for ri in range(2):
            # No need to check validity - output is already masked to zero at invalid positions
            go = grad_output[batch_idx, l_idx, m, ri * num_channels + c]
            o = output[batch_idx, l_idx, m, ri * num_channels + c]
            local_sum = local_sum + go * o

    # Cooperative tile reduction across channels within the block
    t = wp.tile(local_sum)
    s = wp.tile_sum(t)

    # Store result - multiple l-blocks atomically accumulate into the same batch slot
    wp.tile_atomic_add(go_dot_o, s, offset=batch_idx)


@wp.kernel
def rmsnorm_backward_normalize(
    grad_output: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    x: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    norm_stats: wp.array(dtype=float),  # [batch]
    go_dot_o: wp.array(dtype=float),  # [batch]
    affine_weight: wp.array2d(dtype=float),  # [lmax+1, channels]
    balance_weight: wp.array2d(dtype=float),  # [lmax+1, mmax+1]
    grid_mask: wp.array3d(dtype=float),  # [lmax+1, mmax+1, 2]
    grad_x: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels] - output
    grad_affine_weight: wp.array2d(
        dtype=float
    ),  # [lmax+1, channels] - output (zeroed by caller)
    mmax: int,
    num_channels: int,
    inv_num_channels: float,
    eps: float,
):
    """Compute grad_x and grad_affine_weight for backward pass. Pure SIMT, one thread per (b, l, c).

    Launched with wp.launch(dim=(batch, lmax+1, num_channels)).

    Parameters
    ----------
    grad_output : [batch, lmax+1, mmax+1, 2*channels]
        Upstream gradient.
    x : [batch, lmax+1, mmax+1, 2*channels]
        Input from forward pass.
    norm_stats : [batch]
        Raw sum-of-squares from forward reduce kernel.
    go_dot_o : [batch]
        Inner product of grad_output and output from backward reduce kernel.
    affine_weight : [lmax+1, channels]
        Affine scale parameters from forward.
    balance_weight : [lmax+1, mmax+1]
        Degree balancing weights.
    grid_mask : [lmax+1, mmax+1, 2]
        Validity mask (combines m<=l constraint and m=0 imaginary zeroing).
    grad_x : [batch, lmax+1, mmax+1, 2*channels]
        Output: gradient w.r.t. input x.
    grad_affine_weight : [lmax+1, channels]
        Output: gradient w.r.t. affine_weight. Pre-zeroed by caller.
    mmax : int
        Maximum order.
    num_channels : int
        Number of channels.
    inv_num_channels : float
        Pre-computed 1.0 / (2 * num_channels).
    eps : float
        Epsilon for numerical stability.
    """
    batch_idx, l_idx, c = wp.tid()

    # Recompute inv_rms from forward pass
    inv_rms_val = 1.0 / wp.sqrt(norm_stats[batch_idx] * inv_num_channels + eps)
    inv_rms_sq = inv_rms_val * inv_rms_val
    aw = affine_weight[l_idx, c]
    go_dot_o_val = go_dot_o[batch_idx]

    for m in range(mmax + 1):
        for ri in range(2):
            go_val = grad_output[batch_idx, l_idx, m, ri * num_channels + c]
            x_val = x[batch_idx, l_idx, m, ri * num_channels + c]
            mask_val = grid_mask[l_idx, m, ri]
            bw = balance_weight[l_idx, m]

            # Path A: direct gradient (applies everywhere, masked by grid_mask)
            grad_a = go_val * inv_rms_val * aw * mask_val

            # Path B: indirect through norm_stats
            # Only at reduce-valid positions: grid_mask and balance_weight product gives correct mask
            # (balance_weight is 0 for m>l, grid_mask is 0 for m=0 imaginary)
            grad_b = (
                inv_num_channels * inv_rms_sq * go_dot_o_val * bw * x_val * mask_val
            )

            grad_x[batch_idx, l_idx, m, ri * num_channels + c] = grad_a - grad_b

            # Accumulate grad_affine_weight (atomic — threads across b, m, ri contribute)
            grad_aw_contrib = go_val * x_val * inv_rms_val * mask_val
            wp.atomic_add(grad_affine_weight, l_idx, c, grad_aw_contrib)
