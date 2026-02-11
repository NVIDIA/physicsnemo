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

"""Fused Warp GPU kernels and PyTorch custom ops for equivariant normalization layers.

This module provides GPU-accelerated normalization for spherical harmonic features
using NVIDIA Warp kernels, with full PyTorch autograd integration via
``torch.library.custom_op``.

Module Structure
----------------
The module is organized into three sections:

1. **Forward Warp Kernels** - low-level GPU kernels for the forward pass.
2. **Backward Warp Kernels** - hand-written gradient kernels for the backward pass.
3. **Custom Op Wrappers** - ``torch.library.custom_op`` definitions that orchestrate
   kernel launches and register autograd support, making the fused kernels usable
   as differentiable operations within standard PyTorch training loops. This includes
   public forward ops, private backward ops (for ``torch.compile`` compatibility),
   and autograd glue connecting them.

Each normalization variant generally is structured with two kernels:

- **Pass 1 (reduce):** A 3D grid ``(batch, l, channels)`` with cooperative
  tile reduction over the channel dimension accumulates squared-norm statistics
  via atomic adds. Requires ``block_dim=num_channels``.
- **Pass 2 (normalize):** A 3D grid ``(batch, l, channels)`` in pure SIMT mode
  (one thread per element) reads the statistics, computes ``inv_rms``, and writes
  the normalized output.

The backward pass mirrors this with two analogous kernels plus a recomputation
of the forward statistics.

Autograd Integration
--------------------
Each custom op (e.g., ``fused_rmsnorm``) is wired to PyTorch's autograd engine
via six components per normalization variant:

- **Backward/forward custom op** (``@torch.library.custom_op``): Launches forward Warp
  kernels and returns outputs/gradients.
- **Backward/forward fake** (``@<op>.register_fake``): Returns an empty tensor with the
  correct shape/dtype for ``torch.compile`` tracing.
- **Setup context** (``setup_context``): Saves tensors and attributes on the
  autograd context for use in the backward pass.

Kernel Inventory
----------------
**Per-Degree LayerNorm** (used by ``FusedEquivariantLayerNorm``):
    Each spherical harmonic degree is normalized independently with its own
    inverse-RMS statistic. Statistics are stored as ``[batch * (lmax+1)]``
    (flattened) in the reduce pass and read as ``[batch, lmax+1]`` (2D) in the
    normalize pass. Uses ``inv_num_channels = 1 / num_channels``.

    Forward kernels:
        ``layernorm_grid_reduce`` /
        ``layernorm_grid_reduce_submean`` —
            Accumulate per-degree sum-of-squares, optionally subtracting the
            l=0 channel mean first.
        ``layernorm_grid_normalize`` /
        ``layernorm_grid_normalize_submean`` /
        ``layernorm_grid_normalize_submean_bias`` —
            Apply per-degree inv-RMS scaling with affine weight, optional
            mean subtraction, and optional bias at l=0.
    Backward kernels:
        ``layernorm_grid_backward_reduce`` /
        ``layernorm_grid_backward_reduce_submean_bias`` —
            Compute per-degree ``go_dot_o = sum(grad_output * output)``.
        ``layernorm_grid_backward_normalize`` /
        ``layernorm_grid_backward_normalize_submean`` —
            Compute ``grad_x``, ``grad_affine_weight``, and optionally
            ``grad_affine_bias``.
    Custom op:
        ``fused_layernorm`` — Orchestrates all of the above with autograd.
        ``_fused_layernorm_backward`` — Private backward custom op for
            ``torch.compile`` compatibility.

**Global RMSNorm** (used by ``FusedEquivariantRMSNorm``):
    A single inverse-RMS statistic is computed across *all* degrees and used to
    scale every component. Statistics are stored as ``[batch]``.
    Uses ``inv_num_channels = 1 / (2 * num_channels)``.

    Forward kernels:
        ``rmsnorm_grid_reduce`` /
        ``rmsnorm_grid_reduce_submean`` —
            Accumulate global sum-of-squares across all (l, m) positions.
        ``rmsnorm_grid_normalize`` /
        ``rmsnorm_grid_normalize_submean`` /
        ``rmsnorm_grid_normalize_submean_bias`` —
            Apply global inv-RMS scaling with affine weight, optional mean
            subtraction, and optional bias at l=0.
    Backward kernels:
        ``rmsnorm_backward_reduce`` /
        ``rmsnorm_backward_reduce_submean_bias`` —
            Compute global ``go_dot_o = sum(grad_output * output)``.
        ``rmsnorm_backward_normalize`` /
        ``rmsnorm_backward_normalize_submean`` —
            Compute ``grad_x``, ``grad_affine_weight``, and optionally
            ``grad_affine_bias``.
    Custom op:
        ``fused_rmsnorm`` — Orchestrates all of the above with autograd.
        ``_fused_rmsnorm_backward`` — Private backward custom op for
            ``torch.compile`` compatibility.

**LayerNormSH l>0** (used by ``FusedEquivariantLayerNormSH``):
    Normalizes only the l>0 degrees with a single shared inverse-RMS statistic
    (l=0 is handled separately by ``torch.nn.LayerNorm`` in the calling module).
    Input tensors are pre-sliced to ``[batch, lmax, ...]`` (l=0 excluded).
    Statistics are stored as ``[batch]``.
    Uses ``inv_num_channels = 1 / (2 * num_channels)``.

    Forward kernels:
        ``layernormsh_lgt0_reduce`` —
            Accumulate global sum-of-squares over l>0 degrees.
        ``layernormsh_lgt0_normalize`` —
            Apply global inv-RMS scaling with affine weight.
    Backward kernels:
        ``layernormsh_lgt0_backward_reduce`` —
            Compute global ``go_dot_o`` over l>0 slice.
        ``layernormsh_lgt0_backward_normalize`` —
            Compute ``grad_x`` and ``grad_affine_weight`` for l>0.
    Custom op:
        ``fused_layernormsh_lgt0`` — Orchestrates all of the above with autograd.
        ``_fused_layernormsh_lgt0_backward`` — Private backward custom op for
            ``torch.compile`` compatibility.
"""

import warp as wp


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


@wp.kernel
def rmsnorm_backward_reduce_submean_bias(
    grad_output: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    output: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    affine_bias: wp.array(dtype=float),  # [channels]
    go_dot_o: wp.array(dtype=float),  # [batch], pre-zeroed
    mmax: int,
    num_channels: int,
):
    """Compute go_dot_o[b] = sum(grad_output * output_no_bias) for backward pass with bias.

    This kernel is used when the forward pass includes a bias term at (l=0, m=0, ri=0).
    The go_dot_o computation must use output WITHOUT the bias added, so we subtract it inline.

    Launched with wp.launch(dim=(batch, lmax+1, num_channels), block_dim=num_channels).
    Uses tile reduction for efficient accumulation across channels and atomic add across l-blocks.

    Parameters
    ----------
    grad_output : [batch, lmax+1, mmax+1, 2*channels]
        Upstream gradient from loss.
    output : [batch, lmax+1, mmax+1, 2*channels]
        Forward pass output WITH bias (saved from forward).
    affine_bias : [channels]
        Bias parameters from forward (added at l=0, m=0, ri=0).
    go_dot_o : [batch]
        Output: inner product per batch using output_no_bias. Pre-zeroed by caller.
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

            # Subtract bias contribution at (l=0, m=0, ri=0) to get output_no_bias
            if l_idx == 0 and m == 0 and ri == 0:
                o = o - affine_bias[c]

            local_sum = local_sum + go * o

    # Cooperative tile reduction across channels within the block
    t = wp.tile(local_sum)
    s = wp.tile_sum(t)

    # Store result - multiple l-blocks atomically accumulate into the same batch slot
    wp.tile_atomic_add(go_dot_o, s, offset=batch_idx)


@wp.kernel
def rmsnorm_backward_normalize_submean(
    grad_output: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    x: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    l0_mean: wp.array(dtype=float),  # [batch]
    norm_stats: wp.array(dtype=float),  # [batch]
    go_dot_o: wp.array(dtype=float),  # [batch]
    affine_weight: wp.array2d(dtype=float),  # [lmax+1, channels]
    balance_weight: wp.array2d(dtype=float),  # [lmax+1, mmax+1]
    grid_mask: wp.array3d(dtype=float),  # [lmax+1, mmax+1, 2]
    grad_x: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels] - output
    grad_affine_weight: wp.array2d(
        dtype=float
    ),  # [lmax+1, channels] - output (zeroed by caller)
    grad_affine_bias: wp.array(dtype=float),  # [channels] - output (zeroed by caller)
    mmax: int,
    num_channels: int,
    inv_num_channels: float,
    eps: float,
    has_bias: int,  # 1 if bias present, 0 otherwise
):
    """Compute grad_x_hat, grad_affine_weight, and optionally grad_affine_bias for backward pass with subtract_mean.

    This kernel computes gradients with respect to the mean-subtracted input x_hat.
    The gradient w.r.t. x_hat is written to grad_x. The caller must then apply the
    mean-subtraction chain rule correction at (l=0, m=0, ri=0) in PyTorch:

        grad_x_hat_l0_sum = grad_x[:, 0, 0, :num_channels].sum(dim=-1, keepdim=True)
        grad_x[:, 0, 0, :num_channels] -= grad_x_hat_l0_sum / num_channels

    Launched with wp.launch(dim=(batch, lmax+1, num_channels)).

    Parameters
    ----------
    grad_output : [batch, lmax+1, mmax+1, 2*channels]
        Upstream gradient.
    x : [batch, lmax+1, mmax+1, 2*channels]
        Input from forward pass.
    l0_mean : [batch]
        Pre-computed l=0 channel mean from forward pass.
    norm_stats : [batch]
        Raw sum-of-squares from forward reduce kernel.
    go_dot_o : [batch]
        Inner product of grad_output and output_no_bias from backward reduce kernel.
    affine_weight : [lmax+1, channels]
        Affine scale parameters from forward.
    balance_weight : [lmax+1, mmax+1]
        Degree balancing weights.
    grid_mask : [lmax+1, mmax+1, 2]
        Validity mask (combines m<=l constraint and m=0 imaginary zeroing).
    grad_x : [batch, lmax+1, mmax+1, 2*channels]
        Output: gradient w.r.t. x_hat (before mean-correction). Caller applies correction.
    grad_affine_weight : [lmax+1, channels]
        Output: gradient w.r.t. affine_weight. Pre-zeroed by caller.
    grad_affine_bias : [channels]
        Output: gradient w.r.t. affine_bias (if has_bias=1). Pre-zeroed by caller.
    mmax : int
        Maximum order.
    num_channels : int
        Number of channels.
    inv_num_channels : float
        Pre-computed 1.0 / (2 * num_channels).
    eps : float
        Epsilon for numerical stability.
    has_bias : int
        1 if bias is present (accumulate grad_affine_bias), 0 otherwise.
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

            # Compute x_hat: subtract mean at (l=0, m=0, ri=0)
            x_hat_val = x_val
            if l_idx == 0 and m == 0 and ri == 0:
                x_hat_val = x_val - l0_mean[batch_idx]

            mask_val = grid_mask[l_idx, m, ri]
            bw = balance_weight[l_idx, m]

            # Path A: direct gradient (applies everywhere, masked by grid_mask)
            grad_a = go_val * inv_rms_val * aw * mask_val

            # Path B: indirect through norm_stats (uses x_hat, not x)
            grad_b = (
                inv_num_channels * inv_rms_sq * go_dot_o_val * bw * x_hat_val * mask_val
            )

            # Write grad_x_hat into grad_x (caller applies mean-correction later)
            grad_x[batch_idx, l_idx, m, ri * num_channels + c] = grad_a - grad_b

            # Accumulate grad_affine_weight (atomic — uses x_hat, not x)
            grad_aw_contrib = go_val * x_hat_val * inv_rms_val * mask_val
            wp.atomic_add(grad_affine_weight, l_idx, c, grad_aw_contrib)

            # Accumulate grad_affine_bias at (l=0, m=0, ri=0) if bias is present
            if has_bias == 1 and l_idx == 0 and m == 0 and ri == 0:
                wp.atomic_add(grad_affine_bias, c, go_val)


# =============================================================================
# Backward Kernels for LayerNormSH (l>0)
# =============================================================================


@wp.kernel
def layernormsh_lgt0_backward_reduce(
    grad_output: wp.array4d(dtype=float),  # [batch, lmax, mmax+1, 2*channels]
    output: wp.array4d(dtype=float),  # [batch, lmax, mmax+1, 2*channels]
    go_dot_o: wp.array(dtype=float),  # [batch], pre-zeroed
    mmax: int,
    num_channels: int,
):
    """Compute go_dot_o[b] = sum(grad_output * output) for backward pass of l>0 normalization.

    This kernel operates on the l>0 slice only (excluding l=0 scalar component).
    Launched with wp.launch(dim=(batch, lmax, num_channels), block_dim=num_channels).
    Uses tile reduction for efficient accumulation across channels and atomic add across l-blocks.

    Parameters
    ----------
    grad_output : [batch, lmax, mmax+1, 2*channels]
        Upstream gradient from loss (l>0 slice only).
    output : [batch, lmax, mmax+1, 2*channels]
        Forward pass output for l>0 (saved from forward).
    go_dot_o : [batch]
        Output: inner product per batch. Pre-zeroed by caller.
        Multiple l-blocks write to same batch slot via atomic add.
    mmax : int
        Maximum order.
    num_channels : int
        Number of channels (= block_dim).
    """
    batch_idx, l_idx, c = wp.tid()
    # l_idx is 0-based within l>0 slice, so actual degree is l_idx + 1
    l_actual = l_idx + 1
    num_valid_m = wp.min(l_actual, mmax) + 1

    # Each thread accumulates its per-channel contribution to the inner product
    local_sum = float(0.0)
    for m in range(num_valid_m):
        for ri in range(2):
            if m == 0 and ri == 1:
                continue
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
def layernormsh_lgt0_backward_normalize(
    grad_output: wp.array4d(dtype=float),  # [batch, lmax, mmax+1, 2*channels]
    x_lgt0: wp.array4d(dtype=float),  # [batch, lmax, mmax+1, 2*channels]
    norm_stats: wp.array(dtype=float),  # [batch]
    go_dot_o: wp.array(dtype=float),  # [batch]
    affine_weight: wp.array2d(dtype=float),  # [lmax, channels]
    balance_weight_lgt0: wp.array2d(dtype=float),  # [lmax, mmax+1]
    grid_mask_lgt0: wp.array3d(dtype=float),  # [lmax, mmax+1, 2]
    grad_x: wp.array4d(dtype=float),  # [batch, lmax, mmax+1, 2*channels] - output
    grad_affine_weight: wp.array2d(
        dtype=float
    ),  # [lmax, channels] - output (zeroed by caller)
    mmax: int,
    num_channels: int,
    inv_num_channels: float,
    eps: float,
):
    """Compute grad_x and grad_affine_weight for backward pass of l>0 normalization. Pure SIMT.

    This kernel operates on the l>0 slice only (excluding l=0 scalar component).
    Launched with wp.launch(dim=(batch, lmax, num_channels)).

    Parameters
    ----------
    grad_output : [batch, lmax, mmax+1, 2*channels]
        Upstream gradient (l>0 slice only).
    x_lgt0 : [batch, lmax, mmax+1, 2*channels]
        Input from forward pass (l>0 slice only).
    norm_stats : [batch]
        Raw sum-of-squares from forward reduce kernel.
    go_dot_o : [batch]
        Inner product of grad_output and output from backward reduce kernel.
    affine_weight : [lmax, channels]
        Affine scale parameters from forward (l>0 slice).
    balance_weight_lgt0 : [lmax, mmax+1]
        Degree balancing weights for l>0.
    grid_mask_lgt0 : [lmax, mmax+1, 2]
        Validity mask for l>0 (combines m<=l constraint and m=0 imaginary zeroing).
    grad_x : [batch, lmax, mmax+1, 2*channels]
        Output: gradient w.r.t. input x_lgt0.
    grad_affine_weight : [lmax, channels]
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
            x_val = x_lgt0[batch_idx, l_idx, m, ri * num_channels + c]
            mask_val = grid_mask_lgt0[l_idx, m, ri]
            bw = balance_weight_lgt0[l_idx, m]

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


# =============================================================================
# Backward Kernels for Per-Degree LayerNorm
# =============================================================================


@wp.kernel
def layernorm_grid_backward_reduce(
    grad_output: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    output: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    go_dot_o: wp.array(dtype=float),  # [batch * lmax_p1], pre-zeroed
    lmax_p1: int,
    mmax: int,
    num_channels: int,
):
    """Compute go_dot_o[b, l] = sum(grad_output * output) for backward pass of per-degree norm.

    Launched with wp.launch(dim=(batch, lmax+1, num_channels), block_dim=num_channels).
    Uses tile reduction for efficient accumulation across channels and atomic add to per-degree slots.

    Parameters
    ----------
    grad_output : [batch, lmax+1, mmax+1, 2*channels]
        Upstream gradient from loss.
    output : [batch, lmax+1, mmax+1, 2*channels]
        Forward pass output (saved from forward).
    go_dot_o : [batch * lmax_p1]
        Output: inner product per (batch, l) flattened. Pre-zeroed by caller.
        Each (b, l) thread writes to go_dot_o[b * lmax_p1 + l] via atomic add.
    lmax_p1 : int
        lmax + 1.
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

    # Store result - atomic add to per-degree slot [batch * lmax_p1 + l]
    wp.tile_atomic_add(go_dot_o, s, offset=batch_idx * lmax_p1 + l_idx)


@wp.kernel
def layernorm_grid_backward_normalize(
    grad_output: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    x: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    norm_stats: wp.array2d(dtype=float),  # [batch, lmax+1]
    go_dot_o: wp.array2d(dtype=float),  # [batch, lmax+1]
    affine_weight: wp.array2d(dtype=float),  # [lmax+1, channels]
    per_degree_norm_weight: wp.array2d(dtype=float),  # [lmax+1, mmax+1]
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
    """Compute grad_x and grad_affine_weight for backward pass of per-degree norm. Pure SIMT.

    Launched with wp.launch(dim=(batch, lmax+1, num_channels)).

    Parameters
    ----------
    grad_output : [batch, lmax+1, mmax+1, 2*channels]
        Upstream gradient.
    x : [batch, lmax+1, mmax+1, 2*channels]
        Input from forward pass.
    norm_stats : [batch, lmax+1]
        Per-degree raw sum-of-squares from forward reduce kernel.
    go_dot_o : [batch, lmax+1]
        Per-degree inner product of grad_output and output from backward reduce kernel.
    affine_weight : [lmax+1, channels]
        Affine scale parameters from forward.
    per_degree_norm_weight : [lmax+1, mmax+1]
        Per-degree normalization weights.
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
        Pre-computed 1.0 / num_channels (not 1/(2*num_channels) — per-degree uses 1/C).
    eps : float
        Epsilon for numerical stability.
    """
    batch_idx, l_idx, c = wp.tid()

    # Recompute per-degree inv_rms from forward pass
    inv_rms_val = 1.0 / wp.sqrt(norm_stats[batch_idx, l_idx] * inv_num_channels + eps)
    inv_rms_sq = inv_rms_val * inv_rms_val
    aw = affine_weight[l_idx, c]
    go_dot_o_val = go_dot_o[batch_idx, l_idx]

    for m in range(mmax + 1):
        for ri in range(2):
            go_val = grad_output[batch_idx, l_idx, m, ri * num_channels + c]
            x_val = x[batch_idx, l_idx, m, ri * num_channels + c]
            mask_val = grid_mask[l_idx, m, ri]
            pdnw = per_degree_norm_weight[l_idx, m]

            # Path A: direct gradient (applies everywhere, masked by grid_mask)
            grad_a = go_val * inv_rms_val * aw * mask_val

            # Path B: indirect through norm_stats
            # Only at reduce-valid positions: grid_mask and per_degree_norm_weight product gives correct mask
            grad_b = (
                inv_num_channels * inv_rms_sq * go_dot_o_val * pdnw * x_val * mask_val
            )

            grad_x[batch_idx, l_idx, m, ri * num_channels + c] = grad_a - grad_b

            # Accumulate grad_affine_weight (atomic — threads across b, m, ri contribute)
            grad_aw_contrib = go_val * x_val * inv_rms_val * mask_val
            wp.atomic_add(grad_affine_weight, l_idx, c, grad_aw_contrib)


@wp.kernel
def layernorm_grid_backward_reduce_submean_bias(
    grad_output: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    output: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    affine_bias: wp.array(dtype=float),  # [channels]
    go_dot_o: wp.array(dtype=float),  # [batch * lmax_p1], pre-zeroed
    lmax_p1: int,
    mmax: int,
    num_channels: int,
):
    """Compute go_dot_o[b, l] = sum(grad_output * output_no_bias) for backward pass with bias.

    This kernel is used when the forward pass includes a bias term at (l=0, m=0, ri=0).
    The go_dot_o computation must use output WITHOUT the bias added, so we subtract it inline.

    Launched with wp.launch(dim=(batch, lmax+1, num_channels), block_dim=num_channels).
    Uses tile reduction for efficient accumulation across channels and atomic add to per-degree slots.

    Parameters
    ----------
    grad_output : [batch, lmax+1, mmax+1, 2*channels]
        Upstream gradient from loss.
    output : [batch, lmax+1, mmax+1, 2*channels]
        Forward pass output WITH bias (saved from forward).
    affine_bias : [channels]
        Bias parameters from forward (added at l=0, m=0, ri=0).
    go_dot_o : [batch * lmax_p1]
        Output: inner product per (batch, l) using output_no_bias. Pre-zeroed by caller.
    lmax_p1 : int
        lmax + 1.
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

            # Subtract bias contribution at (l=0, m=0, ri=0) to get output_no_bias
            if l_idx == 0 and m == 0 and ri == 0:
                o = o - affine_bias[c]

            local_sum = local_sum + go * o

    # Cooperative tile reduction across channels within the block
    t = wp.tile(local_sum)
    s = wp.tile_sum(t)

    # Store result - atomic add to per-degree slot [batch * lmax_p1 + l]
    wp.tile_atomic_add(go_dot_o, s, offset=batch_idx * lmax_p1 + l_idx)


@wp.kernel
def layernorm_grid_backward_normalize_submean(
    grad_output: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    x: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels]
    l0_mean: wp.array(dtype=float),  # [batch]
    norm_stats: wp.array2d(dtype=float),  # [batch, lmax+1]
    go_dot_o: wp.array2d(dtype=float),  # [batch, lmax+1]
    affine_weight: wp.array2d(dtype=float),  # [lmax+1, channels]
    per_degree_norm_weight: wp.array2d(dtype=float),  # [lmax+1, mmax+1]
    grid_mask: wp.array3d(dtype=float),  # [lmax+1, mmax+1, 2]
    grad_x: wp.array4d(dtype=float),  # [batch, lmax+1, mmax+1, 2*channels] - output
    grad_affine_weight: wp.array2d(
        dtype=float
    ),  # [lmax+1, channels] - output (zeroed by caller)
    grad_affine_bias: wp.array(dtype=float),  # [channels] - output (zeroed by caller)
    mmax: int,
    num_channels: int,
    inv_num_channels: float,
    eps: float,
    has_bias: int,  # 1 if bias present, 0 otherwise
):
    """Compute grad_x_hat, grad_affine_weight, and optionally grad_affine_bias for backward pass with subtract_mean.

    This kernel computes gradients with respect to the mean-subtracted input x_hat.
    The gradient w.r.t. x_hat is written to grad_x. The caller must then apply the
    mean-subtraction chain rule correction at (l=0, m=0, ri=0) in PyTorch:

        grad_x_hat_l0_sum = grad_x[:, 0, 0, :num_channels].sum(dim=-1, keepdim=True)
        grad_x[:, 0, 0, :num_channels] -= grad_x_hat_l0_sum / num_channels

    Launched with wp.launch(dim=(batch, lmax+1, num_channels)).

    Parameters
    ----------
    grad_output : [batch, lmax+1, mmax+1, 2*channels]
        Upstream gradient.
    x : [batch, lmax+1, mmax+1, 2*channels]
        Input from forward pass.
    l0_mean : [batch]
        Pre-computed l=0 channel mean from forward pass.
    norm_stats : [batch, lmax+1]
        Per-degree raw sum-of-squares from forward reduce kernel.
    go_dot_o : [batch, lmax+1]
        Per-degree inner product of grad_output and output_no_bias from backward reduce kernel.
    affine_weight : [lmax+1, channels]
        Affine scale parameters from forward.
    per_degree_norm_weight : [lmax+1, mmax+1]
        Per-degree normalization weights.
    grid_mask : [lmax+1, mmax+1, 2]
        Validity mask (combines m<=l constraint and m=0 imaginary zeroing).
    grad_x : [batch, lmax+1, mmax+1, 2*channels]
        Output: gradient w.r.t. x_hat (before mean-correction). Caller applies correction.
    grad_affine_weight : [lmax+1, channels]
        Output: gradient w.r.t. affine_weight. Pre-zeroed by caller.
    grad_affine_bias : [channels]
        Output: gradient w.r.t. affine_bias (if has_bias=1). Pre-zeroed by caller.
    mmax : int
        Maximum order.
    num_channels : int
        Number of channels.
    inv_num_channels : float
        Pre-computed 1.0 / num_channels (not 1/(2*num_channels) — per-degree uses 1/C).
    eps : float
        Epsilon for numerical stability.
    has_bias : int
        1 if bias is present (accumulate grad_affine_bias), 0 otherwise.
    """
    batch_idx, l_idx, c = wp.tid()

    # Recompute per-degree inv_rms from forward pass
    inv_rms_val = 1.0 / wp.sqrt(norm_stats[batch_idx, l_idx] * inv_num_channels + eps)
    inv_rms_sq = inv_rms_val * inv_rms_val
    aw = affine_weight[l_idx, c]
    go_dot_o_val = go_dot_o[batch_idx, l_idx]

    for m in range(mmax + 1):
        for ri in range(2):
            go_val = grad_output[batch_idx, l_idx, m, ri * num_channels + c]
            x_val = x[batch_idx, l_idx, m, ri * num_channels + c]

            # Compute x_hat: subtract mean at (l=0, m=0, ri=0)
            x_hat_val = x_val
            if l_idx == 0 and m == 0 and ri == 0:
                x_hat_val = x_val - l0_mean[batch_idx]

            mask_val = grid_mask[l_idx, m, ri]
            pdnw = per_degree_norm_weight[l_idx, m]

            # Path A: direct gradient (applies everywhere, masked by grid_mask)
            grad_a = go_val * inv_rms_val * aw * mask_val

            # Path B: indirect through norm_stats (uses x_hat, not x)
            grad_b = (
                inv_num_channels
                * inv_rms_sq
                * go_dot_o_val
                * pdnw
                * x_hat_val
                * mask_val
            )

            # Write grad_x_hat into grad_x (caller applies mean-correction later)
            grad_x[batch_idx, l_idx, m, ri * num_channels + c] = grad_a - grad_b

            # Accumulate grad_affine_weight (atomic — uses x_hat, not x)
            grad_aw_contrib = go_val * x_hat_val * inv_rms_val * mask_val
            wp.atomic_add(grad_affine_weight, l_idx, c, grad_aw_contrib)

            # Accumulate grad_affine_bias at (l=0, m=0, ri=0) if bias is present
            if has_bias == 1 and l_idx == 0 and m == 0 and ri == 0:
                wp.atomic_add(grad_affine_bias, c, go_val)


# =============================================================================
# Custom Op Wrappers for PyTorch Autograd Integration
# =============================================================================

import torch
from physicsnemo.core.function_spec import FunctionSpec


@torch.library.custom_op("physicsnemo::fused_rmsnorm", mutates_args=())
def fused_rmsnorm(
    x: torch.Tensor,  # [batch, lmax+1, mmax+1, 2, channels] - input
    affine_weight: torch.Tensor,  # [lmax+1, channels]
    affine_bias: torch.Tensor,  # [channels] (may be zeros if unused)
    balance_weight: torch.Tensor,  # [lmax+1, mmax+1]
    grid_mask_3d: torch.Tensor,  # [lmax+1, mmax+1, 2]
    lmax: int,
    mmax: int,
    num_channels: int,
    eps: float,
    subtract_mean: bool,
    has_bias: bool,
) -> torch.Tensor:
    """Forward pass for fused RMSNorm custom op.

    Parameters
    ----------
    x : torch.Tensor
        Input features [batch, lmax+1, mmax+1, 2, channels].
    affine_weight : torch.Tensor
        Affine scale parameters [lmax+1, channels].
    affine_bias : torch.Tensor
        Affine bias parameters [channels] (may be zeros if unused).
    balance_weight : torch.Tensor
        Degree balancing weights [lmax+1, mmax+1].
    grid_mask_3d : torch.Tensor
        Validity mask [lmax+1, mmax+1, 2].
    lmax : int
        Maximum degree.
    mmax : int
        Maximum order.
    num_channels : int
        Number of channels.
    eps : float
        Epsilon for numerical stability.
    subtract_mean : bool
        Whether to subtract l=0 mean.
    has_bias : bool
        Whether to add bias at l=0.

    Returns
    -------
    torch.Tensor
        Normalized output [batch, lmax+1, mmax+1, 2, channels].
    """
    # Extract shapes
    batch_size = x.shape[0]
    lmax_p1 = lmax + 1

    # Reshape input to 4D: [batch, lmax+1, mmax+1, 2*channels]
    x_4d = x.contiguous().view(batch_size, lmax_p1, mmax + 1, 2 * num_channels)

    # Allocate output
    output_4d = torch.empty_like(x_4d)

    # Compute l0_mean if subtract_mean is enabled
    if subtract_mean:
        l0_mean = x[:, 0, 0, 0, :].mean(dim=-1)  # Shape: [batch]

    # Allocate norm_stats
    norm_stats = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

    # Compute inv_num_channels
    inv_num_channels = 1.0 / (2 * num_channels)

    # Get Warp context
    wp_device, wp_stream = FunctionSpec.warp_launch_context(x)

    # Launch kernels
    with wp.ScopedStream(wp_stream):
        # Reduce kernel
        if subtract_mean:
            wp.launch(
                rmsnorm_grid_reduce_submean,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    x_4d.detach(),
                    l0_mean.detach(),
                    norm_stats,
                    balance_weight.detach(),
                    mmax,
                    num_channels,
                ],
                device=wp_device,
                block_dim=num_channels,
            )
        else:
            wp.launch(
                rmsnorm_grid_reduce,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    x_4d.detach(),
                    norm_stats,
                    balance_weight.detach(),
                    mmax,
                    num_channels,
                ],
                device=wp_device,
                block_dim=num_channels,
            )

        # Normalize kernel
        if subtract_mean and has_bias:
            wp.launch(
                rmsnorm_grid_normalize_submean_bias,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    x_4d.detach(),
                    output_4d,
                    l0_mean.detach(),
                    norm_stats.detach(),
                    affine_weight.detach(),
                    affine_bias.detach(),
                    grid_mask_3d.detach(),
                    mmax,
                    num_channels,
                    inv_num_channels,
                    eps,
                ],
                device=wp_device,
            )
        elif subtract_mean:
            wp.launch(
                rmsnorm_grid_normalize_submean,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    x_4d.detach(),
                    output_4d,
                    l0_mean.detach(),
                    norm_stats.detach(),
                    affine_weight.detach(),
                    grid_mask_3d.detach(),
                    mmax,
                    num_channels,
                    inv_num_channels,
                    eps,
                ],
                device=wp_device,
            )
        else:
            wp.launch(
                rmsnorm_grid_normalize,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    x_4d.detach(),
                    output_4d,
                    norm_stats.detach(),
                    affine_weight.detach(),
                    grid_mask_3d.detach(),
                    mmax,
                    num_channels,
                    inv_num_channels,
                    eps,
                ],
                device=wp_device,
            )

    # Reshape back to 5D
    output = output_4d.view(batch_size, lmax_p1, mmax + 1, 2, num_channels)

    return output


@fused_rmsnorm.register_fake
def fused_rmsnorm_fake(
    x,
    affine_weight,
    affine_bias,
    balance_weight,
    grid_mask_3d,
    lmax,
    mmax,
    num_channels,
    eps,
    subtract_mean,
    has_bias,
):
    """Fake implementation for torch.compile."""
    return torch.empty_like(x)


def fused_rmsnorm_setup_context(ctx, inputs, output):
    """Setup context for backward pass.

    Saves tensors and non-tensor attributes needed for backward computation.
    """
    (
        x,
        affine_weight,
        affine_bias,
        balance_weight,
        grid_mask_3d,
        lmax,
        mmax,
        num_channels,
        eps,
        subtract_mean,
        has_bias,
    ) = inputs

    # Save tensors needed for backward
    ctx.save_for_backward(
        x, output, affine_weight, affine_bias, balance_weight, grid_mask_3d
    )

    # Save non-tensor attributes
    ctx.lmax = lmax
    ctx.mmax = mmax
    ctx.num_channels = num_channels
    ctx.eps = eps
    ctx.subtract_mean = subtract_mean
    ctx.has_bias = has_bias


@torch.library.custom_op("physicsnemo::_fused_rmsnorm_backward", mutates_args=())
def _fused_rmsnorm_backward(
    grad_output: torch.Tensor,  # [batch, lmax+1, mmax+1, 2, channels]
    x: torch.Tensor,  # [batch, lmax+1, mmax+1, 2, channels]
    output: torch.Tensor,  # [batch, lmax+1, mmax+1, 2, channels]
    affine_weight: torch.Tensor,  # [lmax+1, channels]
    affine_bias: torch.Tensor,  # [channels]
    balance_weight: torch.Tensor,  # [lmax+1, mmax+1]
    grid_mask_3d: torch.Tensor,  # [lmax+1, mmax+1, 2]
    lmax: int,
    mmax: int,
    num_channels: int,
    eps: float,
    subtract_mean: bool,
    has_bias: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Private custom op for RMSNorm backward kernel launches.

    This op wraps all wp.launch() calls from the backward pass, making them
    compatible with torch.compile by providing a register_fake implementation.

    Parameters
    ----------
    grad_output : torch.Tensor
        Upstream gradient [batch, lmax+1, mmax+1, 2, channels].
    x : torch.Tensor
        Input to the forward pass [batch, lmax+1, mmax+1, 2, channels].
    output : torch.Tensor
        Output from the forward pass [batch, lmax+1, mmax+1, 2, channels].
    affine_weight : torch.Tensor
        Learned affine weights [lmax+1, channels].
    affine_bias : torch.Tensor
        Learned affine bias [channels].
    balance_weight : torch.Tensor
        Balancing weights [lmax+1, mmax+1].
    grid_mask_3d : torch.Tensor
        Grid mask [lmax+1, mmax+1, 2].
    lmax : int
        Maximum l degree.
    mmax : int
        Maximum m order.
    num_channels : int
        Number of channels.
    eps : float
        Epsilon for numerical stability.
    subtract_mean : bool
        Whether mean subtraction was applied in forward.
    has_bias : bool
        Whether bias is present.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        - grad_x: Gradient w.r.t. input [batch, lmax+1, mmax+1, 2, channels]
        - grad_affine_weight: Gradient w.r.t. affine weights [lmax+1, channels]
        - grad_affine_bias: Gradient w.r.t. affine bias [channels]

    Notes
    -----
    This private op does NOT apply the mean-subtraction chain rule correction.
    That correction is applied in the outer backward function using pure PyTorch ops.
    """
    # Extract shapes
    batch_size = x.shape[0]
    lmax_p1 = lmax + 1

    # Compute inv_num_channels
    inv_num_channels = 1.0 / (2 * num_channels)

    # Reshape to 4D
    x_4d = x.contiguous().view(batch_size, lmax_p1, mmax + 1, 2 * num_channels)
    output_4d = output.contiguous().view(
        batch_size, lmax_p1, mmax + 1, 2 * num_channels
    )
    grad_output_4d = grad_output.contiguous().view(
        batch_size, lmax_p1, mmax + 1, 2 * num_channels
    )

    # Allocate output tensors
    grad_x_4d = torch.empty_like(x_4d)
    go_dot_o = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
    grad_affine_weight = torch.zeros(
        lmax_p1, num_channels, device=x.device, dtype=x.dtype
    )
    grad_affine_bias = torch.zeros(num_channels, device=x.device, dtype=x.dtype)

    # Recompute l0_mean if subtract_mean is enabled
    if subtract_mean:
        l0_mean = x[:, 0, 0, 0, :].mean(dim=-1)  # Shape: [batch]

    # Get Warp context
    wp_device, wp_stream = FunctionSpec.warp_launch_context(x)

    # Launch kernels
    with wp.ScopedStream(wp_stream):
        # Backward reduce kernel (compute go_dot_o)
        if subtract_mean and has_bias:
            wp.launch(
                rmsnorm_backward_reduce_submean_bias,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    grad_output_4d.detach(),
                    output_4d.detach(),
                    affine_bias.detach(),
                    go_dot_o,
                    mmax,
                    num_channels,
                ],
                device=wp_device,
                block_dim=num_channels,
            )
        else:
            wp.launch(
                rmsnorm_backward_reduce,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    grad_output_4d.detach(),
                    output_4d.detach(),
                    go_dot_o,
                    mmax,
                    num_channels,
                ],
                device=wp_device,
                block_dim=num_channels,
            )

        # Recompute norm_stats for backward normalize kernel
        norm_stats = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        if subtract_mean:
            wp.launch(
                rmsnorm_grid_reduce_submean,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    x_4d.detach(),
                    l0_mean.detach(),
                    norm_stats,
                    balance_weight.detach(),
                    mmax,
                    num_channels,
                ],
                device=wp_device,
                block_dim=num_channels,
            )
        else:
            wp.launch(
                rmsnorm_grid_reduce,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    x_4d.detach(),
                    norm_stats,
                    balance_weight.detach(),
                    mmax,
                    num_channels,
                ],
                device=wp_device,
                block_dim=num_channels,
            )

        # Backward normalize kernel
        if subtract_mean:
            wp.launch(
                rmsnorm_backward_normalize_submean,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    grad_output_4d.detach(),
                    x_4d.detach(),
                    l0_mean.detach(),
                    norm_stats.detach(),
                    go_dot_o.detach(),
                    affine_weight.detach(),
                    balance_weight.detach(),
                    grid_mask_3d.detach(),
                    grad_x_4d,
                    grad_affine_weight,
                    grad_affine_bias,
                    mmax,
                    num_channels,
                    inv_num_channels,
                    eps,
                    1 if has_bias else 0,
                ],
                device=wp_device,
            )
        else:
            wp.launch(
                rmsnorm_backward_normalize,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    grad_output_4d.detach(),
                    x_4d.detach(),
                    norm_stats.detach(),
                    go_dot_o.detach(),
                    affine_weight.detach(),
                    balance_weight.detach(),
                    grid_mask_3d.detach(),
                    grad_x_4d,
                    grad_affine_weight,
                    mmax,
                    num_channels,
                    inv_num_channels,
                    eps,
                ],
                device=wp_device,
            )

    # Reshape grad_x back to 5D
    grad_x = grad_x_4d.view(batch_size, lmax_p1, mmax + 1, 2, num_channels)

    return grad_x, grad_affine_weight, grad_affine_bias


@_fused_rmsnorm_backward.register_fake
def _fused_rmsnorm_backward_fake(
    grad_output,
    x,
    output,
    affine_weight,
    affine_bias,
    balance_weight,
    grid_mask_3d,
    lmax,
    mmax,
    num_channels,
    eps,
    subtract_mean,
    has_bias,
):
    """Fake implementation for torch.compile compatibility.

    Returns tensors with correct shapes and dtypes without executing kernels.
    """
    lmax_p1 = lmax + 1
    return (
        torch.empty_like(x),
        torch.empty(lmax_p1, num_channels, device=x.device, dtype=x.dtype),
        torch.empty(num_channels, device=x.device, dtype=x.dtype),
    )


def fused_rmsnorm_backward(ctx, grad_output):
    """Backward pass for fused RMSNorm custom op.

    Parameters
    ----------
    ctx : torch.autograd.function.BackwardCFunction
        Context with saved tensors and attributes.
    grad_output : torch.Tensor
        Upstream gradient [batch, lmax+1, mmax+1, 2, channels].

    Returns
    -------
    tuple
        Gradients for each input: (grad_x, grad_affine_weight, grad_affine_bias,
        None, None, None, None, None, None, None, None).
    """
    # Unpack saved tensors
    x, output, affine_weight, affine_bias, balance_weight, grid_mask_3d = (
        ctx.saved_tensors
    )

    # Unpack attributes
    lmax = ctx.lmax
    mmax = ctx.mmax
    num_channels = ctx.num_channels
    eps = ctx.eps
    subtract_mean = ctx.subtract_mean
    has_bias = ctx.has_bias

    # Call private custom op for kernel launches
    grad_x, grad_affine_weight, grad_affine_bias = _fused_rmsnorm_backward(
        grad_output,
        x,
        output,
        affine_weight,
        affine_bias,
        balance_weight,
        grid_mask_3d,
        lmax,
        mmax,
        num_channels,
        eps,
        subtract_mean,
        has_bias,
    )

    # Apply mean-subtraction chain rule correction (pure PyTorch, traceable)
    if subtract_mean:
        grad_x_l0_real = grad_x[:, 0, 0, 0, :]  # [batch, channels]
        grad_x_l0_sum = grad_x_l0_real.sum(dim=-1, keepdim=True)  # [batch, 1]
        correction = grad_x_l0_sum / num_channels
        # Clone to avoid modifying custom op output in-place
        grad_x = grad_x.clone()
        grad_x[:, 0, 0, 0, :] = grad_x_l0_real - correction

    return (
        grad_x,
        grad_affine_weight,
        grad_affine_bias if has_bias else None,
        None,  # grad for balance_weight
        None,  # grad for grid_mask_3d
        None,  # grad for lmax
        None,  # grad for mmax
        None,  # grad for num_channels
        None,  # grad for eps
        None,  # grad for subtract_mean
        None,  # grad for has_bias
    )


# Register autograd
fused_rmsnorm.register_autograd(
    fused_rmsnorm_backward, setup_context=fused_rmsnorm_setup_context
)


@torch.library.custom_op("physicsnemo::fused_layernormsh_lgt0", mutates_args=())
def fused_layernormsh_lgt0(
    x_lgt0: torch.Tensor,  # [batch, lmax, mmax+1, 2, channels] - l>0 slice
    affine_weight: torch.Tensor,  # [lmax, channels]
    balance_weight_lgt0: torch.Tensor,  # [lmax, mmax+1]
    grid_mask_lgt0: torch.Tensor,  # [lmax, mmax+1, 2]
    lmax: int,  # actual lmax (not lmax+1)
    mmax: int,
    num_channels: int,
    eps: float,
) -> torch.Tensor:
    """Fused LayerNormSH for l>0 spherical harmonics.

    Parameters
    ----------
    x_lgt0 : torch.Tensor
        Input features for l>0 [batch, lmax, mmax+1, 2, channels]
    affine_weight : torch.Tensor
        Affine scale weights [lmax, channels]
    balance_weight_lgt0 : torch.Tensor
        Degree balancing weights for l>0 [lmax, mmax+1]
    grid_mask_lgt0 : torch.Tensor
        Grid mask for l>0 [lmax, mmax+1, 2]
    lmax : int
        Maximum degree (not lmax+1)
    mmax : int
        Maximum order
    num_channels : int
        Number of channels
    eps : float
        Epsilon for numerical stability

    Returns
    -------
    torch.Tensor
        Normalized features [batch, lmax, mmax+1, 2, channels]
    """
    batch_size = x_lgt0.shape[0]

    # Reshape to 4D for kernel processing
    x_lgt0_4d = x_lgt0.contiguous().view(batch_size, lmax, mmax + 1, 2 * num_channels)
    output_4d = torch.empty_like(x_lgt0_4d)
    norm_stats = torch.zeros(batch_size, device=x_lgt0.device, dtype=x_lgt0.dtype)

    # Pre-compute inv_num_channels
    inv_num_channels = 1.0 / (2 * num_channels)

    # Get Warp context
    wp_device, wp_stream = FunctionSpec.warp_launch_context(x_lgt0)

    # Launch Warp kernels
    with wp.ScopedStream(wp_stream):
        # Reduce kernel
        wp.launch(
            kernel=layernormsh_lgt0_reduce,
            dim=(batch_size, lmax, num_channels),
            inputs=[
                x_lgt0_4d.detach(),
                norm_stats,
                balance_weight_lgt0.detach(),
                mmax,
                num_channels,
            ],
            block_dim=num_channels,
        )

        # Normalize kernel
        wp.launch(
            kernel=layernormsh_lgt0_normalize,
            dim=(batch_size, lmax, num_channels),
            inputs=[
                x_lgt0_4d.detach(),
                output_4d,
                norm_stats.detach(),
                affine_weight.detach(),
                grid_mask_lgt0.detach(),
                mmax,
                num_channels,
                inv_num_channels,
                eps,
            ],
        )

    # Reshape back to 5D
    return output_4d.view(batch_size, lmax, mmax + 1, 2, num_channels)


@fused_layernormsh_lgt0.register_fake
def _(
    x_lgt0: torch.Tensor,
    affine_weight: torch.Tensor,
    balance_weight_lgt0: torch.Tensor,
    grid_mask_lgt0: torch.Tensor,
    lmax: int,
    mmax: int,
    num_channels: int,
    eps: float,
) -> torch.Tensor:
    """Register fake implementation for shape inference."""
    return torch.empty_like(x_lgt0)


def fused_layernormsh_lgt0_setup_context(
    ctx, inputs, output
) -> torch.autograd.graph.saved_tensors_hooks:
    """Setup context for backward pass."""
    (
        x_lgt0,
        affine_weight,
        balance_weight_lgt0,
        grid_mask_lgt0,
        lmax,
        mmax,
        num_channels,
        eps,
    ) = inputs

    # Save tensors and attributes
    ctx.save_for_backward(
        x_lgt0, output, affine_weight, balance_weight_lgt0, grid_mask_lgt0
    )
    ctx.lmax = lmax
    ctx.mmax = mmax
    ctx.num_channels = num_channels
    ctx.eps = eps


@torch.library.custom_op(
    "physicsnemo::_fused_layernormsh_lgt0_backward", mutates_args=()
)
def _fused_layernormsh_lgt0_backward(
    grad_output: torch.Tensor,  # [batch, lmax, mmax+1, 2, channels]
    x_lgt0: torch.Tensor,  # [batch, lmax, mmax+1, 2, channels]
    output: torch.Tensor,  # [batch, lmax, mmax+1, 2, channels]
    affine_weight: torch.Tensor,  # [lmax, channels]
    balance_weight_lgt0: torch.Tensor,  # [lmax, mmax+1]
    grid_mask_lgt0: torch.Tensor,  # [lmax, mmax+1, 2]
    lmax: int,
    mmax: int,
    num_channels: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Private custom op for LayerNormSH l>0 backward kernel launches.

    This private op wraps the Warp kernel launches for the backward pass,
    making them torch.compile compatible. It should only be called by
    fused_layernormsh_lgt0_backward.

    Args:
        grad_output: Gradient w.r.t. output [batch, lmax, mmax+1, 2, channels]
        x_lgt0: Input tensor [batch, lmax, mmax+1, 2, channels]
        output: Forward output [batch, lmax, mmax+1, 2, channels]
        affine_weight: Affine weight [lmax, channels]
        balance_weight_lgt0: Balance weights [lmax, mmax+1]
        grid_mask_lgt0: Grid mask [lmax, mmax+1, 2]
        lmax: Maximum degree
        mmax: Maximum order
        num_channels: Number of channels
        eps: Epsilon for numerical stability

    Returns:
        Tuple of (grad_x, grad_affine_weight)
    """
    batch_size = x_lgt0.shape[0]

    # Reshape to 4D for kernel processing
    x_lgt0_4d = x_lgt0.contiguous().view(batch_size, lmax, mmax + 1, 2 * num_channels)
    output_4d = output.contiguous().view(batch_size, lmax, mmax + 1, 2 * num_channels)
    grad_output_4d = grad_output.contiguous().view(
        batch_size, lmax, mmax + 1, 2 * num_channels
    )

    # Allocate gradient tensors
    grad_x_4d = torch.empty_like(x_lgt0_4d)
    go_dot_o = torch.zeros(batch_size, device=x_lgt0.device, dtype=x_lgt0.dtype)
    grad_affine_weight = torch.zeros(
        lmax, num_channels, device=x_lgt0.device, dtype=x_lgt0.dtype
    )

    # Recompute norm_stats for backward pass
    norm_stats = torch.zeros(batch_size, device=x_lgt0.device, dtype=x_lgt0.dtype)

    # Pre-compute inv_num_channels
    inv_num_channels = 1.0 / (2 * num_channels)

    # Get Warp context
    wp_device, wp_stream = FunctionSpec.warp_launch_context(x_lgt0)

    # Launch Warp kernels for backward pass
    with wp.ScopedStream(wp_stream):
        # Backward reduce kernel
        wp.launch(
            kernel=layernormsh_lgt0_backward_reduce,
            dim=(batch_size, lmax, num_channels),
            inputs=[
                grad_output_4d.detach(),
                output_4d.detach(),
                go_dot_o,
                mmax,
                num_channels,
            ],
            block_dim=num_channels,
        )

        # Recompute forward reduce for norm_stats
        wp.launch(
            kernel=layernormsh_lgt0_reduce,
            dim=(batch_size, lmax, num_channels),
            inputs=[
                x_lgt0_4d.detach(),
                norm_stats,
                balance_weight_lgt0.detach(),
                mmax,
                num_channels,
            ],
            block_dim=num_channels,
        )

        # Backward normalize kernel
        wp.launch(
            kernel=layernormsh_lgt0_backward_normalize,
            dim=(batch_size, lmax, num_channels),
            inputs=[
                grad_output_4d.detach(),
                x_lgt0_4d.detach(),
                norm_stats.detach(),
                go_dot_o.detach(),
                affine_weight.detach(),
                balance_weight_lgt0.detach(),
                grid_mask_lgt0.detach(),
                grad_x_4d,
                grad_affine_weight,
                mmax,
                num_channels,
                inv_num_channels,
                eps,
            ],
        )

    # Reshape grad_x back to 5D
    grad_x = grad_x_4d.view(batch_size, lmax, mmax + 1, 2, num_channels)

    return (grad_x, grad_affine_weight)


@_fused_layernormsh_lgt0_backward.register_fake
def _fused_layernormsh_lgt0_backward_fake(
    grad_output,
    x_lgt0,
    output,
    affine_weight,
    balance_weight_lgt0,
    grid_mask_lgt0,
    lmax,
    mmax,
    num_channels,
    eps,
):
    """Fake implementation for torch.compile."""
    return (
        torch.empty_like(x_lgt0),
        torch.empty(lmax, num_channels, device=x_lgt0.device, dtype=x_lgt0.dtype),
    )


def fused_layernormsh_lgt0_backward(ctx, grad_output: torch.Tensor):
    """Backward pass for fused_layernormsh_lgt0."""
    # Unpack saved tensors and attributes
    x_lgt0, output, affine_weight, balance_weight_lgt0, grid_mask_lgt0 = (
        ctx.saved_tensors
    )
    lmax = ctx.lmax
    mmax = ctx.mmax
    num_channels = ctx.num_channels
    eps = ctx.eps

    grad_x, grad_affine_weight = _fused_layernormsh_lgt0_backward(
        grad_output,
        x_lgt0,
        output,
        affine_weight,
        balance_weight_lgt0,
        grid_mask_lgt0,
        lmax,
        mmax,
        num_channels,
        eps,
    )

    # Return gradients for all 8 inputs (None for non-tensor inputs)
    return (grad_x, grad_affine_weight, None, None, None, None, None, None)


# Register autograd
fused_layernormsh_lgt0.register_autograd(
    fused_layernormsh_lgt0_backward, setup_context=fused_layernormsh_lgt0_setup_context
)


# ============================================================================
# Custom Op: fused_layernorm (per-degree LayerNorm)
# ============================================================================


@torch.library.custom_op("physicsnemo::fused_layernorm", mutates_args=())
def fused_layernorm(
    x: torch.Tensor,  # [batch, lmax+1, mmax+1, 2, channels]
    affine_weight: torch.Tensor,  # [lmax+1, channels]
    affine_bias: torch.Tensor,  # [channels] (zeros if unused)
    per_degree_norm_weight: torch.Tensor,  # [lmax+1, mmax+1]
    grid_mask_3d: torch.Tensor,  # [lmax+1, mmax+1, 2]
    lmax: int,
    mmax: int,
    num_channels: int,
    eps: float,
    subtract_mean: bool,
    has_bias: bool,
) -> torch.Tensor:
    """Per-degree equivariant LayerNorm forward pass.

    Normalizes each degree l independently using fused Warp kernels.

    Parameters
    ----------
    x : torch.Tensor
        Input features [batch, lmax+1, mmax+1, 2, channels].
    affine_weight : torch.Tensor
        Affine weights [lmax+1, channels].
    affine_bias : torch.Tensor
        Affine bias [channels] (zeros if unused).
    per_degree_norm_weight : torch.Tensor
        Per-degree normalization weights [lmax+1, mmax+1].
    grid_mask_3d : torch.Tensor
        Grid mask [lmax+1, mmax+1, 2].
    lmax : int
        Maximum degree.
    mmax : int
        Maximum order.
    num_channels : int
        Number of channels.
    eps : float
        Epsilon for numerical stability.
    subtract_mean : bool
        Whether to subtract l=0 mean.
    has_bias : bool
        Whether to apply bias.

    Returns
    -------
    torch.Tensor
        Normalized features [batch, lmax+1, mmax+1, 2, channels].
    """
    batch_size = x.shape[0]
    lmax_p1 = lmax + 1

    # Reshape to 4D for Warp kernel compatibility
    x_4d = x.contiguous().view(batch_size, lmax_p1, mmax + 1, 2 * num_channels)
    output_4d = torch.empty_like(x_4d)

    # Compute l=0 mean if needed
    l0_mean = None
    if subtract_mean:
        l0_mean = x[:, 0, 0, 0, :].mean(dim=-1)  # [batch]

    # Allocate norm_stats (per-degree, 2D but flattened for atomic ops)
    norm_stats = torch.zeros(batch_size, lmax_p1, device=x.device, dtype=x.dtype)

    inv_num_channels = 1.0 / num_channels

    # Get warp context
    wp_device, wp_stream = FunctionSpec.warp_launch_context(x)

    with wp.ScopedStream(wp_stream):
        # Pass 1: Reduce (compute norm statistics per degree)
        if subtract_mean:
            wp.launch(
                layernorm_grid_reduce_submean,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    x_4d.detach(),
                    l0_mean.detach(),
                    norm_stats.view(-1),  # Flatten to 1D for atomic accumulation
                    per_degree_norm_weight.detach(),
                    lmax_p1,
                    mmax,
                    num_channels,
                    inv_num_channels,
                    eps,
                ],
                block_dim=num_channels,
                device=wp_device,
            )
        else:
            wp.launch(
                layernorm_grid_reduce,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    x_4d.detach(),
                    norm_stats.view(-1),  # Flatten to 1D for atomic accumulation
                    per_degree_norm_weight.detach(),
                    lmax_p1,
                    mmax,
                    num_channels,
                    inv_num_channels,
                    eps,
                ],
                block_dim=num_channels,
                device=wp_device,
            )

        # Pass 2: Normalize (apply normalization)
        if subtract_mean and has_bias:
            wp.launch(
                layernorm_grid_normalize_submean_bias,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    x_4d.detach(),
                    output_4d,
                    l0_mean.detach(),
                    norm_stats.detach(),
                    affine_weight.detach(),
                    affine_bias.detach(),
                    grid_mask_3d.detach(),
                    mmax,
                    num_channels,
                    inv_num_channels,
                    eps,
                ],
                device=wp_device,
            )
        elif subtract_mean:
            wp.launch(
                layernorm_grid_normalize_submean,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    x_4d.detach(),
                    output_4d,
                    l0_mean.detach(),
                    norm_stats.detach(),
                    affine_weight.detach(),
                    grid_mask_3d.detach(),
                    mmax,
                    num_channels,
                    inv_num_channels,
                    eps,
                ],
                device=wp_device,
            )
        else:
            wp.launch(
                layernorm_grid_normalize,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    x_4d.detach(),
                    output_4d,
                    norm_stats.detach(),
                    affine_weight.detach(),
                    grid_mask_3d.detach(),
                    mmax,
                    num_channels,
                    inv_num_channels,
                    eps,
                ],
                device=wp_device,
            )

    # Reshape back to 5D
    return output_4d.view(batch_size, lmax_p1, mmax + 1, 2, num_channels)


@fused_layernorm.register_fake
def _(
    x: torch.Tensor,
    affine_weight: torch.Tensor,
    affine_bias: torch.Tensor,
    per_degree_norm_weight: torch.Tensor,
    grid_mask_3d: torch.Tensor,
    lmax: int,
    mmax: int,
    num_channels: int,
    eps: float,
    subtract_mean: bool,
    has_bias: bool,
) -> torch.Tensor:
    """Fake implementation for meta/tracing."""
    return torch.empty_like(x)


def fused_layernorm_setup_context(ctx, inputs, output) -> None:
    """Save tensors and attributes for backward pass."""
    (
        x,
        affine_weight,
        affine_bias,
        per_degree_norm_weight,
        grid_mask_3d,
        lmax,
        mmax,
        num_channels,
        eps,
        subtract_mean,
        has_bias,
    ) = inputs

    # Save tensors
    ctx.save_for_backward(
        x, output, affine_weight, affine_bias, per_degree_norm_weight, grid_mask_3d
    )

    # Save attributes
    ctx.lmax = lmax
    ctx.mmax = mmax
    ctx.num_channels = num_channels
    ctx.eps = eps
    ctx.subtract_mean = subtract_mean
    ctx.has_bias = has_bias


@torch.library.custom_op("physicsnemo::_fused_layernorm_backward", mutates_args=())
def _fused_layernorm_backward(
    grad_output: torch.Tensor,  # [batch, lmax+1, mmax+1, 2, channels]
    x: torch.Tensor,  # [batch, lmax+1, mmax+1, 2, channels]
    output: torch.Tensor,  # [batch, lmax+1, mmax+1, 2, channels]
    affine_weight: torch.Tensor,  # [lmax+1, channels]
    affine_bias: torch.Tensor,  # [channels]
    per_degree_norm_weight: torch.Tensor,  # [lmax+1, mmax+1]
    grid_mask_3d: torch.Tensor,  # [lmax+1, mmax+1, 2]
    lmax: int,
    mmax: int,
    num_channels: int,
    eps: float,
    subtract_mean: bool,
    has_bias: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Private custom op: per-degree LayerNorm backward kernel launches.

    This op wraps all Warp kernel launches for the per-degree LayerNorm backward pass.
    It does NOT apply the mean-subtraction chain rule correction, which stays in the
    outer backward function as pure PyTorch code (torch.compile-traceable).

    Returns:
        grad_x: [batch, lmax+1, mmax+1, 2, channels]
        grad_affine_weight: [lmax+1, channels]
        grad_affine_bias: [channels]
    """
    batch_size = x.shape[0]
    lmax_p1 = lmax + 1
    inv_num_channels = 1.0 / num_channels

    # Reshape to 4D
    x_4d = x.contiguous().view(batch_size, lmax_p1, mmax + 1, 2 * num_channels)
    output_4d = output.contiguous().view(
        batch_size, lmax_p1, mmax + 1, 2 * num_channels
    )
    grad_output_4d = grad_output.contiguous().view(
        batch_size, lmax_p1, mmax + 1, 2 * num_channels
    )

    # Allocate gradients
    grad_x_4d = torch.empty_like(x_4d)
    go_dot_o = torch.zeros(batch_size, lmax_p1, device=x.device, dtype=x.dtype)
    grad_affine_weight = torch.zeros(
        lmax_p1, num_channels, device=x.device, dtype=x.dtype
    )
    grad_affine_bias = torch.zeros(num_channels, device=x.device, dtype=x.dtype)

    # Recompute l=0 mean if needed
    l0_mean = None
    if subtract_mean:
        l0_mean = x[:, 0, 0, 0, :].mean(dim=-1)  # [batch]

    # Get warp context
    wp_device, wp_stream = FunctionSpec.warp_launch_context(x)

    with wp.ScopedStream(wp_stream):
        # Backward reduce: compute go_dot_o
        if subtract_mean and has_bias:
            wp.launch(
                layernorm_grid_backward_reduce_submean_bias,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    grad_output_4d.detach(),
                    output_4d.detach(),
                    affine_bias.detach(),
                    go_dot_o.view(-1),
                    lmax_p1,
                    mmax,
                    num_channels,
                ],
                block_dim=num_channels,
                device=wp_device,
            )
        else:
            wp.launch(
                layernorm_grid_backward_reduce,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    grad_output_4d.detach(),
                    output_4d.detach(),
                    go_dot_o.view(-1),
                    lmax_p1,
                    mmax,
                    num_channels,
                ],
                block_dim=num_channels,
                device=wp_device,
            )

        # Recompute norm_stats
        norm_stats = torch.zeros(batch_size, lmax_p1, device=x.device, dtype=x.dtype)
        if subtract_mean:
            wp.launch(
                layernorm_grid_reduce_submean,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    x_4d.detach(),
                    l0_mean.detach(),
                    norm_stats.view(-1),
                    per_degree_norm_weight.detach(),
                    lmax_p1,
                    mmax,
                    num_channels,
                    inv_num_channels,
                    eps,
                ],
                block_dim=num_channels,
                device=wp_device,
            )
        else:
            wp.launch(
                layernorm_grid_reduce,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    x_4d.detach(),
                    norm_stats.view(-1),
                    per_degree_norm_weight.detach(),
                    lmax_p1,
                    mmax,
                    num_channels,
                    inv_num_channels,
                    eps,
                ],
                block_dim=num_channels,
                device=wp_device,
            )

        # Backward normalize
        if subtract_mean:
            wp.launch(
                layernorm_grid_backward_normalize_submean,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    grad_output_4d.detach(),
                    x_4d.detach(),
                    l0_mean.detach(),
                    norm_stats.detach(),
                    go_dot_o.detach(),
                    affine_weight.detach(),
                    per_degree_norm_weight.detach(),
                    grid_mask_3d.detach(),
                    grad_x_4d,
                    grad_affine_weight,
                    grad_affine_bias,
                    mmax,
                    num_channels,
                    inv_num_channels,
                    eps,
                    1 if has_bias else 0,
                ],
                device=wp_device,
            )
        else:
            wp.launch(
                layernorm_grid_backward_normalize,
                dim=(batch_size, lmax_p1, num_channels),
                inputs=[
                    grad_output_4d.detach(),
                    x_4d.detach(),
                    norm_stats.detach(),
                    go_dot_o.detach(),
                    affine_weight.detach(),
                    per_degree_norm_weight.detach(),
                    grid_mask_3d.detach(),
                    grad_x_4d,
                    grad_affine_weight,
                    mmax,
                    num_channels,
                    inv_num_channels,
                    eps,
                ],
                device=wp_device,
            )

    # Reshape grad_x back to 5D
    grad_x = grad_x_4d.view(batch_size, lmax_p1, mmax + 1, 2, num_channels)

    return grad_x, grad_affine_weight, grad_affine_bias


@_fused_layernorm_backward.register_fake
def _fused_layernorm_backward_fake(
    grad_output,
    x,
    output,
    affine_weight,
    affine_bias,
    per_degree_norm_weight,
    grid_mask_3d,
    lmax,
    mmax,
    num_channels,
    eps,
    subtract_mean,
    has_bias,
):
    """Fake implementation for meta/tracing."""
    lmax_p1 = lmax + 1
    return (
        torch.empty_like(x),
        torch.empty(lmax_p1, num_channels, device=x.device, dtype=x.dtype),
        torch.empty(num_channels, device=x.device, dtype=x.dtype),
    )


def fused_layernorm_backward(ctx, grad_output: torch.Tensor) -> tuple:
    """Backward pass for fused_layernorm."""
    # Unpack saved tensors
    x, output, affine_weight, affine_bias, per_degree_norm_weight, grid_mask_3d = (
        ctx.saved_tensors
    )

    # Unpack attributes
    lmax = ctx.lmax
    mmax = ctx.mmax
    num_channels = ctx.num_channels
    eps = ctx.eps
    subtract_mean = ctx.subtract_mean
    has_bias = ctx.has_bias

    # Call private custom op to execute kernel launches
    grad_x, grad_affine_weight, grad_affine_bias = _fused_layernorm_backward(
        grad_output,
        x,
        output,
        affine_weight,
        affine_bias,
        per_degree_norm_weight,
        grid_mask_3d,
        lmax,
        mmax,
        num_channels,
        eps,
        subtract_mean,
        has_bias,
    )

    # Apply mean-subtraction chain rule correction (pure PyTorch, traceable)
    if subtract_mean:
        grad_x_l0_real = grad_x[:, 0, 0, 0, :]  # [batch, channels]
        grad_x_l0_sum = grad_x_l0_real.sum(dim=-1, keepdim=True)  # [batch, 1]
        correction = grad_x_l0_sum / num_channels
        grad_x = grad_x.clone()
        grad_x[:, 0, 0, 0, :] = grad_x_l0_real - correction

    # Return gradients for all 11 inputs (None for non-tensor and immutable inputs)
    return (
        grad_x,
        grad_affine_weight,
        grad_affine_bias if has_bias else None,
        None,  # per_degree_norm_weight
        None,  # grid_mask_3d
        None,  # lmax
        None,  # mmax
        None,  # num_channels
        None,  # eps
        None,  # subtract_mean
        None,  # has_bias
    )


# Register autograd
fused_layernorm.register_autograd(
    fused_layernorm_backward, setup_context=fused_layernorm_setup_context
)
