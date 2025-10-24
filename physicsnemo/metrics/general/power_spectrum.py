# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

from typing import Tuple

import torch


def _batch_weighted_histogram(
    data_tensor: torch.Tensor, num_classes: int = -1, weights: torch.Tensor = None
) -> torch.Tensor:
    """
    Computes (optionally weighted) histogram of values in a Tensor, preserving the leading dimensions

    Args:
        data_tensor: torch.Tensor
            a D1 x ... x D_n torch.LongTensor
        num_classes: int, optional
            The number of classes/bins present in data.
            If not provided (set to -1), tensor.max() + 1 is used
        weights: torch.Tensor, optional
            If provided, use values in weights to produce a weighted histogram

    Returns:
        hist: torch.Tensor
            A D1 x ... x D_{n-1} x num_classes 'result' torch.LongTensor,
            containing (weighted) histograms of the last dimension D_n of tensor,
    """
    maxd = data_tensor.max()
    nc = (maxd + 1) if num_classes <= 0 else num_classes
    hist = torch.zeros(
        (*data_tensor.shape[:-1], nc),
        dtype=data_tensor.dtype,
        device=data_tensor.device,
    )
    if weights is not None:
        wts = weights
    else:
        wts = torch.tensor(1, dtype=hist.dtype, device=hist.device).expand(
            data_tensor.shape
        )
    hist.scatter_add_(-1, ((data_tensor * nc) // (maxd + 1)).long(), wts)
    return hist


def power_spectrum(x: torch.Tensor) -> Tuple[torch.Tensor]:
    """Compute the wavenumber-averaged power spectrum of an input tensor x,
    preserving the leading D - 2 dimensions for an input with D dimensions.
    This routine will compute the 2D power from FFT coefficients, then perform
    azimuthal averaging to get the 1D power spectrum as a function of total
    wavenumber.

    Args:
        x: torch.Tensor
            Input tensor with at least three dimensions; the final two dims are
            assumed to be the height and width of a regular 2D spatial domain
            Shape: D1 x D2 x ... x h x w

    Returns:
        k: torch.Tensor
            Centers of the total wavenumber bins after azimuthal averaging
            Number of bins is min(h//2, w//2) - 1, linearly spaced
        power: torch.Tensor
            Azimuthally averaged 1D power spectrum
            Shape: D1 x ... x D_n-2 x min(h//2, w//2) - 1
    """

    leading, (h, w) = x.shape[:-2], x.shape[-2:]
    x = x.reshape(-1, h, w)
    batch = x.shape[0]

    # 2D power
    pwr = torch.fft.fftn(x, dim=(-2, -1), norm="ortho").abs() ** 2
    pwr = torch.fft.fftshift(pwr, dim=(-2, -1)).to(torch.float32)

    # Azimuthal average
    xx, yy = torch.meshgrid(
        torch.arange(h, device=pwr.device),
        torch.arange(w, device=pwr.device),
        indexing="ij",
    )
    k = torch.hypot(xx - h // 2, yy - w / 2).to(torch.float32)

    sort = torch.argsort(k.flatten())
    k_sort = k.flatten()[sort]
    pwr_sort = pwr.reshape(batch, -1)[:, sort]

    nbins = min(h // 2, w // 2)
    k_bins = torch.linspace(0, k_sort.max() + 1, nbins)
    k_bin_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    k_sort_stack = torch.tile(k_sort, dims=(batch, 1))

    pwr_binned = _batch_weighted_histogram(
        k_sort_stack, weights=pwr_sort, num_classes=nbins - 1
    )
    count_binned = _batch_weighted_histogram(k_sort_stack, num_classes=nbins - 1)

    power = pwr_binned / count_binned
    k = k_bin_centers

    power = power.reshape(*leading, nbins - 1)

    return k, power
