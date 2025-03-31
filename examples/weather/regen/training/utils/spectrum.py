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

import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def batch_histogram(data_tensor, num_classes=-1, weights=None):
    """
    From. https://github.com/pytorch/pytorch/issues/99719#issuecomment-1760112194
    Computes histograms of integral values, even if in batches (as opposed to torch.histc and torch.histogram).
    Arguments:
        data_tensor: a D1 x ... x D_n torch.LongTensor
        num_classes (optional): the number of classes present in data.
                                If not provided, tensor.max() + 1 is used (an error is thrown if tensor is empty).
    Returns:
        A D1 x ... x D_{n-1} x num_classes 'result' torch.LongTensor,
        containing histograms of the last dimension D_n of tensor,
        that is, result[d_1,...,d_{n-1}, c] = number of times c appears in tensor[d_1,...,d_{n-1}].
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


def powerspect(x):
    c, h, w = x.shape

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
    pwr_sort = pwr.reshape(c, -1)[:, sort]

    nbins = min(h // 2, w // 2)
    k_bins = torch.linspace(0, k_sort.max() + 1, nbins)
    k_bin_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    k_sort_stack = torch.tile(k_sort, dims=(c, 1))

    pwr_binned = batch_histogram(k_sort_stack, weights=pwr_sort, num_classes=nbins - 1)
    count_binned = batch_histogram(k_sort_stack, num_classes=nbins - 1)

    return (
        k_bin_centers.detach().cpu().numpy(),
        (pwr_binned / count_binned).detach().cpu().numpy(),
    )


def compute_ps1d(generated, target, fields, diffusion_channels):

    assert generated.shape == target.shape

    # Comppute PS1D, all channels
    with torch.no_grad():
        k, Pk_gen = powerspect(generated)
        _, Pk_tar = powerspect(target)

    # Make plots and save metrics
    figs = {}
    ratios = {}
    for i, _f in enumerate(fields):
        cidx = diffusion_channels.index(_f)
        f, (a0, a1) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [2, 1], "hspace": 0}, figsize=(6, 4)
        )
        a0.plot(k, Pk_tar[cidx], "k-", label=_f)
        a0.plot(k, Pk_gen[cidx], "r-", label="prediction")
        a0.set_yscale("log")
        a0.set_xscale("log")
        a0.set_xlabel("Wavenumber")
        a0.set_ylabel("PS1D")
        a0.tick_params(axis="x", direction="in", labelbottom=False, which="both")
        a0.tick_params(axis="x", length=5, which="major")
        a0.tick_params(axis="x", length=3, which="minor")
        a0.legend()

        ratio = Pk_gen[cidx] / Pk_tar[cidx]
        a1.plot(k, ratio, "r-")
        a1.plot(k, np.ones(k.shape), "k--")
        a1.set_xlabel("Wavenumber")
        a1.set_ylabel("Ratio")
        a1.set_xscale("log")
        a1.set_ylim((0, 2))
        a1.minorticks_on()
        a1.tick_params(
            axis="x", top=True, direction="inout", labeltop=False, which="both"
        )
        a1.tick_params(axis="x", length=5, which="major")
        a1.tick_params(axis="x", length=3, which="minor")

        lo = np.argmin(np.abs(k - 10.0))
        hi = np.argmin(np.abs(k - 320.0))
        figs["PS1D_" + _f] = f
        ratios["specratio_" + _f] = np.mean(ratio[lo:hi])

    return figs, ratios
