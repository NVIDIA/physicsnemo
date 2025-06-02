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
import numpy as np
import matplotlib.pyplot as plt
from physicsnemo.metrics.general.power_spectrum import power_spectrum


def ps1d_plots(generated, target, fields, diffusion_channels):

    assert generated.shape == target.shape

    # Comppute PS1D, all channels
    with torch.no_grad():
        k, Pk_gen = power_spectrum(generated)
        _, Pk_tar = power_spectrum(target)

        k = k.detach().cpu().numpy()
        Pk_gen = Pk_gen.detach().cpu().numpy()
        Pk_tar = Pk_tar.detach().cpu().numpy()

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
