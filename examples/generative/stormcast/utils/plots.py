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

from matplotlib import pyplot as plt
import numpy as np


def validation_plot(generated, truth, variable):
    """Produce validation plot created during training."""
    fig, (a, b) = plt.subplots(1, 2)
    im = a.imshow(generated)
    a.set_title("generated, {}.png".format(variable))
    plt.colorbar(im, fraction=0.046, pad=0.04)
    im = b.imshow(truth)
    b.set_title("truth")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    return fig


color_limits = {
    "u10m": (-5, 5),
    "v10": (-5, 5),
    "t2m": (260, 310),
    "tcwv": (0, 60),
    "msl": (0.1, 0.3),
    "refc": (-10, 30),
}


def inference_plot(
    background,
    state_pred,
    state_true,
    plot_var_background,
    plot_var_state,
    initial_time,
    lead_time,
):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    state_error = state_pred - state_true

    if plot_var_state in color_limits:
        im = ax[0].imshow(
            state_pred,
            origin="lower",
            cmap="magma",
            clim=color_limits[plot_var_state],
        )
    else:
        im = ax[0].imshow(state_pred, origin="lower", cmap="magma")

    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    ax[0].set_title(
        "Predicted, {}, \n initial time {} \n lead_time {} hours".format(
            plot_var_state, initial_time, lead_time
        )
    )
    if plot_var_state in color_limits:
        im = ax[1].imshow(
            state_true,
            origin="lower",
            cmap="magma",
            clim=color_limits[plot_var_state],
        )
    else:
        im = ax[1].imshow(state_true, origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    ax[1].set_title("Actual, {}".format(plot_var_state))
    if plot_var_background in color_limits:
        im = ax[2].imshow(
            background,
            origin="lower",
            cmap="magma",
            clim=color_limits[plot_var_background],
        )
    else:
        im = ax[2].imshow(
            background,
            origin="lower",
            cmap="magma",
        )
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    ax[2].set_title("Background, {}".format(plot_var_background))
    maxerror = np.max(np.abs(state_error))
    im = ax[3].imshow(
        state_error,
        origin="lower",
        cmap="RdBu_r",
        vmax=maxerror,
        vmin=-maxerror,
    )
    fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)
    ax[3].set_title("Error, {}".format(plot_var_state))

    return fig
