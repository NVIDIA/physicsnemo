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

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

colormaps = {
    "u10m": "RdBu_r",
    "v10m": "RdBu_r",
    "t2m": "RdYlBu_r",
    "refc": "magma",
}


def comparison_plot(
    panels: list[tuple[str, np.ndarray]], variable: str, title: str | None = None
) -> plt.Figure:
    """Plot 2D fields side by side with the color range of the last panel.

    Args:
        panels: list of (panel title, field) pairs, the last one is the truth
        variable: variable name, used for the color bar and color map
        title: figure title

    Returns:
        matplotlib figure
    """
    (vmin, vmax) = (np.nanmin(panels[-1][1]), np.nanmax(panels[-1][1]))
    cmap = colormaps.get(variable, "viridis")
    if cmap == "RdBu_r":  # center the diverging color map on zero
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
    (fig, axes) = plt.subplots(
        1,
        len(panels),
        figsize=(4.5 * len(panels), 3.2),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, (panel_title, field) in zip(axes[0], panels, strict=True):
        im = ax.imshow(field, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(panel_title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.9, label=variable)
    if title is not None:
        fig.suptitle(title)
    return fig
