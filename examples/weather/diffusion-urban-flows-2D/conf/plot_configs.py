# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Configuration settings for matplotlib plotting of flow field data."""

plot_dict = dict(
    figure=dict(
        figsize=(10, 5.5),
        dpi=300,
        tight_layout=True,
        obs_pos_x=-0.125,
        obs_pos_y=0,
        obs_width=0.25,
        obs_height=1,
        levels=100,
        re_levels=5,
        jpdf_level=100,
    ),
    axes=dict(
        x_label=r"$x/h$",  # X-axis label
        y_label=r"$y/h$",  # Y-axis label
        fluc_label=[r"$u^{\prime}$", r"$v^{\prime}$", r"$w^{\prime}$"],
        re_norm_stresses=[
            r"$\overline{u^{\prime}u^{\prime}}$",
            r"$\overline{v^{\prime}v^{\prime}}$",
            r"$\overline{w^{\prime}w^{\prime}}$",
        ],
        re_norm_stresses_p=[
            r"$\overline{{u^{\prime}_p}{u^{\prime}_p}}$",
            r"$\overline{{v^{\prime}_p}{v^{\prime}_p}}$",
            r"$\overline{{w^{\prime}_p}{w^{\prime}_p}}$",
        ],
        re_sh_stresses=[
            r"$\overline{u^{\prime}v^{\prime}}$",
            r"$\overline{u^{\prime}w^{\prime}}$",
            r"$\overline{v^{\prime}w^{\prime}}$",
        ],
        re_sh_stresses_p=[
            r"$\overline{{u^{\prime}_p}{v^{\prime}_p}}$",
            r"$\overline{{u^{\prime}_p}{w^{\prime}_p}}$",
            r"$\overline{{v^{\prime}_p}{w^{\prime}_p}}$",
        ],
        x_lim=(-1, 5),
        y_lim=(0, 2),
        fontsize=20,
        x_ticks=[-1, 0, 1, 2, 3, 4],
        y_ticks=[0, 1, 1.8],
        ticksize=20,
    ),
    plot=dict(
        snap_cmap="viridis",
        re_stress_cmap="rainbow",
        jpdf_cmap="RdBu",
    ),
    legend=dict(
        comp_labels=["streamwise", "wall-normal", "spanwise"],
        levels=100,
    ),
)


def basic_plt_setup():
    """Configure matplotlib with publication-quality default settings.

    Sets up matplotlib to use serif fonts with LaTeX rendering and
    appropriate font sizes for publication-quality figures.
    """
    import matplotlib.pyplot as plt

    plt.rc("font", family="serif")
    plt.rc("text", usetex="true")
    plt.rc("font", size=30)
    plt.rc("axes", labelsize=30, linewidth=2)
    plt.rc("legend", fontsize=25, handletextpad=0.1)
    plt.rc("xtick", labelsize=25)
    plt.rc("ytick", labelsize=25)

    return
