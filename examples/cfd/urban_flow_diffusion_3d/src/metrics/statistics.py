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
import matplotlib.pyplot as plt
from matplotlib import cm
import os

from src.utils import add_obstacle_patch, plot_subplot, basic_plt_setup
from src.metrics.helpers import get_data_for_stats
# basic_plt_setup()


class StatisticalEvaluation:
    def __init__(
        self,
        gtruth,
        pred,
        x_axis,
        y_axis,
        z_axis,
        time,
        input_data_type="2D",
        data=None,
        config=None,
        Train_data=True,
        Test_data=False,
    ):
        assert gtruth is not None and pred is not None, (
            "Ground truth and prediction must not be None"
        )
        assert gtruth.size > 0 and pred.size > 0, (
            "Ground truth and prediction must not be empty"
        )

        self.gtruth = gtruth
        self.pred = pred
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.z_axis = z_axis
        self.time = time
        self.input_data_type = input_data_type
        self.data = data
        self.config = config
        self.plot_config = config.plots
        self.Train = Train_data
        self.Test = Test_data

        self.pos_x = self.plot_config.common.obs_pos_x
        self.pos_y = self.plot_config.common.obs_pos_y
        self.width = self.plot_config.common.obs_width
        self.height = self.plot_config.common.obs_height

        self.min, self.max = np.min(self.gtruth), np.max(self.gtruth)

        self.eval_dir = os.path.join(
            self.plot_config.common.eval_dir,
            self.plot_config.common.config_name,
            "results",
        )
        self.x_label = self.plot_config.common.x_label
        self.y_label = self.plot_config.common.y_label

        self.ds_ratio = self.plot_config.common.ds_ratio
        self.config_name = self.plot_config.common.config_name

        if input_data_type == "2D":
            assert self.gtruth.shape[1] == self.pred.shape[1] == 2
            assert len(self.gtruth.shape) == len(self.pred.shape) == 4
        elif input_data_type == "3D":
            assert self.gtruth.shape[1] == self.pred.shape[1] == 3
            assert len(self.gtruth.shape) == len(self.pred.shape) == 5

        os.makedirs(self.eval_dir, exist_ok=True)

    def get_figure_config(self, key):
        return self.plot_config.stats.get(key, {})

    # Wrapper methods calling global functions
    def plot_vis_compare(self, num=None, pdf=None):
        return stat_eval_plot_vis_compare(
            gtruth=self.gtruth,
            pred=self.pred,
            x_axis=self.x_axis,
            y_axis=self.y_axis,
            input_data_type=self.input_data_type,
            config=self.config,
            plot_config=self.plot_config,
            eval_dir=self.eval_dir,
            train=self.Train,
            test=self.Test,
            config_name=self.plot_config.common.config_name,
            vmin=self.min,
            vmax=self.max,
            num=num,
            pdf=pdf,
        )

    def plot_reynolds_stresses(self, x=None, y=None, z=None, data="line-x", pdf=None):
        return stat_eval_plot_reynolds_stresses(
            gtruth=self.gtruth,
            pred=self.pred,
            x=x,
            y=y,
            z=z,
            input_data_type=self.input_data_type,
            data=data,
            ds_ratio=self.ds_ratio,
            x_axis=self.x_axis,
            x_label=self.x_label,
            plot_config=self.plot_config,
            eval_dir=self.eval_dir,
            config_name=self.plot_config.common.config_name,
            test=self.Test,
            pdf=pdf,
        )

    def reynolds_stress(self, inp=None, x=None, y=None, z=None, data=None):
        return stat_eval_reynolds_stress(
            inp=inp,
            x=x,
            y=y,
            z=z,
            input_data_type=self.input_data_type,
            data=data,
            ds_ratio=self.ds_ratio,
        )

    def get_mean_vel_profiles(self, inp=None, x=None, y=None, z=None, data=None):
        return stat_eval_get_mean_vel_profiles(
            inp=inp,
            x=x,
            y=y,
            z=z,
            input_data_type=self.input_data_type,
            data=data,
            ds_ratio=self.ds_ratio,
        )

    def plot_mean_vel_profiles_multiple_locations(
        self, locations=None, x=None, y=None, z=None, data="line-y", pdf=None
    ):
        return stat_eval_plot_mean_vel_profiles_multiple_locations(
            gtruth=self.gtruth,
            pred=self.pred,
            locations=locations,
            x=x,
            y=y,
            z=z,
            data=data,
            input_data_type=self.input_data_type,
            ds_ratio=self.ds_ratio,
            x_axis=self.x_axis,
            y_axis=self.y_axis,
            x_label=self.x_label,
            y_label=self.y_label,
            plot_config=self.plot_config,
            eval_dir=self.eval_dir,
            config_name=self.plot_config.common.config_name,
            train=self.Train,
            test=self.Test,
            pdf=pdf,
        )

    def plot_reynolds_stresses_multiple_locations(
        self, locations=None, x=None, y=None, z=None, data="line-y", pdf=None
    ):
        return stat_eval_plot_reynolds_stresses_multiple_locations(
            gtruth=self.gtruth,
            pred=self.pred,
            locations=locations,
            x=x,
            y=y,
            z=z,
            data=data,
            input_data_type=self.input_data_type,
            ds_ratio=self.ds_ratio,
            x_axis=self.x_axis,
            y_axis=self.y_axis,
            x_label=self.x_label,
            y_label=self.y_label,
            plot_config=self.plot_config,
            eval_dir=self.eval_dir,
            config_name=self.plot_config.common.config_name,
            train=self.Train,
            test=self.Test,
            pdf=pdf,
        )

    def plot_reynolds_stress_planes(
        self, x=None, y=None, z=None, data="plane", pdf=None
    ):
        return stat_eval_plot_reynolds_stress_planes(
            gtruth=self.gtruth,
            pred=self.pred,
            x=x,
            y=y,
            z=z,
            data=data,
            input_data_type=self.input_data_type,
            ds_ratio=self.ds_ratio,
            x_axis=self.x_axis,
            y_axis=self.y_axis,
            plot_config=self.plot_config,
            eval_dir=self.eval_dir,
            config_name=self.plot_config.common.config_name,
            train=self.Train,
            test=self.Test,
            pdf=pdf,
        )

    def plot_joint_pdfs(self, x=None, y=None, z=None, data="line-x", pdf=None):
        return stat_eval_plot_joint_pdfs(
            gtruth=self.gtruth,
            pred=self.pred,
            x=x,
            y=y,
            z=z,
            data=data,
            input_data_type=self.input_data_type,
            ds_ratio=self.ds_ratio,
            x_axis=self.x_axis,
            plot_config=self.plot_config,
            eval_dir=self.eval_dir,
            config_name=self.plot_config.common.config_name,
            test=self.Test,
            pdf=pdf,
        )

    def plot_PSD(
        self, locations=None, y=None, z=None, data="point", nperseg=256, pdf=None
    ):
        return stat_eval_plot_psd(
            gtruth=self.gtruth,
            pred=self.pred,
            locations=locations,
            y=y,
            z=z,
            data=data,
            input_data_type=self.input_data_type,
            ds_ratio=self.ds_ratio,
            time=self.time,
            plot_config=self.plot_config,
            eval_dir=self.eval_dir,
            config_name=self.plot_config.common.config_name,
            train=self.Train,
            test=self.Test,
            nperseg=nperseg,
            pdf=pdf,
        )

    def plot_probe_signal(self, n=200, x=None, y=None, z=None):
        return stat_eval_plot_probe_signal(
            gtruth=self.gtruth,
            pred=self.pred,
            x=x,
            y=y,
            z=z,
            input_data_type=self.input_data_type,
            data="point",
            ds_ratio=self.ds_ratio,
            plot_config=self.plot_config,
        )

    def main(self, num=None, locations=None, y=None, pdf=None):
        print("Getting a visual glimpse of prediction")
        pdf = self.plot_vis_compare(num=num, pdf=pdf)

        # If only fluctuations are analyzed, mean velocity profiles might be skipped
        # Uncomment if needed
        # print("Getting mean velocity profiles in wall-normal direction")
        # pdf = self.plot_mean_vel_profiles_multiple_locations(
        #     locations=locations, x=1, y=None, z=None, data='line-y', pdf=pdf
        # )

        print("Computing Reynolds Stresses")
        # pdf = self.plot_reynolds_stresses(x=None, y=0.5, z=None, pdf=pdf)
        pdf = self.plot_reynolds_stresses_multiple_locations(
            locations=locations, x=1, y=None, z=None, data="line-y", pdf=pdf
        )
        pdf = self.plot_reynolds_stress_planes(
            x=None, y=None, z=None, data="plane", pdf=pdf
        )

        # Uncomment to include joint PDFs
        print("Getting PDFs for flow fields")
        for y_val in [0.05, 0.5, 1.0]:
            pdf = self.plot_joint_pdfs(x=None, y=y_val, z=None, data="line-x", pdf=pdf)

        print("Computing Power Spectral Density")
        # Uncomment if you want to plot PSD
        # pdf = self.plot_PSD(locations=locations, y=0.5, z=None, data='point', nperseg=256, pdf=pdf)
        # pdf = self.plot_PSD(locations=None, y=0.5, z=None, data='line-x', nperseg=256, pdf=pdf)

        return pdf


def stat_eval_plot_vis_compare(
    gtruth,
    pred,
    x_axis,
    y_axis,
    input_data_type,
    config,
    plot_config,
    eval_dir,
    train,
    test,
    config_name,
    vmin,
    vmax,
    num=None,
    pdf=None,
):
    labels = plot_config.common.fluc_label
    levels = plot_config.stats.visual_compare.levels

    if input_data_type == "2D":
        fig, axs = plt.subplots(
            2,
            3,
            figsize=(
                3 * plot_config.stats.visual_compare.figsize[0],
                2 * plot_config.stats.visual_compare.figsize[1],
            ),
        )
    elif input_data_type == "3D":
        fig, axs = plt.subplots(
            3,
            3,
            figsize=(
                3 * plot_config.stats.visual_compare.figsize[0],
                3 * plot_config.stats.visual_compare.figsize[1],
            ),
        )
    else:
        raise ValueError("Unsupported input data type")

    for i in range(gtruth.shape[1]):
        extent = [x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()]
        plot_subplot(
            axs[i, 0],
            gtruth[num, i, :, :],
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            plot_config=plot_config,
        )
        plot_subplot(
            axs[i, 1],
            pred[num, i, :, :],
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            plot_config=plot_config,
        )
        plot_subplot(
            axs[i, 2],
            pred[num + 1, i, :, :],
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            plot_config=plot_config,
        )

    save_path = os.path.join(eval_dir)
    filename = (
        f"{plot_config.stats.visual_compare.filename}_num-{num}"
        + ("-test" if test else "")
        + ".pdf"
    )
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path, dpi=plot_config.common.dpi)

    if pdf:
        fig.set_dpi(plot_config.common.dpi)
        pdf.savefig(fig)

    plt.close(fig)
    return pdf


def stat_eval_reynolds_stress(inp, x, y, z, input_data_type, data, ds_ratio):
    if input_data_type == "3D":
        u, v, w = inp[:, 0], inp[:, 1], inp[:, 2]
        assert len(u.shape) == 4, (
            "Expected 3D data to have 4 dimensions (time, x, y, z)"
        )
    elif input_data_type == "2D":
        u, v, w = inp[:, 0], inp[:, 1], None
        assert len(u.shape) == 3, "Expected 2D data to have 3 dimensions (time, x, y)"

    u_pt = get_data_for_stats(
        u,
        x=x,
        y=y,
        z=z,
        input_data_type=input_data_type,
        data=data,
        ds_ratio=ds_ratio,
        mean_over_time=False,
    )
    v_pt = get_data_for_stats(
        v,
        x=x,
        y=y,
        z=z,
        input_data_type=input_data_type,
        data=data,
        ds_ratio=ds_ratio,
        mean_over_time=False,
    )

    rs_uu = np.mean(u_pt * u_pt, axis=0)
    rs_vv = np.mean(v_pt * v_pt, axis=0)
    rs_uv = np.mean(np.abs(u_pt * v_pt), axis=0)

    if w is None:
        rs_ww = rs_vw = rs_uw = None
    else:
        w_pt = get_data_for_stats(
            w,
            x=x,
            y=y,
            z=z,
            input_data_type=input_data_type,
            data=data,
            ds_ratio=ds_ratio,
            mean_over_time=False,
        )
        rs_ww = np.mean(w_pt * w_pt, axis=0)
        rs_vw = np.mean(v_pt * w_pt, axis=0)
        rs_uw = np.mean(u_pt * w_pt, axis=0)

    return rs_uu, rs_vv, rs_ww, rs_uv, rs_vw, rs_uw


def stat_eval_get_mean_vel_profiles(inp, x, y, z, input_data_type, data, ds_ratio):
    if input_data_type == "3D":
        u, v, w = inp[:, 0], inp[:, 1], inp[:, 2]
        assert len(u.shape) == 4, (
            "Expected 3D data to have 4 dimensions (time, x, y, z)"
        )
    elif input_data_type == "2D":
        u, v, w = inp[:, 0], inp[:, 1], None
        assert len(u.shape) == 3, "Expected 2D data to have 3 dimensions (time, x, y)"

    u_pt = get_data_for_stats(
        u,
        x=x,
        y=y,
        z=z,
        input_data_type=input_data_type,
        data=data,
        ds_ratio=ds_ratio,
        mean_over_time=False,
    )
    v_pt = get_data_for_stats(
        v,
        x=x,
        y=y,
        z=z,
        input_data_type=input_data_type,
        data=data,
        ds_ratio=ds_ratio,
        mean_over_time=False,
    )

    u_pt = np.mean(u_pt, axis=0)
    v_pt = np.mean(v_pt, axis=0)

    return u_pt, v_pt


def stat_eval_plot_mean_vel_profiles_multiple_locations(
    gtruth,
    pred,
    locations,
    x,
    y,
    z,
    data,
    input_data_type,
    ds_ratio,
    x_axis,
    y_axis,
    x_label,
    y_label,
    plot_config,
    eval_dir,
    config_name,
    train,
    test,
    pdf,
):
    if locations is None:
        raise ValueError("Please provide valid locations for plotting.")

    u_m, v_m, u_m_pred, v_m_pred = [], [], [], []

    for loc in locations:
        u, v = stat_eval_get_mean_vel_profiles(
            gtruth, loc, y, z, input_data_type, data, ds_ratio
        )
        u_pred, v_pred = stat_eval_get_mean_vel_profiles(
            pred, loc, y, z, input_data_type, data, ds_ratio
        )
        u_m.append(u)
        v_m.append(v)
        u_m_pred.append(u_pred)
        v_m_pred.append(v_pred)

    u_m, v_m = map(np.array, [u_m, v_m])
    u_m_pred, v_m_pred = map(np.array, [u_m_pred, v_m_pred])

    plot_axis = x_axis if x is None and z is None else y_axis
    plot_x_label = x_label if x is None and z is None else y_label

    colors = cm.viridis(np.linspace(0, 1, u_m.shape[0]))
    mean_vel_profiles = [u_m, v_m]
    mean_vel_profiles_p = [u_m_pred, v_m_pred]
    mean_labels = plot_config.common.mean_label

    for j in range(len(mean_vel_profiles)):
        fig, ax = plt.subplots(
            figsize=(
                2.5 * plot_config.mean_profiles.figsize[1],
                2.5 * plot_config.mean_profiles.figsize[1],
            )
        )
        for i, color in enumerate(colors):
            ax.plot(
                plot_axis,
                mean_vel_profiles[j][i],
                label=rf"$x/h = {locations[i]}$",
                linestyle="-",
                color=color,
            )
            ax.plot(
                plot_axis,
                mean_vel_profiles_p[j][i],
                label=rf"$x/h = {locations[i]}$",
                linestyle="None",
                marker="o",
                color=color,
            )

        ax.set_xlabel(plot_x_label)
        ax.set_ylabel(mean_labels[j])
        ax.set_title(mean_labels[j])
        ax.legend(loc="best")

        if plot_config.common.tight_layout:
            fig.tight_layout()

        filename = (
            f"{plot_config.stats.mean_profiles.filename}_comp_{j}"
            + ("-test" if test else "")
            + ".pdf"
        )
        plt.savefig(
            os.path.join(eval_dir, config_name, filename), dpi=plot_config.common.dpi
        )

        if pdf:
            fig.set_dpi(plot_config.common.dpi)
            pdf.savefig(fig)

        plt.close(fig)

    return pdf


def stat_eval_plot_reynolds_stresses_multiple_locations(
    gtruth,
    pred,
    locations,
    x,
    y,
    z,
    data,
    input_data_type,
    ds_ratio,
    x_axis,
    y_axis,
    x_label,
    y_label,
    plot_config,
    eval_dir,
    config_name,
    train,
    test,
    pdf,
):
    if locations is None:
        raise ValueError("Please provide valid locations for plotting.")

    rs_uu, rs_vv, rs_ww, rs_uv, rs_vw, rs_uw = [], [], [], [], [], []
    rs_uu_pred, rs_vv_pred, rs_ww_pred, rs_uv_pred, rs_vw_pred, rs_uw_pred = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for loc in locations:
        stresses = stat_eval_reynolds_stress(
            gtruth, loc, y, z, input_data_type, data, ds_ratio
        )
        stresses_pred = stat_eval_reynolds_stress(
            pred, loc, y, z, input_data_type, data, ds_ratio
        )

        rs_uu.append(stresses[0])
        rs_vv.append(stresses[1])
        rs_ww.append(stresses[2])
        rs_uv.append(stresses[3])
        rs_vw.append(stresses[4])
        rs_uw.append(stresses[5])

        rs_uu_pred.append(stresses_pred[0])
        rs_vv_pred.append(stresses_pred[1])
        rs_ww_pred.append(stresses_pred[2])
        rs_uv_pred.append(stresses_pred[3])
        rs_vw_pred.append(stresses_pred[4])
        rs_uw_pred.append(stresses_pred[5])

    rs_uu, rs_vv, rs_ww, rs_uv, rs_vw, rs_uw = map(
        np.array, [rs_uu, rs_vv, rs_ww, rs_uv, rs_vw, rs_uw]
    )
    rs_uu_pred, rs_vv_pred, rs_ww_pred, rs_uv_pred, rs_vw_pred, rs_uw_pred = map(
        np.array,
        [rs_uu_pred, rs_vv_pred, rs_ww_pred, rs_uv_pred, rs_vw_pred, rs_uw_pred],
    )

    plot_axis = x_axis if x is None and z is None else y_axis
    plot_x_label = x_label if x is None and z is None else y_label
    colors = cm.viridis(np.linspace(0, 1, rs_uu.shape[0]))

    reynolds_stresses = [rs_uu, rs_vv, rs_uv]
    reynolds_stresses_p = [rs_uu_pred, rs_vv_pred, rs_uv_pred]
    stress_labels = plot_config.stats.re_stress_planes.re_norm_stresses
    stress_titles = stress_labels

    for j in range(len(reynolds_stresses)):
        fig, ax = plt.subplots(
            figsize=(
                2.5 * plot_config.stats.re_stress_lines.figsize[1],
                2.5 * plot_config.stats.re_stress_lines.figsize[1],
            )
        )
        for i, color in enumerate(colors):
            ax.plot(
                plot_axis,
                reynolds_stresses[j][i],
                label=rf"$x/h = {locations[i]}$",
                linestyle="-",
                color=color,
            )
            ax.plot(
                plot_axis,
                reynolds_stresses_p[j][i],
                label=rf"$x/h = {locations[i]}$",
                linestyle="None",
                marker="o",
                color=color,
            )

        ax.set_xlabel(plot_x_label)
        ax.set_ylabel(stress_labels[j])
        ax.set_title(stress_titles[j])
        ax.legend()

        if plot_config.common.tight_layout:
            fig.tight_layout()

        filename = (
            f"{plot_config.stats.re_stress_planes.filename}_comp_{j}"
            + ("-test" if test else "")
            + ".pdf"
        )
        save_path = os.path.join(eval_dir, filename)
        plt.savefig(save_path, dpi=plot_config.common.dpi)

        if pdf:
            pdf.savefig(fig)

        plt.close(fig)

    return pdf


def stat_eval_plot_reynolds_stress_planes(
    gtruth,
    pred,
    x,
    y,
    z,
    data,
    input_data_type,
    ds_ratio,
    x_axis,
    y_axis,
    plot_config,
    eval_dir=None,
    pdf=None,
    out_save=False,
    out_show=False,
):
    rs_uu, rs_vv, rs_ww, rs_uv, rs_vw, rs_uw = stat_eval_reynolds_stress(
        gtruth, x, y, z, input_data_type, data, ds_ratio
    )
    rs_uu_pred, rs_vv_pred, rs_ww_pred, rs_uv_pred, rs_vw_pred, rs_uw_pred = (
        stat_eval_reynolds_stress(pred, x, y, z, input_data_type, data, ds_ratio)
    )

    is_3d = rs_ww is not None and rs_ww_pred is not None
    X_grid, Y_grid = np.meshgrid(y_axis, x_axis)
    cmap = plot_config.stats.re_stress_planes.cmap
    fontsize = plot_config.common.fontsize
    levels = plot_config.stats.re_stress_planes.levels

    def _plot_contour(ax, Y_grid, X_grid, data, data_pred, title):
        contour = ax.contourf(Y_grid, X_grid, data, levels=levels, cmap=cmap)
        ax.contour(Y_grid, X_grid, data_pred, levels=levels, colors="black")
        add_obstacle_patch(ax, plot_config)
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(plot_config.common.x_label)
        ax.set_ylabel(plot_config.common.y_label)
        ax.set_xticks(plot_config.common.x_ticks)
        ax.set_yticks(plot_config.common.y_ticks)
        fig = ax.get_figure()
        fig.colorbar(contour, ax=ax)

    if not is_3d:
        fig, axs = plt.subplots(
            1,
            3,
            figsize=(
                1.25 * 2.5 * plot_config.stats.re_stress_planes.figsize[0],
                1.25 * 1 * plot_config.stats.re_stress_planes.figsize[1],
            ),
        )
        _plot_contour(
            axs[0],
            Y_grid,
            X_grid,
            rs_uu,
            rs_uu_pred,
            plot_config.stats.re_stress_planes.re_norm_stresses[0],
        )
        _plot_contour(
            axs[1],
            Y_grid,
            X_grid,
            rs_vv,
            rs_vv_pred,
            plot_config.stats.re_stress_planes.re_norm_stresses[1],
        )
        _plot_contour(
            axs[2],
            Y_grid,
            X_grid,
            rs_uv,
            rs_uv_pred,
            plot_config.stats.re_stress_planes.re_sh_stresses[0],
        )
    else:
        fig, axs = plt.subplots(
            3,
            2,
            figsize=(
                5 * plot_config.stats.re_stress_planes.figsize[0],
                2 * plot_config.stats.re_stress_planes.figsize[1],
            ),
        )
        _plot_contour(
            axs[0, 0],
            Y_grid,
            X_grid,
            rs_uu,
            rs_uu_pred,
            plot_config.stats.re_stress_planes.re_norm_stresses[0],
        )
        _plot_contour(
            axs[0, 1],
            Y_grid,
            X_grid,
            rs_vv,
            rs_vv_pred,
            plot_config.stats.re_stress_planes.re_norm_stresses[1],
        )
        _plot_contour(
            axs[1, 0],
            Y_grid,
            X_grid,
            rs_ww,
            rs_ww_pred,
            plot_config.stats.re_stress_planes.re_norm_stresses[2],
        )
        _plot_contour(
            axs[1, 1],
            Y_grid,
            X_grid,
            rs_uv,
            rs_uv_pred,
            plot_config.stats.re_stress_planes.re_sh_stresses[0],
        )
        _plot_contour(
            axs[2, 0],
            Y_grid,
            X_grid,
            rs_uw,
            rs_uw_pred,
            plot_config.stats.re_stress_planes.re_sh_stresses[1],
        )
        _plot_contour(
            axs[2, 1],
            Y_grid,
            X_grid,
            rs_vw,
            rs_vw_pred,
            plot_config.stats.re_stress_planes.re_sh_stresses[2],
        )

    if plot_config.common.tight_layout:
        fig.tight_layout()

    if out_save and eval_dir is not None:
        filename = f"{plot_config.stats.re_stress_planes.filename}.pdf"
        path = os.path.join(eval_dir, filename)
        plt.savefig(path, dpi=plot_config.common.dpi)

    if pdf is not None:
        pdf.savefig(fig, dpi=plot_config.common.dpi)
        plt.close(fig)
        return pdf

    if out_show:
        plt.show()

    return fig


def stat_eval_plot_joint_pdfs(
    gtruth,
    pred,
    x,
    y,
    z,
    data,
    input_data_type,
    ds_ratio,
    x_axis,
    plot_config,
    eval_dir,
    config_name,
    test,
    pdf,
):
    from scipy.stats import gaussian_kde

    levels = plot_config.stats.jpdfs.levels
    labels = plot_config.common.fluc_label
    fontsize = plot_config.stats.jpdfs.fontsize
    x_skip = 57

    def _joint_pdf(inp_comp, axis):
        u_flat = inp_comp.reshape(-1)
        x_flat = (
            np.tile(axis.reshape(1, inp_comp.shape[1]), (inp_comp.shape[0], 1))
        ).reshape(-1)
        xy = np.vstack([x_flat, u_flat])
        xi, yi = (
            np.linspace(axis.min(), axis.max(), 100),
            np.linspace(inp_comp.min(), inp_comp.max(), 100),
        )
        xi, yi = np.meshgrid(xi, yi)
        zi = gaussian_kde(xy)(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
        return xi, yi, zi / np.max(zi)

    for i in range(gtruth.shape[1]):
        comp = get_data_for_stats(
            gtruth[:, i, x_skip:, :],
            x=x,
            y=y,
            z=z,
            input_data_type=input_data_type,
            data=data,
            ds_ratio=ds_ratio,
        )
        comp_pred = get_data_for_stats(
            pred[:, i, x_skip:, :],
            x=x,
            y=y,
            z=z,
            input_data_type=input_data_type,
            data=data,
            ds_ratio=ds_ratio,
        )

        xi, yi, zi = _joint_pdf(comp, x_axis[x_skip:])
        xi_p, yi_p, zi_p = _joint_pdf(comp_pred, x_axis[x_skip:])

        fig, ax = plt.subplots(
            figsize=(
                plot_config.stats.jpdfs.figsize[0],
                plot_config.stats.jpdfs.figsize[0],
            )
        )
        contour = ax.contour(
            xi, yi, zi, levels=levels, cmap=plot_config.stats.jpdfs.cmap
        )
        ax.contour(xi_p, yi_p, zi_p, levels=levels, colors="black", linestyles="dotted")

        ax.set_xlabel(plot_config.common.x_label, fontsize=fontsize)
        ax.set_ylabel(labels[i], fontsize=fontsize)
        fig.colorbar(contour, ax=ax)
        fig.tight_layout()

        filename = (
            f"{plot_config.stats.jpdfs.filename}-{i}_@y:{y}"
            + ("-test" if test else "")
            + ".pdf"
        )
        plt.savefig(os.path.join(eval_dir, filename), dpi=plot_config.common.dpi)

        if pdf:
            fig.set_dpi(plot_config.common.dpi)
            pdf.savefig(fig)

        plt.close(fig)

    return pdf


def stat_eval_plot_psd(
    gtruth,
    pred,
    locations,
    y,
    z,
    data,
    input_data_type,
    ds_ratio,
    time,
    plot_config,
    eval_dir,
    config_name,
    train,
    test,
    nperseg=256,
    pdf=None,
):
    from scipy.signal import welch

    def _welch_psd(signal):
        signal = signal.flatten()
        fs = time[1] - time[0]
        return welch(signal, fs=fs, nperseg=nperseg)

    colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#e377c2", "#17becf"]
    labels = plot_config.common.fluc_label
    linewidth = 2

    if data == "point":
        for k in range(gtruth.shape[1]):
            fig, ax = plt.subplots(figsize=(12, 8))
            for i, x in enumerate(locations):
                comp = get_data_for_stats(
                    gtruth[:, k],
                    x=x,
                    y=y,
                    z=z,
                    input_data_type=input_data_type,
                    data=data,
                    ds_ratio=ds_ratio,
                )
                comp_pred = get_data_for_stats(
                    pred[:, k],
                    x=x,
                    y=y,
                    z=z,
                    input_data_type=input_data_type,
                    data=data,
                    ds_ratio=ds_ratio,
                )
                f, psd = _welch_psd(comp)
                f_p, psd_p = _welch_psd(comp_pred)

                ax.loglog(
                    f,
                    psd,
                    label=f"GT @ x/h={x}",
                    linestyle="-",
                    linewidth=linewidth,
                    color=colors[i],
                )
                ax.loglog(
                    f_p,
                    psd_p,
                    label=f"pred @ x/h={x}",
                    marker="o",
                    linestyle="none",
                    color=colors[i],
                )

            ax.set_title(
                "Power Spectral Density vs Frequency (Welch Method)", fontsize=16
            )
            ax.set_xlabel("Frequency [Hz]", fontsize=14)
            ax.set_ylabel("PSD [V**2/Hz]", fontsize=14)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.legend(fontsize=12, loc="upper right")
            fig.tight_layout()

            filename = (
                f"{plot_config.stats.psd}-pt-{labels[k]}"
                + ("-test" if test else "")
                + ".pdf"
            )
            plt.savefig(os.path.join(eval_dir, filename), dpi=plot_config.common.dpi)
            if pdf:
                pdf.savefig(fig)
            plt.close(fig)

    elif data == "line-x":
        for k in range(gtruth.shape[1]):
            fig, ax = plt.subplots(figsize=(12, 8))
            comp = get_data_for_stats(
                gtruth[:, k],
                x=None,
                y=y,
                z=z,
                input_data_type=input_data_type,
                data=data,
                ds_ratio=ds_ratio,
            )
            comp_pred = get_data_for_stats(
                pred[:, k],
                x=None,
                y=y,
                z=z,
                input_data_type=input_data_type,
                data=data,
                ds_ratio=ds_ratio,
            )
            f, psd = _welch_psd(comp)
            f_p, psd_p = _welch_psd(comp_pred)

            ax.loglog(
                f,
                psd,
                label=f"GT @ y/h={y}",
                linestyle="-",
                linewidth=linewidth,
                color="k",
            )
            ax.loglog(
                f_p,
                psd_p,
                label=f"pred @ y/h={y}",
                marker="o",
                linestyle="none",
                color="k",
            )

            ax.set_title(
                "Power Spectral Density vs Frequency (Welch Method)", fontsize=16
            )
            ax.set_xlabel("Frequency [Hz]", fontsize=14)
            ax.set_ylabel("PSD [V**2/Hz]", fontsize=14)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.legend(fontsize=12, loc="upper right")
            fig.tight_layout()

            filename = (
                f"{plot_config.stats.psd}-linex-{labels[k]}_{config_name}"
                + ("-test" if test else "")
                + ".pdf"
            )
            plt.savefig(
                os.path.join(eval_dir, config_name, filename),
                dpi=plot_config.common.dpi,
            )
            if pdf:
                pdf.savefig(fig)
            plt.close(fig)

    return pdf


def stat_eval_plot_probe_signal(
    gtruth, pred, x, y, z, input_data_type, data, ds_ratio, plot_config
):
    labels = plot_config.common.fluc_label
    for i in range(gtruth.shape[1]):
        u_pt = get_data_for_stats(
            gtruth[:, i],
            x=x,
            y=y,
            z=z,
            input_data_type=input_data_type,
            data=data,
            mean_over_time=False,
            ds_ratio=ds_ratio,
        )
        u_pt_p = get_data_for_stats(
            pred[:, i],
            x=x,
            y=y,
            z=z,
            input_data_type=input_data_type,
            data=data,
            mean_over_time=False,
            ds_ratio=ds_ratio,
        )

        n = min(200, u_pt.shape[0])
        random_indices = np.random.randint(0, u_pt.shape[0], n)

        plt.figure(figsize=(12, 8))
        plt.plot(
            u_pt[random_indices],
            label=rf"{labels[i]}' @ x/h = 1",
            linestyle="-",
            color="k",
        )
        plt.plot(
            u_pt_p[random_indices],
            label=rf"{labels[i]}_p' @ x/h = 1",
            linestyle="-",
            color="r",
        )

        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.xlabel("num of snapshots", fontsize=14)
        plt.ylabel(rf"{labels[i]}'", fontsize=14)
        plt.legend(loc="best", fontsize=12)
        plt.xlim([0, n])
        plt.ylim([-0.5, 0.5])
        plt.tight_layout()
        plt.show()


# TODO: add different fig sizes, label >> labels
