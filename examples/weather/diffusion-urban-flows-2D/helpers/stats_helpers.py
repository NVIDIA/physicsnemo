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

"""
Statistical evaluation and visualization utilities for flow field predictions.

This module provides comprehensive statistical evaluation tools for comparing ground truth
and predicted flow field data. It includes functionality for:
- Computing Reynolds stress components (normal and shear stresses)
- Generating joint probability density functions (PDFs)
- Computing power spectral density (PSD) using Welch's method
- Creating visual comparisons of predicted vs ground truth fields
- Plotting probe signals over time
- Supporting both 2D and 3D flow field data

The main class `StatisticalEvaluation` provides a comprehensive framework for evaluating
diffusion model predictions from a statistical perspective.
"""

import numpy as np
import matplotlib.pyplot as plt

import time
import os

from helpers.general_helpers import (
    get_data_for_stats,
    add_obstacle_patch,
    plot_subplot,
    dict2namespace,
)
from conf.plot_configs import plot_dict


# basic_plt_setup()
plot_config = dict2namespace(plot_dict)


class StatisticalEvaluation:
    """
    Comprehensive statistical evaluation class for comparing ground truth and predicted flow fields.

    This class provides an integrated framework for evaluating diffusion model predictions
    from a statistical perspective. It generates detailed visualizations and analyses including:
    visual field comparisons, Reynolds stress components, joint probability density functions,
    and power spectral density plots. Outputs are saved as PDF and PNG files.

    The class supports both 2D and 3D flow field data and handles multiple velocity components
    (u, v, and optionally w). It provides methods for computing various statistical metrics
    commonly used in fluid dynamics research.

    Attributes
    ----------
    gtruth : np.ndarray
        Ground truth flow field data with shape (time, components, spatial_dims...).
    pred : np.ndarray
        Predicted flow field data with the same shape as ground truth.
    x_axis : np.ndarray
        X-axis coordinates for the domain.
    y_axis : np.ndarray
        Y-axis coordinates for the domain.
    z_axis : np.ndarray, optional
        Z-axis coordinates for 3D domains.
    time : np.ndarray
        Time steps corresponding to the temporal dimension.
    input_data_type : str
        Type of input data ('2D' or '3D').
    ds_ratio : float
        Downsampling ratio applied to the data.
    snaps : int
        Number of snapshots in the prediction data.

    Notes
    -----
    The class uses global plot configuration from conf.plot_configs to maintain
    consistent visualization styles across all generated plots.

    Examples
    --------
    >>> gtruth = np.random.rand(100, 2, 60, 20)  # 100 time steps, 2 components, 60x20 spatial grid
    >>> pred = np.random.rand(100, 2, 60, 20)
    >>> x_axis = np.linspace(-1, 5, 60)
    >>> y_axis = np.linspace(0, 2, 20)
    >>> time = np.linspace(0, 10, 100)
    >>> evaluator = StatisticalEvaluation(gtruth=gtruth, pred=pred, x_axis=x_axis,
    ...                                    y_axis=y_axis, time=time, input_data_type='2D')
    >>> # Generate statistical evaluation plots
    >>> pdf = evaluator.main(num=0, locations=[1, 2, 3], y=0.5)
    """

    def __init__(
        self,
        gtruth=None,
        pred=None,
        x_axis=None,
        y_axis=None,
        z_axis=None,
        time=None,
        input_data_type="2D",
        data=None,
        ds_ratio=1,
        output_file_path=None,
    ):
        """
        Initialize the StatisticalEvaluation class with the provided parameters and evaluate the DDPM from statistical sense.

        Parameters:
        gtruth (ndarray): Ground truth data.
        pred (ndarray): Predicted data.
        x_axis (ndarray): X-axis values.
        y_axis (ndarray): Y-axis values.
        z_axis (ndarray): Z-axis values.
        time (ndarray): Time values.
        input_data_type (str): Type of input data ('2D' or '3D').
        data (ndarray): Additional data.
        config (object): Configuration object.
        """

        assert gtruth is not None and pred is not None, (
            "Ground truth and prediction must not be None"
        )
        assert gtruth.any() and pred.any(), (
            "Ground truth and prediction must not be empty"
        )

        self.gtruth = gtruth
        self.pred = pred
        self.snaps = pred.shape[0]

        self.x_axis = x_axis
        self.y_axis = y_axis
        self.z_axis = z_axis
        self.time = time

        self.input_data_type = input_data_type
        self.data = data
        self.plot_config = plot_config

        self.ds_ratio = ds_ratio

        # Obstacle dimensions & location
        self.pos_x, self.pos_y = (
            self.plot_config.figure.obs_pos_x,
            self.plot_config.figure.obs_pos_y,
        )
        self.width, self.height = (
            self.plot_config.figure.obs_width,
            self.plot_config.figure.obs_height,
        )  # width, height of the obstacle

        self.min, self.max = np.min(self.gtruth), np.max(self.gtruth)

        self.eval_dir = output_file_path

        self.x_label = self.plot_config.axes.x_label
        self.y_label = self.plot_config.axes.y_label
        self.fontsize = self.plot_config.axes.fontsize

        # plt.rcParams['font.family'] = 'serif'
        # plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering

        if input_data_type == "2D":
            assert self.gtruth.shape[1] == self.pred.shape[1] == 2, (
                "Expected 2 velocity components in 2D data"
            )
            assert len(self.gtruth.shape) == len(self.pred.shape) == 4, (
                "Expected 2D data to have 4 dimensions (time, nc ,u, v)"
            )

        elif input_data_type == "3D":
            assert self.gtruth.shape[1] == self.pred.shape[1] == 3, (
                "Expected 3 velocity components in 3D data"
            )
            assert len(self.gtruth.shape) == len(self.pred.shape) == 5, (
                "Expected 3D data to have 5 dimensions (time, nc ,u, v, w)"
            )

        if not os.path.exists(self.eval_dir):
            os.mkdir(self.eval_dir)

    def plot_vis_compare(self, num=0, pdf=None):
        """
        Plot visual comparison of ground truth and predicted data.

        Parameters:
        num (int): Index for the time step to plot.
        pdf (PdfPages): PDF object to save the plots.
        """

        if self.input_data_type == "2D":
            fig, axs = plt.subplots(
                2,
                3,
                figsize=(
                    3 * self.plot_config.figure.figsize[0],
                    2 * self.plot_config.figure.figsize[1],
                ),
            )

        elif self.input_data_type == "3D":
            fig, axs = plt.subplots(
                2,
                3,
                figsize=(
                    3 * self.plot_config.figure.figsize[0],
                    2 * self.plot_config.figure.figsize[1],
                ),
            )

        else:
            raise ValueError("Unsupported input data type")

        for i in range(self.gtruth.shape[1]):
            # Plot Ground Truth

            extent = [
                self.x_axis.min(),
                self.x_axis.max(),
                self.y_axis.min(),
                self.y_axis.max(),
            ]

            plot_subplot(
                axs[i, 0],
                self.gtruth[num, i, :, :],
                extent=extent,
                vmin=self.min,
                vmax=self.max,
            )
            plot_subplot(
                axs[i, 1],
                self.pred[num, i, :, :],
                extent=extent,
                vmin=self.min,
                vmax=self.max,
            )
            plot_subplot(
                axs[i, 2],
                self.pred[num + 1, i, :, :],
                extent=extent,
                vmin=self.min,
                vmax=self.max,
            )

        if pdf is not None:
            pdf.savefig(fig)
            plt.savefig(
                self.eval_dir
                + f"/pred_snaps-{self.snaps}-visual_comparison_num-{num}.pdf",
                dpi=self.plot_config.figure.dpi,
            )
        else:
            plt.savefig(
                self.eval_dir
                + f"/pred_snaps-{self.snaps}-visual_comparison_num-{num}.png",
                dpi=self.plot_config.figure.dpi,
            )

        plt.close(fig)

        return pdf

    def reynolds_stress(self, inp=None, x=None, y=None, z=None, data=None, pdf=None):
        """
        Compute Reynolds stresses for the given input data.

        Parameters:
        inp (ndarray): Input data.
        x (ndarray): X-axis values.
        y (ndarray): Y-axis values.
        z (ndarray): Z-axis values.
        data (str): Data type ('line-x', 'plane', etc.).
        pdf (PdfPages): PDF object to save the plots.

        Returns:
        tuple: Reynolds stress components.
        """

        if self.input_data_type == "3D":
            u, v, w = inp[:, 0], inp[:, 1], inp[:, 2]
            assert len(u.shape) == 4, (
                "Expected 3D data to have 4 dimensions (time, x, y, z)"
            )

        elif self.input_data_type == "2D":
            u, v, w = inp[:, 0], inp[:, 1], None
            assert len(u.shape) == 3, (
                "Expected 2D data to have 3 dimensions (time, x, y)"
            )

        # Extract the relevant data for statistics
        u_pt = get_data_for_stats(
            u,
            x=x,
            y=y,
            z=z,
            input_data_type=self.input_data_type,
            data=data,
            ds_ratio=self.ds_ratio,
            mean_over_time=False,
        )
        v_pt = get_data_for_stats(
            v,
            x=x,
            y=y,
            z=z,
            input_data_type=self.input_data_type,
            data=data,
            ds_ratio=self.ds_ratio,
            mean_over_time=False,
        )

        # Compute the Reynolds stresses
        rs_uu = np.mean(u_pt * u_pt, axis=0)
        rs_vv = np.mean(v_pt * v_pt, axis=0)
        rs_uv = np.mean(np.abs(u_pt * v_pt), axis=0)

        if w is None:
            rs_ww = None
            rs_vw = None
            rs_uw = None
        else:
            w_pt = get_data_for_stats(
                w,
                x=x,
                y=y,
                z=z,
                input_data_type=self.input_data_type,
                data=data,
                mean_over_time=False,
            )
            rs_ww = np.mean(w_pt * w_pt, axis=0)
            rs_vw = np.mean(v_pt * w_pt, axis=0)
            rs_uw = np.mean(u_pt * w_pt, axis=0)

        return rs_uu, rs_vv, rs_ww, rs_uv, rs_vw, rs_uw

    def plot_reynolds_stresses(self, x=None, y=None, z=None, data="line-x", pdf=None):
        """
        Plot Reynolds stresses for ground truth and predicted data.

        Parameters:
        x (ndarray): X-axis values.
        y (ndarray): Y-axis values.
        z (ndarray): Z-axis values.
        data (str): Data type ('line-x', 'plane', etc.).
        pdf (PdfPages): PDF object to save the plots.
        """
        rs_uu, rs_vv, rs_ww, rs_uv, rs_vw, rs_uw = self.reynolds_stress(
            inp=self.gtruth, x=x, y=y, z=z, data=data
        )
        rs_uu_pred, rs_vv_pred, rs_ww_pred, rs_uv_pred, rs_vw_pred, rs_uw_pred = (
            self.reynolds_stress(inp=self.pred, x=x, y=y, z=z, data=data)
        )

        # Check if rs_ww and rs_ww_pred are None
        is_3d = rs_ww is not None and rs_ww_pred is not None

        # Plotting
        fig, ax = plt.subplots(
            figsize=(
                2 * self.plot_config.figure.figsize[1],
                2 * self.plot_config.figure.figsize[1],
            )
        )

        ax.plot(
            self.x_axis,
            rs_uu,
            label=self.plot_config.axes.re_norm_stresses[0],
            linestyle="-",
            color="b",
        )
        ax.plot(
            self.x_axis,
            rs_uu_pred,
            label=self.plot_config.axes.re_norm_stresses_p[0],
            linestyle="None",
            marker="o",
            color="b",
        )

        ax.plot(
            self.x_axis,
            rs_vv,
            label=self.plot_config.axes.re_norm_stresses[1],
            linestyle="-",
            color="g",
        )
        ax.plot(
            self.x_axis,
            rs_vv_pred,
            label=self.plot_config.axes.re_norm_stresses_p[1],
            linestyle="None",
            marker="^",
            color="g",
        )

        ax.plot(
            self.x_axis,
            rs_uv,
            label=self.plot_config.axes.re_sh_stresses[0],
            linestyle="-",
            color="r",
        )
        ax.plot(
            self.x_axis,
            rs_uv_pred,
            label=self.plot_config.axes.re_sh_stresses_p[0],
            linestyle="None",
            marker="v",
            color="r",
        )

        if is_3d:
            ax.plot(
                self.x_axis,
                rs_ww,
                label=self.plot_config.axes.re_norm_stresses[2],
                linestyle="-",
                color="c",
            )
            ax.plot(
                self.x_axis,
                rs_ww_pred,
                label=self.plot_config.axes.re_norm_stresses_p[2],
                linestyle="None",
                marker="s",
                color="c",
            )

            ax.plot(
                self.x_axis,
                rs_uw,
                label=self.plot_config.axes.re_sh_stresses[1],
                linestyle="-",
                color="m",
            )
            ax.plot(
                self.x_axis,
                rs_uw_pred,
                label=self.plot_config.axes.re_sh_stresses_p[1],
                linestyle="None",
                marker="d",
                color="m",
            )

            ax.plot(
                self.x_axis,
                rs_vw,
                label=self.plot_config.axes.re_sh_stresses[2],
                linestyle="-",
                color="y",
            )
            ax.plot(
                self.x_axis,
                rs_vw_pred,
                label=self.plot_config.axes.re_sh_stresses_p[2],
                linestyle="None",
                marker="*",
                color="y",
            )

        ax.set_xlabel(self.x_label)  # , fontsize = self.plot_config.axes.fontsize)
        ax.set_ylabel(
            r"$\overline{{u_i}^{\prime}{u_j}^{\prime}}$"
        )  # , fontsize=self.plot_config.axes.fontsize)
        ax.legend()  # prop={'size': self.plot_config.axes.fontsize})
        ax.set_title(
            "Reynolds Stress Components"
        )  # , fontsize = self.plot_config.axes.fontsize)

        if self.plot_config.figure.tight_layout:
            fig.tight_layout()

        if pdf is not None:
            pdf.savefig(fig)
            plt.savefig(
                self.eval_dir + f"/pred_snaps-{self.snaps}-Reynolds_stresses1.pdf",
                dpi=self.plot_config.figure.dpi,
            )
        else:
            plt.savefig(
                self.eval_dir + f"/pred_snaps-{self.snaps}-Reynolds_stresses1.pdf",
                dpi=self.plot_config.figure.dpi,
            )
        plt.close(fig)

        return pdf

    def _plot_contour(
        self,
        ax,
        Y_grid,
        X_grid,
        data,
        data_pred,
        title,
        levels,
        cmap,
        fontsize,
        cbar_orientation="vertical",
    ):
        """
        Helper method to plot a single contour plot with ground truth and predicted data.
        #TODO: Move this to utils
        Parameters:
        ax (Axes): The matplotlib Axes object to plot on.
        Y_grid (ndarray): Y-axis grid values.
        X_grid (ndarray): X-axis grid values.
        data (ndarray): Ground truth data.
        data_pred (ndarray): Predicted data.
        title (str): Title of the plot.
        levels (int): Number of contour levels.
        cmap (str): Colormap.
        fontsize (int): Font size for the title.
        cbar_orientation (str): Orientation of the colorbar ('vertical' or 'horizontal'). Default is 'vertical'.
        """

        # Create filled contour plot for ground truth
        contour = ax.contourf(Y_grid, X_grid, data, levels=levels, cmap=cmap)

        # Create contour plot for predicted data in black
        ax.contour(Y_grid, X_grid, data_pred, levels=levels, colors="black")

        # Add obstacle patch to the plot (custom method)
        add_obstacle_patch(ax)

        # Set title and axis labels
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(
            self.plot_config.axes.x_label
        )  # , fontsize=self.plot_config.axes.fontsize)
        ax.set_ylabel(
            self.plot_config.axes.y_label
        )  # , fontsize=self.plot_config.axes.fontsize)

        # Set ticks and their sizes
        ax.set_xticks(self.plot_config.axes.x_ticks)
        ax.set_yticks(self.plot_config.axes.y_ticks)
        # ax.tick_params(axis='both', labelsize=self.plot_config.axes.ticksize)

        # Get the figure object from the axes
        fig = ax.get_figure()

        # Add a colorbar with the correct orientation and tick size
        fig.colorbar(contour, ax=ax, orientation=cbar_orientation)
        # cbar.ax.tick_params(labelsize=self.plot_config.axes.ticksize)

    def plot_reynolds_stress_planes(
        self, x=None, y=None, z=None, data="plane", pdf=None
    ):
        """
        Plot Reynolds stress planes for ground truth and predicted data.

        Parameters:
        x (ndarray): X-axis values.
        y (ndarray): Y-axis values.
        z (ndarray): Z-axis values.
        data (str): Data type ('plane', etc.).
        levels (int): Number of contour levels.
        pdf (PdfPages): PDF object to save the plots.

        Returns:
        pdf (PdfPages): PDF object with the saved plots.
        """

        rs_uu, rs_vv, rs_ww, rs_uv, rs_vw, rs_uw = self.reynolds_stress(
            self.gtruth, x=x, y=y, z=z, data=data
        )
        rs_uu_pred, rs_vv_pred, rs_ww_pred, rs_uv_pred, rs_vw_pred, rs_uw_pred = (
            self.reynolds_stress(self.pred, x=x, y=y, z=z, data=data)
        )

        # Check if rs_ww and rs_ww_pred are None
        is_3d = rs_ww is not None and rs_ww_pred is not None

        # Create mesh grid for plotting
        X_grid, Y_grid = np.meshgrid(self.y_axis, self.x_axis)

        cmap = self.plot_config.plot.re_stress_cmap
        fontsize = self.plot_config.axes.fontsize
        levels = self.plot_config.figure.re_levels

        if not is_3d:
            fig, axs = plt.subplots(
                1,
                3,
                figsize=(
                    2.5 * self.plot_config.figure.figsize[0],
                    1 * self.plot_config.figure.figsize[1],
                ),
            )

            # Plot uu
            self._plot_contour(
                axs[0],
                Y_grid,
                X_grid,
                rs_uu,
                rs_uu_pred,
                self.plot_config.axes.re_norm_stresses[0],
                levels,
                cmap,
                fontsize,
            )

            # Plot vv
            self._plot_contour(
                axs[1],
                Y_grid,
                X_grid,
                rs_vv,
                rs_vv_pred,
                self.plot_config.axes.re_norm_stresses[1],
                levels,
                cmap,
                fontsize,
            )

            # Plot uv
            self._plot_contour(
                axs[2],
                Y_grid,
                X_grid,
                rs_uv,
                rs_uv_pred,
                self.plot_config.axes.re_sh_stresses[0],
                levels,
                cmap,
                fontsize,
            )

        else:
            fig, axs = plt.subplots(
                3,
                2,
                figsize=(
                    5 * self.plot_config.figure.figsize[0],
                    2 * self.plot_config.figure.figsize[1],
                ),
            )
            # Plot uu
            self._plot_contour(
                axs[0, 0],
                Y_grid,
                X_grid,
                rs_uu,
                rs_uu_pred,
                self.plot_config.axes.re_norm_stresses[0],
                levels,
                cmap,
                fontsize,
            )

            # Plot vv
            self._plot_contour(
                axs[0, 1],
                Y_grid,
                X_grid,
                rs_vv,
                rs_vv_pred,
                self.plot_config.axes.re_norm_stresses[1],
                levels,
                cmap,
                fontsize,
            )

            # Plot ww
            self._plot_contour(
                axs[1, 0],
                Y_grid,
                X_grid,
                rs_ww,
                rs_ww_pred,
                self.plot_config.axes.re_norm_stresses[2],
                levels,
                cmap,
                fontsize,
            )

            # Plot uv
            self._plot_contour(
                axs[1, 1],
                Y_grid,
                X_grid,
                rs_uv,
                rs_uv_pred,
                self.plot_config.axes.re_sh_stresses[0],
                levels,
                cmap,
                fontsize,
            )

            # Plot uw
            self._plot_contour(
                axs[2, 0],
                Y_grid,
                X_grid,
                rs_uw,
                rs_uw_pred,
                self.plot_config.axes.re_sh_stresses[1],
                levels,
                cmap,
                fontsize,
            )

            # Plot vw
            self._plot_contour(
                axs[2, 1],
                Y_grid,
                X_grid,
                rs_vw,
                rs_vw_pred,
                self.plot_config.axes.re_sh_stresses[2],
                levels,
                cmap,
                fontsize,
            )

        if self.plot_config.figure.tight_layout:
            fig.tight_layout()

        if pdf is not None:
            pdf.savefig(fig)
            plt.savefig(
                self.eval_dir + f"/pred_snaps-{self.snaps}-Reynolds_stresses2.pdf",
                dpi=self.plot_config.figure.dpi,
            )
        else:
            plt.savefig(
                self.eval_dir + f"/pred_snaps-{self.snaps}-Reynolds_stresses2.pdf",
                dpi=self.plot_config.figure.dpi,
            )

        plt.close(fig)

        return pdf

    def joint_pdfs(self, inp_comp=None, axis=None):
        """
        Calculate the joint probability density function (PDF) of input components and their corresponding axis values.

        Parameters:
        inp_comp (ndarray): The input components to calculate the PDF for.
        axis (ndarray): The axis values corresponding to the input components.

        Returns:
        xi (ndarray): The meshgrid x-values for contour plotting.
        yi (ndarray): The meshgrid y-values for contour plotting.
        zi_norm (ndarray): The normalized joint PDF values for the meshgrid.
        """
        from scipy.stats import gaussian_kde

        u_copy = inp_comp.reshape(-1)
        x_copy = (
            np.tile(axis.reshape(1, inp_comp.shape[1]), (inp_comp.shape[0], 1))
        ).reshape(-1)

        # Calculate the point density
        xy = np.vstack([x_copy, u_copy])
        gaussian_kde(xy)

        # Create a grid for contour plotting
        # print(u_.min())
        # xi, yi = np.linspace(axis.min(), axis.max(), 100), np.linspace(inp_comp.min(), inp_comp.max(), 100)
        xi, yi = np.linspace(-1, 5, 100), np.linspace(-0.4, 0.4, 100)
        xi, yi = np.meshgrid(xi, yi)
        zi = gaussian_kde(xy)(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

        zi_norm = zi / np.max(zi)

        return xi, yi, zi_norm

    def plot_joint_pdfs(self, x=None, y=None, z=None, data="line-x", pdf=None):
        """
        Plot the joint probability density functions (PDFs) for the components of the ground truth and predicted data.

        Parameters:
        x (ndarray): The x-coordinates for data selection.
        y (ndarray): The y-coordinates for data selection.
        z (ndarray): The z-coordinates for data selection.
        data (str): The type of data to process ('line-x', 'line-y', etc.).
        levels (int): The number of contour levels to plot.
        pdf (PdfPages): An optional PdfPages object to save the plots to a PDF file.

        Returns:
        pdf (PdfPages): The PdfPages object if provided, with the plots saved.
        """
        levels = self.plot_config.figure.jpdf_level
        labels = self.plot_config.axes.fluc_label
        fontsize = self.plot_config.axes.fontsize

        for i in range(self.gtruth.shape[1]):
            component = get_data_for_stats(
                self.gtruth[:, i],
                x=x,
                y=y,
                z=z,
                input_data_type=self.input_data_type,
                data=data,
                ds_ratio=self.ds_ratio,
            )
            component_pred = get_data_for_stats(
                self.pred[:, i],
                x=x,
                y=y,
                z=z,
                input_data_type=self.input_data_type,
                data=data,
                ds_ratio=self.ds_ratio,
            )

            # print("creating pdfs")
            xi, yi, zi_norm = self.joint_pdfs(
                component, axis=self.x_axis
            )  # print("gtruth done")
            xi, yi, zi_norm_pred = self.joint_pdfs(
                component_pred, axis=self.x_axis
            )  # print("pred done")

            # Plotting
            fig, ax = plt.subplots(
                figsize=(
                    self.plot_config.figure.figsize[0],
                    self.plot_config.figure.figsize[0],
                )
            )

            contour = ax.contour(
                xi, yi, zi_norm, levels=levels, cmap=self.plot_config.plot.jpdf_cmap
            )
            ax.contour(
                xi, yi, zi_norm_pred, levels=levels, colors="black", linestyles="dotted"
            )

            # plt.title("Joint PDF of $x$ and $u'$", fontsize=self.fontsize)
            ax.set_xlabel(self.plot_config.axes.x_label, fontsize=fontsize)
            ax.set_ylabel(labels[i], fontsize=fontsize)
            fig.colorbar(contour, ax=ax)
            # ax.ylim(-0.4, 0.4)

            fig.tight_layout()

            if pdf is not None:
                pdf.savefig(fig)
                plt.savefig(
                    self.eval_dir + f"/pred_snaps-{self.snaps}-jpdfs-{i}.pdf",
                    dpi=self.plot_config.figure.dpi,
                )
            else:
                plt.savefig(
                    self.eval_dir + f"/pred_snaps-{self.snaps}-jpdfs-{i}.pdf",
                    dpi=self.plot_config.figure.dpi,
                )

            plt.close(fig)

        return pdf

    def PSD_welch(self, inp_comp=None, nperseg=256):
        """
        Compute the Power Spectral Density (PSD) of an input component using the Welch method.

        Parameters:
        inp_comp (ndarray): The input component for which the PSD is to be computed.
        nperseg (int): Length of each segment for the Welch method (default is 256).

        Returns:
        frequencies (ndarray): Array of sample frequencies.
        psd (ndarray): Power spectral density of the input component.
        """
        from scipy.signal import welch

        inp_comp = inp_comp.flatten()

        fs = self.time[1] - self.time[0]

        frequencies, psd = welch(inp_comp, fs=fs, nperseg=nperseg)

        return frequencies, psd

    def plot_PSD(
        self, locations=None, y=None, z=None, data="point", nperseg=256, pdf=None
    ):
        """
        Plot the Power Spectral Density (PSD) for ground truth and predicted data using the Welch method.

        Parameters:
        locations (list): List of x locations for point data.
        y (float): y-coordinate for line-x data.
        z (float): z-coordinate for line-x data.
        data (str): Type of data, either 'point' or 'line-x'.
        nperseg (int): Length of each segment for the Welch method (default is 256).
        pdf (PdfPages): PdfPages object to save plots to a PDF file.

        Returns:
        pdf (PdfPages): PdfPages object with saved figures.
        """
        colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#e377c2", "#17becf"]
        # Blue, Green, Red, Orange, Magenta, Cyan
        labels = ["u", "v", "w"]
        # Create the fig and axis
        linewidth = 2

        if data == "point":
            assert self.gtruth.shape[1] == self.pred.shape[1]
            for k in range(self.gtruth.shape[1]):
                fig, ax = plt.subplots(figsize=(12, 8))

                for i, x in enumerate(locations):
                    # Get ground truth and predicted data
                    comp_x = get_data_for_stats(
                        self.gtruth[:, k],
                        x=x,
                        y=y,
                        z=z,
                        input_data_type=self.input_data_type,
                        data=data,
                        ds_ratio=self.ds_ratio,
                    )
                    comp_x_pred = get_data_for_stats(
                        self.pred[:, k],
                        x=x,
                        y=y,
                        z=z,
                        input_data_type=self.input_data_type,
                        data=data,
                        ds_ratio=self.ds_ratio,
                    )

                    # Compute PSD using Welch's method
                    frequency, psd = self.PSD_welch(comp_x, nperseg=nperseg)
                    frequency, psd_p = self.PSD_welch(comp_x_pred, nperseg=nperseg)

                    # Plot the PSD
                    ax.loglog(
                        frequency,
                        psd,
                        label=rf"GT @ $\frac{{x}}{{h}}=$ {x}",
                        linestyle="-",
                        linewidth=linewidth,
                        color=colors[i],
                    )
                    ax.loglog(
                        frequency,
                        psd_p,
                        label=rf"pred @ $\frac{{x}}{{h}}=$ {x}",
                        color=colors[i],
                        marker="o",
                        markersize=5,
                        linestyle="none",
                    )

                # Add titles and labels
                ax.set_title(
                    "Power Spectral Density vs Frequency (Welch Method)",
                    fontsize=self.fontsize,
                )
                ax.set_xlabel("Frequency [Hz]", fontsize=self.fontsize)
                ax.set_ylabel("PSD [V**2/Hz]", fontsize=self.fontsize)

                # Customize grid and legend
                ax.grid(True, which="both", linestyle="--", linewidth=0.5)
                ax.legend(fontsize=12, loc="upper right", framealpha=0.9)

                # Tight layout for better spacing
                fig.tight_layout()
                if pdf is not None:
                    pdf.savefig(fig)
                else:
                    plt.savefig(
                        self.eval_dir
                        + f"/pred_snaps-{self.snaps}-PSD-pt-{labels[k]}.png",
                        dpi=self.plot_config.figure.dpi,
                    )
                plt.close(fig)

        elif data == "line-x":
            for k in range(self.gtruth.shape[1]):
                fig, ax = plt.subplots(figsize=(12, 8))

                # Get ground truth and predicted data
                comp_x = get_data_for_stats(
                    self.gtruth[:, k],
                    x=None,
                    y=y,
                    z=z,
                    input_data_type=self.input_data_type,
                    data=data,
                    ds_ratio=self.ds_ratio,
                )
                comp_x_pred = get_data_for_stats(
                    self.pred[:, k],
                    x=None,
                    y=y,
                    z=z,
                    input_data_type=self.input_data_type,
                    data=data,
                    ds_ratio=self.ds_ratio,
                )

                # Compute PSD using Welch's method
                frequency, psd = self.PSD_welch(comp_x, nperseg=nperseg)
                frequency, psd_p = self.PSD_welch(comp_x_pred, nperseg=nperseg)

                # Plot the PSD
                ax.loglog(
                    frequency,
                    psd,
                    label=rf"GT @ $\frac{{y}}{{h}}=$ {y}",
                    linestyle="-",
                    linewidth=linewidth,
                    color="k",
                )
                ax.loglog(
                    frequency,
                    psd_p,
                    label=rf"pred @ $\frac{{y}}{{h}}=$ {y}",
                    color="k",
                    marker="o",
                    markersize=5,
                    linestyle="none",
                )

                # Add titles and labels
                ax.set_title(
                    "Power Spectral Density vs Frequency (Welch Method)", fontsize=16
                )
                ax.set_xlabel("Frequency [Hz]", fontsize=14)
                ax.set_ylabel("PSD [V**2/Hz]", fontsize=14)

                # Customize grid and legend
                ax.grid(True, which="both", linestyle="--", linewidth=0.5)
                ax.legend(fontsize=12, loc="upper right", framealpha=0.9)

                # Tight layout for better spacing
                fig.tight_layout()

                if pdf is not None:
                    pdf.savefig(fig)
                else:
                    plt.savefig(
                        self.eval_dir
                        + f"/pred_snaps-{self.snaps}-PSD_linex-{labels[k]}.png",
                        dpi=self.plot_config.figure.dpi,
                    )
                plt.close(fig)

        return pdf

    def plot_probe_signal(self, n=200, x=None, y=None, z=None):
        """
        Plot time series of velocity components at a probe location.

        This method creates visualization of velocity fluctuations over time at a
        specified spatial location. It compares ground truth and predicted velocity
        signals side-by-side for validation. Multiple snapshots are randomly selected
        for display.

        Parameters
        ----------
        n : int, optional
            Number of time steps to display in the plot. Default is 200.
        x : float, optional
            X-coordinate of the probe location in physical units. If None, data
            extraction behavior depends on input_data_type.
        y : float, optional
            Y-coordinate of the probe location in physical units.
        z : float, optional
            Z-coordinate of the probe location in physical units (for 3D data).

        Returns
        -------
        None

        Notes
        -----
        The method plots three subplots if the data contains three velocity components
        (u, v, w), or two subplots for 2D data (u, v). Each subplot shows:
        - Ground truth velocity fluctuations (black line)
        - Predicted velocity fluctuations (red line)
        - Grid lines for easier reading
        - Legend with location information

        The plot includes:
        - X-axis: Number of snapshots
        - Y-axis: Velocity fluctuation magnitude
        - Y-axis limits: [-0.5, 0.5] for standard normalization

        Uses the internal ds_ratio attribute for coordinate conversion.

        Examples
        --------
        >>> evaluator = StatisticalEvaluation(gtruth=gtruth, pred=pred, ...)
        >>> evaluator.plot_probe_signal(n=200, x=1.0, y=0.5, z=None)
        """

        labels = ["u", "v", "w"]
        for i in range(self.gtruth.shape[1]):
            u_pt = get_data_for_stats(
                self.gtruth[:, i],
                x=x,
                y=y,
                z=z,
                input_data_type="2D",
                data="point",
                mean_over_time=False,
                ds_ratio=config.dataset.ds_ratio,
            )  # Check this while integrating the function
            u_pt_p = get_data_for_stats(
                self.pred[:, i],
                x=x,
                y=y,
                z=z,
                input_data_type="2D",
                data="point",
                mean_over_time=False,
                ds_ratio=config.dataset.ds_ratio,
            )

            # print(u_pt.shape); print(u_pt_p.shape)
            random_integers = np.random.randint(0, u_pt.shape[0], n)

            # Plot settings
            plt.figure(figsize=(12, 8))

            plt.plot(
                u_pt[random_integers],
                label=rf"${labels[i]}'$ @" + r"$\frac{x}{h}= 1$ ",
                linestyle="-",
                color="k",
            )
            plt.plot(
                u_pt_p[:n],
                label=rf"${labels[i]}_p'$ @" + r"$\frac{x}{h}= 1$ ",
                linestyle="-",
                color="r",
            )

            # Add grid lines
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)

            # Add titles and labels
            plt.xlabel("num of snapshots", fontsize=14)
            plt.ylabel(rf"${labels[i]}'$", fontsize=14)

            # Add a legend
            plt.legend(loc="best", fontsize=12)

            # Add limits for x-axis to ensure both series are easily comparable
            plt.xlim([0, n])
            plt.ylim([-0.5, 0.5])
            # Add a scientific look with a tighter layout
            plt.tight_layout()

            # Show the plot
            plt.show()

    def main(self, num=None, locations=None, y=None, pdf=None):
        """
        Main function to execute various plotting and analysis routines.

        Parameters:
        num (int): The index or identifier for the visual comparison (optional).
        locations (list): List of x locations for PSD plotting (optional).
        y (float): y-coordinate for line-x data in various plots (optional).
        pdf (PdfPages): PdfPages object to save plots to a PDF file (optional).

        Returns:
        pdf (PdfPages): PdfPages object with saved figures.
        """
        print("Getting a visual glimps of prediction")
        pdf = self.plot_vis_compare(num=0, pdf=pdf)

        print("Computing Reynolds Stresses")
        pdf = self.plot_reynolds_stresses(x=None, y=0.5, z=None, pdf=pdf)
        pdf = self.plot_reynolds_stress_planes(
            x=None, y=None, z=None, data="plane", pdf=pdf
        )

        print("Getting pdfs for flow fields")
        pdf = self.plot_joint_pdfs(x=None, y=0.5, z=None, data="line-x", pdf=pdf)

        print("Computing Power Spectral Density")
        pdf = self.plot_PSD(
            locations=None, y=0.5, z=None, data="line-x", nperseg=256, pdf=pdf
        )

        return pdf


if __name__ == "__main__":
    import sys

    sys.path.append("../")

    from configs.OneObs2D_ds1_10M import config_dict
    from libs import runner
    from libs.utils import get_data_for_stats

    from matplotlib.backends.backend_pdf import PdfPages

    print(
        "Class to evaluate statistics for given ground truth and predictions. Outputs a pdf file with relevant statistics"
    )
    print("Starting dummy implementation!! ")
    # Dummy data and configuration
    gtruth = np.random.rand(5, 2, 60, 20)  # Example ground truth data
    pred = np.random.rand(5, 2, 60, 20)  # Example predicted data
    x_axis = np.linspace(-1, 5, 60)
    y_axis = np.linspace(0, 2, 20)
    time = np.linspace(0, 10, 5)  # Example time array

    config = runner.dict2namespace(config_dict)

    # Instantiate the class
    evaluator = StatisticalEvaluation(
        gtruth=gtruth,
        pred=pred,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=None,
        time=time,
        input_data_type="2D",
        data="line-x",
        config=config,
    )

    # Setup PDF output
    pdf = PdfPages("./test_stats_eval.pdf")
    pdf = evaluator.main(
        num=0, locations=[0.5], y=0.5, pdf=pdf
    )  # locations = locations along x, where the PSD needs to be calculated
    pdf.close()
