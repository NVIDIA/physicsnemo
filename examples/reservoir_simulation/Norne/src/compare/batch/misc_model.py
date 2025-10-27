"""
SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

=====================================================================
                    MACHINE LEARNING MODEL UTILITIES
=====================================================================

This module provides utilities for machine learning model operations in reservoir
simulation. It includes functions for model creation, training, evaluation,
and visualization.

Key Features:
- Model creation and configuration
- Training and evaluation utilities
- Visualization and plotting functions
- Performance metrics calculation
- Model persistence and loading
- Data preprocessing for ML models

Usage:
    from compare.batch.misc_model import (
        create_model,
        train_model,
        evaluate_model,
        plot_model_results
    )

Inputs:
    - Model configuration parameters
    - Training data arrays
    - Model specifications
    - Visualization parameters

Outputs:
    - Trained models
    - Evaluation metrics
    - Visualization plots
    - Performance statistics

@Author : Clement Etienam
"""

# 🛠 Standard Library
import os
import time
import logging
from collections import OrderedDict

# 🔧 Third-party Libraries
import numpy as np
import numpy.matlib
import torch
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib import cm

# 📦 Local Modules
from physicsnemo.models.fno import FNO
from physicsnemo.models.module import Module
from compare.batch.misc_operations import (
    ProgressBar,
    ShowBar,
)


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger("Forward problem")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def Plot_PhyNeMo(
    ax, nx, ny, nz, Truee, N_injw, N_pr, N_injg, varii, injectors, producers, gass
):
    # matplotlib.use('Agg')
    Pressz = np.reshape(Truee, (nx, ny, nz), "F")
    avg_2d = np.mean(Pressz, axis=2)
    avg_2d[avg_2d == 0] = np.nan
    maxii = max(Pressz.ravel())
    minii = min(Pressz.ravel())
    Pressz = Pressz / maxii

    masked_Pressz = Pressz
    colors = plt.cm.jet(masked_Pressz)
    colors[np.isnan(Pressz), :3] = 1  # set color to white for NaN values
    # alpha = np.where(np.isnan(Pressz), 0.0, 0.8) # set alpha to 0 for NaN values
    norm = mpl.colors.Normalize(vmin=minii, vmax=maxii)

    arr_3d = Pressz

    x, y, z = np.indices((arr_3d.shape))
    x = x + 0.5
    y = y + 0.5
    z = z + 0.5
    ax.voxels(arr_3d, facecolors=colors, alpha=0.5, edgecolor="none", shade=True)
    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    ax.set_xlim(0, arr_3d.shape[0])
    ax.set_ylim(0, arr_3d.shape[1])
    ax.set_zlim(0, arr_3d.shape[2])
    ax.grid(False)
    ax.set_box_aspect([nx, ny, nz])
    ax.set_proj_type("ortho")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Remove the tick lines on each axis
    ax.xaxis._axinfo["tick"]["inward_factor"] = 0
    ax.xaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax.yaxis._axinfo["tick"]["inward_factor"] = 0
    ax.yaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax.zaxis._axinfo["tick"]["inward_factor"] = 0
    ax.zaxis._axinfo["tick"]["outward_factor"] = 0.4
    # Set the azimuth and elevation to make the plot brighter
    ax.view_init(elev=30, azim=45)

    n_inj = N_injw  # Number of injectors
    n_prod = N_pr  # Number of producers

    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[-1])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 7)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 7], "blue", linewidth=1)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="blue",
            weight="bold",
            fontsize=5,
        )

    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[-1])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 5)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 5], "g", linewidth=1)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="g",
            weight="bold",
            fontsize=5,
        )
    for mm in range(N_injg):
        usethis = gass[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[-1])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 5)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "red", linewidth=1)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="red",
            weight="bold",
            fontsize=5,
        )
    blue_line = mlines.Line2D([], [], color="blue", linewidth=2, label="water injector")
    green_line = mlines.Line2D(
        [], [], color="green", linewidth=2, label="oil/water/gas producer"
    )
    red_line = mlines.Line2D([], [], color="red", linewidth=2, label="gas injectors")
    ax.legend(handles=[blue_line, green_line, red_line], loc="lower left", fontsize=9)

    cbar = plt.colorbar(m, ax=ax, orientation="horizontal", shrink=0.5)
    if varii == "perm":
        cbar.set_label("Log K(mD)", fontsize=12)
        ax.set_title(
            "Permeability Field with well locations", fontsize=12, weight="bold"
        )
    elif varii == "water PhyNeMo":
        cbar.set_label("water saturation", fontsize=12)
        ax.set_title("water saturation -PhyNeMo", fontsize=12, weight="bold")
    elif varii == "water Numerical":
        cbar.set_label("water saturation", fontsize=12)
        ax.set_title("water saturation - Numerical(Flow)", fontsize=12, weight="bold")
    elif varii == "water diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "water saturation - (Numerical(Flow) -PhyNeMo))", fontsize=12, weight="bold"
        )

    elif varii == "oil PhyNeMo":
        cbar.set_label("Oil saturation", fontsize=12)
        ax.set_title("Oil saturation -PhyNeMo", fontsize=12, weight="bold")

    elif varii == "oil Numerical":
        cbar.set_label("Oil saturation", fontsize=12)
        ax.set_title("Oil saturation - Numerical(Flow)", fontsize=12, weight="bold")

    elif varii == "oil diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "oil saturation - (Numerical(Flow) -PhyNeMo))", fontsize=12, weight="bold"
        )

    elif varii == "gas PhyNeMo":
        cbar.set_label("Gas saturation", fontsize=12)
        ax.set_title("Gas saturation -PhyNeMo", fontsize=12, weight="bold")

    elif varii == "gas Numerical":
        cbar.set_label("Gas saturation", fontsize=12)
        ax.set_title("Gas saturation - Numerical(Flow)", fontsize=12, weight="bold")

    elif varii == "gas diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "gas saturation - (Numerical(Flow) -PhyNeMo))", fontsize=12, weight="bold"
        )

    elif varii == "pressure PhyNeMo":
        cbar.set_label("pressure", fontsize=12)
        ax.set_title("Pressure -PhyNeMo", fontsize=12, weight="bold")

    elif varii == "pressure Numerical":
        cbar.set_label("pressure", fontsize=12)
        ax.set_title("Pressure -Numerical(Flow)", fontsize=12, weight="bold")

    elif varii == "pressure diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "Pressure - (Numerical(Flow) -PhyNeMo))", fontsize=12, weight="bold"
        )

    elif varii == "porosity":
        cbar.set_label("porosity", fontsize=12)
        ax.set_title("Porosity Field", fontsize=12, weight="bold")
    cbar.mappable.set_clim(minii, maxii)


def compute_metrics(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    TSS = np.sum((y_true - y_true_mean) ** 2)
    RSS = np.sum((y_true - y_pred) ** 2)

    R2 = 1 - (RSS / TSS)
    L2_accuracy = 1 - np.sqrt(RSS) / np.sqrt(TSS)
    return R2, L2_accuracy


def process_step(
    kk,
    steppi,
    dt,
    pressure,
    effectiy,
    pressure_true,
    Swater,
    Swater_true,
    Soil,
    Soil_true,
    Sgas,
    Sgas_true,
    nx,
    ny,
    nz,
    N_injw,
    N_pr,
    N_injg,
    injectors,
    producers,
    gass,
    fol,
    fol1,
    Accuracy_presure,
    Accuracy_oil,
    Accuracy_water,
    Accuracy_gas,
):
    os.chdir(fol)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk - 1, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)

    current_time = dt[kk]
    # Time_vector[kk] = current_time

    f_3 = plt.figure(figsize=(20, 20), dpi=200)

    look = ((pressure[0, kk, :, :, :]) * effectiy)[:, :, ::-1]

    lookf = ((pressure_true[0, kk, :, :, :]) * effectiy)[:, :, ::-1]
    # lookf = lookf * pini_alt
    diff1 = ((abs(look - lookf)) * effectiy)[:, :, ::-1]

    ax1 = f_3.add_subplot(4, 3, 1, projection="3d")
    Plot_PhyNeMo(
        ax1,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "pressure PhyNeMo",
        injectors,
        producers,
        gass,
    )
    ax2 = f_3.add_subplot(4, 3, 2, projection="3d")
    Plot_PhyNeMo(
        ax2,
        nx,
        ny,
        nz,
        lookf,
        N_injw,
        N_pr,
        N_injg,
        "pressure Numerical",
        injectors,
        producers,
        gass,
    )
    ax3 = f_3.add_subplot(4, 3, 3, projection="3d")
    Plot_PhyNeMo(
        ax3,
        nx,
        ny,
        nz,
        diff1,
        N_injw,
        N_pr,
        N_injg,
        "pressure diff",
        injectors,
        producers,
        gass,
    )
    R2p, L2p = compute_metrics(look.ravel(), lookf.ravel())
    Accuracy_presure[kk, 0] = R2p
    Accuracy_presure[kk, 1] = L2p

    look = ((Swater[0, kk, :, :, :]) * effectiy)[:, :, ::-1]
    lookf = ((Swater_true[0, kk, :, :, :]) * effectiy)[:, :, ::-1]
    diff1 = ((abs(look - lookf)) * effectiy)[:, :, ::-1]
    ax1 = f_3.add_subplot(4, 3, 4, projection="3d")
    Plot_PhyNeMo(
        ax1,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "water PhyNeMo",
        injectors,
        producers,
        gass,
    )
    ax2 = f_3.add_subplot(4, 3, 5, projection="3d")
    Plot_PhyNeMo(
        ax2,
        nx,
        ny,
        nz,
        lookf,
        N_injw,
        N_pr,
        N_injg,
        "water Numerical",
        injectors,
        producers,
        gass,
    )
    ax3 = f_3.add_subplot(4, 3, 6, projection="3d")
    Plot_PhyNeMo(
        ax3,
        nx,
        ny,
        nz,
        diff1,
        N_injw,
        N_pr,
        N_injg,
        "water diff",
        injectors,
        producers,
        gass,
    )
    R2w, L2w = compute_metrics(look.ravel(), lookf.ravel())
    Accuracy_water[kk, 0] = R2w
    Accuracy_water[kk, 1] = L2w

    look = Soil[0, kk, :, :, :]
    look = (look)[:, :, ::-1]
    lookf = Soil_true[0, kk, :, :, :]
    lookf = (lookf)[:, :, ::-1]
    diff1 = (abs(look - lookf))[:, :, ::-1]
    ax1 = f_3.add_subplot(4, 3, 7, projection="3d")
    Plot_PhyNeMo(
        ax1,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "oil PhyNeMo",
        injectors,
        producers,
        gass,
    )
    ax2 = f_3.add_subplot(4, 3, 8, projection="3d")
    Plot_PhyNeMo(
        ax2,
        nx,
        ny,
        nz,
        lookf,
        N_injw,
        N_pr,
        N_injg,
        "oil Numerical",
        injectors,
        producers,
        gass,
    )
    ax3 = f_3.add_subplot(4, 3, 9, projection="3d")
    Plot_PhyNeMo(
        ax3,
        nx,
        ny,
        nz,
        diff1,
        N_injw,
        N_pr,
        N_injg,
        "oil diff",
        injectors,
        producers,
        gass,
    )
    R2o, L2o = compute_metrics(look.ravel(), lookf.ravel())
    Accuracy_oil[kk, 0] = R2o
    Accuracy_oil[kk, 1] = L2o
    look = ((Sgas[0, kk, :, :, :]) * effectiy)[:, :, ::-1]
    lookf = ((Sgas_true[0, kk, :, :, :]) * effectiy)[:, :, ::-1]
    diff1 = (abs(look - lookf))[:, :, ::-1]
    ax1 = f_3.add_subplot(4, 3, 10, projection="3d")
    Plot_PhyNeMo(
        ax1,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "gas PhyNeMo",
        injectors,
        producers,
        gass,
    )
    ax2 = f_3.add_subplot(4, 3, 11, projection="3d")
    Plot_PhyNeMo(
        ax2,
        nx,
        ny,
        nz,
        lookf,
        N_injw,
        N_pr,
        N_injg,
        "gas Numerical",
        injectors,
        producers,
        gass,
    )
    ax3 = f_3.add_subplot(4, 3, 12, projection="3d")
    Plot_PhyNeMo(
        ax3,
        nx,
        ny,
        nz,
        diff1,
        N_injw,
        N_pr,
        N_injg,
        "gas diff",
        injectors,
        producers,
        gass,
    )
    R2g, L2g = compute_metrics(look.ravel(), lookf.ravel())
    Accuracy_gas[kk, 0] = R2g
    Accuracy_gas[kk, 1] = L2g
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    tita = "Timestep --" + str(current_time) + " days"
    plt.suptitle(tita, fontsize=16)
    # plt.savefig('Dynamic' + str(int(kk)))
    plt.savefig("Dynamic" + str(int(kk)))
    plt.clf()
    plt.close()
    return R2p, L2p, R2w, L2w, R2o, L2o, R2g, L2g
    os.chdir(fol1)


def load_modell(model, model_path, is_distributed, device, express, namee):
    logger = setup_logging()
    logger.info(f"🔄 Loading model from: {model_path}")

    if express == 1:
        # 🔥 If it fails, load as a state_dict model
        state_dict = torch.load(model_path, map_location=device)

        # ✅ Handle Distributed Data Parallel (Remove `module.` prefix if needed)
        if is_distributed == 1:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:] if k.startswith("module.") else k] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

        # ✅ Move model to correct device & set to eval mode
        model = model.to(device)
        model.eval()
    else:
        checkpoint = torch.load(model_path, map_location=device)
        if namee == "PRESSURE":
            state_dict = checkpoint["surrogate_pressure_state_dict"]
        if namee == "SWAT":
            state_dict = checkpoint["surrogate_saturation_state_dict"]

        if namee == "SOIL":
            state_dict = checkpoint["surrogate_oil_state_dict"]

        if namee == "SGAS":
            state_dict = checkpoint["surrogate_gas_state_dict"]
        if namee == "PEACEMANN":
            state_dict = checkpoint["surrogate_peacemann_state_dict"]
        # ✅ Handle Distributed Data Parallel (Remove `module.` prefix if needed)
        if is_distributed == 1:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:] if k.startswith("module.") else k] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

        # ✅ Move model to correct device & set to eval mode
        model = model.to(device)
        model.eval()

    return model


class FNOModel(Module):
    def __init__(
        self,
        input_dim,
        steppi,
        output_shape,
        device,
        num_layers=4,
        decoder_layers=1,
        decoder_layer_size=32,
        dimension=3,
        latent_channels=32,
        num_fno_layers=4,
        padding=8,
        num_fno_modes=16,
    ):
        super().__init__()

        # Initialize FNO model
        self.fno = FNO(
            in_channels=input_dim,
            out_channels=output_shape * steppi,
            decoder_layers=decoder_layers,
            decoder_layer_size=decoder_layer_size,
            dimension=dimension,
            latent_channels=latent_channels,
            num_fno_layers=num_layers,
            padding=padding,
            num_fno_modes=num_fno_modes,
        ).to(torch.device(device))  # Explicit device conversion

        # Meta attribute for PhyNeMo compatibility
        self.meta = type("", (), {})()  # Empty object
        self.meta.name = "fno_model"

    def forward(self, x):
        return self.fno(x)


def create_fno_model(
    input_dim,
    steppi,
    output_shape,
    device,
    num_layers=4,
    decoder_layers=1,
    decoder_layer_size=32,
    dimension=3,
    latent_channels=32,
    num_fno_layers=4,
    padding=8,
    num_fno_modes=16,
):
    """
    Create a Fourier Neural Operator (FNO) model wrapped in a compatible PhyNeMo Module.

    Parameters:
    -----------
    input_dim : int
        The number of input features (e.g., the number of spatial points).
    steppi : int
        Number of time steps or resolution in the output.
    output_shape : int
        The number of distinct outputs to predict.
    device : str
        Device to create the model on ('cpu' or 'cuda').
    num_layers : int, optional
        The number of layers in the FNO. Default is 4.
    decoder_layers : int, optional
        The number of decoder layers. Default is 1.
    decoder_layer_size : int, optional
        The size of each decoder layer. Default is 32.
    dimension : int, optional
        The dimensionality of the FNO (2D or 3D). Default is 2.
    latent_channels : int, optional
        Number of latent channels. Default is 32.
    num_fno_layers : int, optional
        Number of FNO layers. Default is 4.
    padding : int, optional
        Padding for the FNO. Default is 8.
    num_fno_modes : int, optional
        Number of Fourier modes to use. Default is 16.

    Returns:
    --------
    fno_model : FNOModel
        Initialized FNO model ready for inference or training.
    """
    # Validate arguments
    if dimension not in [1, 2, 3]:
        raise ValueError(f"Invalid dimension: {dimension}. Must be 1, 2 or 3.")

    # Create and return FNOModel instance
    return FNOModel(
        input_dim=input_dim,
        steppi=steppi,
        output_shape=output_shape,
        device=device,
        num_layers=num_layers,
        decoder_layers=decoder_layers,
        decoder_layer_size=decoder_layer_size,
        dimension=dimension,
        latent_channels=latent_channels,
        num_fno_layers=num_fno_layers,
        padding=padding,
        num_fno_modes=num_fno_modes,
    )
