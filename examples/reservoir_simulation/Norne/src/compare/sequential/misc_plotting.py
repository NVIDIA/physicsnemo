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
                    SEQUENTIAL PLOTTING UTILITIES
=====================================================================

This module provides plotting utilities for sequential processing of FVM
surrogate model comparisons. It includes functions for data visualization,
result plotting, and performance analysis.

Key Features:
- Data type definitions for simulation data
- Plotting functions for RSM percentiles
- Marker and visualization utilities
- 2D plotting capabilities

Usage:
    from compare.sequential.misc_plotting import (
        simulation_data_types,
        Plot_RSM_percentile,
        Add_marker,
        Plot_2D
    )

@Author : Clement Etienam
"""

# 🛠 Standard Library
import logging

# 🔧 Third-party Libraries
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib import cm

# 📦 Local Modules


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def simulation_data_types():
    SUPPORTED_DATA_TYPES = {
        "INTE": (4, "i", 1000),
        "REAL": (4, "f", 1000),
        "LOGI": (4, "i", 1000),
        "DOUB": (8, "d", 1000),
        "CHAR": (8, "8s", 105),
        "MESS": (8, "8s", 105),
        "C008": (8, "8s", 105),
    }

    type_dict = {
        b"INTE": "i",
        b"CHAR": "8s",
        b"REAL": "f",
        b"DOUB": "d",
        b"LOGI": "4s",
        b"MESS": "?",
    }

    ecl_extensions = [
        ".DATA",
        ".DBG",
        ".ECLEND",
        ".EGRID",
        ".FEGRID",
        ".FGRID",
        ".FINIT",
        ".FINSPEC",
        ".FRFT",
        ".FRSSPEC",
        ".FSMSPEC",
        ".FUNRST",
        ".FUNSMRY",
        ".GRID",
        ".INIT",
        ".INSPEC",
        ".MSG",
        ".PRT",
        ".RFT",
        ".RSM",
        ".RSSPEC",
        ".SMSPEC",
        ".UNRST",
        ".UNSMRY",
        ".dbprtx",
    ]

    dynamic_props = [
        "SEQNUM",
        "PRESSURE",
        "SWAT",
        "SGAS",
        "SOIL",
        "RS",
        "RV",
        "RSSAT",
        "RVSAT",
        "STATES",
        "OWC",
        "OGC",
        "GWC",
        "EOWC",
        "EOGC",
        "OILAPI",
        "SDENO",
        "FIPOIL",
        "RFIPOIL",
        "FIPGAS",
        "RFIPGAS",
        "FIPWAT",
        "RFIPWAT",
        "SFIPOIL",
        "SFIPGAS",
        "SFIPWAT",
        "SFIPPLY",
        "RFIPPLY",
        "SFIPSAL",
        "RFIPSAL",
        "SFIPSOL",
        "SFIPGGI",
        "RFIPOIL",
        "RFIPGAS",
        "RFIPWAT",
        "RFIPSOL",
        "RFIPGGI",
        "OIL-POTN",
        "GAS-POTN",
        "WAT-POTN",
        "POLYMER",
        "PADS",
        "PLYTRRFA",
        "POLYMAX",
        "SALT",
        "TEMP",
        "XMF",
        "YMF",
        "ZMF",
        "SSOL",
        "PBUB",
        "PDEW",
        "SURFACT",
        "SURFADS",
        "SURFMAX",
        "SURFCNM",
        "SURFST",
        "GGI",
        "WAT-PRES",
        "GAS-PRES",
        "OIL-VISC",
        "WAT-VISC",
        "GAS-VISC",
        "OIL-DEN",
        "WAT-DEN",
        "GAS-DEN",
        "DRAINAGE",
        "DRAINMIN",
        "PCOW",
        "PCOG",
        "1OVERBO",
        "1OVERBW",
        "1OVERBG",
        "POT_CORR",
        "OILKR",
        "WATKR",
        "GASKR",
        "HYDH",
        "HYDHFW",
        "PORV",
        "RPORV",
        "FOAM",
        "FOAMADS",
        "FOAMMAX",
        "FOAMDCY",
        "FOAMCNM",
        "FOAM_HL",
        "FOAMMOB",
        "ALKALINE",
        "ALKADS",
        "ALKMAX",
        "STMALK",
        "SFADALK",
        "PLADALK",
        "PADMAX",
        "CATSURF",
        "CATROCK",
        "ESALSUR",
        "ESALPLY",
        "COALGAS",
        "COALSOLV",
        "GASSATC",
        "MLANG",
        "MLANGSLV",
        "SWMIN",
        "SWMAX",
        "ISTHW",
        "SOMAX",
        "ISTHG",
        "SGMIN",
        "SGMAX",
        "PRESROCC",
        "CNV_OIL",
        "CNV_WAT",
        "CNV_GAS",
        "CNV_PLY",
        "TRANEXX",
        "TRANEXY",
        "TRANEXZ",
        "EXCAVNUM",
        "CNV_SAL",
        "CNV_SOL",
        "CNV_GGI",
        "CNV_DPRE",
        "CNV_DWAT",
        "CNV_DGAS",
        "CNV_DPLY",
        "CNV_DSAL",
        "CNV_DSOL",
        "CNV_DGGI",
        "CONV_VBR",
        "CONV_PRU",
        "CONV_NEW",
        "FLOOILI+",
        "FLOOILJ+",
        "FLOOILK+",
        "FLOGASI+",
        "FLOGASJ+",
        "FLOGASK+",
        "FLOWATI+",
        "FLOWATJ+",
        "FLOWATK+",
        "FLROILI+",
        "FLROILJ+",
        "FLROILK+",
        "FLRGASI+",
        "FLRGASJ+",
        "FLRGASK+",
        "FLRWATI+",
        "FLRWATJ+",
        "FLRWATK+",
        "VOILI+",
        "VOILJ+",
        "VOILK+",
        "VGASI+",
        "VGASJ+",
        "VGASK+",
        "VWATI+",
        "VWATJ+",
        "VWATK+",
        "FLOOILN+",
        "FLOGASN+",
        "FLOWATN+",
        "FLOOILL+",
        "FLOGASL+",
        "FLOWATL+",
        "FLOOILA+",
        "FLOGASA+",
        "FLOWATA+",
        "FLROILN+",
        "FLRGASN+",
        "FLRWATN+",
        "FLROILL+",
        "FLRGASL+",
        "FLRWATL+",
        "FLROILA+",
        "FLRGASA+",
        "FLRWATA+",
    ]

    ecl_vectors = [
        "COPR",
        "COPT",
        "CWFR",
        "CWIR",
        "CWPR",
        "CWPT",
        "FGIR",
        "FGIT",
        "FGLIR",
        "FGOR",
        "FGORH",
        "FGPR",
        "FGPT",
        "FLPR",
        "FLPT",
        "FMCTP",
        "FMWWO",
        "FMWWT",
        "FODEN",
        "FOE",
        "FOIP",
        "FOPR",
        "FOPRF",
        "FOPRH",
        "FOPRS",
        "FOPT",
        "FOPTH",
        "FPR",
        "FVIR",
        "FVIT",
        "FVPR",
        "FVPT",
        "FWCT",
        "FWCTH",
        "FWIP",
        "FWIR",
        "FWIT",
        "FWPR",
        "FWPT",
        "GGOR",
        "GGPR",
        "GGPT",
        "GOPR",
        "GOPT",
        "GVIR",
        "GVIT",
        "GVPR",
        "GVPT",
        "GWCT",
        "GWIR",
        "GWPR",
        "MSUMLINS",
        "RGPV",
        "RHPV",
        "ROE",
        "ROEW",
        "ROPV",
        "ROSAT",
        "RPR",
        "RRPV",
        "RWPV",
        "TCPU",
        "TIME",
        "WBHP",
        "WBHPH",
        "WBP",
        "WBP4",
        "WBP9",
        "WGIR",
        "WGIT",
        "WGLIR",
        "WGOR",
        "WGORH",
        "WGPR",
        "WGPRH",
        "WGPTH",
        "WLPR",
        "WLPRH",
        "WLPT",
        "WLPTH",
        "WMCON",
        "WMCTL",
        "WOPR",
        "WOPRH",
        "WOPT",
        "WOPTH",
        "WPI",
        "WTHP",
        "WTICIW1",
        "WTICIW2",
        "WTIRIW1",
        "WTIRIW2",
        "WTPCIW1",
        "WTPCIW2",
        "WTPRIW1",
        "WTPRIW2",
        "WWCT",
        "WWCTH",
        "WWIR",
        "WWIRH",
        "WWIT",
        "WWITH",
        "WWPR",
        "WWPRH",
        "WWPT",
        "WWPTH",
        "YEARS",
    ]

    static_props = [
        "DEPTH",
        "DX",
        "DR",
        "DY",
        "DTHETA",
        "DZ",
        "PORO",
        "PERMX",
        "PERMR",
        "PERMI",
        "PERMY",
        "PERMTHT",
        "PERMJ",
        "PERMZ",
        "PERMK",
        "MULTX",
        "MULTR",
        "MULTI",
        "MULTY",
        "MULTTHT",
        "MULTJ",
        "MULTZ",
        "MULTK",
        "TRANX",
        "TRANR",
        "TRANI",
        "TRANY",
        "TRANTHT",
        "TRANJ",
        "TRANZ",
        "TRANK",
        "DIFFMX",
        "DIFFMR",
        "DIFFMI",
        "DIFFMY",
        "DIFFMTHT",
        "DIFFMJ",
        "DIFFMZ",
        "DIFFMK",
        "DIFFX",
        "DIFFR",
        "DIFFI",
        "DIFFY",
        "DIFFTHT",
        "DIFFJ",
        "DIFFZ",
        "DIFFK",
        "DIFFTX",
        "DIFFTR",
        "DIFFTI",
        "DIFFTY",
        "DIFFTTHT",
        "DIFFTJ",
        "DIFFTZ",
        "DIFFTK",
        "HEATTX",
        "HEATTR",
        "HEATTY",
        "HEATTTHT",
        "MLANGI",
        "GASSATC",
        "MLNGSLVI",
        "MLANG",
        "GASSATC",
        "MLANGSLV",
        "AQUIFERN",
        "DOMAINS",
        "ENDNUM",
        "EQLNUM",
        "FIPNUM",
        "FLUXNUM",
        "KRO",
        "KRORW",
        "KRW",
        "KRWR",
        "MINPVV",
        "MULTNUM",
        "MULTPV",
        "MULTX",
        "MULTX-",
        "MULTY",
        "MULTY-",
        "MULTZ",
        "MULTZ-",
        "NTG",
        "OPERNUM",
        "PCW",
        "PORV",
        "PVTNUM",
        "SATNUM",
        "SOWCR",
        "SWATINIT",
        "SWCR",
        "SWL",
        "SWLPC",
        "SWU",
        "TOPS",
        "TRANNNC",
    ]
    return (
        type_dict,
        ecl_extensions,
        dynamic_props,
        ecl_vectors,
        static_props,
        SUPPORTED_DATA_TYPES,
    )


def Plot3DNorne(
    nx, ny, nz, Truee, N_injw, N_pr, N_injg, cgrid, varii, injectors, producers, gass
):
    Pressz = np.reshape(Truee, (nx, ny, nz), "F")
    avg_2d = np.mean(Pressz, axis=2)
    avg_2d[np.isclose(avg_2d, 0)] = np.nan  # Convert values close to 0 to NaNs
    maxii = max(Pressz.ravel())
    minii = min(Pressz.ravel())
    Pressz = Pressz / maxii
    masked_Pressz = np.ma.masked_invalid(Pressz)
    colors = plt.cm.jet(masked_Pressz)
    colors[np.isnan(Pressz), :3] = 1  # set color to white for NaN values
    # alpha = np.where(np.isnan(Pressz), 0.0, 0.8) # set alpha to 0 for NaN values
    norm = mpl.colors.Normalize(vmin=minii, vmax=maxii)
    arr_3d = Pressz
    fig = plt.figure(figsize=(20, 20), dpi=200)
    ax = fig.add_subplot(221, projection="3d")
    # Shift the coordinates to center the points at the voxel locations
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
    ax.xaxis._axinfo["tick"]["inward_factor"] = 0
    ax.xaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax.yaxis._axinfo["tick"]["inward_factor"] = 0
    ax.yaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax.zaxis._axinfo["tick"]["inward_factor"] = 0
    ax.zaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax.view_init(elev=30, azim=45)
    n_inj = N_injw  # Number of injectors
    n_prod = N_pr  # Number of producers
    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        line_dir = (0, 0, (nz * 2) + 2)
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "blue", linewidth=1)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="blue",
            weight="bold",
            fontsize=5,
        )
    for mm in range(N_injg):
        usethis = gass[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        line_dir = (0, 0, (nz * 2) + 2)
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
    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        line_dir = (0, 0, (nz * 2) + 2)
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "g", linewidth=1)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="g",
            weight="bold",
            fontsize=5,
        )
    blue_line = mlines.Line2D([], [], color="blue", linewidth=2, label="water injector")
    red_line = mlines.Line2D([], [], color="red", linewidth=2, label="Gas injector")
    green_line = mlines.Line2D(
        [], [], color="green", linewidth=2, label="oil/water/gas Producer"
    )
    ax.legend(handles=[blue_line, red_line, green_line], loc="lower left", fontsize=9)
    cbar = plt.colorbar(m, orientation="horizontal", shrink=0.5)
    if varii == "perm":
        cbar.set_label("Log K(mD)", fontsize=9)
        ax.set_title(
            "Permeability Field with well locations", fontsize=16, weight="bold"
        )
    elif varii == "water":
        cbar.set_label("water saturation", fontsize=9)
        ax.set_title(
            "water saturation Field with well locations", fontsize=16, weight="bold"
        )
    elif varii == "oil":
        cbar.set_label("Oil saturation", fontsize=9)
        ax.set_title(
            "Oil saturation Field with well locations", fontsize=16, weight="bold"
        )
    elif varii == "porosity":
        cbar.set_label("porosity", fontsize=9)
        ax.set_title("Porosity Field with well locations", fontsize=16, weight="bold")
    cbar.mappable.set_clim(minii, maxii)
    kxy2b = cgrid
    kxy2b[:, 6] = Truee.ravel()
    m = nx
    n = ny
    nn = nz
    Xcor = np.full((m, n, nn), np.nan)
    Ycor = np.full((m, n, nn), np.nan)
    Zcor = np.full((m, n, nn), np.nan)
    poroj = np.full((m, n, nn), np.nan)
    for j in range(kxy2b.shape[0]):
        index = int(kxy2b[j, 0] - 1)
        indey = int(kxy2b[j, 1] - 1)
        indez = int(kxy2b[j, 2] - 1)
        Xcor[index, indey, indez] = kxy2b[j, 3]
        Ycor[index, indey, indez] = kxy2b[j, 4]
        Zcor[index, indey, indez] = kxy2b[j, 5]
        poroj[index, indey, indez] = kxy2b[j, 6]
    ax1 = fig.add_subplot(222, projection="3d")
    ax1.set_xlim(0, nx)
    ax1.set_ylim(0, ny)
    ax1.set_zlim(0, nz)
    ax1.set_facecolor("white")
    maxii = np.nanmax(poroj)
    minii = np.nanmin(poroj)
    colors = plt.cm.jet(np.linspace(0, 1, 256))
    colors[0, :] = (1, 1, 1, 1)  # set color for NaN values to white
    for j in range(nz):
        Xcor2D = Xcor[:, :, j]
        Ycor2D = Ycor[:, :, j]
        Zcor2D = Zcor[:, :, j]
        poroj2D = poroj[:, :, j]
        Pressz = poroj2D / maxii
        Pressz[Pressz == 0] = np.nan
        masked_Pressz = np.ma.masked_invalid(Pressz)
        colors = plt.cm.jet(masked_Pressz)
        colors[np.isnan(Pressz), :3] = 1  # set color to white for NaN values
        norm = mpl.colors.Normalize(vmin=minii, vmax=maxii)
        h1 = ax1.plot_surface(
            Xcor2D,
            Ycor2D,
            Zcor2D,
            cmap="jet",
            facecolors=colors,
            edgecolor="none",
            shade=True,
        )
        ax1.patch.set_facecolor("white")  # set the facecolor of the figure to white
        ax1.set_facecolor("white")
    cbar = fig.colorbar(h1, orientation="horizontal", shrink=0.5)
    if varii == "perm":
        cbar.ax.set_ylabel("Log K(mD)", fontsize=9)
        ax1.set_title("Permeability Field - side view", weight="bold", fontsize=16)
    if varii == "porosity":
        cbar.ax.set_ylabel("porosity", fontsize=9)
        ax1.set_title("porosity Field - side view", weight="bold", fontsize=16)
    if varii == "oil":
        cbar.ax.set_ylabel("Oil saturation", fontsize=9)
        ax1.set_title("Oil saturation Field - side view", weight="bold", fontsize=16)
    if varii == "water":
        cbar.ax.set_ylabel("Water saturation", fontsize=9)
        ax1.set_title("Water saturation - side view", weight="bold", fontsize=16)
    cbar.mappable.set_clim(minii, maxii)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_xlim(Xcor.min(), Xcor.max())
    ax1.set_ylim(Ycor.min(), Ycor.max())
    ax1.set_zlim(Zcor.min(), Zcor.max())
    ax1.view_init(30, 30)
    n_inj = N_injw  # Number of injectors
    n_prod = N_pr  # Number of producers
    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        line_dir = (0, 0, (nz * 2) + 2)
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax1.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "black", linewidth=2)
        ax1.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="black",
            weight="bold",
            fontsize=16,
        )
    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        line_dir = (0, 0, (nz * 2) + 2)
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax1.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "r", linewidth=2)
        ax1.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="r",
            weight="bold",
            fontsize=16,
        )
    ax1.grid(False)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.xaxis._axinfo["tick"]["inward_factor"] = 0
    ax1.xaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax1.yaxis._axinfo["tick"]["inward_factor"] = 0
    ax1.yaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax1.zaxis._axinfo["tick"]["inward_factor"] = 0
    ax1 = fig.add_subplot(223, projection="3d")
    ax1.set_xlim(0, nx)
    ax1.set_ylim(0, ny)
    ax1.set_zlim(0, nz)
    ax1.set_facecolor("white")
    maxii = np.nanmax(poroj)
    minii = np.nanmin(poroj)
    colors = plt.cm.jet(np.linspace(0, 1, 256))
    colors[0, :] = (1, 1, 1, 1)  # set color for NaN values to white
    for j in range(nz):
        Xcor2D = Xcor[:, :, j]
        Ycor2D = Ycor[:, :, j]
        Zcor2D = Zcor[:, :, j]
        poroj2D = poroj[:, :, j]
        Pressz = poroj2D / maxii
        Pressz[Pressz == 0] = np.nan
        masked_Pressz = np.ma.masked_invalid(Pressz)
        colors = plt.cm.jet(masked_Pressz)
        colors[np.isnan(Pressz), :3] = 1  # set color to white for NaN values
        norm = mpl.colors.Normalize(vmin=minii, vmax=maxii)
        h1 = ax1.plot_surface(
            Xcor2D,
            Ycor2D,
            Zcor2D,
            cmap="jet",
            facecolors=colors,
            edgecolor="none",
            shade=True,
        )
        ax1.patch.set_facecolor("white")  # set the facecolor of the figure to white
        ax1.set_facecolor("white")
    cbar = fig.colorbar(h1, orientation="horizontal", shrink=0.5)
    if varii == "perm":
        cbar.ax.set_ylabel("Log K(mD)", fontsize=9)
        ax1.set_title("Permeability Field - Top view", weight="bold", fontsize=16)
    if varii == "porosity":
        cbar.ax.set_ylabel("porosity", fontsize=9)
        ax1.set_title("porosity Field - Top view", weight="bold", fontsize=16)
    if varii == "oil":
        cbar.ax.set_ylabel("Oil saturation", fontsize=9)
        ax1.set_title("Oil saturation Field - Top view", weight="bold", fontsize=16)
    if varii == "water":
        cbar.ax.set_ylabel("Water saturation", fontsize=9)
        ax1.set_title("Water saturation - Top view", weight="bold", fontsize=16)
    cbar.mappable.set_clim(minii, maxii)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_xlim(Xcor.min(), Xcor.max())
    ax1.set_ylim(Ycor.min(), Ycor.max())
    ax1.set_zlim(Zcor.min(), Zcor.max())
    ax1.view_init(90, -90)
    n_inj = N_injw  # Number of injectors
    n_prod = N_pr  # Number of producers
    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        line_dir = (0, 0, (nz * 2) + 2)
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax1.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "black", linewidth=2)
        ax1.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="black",
            weight="bold",
            fontsize=16,
        )
    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        line_dir = (0, 0, (nz * 2) + 2)
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax1.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "r", linewidth=2)
        ax1.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="r",
            weight="bold",
            fontsize=16,
        )
    ax1.grid(False)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.xaxis._axinfo["tick"]["inward_factor"] = 0
    ax1.xaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax1.yaxis._axinfo["tick"]["inward_factor"] = 0
    ax1.yaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax1.zaxis._axinfo["tick"]["inward_factor"] = 0
    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
    plt.subplot(224)
    plt.pcolormesh(XX.T, YY.T, avg_2d, cmap="jet")
    cbar = plt.colorbar()
    if varii == "perm":
        cbar.ax.set_ylabel("Log K(mD)", fontsize=9)
        plt.title(r"Average Permeability Field ", fontsize=16, weight="bold")
    if varii == "porosity":
        cbar.ax.set_ylabel("porosity", fontsize=9)
        plt.title("Average porosity Field", weight="bold", fontsize=16)
    if varii == "oil":
        cbar.ax.set_ylabel("Oil saturation", fontsize=9)
        plt.title("Average Oil saturation Field - Top view", weight="bold", fontsize=16)
    if varii == "water":
        cbar.ax.set_ylabel("Water saturation", fontsize=9)
        plt.title("Average Water saturation - Top view", weight="bold", fontsize=16)
    plt.ylabel("Y", fontsize=16)
    plt.xlabel("X", fontsize=16)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    Add_marker2(plt, XX, YY, injectors, producers, gass)
    if varii == "perm":
        fig.suptitle(
            r"3D Permeability NORNE FIELD [$N_x = 46$, $N_y = 112$, $N_z = 22$]",
            weight="bold",
            fontsize=15,
        )
    elif varii == "water":
        fig.suptitle(
            r"3D Water Saturation NORNE FIELD [$N_x = 46$, $N_y = 112$, $N_z = 22$]",
            weight="bold",
            fontsize=15,
        )
    elif varii == "oil":
        fig.suptitle(
            r"3D Oil Saturation NORNE FIELD [$N_x = 46$, $N_y = 112$, $N_z = 22$]",
            weight="bold",
            fontsize=15,
        )
    elif varii == "porosity":
        fig.suptitle(
            r"3D Porosity NORNE FIELD [$N_x = 46$, $N_y = 112$, $N_z = 22$]",
            weight="bold",
            fontsize=15,
        )
    plt.savefig("All1.png")
    plt.clf()


def Plot_all_layesr(nx, ny, nz, see, injectors, producers, gass, varii):
    see[see == 0] = np.nan  # Convert zeros to NaNs
    plt.figure(figsize=(20, 20), dpi=300)
    Pressz = np.reshape(see, (nx, ny, nz), "F")
    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
    for i in range(nz):
        plt.subplot(5, 5, i + 1)
        plt.pcolormesh(XX.T, YY.T, Pressz[:, :, i], cmap="jet")
        cbar = plt.colorbar()
        if varii == "perm":
            cbar.ax.set_ylabel("Log K(mD)", fontsize=9)
            plt.title(
                "Permeability Field Layer_" + str(i + 1), fontsize=11, weight="bold"
            )
        if varii == "porosity":
            cbar.ax.set_ylabel("porosity", fontsize=9)
            plt.title("porosity Field Layer_" + str(i + 1), weight="bold", fontsize=11)
        if varii == "oil":
            cbar.ax.set_ylabel("Oil saturation", fontsize=9)
            plt.title(
                "Oil saturation Field Layer_" + str(i + 1), weight="bold", fontsize=11
            )
        if varii == "water":
            cbar.ax.set_ylabel("Water saturation", fontsize=9)
            plt.title(
                "Water saturation Field Layer_" + str(i + 1), weight="bold", fontsize=11
            )
        plt.ylabel("Y", fontsize=11)
        plt.xlabel("X", fontsize=11)
        plt.axis([0, (nx - 1), 0, (ny - 1)])
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        Add_marker3(plt, XX, YY, injectors, producers, gass)
    if varii == "perm":
        plt.suptitle(
            r"Permeability NORNE FIELD [$N_x = 46$, $N_y = 112$, $N_z = 22$]",
            weight="bold",
            fontsize=15,
        )
    elif varii == "water":
        plt.suptitle(
            r"Water Saturation NORNE FIELD [$N_x = 46$, $N_y = 112$, $N_z = 22$]",
            weight="bold",
            fontsize=15,
        )
    elif varii == "oil":
        plt.suptitle(
            r"Oil Saturation NORNE FIELD [$N_x = 46$, $N_y = 112$, $N_z = 22$]",
            weight="bold",
            fontsize=15,
        )
    elif varii == "porosity":
        plt.suptitle(
            r"Porosity NORNE FIELD [$N_x = 46$, $N_y = 112$, $N_z = 22$]",
            weight="bold",
            fontsize=15,
        )
    plt.savefig("All.png")
    plt.clf()


def Add_marker2(plt, XX, YY, injectors, producers, gass):
    n_inj = len(injectors)  # Number of injectors
    n_prod = len(producers)  # Number of producers
    n_injg = len(gass)  # Number of gas injectors
    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=200,
            marker="v",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )
    for mm in range(n_injg):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=200,
            marker="v",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )
    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=200,
            marker="^",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )


def Add_marker3(plt, XX, YY, injectors, producers, gass):
    n_inj = len(injectors)  # Number of injectors
    n_prod = len(producers)  # Number of producers
    n_injg = len(gass)  # Number of gas injectors
    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=100,
            marker="v",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=9,
        )
    for mm in range(n_injg):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=100,
            marker="v",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=9,
        )
    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=100,
            marker="^",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=9,
        )
