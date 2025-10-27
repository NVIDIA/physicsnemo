"""
SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
                    MISCELLANEOUS UTILITIES FOR BATCH COMPARISON
=====================================================================

This module provides utility functions for batch comparison operations in reservoir
simulation. It includes functions for data processing, visualization, file operations,
and system information gathering.

Key Features:
- System information detection (GPU/CPU availability and specifications)
- Data processing and manipulation utilities
- File I/O operations with proper error handling
- Visualization and plotting utilities
- Parallel processing support
- Memory management and optimization

Usage:
    from compare.batch.utils.misc_utils import (
        is_available,
        get_system_info,
        process_data,
        create_visualization
    )

Inputs:
    - System configuration data
    - Data arrays for processing
    - File paths for I/O operations

Outputs:
    - Processed data arrays
    - Visualization plots and images
    - System information reports
    - Logged status messages

@Author : Clement Etienam
"""

# 🛠 Standard Library
import os
import time
import math
import shutil
import logging
import warnings
from datetime import timedelta
from multiprocessing import cpu_count
import re

# 🔧 Third-party Libraries
import numpy as np
import numpy.matlib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image
from cpuinfo import get_cpu_info

# 📦 Local Modules
from hydra.utils import to_absolute_path
from compare.batch.misc_plotting_utils import (
    simulation_data_types,
    setup_logging,
    Get_Time,
)

from compare.batch.misc_operations import (
    ProgressBar,
    ShowBar,
    Plot_RSM_percentile,
)

from compare.batch.misc_model import (
    process_step,
)

from compare.batch.misc_forward import (
    write_RSM,
)

from compare.batch.misc_forward_enact import (
    Forward_model_ensemble,
)


logger = setup_logging()
warnings.filterwarnings("ignore")


# 🖥️ Detect GPU
def is_available() -> bool:
    """Check if NVIDIA GPU is available using native Python methods."""
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def initialize_environment() -> tuple[bool, int, logging.Logger]:
    """Initialize the environment and return GPU availability, operation mode, and logger."""
    logger = setup_logging()

    # Check GPU availability
    gpu_available = is_available()
    if gpu_available:
        logger.info("GPU Available with CUDA")
        operation_mode = 0
    else:
        logger.info("No GPU Available")
        operation_mode = 1

    # Log CPU information
    cpu_info = get_cpu_info()
    logger.info("CPU Info:")
    for key, value in cpu_info.items():
        logger.info(f"\t{key}: {value}")

    cores = cpu_count()
    logger.info(
        f"This computer has {cores} cores, which will all be utilised in parallel"
    )

    return gpu_available, operation_mode, logger


(
    type_dict,
    ecl_extensions,
    dynamic_props,
    ecl_vectors,
    static_props,
    SUPPORTED_DATA_TYPES,
) = simulation_data_types()


# Sorting function for row-major order
def sort_key(path):
    # Extract row and column indices from the filename using regex
    match = re.search(r"_(\d+)_(\d+)\.png", path)  # Match pattern like "_row_col.png"
    if match:
        row_index, col_index = int(match.group(1)), int(match.group(2))
        # Sort by row first, then by column (row-major order)
        return (row_index, col_index)
    return float("inf"), float("inf")  # Handle unexpected filenames


def compare_and_analyze_results(
    # Timing data
    physicsnemo_time,
    flow_time,
    # Model parameters
    nx,
    ny,
    nz,
    steppi,
    steppi_indices,
    Ne,
    # Results data
    pressure,
    pressure_true,
    Swater,
    Swater_true,
    Soil,
    Soil_true,
    Sgas,
    Sgas_true,
    ouut_peacemann,
    out_fcn_true,
    # Configuration
    cfg,
    device,
    num_cores,
    oldfolder,
    folderr,
    # Well configuration
    N_injw,
    N_pr,
    N_injg,
    injectors,
    producers,
    gass,
    well_names,
    # Forward model parameters
    inn,
    min_inn_fcn,
    max_inn_fcn,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    models,
    min_out_fcn,
    max_out_fcn,
    Time,
    effective_abi,
    degg,
    experts,
    min_out_fcn2,
    max_out_fcn2,
    min_inn_fcn2,
    max_inn_fcn2,
    compdat_data,
    output_variables,
    well_measurements,
    # Additional parameters
    effectiveuse,
    columns,
    lenwels,
):
    if physicsnemo_time < flow_time:
        slower_time = physicsnemo_time
        faster_time = flow_time
        slower = "Nvidia physicsnemo Surrogate"
        faster = "flow Reservoir simulator"
        speedup = math.ceil(flow_time / physicsnemo_time)
        os.chdir(folderr)
        tasks = ["Flow", "physicsnemo"]
        times = [faster_time, slower_time]
        colors = ["green", "red"]
        plt.figure(figsize=(10, 6))
        bars = plt.bar(tasks, times, color=colors)
        plt.ylabel("Time (seconds)", fontweight="bold")
        plt.title("Execution Time Comparison for PhyNeMo vs. Flow", fontweight="bold")
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 20,
                round(yval, 2),
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        plt.text(
            0.5,
            550,
            f"Speedup: {speedup}x",
            ha="center",
            fontsize=12,
            fontweight="bold",
            color="blue",
        )
        namez = "Compare_time.png"
        plt.savefig(namez)
        plt.clf()
        plt.close()
        os.chdir(oldfolder)
    else:
        slower_time = flow_time
        faster_time = physicsnemo_time
        slower = "flow Reservoir simulator"
        faster = "Nvidia physicsnemo Surrogate"
        speedup = math.ceil(physicsnemo_time / flow_time)
        os.chdir(folderr)
        tasks = ["Flow", "physicsnemo"]
        times = [slower_time, faster_time]
        colors = ["green", "red"]
        plt.figure(figsize=(10, 6))
        bars = plt.bar(tasks, times, color=colors)
        plt.ylabel("Time (seconds)", fontweight="bold")
        plt.title("Execution Time Comparison for PhyNeMo vs. Flow", fontweight="bold")
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 20,
                round(yval, 2),
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        plt.text(
            0.5,
            550,
            f"Speedup: {speedup}x",
            ha="center",
            fontsize=12,
            fontweight="bold",
            color="blue",
        )
        namez = "Compare_time.png"
        plt.savefig(namez)
        plt.clf()
        plt.close()
        os.chdir(oldfolder)
    message = (
        f"{slower} execution took: {slower_time} seconds\n"
        f"{faster} execution took: {faster_time} seconds\n"
        f"Speedup =  {speedup}X  "
    )
    logger.info(message)
    os.chdir(to_absolute_path("../simulator_data"))
    Time = Get_Time(nx, ny, nz, steppi, steppi_indices, Ne)
    Time_unie = np.zeros((steppi))
    for i in range(steppi):
        Time_unie[i] = Time[0, i, 0, 0, 0]
    os.chdir(to_absolute_path(oldfolder))
    dt = Time_unie
    # Runs = steppi
    # ty = np.arange(1, Runs + 1)
    Time_vector = Time_unie
    Accuracy_presure = np.zeros((steppi, 2))
    Accuracy_oil = np.zeros((steppi, 2))
    Accuracy_water = np.zeros((steppi, 2))
    Accuracy_gas = np.zeros((steppi, 2))
    results = Parallel(n_jobs=num_cores, backend="loky", verbose=10)(
        delayed(process_step)(
            kk,
            steppi,
            dt,
            pressure,
            effectiveuse,
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
            folderr,
            oldfolder,
            Accuracy_presure,
            Accuracy_oil,
            Accuracy_water,
            Accuracy_gas,
        )
        for kk in range(steppi)
    )
    os.chdir(to_absolute_path(oldfolder))
    progressBar = "\rPlotting Progress: " + ProgressBar(
        steppi - 1, steppi - 1, steppi - 1
    )
    ShowBar(progressBar)
    time.sleep(1)
    for kk, (R2p, L2p, R2w, L2w, R2o, L2o, R2g, L2g) in enumerate(results):
        Accuracy_presure[kk, 0] = R2p
        Accuracy_presure[kk, 1] = L2p
        Accuracy_water[kk, 0] = R2w
        Accuracy_water[kk, 1] = L2w
        Accuracy_oil[kk, 0] = R2o
        Accuracy_oil[kk, 1] = L2o
        Accuracy_gas[kk, 0] = R2g
        Accuracy_gas[kk, 1] = L2g
    os.chdir(to_absolute_path(folderr))
    fig4 = plt.figure(figsize=(20, 20), dpi=100)
    font = FontProperties()
    font.set_family("Helvetica")
    font.set_weight("bold")
    fig4.text(
        0.5,
        0.98,
        "R2(%) Accuracy - PhyNeMo/Numerical(GPU)",
        ha="center",
        va="center",
        fontproperties=font,
        fontsize=11,
    )
    fig4.text(
        0.5,
        0.49,
        "L2(%) Accuracy - PhyNeMo/Numerical(GPU)",
        ha="center",
        va="center",
        fontproperties=font,
        fontsize=11,
    )
    plt.subplot(2, 4, 1)
    plt.plot(
        Time_vector,
        Accuracy_presure[:, 0],
        label="R2",
        marker="*",
        markerfacecolor="red",
        markeredgecolor="red",
        linewidth=0.5,
    )
    plt.title("Pressure", fontproperties=font)
    plt.xlabel("Time (days)", fontproperties=font)
    plt.ylabel("R2(%)", fontproperties=font)
    plt.subplot(2, 4, 2)
    plt.plot(
        Time_vector,
        Accuracy_water[:, 0],
        label="R2",
        marker="*",
        markerfacecolor="red",
        markeredgecolor="red",
        linewidth=0.5,
    )
    plt.title("water saturation", fontproperties=font)
    plt.xlabel("Time (days)", fontproperties=font)
    plt.ylabel("R2(%)", fontproperties=font)
    plt.subplot(2, 4, 3)
    plt.plot(
        Time_vector,
        Accuracy_oil[:, 0],
        label="R2",
        marker="*",
        markerfacecolor="red",
        markeredgecolor="red",
        linewidth=0.5,
    )
    plt.title("oil saturation", fontproperties=font)
    plt.xlabel("Time (days)", fontproperties=font)
    plt.ylabel("R2(%)", fontproperties=font)
    plt.subplot(2, 4, 4)
    plt.plot(
        Time_vector,
        Accuracy_gas[:, 0],
        label="R2",
        marker="*",
        markerfacecolor="red",
        markeredgecolor="red",
        linewidth=0.5,
    )
    plt.title("gas saturation", fontproperties=font)
    plt.xlabel("Time (days)", fontproperties=font)
    plt.ylabel("R2(%)", fontproperties=font)
    plt.subplot(2, 4, 5)
    plt.plot(
        Time_vector,
        Accuracy_presure[:, 1],
        label="L2",
        marker="*",
        markerfacecolor="red",
        markeredgecolor="red",
        linewidth=0.5,
    )
    plt.title("Pressure", fontproperties=font)
    plt.xlabel("Time (days)", fontproperties=font)
    plt.ylabel("L2(%)", fontproperties=font)
    plt.subplot(2, 4, 6)
    plt.plot(
        Time_vector,
        Accuracy_water[:, 1],
        label="L2",
        marker="*",
        markerfacecolor="red",
        markeredgecolor="red",
        linewidth=0.5,
    )
    plt.title("water saturation", fontproperties=font)
    plt.xlabel("Time (days)", fontproperties=font)
    plt.ylabel("L2(%)", fontproperties=font)
    plt.subplot(2, 4, 7)
    plt.plot(
        Time_vector,
        Accuracy_oil[:, 1],
        label="L2",
        marker="*",
        markerfacecolor="red",
        markeredgecolor="red",
        linewidth=0.5,
    )
    plt.title("oil saturation", fontproperties=font)
    plt.xlabel("Time (days)", fontproperties=font)
    plt.ylabel("L2(%)", fontproperties=font)
    plt.subplot(2, 4, 8)
    plt.plot(
        Time_vector,
        Accuracy_gas[:, 1],
        label="L2",
        marker="*",
        markerfacecolor="red",
        markeredgecolor="red",
        linewidth=0.5,
    )
    plt.title("gas saturation", fontproperties=font)
    plt.xlabel("Time (days)", fontproperties=font)
    plt.ylabel("L2(%)", fontproperties=font)
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    namez = "R2L2.png"
    plt.savefig(namez)
    plt.clf()
    plt.close()
    logger.info("Now - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*Dynamic*"), key=sort_key)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)
    frames[0].save(
        "Evolution.gif",
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=500,
        loop=0,
    )
    print("Saving prediction in CSV file")
    write_RSM(ouut_peacemann[0, :, :], Time_vector, "PhyNeMo", well_names, N_pr)
    write_RSM(out_fcn_true[0, :, :], Time_vector, "Flow", well_names, N_pr)
    CCRhard = ouut_peacemann[0, :, :]
    Truedata = out_fcn_true[0, :, :]
    print("Plotting well responses and accuracies")
    Plot_RSM_percentile(
        ouut_peacemann[0, :, :], out_fcn_true[0, :, :], Time_vector, well_names, N_pr
    )
    os.chdir(to_absolute_path(oldfolder))
    Trainmoe = "FNO"
    print("----------------------------------------------------------------------")
    print("Using FNO for peacemann model           ")

    pred_type = 1
    if cfg.custom.fno_type == "PINO":
        folderr = "../RESULTS/FORWARD_RESULTS_BATCH/RESULTS/COMPARE_RESULTS/PINO/PEACEMANN_FNO"
        if not os.path.exists(
            to_absolute_path(
                "../RESULTS/FORWARD_RESULTS_BATCH/RESULTS/COMPARE_RESULTS/PINO/PEACEMANN_FNO"
            )
        ):
            os.makedirs(
                to_absolute_path(
                    "../RESULTS/FORWARD_RESULTS_BATCH/RESULTS/COMPARE_RESULTS/PINO/PEACEMANN_FNO"
                ),
                exist_ok=True,
            )
        else:
            shutil.rmtree(
                to_absolute_path(
                    "../RESULTS/FORWARD_RESULTS_BATCH/RESULTS/COMPARE_RESULTS/PINO/PEACEMANN_FNO"
                )
            )
            os.makedirs(
                to_absolute_path(
                    "../RESULTS/FORWARD_RESULTS_BATCH/RESULTS/COMPARE_RESULTS/PINO/PEACEMANN_FNO"
                ),
                exist_ok=True,
            )
        source_directory = to_absolute_path(
            "../RESULTS/FORWARD_RESULTS_BATCH/RESULTS/COMPARE_RESULTS/PINO/PEACEMANN_CCR"
        )
        destination_directory = to_absolute_path(
            "../RESULTS/FORWARD_RESULTS_BATCH/RESULTS/COMPARE_RESULTS/PINO/PEACEMANN_FNO"
        )
    else:
        folderr = (
            "../RESULTS/FORWARD_RESULTS_BATCH/RESULTS/COMPARE_RESULTS/FNO/PEACEMANN_FNO"
        )
        if not os.path.exists(
            to_absolute_path(
                "../RESULTS/FORWARD_RESULTS_BATCH/RESULTS/COMPARE_RESULTS/FNO/PEACEMANN_FNO"
            )
        ):
            os.makedirs(
                to_absolute_path(
                    "../RESULTS/FORWARD_RESULTS_BATCH/RESULTS/COMPARE_RESULTS/FNO/PEACEMANN_FNO"
                ),
                exist_ok=True,
            )
        else:
            shutil.rmtree(
                to_absolute_path(
                    "../RESULTS/FORWARD_RESULTS_BATCH/RESULTS/COMPARE_RESULTS/FNO/PEACEMANN_FNO"
                )
            )
            os.makedirs(
                to_absolute_path(
                    "../RESULTS/FORWARD_RESULTS_BATCH/RESULTS/COMPARE_RESULTS/FNO/PEACEMANN_FNO"
                ),
                exist_ok=True,
            )
        source_directory = to_absolute_path(
            "../RESULTS/FORWARD_RESULTS_BATCH/RESULTS/COMPARE_RESULTS/FNO/PEACEMANN_CCR"
        )
        destination_directory = to_absolute_path(
            "../RESULTS/FORWARD_RESULTS_BATCH/RESULTS/COMPARE_RESULTS/FNO/PEACEMANN_FNO"
        )
    filename = "Evolution.gif"
    source_path = to_absolute_path(os.path.join(source_directory, filename))
    destination_path = to_absolute_path(os.path.join(destination_directory, filename))
    shutil.copy(source_path, destination_path)
    filename = "R2L2.png"
    source_path = to_absolute_path(os.path.join(source_directory, filename))
    destination_path = to_absolute_path(os.path.join(destination_directory, filename))
    shutil.copy(source_path, destination_path)
    filename = "Evolution.gif"
    source_path = to_absolute_path(os.path.join(source_directory, filename))
    destination_path = to_absolute_path(os.path.join(destination_directory, filename))
    shutil.copy(source_path, destination_path)
    start_time_plots2 = time.time()
    simout = Forward_model_ensemble(
        Ne,
        inn,
        steppi,
        min_inn_fcn,
        max_inn_fcn,
        target_min,
        target_max,
        minK,
        maxK,
        minT,
        maxT,
        minP,
        maxP,
        models,
        device,
        min_out_fcn,
        max_out_fcn,
        Time,
        effective_abi,
        Trainmoe,
        num_cores,
        pred_type,
        oldfolder,
        degg,
        experts,
        min_out_fcn2,
        max_out_fcn2,
        min_inn_fcn2,
        max_inn_fcn2,
        producers,
        compdat_data,
        output_variables,
        well_measurements,
        cfg,
        N_pr,
        lenwels,
        effective_abi,
        nx,
        ny,
        nz,
    )
    elapsed_time_secs2 = time.time() - start_time_plots2
    msg = (
        "Reservoir simulation with NVIDIA PhyNeMo (FNO)  took: %s secs (Wall clock time)"
        % timedelta(seconds=round(elapsed_time_secs2))
    )
    print(msg)
    if "PRESSURE" in output_variables:
        pressure = simout["PRESSURE"]
    if "SWAT" in output_variables:
        Swater = simout["SWAT"]
    if "SOIL" in output_variables:
        Soil = simout["SOIL"]
    if "SGAS" in output_variables:
        Sgas = simout["SGAS"]
    ouut_peacemann = simout["ouut_p"]
    physicsnemo_time = elapsed_time_secs2
    # flow_time = elapsed_time_secs
    if physicsnemo_time < flow_time:
        slower_time = physicsnemo_time
        faster_time = flow_time
        slower = "Nvidia physicsnemo Surrogate"
        faster = "flow Reservoir simulator"
        speedup = math.ceil(flow_time / physicsnemo_time)
        os.chdir(to_absolute_path(folderr))
        tasks = ["Flow", "physicsnemo"]
        times = [faster_time, slower_time]
        colors = ["green", "red"]
        plt.figure(figsize=(10, 6))
        bars = plt.bar(tasks, times, color=colors)
        plt.ylabel("Time (seconds)", fontweight="bold")
        plt.title("Execution Time Comparison for PhyNeMo vs. Flow", fontweight="bold")
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 20,
                round(yval, 2),
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        plt.text(
            0.5,
            550,
            f"Speedup: {speedup}x",
            ha="center",
            fontsize=12,
            fontweight="bold",
            color="blue",
        )
        namez = "Compare_time.png"
        plt.savefig(namez)
        plt.clf()
        plt.close()
        os.chdir(to_absolute_path(oldfolder))
    else:
        slower_time = flow_time
        faster_time = physicsnemo_time
        slower = "flow Reservoir simulator"
        faster = "Nvidia physicsnemo Surrogate"
        speedup = math.ceil(physicsnemo_time / flow_time)
        os.chdir(to_absolute_path(folderr))
        tasks = ["Flow", "physicsnemo"]
        times = [slower_time, faster_time]
        colors = ["green", "red"]
        plt.figure(figsize=(10, 6))
        bars = plt.bar(tasks, times, color=colors)
        plt.ylabel("Time (seconds)", fontweight="bold")
        plt.title("Execution Time Comparison for PhyNeMo vs. Flow", fontweight="bold")
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 20,
                round(yval, 2),
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        plt.text(
            0.5,
            550,
            f"Speedup: {speedup}x",
            ha="center",
            fontsize=12,
            fontweight="bold",
            color="blue",
        )
        namez = "Compare_time.png"
        plt.savefig(namez)
        plt.clf()
        plt.close()
        os.chdir(to_absolute_path(oldfolder))
    message = (
        f"{slower} execution took: {slower_time} seconds\n"
        f"{faster} execution took: {faster_time} seconds\n"
        f"Speedup =  {speedup}X  "
    )
    logger.info(message)
    os.chdir(to_absolute_path("../simulator_data"))
    Time = Get_Time(nx, ny, nz, steppi, steppi_indices, Ne)
    Time_unie = np.zeros((steppi))
    for i in range(steppi):
        Time_unie[i] = Time[0, i, 0, 0, 0]
    os.chdir((oldfolder))
    dt = Time_unie
    print("Plotting outputs")
    os.chdir(to_absolute_path(folderr))
    # Runs = steppi
    # ty = np.arange(1, Runs + 1)
    Time_vector = np.zeros((steppi))
    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        current_time = dt[kk]
        Time_vector[kk] = current_time
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Saving prediction in CSV file")
    write_RSM(ouut_peacemann[0, :, :], Time_vector, "PhyNeMo", well_names, N_pr)
    write_RSM(out_fcn_true[0, :, :], Time_vector, "Flow", well_names, N_pr)
    print("Plotting well responses and accuracies")
    Plot_RSM_percentile(
        ouut_peacemann[0, :, :], out_fcn_true[0, :, :], Time_vector, well_names, N_pr
    )
    FNOpred = ouut_peacemann[0, :, :]
    os.chdir((oldfolder))
    os.chdir(
        to_absolute_path("../RESULTS/FORWARD_RESULTS_BATCH/RESULTS/COMPARE_RESULTS")
    )
    P10 = CCRhard
    P90 = FNOpred
    True_mat = Truedata
    timezz = Time_vector
    plt.figure(figsize=(40, 40))
    for k in range(N_pr):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k], color="red", lw="2", label="Flow")
        if cfg.custom.fno_type == "FNO":
            plt.plot(timezz, P10[:, k], color="blue", lw="2", label="PINO -CCR(hard)")
            plt.plot(timezz, P90[:, k], color="orange", lw="2", label="PINO - FNO")
        else:
            plt.plot(timezz, P10[:, k], color="blue", lw="2", label="FNO -CCR(hard)")
            plt.plot(timezz, P90[:, k], color="orange", lw="2", label="FNO - FNO")

        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()
    plt.savefig("Oil.png")
    plt.clf()
    plt.close()
    plt.figure(figsize=(40, 40))
    for k in range(N_pr):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + N_pr], color="red", lw="2", label="Flow")
        plt.plot(
            timezz, P10[:, k + N_pr], color="blue", lw="2", label="PINO -CCR(hard)"
        )
        plt.plot(timezz, P90[:, k + N_pr], color="orange", lw="2", label="PINO - FNO")
        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()
    plt.savefig("Water.png")
    plt.clf()
    plt.close()
    plt.figure(figsize=(40, 40))
    for k in range(N_pr):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + 2 * N_pr], color="red", lw="2", label="Flow")
        plt.plot(
            timezz, P10[:, k + 2 * N_pr], color="blue", lw="2", label="PINO -CCR(hard)"
        )
        plt.plot(
            timezz, P90[:, k + 2 * N_pr], color="orange", lw="2", label="PINO - FNO"
        )

        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()
    plt.savefig("Gas.png")
    plt.clf()
    plt.close()
    True_data = np.reshape(Truedata, (-1, 1), "F")
    CCRhard = np.reshape(CCRhard, (-1, 1), "F")
    cc1 = ((np.sum((((CCRhard) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of PINO - CCR (hard prediction) reservoir model  =  " + str(cc1))
    FNO = np.reshape(FNOpred, (-1, 1), "F")
    cc3 = ((np.sum((((FNO) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of  PINO - FNO reservoir model  =  " + str(cc3))
    plt.figure(figsize=(10, 10))
    values = [cc1, cc3]
    model_names = ["PINO CCR-hard", "PINO - FNO"]
    colors = ["b", "orange"]
    min_rmse_index = np.argmin(values)
    min_rmse = values[min_rmse_index]
    best_model = model_names[min_rmse_index]
    print(f"The minimum RMSE value = {min_rmse}")
    print(
        f"Recommended reservoir forward model workflow = {best_model} reservoir model."
    )
    plt.bar(model_names, values, color=colors)
    plt.xlabel("Reservoir Models")
    plt.ylabel("RMSE")
    plt.title(
        "Histogram of RMSE Values for Different Reservoir Surrogate Model workflow"
    )
    plt.legend(model_names)
    plt.savefig(
        "Histogram.png"
    )  # save as png                                  # preventing the figures from showing
    plt.clf()
    plt.close()
    os.chdir((oldfolder))
