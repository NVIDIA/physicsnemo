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
                    ENSEMBLE RESULTS UTILITIES MODULE
=====================================================================

This module provides ensemble results processing utilities for inverse
problems in reservoir simulation. It includes result analysis,
visualization, and data processing utilities.

Key Features:
- Final results processing
- Data analysis and visualization
- Result comparison utilities
- Statistical analysis tools

@Author : Clement Etienam
"""

# 🛠 Standard Library
import os
import time
import pickle
import gzip
import shutil
import logging

# 🔧 Third-party Libraries
import numpy as np
import numpy.matlib
from hydra.utils import to_absolute_path
from PIL import Image
from joblib import Parallel, delayed

# 📦 Local Modules
from inverse.inversion_operation_surrogate import (
    Forward_model_ensemble,
    remove_rows,
)
from inverse.inversion_operation_ensemble import (
    honour2,
    funcGetDataMismatch,
    ensemble_pytorch,
)
from inverse.inversion_operation_gather import (
    Plot_RSM,
    Plot_RSM_percentile_model,
    write_RSM,
    Plot_petrophysical,
    Plot_RSM_single,
    Plot_mean,
    Plot_Histogram_now,
)
from inverse.inversion_operation_misc import (
    process_step,
    ProgressBar,
    ShowBar,
    sort_key,
)


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger("Inverse problem")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logging()


# avoid redefining remove_rows; import from surrogate module already


def process_final_results(
    input_variables,
    output_variables,
    # Ensemble tracking arrays
    ensemble_bestK,
    ensemble_meanK,
    ensemble_bestP,
    ensemble_meanP,
    ensemble_bestf,
    ensemble_meanf,
    # Current ensembles
    ensemble,
    ensemblep,
    ensemblef,
    # Best models
    use_k,
    use_p,
    use_f,
    # Configuration parameters
    chm,
    Ne,
    nx,
    ny,
    nz,
    N_ens,
    High_K1,
    Low_K1,
    High_P,
    Low_P,
    effec,
    # Forward model parameters
    effective,
    oldfolder,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    minQ,
    maxQ,
    minQw,
    maxQw,
    minQg,
    maxQg,
    steppi,
    device,
    steppi_indices,
    cfg,
    # Models and data
    models,
    min_inn_fcn,
    max_inn_fcn,
    min_out_fcn,
    max_out_fcn,
    Time,
    effectiveuse,
    Trainmoe,
    num_cores,
    pred_type,
    degg,
    experts,
    min_out_fcn2,
    max_out_fcn2,
    min_inn_fcn2,
    max_inn_fcn2,
    producers,
    compdat_data,
    quant_big,
    N_pr,
    lenwels,
    effective_abi,
    rows_to_remove,
    # True data
    True_mat,
    True_data,
    Time_unie1,
    well_names,
    dt,
    # Additional parameters
    dist,
    cc_ini,
    mean_cost,
    best_cost,
    ini_ensemble,
    N_injw,
    N_injg,
    injectors,
    gass,
    True_K,
    yes_best,
    ensemble_best,
    yes_mean,
    ensemble_mean,
    all_ensemble,
    ensemble_dict,
):
    """Aggregate, plot, and persist ensemble results and diagnostics.

    Consumes best/mean ensembles and configuration to generate RSM plots,
    progress bars, and final artefacts for downstream analysis.
    """
    if "PERM" in input_variables:
        ensemble_bestK = np.hstack(ensemble_bestK)
        ensemble_meanK = np.hstack(ensemble_meanK)
        ensemble_best["PERM"] = np.hstack(ensemble_bestK)
        yes_best["PERM"] = ensemble_bestK[:, chm].reshape(-1, 1)
        ensemble_mean["PERM"] = np.hstack(ensemble_meanK)
        yes_mean["PERM"] = ensemble_meanK[:, chm].reshape(-1, 1)
        all_ensemble["PERM"] = use_k
        ensemble_dict["PERM"] = ensemble
    if "PORO" in input_variables:
        ensemble_bestP = np.hstack(ensemble_bestP)
        ensemble_meanP = np.hstack(ensemble_meanP)
        ensemble_best["PORO"] = np.hstack(ensemble_bestP)
        yes_best["PORO"] = ensemble_bestP[:, chm].reshape(-1, 1)
        ensemble_mean["PORO"] = np.hstack(ensemble_meanP)
        yes_mean["PORO"] = ensemble_meanP[:, chm].reshape(-1, 1)
        all_ensemble["PORO"] = use_p
        ensemble_dict["PORO"] = ensemblep
    if "FAULT" in input_variables:
        ensemble_bestf = np.hstack(ensemble_bestf)
        ensemble_meanf = np.hstack(ensemble_meanf)
        ensemble_best["FAULT"] = np.hstack(ensemble_bestf)
        yes_best["FAULT"] = ensemble_bestf[:, chm].reshape(-1, 1)
        ensemble_mean["FAULT"] = np.hstack(ensemble_meanf)
        yes_mean["FAULT"] = ensemble_meanf[:, chm].reshape(-1, 1)
        use_f = np.clip(use_f, 0, 1)
        all_ensemble["FAULT"] = use_f
        ensemble_dict["FAULT"] = ensemblef
    ensemble = ensemble_dict
    if "PERM" in input_variables or "PORO" in input_variables:
        ensemble_dict = honour2(
            ensemble_dict, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P, effec
        )
    ensemble = ensemble_dict
    if "PERM" in input_variables or "PORO" in input_variables:
        all_ensemble = honour2(
            all_ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P, effec
        )
    if "FAULT" in input_variables:
        all_ensemble["FAULT"] = np.clip(all_ensemble["FAULT"], 0, 1)
    meann = {}
    if "PERM" in input_variables:
        meann["PERM"] = np.reshape(np.mean(ensemble["PERM"], axis=1), (-1, 1), "F")
    if "PORO" in input_variables:
        meann["PORO"] = np.reshape(np.mean(ensemble["PORO"], axis=1), (-1, 1), "F")
    if "FAULT" in input_variables:
        meann["FAULT"] = np.reshape(np.mean(ensemble["FAULT"], axis=1), (-1, 1), "F")
    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj = {}
    if "PERM" in input_variables:
        controljj["PERM"] = np.reshape(meann["PERM"], (-1, 1), "F")
    if "PERM" in input_variables:
        controljj["PORO"] = np.reshape(meann["PORO"], (-1, 1), "F")
    if "PERM" in input_variables:
        controljj["FAULT"] = np.reshape(meann["FAULT"], (-1, 1), "F")
    ensemblepy = ensemble_pytorch(
        ensemble,
        nx,
        ny,
        nz,
        ensemble["PERM"].shape[1],
        effective,
        oldfolder,
        target_min,
        target_max,
        minK,
        maxK,
        minT,
        maxT,
        minP,
        maxP,
        minQ,
        maxQ,
        minQw,
        maxQw,
        minQg,
        maxQg,
        steppi,
        device,
        steppi_indices,
        input_variables,
        cfg,
    )
    # mazw = 0  # Dont smooth the presure field
    simout = Forward_model_ensemble(
        ensemble["PERM"].shape[1],
        ensemblepy,
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
        effectiveuse,
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
        quant_big,
        cfg,
        N_pr,
        lenwels,
        effective_abi,
        rows_to_remove,
        nx,
        ny,
        nz,
    )
    simDatafinal = simout["sim"]
    predMatrix = simout["ouut_p"]
    if "PRESSURE" in output_variables:
        pressure_ensemble = simout["PRESSURE"]
    if "SWAT" in output_variables:
        water_ensemble = simout["SWAT"]
    if "SOIL" in output_variables:
        oil_ensemble = simout["SOIL"]
    if "SGAS" in output_variables:
        gas_ensemble = simout["SGAS"]
    ensemblepya = ensemble_pytorch(
        all_ensemble,
        nx,
        ny,
        nz,
        all_ensemble["PERM"].shape[1],
        effective,
        oldfolder,
        target_min,
        target_max,
        minK,
        maxK,
        minT,
        maxT,
        minP,
        maxP,
        minQ,
        maxQ,
        minQw,
        maxQw,
        minQg,
        maxQg,
        steppi,
        device,
        steppi_indices,
        input_variables,
        cfg,
    )
    # mazw = 0  # Dont smooth the presure field
    simout = Forward_model_ensemble(
        all_ensemble["PERM"].shape[1],
        ensemblepya,
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
        effectiveuse,
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
        quant_big,
        cfg,
        N_pr,
        lenwels,
        effective_abi,
        rows_to_remove,
        nx,
        ny,
        nz,
    )
    simDatafinala = simout["sim"]
    predMatrixa = simout["ouut_p"]
    if "PRESSURE" in output_variables:
        pressure_ensemblea = simout["PRESSURE"]
    if "SWAT" in output_variables:
        water_ensemblea = simout["SWAT"]
    if "SOIL" in output_variables:
        oil_ensemblea = simout["SOIL"]
    if "SGAS" in output_variables:
        gas_ensemblea = simout["SGAS"]
    os.chdir("../RESULTS/HM_RESULTS")
    if dist.rank == 0:
        Plot_RSM(predMatrixa, True_mat, "Final.png", Ne, Time_unie1, N_pr, well_names)
    os.chdir(oldfolder)
    aa, bb, cc = funcGetDataMismatch(simDatafinal, True_data)
    muv = np.argmin(cc)
    # shpw = cc[muv]
    controlbest = {}
    if "PERM" in input_variables:
        controlbest["PERM"] = np.reshape(ensemble["PERM"][:, muv], (-1, 1), "F")
    if "PORO" in input_variables:
        controlbest["PORO"] = np.reshape(ensemble["PORO"][:, muv], (-1, 1), "F")
    if "FAULT" in input_variables:
        controlbest["FAULT"] = np.reshape(ensemble["FAULT"][:, muv], (-1, 1), "F")
    controlbest2 = {}
    if "PERM" in input_variables:
        controlbest2["PERM"] = controljj["PERM"]  # controlbest
    if "PORO" in input_variables:
        controlbest2["PORO"] = controljj["PORO"]  # controlbest
    if "FAULT" in input_variables:
        controlbest2["FAULT"] = controljj["FAULT"]  # controlbest
    muvbad = np.argmax(cc)
    controlbad = {}
    if "PERM" in input_variables:
        controlbad["PERM"] = np.reshape(ensemble["PERM"][:, muvbad], (-1, 1), "F")
    if "PORO" in input_variables:
        controlbad["PORO"] = np.reshape(ensemble["PORO"][:, muvbad], (-1, 1), "F")
    if "FAULT" in input_variables:
        controlbad["FAULT"] = np.reshape(ensemble["FAULT"][:, muvbad], (-1, 1), "F")
    if dist.rank == 0:
        Plot_Histogram_now(Ne, cc_ini, cc, mean_cost, best_cost, oldfolder)
    if dist.rank == 0:
        if not os.path.exists(to_absolute_path("../RESULTS/HM_RESULTS/ADAPT_REKI")):
            os.makedirs(
                to_absolute_path("../RESULTS/HM_RESULTS/ADAPT_REKI"), exist_ok=True
            )
        else:
            shutil.rmtree(to_absolute_path("../RESULTS/HM_RESULTS/ADAPT_REKI"))
            os.makedirs(
                to_absolute_path("../RESULTS/HM_RESULTS/ADAPT_REKI"), exist_ok=True
            )
        logger.info(
            "**********************************************************************"
        )
        logger.info(
            "                   ANALYSIS FOR MLE RESERVOIR_MODEL                    "
        )
        logger.info(
            "***********************************************************************"
        )
    ensemblepy = ensemble_pytorch(
        controlbest,
        nx,
        ny,
        nz,
        controlbest["PERM"].shape[1],
        effective,
        oldfolder,
        target_min,
        target_max,
        minK,
        maxK,
        minT,
        maxT,
        minP,
        maxP,
        minQ,
        maxQ,
        minQw,
        maxQw,
        minQg,
        maxQg,
        steppi,
        device,
        steppi_indices,
        input_variables,
        cfg,
    )
    # removed unused variable mazw
    simout = Forward_model_ensemble(
        controlbest["PERM"].shape[1],
        ensemblepy,
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
        effectiveuse,
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
        quant_big,
        cfg,
        N_pr,
        lenwels,
        effective_abi,
        rows_to_remove,
        nx,
        ny,
        nz,
    )
    yycheck = simout["ouut_p"]
    if "PRESSURE" in output_variables:
        pree = simout["PRESSURE"]
    if "SWAT" in output_variables:
        wats = simout["SWAT"]
    if "SOIL" in output_variables:
        oilss = simout["SOIL"]
    if "SGAS" in output_variables:
        gasss = simout["SGAS"]
    os.chdir(to_absolute_path("../RESULTS/HM_RESULTS/ADAPT_REKI"))
    if dist.rank == 0:
        Plot_RSM_single(yycheck, Time_unie1, N_pr, well_names)
        Plot_petrophysical(
            controlbest["PERM"],
            controlbest["PORO"],
            nx,
            ny,
            nz,
            Low_K1,
            High_K1,
            effectiveuse,
            N_injw,
            N_pr,
            N_injg,
            injectors,
            producers,
            gass,
            Low_P,
            High_P,
        )
    X_data1 = {}
    if "PERM" in input_variables:
        X_data1["PERM"] = controlbest["PERM"]
    if "PORO" in input_variables:
        X_data1["PORO"] = controlbest["PORO"]
    if "FAULT" in input_variables:
        X_data1["FAULT"] = controlbest["FAULT"]
    if "PRESSURE" in output_variables:
        X_data1["PRESSURE"] = simout["PRESSURE"]
    if "SWAT" in output_variables:
        X_data1["SWAT"] = simout["SWAT"]
    if "SOIL" in output_variables:
        X_data1["SOIL"] = simout["SOIL"]
    if "SGAS" in output_variables:
        X_data1["SGAS"] = simout["SGAS"]
    X_data1["Simulated_data_plots"] = yycheck
    if dist.rank == 0:
        with gzip.open("RESERVOIR_MODEL.pkl.gz", "wb") as f1:
            pickle.dump(X_data1, f1)
    os.chdir(oldfolder)
    Time_vector = np.zeros((steppi))
    for kk in range(steppi):
        current_time = dt[kk]
        Time_vector[kk] = current_time
    folderrin = os.path.join(oldfolder, "..", "RESULTS", "HM_RESULTS", "ADAPT_REKI")
    if dist.rank == 0:
        Parallel(n_jobs=num_cores, backend="loky", verbose=10)(
            delayed(process_step)(
                kk,
                steppi,
                dt,
                pree,
                effectiveuse,
                wats,
                oilss,
                gasss,
                nx,
                ny,
                nz,
                N_injw,
                N_pr,
                N_injg,
                injectors,
                producers,
                gass,
                to_absolute_path(folderrin),
                oldfolder,
            )
            for kk in range(steppi)
        )
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, steppi - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
    os.chdir(to_absolute_path("../RESULTS/HM_RESULTS/ADAPT_REKI"))
    import glob

    if dist.rank == 0:
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
        from glob import glob

        for f3 in glob("*Dynamic*"):
            os.remove(f3)
        write_RSM(
            yycheck[0, :, : lenwels * N_pr], Time_vector, "PhyNeMo", well_names, N_pr
        )
        Plot_RSM_percentile_model(
            yycheck[0, :, : lenwels * N_pr], True_mat, Time_unie1, N_pr, well_names
        )
    os.chdir(oldfolder)
    yycheck = yycheck[0, :, : lenwels * N_pr]
    jesuni = []
    for k in range(lenwels):
        quantt = quant_big[f"K_{k}"]
        # ajes = quantt["value"]
        if quantt["boolean"] == 1:
            kodsval = yycheck[:, k * N_pr : (k + 1) * N_pr] / quantt["scale"]
        else:
            kodsval = yycheck[:, k * N_pr : (k + 1) * N_pr] * quantt["scale"]
        jesuni.append(kodsval)
    usesim = np.hstack(jesuni)
    usesim = np.reshape(usesim, (-1, 1), "F")
    usesim = remove_rows(usesim, rows_to_remove).reshape(-1, 1)
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim
    cc = ((np.sum((((usesim) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    logger.info("RMSE OF MLE_RESERVOIR_MODEL   =  %s", str(cc))
    os.chdir(to_absolute_path("../RESULTS/HM_RESULTS"))
    if dist.rank == 0:
        Plot_mean(
            controlbest["PERM"],
            yes_mean["PERM"],
            meanini,
            nx,
            ny,
            nz,
            Low_K1,
            High_K1,
            True_K,
            effectiveuse,
            injectors,
            producers,
            gass,
            N_injw,
            N_pr,
            N_injg,
        )
    os.chdir(oldfolder)
    if dist.rank == 0:
        logger.info(
            "**********************************************************************"
        )
        logger.info(
            "                ANALYSIS FOR BEST_RESERVOIR_MODEL                      "
        )
        logger.info(
            "***********************************************************************"
        )
        if not os.path.exists(
            to_absolute_path("../RESULTS/HM_RESULTS/BEST_RESERVOIR_MODEL")
        ):
            os.makedirs(
                to_absolute_path("../RESULTS/HM_RESULTS/BEST_RESERVOIR_MODEL"),
                exist_ok=True,
            )
        else:
            shutil.rmtree(
                to_absolute_path("../RESULTS/HM_RESULTS/BEST_RESERVOIR_MODEL")
            )
            os.makedirs(
                to_absolute_path("../RESULTS/HM_RESULTS/BEST_RESERVOIR_MODEL"),
                exist_ok=True,
            )
    ensemblepy = ensemble_pytorch(
        yes_best,
        nx,
        ny,
        nz,
        controlbest["PERM"].shape[1],
        effective,
        oldfolder,
        target_min,
        target_max,
        minK,
        maxK,
        minT,
        maxT,
        minP,
        maxP,
        minQ,
        maxQ,
        minQw,
        maxQw,
        minQg,
        maxQg,
        steppi,
        device,
        steppi_indices,
        input_variables,
        cfg,
    )
    os.chdir(oldfolder)
    # removed unused variable mazw
    simout = Forward_model_ensemble(
        controlbest["PERM"].shape[1],
        ensemblepy,
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
        effectiveuse,
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
        quant_big,
        cfg,
        N_pr,
        lenwels,
        effective_abi,
        rows_to_remove,
        nx,
        ny,
        nz,
    )
    yycheck = simout["ouut_p"]
    if "PRESSURE" in output_variables:
        preebest = simout["PRESSURE"]
    if "SWAT" in output_variables:
        watsbest = simout["SWAT"]
    if "SOIL" in output_variables:
        oilssbest = simout["SOIL"]
    if "SGAS" in output_variables:
        gasbest = simout["SGAS"]
    os.chdir(to_absolute_path("../RESULTS/HM_RESULTS/BEST_RESERVOIR_MODEL"))
    if dist.rank == 0:
        Plot_RSM_single(yycheck, Time_unie1, N_pr, well_names)
        Plot_petrophysical(
            yes_best["PERM"],
            yes_best["PORO"],
            nx,
            ny,
            nz,
            Low_K1,
            High_K1,
            effectiveuse,
            N_injw,
            N_pr,
            N_injg,
            injectors,
            producers,
            gass,
            Low_P,
            High_P,
        )
    X_data1 = {}
    if "PERM" in input_variables:
        X_data1["PERM"] = yes_best["PERM"]
    if "PORO" in input_variables:
        X_data1["PORO"] = yes_best["PORO"]
    if "FAULT" in input_variables:
        X_data1["FAULT"] = yes_best["FAULT"]
    if "PRESSURE" in output_variables:
        X_data1["PRESSURE"] = simout["PRESSURE"]
    if "SWAT" in output_variables:
        X_data1["SWAT"] = simout["SWAT"]
    if "SOIL" in output_variables:
        X_data1["SOIL"] = simout["SOIL"]
    if "SGAS" in output_variables:
        X_data1["SGAS"] = simout["SGAS"]
    X_data1["Simulated_data_plots"] = yycheck
    if dist.rank == 0:
        with gzip.open("BEST_RESERVOIR_MODEL.pkl.gz", "wb") as f1:
            pickle.dump(X_data1, f1)
    os.chdir(oldfolder)
    folderrin = os.path.join(
        oldfolder, "..", "RESULTS", "HM_RESULTS", "BEST_RESERVOIR_MODEL"
    )
    import glob

    if dist.rank == 0:
        Parallel(n_jobs=num_cores, backend="loky", verbose=10)(
            delayed(process_step)(
                kk,
                steppi,
                dt,
                preebest,
                effectiveuse,
                watsbest,
                oilssbest,
                gasbest,
                nx,
                ny,
                nz,
                N_injw,
                N_pr,
                N_injg,
                injectors,
                producers,
                gass,
                to_absolute_path(folderrin),
                oldfolder,
            )
            for kk in range(steppi)
        )

        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, steppi - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        os.chdir(to_absolute_path("../RESULTS/HM_RESULTS/BEST_RESERVOIR_MODEL"))
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
        from glob import glob

        for f3 in glob("*Dynamic*"):
            os.remove(f3)

        write_RSM(
            yycheck[0, :, : lenwels * N_pr], Time_vector, "PhyNeMo", well_names, N_pr
        )
        Plot_RSM_percentile_model(
            yycheck[0, :, : lenwels * N_pr], True_mat, Time_unie1, N_pr, well_names
        )
    os.chdir(oldfolder)
    yycheck = yycheck[0, :, : lenwels * N_pr]
    jesuni = []
    for k in range(lenwels):
        quantt = quant_big[f"K_{k}"]
        # ajes = quantt["value"]
        if quantt["boolean"] == 1:
            kodsval = yycheck[:, k * N_pr : (k + 1) * N_pr] / quantt["scale"]
        else:
            kodsval = yycheck[:, k * N_pr : (k + 1) * N_pr] * quantt["scale"]
        jesuni.append(kodsval)
    usesim = np.hstack(jesuni)
    usesim = np.reshape(usesim, (-1, 1), "F")
    usesim = remove_rows(usesim, rows_to_remove).reshape(-1, 1)
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim
    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    if dist.rank == 0:
        logger.info("RMSE OF BEST RESERVOIR MODEL  =  %s", str(cc))
        logger.info(
            "**********************************************************************"
        )
        logger.info(
            "              ANALYSIS FOR MEAN_RESERVOIR_MODEL                       "
        )
        logger.info(
            "**********************************************************************"
        )
        if not os.path.exists(
            to_absolute_path("../RESULTS/HM_RESULTS/MEAN_RESERVOIR_MODEL")
        ):
            os.makedirs(
                to_absolute_path("../RESULTS/HM_RESULTS/MEAN_RESERVOIR_MODEL"),
                exist_ok=True,
            )
        else:
            shutil.rmtree(
                to_absolute_path("../RESULTS/HM_RESULTS/MEAN_RESERVOIR_MODEL")
            )
            os.makedirs(
                to_absolute_path("../RESULTS/HM_RESULTS/MEAN_RESERVOIR_MODEL"),
                exist_ok=True,
            )
    ensemblepy = ensemble_pytorch(
        yes_mean,
        nx,
        ny,
        nz,
        yes_mean["PERM"].shape[1],
        effective,
        oldfolder,
        target_min,
        target_max,
        minK,
        maxK,
        minT,
        maxT,
        minP,
        maxP,
        minQ,
        maxQ,
        minQw,
        maxQw,
        minQg,
        maxQg,
        steppi,
        device,
        steppi_indices,
        input_variables,
        cfg,
    )
    os.chdir(oldfolder)
    # mazw = 0  # Dont smooth the presure field
    simout = Forward_model_ensemble(
        controlbest2["PERM"].shape[1],
        ensemblepy,
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
        effectiveuse,
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
        quant_big,
        cfg,
        N_pr,
        lenwels,
        effective_abi,
        rows_to_remove,
        nx,
        ny,
        nz,
    )
    yycheck = simout["ouut_p"]
    if "PRESSURE" in output_variables:
        preebest = simout["PRESSURE"]
    if "SWAT" in output_variables:
        watsbest = simout["SWAT"]
    if "SOIL" in output_variables:
        oilssbest = simout["SOIL"]
    if "SGAS" in output_variables:
        gasbest = simout["SGAS"]
    os.chdir(to_absolute_path("../RESULTS/HM_RESULTS/MEAN_RESERVOIR_MODEL"))
    if dist.rank == 0:
        Plot_RSM_single(yycheck, Time_unie1, N_pr, well_names)
        Plot_petrophysical(
            yes_mean["PERM"],
            yes_mean["PORO"],
            nx,
            ny,
            nz,
            Low_K1,
            High_K1,
            effectiveuse,
            N_injw,
            N_pr,
            N_injg,
            injectors,
            producers,
            gass,
            Low_P,
            High_P,
        )
    X_data1 = {}
    if "PERM" in input_variables:
        X_data1["PERM"] = yes_mean["PERM"]
    if "PORO" in input_variables:
        X_data1["PORO"] = yes_mean["PORO"]
    if "FAULT" in input_variables:
        X_data1["FAULT"] = yes_mean["FAULT"]
    if "PRESSURE" in output_variables:
        X_data1["PRESSURE"] = simout["PRESSURE"]
    if "SWAT" in output_variables:
        X_data1["SWAT"] = simout["SWAT"]
    if "SOIL" in output_variables:
        X_data1["SOIL"] = simout["SOIL"]
    if "SGAS" in output_variables:
        X_data1["SGAS"] = simout["SGAS"]
    X_data1["Simulated_data_plots"] = yycheck
    if dist.rank == 0:
        with gzip.open("MEAN_RESERVOIR_MODEL.pkl.gz", "wb") as f1:
            pickle.dump(X_data1, f1)
    os.chdir(oldfolder)
    folderrin = os.path.join(
        oldfolder, "..", "RESULTS", "HM_RESULTS", "MEAN_RESERVOIR_MODEL"
    )
    import glob

    if dist.rank == 0:
        Parallel(n_jobs=num_cores, backend="loky", verbose=10)(
            delayed(process_step)(
                kk,
                steppi,
                dt,
                preebest,
                effectiveuse,
                watsbest,
                oilssbest,
                gasbest,
                nx,
                ny,
                nz,
                N_injw,
                N_pr,
                N_injg,
                injectors,
                producers,
                gass,
                to_absolute_path(folderrin),
                oldfolder,
            )
            for kk in range(steppi)
        )

        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, steppi - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        os.chdir(to_absolute_path("../RESULTS/HM_RESULTS/MEAN_RESERVOIR_MODEL"))
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
        from glob import glob

        for f3 in glob("*Dynamic*"):
            os.remove(f3)

        write_RSM(
            yycheck[0, :, : lenwels * N_pr], Time_vector, "PhyNeMo", well_names, N_pr
        )
        Plot_RSM_percentile_model(
            yycheck[0, :, : lenwels * N_pr], True_mat, Time_unie1, N_pr, well_names
        )
    os.chdir(oldfolder)
    yycheck = yycheck[0, :, : lenwels * N_pr]
    jesuni = []
    for k in range(lenwels):
        quantt = quant_big[f"K_{k}"]
        # ajes = quantt["value"]
        if quantt["boolean"] == 1:
            kodsval = yycheck[:, k * N_pr : (k + 1) * N_pr] / quantt["scale"]
        else:
            kodsval = yycheck[:, k * N_pr : (k + 1) * N_pr] * quantt["scale"]
        jesuni.append(kodsval)
    usesim = np.hstack(jesuni)
    usesim = np.reshape(usesim, (-1, 1), "F")
    usesim = remove_rows(usesim, rows_to_remove).reshape(-1, 1)
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim
    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    if dist.rank == 0:
        logger.info("RMSE of MAP RESERVOIR MODEL  =  %s", str(cc))
    os.chdir(to_absolute_path("../RESULTS/HM_RESULTS"))
    X_data1 = {
        "PERM_Reali": ensemble["PERM"],
        "FAULT_Reali": ensemble["FAULT"],
        "PORO_Reali": ensemble["PORO"],
        "P10_Perm": controlbest["PERM"],
        "P50_Perm": controljj["PERM"],
        "P90_Perm": controlbad["PERM"],
        "P10_Poro": controlbest["PORO"],
        "P50_Poro": controljj["PORO"],
        "P90_Poro": controlbad["PORO"],
        "P10_Fault": controlbest["FAULT"],
        "P50_Fault": controljj["FAULT"],
        "P90_Fault": controlbad["FAULT"],
        "Simulated_data": simDatafinal,
        "Simulated_data_plots": predMatrix,
        "Pressures": pressure_ensemble,
        "Water_saturation": water_ensemble,
        "Oil_saturation": oil_ensemble,
        "Gas_saturation": gas_ensemble,
        "Simulated_data_best_ensemble": simDatafinala,
        "Simulated_data_plots_best_ensemble": predMatrixa,
        "Pressures_best_ensemble": pressure_ensemblea,
        "Water_saturation_best_ensemble": water_ensemblea,
        "Oil_saturation_best_ensemble": oil_ensemblea,
        "Gas_saturation_best_ensemble": gas_ensemblea,
        "ensemble_best": ensemble_best,
        "yes_best": yes_best,
        "ensemble_mean": ensemble_mean,
        "yes_mean": yes_mean,
        "all_ensemble": all_ensemble,
        "ensemble_dict": ensemble_dict,
    }

    return X_data1
