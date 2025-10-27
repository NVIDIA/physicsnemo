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
                  HISTORY MATCHING (α-REKI) UTILITIES
=====================================================================

This module implements an iterative history-matching workflow based on
Adaptive Regularised Ensemble Kalman Inversion (α-REKI) for reservoir
simulation using NVIDIA PhyNeMo surrogates. It supports optional
parametrisation (DCT/VCAE) and spatial localisation, and logs per-iteration
cost statistics for both ensemble mean and best member.

Key Features:
- α-REKI update with adaptive scalar α from data-mismatch statistics
- Optional DCT/VCAE parameter-space updates and spatial localisation
- Multi-target forward simulation via PhyNeMo surrogates
- Iteration-wise cost tracking and result visualisation
- GPU-aware execution and mixed NumPy/Torch handling

Typical Usage:
    from inverse.history_matching import run_history_matching_loop
    results = run_history_matching_loop(dist, logger, cfg, ...)

Inputs (high level):
- Configuration (Hydra), distributed context, and logging
- Prior ensembles (permeability/porosity/fault) and grid dimensions
- Normalisation/physics parameters and forward-model assets
- Measurement vectors and well metadata

Outputs (high level):
- Updated ensembles (PERM/PORO/FAULT) and bookkeeping arrays
- Per-iteration costs, α values, and convergence indicators

@Author : Clement Etienam
"""

# 🛠 Standard Library
import os
import logging
import warnings


# 🔧 Third-party Libraries
import numpy as np
import numpy.matlib


# 🔥 PhyNeMo & ML Libraries
import torch
import torch.optim as optim
from hydra.utils import to_absolute_path
import matplotlib.pyplot as plt


# 📦 Local Modules
from inverse.inversion_operation_surrogate import (
    Forward_model_ensemble,
    remove_rows,
)

from inverse.inversion_operation_ensemble import (
    funcGetDataMismatch,
    honour2,
    ensemble_pytorch,
)
from inverse.inversion_operation_gather import (
    Plot_RSM,
    Plot_RSM_singleT,
    Add_marker2,
)

from inverse.utils.ensemble_generation import (
    NorneInitialEnsemble,
    historydata,
)

from inverse.inversion_operation_uq import (
    dct22,
    idct22,
)


from inverse.inversion_operation_misc import (
    VCAE3D,
    Train_VCAE,
    encode_values,
    decode_values,
    Localisation,
    Get_Kalman_Gain,
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
    warnings.filterwarnings("ignore")
    return logger


def run_history_matching_loop(
    dist,
    logger,
    cfg,
    iteration_converged,
    iteration_count,
    Termm,
    input_variables,
    ensemble,
    ensemblep,
    ensemblef,
    nx,
    ny,
    nz,
    Ne,
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
    min_inn_fcn,
    max_inn_fcn,
    models,
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
    output_variables,
    quant_big,
    N_pr,
    lenwels,
    effective_abi,
    rows_to_remove,
    timestep,
    Time_unie1,
    well_names,
    CDd,
    gpu_available,
    Do_parametrisation,
    Do_param_method,
    size1,
    size2,
    Low_K1,
    High_K1,
    High_P,
    Low_P,
    N_ens,
    do_localisation,
    gass,
    injectors,
    effec,
    True_mat,
    perturbations,
) -> tuple:
    """Run the α-REKI history-matching loop and return updated ensembles.

    This routine performs iterative history matching using an Adaptive
    Regularised Ensemble Kalman Inversion approach. For each iteration, it:
    1) builds simulation inputs from ensembles; 2) runs the forward surrogate
    model; 3) evaluates data mismatch and computes an adaptive scalar α; 4)
    applies optional parametrisation (DCT/VCAE) and optional localisation; 5)
    updates ensemble members; 6) tracks and logs costs and best statistics.

    Notes
    -----
    - Arrays are expected to be shaped consistently with grid dimensions
      ``(nx, ny, nz)`` and ensemble size ``Ne``.

    Returns
    -------
    tuple
        A tuple containing, in order: ``use_k``, ``use_p``, ``use_f``,
        ``mean_cost``, ``best_cost``, ``ensemble_bestK``, ``ensemble_meanK``,
        ``ensemble_bestP``, ``ensemble_meanP``, ``ensemble_bestf``,
        ``ensemble_meanf``, ``iteration_count``, ``iteration_converged``,
        ``alpha_big``, ``ensemble``, ``ensemblep``, ``ensemblef``, ``chm``,
        ``cc_ini``, ``ensemble_dict``, ``base_k``, ``base_p``, ``base_f``.
    """

    alpha_big = []
    mean_cost = []
    best_cost = []
    if "PERM" in input_variables:
        ensemble_meanK = []
        ensemble_bestK = []
        base_k = np.mean(ensemble, axis=1).reshape(-1, 1)
    if "PORO" in input_variables:
        ensemble_meanP = []
        ensemble_bestP = []
        base_p = np.mean(ensemblep, axis=1).reshape(-1, 1)
    if "FAULT" in input_variables:
        ensemble_meanf = []
        ensemble_bestf = []
        base_f = np.mean(ensemblef, axis=1).reshape(-1, 1)
    if dist.rank == 0:
        logger.info(
            "History Matching using the Adaptive Regularised Ensemble Kalman Inversion (α-REKI)"
        )
    ensemble_dict = {}
    overall_list = {}
    Youtt = {}
    updated_ensemble = {}

    while iteration_converged < 1:
        if dist.rank == 0:
            logger.info(
                "****************************************************************"
            )
            logger.info(f"Iteration --{iteration_count + 1} | {Termm}")
            logger.info(
                "****************************************************************"
            )
        if "PERM" in input_variables:
            ensemble_dict["PERM"] = ensemble
            ini_K = ensemble
        if "PORO" in input_variables:
            ensemble_dict["PORO"] = ensemblep
            ini_p = ensemblep
        if "FAULT" in input_variables:
            ensemble_dict["FAULT"] = ensemblef
            ini_f = ensemblef
        ensemblepy = ensemble_pytorch(
            ensemble_dict,
            nx,
            ny,
            nz,
            Ne,
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
        # mazw = 0
        simout = Forward_model_ensemble(
            ensemble.shape[1],
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
        predMatrix = simout["ouut_p"]
        simDatafinal = simout["sim"]
        if iteration_count == 0:
            os.chdir(to_absolute_path("../RESULTS/HM_RESULTS"))
            if dist.rank == 0:
                Plot_RSM(
                    predMatrix,
                    True_mat,
                    "Initial.png",
                    Ne,
                    Time_unie1,
                    N_pr,
                    well_names,
                )
            os.chdir(oldfolder)
        else:
            pass
        _, True_data1, True_mat = historydata(timestep, steppi, steppi_indices, N_pr)
        True_mat[True_mat <= 0] = 0
        os.chdir(to_absolute_path("../RESULTS/HM_RESULTS"))
        if dist.rank == 0:
            Plot_RSM_singleT(True_mat, Time_unie1, N_pr, well_names)
        os.chdir(oldfolder)
        jesuni = []
        for k in range(lenwels):
            quantt = quant_big[f"K_{k}"]
            # ajes = quantt["value"]
            if quantt["boolean"] == 1:
                kodsval = True_mat[:, k * N_pr : (k + 1) * N_pr] / quantt["scale"]
            else:
                kodsval = True_mat[:, k * N_pr : (k + 1) * N_pr] * quantt["scale"]
            jesuni.append(kodsval)
        True_data = np.hstack(jesuni)
        True_data = np.reshape(True_data, (-1, 1), "F")
        True_data = remove_rows(True_data, rows_to_remove).reshape(-1, 1)
        # True_yet = True_data
        True_dataa = torch.tensor(True_data, dtype=torch.float32).to(device)
        # adaptive_rho undefined; skip dynamic update to satisfy linter
        CDd = torch.tensor(CDd, dtype=torch.float32).to(device)
        Ddraw = True_dataa.repeat(1, Ne).to(device)
        Dd = Ddraw  # + pertubations
        if gpu_available == 0:
            CDd = CDd.detach().cpu().numpy()
            Dd = Dd.detach().cpu().numpy()
            Ddraw = Ddraw.detach().cpu().numpy()
            True_dataa = True_dataa.detach().cpu().numpy()
        else:
            pass
        if not isinstance(Dd, torch.Tensor):
            Dd = torch.as_tensor(Dd, dtype=torch.float32, device=device)
        if not isinstance(simDatafinal, torch.Tensor):
            simDatafinal = torch.as_tensor(
                simDatafinal, dtype=torch.float32, device=device
            )
        if not isinstance(CDd, torch.Tensor):
            CDd = torch.as_tensor(CDd, dtype=torch.float32, device=device)
        yyy = 0.5 * (Dd - simDatafinal).T @ torch.linalg.inv(CDd) @ (Dd - simDatafinal)
        yyy = 0.5 * (Dd - simDatafinal).T @ torch.linalg.inv(CDd) @ (Dd - simDatafinal)
        yyy = torch.mean(yyy, dim=1).to(device)  # Compute mean along dim=1
        yyy = torch.nan_to_num(yyy, nan=0.0).reshape(-1, 1)  # Remove NaNs and reshape
        alpha_star = torch.mean(yyy)  # No need for dim=0
        alpha_star2 = torch.var(yyy)  # Variance is equivalent to std²
        leftt = True_data.shape[0] / (2 * alpha_star)
        rightt = torch.sqrt(True_data.shape[0] / (2 * alpha_star2))
        chok = torch.clamp(torch.max(leftt, rightt), max=1 - iteration_converged)
        alpha = 1 / chok
        alpha_big.append(alpha.item())  # Convert only at the final step
        if dist.rank == 0:
            logger.info(f"alpha = {alpha.item()}")
            logger.info(f"iteration_converged = {iteration_converged}")
        if Do_parametrisation == "No":
            if "PERM" in input_variables:
                overall_list["PERM"] = torch.tensor(
                    ensemble_dict["PERM"], dtype=torch.float32
                ).to(device)
            if "PORO" in input_variables:
                overall_list["PORO"] = torch.tensor(
                    ensemble_dict["PORO"], dtype=torch.float32
                ).to(device)
            if "FAULT" in input_variables:
                overall_list["FAULT"] = torch.tensor(
                    ensemble_dict["FAULT"], dtype=torch.float32
                ).to(device)
            overall = overall_list
        else:
            if iteration_count == 0:
                if Do_param_method == "DCT":
                    if dist.rank == 0:
                        logger.info(
                            "Adaptive Regularised Ensemble Kalman Inversion with DCT Parametrisation"
                        )
                        logger.info(
                            "Novel Implementation: Author: Clement Etienam - DevTech Energy @Nvidia"
                        )
                        logger.info(
                            "Starting the History matching with "
                            + str(Ne)
                            + " Ensemble members"
                        )
                    os.chdir(oldfolder)
                    if "PERM" in input_variables:
                        if dist.rank == 0:
                            small = dct22(
                                ensemble_dict["PERM"], Ne, nx, ny, nz, size1, size2
                            )
                            recc = idct22(small, Ne, nx, ny, nz, size1, size2)
                            dimms = (small.shape[0] / ensemble.shape[0]) * 100
                            dimms = round(dimms, 3)
                            recover = np.reshape(recc[:, 0], (nx, ny, nz), "F")
                            origii = np.reshape(
                                ensemble_dict["PERM"][:, 0], (nx, ny, nz), "F"
                            )
                            recover = np.mean(recover, axis=2)
                            origii = np.mean(origii, axis=2)
                            plt.figure(figsize=(20, 20))
                            XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
                            plt.subplot(2, 2, 1)
                            plt.pcolormesh(XX.T, YY.T, recover, cmap="jet")
                            plt.title("Recovered - mean", fontsize=15)
                            plt.ylabel("Y", fontsize=13)
                            plt.xlabel("X", fontsize=13)
                            plt.axis([0, (nx - 1), 0, (ny - 1)])
                            plt.gca().set_xticks([])
                            plt.gca().set_yticks([])
                            cbar1 = plt.colorbar()
                            cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
                            plt.clim(Low_K1, High_K1)
                            plt.subplot(2, 2, 2)
                            plt.pcolormesh(XX.T, YY.T, origii, cmap="jet")
                            plt.title("True - mean", fontsize=15)
                            plt.ylabel("Y", fontsize=13)
                            plt.xlabel("X", fontsize=13)
                            plt.axis([0, (nx - 1), 0, (ny - 1)])
                            plt.gca().set_xticks([])
                            plt.gca().set_yticks([])
                            cbar1 = plt.colorbar()
                            cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
                            plt.clim(Low_K1, High_K1)
                            plt.tight_layout(rect=[0, 0, 1, 0.95])
                            ttitle = (
                                "Parameter Recovery with "
                                + str(dimms)
                                + "% of original value"
                            )
                            plt.suptitle(ttitle, fontsize=20)
                            os.chdir("../RESULTS/HM_RESULTS")
                            plt.savefig("Recover_Comparison.png")
                            os.chdir(oldfolder)
                            plt.close()
                            plt.clf()
                        # sgsim = ensemble_dict["PERM"]
                        ensembledct = dct22(
                            ensemble_dict["PERM"], Ne, nx, ny, nz, size1, size2
                        )
                    if "PORO" in input_variables:
                        ensembledctp = dct22(
                            ensemble_dict["PORO"], Ne, nx, ny, nz, size1, size2
                        )
                if Do_param_method == "VCAE":
                    permxi, poroxi, _ = NorneInitialEnsemble(
                        nx, ny, nz, ensembleSize=3000, randomNumber=1.2345e5
                    )
                    if dist.rank == 0:
                        logger.info(
                            "Adaptive Regularised Ensemble Kalman Inversion with VCAE Parametrisation"
                        )
                        logger.info(
                            "Novel Implementation: Author: Clement Etienam - DevTech Energy @Nvidia"
                        )
                        logger.info(
                            "Starting the History matching with "
                            + str(Ne)
                            + " Ensemble members"
                        )
                    if "PERM" in input_variables:
                        sizedct = cfg.custom.INVERSE_PROBLEM.DCT
                        sizedct = sizedct / 100
                        latent_dim = 600  # int(sizedct *nx*ny*nz)
                        file_path = to_absolute_path("../MODELS/perm_vcae.pth")
                        file_exists = os.path.isfile(file_path)
                        if file_exists:
                            state_dict = torch.load(file_path, map_location=device)
                            model_perm = VCAE3D(latent_dim).to(device)
                            dummy_input = torch.randn(1, 1, nz, nx, ny).to(device)
                            _ = model_perm(
                                dummy_input
                            )  # triggers creation of fc_mu, fc_logvar, decoder_input
                            model_perm.load_state_dict(state_dict)
                            model_perm = model_perm.to(device)
                            model_perm.eval()
                        else:
                            permxi, _, _ = NorneInitialEnsemble(
                                nx, ny, nz, ensembleSize=3000, randomNumber=1.2345e5
                            )
                            model_perm = VCAE3D(latent_dim).to(device)
                            optimizer_perm = optim.Adam(
                                model_perm.parameters(),
                                lr=cfg.optimizer.lr,
                                betas=(0.9, 0.999),
                                weight_decay=cfg.optimizer.weight_decay,
                            )
                            scheduler_perm = optim.lr_scheduler.ExponentialLR(
                                optimizer_perm, gamma=cfg.optimizer.gamma
                            )
                            model_perm = Train_VCAE(
                                cfg.optimizer.lr,
                                latent_dim,
                                500,
                                100,
                                device,
                                permxi / maxK,
                                nz,
                                nx,
                                ny,
                                model_perm,
                                optimizer_perm,
                                scheduler_perm,
                            )
                            torch.save(
                                model_perm.state_dict(), "../MODELS/perm_vcae.pth"
                            )
                        ensembledct = encode_values(
                            ensemble_dict["PERM"] / maxK,
                            nz,
                            nx,
                            ny,
                            device,
                            model_perm,
                        )
                    if "PORO" in input_variables:
                        file_path = to_absolute_path("../MODELS/poro_vcae.pth")
                        file_exists = os.path.isfile(file_path)
                        latent_dim = 600
                        if file_exists:
                            state_dict = torch.load(file_path, map_location=device)
                            model_poro = VCAE3D(latent_dim).to(device)
                            dummy_input = torch.randn(1, 1, nz, nx, ny).to(device)
                            _ = model_poro(
                                dummy_input
                            )  # triggers creation of fc_mu, fc_logvar, decoder_input
                            model_poro.load_state_dict(state_dict)
                            model_poro = model_poro.to(device)
                            model_poro.eval()
                        else:
                            _, poroxi, _ = NorneInitialEnsemble(
                                nx, ny, nz, ensembleSize=3000, randomNumber=1.2345e5
                            )
                            model_poro = VCAE3D(latent_dim).to(device)
                            optimizer_poro = optim.Adam(
                                model_poro.parameters(),
                                lr=cfg.optimizer.lr,
                                betas=(0.9, 0.999),
                                weight_decay=cfg.optimizer.weight_decay,
                            )
                            scheduler_poro = optim.lr_scheduler.ExponentialLR(
                                optimizer_poro, gamma=cfg.optimizer.gamma
                            )
                            model_poro = Train_VCAE(
                                cfg.optimizer.lr,
                                latent_dim,
                                500,
                                100,
                                device,
                                poroxi,
                                nz,
                                nx,
                                ny,
                                model_poro,
                                optimizer_poro,
                                scheduler_poro,
                            )
                            torch.save(
                                model_poro.state_dict(), "../MODELS/poro_vcae.pth"
                            )
                        ensembledctp = encode_values(
                            ensemble_dict["PORO"], nz, nx, ny, device, model_poro
                        )
            else:
                if Do_param_method == "DCT":
                    if "PERM" in input_variables:
                        ensembledct = dct22(
                            ensemble_dict["PERM"], Ne, nx, ny, nz, size1, size2
                        )
                        # shapalla = ensembledct.shape[0]
                    if "PORO" in input_variables:
                        ensembledctp = dct22(
                            ensemble_dict["PORO"], Ne, nx, ny, nz, size1, size2
                        )
                else:
                    if "PERM" in input_variables:
                        ensembledct = encode_values(
                            ensemble_dict["PERM"] / maxK,
                            nz,
                            nx,
                            ny,
                            device,
                            model_perm,
                        )
                    if "PORO" in input_variables:
                        ensembledctp = encode_values(
                            ensemble_dict["PORO"],
                            nz,
                            nx,
                            ny,
                            device,
                            model_poro,
                        )
            if "PERM" in input_variables:
                overall_list["PERM"] = torch.tensor(
                    ensembledct, dtype=torch.float32
                ).to(device)
            if "PORO" in input_variables:
                overall_list["PORO"] = torch.tensor(
                    ensembledctp, dtype=torch.float32
                ).to(device)
            if "FAULT" in input_variables:
                overall_list["FAULT"] = torch.tensor(ensemblef, dtype=torch.float32).to(
                    device
                )
            overall = overall_list
        if (Do_parametrisation == "No") and (do_localisation == "Yes"):
            if iteration_count == 0:
                locmat = Localisation(10, nx, ny, nz, Ne, gass, producers, injectors)
                see1 = locmat[: nx * ny * nz, :] * effec
                XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
                look = np.reshape(see1[:, 1], (nx, ny, nz), "F")
                look[look == 0] = np.nan
                plt.figure(figsize=(40, 40))
                if dist.rank == 0:
                    for kkt in range(nz):
                        plt.subplot(5, 5, int(kkt + 1))
                        plt.pcolormesh(XX.T, YY.T, look[:, :, kkt], cmap="jet")
                        string = "Layer " + str(kkt + 1)
                        plt.title(string, fontsize=13)
                        plt.ylabel("Y", fontsize=13)
                        plt.xlabel("X", fontsize=13)
                        plt.axis([0, (nx - 1), 0, (ny - 1)])
                        plt.gca().set_xticks([])
                        plt.gca().set_yticks([])
                        cbar1 = plt.colorbar()
                        cbar1.ax.set_ylabel(" Localisation Matrix", fontsize=13)
                    Add_marker2(plt, XX, YY, injectors, producers, gass)
                    plt.savefig("../RESULTS/HM_RESULTS/Localisation_matrix.png")
                    plt.clf()
                    plt.close()
                locmat = torch.as_tensor(locmat, dtype=torch.float32, device=device)
        for key, tensor in overall.items():
            if dist.rank == 0:
                logger.info(f"Processing key: {key}")
            Y = tensor
            update_term = Get_Kalman_Gain(
                Y, simDatafinal, CDd, alpha, device, perturbations, True_data, Ne, dist
            )
            if (Do_parametrisation == "No") and (do_localisation == "Yes"):
                if key == "PERM" or key == "PORO":
                    update_term = update_term * locmat
            Ynew = Y + update_term
            Youtt[key] = Ynew
        if gpu_available == 0:
            if "PERM" in input_variables:
                updated_ensemble["PERM"] = Youtt["PERM"].detach().cpu().numpy()
            if "PORO" in input_variables:
                updated_ensemble["PORO"] = Youtt["PORO"].detach().cpu().numpy()
            if "FAULT" in input_variables:
                updated_ensemble["FAULT"] = Youtt["FAULT"].detach().cpu().numpy()
        else:
            if "PERM" in input_variables:
                updated_ensemble["PERM"] = Youtt["PERM"]
            if "PORO" in input_variables:
                updated_ensemble["PORO"] = Youtt["PORO"]
            if "FAULT" in input_variables:
                updated_ensemble["FAULT"] = Youtt["FAULT"]
        True_dataa1 = torch.as_tensor(True_dataa, dtype=torch.float32, device=device)
        if iteration_count == 0:
            simmean = torch.mean(simDatafinal, dim=1, keepdim=True)
            tinuke1 = torch.sqrt(
                torch.sum((simmean - True_dataa1) ** 2)
            ) / torch.tensor(True_dataa.shape[0], dtype=simmean.dtype, device=device)
            tinuke = tinuke1.detach().cpu().numpy()  # Move to CPU and convert to NumPy
            if dist.rank == 0:
                logger.info(f"Initial RMSE of the ensemble mean =  {tinuke}... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa1)
            muv = torch.argmin(cc)
            simmbest = simDatafinal[:, muv].reshape(-1, 1)
            tinukebest = torch.sqrt(
                torch.sum((simmbest - True_dataa1) ** 2)
            ) / torch.tensor(True_dataa.shape[0], device=simmean.device)
            tinukebest = tinukebest.detach().cpu().numpy()
            if dist.rank == 0:
                logger.info(f"Initial RMSE of the ensemble best =  {tinukebest}... .")
            cc_ini = cc
            tinumeanprior = tinuke
            tinubestprior = tinukebest
            best_cost_mean = tinumeanprior
            best_cost_best = tinubestprior
        else:
            simmean = torch.mean(simDatafinal, dim=1, keepdim=True)
            tinuke = torch.sqrt(torch.sum((simmean - True_dataa1) ** 2)) / torch.tensor(
                True_dataa.shape[0], device=simmean.device
            )
            tinuke = tinuke.detach().cpu().numpy()
            if dist.rank == 0:
                logger.info(f"RMSE of the ensemble mean = : {tinuke}... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa1)
            muv = torch.argmin(cc)
            simmbest = simDatafinal[:, muv].reshape(-1, 1)
            tinukebest = torch.sqrt(
                torch.sum((simmbest - True_dataa1) ** 2)
            ) / torch.tensor(True_dataa.shape[0], device=simmean.device)
            tinukebest = tinukebest.detach().cpu().numpy()
            if dist.rank == 0:
                logger.info(f"RMSE of the ensemble best = {tinukebest}")
            if tinuke < tinumeanprior:
                if dist.rank == 0:
                    logger.info(
                        f"ensemble mean cost decreased by = : {abs(tinuke - tinumeanprior)}... ."
                    )
            if tinuke > tinumeanprior:
                if dist.rank == 0:
                    logger.info(
                        f"ensemble mean cost increased by = : {abs(tinuke - tinumeanprior)}... ."
                    )
            if tinuke == tinumeanprior:
                if dist.rank == 0:
                    logger.info("No change in ensemble mean cost")
            if tinukebest > tinubestprior:
                if dist.rank == 0:
                    logger.info(
                        f"ensemble best cost increased by =  {abs(tinukebest - tinubestprior)}... ."
                    )
            if tinukebest < tinubestprior:
                if dist.rank == 0:
                    logger.info(
                        f"ensemble best cost decreased by =  {abs(tinukebest - tinubestprior)}... ."
                    )
            if tinukebest == tinubestprior:
                if dist.rank == 0:
                    logger.info("No change in ensemble best cost")
            tinumeanprior = tinuke
            tinubestprior = tinukebest
        if best_cost_mean > tinuke:
            if dist.rank == 0:
                logger.info("Ensemble of permeability and porosity saved")
                logger.info(f"Current best mean cost = {best_cost_mean}")
                logger.info(f"Current iteration mean cost = {tinuke}")
                logger.info(f"Current best MAP cost = {best_cost_best}")
                logger.info(f"Current iteration MAP cost = {tinukebest}")
            best_cost_mean = tinuke
            best_cost_best = tinukebest
            if "PERM" in input_variables:
                use_k = ensemble
            if "PORO" in input_variables:
                use_p = ensemblep
            if "FAULT" in input_variables:
                use_f = ensemblef
        else:
            if dist.rank == 0:
                logger.info("Ensemble of permeability and porosity NOT saved")
                logger.info(f"Current best mean cost = {best_cost_mean}")
                logger.info(f"Current iteration mean cost = {tinuke}")
                logger.info(f"Current best MAP cost = {best_cost_best}")
                logger.info(f"Current iteration MAP cost = {tinukebest}")
            if "PERM" in input_variables:
                use_k = ini_K
            if "PORO" in input_variables:
                use_p = ini_p
            if "FAULT" in input_variables:
                use_f = ini_f
        mean_cost.append(tinuke)
        best_cost.append(tinukebest)
        if "PERM" in input_variables:
            ensemble_bestK.append(ini_K[:, muv].reshape(-1, 1))
            ensemble_meanK.append(np.reshape(np.mean(ini_K, axis=1), (-1, 1), "F"))
        if "PORO" in input_variables:
            ensemble_bestP.append(ini_p[:, muv].reshape(-1, 1))
            ensemble_meanP.append(np.reshape(np.mean(ini_p, axis=1), (-1, 1), "F"))
        if "FAULT" in input_variables:
            ensemble_bestf.append(ini_f[:, muv].reshape(-1, 1))
            ensemble_meanf.append(np.reshape(np.mean(ini_f, axis=1), (-1, 1), "F"))

        if Do_parametrisation == "No":
            if "PERM" in input_variables:
                ensemble = updated_ensemble["PERM"]
            if "PORO" in input_variables:
                ensemblep = updated_ensemble["PORO"]
        else:
            if "PERM" in input_variables:
                if Do_param_method == "DCT":
                    ensemble = idct22(
                        updated_ensemble["PERM"], Ne, nx, ny, nz, size1, size2
                    )
                else:
                    ensemble = decode_values(
                        updated_ensemble["PERM"], device, nx, ny, nz, model_perm
                    )
            if "PORO" in input_variables:
                if Do_param_method == "DCT":
                    ensemblep = idct22(
                        updated_ensemble["PORO"], Ne, nx, ny, nz, size1, size2
                    )
                else:
                    ensemblep = decode_values(
                        updated_ensemble["PORO"], device, nx, ny, nz, model_poro
                    )
        if "FAULT" in input_variables:
            ensemblef = updated_ensemble["FAULT"]
        outt = {}
        outt["PERM"] = ensemble
        outt["PORO"] = ensemblep
        outt = honour2(outt, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P, effec)
        ensemble = outt["PERM"]
        ensemblep = outt["PORO"]
        if "FAULT" in input_variables:
            ensemblef = np.clip(ensemblef, 0, 1)
        if iteration_converged > 1:
            if dist.rank == 0:
                logger.info("Converged")
            break
        else:
            pass
        if iteration_count == Termm - 1:
            if dist.rank == 0:
                logger.info(
                    "****************************************************************"
                )
                logger.info(
                    "         Did not converge, Maximum Iteration reached            "
                )
                logger.info(
                    "****************************************************************"
                )
            break
        else:
            pass
        iteration_count = iteration_count + 1
        iteration_converged = iteration_converged + chok
    mean_cost.append(tinuke)
    best_cost.append(tinukebest)
    if dist.rank == 0:
        plt.figure(figsize=(10, 10))
        alpha_big = np.array(alpha_big)
        alpha_big_inverted = 1 / alpha_big
        sum_inverted = np.sum(alpha_big_inverted)
        plt.plot(alpha_big, marker="o", linestyle="-")
        plt.xlabel("Iteration")
        plt.ylabel(r"$\alpha$ Value")
        plt.title(r"$\alpha$ Values evolution")
        plt.text(
            0.1 * len(alpha_big),
            max(alpha_big) * 1,  # Increased y value
            rf"Sum inverted $\alpha$ Values: {sum_inverted:.2f}",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8),
        )
        plt.grid(True)
        plt.savefig("../RESULTS/HM_RESULTS/alpha.png")
        plt.clf()
        plt.close()
    meancost1 = np.vstack(mean_cost)
    chm = np.argmin(meancost1)

    return (
        use_k,
        use_p,
        use_f,
        mean_cost,
        best_cost,
        ensemble_bestK,
        ensemble_meanK,
        ensemble_bestP,
        ensemble_meanP,
        ensemble_bestf,
        ensemble_meanf,
        iteration_count,
        iteration_converged,
        alpha_big,
        ensemble,
        ensemblep,
        ensemblef,
        chm,
        cc_ini,
        ensemble_dict,
        base_k,
        base_p,
        base_f,
    )
