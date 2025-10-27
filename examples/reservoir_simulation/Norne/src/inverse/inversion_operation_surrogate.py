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
                    INVERSE SURROGATE OPERATIONS MODULE
=====================================================================

This module provides surrogate model operations for inverse problems
in reservoir simulation. It includes neural network models, ensemble
operations, and data processing utilities.

Key Features:
- Forward model ensemble operations
- Neural network surrogate models
- Data processing and conversion utilities
- Ensemble Kalman filter operations

@Author : Clement Etienam
"""

# 🛠 Standard Library
import os
import pickle
import logging
import warnings
from collections import OrderedDict
# Removed unused imports

# 🔧 Third-party Libraries
import numpy as np
import numpy.matlib
import pandas as pd
import scipy.io as sio
import torch
import xgboost as xgb
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from hydra.utils import to_absolute_path
from joblib import Parallel, delayed

# 📦 Local Modules
# Removed unused imports from inverse.inversion_operation_ensemble
from compare.batch.misc_forward import Split_Matrix
from compare.batch.misc_forward_enact import (
    Make_correct,
    convert_back,
    fit_operation,
)
from gpytorch.likelihoods import GaussianLikelihood
from shutil import rmtree
# Removed unused imports from inverse.inversion_operation_gather
# Removed unused imports from inverse.inversion_operation_misc


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger("Inverse problem")
    if not logger.handlers:
        f_handler = logging.FileHandler(filename="read_vectors.log", mode="w")
        formatter = logging.Formatter(" %(asctime)s - %(levelname)s - %(message)s")
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)
        logger.setLevel(logging.INFO)
    warnings.filterwarnings("ignore")
    return logger


logger = setup_logging()


def process_data(data):
    well_indices = {}
    for entry in data:
        if entry[0] not in well_indices:
            well_indices[entry[0]] = []
        well_indices[entry[0]].append(
            (int(entry[1]) - 1, int(entry[2]) - 1, int(entry[3]) - 1, int(entry[4]) - 1)
        )
    return well_indices


def get_dyna(steppi, well_indices, swatuse):
    mean_big_all = []
    for xx in range(steppi):
        mean_big = []  # Collects mean values for this particular timestep
        for idx, list1 in well_indices.items():  # Direct access to lists via .items()
            temp_perm_values = [
                swatuse[xx, i_idx, j_idx, k_idx]
                if k_idx == l_idx
                else swatuse[xx, i_idx, j_idx, k_idx : l_idx + 1]
                for i_idx, j_idx, k_idx, l_idx in list1
            ]
            mean_all = np.mean(temp_perm_values)
            mean_big.append(mean_all)
        mean_big_all.append(mean_big)
    outt2 = np.array(mean_big_all)
    return outt2


def get_dyna2(
    steppi, well_indices, well_indicesg, well_indiceso, swatuse, gasuse, oiluse, Q, Qg
):
    unique_well_names = OrderedDict()
    for idx, tuple_entry in enumerate(well_indices):
        well_name = tuple_entry[0]
        if well_name not in unique_well_names:
            unique_well_names[well_name] = len(unique_well_names)
    well_name_to_index = {name: index for index, name in enumerate(unique_well_names)}
    for xx in range(steppi):
        for well_name, q_idx in well_name_to_index.items():
            entries_for_well = [t for t in well_indices if t[0] == well_name]
            total_value = Q[xx, q_idx]
            average_value = (
                total_value / len(entries_for_well) if entries_for_well else 0
            )
            for _, i_idx, j_idx, k_idx, l_idx in entries_for_well:
                if int(k_idx) - 1 == int(l_idx) - 1:
                    swatuse[xx, int(i_idx) - 1, int(j_idx) - 1, int(k_idx) - 1] = (
                        average_value
                    )
                else:
                    swatuse[
                        xx,
                        int(i_idx) - 1,
                        int(j_idx) - 1,
                        int(k_idx) - 1 : int(l_idx) - 1 + 1,
                    ] = average_value
    unique_well_namesg = OrderedDict()
    for idx, tuple_entry in enumerate(well_indicesg):
        well_nameg = tuple_entry[0]
        if well_nameg not in unique_well_namesg:
            unique_well_namesg[well_nameg] = len(unique_well_namesg)
    well_name_to_indexg = {name: index for index, name in enumerate(unique_well_namesg)}
    for xx in range(steppi):
        for well_nameg, q_idxg in well_name_to_indexg.items():
            entries_for_wellg = [t for t in well_indicesg if t[0] == well_nameg]
            total_valueg = Q[xx, q_idxg]
            average_valueg = (
                total_valueg / len(entries_for_wellg) if entries_for_wellg else 0
            )
            for _, i_idx, j_idx, k_idx, l_idx in entries_for_wellg:
                if int(k_idx) - 1 == int(l_idx) - 1:
                    gasuse[xx, int(i_idx) - 1, int(j_idx) - 1, int(k_idx) - 1] = (
                        average_valueg
                    )
                else:
                    gasuse[
                        xx,
                        int(i_idx) - 1,
                        int(j_idx) - 1,
                        int(k_idx) - 1 : int(l_idx) - 1 + 1,
                    ] = average_valueg
    unique_well_nameso = OrderedDict()
    for idx, tuple_entry in enumerate(well_indiceso):
        well_nameo = tuple_entry[0]
        if well_nameo not in unique_well_nameso:
            unique_well_nameso[well_nameo] = len(unique_well_nameso)
    well_name_to_indexo = {name: index for index, name in enumerate(unique_well_nameso)}
    for xx in range(steppi):
        for well_nameo, q_idxo in well_name_to_indexo.items():
            entries_for_wello = [t for t in well_indiceso if t[0] == well_nameo]
            for _, i_idx, j_idx, k_idx, l_idx in entries_for_wello:
                if int(k_idx) - 1 == int(l_idx) - 1:
                    oiluse[xx, int(i_idx) - 1, int(j_idx) - 1, int(k_idx) - 1] = -1
                else:
                    oiluse[
                        xx,
                        int(i_idx) - 1,
                        int(j_idx) - 1,
                        int(k_idx) - 1 : int(l_idx) - 1 + 1,
                    ] = -1
    return swatuse, gasuse, oiluse


def read_compdats(filename, well_names):
    with open(filename, "r") as file:
        start_collecting = False
        data = []  # List to collect all entries
        for line in file:
            stripped_line = line.strip()
            if stripped_line.startswith("--"):
                continue
            if "COMPDAT" in stripped_line:
                start_collecting = True
                continue
            if start_collecting and stripped_line.startswith("/"):
                start_collecting = False
                continue
            if start_collecting and stripped_line:
                parts = stripped_line.split()
                well_name = parts[0].strip("'")
                if well_name in well_names:
                    data.append((well_name, parts[1], parts[2], parts[3], parts[4]))
    return data


def read_compdats2(filename, file_path):
    with open(filename, "r") as file:
        data_gas = []  # List to collect gas entries
        data_water = []  # List to collect water entries
        data_oil = []  # List to collect oil entries
        injector_gas = set()  # Set to collect gas injector well names
        injector_water = set()  # Set to collect water injector well names
        producer_oil = set()
        start_collecting_welspecs = False
        start_collecting_wconinje = False
        start_collecting_wconhist = False
        for line in file:
            stripped_line = line.strip()
            if stripped_line.startswith("--"):
                continue
            if "WELSPECS" in stripped_line:
                start_collecting_welspecs = True
                continue
            if start_collecting_welspecs and stripped_line.startswith("/"):
                start_collecting_welspecs = False
                continue
            if "WCONINJE" in stripped_line:
                start_collecting_wconinje = True
                continue
            if start_collecting_wconinje and stripped_line.startswith("/"):
                start_collecting_wconinje = False
                continue
            if "WCONHIST" in stripped_line:
                start_collecting_wconhist = True
                continue
            if start_collecting_wconhist and stripped_line.startswith("/"):
                start_collecting_wconhist = False
                continue
            if start_collecting_welspecs:
                parts = stripped_line.split()
                if (
                    len(parts) > 5
                ):  # Ensure the line has enough parts to avoid index errors
                    well_name = parts[0].strip("'")
                    i = parts[2]
                    j = parts[3]
                    if parts[5].strip("'") == "GAS":
                        data_gas.append((well_name, i, j))
                    elif parts[5].strip("'") == "WATER":
                        data_water.append((well_name, i, j))
                    elif parts[5].strip("'") == "OIL":
                        data_oil.append((well_name, i, j))
            if start_collecting_wconinje:
                parts = stripped_line.split()
                if (
                    len(parts) > 3
                ):  # Ensure the line has enough parts to avoid index errors
                    well_name = parts[0].strip("'")
                    fluid_type = parts[1].strip("'")
                    if fluid_type == "GAS":
                        injector_gas.add(well_name)

                    elif fluid_type == "WATER":
                        injector_water.add(well_name)
            if start_collecting_wconhist:
                parts = stripped_line.split()
                if (
                    len(parts) > 3
                ):  # Ensure the line has enough parts to avoid index errors
                    well_name = parts[0].strip("'")
                    producer_oil.add(well_name)
    data = convert_to_list(process_data2(data_oil))
    data.sort(key=lambda x: x[2])
    with open(file_path, "r") as file:
        lines = file.readlines()
    well_namesoil = set()
    capture = False
    for line in lines:
        line = line.strip()
        if line == "WOPR":
            capture = True
            continue
        if capture:
            if line == "/":
                break
            well_name = line.strip(" '")
            well_namesoil.add(well_name)
    gass, water, oil = extract_tuples(injector_gas, injector_water, well_namesoil, data)
    return gass, oil, water


def convert_to_list(well_data):
    output_list = []
    for well_name, indices in well_data.items():
        for i, j in indices:
            output_list.append((i, j, well_name))
    return output_list


def extract_tuples(set1, set2, set3, tuples_list):
    extracted_set1 = [tup for tup in tuples_list if tup[2] in set1]
    extracted_set1.sort(key=lambda x: x[2])
    extracted_set2 = [tup for tup in tuples_list if tup[2] in set2]
    extracted_set2.sort(key=lambda x: x[2])
    combined_set = list(set1) + list(set2)
    extracted_set3 = [tup for tup in tuples_list if tup[2] in set3]
    extracted_set3.sort(key=lambda x: x[2])
    final_remaining_list = [tup for tup in extracted_set3 if tup[2] not in combined_set]
    final_remaining_list.sort(key=lambda x: x[2])
    return extracted_set1, extracted_set2, final_remaining_list


def process_data2(data):
    well_indices = {}
    for entry in data:
        well_name = entry[0]
        if well_name not in well_indices:
            well_indices[well_name] = []
        i_index = int(entry[1]) - 1  # Convert to zero-based index
        j_index = int(entry[2]) - 1  # Convert to zero-based index
        well_indices[well_name].append((i_index, j_index))
    return well_indices


def remove_rows(matrix, indices_to_remove):
    matrix = np.delete(matrix, indices_to_remove, axis=0)
    return matrix


def Forward_model_ensemble(
    N,
    x_true,
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
    producer_wells,
    unique_entries,
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
):
    #### ===================================================================== ####
    #                     RESERVOIR SIMULATOR WITH MODULUS
    #
    #### ===================================================================== ####
    modelPe = models["peacemann"]
    if "PRESSURE" in output_variables:
        modelP = models["pressure"]
        pressure = torch.zeros(N, steppi, nz, nx, ny).to(device, torch.float32)
    if "SWAT" in output_variables:
        output_keys_saturation = []
        modelS = models["saturation"]
        output_keys_saturation.append("water_sat")
        Swater = torch.zeros(N, steppi, nz, nx, ny).to(device, torch.float32)
    if "SOIL" in output_variables:
        output_keys_oil = []
        modelO = models["oil"]
        output_keys_oil.append("oil_sat")
        Soil = torch.zeros(N, steppi, nz, nx, ny).to(device, torch.float32)
    if "SGAS" in output_variables:
        output_keys_gas = []
        output_keys_gas.append("gas_sat")
        modelG = models["gas"]
        Sgas = torch.zeros(N, steppi, nz, nx, ny).to(device, torch.float32)

    for mv in range(N):
        temp = {
            "perm": x_true["perm"][mv, :, :, :, :][None, :, :, :, :],
            "poro": x_true["poro"][mv, :, :, :, :][None, :, :, :, :],
            "fault": x_true["fault"][mv, :, :, :, :][None, :, :, :, :],
            "pini": x_true["pini"][mv, :, :, :, :][None, :, :, :, :],
            "sini": x_true["sini"][mv, :, :, :, :][None, :, :, :, :],
        }

        with torch.no_grad():
            tensors = [
                value for value in temp.values() if isinstance(value, torch.Tensor)
            ]
            if not tensors:
                raise ValueError("🚨 No valid input tensors found for the model!")

            input_tensor = torch.cat(tensors, dim=1)
            nz_current = input_tensor.shape[2]

            # Setup chunking - only if nz > 30
            if nz_current > cfg.custom.allowable_size:
                chunk_size = max(1, int(nz_current * 0.1))
                num_chunks = (nz_current + chunk_size - 1) // chunk_size
            else:
                chunk_size = nz_current
                num_chunks = 1

            # Process in chunks
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, nz_current)
                current_chunk_size = end_idx - start_idx

                # Extract chunk
                input_temp = input_tensor[:, :, start_idx:end_idx, :, :]

                # Pad if needed (only when chunking)
                if nz_current > cfg.custom.allowable_size:
                    pad_size = chunk_size - current_chunk_size
                    if pad_size > 0:
                        input_temp = torch.nn.functional.pad(
                            input_temp, (0, 0, 0, 0, 0, pad_size)
                        )
                else:
                    pad_size = 0

                # Model predictions
                if "PRESSURE" in output_variables and modelP is not None:
                    ouut_p1 = modelP(input_temp)
                if "SGAS" in output_variables and modelG is not None:
                    ouut_sg1 = modelG(input_temp)
                if "SWAT" in output_variables and modelS is not None:
                    ouut_s1 = modelS(input_temp)
                if "SOIL" in output_variables and modelO is not None:
                    ouut_so1 = modelO(input_temp)

                # Remove padding if applied
                if nz_current > 30 and pad_size > 0:
                    if "PRESSURE" in output_variables and modelP is not None:
                        ouut_p1 = ouut_p1[:, :, :current_chunk_size, :, :]
                    if "SGAS" in output_variables and modelG is not None:
                        ouut_sg1 = ouut_sg1[:, :, :current_chunk_size, :, :]
                    if "SWAT" in output_variables and modelS is not None:
                        ouut_s1 = ouut_s1[:, :, :current_chunk_size, :, :]
                    if "SOIL" in output_variables and modelO is not None:
                        ouut_so1 = ouut_so1[:, :, :current_chunk_size, :, :]

                # Store results directly
                if "PRESSURE" in output_variables and modelP is not None:
                    pressure[mv, :, start_idx:end_idx, :, :] = ouut_p1
                if "SWAT" in output_variables and modelS is not None:
                    Swater[mv, :, start_idx:end_idx, :, :] = ouut_s1
                if "SOIL" in output_variables and modelO is not None:
                    Soil[mv, :, start_idx:end_idx, :, :] = ouut_so1
                if "SGAS" in output_variables and modelG is not None:
                    Sgas[mv, :, start_idx:end_idx, :, :] = ouut_sg1

                # Clean up chunk variables
                for var in ["ouut_p1", "ouut_s1", "ouut_sg1", "ouut_so1"]:
                    if var in locals():
                        del locals()[var]

            # Clean up after each sample
            del temp, input_tensor
            if (
                torch.cuda.is_available()
                and torch.cuda.memory_reserved()
                > 0.9 * torch.cuda.max_memory_allocated()
            ):
                torch.cuda.empty_cache()

    # Convert tensors to NumPy arrays and process
    if "PRESSURE" in output_variables:
        pressure_np = pressure.detach().cpu().numpy()
        pressure_np = Make_correct(pressure_np)
        pressure_np = convert_back(pressure_np, target_min, target_max, minP, maxP)
        pressure_np = np.clip(pressure_np, a_min=0, a_max=None)

    if "SWAT" in output_variables:
        Swater_np = Swater.detach().cpu().numpy()
        Swater_np = Make_correct(Swater_np)
        Swater_np = np.clip(Swater_np, 0, 1)

    if "SGAS" in output_variables:
        Sgas_np = Sgas.detach().cpu().numpy()
        Sgas_np = Make_correct(Sgas_np)
        Sgas_np = np.clip(Sgas_np, 0, 1)

    if "SOIL" in output_variables:
        Soil_np = Soil.detach().cpu().numpy()
        Soil_np = Make_correct(Soil_np)
        Soil_np = np.clip(Soil_np, 0, 1)

    perm = convert_back(
        x_true["perm"].detach().cpu().numpy(), target_min, target_max, minK, maxK
    )
    perm = Make_correct(perm)

    effective_abi = effective_abi[None, None, :, :, :]
    resultss = {}

    if "PRESSURE" in output_variables:
        resultss["PRESSURE"] = pressure_np * effective_abi
    if "SWAT" in output_variables:
        resultss["SWAT"] = Swater_np * effective_abi
    if "SOIL" in output_variables:
        resultss["SOIL"] = Soil_np * effective_abi
    if "SGAS" in output_variables:
        resultss["SGAS"] = Sgas_np * effective_abi
    if Trainmoe == "FNO":
        innn = np.zeros((N, (N_pr * 4) + 2, steppi))
    else:
        innn = np.zeros((N, steppi, (N_pr * 4) + 2))

    well_indices = process_data(unique_entries)
    for i in range(N):
        permuse = perm[i, 0, :, :, :]
        mean_big = []
        for idx, indices_list in well_indices.items():
            values = [
                permuse[i_idx, j_idx, k_idx]
                if k_idx == l_idx
                else permuse[i_idx, j_idx, k_idx : l_idx + 1]
                for i_idx, j_idx, k_idx, l_idx in indices_list
            ]
            mean_big.append(np.mean(values))
        permxx = np.tile(mean_big, (steppi, 1))
        presure_use = pressure_np[i, :, :, :, :]
        gas_use = Sgas_np[i, :, :, :, :]
        water_use = Swater_np[i, :, :, :, :]
        oil_use = Soil_np[i, :, :, :, :]
        Time_usee = Time[i, :, :, :, :]
        a3 = get_dyna(steppi, well_indices, water_use)
        a2 = get_dyna(steppi, well_indices, gas_use)
        a5 = get_dyna(steppi, well_indices, oil_use)
        a1 = np.zeros((steppi, 1))
        a4 = np.zeros((steppi, 1))
        for k in range(steppi):
            uniep = presure_use[k, :, :, :]
            permuse = uniep
            a1[k, 0] = np.mean(permuse)
            unietime = Time_usee[k, :, :, :]
            permuse = unietime
            a4[k, 0] = permuse[0, 0, 0]
        inn1 = np.hstack((permxx, a1, a5, a2, a3, a4))
        if Trainmoe == "FNO":
            inn1 = fit_operation(inn1, target_min, target_max, min_inn_fcn, max_inn_fcn)
            innn[i, :, :] = inn1.T
        else:
            inn1 = convert_backsin(inn1, max_inn_fcn2, N_pr)
            innn[i, :, :] = inn1
    if Trainmoe == "FNO":
        innn = torch.from_numpy(innn).to(device, torch.float32)
        ouut_p = []
        for mv in range(N):
            temp = innn[mv, :, :][None, :, :]
            with torch.no_grad():
                ouut_p1 = modelPe(temp)
                ouut_p1 = ouut_p1.detach().cpu().numpy() * max_out_fcn
            ouut_p.append(ouut_p1)
            del temp
            torch.cuda.empty_cache()
        ouut_p = np.vstack(ouut_p)
        ouut_p = np.transpose(ouut_p, (0, 2, 1))
        ouut_p[ouut_p <= 0] = 0
    else:
        useq = lenwels * N_pr
        innn = np.vstack(innn)
        cluster_all = sio.loadmat(
            to_absolute_path("../ML_MACHINE/clustersizescost.mat")
        )["cluster"]
        cluster_all = np.reshape(cluster_all, (-1, 1), "F")

        mves = Parallel(n_jobs=num_cores, backend="loky")(
            delayed(PREDICTION_CCR__MACHINE)(
                ib,
                int(cluster_all[ib, :]),
                innn,
                innn.shape[1],
                to_absolute_path("../ML_MACHINE"),
                oldfolder,
                pred_type,
                degg,
                experts,
                device,
            )
            for ib in range(useq)
        )

        ouut_p = np.array(Split_Matrix(np.hstack(mves), N))
        ouut_p = convert_backs(ouut_p, max_out_fcn2, N_pr, lenwels)
        ouut_p[ouut_p <= 0] = 0
    sim = []
    for zz in range(ouut_p.shape[0]):
        jesuni = []
        for k in range(lenwels):
            quantt = quant_big[f"K_{k}"]
            if quantt["boolean"] == 1:
                kodsval = (ouut_p[zz, :, k * N_pr : (k + 1) * N_pr]) / quantt["scale"]
            else:
                kodsval = (ouut_p[zz, :, k * N_pr : (k + 1) * N_pr]) * quantt["scale"]
            jesuni.append(kodsval)
        spit = np.hstack(jesuni)
        spit = np.reshape(spit, (-1, 1), "F")
        spit = remove_rows(spit, rows_to_remove).reshape(-1, 1)
        use = np.reshape(spit, (-1, 1), "F")
        sim.append(use)
    sim = np.hstack(sim)
    resultss["sim"] = sim
    resultss["ouut_p"] = ouut_p
    return resultss


def convert_backs(rescaled_tensor, max_val, N_pr, lenwels):
    C = []
    for k in range(lenwels):
        # rescaled_tensorr = rescaled_tensor[:, :, k*10: (k+1)*10]*max_val[:, k]
        rescaled_tensorr = (
            rescaled_tensor[:, :, k * N_pr : (k + 1) * N_pr] * max_val[:, k]
        )
        C.append(rescaled_tensorr)
    get_it2 = np.concatenate(C, axis=-1)
    return get_it2


def convert_backsin(rescaled_tensor, max_val, N_pr):
    C = []
    Anow = rescaled_tensor[:, :N_pr]
    max_vall = max_val[:, 0]
    rescaled_tensorr = Anow / max_vall
    C.append(rescaled_tensorr)
    Anow = rescaled_tensor[:, N_pr : N_pr + 1]
    max_vall = max_val[:, 1]
    rescaled_tensorr = Anow / max_vall
    C.append(rescaled_tensorr)
    Anow = rescaled_tensor[:, N_pr + 1 : 2 * N_pr + 1]
    max_vall = max_val[:, 2]
    rescaled_tensorr = Anow / max_vall
    C.append(rescaled_tensorr)
    Anow = rescaled_tensor[:, 2 * N_pr + 1 : 3 * N_pr + 1]
    max_vall = max_val[:, 3]
    rescaled_tensorr = Anow / max_vall
    C.append(rescaled_tensorr)
    Anow = rescaled_tensor[:, 3 * N_pr + 1 : 4 * N_pr + 1]
    max_vall = max_val[:, 4]
    rescaled_tensorr = Anow / max_vall
    C.append(rescaled_tensorr)
    Anow = rescaled_tensor[:, 4 * N_pr + 1 : 4 * N_pr + 2]
    max_vall = max_val[:, 5]
    rescaled_tensorr = Anow / max_vall
    C.append(rescaled_tensorr)
    get_it2 = np.concatenate(C, axis=-1)
    return get_it2


def cov(G):
    return torch.cov(G)


def KalmanGain(G, params, Gamma, N, alpha):
    CnGG = cov(G)
    mean_params = torch.mean(params, dim=1, keepdim=True)
    mean_G = torch.mean(G, dim=1, keepdim=True)
    Cyd = (params - mean_params) @ ((G - mean_G).T)
    U, s, Vh = torch.linalg.svd(CnGG + (alpha * Gamma))
    s_inv = 1.0 / s
    s_inv[s_inv < 1e-15] = 0
    inv_denominator = Vh.T @ (s_inv[:, None] * U.T)
    K = (1 / (N - 1)) * Cyd @ inv_denominator
    return K


def predict_machine(a0, model):
    ynew = model.predict(xgb.DMatrix(a0))
    return ynew


def predict_machine3(a0, deg, model, poly):
    predicted = model.predict(poly.fit_transform(a0))
    return predicted


class SparseGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing_points):
        super(SparseGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        self.variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            self.variational_distribution,
            learn_inducing_locations=True,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def PREDICTION_CCR__MACHINE(
    ii,
    nclusters,
    inputtest,
    numcols,
    training_master,
    oldfolder,
    pred_type,
    deg,
    experts,
    device,
):
    filenamex = "clfx_%d.asv" % ii
    filenamey = "clfy_%d.asv" % ii
    os.chdir(training_master)
    if experts == 1 or experts == 3:
        filename1 = "Classifier_%d.bin" % ii
        loaded_model = xgb.Booster({"nthread": 4})  # init model
        loaded_model.load_model(filename1)  # load data
    if experts == 2:
        filename1 = "Classifier_%d.pkl" % ii
        try:
            with open(filename1, "rb") as file:
                loaded_model = pickle.load(file)
        except (pickle.PickleError, EOFError, FileNotFoundError) as e:
            logger.error(f"Error loading model pickle file: {e}")
            raise
    try:
        clfx = pickle.load(open(filenamex, "rb"))
        clfy = pickle.load(open(filenamey, "rb"))
    except (pickle.PickleError, EOFError, FileNotFoundError) as e:
        logger.error(f"Error loading classifier pickle files: {e}")
        raise
    os.chdir(oldfolder)
    inputtest = clfx.transform(inputtest)
    if experts == 2:
        labelDA = loaded_model.predict(inputtest)
    else:
        labelDA = loaded_model.predict(xgb.DMatrix(inputtest))
        if nclusters == 2:
            labelDAX = 1 - labelDA
            labelDA = np.reshape(labelDA, (-1, 1))
            labelDAX = np.reshape(labelDAX, (-1, 1))
            labelDA = np.concatenate((labelDAX, labelDA), axis=1)
            labelDA = np.argmax(labelDA, axis=-1)
        else:
            labelDA = np.argmax(labelDA, axis=-1)
        labelDA = np.reshape(labelDA, (-1, 1), "F")
    numrowstest = len(inputtest)
    operationanswer = np.zeros((numrowstest, 1))
    labelDA = np.reshape(labelDA, (-1, 1), "F")
    for i in range(nclusters):
        if experts == 1:  # Polynomial regressor experts
            filename2 = "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
            filename2b = "polfeat_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
            os.chdir(training_master)
            try:
                with open(filename2, "rb") as file:
                    model0 = pickle.load(file)
                with open(filename2b, "rb") as filex:
                    poly0 = pickle.load(filex)
            except (pickle.PickleError, EOFError, FileNotFoundError) as e:
                logger.error(f"Error loading regressor pickle files: {e}")
                raise
            os.chdir(oldfolder)
            labelDA0 = (np.asarray(np.where(labelDA == i))).T
            #    ##----------------------##------------------------##
            a00 = inputtest[labelDA0[:, 0], :]
            a00 = np.reshape(a00, (-1, numcols), "F")
            if a00.shape[0] != 0:
                operationanswer[labelDA0[:, 0], :] = np.reshape(
                    predict_machine3(a00, deg, model0, poly0), (-1, 1)
                )
        elif experts == 2:
            labelDA0 = (np.asarray(np.where(labelDA == i))).T
            #    ##----------------------##------------------------##
            a00 = inputtest[labelDA0[:, 0], :]
            a00 = np.reshape(a00, (-1, numcols), "F")
            a00 = torch.tensor(a00, dtype=torch.float32).to(device)
            filename2 = "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".pth"
            os.chdir(training_master)
            checkpoint = torch.load(filename2, map_location=device)
            num_inducing_points = checkpoint[
                "variational_strategy.inducing_points"
            ].shape[0]
            input_dim = checkpoint["variational_strategy.inducing_points"].shape[1]
            # output_dim = 1  # Assuming a single output per sample
            train_x = torch.zeros(a00.shape[0], input_dim).to(device)
            train_y = torch.zeros(a00.shape[0], 1).to(device)
            train_y = train_y.squeeze(-1)
            likelihood = GaussianLikelihood().to(device)
            inducing_points = torch.zeros(num_inducing_points, input_dim).to(device)
            model = SparseGPModel(train_x, train_y, likelihood, inducing_points).to(
                device
            )
            model.load_state_dict(checkpoint, strict=False)  # ✅ Pass strict=False here
            os.chdir(oldfolder)
            model = model.to(device)
            model.eval()
            batch_size = 1  # Adjust based on memory availability
            predictions = []
            if a00.shape[0] != 0:
                with torch.no_grad():
                    for i in range(0, a00.shape[0], batch_size):
                        batch = a00[i : i + batch_size]  # Take a batch of inputs
                        prediction = model(batch)  # Forward pass
                        pred = prediction.mean.detach().cpu().numpy()
                        predictions.append(pred)  # Store batch predictions
                operationanswer[labelDA0[:, 0], :] = np.vstack(predictions)
            del model
            torch.cuda.empty_cache()  # Free unused GPU memory
        else:  # XGBoost experts
            loaded_modelr = xgb.Booster({"nthread": 4})  # init model
            filename2 = "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".bin"
            os.chdir(training_master)
            loaded_modelr.load_model(filename2)  # load data
            os.chdir(oldfolder)
            labelDA0 = (np.asarray(np.where(labelDA == i))).T
            #    ##----------------------##------------------------##
            a00 = inputtest[labelDA0[:, 0], :]
            a00 = np.reshape(a00, (-1, numcols), "F")
            if a00.shape[0] != 0:
                operationanswer[labelDA0[:, 0], :] = np.reshape(
                    predict_machine(a00, loaded_modelr), (-1, 1)
                )
    operationanswer = clfy.inverse_transform(operationanswer)
    return operationanswer


def predict_machine11(a0, model):
    ynew = model.predict(xgb.DMatrix(a0))
    return ynew


def Remove_folder(N_ens, straa):
    for jj in range(N_ens):
        folderr = straa + str(jj)
        rmtree(folderr)


def historydata(timestep, steppi, steppi_indices, N_pr):
    file_path = "../simulator_data/Flow.xlsx"
    df = pd.read_excel(file_path, skiprows=1)
    data_array = df.to_numpy()[:10, 1:]  # Skips first column (assuming it's time)
    WOIL1 = data_array[:, :N_pr]
    WWATER1 = data_array[:, N_pr : 2 * N_pr]
    WGAS1 = data_array[:, 2 * N_pr : 3 * N_pr]
    DATA = {"OIL": WOIL1, "WATER": WWATER1, "GAS": WGAS1}
    oil = WOIL1.reshape(-1, 1, order="F")  # Column-major reshaping
    water = WWATER1.reshape(-1, 1, order="F")
    gas = WGAS1.reshape(-1, 1, order="F")
    DATA2 = np.vstack([oil, water, gas])  # History-matching data
    new = np.hstack([WOIL1, WWATER1, WGAS1])  # Original structure but stacked
    return DATA, DATA2, new


def historydatano(timestep, steppi, steppi_indices, N_pr):
    WOIL1 = np.zeros((steppi, N_pr))
    WWATER1 = np.zeros((steppi, N_pr))
    WGAS1 = np.zeros((steppi, N_pr))
    steppii = 246
    A2oilsim = pd.read_csv(
        "../simulator_data/FULLNORNE.RSM", skiprows=1545, sep="\s+", header=None
    )

    B_1BHoilsim = A2oilsim[5].values[:steppii]
    B_1Hoilsim = A2oilsim[6].values[:steppii]
    B_2Hoilsim = A2oilsim[7].values[:steppii]
    B_3Hoilsim = A2oilsim[8].values[:steppii]
    B_4BHoilsim = A2oilsim[9].values[:steppii]
    A22oilsim = pd.read_csv(
        "../simulator_data/FULLNORNE.RSM", skiprows=1801, sep="\s+", header=None
    )
    B_4DHoilsim = A22oilsim[1].values[:steppii]
    B_4Hoilsim = A22oilsim[2].values[:steppii]
    D_1CHoilsim = A22oilsim[3].values[:steppii]
    D_1Hoilsim = A22oilsim[4].values[:steppii]
    D_2Hoilsim = A22oilsim[5].values[:steppii]
    D_3AHoilsim = A22oilsim[6].values[:steppii]
    D_3BHoilsim = A22oilsim[7].values[:steppii]
    D_4AHoilsim = A22oilsim[8].values[:steppii]
    D_4Hoilsim = A22oilsim[9].values[:steppii]
    A222oilsim = pd.read_csv(
        "../simulator_data/FULLNORNE.RSM", skiprows=2057, sep="\s+", header=None
    )

    E_1Hoilsim = A222oilsim[1].values[:steppii]
    E_2AHoilsim = A222oilsim[2].values[:steppii]
    E_2Hoilsim = A222oilsim[3].values[:steppii]
    E_3AHoilsim = A222oilsim[4].values[:steppii]
    E_3CHoilsim = A222oilsim[5].values[:steppii]
    E_3Hoilsim = A222oilsim[6].values[:steppii]
    E_4AHoilsim = A222oilsim[7].values[:steppii]
    K_3Hoilsim = A222oilsim[8].values[:steppii]

    WOIL1[:, 0] = B_1BHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 1] = B_1Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 2] = B_2Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 3] = B_3Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 4] = B_4BHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 5] = B_4DHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 6] = B_4Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 7] = D_1CHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 8] = D_1Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 9] = D_2Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 10] = D_3AHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 11] = D_3BHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 12] = D_4AHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 13] = D_4Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 14] = E_1Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 15] = E_2AHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 16] = E_2Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 17] = E_3AHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 18] = E_3CHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 19] = E_3Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 20] = E_4AHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 21] = K_3Hoilsim.ravel()[steppi_indices - 1]
    # IMPORT FOR WATER
    A2watersim = pd.read_csv(
        "../simulator_data/FULLNORNE.RSM", skiprows=2313, sep="\s+", header=None
    )
    B_1BHwatersim = A2watersim[9].values[:steppii]

    A22watersim = pd.read_csv(
        "../simulator_data/FULLNORNE.RSM", skiprows=2569, sep="\s+", header=None
    )
    B_1Hwatersim = A22watersim[1].values[:steppii]
    B_2Hwatersim = A22watersim[2].values[:steppii]
    B_3Hwatersim = A22watersim[3].values[:steppii]
    B_4BHwatersim = A22watersim[4].values[:steppii]
    B_4DHwatersim = A22watersim[5].values[:steppii]
    B_4Hwatersim = A22watersim[6].values[:steppii]
    D_1CHwatersim = A22watersim[7].values[:steppii]
    D_1Hwatersim = A22watersim[8].values[:steppii]
    D_2Hwatersim = A22watersim[9].values[:steppii]

    A222watersim = pd.read_csv(
        "../simulator_data/FULLNORNE.RSM", skiprows=2825, sep="\s+", header=None
    )
    D_3AHwatersim = A222watersim[1].values[:steppii]
    D_3BHwatersim = A222watersim[2].values[:steppii]
    D_4AHwatersim = A222watersim[3].values[:steppii]
    D_4Hwatersim = A222watersim[4].values[:steppii]
    E_1Hwatersim = A222watersim[5].values[:steppii]
    E_2AHwatersim = A222watersim[6].values[:steppii]
    E_2Hwatersim = A222watersim[7].values[:steppii]
    E_3AHwatersim = A222watersim[8].values[:steppii]
    E_3CHwatersim = A222watersim[9].values[:steppii]

    A222watersim = pd.read_csv(
        "../simulator_data/FULLNORNE.RSM", skiprows=3081, sep="\s+", header=None
    )
    E_3Hwatersim = A222watersim[1].values[:steppii]
    E_4AHwatersim = A222watersim[2].values[:steppii]
    K_3Hwatersim = A222watersim[3].values[:steppii]

    WWATER1[:, 0] = B_1BHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 1] = B_1Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 2] = B_2Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 3] = B_3Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 4] = B_4BHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 5] = B_4DHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 6] = B_4Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 7] = D_1CHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 8] = D_1Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 9] = D_2Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 10] = D_3AHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 11] = D_3BHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 12] = D_4AHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 13] = D_4Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 14] = E_1Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 15] = E_2AHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 16] = E_2Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 17] = E_3AHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 18] = E_3CHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 19] = E_3Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 20] = E_4AHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 21] = K_3Hwatersim.ravel()[steppi_indices - 1]

    # GAS PRODUCTION RATE
    A2gassim = pd.read_csv(
        "../simulator_data/FULLNORNE.RSM", skiprows=1033, sep="\s+", header=None
    )
    B_1BHgassim = A2gassim[1].values[:steppii]
    B_1Hgassim = A2gassim[2].values[:steppii]
    B_2Hgassim = A2gassim[3].values[:steppii]
    B_3Hgassim = A2gassim[4].values[:steppii]
    B_4BHgassim = A2gassim[5].values[:steppii]
    B_4DHgassim = A2gassim[6].values[:steppii]
    B_4Hgassim = A2gassim[7].values[:steppii]
    D_1CHgassim = A2gassim[8].values[:steppii]
    D_1Hgassim = A2gassim[9].values[:steppii]

    A22gassim = pd.read_csv(
        "../simulator_data/FULLNORNE.RSM", skiprows=1289, sep="\s+", header=None
    )
    D_2Hgassim = A22gassim[1].values[:steppii]
    D_3AHgassim = A22gassim[2].values[:steppii]
    D_3BHgassim = A22gassim[3].values[:steppii]
    D_4AHgassim = A22gassim[4].values[:steppii]
    D_4Hgassim = A22gassim[5].values[:steppii]
    E_1Hgassim = A22gassim[6].values[:steppii]
    E_2AHgassim = A22gassim[7].values[:steppii]
    E_2Hgassim = A22gassim[8].values[:steppii]
    E_3AHgassim = A22gassim[9].values[:steppii]

    A222gassim = pd.read_csv(
        "../simulator_data/FULLNORNE.RSM", skiprows=1545, sep="\s+", header=None
    )
    E_3CHgassim = A222gassim[1].values[:steppii]
    E_3Hgassim = A222gassim[2].values[:steppii]
    E_4AHgassim = A222gassim[3].values[:steppii]
    K_3Hgassim = A222gassim[4].values[:steppii]

    WGAS1[:, 0] = B_1BHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 1] = B_1Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 2] = B_2Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 3] = B_3Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 4] = B_4BHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 5] = B_4DHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 6] = B_4Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 7] = D_1CHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 8] = D_1Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 9] = D_2Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 10] = D_3AHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 11] = D_3BHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 12] = D_4AHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 13] = D_4Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 14] = E_1Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 15] = E_2AHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 16] = E_2Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 17] = E_3AHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 18] = E_3CHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 19] = E_3Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 20] = E_4AHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 21] = K_3Hgassim.ravel()[steppi_indices - 1]

    DATA = {"OIL": WOIL1, "WATER": WWATER1, "GAS": WGAS1}

    oil = np.reshape(WOIL1, (-1, 1), "F")
    water = np.reshape(WWATER1, (-1, 1), "F")
    gas = np.reshape(WGAS1, (-1, 1), "F")

    # Get data for history matching
    DATA2 = np.vstack([oil, water, gas])
    new = np.hstack([WOIL1, WWATER1, WGAS1])
    return DATA, DATA2, new


def historydata2(timestep, steppi, steppi_indices):
    WOIL1 = np.zeros((steppi, 22))
    WWATER1 = np.zeros((steppi, 22))
    WGAS1 = np.zeros((steppi, 22))
    WWINJ1 = np.zeros((steppi, 9))
    WGASJ1 = np.zeros((steppi, 4))
    indices = timestep
    lines = []
    with open("../simulator_data/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 47873:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A1 = df[[2, 3, 4, 5, 6, 8]].values

    B_2H = A1[:, 0][indices - 1]
    D_1H = A1[:, 1][indices - 1]
    D_2H = A1[:, 2][indices - 1]
    B_4H = A1[:, 3][indices - 1]
    D_4H = A1[:, 4][indices - 1]
    E_3H = A1[:, 5][indices - 1]
    lines = []
    with open("../simulator_data/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 48743:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A2 = df[[1, 4, 5, 7, 9]].values

    B_1H = A2[:, 0][indices - 1]
    B_3H = A2[:, 1][indices - 1]
    E_1H = A2[:, 2][indices - 1]
    E_2H = A2[:, 3][indices - 1]
    E_4AH = A2[:, 4][indices - 1]

    # Open the file and read lines until '---' is found
    lines = []
    with open("../simulator_data/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 49613:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A3 = df[[2, 4, 7, 8, 9]].values
    D_3AH = A3[:, 0][indices - 1]
    E_3AH = A3[:, 1][indices - 1]
    B_4BH = A3[:, 2][indices - 1]
    D_4AH = A3[:, 3][indices - 1]
    D_1CH = A3[:, 4][indices - 1]
    lines = []
    with open("../simulator_data/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 50483:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A4 = df[[2, 4, 5, 6, 8, 9]].values

    B_4DH = A4[:, 0][indices - 1]
    E_3CH = A4[:, 1][indices - 1]
    E_2AH = A4[:, 2][indices - 1]
    D_3BH = A4[:, 3][indices - 1]
    B_1BH = A4[:, 4][indices - 1]
    K_3H = A4[:, 5][indices - 1]

    WOIL1[:, 0] = B_1BH.ravel()[steppi_indices - 1]
    WOIL1[:, 1] = B_1H.ravel()[steppi_indices - 1]
    WOIL1[:, 2] = B_2H.ravel()[steppi_indices - 1]
    WOIL1[:, 3] = B_3H.ravel()[steppi_indices - 1]
    WOIL1[:, 4] = B_4BH.ravel()[steppi_indices - 1]
    WOIL1[:, 5] = B_4DH.ravel()[steppi_indices - 1]
    WOIL1[:, 6] = B_4H.ravel()[steppi_indices - 1]
    WOIL1[:, 7] = D_1CH.ravel()[steppi_indices - 1]
    WOIL1[:, 8] = D_1H.ravel()[steppi_indices - 1]
    WOIL1[:, 9] = D_2H.ravel()[steppi_indices - 1]
    WOIL1[:, 10] = D_3AH.ravel()[steppi_indices - 1]
    WOIL1[:, 11] = D_3BH.ravel()[steppi_indices - 1]
    WOIL1[:, 12] = D_4AH.ravel()[steppi_indices - 1]
    WOIL1[:, 13] = D_4H.ravel()[steppi_indices - 1]
    WOIL1[:, 14] = E_1H.ravel()[steppi_indices - 1]
    WOIL1[:, 15] = E_2AH.ravel()[steppi_indices - 1]
    WOIL1[:, 16] = E_2H.ravel()[steppi_indices - 1]
    WOIL1[:, 17] = E_3AH.ravel()[steppi_indices - 1]
    WOIL1[:, 18] = E_3CH.ravel()[steppi_indices - 1]
    WOIL1[:, 19] = E_3H.ravel()[steppi_indices - 1]
    WOIL1[:, 20] = E_4AH.ravel()[steppi_indices - 1]
    WOIL1[:, 21] = K_3H.ravel()[steppi_indices - 1]
    lines = []
    with open("../simulator_data/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 40913:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A1w = df[[2, 3, 4, 5, 6, 8]].values

    B_2Hw = A1w[:, 0][indices - 1]
    D_1Hw = A1w[:, 1][indices - 1]
    D_2Hw = A1w[:, 2][indices - 1]
    B_4Hw = A1w[:, 3][indices - 1]
    D_4Hw = A1w[:, 4][indices - 1]
    E_3Hw = A1w[:, 5][indices - 1]
    lines = []
    with open("../simulator_data/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 41783:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A2w = df[[1, 4, 5, 7, 9]].values

    B_1Hw = A2w[:, 0][indices - 1]
    B_3Hw = A2w[:, 1][indices - 1]
    E_1Hw = A2w[:, 2][indices - 1]
    E_2Hw = A2w[:, 3][indices - 1]
    E_4AHw = A2w[:, 4][indices - 1]
    lines = []
    with open("../simulator_data/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 42653:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A3w = df[[2, 4, 7, 8, 9]].values

    D_3AHw = A3w[:, 0][indices - 1]
    E_3AHw = A3w[:, 1][indices - 1]
    B_4BHw = A3w[:, 2][indices - 1]
    D_4AHw = A3w[:, 3][indices - 1]
    D_1CHw = A3w[:, 4][indices - 1]
    lines = []
    with open("../simulator_data/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 43523:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A4w = df[[2, 4, 5, 6, 8, 9]].values

    B_4DHw = A4w[:, 0][indices - 1]
    E_3CHw = A4w[:, 1][indices - 1]
    E_2AHw = A4w[:, 2][indices - 1]
    D_3BHw = A4w[:, 3][indices - 1]
    B_1BHw = A4w[:, 4][indices - 1]
    K_3Hw = A4w[:, 5][indices - 1]

    WWATER1[:, 0] = B_1BHw.ravel()[steppi_indices - 1]
    WWATER1[:, 1] = B_1Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 2] = B_2Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 3] = B_3Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 4] = B_4BHw.ravel()[steppi_indices - 1]
    WWATER1[:, 5] = B_4DHw.ravel()[steppi_indices - 1]
    WWATER1[:, 6] = B_4Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 7] = D_1CHw.ravel()[steppi_indices - 1]
    WWATER1[:, 8] = D_1Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 9] = D_2Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 10] = D_3AHw.ravel()[steppi_indices - 1]
    WWATER1[:, 11] = D_3BHw.ravel()[steppi_indices - 1]
    WWATER1[:, 12] = D_4AHw.ravel()[steppi_indices - 1]
    WWATER1[:, 13] = D_4Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 14] = E_1Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 15] = E_2AHw.ravel()[steppi_indices - 1]
    WWATER1[:, 16] = E_2Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 17] = E_3AHw.ravel()[steppi_indices - 1]
    WWATER1[:, 18] = E_3CHw.ravel()[steppi_indices - 1]

    WWATER1[:, 19] = E_3Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 20] = E_4AHw.ravel()[steppi_indices - 1]
    WWATER1[:, 21] = K_3Hw.ravel()[steppi_indices - 1]
    lines = []
    with open("../simulator_data/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 54833:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A1g = df[[2, 3, 4, 5, 6, 8]].values

    B_2Hg = A1g[:, 0][indices - 1]
    D_1Hg = A1g[:, 1][indices - 1]
    D_2Hg = A1g[:, 2][indices - 1]
    B_4Hg = A1g[:, 3][indices - 1]
    D_4Hg = A1g[:, 4][indices - 1]
    E_3Hg = A1g[:, 5][indices - 1]
    lines = []
    with open("../simulator_data/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 55703:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A2g = df[[1, 4, 5, 7, 9]].values

    B_1Hg = A2g[:, 0][indices - 1]
    B_3Hg = A2g[:, 1][indices - 1]
    E_1Hg = A2g[:, 2][indices - 1]
    E_2Hg = A2g[:, 3][indices - 1]
    E_4AHg = A2g[:, 4][indices - 1]
    lines = []
    with open("../simulator_data/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 56573:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A3g = df[[2, 4, 7, 8, 9]].values

    D_3AHg = A3g[:, 0][indices - 1]
    E_3AHg = A3g[:, 1][indices - 1]
    B_4BHg = A3g[:, 2][indices - 1]
    D_4AHg = A3g[:, 3][indices - 1]
    D_1CHg = A3g[:, 4][indices - 1]
    lines = []
    with open("../simulator_data/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 57443:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A4g = df[[2, 4, 5, 6, 8, 9]].values
    B_4DHg = A4g[:, 0][indices - 1]
    E_3CHg = A4g[:, 1][indices - 1]
    E_2AHg = A4g[:, 2][indices - 1]
    D_3BHg = A4g[:, 3][indices - 1]
    B_1BHg = A4g[:, 4][indices - 1]
    K_3Hg = A4g[:, 5][indices - 1]

    WGAS1[:, 0] = B_1BHg.ravel()[steppi_indices - 1]
    WGAS1[:, 1] = B_1Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 2] = B_2Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 3] = B_3Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 4] = B_4BHg.ravel()[steppi_indices - 1]
    WGAS1[:, 5] = B_4DHg.ravel()[steppi_indices - 1]
    WGAS1[:, 6] = B_4Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 7] = D_1CHg.ravel()[steppi_indices - 1]
    WGAS1[:, 8] = D_1Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 9] = D_2Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 10] = D_3AHg.ravel()[steppi_indices - 1]
    WGAS1[:, 11] = D_3BHg.ravel()[steppi_indices - 1]
    WGAS1[:, 12] = D_4AHg.ravel()[steppi_indices - 1]
    WGAS1[:, 13] = D_4Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 14] = E_1Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 15] = E_2AHg.ravel()[steppi_indices - 1]
    WGAS1[:, 16] = E_2Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 17] = E_3AHg.ravel()[steppi_indices - 1]
    WGAS1[:, 18] = E_3CHg.ravel()[steppi_indices - 1]
    WGAS1[:, 19] = E_3Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 20] = E_4AHg.ravel()[steppi_indices - 1]
    WGAS1[:, 21] = K_3Hg.ravel()[steppi_indices - 1]
    lines = []
    with open("../simulator_data/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 72237:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A1win = df[[1, 2, 3, 4, 5, 6, 7, 8, 9]].values

    C_1Hwin = A1win[:, 0][indices - 1]
    C_2Hwin = A1win[:, 1][indices - 1]
    C_3Hwin = A1win[:, 2][indices - 1]
    C_4Hwin = A1win[:, 3][indices - 1]
    C_4AHwin = A1win[:, 4][indices - 1]
    F_1Hwin = A1win[:, 5][indices - 1]
    F_2Hwin = A1win[:, 6][indices - 1]
    F_3Hwin = A1win[:, 7][indices - 1]
    F_4Hwin = A1win[:, 8][indices - 1]

    WWINJ1[:, 0] = C_1Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 1] = C_2Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 2] = C_3Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 3] = C_4AHwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 4] = C_4Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 5] = F_1Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 6] = F_2Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 7] = F_3Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 8] = F_4Hwin.ravel()[steppi_indices - 1]
    lines = []
    with open("../simulator_data/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 73977:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)
    df = pd.DataFrame([line.split() for line in lines])
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A1gin = df[[1, 3, 4, 5]].values
    C_1Hgin = A1gin[:, 0][indices - 1]
    C_3Hgin = A1gin[:, 1][indices - 1]
    C_4Hgin = A1gin[:, 2][indices - 1]
    C_4AHgin = A1gin[:, 3][indices - 1]

    WGASJ1[:, 0] = C_1Hgin.ravel()[steppi_indices - 1]
    WGASJ1[:, 1] = C_3Hgin.ravel()[steppi_indices - 1]
    WGASJ1[:, 2] = C_4AHgin.ravel()[steppi_indices - 1]
    WGASJ1[:, 3] = C_4Hgin.ravel()[steppi_indices - 1]
    DATA = {
        "OIL": WOIL1,
        "WATER": WWATER1,
        "GAS": WGAS1,
        "WATER_INJ": WWINJ1,
        "WGAS_inj": WGASJ1,
    }
    oil = np.reshape(WOIL1, (-1, 1), "F")
    water = np.reshape(WWATER1, (-1, 1), "F")
    gas = np.reshape(WGAS1, (-1, 1), "F")
    DATA2 = np.vstack([oil, water, gas])
    new = np.hstack([WOIL1, WWATER1, WGAS1])
    return DATA, DATA2, new
