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

"""
=====================================================================
                            BINARIES EXTRACTION
=====================================================================

This module provides binary data extraction capabilities for reservoir
simulation forward modeling. It includes functions for processing binary
data files, extracting simulation results, and preparing data for
machine learning models.

Key Features:
- Binary data extraction and processing
- Black oil model calculations
- Data validation and quality checks
- Integration with simulation workflows

Usage:
    from forward.binaries_extract import (
        Black_oil2,
        process_and_print,
        extract_binary_data
    )

@Author : Clement Etienam
"""

import logging
import sys

import numpy as np
import numpy.linalg
import numpy.matlib
from scipy.interpolate import interp1d

import torch
import torch.optim as optim
import torch.nn as nn

from forward.simulator import (
    calc_mu_g,
    calc_rs,
    calc_bg,
    calc_bo,
    linear_interp,
    StoneIIModel,
    compute_peacemannoil,
)
from data_extract.opm_extract_rates import normalize_tensors_adjusted


def setup_logging() -> logging.Logger:
    """Configure and return the main logger with green INFO console output."""
    logger = logging.getLogger("forward problem")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers on re-entry

    formatter = logging.Formatter(" %(asctime)s - %(levelname)s - %(message)s")

    # File handler — plain, no colors (keeps log file clean)
    f_handler = logging.FileHandler(filename="read_vectors.log", mode="w")
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

    # Console handler — colored
    class _ColorFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG: "\033[0;36m",  # cyan
            logging.INFO: "\033[0;32m",  # green
            logging.WARNING: "\033[1;33m",  # yellow
            logging.ERROR: "\033[0;31m",  # red
            logging.CRITICAL: "\033[1;31m",  # bold red
        }
        RESET = "\033[0m"

        def format(self, record):
            msg = super().format(record)
            if sys.stdout.isatty():
                color = self.COLORS.get(record.levelno, "")
                return f"{color}{msg}{self.RESET}"
            return msg

    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(
        _ColorFormatter(" %(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(c_handler)

    return logger


def Black_oil_seq(
    input_var,
    neededM,
    SWI,
    SWR,
    UW,
    BW,
    UO,
    BO,
    nx,
    ny,
    nz,
    SWOW,
    SWOG,
    target_min,
    target_max,
    minK,
    maxK,
    minP,
    maxP,
    p_bub,
    p_atm,
    CFO,
    Relperm,
    params,
    pde_method,
    RE,
    max_inn_fcn,
    max_out_fcn,
    DZ,
    device,
    params1_swow,
    params2_swow,
    params1_swog,
    params2_swog,
    maxQw,
    maxQg,
    maxQ,
    maxT,
):
    """Compute black-oil PDE residuals using sequential (time-varying) rate inputs.

    Parameters
    ----------
    input_var : dict
        Dict of torch.Tensor fields: ``"pressure"``, ``"perm"``, ``"Q"``, ``"Qw"``,
        ``"dt"``, ``"pini"``, ``"poro"``, ``"sini"``, ``"sgini"``, ``"water_sat"``,
        ``"gas_sat"``, ``"Qg"``, ``"fault"``.
    neededM : dict
        Dict containing ``"actnum"`` active-cell mask as torch.Tensor.
    SWI : torch.Tensor
        Irreducible water saturation.
    SWR : torch.Tensor
        Residual water saturation.
    UW : float
        Water viscosity.
    BW : float
        Water formation volume factor.
    UO : float
        Oil viscosity.
    BO : float
        Oil formation volume factor (overwritten internally).
    nx : int
        Number of grid cells in the x direction.
    ny : int
        Number of grid cells in the y direction.
    nz : int
        Number of grid cells in the z direction.
    SWOW : torch.Tensor
        Water-oil relative permeability table of shape ``(n, 3)``.
    SWOG : torch.Tensor
        Gas-oil relative permeability table of shape ``(n, 3)``.
    target_min : float
        Normalisation target minimum (currently unused).
    target_max : float
        Normalisation target maximum (currently unused).
    minK : float
        Minimum permeability (currently unused).
    maxK : float or np.ndarray
        Permeability scale factor for denormalisation.
    minP : float
        Minimum pressure (currently unused).
    maxP : float or np.ndarray
        Pressure scale factor for denormalisation.
    p_bub : torch.Tensor
        Bubble-point pressure scalar.
    p_atm : torch.Tensor
        Atmospheric pressure scalar.
    CFO : float
        Oil compressibility above bubble point.
    Relperm : int
        Flag selecting relative-permeability model (1 = table, else Stone-II).
    params : dict
        Stone-II model parameters (used when ``Relperm != 1``).
    pde_method : int
        PDE discretisation method flag (currently unused; FDM always applied).
    RE : float
        Drainage radius (currently unused in this variant).
    max_inn_fcn : float
        Input normalisation scale (currently unused).
    max_out_fcn : float
        Output normalisation scale (currently unused).
    DZ : float
        Perforation thickness (currently unused in this variant).
    device : str
        PyTorch device string.
    params1_swow : tuple
        Cubic polynomial coefficients for ``KROW`` vs. water saturation.
    params2_swow : tuple
        Cubic polynomial coefficients for ``KRW`` vs. water saturation.
    params1_swog : tuple
        Cubic polynomial coefficients for ``KROG`` vs. gas saturation.
    params2_swog : tuple
        Cubic polynomial coefficients for ``KRG`` vs. gas saturation.
    maxQw : np.ndarray
        Scale factors for water injection rates.
    maxQg : np.ndarray
        Scale factors for gas injection rates.
    maxQ : np.ndarray
        Scale factors for total injection rates.
    maxT : np.ndarray
        Scale factors for time steps.

    Returns
    -------
    dict
        Normalised residual dictionary with keys ``"pressured"``, ``"saturationd"``,
        and ``"gasd"`` as torch.Tensor fields.
    """
    from forward.gradients_extract import (
        compute_gradient_3d,
        compute_second_order_gradient_3d,
    )

    maxQw = torch.from_numpy(maxQw).to(device)
    maxQg = torch.from_numpy(maxQg).to(device)
    maxQ = torch.from_numpy(maxQ).to(device)
    maxT = torch.from_numpy(maxT).to(device)
    u = input_var["pressure"]  # .to(torch.float32)
    perm = input_var["perm"]  # .to(torch.float32)
    fin = input_var["Q"].to(torch.float32).clamp(min=1e-6, max=1e6)
    fin[fin == 0.1] = 0
    fin = fin * maxQ
    finwater = input_var["Qw"].to(torch.float32).clamp(min=1e-6, max=1e6)
    finwater[finwater == 0.1] = 0
    finwater = finwater * maxQw
    dt = input_var["dt"]  # .to(torch.float32)
    dt = dt * maxT
    pini = input_var["pini"]  # .to(torch.float32)
    poro = input_var["poro"]  # .to(torch.float32)
    sini = input_var["sini"]  # .to(torch.float32)
    sgini = input_var["sgini"]  # .to(torch.float32)
    sat = input_var["water_sat"]  # .to(torch.float32)
    satg = input_var["gas_sat"]  # .to(torch.float32)
    fingas = input_var["Qg"] * maxQg
    fingas = fingas.repeat(u.shape[0], 1, 1, 1, 1)
    fingas = input_var["Qg"].to(torch.float32).clamp(min=1e-6, max=1e6)
    fingas[fingas == 0.1] = 0
    fingas = fingas * maxQg
    finoil = fin - (finwater + fingas)
    finoil = finoil.repeat(u.shape[0], 1, 1, 1, 1)
    actnum = (
        neededM["actnum"]
        .to(torch.float32)
        .repeat(u.shape[0], 1, 1, 1, 1)
        .clamp(min=1e-6)
    )
    finoil = finoil.clamp(max=1e6)
    actnum = neededM["actnum"]
    actnum = actnum.repeat(u.shape[0], 1, 1, 1, 1)
    actnum = actnum.clamp(min=1e-6)
    sato = 1 - (sat + satg)
    dxf = 1e-2
    fault = input_var["fault"].to(torch.float32)
    if isinstance(maxP, np.ndarray):
        maxP = torch.tensor(maxP, dtype=torch.float32, device=u.device)
    if isinstance(maxK, np.ndarray):
        maxK = torch.tensor(maxK, dtype=torch.float32, device=perm.device)
    u = u * maxP.to(u.dtype)
    pini = pini * maxP.to(pini.dtype)
    a = (perm * maxK.to(perm.dtype)).to(torch.float32)
    p_loss = torch.zeros_like(u, dtype=torch.float32, device=device)
    s_loss = torch.zeros_like(u, dtype=torch.float32, device=device)
    prior_pressure = torch.zeros(
        sat.shape[0], sat.shape[1], nz, nx, ny, dtype=torch.float32, device=device
    )
    prior_pressure = pini

    avg_p = prior_pressure.mean(dim=(2, 3, 4), keepdim=True).to(torch.float32)
    avg_p = replace_with_mean(avg_p)
    UG = calc_mu_g(avg_p).to(torch.float32)
    RS = calc_rs(p_bub, avg_p, device).to(torch.float32)
    BG = calc_bg(p_bub, p_atm, avg_p).to(torch.float32)
    BO = calc_bo(p_bub, p_atm, CFO, avg_p).to(torch.float32)
    avg_p = replace_with_mean(avg_p)
    UG = replace_with_mean(UG)
    BG = replace_with_mean(BG)
    RS = replace_with_mean(RS)
    BO = replace_with_mean(BO)
    prior_sat = sini
    prior_gas = sgini
    dsw = sat - prior_sat  # ds
    dtime = dt  # ds
    dtime = replace_with_mean(dtime)
    if Relperm == 1:
        one_minus_swi_swr = (1 - (SWI + SWR)).to(torch.float32)
        soa = (torch.divide((1 - (prior_sat + prior_gas) - SWR), one_minus_swi_swr)).to(
            torch.float32
        )
        swa = (torch.divide((prior_sat - SWI), one_minus_swi_swr)).to(torch.float32)
        sga = (torch.divide(prior_gas, one_minus_swi_swr)).to(torch.float32)
        KROW = linear_interp(prior_sat, SWOW[:, 0], SWOW[:, 1]).to(torch.float32)
        KRW = linear_interp(prior_sat, SWOW[:, 0], SWOW[:, 2]).to(torch.float32)
        KROG = linear_interp(prior_gas, SWOG[:, 0], SWOG[:, 1]).to(torch.float32)
        KRG = linear_interp(prior_gas, SWOG[:, 0], SWOG[:, 2]).to(torch.float32)
        KRO = (
            (torch.divide(KROW, (1 - swa)) * torch.divide(KROG, (1 - sga))) * soa
        ).to(torch.float32)
    else:
        KRW, KRO, KRG = StoneIIModel(params, device, prior_gas, prior_sat)
    Mw = (torch.divide(KRW, (UW * BW))).to(torch.float32)
    Mo = (torch.divide(KRO, (UO * BO))).to(torch.float32)
    Mg = (torch.divide(KRG, (UG * BG))).to(torch.float32)
    Mg = replace_with_mean(Mg)
    Mw = replace_with_mean(Mw)
    Mo = replace_with_mean(Mo)
    Mt = (torch.add(torch.add(torch.add(Mw, Mo), Mg), Mo * RS)).to(torch.float32)
    a1 = (Mt * a * fault).to(torch.float32)  # overall Effective permeability
    a1water = (Mw * a * fault).to(torch.float32)  # water Effective permeability
    a1gas = (Mg * a * fault).to(torch.float32)  # gas Effective permeability
    a1oil = (Mo * a * fault).to(torch.float32)  # oil Effective permeability
    # if pde_method == 1:
    dudx_fdm = compute_gradient_3d(u, dx=dxf, dim=0, order=1, padding="replication").to(
        torch.float32
    )
    dudy_fdm = compute_gradient_3d(u, dx=dxf, dim=1, order=1, padding="replication").to(
        torch.float32
    )
    dudz_fdm = compute_gradient_3d(u, dx=dxf, dim=2, order=1, padding="replication").to(
        torch.float32
    )
    dduddx_fdm = compute_second_order_gradient_3d(
        u, dx=dxf, dim=0, padding="replication"
    ).to(torch.float32)
    dduddy_fdm = compute_second_order_gradient_3d(
        u, dx=dxf, dim=1, padding="replication"
    ).to(torch.float32)
    dduddz_fdm = compute_second_order_gradient_3d(
        u, dx=dxf, dim=2, padding="replication"
    ).to(torch.float32)
    dcdx = compute_gradient_3d(a1, dx=dxf, dim=0, order=1, padding="replication").to(
        torch.float32
    )
    dcdy = compute_gradient_3d(a1, dx=dxf, dim=1, order=1, padding="replication").to(
        torch.float32
    )
    dcdz = compute_gradient_3d(a1, dx=dxf, dim=2, order=1, padding="replication").to(
        torch.float32
    )
    darcy_pressure = (
        torch.mul(
            actnum,
            (
                fin
                + dcdx * dudx_fdm
                + a1 * dduddx_fdm
                + dcdy * dudy_fdm
                + a1 * dduddy_fdm
                + dcdz * dudz_fdm
                + a1 * dduddz_fdm
            ),
        )
    ).to(torch.float32)
    darcy_pressure.mul_(dxf)
    p_loss = darcy_pressure
    dudx = dudx_fdm
    dudy = dudy_fdm
    dudz = dudz_fdm
    dduddx = dduddx_fdm
    dduddy = dduddy_fdm
    dduddz = dduddz_fdm
    dadx = compute_gradient_3d(
        a1water, dx=dxf, dim=0, order=1, padding="replication"
    ).to(torch.float32)
    dady = compute_gradient_3d(
        a1water, dx=dxf, dim=1, order=1, padding="replication"
    ).to(torch.float32)
    dadz = compute_gradient_3d(
        a1water, dx=dxf, dim=2, order=1, padding="replication"
    ).to(torch.float32)
    inner_diff = (
        dadx * dudx
        + a1water * dduddx
        + dady * dudy
        + a1water * dduddy
        + dadz * dudz
        + a1water * dduddz
        + finwater
    ).to(torch.float32)
    darcy_saturation = (
        torch.mul(actnum, (poro * torch.divide(dsw, dtime) - inner_diff))
    ).to(torch.float32)
    darcy_saturation.mul_(dxf)
    s_loss = darcy_saturation
    Ugx = (a1gas * dudx_fdm).to(torch.float32)
    Ugy = (a1gas * dudy_fdm).to(torch.float32)
    Ugz = (a1gas * dudz_fdm).to(torch.float32)
    Uox = (a1oil * dudx_fdm * RS).to(torch.float32)
    Uoy = (a1oil * dudy_fdm * RS).to(torch.float32)
    Uoz = (a1oil * dudz_fdm * RS).to(torch.float32)
    Ubigx = (Ugx + Uox).to(torch.float32)
    Ubigy = (Ugy + Uoy).to(torch.float32)
    Ubigz = (Ugz + Uoz).to(torch.float32)
    dubigxdx = compute_gradient_3d(
        Ubigx, dx=dxf, dim=0, order=1, padding="replication"
    ).to(torch.float32)
    dubigxdy = compute_gradient_3d(
        Ubigx, dx=dxf, dim=1, order=1, padding="replication"
    ).to(torch.float32)
    dubigxdz = compute_gradient_3d(
        Ubigx, dx=dxf, dim=2, order=1, padding="replication"
    ).to(torch.float32)
    dubigydx = compute_gradient_3d(
        Ubigy, dx=dxf, dim=0, order=1, padding="replication"
    ).to(torch.float32)
    dubigydy = compute_gradient_3d(
        Ubigy, dx=dxf, dim=1, order=1, padding="replication"
    ).to(torch.float32)
    dubigydz = compute_gradient_3d(
        Ubigy, dx=dxf, dim=2, order=1, padding="replication"
    ).to(torch.float32)
    dubigzdx = compute_gradient_3d(
        Ubigz, dx=dxf, dim=0, order=1, padding="replication"
    ).to(torch.float32)
    dubigzdy = compute_gradient_3d(
        Ubigz, dx=dxf, dim=1, order=1, padding="replication"
    ).to(torch.float32)
    dubigzdz = compute_gradient_3d(
        Ubigz, dx=dxf, dim=2, order=1, padding="replication"
    ).to(torch.float32)
    inner_sum = (
        dubigxdx
        + dubigxdy
        + dubigxdz
        + dubigydx
        + dubigydy
        + dubigydz
        + dubigzdx
        + dubigzdy
        + dubigzdz
        - 9 * fingas
    ).to(torch.float32)
    div_term = (
        torch.divide(
            torch.mul(
                poro,
                (torch.divide(satg, BG) + torch.mul(torch.divide(sato, BO), RS)),
            ),
            dtime,
        )
    ).to(torch.float32)
    darcy_saturationg = (torch.mul(actnum, (inner_sum + div_term))).to(torch.float32)
    sg_loss = (dxf * darcy_saturationg).to(torch.float32)
    p_loss = replace_with_mean(p_loss).to(torch.float32)
    s_loss = replace_with_mean(s_loss).to(torch.float32)
    sg_loss = replace_with_mean(sg_loss).to(torch.float32)
    output_var = {"pressured": p_loss, "saturationd": s_loss, "gasd": sg_loss}
    return normalize_tensors_adjusted(output_var)


def Black_oil(
    input_var,
    neededM,
    SWI,
    SWR,
    UW,
    BW,
    UO,
    BO,
    nx,
    ny,
    nz,
    SWOW,
    SWOG,
    target_min,
    target_max,
    minK,
    maxK,
    minP,
    maxP,
    p_bub,
    p_atm,
    CFO,
    Relperm,
    params,
    pde_method,
    RE,
    max_inn_fcn,
    max_out_fcn,
    DZ,
    device,
    params1_swow,
    params2_swow,
    params1_swog,
    params2_swog,
):
    """Compute black-oil PDE residuals using static (neededM) rate and time inputs.

    Parameters
    ----------
    input_var : dict
        Dict of torch.Tensor fields: ``"pressure"``, ``"perm"``, ``"pini"``,
        ``"poro"``, ``"sini"``, ``"water_sat"``, ``"gas_sat"``, ``"fault"``.
    neededM : dict
        Dict with static fields: ``"Q"``, ``"Qw"``, ``"Qg"``, ``"Qo"``,
        ``"Time"``, ``"actnum"`` as torch.Tensor.
    SWI : torch.Tensor
        Irreducible water saturation.
    SWR : torch.Tensor
        Residual water saturation.
    UW : float
        Water viscosity.
    BW : float
        Water formation volume factor.
    UO : float
        Oil viscosity.
    BO : float
        Oil FVF (overwritten internally by ``calc_bo``).
    nx : int
        Number of grid cells in the x direction.
    ny : int
        Number of grid cells in the y direction.
    nz : int
        Number of grid cells in the z direction.
    SWOW : torch.Tensor
        Water-oil relative permeability table of shape ``(n, 3)``.
    SWOG : torch.Tensor
        Gas-oil relative permeability table of shape ``(n, 3)``.
    target_min : float
        Normalisation target minimum (currently unused).
    target_max : float
        Normalisation target maximum (currently unused).
    minK : float
        Minimum permeability (currently unused).
    maxK : float or np.ndarray
        Permeability scale factor for denormalisation.
    minP : float
        Minimum pressure (currently unused).
    maxP : float or np.ndarray
        Pressure scale factor for denormalisation.
    p_bub : torch.Tensor
        Bubble-point pressure scalar.
    p_atm : torch.Tensor
        Atmospheric pressure scalar.
    CFO : float
        Oil compressibility above bubble point.
    Relperm : int
        Flag selecting relative-permeability model (1 = table, else Stone-II).
    params : dict
        Stone-II model parameters (used when ``Relperm != 1``).
    pde_method : int
        PDE discretisation flag (currently unused; FDM always applied).
    RE : float
        Drainage radius passed to ``compute_peacemannoil``.
    max_inn_fcn : float
        Input normalisation scale passed to ``compute_peacemannoil``.
    max_out_fcn : float
        Output normalisation scale passed to ``compute_peacemannoil``.
    DZ : float
        Perforation thickness passed to ``compute_peacemannoil``.
    device : str
        PyTorch device string.
    params1_swow : tuple
        Cubic polynomial coefficients for ``KROW`` vs. water saturation.
    params2_swow : tuple
        Cubic polynomial coefficients for ``KRW`` vs. water saturation.
    params1_swog : tuple
        Cubic polynomial coefficients for ``KROG`` vs. gas saturation.
    params2_swog : tuple
        Cubic polynomial coefficients for ``KRG`` vs. gas saturation.

    Returns
    -------
    dict
        Normalised residual dictionary with keys ``"pressured"``, ``"saturationd"``,
        and ``"gasd"`` as torch.Tensor fields.
    """
    from forward.gradients_extract import (
        compute_gradient_3d,
        compute_second_order_gradient_3d,
    )

    u = input_var["pressure"]  # .to(torch.float32)
    perm = input_var["perm"]  # .to(torch.float32)
    fin = (
        neededM["Q"]
        .to(torch.float32)
        .repeat(u.shape[0], 1, 1, 1, 1)
        .clamp(min=1e-6, max=1e6)
    )
    finwater = (
        neededM["Qw"]
        .to(torch.float32)
        .repeat(u.shape[0], 1, 1, 1, 1)
        .clamp(min=1e-6, max=1e6)
    )
    dt = neededM["Time"]  # .to(torch.float32)
    pini = input_var["pini"]  # .to(torch.float32)
    poro = input_var["poro"]  # .to(torch.float32)
    sini = input_var["sini"]  # .to(torch.float32)
    sat = input_var["water_sat"]  # .to(torch.float32)
    satg = input_var["gas_sat"]  # .to(torch.float32)
    fingas = neededM["Qg"]
    fingas = fingas.repeat(u.shape[0], 1, 1, 1, 1)
    finoil = neededM["Qo"]
    finoil = finoil.repeat(u.shape[0], 1, 1, 1, 1)
    fingas = (
        neededM["Qg"]
        .to(torch.float32)
        .repeat(u.shape[0], 1, 1, 1, 1)
        .clamp(min=1e-6, max=1e6)
    )
    actnum = (
        neededM["actnum"]
        .to(torch.float32)
        .repeat(u.shape[0], 1, 1, 1, 1)
        .clamp(min=1e-6)
    )
    finoil = finoil.clamp(max=1e6)
    actnum = neededM["actnum"]
    actnum = actnum.repeat(u.shape[0], 1, 1, 1, 1)
    actnum = actnum.clamp(min=1e-6)
    sato = 1 - (sat + satg)
    siniuse = sini[0, 0, 0, 0, 0]
    dxf = 1e-2
    fault = input_var["fault"].to(torch.float32)
    if isinstance(maxP, np.ndarray):
        maxP = torch.tensor(maxP, dtype=torch.float32, device=u.device)
    if isinstance(maxK, np.ndarray):
        maxK = torch.tensor(maxK, dtype=torch.float32, device=perm.device)
    u = u * maxP.to(u.dtype)
    pini = pini * maxP.to(pini.dtype)
    a = (perm * maxK.to(perm.dtype)).to(torch.float32)
    p_loss = torch.zeros_like(u, dtype=torch.float32, device=device)
    s_loss = torch.zeros_like(u, dtype=torch.float32, device=device)
    prior_pressure = torch.zeros(
        sat.shape[0], sat.shape[1], nz, nx, ny, dtype=torch.float32, device=device
    )
    prior_pressure[:, 0, :, :, :] = pini[:, 0, :, :, :]
    prior_pressure[:, 1:, :, :, :] = u[:, :-1, :, :, :]
    avg_p = prior_pressure.mean(dim=(2, 3, 4), keepdim=True).to(torch.float32)
    avg_p = replace_with_mean(avg_p)
    UG = calc_mu_g(avg_p).to(torch.float32)
    RS = calc_rs(p_bub, avg_p, device).to(torch.float32)
    BG = calc_bg(p_bub, p_atm, avg_p).to(torch.float32)
    BO = calc_bo(p_bub, p_atm, CFO, avg_p).to(torch.float32)
    avg_p = replace_with_mean(avg_p)
    UG = replace_with_mean(UG)
    BG = replace_with_mean(BG)
    RS = replace_with_mean(RS)
    BO = replace_with_mean(BO)
    prior_sat = torch.zeros(
        sat.shape[0], sat.shape[1], nz, nx, ny, dtype=torch.float32, device=device
    )
    prior_sat[:, 0, :, :, :] = siniuse * torch.ones(
        sat.shape[0], nz, nx, ny, dtype=torch.float32, device=device
    )
    prior_sat[:, 1:, :, :, :] = sat[:, :-1, :, :, :]
    prior_gas = torch.zeros(
        sat.shape[0], sat.shape[1], nz, nx, ny, dtype=torch.float32, device=device
    )
    prior_gas[:, 0, :, :, :] = torch.zeros(
        sat.shape[0], nz, nx, ny, dtype=torch.float32, device=device
    )
    prior_gas[:, 1:, :, :, :] = satg[:, :-1, :, :, :]
    prior_time = torch.zeros(
        sat.shape[0], sat.shape[1], nz, nx, ny, dtype=torch.float32, device=device
    )
    prior_time[:, 0, :, :, :] = torch.zeros(
        sat.shape[0], nz, nx, ny, dtype=torch.float32, device=device
    )
    prior_time[:, 1:, :, :, :] = dt[:, :-1, :, :, :]
    dsw = sat - prior_sat  # ds
    dtime = dt - prior_time  # ds
    dtime = replace_with_mean(dtime)
    if Relperm == 1:
        one_minus_swi_swr = (1 - (SWI + SWR)).to(torch.float32)
        soa = (torch.divide((1 - (prior_sat + prior_gas) - SWR), one_minus_swi_swr)).to(
            torch.float32
        )
        swa = (torch.divide((prior_sat - SWI), one_minus_swi_swr)).to(torch.float32)
        sga = (torch.divide(prior_gas, one_minus_swi_swr)).to(torch.float32)
        KROW = linear_interp(prior_sat, SWOW[:, 0], SWOW[:, 1]).to(torch.float32)
        KRW = linear_interp(prior_sat, SWOW[:, 0], SWOW[:, 2]).to(torch.float32)
        KROG = linear_interp(prior_gas, SWOG[:, 0], SWOG[:, 1]).to(torch.float32)
        KRG = linear_interp(prior_gas, SWOG[:, 0], SWOG[:, 2]).to(torch.float32)
        KRO = (
            (torch.divide(KROW, (1 - swa)) * torch.divide(KROG, (1 - sga))) * soa
        ).to(torch.float32)
    else:
        KRW, KRO, KRG = StoneIIModel(params, device, prior_gas, prior_sat)
    Mw = (torch.divide(KRW, (UW * BW))).to(torch.float32)
    Mo = (torch.divide(KRO, (UO * BO))).to(torch.float32)
    Mg = (torch.divide(KRG, (UG * BG))).to(torch.float32)
    Mg = replace_with_mean(Mg)
    Mw = replace_with_mean(Mw)
    Mo = replace_with_mean(Mo)
    Mt = (torch.add(torch.add(torch.add(Mw, Mo), Mg), Mo * RS)).to(torch.float32)
    a1 = (Mt * a * fault).to(torch.float32)  # overall Effective permeability
    a1water = (Mw * a * fault).to(torch.float32)  # water Effective permeability
    a1gas = (Mg * a * fault).to(torch.float32)  # gas Effective permeability
    a1oil = (Mo * a * fault).to(torch.float32)  # oil Effective permeability
    dudx_fdm = compute_gradient_3d(u, dx=dxf, dim=0, order=1, padding="replication").to(
        torch.float32
    )
    dudy_fdm = compute_gradient_3d(u, dx=dxf, dim=1, order=1, padding="replication").to(
        torch.float32
    )
    dudz_fdm = compute_gradient_3d(u, dx=dxf, dim=2, order=1, padding="replication").to(
        torch.float32
    )
    dduddx_fdm = compute_second_order_gradient_3d(
        u, dx=dxf, dim=0, padding="replication"
    ).to(torch.float32)
    dduddy_fdm = compute_second_order_gradient_3d(
        u, dx=dxf, dim=1, padding="replication"
    ).to(torch.float32)
    dduddz_fdm = compute_second_order_gradient_3d(
        u, dx=dxf, dim=2, padding="replication"
    ).to(torch.float32)
    dcdx = compute_gradient_3d(a1, dx=dxf, dim=0, order=1, padding="replication").to(
        torch.float32
    )
    dcdy = compute_gradient_3d(a1, dx=dxf, dim=1, order=1, padding="replication").to(
        torch.float32
    )
    dcdz = compute_gradient_3d(a1, dx=dxf, dim=2, order=1, padding="replication").to(
        torch.float32
    )
    finoil = compute_peacemannoil(
        UO,
        BO,
        UW,
        BW,
        DZ,
        RE,
        device,
        max_inn_fcn,
        max_out_fcn,
        params,
        p_bub,
        p_atm,
        prior_gas.shape[1],
        CFO,
        prior_gas,
        prior_sat,
        prior_pressure,
        a,
    ).to(torch.float32)
    fin = (finoil + fingas + finwater).to(torch.float32)
    darcy_pressure = (
        torch.mul(
            actnum,
            (
                fin
                + dcdx * dudx_fdm
                + a1 * dduddx_fdm
                + dcdy * dudy_fdm
                + a1 * dduddy_fdm
                + dcdz * dudz_fdm
                + a1 * dduddz_fdm
            ),
        )
    ).to(torch.float32)
    darcy_pressure.mul_(dxf)
    p_loss = darcy_pressure
    dudx = dudx_fdm
    dudy = dudy_fdm
    dudz = dudz_fdm
    dduddx = dduddx_fdm
    dduddy = dduddy_fdm
    dduddz = dduddz_fdm
    dadx = compute_gradient_3d(
        a1water, dx=dxf, dim=0, order=1, padding="replication"
    ).to(torch.float32)
    dady = compute_gradient_3d(
        a1water, dx=dxf, dim=1, order=1, padding="replication"
    ).to(torch.float32)
    dadz = compute_gradient_3d(
        a1water, dx=dxf, dim=2, order=1, padding="replication"
    ).to(torch.float32)
    inner_diff = (
        dadx * dudx
        + a1water * dduddx
        + dady * dudy
        + a1water * dduddy
        + dadz * dudz
        + a1water * dduddz
        + finwater
    ).to(torch.float32)
    darcy_saturation = (
        torch.mul(actnum, (poro * torch.divide(dsw, dtime) - inner_diff))
    ).to(torch.float32)
    darcy_saturation.mul_(dxf)
    s_loss = darcy_saturation
    Ugx = (a1gas * dudx_fdm).to(torch.float32)
    Ugy = (a1gas * dudy_fdm).to(torch.float32)
    Ugz = (a1gas * dudz_fdm).to(torch.float32)
    Uox = (a1oil * dudx_fdm * RS).to(torch.float32)
    Uoy = (a1oil * dudy_fdm * RS).to(torch.float32)
    Uoz = (a1oil * dudz_fdm * RS).to(torch.float32)
    Ubigx = (Ugx + Uox).to(torch.float32)
    Ubigy = (Ugy + Uoy).to(torch.float32)
    Ubigz = (Ugz + Uoz).to(torch.float32)
    dubigxdx = compute_gradient_3d(
        Ubigx, dx=dxf, dim=0, order=1, padding="replication"
    ).to(torch.float32)
    dubigxdy = compute_gradient_3d(
        Ubigx, dx=dxf, dim=1, order=1, padding="replication"
    ).to(torch.float32)
    dubigxdz = compute_gradient_3d(
        Ubigx, dx=dxf, dim=2, order=1, padding="replication"
    ).to(torch.float32)
    dubigydx = compute_gradient_3d(
        Ubigy, dx=dxf, dim=0, order=1, padding="replication"
    ).to(torch.float32)
    dubigydy = compute_gradient_3d(
        Ubigy, dx=dxf, dim=1, order=1, padding="replication"
    ).to(torch.float32)
    dubigydz = compute_gradient_3d(
        Ubigy, dx=dxf, dim=2, order=1, padding="replication"
    ).to(torch.float32)
    dubigzdx = compute_gradient_3d(
        Ubigz, dx=dxf, dim=0, order=1, padding="replication"
    ).to(torch.float32)
    dubigzdy = compute_gradient_3d(
        Ubigz, dx=dxf, dim=1, order=1, padding="replication"
    ).to(torch.float32)
    dubigzdz = compute_gradient_3d(
        Ubigz, dx=dxf, dim=2, order=1, padding="replication"
    ).to(torch.float32)
    inner_sum = (
        dubigxdx
        + dubigxdy
        + dubigxdz
        + dubigydx
        + dubigydy
        + dubigydz
        + dubigzdx
        + dubigzdy
        + dubigzdz
        - 9 * fingas
    ).to(torch.float32)
    div_term = (
        torch.divide(
            torch.mul(
                poro,
                (torch.divide(satg, BG) + torch.mul(torch.divide(sato, BO), RS)),
            ),
            dtime,
        )
    ).to(torch.float32)
    darcy_saturationg = (torch.mul(actnum, (inner_sum + div_term))).to(torch.float32)
    sg_loss = (dxf * darcy_saturationg).to(torch.float32)
    p_loss = replace_with_mean(p_loss).to(torch.float32)
    s_loss = replace_with_mean(s_loss).to(torch.float32)
    sg_loss = replace_with_mean(sg_loss).to(torch.float32)
    output_var = {"pressured": p_loss, "saturationd": s_loss, "gasd": sg_loss}
    return normalize_tensors_adjusted(output_var)


def interpolate_pytorch_gpu(x_new, x, y):
    """Interpolate a 1-D function on GPU tensors via a CPU-side scipy interpolator.

    Parameters
    ----------
    x_new : torch.Tensor
        Query points at which to evaluate the interpolant.
    x : torch.Tensor
        Known x-coordinates of the function (sorted, finite values used).
    y : torch.Tensor
        Known y-values corresponding to ``x``.

    Returns
    -------
    torch.Tensor
        Interpolated values at ``x_new``, clamped to ``[0, 1]`` and on the same device as ``x``.
    """
    x_np = x.float().detach().cpu().numpy()
    y_np = y.float().detach().cpu().numpy()
    x_new_np = x_new.float().detach().cpu().numpy()
    # Remove NaNs and Infs from x and y
    valid_indices = np.isfinite(x_np) & np.isfinite(y_np)
    x_np_clean = x_np[valid_indices]
    y_np_clean = y_np[valid_indices]
    interpolator = interp1d(
        x_np_clean, y_np_clean, kind="linear", fill_value="extrapolate"
    )
    y_new_np = interpolator(x_new_np)
    y_new_tensor = torch.from_numpy(y_new_np).float()
    y_new_tensor = torch.clamp(y_new_tensor, min=0, max=1)
    y_new_tensor = replace_with_mean(y_new_tensor)
    y_new_tensor = y_new_tensor.to(x.device)
    return torch.clamp(y_new_tensor, 0, 1)


def linear_interp2D(x, xp, fp):
    """Apply column-wise 1-D linear interpolation to a 2-D tensor.

    Parameters
    ----------
    x : torch.Tensor
        Query tensor of shape ``(T, num_vars)``; each column interpolated independently.
    xp : torch.Tensor
        1-D sorted breakpoint coordinate tensor.
    fp : torch.Tensor
        1-D function values at each breakpoint in ``xp``.

    Returns
    -------
    torch.Tensor
        Interpolated tensor of shape ``(T, num_vars)``, clamped to ``[0, 1]``.
    """
    # Ensure inputs are contiguous for performance
    xp_contiguous = xp.contiguous()
    fp_contiguous = fp.contiguous()
    _T, num_vars = x.shape  # num_vars is 10 in this context
    interpolated_values = torch.zeros_like(x)
    for var in range(num_vars):
        x_col = x[:, var]
        left_indices = torch.clamp(
            torch.searchsorted(xp_contiguous, x_col) - 1, 0, len(xp_contiguous) - 2
        )
        denominators = xp_contiguous[left_indices + 1] - xp_contiguous[left_indices]
        close_to_zero = denominators.abs() < 1e-6
        denominators[close_to_zero] = 1  # Avoid division by zero
        interpolated_col = (
            (
                (fp_contiguous[left_indices + 1] - fp_contiguous[left_indices])
                / denominators
            )
            * (x_col - xp_contiguous[left_indices])
        ) + fp_contiguous[left_indices]
        interpolated_col = torch.where(
            torch.isnan(interpolated_col) | torch.isinf(interpolated_col),
            torch.zeros_like(interpolated_col),
            interpolated_col,
        )
        interpolated_values[:, var] = interpolated_col
    interpolated_values = replace_with_mean(interpolated_values)
    return torch.clamp(interpolated_values, 0, 1)


class Polynomial3(nn.Module):
    """Polynomial3."""

    def __init__(self):
        """Initialise learnable cubic polynomial parameters (a, b, c, d)."""
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))
        self.d = nn.Parameter(torch.randn(1))

    def forward(self, x):
        """Evaluate ``a*x³ + b*x² + c*x + d``.

        Parameters
        ----------
        x : torch.Tensor
            Input values of any shape.

        Returns
        -------
        torch.Tensor
            Polynomial output of the same shape as ``x``.
        """
        return self.a * x**3 + self.b * x**2 + self.c * x + self.d


def train(model, optimizer, x, y, criterion, epochs=5000):
    """Run a simple training loop for a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train; modified in-place via the optimizer.
    optimizer : torch.optim.Optimizer
        Optimiser used to update model parameters.
    x : torch.Tensor
        Input training tensor.
    y : torch.Tensor
        Target training tensor.
    criterion : torch.nn.Module
        Loss function (e.g., ``nn.MSELoss``).
    epochs : int, optional
        Number of training iterations. Default is 5000.
    """
    for _epoch in range(epochs):
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_polynomial_models(data, device):
    """Fit two independent cubic polynomial models to relative-permeability table data.

    Parameters
    ----------
    data : np.ndarray
        Array of shape ``(n, 3)`` where column 0 is input saturation,
        column 1 is the first output (e.g., ``KROW``), and column 2 is the second
        (e.g., ``KRW``).
    device : str
        PyTorch device string for model placement.

    Returns
    -------
    params1 : list of float
        Fitted ``[a, b, c, d]`` coefficients for the first polynomial model.
    params2 : list of float
        Fitted ``[a, b, c, d]`` coefficients for the second polynomial model.
    """
    data = torch.tensor(data, dtype=torch.float32)
    x = data[:, 0].unsqueeze(1)  # Input
    y1 = data[:, 1].unsqueeze(1)  # Output for machine 1
    y2 = data[:, 2].unsqueeze(1)  # Output for machine 2
    machine1 = Polynomial3().to(device)
    machine2 = Polynomial3().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer1 = optim.SGD(machine1.parameters(), lr=0.01)
    optimizer2 = optim.SGD(machine2.parameters(), lr=0.01)
    train(machine1, optimizer1, x, y1, criterion)
    train(machine2, optimizer2, x, y2, criterion)
    params1 = [
        machine1.a.item(),
        machine1.b.item(),
        machine1.c.item(),
        machine1.d.item(),
    ]
    params2 = [
        machine2.a.item(),
        machine2.b.item(),
        machine2.c.item(),
        machine2.d.item(),
    ]
    return params1, params2


def replace_with_mean(tensor):
    """Replace NaN and Inf values in a tensor with a perturbed finite mean.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of any shape that may contain NaN or Inf values.

    Returns
    -------
    torch.Tensor
        Float32 tensor with invalid entries replaced and values clamped to 1e-6.
    """
    tensor = tensor.to(torch.float32)
    valid_elements = tensor[torch.isfinite(tensor)]
    if valid_elements.numel() > 0:  # Check if there are any valid elements
        mean_value = valid_elements.mean()  # ✅ Retains gradients
        perturbation = torch.normal(mean=0.0, std=0.01, size=(1,), device=tensor.device)
        perturbed_mean_value = mean_value + perturbation
    else:
        perturbed_mean_value = torch.tensor(
            1e-4, device=tensor.device, dtype=torch.float32, requires_grad=True
        )
    ouut = torch.where(
        torch.isnan(tensor) | torch.isinf(tensor), perturbed_mean_value, tensor
    )
    return torch.clamp(ouut, min=1e-6)  # ✅ Keeps gradients flowing


def safe_mean_std(data):
    """Compute the global mean and standard deviation of an array, guarding against Inf std.

    Parameters
    ----------
    data : np.ndarray
        Input array of any shape.

    Returns
    -------
    mean : float
        Global mean of all elements.
    std : float or None
        Global standard deviation (ddof=1), or ``None`` if the result is infinite.
    """
    mean = np.mean(data, axis=None)
    std = np.std(data, axis=None, ddof=1)
    if np.isinf(std):
        std = None
    return mean, std
