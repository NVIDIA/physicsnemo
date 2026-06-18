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

# ruff: noqa: RUF001, RUF002, RUF003
# Greek letters (alpha, etc.) appear in this module's docstrings and log
# messages because they refer to the alpha-REKI algorithm by its conventional
# notation; ruff's confusable-character rule is suppressed for the file.
"""
SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES.
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
simulation using NVIDIA PhysicsNeMo surrogates. It supports optional
parametrisation (DCT/VCAE) and spatial localisation, and logs per-iteration
cost statistics for both ensemble mean and best member.

This version is rank-aware for multi-GPU execution under torchrun:
- Forward model is sharded across ranks via Forward_model_ensemble.
- All file/IO and plotting operations are guarded with `if dist.rank == 0`.
- VCAE training (if used) runs on rank 0 only; all ranks then load weights.
- Kalman updates run on every rank deterministically (gathered inputs).

Typical Usage:
    from inverse.history_matching import run_history_matching_loop
    results = run_history_matching_loop(dist, logger, cfg, ...)

@Author : Clement Etienam
"""

import os
import numpy as np
import numpy.matlib
import torch
import torch.optim as optim
import torch.distributed as torchdist
from hydra.utils import to_absolute_path
import matplotlib.pyplot as plt
from utils.ccr_utils import (
    Forward_model_ensemble,
)

from inverse.inversion_operation_ensemble import (
    compute_data_mismatch,
    clip_ensemble_params,
    ensemble_pytorch,
)
from inverse.inversion_operation_gather import (
    plot_rsm,
    plot_rsm_singleT,
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
    Get_Kalman_Gain_EKI,
    Get_Kalman_Gain_ESMDA,
)


from inverse.utils.inverse_config import (
    GridConfig,
    NormBounds,
    PermBounds,
    EnsembleSetup,
    WellConfig,
    SurrogateConfig,
    InversionParams,
    TimeArrays,
    FlowArrays,
)


def _is_dist_active():
    """Return True if torch.distributed has been initialised."""
    return torchdist.is_available() and torchdist.is_initialized()


def _barrier():
    """Distributed barrier that no-ops in single-GPU mode."""
    if _is_dist_active():
        torchdist.barrier()


def compute_rowwise_scaling(True_data: np.ndarray) -> np.ndarray:
    """
    Build a (Nop, 1) scaling vector that brings each value into roughly 1-digit range
    (i.e. magnitude in [1, 10)). Works in both directions:

    Examples:
        60000  → scale = 0.0001   → scaled = 6.0    ✓
        500    → scale = 0.01     → scaled = 5.0    ✓
        50     → scale = 0.1      → scaled = 5.0    ✓
        5      → scale = 1.0      → scaled = 5.0    ✓  (already 1-digit)
        0.5    → scale = 10.0     → scaled = 5.0    ✓
        0.001  → scale = 1000.0   → scaled = 1.0    ✓
        0.0    → scale = 1.0      → scaled = 0.0    ✓  (guard, untouched)
    """
    Nop = True_data.shape[0]
    scale_vec = np.ones((Nop, 1), dtype=np.float64)

    for i in range(Nop):
        val = abs(float(True_data[i, 0]))
        if val == 0.0:
            continue
        # Bring val into [1, 10): find the exponent needed
        exp = np.floor(np.log10(val))
        scale_vec[i, 0] = 10.0 ** (-exp)

    return scale_vec  # (Nop, 1)


def get_keep_mask(data, threshold=1e-4, reshape_fortran=True):
    """Return boolean mask: True = keep (value > threshold)."""
    is_torch = isinstance(data, torch.Tensor)

    if is_torch:
        if reshape_fortran:
            flat = data.permute(*reversed(range(data.ndim))).contiguous().reshape(-1)
        else:
            flat = data.reshape(-1)
        return flat > threshold
    else:
        order = "F" if reshape_fortran else "C"
        return np.reshape(data, -1, order) > threshold


def run_history_matching_loop(
    dist,
    logger,
    cfg,
    device,
    oldfolder: str,
    gpu_available: bool,
    iteration_converged: float,
    iteration_count: int,
    input_variables: list,
    output_variables: list,
    ensemble,
    ensemblep,
    ensemblef,
    CDd,
    True_mat,
    perturbations,
    grid: "GridConfig",
    norm: "NormBounds",
    perm: "PermBounds",
    ens: "EnsembleSetup",
    well: "WellConfig",
    surrogate: "SurrogateConfig",
    inversion: "InversionParams",
    time_arr: "TimeArrays",
    flow: "FlowArrays",
) -> tuple:
    """Run the history-matching loop (aREKI or ES-MDA), rank-aware.

    Performs iterative history matching using either:
    - **aREKI** — Adaptive Regularised Ensemble Kalman Inversion, which
      adaptively selects an inflation scalar α at each iteration and
      terminates when Σ(1/α) ≥ 1.
    - **ES-MDA** — Ensemble Smoother with Multiple Data Assimilation, which
      runs a fixed number of iterations with constant α = Termm.

    Method is selected via ``cfg.custom.INVERSE_PROBLEM.assimilation``
    (``"aREKI"`` or ``"ESMDA"``).

    Distributed behaviour
    ---------------------
    - The forward surrogate (`Forward_model_ensemble`) shards ensemble members
      across ranks and gathers results — every rank ends up with the same data.
    - Plotting, I/O, and disk writes are guarded with ``if dist.rank == 0``.
    - VCAE model training (if `Do_param_method == 'VCAE'`) runs on rank 0 only;
      other ranks wait at a barrier and then load the trained weights.
    - The Kalman update runs on every rank because inputs are identical.

    Returns
    -------
    tuple
        ``(use_k, use_p, use_f, mean_cost, best_cost,
        ensemble_bestK, ensemble_meanK, ensemble_bestP, ensemble_meanP,
        ensemble_bestf, ensemble_meanf, iteration_count, iteration_converged,
        alpha_big, ensemble, ensemblep, ensemblef, chm, cc_ini,
        ensemble_dict, base_k, base_p, base_f)``
    """
    # --- unpack dataclasses to local names expected by the function body ---
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    steppi = grid.steppi
    steppi_indices = grid.steppi_indices
    effective = ens.effective
    Ne = ens.Ne
    N_ens = ens.N_ens
    effec = ens.effec
    active_cells_ensemble = ens.active_cells_ensemble
    active_mask_3d = ens.active_mask_3d
    target_min = norm.target_min
    target_max = norm.target_max
    minK, maxK = norm.minK, norm.maxK
    minT, maxT = norm.minT, norm.maxT
    minP, maxP = norm.minP, norm.maxP
    minQ, maxQ = norm.minQ, norm.maxQ
    minQw, maxQw = norm.minQw, norm.maxQw
    minQg, maxQg = norm.minQg, norm.maxQg
    min_inn_fcn, max_inn_fcn = norm.min_inn_fcn, norm.max_inn_fcn
    min_out_fcn, max_out_fcn = norm.min_out_fcn, norm.max_out_fcn
    min_inn_fcn2, max_inn_fcn2 = norm.min_inn_fcn2, norm.max_inn_fcn2
    min_out_fcn2, max_out_fcn2 = norm.min_out_fcn2, norm.max_out_fcn2
    High_K1, Low_K1 = perm.High_K1, perm.Low_K1
    High_P, Low_P = perm.High_P, perm.Low_P
    models = surrogate.models
    Trainmoe = surrogate.Trainmoe
    pred_type = surrogate.pred_type
    degg = surrogate.degg
    experts = surrogate.experts
    num_cores = surrogate.num_cores
    producers = well.producers
    injectors = well.injectors
    gas_injectors = well.gas_injectors
    well_names = well.well_names
    N_pr = well.N_pr
    lenwels = well.lenwels
    compdat_data = well.compdat_data
    Time = time_arr.Time
    Time_unie1 = time_arr.Time_unie1
    timestep = time_arr.timestep
    awater = flow.awater
    agas = flow.agas
    aoil = flow.aoil
    aqq = flow.aqq
    Termm = inversion.Termm
    Do_parametrisation = inversion.Do_parametrisation
    Do_param_method = inversion.Do_param_method
    do_localisation = inversion.do_localisation
    size1 = inversion.size1
    size2 = inversion.size2
    # -----------------------------------------------------------------------

    # ── Ensure RESULTS/HM_RESULTS exists (rank 0 only, then barrier) ─────────
    if dist.rank == 0:
        os.makedirs(to_absolute_path("../RESULTS/HM_RESULTS"), exist_ok=True)
    # _barrier()

    # ── Helper: coerce tensor/array to numpy ─────────────────────────────────
    def _to_np(x):
        """Coerce a numpy array or torch tensor to a numpy array."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # ── bookkeeping ───────────────────────────────────────────────────────────
    alpha_big, mean_cost, best_cost = [], [], []
    ensemble_bestK, ensemble_meanK = [], []
    ensemble_bestP, ensemble_meanP = [], []
    ensemble_bestf, ensemble_meanf = [], []
    ensemble_dict = {}

    # initialise base / use_* defaults so all return values are defined
    base_k = base_p = base_f = None
    use_k = use_p = use_f = None

    if "PERM" in input_variables:
        ens_np = _to_np(ensemble)
        base_k = np.mean(ens_np, axis=1).reshape(-1, 1)
        use_k = ens_np.copy()
    if "PORO" in input_variables:
        ensp_np = _to_np(ensemblep)
        base_p = np.mean(ensp_np, axis=1).reshape(-1, 1)
        use_p = ensp_np.copy()
    if "FAULT" in input_variables:
        ensf_np = _to_np(ensemblef)
        base_f = np.mean(ensf_np, axis=1).reshape(-1, 1)
        use_f = ensf_np.copy()

    assimilation = cfg.custom.INVERSE_PROBLEM.assimilation
    use_areki = assimilation == "aREKI"

    if dist.rank == 0:
        logger.info("=" * 64)
        method_label = (
            "Adaptive Regularised Ensemble Kalman Inversion (α-REKI)"
            if use_areki
            else "Ensemble Smoother Multiple Data Assimilation (ES-MDA)"
        )
        logger.info(f"  {method_label}")
        logger.info("=" * 64)

    # ── fixed α for ES-MDA ────────────────────────────────────────────────────
    n_iterations = Termm

    # ── shared forward-model call ─────────────────────────────────────────────
    def _forward(ensemblepy):
        """Run the forward surrogate model on the current ensemble.

        `Forward_model_ensemble` is rank-aware: it shards work across ranks
        and gathers identical full results to every rank.
        """
        return Forward_model_ensemble(
            ensemble.shape[1] if hasattr(ensemble, "shape") else len(ensemble),
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
            active_cells_ensemble,
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
            well_names,
            cfg,
            N_pr,
            lenwels,
            active_mask_3d,
            awater,
            agas,
            aoil,
            aqq,
            nx,
            ny,
            nz,
            minQ,
            maxQ,
            minQw,
            maxQw,
            minQg,
            maxQg,
        )

    # ── shared parametrisation helpers (PERM/PORO only) ──────────────────────
    def _encode(ens_dict):
        """Encode ensemble to parametrised space (DCT or VCAE)."""
        out = {}
        if Do_param_method == "DCT":
            if "PERM" in input_variables:
                out["PERM"] = dct22(ens_dict["PERM"], Ne, nx, ny, nz, size1, size2)
            if "PORO" in input_variables:
                out["PORO"] = dct22(ens_dict["PORO"], Ne, nx, ny, nz, size1, size2)
        else:  # VCAE
            if "PERM" in input_variables:
                out["PERM"] = encode_values(
                    ens_dict["PERM"] / maxK, nz, nx, ny, device, model_perm
                )
            if "PORO" in input_variables:
                out["PORO"] = encode_values(
                    ens_dict["PORO"], nz, nx, ny, device, model_poro
                )
        return out

    def _decode(upd_dict):
        """Decode updated ensemble back to physical space."""
        out = {}
        if Do_param_method == "DCT":
            if "PERM" in input_variables:
                out["PERM"] = idct22(upd_dict["PERM"], Ne, nx, ny, nz, size1, size2)
            if "PORO" in input_variables:
                out["PORO"] = idct22(upd_dict["PORO"], Ne, nx, ny, nz, size1, size2)
        else:  # VCAE
            if "PERM" in input_variables:
                out["PERM"] = decode_values(
                    upd_dict["PERM"], device, nx, ny, nz, model_perm
                )
            if "PORO" in input_variables:
                out["PORO"] = decode_values(
                    upd_dict["PORO"], device, nx, ny, nz, model_poro
                )
        return out

    # ── VCAE model initialisation (once, before the loop, rank-aware) ────────
    model_perm = model_poro = None
    if Do_parametrisation != "No" and Do_param_method == "VCAE":
        latent_dim = 600
        model_perm, model_poro = _init_vcae_models(
            latent_dim, nz, nx, ny, maxK, device, cfg, logger, dist
        )

    # ── localisation matrix (built once, rank-aware) ─────────────────────────
    locmat = None

    # ─────────────────────────────────────────────────────────────────────────
    # shared single-iteration body
    # ─────────────────────────────────────────────────────────────────────────
    def _run_one_iteration(
        iteration_count,
        iteration_converged,
        ensemble,
        ensemblep,
        ensemblef,
        tinumeanprior,
        tinubestprior,
        best_cost_mean,
        best_cost_best,
        cc_ini,
        alpha_fixed=None,
    ):
        """Execute one assimilation iteration.

        Parameters
        ----------
        alpha_fixed : float or None
            If not None, use this fixed α (ES-MDA). If None, compute
            adaptively (aREKI).
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        nonlocal use_k, use_p, use_f, locmat

        if dist.rank == 0:
            logger.info("*" * 64)
            logger.info(f"  Iteration {iteration_count + 1} / {Termm}")
            logger.info("*" * 64)

        # ── build ensemble dict ───────────────────────────────────────────────
        ens_dict = {}
        ini_K = ini_p = ini_f = None
        if "PERM" in input_variables:
            ens_dict["PERM"] = ensemble
            ini_K = ensemble
        if "PORO" in input_variables:
            ens_dict["PORO"] = ensemblep
            ini_p = ensemblep
        if "FAULT" in input_variables:
            ens_dict["FAULT"] = ensemblef
            ini_f = ensemblef

        # ── forward model (rank-aware: shards & gathers internally) ──────────
        ensemblepy = ensemble_pytorch(
            ens_dict,
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
        simout = _forward(ensemblepy)
        predMatrix = simout["ouut_p"]

        # ── prior plots (iteration 0, rank 0 only) ───────────────────────────
        if iteration_count == 0 and dist.rank == 0:
            os.chdir(to_absolute_path("../RESULTS/HM_RESULTS"))
            # plot_rsm(predMatrix[:, :, :N_pr],
            # True_mat[:, :N_pr],
            # "PRIOR_ENSEMBLE_WOPR",
            # Ne, Time_unie1, N_pr, well_names, "WOPR")
            # plot_rsm(predMatrix[:, :, N_pr:2 * N_pr],
            # True_mat[:, N_pr:2 * N_pr],
            # "PRIOR_ENSEMBLE_WWPR",
            # Ne, Time_unie1, N_pr, well_names, "WWPR")
            plot_rsm(
                predMatrix[:, :, 2 * N_pr : 3 * N_pr],
                True_mat[:, 2 * N_pr : 3 * N_pr],
                "PRIOR_ENSEMBLE_WGPR",
                Ne,
                Time_unie1,
                N_pr,
                well_names,
                "WGPR",
            )
            os.chdir(oldfolder)

        _, _True_data1, True_mat_local = historydata(
            timestep, steppi, steppi_indices, N_pr
        )
        True_mat_local[True_mat_local <= 0] = 0
        if dist.rank == 0:
            os.chdir(to_absolute_path("../RESULTS/HM_RESULTS"))
            # plot_rsm_singleT(True_mat_local[:, :N_pr], Time_unie1, N_pr, well_names, "WOPR")
            # plot_rsm_singleT(True_mat_local[:, N_pr:2 * N_pr], Time_unie1, N_pr, well_names, "WWPR")
            plot_rsm_singleT(
                True_mat_local[:, 2 * N_pr : 3 * N_pr],
                Time_unie1,
                N_pr,
                well_names,
                "WGPR",
            )
            os.chdir(oldfolder)

        # ── assemble scaled True_data vector ──────────────────────────────────
        flat_true = np.reshape(_True_data1, (-1, 1), "F")  # (N, 1)
        mask = get_keep_mask(flat_true, threshold=1e-4)  # (N,) bool

        True_data = flat_true[mask]  # (n_kept, 1)
        scale_mat = compute_rowwise_scaling(True_data)  # (n_kept, 1)
        True_data = True_data * scale_mat

        assert simout["sim"].shape[0] == flat_true.shape[0], "row mismatch"
        simDatafinal = simout["sim"][mask] * scale_mat

        # ── tensor conversions ───────────────────────────────────────────────
        Nop = True_data.shape[0]
        True_dataa = torch.as_tensor(True_data, dtype=torch.float32, device=device)
        CDd_t = torch.as_tensor(CDd, dtype=torch.float32, device=device)
        Dd = True_dataa.repeat(1, Ne)
        simDatafinal = torch.as_tensor(simDatafinal, dtype=torch.float32, device=device)

        # ── data mismatch and α ───────────────────────────────────────────────
        yyy = (
            0.5 * (Dd - simDatafinal).T @ torch.linalg.inv(CDd_t) @ (Dd - simDatafinal)
        )
        yyy = torch.nan_to_num(torch.mean(yyy, dim=1), nan=0.0).reshape(-1, 1)
        alpha_star = torch.mean(yyy)
        alpha_star2 = torch.var(yyy)
        leftt = Nop / (2 * alpha_star)
        rightt = torch.sqrt(
            torch.tensor(Nop, dtype=torch.float32, device=device) / (2 * alpha_star2)
        )

        if alpha_fixed is None:
            # aREKI — compute adaptive α on rank 0, broadcast to all
            if dist.rank == 0:
                chok = torch.clamp(
                    torch.max(leftt, rightt), max=1.0 - iteration_converged
                )
                alpha = 1.0 / chok
            else:
                chok = torch.zeros(1, device=device)
                alpha = torch.zeros(1, device=device)
            if _is_dist_active() and torchdist.get_world_size() > 1:
                # ensure scalars are 1-element tensors and contiguous
                chok_t = chok.reshape(1).contiguous()
                alpha_t = alpha.reshape(1).contiguous()
                torchdist.broadcast(chok_t, src=0)
                torchdist.broadcast(alpha_t, src=0)
                chok = chok_t.reshape(())
                alpha = alpha_t.reshape(())
            alpha_big.append(alpha.item())
        else:
            # ES-MDA — fixed α, identical on every rank by construction
            alpha = torch.tensor(float(alpha_fixed), device=device)
            chok = torch.tensor(1.0 / alpha_fixed, device=device)
            alpha_big.append(alpha_fixed)

        if dist.rank == 0:
            logger.info(
                f"  α = {float(alpha):.4f}   iteration_converged = {float(iteration_converged):.4f}"
            )

        overall = {}
        if Do_parametrisation == "No":
            if "PERM" in input_variables:
                overall["PERM"] = torch.as_tensor(
                    _to_np(ens_dict["PERM"]), dtype=torch.float32, device=device
                )
            if "PORO" in input_variables:
                overall["PORO"] = torch.as_tensor(
                    _to_np(ens_dict["PORO"]), dtype=torch.float32, device=device
                )
            if "FAULT" in input_variables:
                overall["FAULT"] = torch.as_tensor(
                    _to_np(ens_dict["FAULT"]), dtype=torch.float32, device=device
                )
        else:
            encoded = _encode(ens_dict)
            for k, v in encoded.items():
                overall[k] = torch.as_tensor(
                    _to_np(v), dtype=torch.float32, device=device
                )

            if "FAULT" in input_variables:
                overall["FAULT"] = torch.as_tensor(
                    _to_np(ens_dict["FAULT"]), dtype=torch.float32, device=device
                )

        # ── localisation (built once on rank 0, broadcast to all) ────────────
        if Do_parametrisation == "No" and do_localisation == "Yes" and locmat is None:
            locmat_np = _build_localisation_rank_aware(
                nx,
                ny,
                nz,
                Ne,
                gas_injectors,
                producers,
                injectors,
                effec,
                dist,
                oldfolder,
                device,
            )
            locmat = torch.as_tensor(locmat_np, dtype=torch.float32, device=device)

        # ── Kalman update (runs on every rank deterministically) ─────────────
        kalman_fn = (
            Get_Kalman_Gain_EKI if alpha_fixed is None else Get_Kalman_Gain_ESMDA
        )
        Youtt = {}
        for key, Y in overall.items():
            if dist.rank == 0:
                logger.info(f"  Processing key: {key}")
            upd = kalman_fn(
                Y,
                simDatafinal,
                CDd_t,
                alpha,
                device,
                perturbations,
                True_data,
                Ne,
                dist,
            )
            if (
                Do_parametrisation == "No"
                and do_localisation == "Yes"
                and key in ("PERM", "PORO")
            ):
                upd = upd * locmat
            Youtt[key] = Y + upd

        updated = {k: _to_np(v) for k, v in Youtt.items()}

        # ── RMSE tracking ─────────────────────────────────────────────────────
        simmean = torch.mean(simDatafinal, dim=1, keepdim=True)
        tinuke = (
            (torch.sqrt(torch.sum((simmean - True_dataa) ** 2)) / Nop)
            .detach()
            .cpu()
            .numpy()
        )

        _aa, _bb, cc = compute_data_mismatch(simDatafinal, True_dataa)
        muv = torch.argmin(cc)
        muv_idx = int(muv.item())

        tinukebest = (
            (
                torch.sqrt(
                    torch.sum(
                        (simDatafinal[:, muv_idx : muv_idx + 1] - True_dataa) ** 2
                    )
                )
                / Nop
            )
            .detach()
            .cpu()
            .numpy()
        )

        if iteration_count == 0:
            cc_ini = cc
            tinumeanprior = tinuke
            tinubestprior = tinukebest
            best_cost_mean = tinuke
            best_cost_best = tinukebest
            if dist.rank == 0:
                logger.info(f"  Initial RMSE mean = {tinuke}   best = {tinukebest}")
        else:
            _log_rmse_change(
                tinuke, tinukebest, tinumeanprior, tinubestprior, dist, logger
            )
            tinumeanprior = tinuke
            tinubestprior = tinukebest

        # ── save best ─────────────────────────────────────────────────────────
        # use_k/use_p/use_f are kept as numpy for downstream consumers
        if best_cost_mean > tinuke:
            if dist.rank == 0:
                logger.info("  ✔ Ensemble saved (improved)")
                logger.info(f"  Current best mean cost = {best_cost_mean}")
                logger.info(f"  Current iteration mean cost = {tinuke}")
                logger.info(f"  Current best MAP cost = {best_cost_best}")
                logger.info(f"  Current iteration MAP cost = {tinukebest}")
            best_cost_mean = tinuke
            best_cost_best = tinukebest
            if "PERM" in input_variables:
                use_k = _to_np(ensemble)
            if "PORO" in input_variables:
                use_p = _to_np(ensemblep)
            if "FAULT" in input_variables:
                use_f = _to_np(ensemblef)
        else:
            if dist.rank == 0:
                logger.info("  ✗ Ensemble NOT saved (no improvement)")
                logger.info(f"  Current best mean cost = {best_cost_mean}")
                logger.info(f"  Current iteration mean cost = {tinuke}")
                logger.info(f"  Current best MAP cost = {best_cost_best}")
                logger.info(f"  Current iteration MAP cost = {tinukebest}")
            if "PERM" in input_variables:
                use_k = _to_np(ini_K)
            if "PORO" in input_variables:
                use_p = _to_np(ini_p)
            if "FAULT" in input_variables:
                use_f = _to_np(ini_f)

        mean_cost.append(tinuke)
        best_cost.append(tinukebest)

        # Coerce ini_* to numpy once for the bookkeeping; used multiple times below.
        if "PERM" in input_variables:
            ini_K_np = _to_np(ini_K)
            ensemble_bestK.append(ini_K_np[:, muv_idx].reshape(-1, 1))
            ensemble_meanK.append(np.reshape(np.mean(ini_K_np, axis=1), (-1, 1), "F"))
        if "PORO" in input_variables:
            ini_p_np = _to_np(ini_p)
            ensemble_bestP.append(ini_p_np[:, muv_idx].reshape(-1, 1))
            ensemble_meanP.append(np.reshape(np.mean(ini_p_np, axis=1), (-1, 1), "F"))
        if "FAULT" in input_variables:
            ini_f_np = _to_np(ini_f)
            ensemble_bestf.append(ini_f_np[:, muv_idx].reshape(-1, 1))
            ensemble_meanf.append(np.reshape(np.mean(ini_f_np, axis=1), (-1, 1), "F"))

        # ── decode & honour bounds ────────────────────────────────────────────
        if Do_parametrisation == "No":
            if "PERM" in input_variables:
                ensemble = updated["PERM"]
            if "PORO" in input_variables:
                ensemblep = updated["PORO"]
        else:
            decoded = _decode(updated)
            if "PERM" in input_variables:
                ensemble = decoded["PERM"]
            if "PORO" in input_variables:
                ensemblep = decoded["PORO"]
        if "FAULT" in input_variables:
            ensemblef = updated["FAULT"]

        outt = clip_ensemble_params(
            {"PERM": ensemble, "PORO": ensemblep},
            nx,
            ny,
            nz,
            N_ens,
            High_K1,
            Low_K1,
            High_P,
            Low_P,
            effec,
        )
        ensemble = outt["PERM"]
        ensemblep = outt["PORO"]
        if "FAULT" in input_variables:
            if isinstance(ensemblef, torch.Tensor):
                ensemblef = torch.clamp(ensemblef, min=0.0, max=1.0)
            else:
                ensemblef = np.clip(ensemblef, 0, 1)

        return dict(
            ensemble=ensemble,
            ensemblep=ensemblep,
            ensemblef=ensemblef,
            tinumeanprior=tinumeanprior,
            tinubestprior=tinubestprior,
            best_cost_mean=best_cost_mean,
            best_cost_best=best_cost_best,
            cc_ini=cc_ini,
            chok=chok,
            muv=muv,
        )

    # ── run the loop (unchanged) ──────────────────────────────────────────────
    tinumeanprior = tinubestprior = best_cost_mean = best_cost_best = None
    cc_ini = None

    if use_areki:
        # ─── aREKI: adaptive α, terminates when Σ(1/α) ≥ 1 ───────────────────
        while iteration_converged < 1.0:
            state = _run_one_iteration(
                iteration_count,
                iteration_converged,
                ensemble,
                ensemblep,
                ensemblef,
                tinumeanprior,
                tinubestprior,
                best_cost_mean,
                best_cost_best,
                cc_ini,
                alpha_fixed=None,
            )
            ensemble = state["ensemble"]
            ensemblep = state["ensemblep"]
            ensemblef = state["ensemblef"]
            tinumeanprior = state["tinumeanprior"]
            tinubestprior = state["tinubestprior"]
            best_cost_mean = state["best_cost_mean"]
            best_cost_best = state["best_cost_best"]
            cc_ini = state["cc_ini"]
            chok = state["chok"]

            iteration_converged += chok.item() if torch.is_tensor(chok) else float(chok)
            iteration_count += 1

            if iteration_converged >= 1.0:
                if dist.rank == 0:
                    logger.info("  ✔ Converged (Σ1/α ≥ 1)")
                break
            if iteration_count >= Termm:
                if dist.rank == 0:
                    logger.info("  ⚠ Max iterations reached without convergence")
                break
    else:
        # ─── ES-MDA: fixed α = n_iterations, runs all iterations ─────────────
        for iteration_count in range(n_iterations):
            state = _run_one_iteration(
                iteration_count,
                iteration_converged,
                ensemble,
                ensemblep,
                ensemblef,
                tinumeanprior,
                tinubestprior,
                best_cost_mean,
                best_cost_best,
                cc_ini,
                alpha_fixed=n_iterations,
            )
            ensemble = state["ensemble"]
            ensemblep = state["ensemblep"]
            ensemblef = state["ensemblef"]
            tinumeanprior = state["tinumeanprior"]
            tinubestprior = state["tinubestprior"]
            best_cost_mean = state["best_cost_mean"]
            best_cost_best = state["best_cost_best"]
            cc_ini = state["cc_ini"]

            iteration_converged += 1.0 / n_iterations

        iteration_count = n_iterations

        if dist.rank == 0:
            logger.info("  ✔ ES-MDA: all iterations complete")

    # ── α evolution plot (rank 0 only) ───────────────────────────────────────
    if dist.rank == 0:
        alpha_arr = np.array(alpha_big, dtype=float)
        sum_inv = np.sum(1.0 / (alpha_arr + 1e-12))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(alpha_arr, marker="o", linestyle="-", color="#2471A3", lw=2)
        ax.set_xlabel("Iteration", fontsize=13, fontweight="bold")
        ax.set_ylabel(r"$\alpha$", fontsize=13, fontweight="bold")
        ax.set_title(r"$\alpha$ value evolution", fontsize=14, fontweight="bold")
        ax.text(
            0.05,
            0.92,
            rf"$\Sigma (1/\alpha)$ = {sum_inv:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="#AED6F1", alpha=0.9),
        )
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        plt.savefig(
            "../RESULTS/HM_RESULTS/alpha.png",
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close()

    # _barrier()

    # Final ensembles returned as numpy for downstream consumers
    ensemble = _to_np(ensemble)
    ensemblep = _to_np(ensemblep)
    ensemblef = _to_np(ensemblef)

    mean_cost.append(tinumeanprior)
    best_cost.append(tinubestprior)
    chm = int(np.argmin(np.vstack(mean_cost)))

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


def _log_rmse_change(tinuke, tinukebest, prev_mean, prev_best, dist, logger):
    """Log RMSE change direction for mean and best ensemble members."""
    if dist.rank != 0:
        return
    for label, curr, prev in [
        ("mean", tinuke, prev_mean),
        ("best", tinukebest, prev_best),
    ]:
        diff = abs(curr - prev)
        logger.info(f"  PREVIOUS RMSE {label} = {prev}")
        logger.info(f"  CURRENT  RMSE {label} = {curr}")
        if curr < prev:
            logger.info(f"  RMSE {label} ↓ decreased by {diff}")
        elif curr > prev:
            logger.info(f"  RMSE {label} ↑ increased by {diff}")
        else:
            logger.info(f"  RMSE {label} — no change")


def _build_localisation_rank_aware(
    nx, ny, nz, Ne, gas_injectors, producers, injectors, effec, dist, oldfolder, device
):
    """Build the spatial localisation matrix on every rank (deterministic).

    Localisation construction depends only on grid geometry and well
    positions — all of which are identical across ranks. We compute it
    locally on every rank and avoid the broadcast altogether. Plotting
    and disk write happen on rank 0 only.
    """
    # Every rank computes its own copy — deterministic, no comm needed
    locmat_np = Localisation(10, nx, ny, nz, Ne, gas_injectors, producers, injectors)

    # Plot + save only on rank 0
    if dist.rank == 0:
        see1 = locmat_np[: nx * ny * nz, :] * effec
        XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
        look = np.reshape(see1[:, 1], (nx, ny, nz), "F")
        look[look == 0] = np.nan
        plt.figure(figsize=(40, 40))
        for kkt in range(nz):
            plt.subplot(5, 5, int(kkt + 1))
            plt.pcolormesh(XX.T, YY.T, look[:, :, kkt], cmap="jet")
            plt.title(f"Layer {kkt + 1}", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (nx - 1), 0, (ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" Localisation Matrix", fontsize=13)
        Add_marker2(plt, XX, YY, injectors, producers, gas_injectors)
        plt.savefig("../RESULTS/HM_RESULTS/Localisation_matrix.png")
        plt.clf()
        plt.close()

    return locmat_np


def _init_vcae_models(latent_dim, nz, nx, ny, maxK, device, cfg, logger, dist):
    """Load or train VCAE models for permeability and porosity, rank-aware.

    Training and saving happens on rank 0 only. Other ranks wait at a
    barrier and then load the trained weights from disk.
    """
    import os

    models = {}
    for field, fname in [
        ("perm", "../MODELS/perm_vcae.pth"),
        ("poro", "../MODELS/poro_vcae.pth"),
    ]:
        path = to_absolute_path(fname)
        model = VCAE3D(latent_dim).to(device)
        _ = model(torch.randn(1, 1, nz, nx, ny, device=device))  # init lazy layers

        # Decide once on every rank whether the file exists *before* any rank
        # writes it. If it doesn't exist, only rank 0 trains + saves; others wait.
        file_exists_before = os.path.isfile(path)

        if not file_exists_before:
            if dist.rank == 0:
                if logger is not None:
                    logger.info(f"  Training VCAE for {field} …")
                permxi, poroxi, _ = NorneInitialEnsemble(
                    nx, ny, nz, ensembleSize=3000, randomNumber=1.2345e5
                )
                data = (permxi / maxK) if field == "perm" else poroxi
                opt = optim.Adam(
                    model.parameters(),
                    lr=cfg.optimizer.lr,
                    betas=(0.9, 0.999),
                    weight_decay=cfg.optimizer.weight_decay,
                )
                sched = optim.lr_scheduler.ExponentialLR(opt, gamma=cfg.optimizer.gamma)
                model = Train_VCAE(
                    cfg.optimizer.lr,
                    latent_dim,
                    500,
                    100,
                    device,
                    data,
                    nz,
                    nx,
                    ny,
                    model,
                    opt,
                    sched,
                )
                # Make sure the directory exists before saving
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(model.state_dict(), path)
                if logger is not None:
                    logger.info(f"  Saved VCAE for {field} to {path}")

            # Wait for rank 0 to finish training/saving, then everyone loads
            # _barrier()

        # Now every rank loads the weights from disk
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        if logger is not None and dist.rank == 0:
            logger.info(f"  Loaded VCAE for {field} from {path}")

        models[field] = model

    return models["perm"], models["poro"]
