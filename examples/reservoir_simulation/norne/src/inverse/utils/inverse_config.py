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

# ruff: noqa: RUF002
# Greek letters appear in docstrings to match conventional algorithm notation;
# ruff's confusable-character rule is suppressed for the file.
"""
SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0

Typed configuration dataclasses for the inverse problem workflow.

Each dataclass groups logically related parameters that are passed to the four
main pipeline functions: ``generate_ensemble``, ``run_history_matching_loop``,
``process_final_results``, and ``plot_percentile_models``.  Using dataclasses
instead of 25-62 positional arguments makes call sites readable and lets IDEs
provide completion and type checking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GridConfig:
    """Reservoir grid dimensions and time-step indexing.

    Parameters
    ----------
    nx : int
        Number of grid cells in the x-direction.
    ny : int
        Number of grid cells in the y-direction.
    nz : int
        Number of grid cells in the z-direction (layers).
    steppi : int
        Total number of report time steps used by the surrogate model.
    steppi_indices : numpy.ndarray
        1-D integer array of shape (steppi,) mapping surrogate time steps
        to simulator time-step indices.
    """

    nx: int = 0
    ny: int = 0
    nz: int = 0
    steppi: int = 0
    steppi_indices: Any = None


@dataclass
class NormBounds:
    """Normalisation scalers loaded from ``conversions.mat``.

    These bounds are used to map simulator outputs and neural-network inputs /
    outputs to the [target_min, target_max] interval and back.

    Parameters
    ----------
    minK, maxK : numpy.ndarray
        Minimum and maximum permeability (mD) in the training dataset.
    minT, maxT : numpy.ndarray
        Minimum and maximum time values.
    minP, maxP : numpy.ndarray
        Minimum and maximum reservoir pressure (psi or bar).
    minQ, maxQ : numpy.ndarray
        Minimum and maximum oil production rate (STB/day or Sm³/day).
    minQw, maxQw : numpy.ndarray
        Minimum and maximum water injection/production rate.
    minQg, maxQg : numpy.ndarray
        Minimum and maximum gas rate.
    min_inn_fcn, max_inn_fcn : numpy.ndarray
        Per-channel minimum/maximum for the primary surrogate input tensor.
    min_out_fcn, max_out_fcn : numpy.ndarray
        Per-channel minimum/maximum for the primary surrogate output tensor.
    min_inn_fcn2, max_inn_fcn2 : numpy.ndarray
        Per-channel min/max for the secondary surrogate (Peacemann) inputs.
    min_out_fcn2, max_out_fcn2 : numpy.ndarray
        Per-channel min/max for the secondary surrogate (Peacemann) outputs.
    target_min : float
        Lower bound of the normalised output range (default 0.01).
    target_max : float
        Upper bound of the normalised output range (default 1.0).
    """

    minK: Any = None
    maxK: Any = None
    minT: Any = None
    maxT: Any = None
    minP: Any = None
    maxP: Any = None
    minQ: Any = None
    maxQ: Any = None
    minQw: Any = None
    maxQw: Any = None
    minQg: Any = None
    maxQg: Any = None
    min_inn_fcn: Any = None
    max_inn_fcn: Any = None
    min_out_fcn: Any = None
    max_out_fcn: Any = None
    min_inn_fcn2: Any = None
    max_inn_fcn2: Any = None
    min_out_fcn2: Any = None
    max_out_fcn2: Any = None
    target_min: float = 0.01
    target_max: float = 1.0


@dataclass
class PermBounds:
    """Prior bounds for permeability and porosity used during ensemble generation.

    Parameters
    ----------
    High_K : float
        Upper bound of the log-permeability prior (Gaussian truncation).
    Low_K : float
        Lower bound of the log-permeability prior.
    High_K1 : float
        Upper hard-clipping bound applied after mapping (used in DCT / VAE
        parametrisation).
    Low_K1 : float
        Lower hard-clipping bound applied after mapping.
    High_P : float
        Upper bound of the porosity prior.
    Low_P : float
        Lower bound of the porosity prior.
    """

    High_K: float = 0.0
    Low_K: float = 0.0
    High_K1: float = 0.0
    Low_K1: float = 0.0
    High_P: float = 0.0
    Low_P: float = 0.0


@dataclass
class EnsembleSetup:
    """Ensemble geometry, masks, and supporting arrays.

    Parameters
    ----------
    Ne : int
        Ensemble size (number of realisations).
    N_ens : int
        Alias for Ne kept for backward compatibility with inner functions.
    effective : numpy.ndarray
        Active-cell mask reshaped to (nx*ny*nz, 1) in Fortran order.
    effec : numpy.ndarray
        Active-cell mask flattened to a column vector of shape (nx*ny*nz, 1).
    active_cells_ensemble : numpy.ndarray
        Active-cell mask reshaped to (nx, ny, nz) in Fortran order.
    active_mask_3d : numpy.ndarray
        Same mask as active_cells_ensemble; kept as a named alias used by
        plotting helpers.
    quant_big : numpy.ndarray
        Quantile grid used by the REKI observation operator (shape depends on
        lenwels and steppi).
    rows_to_remove : list[int]
        Row indices to drop from the observation vector (inactive wells).
    indii : numpy.ndarray or list[int]
        Column indices to delete from the static ensemble arrays when the
        ensemble size differs from the training set.
    """

    Ne: int = 0
    N_ens: int = 0
    effective: Any = None
    effec: Any = None
    active_cells_ensemble: Any = None
    active_mask_3d: Any = None
    rows_to_remove: Any = None
    indii: Any = None


@dataclass
class WellConfig:
    """Well identifiers, counts, and completion data.

    Parameters
    ----------
    producers : list[tuple]
        List of producer well records from the COMPDAT section; each entry is a
        tuple of (I, J, K1, K2, well_name, ...).
    injectors : list[tuple]
        Water injector well records (same format as producers).
    gas_injectors : list[tuple]
        Gas injector well records.
    well_names : list[str]
        Names of producer wells in the order returned by ``read_compdats2``.
    N_pr : int
        Number of producer wells.
    N_injw : int
        Number of water injector wells.
    N_injg : int
        Number of gas injector wells.
    lenwels : int
        Total number of measurement channels
        (= len(cfg.custom.well_measurements)).
    compdat_data : dict
        Parsed COMPDAT data for producer wells, keyed by well name.
    """

    producers: list[Any] = field(default_factory=list)
    injectors: list[Any] = field(default_factory=list)
    gas_injectors: list[Any] = field(default_factory=list)
    well_names: list[str] = field(default_factory=list)
    N_pr: int = 0
    N_injw: int = 0
    N_injg: int = 0
    lenwels: int = 0
    compdat_data: Any | None = None


@dataclass
class SurrogateConfig:
    """Surrogate model references and evaluation settings.

    Parameters
    ----------
    models : object
        Loaded PhysicsNeMo neural-operator model(s); the exact type depends on
        ``pred_type`` (FNO / PINO / Transolver).
    Trainmoe : str
        Surrogate mode selector: ``"MoE"`` for Mixture-of-Experts CCR, or
        ``"FNO"`` for a plain FNO forward pass.
    pred_type : str
        Model architecture tag returned by ``setup_models_and_data``.
    degg : int
        Polynomial degree used by the CCR Gaussian-process kernel.
    experts : int
        Number of expert sub-models in the Mixture-of-Experts setup.
    num_cores : int
        Number of CPU cores used for parallelised ensemble evaluations.
    """

    models: Any = None
    Trainmoe: str = "FNO"
    pred_type: str = ""
    degg: int = 3
    experts: int = 5
    num_cores: int = 1


@dataclass
class InversionParams:
    """REKI / ES-MDA algorithm settings and parametrisation options.

    Parameters
    ----------
    Termm : int
        Maximum number of history-matching iterations (aREKI stops earlier
        when Σ(1/α) ≥ 1).
    Do_parametrisation : str
        ``"Yes"`` to reparametrise ensemble members via DCT or VAE before each
        Kalman update; ``"No"`` to update in physical space.
    Do_param_method : str
        Parametrisation method when Do_parametrisation is ``"Yes"``:
        ``"DCT"`` or ``"VAE"``.
    do_localisation : str
        ``"Yes"`` to apply covariance localisation during the Kalman update.
    size1 : int
        Number of DCT coefficients retained in the x-direction.
    size2 : int
        Number of DCT coefficients retained in the y-direction.
    noise_level : float
        Fractional observation noise added to the ensemble (0–1 range;
        e.g. 0.05 for 5 % noise).
    Deccor : str
        ``"Yes"`` to apply ensemble decorrelation before the first iteration.
    """

    Termm: int = 20
    Do_parametrisation: str = "No"
    Do_param_method: str = "DCT"
    do_localisation: str = "Yes"
    size1: int = 0
    size2: int = 0
    noise_level: float = 0.05
    Deccor: str = "No"


@dataclass
class TimeArrays:
    """Time-series arrays shared across ensemble and plotting functions.

    Parameters
    ----------
    Time : numpy.ndarray
        Simulation time values (days) for all report steps, shape (steppi,).
    Time_unie1 : numpy.ndarray
        Unique time values used for plotting well responses.
    timestep : numpy.ndarray or int
        Time-step size(s) used internally by the data-extraction helpers.
    """

    Time: Any = None
    Time_unie1: Any = None
    timestep: Any = None


@dataclass
class EnsembleRuntime:
    """Runtime context bundle for the ensemble-setup pipeline.

    Bundles the Hydra config, distributed manager, compute device, and
    user-facing toggles that travel together through ``setup_models_and_data``
    and the surrogate-loading helpers.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    dist : DistributedManager
        PhysicsNeMo distributed-execution context.
    device : torch.device or str
        Compute device for surrogate inference.
    oldfolder : str
        Original working directory at job launch (used for chdir-pair restoration).
    DEFAULT : str
        ``"Yes"`` to use the recommended hard-coded inverse-problem defaults,
        otherwise ``"No"`` to read the values from cfg.
    excel : int
        ``1`` if the user provided an Excel results template, else ``0``.
    TEMPLATEFILE : dict
        Mutable run-summary dictionary populated by the pipeline for the
        Excel/Word reporting step.
    """

    cfg: Any = None
    dist: Any = None
    device: Any = None
    oldfolder: str = ""
    DEFAULT: str = "Yes"
    excel: int = 0
    TEMPLATEFILE: Any = None


@dataclass
class PriorEnsembles:
    """Prior ensemble realisations for permeability, porosity, and fault multipliers.

    Parameters
    ----------
    perm : numpy.ndarray
        Prior permeability ensemble of shape ``(n_cells, Ne)``.
    poro : numpy.ndarray
        Prior porosity ensemble of shape ``(n_cells, Ne)``.
    fault : numpy.ndarray
        Prior fault-multiplier ensemble of shape ``(n_faults, Ne)``.
    """

    perm: Any = None
    poro: Any = None
    fault: Any = None


@dataclass
class FlowArrays:
    """Per-cell flow-rate arrays for water, gas, and oil phases.

    These arrays are built by ``get_dyna2`` from per-well completion rates and
    are passed to the surrogate forward model as source-term tensors.

    Parameters
    ----------
    awater : numpy.ndarray
        Water injection rate array, shape (steppi, nx, ny, nz).
    agas : numpy.ndarray
        Gas injection rate array, shape (steppi, nx, ny, nz).
    aoil : numpy.ndarray
        Oil production rate array, shape (steppi, nx, ny, nz).
    aqq : numpy.ndarray
        Total rate array (awater + agas + aoil), shape (steppi, nx, ny, nz).
    """

    awater: Any = None
    agas: Any = None
    aoil: Any = None
    aqq: Any = None
