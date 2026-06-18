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
Typed configuration dataclasses for the FVM-vs-surrogate comparison workflow.

These bundles let ``compare_and_analyze_results`` take ~8 grouped arguments
instead of 68 positional ones, mirroring the dataclass pattern already used
by the inverse workflow (``inverse.utils.inverse_config``) and the forward
training loop (``forward.utils.sequential.training_config``).

Each dataclass groups parameters that travel together through the pipeline,
which makes call sites readable, gives IDEs proper completion, and lets
static analysers verify the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompareGrid:
    """Grid dimensions, time-step indexing, and ensemble size."""

    nx: int = 0
    ny: int = 0
    nz: int = 0
    steppi: int = 0
    steppi_indices: Any = None
    Ne: int = 0


@dataclass
class CompareWells:
    """Producer/injector counts, locations, names, and completion data."""

    N_pr: int = 0
    N_injw: int = 0
    N_injg: int = 0
    lenwels: int = 0
    injectors: list[Any] = field(default_factory=list)
    producers: list[Any] = field(default_factory=list)
    gas_injectors: list[Any] = field(default_factory=list)
    well_names: list[str] = field(default_factory=list)
    columns: Any = None
    compdat_data: Any = None


@dataclass
class CompareNorms:
    """Normalisation min/max bounds for every input/output channel."""

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


@dataclass
class CompareSurrogate:
    """Surrogate model handles and CCR-specific scalar settings."""

    models: Any = None
    degg: int = 3
    experts: int = 5
    inn: Any = None


@dataclass
class CompareFlow:
    """Active-cell masks and per-phase rate arrays + simulation time vector."""

    active_cells_ensemble: Any = None
    active_mask_3d: Any = None
    awater: Any = None
    agas: Any = None
    aoil: Any = None
    aqq: Any = None
    Time: Any = None


@dataclass
class CompareRuntime:
    """Hydra config, compute device, and runtime path/job settings."""

    cfg: Any = None
    device: Any = None
    num_cores: int = 1
    oldfolder: str = ""
    folderr: str = ""
    output_variables: list[str] = field(default_factory=list)
    well_measurements: Any = None


@dataclass
class CompareFields:
    """Predicted vs. ground-truth 3-D pressure and saturation fields."""

    pressure_pred: Any = None
    pressure_true: Any = None
    water_pred: Any = None
    water_true: Any = None
    oil_pred: Any = None
    oil_true: Any = None
    gas_pred: Any = None
    gas_true: Any = None


@dataclass
class CompareWellResults:
    """Predicted vs. observed concatenated well-rate matrices."""

    ouut_peacemann: Any = None
    out_fcn_true: Any = None


@dataclass
class CompareTiming:
    """Wall-clock execution times for the surrogate vs. the reservoir simulator."""

    physicsnemo_time: float = 0.0
    flow_time: float = 0.0
