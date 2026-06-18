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
             TRAINING LOOP CONFIGURATION DATACLASSES
=====================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable


@dataclass
class SurrogateModels:
    """Per-target surrogate models and their best-so-far snapshots."""

    composite_model: Any
    surrogate_pressure: Any | None = None
    surrogate_gas: Any | None = None
    surrogate_saturation: Any | None = None
    surrogate_oil: Any | None = None
    surrogate_peacemann: Any | None = None
    best_pressure: Any | None = None
    best_gas: Any | None = None
    best_saturation: Any | None = None
    best_oil: Any | None = None
    best_peacemann: Any | None = None


@dataclass
class DataLoaders:
    """Training and validation data loaders."""

    labelled_loader_train: Any
    labelled_loader_testt: Any
    labelled_loader_trainp: Any | None = None
    labelled_loader_testtp: Any | None = None


@dataclass
class ModelKeys:
    """Input/output tensor key lists for the composite model."""

    output_variables: list[str]
    input_keys: list[Any]
    input_keys_peacemann: list[Any]
    output_keys_pressure: list[Any] = field(default_factory=list)
    output_keys_gas: list[Any] = field(default_factory=list)
    output_keys_saturation: list[Any] = field(default_factory=list)
    output_keys_oil: list[Any] = field(default_factory=list)
    output_keys_peacemann: list[Any] = field(default_factory=list)


@dataclass
class PhysicsParams:
    """Physical and dimensional constants forwarded to the step functions."""

    # Grid dimensions
    nx: int
    ny: int
    nz: int
    steppi: int
    # Well / connection counts
    N_pr: int
    lenwels: int
    neededM: Any
    neededMx: Any
    neededMxt: Any
    # Fluid viscosities and formation-volume factors
    UO: float
    BO: float
    UW: float
    BW: float
    DZ: float
    RE: float
    # PVT / bubble-point
    params: Any
    p_bub: float
    p_atm: float
    CFO: float
    # Relative permeability tables
    Relperm: Any
    SWI: float
    SWR: float
    SWOW: Any
    SWOG: Any
    params1_swow: Any
    params2_swow: Any
    params1_swog: Any
    params2_swog: Any
    # Step-function controls
    pde_method: Any = None
    unique_entries: Any = None
    time_physics: Any = None


@dataclass
class NormParams:
    """Normalisation bounds for inputs, outputs, permeability, and pressure."""

    max_inn_fcnx: Any
    max_out_fcnx: Any
    max_inn_fcn: Any
    max_out_fcn: Any
    target_min: Any
    target_max: Any
    minK: float
    maxK: float
    minP: float
    maxP: float
    # Time / rate normalisation bounds
    maxT: Any = None
    maxQ: Any = None
    maxQw: Any = None
    maxQg: Any = None


@dataclass
class Optimizers:
    """Per-target optimizers."""

    optimizer_peacemann: Any
    optimizer_pressure: Any | None = None
    optimizer_saturation: Any | None = None
    optimizer_oil: Any | None = None
    optimizer_gas: Any | None = None


@dataclass
class Schedulers:
    """Per-target learning-rate schedulers."""

    scheduler_peacemann: Any
    scheduler_pressure: Any | None = None
    scheduler_saturation: Any | None = None
    scheduler_oil: Any | None = None
    scheduler_gas: Any | None = None


@dataclass
class TrainingState:
    """Step functions and mutable metric dicts for one training loop."""

    training_step: Callable
    validation_step: Callable
    training_step_metrics: dict
    val_step_metrics: dict
