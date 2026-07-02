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

"""End-to-end FP-DDM thermal solve pipeline."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .domain import Domain, Fields, Subdomain, initialize_thermal_fields
from .interfaces import (
    ExchangeInterfaceHandler,
    GradientInterfaceHandler,
    InterfaceConsistencyMonitor,
    ParallelGradientInterfaceHandler,
)
from .observers import (
    AttributeVisualizer,
    MetricsLogger,
    NRMAMetric,
    R2Metric,
    plot_heat_flux,
    visualize_array,
)
from .schwarz import EarlyStopping, PhysicsInformedAdapter, SchwarzMethod
from .solvers import NeuralDomainSolver, ThermalFEMSolver


@dataclass
class RunResult:
    """Outputs from one completed FP-DDM run."""

    grid: list[list[Subdomain]]
    metrics: list[dict[str, float | int]]
    reference: torch.Tensor | None
    elapsed_seconds: float


def _downsample(
    values: torch.Tensor, factor: int = 9, preserve_boundary: bool = True
) -> torch.Tensor:
    pooled = F.max_pool2d(
        values[None, None].float(),
        kernel_size=factor,
        stride=factor,
        ceil_mode=True,
    )[0, 0]
    if preserve_boundary:
        pooled = pooled.clone()
        pooled[0] = values[0, ::factor]
        pooled[-1] = values[-1, ::factor]
        pooled[:, 0] = values[::factor, 0]
        pooled[:, -1] = values[::factor, -1]
    return pooled.to(values)


def _upsample(values: torch.Tensor, factor: int = 9) -> torch.Tensor:
    return F.interpolate(values[None, None], scale_factor=factor, mode="nearest")[0, 0]


def _reference_solve(
    domain: Domain,
    solver: ThermalFEMSolver,
    output_dir: Path,
    *,
    coarse: bool = False,
    render: bool = True,
) -> torch.Tensor | None:
    if domain.rows * domain.columns > 26:
        return None

    conductivity = domain.fields[Fields.CONDUCTIVITY]
    temperature_bc = domain.fields[Fields.TEMPERATURE_BC]
    heat_source = domain.fields[Fields.HEAT_SOURCE]
    original_shape = conductivity.shape
    if coarse:
        factor = 9
        conductivity = _downsample(conductivity, factor)
        temperature_bc = _downsample(temperature_bc, factor)
        heat_source = _downsample(heat_source, factor)
        output_dir = output_dir / "coarse_mesh"
    output_dir.mkdir(parents=True, exist_ok=True)
    temperature = solver.solve(conductivity, temperature_bc, heat_source)[0].cpu()
    if coarse:
        temperature = _upsample(temperature, factor)[
            : original_shape[0], : original_shape[1]
        ]

    np.save(
        output_dir / "inputs.npy",
        torch.stack([conductivity, temperature_bc, heat_source]).cpu().numpy(),
    )
    np.save(output_dir / "true_solution.npy", temperature.numpy())
    if render:
        visualize_array(temperature, output_dir / "output_true.png")
    if render and not coarse:
        plot_heat_flux(
            domain.fields[Fields.CONDUCTIVITY],
            temperature,
            output_dir / "heat_flux.png",
        )
    return temperature


def _build_interface_handler(run_config: Mapping[str, object]):
    name = str(run_config.get("handler", "parallel"))
    learning_rate = float(run_config.get("handler_learning_rate", 10.0))
    if name == "parallel":
        return ParallelGradientInterfaceHandler(learning_rate)
    if name == "gradient":
        return GradientInterfaceHandler(learning_rate)
    if name == "exchange":
        return ExchangeInterfaceHandler(float(run_config.get("exchange_alpha", 1.0)))
    raise ValueError(f"Unknown interface handler: {name}")


def run_fpddm(
    config: Mapping[str, object], *, device: str | torch.device | None = None
) -> RunResult:
    """Run FP-DDM with either finite-volume or PhysicsNeMo FNO local solves."""

    run_config = config["run"]
    domain_config = config["domain"]
    dataset_config = config["dataset"]
    model_config = config["model"]
    visualization_config = config["visualization"]
    fem_config = config["fem"]
    if not all(
        isinstance(section, Mapping)
        for section in (
            run_config,
            domain_config,
            dataset_config,
            model_config,
            visualization_config,
            fem_config,
        )
    ):
        raise TypeError("FP-DDM config sections must be mappings")

    output_dir = Path(str(run_config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    domain = Domain(
        rows=int(domain_config["rows"]),
        columns=int(domain_config["columns"]),
        width=int(domain_config["width"]),
        height=int(domain_config["height"]),
        overlap=int(domain_config["overlap"]),
    )
    initialize_thermal_fields(
        domain,
        domain_config["outer_boundary"],
        dataset_config,
        device="cpu",
    )
    reference_solver = ThermalFEMSolver(
        batch_size=int(fem_config.get("batch_size", 10)),
        max_iter=int(fem_config.get("max_iter", 1000)),
        tolerance=float(fem_config.get("tolerance", 1.0e-12)),
        device=device,
    )
    render = bool(run_config.get("visualize", True))
    if bool(domain_config.get("coarse_initialization", False)):
        interior = _reference_solve(
            domain,
            reference_solver,
            output_dir,
            coarse=True,
            render=render,
        )
        if interior is not None:
            initialize_thermal_fields(
                domain,
                domain_config["outer_boundary"],
                dataset_config,
                interior_temperature=interior,
                device="cpu",
            )
    reference = (
        _reference_solve(domain, reference_solver, output_dir, render=render)
        if bool(run_config.get("ground_truth", True))
        else None
    )

    solver_name = str(run_config.get("solver", "fem"))
    if solver_name == "fem":
        local_solver = reference_solver
        model_updates = []
    elif solver_name == "fno":
        checkpoint_dir = run_config.get("checkpoint_dir")
        if not checkpoint_dir:
            raise ValueError("run.checkpoint_dir is required for the FNO solver")
        local_solver = NeuralDomainSolver(
            model_config,
            checkpoint_dir=Path(str(checkpoint_dir)),
            batch_size=int(run_config.get("batch_size", 128)),
            device=device,
        )
        adapter = PhysicsInformedAdapter(
            local_solver,
            steps=int(run_config.get("ttt_steps", 0)),
            learning_rate=float(run_config.get("ttt_learning_rate", 5.0e-5)),
            batch_size=int(run_config.get("ttt_batch_size", 1000)),
            pde_weight=float(run_config.get("ttt_pde_weight", 1.0)),
            boundary_weight=float(run_config.get("ttt_boundary_weight", 1.0)),
        )
        model_updates = [adapter]
    else:
        raise ValueError(f"Unknown local solver: {solver_name}")

    interface_handler = _build_interface_handler(run_config)
    consistency = InterfaceConsistencyMonitor()
    observers = []
    if bool(run_config.get("visualize", True)):
        observers.extend(
            [
                AttributeVisualizer(
                    Fields.TEMPERATURE,
                    output_dir,
                    save_name="temperature",
                    fps=int(visualization_config.get("fps", 10)),
                    vmin=float(visualization_config.get("temperature_min", 300.0)),
                    vmax=float(visualization_config.get("temperature_max", 400.0)),
                ),
                AttributeVisualizer(
                    Fields.CONDUCTIVITY,
                    output_dir,
                    save_name="conductivity",
                    fps=int(visualization_config.get("fps", 10)),
                    vmin=float(visualization_config.get("conductivity_min", 0.5)),
                    vmax=float(visualization_config.get("conductivity_max", 200.0)),
                ),
            ]
        )
    if reference is not None:
        observers.extend(
            [
                NRMAMetric(reference, Fields.TEMPERATURE),
                R2Metric(reference, Fields.TEMPERATURE),
            ]
        )

    metrics = MetricsLogger(output_dir / "metrics.csv")
    early_stopping = EarlyStopping(
        consistency.get_metric,
        tolerance=float(run_config.get("tolerance", 1.0e-9)),
        max_iterations=int(run_config.get("max_iterations", 50)),
        output_dir=output_dir if render else None,
        patience=int(run_config.get("patience", 10000)),
    )
    method = SchwarzMethod(
        output_dir,
        metrics,
        early_stopper=early_stopping,
        model_update_handlers=model_updates,
        domain_solve_handlers=[local_solver],
        interface_handlers=[interface_handler, consistency],
        observation_handlers=observers,
    )
    grid = method.run(domain)
    elapsed = time.perf_counter() - started
    summary = {
        "solver": solver_name,
        "iterations": len(metrics.rows),
        "elapsed_seconds": elapsed,
        "converged": early_stopping.converged,
        "metrics": metrics.rows[-1] if metrics.rows else {},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)
    return RunResult(grid, metrics.rows, reference, elapsed)
