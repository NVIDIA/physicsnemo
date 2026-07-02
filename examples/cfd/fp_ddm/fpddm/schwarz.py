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

"""Schwarz iteration and physics-guided test-time adaptation for FP-DDM."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import torch
from tqdm import tqdm

from .domain import Domain, Subdomain
from .model import thermal_losses
from .solvers import NeuralDomainSolver


class PhysicsInformedAdapter:
    """Adapt the local FNO using PDE and boundary losses during FP-DDM."""

    def __init__(
        self,
        solver: NeuralDomainSolver,
        *,
        steps: int = 10,
        learning_rate: float = 5.0e-5,
        batch_size: int = 8,
        pde_weight: float = 1.0,
        boundary_weight: float = 1.0,
    ) -> None:
        """Configure adaptation steps, batching, weights, and learning rate."""

        self.solver = solver
        self.steps = steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.pde_weight = pde_weight
        self.boundary_weight = boundary_weight

    def __call__(self, subdomains: list[Subdomain]) -> None:
        """Run physics-guided adaptation and restore inference model modes."""

        if self.steps <= 0 or self.learning_rate <= 0.0:
            return

        model = self.solver.model
        previous_modes = (
            model.use_dimensional_input,
            model.use_dimensional_output,
            model.enforce_boundary,
            model.training,
        )
        model.set_boundary_enforcement(False)
        model.set_dimensional_mode(False, False)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        try:
            for _ in range(self.steps):
                optimizer.zero_grad(set_to_none=True)
                for _, inputs in self.solver.iter_batches(subdomains, self.batch_size):
                    normalized = model.normalize_input(inputs.detach())
                    prediction = model(normalized)
                    pde_loss, boundary_loss = thermal_losses(
                        prediction,
                        normalized,
                        source_scale=model.nondimensional_source_scale,
                    )
                    total = (
                        self.pde_weight * pde_loss
                        + self.boundary_weight * boundary_loss
                    )
                    total.backward()
                optimizer.step()

            total_loss = 0.0
            batches = 0
            model.eval()
            with torch.no_grad():
                for _, inputs in self.solver.iter_batches(subdomains, self.batch_size):
                    normalized = model.normalize_input(inputs)
                    prediction = model(normalized)
                    pde_loss, boundary_loss = thermal_losses(
                        prediction,
                        normalized,
                        source_scale=model.nondimensional_source_scale,
                    )
                    total_loss += float(
                        self.pde_weight * pde_loss
                        + self.boundary_weight * boundary_loss
                    )
                    batches += 1
            self.solver.loss = total_loss / max(batches, 1)
        finally:
            model.set_dimensional_mode(previous_modes[0], previous_modes[1])
            model.set_boundary_enforcement(previous_modes[2])
            model.train(previous_modes[3])


class EarlyStopping:
    """Stop Schwarz iteration on convergence, iteration limit, or patience."""

    def __init__(
        self,
        metric_fn: Callable[[], tuple[str, float]],
        *,
        tolerance: float = 1.0e-4,
        max_iterations: int = 1000,
        output_dir: str | Path | None = None,
        patience: int | None = None,
    ) -> None:
        """Configure convergence and history output."""

        self.metric_fn = metric_fn
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.patience = patience
        self.iteration = 0
        self.current_loss = float("inf")
        self.best_loss = float("inf")
        self.patience_count = 0
        self.converged = False
        self.loss_history: list[float] = []

    @torch.no_grad()
    def step(self) -> bool:
        """Record the current metric and return whether iteration should stop."""

        self.iteration += 1
        self.current_loss = float(self.metric_fn()[1])
        self.loss_history.append(self.current_loss)
        if self.current_loss < self.best_loss:
            self.best_loss = self.current_loss
            self.patience_count = 0
        elif self.patience is not None:
            self.patience_count += 1

        should_stop = (
            self.current_loss < self.tolerance
            or self.iteration >= self.max_iterations
            or (self.patience is not None and self.patience_count >= self.patience)
        )
        if should_stop:
            self.converged = self.current_loss < self.tolerance
            if self.output_dir is not None:
                self._plot_history()
        return should_stop

    def _plot_history(self) -> None:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        axis.semilogy(range(1, len(self.loss_history) + 1), self.loss_history)
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Interface RMSE")
        axis.set_title("FP-DDM convergence")
        axis.grid(True)
        figure.savefig(self.output_dir / "convergence_history.png")
        plt.close(figure)


class SchwarzMethod:
    """Orchestrate model adaptation, local solves, and interface updates."""

    def __init__(
        self,
        output_dir: str | Path,
        metric_logger,
        *,
        early_stopper: EarlyStopping,
        model_update_handlers: Iterable[Callable] = (),
        domain_solve_handlers: Iterable[object] = (),
        interface_handlers: Iterable[Callable] = (),
        observation_handlers: Iterable[Callable] = (),
    ) -> None:
        """Configure the ordered callbacks for each Schwarz iteration."""

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metric_logger = metric_logger
        self.early_stopper = early_stopper
        self.model_update_handlers = list(model_update_handlers)
        self.domain_solve_handlers = list(domain_solve_handlers)
        self.interface_handlers = list(interface_handlers)
        self.observation_handlers = list(observation_handlers)
        self.handlers = (
            [metric_logger]
            + self.model_update_handlers
            + self.domain_solve_handlers
            + self.interface_handlers
            + self.observation_handlers
        )
        for handler in self.handlers[1:]:
            self.metric_logger.register_metric(handler)

    def execute_iteration(
        self,
        subdomains: list[Subdomain],
        grid: list[list[Subdomain]],
    ) -> None:
        """Execute one ordered FP-DDM iteration."""

        for handler in self.model_update_handlers:
            handler(subdomains)
        for solver in self.domain_solve_handlers:
            solver.solve_batch(subdomains)
        for handler in self.interface_handlers:
            handler(subdomains)
        for observer in self.observation_handlers:
            observer(grid)

    def finalize(self) -> None:
        """Finalize metric and visualization handlers."""

        for handler in self.handlers:
            finalize = getattr(handler, "finalize", None)
            if callable(finalize):
                finalize()

    def run(self, domain: Domain) -> list[list[Subdomain]]:
        """Run Schwarz iteration and return the final subdomain grid."""

        grid = domain.build_subdomains()
        subdomains = [subdomain for row in grid for subdomain in row]
        progress = tqdm(
            range(self.early_stopper.max_iterations), desc="FP-DDM iteration"
        )
        for _ in progress:
            self.execute_iteration(subdomains, grid)
            progress.set_postfix(self.metric_logger.step())
            if self.early_stopper.step():
                break
        self.finalize()
        return grid
