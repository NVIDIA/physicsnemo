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

"""Finite-volume and PhysicsNeMo FNO local solvers for FP-DDM."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import torch

from physicsnemo.utils import load_checkpoint

from .domain import BC_FIELDS, INPUT_FIELDS, OUTPUT_FIELDS, Fields, Subdomain
from .model import ThermalPINO, thermal_losses
from .thermal import Heat2DSolver


class DomainSolver:
    """Build batched five-channel model inputs from FP-DDM subdomains."""

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        input_fields: tuple[Fields, ...] = INPUT_FIELDS,
        boundary_fields: tuple[Fields, ...] = BC_FIELDS,
        output_fields: tuple[Fields, ...] = OUTPUT_FIELDS,
    ) -> None:
        """Configure field mappings and the local-solver device."""

        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.input_fields = input_fields
        self.boundary_fields = boundary_fields
        self.output_fields = output_fields
        self._coordinate_cache: dict[tuple, torch.Tensor] = {}

    def _coordinates(self, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        key = (height, width, self.device.type, self.device.index, dtype)
        if key not in self._coordinate_cache:
            y_axis = torch.linspace(0.0, 1.0, height, device=self.device, dtype=dtype)
            x_axis = torch.linspace(0.0, 1.0, width, device=self.device, dtype=dtype)
            yy, xx = torch.meshgrid(y_axis, x_axis, indexing="ij")
            self._coordinate_cache[key] = torch.stack([xx, yy], dim=0)
        return self._coordinate_cache[key]

    def _input_from_subdomain(self, subdomain: Subdomain) -> torch.Tensor:
        fields = []
        for field in self.input_fields:
            values = subdomain.fields[field]
            if field in self.boundary_fields:
                values = values.clone()
                values[..., 1:-1, 1:-1] = 0.0
            fields.append(values)
        return self._input_from_fields(fields)

    def _input_from_fields(self, fields: list[torch.Tensor]) -> torch.Tensor:
        height, width = fields[0].shape[-2:]
        channels = [self._coordinates(height, width, torch.float32)]
        channels.extend(
            field.to(self.device, torch.float32).reshape(-1, height, width)
            for field in fields
        )
        return torch.cat(channels, dim=0)

    def iter_batches(
        self, subdomains: list[Subdomain], batch_size: int
    ) -> Iterable[tuple[list[Subdomain], torch.Tensor]]:
        """Yield subdomain chunks and their batched model inputs."""

        for start in range(0, len(subdomains), batch_size):
            chunk = subdomains[start : start + batch_size]
            inputs = torch.stack(
                [self._input_from_subdomain(subdomain) for subdomain in chunk]
            )
            yield chunk, inputs


class ThermalFEMSolver(DomainSolver):
    """Use the matrix-free thermal solver on every local FP-DDM patch."""

    def __init__(
        self,
        batch_size: int = 15,
        *,
        max_iter: int = 1000,
        tolerance: float = 1.0e-12,
        device: str | torch.device | None = None,
    ) -> None:
        """Configure local solve batching and convergence."""

        super().__init__(device=device)
        self.batch_size = batch_size
        self.solver = Heat2DSolver(
            max_iter=max_iter, tolerance=tolerance, device=self.device
        )

    def model(self, inputs: torch.Tensor) -> torch.Tensor:
        """Solve a batch of five-channel local thermal inputs."""

        batch, _, height, width = inputs.shape
        conductivity = inputs[:, 2]
        temperature_bc = inputs[:, 3]
        heat_source = inputs[:, 4]
        dirichlet = torch.ones(
            (batch, height, width), dtype=torch.bool, device=inputs.device
        )
        dirichlet[:, 1:-1, 1:-1] = False
        neumann = torch.zeros_like(dirichlet)
        temperature = self.solver.solve(
            conductivity,
            heat_source,
            temperature_bc,
            dirichlet,
            neumann,
        )
        return temperature[:, None]

    @torch.no_grad()
    def solve_batch(self, subdomains: list[Subdomain]) -> None:
        """Solve and write temperature for every supplied subdomain."""

        for chunk, inputs in self.iter_batches(subdomains, self.batch_size):
            output = self.model(inputs).cpu()
            for index, subdomain in enumerate(chunk):
                subdomain.fields[Fields.TEMPERATURE] = output[index, 0].clone()

    @torch.no_grad()
    def solve(
        self,
        conductivity: torch.Tensor,
        temperature_bc: torch.Tensor,
        heat_source: torch.Tensor,
    ) -> torch.Tensor:
        """Solve one full-domain thermal system from unbatched fields."""

        inputs = self._input_from_fields([conductivity, temperature_bc, heat_source])
        return self.model(inputs[None])[:, 0]


class NeuralDomainSolver(DomainSolver):
    """Use a checkpointed PhysicsNeMo FNO as the local FP-DDM solver."""

    def __init__(
        self,
        model_config: Mapping[str, object],
        checkpoint_dir: str | Path | None = None,
        *,
        batch_size: int = 128,
        device: str | torch.device | None = None,
    ) -> None:
        """Build the thermal PINO and optionally restore a checkpoint."""

        super().__init__(device=device)
        self.batch_size = batch_size
        self.model = ThermalPINO(model_config).to(self.device)
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            if not checkpoint_dir.is_dir():
                raise FileNotFoundError(
                    f"Checkpoint directory not found: {checkpoint_dir}"
                )
            if not any(checkpoint_dir.glob("ThermalPINO.*")):
                raise FileNotFoundError(
                    f"No ThermalPINO checkpoint found in: {checkpoint_dir}"
                )
            load_checkpoint(checkpoint_dir, models=self.model, device=self.device)
        self.model.set_dimensional_mode(True, True)
        self.model.set_boundary_enforcement(True)
        self.model.eval()
        self.loss = 0.0

    def set_dimensional_mode(self, input_enabled: bool, output_enabled: bool) -> None:
        """Set dimensional conversion on the wrapped thermal PINO."""

        self.model.set_dimensional_mode(input_enabled, output_enabled)

    @torch.no_grad()
    def solve_batch(self, subdomains: list[Subdomain]) -> None:
        """Predict and write temperature for every supplied subdomain."""

        self.model.eval()
        physics_rmse = 0.0
        batches = 0
        for chunk, inputs in self.iter_batches(subdomains, self.batch_size):
            output = self.model(inputs)
            normalized_inputs = self.model.normalize_input(inputs)
            normalized_output = (
                output - self.model.temperature_reference
            ) / self.model.temperature_scale
            pde_loss, _ = thermal_losses(
                normalized_output,
                normalized_inputs,
                source_scale=self.model.nondimensional_source_scale,
            )
            physics_rmse += float(pde_loss.sqrt())
            batches += 1
            output = output.cpu()
            inputs_cpu = inputs.cpu()
            for index, subdomain in enumerate(chunk):
                temperature = output[index, 0].clone()
                for boundary_field in self.boundary_fields:
                    channel = self.input_fields.index(boundary_field) + 2
                    boundary = inputs_cpu[index, channel]
                    temperature[[0, -1], :] = boundary[[0, -1], :]
                    temperature[:, [0, -1]] = boundary[:, [0, -1]]
                subdomain.fields[Fields.TEMPERATURE] = temperature
        self.loss = physics_rmse / max(batches, 1)

    def get_metric(self) -> tuple[str, float]:
        """Return the latest test-time physics loss."""

        return "loss_pde", self.loss
