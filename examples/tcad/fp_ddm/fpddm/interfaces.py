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

"""Interface-update algorithms and overlap metrics for FP-DDM."""

from __future__ import annotations

import torch

from .domain import DIRECTIONS, Fields, Subdomain


def _zero_interior(values: torch.Tensor, depth: int = 1) -> None:
    with torch.no_grad():
        values[..., depth:-depth, depth:-depth] = 0.0


def _domain_interface_mse(
    subdomain: Subdomain,
    solution_field: Fields = Fields.TEMPERATURE,
    boundary_field: Fields = Fields.TEMPERATURE_BC,
    boundary_depth: int = 1,
) -> torch.Tensor:
    boundary = subdomain.fields[boundary_field]
    squared_error = boundary.new_tensor(0.0)
    count = 0
    for direction in DIRECTIONS:
        neighbor = subdomain.neighbors[direction]
        if neighbor is None:
            continue
        current = boundary[subdomain.boundary_slice(direction, boundary_depth)]
        target = (
            neighbor.fields[solution_field][
                neighbor.overlap_slice(direction.opposite(), boundary_depth)
            ]
            .detach()
            .to(current.device)
        )
        squared_error += (current - target).square().sum()
        count += current.numel()
    return squared_error / max(count, 1)


def _make_optimizer(
    name: str, parameters: list[torch.Tensor], learning_rate: float
) -> torch.optim.Optimizer:
    normalized = name.lower()
    if normalized == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if normalized == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate)
    if normalized in {"momentum", "sgd_momentum", "heavyball"}:
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)
    if normalized in {"nesterov", "sgd_nesterov"}:
        return torch.optim.SGD(
            parameters, lr=learning_rate, momentum=0.9, nesterov=True
        )
    if normalized == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=learning_rate)
    raise ValueError(f"Unknown interface optimizer: {name}")


class GradientInterfaceHandler:
    """Optimize each subdomain boundary independently against its neighbors."""

    def __init__(self, learning_rate: float = 1.0e-3) -> None:
        """Configure the boundary optimizer learning rate."""

        self.learning_rate = learning_rate
        self.total_loss = 100.0

    def __call__(self, subdomains: list[Subdomain]) -> None:
        """Apply one sequential interface optimization step."""

        rmse = 0.0
        for subdomain in subdomains:
            temperature_bc = subdomain.fields[Fields.TEMPERATURE_BC]
            _zero_interior(temperature_bc)
            if subdomain.optimizer is None:
                subdomain.optimizer = torch.optim.Adam(
                    [temperature_bc], lr=self.learning_rate
                )
            loss = _domain_interface_mse(subdomain)
            if not loss.requires_grad:
                continue
            torch.autograd.backward(loss, inputs=temperature_bc)
            subdomain.optimizer.step()
            subdomain.optimizer.zero_grad()
            rmse += float(loss.detach().sqrt())
        self.total_loss = rmse / max(len(subdomains), 1)

    def get_metric(self) -> tuple[str, float]:
        """Return the mean interface RMSE from the latest update."""

        return "loss_interface", self.total_loss


class ParallelGradientInterfaceHandler:
    """Optimize all subdomain boundaries from one batched interface loss."""

    def __init__(self, learning_rate: float = 1.0e-3, optimizer: str = "adam") -> None:
        """Configure the batched boundary optimizer."""

        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.total_loss = 100.0

    def __call__(self, subdomains: list[Subdomain]) -> None:
        """Apply one parallel interface optimization step."""

        if not subdomains:
            self.total_loss = 0.0
            return

        template = subdomains[0].fields[Fields.TEMPERATURE_BC]
        per_domain_sse = [template.new_tensor(0.0) for _ in subdomains]
        per_domain_count = [0 for _ in subdomains]
        total_sse = subdomains[0].fields[Fields.TEMPERATURE_BC].new_tensor(0.0)
        total_count = 0

        # Each patch owns its boundary tensor and optimizer state across
        # Schwarz iterations.
        for subdomain in subdomains:
            temperature_bc = subdomain.fields[Fields.TEMPERATURE_BC]
            _zero_interior(temperature_bc)
            if not temperature_bc.requires_grad:
                temperature_bc.requires_grad_(True)
            if subdomain.optimizer is None:
                subdomain.optimizer = _make_optimizer(
                    self.optimizer_name, [temperature_bc], self.learning_rate
                )

        # Stack interfaces with the same orientation so one backward pass can
        # update every artificial boundary.
        for direction in DIRECTIONS:
            current_values = []
            target_values = []
            owners = []
            for index, subdomain in enumerate(subdomains):
                neighbor = subdomain.neighbors[direction]
                if neighbor is None:
                    continue
                current = subdomain.fields[Fields.TEMPERATURE_BC][
                    subdomain.boundary_slices[direction]
                ]
                target = (
                    neighbor.fields[Fields.TEMPERATURE][
                        neighbor.overlap_slices[direction.opposite()]
                    ]
                    .detach()
                    .to(current.device)
                )
                current_values.append(current)
                target_values.append(target)
                owners.append(index)
            if not current_values:
                continue
            difference = torch.stack(current_values) - torch.stack(target_values)
            total_sse += difference.square().sum()
            total_count += difference.numel()
            pair_sse = difference.flatten(1).square().sum(1)
            pair_count = difference[0].numel()
            for owner, value in zip(owners, pair_sse):
                per_domain_sse[owner] = per_domain_sse[owner] + value
                per_domain_count[owner] += pair_count

        if total_count == 0:
            self.total_loss = 0.0
            return

        loss = total_sse / total_count
        loss.backward()
        for subdomain in subdomains:
            subdomain.optimizer.step()
            subdomain.optimizer.zero_grad()

        rmses = [
            torch.sqrt(value / max(count, 1))
            for value, count in zip(per_domain_sse, per_domain_count)
        ]
        self.total_loss = float(torch.stack(rmses).mean().detach())

    def get_metric(self) -> tuple[str, float]:
        """Return the mean interface RMSE from the latest update."""

        return "loss_interface", self.total_loss


class ExchangeInterfaceHandler:
    """Copy neighboring overlap values directly onto subdomain boundaries."""

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        solution_field: Fields = Fields.TEMPERATURE,
        boundary_field: Fields = Fields.TEMPERATURE_BC,
        boundary_depth: int = 1,
    ) -> None:
        """Set the relaxation fraction used for each boundary exchange."""

        self.alpha = alpha
        self.solution_field = solution_field
        self.boundary_field = boundary_field
        self.boundary_depth = boundary_depth
        self.total_loss = 0.0

    @torch.no_grad()
    def __call__(self, subdomains: list[Subdomain]) -> None:
        """Apply one relaxed Dirichlet exchange step."""

        losses = [
            _domain_interface_mse(
                subdomain,
                self.solution_field,
                self.boundary_field,
                self.boundary_depth,
            ).sqrt()
            for subdomain in subdomains
        ]
        self.total_loss = float(torch.stack(losses).mean()) if losses else 0.0
        for subdomain in subdomains:
            boundary = subdomain.fields[self.boundary_field]
            for direction in DIRECTIONS:
                neighbor = subdomain.neighbors[direction]
                if neighbor is None:
                    continue
                current = boundary[
                    subdomain.boundary_slice(direction, self.boundary_depth)
                ]
                target = neighbor.fields[self.solution_field][
                    neighbor.overlap_slice(direction.opposite(), self.boundary_depth)
                ].to(current.device)
                current.lerp_(target, self.alpha)
            _zero_interior(boundary, self.boundary_depth)

    def get_metric(self) -> tuple[str, float]:
        """Return the mean interface RMSE before the latest exchange."""

        return "loss_interface", self.total_loss


class InterfaceConsistencyMonitor:
    """Measure temperature disagreement across every directed interface."""

    def __init__(self) -> None:
        """Initialize the overlap RMSE metric."""

        self.total_rmse = 0.0

    @torch.no_grad()
    def __call__(self, subdomains: list[Subdomain]) -> None:
        """Update the mean overlap RMSE for the current subdomain fields."""

        total = 0.0
        count = 0
        for subdomain in subdomains:
            for direction in DIRECTIONS:
                neighbor = subdomain.neighbors[direction]
                if neighbor is None:
                    continue
                current = subdomain.fields[Fields.TEMPERATURE][
                    subdomain.boundary_slices[direction]
                ]
                target = neighbor.fields[Fields.TEMPERATURE][
                    neighbor.overlap_slices[direction.opposite()]
                ].to(current.device)
                total += float((current - target).square().mean().sqrt())
                count += 1
        self.total_rmse = total / max(count, 1)

    def get_metric(self) -> tuple[str, float]:
        """Return the most recently measured overlap RMSE."""

        return "loss_overlap", self.total_rmse
