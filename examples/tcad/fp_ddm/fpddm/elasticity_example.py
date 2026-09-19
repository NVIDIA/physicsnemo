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

"""Numerical plane-stress baseline for the FP-DDM workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .domain import DIRECTIONS, Domain, Fields, Subdomain, assemble_avg
from .elasticity import Elasticity2DSolver
from .interfaces import ExchangeInterfaceHandler
from .observers import MetricsLogger
from .schwarz import EarlyStopping, SchwarzMethod


class ElasticityDomainSolver:
    """Apply the numerical elasticity solver to a batch of subdomains."""

    def __init__(self, solver: Elasticity2DSolver) -> None:
        self.solver = solver

    @torch.no_grad()
    def solve_batch(self, subdomains: list[Subdomain]) -> None:
        """Solve each supplied patch and write its displacement field."""

        radius = self.solver.stencil_radius
        boundary = torch.stack(
            [patch.fields[Fields.DISPLACEMENT_BC] for patch in subdomains]
        )
        masks = []
        for patch in subdomains:
            mask = torch.zeros_like(
                patch.fields[Fields.DISPLACEMENT_BC], dtype=torch.bool
            )
            for direction in DIRECTIONS:
                depth = radius if patch.neighbors[direction] is not None else 1
                mask[patch.boundary_slice(direction, depth)] = True
            masks.append(mask)
        displacement = self.solver.solve(
            torch.stack([patch.fields[Fields.YOUNG_MODULUS] for patch in subdomains]),
            torch.stack([patch.fields[Fields.POISSON_RATIO] for patch in subdomains]),
            torch.zeros_like(boundary),
            boundary,
            torch.stack(masks),
            initial=torch.stack(
                [patch.fields[Fields.DISPLACEMENT] for patch in subdomains]
            ),
        ).cpu()
        for patch, values in zip(subdomains, displacement):
            patch.fields[Fields.DISPLACEMENT] = values.clone()


@dataclass
class ElasticityRunResult:
    """Monolithic and decomposed outputs from the elasticity example."""

    exact_displacement: torch.Tensor
    reference_displacement: torch.Tensor
    displacement: torch.Tensor
    reference_stress: torch.Tensor
    stress: torch.Tensor
    metrics: list[dict[str, float | int]]
    converged: bool
    displacement_relative_error: float
    stress_relative_error: float
    comparison_path: Path | None


def _relative_error(prediction: torch.Tensor, reference: torch.Tensor) -> float:
    difference = torch.linalg.vector_norm(prediction - reference)
    scale = torch.linalg.vector_norm(reference).clamp_min(1.0e-15)
    return float(difference / scale)


def _initialize_problem(
    domain: Domain, solver: Elasticity2DSolver
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a zero-force bending problem with an exact solution."""

    height, width = domain.total_height, domain.total_width
    y = torch.linspace(0.0, 1.0, height, dtype=solver.dtype)
    x = torch.linspace(0.0, 1.0, width, dtype=solver.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    # The -3.2 ratio satisfies Navier's equation for poisson_ratio=0.25.
    exact = torch.stack(
        (
            0.01 * yy.square(),
            -0.032 * xx * yy,
        )
    )
    young_modulus = torch.full_like(xx, 120.0)
    poisson_ratio = torch.full_like(young_modulus, 0.25)

    physical_mask = torch.ones_like(exact, dtype=torch.bool)
    physical_mask[..., 1:-1, 1:-1] = False
    boundary = torch.where(physical_mask, exact, torch.zeros_like(exact))
    domain.fields = {
        Fields.DISPLACEMENT: boundary.clone(),
        Fields.YOUNG_MODULUS: young_modulus,
        Fields.POISSON_RATIO: poisson_ratio,
        Fields.DISPLACEMENT_BC: boundary,
    }
    return exact, physical_mask


def _plot_comparison(
    reference_displacement: torch.Tensor,
    displacement: torch.Tensor,
    reference_von_mises: torch.Tensor,
    von_mises: torch.Tensor,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    reference_magnitude = torch.linalg.vector_norm(reference_displacement, dim=0)
    magnitude = torch.linalg.vector_norm(displacement, dim=0)
    rows = (
        (reference_magnitude, magnitude, (magnitude - reference_magnitude).abs()),
        (reference_von_mises, von_mises, (von_mises - reference_von_mises).abs()),
    )
    titles = (
        "Monolithic",
        "Numerical DDM (2 x 2 patches)",
        "Absolute difference",
    )
    labels = ("Displacement magnitude", "Von Mises stress")
    figure, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for row, values in enumerate(rows):
        shared_max = max(float(values[0].max()), float(values[1].max()))
        for column, field in enumerate(values):
            image = axes[row, column].imshow(
                field.cpu(),
                cmap="magma",
                vmin=0.0,
                vmax=None if column == 2 else shared_max,
            )
            axes[row, column].set_title(f"{titles[column]}\n{labels[row]}")
            axes[row, column].set_axis_off()
            figure.colorbar(image, ax=axes[row, column], fraction=0.046)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_elasticity(
    output_dir: str | Path = "outputs/fp_ddm/elasticity",
    *,
    size: int = 32,
    overlap: int = 4,
    max_iterations: int = 100,
    tolerance: float = 1.0e-8,
    device: str | torch.device | None = None,
    visualize: bool = True,
) -> ElasticityRunResult:
    """Compare a 2 x 2 overlapping DDM solve with a monolithic solve."""

    minimum_overlap = 2 * Elasticity2DSolver.stencil_radius
    if overlap < minimum_overlap:
        raise ValueError(
            f"elasticity requires at least {minimum_overlap} shared grid cells"
        )
    if (size + overlap) % 2:
        raise ValueError("size + overlap must be even for a 2 x 2 decomposition")
    patch_size = (size + overlap) // 2
    spacing = (1.0 / (size - 1), 1.0 / (size - 1))
    solver = Elasticity2DSolver(
        spacing=spacing,
        max_iter=1000,
        tolerance=1.0e-11,
        device=device,
        dtype=torch.float64,
    )
    domain = Domain(2, 2, patch_size, patch_size, overlap)
    exact_displacement, physical_mask = _initialize_problem(domain, solver)

    reference_displacement = solver.solve(
        domain.fields[Fields.YOUNG_MODULUS][None],
        domain.fields[Fields.POISSON_RATIO][None],
        torch.zeros_like(domain.fields[Fields.DISPLACEMENT])[None],
        domain.fields[Fields.DISPLACEMENT_BC][None],
        physical_mask[None],
        initial=domain.fields[Fields.DISPLACEMENT][None],
    )[0].cpu()

    output_dir = Path(output_dir)
    exchange = ExchangeInterfaceHandler(
        solution_field=Fields.DISPLACEMENT,
        boundary_field=Fields.DISPLACEMENT_BC,
        boundary_depth=solver.stencil_radius,
    )
    metrics = MetricsLogger(output_dir / "elasticity_metrics.csv")
    early_stopping = EarlyStopping(
        exchange.get_metric,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    method = SchwarzMethod(
        output_dir,
        metrics,
        early_stopper=early_stopping,
        domain_solve_handlers=[ElasticityDomainSolver(solver)],
        interface_handlers=[exchange],
    )
    grid = method.run(domain)
    displacement = assemble_avg(grid, Fields.DISPLACEMENT).detach().cpu()

    young_modulus = domain.fields[Fields.YOUNG_MODULUS][None]
    poisson_ratio = domain.fields[Fields.POISSON_RATIO][None]
    reference_stress = solver.stress(
        reference_displacement[None], young_modulus, poisson_ratio
    )[0].cpu()
    stress = solver.stress(displacement[None], young_modulus, poisson_ratio)[0].cpu()
    reference_von_mises = solver.von_mises(
        reference_displacement[None], young_modulus, poisson_ratio
    )[0].cpu()
    von_mises = solver.von_mises(displacement[None], young_modulus, poisson_ratio)[
        0
    ].cpu()

    comparison_path = output_dir / "elasticity_comparison.png" if visualize else None
    if comparison_path is not None:
        _plot_comparison(
            reference_displacement,
            displacement,
            reference_von_mises,
            von_mises,
            comparison_path,
        )
    return ElasticityRunResult(
        exact_displacement=exact_displacement,
        reference_displacement=reference_displacement,
        displacement=displacement,
        reference_stress=reference_stress,
        stress=stress,
        metrics=metrics.rows,
        converged=early_stopping.converged,
        displacement_relative_error=_relative_error(
            displacement, reference_displacement
        ),
        stress_relative_error=_relative_error(stress, reference_stress),
        comparison_path=comparison_path,
    )
