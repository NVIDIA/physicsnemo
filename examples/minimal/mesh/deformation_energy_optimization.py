# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optimize an RBF deformation with differentiable geometric penalties.

The target handle and corner anchors are fixed in the parameterization. Adam
optimizes four auxiliary RBF controls to reduce strain, total-area change, and
element inversion. The example uses Warp on CUDA and Torch on CPU.
"""

import torch

from physicsnemo.mesh.deformation import (
    simplex_inversion_energy,
    simplex_strain_energy,
    total_measure_energy,
)
from physicsnemo.mesh.primitives.planar import unit_square
from physicsnemo.nn.functional import radial_basis_function_deform_points


def _signed_triangle_ratios(
    points: torch.Tensor,
    reference_points: torch.Tensor,
    cells: torch.Tensor,
) -> torch.Tensor:
    """Return signed current-to-reference area ratios for 2D triangles."""

    current = points[cells]
    reference = reference_points[cells]
    current_twice_area = torch.linalg.det(
        torch.stack(
            (current[:, 1] - current[:, 0], current[:, 2] - current[:, 0]), dim=-1
        )
    )
    reference_twice_area = torch.linalg.det(
        torch.stack(
            (
                reference[:, 1] - reference[:, 0],
                reference[:, 2] - reference[:, 0],
            ),
            dim=-1,
        )
    )
    return current_twice_area / reference_twice_area


def main() -> None:
    """Run the penalty-regularized shape-optimization example."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    implementation = "warp" if device.type == "cuda" else "torch"

    reference = unit_square.load(subdivisions=3).to(device=device, dtype=torch.float64)
    controls = reference.points.new_tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [1.0, 0.5],
            [0.0, 0.5],
            [0.5, 0.5],
            [0.5, 1.0],
        ]
    )
    target_displacement = controls.new_tensor([0.36, 0.48])
    free_displacements = torch.nn.Parameter(reference.points.new_zeros((4, 2)))

    def deformed_points() -> torch.Tensor:
        """Evaluate the radial-basis deformation for the current parameters."""

        displacements = torch.cat(
            (
                torch.zeros_like(controls[:4]),
                free_displacements,
                target_displacement.unsqueeze(0),
            )
        )
        return radial_basis_function_deform_points(
            reference.points,
            controls,
            displacements,
            kernel="thin_plate_spline",
            polynomial=True,
            smoothing=0.0,
            implementation=implementation,
        )

    optimizer = torch.optim.Adam([free_displacements], lr=3.0e-2)
    for _ in range(350):
        optimizer.zero_grad()
        points = deformed_points()
        loss = (
            0.05
            * simplex_strain_energy(
                reference,
                points,
                implementation=implementation,
            )
            + 80.0
            * total_measure_energy(
                reference,
                points,
                implementation=implementation,
            )
            + 500.0
            * simplex_inversion_energy(
                reference,
                points,
                minimum_jacobian=0.4,
                implementation=implementation,
            )
        )
        loss.backward()
        if (
            free_displacements.grad is None
            or not torch.isfinite(free_displacements.grad).all()
        ):
            raise RuntimeError(
                "The deformation objective produced a non-finite gradient"
            )
        optimizer.step()

    points = deformed_points().detach()
    ratios = _signed_triangle_ratios(points, reference.points, reference.cells)
    current = points[reference.cells]
    rest = reference.points[reference.cells]
    current_twice_area = torch.linalg.det(
        torch.stack(
            (current[:, 1] - current[:, 0], current[:, 2] - current[:, 0]),
            dim=-1,
        )
    ).abs()
    rest_twice_area = torch.linalg.det(
        torch.stack((rest[:, 1] - rest[:, 0], rest[:, 2] - rest[:, 0]), dim=-1)
    ).abs()
    area_ratio = current_twice_area.sum() / rest_twice_area.sum()
    area_error = abs(area_ratio.item() - 1.0)
    minimum_ratio = ratios.min().item()
    if area_error >= 1.0e-2 or minimum_ratio <= 0.0:
        raise RuntimeError(
            "Optimization did not satisfy the area and orientation checks: "
            f"relative area error={area_error:.3e}, minimum signed "
            f"ratio={minimum_ratio:.3f}"
        )
    print(f"device: {device}; backend: {implementation}")
    print(f"relative total-area error: {area_error:.3e}")
    print(f"minimum signed element ratio: {minimum_ratio:.3f}")
    print("optimized auxiliary control displacements:")
    print(free_displacements.detach().cpu())


if __name__ == "__main__":
    main()
