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

"""Minimal differentiable shape-update loop with geometric safeguards.

The objective below is an illustrative projected-area proxy, not a CFD drag
model. It keeps the example checkpoint-free while exercising the same handoff:
a model/objective supplies point vectors, then ``guarded_normal_update`` projects,
smooths, masks, clips, validates, and backtracks before changing the mesh.

An ordinary objective gradient is negated for minimization. A sensitivity that
is already signed in the improving direction, such as DoMINO's
``d(-drag)/dX``, should instead be passed directly.
"""

import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral


def projected_area_proxy(mesh: Mesh) -> torch.Tensor:
    """Return a differentiable flow-aligned projected-area proxy."""
    flow_direction = mesh.points.new_tensor([1.0, 0.0, 0.0])
    alignment = mesh.cell_normals @ flow_direction
    return (mesh.cell_areas * alignment.square()).sum()


def main() -> None:
    """Run a guarded optimization of the illustrative area objective."""
    mesh = sphere_icosahedral.load(subdivisions=2)
    initial_points = mesh.points.clone()

    # Freeze the lower cap and smoothly ramp to full motion above it. The
    # weights are applied after scalar smoothing and before global clipping.
    initial_z = initial_points[:, 2]
    design_weights = ((initial_z + 0.75) / 0.25).clamp(0.0, 1.0)
    protected = design_weights == 0

    initial_objective = float(projected_area_proxy(mesh).detach())
    print(f"initial objective: {initial_objective:.6f}")

    for iteration in range(8):
        # Start a fresh graph each design iteration to avoid retaining the full
        # optimization history.
        points = mesh.points.detach().requires_grad_(True)
        differentiable_mesh = Mesh(points=points, cells=mesh.cells)
        objective = projected_area_proxy(differentiable_mesh)
        gradient = torch.autograd.grad(objective, points)[0]

        updated, diagnostics = differentiable_mesh.guarded_normal_update(
            -gradient.detach(),
            point_weights=design_weights,
            smoothing_iterations=5,
            smoothing_relaxation=0.2,
            max_step=0.02,
            backtracking_factor=0.5,
            max_backtracks=8,
        )
        if not diagnostics.accepted:
            raise RuntimeError(
                f"No geometrically valid update was found: {diagnostics.validation}"
            )

        new_objective = float(projected_area_proxy(updated).detach())
        # The helper backtracks on geometry validity, not objective value. A
        # production optimizer can add its own Armijo/trust-region acceptance
        # test around this call when monotonic decrease is required.
        print(
            f"iteration {iteration + 1:02d}: objective={new_objective:.6f}, "
            f"max_step={diagnostics.applied_max_step:.4f}, "
            f"backtracks={diagnostics.n_backtracks}"
        )
        mesh = Mesh(points=updated.points.detach(), cells=updated.cells)

    final_objective = float(projected_area_proxy(mesh).detach())
    final_report = mesh.validate(check_duplicate_vertices=False)
    torch.testing.assert_close(mesh.points[protected], initial_points[protected])
    if final_objective >= initial_objective:
        raise RuntimeError("The illustrative objective did not decrease")
    if not final_report["valid"]:
        raise RuntimeError(f"Final mesh is invalid: {final_report}")
    print(f"final objective:   {final_objective:.6f}")


if __name__ == "__main__":
    main()
