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

"""PyTorch adapter for regular-lattice point deformation."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from physicsnemo.nn.functional.interpolation import grid_to_point_interpolation

from .utils import (
    LatticeInterpolation,
    apply_lattice_displacement,
    empty_lattice_deformation,
    prepare_lattice_deform_inputs,
)


def lattice_deform_points_torch(
    points: torch.Tensor,
    control_displacements: torch.Tensor,
    lattice_bounds: Sequence[tuple[float, float]],
    *,
    interpolation_type: LatticeInterpolation = "smooth_step_2",
    point_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Deform points using the PyTorch grid-interpolation backend."""
    grid, normalized_points, point_weights = prepare_lattice_deform_inputs(
        points,
        control_displacements,
        lattice_bounds,
        interpolation_type,
        point_weights,
    )
    if points.shape[0] == 0:
        return empty_lattice_deformation(
            points,
            control_displacements,
            point_weights,
        )

    displacement = grid_to_point_interpolation(
        normalized_points,
        control_displacements,
        grid,
        interpolation_type=interpolation_type,
        implementation="torch",
    )
    return apply_lattice_displacement(points, displacement, point_weights)


__all__ = ["lattice_deform_points_torch"]
