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

"""Uncompiled PhysicsNeMo-Mesh operations for benchmarking.

Each function takes **raw tensors** (points, cells, and any extra args) and
constructs a fresh :class:`~physicsnemo.mesh.Mesh` internally.  This is the
single source of truth for operation logic; ``compiled_ops`` applies
``torch.compile`` to these same functions.
"""

import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus import compute_gradient_points_lsq
from physicsnemo.mesh.smoothing import smooth_laplacian


def cell_normals(points, cells):
    """Compute per-cell normals from raw point/cell tensors."""
    return Mesh(points=points, cells=cells).cell_normals


def gaussian_curvature(points, cells):
    """Compute per-vertex Gaussian curvature from raw point/cell tensors."""
    return Mesh(points=points, cells=cells).gaussian_curvature_vertices


def gradient(points, cells, scalar_field):
    """Compute least-squares gradient of a scalar field on vertices."""
    m = Mesh(points=points, cells=cells)
    return compute_gradient_points_lsq(m, point_values=scalar_field)


def subdivide(points, cells):
    """Loop-subdivide once and return (new_points, new_cells)."""
    r = Mesh(points=points, cells=cells).subdivide(levels=1, filter="loop")
    return r.points, r.cells


def p2p_neighbors(points, cells):
    """Compute point-to-point adjacency."""
    return Mesh(points=points, cells=cells).get_point_to_points_adjacency()


def c2c_neighbors(points, cells):
    """Compute cell-to-cell adjacency."""
    return Mesh(points=points, cells=cells).get_cell_to_cells_adjacency()


def sample_points(points, cells, cell_indices):
    """Sample random points on specified cells."""
    return Mesh(points=points, cells=cells).sample_random_points_on_cells(
        cell_indices=cell_indices,
    )


def sample_points_area_weighted(points, cells, n_samples):
    """Area-weighted random surface sampling (end-to-end)."""
    m = Mesh(points=points, cells=cells)
    areas = m.cell_areas
    cell_indices = torch.multinomial(areas, n_samples, replacement=True)
    return m.sample_random_points_on_cells(cell_indices=cell_indices)


def smooth(points, cells, n_iter, relaxation_factor):
    """Run Laplacian smoothing and return the smoothed point positions."""
    m = Mesh(points=points, cells=cells)
    return smooth_laplacian(
        m, n_iter=n_iter, relaxation_factor=relaxation_factor
    ).points


def transforms(points, cells, offset, angle_x, angle_y, scale_factor):
    """Apply translate -> rotate_x -> rotate_y -> scale and return points."""
    m = Mesh(points=points, cells=cells)
    m = m.translate(offset=offset)
    m = m.rotate(axis="x", angle=angle_x)
    m = m.rotate(axis="y", angle=angle_y)
    m = m.scale(factor=scale_factor)
    return m.points
