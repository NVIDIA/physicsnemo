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

r"""Discrete integration measure for meshes.

A mesh discretizes a manifold, and every integral over it is a
measure-weighted sum: each cell contributes its field value times the
measure that cell represents.  For a mesh that represents exactly the
geometry it stores, that measure is the geometric simplex measure,
:attr:`~physicsnemo.mesh.Mesh.cell_areas`.  But the two can diverge -- a
cell may stand for more, less, or other than itself.  This module defines
the resulting contract: the effective **cell measure**

.. math::
    \mu_c = |\sigma_c| \, w_c,

where :math:`|\sigma_c|` is the geometric measure of cell :math:`c` and
:math:`w_c` is a dimensionless per-cell **measure weight** -- the ratio of
represented to geometric measure -- defaulting to one.

Non-unit weights arise whenever representation and geometry part ways:
cells standing in for symmetric or periodic images of themselves,
fractional weights that de-duplicate overlapping patches, corrections for
curved geometry that a straight simplex under-resolves, or coarse cells
representing a partition of finer ones.  The canonical source in this
package is random cell subsampling: a cell retained with inclusion
probability :math:`\pi_c` statistically represents :math:`1/\pi_c` cells
of the original mesh, so the subsampling stage records the
Horvitz-Thompson weight :math:`w_c = 1/\pi_c` (exactly ``N/k`` for a
uniform ``k``-of-``N`` subsample).  Integrals computed with the effective
measure are then unbiased estimates of the full-mesh integrals; computed
with the bare geometric measure they shrink by ``~k/N``, compounded once
per stage in any computation that chains measure-weighted sums.

The contract has three parts:

- Measure weights are stored in ``cell_data`` under the reserved key
  :data:`MEASURE_WEIGHTS_KEY`.  Living in ``cell_data``, they survive
  cell slicing, serialization, device transfer, and rigid transforms
  automatically; being dimensionless, they are also invariant under
  geometric rescaling (``cell_areas`` alone picks up the appropriate power
  of length).  The underscore prefix marks the field as bookkeeping:
  feature-selection code that consumes ``cell_data`` wholesale should
  exclude it.
- Producers record their weight contribution via
  :func:`compose_measure_weights`; successive contributions compose
  multiplicatively, so e.g. chained subsampling stays exact.  Point
  subsampling on meshes with cells does **not** maintain weights (cells
  dropped implicitly have no per-cell inclusion probability).
- Integral consumers (:meth:`Mesh.integrate`, :meth:`Mesh.integrate_flux`,
  :func:`~physicsnemo.mesh.calculus.integration.integrate_moment`) read
  :func:`cell_measures`.  Meshes without recorded weights pass through
  with the bare geometric measure, bit-identically.
"""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh

### Reserved `cell_data` key holding dimensionless per-cell measure weights
### (ratios of represented to geometric measure). See the module docstring
### for the contract.
MEASURE_WEIGHTS_KEY: str = "_measure_weights"


def cell_measure_weights(mesh: "Mesh") -> Float[torch.Tensor, " n_cells"]:
    r"""Per-cell measure weights of *mesh* (ones when none are recorded).

    Returns
    -------
    torch.Tensor
        Dimensionless weights of shape ``(n_cells,)``.  If no weights have
        been recorded, returns ones: every cell represents exactly itself.
    """
    weights = mesh.cell_data.get(MEASURE_WEIGHTS_KEY, None)
    if weights is None:
        return torch.ones(
            mesh.n_cells, dtype=mesh.points.dtype, device=mesh.points.device
        )
    return weights


def cell_measures(mesh: "Mesh") -> Float[torch.Tensor, " n_cells"]:
    r"""Effective per-cell integration measure: ``cell_areas * measure_weights``.

    This is what integral consumers should weight by.  Skips the
    multiplication when no measure weights are recorded, so meshes without
    weights pay nothing and results are bit-identical to the bare geometric
    measure.

    Returns
    -------
    torch.Tensor
        Effective measure of shape ``(n_cells,)``.
    """
    cell_areas = mesh.cell_areas
    weights = mesh.cell_data.get(MEASURE_WEIGHTS_KEY, None)
    if weights is None:
        return cell_areas
    return cell_areas * weights


def compose_measure_weights(mesh: "Mesh", factor: float | torch.Tensor) -> None:
    r"""Multiply *mesh*'s measure weights by *factor*, in place.

    Called by each producer with its contribution to the represented-to-
    geometric measure ratio -- a sampling stage, for example, passes its
    inverse inclusion probability (``n_cells_before / n_cells_after`` for
    a uniform sample).  Contributions compose multiplicatively: a stage
    keeping ``k1`` of ``N`` cells followed by one keeping ``k2`` of ``k1``
    yields exactly ``N/k2``.

    Parameters
    ----------
    mesh : Mesh
        Mesh to update.  Its ``cell_data`` is modified in place.
    factor : float or torch.Tensor
        This producer's weight contribution; a scalar or a per-cell tensor
        broadcast against the existing weights.
    """
    mesh.cell_data[MEASURE_WEIGHTS_KEY] = cell_measure_weights(mesh) * factor
