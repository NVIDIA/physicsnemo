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

r"""Quadrature measure for subsampled meshes.

When a mesh is produced by randomly subsampling the cells of a larger mesh,
each retained cell statistically represents more of the domain than its own
geometric area: a cell kept with inclusion probability :math:`\pi_i` carries
the Horvitz-Thompson **sampling weight** :math:`w_i = 1/\pi_i` (exactly
``N/k`` for a uniform ``k``-of-``N`` cell subsample).  A discrete integral

.. math::
    \int_\Omega f\,d\Omega \approx \sum_c f_c\,|\sigma_c|\,w_c

computed with the effective **quadrature areas**
``cell_areas * sampling_weights`` is then an unbiased estimate of the
full-mesh integral.  Without this, integrals over a ``k``-of-``N`` subsample
silently shrink by ``~k/N`` -- and models that compound area-weighted
integrals across several stages (e.g. GLOBE's boundary-integral hyperlayers)
attenuate their outputs by that factor *per stage*.

This module owns the whole concept, so :class:`~physicsnemo.mesh.Mesh`
itself stays a general-purpose container:

- Sampling weights are stored in ``cell_data`` under the reserved key
  :data:`SAMPLING_WEIGHTS_KEY`.  Living in ``cell_data``, they survive cell
  slicing, serialization, device transfer, and rigid transforms
  automatically; being dimensionless, they are also invariant under
  geometric rescaling (``cell_areas`` alone picks up the appropriate power
  of length).  The underscore prefix marks the field as bookkeeping:
  feature-selection code that consumes ``cell_data`` wholesale should
  exclude it.
- Cell-subsampling operations (``MeshReader``/``DomainMeshReader`` cell
  subsampling, the ``SubsampleMesh`` transform) record each stage's inverse
  inclusion probability via :func:`compose_sampling_weights`; multiple
  stages compose multiplicatively, so chained subsampling stays exact.
  Point subsampling on meshes with cells does **not** maintain weights
  (cells dropped implicitly have no per-cell inclusion probability).
- Integral consumers (:meth:`Mesh.integrate`, :meth:`Mesh.integrate_flux`,
  ``integrate_moment``, GLOBE) read :func:`cell_quadrature_areas`.  Meshes
  without recorded weights pass through with the bare geometric measure,
  bit-identically to the no-weights behavior.

Terminology follows survey statistics and finite elements: the dimensionless
*sampling weight* is :math:`1/\pi_i`; the *quadrature area* (the full
measure carried by a retained cell) is its geometric area times its sampling
weight.
"""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh

### Reserved `cell_data` key holding dimensionless per-cell sampling weights
### (inverse inclusion probabilities) recorded by cell-subsampling
### operations. See the module docstring for the contract.
SAMPLING_WEIGHTS_KEY: str = "_sampling_weights"


def cell_sampling_weights(mesh: "Mesh") -> Float[torch.Tensor, " n_cells"]:
    r"""Per-cell sampling weights of *mesh* (ones when none are recorded).

    Returns
    -------
    torch.Tensor
        Dimensionless weights of shape ``(n_cells,)``.  If no weights have
        been recorded, returns ones: every cell represents exactly itself.
    """
    weights = mesh.cell_data.get(SAMPLING_WEIGHTS_KEY, None)
    if weights is None:
        return torch.ones(
            mesh.n_cells, dtype=mesh.points.dtype, device=mesh.points.device
        )
    return weights


def cell_quadrature_areas(mesh: "Mesh") -> Float[torch.Tensor, " n_cells"]:
    r"""Effective per-cell quadrature measure: ``cell_areas * sampling_weights``.

    This is what integral consumers should weight by.  Skips the
    multiplication when no sampling weights are recorded, so meshes without
    weights pay nothing and results are bit-identical to the bare geometric
    measure.

    Returns
    -------
    torch.Tensor
        Effective measure of shape ``(n_cells,)``.
    """
    cell_areas = mesh.cell_areas
    weights = mesh.cell_data.get(SAMPLING_WEIGHTS_KEY, None)
    if weights is None:
        return cell_areas
    return cell_areas * weights


def compose_sampling_weights(mesh: "Mesh", factor: float | torch.Tensor) -> None:
    r"""Multiply *mesh*'s sampling weights by *factor*, in place.

    Called by each cell-sampling stage with its inverse inclusion
    probability (``n_cells_before / n_cells_after`` for a uniform sample).
    Stages compose multiplicatively: a reader keeping ``k1`` of ``N`` cells
    followed by a transform keeping ``k2`` of ``k1`` yields exactly
    ``N/k2``.

    Parameters
    ----------
    mesh : Mesh
        Mesh to update.  Its ``cell_data`` is modified in place.
    factor : float or scalar torch.Tensor
        This stage's inverse inclusion probability, broadcast over all
        cells.
    """
    mesh.cell_data[SAMPLING_WEIGHTS_KEY] = cell_sampling_weights(mesh) * factor
