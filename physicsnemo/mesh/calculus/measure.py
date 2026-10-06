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

r"""Complete integration measures for cells and point samples.

``_effective_measure`` always stores the complete contribution to an integral,
never a multiplier that must still be multiplied by geometry. Cells without an
explicit measure use their geometric simplex measure. Point measures are always
explicit, independent of connectivity; counting measure is an explicit choice.

Sampling producers multiply effective measures by their inverse inclusion
probability. Cell-to-centroid conversion transfers those measures to points.
All such producers use the helpers here rather than inventing storage keys.

Point quadrature also records its represented dimension in ``global_data``:
zero for counting, one for length, two for area, etc. This describes the measure,
not the topology of the point cloud. It suffices for uniform scaling; arbitrary
changes of point positions require support geometry or an explicit decision to
preserve the measure. Cell measures follow the ratio of new to old geometric
measure when geometry changes, preserving their represented/geometric ratio.
"""

from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh

EFFECTIVE_MEASURE_KEY = "_effective_measure"
POINT_MEASURE_DIMENSION_KEY = "_point_measure_dimension"


def _validate_measures(
    mesh: "Mesh", values: torch.Tensor, association: str
) -> torch.Tensor:
    n = mesh.n_cells if association == "cells" else mesh.n_points
    if not isinstance(values, torch.Tensor) or values.shape != (n,):
        shape = (
            tuple(values.shape)
            if isinstance(values, torch.Tensor)
            else type(values).__name__
        )
        raise ValueError(
            f"{EFFECTIVE_MEASURE_KEY!r} must have shape {(n,)} (one scalar per {association}), got {shape}"
        )
    if not values.is_floating_point():
        raise TypeError("Effective measures must be floating-point tensors")
    if values.device != mesh.points.device:
        raise ValueError(
            "Effective measures and mesh points must be on the same device"
        )
    return values


def cell_measures(mesh: "Mesh") -> torch.Tensor:
    """Return complete per-cell measures, falling back to geometric measures.

    The result has shape ``(n_cells,)``. The fallback does not materialize a
    stored field, so unweighted meshes retain lazy geometric computation.
    """
    if "_measure_weights" in mesh.cell_data:
        raise ValueError(
            "Legacy _measure_weights must be converted to _effective_measure = "
            "cell_areas * weights before use; the new field stores complete measures"
        )
    values = mesh.cell_data.get(EFFECTIVE_MEASURE_KEY, None)
    return (
        mesh.cell_areas if values is None else _validate_measures(mesh, values, "cells")
    )


def point_measures(mesh: "Mesh") -> torch.Tensor:
    """Return explicit per-point measures of shape ``(n_points,)``.

    Raises ``KeyError`` when absent, even if cells exist. Use an ordinary sum
    for counting measure, or explicitly install ones with dimension zero.
    Use :func:`lumped_point_measures` to construct nodal quadrature from cells.
    """
    values = mesh.point_data.get(EFFECTIVE_MEASURE_KEY, None)
    if values is None:
        raise KeyError(
            "Point quadrature requires explicit effective measures; use set_point_measures or an ordinary sum for counting measure"
        )
    return _validate_measures(mesh, values, "points")


def set_cell_measures(mesh: "Mesh", values: torch.Tensor) -> None:
    """Set complete per-cell measures in place, without changing geometry."""
    mesh.cell_data[EFFECTIVE_MEASURE_KEY] = _validate_measures(mesh, values, "cells")


def set_point_measures(mesh: "Mesh", values: torch.Tensor, *, dimension: int) -> None:
    """Set complete point measures and their represented dimension in place.

    ``dimension`` is the power of length: 0 for counting, 1 for length, 2 for
    area, 3 for volume. It never depends on whether this mesh has cells.
    """
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or not 0 <= dimension <= mesh.n_spatial_dims
    ):
        raise ValueError(
            f"Measure dimension must be an integer in [0, {mesh.n_spatial_dims}]"
        )
    values = _validate_measures(mesh, values, "points")
    mesh.point_data[EFFECTIVE_MEASURE_KEY] = values
    mesh.global_data[POINT_MEASURE_DIMENSION_KEY] = torch.tensor(
        dimension, device=mesh.points.device
    )


def point_measure_dimension(mesh: "Mesh") -> torch.Tensor:
    """Read the scalar dimension metadata required to transform point measures."""
    dimension = mesh.global_data.get(POINT_MEASURE_DIMENSION_KEY, None)
    if (
        dimension is None
        or not isinstance(dimension, torch.Tensor)
        or dimension.ndim != 0
    ):
        raise ValueError(
            "Point measure dimension is missing or not scalar; use set_point_measures"
        )
    if not torch.compiler.is_compiling() and not bool(
        (dimension >= 0)
        & (dimension <= mesh.n_spatial_dims)
        & (dimension == dimension.round())
    ):
        raise ValueError(
            "Point measure dimension must be a nonnegative integer no larger than the spatial dimension"
        )
    return dimension


def scale_measures(
    mesh: "Mesh",
    factor: float | torch.Tensor,
    *,
    association: Literal["cells", "points"] = "cells",
) -> None:
    """Multiply effective measures in place by a scalar or per-entity factor.

    For example, a sampling stage keeping k of N cells uses N/k; subsequent
    stages multiply the measures already recorded. Point measures must exist
    before reweighting: this function never invents a point base measure.
    """
    if association not in ("cells", "points"):
        raise ValueError("association must be 'cells' or 'points'")
    values = cell_measures(mesh) if association == "cells" else point_measures(mesh)
    if (
        isinstance(factor, torch.Tensor)
        and factor.ndim != 0
        and factor.shape != values.shape
    ):
        raise ValueError(
            f"Measure factor must be a scalar or have shape {tuple(values.shape)}, got {tuple(factor.shape)}"
        )
    getattr(mesh, "cell_data" if association == "cells" else "point_data")[
        EFFECTIVE_MEASURE_KEY
    ] = values * factor


def lumped_point_measures(mesh: "Mesh") -> torch.Tensor:
    """Construct P1 nodal quadrature by dividing each cell's measure equally.

    Each vertex receives the sum of its incident cells' contributions. This is
    an explicit construction, not the default point measure. For finite nodal
    values it reproduces the piecewise-linear integral. Cell-based NaN omission
    still requires :func:`integrate`, rather than a sum of nodal contributions.
    """
    measures = cell_measures(mesh)
    result = measures.new_zeros(mesh.n_points)
    shares = (measures / mesh.cells.shape[1]).unsqueeze(-1).expand_as(mesh.cells)
    return result.scatter_add(0, mesh.cells.flatten(), shares.flatten())


def _transfer_cell_measures(
    source: "Mesh", result: "Mesh", parents: torch.Tensor | None = None
) -> None:
    """Transfer explicit cell measures through geometric changes/subdivision."""
    if EFFECTIVE_MEASURE_KEY not in source.cell_data:
        return
    measures = cell_measures(source)
    old_areas = source.cell_areas
    if parents is not None:
        measures, old_areas = measures[parents], old_areas[parents]
    if not torch.compiler.is_compiling() and bool(
        ((old_areas == 0) & (measures != 0)).any()
    ):
        raise ValueError(
            "Cannot transform nonzero effective measure on a zero-measure cell; supply replacement measures"
        )
    denominator = torch.where(old_areas != 0, old_areas, torch.ones_like(old_areas))
    set_cell_measures(result, measures * (result.cell_areas / denominator))


def _require_preserved_point_measures(mesh: "Mesh") -> None:
    """Reject unknown geometric changes of dimensional point quadrature."""
    if EFFECTIVE_MEASURE_KEY in mesh.point_data and bool(
        point_measure_dimension(mesh) != 0
    ):
        raise ValueError(
            "Changing quadrature point geometry requires replacement measures or an explicit preservation policy; use with_points(..., preserve_measures=True) to retain reference measures"
        )


def _transform_point_measures(
    source: "Mesh", result: "Mesh", matrix: torch.Tensor
) -> None:
    """Transform counting, full-dimensional, or similarity-mapped point measures."""
    if EFFECTIVE_MEASURE_KEY not in source.point_data:
        return
    dimension = point_measure_dimension(source)
    if bool(dimension == 0):
        return
    if matrix.shape[0] == matrix.shape[1] and bool(dimension == source.n_spatial_dims):
        factor = matrix.det().abs()
    else:
        from physicsnemo.mesh.transformations.geometric import _is_similarity_transform

        if not _is_similarity_transform(matrix):
            raise ValueError(
                "Anisotropic transformation of point measures requires support geometry; transform the source cells before constructing quadrature, or explicitly preserve reference measures"
            )
        length_scale = (matrix.T @ matrix).diagonal().mean().clamp_min(0).sqrt()
        factor = length_scale**dimension
    result.point_data[EFFECTIVE_MEASURE_KEY] = point_measures(source) * factor
