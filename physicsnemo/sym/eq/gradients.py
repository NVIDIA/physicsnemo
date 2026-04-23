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

"""Spatial gradient modules that adapt modulus functionals for PhysicsInformer.

Each module follows the contract ``forward(input_dict) -> dict[str, Tensor]``,
producing derivative entries keyed by the ``u__x`` naming convention.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, List, Union

import torch

logger = logging.getLogger(__name__)

_AXIS_NAMES = ["x", "y", "z"]


class GradientsAutoDiff(torch.nn.Module):
    """Compute spatial derivatives via ``torch.autograd.grad``.

    Parameters
    ----------
    invar : str
        Name of the variable to differentiate (e.g. ``"u"``).
    dim : int
        Spatial dimensionality (1, 2, or 3).
    order : int
        Derivative order (1 or 2).
    return_mixed_derivs : bool
        If True and ``order=2``, include cross-derivatives like ``u__x__y``.
    """

    def __init__(
        self,
        invar: str,
        dim: int = 3,
        order: int = 1,
        return_mixed_derivs: bool = False,
    ):
        super().__init__()
        self.invar = invar
        self.dim = dim
        self.order = order
        self.return_mixed_derivs = return_mixed_derivs

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        y = input_dict[self.invar]
        x = input_dict["coordinates"]

        grad = _gradient_autodiff(y, [x])
        result: Dict[str, torch.Tensor] = {}

        if self.order == 1:
            for axis in range(self.dim):
                result[f"{self.invar}__{_AXIS_NAMES[axis]}"] = grad[0][
                    :, axis : axis + 1
                ]
        elif self.order == 2:
            for axis in range(self.dim):
                second = _gradient_autodiff(grad[0][:, axis : axis + 1], [x])
                result[f"{self.invar}__{_AXIS_NAMES[axis]}__{_AXIS_NAMES[axis]}"] = (
                    second[0][:, axis : axis + 1]
                )

            if self.return_mixed_derivs:
                for ai, aj in combinations(range(self.dim), 2):
                    mixed = _gradient_autodiff(grad[0][:, ai : ai + 1], [x])[0][
                        :, aj : aj + 1
                    ]
                    result[f"{self.invar}__{_AXIS_NAMES[ai]}__{_AXIS_NAMES[aj]}"] = (
                        mixed
                    )
                    result[f"{self.invar}__{_AXIS_NAMES[aj]}__{_AXIS_NAMES[ai]}"] = (
                        mixed
                    )
        return result


class GradientsFiniteDifference(torch.nn.Module):
    """Compute spatial derivatives on uniform grids via ``UniformGridGradient``.

    Parameters
    ----------
    invar : str
        Name of the variable to differentiate (e.g. ``"u"``).
    dx : float or list[float]
        Uniform grid spacing per axis.
    dim : int
        Spatial dimensionality (1, 2, or 3).
    order : int
        Derivative order (1 or 2).
    return_mixed_derivs : bool
        If True and ``order=2``, include cross-derivatives like ``u__x__y``.
    """

    def __init__(
        self,
        invar: str,
        dx: Union[float, List[float]],
        dim: int = 3,
        order: int = 1,
        return_mixed_derivs: bool = False,
    ):
        super().__init__()
        self.invar = invar
        self.dim = dim
        self.order = order
        self.return_mixed_derivs = return_mixed_derivs
        self.dx = [dx] * dim if isinstance(dx, (float, int)) else list(dx)

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Lazy import: nn.functional.derivatives sits in a lower layer and pulls
        # in warp/FunctionSpec machinery; deferring keeps `import physicsnemo.sym`
        # lightweight and avoids import-linter layer violations.
        from physicsnemo.nn.functional.derivatives import uniform_grid_gradient

        u = input_dict[self.invar]
        field = u[0, 0]

        result: Dict[str, torch.Tensor] = {}

        if self.order == 1:
            grads = uniform_grid_gradient(
                field,
                spacing=self.dx,
                derivative_orders=1,
                include_mixed=False,
            )
            for axis in range(self.dim):
                result[f"{self.invar}__{_AXIS_NAMES[axis]}"] = (
                    grads[axis].unsqueeze(0).unsqueeze(0)
                )
        elif self.order == 2:
            grads = uniform_grid_gradient(
                field,
                spacing=self.dx,
                derivative_orders=2,
                include_mixed=self.return_mixed_derivs,
            )
            idx = 0
            for axis in range(self.dim):
                result[f"{self.invar}__{_AXIS_NAMES[axis]}__{_AXIS_NAMES[axis]}"] = (
                    grads[idx].unsqueeze(0).unsqueeze(0)
                )
                idx += 1

            if self.return_mixed_derivs:
                for ai, aj in combinations(range(self.dim), 2):
                    val = grads[idx].unsqueeze(0).unsqueeze(0)
                    result[f"{self.invar}__{_AXIS_NAMES[ai]}__{_AXIS_NAMES[aj]}"] = val
                    result[f"{self.invar}__{_AXIS_NAMES[aj]}__{_AXIS_NAMES[ai]}"] = val
                    idx += 1
        return result


class GradientsSpectral(torch.nn.Module):
    """Compute spatial derivatives via ``SpectralGridGradient``."""

    def __init__(
        self,
        invar: str,
        ell: Union[float, List[float]],
        dim: int = 3,
        order: int = 1,
        return_mixed_derivs: bool = False,
    ):
        super().__init__()
        self.invar = invar
        self.dim = dim
        self.order = order
        self.return_mixed_derivs = return_mixed_derivs
        self.ell = [ell] * dim if isinstance(ell, (float, int)) else list(ell)

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Lazy import (see GradientsFiniteDifference.forward for rationale).
        from physicsnemo.nn.functional.derivatives import spectral_grid_gradient

        u = input_dict[self.invar]
        field = u[0, 0]

        result: Dict[str, torch.Tensor] = {}

        if self.order == 1:
            grads = spectral_grid_gradient(
                field,
                lengths=self.ell[: self.dim],
                derivative_orders=1,
                include_mixed=False,
            )
            for axis in range(self.dim):
                result[f"{self.invar}__{_AXIS_NAMES[axis]}"] = (
                    grads[axis].unsqueeze(0).unsqueeze(0)
                )
        elif self.order == 2:
            grads = spectral_grid_gradient(
                field,
                lengths=self.ell[: self.dim],
                derivative_orders=2,
                include_mixed=self.return_mixed_derivs,
            )
            idx = 0
            for axis in range(self.dim):
                result[f"{self.invar}__{_AXIS_NAMES[axis]}__{_AXIS_NAMES[axis]}"] = (
                    grads[idx].unsqueeze(0).unsqueeze(0)
                )
                idx += 1

            if self.return_mixed_derivs:
                for ai, aj in combinations(range(self.dim), 2):
                    val = grads[idx].unsqueeze(0).unsqueeze(0)
                    result[f"{self.invar}__{_AXIS_NAMES[ai]}__{_AXIS_NAMES[aj]}"] = val
                    result[f"{self.invar}__{_AXIS_NAMES[aj]}__{_AXIS_NAMES[ai]}"] = val
                    idx += 1
        return result


class GradientsMeshlessFiniteDifference(torch.nn.Module):
    """Compute spatial derivatives using meshless central differences.

    Expects stencil values in the input dict keyed as ``u>>x::1``, ``u>>x::-1``, etc.
    """

    def __init__(
        self,
        invar: str,
        dx: Union[float, List[float]],
        dim: int = 3,
        order: int = 1,
        return_mixed_derivs: bool = False,
    ):
        super().__init__()
        self.invar = invar
        self.dim = dim
        self.order = order
        self.return_mixed_derivs = return_mixed_derivs
        self.dx = [dx] * dim if isinstance(dx, (float, int)) else list(dx)

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        v = self.invar

        if self.order == 1:
            for axis in range(self.dim):
                a = _AXIS_NAMES[axis]
                pos = input_dict[f"{v}>>{a}::1"]
                neg = input_dict[f"{v}>>{a}::-1"]
                result[f"{v}__{a}"] = (pos - neg) / (2 * self.dx[axis])

        elif self.order == 2:
            center = input_dict[v]
            for axis in range(self.dim):
                a = _AXIS_NAMES[axis]
                pos = input_dict[f"{v}>>{a}::1"]
                neg = input_dict[f"{v}>>{a}::-1"]
                result[f"{v}__{a}__{a}"] = (pos - 2 * center + neg) / (
                    self.dx[axis] ** 2
                )

            if self.return_mixed_derivs:
                for ai, aj in combinations(range(self.dim), 2):
                    an_i, an_j = _AXIS_NAMES[ai], _AXIS_NAMES[aj]
                    pp = input_dict[f"{v}>>{an_i}::1&&{an_j}::1"]
                    pn = input_dict[f"{v}>>{an_i}::1&&{an_j}::-1"]
                    np_ = input_dict[f"{v}>>{an_i}::-1&&{an_j}::1"]
                    nn = input_dict[f"{v}>>{an_i}::-1&&{an_j}::-1"]
                    mixed = (pp - pn - np_ + nn) / (4 * self.dx[ai] * self.dx[aj])
                    result[f"{v}__{an_i}__{an_j}"] = mixed
                    result[f"{v}__{an_j}__{an_i}"] = mixed
        return result


class GradientsLeastSquares(torch.nn.Module):
    """Compute spatial derivatives using least-squares gradient reconstruction.

    Uses ``MeshLSQGradient`` for first-order gradients and composes calls for
    second-order (same approach as the original physicsnemo-sym implementation).
    """

    def __init__(
        self,
        invar: str,
        dim: int = 3,
        order: int = 1,
        return_mixed_derivs: bool = False,
    ):
        super().__init__()
        self.invar = invar
        self.dim = dim
        self.order = order
        self.return_mixed_derivs = return_mixed_derivs

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Lazy import (see GradientsFiniteDifference.forward for rationale).
        from physicsnemo.nn.functional.derivatives import mesh_lsq_gradient

        coords = input_dict["coordinates"].detach()
        connectivity = input_dict["connectivity_tensor"]
        offsets, indices = connectivity[0], connectivity[1]

        values = input_dict[self.invar].squeeze(-1)
        first_grads = mesh_lsq_gradient(coords, values, offsets, indices)

        result: Dict[str, torch.Tensor] = {}

        if self.order == 1:
            for axis in range(self.dim):
                result[f"{self.invar}__{_AXIS_NAMES[axis]}"] = first_grads[
                    :, axis : axis + 1
                ]
            return result

        derivs = [first_grads[:, a : a + 1] for a in range(self.dim)]

        if self.order == 2:
            second_grads = []
            for a in range(self.dim):
                sg = mesh_lsq_gradient(coords, derivs[a].squeeze(-1), offsets, indices)
                second_grads.append(sg)

            for a in range(self.dim):
                result[f"{self.invar}__{_AXIS_NAMES[a]}__{_AXIS_NAMES[a]}"] = (
                    second_grads[a][:, a : a + 1]
                )

            if self.return_mixed_derivs:
                for ai, aj in combinations(range(self.dim), 2):
                    mixed = second_grads[ai][:, aj : aj + 1]
                    result[f"{self.invar}__{_AXIS_NAMES[ai]}__{_AXIS_NAMES[aj]}"] = (
                        mixed
                    )
                    result[f"{self.invar}__{_AXIS_NAMES[aj]}__{_AXIS_NAMES[ai]}"] = (
                        mixed
                    )
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class GradientCalculator:
    """Factory for spatial gradient modules.

    Parameters
    ----------
    device : str or torch.device or None
        Target device for the created gradient modules.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.sym.eq.gradients import GradientCalculator
    >>> calc = GradientCalculator(device="cpu")
    >>> module = calc.get_gradient_module("autodiff", invar="u", dim=2, order=1)
    """

    def __init__(self, device=None):
        self.device = device if device is not None else torch.device("cpu")
        self._registry = {
            "autodiff": GradientsAutoDiff,
            "meshless_finite_difference": GradientsMeshlessFiniteDifference,
            "finite_difference": GradientsFiniteDifference,
            "spectral": GradientsSpectral,
            "least_squares": GradientsLeastSquares,
        }

    def get_gradient_module(self, method_name: str, invar: str, **kwargs):
        """Return a gradient ``torch.nn.Module`` for the given method and variable."""
        module = self._registry[method_name](invar, **kwargs)
        module.to(self.device)
        return module

    def compute_gradients(self, input_dict, method_name=None, invar=None, **kwargs):
        """Compute gradients in one shot (convenience wrapper)."""
        module = self.get_gradient_module(method_name, invar, **kwargs)
        return module.forward(input_dict)


# ---------------------------------------------------------------------------
# Utilities (used by tests / PhysicsInformer)
# ---------------------------------------------------------------------------


def _compute_stencil3d(
    coords: torch.Tensor,
    model: torch.nn.Module,
    dx: float,
    return_mixed_derivs: bool = False,
):
    """Evaluate *model* at axis-aligned (and optionally diagonal) offset points.

    Returns a tuple of model outputs at shifted coordinates.  Without mixed
    derivs: 6 evaluations ``(+x, -x, +y, -y, +z, -z)``.  With mixed derivs:
    18 evaluations (6 axis-aligned + 12 diagonal pairs).
    """
    base = [coords[:, i : i + 1] for i in range(3)]

    def _eval(offsets):
        shifted = [base[i] + offsets[i] * dx for i in range(3)]
        return model(torch.cat(shifted, dim=1))

    axis_offsets = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]
    results = tuple(_eval(o) for o in axis_offsets)

    if not return_mixed_derivs:
        return results

    diag_offsets = [
        (1, 1, 0),
        (1, -1, 0),
        (-1, 1, 0),
        (-1, -1, 0),
        (1, 0, 1),
        (1, 0, -1),
        (-1, 0, 1),
        (-1, 0, -1),
        (0, 1, 1),
        (0, 1, -1),
        (0, -1, 1),
        (0, -1, -1),
    ]
    return results + tuple(_eval(o) for o in diag_offsets)


def compute_connectivity_tensor(
    nodes: torch.Tensor,
    edges: torch.Tensor,
    max_neighbors: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build CSR adjacency from node/edge lists.

    Uses vectorized PyTorch ops (argsort + bincount) via the
    ``physicsnemo.mesh.neighbors`` utilities.

    Parameters
    ----------
    nodes : torch.Tensor
        Node IDs with shape ``(N, 1)``.
    edges : torch.Tensor
        Edge pairs with shape ``(M, 2)``.
    max_neighbors : int or None
        Pad neighbor matrix to this width. If None, uses the maximum found.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(offsets, indices, neighbor_matrix)`` — CSR representation plus a
        padded ``(N, max_neighbors)`` neighbor matrix for batched computation.
    """
    from physicsnemo.mesh.neighbors._adjacency import build_adjacency_from_pairs

    num_nodes = nodes.numel()
    device = nodes.device

    bidir = torch.cat([edges, edges.flip(1)], dim=0)

    sort_by_target = torch.argsort(bidir[:, 1], stable=True)
    sort_indices = sort_by_target[torch.argsort(bidir[sort_by_target, 0], stable=True)]
    sorted_edges = bidir[sort_indices]

    mask = torch.ones(len(sorted_edges), dtype=torch.bool, device=device)
    mask[1:] = (sorted_edges[:-1] != sorted_edges[1:]).any(dim=1)
    unique_edges = sorted_edges[mask]

    adj = build_adjacency_from_pairs(unique_edges[:, 0], unique_edges[:, 1], num_nodes)

    offsets = adj.offsets
    indices = adj.indices

    if max_neighbors is None:
        counts = offsets[1:] - offsets[:-1]
        max_neighbors = int(counts.max().item()) if len(counts) > 0 else 0

    neighbor_matrix = torch.full(
        (num_nodes, max_neighbors), -1, dtype=torch.long, device=device
    )
    for i in range(num_nodes):
        s, e = offsets[i].item(), offsets[i + 1].item()
        n_neigh = e - s
        if n_neigh > 0:
            neighbor_matrix[i, :n_neigh] = indices[s:e]

    return offsets, indices, neighbor_matrix


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gradient_autodiff(y: torch.Tensor, x: List[torch.Tensor]) -> List[torch.Tensor]:
    grad_outputs = [torch.ones_like(y, device=y.device)]
    grad = torch.autograd.grad(
        [y],
        x,
        grad_outputs=grad_outputs,
        create_graph=True,
        allow_unused=True,
    )
    if grad is None:
        return [torch.zeros_like(xx) for xx in x]
    return [g if g is not None else torch.zeros_like(x[i]) for i, g in enumerate(grad)]
