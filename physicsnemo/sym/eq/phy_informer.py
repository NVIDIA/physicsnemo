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

"""PhysicsInformer: PDE residual evaluator using modulus derivative functionals."""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from physicsnemo.sym.computation import Computation
from physicsnemo.sym.eq.gradients import (
    GradientCalculator,
    compute_connectivity_tensor,
)
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.graph import Graph

logger = logging.getLogger(__name__)


class PhysicsInformer:
    """Compute the residual of a PDE using automatic spatial derivative computation.

    Given a :class:`PDE` and a list of ``required_outputs``, this class builds a
    computational graph that automatically computes spatial derivatives and evaluates
    equation residuals.

    Parameters
    ----------
    required_outputs : list[str]
        Equation names to compute (e.g. ``["continuity", "momentum_x"]``).
    equations : PDE
        The PDE whose ``equations`` dict defines the symbolic residuals.
    grad_method : str
        One of ``"autodiff"``, ``"meshless_finite_difference"``,
        ``"finite_difference"``, ``"spectral"``, ``"least_squares"``.
    fd_dx : float or list[float]
        Grid spacing for FD / meshless FD methods.
    bounds : list[float]
        Domain lengths for spectral method.
    compute_connectivity : bool
        If True and using ``"least_squares"``, build the connectivity tensor
        on the fly from ``"nodes"`` and ``"edges"`` in the input dict.
    device : str or torch.device or None
        Target device.

    Examples
    --------
    >>> import torch
    >>> from sympy import Symbol, Function, Number
    >>> from physicsnemo.sym.eq.pde import PDE
    >>> from physicsnemo.sym.eq.phy_informer import PhysicsInformer
    >>>
    >>> class Diffusion(PDE):
    ...     def __init__(self, D=0.1):
    ...         self.dim = 2
    ...         x, y = Symbol("x"), Symbol("y")
    ...         u = Function("u")(x, y)
    ...         self.equations = {
    ...             "diffusion": -D * (u.diff(x, 2) + u.diff(y, 2)),
    ...         }
    ...
    >>> pde = Diffusion(D=0.01)
    >>> pi = PhysicsInformer(
    ...     required_outputs=["diffusion"],
    ...     equations=pde,
    ...     grad_method="finite_difference",
    ...     fd_dx=0.01,
    ... )
    >>> field = torch.rand(1, 1, 32, 32)
    >>> result = pi.forward({"u": field})
    >>> result["diffusion"].shape
    torch.Size([1, 1, 32, 32])
    """

    def __init__(
        self,
        required_outputs: List[str],
        equations: PDE,
        grad_method: str,
        fd_dx: Union[float, List[float]] = 0.001,
        bounds: List[float] | None = None,
        compute_connectivity: bool = True,
        device: Optional[str] = None,
    ):
        if bounds is None:
            bounds = [2 * np.pi, 2 * np.pi, 2 * np.pi]

        self.required_outputs = required_outputs
        self.equations = equations
        self.dim = equations.dim
        self.grad_method = grad_method
        self.fd_dx = fd_dx
        self.bounds = bounds
        self.compute_connectivity = compute_connectivity
        self.device = device if device is not None else torch.device("cpu")

        self.grad_calc = GradientCalculator(device=self.device)
        self.computations = self.equations.make_computations()

        self.require_mixed_derivs = False
        self.graph = self._create_graph()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def required_inputs(self) -> list[str]:
        """Return the list of tensor names the ``forward`` call expects."""
        comp_outputs = [c.outputs[0] for c in self.computations]
        node_inputs: set[str] = set()

        for name in self.required_outputs:
            if name not in comp_outputs:
                raise ValueError(
                    f"{name} not in equation outputs. Choose from {comp_outputs}"
                )

        fd, sd, others = self._extract_derivatives()
        node_inputs.update(fd | sd | others)

        for comp in self.computations:
            if comp.outputs[0] in self.required_outputs and comp.inputs:
                node_inputs.update(comp.inputs)

        inputs = list(node_inputs)

        if self.grad_method == "meshless_finite_difference":
            inputs = self._expand_for_meshless_fd(inputs)
        elif self.grad_method == "autodiff":
            inputs.append("coordinates")
        elif self.grad_method == "least_squares":
            inputs.append("coordinates")
            if self.compute_connectivity:
                inputs.extend(["nodes", "edges"])
            else:
                inputs.append("connectivity_tensor")
        return inputs

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _create_graph(self) -> Graph:
        first_deriv, second_deriv, _ = self._extract_derivatives()

        input_keys = list(self.required_inputs)
        output_keys = list(self.required_outputs)

        diff_nodes = self._create_diff_nodes(first_deriv, dim=self.dim, order=1)
        diff_nodes += self._create_diff_nodes(second_deriv, dim=self.dim, order=2)

        return Graph(
            self.computations, input_keys, output_keys, diff_nodes=diff_nodes
        ).to(self.device)

    def _extract_derivatives(self):
        first_deriv: set[str] = set()
        second_deriv: set[str] = set()
        other_derivs: set[str] = set()

        for comp in self.computations:
            if comp.outputs[0] in self.required_outputs:
                for d in comp.derivatives:
                    self._process_derivative(d, first_deriv, second_deriv, other_derivs)

        first_consolidated = {s.split("__")[0] for s in first_deriv}
        second_consolidated = {s.split("__")[0] for s in second_deriv}
        return first_consolidated, second_consolidated, other_derivs

    def _process_derivative(self, d, first_deriv, second_deriv, other_derivs):
        parts = d.split("__")
        if len(parts) - 1 > 2:
            raise ValueError("Only up to second-order PDEs are supported")

        allowed = {"x", "y", "z"}
        for var in parts[1:]:
            if var not in allowed:
                logger.warning(
                    "Derivative w.r.t %s detected — must be supplied manually.", var
                )
                other_derivs.add(d)
                return

        if len(parts) - 1 == 2 and parts[1] != parts[2]:
            self.require_mixed_derivs = True

        count = len(parts) - 1
        if count == 1:
            first_deriv.add(d)
        elif count == 2:
            second_deriv.add(d)

    def _create_diff_nodes(self, derivatives, dim, order):
        nodes: list[Computation] = []
        for var in derivatives:
            node = self._create_diff_node(var, dim, order)
            if node is not None:
                nodes.append(node)
        return nodes

    def _create_diff_node(self, var, dim, order):
        methods = {
            "finite_difference": self._fd_module,
            "spectral": self._spectral_module,
            "least_squares": self._ls_module,
            "autodiff": self._autodiff_module,
            "meshless_finite_difference": self._meshless_fd_module,
        }
        if self.grad_method not in methods:
            return None

        module = methods[self.grad_method](var, dim, order)
        output_keys = self._derivative_keys(
            var, dim, order, return_mixed_derivs=self.require_mixed_derivs
        )
        return Computation([var], output_keys, module)

    def _derivative_keys(self, var, dim, order, return_mixed_derivs=False):
        base = ["__x", "__y", "__z"][:dim]
        keys = [f"{var}{k * order}" for k in base]
        if return_mixed_derivs and order == 2:
            from itertools import combinations

            for ai, aj in combinations(range(dim), 2):
                an = ["x", "y", "z"]
                keys.append(f"{var}__{an[ai]}__{an[aj]}")
                keys.append(f"{var}__{an[aj]}__{an[ai]}")
        return keys

    # --- Module builders --------------------------------------------------

    def _fd_module(self, var, dim, order):
        return self.grad_calc.get_gradient_module(
            "finite_difference",
            var,
            dx=self.fd_dx,
            dim=dim,
            order=order,
            return_mixed_derivs=self.require_mixed_derivs and order == 2,
        )

    def _spectral_module(self, var, dim, order):
        return self.grad_calc.get_gradient_module(
            "spectral",
            var,
            ell=self.bounds,
            dim=dim,
            order=order,
            return_mixed_derivs=self.require_mixed_derivs and order == 2,
        )

    def _ls_module(self, var, dim, order):
        return self.grad_calc.get_gradient_module(
            "least_squares",
            var,
            dim=dim,
            order=order,
            return_mixed_derivs=self.require_mixed_derivs and order == 2,
        )

    def _autodiff_module(self, var, dim, order):
        return self.grad_calc.get_gradient_module(
            "autodiff",
            var,
            dim=dim,
            order=order,
            return_mixed_derivs=self.require_mixed_derivs and order == 2,
        )

    def _meshless_fd_module(self, var, dim, order):
        return self.grad_calc.get_gradient_module(
            "meshless_finite_difference",
            var,
            dx=self.fd_dx,
            dim=dim,
            order=order,
            return_mixed_derivs=self.require_mixed_derivs and order == 2,
        )

    # ------------------------------------------------------------------
    # Meshless FD helpers
    # ------------------------------------------------------------------

    def _expand_for_meshless_fd(self, node_inputs):
        expanded = copy.deepcopy(node_inputs)
        for name in node_inputs:
            mfd_vars = [
                f"{name}>>x::1",
                f"{name}>>x::-1",
                f"{name}>>y::1",
                f"{name}>>y::-1",
                f"{name}>>z::1",
                f"{name}>>z::-1",
            ]
            expanded.extend(mfd_vars[: 2 * self.dim])
        return expanded

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs = dict(inputs)
        if self.grad_method == "least_squares" and self.compute_connectivity:
            connectivity = compute_connectivity_tensor(inputs["nodes"], inputs["edges"])
            inputs["connectivity_tensor"] = connectivity
        return self.graph.forward(inputs)
