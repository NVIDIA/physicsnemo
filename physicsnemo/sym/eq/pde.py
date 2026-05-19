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

"""Base class for symbolic PDE definitions."""

from __future__ import annotations

from sympy import Eq, Function, Matrix, init_printing, preview

from physicsnemo.sym.computation import Computation


class PDE:
    """Base class for all partial differential equations.

    Subclasses must populate ``self.equations`` (a ``dict[str, sympy.Expr]``)
    and set ``self.dim``.

    Examples
    --------
    Define a 2-D advection-diffusion equation:

    >>> from sympy import Symbol, Function, Number
    >>> from physicsnemo.sym.eq.pde import PDE
    >>>
    >>> class AdvectionDiffusion(PDE):
    ...     def __init__(self, D=0.1):
    ...         self.dim = 2
    ...         x, y = Symbol("x"), Symbol("y")
    ...         T = Function("T")(x, y)
    ...         u = Function("u")(x, y)
    ...         v = Function("v")(x, y)
    ...         self.equations = {
    ...             "advection_diffusion": (
    ...                 u * T.diff(x) + v * T.diff(y)
    ...                 - D * (T.diff(x, 2) + T.diff(y, 2))
    ...             ),
    ...         }
    ...
    >>> pde = AdvectionDiffusion(D=0.01)
    >>> pde.pprint()
    advection_diffusion: ...
    """

    name = "PDE"

    def pprint(self, print_latex: bool = False) -> None:
        """Pretty-print the equations."""
        init_printing(use_latex=True)
        for key, value in self.equations.items():
            print(str(key) + ": " + str(value))
        if print_latex:
            preview(
                Matrix(
                    [
                        Eq(Function(name, real=True), eq)
                        for name, eq in self.equations.items()
                    ]
                ),
                mat_str="cases",
                mat_delim="",
            )

    def subs(self, x, y):
        """Substitute *x* with *y* in all equations (calls SymPy ``subs``)."""
        for name, eq in self.equations.items():
            self.equations[name] = eq.subs(x, y).doit()

    def make_computations(
        self,
        detach_names: list[str] | None = None,
    ) -> list[Computation]:
        """Convert each equation into a :class:`Computation`.

        Returns
        -------
        list[Computation]
            One computation per equation in ``self.equations``.
        """
        if detach_names is None:
            detach_names = []

        return [
            Computation.from_sympy(eq, str(name), detach_names=detach_names)
            for name, eq in self.equations.items()
        ]
