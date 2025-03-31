# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

from physicsnemo.sym.eq.pde import PDE
from sympy import Symbol, Function, Number


class SWE_NL(PDE):
    """SWE Nonlinear PDE using PhysicsNeMo Sym"""

    name = "SWE_NL"

    def __init__(self, g=1.0, nu=1.0e-3):

        # x, y, time
        x, y, t = Symbol("x"), Symbol("y"), Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "t": t}

        # make functions
        h = Function("h")(*input_variables)
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        hu = Function("hu")(*input_variables)
        hv = Function("hv")(*input_variables)
        hh = Function("hh")(*input_variables)
        huu = Function("huu")(*input_variables)
        huv = Function("huv")(*input_variables)
        hvv = Function("hvv")(*input_variables)

        # initialize constants
        g = Number(g)
        nu = Number(nu)

        # set equations
        self.equations = {}
        self.equations["Dh"] = +h.diff(t) + hu.diff(x) + hv.diff(y)
        self.equations["Du"] = (
            +u.diff(t)
            + (huu.diff(x) + 0.5 * g * hh.diff(x))
            + huv.diff(y)
            - nu * (u.diff(x, 2) + u.diff(y, 2))
        )
        self.equations["Dv"] = (
            +v.diff(t)
            + (hvv.diff(y) + 0.5 * g * hh.diff(y))
            + huv.diff(x)
            - nu * (v.diff(x, 2) + v.diff(y, 2))
        )
